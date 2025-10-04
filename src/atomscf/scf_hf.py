r"""Hartree-Fock SCF 实现

本模块实现原子 Hartree-Fock 自洽场计算。

实现策略
========

采用**算子-作用式** + 显式矩阵对角化的混合方案：

1. 交换算子 K 实现为算子作用（避免 O(N²) 存储占据态间积分）
2. Fock 矩阵通过在网格基上应用算子构造（F_ij = <i|F|j>）
3. 使用标准对称本征求解器（np.linalg.eigh）

与 LinearOperator + LOBPCG 的区别
==================================

虽然 plan 中建议使用 LinearOperator + 迭代求解器，但当前实现选择显式矩阵化：

**优势**:
- 实现简单，调试容易
- 对小规模网格（n ~ 1000）性能可接受
- 避免 LOBPCG 收敛性问题

**劣势**:
- 每次 SCF 迭代需 O(n²) 时间构造交换矩阵
- 内存占用 O(n²)（但对 n=1000，仅 8MB）

**后续优化**: 可在 M6.2 阶段切换到 LinearOperator 以提升性能

Physical Background
===================

Hartree-Fock 方程（径向中心场形式）：

.. math::

    \\left[-\\frac{1}{2}\\frac{d^2}{dr^2} + \\frac{\\ell(\\ell+1)}{2r^2}
    + v_{ext}(r) + v_H(r) + K_\\ell\\right] u_\\ell(r) = \\varepsilon_\\ell u_\\ell(r)

其中：
- v_H(r) 是 Hartree 势（局域）
- K_ℓ 是交换算子（非局域）

References
----------
.. [HFMethod] Szabo & Ostlund (1996)
   "Modern Quantum Chemistry"
   Dover, Chapter 3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .grid import trapezoid_weights
from .hartree import v_hartree
from .hf import SlaterIntegralCache, exchange_operator_s
from .operator import radial_hamiltonian_matrix, solve_bound_states_fd
from .utils import trapz

__all__ = ["HFConfig", "HFResult", "run_hf_minimal", "run_hf_scf_s"]


@dataclass
class HFConfig:
    r"""HF 最小实现配置（教学版，仅支持 H）。

    Attributes
    ----------
    Z : int
        原子序数（当前仅支持 1）。
    r : numpy.ndarray
        径向网格。
    w : numpy.ndarray
        梯形权重。
    tol : float
        收敛阈值。
    maxiter : int
        最大迭代数。
    """

    Z: int
    r: np.ndarray
    w: np.ndarray
    tol: float = 1e-7
    maxiter: int = 50


@dataclass
class HFResult:
    r"""HF 最小实现结果容器（H 验证）。"""

    converged: bool
    iterations: int
    eps_1s: float
    u_1s: np.ndarray


def run_hf_minimal(cfg: HFConfig) -> HFResult:
    r"""对 H 原子运行最小 HF 自洽：交换与 Hartree 相消，仅剩外势。

    该实现用于快速验证 HF 思想与数值骨架。对多电子（如 C）不适用。
    """
    if cfg.Z != 1:
        raise NotImplementedError("最小 HF 实现当前仅支持 Z=1（氢原子）")
    r = cfg.r
    v_ext = -1.0 / np.maximum(r, 1e-12)
    # 直接在外势下求解 1s 态（HF 下与单电子精确解一致）
    eps, U = solve_bound_states_fd(r, l=0, v_of_r=v_ext, k=1)
    return HFResult(converged=True, iterations=1, eps_1s=float(eps[0]), u_1s=U[0])


@dataclass
class HFSCFConfig:
    r"""完整 HF SCF 配置（支持 s 轨道）。

    Attributes
    ----------
    Z : int
        原子序数
    r : np.ndarray
        径向网格
    w : np.ndarray
        积分权重
    n_occ : int
        占据态数量（例 He: n_occ=1 表示 1s²）
    occ_nums : list[float]
        各占据态的占据数（例 He: [2.0] 表示 1s²）
    mix_alpha : float
        密度混合参数（0=不混合，1=完全更新）
    tol : float
        收敛阈值（波函数 RMS 差）
    maxiter : int
        最大 SCF 迭代数
    initial_guess : str
        初始猜测方式："hydrogen" 或 "lsda"
    """

    Z: int
    r: np.ndarray
    w: np.ndarray
    n_occ: int = 1
    occ_nums: list[float] | None = None
    mix_alpha: float = 0.3
    tol: float = 1e-6
    maxiter: int = 100
    initial_guess: str = "hydrogen"


@dataclass
class HFSCFResult:
    r"""HF SCF 结果容器。

    Attributes
    ----------
    converged : bool
        是否收敛
    iterations : int
        实际迭代次数
    eigenvalues : np.ndarray
        本征能量（占据态）
    orbitals : list[np.ndarray]
        占据态波函数
    E_total : float
        总能量
    E_kinetic : float
        动能
    E_ext : float
        外势能
    E_hartree : float
        Hartree 能量
    E_exchange : float
        交换能量
    """

    converged: bool
    iterations: int
    eigenvalues: np.ndarray
    orbitals: list[np.ndarray]
    E_total: float
    E_kinetic: float
    E_ext: float
    E_hartree: float
    E_exchange: float


def run_hf_scf_s(cfg: HFSCFConfig) -> HFSCFResult:
    r"""运行 s 轨道 HF SCF 计算。

    实现完整的 Hartree-Fock 自洽场循环：
    1. 初始化波函数猜测
    2. 计算 Hartree 势
    3. 构造 Fock 矩阵（包含交换算子）
    4. 对角化求解本征态
    5. 检查收敛并混合
    6. 重复直到收敛

    Parameters
    ----------
    cfg : HFSCFConfig
        HF SCF 配置

    Returns
    -------
    HFSCFResult
        收敛的 HF 结果

    Notes
    -----
    **收敛判据**: 波函数的 RMS 变化 < tol

    **密度混合**: u_new = α * u_scf + (1-α) * u_old

    **能量计算**:
        - E_kin = Σ_i n_i <u_i| -∇²/2 |u_i>
        - E_ext = Σ_i n_i <u_i| v_ext |u_i>
        - E_H = (1/2) ∫ n(r) v_H(r) 4πr² dr
        - E_x = (1/2) Σ_i n_i <u_i| K |u_i>
        - E_total = E_kin + E_ext + E_H + E_x

    Examples
    --------
    氢原子 HF::

        >>> from atomscf.grid import radial_grid_linear, trapezoid_weights
        >>> r, _ = radial_grid_linear(n=1000, rmin=1e-6, rmax=50.0)
        >>> w = trapezoid_weights(r)
        >>> cfg = HFSCFConfig(Z=1, r=r, w=w, n_occ=1, occ_nums=[1.0])
        >>> res = run_hf_scf_s(cfg)
        >>> print(f"E_1s = {res.eigenvalues[0]:.6f} Ha")  # 应约 -0.5 Ha
        >>> print(f"E_total = {res.E_total:.6f} Ha")      # 应约 -0.5 Ha

    See Also
    --------
    exchange_operator_s : s 轨道交换算子
    v_hartree : Hartree 势计算
    """
    r = cfg.r
    w = cfg.w
    Z = cfg.Z
    n_occ = cfg.n_occ

    # 默认占据数（例 H: [1.0], He: [2.0]）
    if cfg.occ_nums is None:
        occ_nums = [2.0] * n_occ  # 默认闭壳层
    else:
        occ_nums = cfg.occ_nums

    if len(occ_nums) != n_occ:
        raise ValueError(f"占据数列表长度 ({len(occ_nums)}) 与 n_occ ({n_occ}) 不匹配")

    # 外势
    v_ext = -Z / np.maximum(r, 1e-12)

    # 初始化：氢样波函数猜测
    if cfg.initial_guess == "hydrogen":
        # 1s: u(r) = 2*sqrt(Z) * exp(-Z*r)
        # 但必须满足 Dirichlet 边界条件：u[0] = u[-1] = 0
        u_initial = 2.0 * np.sqrt(Z) * np.exp(-Z * r)
        u_initial[0] = 0.0  # 强制边界条件
        u_initial[-1] = 0.0
        # 归一化
        norm = np.sqrt(trapz(u_initial**2, r, w))
        u_initial /= norm
        u_orbitals = [u_initial.copy() for _ in range(n_occ)]
    else:
        raise NotImplementedError(f"初始猜测方式 '{cfg.initial_guess}' 未实现")

    # Slater 积分缓存
    cache = SlaterIntegralCache()

    # SCF 主循环
    converged = False
    for iteration in range(cfg.maxiter):
        # 1. 计算电子密度（s 轨道球对称）
        # 注意：n_r 是径向密度 (u²)，需转换为 3D 数密度 n₃D = u²/(4πr²)
        n_radial = np.zeros_like(r)
        for u_i, n_i in zip(u_orbitals, occ_nums):
            n_radial += n_i * u_i**2

        # 转换为 3D 数密度（v_hartree 需要此格式）
        n_3d = n_radial / (4 * np.pi * np.maximum(r**2, 1e-30))

        # 2. 计算 Hartree 势
        v_H = v_hartree(n_3d, r, w)

        # 3. 构造 Fock 矩阵
        F_matrix = _build_fock_matrix_s(
            r, w, l=0, v_ext=v_ext, v_H=v_H, u_occ=u_orbitals, occ_nums=occ_nums, cache=cache
        )

        # 4. 对角化
        eigs, vecs = np.linalg.eigh(F_matrix)

        # 5. 提取占据态（从内部点重构完整波函数）
        u_new_orbitals = []
        for i_occ in range(n_occ):
            u_inner = vecs[:, i_occ]
            # 边界补零（Dirichlet）
            u_full = np.zeros(len(r))
            u_full[1:-1] = u_inner
            # 归一化
            norm = np.sqrt(trapz(u_full**2, r, w))
            u_full /= norm
            u_new_orbitals.append(u_full)

        # 6. 检查收敛（波函数 RMS 差）
        rms_diff = 0.0
        for u_new, u_old in zip(u_new_orbitals, u_orbitals):
            rms_diff += np.sqrt(np.mean((u_new - u_old) ** 2))
        rms_diff /= n_occ

        if rms_diff < cfg.tol:
            converged = True
            u_orbitals = u_new_orbitals
            break

        # 7. 密度混合
        u_orbitals = [
            cfg.mix_alpha * u_new + (1 - cfg.mix_alpha) * u_old
            for u_new, u_old in zip(u_new_orbitals, u_orbitals)
        ]

        # 重归一化
        for i in range(n_occ):
            norm = np.sqrt(trapz(u_orbitals[i] ** 2, r, w))
            u_orbitals[i] /= norm

    # 计算最终能量
    E_kin, E_ext, E_H, E_x = _compute_hf_energies(
        r, w, v_ext, u_orbitals, occ_nums, cache
    )
    E_total = E_kin + E_ext + E_H + E_x

    # 提取最终本征值（重构最后一次的 Fock 矩阵并对角化）
    final_eigs = eigs[:n_occ]

    return HFSCFResult(
        converged=converged,
        iterations=iteration + 1 if converged else cfg.maxiter,
        eigenvalues=final_eigs,
        orbitals=u_orbitals,
        E_total=E_total,
        E_kinetic=E_kin,
        E_ext=E_ext,
        E_hartree=E_H,
        E_exchange=E_x,
    )


def _build_fock_matrix_s(
    r: np.ndarray,
    w: np.ndarray,
    l: int,
    v_ext: np.ndarray,
    v_H: np.ndarray,
    u_occ: list[np.ndarray],
    occ_nums: list[float],
    cache: SlaterIntegralCache,
) -> np.ndarray:
    r"""构造 s 轨道的 Fock 矩阵。

    Fock 矩阵定义为：
    F = H_loc + K

    其中：
    - H_loc: 局域哈密顿（动能 + 外势 + Hartree 势）
    - K: 交换矩阵

    Parameters
    ----------
    r : np.ndarray
        径向网格
    w : np.ndarray
        积分权重
    l : int
        角动量（s 轨道为 0）
    v_ext : np.ndarray
        外势
    v_H : np.ndarray
        Hartree 势
    u_occ : list[np.ndarray]
        占据态波函数
    occ_nums : list[float]
        占据数
    cache : SlaterIntegralCache
        Slater 积分缓存

    Returns
    -------
    np.ndarray
        Fock 矩阵（内部点，形状 (n-2, n-2)）
    """
    # 1. 局域哈密顿矩阵（内部点）
    H_loc, r_inner = radial_hamiltonian_matrix(r, l, v_ext + v_H)
    n_inner = len(r_inner)

    # 2. 构造交换矩阵（通过应用算子到基向量）
    K_op = exchange_operator_s(r, w, u_occ, occ_nums, cache=cache)

    K_matrix = np.zeros((n_inner, n_inner))
    for j in range(n_inner):
        # 构造第 j 个基向量（完整网格）
        e_j_full = np.zeros(len(r))
        e_j_full[j + 1] = 1.0  # 注意：内部点从索引 1 开始

        # 应用交换算子
        K_e_j_full = K_op(e_j_full)

        # 提取内部点
        K_matrix[:, j] = K_e_j_full[1:-1]

    # 3. Fock 矩阵 = 局域 + 交换
    F_matrix = H_loc + K_matrix

    return F_matrix


def _compute_hf_energies(
    r: np.ndarray,
    w: np.ndarray,
    v_ext: np.ndarray,
    u_orbitals: list[np.ndarray],
    occ_nums: list[float],
    cache: SlaterIntegralCache,
) -> tuple[float, float, float, float]:
    r"""计算 HF 能量分量。

    Returns
    -------
    tuple[float, float, float, float]
        (E_kinetic, E_ext, E_hartree, E_exchange)

    Notes
    -----
    **动能**: 使用差分计算 <u| -∇²/2 |u>

    **交换能**: E_x = (1/2) Σ_i n_i <u_i| K |u_i>

    **Hartree 能**: E_H = (1/2) ∫ ρ(r) v_H(r) d³r
    """
    # 电子密度（径向）
    n_radial = np.zeros_like(r)
    for u_i, n_i in zip(u_orbitals, occ_nums):
        n_radial += n_i * u_i**2

    # 1. 外势能
    E_ext = 0.0
    for u_i, n_i in zip(u_orbitals, occ_nums):
        E_ext += n_i * trapz(u_i**2 * v_ext, r, w)

    # 2. 动能（用差分近似 d²u/dr²）
    E_kin = 0.0
    for u_i, n_i in zip(u_orbitals, occ_nums):
        # 差分计算 -∇²/2
        d2u_dr2 = np.zeros_like(u_i)
        for i in range(1, len(r) - 1):
            dr_left = r[i] - r[i - 1]
            dr_right = r[i + 1] - r[i]
            dr_avg = (dr_left + dr_right) / 2.0
            d2u_dr2[i] = (
                (u_i[i + 1] - u_i[i]) / dr_right - (u_i[i] - u_i[i - 1]) / dr_left
            ) / dr_avg

        T_u = -0.5 * d2u_dr2
        E_kin += n_i * trapz(u_i * T_u, r, w)

    # 3. Hartree 能量
    # 转换为 3D 数密度
    n_3d = n_radial / (4 * np.pi * np.maximum(r**2, 1e-30))
    v_H = v_hartree(n_3d, r, w)
    # E_H = (1/2) ∫ ρ(r) v_H(r) d³r = (1/2) ∫ [n_rad/(4πr²)] v_H 4πr² dr
    E_hartree = 0.5 * trapz(n_radial * v_H, r, w)

    # 4. 交换能量
    K_op = exchange_operator_s(r, w, u_orbitals, occ_nums, cache=cache)
    E_exchange = 0.0
    for u_i, n_i in zip(u_orbitals, occ_nums):
        K_u_i = K_op(u_i)
        E_exchange += 0.5 * n_i * trapz(u_i * K_u_i, r, w)

    return E_kin, E_ext, E_hartree, E_exchange

