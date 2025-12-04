r"""Hartree-Fock SCF 实现

本模块实现原子 Hartree-Fock 自洽场计算。

实现方法
========

采用算子-作用式构造 Fock 矩阵：

1. 交换算子 K 作用于基向量得到矩阵元
2. 组合局域哈密顿与交换矩阵
3. 对角化求解占据态

物理背景
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

import numpy as np

from .hartree import v_hartree
from .hf import SlaterIntegralCache, exchange_operator_general, exchange_operator_s
from .operator import build_transformed_hamiltonian, radial_hamiltonian_matrix, solve_bound_states_fd
from .utils import trapz

__all__ = [
    "HFConfig",
    "HFResult",
    "run_hf_minimal",
    "HFSCFConfig",
    "HFSCFResult",
    "run_hf_scf_s",
    "HFSCFGeneralConfig",
    "HFSCFGeneralResult",
    "run_hf_scf",
]


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
    delta : float | None
        变量变换参数 δ（仅用于指数网格）
    Rp : float | None
        变量变换参数 R_p（仅用于指数网格）
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
    delta: float | None = None
    Rp: float | None = None


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

    # 边界处理（FD2: 两端各去掉 1 个点）
    n_boundary_left = 1
    n_boundary_right = 1

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
        F_matrix, B_matrix = _build_fock_matrix_s(
            r,
            w,
            l=0,
            v_ext=v_ext,
            v_H=v_H,
            u_occ=u_orbitals,
            occ_nums=occ_nums,
            cache=cache,
            delta=cfg.delta,
            Rp=cfg.Rp,
        )

        # 4. 对角化（广义或标准特征值问题）
        use_transformed = B_matrix is not None
        if use_transformed:
            from scipy.linalg import eigh

            eigs, vecs_v = eigh(F_matrix, B_matrix)
        else:
            eigs, vecs_v = np.linalg.eigh(F_matrix)

        # 5. 提取占据态（从内部点重构完整波函数）
        u_new_orbitals = []
        for i_occ in range(n_occ):
            v_inner = vecs_v[:, i_occ]

            if use_transformed:
                # 变换回 u 空间：u(j) = v(j) * exp(j*delta/2), j=1..n-1
                j_vals = np.arange(1, len(r))
                transform = np.exp(j_vals * cfg.delta / 2.0)
                u_inner = v_inner * transform

                # 边界补零
                u_full = np.zeros(len(r))
                u_full[0] = 0.0
                u_full[1:] = u_inner
            else:
                # 标准 FD2：边界补零
                u_full = np.zeros(len(r))
                u_full[n_boundary_left : len(r) - n_boundary_right] = v_inner

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
    delta: float | None = None,
    Rp: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
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
    delta : float | None
        变量变换参数（如提供则使用变换方法）
    Rp : float | None
        变量变换参数

    Returns
    -------
    F_matrix : np.ndarray
        Fock 矩阵（内部点）
    B_matrix : np.ndarray | None
        质量矩阵（仅变换方法返回，否则为 None）
    """
    use_transformed = (delta is not None) and (Rp is not None)

    if use_transformed:
        # 使用变量变换构造局域 Hamiltonian
        H_loc, B, r_inner = build_transformed_hamiltonian(r, l, v_ext + v_H, delta, Rp)
        n_inner = len(r_inner)

        # 构造交换矩阵（在 v 空间，需要 u ↔ v 转换）
        K_op = exchange_operator_s(r, w, u_occ, occ_nums, cache=cache)
        K_matrix = np.zeros((n_inner, n_inner))

        # 变换因子: u(j) = v(j) * exp(j*delta/2)，j=1..n-1
        j_vals = np.arange(1, len(r))
        transform = np.exp(j_vals * delta / 2.0)

        # 预计算所有基函数的 K 作用结果（在 v 空间）
        K_v_basis = []
        for j in range(n_inner):
            # 第 j 个 v 空间基向量（内部点）
            v_j_inner = np.zeros(n_inner)
            v_j_inner[j] = 1.0

            # 转换到 u 空间（完整网格，包含边界）
            u_j_full = np.zeros(len(r))
            u_j_full[0] = 0.0
            u_j_full[1:] = v_j_inner * transform

            # 应用交换算子（在 u 空间）
            K_u_j_full = K_op(u_j_full)

            # 转换回 v 空间（仅内部点）
            K_v_j_inner = K_u_j_full[1:] / transform
            K_v_basis.append(K_v_j_inner)

        # 计算矩阵元：<v_i | K | v_j>_B = sum_k v_i[k] * (K v_j)[k] * B[k,k]
        # 由于基是 delta 函数，v_i 只在 i 处为 1，所以简化为：
        for i in range(n_inner):
            for j in range(n_inner):
                K_matrix[i, j] = K_v_basis[j][i] * B[i, i]

        # 对称化
        K_matrix = 0.5 * (K_matrix + K_matrix.T)
        F_matrix = H_loc + K_matrix
        F_matrix = 0.5 * (F_matrix + F_matrix.T)

        return F_matrix, B

    else:
        # 标准 FD2 方法
        H_loc, r_inner = radial_hamiltonian_matrix(r, l, v_ext + v_H)
        n_inner = len(r_inner)
        n_boundary_left = 1
        n_boundary_right = 1

        K_op = exchange_operator_s(r, w, u_occ, occ_nums, cache=cache)
        K_matrix = np.zeros((n_inner, n_inner))

        for j in range(n_inner):
            e_j_full = np.zeros(len(r))
            e_j_full[j + n_boundary_left] = 1.0
            K_e_j_full = K_op(e_j_full)
            K_matrix[:, j] = K_e_j_full[n_boundary_left : len(r) - n_boundary_right]

        K_matrix = 0.5 * (K_matrix + K_matrix.T)
        F_matrix = H_loc + K_matrix
        F_matrix = 0.5 * (F_matrix + F_matrix.T)

        return F_matrix, None


def _compute_hf_energies(
    r: np.ndarray,
    w: np.ndarray,
    v_ext: np.ndarray,
    u_orbitals: list[np.ndarray],
    occ_nums: list[float],
    cache: SlaterIntegralCache,
) -> tuple[float, float, float, float]:
    r"""计算 HF 能量分量（仅适用于 s 轨道）。

    Returns
    -------
    tuple[float, float, float, float]
        (E_kinetic, E_ext, E_hartree, E_exchange)

    Notes
    -----
    **动能**: 使用差分计算 <u| -∇²/2 |u>

    **交换能**: E_x = (1/2) Σ_i n_i <u_i| K |u_i>

    **Hartree 能**: E_H = (1/2) ∫ ρ(r) v_H(r) d³r

    **警告**: 此函数仅适用于纯 s 轨道体系！多 l 通道应使用 _compute_hf_energies_general。
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

    # 4. 交换能量（仅 s 轨道）
    K_op = exchange_operator_s(r, w, u_orbitals, occ_nums, cache=cache)
    E_exchange = 0.0
    for u_i, n_i in zip(u_orbitals, occ_nums):
        K_u_i = K_op(u_i)
        E_exchange += 0.5 * n_i * trapz(u_i * K_u_i, r, w)

    return E_kin, E_ext, E_hartree, E_exchange


def _compute_hf_energies_general(
    r: np.ndarray,
    w: np.ndarray,
    v_ext: np.ndarray,
    u_orbitals_by_l: dict[int, list[np.ndarray]],
    occ_by_l: dict[int, list[float]],
    cache: SlaterIntegralCache,
) -> tuple[float, float, float, float]:
    r"""计算通用 HF 能量分量（支持多 l 通道）。

    Parameters
    ----------
    r : np.ndarray
        径向网格
    w : np.ndarray
        积分权重
    v_ext : np.ndarray
        外势
    u_orbitals_by_l : dict[int, list[np.ndarray]]
        按 l 分组的占据态波函数
    occ_by_l : dict[int, list[float]]
        按 l 分组的占据数
    cache : SlaterIntegralCache
        Slater 积分缓存

    Returns
    -------
    tuple[float, float, float, float]
        (E_kinetic, E_ext, E_hartree, E_exchange)
    """
    # 电子密度（径向）
    n_radial = np.zeros_like(r)
    for l_val, u_list in u_orbitals_by_l.items():
        occ_list = occ_by_l[l_val]
        for u_i, n_i in zip(u_list, occ_list):
            n_radial += n_i * u_i**2

    # 1. 外势能
    E_ext = 0.0
    for l_val, u_list in u_orbitals_by_l.items():
        occ_list = occ_by_l[l_val]
        for u_i, n_i in zip(u_list, occ_list):
            E_ext += n_i * trapz(u_i**2 * v_ext, r, w)

    # 2. 动能
    E_kin = 0.0
    for l_val, u_list in u_orbitals_by_l.items():
        occ_list = occ_by_l[l_val]
        for u_i, n_i in zip(u_list, occ_list):
            d2u_dr2 = np.zeros_like(u_i)
            for i in range(1, len(r) - 1):
                dr_left = r[i] - r[i - 1]
                dr_right = r[i + 1] - r[i]
                dr_avg = (dr_left + dr_right) / 2.0
                d2u_dr2[i] = (
                    (u_i[i + 1] - u_i[i]) / dr_right - (u_i[i] - u_i[i - 1]) / dr_left
                ) / dr_avg

            # 动能算符：-1/2 d²/dr² + l(l+1)/(2r²)
            r_safe = np.maximum(r, 1e-12)
            T_u = -0.5 * d2u_dr2 + (l_val * (l_val + 1)) / (2.0 * r_safe**2) * u_i
            E_kin += n_i * trapz(u_i * T_u, r, w)

    # 3. Hartree 能量
    n_3d = n_radial / (4 * np.pi * np.maximum(r**2, 1e-30))
    v_H = v_hartree(n_3d, r, w)
    E_hartree = 0.5 * trapz(n_radial * v_H, r, w)

    # 4. 交换能量（通用，支持多 l）
    E_exchange = 0.0
    for l_val, u_list in u_orbitals_by_l.items():
        occ_list = occ_by_l[l_val]

        # 为当前 l 创建交换算子（包含所有 l' 的耦合）
        K_op = exchange_operator_general(
            r, w, l_val, u_orbitals_by_l, occ_by_l, cache=cache
        )

        # 计算交换能贡献
        for u_i, n_i in zip(u_list, occ_list):
            K_u_i = K_op(u_i)
            E_exchange += 0.5 * n_i * trapz(u_i * K_u_i, r, w)

    return E_kin, E_ext, E_hartree, E_exchange



# ============================================================================
# 通用 HF SCF（支持多 l 通道）
# ============================================================================


@dataclass
class HFSCFGeneralConfig:
    r"""通用 HF SCF 配置（支持 RHF/UHF 和多 l 通道）。

    Attributes
    ----------
    Z : int
        原子序数
    r : np.ndarray
        径向网格
    w : np.ndarray
        积分权重
    occ_by_l : dict[int, list[float]]
        按 l 分组的占据数配置（RHF 模式）
        格式：{l: [n_1, n_2, ...]} 表示该 l 通道的各占据态占据数

        例::

            C (1s² 2s² 2p²) → {0: [2.0, 2.0], 1: [2.0]}
    eigs_per_l : dict[int, int]
        每个 l 通道求解的本征态数量
        例：{0: 2, 1: 1} 表示求解 2 个 s 态和 1 个 p 态
    spin_mode : str, optional
        自旋处理模式，默认 'RHF'
        - 'RHF': 限制性 Hartree-Fock（闭壳层精确，开壳层近似）
        - 'UHF': 非限制性 Hartree-Fock（自旋极化，适用于开壳层）
    occ_by_l_spin : dict[int, dict[str, list[float]]] | None, optional
        自旋分辨占据数配置（UHF 模式）
        格式：{l: {'up': [n_1, ...], 'down': [n_1, ...]}}
        若为 None 且 spin_mode='UHF'，自动从 occ_by_l 均分生成

        例（C ³P 态）::

            {0: {'up': [1.0, 1.0], 'down': [1.0, 1.0]},
             1: {'up': [2.0], 'down': [0.0]}}
    mix_alpha : float
        密度混合参数
    tol : float
        收敛阈值
    maxiter : int
        最大迭代数
    delta : float | None
        变量变换参数（可选）
    Rp : float | None
        变量变换参数（可选）
    """

    Z: int
    r: np.ndarray
    w: np.ndarray
    occ_by_l: dict[int, list[float]]
    eigs_per_l: dict[int, int]
    spin_mode: str = 'RHF'
    occ_by_l_spin: dict[int, dict[str, list[float]]] | None = None
    mix_alpha: float = 0.3
    tol: float = 1e-6
    maxiter: int = 100
    delta: float | None = None
    Rp: float | None = None


@dataclass
class HFSCFGeneralResult:
    r"""通用 HF SCF 结果容器（支持 RHF/UHF）。

    Attributes
    ----------
    converged : bool
        是否收敛
    iterations : int
        实际迭代次数
    eigenvalues_by_l : dict[int, np.ndarray]
        按 l 分组的本征能量（RHF 模式）
    orbitals_by_l : dict[int, list[np.ndarray]]
        按 l 分组的占据态波函数（RHF 模式）
    eigenvalues_by_l_spin : dict[tuple[int, str], np.ndarray] | None
        按 (l, spin) 分组的本征能量（UHF 模式）
    orbitals_by_l_spin : dict[tuple[int, str], list[np.ndarray]] | None
        按 (l, spin) 分组的占据态波函数（UHF 模式）
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
    spin_mode : str
        使用的自旋模式（'RHF' 或 'UHF'）
    """

    converged: bool
    iterations: int
    eigenvalues_by_l: dict[int, np.ndarray]
    orbitals_by_l: dict[int, list[np.ndarray]]
    E_total: float
    E_kinetic: float
    E_ext: float
    E_hartree: float
    E_exchange: float
    spin_mode: str = 'RHF'
    eigenvalues_by_l_spin: dict[tuple[int, str], np.ndarray] | None = None
    orbitals_by_l_spin: dict[tuple[int, str], list[np.ndarray]] | None = None


def run_hf_scf(cfg: HFSCFGeneralConfig) -> HFSCFGeneralResult:
    r"""运行通用 HF SCF 计算（支持 s, p 等多 l 通道）。

    实现按 l 分组的自洽场循环，每个 l 通道独立求解 Fock 方程，
    但通过 Hartree 势和交换算子耦合。

    Parameters
    ----------
    cfg : HFSCFGeneralConfig
        通用 HF SCF 配置

    Returns
    -------
    HFSCFGeneralResult
        收敛的 HF 结果

    Examples
    --------
    碳原子 HF (1s² 2s² 2p²)::

        >>> cfg = HFSCFGeneralConfig(
        ...     Z=6,
        ...     r=r, w=w,
        ...     occ_by_l={0: [2.0, 2.0], 1: [2.0]},  # 1s², 2s², 2p²
        ...     eigs_per_l={0: 2, 1: 1},              # 求解 2 个 s + 1 个 p
        ... )
        >>> res = run_hf_scf(cfg)
    """
    # 分支：UHF 或 RHF
    if cfg.spin_mode == 'UHF':
        return _run_uhf_scf_impl(cfg)
    elif cfg.spin_mode != 'RHF':
        raise ValueError(f"不支持的 spin_mode: {cfg.spin_mode}，仅支持 'RHF' 或 'UHF'")

    # 以下是 RHF 实现
    r = cfg.r
    w = cfg.w
    Z = cfg.Z

    # 外势
    v_ext = -Z / np.maximum(r, 1e-12)

    # 初始化：改进的氢样波函数猜测（使用 Slater 屏蔽估计）
    u_orbitals_by_l = {}
    for l_val, n_occ in cfg.eigs_per_l.items():
        u_orbitals_by_l[l_val] = []
        for idx in range(n_occ):
            n_eff = idx + l_val + 1  # 有效主量子数

            # 使用 Slater 规则估计 Z_eff
            # 标准 Slater 规则：同层 0.35，n-1 层 0.85，更内层 1.00
            if l_val == 0 and idx == 0:
                # 1s: 同层其他电子屏蔽
                Z_eff = Z - 0.30
            elif l_val == 0 and idx == 1:
                # 2s: 1s² 屏蔽 0.85×2，同层（2s+2p）屏蔽 0.35×3
                Z_eff = Z - 0.85 * 2 - 0.35 * 3
            elif l_val == 1:
                # 2p: 1s² 屏蔽 0.85×2，同层（2s+2p）屏蔽 0.35×3
                Z_eff = Z - 0.85 * 2 - 0.35 * 3
            else:
                Z_eff = Z / (n_eff**0.7)  # 通用回退

            u_init = r**l_val * np.exp(-Z_eff * r / n_eff)
            u_init[0] = 0.0
            u_init[-1] = 0.0
            norm = np.sqrt(trapz(u_init**2, r, w))
            u_init /= norm
            u_orbitals_by_l[l_val].append(u_init)

    # Slater 积分缓存
    cache = SlaterIntegralCache()

    # 判断是否使用变换方法
    use_transformed = (cfg.delta is not None) and (cfg.Rp is not None)
    if use_transformed:
        n_boundary_left = 1  # 变换方法去掉 j=0
        n_boundary_right = 0
    else:
        n_boundary_left = 1  # FD2 两端各去 1
        n_boundary_right = 1

    # SCF 主循环
    converged = False
    for iteration in range(cfg.maxiter):
        # 1. 计算总电子密度（所有 l 通道叠加）
        n_radial = np.zeros_like(r)
        for l_val, u_list in u_orbitals_by_l.items():
            occ_list = cfg.occ_by_l[l_val]
            for u_i, n_i in zip(u_list, occ_list):
                n_radial += n_i * u_i**2

        # 转换为 3D 数密度
        n_3d = n_radial / (4 * np.pi * np.maximum(r**2, 1e-30))

        # 2. 计算 Hartree 势（全局，所有 l 共享）
        v_H = v_hartree(n_3d, r, w)

        # 3. 为每个 l 通道构造并对角化 Fock 矩阵
        u_new_by_l = {}
        eigs_by_l = {}

        for l_val, n_eigs in cfg.eigs_per_l.items():
            # 构造 Fock 矩阵
            if use_transformed:
                H_loc, B, r_inner = build_transformed_hamiltonian(
                    r, l_val, v_ext + v_H, cfg.delta, cfg.Rp
                )
                n_inner = len(r_inner)
            else:
                H_loc, r_inner = radial_hamiltonian_matrix(r, l_val, v_ext + v_H)
                n_inner = len(r_inner)
                B = None

            # 构造交换矩阵（使用通用交换算子）
            K_op = exchange_operator_general(
                r, w, l_val, u_orbitals_by_l, cfg.occ_by_l, cache=cache
            )

            K_matrix = np.zeros((n_inner, n_inner))

            if use_transformed:
                # 变换方法：u(j) = v(j) * exp(jδ/2)
                j_vals = np.arange(1, len(r))
                transform = np.exp(j_vals * cfg.delta / 2.0)

                K_v_basis = []
                for j in range(n_inner):
                    v_j_inner = np.zeros(n_inner)
                    v_j_inner[j] = 1.0

                    u_j_full = np.zeros(len(r))
                    u_j_full[0] = 0.0
                    u_j_full[1:] = v_j_inner * transform

                    K_u_j_full = K_op(u_j_full)
                    K_v_j_inner = K_u_j_full[1:] / transform
                    K_v_basis.append(K_v_j_inner)

                for i in range(n_inner):
                    for j in range(n_inner):
                        K_matrix[i, j] = K_v_basis[j][i] * B[i, i]
            else:
                # 标准 FD2 方法
                for j in range(n_inner):
                    e_j_full = np.zeros(len(r))
                    e_j_full[j + n_boundary_left] = 1.0
                    K_e_j_full = K_op(e_j_full)
                    K_matrix[:, j] = K_e_j_full[
                        n_boundary_left : len(r) - n_boundary_right
                    ]

            # 对称化
            K_matrix = 0.5 * (K_matrix + K_matrix.T)
            F_matrix = H_loc + K_matrix
            F_matrix = 0.5 * (F_matrix + F_matrix.T)

            # 对角化
            if use_transformed:
                from scipy.linalg import eigh
                eigs, vecs_v = eigh(F_matrix, B)
            else:
                eigs, vecs_v = np.linalg.eigh(F_matrix)

            # 提取占据态
            u_new_list = []
            for i_occ in range(n_eigs):
                v_inner = vecs_v[:, i_occ]

                if use_transformed:
                    u_inner = v_inner * transform
                    u_full = np.zeros(len(r))
                    u_full[0] = 0.0
                    u_full[1:] = u_inner
                else:
                    u_full = np.zeros(len(r))
                    u_full[n_boundary_left : len(r) - n_boundary_right] = v_inner

                norm = np.sqrt(trapz(u_full**2, r, w))
                u_full /= norm
                u_new_list.append(u_full)

            u_new_by_l[l_val] = u_new_list
            eigs_by_l[l_val] = eigs[:n_eigs]

        # 4. 检查收敛（所有 l 通道的 RMS 差）
        rms_diff_total = 0.0
        n_orbitals_total = 0

        for l_val in cfg.eigs_per_l.keys():
            for u_new, u_old in zip(u_new_by_l[l_val], u_orbitals_by_l[l_val]):
                rms_diff_total += np.sqrt(np.mean((u_new - u_old) ** 2))
                n_orbitals_total += 1

        rms_diff_avg = rms_diff_total / n_orbitals_total

        if rms_diff_avg < cfg.tol:
            converged = True
            u_orbitals_by_l = u_new_by_l
            break

        # 5. 密度混合
        for l_val in cfg.eigs_per_l.keys():
            u_orbitals_by_l[l_val] = [
                cfg.mix_alpha * u_new + (1 - cfg.mix_alpha) * u_old
                for u_new, u_old in zip(u_new_by_l[l_val], u_orbitals_by_l[l_val])
            ]

            # 重归一化
            for i in range(len(u_orbitals_by_l[l_val])):
                norm = np.sqrt(trapz(u_orbitals_by_l[l_val][i] ** 2, r, w))
                u_orbitals_by_l[l_val][i] /= norm

    # 计算最终能量（使用通用能量计算）
    E_kin, E_ext, E_H, E_x = _compute_hf_energies_general(
        r, w, v_ext, u_orbitals_by_l, cfg.occ_by_l, cache
    )
    E_total = E_kin + E_ext + E_H + E_x

    return HFSCFGeneralResult(
        converged=converged,
        iterations=iteration + 1 if converged else cfg.maxiter,
        eigenvalues_by_l=eigs_by_l,
        orbitals_by_l=u_orbitals_by_l,
        E_total=E_total,
        E_kinetic=E_kin,
        E_ext=E_ext,
        E_hartree=E_H,
        E_exchange=E_x,
    )

# ============================================================================
# UHF 实现（自旋极化）
# ============================================================================


def _compute_hf_energies_general_spin(
    r: np.ndarray,
    w: np.ndarray,
    v_ext: np.ndarray,
    u_orbitals_by_l_spin: dict[tuple[int, str], list[np.ndarray]],
    occ_by_l_spin: dict[int, dict[str, list[float]]],
    cache,
) -> dict[str, float]:
    """计算自旋分辨的 HF 能量分解（UHF）。
    
    Parameters
    ----------
    r : np.ndarray
        径向网格
    w : np.ndarray
        积分权重
    v_ext : np.ndarray
        外势
    u_orbitals_by_l_spin : dict[tuple[int, str], list[np.ndarray]]
        自旋分辨轨道，按 (l, spin) 索引
    occ_by_l_spin : dict[int, dict[str, list[float]]]
        自旋分辨占据数，{l: {'up': [...], 'down': [...]}}
    cache : SlaterIntegralCache
        Slater 积分缓存
    
    Returns
    -------
    dict[str, float]
        能量分解：{'E_total', 'E_kin', 'E_ext', 'E_H', 'E_x'}
    """
    from .hf.exchange import exchange_operator_general_spin
    from .hartree import v_hartree
    from .utils import trapz
    
    # 1. 构造总径向密度
    n_radial = np.zeros_like(r)
    for (l_val, spin), u_list in u_orbitals_by_l_spin.items():
        occ_list = occ_by_l_spin[l_val][spin]
        for u_i, n_i in zip(u_list, occ_list):
            n_radial += n_i * u_i**2
    
    # 2. 动能（带离心项）
    E_kin = 0.0
    for (l_val, spin), u_list in u_orbitals_by_l_spin.items():
        occ_list = occ_by_l_spin[l_val][spin]
        for u_i, n_i in zip(u_list, occ_list):
            d2u_dr2 = np.zeros_like(u_i)
            for i in range(1, len(r) - 1):
                dr_left = r[i] - r[i - 1]
                dr_right = r[i + 1] - r[i]
                dr_avg = (dr_left + dr_right) / 2.0
                d2u_dr2[i] = (
                    (u_i[i + 1] - u_i[i]) / dr_right - (u_i[i] - u_i[i - 1]) / dr_left
                ) / dr_avg
            
            # 完整径向动能算符
            r_safe = np.maximum(r, 1e-12)
            T_u = -0.5 * d2u_dr2 + (l_val * (l_val + 1)) / (2.0 * r_safe**2) * u_i
            E_kin += n_i * trapz(u_i * T_u, r, w)
    
    # 3. Hartree 能量（用总密度）
    n_3d = n_radial / (4 * np.pi * np.maximum(r**2, 1e-30))
    v_H = v_hartree(n_3d, r, w)
    E_hartree = 0.5 * trapz(n_radial * v_H, r, w)
    
    # 4. 交换能量（仅同自旋耦合）
    E_exchange = 0.0
    
    # 准备 UHF 格式的占据态字典
    u_occ_by_l_spin_dict = {
        (l, s): u_orbitals_by_l_spin[(l, s)]
        for (l, s) in u_orbitals_by_l_spin.keys()
    }
    occ_nums_by_l_spin_dict = {
        (l, s): occ_by_l_spin[l][s]
        for l in occ_by_l_spin.keys()
        for s in ['up', 'down']
    }
    
    for (l_val, spin), u_list in u_orbitals_by_l_spin.items():
        occ_list = occ_by_l_spin[l_val][spin]
        
        # 创建该 (l, spin) 的交换算子
        K_op = exchange_operator_general_spin(
            r, w, l_val, spin,
            u_occ_by_l_spin_dict,
            occ_nums_by_l_spin_dict,
            cache=cache
        )
        
        # 计算交换能贡献
        for u_i, n_i in zip(u_list, occ_list):
            K_u_i = K_op(u_i)
            E_exchange += 0.5 * n_i * trapz(u_i * K_u_i, r, w)
    
    # 5. 外势能
    E_ext = trapz(n_radial * v_ext, r, w)
    
    # 6. 总能量
    E_total = E_kin + E_ext + E_hartree + E_exchange

    return {
        'E_total': E_total,
        'E_kin': E_kin,
        'E_ext': E_ext,
        'E_H': E_hartree,
        'E_x': E_exchange,
    }


def _run_uhf_scf_impl(cfg: HFSCFGeneralConfig) -> HFSCFGeneralResult:
    """UHF SCF 实现（自旋极化）。

    核心逻辑：
    1. 自旋分辨的初始猜测
    2. SCF 循环（两套独立 Fock 矩阵）
    3. 自旋分辨能量计算
    """
    from .hf.exchange import exchange_operator_general_spin, SlaterIntegralCache
    from .hf.slater import slater_integral_radial
    from .operator import build_transformed_hamiltonian, radial_hamiltonian_matrix
    from .hartree import v_hartree
    from .utils import trapz

    r, w, Z = cfg.r, cfg.w, cfg.Z
    v_ext = -Z / np.maximum(r, 1e-12)
    use_transformed = (cfg.delta is not None and cfg.Rp is not None)

    # 步骤 1: 准备自旋分辨占据数
    if cfg.occ_by_l_spin is None:
        # 从 occ_by_l 自动均分生成（闭壳层近似）
        occ_by_l_spin = {}
        for l_val, occ_list in cfg.occ_by_l.items():
            occ_by_l_spin[l_val] = {
                'up': [n / 2.0 for n in occ_list],
                'down': [n / 2.0 for n in occ_list],
            }
    else:
        occ_by_l_spin = cfg.occ_by_l_spin

    # 步骤 2: 初始化自旋分辨轨道
    u_orbitals_by_l_spin = {}
    for l_val, n_occ in cfg.eigs_per_l.items():
        for spin in ['up', 'down']:
            u_orbitals_by_l_spin[(l_val, spin)] = []
            for idx in range(n_occ):
                n_eff = idx + l_val + 1
                # Slater 屏蔽
                if l_val == 0 and idx == 0:
                    Z_eff = Z - 0.30
                elif l_val == 0 and idx == 1:
                    Z_eff = Z - 0.85 * 2 - 0.35 * 3
                elif l_val == 1:
                    Z_eff = Z - 0.85 * 2 - 0.35 * 3
                else:
                    Z_eff = Z / (n_eff**0.7)

                u_init = r**l_val * np.exp(-Z_eff * r / n_eff)
                u_init[0] = 0.0
                u_init[-1] = 0.0
                norm = np.sqrt(trapz(u_init**2, r, w))
                u_init /= norm
                u_orbitals_by_l_spin[(l_val, spin)].append(u_init)

    cache = SlaterIntegralCache()

    # 步骤 3: SCF 循环
    converged = False
    for iteration in range(cfg.maxiter):
        # 3.1 构造总径向密度（两个自旋求和）
        n_radial = np.zeros_like(r)
        for (l_val, spin), u_list in u_orbitals_by_l_spin.items():
            occ_list = occ_by_l_spin[l_val][spin]
            for u_i, n_i in zip(u_list, occ_list):
                n_radial += n_i * u_i**2

        # 3.2 计算 Hartree 势（用总密度）
        n_3d = n_radial / (4 * np.pi * np.maximum(r**2, 1e-30))
        v_H = v_hartree(n_3d, r, w)

        # 3.3 对每个 (l, spin) 通道求解 Fock 方程
        new_orbitals = {}
        new_eigenvalues = {}
        max_delta = 0.0

        for l_val in cfg.eigs_per_l.keys():
            for spin in ['up', 'down']:
                key = (l_val, spin)
                n_occ = cfg.eigs_per_l[l_val]

                # 3.3.1 准备占据态字典（UHF 格式）
                u_occ_by_l_spin_dict = {
                    (l, s): u_orbitals_by_l_spin[(l, s)]
                    for (l, s) in u_orbitals_by_l_spin.keys()
                }
                occ_nums_by_l_spin_dict = {
                    (l, s): occ_by_l_spin[l][s]
                    for l in occ_by_l_spin.keys()
                    for s in ['up', 'down']
                }

                # 3.3.2 构造交换算子（仅同自旋）
                K_op = exchange_operator_general_spin(
                    r, w, l_val, spin,
                    u_occ_by_l_spin_dict,
                    occ_nums_by_l_spin_dict,
                    cache=cache
                )

                # 3.3.3 构造 Fock 矩阵
                if use_transformed:
                    H_loc, B, r_inner = build_transformed_hamiltonian(
                        r, l_val, v_ext + v_H, cfg.delta, cfg.Rp
                    )
                    n_inner = len(r_inner)
                    K_matrix = np.zeros((n_inner, n_inner))

                    # 计算交换矩阵元（在 v 空间）
                    j_vals = np.arange(1, len(r))
                    transform = np.exp(j_vals * cfg.delta / 2.0)

                    K_v_basis = []
                    for j in range(n_inner):
                        v_j_inner = np.zeros(n_inner)
                        v_j_inner[j] = 1.0

                        u_j_full = np.zeros(len(r))
                        u_j_full[0] = 0.0
                        u_j_full[1:] = v_j_inner * transform

                        K_u_j_full = K_op(u_j_full)
                        K_v_j_inner = K_u_j_full[1:] / transform
                        K_v_basis.append(K_v_j_inner)

                    for i in range(n_inner):
                        for j in range(n_inner):
                            K_matrix[i, j] = K_v_basis[j][i] * B[i, i]

                    K_matrix = 0.5 * (K_matrix + K_matrix.T)
                    F_matrix = H_loc + K_matrix
                    F_matrix = 0.5 * (F_matrix + F_matrix.T)

                    from scipy.linalg import eigh
                    eigs, vecs_v = eigh(F_matrix, B)

                    # 转换回 u 空间
                    new_orbitals[key] = []
                    for i_eig in range(n_occ):
                        v_inner = vecs_v[:, i_eig]
                        u_inner = v_inner * transform
                        u_full = np.zeros(len(r))
                        u_full[0] = 0.0
                        u_full[1:] = u_inner
                        norm = np.sqrt(trapz(u_full**2, r, w))
                        u_full /= norm
                        new_orbitals[key].append(u_full)
                else:
                    # 标准 FD2 方法（使用 operator 模块）
                    H_loc, r_inner = radial_hamiltonian_matrix(r, l_val, v_ext + v_H)
                    n_inner = len(r_inner)
                    n_boundary_left = 1
                    n_boundary_right = 1

                    K_matrix = np.zeros((n_inner, n_inner))

                    # 交换矩阵
                    for j in range(n_inner):
                        e_j_full = np.zeros(len(r))
                        e_j_full[j + n_boundary_left] = 1.0
                        K_e_j_full = K_op(e_j_full)
                        K_matrix[:, j] = K_e_j_full[
                            n_boundary_left : len(r) - n_boundary_right
                        ]

                    K_matrix = 0.5 * (K_matrix + K_matrix.T)
                    F_matrix = H_loc + K_matrix
                    F_matrix = 0.5 * (F_matrix + F_matrix.T)

                    eigs, vecs_v = np.linalg.eigh(F_matrix)

                    new_orbitals[key] = []
                    for i_eig in range(n_occ):
                        v_inner = vecs_v[:, i_eig]
                        u_full = np.zeros(len(r))
                        u_full[n_boundary_left : len(r) - n_boundary_right] = v_inner
                        norm = np.sqrt(trapz(u_full**2, r, w))
                        u_full /= norm
                        new_orbitals[key].append(u_full)

                new_eigenvalues[key] = eigs[:n_occ]

                # 3.3.4 计算变化量
                for i in range(n_occ):
                    old_u = u_orbitals_by_l_spin[key][i]
                    new_u = new_orbitals[key][i]
                    delta = trapz((old_u - new_u)**2, r, w)
                    max_delta = max(max_delta, delta)

        # 3.4 更新轨道（密度混合 + 重归一化）
        for key in new_orbitals.keys():
            for i in range(len(new_orbitals[key])):
                old_u = u_orbitals_by_l_spin[key][i]
                new_u = new_orbitals[key][i]
                u_orbitals_by_l_spin[key][i] = (
                    cfg.mix_alpha * new_u + (1 - cfg.mix_alpha) * old_u
                )

            # 重归一化（必须在混合后进行）
            for i in range(len(u_orbitals_by_l_spin[key])):
                norm = np.sqrt(trapz(u_orbitals_by_l_spin[key][i]**2, r, w))
                u_orbitals_by_l_spin[key][i] /= norm

        # 3.5 收敛检查
        if max_delta < cfg.tol:
            converged = True
            break

    # 步骤 4: 计算能量
    energies = _compute_hf_energies_general_spin(
        r, w, v_ext,
        u_orbitals_by_l_spin,
        occ_by_l_spin,
        cache
    )

    # 步骤 5: 返回结果
    # 为兼容性，也构造 RHF 格式的 orbitals_by_l（合并自旋）
    orbitals_by_l = {}
    eigenvalues_by_l = {}
    for l_val in cfg.eigs_per_l.keys():
        orbitals_by_l[l_val] = u_orbitals_by_l_spin[(l_val, 'up')]
        eigenvalues_by_l[l_val] = new_eigenvalues[(l_val, 'up')]

    return HFSCFGeneralResult(
        converged=converged,
        iterations=iteration + 1,
        eigenvalues_by_l=eigenvalues_by_l,
        orbitals_by_l=orbitals_by_l,
        E_total=energies['E_total'],
        E_kinetic=energies['E_kin'],
        E_ext=energies['E_ext'],
        E_hartree=energies['E_H'],
        E_exchange=energies['E_x'],
        spin_mode='UHF',
        eigenvalues_by_l_spin=new_eigenvalues,
        orbitals_by_l_spin=u_orbitals_by_l_spin,
    )
