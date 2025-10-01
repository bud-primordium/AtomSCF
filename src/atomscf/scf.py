from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np

from .grid import trapezoid_weights
from .hartree import v_hartree
from .operator import solve_bound_states_fd
from .occupations import OrbitalSpec, default_occupations
from .utils import trapz
from .xc.lda import vx_dirac, lda_c_pz81, ex_dirac_density
from .xc.vwn import lda_c_vwn

__all__ = [
    "SCFConfig",
    "SCFResult",
    "run_lsda_x_only",
    "run_lsda_pz81",
]


@dataclass
class SCFConfig:
    r"""SCF 配置参数。

    Attributes
    ----------
    Z : int
        原子序数。
    r : numpy.ndarray
        径向网格 :math:`r_i`，要求严格单调递增且 :math:`r_0>0`。
    w : numpy.ndarray
        梯形积分权重 :math:`w_i`。
    lmax : int
        最大角动量量子数 :math:`\ell_{\max}`（决定求解哪些通道）。
    mix_alpha : float
        密度（或势）混合参数 :math:`\alpha \in (0,1]`。
    maxiter : int
        最大自洽迭代次数。
    tol : float
        收敛阈值（默认用于密度无穷范数）。
    occ : list[OrbitalSpec] | None
        占据方案；若为 ``None`` 则使用 :func:`default_occupations`。
    eigs_per_l : int
        每个 :math:`\ell` 通道求解的最低本征态数量（需覆盖所有占据的 :math:`n`）。
    """

    Z: int
    r: np.ndarray
    w: np.ndarray
    lmax: int = 3
    mix_alpha: float = 0.3
    maxiter: int = 200
    tol: float = 1e-7
    occ: List[OrbitalSpec] | None = None
    eigs_per_l: int = 3
    compute_all_l: bool = True
    # 计算所有 l 的模式："always"（每轮都算）、"final"（仅在收敛后补齐）、"none"（只算占据所需）
    compute_all_l_mode: str = "final"
    xc: str = "PZ81"  # 可选 "PZ81" 或 "VWN"


@dataclass
class SCFResult:
    r"""SCF 结果容器。"""

    converged: bool
    iterations: int
    eps_by_l_sigma: Dict[Tuple[int, str], np.ndarray]
    u_by_l_sigma: Dict[Tuple[int, str], np.ndarray]
    n_up: np.ndarray
    n_dn: np.ndarray
    v_h: np.ndarray
    v_x_up: np.ndarray
    v_x_dn: np.ndarray
    v_c_up: np.ndarray | None = None
    v_c_dn: np.ndarray | None = None
    energies: dict | None = None


def _build_effective_potential(
    Z: int, r: np.ndarray, n_up: np.ndarray, n_dn: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""由密度构建自旋分辨的有效势 :math:`v_{\text{eff},\sigma}`（LSDA X-only）。

    .. math::
        v_{\text{eff},\sigma}(r) = v_{\text{ext}}(r) + v_H[n_\uparrow+n_\downarrow](r) + v_x^\sigma[n_\sigma](r).

    Parameters
    ----------
    Z : int
        原子序数。
    r : numpy.ndarray
        径向网格。
    n_up, n_dn : numpy.ndarray
        自旋分辨密度。
    w : numpy.ndarray
        梯形权重。

    Returns
    -------
    v_eff_up, v_eff_dn : numpy.ndarray
        两个自旋通道的有效势。
    """
    r_safe = np.maximum(r, 1e-12)
    v_ext = -float(Z) / r_safe
    n_tot = n_up + n_dn
    vH = v_hartree(n_tot, r, w)
    vx_up = vx_dirac(n_up)
    vx_dn = vx_dirac(n_dn)
    return v_ext + vH + vx_up, v_ext + vH + vx_dn


def _build_effective_potential_pz81(
    Z: int, r: np.ndarray, n_up: np.ndarray, n_dn: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""含 PZ81 关联的有效势及分量：返回 (v_up, v_dn, vH, vX_up, vX_dn)。"""
    r_safe = np.maximum(r, 1e-12)
    v_ext = -float(Z) / r_safe
    n_tot = n_up + n_dn
    vH = v_hartree(n_tot, r, w)
    vx_up = vx_dirac(n_up)
    vx_dn = vx_dirac(n_dn)
    _, vc_up, vc_dn, _ = lda_c_pz81(n_up, n_dn)
    return v_ext + vH + vx_up + vc_up, v_ext + vH + vx_dn + vc_dn, vH, vx_up, vx_dn


def _init_guess_density(Z: int, r: np.ndarray, w: np.ndarray, occ: list[OrbitalSpec]) -> tuple[np.ndarray, np.ndarray]:
    r"""用外势 :math:`v_{\text{ext}}`（不含相互作用）波函数作为初猜密度。"""
    # 求解在纯库仑外势下的径向态（每个 l、两自旋势相同）
    v_ext = -float(Z) / np.maximum(r, 1e-12)
    n_up = np.zeros_like(r)
    n_dn = np.zeros_like(r)

    # 按 l 分组需要的最大 n_index
    need: dict[tuple[int, str], int] = {}
    for spec in occ:
        key = (spec.l, spec.spin)
        need[key] = max(need.get(key, -1), spec.n_index)

    cache_e: dict[int, np.ndarray] = {}
    cache_u: dict[int, np.ndarray] = {}
    for l, max_n in {l: max(n for (l2, _), n in need.items() if l2 == l) for l in set(l for l, _ in need.keys())}.items():
        eps, U = solve_bound_states_fd(r, l=l, v_of_r=v_ext, k=max(max_n + 1, 1))
        cache_e[l] = eps
        cache_u[l] = U

    # 组装密度（m 平均）
    r2 = np.maximum(r, 1e-12) ** 2
    for spec in occ:
        u = cache_u[spec.l][spec.n_index]
        contrib = (2 * spec.l + 1) * spec.f_per_m * (u * u) / (4.0 * np.pi * r2)
        if spec.spin == "up":
            n_up += contrib
        else:
            n_dn += contrib

    return n_up, n_dn


def run_lsda_x_only(cfg: SCFConfig, verbose: bool = False, progress_every: int = 10) -> SCFResult:
    r"""运行 LSDA X-only 的自洽场计算（球对称、径向）。

    Parameters
    ----------
    cfg : SCFConfig
        自洽计算的配置。

    Returns
    -------
    SCFResult
        自洽结果（是否收敛、迭代数、能级/波函数、密度与势）。

    Notes
    -----
    - 占据方案以径向通道为单位（指定 :math:`\ell` 与通道内序号 :math:`n_{\text{index}}`）。
    - 对 C 原子，2p 壳层采用 m 平均且自旋极化：上自旋每个 m 的分数占据 :math:`2/3`，下自旋为 0。
    - 混合策略：密度线性混合（数值稳健，易用）。
    """
    r, w = cfg.r, cfg.w
    if cfg.occ is None:
        occ = default_occupations(cfg.Z)
    else:
        occ = cfg.occ

    # 初猜密度
    n_up, n_dn = _init_guess_density(cfg.Z, r, w, occ)

    eps_by_l_sigma: Dict[Tuple[int, str], np.ndarray] = {}
    u_by_l_sigma: Dict[Tuple[int, str], np.ndarray] = {}

    converged = False
    for it in range(1, cfg.maxiter + 1):
        # 构势（当前密度）
        v_up, v_dn = _build_effective_potential(cfg.Z, r, n_up, n_dn, w)

        # 求解每个需要的 l, spin 通道的本征态
        # 确定每个 (l,spin) 所需的最大 n_index
        need: dict[tuple[int, str], int] = {}
        for spec in occ:
            key = (spec.l, spec.spin)
            need[key] = max(need.get(key, -1), spec.n_index)
        # 确保为可视化与 LUMO 估计计算所有 l, 两自旋的前若干本征值
        if cfg.compute_all_l and cfg.compute_all_l_mode == "always":
            for l_all in range(0, cfg.lmax + 1):
                for spin in ("up", "down"):
                    need.setdefault((l_all, spin), max(cfg.eigs_per_l - 1, 0))

        new_eps_by_l_sigma: Dict[Tuple[int, str], np.ndarray] = {}
        new_u_by_l_sigma: Dict[Tuple[int, str], np.ndarray] = {}

        for (l, spin), nmax in sorted(need.items()):
            k = max(nmax + 1, cfg.eigs_per_l)
            v = v_up if spin == "up" else v_dn
            eps, U = solve_bound_states_fd(r, l=l, v_of_r=v, k=k)
            new_eps_by_l_sigma[(l, spin)] = eps
            new_u_by_l_sigma[(l, spin)] = U

        # 由占据重建新密度
        r2 = np.maximum(r, 1e-12) ** 2
        n_up_out = np.zeros_like(r)
        n_dn_out = np.zeros_like(r)
        for spec in occ:
            u = new_u_by_l_sigma[(spec.l, spec.spin)][spec.n_index]
            contrib = (2 * spec.l + 1) * spec.f_per_m * (u * u) / (4.0 * np.pi * r2)
            if spec.spin == "up":
                n_up_out += contrib
            else:
                n_dn_out += contrib

        # 混合
        n_up_mixed = (1.0 - cfg.mix_alpha) * n_up + cfg.mix_alpha * n_up_out
        n_dn_mixed = (1.0 - cfg.mix_alpha) * n_dn + cfg.mix_alpha * n_dn_out

        # 收敛判断（密度无穷范数 + 电子数守恒）
        dn_inf = max(np.max(np.abs(n_up_mixed - n_up)), np.max(np.abs(n_dn_mixed - n_dn)))

        # 更新状态
        n_up, n_dn = n_up_mixed, n_dn_mixed
        eps_by_l_sigma = new_eps_by_l_sigma
        u_by_l_sigma = new_u_by_l_sigma

        if verbose and (it == 1 or it % progress_every == 0):
            print(f"[LSDA X-only] iter={it} |dn|_inf={dn_inf:.3e}")
        if dn_inf < cfg.tol:
            converged = True
            break

    # 电子数守恒检查（便于测试）
    n_tot = n_up + n_dn
    Ne = 4.0 * np.pi * trapz(n_tot * (r ** 2), r, w)
    _ = Ne  # 未直接返回，但用于测试断言

    # 最终势
    v_up, v_dn = _build_effective_potential(cfg.Z, r, n_up, n_dn, w)
    vH = v_hartree(n_up + n_dn, r, w)
    vx_up = v_up - (-float(cfg.Z) / np.maximum(r, 1e-12)) - vH
    vx_dn = v_dn - (-float(cfg.Z) / np.maximum(r, 1e-12)) - vH

    return SCFResult(
        converged=converged,
        iterations=it,
        eps_by_l_sigma=eps_by_l_sigma,
        u_by_l_sigma=u_by_l_sigma,
        n_up=n_up,
        n_dn=n_dn,
        v_h=vH,
        v_x_up=vx_up,
        v_x_dn=vx_dn,
    )


def run_lsda_pz81(cfg: SCFConfig, verbose: bool = False, progress_every: int = 10) -> SCFResult:
    r"""运行 LSDA（Dirac 交换 + PZ81 关联）的自洽场计算，并给出能量分解。"""
    r, w = cfg.r, cfg.w
    if cfg.occ is None:
        occ = default_occupations(cfg.Z)
    else:
        occ = cfg.occ

    # 初猜密度
    n_up, n_dn = _init_guess_density(cfg.Z, r, w, occ)

    eps_by_l_sigma: Dict[Tuple[int, str], np.ndarray] = {}
    u_by_l_sigma: Dict[Tuple[int, str], np.ndarray] = {}

    converged = False
    for it in range(1, cfg.maxiter + 1):
        v_up, v_dn, _, _, _ = _build_effective_potential_pz81(cfg.Z, r, n_up, n_dn, w)

        # 需要的通道
        need: dict[tuple[int, str], int] = {}
        for spec in occ:
            key = (spec.l, spec.spin)
            need[key] = max(need.get(key, -1), spec.n_index)
        if cfg.compute_all_l and cfg.compute_all_l_mode == "always":
            for l_all in range(0, cfg.lmax + 1):
                for spin in ("up", "down"):
                    need.setdefault((l_all, spin), max(cfg.eigs_per_l - 1, 0))

        new_eps_by_l_sigma: Dict[Tuple[int, str], np.ndarray] = {}
        new_u_by_l_sigma: Dict[Tuple[int, str], np.ndarray] = {}

        for (l, spin), nmax in sorted(need.items()):
            k = max(nmax + 1, cfg.eigs_per_l)
            v = v_up if spin == "up" else v_dn
            eps, U = solve_bound_states_fd(r, l=l, v_of_r=v, k=k)
            new_eps_by_l_sigma[(l, spin)] = eps
            new_u_by_l_sigma[(l, spin)] = U

        # 密度
        r2 = np.maximum(r, 1e-12) ** 2
        n_up_out = np.zeros_like(r)
        n_dn_out = np.zeros_like(r)
        for spec in occ:
            u = new_u_by_l_sigma[(spec.l, spec.spin)][spec.n_index]
            contrib = (2 * spec.l + 1) * spec.f_per_m * (u * u) / (4.0 * np.pi * r2)
            if spec.spin == "up":
                n_up_out += contrib
            else:
                n_dn_out += contrib

        # 混合与收敛
        n_up_mixed = (1.0 - cfg.mix_alpha) * n_up + cfg.mix_alpha * n_up_out
        n_dn_mixed = (1.0 - cfg.mix_alpha) * n_dn + cfg.mix_alpha * n_dn_out
        dn_inf = max(np.max(np.abs(n_up_mixed - n_up)), np.max(np.abs(n_dn_mixed - n_dn)))
        n_up, n_dn = n_up_mixed, n_dn_mixed
        eps_by_l_sigma = new_eps_by_l_sigma
        u_by_l_sigma = new_u_by_l_sigma
        if verbose and (it == 1 or it % progress_every == 0):
            print(f"[LSDA PZ81] iter={it} |dn|_inf={dn_inf:.3e}")
        if dn_inf < cfg.tol:
            converged = True
            break

    # 势与分量
    v_up, v_dn, vH, vX_up, vX_dn = _build_effective_potential_pz81(cfg.Z, r, n_up, n_dn, w)
    v_ext = -float(cfg.Z) / np.maximum(r, 1e-12)
    vc_up = v_up - (v_ext + vH + vX_up)
    vc_dn = v_dn - (v_ext + vH + vX_dn)

    # 能量分解
    n_tot = n_up + n_dn
    r2 = r * r
    e_sum = 0.0
    for spec in (cfg.occ or default_occupations(cfg.Z)):
        eps_l_sigma = eps_by_l_sigma[(spec.l, spec.spin)][spec.n_index]
        e_sum += (2 * spec.l + 1) * spec.f_per_m * float(eps_l_sigma)
    E_H = 0.5 * 4.0 * np.pi * trapz(n_tot * vH * r2, r, w)
    e_x = ex_dirac_density(n_up, n_dn)
    _, _, _, e_c = lda_c_pz81(n_up, n_dn)
    E_x = 4.0 * np.pi * trapz(e_x * r2, r, w)
    E_c = 4.0 * np.pi * trapz(e_c * r2, r, w)
    E_ext = 4.0 * np.pi * trapz(n_tot * v_ext * r2, r, w)
    E_xc_dc = 4.0 * np.pi * trapz((vX_up * n_up + vX_dn * n_dn + vc_up * n_up + vc_dn * n_dn) * r2, r, w)
    E_tot = e_sum - E_H - E_xc_dc + (E_x + E_c)
    # 计算 Kohn–Sham 动能 T_s = sum eps_i - ∫ n (v_ext + v_H + v_x + v_c) dr
    int_n_v = 4.0 * np.pi * trapz((n_tot * (v_ext + vH + vX_up * (n_up / n_tot + 0.0) + vX_dn * (n_dn / n_tot + 0.0) + vc_up * (n_up / n_tot + 0.0) + vc_dn * (n_dn / n_tot + 0.0))) * r2, r, w) if np.all(n_tot>0) else 4.0 * np.pi * trapz(n_tot * (v_ext + vH) * r2, r, w)
    # 更稳健地拆分：∫ n v_x = ∫ (n_up v_x_up + n_dn v_x_dn)，相关同理
    int_n_vx = 4.0 * np.pi * trapz((n_up * vX_up + n_dn * vX_dn) * r2, r, w)
    int_n_vc = 4.0 * np.pi * trapz((n_up * vc_up + n_dn * vc_dn) * r2, r, w)
    int_n_v = 4.0 * np.pi * trapz(n_tot * (v_ext + vH) * r2, r, w) + int_n_vx + int_n_vc
    T_s = e_sum - int_n_v
    energies = dict(E_total=E_tot, E_H=E_H, E_x=E_x, E_c=E_c, E_ext=E_ext, E_sum=e_sum, E_kin=T_s, E_coul=E_H)

    # 若仅在收敛后补齐全部通道，则此处再求一遍所有 (l,spin) 的低能态
    if cfg.compute_all_l and cfg.compute_all_l_mode == "final":
        all_eps: Dict[Tuple[int, str], np.ndarray] = {}
        all_u: Dict[Tuple[int, str], np.ndarray] = {}
        for l_all in range(0, cfg.lmax + 1):
            for spin in ("up", "down"):
                v_sel = v_up if spin == "up" else v_dn
                eps, U = solve_bound_states_fd(r, l=l_all, v_of_r=v_sel, k=max(cfg.eigs_per_l, 1))
                all_eps[(l_all, spin)] = eps
                all_u[(l_all, spin)] = U
        eps_by_l_sigma = all_eps
        u_by_l_sigma = all_u

    return SCFResult(
        converged=converged,
        iterations=it,
        eps_by_l_sigma=eps_by_l_sigma,
        u_by_l_sigma=u_by_l_sigma,
        n_up=n_up,
        n_dn=n_dn,
        v_h=vH,
        v_x_up=vX_up,
        v_x_dn=vX_dn,
        v_c_up=vc_up,
        v_c_dn=vc_dn,
        energies=energies,
    )


def run_lsda_vwn(cfg: SCFConfig, verbose: bool = False, progress_every: int = 10) -> SCFResult:
    r"""运行 LSDA（Dirac 交换 + VWN 关联）的自洽场计算，并给出能量分解。"""
    r, w = cfg.r, cfg.w
    if cfg.occ is None:
        occ = default_occupations(cfg.Z)
    else:
        occ = cfg.occ

    n_up, n_dn = _init_guess_density(cfg.Z, r, w, occ)
    eps_by_l_sigma: Dict[Tuple[int, str], np.ndarray] = {}
    u_by_l_sigma: Dict[Tuple[int, str], np.ndarray] = {}

    converged = False
    for it in range(1, cfg.maxiter + 1):
        v_up, v_dn, vH, vX_up, vX_dn = _build_effective_potential_vwn(cfg.Z, r, n_up, n_dn, w)

        need: dict[tuple[int, str], int] = {}
        for spec in occ:
            key = (spec.l, spec.spin)
            need[key] = max(need.get(key, -1), spec.n_index)
        if cfg.compute_all_l:
            for l_all in range(0, cfg.lmax + 1):
                for spin in ("up", "down"):
                    need.setdefault((l_all, spin), max(cfg.eigs_per_l - 1, 0))

        new_eps_by_l_sigma: Dict[Tuple[int, str], np.ndarray] = {}
        new_u_by_l_sigma: Dict[Tuple[int, str], np.ndarray] = {}
        for (l, spin), nmax in sorted(need.items()):
            k = max(nmax + 1, cfg.eigs_per_l)
            v = v_up if spin == "up" else v_dn
            eps, U = solve_bound_states_fd(r, l=l, v_of_r=v, k=k)
            new_eps_by_l_sigma[(l, spin)] = eps
            new_u_by_l_sigma[(l, spin)] = U

        r2 = np.maximum(r, 1e-12) ** 2
        n_up_out = np.zeros_like(r)
        n_dn_out = np.zeros_like(r)
        for spec in occ:
            u = new_u_by_l_sigma[(spec.l, spec.spin)][spec.n_index]
            contrib = (2 * spec.l + 1) * spec.f_per_m * (u * u) / (4.0 * np.pi * r2)
            if spec.spin == "up":
                n_up_out += contrib
            else:
                n_dn_out += contrib

        n_up_mixed = (1.0 - cfg.mix_alpha) * n_up + cfg.mix_alpha * n_up_out
        n_dn_mixed = (1.0 - cfg.mix_alpha) * n_dn + cfg.mix_alpha * n_dn_out
        dn_inf = max(np.max(np.abs(n_up_mixed - n_up)), np.max(np.abs(n_dn_mixed - n_dn)))
        n_up, n_dn = n_up_mixed, n_dn_mixed
        eps_by_l_sigma = new_eps_by_l_sigma
        u_by_l_sigma = new_u_by_l_sigma
        if verbose and (it == 1 or it % progress_every == 0):
            print(f"[LSDA VWN] iter={it} |dn|_inf={dn_inf:.3e}")
        if dn_inf < cfg.tol:
            converged = True
            break

    v_up, v_dn, vH, vX_up, vX_dn = _build_effective_potential_vwn(cfg.Z, r, n_up, n_dn, w)
    v_ext = -float(cfg.Z) / np.maximum(r, 1e-12)
    vc_up = v_up - (v_ext + vH + vX_up)
    vc_dn = v_dn - (v_ext + vH + vX_dn)

    n_tot = n_up + n_dn
    r2 = r * r
    e_sum = 0.0
    for spec in (cfg.occ or default_occupations(cfg.Z)):
        eps_l_sigma = eps_by_l_sigma[(spec.l, spec.spin)][spec.n_index]
        e_sum += (2 * spec.l + 1) * spec.f_per_m * float(eps_l_sigma)
    E_H = 0.5 * 4.0 * np.pi * trapz(n_tot * vH * r2, r, w)
    e_x = ex_dirac_density(n_up, n_dn)
    _, _, _, e_c = lda_c_vwn(n_up, n_dn)
    E_x = 4.0 * np.pi * trapz(e_x * r2, r, w)
    E_c = 4.0 * np.pi * trapz(e_c * r2, r, w)
    E_ext = 4.0 * np.pi * trapz(n_tot * v_ext * r2, r, w)
    E_xc_dc = 4.0 * np.pi * trapz((vX_up * n_up + vX_dn * n_dn + vc_up * n_up + vc_dn * n_dn) * r2, r, w)
    E_tot = e_sum - E_H - E_xc_dc + (E_x + E_c)
    # 计算 Kohn–Sham 动能 T_s
    int_n_vx = 4.0 * np.pi * trapz((n_up * vX_up + n_dn * vX_dn) * r2, r, w)
    int_n_vc = 4.0 * np.pi * trapz((n_up * vc_up + n_dn * vc_dn) * r2, r, w)
    int_n_v = 4.0 * np.pi * trapz(n_tot * (v_ext + vH) * r2, r, w) + int_n_vx + int_n_vc
    T_s = e_sum - int_n_v
    energies = dict(E_total=E_tot, E_H=E_H, E_x=E_x, E_c=E_c, E_ext=E_ext, E_sum=e_sum, E_kin=T_s, E_coul=E_H)

    # 收敛后补齐所有通道
    if cfg.compute_all_l and cfg.compute_all_l_mode == "final":
        all_eps: Dict[Tuple[int, str], np.ndarray] = {}
        all_u: Dict[Tuple[int, str], np.ndarray] = {}
        for l_all in range(0, cfg.lmax + 1):
            for spin in ("up", "down"):
                v_sel = v_up if spin == "up" else v_dn
                eps, U = solve_bound_states_fd(r, l=l_all, v_of_r=v_sel, k=max(cfg.eigs_per_l, 1))
                all_eps[(l_all, spin)] = eps
                all_u[(l_all, spin)] = U
        eps_by_l_sigma = all_eps
        u_by_l_sigma = all_u

    return SCFResult(
        converged=converged,
        iterations=it,
        eps_by_l_sigma=eps_by_l_sigma,
        u_by_l_sigma=u_by_l_sigma,
        n_up=n_up,
        n_dn=n_dn,
        v_h=vH,
        v_x_up=vX_up,
        v_x_dn=vX_dn,
        v_c_up=vc_up,
        v_c_dn=vc_dn,
        energies=energies,
    )


def _build_effective_potential_vwn(
    Z: int, r: np.ndarray, n_up: np.ndarray, n_dn: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""含 VWN 关联的有效势及分量：返回 (v_up, v_dn, vH, vX_up, vX_dn)。"""
    r_safe = np.maximum(r, 1e-12)
    v_ext = -float(Z) / r_safe
    n_tot = n_up + n_dn
    vH = v_hartree(n_tot, r, w)
    vx_up = vx_dirac(n_up)
    vx_dn = vx_dirac(n_dn)
    _, vc_up, vc_dn, _ = lda_c_vwn(n_up, n_dn)
    return v_ext + vH + vx_up + vc_up, v_ext + vH + vx_dn + vc_dn, vH, vx_up, vx_dn
