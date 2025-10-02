from __future__ import annotations

import numpy as np
from .constants import VWN5_RPA_PARAMS

__all__ = [
    "lda_c_vwn",
]


def _vwn_params(polarized: bool):
    """VWN5/RPA 参数（集中于 constants.VWN5_RPA_PARAMS）。"""
    key = "polarized" if polarized else "unpolarized"
    return VWN5_RPA_PARAMS[key]


def _vwn_eps_and_depsdrs(rs: np.ndarray, polarized: bool) -> tuple[np.ndarray, np.ndarray]:
    r"""VWN5 关联能及其对 :math:`r_s` 的解析导数。

    采用 :math:`x=\sqrt{r_s}` 变量，写出 :math:`\varepsilon_c(x)` 及 :math:`\partial \varepsilon_c/\partial x` 的闭式表达，
    再由 :math:`\partial x/\partial r_s = 1/(2\sqrt{r_s})` 得到 :math:`\partial \varepsilon_c/\partial r_s`。
    """
    A, x0, b, c = _vwn_params(polarized)
    rs = np.asarray(rs, dtype=float)
    x = np.sqrt(np.maximum(rs, 1e-30))
    Q = np.sqrt(max(4.0 * c - b * b, 1e-30))

    def X(xx: np.ndarray) -> np.ndarray:
        return xx * xx + b * xx + c

    # epsilon_c(x)
    term_log = np.log((x * x) / X(x))
    term_atan = (2.0 * b / Q) * np.arctan(Q / (2.0 * x + b))
    denom0 = X(x0)
    # 常数项：避免除零
    denom0 = denom0 if np.isscalar(denom0) else np.where(np.abs(denom0) < 1e-30, 1e-30, denom0)
    const_pref = (b * x0) / denom0
    term_log2 = np.log(((x - x0) * (x - x0)) / X(x))
    term_atan2 = (2.0 * (b + 2.0 * x0) / Q) * np.arctan(Q / (2.0 * x + b))
    eps = A * (term_log + term_atan - const_pref * (term_log2 + term_atan2))

    # d eps / dx（解析导数）
    # d/dx ln(x^2 / X) = 2/x - (2x+b)/X
    d_log = 2.0 / np.maximum(x, 1e-30) - (2.0 * x + b) / np.maximum(X(x), 1e-30)
    # d/dx [ 2b/Q arctan(Q/(2x+b)) ] = -4b / ( (2x+b)^2 + Q^2 )
    denom_sq = (2.0 * x + b) ** 2 + Q * Q
    d_atan = -4.0 * b / np.maximum(denom_sq, 1e-30)
    # d/dx ln( (x-x0)^2 / X ) = 2/(x-x0) - (2x+b)/X
    d_log2 = 2.0 / np.maximum(x - x0, 1e-30) - (2.0 * x + b) / np.maximum(X(x), 1e-30)
    # d/dx [ 2(b+2x0)/Q arctan(Q/(2x+b)) ] = -4(b+2x0) / ( (2x+b)^2 + Q^2 )
    d_atan2 = -4.0 * (b + 2.0 * x0) / np.maximum(denom_sq, 1e-30)
    deps_dx = A * (d_log + d_atan - const_pref * (d_log2 + d_atan2))
    deps_drs = deps_dx / (2.0 * np.maximum(x, 1e-30))
    return eps, deps_drs


def _num_grad_y_x(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    r"""对非均匀单调网格计算数值导数 :math:`dy/dx`（中心差分，端点前/后向差分）。

    相比 ``numpy.gradient``，显式避免了极小间距下的除零警告。
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = y.size
    g = np.empty_like(y)
    eps = 1e-30
    if n == 1:
        g[...] = 0.0
        return g
    # 前向差分 at 0
    dx = max(x[1] - x[0], eps)
    g[0] = (y[1] - y[0]) / dx
    # 中心差分
    for i in range(1, n - 1):
        dxl = max(x[i] - x[i - 1], eps)
        dxr = max(x[i + 1] - x[i], eps)
        g[i] = (dxl * (y[i + 1] - y[i]) / dxr + dxr * (y[i] - y[i - 1]) / dxl) / (dxl + dxr)
    # 后向差分 at n-1
    dx = max(x[-1] - x[-2], eps)
    g[-1] = (y[-1] - y[-2]) / dx
    return g


def lda_c_vwn(n_up: np.ndarray, n_dn: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""VWN 关联：返回 :math:`\varepsilon_c, v_c^\uparrow, v_c^\downarrow, e_c`。

    采用自旋插值：
    :math:`\varepsilon_c(n,\zeta)=\varepsilon_c^0(r_s) + [\varepsilon_c^1(r_s)-\varepsilon_c^0(r_s)] f(\zeta)`，
    其中 :math:`r_s=(3/(4\pi n))^{1/3}`，:math:`\zeta=(n_\uparrow-n_\downarrow)/n`，
    :math:`f(\zeta)=((1+\zeta)^{4/3}+(1-\zeta)^{4/3}-2)/(2^{4/3}-2)`。

    :math:`v_c^\sigma` 通过链式法则得到，其中 :math:`\partial\varepsilon/\partial n` 用数值微分实现（对 :math:`r_s` 的导数经 :math:`dr_s/dn` 链接）。
    """
    up = np.clip(n_up, 0.0, None)
    dn = np.clip(n_dn, 0.0, None)
    n = up + dn
    n_safe = np.maximum(n, 1e-30)

    rs = (3.0 / (4.0 * np.pi * n_safe)) ** (1.0 / 3.0)
    zeta = np.clip((up - dn) / n_safe, -1.0, 1.0)

    eps0, deps0_drs = _vwn_eps_and_depsdrs(rs, polarized=False)
    eps1, deps1_drs = _vwn_eps_and_depsdrs(rs, polarized=True)

    two43 = 2.0 ** (4.0 / 3.0)
    denom = two43 - 2.0
    f = ((1.0 + zeta) ** (4.0 / 3.0) + (1.0 - zeta) ** (4.0 / 3.0) - 2.0) / denom
    fp = ((4.0 / 3.0) * ((1.0 + zeta) ** (1.0 / 3.0) - (1.0 - zeta) ** (1.0 / 3.0))) / denom

    eps = eps0 + (eps1 - eps0) * f
    deps_drs = deps0_drs + (deps1_drs - deps0_drs) * f
    deps_dz = (eps1 - eps0) * fp

    drs_dn = -rs / (3.0 * n_safe)
    deps_dn = deps_drs * drs_dn

    dz_up = (1.0 - zeta) / n_safe
    dz_dn = -(1.0 + zeta) / n_safe

    vcu = eps + n * deps_dn + n * deps_dz * dz_up
    vcd = eps + n * deps_dn + n * deps_dz * dz_dn

    e_c = n * eps
    return eps, vcu, vcd, e_c
