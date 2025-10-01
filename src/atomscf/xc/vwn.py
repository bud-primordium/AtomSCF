from __future__ import annotations

import numpy as np

__all__ = [
    "lda_c_vwn",
]


def _vwn_params(polarized: bool):
    """VWN5 参数（常见取值），用于 :math:`\varepsilon_c(r_s)` 计算。

    参考常用实现的参数集合（RPA 版本，VWN5），用于教学对比。
    注意：不同资料存在细微差异，后续将据文献校正。
    """
    if not polarized:
        # 非极化
        A = 0.0310907
        x0 = -0.10498
        b = 3.72744
        c = 12.9352
    else:
        # 全极化
        A = 0.01554535
        x0 = -0.32500
        b = 7.06042
        c = 18.0578
    return A, x0, b, c


def _vwn_eps(rs: np.ndarray, polarized: bool) -> np.ndarray:
    r"""VWN 每电子关联能 :math:`\varepsilon_c(r_s)`（使用 :math:`x=\sqrt{r_s}` 的常见表达）。"""
    A, x0, b, c = _vwn_params(polarized)
    rs = np.asarray(rs, dtype=float)
    x = np.sqrt(np.maximum(rs, 1e-30))
    Q = np.sqrt(4.0 * c - b * b)

    def g(xx: np.ndarray) -> np.ndarray:
        return np.log((xx * xx) / (xx * xx + b * xx + c)) + (2.0 * b / Q) * np.arctan((2.0 * xx + b) / Q)

    term0 = g(x)
    denom0 = x0 * x0 + b * x0 + c
    # 避免 x0 奇点的数值问题（理论上固定常数，数值无问题）
    term1 = (x0 * b / denom0) * (np.log(((x - x0) * (x - x0)) / denom0) + (2.0 * (b + 2.0 * x0) / Q) * np.arctan((2.0 * x + b) / Q))
    return A * (term0 - term1)


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

    eps0 = _vwn_eps(rs, polarized=False)
    eps1 = _vwn_eps(rs, polarized=True)

    # 数值求 d eps / d rs
    deps0_drs = _num_grad_y_x(eps0, rs)
    deps1_drs = _num_grad_y_x(eps1, rs)

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
