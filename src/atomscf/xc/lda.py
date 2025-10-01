from __future__ import annotations

import numpy as np

__all__ = [
    "vx_dirac",
    "ex_dirac_density",
    "lda_c_pz81",
]


def vx_dirac(n_sigma: np.ndarray) -> np.ndarray:
    r"""Dirac 交换势（LSDA，自旋分辨）。

    对于每个自旋通道 :math:`\sigma`，Dirac 交换势为：

    .. math::
        v_x^\sigma(r) = -\left(\frac{3}{\pi}\right)^{1/3} n_\sigma^{1/3}(r).

    Parameters
    ----------
    n_sigma : numpy.ndarray
        自旋分辨数密度 :math:`n_\sigma(r_i)`；当密度非正时，势定义为 0。

    Returns
    -------
    vx : numpy.ndarray
        Dirac 交换势 :math:`v_x^\sigma(r_i)`。

    Notes
    -----
    - 本函数不包含关联（correlation）贡献；作为 X-only 的最小实现。
    - 为避免数值不稳定，对小于 0 的密度会裁剪为 0 后再取立方根。
    """
    c = (3.0 / np.pi) ** (1.0 / 3.0)
    n_pos = np.clip(n_sigma, 0.0, None)
    return -c * np.cbrt(n_pos)


def ex_dirac_density(n_up: np.ndarray, n_dn: np.ndarray) -> np.ndarray:
    r"""Dirac 交换能量密度（体密度），单位 Hartree/a0^3。

    .. math::
        e_x(n_\uparrow, n_\downarrow) = -\frac{3}{4}\left(\frac{3}{\pi}\right)^{1/3}\left(n_\uparrow^{4/3}+n_\downarrow^{4/3}\right).

    Parameters
    ----------
    n_up, n_dn : numpy.ndarray
        自旋分辨数密度。

    Returns
    -------
    ex : numpy.ndarray
        交换能量密度 :math:`e_x`。
    """
    c = (3.0 / np.pi) ** (1.0 / 3.0)
    up = np.clip(n_up, 0.0, None)
    dn = np.clip(n_dn, 0.0, None)
    return -0.75 * c * (np.power(up, 4.0 / 3.0) + np.power(dn, 4.0 / 3.0))


def _pz81_params(polarized: bool):
    """PZ81 参数集。

    返回 (A, B, C, D, gamma, beta1, beta2)。
    取值来自 Perdew–Zunger 1981（Ceperley–Alder 拟合）。
    """
    if not polarized:
        # 非极化（zeta=0）
        return 0.031091, -0.048, 0.0020, -0.0116, -0.1423, 1.0529, 0.3334
    # 全极化（zeta=1）
    return 0.015545, -0.0269, 0.0007, -0.0048, -0.0843, 1.3981, 0.2611


def _pz81_eps_and_depsdrs(rs: np.ndarray, polarized: bool) -> tuple[np.ndarray, np.ndarray]:
    r"""PZ81 的 :math:`\varepsilon_c(r_s)` 与对 :math:`r_s` 的导数。

    分段形式：

    .. math::
        \varepsilon_c(r_s) = \begin{cases}
        A\ln r_s + B + C r_s\ln r_s + D r_s, & r_s < 1,\\
        \dfrac{\gamma}{1+\beta_1\sqrt{r_s}+\beta_2 r_s}, & r_s \ge 1.\end{cases}

    Parameters
    ----------
    rs : numpy.ndarray
        Wigner–Seitz 半径 :math:`r_s`。
    polarized : bool
        ``False`` 非极化；``True`` 全极化。

    Returns
    -------
    eps : numpy.ndarray
        每电子关联能量 :math:`\varepsilon_c`。
    depsdrs : numpy.ndarray
        :math:`\partial \varepsilon_c/\partial r_s`。
    """
    A, B, C, D, gamma, beta1, beta2 = _pz81_params(polarized)
    rs = np.asarray(rs)
    eps = np.empty_like(rs, dtype=float)
    deps = np.empty_like(rs, dtype=float)

    mask = rs < 1.0
    rs1 = rs[mask]
    rs2 = rs[~mask]

    # rs < 1
    if rs1.size:
        eps1 = A * np.log(rs1) + B + C * rs1 * np.log(rs1) + D * rs1
        deps1 = A / np.maximum(rs1, 1e-30) + C * (np.log(rs1) + 1.0) + D
        eps[mask] = eps1
        deps[mask] = deps1

    # rs >= 1
    if rs2.size:
        sqrt_rs = np.sqrt(rs2)
        denom = (1.0 + beta1 * sqrt_rs + beta2 * rs2)
        eps2 = gamma / denom
        deps2 = gamma * (-1.0) * (0.5 * beta1 / sqrt_rs + beta2) / (denom * denom)
        eps[~mask] = eps2
        deps[~mask] = deps2

    return eps, deps


def lda_c_pz81(n_up: np.ndarray, n_dn: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""PZ81 关联：返回 :math:`\varepsilon_c, v_c^\uparrow, v_c^\downarrow, e_c`。

    自旋插值：

    .. math::
        \varepsilon_c(n,\zeta)=\varepsilon_c^0(r_s) + [\varepsilon_c^1(r_s)-\varepsilon_c^0(r_s)] f(\zeta),

    其中 :math:`r_s=(3/(4\pi n))^{1/3}`，:math:`\zeta=(n_\uparrow-n_\downarrow)/n`，
    :math:`f(\zeta)=\dfrac{(1+\zeta)^{4/3}+(1-\zeta)^{4/3}-2}{2^{4/3}-2}`。

    势的链式法则：

    .. math::
        v_c^\sigma=\varepsilon_c + n\frac{\partial\varepsilon_c}{\partial n}
        + n\frac{\partial\varepsilon_c}{\partial \zeta}\frac{\partial \zeta}{\partial n_\sigma},\quad
        \frac{\partial \zeta}{\partial n_\uparrow}=\frac{1-\zeta}{n},\ \ 
        \frac{\partial \zeta}{\partial n_\downarrow}=-\frac{1+\zeta}{n}.

    返回的 :math:`e_c = n\,\varepsilon_c` 为关联能量密度（体密度）。
    """
    up = np.clip(n_up, 0.0, None)
    dn = np.clip(n_dn, 0.0, None)
    n = up + dn
    n_safe = np.maximum(n, 1e-30)

    rs = (3.0 / (4.0 * np.pi * n_safe)) ** (1.0 / 3.0)
    zeta = np.clip((up - dn) / n_safe, -1.0, 1.0)

    eps0, deps0 = _pz81_eps_and_depsdrs(rs, polarized=False)
    eps1, deps1 = _pz81_eps_and_depsdrs(rs, polarized=True)

    # 自旋插值函数 f(zeta) 及导数
    two43 = 2.0 ** (4.0 / 3.0)
    denom = two43 - 2.0
    f = ((1.0 + zeta) ** (4.0 / 3.0) + (1.0 - zeta) ** (4.0 / 3.0) - 2.0) / denom
    fp = ((4.0 / 3.0) * ((1.0 + zeta) ** (1.0 / 3.0) - (1.0 - zeta) ** (1.0 / 3.0))) / denom

    eps = eps0 + (eps1 - eps0) * f

    # d eps / d rs
    depsdrs = deps0 + (deps1 - deps0) * f
    # d eps / d zeta
    depsdz = (eps1 - eps0) * fp

    # drs/dn = -(rs)/(3n)
    drs_dn = -rs / (3.0 * n_safe)
    depsdn = depsdrs * drs_dn

    # d zeta / d n_sigma
    dz_up = (1.0 - zeta) / n_safe
    dz_dn = -(1.0 + zeta) / n_safe

    vcu = eps + n * depsdn + n * depsdz * dz_up
    vcd = eps + n * depsdn + n * depsdz * dz_dn

    e_c = n * eps
    return eps, vcu, vcd, e_c

