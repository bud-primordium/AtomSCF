from __future__ import annotations

import numpy as np

from .grid import trapezoid_weights

__all__ = ["v_hartree"]


def v_hartree(n: np.ndarray, r: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    r"""由总电子数密度 :math:`n(r)` 计算径向 Hartree 势 :math:`v_H(r)`。

    采用数值稳定的分段积分公式（原子单位）：

    .. math::
        v_H(r) = \frac{4\pi}{r}\int_0^r n(r')r'^2\,dr' + 4\pi \int_r^{\infty} n(r')r'\,dr'.

    在有限区间 :math:`[r_\min,r_\max]` 上，第二项的上限以 :math:`r_\max` 近似，
    当 :math:`r\to r_\max` 时，:math:`v_H(r) \approx Q/r`，其中 :math:`Q=\int 4\pi r'^2 n(r')dr'` 为电子总数。

    Parameters
    ----------
    n : numpy.ndarray
        径向数密度 :math:`n(r_i)`，单位为 :math:`a_0^{-3}`。
    r : numpy.ndarray
        径向网格 :math:`r_i`，需严格单调递增，且 :math:`r_i>0`（建议避开 0 点）。
    w : numpy.ndarray, optional
        梯形积分权重 :math:`w_i`；若为 ``None`` 则内部计算一次。

    Returns
    -------
    vH : numpy.ndarray
        Hartree 势 :math:`v_H(r_i)`，单位 Hartree。

    Notes
    -----
    - 在极小 :math:`r` 处对 :math:`1/r` 做安全下界裁剪。
    - 对于单电子体系（如氢原子），严格 HF 下 Hartree 与交换应相消；在 LSDA 中仅近似抵消。
    """
    if n.shape != r.shape:
        raise ValueError("n 与 r 的形状必须一致")
    if np.any(np.diff(r) <= 0):
        raise ValueError("r 必须严格单调递增")
    # 采用分段梯形法逐段累积，避免直接使用全局权重导致的端点半权误差扩散
    r_safe = np.maximum(r, 1e-12)
    f1 = n * (r ** 2)  # 被积函数：n(r) r^2
    f2 = n * r  # 被积函数：n(r) r

    dr = np.diff(r)
    # 前缀积分 I1[i] = ∫_0^{r_i} f1 dr
    inc1 = 0.5 * (f1[:-1] + f1[1:]) * dr
    I1 = np.zeros_like(r)
    I1[1:] = np.cumsum(inc1)

    # 后缀积分 I2[i] = ∫_{r_i}^{r_max} f2 dr
    inc2 = 0.5 * (f2[:-1] + f2[1:]) * dr
    I2 = np.zeros_like(r)
    I2[:-1] = np.cumsum(inc2[::-1])[::-1]

    vH = 4.0 * np.pi * (I1 / r_safe + I2)
    return vH
