from __future__ import annotations

import numpy as np

from .grid import trapezoid_weights

__all__ = ["v_hartree"]


def v_hartree(n: np.ndarray, r: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    r"""由总电子数密度 :math:`n(r)` 计算径向 Hartree 势 :math:`v_H(r)`。

    采用与 Slater 积分一致的权重累积算法（原子单位）：

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
    - **重要**: 本版本使用与 Slater 积分相同的累积算法，确保单电子一致性。
    """
    if n.shape != r.shape:
        raise ValueError("n 与 r 的形状必须一致")
    if np.any(np.diff(r) <= 0):
        raise ValueError("r 必须严格单调递增")

    # 自动计算权重（如未提供）
    if w is None:
        w = trapezoid_weights(r)

    # 安全半径（避免除零）
    r_safe = np.maximum(r, 1e-12)
    eps = 1e-30  # 与 Slater 一致的安全常数

    # 被积函数（与 Slater k=0 公式对应）
    # Y: ∫_0^r n(r') r'^2 dr'（对应 r^k 项，k=0 时简化）
    # Z: ∫_r^∞ n(r') r' dr'（对应 r'^(-k-1) 项，k=0 时为 r'^(-1)）

    # 向前累积：Y = ∫_0^r n r^2 dr
    integrand_Y = n * (r**2) * w
    Y = np.cumsum(integrand_Y)

    # 向后累积：Z = ∫_r^∞ n r dr
    integrand_Z = n * r * w
    Z = np.cumsum(integrand_Z[::-1])[::-1]

    # 组合：v_H = 4π (Y/r + Z)
    vH = 4.0 * np.pi * (Y / (r_safe + eps) + Z)
    return vH
