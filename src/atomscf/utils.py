from __future__ import annotations

import numpy as np

__all__ = [
    "trapz",
    "normalize_radial_u",
]


def trapz(y: np.ndarray, r: np.ndarray, w: np.ndarray | None = None) -> float:
    r"""使用梯形权重对函数进行一维数值积分。

    若提供 :data:`w`，则直接返回 :math:`\sum_i w_i y_i`；否则退化为
    :func:`numpy.trapz(y, r)` 的行为（均匀或非均匀步长）。

    Parameters
    ----------
    y : numpy.ndarray
        被积函数离散值 :math:`y(r_i)`。
    r : numpy.ndarray
        网格坐标 :math:`r_i`。
    w : numpy.ndarray, optional
        梯形权重 :math:`w_i`；若为 ``None`` 则内部用 ``numpy.trapz`` 计算。

    Returns
    -------
    float
        积分近似值。
    """
    if w is not None:
        if w.shape != y.shape:
            raise ValueError("w 与 y 的形状必须一致")
        return float(np.sum(w * y))
    return float(np.trapz(y, r))


def normalize_radial_u(u: np.ndarray, r: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    r"""将径向函数 :math:`u(r)` 归一化到 :math:`\int u^2\,dr=1`。

    Parameters
    ----------
    u : numpy.ndarray
        径向函数离散值 :math:`u(r_i)`。
    r : numpy.ndarray
        网格坐标 :math:`r_i`。
    w : numpy.ndarray
        梯形权重 :math:`w_i`。

    Returns
    -------
    u_norm : numpy.ndarray
        归一化后的函数值数组。
    norm : float
        原始函数的范数 :math:`\sqrt{\int u^2\,dr}`。

    Notes
    -----
    - 该归一化对应径向方程中 :math:`u(r)` 的标准内积，不包含体积分因子 :math:`4\pi r^2`。
    """
    if not (u.shape == r.shape == w.shape):
        raise ValueError("u, r, w 的形状必须一致")
    norm2 = trapz(u * u, r, w)
    norm = float(np.sqrt(max(norm2, 1e-300)))
    return u / norm, norm

