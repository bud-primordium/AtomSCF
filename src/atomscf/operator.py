from __future__ import annotations

import numpy as np

from .grid import trapezoid_weights
from .utils import normalize_radial_u

__all__ = [
    "radial_hamiltonian_matrix",
    "solve_bound_states_fd",
]


def _second_derivative_tridiag_nonuniform(r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""构造非均匀网格上二阶导数算子的三对角表示（内部点）。

    对于非均匀一维网格 :math:`(r_0,\dots,r_{N-1})`，在内部点 :math:`i=1,\dots,N-2` 上，
    二阶导数可用以下一致格式近似：

    .. math::
        \left.\frac{d^2u}{dr^2}\right|_{r_i}
        \approx \frac{2}{h_{i-1}+h_i}\left[\frac{u_{i+1}-u_i}{h_i} - \frac{u_i-u_{i-1}}{h_{i-1}}\right],

    其中 :math:`h_{i-1}=r_i-r_{i-1},\ h_i=r_{i+1}-r_i`。

    本函数返回内部点上的三对角系数数组 :math:`(a, b, c)`，对应 :math:`u_{i-1}, u_i, u_{i+1}` 的系数，
    以及内部坐标数组 :math:`r_{\text{inner}}=(r_1,\dots,r_{N-2})`。

    Parameters
    ----------
    r : numpy.ndarray
        单调递增的一维网格坐标。

    Returns
    -------
    a : numpy.ndarray
        下对角系数数组，长度 :math:`N-2`，首元素对物理上无意义（可忽略或视为填充）。
    b : numpy.ndarray
        对角系数数组，长度 :math:`N-2`。
    c : numpy.ndarray
        上对角系数数组，长度 :math:`N-2`，末元素对物理上无意义（可忽略或视为填充）。
    r_inner : numpy.ndarray
        内部网格坐标 :math:`(r_1,\dots,r_{N-2})`。

    Notes
    -----
    - 该离散对应 Dirichlet 边界条件 :math:`u(r_0)=u(r_{N-1})=0`，因此仅对内部点建立方程。
    - 返回系数对应算子 :math:`\frac{d^2}{dr^2}`，后续 Hamiltonian 需乘以 :math:`-\tfrac{1}{2}`。
    """
    if r.ndim != 1:
        raise ValueError("r 必须是一维数组")
    if np.any(np.diff(r) <= 0):
        raise ValueError("r 必须严格单调递增")
    n = r.size
    if n < 3:
        raise ValueError("至少需要 3 个网格点以建立内部二阶导数")

    r_inner = r[1:-1]
    h_left = np.diff(r[:-1])  # r[i] - r[i-1], len = n-2 (for i=1..n-2)
    h_right = np.diff(r[1:])  # r[i+1] - r[i], len = n-2 (for i=1..n-2)

    a = np.empty(n - 2, dtype=float)
    b = np.empty(n - 2, dtype=float)
    c = np.empty(n - 2, dtype=float)

    denom = h_left + h_right
    a[:] = 2.0 / (h_left * denom)
    c[:] = 2.0 / (h_right * denom)
    b[:] = -2.0 / (h_left * h_right)

    return a, b, c, r_inner


def radial_hamiltonian_matrix(
    r: np.ndarray,
    l: int,
    v_of_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""构造径向 Hamiltonian（内部点）并返回矩阵与内部网格坐标。

    径向方程（原子单位）在 :math:`u(r)` 表示下为：

    .. math::
        \left[-\frac{1}{2}\frac{d^2}{dr^2}+\frac{\ell(\ell+1)}{2r^2}+v(r)\right]u(r)=\varepsilon\,u(r),

    本函数在非均匀网格上采用二阶差分对 :math:`\tfrac{d^2}{dr^2}` 离散，并以 Dirichlet 边界
    :math:`u(r_0)=u(r_{N-1})=0` 构造内部点的对称三对角矩阵。

    Parameters
    ----------
    r : numpy.ndarray
        单调递增的径向网格 :math:`(r_0,\dots,r_{N-1})`。
    l : int
        角动量量子数 :math:`\ell`（非负整数）。
    v_of_r : numpy.ndarray
        势能数组 :math:`v(r_i)`，长度与 :data:`r` 一致。

    Returns
    -------
    H : numpy.ndarray
        内部点上的 Hamiltonian 矩阵，形状 :math:`(N-2, N-2)`。
    r_inner : numpy.ndarray
        内部网格坐标 :math:`(r_1,\dots,r_{N-2})`。

    Notes
    -----
    - 矩阵是对称实矩阵，适用于标准本征求解器。
    - 若需得到完整长度的 :math:`u(r)`，可在内部解向量两端补零以满足 Dirichlet 边界。
    """
    if v_of_r.shape != r.shape:
        raise ValueError("v_of_r 的形状必须与 r 相同")
    if l < 0:
        raise ValueError("l 必须为非负整数")

    a, b, c, r_inner = _second_derivative_tridiag_nonuniform(r)
    n_in = r_inner.size

    # 组装二阶导数三对角矩阵 D2
    D2 = np.zeros((n_in, n_in), dtype=float)
    # 对角
    np.fill_diagonal(D2, b)
    # 上/下对角
    idx = np.arange(n_in - 1)
    D2[idx, idx + 1] = c[:-1]
    D2[idx + 1, idx] = a[1:]

    # 动能算子：T = -1/2 D2
    T = -0.5 * D2

    # 势能 + 离心项（内部点）
    v_inner = v_of_r[1:-1]
    lterm = 0.5 * l * (l + 1) / (r_inner**2)
    V = np.diag(v_inner + lterm)

    H = T + V
    return H, r_inner


def solve_bound_states_fd(
    r: np.ndarray,
    l: int,
    v_of_r: np.ndarray,
    k: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    r"""基于有限差分 Hamiltonian 的径向束缚态求解（取低端 :math:`k` 个本征对）。

    该方法将径向 Hamiltonian 离散为内部点的对称矩阵，调用密集线性代数本征求解。
    适合教学验证（例如氢样势下的 1s 能级），在网格较大时可能较慢。

    Parameters
    ----------
    r : numpy.ndarray
        单调递增的径向网格 :math:`(r_0,\dots,r_{N-1})`。
    l : int
        角动量量子数 :math:`\ell`。
    v_of_r : numpy.ndarray
        势能数组 :math:`v(r_i)`，长度与 :data:`r` 一致。
    k : int, optional
        返回最低能的本征态个数（默认 4）。

    Returns
    -------
    eps : numpy.ndarray
        低端 :math:`k` 个本征值（按升序）。
    U : numpy.ndarray
        对应的径向函数矩阵 :math:`U`，形状 :math:`(k, N)`，已在 :math:`[r_0,r_{N-1}]` 上按
        :math:`\int u^2\,dr=1` 归一，并在两端补零以满足 Dirichlet 边界。

    Notes
    -----
    - 实际内部计算在 :math:`(r_1,\dots,r_{N-2})` 上进行，端点边界条件 :math:`u=0`。
    - 若只需氢样势 1s 态，建议选择较大的 :math:`r_\max`（如 50–100）与足够细的网格以降低边界误差。
    """
    H, r_inner = radial_hamiltonian_matrix(r, l, v_of_r)
    # 全量密集本征求解（教学简化版）；后续可换用稀疏低端本征器
    e_all, U_inner = np.linalg.eigh(H)
    idx = np.argsort(e_all)
    e_sorted = e_all[idx]
    U_inner = U_inner[:, idx]

    k = min(k, e_sorted.size)
    eps = e_sorted[:k]
    U_out = np.zeros((k, r.size), dtype=float)
    w = trapezoid_weights(r)

    for j in range(k):
        u = np.zeros_like(r)
        u[1:-1] = U_inner[:, j]
        u, _ = normalize_radial_u(u, r, w)
        U_out[j] = u

    return eps, U_out

