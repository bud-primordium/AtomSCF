import numpy as np

__all__ = [
    "radial_grid_linear",
    "radial_grid_log",
    "trapezoid_weights",
]


def trapezoid_weights(r: np.ndarray) -> np.ndarray:
    r"""为给定单调递增的径向网格计算梯形积分权重。

    使用一维梯形规则近似积分：

    .. math::
        \int_{r_\min}^{r_\max} f(r)\,\mathrm{d}r \approx \sum_{i=0}^{N-1} w_i f(r_i)

    其中权重 :math:`w_i` 由相邻网格间距确定。端点权重为半步长，内部点为左右间距的平均值。

    Parameters
    ----------
    r : numpy.ndarray
        单调递增的径向坐标数组 :math:`(r_0, r_1, \dots, r_{N-1})`，要求 :math:`r_i < r_{i+1}`。

    Returns
    -------
    w : numpy.ndarray
        梯形积分权重 :math:`(w_0, \dots, w_{N-1})`，满足上述积分近似式。

    Notes
    -----
    - 该权重适用于一维径向函数 :math:`u(r)` 的普通积分（例如归一化 :math:`\int u^2\,dr=1`）。
    - 若用于三维体积分（如 :math:`\int 4\pi r^2 n(r)\,dr`），需显式将体积因子 :math:`4\pi r^2` 乘入被积函数。
    """
    if r.ndim != 1:
        raise ValueError("r 必须是一维数组")
    if np.any(np.diff(r) <= 0):
        raise ValueError("r 必须严格单调递增")
    n = r.size
    w = np.empty_like(r)
    if n == 1:
        w[0] = 0.0
        return w
    dr = np.diff(r)
    w[0] = 0.5 * dr[0]
    w[1:-1] = 0.5 * (dr[1:] + dr[:-1])
    w[-1] = 0.5 * dr[-1]
    return w


def radial_grid_linear(n: int, rmin: float, rmax: float) -> tuple[np.ndarray, np.ndarray]:
    r"""生成线性（等间隔）径向网格及其梯形积分权重。

    线性网格的定义：

    .. math::
        r_i = r_\min + i\,\Delta r, \quad i=0,\dots, N-1,\ \ \Delta r = \frac{r_\max - r_\min}{N-1}.

    Parameters
    ----------
    n : int
        网格点数 :math:`N`，要求 :math:`N\ge 2`。
    rmin : float
        径向下限 :math:`r_\min`，应为非负数（通常取非常小的正数以避免 :math:`r=0` 奇点）。
    rmax : float
        径向上限 :math:`r_\max`，应满足 :math:`r_\max > r_\min`。

    Returns
    -------
    r : numpy.ndarray
        线性径向网格坐标 :math:`r_i`。
    w : numpy.ndarray
        对应的梯形积分权重 :math:`w_i`。

    Examples
    --------
    >>> import numpy as np
    >>> from atom_scf.grid import radial_grid_linear
    >>> r, w = radial_grid_linear(5, 0.0, 1.0)
    >>> np.allclose(np.sum(w), 1.0)
    True
    """
    if n < 2:
        raise ValueError("n 必须 >= 2")
    if rmax <= rmin:
        raise ValueError("要求 rmax > rmin")
    r = np.linspace(rmin, rmax, n)
    w = trapezoid_weights(r)
    return r, w


def radial_grid_log(n: int, rmin: float, rmax: float) -> tuple[np.ndarray, np.ndarray]:
    r"""生成对数（几何）径向网格及其梯形积分权重。

    对数网格的定义：

    .. math::
        r_i = r_\min\,\exp\!\left( i\,\Delta x \right),\ \ \Delta x = \frac{\ln(r_\max) - \ln(r_\min)}{N-1}.

    Parameters
    ----------
    n : int
        网格点数 :math:`N`，要求 :math:`N\ge 2`。
    rmin : float
        径向下限 :math:`r_\min>0`，需严格大于 0 才能取对数。
    rmax : float
        径向上限 :math:`r_\max`，应满足 :math:`r_\max > r_\min`。

    Returns
    -------
    r : numpy.ndarray
        对数径向网格坐标 :math:`r_i`。
    w : numpy.ndarray
        对应的梯形积分权重 :math:`w_i`。

    Notes
    -----
    - 对数网格能在小 :math:`r` 处加密采样，适合处理库仑势与核附近行为。
    - 若后续采用有限差分离散二阶导数，需使用非均匀网格的差分公式（本包已支持）。
    """
    if n < 2:
        raise ValueError("n 必须 >= 2")
    if rmin <= 0:
        raise ValueError("对数网格要求 rmin > 0")
    if rmax <= rmin:
        raise ValueError("要求 rmax > rmin")
    x = np.linspace(np.log(rmin), np.log(rmax), n)
    r = np.exp(x)
    w = trapezoid_weights(r)
    return r, w

