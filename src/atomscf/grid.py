import numpy as np

__all__ = [
    "radial_grid_linear",
    "radial_grid_log",
    "radial_grid_exp_transformed",
    "trapezoid_weights",
    "radial_grid_mixed",
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

    对数网格的定义（对数等差，适用于 Numerov）：

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
    - 该网格满足 ln(r) 等差，适用于 Numerov 方法。
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


def radial_grid_exp_transformed(
    n: int,
    rmin: float,
    rmax: float,
    total_span: float = 6.0
) -> tuple[np.ndarray, np.ndarray, float, float]:
    r"""生成用于变量变换方法的指数网格。

    网格公式参考文献 [1]_：

    .. math::
        r(j) = R_p(\exp(j\delta) - 1) + r_{\min}, \quad j=0,1,\ldots,j_{\max}

    其中 :math:`R_p` 由边界条件 :math:`r(j_{\max}) = r_{\max}` 确定，
    :math:`\delta` 由 :math:`j_{\max} \cdot \delta = \text{total\_span}` 确定（默认6.0）。

    该网格配合变量变换 :math:`v(j) = u(j) / \exp(j\delta/2)` 使用，
    可以消除坐标变换引入的一阶导数项。

    Parameters
    ----------
    n : int
        网格点数 :math:`j_{\max} + 1`，要求 :math:`n \ge 2`。
    rmin : float
        径向下限 :math:`r_{\min} \ge 0`（可以为0，物理上核在原点）。
    rmax : float
        径向上限 :math:`r_{\max}`，应满足 :math:`r_{\max} > r_{\min}`。
    total_span : float, optional
        控制参数 :math:`j_{\max} \cdot \delta`，默认6.0。

    Returns
    -------
    r : numpy.ndarray
        径向网格坐标，满足 r[0] = rmin, r[-1] = rmax。
    w : numpy.ndarray
        对应的梯形积分权重。
    delta : float
        网格参数 :math:`\delta`。
    Rp : float
        网格参数 :math:`R_p`。

    Notes
    -----
    - 该网格确保 r[0] = rmin（可以为0，物理上正确）
    - 需配合 operator.py 中的 solve_bound_states_transformed 使用
    - 变量变换后的方程没有一阶导数项，数值稳定性好

    References
    ----------
    .. [ExpGridTransform] 指数网格变量变换方法
       来源：Computational Physics Fall 2024, Assignment 7, Problem 2
       https://github.com/bud-primordium/Computational-Physics-Fall-2024/tree/main/Assignment_7/Problem_2
       problem_2.tex 第32-46行
    """
    if n < 2:
        raise ValueError("n 必须 >= 2")
    if rmin < 0:
        raise ValueError("rmin 必须 >= 0")
    if rmax <= rmin:
        raise ValueError("要求 rmax > rmin")

    j_max = n - 1
    delta = total_span / j_max

    # 从边界条件反推 Rp
    # r_max = Rp * (exp(j_max * delta) - 1) + rmin
    Rp = (rmax - rmin) / (np.exp(j_max * delta) - 1.0)

    # 生成网格
    j = np.arange(n)
    r = Rp * (np.exp(j * delta) - 1.0) + rmin

    # 验证边界条件
    assert np.abs(r[0] - rmin) < 1e-12, f"r[0] 应为 {rmin}，实际为 {r[0]}"
    assert np.abs(r[-1] - rmax) < 1e-10 * rmax, f"r[-1] 应为 {rmax}，实际为 {r[-1]}"

    w = trapezoid_weights(r)
    return r, w, delta, Rp


def radial_grid_mixed(
    n_inner: int,
    n_outer: int,
    rmin: float,
    r_switch: float,
    rmax: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""生成混合径向网格：核附近为对数网格，外层为线性网格。

    混合网格旨在在小 :math:`r` 处加密，同时控制总点数与外层行为：

    - 内层（对数）：:math:`r_i = r_{\min} e^{i\Delta},\ i=0..n_{\text{inner}}-1`，末点尽量接近 :math:`r_{\text{switch}}`；
    - 外层（线性）：从 :math:`r_{\text{switch}}` 到 :math:`r_{\max}` 等间距 :math:`n_{\text{outer}}` 个点（含端点）。

    Parameters
    ----------
    n_inner : int
        内层对数段点数（>=2）。
    n_outer : int
        外层线性段点数（>=2）。
    rmin : float
        最小半径（>0）。
    r_switch : float
        切换半径（满足 rmin < r_switch < rmax）。
    rmax : float
        最大半径（> r_switch）。

    Returns
    -------
    r : numpy.ndarray
        合并后的单调递增网格，重复点已去重。
    w : numpy.ndarray
        对应梯形积分权重。
    """
    if not (n_inner >= 2 and n_outer >= 2):
        raise ValueError("n_inner 与 n_outer 均需 >= 2")
    if not (rmin > 0 and rmax > rmin and r_switch > rmin and r_switch < rmax):
        raise ValueError("半径需满足 0<rmin<r_switch<rmax")

    # 内层对数：从 rmin 到 r_switch
    r_log, _ = radial_grid_log(n_inner, rmin, r_switch)
    # 外层线性：从 r_switch 到 rmax
    r_lin, _ = radial_grid_linear(n_outer, r_switch, rmax)

    # 合并并去重（避免 r_switch 重复）
    # 使用显式拼接：去掉r_log的最后一点（因为r_lin的第一点就是r_switch）
    r = np.concatenate([r_log[:-1], r_lin])
    # 验证严格单调
    if not np.all(np.diff(r) > 0):
        raise ValueError("混合网格生成失败：未能保证严格单调")
    # 权重
    w = trapezoid_weights(r)
    return r, w
