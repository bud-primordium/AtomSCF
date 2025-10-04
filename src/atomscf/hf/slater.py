"""Slater 径向积分模块

本模块实现 HF 非局域交换所需的 Slater 径向积分核计算。

核心算法：两段累积（Y^k/Z^k）
=========================================

Slater 径向积分定义：

.. math::

    R^k_{ij}(r) = \\int_0^\\infty \\frac{r_<^k}{r_>^{k+1}} u_i(r') u_j(r') \\, dr'

其中 :math:`r_< = \\min(r, r')`, :math:`r_> = \\max(r, r')`。

分解为向前/向后累积：

.. math::

    Y^k(r) &= \\int_0^r r'^k u_i(r') u_j(r') \\, dr' \\\\
    Z^k(r) &= \\int_r^\\infty r'^{-k-1} u_i(r') u_j(r') \\, dr' \\\\
    R^k(r) &= \\frac{Y^k(r)}{r^{k+1}} + Z^k(r) \\cdot r^k

数值稳定性
==========

- **r→0 处理**: 使用 :math:`r_{\\mathrm{safe}} = \\max(r, \\epsilon)` 避免除零
- **r→∞ 处理**: 边界显式设极限值（通常为零，因波函数衰减）
- **权重一致性**: 使用项目梯形权重 `w`

References
----------
.. [SlaterHF] Herman, F. & Skillman, S. (1963)
   "Atomic Structure Calculations"
   Prentice-Hall
.. [KoellingHarmon] Koelling, D. D. & Harmon, B. N. (1977)
   "A technique for relativistic spin-polarised calculations"
   J. Phys. C: Solid State Phys. 10, 3107
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "slater_integral_radial",
    "slater_integral_k0",
    "SlaterIntegralCache",
]


def slater_integral_radial(
    r: np.ndarray,
    w: np.ndarray,
    u_i: np.ndarray,
    u_j: np.ndarray,
    k: int,
    eps: float = 1e-30,
) -> np.ndarray:
    r"""计算 Slater 径向积分 :math:`R^k_{ij}(r)`。

    使用两段累积算法（向前 Y^k + 向后 Z^k）计算非局域交换核。

    Parameters
    ----------
    r : numpy.ndarray
        径向网格 (shape: [n])
    w : numpy.ndarray
        积分权重（梯形权重，shape: [n]）
    u_i : numpy.ndarray
        第一个径向波函数 (shape: [n])
    u_j : numpy.ndarray
        第二个径向波函数 (shape: [n])
    k : int
        多极指标 (k ≥ 0)
    eps : float, optional
        数值安全常数（避免除零），默认 1e-30

    Returns
    -------
    numpy.ndarray
        Slater 积分 :math:`R^k_{ij}(r)` (shape: [n])

    Notes
    -----
    **算法复杂度**: :math:`O(n)` 时间，:math:`O(n)` 空间

    **数值稳定性**:
        - r→0: 使用 r_safe = max(r, eps) 避免 :math:`r^{-k-1}` 发散
        - r→∞: 边界条件由波函数衰减自然满足

    **物理意义**:
        - k=0: 库仑积分（s-s 交换）
        - k=1: s-p 交叉项
        - k=2: p-p 交换

    Examples
    --------
    >>> r = np.linspace(1e-6, 50, 1000)
    >>> w = trapezoid_weights(r)
    >>> u_1s = ... # 氢样 1s 波函数
    >>> R0 = slater_integral_radial(r, w, u_1s, u_1s, k=0)
    """
    n = len(r)
    if len(w) != n or len(u_i) != n or len(u_j) != n:
        raise ValueError(f"输入数组长度不一致: r={len(r)}, w={len(w)}, u_i={len(u_i)}, u_j={len(u_j)}")
    if k < 0:
        raise ValueError(f"多极指标 k 必须非负，当前值: {k}")

    # 安全半径（避免除零）
    r_safe = np.maximum(r, eps)

    # 被积函数
    product = u_i * u_j

    # 向前累积 Y^k(r) = ∫_0^r r'^k u_i u_j dr'
    integrand_Y = (r**k) * product * w
    Y_k = np.cumsum(integrand_Y)

    # 向后累积 Z^k(r) = ∫_r^∞ r'^(-k-1) u_i u_j dr'
    integrand_Z = (r_safe ** (-k - 1)) * product * w
    Z_k = np.cumsum(integrand_Z[::-1])[::-1]  # 反向累积后翻转

    # 组合 R^k(r) = Y^k(r)/r^(k+1) + Z^k(r) * r^k
    R_k = Y_k / (r_safe ** (k + 1)) + Z_k * (r_safe**k)

    return R_k


def slater_integral_k0(
    r: np.ndarray,
    w: np.ndarray,
    u_i: np.ndarray,
    u_j: np.ndarray,
    eps: float = 1e-30,
) -> np.ndarray:
    r"""计算 k=0 Slater 积分（库仑积分，s-s 交换）。

    k=0 特化版本，物理意义为库仑相互作用：

    .. math::

        R^0(r) = \\frac{1}{r} \\int_0^r u_i u_j r'^2 \\, dr' + r \\int_r^\\infty u_i u_j \\, dr'

    Parameters
    ----------
    r : numpy.ndarray
        径向网格
    w : numpy.ndarray
        积分权重
    u_i : numpy.ndarray
        第一个径向波函数
    u_j : numpy.ndarray
        第二个径向波函数
    eps : float, optional
        数值安全常数，默认 1e-30

    Returns
    -------
    numpy.ndarray
        k=0 Slater 积分

    Notes
    -----
    此函数是 `slater_integral_radial(r, w, u_i, u_j, k=0)` 的等价实现，
    但公式更清晰地表达了库仑积分的物理意义。

    Examples
    --------
    >>> # 氢原子 1s 态自交换
    >>> r = radial_grid_linear(1000, 1e-6, 50.0)[0]
    >>> w = trapezoid_weights(r)
    >>> u_1s = 2 * np.exp(-r) / np.sqrt(4*np.pi)  # 归一化 1s
    >>> R0 = slater_integral_k0(r, w, u_1s, u_1s)
    >>> # 验证: 对 r→∞, R0 → 1/r (总电荷为1)
    """
    # 直接调用通用函数
    return slater_integral_radial(r, w, u_i, u_j, k=0, eps=eps)


class SlaterIntegralCache:
    """Slater 积分缓存管理器。

    用于缓存占据态之间的 Slater 积分，避免重复计算。

    Attributes
    ----------
    cache : dict
        缓存字典，键为 (i, j, k) 三元组

    Notes
    -----
    **缓存策略**:
        - 缓存占据态之间的 R^k_{ij}(r)（数量有限）
        - 不缓存涉及目标态 u 的积分（每次 SCF 迭代都变）

    **内存估计**:
        - 对 C 原子（6 个占据态），k_max=2
        - 最多 6×6×3 = 108 个积分
        - 每个积分 ~8KB（1000 点网格），总计 ~1MB

    Examples
    --------
    >>> cache = SlaterIntegralCache()
    >>> # 第一次计算并缓存
    >>> R0 = cache.get(r, w, u_1s, u_2s, k=0)
    >>> # 第二次直接从缓存读取
    >>> R0_cached = cache.get(r, w, u_1s, u_2s, k=0)  # 快速
    """

    def __init__(self):
        """初始化空缓存。"""
        self.cache: dict[tuple[int, int, int], np.ndarray] = {}

    def get(
        self,
        r: np.ndarray,
        w: np.ndarray,
        u_i: np.ndarray,
        u_j: np.ndarray,
        k: int,
        i_index: int | None = None,
        j_index: int | None = None,
    ) -> np.ndarray:
        """获取 Slater 积分（自动缓存）。

        Parameters
        ----------
        r, w, u_i, u_j, k
            同 `slater_integral_radial` 参数
        i_index : int, optional
            轨道 i 的索引（用于缓存键）
        j_index : int, optional
            轨道 j 的索引（用于缓存键）

        Returns
        -------
        numpy.ndarray
            Slater 积分

        Notes
        -----
        若 i_index 和 j_index 均提供，则启用缓存；否则直接计算。
        """
        # 检查缓存
        if i_index is not None and j_index is not None:
            key = (i_index, j_index, k)
            if key in self.cache:
                return self.cache[key]

        # 计算
        R_k = slater_integral_radial(r, w, u_i, u_j, k)

        # 存入缓存
        if i_index is not None and j_index is not None:
            self.cache[key] = R_k

        return R_k

    def clear(self):
        """清空缓存。"""
        self.cache.clear()

    def __len__(self):
        """返回缓存项数量。"""
        return len(self.cache)
