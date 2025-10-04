r"""HF 交换算子模块

本模块实现 Hartree-Fock 非局域交换算子的径向中心场形式。

物理背景
========

在径向 HF 中心场近似下，交换算子对目标态 u_l(r) 的作用为：

.. math::

    K_\ell[u](r) = -\sum_{\ell'} \sum_{k} a_k(\ell,\ell')
    \sum_i n_i \cdot R^k_{i,u}(r) \cdot u_{\ell',i}(r)

其中：
- :math:`a_k(\ell,\ell')` 是角动量耦合因子
- :math:`R^k_{i,u}(r)` 是 Slater 径向积分
- :math:`n_i` 是占据态 i 的占据数
- 求和遍历所有占据态 i（轨道 :math:`\ell'`）

s 轨道特化
==========

对于 s 轨道（l=0），选择规则简化为：
- 仅允许 k=0（库仑项）
- a_0(0,0) = 1.0
- 交换仅与其他 s 轨道耦合

交换公式变为：

.. math::

    K_0[u](r) = -\sum_{i \in s} n_i \cdot R^0_{i,u}(r) \cdot u_i(r)

实现策略
========

本模块采用**算子-作用式**（operator-action formulation）：

1. 交换算子返回闭包函数 `K[·]`，而非显式矩阵
2. 每次调用 `K[u]` 时实时计算 Slater 积分（避免 O(N²) 存储）
3. 占据态间的 R^k_{i,j} 可缓存（数量有限）
4. 目标态 u 的 R^k_{i,u} 每次重算（避免存储爆炸）

使用示例
========

.. code-block:: python

    from atomscf.hf import SlaterIntegralCache, exchange_operator_s
    from atomscf.grid import radial_grid_linear, trapezoid_weights

    # 准备网格
    r, _ = radial_grid_linear(n=1000, rmin=1e-6, rmax=50.0)
    w = trapezoid_weights(r)

    # 占据态（例：氢原子 1s）
    u_1s = np.sqrt(2) * np.exp(-r)
    u_occ = [u_1s]
    occ_nums = [1.0]

    # 创建交换算子
    cache = SlaterIntegralCache()
    K = exchange_operator_s(r, w, u_occ, occ_nums, cache=cache)

    # 应用到目标态
    Ku = K(u_1s)  # 返回 K[u_1s](r)

References
----------
.. [HFExchange] Cowan, R. D. (1981)
   "The Theory of Atomic Structure and Spectra"
   University of California Press, Chapter 7
.. [SlaterIntegrals] Slater, J. C. (1960)
   "Quantum Theory of Atomic Structure"
   McGraw-Hill, Volume 1
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .angular import get_coupling_factor
from .slater import SlaterIntegralCache, slater_integral_radial

__all__ = [
    "exchange_operator_s",
    "exchange_operator_general",
    "exchange_operator_general_spin",
]


def exchange_operator_s(
    r: np.ndarray,
    w: np.ndarray,
    u_occ: list[np.ndarray],
    occ_nums: list[float],
    cache: SlaterIntegralCache | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    r"""创建 s 轨道 (l=0) 交换算子。

    返回闭包函数 K[u]，对目标波函数 u 应用 HF 交换作用：

    .. math::

        K[u](r) = -\sum_i n_i \cdot a_0(0,0) \cdot R^0_{i,u}(r) \cdot u_i(r)

    由于 a_0(0,0) = 1.0，公式简化为：

    .. math::

        K[u](r) = -\sum_i n_i \cdot R^0_{i,u}(r) \cdot u_i(r)

    Parameters
    ----------
    r : np.ndarray
        径向网格点（长度 n）
    w : np.ndarray
        积分权重（长度 n）
    u_occ : list[np.ndarray]
        占据态径向波函数列表（每个长度 n）
    occ_nums : list[float]
        占据数列表（对应 u_occ）
    cache : SlaterIntegralCache | None, optional
        Slater 积分缓存（默认 None，内部创建新缓存）

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        交换算子闭包 K[u]，接受波函数 u(r) 返回 K[u](r)

    Notes
    -----
    **性能优化**:
        - 占据态间 R^0_{i,j} 可预先缓存
        - 目标态 u 的 R^0_{i,u} 实时计算（避免存储）

    **符号约定**:
        - 负号来自 HF 交换的物理定义
        - 对闭壳层自旋求和已隐含在 n_i 中

    Examples
    --------
    氢原子 1s 自交换::

        >>> r, _ = radial_grid_linear(n=1000, rmin=1e-6, rmax=50.0)
        >>> w = trapezoid_weights(r)
        >>> u_1s = np.sqrt(2) * np.exp(-r)  # 归一化径向波函数
        >>> K = exchange_operator_s(r, w, [u_1s], [1.0])
        >>> Ku = K(u_1s)
        >>> # Ku(r) ≈ -u_1s(r) / r  在远处

    See Also
    --------
    exchange_operator_general : 任意 l 的通用交换算子
    slater_integral_k0 : k=0 Slater 积分特化版本
    """
    if cache is None:
        cache = SlaterIntegralCache()

    if len(u_occ) != len(occ_nums):
        raise ValueError(
            f"占据态数量 ({len(u_occ)}) 与占据数数量 ({len(occ_nums)}) 不匹配"
        )

    # 预计算耦合因子（s-s 仅 k=0，且 a_0 = 1.0）
    a_0 = get_coupling_factor(0, 0, 0, use_cache=True)
    assert np.isclose(a_0, 1.0), "s-s 耦合因子应为 1.0"

    def apply_exchange(u_target: np.ndarray) -> np.ndarray:
        """应用交换算子到目标波函数。

        Parameters
        ----------
        u_target : np.ndarray
            目标径向波函数（长度 n）

        Returns
        -------
        np.ndarray
            K[u_target](r)
        """
        if u_target.shape != r.shape:
            raise ValueError(
                f"目标波函数形状 {u_target.shape} 与网格 {r.shape} 不匹配"
            )

        # 初始化交换势
        Ku = np.zeros_like(r)

        # 遍历所有占据态
        for i, (u_i, n_i) in enumerate(zip(u_occ, occ_nums)):
            # 计算 Slater 积分 R^0_{i,u}(r)
            # 注意：这里 i_index 用于缓存判断（实际上目标态不缓存）
            R0 = slater_integral_radial(r, w, u_i, u_target, k=0)

            # 累加交换贡献：-n_i * a_0 * R^0 * u_i
            # 由于 a_0 = 1.0，简化为：-n_i * R^0 * u_i
            Ku -= n_i * R0 * u_i

        return Ku

    return apply_exchange


def exchange_operator_general(
    r: np.ndarray,
    w: np.ndarray,
    l_target: int,
    u_occ_by_l: dict[int, list[np.ndarray]],
    occ_nums_by_l: dict[int, list[float]],
    cache: SlaterIntegralCache | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    r"""创建任意角动量 l 的通用交换算子。

    返回闭包函数 K[u]，对目标态 u_l(r) 应用完整 HF 交换：

    .. math::

        K_\ell[u](r) = -\sum_{\ell'} \sum_{k} a_k(\ell,\ell')
        \sum_i n_i \cdot R^k_{i,u}(r) \cdot u_{\ell',i}(r)

    Parameters
    ----------
    r : np.ndarray
        径向网格点
    w : np.ndarray
        积分权重
    l_target : int
        目标态的角动量量子数
    u_occ_by_l : dict[int, list[np.ndarray]]
        按角动量分组的占据态波函数
        格式：{l: [u_1, u_2, ...]}
    occ_nums_by_l : dict[int, list[float]]
        按角动量分组的占据数
        格式：{l: [n_1, n_2, ...]}
    cache : SlaterIntegralCache | None, optional
        Slater 积分缓存

    Returns
    -------
    Callable[[np.ndarray], np.ndarray]
        交换算子闭包 K[u]

    Notes
    -----
    **角动量耦合**:
        - 不同 l 通道通过允许的 k 值耦合
        - 例：p-p 允许 k=[0,2]，s-p 允许 k=[1]

    **性能**:
        - 复杂度 O(N_occ * n)，n 为网格点数
        - 缓存占据态间积分减少重复计算

    Examples
    --------
    碳原子 HF（1s² 2s² 2p²）::

        >>> u_occ_by_l = {
        ...     0: [u_1s, u_2s],  # s 轨道
        ...     1: [u_2p],        # p 轨道
        ... }
        >>> occ_nums_by_l = {
        ...     0: [2.0, 2.0],  # 1s² 2s²
        ...     1: [2.0],       # 2p²（球对称平均）
        ... }
        >>> K_p = exchange_operator_general(r, w, l_target=1,
        ...                                  u_occ_by_l, occ_nums_by_l)
        >>> Ku_2p = K_p(u_2p)

    See Also
    --------
    exchange_operator_s : s 轨道特化版本
    allowed_k_values : k 值选择规则
    """
    if cache is None:
        cache = SlaterIntegralCache()

    # 验证输入一致性
    if set(u_occ_by_l.keys()) != set(occ_nums_by_l.keys()):
        raise ValueError("u_occ_by_l 和 occ_nums_by_l 的 l 键不一致")

    for l_val in u_occ_by_l:
        if len(u_occ_by_l[l_val]) != len(occ_nums_by_l[l_val]):
            raise ValueError(f"l={l_val} 的波函数数量与占据数数量不匹配")

    def apply_exchange(u_target: np.ndarray) -> np.ndarray:
        """应用通用交换算子。"""
        if u_target.shape != r.shape:
            raise ValueError(f"目标波函数形状 {u_target.shape} 与网格 {r.shape} 不匹配")

        Ku = np.zeros_like(r)

        # 遍历所有占据态（按 l 分组）
        for l_occ, u_list in u_occ_by_l.items():
            n_list = occ_nums_by_l[l_occ]

            # 导入允许的 k 值
            from .angular import allowed_k_values

            # 获取允许的 k 值
            k_values = allowed_k_values(l_target, l_occ)

            # 遍历允许的多极指标 k
            for k in k_values:
                # 获取耦合因子
                a_k = get_coupling_factor(l_target, k, l_occ, use_cache=True)

                if np.abs(a_k) < 1e-15:
                    continue  # 跳过零耦合

                # 遍历该 l 通道的所有占据态
                for i, (u_i, n_i) in enumerate(zip(u_list, n_list)):
                    # 计算 Slater 积分 R^k_{i,u}(r)
                    R_k = slater_integral_radial(r, w, u_i, u_target, k=k)

                    # 占据数归一化：考虑空间简并和自旋简并
                    # n_i 为子壳层总占据数，需除以 (2l+1) 个 m 态和 2 个自旋态
                    g_m = 2 * l_occ + 1  # 空间简并度
                    n_eff = n_i / (g_m * 2.0)  # 每个 (m, σ) 的平均占据
                    # 累加交换贡献
                    Ku -= n_eff * a_k * R_k * u_i

        return Ku

    return apply_exchange


def exchange_operator_general_spin(
    r: np.ndarray,
    w: np.ndarray,
    l_target: int,
    spin_target: str,
    u_occ_by_l_spin: dict[tuple[int, str], list[np.ndarray]],
    occ_nums_by_l_spin: dict[tuple[int, str], list[float]],
    cache: SlaterIntegralCache | None = None,
) -> callable:
    r"""构造自旋分辨的通用交换算子（UHF）。

    Parameters
    ----------
    r : np.ndarray
        径向网格点
    w : np.ndarray
        积分权重
    l_target : int
        目标轨道的角动量量子数
    spin_target : str
        目标轨道的自旋通道（'up' 或 'down'）
    u_occ_by_l_spin : dict[tuple[int, str], list[np.ndarray]]
        占据态波函数，按 (l, spin) 索引
        例如：{(0, 'up'): [u_1s_up], (0, 'down'): [u_1s_down], ...}
    occ_nums_by_l_spin : dict[tuple[int, str], list[float]]
        占据数，按 (l, spin) 索引
        例如：{(0, 'up'): [1.0], (0, 'down'): [1.0], ...}
    cache : SlaterIntegralCache, optional
        Slater 积分缓存（可选）

    Returns
    -------
    callable
        交换算子闭包 K[u]，接受目标波函数返回交换作用后的结果

    Notes
    -----
    UHF 交换仅在**同自旋**占据态间发生：

    .. math::

        K_{\ell,\sigma}[u](r) = -\sum_{\ell'} \sum_{k \in \text{allowed}}
        a_k(\ell,\ell') \sum_{i \in \sigma} n_i \cdot R^k_{i,u}(r) \cdot u_{\ell',i}(r)

    与 RHF 的区别：
    - 求和仅遍历与目标态同自旋的占据态
    - 占据数 n_i 已按自旋分离，无需除以 2
    - 保持 m 简并度归一化（除以 2l+1）
    """
    from .slater import slater_integral_radial

    def apply_exchange(u_target: np.ndarray) -> np.ndarray:
        """对目标波函数应用交换算子。

        Parameters
        ----------
        u_target : np.ndarray
            目标径向波函数

        Returns
        -------
        np.ndarray
            交换作用后的波函数
        """
        if u_target.shape != r.shape:
            raise ValueError(f"目标波函数形状 {u_target.shape} 与网格 {r.shape} 不匹配")

        Ku = np.zeros_like(r)

        # 仅遍历同自旋的占据态
        for (l_occ, spin_occ), u_list in u_occ_by_l_spin.items():
            if spin_occ != spin_target:
                continue  # 跳过异自旋态（UHF 交换选择规则）

            n_list = occ_nums_by_l_spin[(l_occ, spin_occ)]

            # 获取允许的 k 值
            from .angular import allowed_k_values, get_coupling_factor

            k_values = allowed_k_values(l_target, l_occ)

            # 遍历允许的多极指标 k
            for k in k_values:
                # 获取耦合因子
                a_k = get_coupling_factor(l_target, k, l_occ, use_cache=True)

                if np.abs(a_k) < 1e-15:
                    continue  # 跳过零耦合

                # 遍历该 (l, spin) 通道的所有占据态
                for i, (u_i, n_i) in enumerate(zip(u_list, n_list)):
                    # 计算 Slater 积分 R^k_{i,u}(r)
                    R_k = slater_integral_radial(r, w, u_i, u_target, k=k)

                    # 占据数归一化：仅考虑空间简并度（自旋已分离）
                    g_m = 2 * l_occ + 1  # 空间简并度
                    n_eff = n_i / g_m  # 每个 m 的占据（无需除以 2）
                    # 累加交换贡献
                    Ku -= n_eff * a_k * R_k * u_i

        return Ku

    return apply_exchange
