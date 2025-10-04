r"""角动量耦合系数模块

本模块提供 HF 非局域交换所需的角动量耦合系数计算：

- Wigner-3j 系数（通过 sympy 包装）
- 径向中心场耦合因子 a_k(l, l')
- k 值选择规则（三角条件 + 奇偶性）

物理背景
========

在径向 HF 中心场近似下，非局域交换通过多极展开表示：

.. math::

    \frac{1}{|\mathbf{r} - \mathbf{r}'|} = \sum_{k=0}^\infty \frac{4\pi}{2k+1}
    \frac{r_<^k}{r_>^{k+1}} \sum_{q=-k}^k Y_{kq}(\hat{r}) Y_{kq}^*(\hat{r}')

对径向波函数的作用经球谐积分后，角动量耦合因子为：

.. math::

    a_k(\ell, \ell') = (2\ell'+1) \cdot
    \begin{pmatrix}
    \ell & k & \ell' \\
    0 & 0 & 0
    \end{pmatrix}^2

选择规则
========

k 值必须满足：

1. **三角条件**: :math:`|\ell - \ell'| \leq k \leq \ell + \ell'`
2. **奇偶性**: :math:`\ell + \ell' + k` 为偶数

这确保了 Wigner-3j 系数非零且物理合理。

References
----------
.. [Wigner3j] Varshalovich, D. A., Moskalev, A. N., & Khersonskii, V. K. (1988)
   "Quantum Theory of Angular Momentum"
   World Scientific
.. [HFAngular] Cowan, R. D. (1981)
   "The Theory of Atomic Structure and Spectra"
   University of California Press, Chapter 7
"""

from __future__ import annotations

import numpy as np
from sympy.physics.wigner import wigner_3j as _sympy_wigner_3j

__all__ = [
    "allowed_k_values",
    "coupling_factor_ak",
    "wigner_3j_squared",
]


def allowed_k_values(l: int, l_prime: int) -> list[int]:
    """计算允许的多极指标 k 值列表。

    Parameters
    ----------
    l : int
        第一个角动量量子数（l ≥ 0）
    l_prime : int
        第二个角动量量子数（l' ≥ 0）

    Returns
    -------
    list[int]
        允许的 k 值列表（按升序）

    Notes
    -----
    **选择规则**:
        1. 三角条件: :math:`|\ell - \ell'| \\leq k \\leq \ell + \ell'`
        2. 奇偶性: :math:`\ell + \ell' + k` 为偶数

    **物理意义**:
        - s-s (l=0, l'=0): k = [0]（仅库仑项）
        - s-p (l=0, l'=1): k = [1]（单个偶极项）
        - p-p (l=1, l'=1): k = [0, 2]（库仑 + 四极）

    Examples
    --------
    >>> allowed_k_values(0, 0)  # s-s
    [0]
    >>> allowed_k_values(0, 1)  # s-p
    [1]
    >>> allowed_k_values(1, 1)  # p-p
    [0, 2]
    >>> allowed_k_values(1, 2)  # p-d
    [1, 3]
    """
    if l < 0 or l_prime < 0:
        raise ValueError(f"角动量量子数必须非负: l={l}, l'={l_prime}")

    k_min = abs(l - l_prime)
    k_max = l + l_prime

    # 应用三角条件 + 奇偶性约束
    k_list = [
        k
        for k in range(k_min, k_max + 1)
        if (l + l_prime + k) % 2 == 0  # 奇偶性
    ]

    return k_list


def wigner_3j_squared(l: int, k: int, l_prime: int) -> float:
    r"""计算 Wigner-3j 系数的平方。

    计算:

    .. math::

        W^2 = \begin{pmatrix}
        \ell & k & \ell' \\
        0 & 0 & 0
        \end{pmatrix}^2

    Parameters
    ----------
    l : int
        第一个角动量
    k : int
        多极指标
    l_prime : int
        第二个角动量

    Returns
    -------
    float
        Wigner-3j 系数的平方

    Notes
    -----
    **符号约定**: 使用 `sympy.physics.wigner.wigner_3j` 标准相位

    **对称性**:
        - 偶置换不变: (l k l') = (l' k l) = (k l l')
        - 奇偶性: 若 l+k+l' 为奇数，结果为 0

    **实现**: sympy 返回符号表达式，需转换为浮点数

    Examples
    --------
    >>> wigner_3j_squared(0, 0, 0)  # s-s, k=0
    1.0
    >>> wigner_3j_squared(1, 0, 1)  # p-p, k=0
    0.1111...  # 1/9
    """
    # sympy.wigner_3j 返回 sympy 符号对象，需转换为浮点
    w3j_sym = _sympy_wigner_3j(l, k, l_prime, 0, 0, 0)
    w3j = float(w3j_sym)
    return w3j**2


def coupling_factor_ak(l: int, k: int, l_prime: int) -> float:
    r"""计算径向中心场角动量耦合因子 a_k(l, l')。

    定义:

    .. math::

        a_k(\ell, \ell') = (2\ell'+1) \cdot
        \begin{pmatrix}
        \ell & k & \ell' \\
        0 & 0 & 0
        \end{pmatrix}^2

    Parameters
    ----------
    l : int
        第一个角动量（目标态）
    k : int
        多极指标
    l_prime : int
        第二个角动量（占据态）

    Returns
    -------
    float
        耦合因子 a_k

    Notes
    -----
    **归一化**: 因子 :math:`(2\ell'+1)` 来源于 m 球对称平均

    **选择规则**: 若 k 不在 `allowed_k_values(l, l')` 中，返回 0

    **物理意义**: 该因子出现在 Fock 交换矩阵元中：

    .. math::

        K_{\ell}[u](r) = -\sum_{\ell'} \sum_k a_k(\ell,\ell')
        R^k(r) u_{\ell'}(r)

    Examples
    --------
    >>> coupling_factor_ak(0, 0, 0)  # s-s, k=0
    1.0
    >>> coupling_factor_ak(1, 0, 1)  # p-p, k=0
    0.3333...  # 1/3
    >>> coupling_factor_ak(1, 2, 1)  # p-p, k=2
    0.6666...  # 2/3
    """
    # 检查选择规则
    allowed_k = allowed_k_values(l, l_prime)
    if k not in allowed_k:
        return 0.0

    # 计算 a_k = (2l'+1) * W^2
    w_squared = wigner_3j_squared(l, k, l_prime)
    a_k = (2 * l_prime + 1) * w_squared

    return a_k


# 预计算常用值（s, p 轨道）供快速查表
_PRECOMPUTED_AK = {
    # (l, k, l'): a_k
    (0, 0, 0): 1.0,  # s-s, k=0
    (1, 0, 1): 1.0 / 3.0,  # p-p, k=0
    (1, 2, 1): 2.0 / 3.0,  # p-p, k=2
    (0, 1, 1): 1.0,  # s-p, k=1 (or p-s)
    (1, 1, 0): 1.0,  # p-s, k=1 (symmetric)
}


def get_coupling_factor(l: int, k: int, l_prime: int, use_cache: bool = True) -> float:
    """获取耦合因子（带预计算缓存）。

    Parameters
    ----------
    l, k, l_prime
        同 `coupling_factor_ak`
    use_cache : bool, optional
        是否使用预计算表（默认 True）

    Returns
    -------
    float
        耦合因子

    Notes
    -----
    对常用的 s/p 轨道组合（l, l' ≤ 1），直接查表；
    其他情况调用 `coupling_factor_ak` 实时计算。
    """
    if use_cache:
        # 尝试对称查表 (l,k,l') 或 (l',k,l)
        key = (l, k, l_prime)
        if key in _PRECOMPUTED_AK:
            return _PRECOMPUTED_AK[key]

        key_sym = (l_prime, k, l)
        if key_sym in _PRECOMPUTED_AK:
            return _PRECOMPUTED_AK[key_sym]

    # 回退到实时计算
    return coupling_factor_ak(l, k, l_prime)
