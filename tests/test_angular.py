"""角动量耦合系数单元测试

测试 hf/angular.py 模块的正确性。
"""

import numpy as np
import pytest

from atomscf.hf.angular import (
    allowed_k_values,
    coupling_factor_ak,
    get_coupling_factor,
    wigner_3j_squared,
)


@pytest.mark.operator
@pytest.mark.quick
def test_allowed_k_ss():
    """测试 s-s (l=0, l'=0) 的允许 k 值。"""
    k_list = allowed_k_values(0, 0)
    assert k_list == [0], f"s-s 应仅允许 k=0，实际: {k_list}"


@pytest.mark.operator
@pytest.mark.quick
def test_allowed_k_sp():
    """测试 s-p (l=0, l'=1) 的允许 k 值。"""
    k_list = allowed_k_values(0, 1)
    assert k_list == [1], f"s-p 应仅允许 k=1，实际: {k_list}"


@pytest.mark.operator
@pytest.mark.quick
def test_allowed_k_pp():
    """测试 p-p (l=1, l'=1) 的允许 k 值。"""
    k_list = allowed_k_values(1, 1)
    assert k_list == [0, 2], f"p-p 应允许 k=[0,2]，实际: {k_list}"


@pytest.mark.operator
def test_allowed_k_pd():
    """测试 p-d (l=1, l'=2) 的允许 k 值。"""
    k_list = allowed_k_values(1, 2)
    # |1-2| = 1, 1+2 = 3, 奇偶性: 1+2+k 为偶 => k=1,3
    assert k_list == [1, 3], f"p-d 应允许 k=[1,3]，实际: {k_list}"


@pytest.mark.operator
def test_allowed_k_dd():
    """测试 d-d (l=2, l'=2) 的允许 k 值。"""
    k_list = allowed_k_values(2, 2)
    # |2-2| = 0, 2+2 = 4, 奇偶性: 2+2+k 为偶 => k=0,2,4
    assert k_list == [0, 2, 4], f"d-d 应允许 k=[0,2,4]，实际: {k_list}"


@pytest.mark.operator
def test_allowed_k_invalid_negative():
    """测试负角动量的异常检测。"""
    with pytest.raises(ValueError, match="角动量量子数必须非负"):
        allowed_k_values(-1, 0)


@pytest.mark.operator
@pytest.mark.quick
def test_wigner_3j_squared_s_s():
    """测试 s-s (0 0 0) Wigner-3j 系数。

    理论值: W(0,0,0; 0,0,0) = 1
    """
    w2 = wigner_3j_squared(0, 0, 0)
    assert np.isclose(w2, 1.0), f"s-s Wigner-3j^2 应为 1.0，实际: {w2}"


@pytest.mark.operator
@pytest.mark.quick
def test_wigner_3j_squared_p_p_k0():
    """测试 p-p (1 0 1) Wigner-3j 系数。

    理论值: W(1,0,1; 0,0,0)^2 = 1/3
    """
    w2 = wigner_3j_squared(1, 0, 1)
    expected = 1.0 / 3.0
    assert np.isclose(w2, expected), f"p-p k=0 Wigner-3j^2 应为 {expected}，实际: {w2}"


@pytest.mark.operator
@pytest.mark.quick
def test_wigner_3j_squared_p_p_k2():
    """测试 p-p (1 2 1) Wigner-3j 系数。

    理论值: W(1,2,1; 0,0,0)^2 = 2/15
    """
    w2 = wigner_3j_squared(1, 2, 1)
    expected = 2.0 / 15.0
    assert np.isclose(w2, expected, rtol=1e-6), f"p-p k=2 Wigner-3j^2 应为 {expected}，实际: {w2}"


@pytest.mark.operator
def test_wigner_3j_parity_violation():
    """测试奇偶性违反时 Wigner-3j 为零。

    例: (0,1,0) 中 0+1+0=1 为奇数，应返回 0
    """
    w2 = wigner_3j_squared(0, 1, 0)
    assert np.isclose(w2, 0.0, atol=1e-15), f"奇偶性违反应返回 0，实际: {w2}"


@pytest.mark.operator
@pytest.mark.quick
def test_coupling_factor_ss():
    """测试 s-s 耦合因子 a_0(0,0)。

    a_0 = (2*0+1) * W(0,0,0)^2 = 1 * 1 = 1
    """
    a_k = coupling_factor_ak(0, 0, 0)
    assert np.isclose(a_k, 1.0), f"s-s a_0 应为 1.0，实际: {a_k}"


@pytest.mark.operator
@pytest.mark.quick
def test_coupling_factor_pp_k0():
    """测试 p-p 耦合因子 a_0(1,1)。

    a_0 = (2*1+1) * W(1,0,1)^2 = 3 * (1/3) = 1.0
    """
    a_k = coupling_factor_ak(1, 0, 1)
    expected = 1.0
    assert np.isclose(a_k, expected), f"p-p a_0 应为 {expected}，实际: {a_k}"


@pytest.mark.operator
@pytest.mark.quick
def test_coupling_factor_pp_k2():
    """测试 p-p 耦合因子 a_2(1,1)。

    a_2 = (2*1+1) * W(1,2,1)^2 = 3 * (2/15) = 2/5
    """
    a_k = coupling_factor_ak(1, 2, 1)
    expected = 2.0 / 5.0
    assert np.isclose(a_k, expected, rtol=1e-6), f"p-p a_2 应为 {expected}，实际: {a_k}"


@pytest.mark.operator
def test_coupling_factor_forbidden_k():
    """测试禁戒 k 值时返回 0。

    例: s-s (l=0, l'=0) 不允许 k=1（奇偶性违反）
    """
    a_k = coupling_factor_ak(0, 1, 0)
    assert np.isclose(a_k, 0.0, atol=1e-15), f"禁戒 k 应返回 0，实际: {a_k}"


@pytest.mark.operator
@pytest.mark.quick
def test_coupling_factor_sp():
    """测试 s-p 耦合因子 a_1(0,1)。

    a_1 = (2*1+1) * W(0,1,1)^2 = 3 * (1/3) = 1
    """
    a_k = coupling_factor_ak(0, 1, 1)
    assert np.isclose(a_k, 1.0), f"s-p a_1 应为 1.0，实际: {a_k}"


@pytest.mark.operator
def test_wigner_3j_symmetry():
    """测试 Wigner-3j 的对称性 W(l,k,l') = W(l',k,l)。"""
    # Wigner-3j 对 (l,l') 对换对称
    w_01 = wigner_3j_squared(0, 1, 1)
    w_10 = wigner_3j_squared(1, 1, 0)
    assert np.isclose(w_01, w_10), f"Wigner-3j 对称性失败: W(0,1,1)^2={w_01}, W(1,1,0)^2={w_10}"


@pytest.mark.operator
@pytest.mark.quick
def test_get_coupling_factor_cache():
    """测试预计算缓存功能。"""
    # 使用缓存（s-s）
    a_cached = get_coupling_factor(0, 0, 0, use_cache=True)
    assert np.isclose(a_cached, 1.0)

    # 禁用缓存（应得到相同结果）
    a_direct = get_coupling_factor(0, 0, 0, use_cache=False)
    assert np.isclose(a_cached, a_direct)

    # 未预计算的值（d-d）
    a_dd = get_coupling_factor(2, 0, 2, use_cache=True)
    assert np.isfinite(a_dd) and a_dd > 0


@pytest.mark.operator
def test_coupling_factor_normalization():
    """测试耦合因子的归一化和。

    对 p-p (l=1, l'=1)，所有允许 k 的 a_k 之和应满足物理约束。
    """
    # p-p 允许 k=[0,2]
    a0 = coupling_factor_ak(1, 0, 1)
    a2 = coupling_factor_ak(1, 2, 1)

    # a_0 + a_2 = 1.0 + 2/5 = 1.0 + 0.4 = 1.4
    total = a0 + a2
    expected = 1.4
    assert np.isclose(total, expected, rtol=1e-6), \
        f"p-p 耦合因子归一化失败: {total} vs {expected}"
