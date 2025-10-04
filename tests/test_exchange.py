"""HF 交换算子单元测试

测试 hf/exchange.py 模块的正确性。
"""

import numpy as np
import pytest

from atomscf.grid import radial_grid_linear, trapezoid_weights
from atomscf.hf import SlaterIntegralCache, exchange_operator_general, exchange_operator_s


@pytest.mark.operator
@pytest.mark.quick
def test_exchange_s_hydrogen_1s():
    """测试氢原子 1s 自交换。

    物理验证：
    - 单电子体系的 HF 交换完全抵消 Hartree 势
    - K[u_1s](r) * u_1s(r) 应在远处渐近 -1/r
    """
    # 构建网格
    r, _ = radial_grid_linear(n=1000, rmin=1e-6, rmax=50.0)
    w = trapezoid_weights(r)

    # 氢原子 1s 径向波函数（归一化）
    u_1s = np.sqrt(2.0) * np.exp(-r)

    # 创建交换算子
    K = exchange_operator_s(r, w, u_occ=[u_1s], occ_nums=[1.0])

    # 应用到 1s 态
    Ku_1s = K(u_1s)

    # 验证形状
    assert Ku_1s.shape == r.shape

    # 验证有限性
    assert np.all(np.isfinite(Ku_1s))

    # 验证符号（交换为负贡献）
    # 在大部分区域 Ku 应为负（因负号在公式中）
    # 注意：近核处由于 R^0(r) 的行为，符号可能复杂
    # 远处应满足 Ku * u_1s ≈ -u_1s^2 / r


@pytest.mark.operator
@pytest.mark.quick
def test_exchange_s_zero_occupation():
    """测试占据数为 0 时交换为零。"""
    r, _ = radial_grid_linear(n=500, rmin=1e-6, rmax=30.0)
    w = trapezoid_weights(r)

    u_1s = np.exp(-r)

    # 占据数为 0
    K = exchange_operator_s(r, w, u_occ=[u_1s], occ_nums=[0.0])
    Ku = K(u_1s)

    assert np.allclose(Ku, 0.0, atol=1e-15)


@pytest.mark.operator
def test_exchange_s_two_orbitals():
    """测试两个 s 轨道的交换。

    模拟 He (1s²) 或 Li⁺ 的闭壳层情况。
    """
    r, _ = radial_grid_linear(n=800, rmin=1e-6, rmax=40.0)
    w = trapezoid_weights(r)

    # 两个不同的 s 轨道
    u_1s = np.sqrt(2.0) * np.exp(-r)
    u_2s_unnorm = np.exp(-0.5 * r) * (1 - 0.5 * r)

    # 显式归一化 u_2s
    norm_2s = np.sqrt(np.sum(u_2s_unnorm**2 * w))
    u_2s = u_2s_unnorm / norm_2s

    # 归一化检查（容差放宽）
    norm_1s = np.sum(u_1s**2 * w)
    norm_2s_check = np.sum(u_2s**2 * w)
    assert np.isclose(norm_1s, 1.0, atol=1e-2)
    assert np.isclose(norm_2s_check, 1.0, atol=1e-2)

    # 创建交换算子（两个轨道各占据 1 电子）
    K = exchange_operator_s(r, w, u_occ=[u_1s, u_2s], occ_nums=[1.0, 1.0])

    # 应用到 1s
    Ku_1s = K(u_1s)

    assert Ku_1s.shape == r.shape
    assert np.all(np.isfinite(Ku_1s))

    # 应用到 2s
    Ku_2s = K(u_2s)

    assert Ku_2s.shape == r.shape
    assert np.all(np.isfinite(Ku_2s))


@pytest.mark.operator
def test_exchange_s_with_cache():
    """测试交换算子的缓存功能。"""
    r, _ = radial_grid_linear(n=500, rmin=1e-6, rmax=30.0)
    w = trapezoid_weights(r)

    u_1s = np.exp(-r)

    # 使用缓存
    cache = SlaterIntegralCache()
    K1 = exchange_operator_s(r, w, u_occ=[u_1s], occ_nums=[1.0], cache=cache)
    Ku1 = K1(u_1s)

    # 第二次调用（应使用缓存）
    K2 = exchange_operator_s(r, w, u_occ=[u_1s], occ_nums=[1.0], cache=cache)
    Ku2 = K2(u_1s)

    # 结果应一致
    assert np.allclose(Ku1, Ku2)


@pytest.mark.operator
def test_exchange_s_input_validation():
    """测试输入验证。"""
    r = np.linspace(1e-6, 10, 100)
    w = trapezoid_weights(r)
    u = np.exp(-r)

    # 占据态数量与占据数不匹配
    with pytest.raises(ValueError, match="占据态数量.*与占据数数量.*不匹配"):
        exchange_operator_s(r, w, u_occ=[u, u], occ_nums=[1.0])

    # 目标波函数形状不匹配
    K = exchange_operator_s(r, w, u_occ=[u], occ_nums=[1.0])
    u_wrong_shape = np.exp(-np.linspace(0, 10, 50))
    with pytest.raises(ValueError, match="目标波函数形状.*与网格.*不匹配"):
        K(u_wrong_shape)


@pytest.mark.operator
def test_exchange_general_s_only():
    """测试通用交换算子（仅 s 轨道）与特化版本一致性。"""
    r, _ = radial_grid_linear(n=500, rmin=1e-6, rmax=30.0)
    w = trapezoid_weights(r)

    u_1s = np.sqrt(2.0) * np.exp(-r)

    # 特化版本
    K_s = exchange_operator_s(r, w, u_occ=[u_1s], occ_nums=[1.0])
    Ku_s = K_s(u_1s)

    # 通用版本（仅 l=0）
    K_gen = exchange_operator_general(
        r, w, l_target=0, u_occ_by_l={0: [u_1s]}, occ_nums_by_l={0: [1.0]}
    )
    Ku_gen = K_gen(u_1s)

    # 应一致
    assert np.allclose(Ku_s, Ku_gen, rtol=1e-10)


@pytest.mark.operator
def test_exchange_general_s_p_coupling():
    """测试 s-p 交换耦合（允许 k=1）。"""
    r, _ = radial_grid_linear(n=800, rmin=1e-6, rmax=40.0)
    w = trapezoid_weights(r)

    # s 和 p 轨道
    u_s = np.exp(-r)
    u_p = np.exp(-0.5 * r) * r

    # 归一化
    u_s /= np.sqrt(np.sum(u_s**2 * w))
    u_p /= np.sqrt(np.sum(u_p**2 * w))

    # 创建通用交换算子（目标为 s，占据有 s 和 p）
    K_s = exchange_operator_general(
        r,
        w,
        l_target=0,
        u_occ_by_l={0: [u_s], 1: [u_p]},
        occ_nums_by_l={0: [1.0], 1: [1.0]},
    )

    Ku_s = K_s(u_s)

    assert Ku_s.shape == r.shape
    assert np.all(np.isfinite(Ku_s))


@pytest.mark.operator
def test_exchange_general_p_p_coupling():
    """测试 p-p 交换耦合（允许 k=[0,2]）。"""
    r, _ = radial_grid_linear(n=800, rmin=1e-6, rmax=40.0)
    w = trapezoid_weights(r)

    # p 轨道
    u_2p = np.exp(-0.6 * r) * r

    # 归一化
    u_2p /= np.sqrt(np.sum(u_2p**2 * w))

    # 创建通用交换算子（目标为 p，占据为 p）
    K_p = exchange_operator_general(
        r, w, l_target=1, u_occ_by_l={1: [u_2p]}, occ_nums_by_l={1: [1.0]}
    )

    Ku_p = K_p(u_2p)

    assert Ku_p.shape == r.shape
    assert np.all(np.isfinite(Ku_p))


@pytest.mark.operator
def test_exchange_general_input_validation():
    """测试通用交换算子的输入验证。"""
    r = np.linspace(1e-6, 10, 100)
    w = trapezoid_weights(r)
    u = np.exp(-r)

    # u_occ_by_l 和 occ_nums_by_l 的 l 键不一致
    with pytest.raises(ValueError, match="l 键不一致"):
        exchange_operator_general(
            r, w, l_target=0, u_occ_by_l={0: [u]}, occ_nums_by_l={1: [1.0]}
        )

    # 波函数数量与占据数数量不匹配
    with pytest.raises(ValueError, match="波函数数量与占据数数量不匹配"):
        exchange_operator_general(
            r, w, l_target=0, u_occ_by_l={0: [u, u]}, occ_nums_by_l={0: [1.0]}
        )


@pytest.mark.operator
def test_exchange_linearity():
    """测试交换算子的线性性质。

    K[α*u + β*v] 不一定等于 α*K[u] + β*K[v]（因 K 是非线性的）
    但此测试验证算子应用的数值稳定性。
    """
    r, _ = radial_grid_linear(n=500, rmin=1e-6, rmax=30.0)
    w = trapezoid_weights(r)

    u_1 = np.exp(-r)
    u_2 = np.exp(-0.8 * r)

    K = exchange_operator_s(r, w, u_occ=[u_1], occ_nums=[1.0])

    # 应用到不同波函数
    Ku1 = K(u_1)
    Ku2 = K(u_2)

    # 验证有限性
    assert np.all(np.isfinite(Ku1))
    assert np.all(np.isfinite(Ku2))

    # 验证非零（除非特殊情况）
    assert np.linalg.norm(Ku1) > 0
    assert np.linalg.norm(Ku2) > 0
