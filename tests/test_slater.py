"""Slater 径向积分单元测试

测试 hf/slater.py 模块的正确性和数值稳定性。
"""

import numpy as np
import pytest

from atomscf.grid import radial_grid_linear, trapezoid_weights
from atomscf.hf.slater import (
    SlaterIntegralCache,
    slater_integral_k0,
    slater_integral_radial,
)


@pytest.mark.operator
@pytest.mark.quick
def test_slater_k0_hydrogen_1s():
    """测试 k=0 Slater 积分：氢原子 1s 态自交换。

    理论验证：
    对氢样 1s: u(r) = 2 exp(-r) / sqrt(4π)
    归一化: ∫ u² dr = 1
    库仑积分: R⁰(r→∞) → 1/r（总电荷为1）
    """
    # 构建网格
    r, _ = radial_grid_linear(n=1000, rmin=1e-6, rmax=50.0)
    w = trapezoid_weights(r)

    # 氢样 1s 径向波函数（解析形式）
    # u(r) = r * R(r)，对 1s: R₁₀(r) = 2*exp(-r)
    # 归一化: ∫ u² dr = 1  =>  u(r) = sqrt(2) * exp(-r)
    u_1s = np.sqrt(2.0) * np.exp(-r)

    # 归一化检查（允许数值积分误差）
    norm = np.sum(u_1s**2 * w)
    assert np.isclose(norm, 1.0, atol=1e-3), f"归一化失败: {norm}"

    # 计算 Slater 积分
    R0 = slater_integral_k0(r, w, u_1s, u_1s)

    # 验证1: 远处渐近行为 R⁰(r→∞) ≈ 1/r
    # 取 r = 30 处（足够远，波函数已衰减）
    idx_far = np.argmin(np.abs(r - 30.0))
    expected_far = 1.0 / r[idx_far]
    assert np.isclose(R0[idx_far], expected_far, rtol=0.05), \
        f"远处渐近失败: R0({r[idx_far]:.1f}) = {R0[idx_far]:.6f}, 期望 ≈ {expected_far:.6f}"

    # 验证2: 单调性（R⁰ 应单调递减）
    assert np.all(np.diff(R0) <= 0), "R⁰ 应单调递减"

    # 验证3: 正定性
    assert np.all(R0 > 0), "R⁰ 必须为正"


@pytest.mark.operator
def test_slater_k1_s_p_cross():
    """测试 k=1 Slater 积分：s-p 交叉项。

    物理约束：
    - s (l=0) 与 p (l=1) 的交叉项允许 k=1
    - 数值应收敛
    """
    r, _ = radial_grid_linear(n=800, rmin=1e-6, rmax=40.0)
    w = trapezoid_weights(r)

    # 简化波函数（不需要精确）
    u_s = np.exp(-r) * r**0  # s 型
    u_p = np.exp(-0.5 * r) * r**1  # p 型

    # 归一化
    u_s /= np.sqrt(np.sum(u_s**2 * w))
    u_p /= np.sqrt(np.sum(u_p**2 * w))

    # 计算 k=1 积分
    R1 = slater_integral_radial(r, w, u_s, u_p, k=1)

    # 基本检查
    assert R1.shape == r.shape
    assert np.all(np.isfinite(R1)), "R¹ 包含 NaN 或 Inf"
    assert np.all(R1 >= 0), "R¹ 应为非负（对归一化波函数）"


@pytest.mark.operator
def test_slater_k2_p_p():
    """测试 k=2 Slater 积分：p-p 交换。"""
    r, _ = radial_grid_linear(n=800, rmin=1e-6, rmax=40.0)
    w = trapezoid_weights(r)

    # p 型波函数
    u_2p = np.exp(-0.6 * r) * r

    # 归一化
    u_2p /= np.sqrt(np.sum(u_2p**2 * w))

    # 计算 k=2 积分
    R2 = slater_integral_radial(r, w, u_2p, u_2p, k=2)

    # 基本检查
    assert R2.shape == r.shape
    assert np.all(np.isfinite(R2))
    assert np.all(R2 >= 0)


@pytest.mark.operator
def test_slater_boundary_r_near_zero():
    """测试边界情况：r→0 数值稳定性。"""
    # 包含非常小的 r 值
    r = np.logspace(-10, 1, 500)
    w = trapezoid_weights(r)

    # 简单波函数
    u = np.exp(-r) * (r + 1e-12) ** 0.5  # 避免 r=0 处发散

    # 计算 k=0,1,2
    R0 = slater_integral_radial(r, w, u, u, k=0)
    R1 = slater_integral_radial(r, w, u, u, k=1)
    R2 = slater_integral_radial(r, w, u, u, k=2)

    # 验证无 NaN/Inf
    assert np.all(np.isfinite(R0))
    assert np.all(np.isfinite(R1))
    assert np.all(np.isfinite(R2))


@pytest.mark.operator
def test_slater_cache():
    """测试 Slater 积分缓存功能。"""
    r, _ = radial_grid_linear(n=500, rmin=1e-6, rmax=30.0)
    w = trapezoid_weights(r)

    u_1s = np.exp(-r)
    u_2s = np.exp(-0.5 * r) * (1 - 0.5 * r)

    cache = SlaterIntegralCache()

    # 第一次计算（应缓存）
    R0_first = cache.get(r, w, u_1s, u_2s, k=0, i_index=0, j_index=1)

    # 第二次获取（应从缓存读取）
    R0_second = cache.get(r, w, u_1s, u_2s, k=0, i_index=0, j_index=1)

    # 验证结果一致
    assert np.allclose(R0_first, R0_second)

    # 验证缓存命中
    assert len(cache) == 1

    # 清空缓存
    cache.clear()
    assert len(cache) == 0


@pytest.mark.operator
def test_slater_invalid_inputs():
    """测试无效输入的错误处理。"""
    r = np.linspace(1e-6, 10, 100)
    w = trapezoid_weights(r)
    u = np.exp(-r)

    # 负 k 值
    with pytest.raises(ValueError, match="k 必须非负"):
        slater_integral_radial(r, w, u, u, k=-1)

    # 长度不匹配
    with pytest.raises(ValueError, match="输入数组长度不一致"):
        slater_integral_radial(r, w[:-10], u, u, k=0)


@pytest.mark.operator
def test_slater_symmetry():
    """测试对称性：R^k_{ij} = R^k_{ji}。"""
    r, _ = radial_grid_linear(n=500, rmin=1e-6, rmax=30.0)
    w = trapezoid_weights(r)

    u_i = np.exp(-r)
    u_j = np.exp(-0.8 * r) * r

    for k in [0, 1, 2]:
        R_ij = slater_integral_radial(r, w, u_i, u_j, k)
        R_ji = slater_integral_radial(r, w, u_j, u_i, k)

        assert np.allclose(R_ij, R_ji, atol=1e-10), f"对称性失败 k={k}"
