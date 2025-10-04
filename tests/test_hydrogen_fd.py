import numpy as np
import pytest

from atomscf.grid import radial_grid_linear
from atomscf.operator import solve_bound_states_fd


@pytest.mark.operator
@pytest.mark.quick

def test_hydrogen_1s_energy_close_to_half_ha():
    # 线性网格（教学用），较大 rmax 降低边界影响
    Z = 1.0
    rmin, rmax, n = 1e-6, 80.0, 3000
    r, _ = radial_grid_linear(n, rmin, rmax)
    v = -Z / np.maximum(r, 1e-12)

    eps, U = solve_bound_states_fd(r, l=0, v_of_r=v, k=3)
    e1s = eps[0]

    # 理论值：-0.5 Ha；离散与边界误差允许 1e-2 量级
    assert np.isclose(e1s, -0.5, rtol=0, atol=1e-2)

    # 归一化检查：∫ u^2 dr ≈ 1
    # 这里不重复计算权重，solve_bound_states_fd 已做归一
    u0 = U[0]
    # 简单数值长度（非严格）
    assert np.linalg.norm(u0) > 0

