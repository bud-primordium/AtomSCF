import numpy as np

from atomscf.grid import radial_grid_log
from atomscf.operator import solve_bound_states_fd5_auxlinear


def test_hydrogen_log_grid_fd5_aux_1s_energy():
    # 对数等距网格 + 辅助线性 FD5，应能较准确给出氢样 1s ≈ -0.5 Ha
    r, _ = radial_grid_log(n=800, rmin=1e-6, rmax=60.0)
    v = -1.0 / np.maximum(r, 1e-30)
    eps, _ = solve_bound_states_fd5_auxlinear(r, l=0, v_of_r=v, k=1)
    # 允许 ~7e-2 的误差（快速参数，n_aux 默认为 >= len(r)）
    assert np.isclose(float(eps[0]), -0.5, atol=7e-2, rtol=0)
