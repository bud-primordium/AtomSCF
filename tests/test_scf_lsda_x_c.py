import numpy as np
import pytest

from atomscf.grid import radial_grid_linear
from atomscf.scf import SCFConfig, run_lsda_x_only


@pytest.mark.scf

def test_lsda_x_only_carbon_ordering_and_electron_count():
    r, w = radial_grid_linear(n=1000, rmin=1e-6, rmax=60.0)
    cfg = SCFConfig(Z=6, r=r, w=w, mix_alpha=0.5, tol=5e-5, maxiter=100, eigs_per_l=2)
    res = run_lsda_x_only(cfg)

    assert res.converged, f"SCF 未收敛，迭代步数={res.iterations}"

    # 电子数守恒（C: 6）
    n_tot = res.n_up + res.n_dn
    Ne = 4.0 * np.pi * np.sum(w * n_tot * (r ** 2))
    assert np.isclose(Ne, 6.0, atol=6e-2)

    # 能级序：1s < 2s < 2p （取上自旋通道，各自第一个径向态）
    e_1s = res.eps_by_l_sigma[(0, "up")][0]
    e_2s = res.eps_by_l_sigma[(0, "up")][1]
    e_2p = res.eps_by_l_sigma[(1, "up")][0]
    assert e_1s < e_2s < e_2p
