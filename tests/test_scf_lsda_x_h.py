import numpy as np
import pytest

from atomscf.grid import radial_grid_linear
from atomscf.scf import SCFConfig, run_lsda_x_only


@pytest.mark.scf
@pytest.mark.quick

def test_lsda_x_only_hydrogen_electron_count_and_convergence():
    # 对数网格：小 r 加密，适合库仑势
    r, w = radial_grid_linear(n=1000, rmin=1e-6, rmax=60.0)
    cfg = SCFConfig(Z=1, r=r, w=w, mix_alpha=0.35, tol=1e-5, maxiter=80, eigs_per_l=1)
    res = run_lsda_x_only(cfg)

    assert res.converged, f"SCF 未收敛，迭代步数={res.iterations}"

    # 电子数守恒（H: 1）
    n_tot = res.n_up + res.n_dn
    Ne = 4.0 * np.pi * np.sum(w * n_tot * (r ** 2))
    assert np.isclose(Ne, 1.0, atol=2e-2)

    # 1s 能级应为最低能，负值（Kohn-Sham 本征值，不强行验证精确数值）
    e_s_up = res.eps_by_l_sigma[(0, "up")][0]
    assert e_s_up < 0.0
