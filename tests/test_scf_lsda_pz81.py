import numpy as np
import pytest

from atomscf.grid import radial_grid_linear
from atomscf.scf import SCFConfig, run_lsda_pz81


@pytest.mark.scf

def test_lsda_pz81_hydrogen_energy_components_and_homo():
    r, w = radial_grid_linear(n=1000, rmin=1e-6, rmax=60.0)
    cfg = SCFConfig(Z=1, r=r, w=w, mix_alpha=0.35, tol=1e-5, maxiter=80, eigs_per_l=1)
    res = run_lsda_pz81(cfg)

    assert res.converged
    # 电子数守恒
    n_tot = res.n_up + res.n_dn
    Ne = 4.0 * np.pi * np.sum(w * n_tot * (r ** 2))
    assert np.isclose(Ne, 1.0, atol=2e-2)
    # HOMO < 0
    e_1s = res.eps_by_l_sigma[(0, "up")][0]
    assert e_1s < 0.0
    # 能量项存在且有限
    E = res.energies
    assert E is not None
    for k in ["E_total", "E_H", "E_x", "E_c", "E_ext", "E_sum"]:
        assert k in E and np.isfinite(E[k])


@pytest.mark.scf

def test_lsda_pz81_carbon_quick():
    r, w = radial_grid_linear(n=1000, rmin=1e-6, rmax=70.0)
    cfg = SCFConfig(Z=6, r=r, w=w, mix_alpha=0.5, tol=5e-5, maxiter=120, eigs_per_l=2)
    res = run_lsda_pz81(cfg)
    assert res.converged
    n_tot = res.n_up + res.n_dn
    Ne = 4.0 * np.pi * np.sum(w * n_tot * (r ** 2))
    assert np.isclose(Ne, 6.0, atol=8e-2)

