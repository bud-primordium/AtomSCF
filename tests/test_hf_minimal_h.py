import numpy as np

from atomscf.grid import radial_grid_linear
from atomscf.scf_hf import HFConfig, run_hf_minimal


def test_hf_minimal_hydrogen_1s():
    r, w = radial_grid_linear(1200, 1e-6, 80.0)
    res = run_hf_minimal(HFConfig(Z=1, r=r, w=w))
    assert res.converged
    # HF 单电子 1s 能级 ~ -0.5 Ha
    assert np.isclose(res.eps_1s, -0.5, atol=1e-2)
    # 波函数非零且有限
    assert np.isfinite(res.u_1s).all()

