import numpy as np
import pytest

from atomscf.grid import radial_grid_linear, radial_grid_log, trapezoid_weights


@pytest.mark.grid
@pytest.mark.quick
def test_linear_grid_integral_of_one():
    r, w = radial_grid_linear(101, 0.0, 2.0)
    # ∫_0^2 1 dr = 2
    assert np.isclose(np.sum(w), 2.0, rtol=0, atol=1e-12)


@pytest.mark.grid
@pytest.mark.quick
def test_log_grid_monotonic_and_weights():
    r, w = radial_grid_log(101, 1e-5, 1.0)
    assert np.all(np.diff(r) > 0)
    assert np.all(w > 0)


@pytest.mark.grid
@pytest.mark.quick
def test_trapezoid_weights_match_np_trapz():
    r = np.linspace(0.0, 1.0, 1001)
    w = trapezoid_weights(r)
    f = np.sin(2 * np.pi * r)
    int1 = np.sum(w * f)
    int2 = np.trapz(f, r)
    assert np.isclose(int1, int2, rtol=0, atol=1e-12)

