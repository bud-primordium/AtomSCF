import numpy as np

from atomscf.xc.vwn import lda_c_vwn


def finite_diff_eps(rs: np.ndarray, up: float, dn: float, delta: float = 1e-3):
    """对 n 做小扰动，检查 e_c 的稳定性（冒烟测试）。"""
    n = 3.0 / (4.0 * np.pi * rs ** 3)
    n_up = up * n
    n_dn = dn * n
    eps, vcu, vcd, ec = lda_c_vwn(n_up, n_dn)
    # 轻微扰动 rs，查看 eps 变化是否有限且无 NaN
    rs2 = rs * (1.0 + delta)
    n2 = 3.0 / (4.0 * np.pi * rs2 ** 3)
    eps2, _, _, _ = lda_c_vwn(up * n2, dn * n2)
    return eps, eps2


def test_vwn_eps_no_nan_and_reasonable_change():
    rs_vals = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    eps1, eps2 = finite_diff_eps(rs_vals, up=0.6, dn=0.4)
    assert np.all(np.isfinite(eps1))
    assert np.all(np.isfinite(eps2))
    # 变化不应巨大（粗略阈值，仅冒烟）
    assert np.all(np.abs(eps2 - eps1) < 1.0)

