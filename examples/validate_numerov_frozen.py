"""Numerov 冻结势验证脚本（对数等距网格）。

用途：
- 在纯库仑势 v(r)=-Z/r 与一次性冻结的 v_eff(r) 上，用 Numerov（匹配法）求解 l=0/1 的低能态，
  验证能级与波函数的数值质量；

运行：

  PYTHONPATH=src python examples/validate_numerov_frozen.py
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from atomscf.grid import radial_grid_log
from atomscf.numerov import numerov_find_k_log_matching
from atomscf.scf import SCFConfig, run_lsda_vwn
from atomscf.occupations import default_occupations
from atomscf.utils import trapz


def hydrogen_like_check(Z: int = 1) -> None:
    """在纯库仑势 -Z/r 上验证氢样能级（1s/2s/2p）。"""
    r, w = radial_grid_log(n=2000, rmin=1e-7, rmax=100.0)
    v = -float(Z) / np.maximum(r, 1e-30)
    # s 通道：取前 2 个（1s, 2s）
    eps_s, _ = numerov_find_k_log_matching(r, v, l=0, k=2, eps_min=-2.0 * Z * Z, eps_max=-1e-6, samples=180)
    # p 通道：取前 1 个（2p）
    eps_p, _ = numerov_find_k_log_matching(r, v, l=1, k=1, eps_min=-2.0 * Z * Z, eps_max=-1e-6, samples=180)

    # 理论：E_n = -Z^2/(2 n^2)
    e_1s_th = -Z * Z / 2.0
    e_2_th = -Z * Z / 8.0
    print(f"[Hydrogen-like Z={Z}] 1s(num)={eps_s[0]: .6f} vs th={e_1s_th: .6f} | 2s(num)={eps_s[1]: .6f}, 2p(num)={eps_p[0]: .6f}, th={e_2_th: .6f}")


def frozen_veff_check_carbon() -> None:
    """在碳原子一次性冻结的 v_eff 上用 Numerov 求解，检查深束缚态是否合理。"""
    # 先用线性基线做一次 SCF（目标只是得到一个冻结 v_eff，不追求严收敛）
    from atomscf.grid import radial_grid_linear
    from atomscf.scf import _build_effective_potential_vwn

    r_lin, w_lin = radial_grid_linear(n=1200, rmin=1e-6, rmax=70.0)
    cfg0 = SCFConfig(Z=6, r=r_lin, w=w_lin, mix_alpha=0.5, tol=5e-5, maxiter=60, eigs_per_l=2, lmax=2, compute_all_l=True, xc="VWN")
    res0 = run_lsda_vwn(cfg0, verbose=False)
    v_up, v_dn, *_ = _build_effective_potential_vwn(cfg0.Z, r_lin, res0.n_up, res0.n_dn, w_lin)
    # 插值到对数网格
    r_log, w_log = radial_grid_log(n=2000, rmin=1e-7, rmax=100.0)
    v_up_log = np.interp(r_log, r_lin, v_up)
    v_dn_log = np.interp(r_log, r_lin, v_dn)

    # s 通道取 2 个，p 通道取 1 个
    eps_s_up, _ = numerov_find_k_log_matching(r_log, v_up_log, l=0, k=2, eps_min=-50.0, eps_max=-1e-3, samples=160)
    eps_p_up, _ = numerov_find_k_log_matching(r_log, v_up_log, l=1, k=1, eps_min=-5.0, eps_max=-1e-3, samples=160)
    print(f"[Frozen v_eff (C, up)] 1s={eps_s_up[0]: .6f}, 2s={eps_s_up[1]: .6f}, 2p={eps_p_up[0]: .6f}")


def main() -> None:
    # 1) 氢样验证（Z=1,6）
    hydrogen_like_check(Z=1)
    hydrogen_like_check(Z=6)
    # 2) 冻结势（碳，VWN）
    frozen_veff_check_carbon()


if __name__ == "__main__":
    main()

