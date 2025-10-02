"""碳原子（Z=6）LSDA（Dirac 交换 + VWN 关联）自洽示例（对数网格）。

运行：

  PYTHONPATH=src python examples/run_c_lsda_vwn_log.py
"""
from __future__ import annotations

import numpy as np

from atomscf.grid import radial_grid_log, radial_grid_mixed
from atomscf.scf import SCFConfig, run_lsda_vwn
from atomscf.occupations import default_occupations
from atomscf.io import export_levels_csv, export_energies_json


def main() -> None:
    # 标准档“混合网格”：更高近核分辨率与更平滑的切换点
    r, w = radial_grid_mixed(
        n_inner=1500,
        n_outer=500,
        rmin=1e-7,
        r_switch=10.0,
        rmax=100.0,
    )
    cfg = SCFConfig(
        Z=6,
        r=r,
        w=w,
        mix_alpha=0.35,
        tol=1e-6,
        maxiter=160,
        eigs_per_l=2,
        lmax=2,
        compute_all_l=True,
        compute_all_l_mode="final",
        mix_kind="density",
        adapt_mixing=True,
        xc="VWN",
    )
    res = run_lsda_vwn(cfg, verbose=True, progress_every=10)

    # 能级（自旋分辨）
    e_1s_up = res.eps_by_l_sigma[(0, "up")][0]
    e_1s_dn = res.eps_by_l_sigma[(0, "down")][0]
    e_2s_up = res.eps_by_l_sigma[(0, "up")][1]
    e_2s_dn = res.eps_by_l_sigma[(0, "down")][1]
    e_2p_up = res.eps_by_l_sigma[(1, "up")][0]
    e_2p_dn = res.eps_by_l_sigma[(1, "down")][0]
    print(f"能级（Ha） up: 1s={e_1s_up:.6f}, 2s={e_2s_up:.6f}, 2p={e_2p_up:.6f}")
    print(f"能级（Ha） dn: 1s={e_1s_dn:.6f}, 2s={e_2s_dn:.6f}, 2p={e_2p_dn:.6f}")

    # 导出
    export_levels_csv("outputs/levels_c_lsda_vwn_log.csv", res.eps_by_l_sigma, default_occupations(6))
    if res.energies:
        export_energies_json("outputs/energies_c_lsda_vwn_log.json", res.energies)
    print("已导出: outputs/levels_c_lsda_vwn_log.csv, outputs/energies_c_lsda_vwn_log.json")


if __name__ == "__main__":
    main()
