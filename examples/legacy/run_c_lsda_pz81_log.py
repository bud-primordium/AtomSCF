"""碳原子（Z=6）LSDA（Dirac 交换 + PZ81 关联）自洽示例（对数网格）。

运行：

  PYTHONPATH=src python examples/run_c_lsda_pz81_log.py
"""
from __future__ import annotations

import numpy as np

from atomscf.grid import radial_grid_log
from atomscf.scf import SCFConfig, run_lsda_pz81
from atomscf.occupations import default_occupations
from atomscf.io import export_levels_csv, export_energies_json


def main() -> None:
    # 对数网格：更小 rmin、更大 rmax 与更多点有助于近核与尾部精度
    r, w = radial_grid_log(n=2500, rmin=1e-8, rmax=90.0)
    cfg = SCFConfig(
        Z=6,
        r=r,
        w=w,
        mix_alpha=0.35,
        tol=3e-5,
        maxiter=150,
        eigs_per_l=2,
        lmax=2,
        compute_all_l=True,
    )
    res = run_lsda_pz81(cfg, verbose=True, progress_every=10)

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
    export_levels_csv("outputs/levels_c_lsda_pz81_log.csv", res.eps_by_l_sigma, default_occupations(6))
    if res.energies:
        export_energies_json("outputs/energies_c_lsda_pz81_log.json", res.energies)
    print("已导出: outputs/levels_c_lsda_pz81_log.csv, outputs/energies_c_lsda_pz81_log.json")


if __name__ == "__main__":
    main()
