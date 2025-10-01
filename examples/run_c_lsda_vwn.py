"""碳原子（Z=6）LSDA（Dirac 交换 + VWN 关联）自洽与导出示例。

运行：
  PYTHONPATH=src python examples/run_c_lsda_vwn.py
"""
from __future__ import annotations

import numpy as np

from atomscf.grid import radial_grid_linear
from atomscf.scf import SCFConfig, run_lsda_vwn
from atomscf.occupations import default_occupations
from atomscf.io import export_levels_csv, export_energies_json


def main() -> None:
    r, w = radial_grid_linear(n=1200, rmin=1e-6, rmax=70.0)
    cfg = SCFConfig(Z=6, r=r, w=w, mix_alpha=0.5, tol=5e-5, maxiter=140, eigs_per_l=2, lmax=2, compute_all_l=True, xc="VWN")
    res = run_lsda_vwn(cfg, verbose=True, progress_every=5)

    # 能级（自旋分辨）
    def e(ls):
        return float(res.eps_by_l_sigma[ls][0]), float(res.eps_by_l_sigma[ls][1]) if len(res.eps_by_l_sigma[ls])>1 else np.nan
    e1s_up = res.eps_by_l_sigma[(0, "up")][0]
    e1s_dn = res.eps_by_l_sigma[(0, "down")][0]
    e2s_up = res.eps_by_l_sigma[(0, "up")][1]
    e2s_dn = res.eps_by_l_sigma[(0, "down")][1]
    e2p_up = res.eps_by_l_sigma[(1, "up")][0]
    e2p_dn = res.eps_by_l_sigma[(1, "down")][0]
    print(f"能级（Ha） up: 1s={e1s_up:.6f}, 2s={e2s_up:.6f}, 2p={e2p_up:.6f}")
    print(f"能级（Ha） dn: 1s={e1s_dn:.6f}, 2s={e2s_dn:.6f}, 2p={e2p_dn:.6f}")

    # 导出
    export_levels_csv("outputs/levels_c_lsda_vwn.csv", res.eps_by_l_sigma, default_occupations(6))
    if res.energies:
        export_energies_json("outputs/energies_c_lsda_vwn.json", res.energies)
    print("已导出: outputs/levels_c_lsda_vwn.csv, outputs/energies_c_lsda_vwn.json")


if __name__ == "__main__":
    main()

