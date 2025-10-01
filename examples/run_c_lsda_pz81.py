"""碳原子（Z=6）LSDA（Dirac 交换 + PZ81 关联）自洽示例。

运行示例：

    PYTHONPATH=src python examples/run_c_lsda_pz81.py

会打印 SCF 迭代进度与能量分解摘要。
"""
from __future__ import annotations

import numpy as np

from atomscf.grid import radial_grid_linear
from atomscf.scf import SCFConfig, run_lsda_pz81
from atomscf.occupations import default_occupations
from atomscf.io import export_levels_csv, export_energies_json, export_wavefunctions_csv


def main() -> None:
    r, w = radial_grid_linear(n=1000, rmin=1e-6, rmax=70.0)
    cfg = SCFConfig(Z=6, r=r, w=w, mix_alpha=0.5, tol=5e-5, maxiter=120, eigs_per_l=2, lmax=2, compute_all_l=True)
    res = run_lsda_pz81(cfg, verbose=True, progress_every=5)

    # 能级摘要（自旋分辨；m 已平均，报告径向通道）
    e_1s_up = res.eps_by_l_sigma[(0, "up")][0]
    e_2s_up = res.eps_by_l_sigma[(0, "up")][1]
    e_2p_up = res.eps_by_l_sigma[(1, "up")][0]
    e_1s_dn = res.eps_by_l_sigma.get((0, "down"), [np.nan]*2)[0]
    e_2s_dn = res.eps_by_l_sigma.get((0, "down"), [np.nan]*2)[1]
    e_2p_dn = res.eps_by_l_sigma.get((1, "down"), [np.nan]) [0]
    print("能级（Ha，径向通道，自旋分辨）：")
    print(f"  up: 1s={e_1s_up:.6f}, 2s={e_2s_up:.6f}, 2p={e_2p_up:.6f}")
    print(f"  dn: 1s={e_1s_dn:.6f}, 2s={e_2s_dn:.6f}, 2p={e_2p_dn:.6f}")

    # 导出数据
    occ = default_occupations(6)
    export_levels_csv("outputs/levels_c_lsda_pz81.csv", res.eps_by_l_sigma, occ)
    if res.energies:
        export_energies_json("outputs/energies_c_lsda_pz81.json", res.energies)
    export_wavefunctions_csv("outputs/wavefuncs_c", r, res.u_by_l_sigma, which=[(0,"up",0),(0,"up",1),(1,"up",0)])
    print("已导出: outputs/levels_c_lsda_pz81.csv, outputs/energies_c_lsda_pz81.json, outputs/wavefuncs_c/*.csv")

    # 电子数
    r2 = r * r
    Ne = 4.0 * np.pi * np.sum(w * (res.n_up + res.n_dn) * r2)
    print(f"电子数 ≈ {Ne:.6f}")

    # 能量分解
    E = res.energies or {}
    print("能量分解 (Ha):")
    for k in ["E_total", "E_ext", "E_H", "E_x", "E_c", "E_sum"]:
        if k in E:
            print(f"  {k}: {E[k]:.6f}")


if __name__ == "__main__":
    main()
