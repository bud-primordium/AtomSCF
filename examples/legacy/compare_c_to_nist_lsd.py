"""对比碳原子（Z=6）LSDA（PZ81 当前实现）与 NIST LSD（VWN）参考数据。

运行：

  PYTHONPATH=src python examples/compare_c_to_nist_lsd.py

注意：NIST 的 LSD 使用 VWN 关联，本项目当前脚本使用 PZ81，存在系统差异；
后续将加入 VWN 选项以更好对齐 NIST。
"""
from __future__ import annotations

import numpy as np

from atomscf.grid import radial_grid_linear
from atomscf.scf import SCFConfig, run_lsda_pz81


def main() -> None:
    # 计算
    r, w = radial_grid_linear(n=1000, rmin=1e-6, rmax=70.0)
    cfg = SCFConfig(Z=6, r=r, w=w, mix_alpha=0.5, tol=5e-5, maxiter=120, eigs_per_l=2, lmax=2, compute_all_l=True)
    res = run_lsda_pz81(cfg)

    # NIST LSD (VWN) 参考（单位 Ha）
    nist = {
        "E_total": -37.470031,
        "1s_up": -9.940546,
        "1s_dn": -9.905802,
        "2s_up": -0.531276,
        "2s_dn": -0.435066,
        "2p_up": -0.227557,
        "2p_dn": -0.139285,
    }

    # 提取本项目结果
    e_1s_up = float(res.eps_by_l_sigma[(0, "up")][0])
    e_1s_dn = float(res.eps_by_l_sigma[(0, "down")][0])
    e_2s_up = float(res.eps_by_l_sigma[(0, "up")][1])
    e_2s_dn = float(res.eps_by_l_sigma[(0, "down")][1])
    e_2p_up = float(res.eps_by_l_sigma[(1, "up")][0])
    e_2p_dn = float(res.eps_by_l_sigma[(1, "down")][0])
    E_total = float(res.energies.get("E_total", np.nan)) if res.energies else np.nan

    ours = {
        "E_total": E_total,
        "1s_up": e_1s_up,
        "1s_dn": e_1s_dn,
        "2s_up": e_2s_up,
        "2s_dn": e_2s_dn,
        "2p_up": e_2p_up,
        "2p_dn": e_2p_dn,
    }

    # 打印对比
    print("对比（本项目 PZ81 vs NIST LSD=VWN）：")
    for k in ["E_total", "1s_up", "1s_dn", "2s_up", "2s_dn", "2p_up", "2p_dn"]:
        v_ref = nist[k]
        v_ours = ours[k]
        diff = v_ours - v_ref
        print(f"  {k:7s}: ours={v_ours: .6f}, nist={v_ref: .6f}, diff={diff: .6f}")


if __name__ == "__main__":
    main()

