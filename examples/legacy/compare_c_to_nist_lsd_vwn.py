"""对比碳原子（Z=6）LSDA（VWN）与 NIST LSD（VWN）参考数据。

运行：

  PYTHONPATH=src python examples/compare_c_to_nist_lsd_vwn.py
"""
from __future__ import annotations

import numpy as np

from atomscf.grid import radial_grid_linear
from atomscf.scf import SCFConfig, run_lsda_vwn


def main() -> None:
    # 计算（与 run_c_lsda_vwn.py 一致的轻量配置）
    r, w = radial_grid_linear(n=1200, rmin=1e-6, rmax=70.0)
    cfg = SCFConfig(Z=6, r=r, w=w, mix_alpha=0.5, tol=5e-5, maxiter=140, eigs_per_l=2, lmax=2, compute_all_l=True, xc="VWN")
    res = run_lsda_vwn(cfg)

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
    print("对比（本项目 VWN vs NIST LSD=VWN）：")
    for k in ["E_total", "1s_up", "1s_dn", "2s_up", "2s_dn", "2p_up", "2p_dn"]:
        v_ref = nist[k]
        v_ours = ours[k]
        diff = v_ours - v_ref
        print(f"  {k:7s}: ours={v_ours: .6f}, nist={v_ref: .6f}, diff={diff: .6f}")

    # 输出我方能量分解（无 NIST 参考项）
    if res.energies:
        E = res.energies
        print("\n我方能量分解（Ha）：")
        for k in ["E_total", "E_kin", "E_coul", "E_ext", "E_x", "E_c"]:
            if k in E:
                print(f"  {k:7s}: {E[k]: .6f}")


if __name__ == "__main__":
    main()
