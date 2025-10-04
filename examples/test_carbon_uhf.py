#!/usr/bin/env python3
"""测试碳原子 UHF（³P 基态）"""

import sys
import numpy as np

sys.path.insert(0, "src")

from atomscf.grid import radial_grid_linear
from atomscf.scf_hf import run_hf_scf, HFSCFGeneralConfig

print("=" * 70)
print("碳原子 (Z=6) UHF 测试: 1s² 2s² 2p²")
print("=" * 70)

r, w = radial_grid_linear(n=1200, rmin=1e-6, rmax=70.0)

# UHF 配置：模拟 ³P 态（2 个未配对 p 电子）
cfg_uhf = HFSCFGeneralConfig(
    Z=6,
    r=r, w=w,
    occ_by_l={
        0: [2.0, 2.0],  # 1s² 2s²
        1: [2.0],       # 2p² (总占据)
    },
    occ_by_l_spin={
        0: {
            'up': [1.0, 1.0],    # 1s↑ 2s↑
            'down': [1.0, 1.0],  # 1s↓ 2s↓
        },
        1: {
            'up': [2.0],    # 2p↑↑ (2 个未配对电子)
            'down': [0.0],  # 2p↓ 无占据
        },
    },
    eigs_per_l={0: 2, 1: 1},
    spin_mode='UHF',
    mix_alpha=0.3,
    tol=1e-6,
    maxiter=150
)

print("\n运行 UHF SCF...")
res_uhf = run_hf_scf(cfg_uhf)

print(f"\n收敛状态: {res_uhf.converged}, 迭代次数: {res_uhf.iterations}")
print(f"\n总能量: E_total = {res_uhf.E_total:.6f} Ha")
print("\n能量分解:")
print(f"  E_kinetic  = {res_uhf.E_kinetic:.6f} Ha")
print(f"  E_ext      = {res_uhf.E_ext:.6f} Ha")
print(f"  E_hartree  = {res_uhf.E_hartree:.6f} Ha")
print(f"  E_exchange = {res_uhf.E_exchange:.6f} Ha")

print("\n轨道能级 (自旋分辨):")
print(f"  ε_1s(up)  = {res_uhf.eigenvalues_by_l_spin[(0, 'up')][0]:.6f} Ha")
print(f"  ε_2s(up)  = {res_uhf.eigenvalues_by_l_spin[(0, 'up')][1]:.6f} Ha")
print(f"  ε_2p(up)  = {res_uhf.eigenvalues_by_l_spin[(1, 'up')][0]:.6f} Ha")
print(f"  ε_1s(dn)  = {res_uhf.eigenvalues_by_l_spin[(0, 'down')][0]:.6f} Ha")
print(f"  ε_2s(dn)  = {res_uhf.eigenvalues_by_l_spin[(0, 'down')][1]:.6f} Ha")
print(f"  ε_2p(dn)  = {res_uhf.eigenvalues_by_l_spin[(1, 'down')][0]:.6f} Ha")

# 对比 Clementi & Roetti (1974) ROHF 参考值
E_clementi = -37.6886  # Ha (ROHF, ³P)
eps_1s_clementi = -11.32554
eps_2s_clementi = -0.70563
eps_2p_clementi = -0.43335

print("\n" + "=" * 70)
print("与 Clementi & Roetti (1974) ROHF 对比:")
print("=" * 70)
print(f"参考值: E_total = {E_clementi:.4f} Ha")
print(f"UHF:    E_total = {res_uhf.E_total:.4f} Ha")
print(f"误差:   ΔE = {(res_uhf.E_total - E_clementi)*1000:.2f} mHa")
print(f"相对误差: {abs((res_uhf.E_total - E_clementi)/E_clementi)*100:.3f}%")

print(f"\n轨道能级对比:")
print(f"  ε_2p(Clementi) = {eps_2p_clementi:.5f} Ha")
print(f"  ε_2p(UHF↑)     = {res_uhf.eigenvalues_by_l_spin[(1, 'up')][0]:.5f} Ha")
print(f"  Δε_2p          = {abs(res_uhf.eigenvalues_by_l_spin[(1, 'up')][0] - eps_2p_clementi)*1000:.2f} mHa")
