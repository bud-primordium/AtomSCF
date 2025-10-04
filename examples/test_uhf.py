#!/usr/bin/env python3
"""测试 UHF 实现：He (闭壳层) 和 Li (开壳层)"""

import sys
import numpy as np

sys.path.insert(0, "src")

from atomscf.grid import radial_grid_linear
from atomscf.scf_hf import run_hf_scf, HFSCFGeneralConfig

def test_he_uhf_vs_rhf():
    """He 原子：UHF 应该等于 RHF（闭壳层）"""
    print("=" * 70)
    print("测试 1: He 原子 (闭壳层，1s²)")
    print("=" * 70)

    r, w = radial_grid_linear(n=800, rmin=1e-6, rmax=50.0)

    # RHF 配置
    cfg_rhf = HFSCFGeneralConfig(
        Z=2,
        r=r, w=w,
        occ_by_l={0: [2.0]},
        eigs_per_l={0: 1},
        spin_mode='RHF',
        mix_alpha=0.5,
        tol=1e-6,
        maxiter=100
    )

    # UHF 配置
    cfg_uhf = HFSCFGeneralConfig(
        Z=2,
        r=r, w=w,
        occ_by_l={0: [2.0]},
        eigs_per_l={0: 1},
        spin_mode='UHF',
        mix_alpha=0.5,
        tol=1e-6,
        maxiter=100
    )

    print("\n运行 RHF...")
    res_rhf = run_hf_scf(cfg_rhf)
    print(f"  收敛: {res_rhf.converged}, 迭代: {res_rhf.iterations}")
    print(f"  E_total = {res_rhf.E_total:.6f} Ha")
    print(f"  ε_1s = {res_rhf.eigenvalues_by_l[0][0]:.6f} Ha")

    print("\n运行 UHF...")
    res_uhf = run_hf_scf(cfg_uhf)
    print(f"  收敛: {res_uhf.converged}, 迭代: {res_uhf.iterations}")
    print(f"  E_total = {res_uhf.E_total:.6f} Ha")
    print(f"  ε_1s(up) = {res_uhf.eigenvalues_by_l_spin[(0, 'up')][0]:.6f} Ha")
    print(f"  ε_1s(dn) = {res_uhf.eigenvalues_by_l_spin[(0, 'down')][0]:.6f} Ha")

    # 对比
    print("\n对比分析:")
    delta_E = abs(res_uhf.E_total - res_rhf.E_total)
    delta_eps_up = abs(res_uhf.eigenvalues_by_l_spin[(0, 'up')][0] - res_rhf.eigenvalues_by_l[0][0])
    delta_eps_dn = abs(res_uhf.eigenvalues_by_l_spin[(0, 'down')][0] - res_rhf.eigenvalues_by_l[0][0])

    print(f"  |E_UHF - E_RHF| = {delta_E:.2e} Ha")
    print(f"  |ε_up - ε_RHF| = {delta_eps_up:.2e} Ha")
    print(f"  |ε_dn - ε_RHF| = {delta_eps_dn:.2e} Ha")

    if delta_E < 1e-5:
        print("  ✓ 闭壳层 UHF = RHF (通过)")
    else:
        print(f"  ✗ 闭壳层 UHF ≠ RHF (失败)")

    return res_rhf, res_uhf


def test_li_uhf_vs_rhf():
    """Li 原子：UHF 应该低于 RHF（开壳层）"""
    print("\n" + "=" * 70)
    print("测试 2: Li 原子 (开壳层，1s² 2s¹)")
    print("=" * 70)

    r, w = radial_grid_linear(n=1000, rmin=1e-6, rmax=60.0)

    # RHF 配置
    cfg_rhf = HFSCFGeneralConfig(
        Z=3,
        r=r, w=w,
        occ_by_l={0: [2.0, 1.0]},
        eigs_per_l={0: 2},
        spin_mode='RHF',
        mix_alpha=0.3,
        tol=1e-6,
        maxiter=120
    )

    # UHF 配置（显式指定自旋占据）
    cfg_uhf = HFSCFGeneralConfig(
        Z=3,
        r=r, w=w,
        occ_by_l={0: [2.0, 1.0]},
        occ_by_l_spin={
            0: {
                'up': [1.0, 1.0],    # 1s↑ 2s↑
                'down': [1.0, 0.0],  # 1s↓ (2s 无占据)
            }
        },
        eigs_per_l={0: 2},
        spin_mode='UHF',
        mix_alpha=0.3,
        tol=1e-6,
        maxiter=120
    )

    print("\n运行 RHF...")
    res_rhf = run_hf_scf(cfg_rhf)
    print(f"  收敛: {res_rhf.converged}, 迭代: {res_rhf.iterations}")
    print(f"  E_total = {res_rhf.E_total:.6f} Ha")
    print(f"  ε_1s = {res_rhf.eigenvalues_by_l[0][0]:.6f} Ha")
    print(f"  ε_2s = {res_rhf.eigenvalues_by_l[0][1]:.6f} Ha")

    print("\n运行 UHF...")
    res_uhf = run_hf_scf(cfg_uhf)
    print(f"  收敛: {res_uhf.converged}, 迭代: {res_uhf.iterations}")
    print(f"  E_total = {res_uhf.E_total:.6f} Ha")
    print(f"  ε_1s(up) = {res_uhf.eigenvalues_by_l_spin[(0, 'up')][0]:.6f} Ha")
    print(f"  ε_2s(up) = {res_uhf.eigenvalues_by_l_spin[(0, 'up')][1]:.6f} Ha")
    print(f"  ε_1s(dn) = {res_uhf.eigenvalues_by_l_spin[(0, 'down')][0]:.6f} Ha")
    print(f"  ε_2s(dn) = {res_uhf.eigenvalues_by_l_spin[(0, 'down')][1]:.6f} Ha")

    # 对比
    print("\n对比分析:")
    delta_E = res_uhf.E_total - res_rhf.E_total
    print(f"  E_UHF - E_RHF = {delta_E:.6f} Ha")

    if delta_E < -1e-4:
        print(f"  ✓ 开壳层 UHF < RHF (通过，UHF 降低 {abs(delta_E)*1000:.2f} mHa)")
    else:
        print(f"  ✗ 开壳层 UHF 未降低能量 (失败)")

    # 参考值（Clementi & Roetti, 1974）
    E_ref = -7.4327 # Ha (ROHF)
    print(f"\n  参考值 (Clementi ROHF): {E_ref:.4f} Ha")
    print(f"  RHF 误差: {abs(res_rhf.E_total - E_ref)*1000:.2f} mHa")
    print(f"  UHF 误差: {abs(res_uhf.E_total - E_ref)*1000:.2f} mHa")

    return res_rhf, res_uhf


if __name__ == '__main__':
    test_he_uhf_vs_rhf()
    test_li_uhf_vs_rhf()

    print("\n" + "=" * 70)
    print("UHF 实现测试完成")
    print("=" * 70)
