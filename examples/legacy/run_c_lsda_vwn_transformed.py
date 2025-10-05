"""碳原子（Z=6）LSDA（VWN）自洽计算（变量变换方法 + 指数网格）

运行：
    python test_carbon_lsda_transformed.py
"""

import sys

sys.path.insert(0, "../src")

import numpy as np
from atomscf.grid import radial_grid_exp_transformed
from atomscf.scf import SCFConfig, run_lsda_vwn
from atomscf.occupations import default_occupations

print("=" * 70)
print("碳原子 LSDA-VWN 自洽计算（变量变换方法 + 指数网格）")
print("=" * 70)
print()

# 生成指数变换网格
r, w, delta, Rp = radial_grid_exp_transformed(
    n=1000, rmin=0.0, rmax=100.0, total_span=6.0
)

print(f"网格参数:")
print(f"  点数: n = {len(r)}")
print(f"  范围: r ∈ [{r[0]:.2e}, {r[-1]:.1f}] Bohr")
print(f"  变换参数: δ = {delta:.6f}, R_p = {Rp:.6f}")
print(f"  r[0] = {r[0]:.2e}, r[1] = {r[1]:.2e}, r[2] = {r[2]:.2e}")
print()

# SCF 配置
cfg = SCFConfig(
    Z=6,
    r=r,
    w=w,
    mix_alpha=0.3,
    tol=1e-6,
    maxiter=150,
    eigs_per_l=2,
    lmax=2,
    compute_all_l=True,
    compute_all_l_mode="final",
    mix_kind="density",
    adapt_mixing=True,
    xc="VWN",
    eig_solver="transformed",  # 使用变量变换方法
    delta=delta,  # 传入网格参数
    Rp=Rp,
)

print("SCF 配置:")
print(f"  原子: Z = {cfg.Z} (碳)")
print(f"  求解器: {cfg.eig_solver}")
print(f"  交换关联泛函: {cfg.xc}")
print(f"  混合方式: {cfg.mix_kind}, α = {cfg.mix_alpha}")
print(f"  收敛阈值: {cfg.tol:.1e}")
print()

print("开始 SCF 迭代...")
print("-" * 70)
res = run_lsda_vwn(cfg, verbose=True, progress_every=10)
print("-" * 70)
print()

if res.converged:
    print("✓ SCF 收敛成功！")
else:
    print("✗ SCF 未收敛，达到最大迭代次数")
print(f"  总迭代次数: {res.iterations}")
print()

# 提取能级
e_1s_up = res.eps_by_l_sigma[(0, "up")][0]
e_1s_dn = res.eps_by_l_sigma[(0, "down")][0]
e_2s_up = res.eps_by_l_sigma[(0, "up")][1]
e_2s_dn = res.eps_by_l_sigma[(0, "down")][1]
e_2p_up = res.eps_by_l_sigma[(1, "up")][0]
e_2p_dn = res.eps_by_l_sigma[(1, "down")][0]

print("能级结果:")
print("  自旋向上:")
print(f"    1s: {e_1s_up:12.6f} Ha")
print(f"    2s: {e_2s_up:12.6f} Ha")
print(f"    2p: {e_2p_up:12.6f} Ha")
print("  自旋向下:")
print(f"    1s: {e_1s_dn:12.6f} Ha")
print(f"    2s: {e_2s_dn:12.6f} Ha")
print(f"    2p: {e_2p_dn:12.6f} Ha")
print()

# 与 NIST 参考值比较
print("与 NIST 参考值比较:")
nist_1s = -9.940546  # 近似值
nist_2s = -0.531276
nist_2p = -0.227557

print(f"  1s: 计算值 = {e_1s_up:.4f} Ha, NIST ≈ {nist_1s:.2f} Ha")
print(f"  2s: 计算值 = {e_2s_up:.4f} Ha, NIST ≈ {nist_2s:.2f} Ha")
print(f"  2p: 计算值 = {e_2p_up:.4f} Ha, NIST ≈ {nist_2p:.2f} Ha")
print()

# 总能量
if res.energies and "total" in res.energies:
    E_tot = res.energies["total"]
    print(f"总能量: E_tot = {E_tot:.6f} Ha")
    print()
elif res.energies:
    print("能量分量:")
    for key, val in res.energies.items():
        print(f"  {key}: {val:.6f} Ha")
    print()

print("=" * 70)
print("测试完成！")
print("=" * 70)
