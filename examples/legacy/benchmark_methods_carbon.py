"""对比测试：不同方法在碳原子 LSDA 计算中的表现

比较三种方法：
1. 变量变换方法 (transformed) + 指数网格
2. FD5-aux 方法 + 混合网格
3. FD5-aux 方法 + 对数网格

运行：python compare_methods_carbon.py
"""
import sys
sys.path.insert(0, '../src')

import numpy as np
import time
from atomscf.grid import radial_grid_exp_transformed, radial_grid_mixed, radial_grid_log
from atomscf.scf import SCFConfig, run_lsda_vwn

print('=' * 80)
print('碳原子 LSDA-VWN 计算方法对比测试')
print('=' * 80)
print()

# 测试配置列表
test_configs = []

# ============================================================================
# 方法 1: 变量变换方法 + 指数网格
# ============================================================================
print('[1/3] 准备配置：变量变换方法 + 指数网格')
r1, w1, delta1, Rp1 = radial_grid_exp_transformed(
    n=1000, rmin=0.0, rmax=100.0, total_span=6.0
)
cfg1 = SCFConfig(
    Z=6, r=r1, w=w1,
    mix_alpha=0.3, tol=1e-6, maxiter=150,
    eigs_per_l=2, lmax=2,
    compute_all_l=True, compute_all_l_mode="final",
    mix_kind="density", adapt_mixing=True, xc="VWN",
    eig_solver="transformed", delta=delta1, Rp=Rp1,
)
test_configs.append(("变量变换 + 指数网格", cfg1))
print(f'  网格点数: {len(r1)}, rmax: {r1[-1]:.1f}, δ: {delta1:.6f}')

# ============================================================================
# 方法 2: FD5-aux + 混合网格（之前的"标准"方法）
# ============================================================================
print('[2/3] 准备配置：FD5-aux + 混合网格')
r2, w2 = radial_grid_mixed(
    n_inner=1500, n_outer=500,
    rmin=1e-7, r_switch=10.0, rmax=100.0,
)
cfg2 = SCFConfig(
    Z=6, r=r2, w=w2,
    mix_alpha=0.35, tol=1e-6, maxiter=160,
    eigs_per_l=2, lmax=2,
    compute_all_l=True, compute_all_l_mode="final",
    mix_kind="density", adapt_mixing=True, xc="VWN",
    eig_solver="fd5_aux",
)
test_configs.append(("FD5-aux + 混合网格", cfg2))
print(f'  网格点数: {len(r2)}, rmin: {r2[0]:.2e}, r_switch: 10.0, rmax: {r2[-1]:.1f}')

# ============================================================================
# 方法 3: FD5-aux + 对数网格
# ============================================================================
print('[3/3] 准备配置：FD5-aux + 对数网格')
r3, w3 = radial_grid_log(n=2000, rmin=1e-7, rmax=100.0)
cfg3 = SCFConfig(
    Z=6, r=r3, w=w3,
    mix_alpha=0.35, tol=1e-6, maxiter=160,
    eigs_per_l=2, lmax=2,
    compute_all_l=True, compute_all_l_mode="final",
    mix_kind="density", adapt_mixing=True, xc="VWN",
    eig_solver="fd5_aux",
)
test_configs.append(("FD5-aux + 对数网格", cfg3))
print(f'  网格点数: {len(r3)}, rmin: {r3[0]:.2e}, rmax: {r3[-1]:.1f}')
print()

# ============================================================================
# 运行测试
# ============================================================================
results = []

for i, (name, cfg) in enumerate(test_configs, 1):
    print('=' * 80)
    print(f'测试 {i}/{len(test_configs)}: {name}')
    print('=' * 80)

    t_start = time.time()
    try:
        res = run_lsda_vwn(cfg, verbose=True, progress_every=20)
        t_elapsed = time.time() - t_start

        # 提取能级
        e_1s = res.eps_by_l_sigma[(0, "up")][0]
        e_2s = res.eps_by_l_sigma[(0, "up")][1]
        e_2p = res.eps_by_l_sigma[(1, "up")][0]

        # 总能量
        E_tot = res.energies.get('E_total', None) if res.energies else None

        results.append({
            'name': name,
            'converged': res.converged,
            'iterations': res.iterations,
            'time': t_elapsed,
            'e_1s': e_1s,
            'e_2s': e_2s,
            'e_2p': e_2p,
            'E_tot': E_tot,
            'n_points': len(cfg.r),
        })

        print(f'\n✓ 完成 ({t_elapsed:.2f} 秒)')
        print(f'  收敛: {"是" if res.converged else "否"}, 迭代次数: {res.iterations}')
        print(f'  1s: {e_1s:.6f} Ha, 2s: {e_2s:.6f} Ha, 2p: {e_2p:.6f} Ha')

    except Exception as e:
        print(f'\n✗ 失败: {e}')
        results.append({
            'name': name,
            'converged': False,
            'iterations': 0,
            'time': 0,
            'e_1s': None,
            'e_2s': None,
            'e_2p': None,
            'E_tot': None,
            'n_points': len(cfg.r),
        })

    print()

# ============================================================================
# 汇总对比
# ============================================================================
print('=' * 80)
print('对比汇总')
print('=' * 80)
print()

# NIST 参考值
nist_1s = -11.33
nist_2s = -0.71
nist_2p = -0.44

print('方法                         网格点   收敛  迭代  时间(s)  1s (Ha)    2s (Ha)    2p (Ha)')
print('-' * 100)

for r in results:
    conv_str = "是" if r['converged'] else "否"
    e_1s_str = f"{r['e_1s']:10.4f}" if r['e_1s'] is not None else "    -     "
    e_2s_str = f"{r['e_2s']:10.4f}" if r['e_2s'] is not None else "    -     "
    e_2p_str = f"{r['e_2p']:10.4f}" if r['e_2p'] is not None else "    -     "

    print(f"{r['name']:28s} {r['n_points']:5d}    {conv_str:2s}  {r['iterations']:4d}  "
          f"{r['time']:6.2f}  {e_1s_str}  {e_2s_str}  {e_2p_str}")

print()
print(f"NIST 参考值 (近似)                                        {nist_1s:10.2f}  "
      f"{nist_2s:10.2f}  {nist_2p:10.2f}")
print()

# 误差分析
print('与 NIST 参考值的相对误差 (%):')
print('-' * 80)
for r in results:
    if r['e_1s'] is not None:
        err_1s = abs(r['e_1s'] - nist_1s) / abs(nist_1s) * 100
        err_2s = abs(r['e_2s'] - nist_2s) / abs(nist_2s) * 100
        err_2p = abs(r['e_2p'] - nist_2p) / abs(nist_2p) * 100
        print(f"{r['name']:28s}  1s: {err_1s:6.2f}%   2s: {err_2s:6.2f}%   2p: {err_2p:6.2f}%")
    else:
        print(f"{r['name']:28s}  计算失败")

print()
print('=' * 80)
print('结论：')
print('=' * 80)

# 找到最好的结果（基于 1s 能级）
best_idx = 0
best_e1s_error = float('inf')
for i, r in enumerate(results):
    if r['e_1s'] is not None:
        error = abs(r['e_1s'] - nist_1s)
        if error < best_e1s_error:
            best_e1s_error = error
            best_idx = i

print(f"最佳方法（基于 1s 能级精度）: {results[best_idx]['name']}")
print(f"  1s 误差: {abs(results[best_idx]['e_1s'] - nist_1s):.4f} Ha "
      f"({abs(results[best_idx]['e_1s'] - nist_1s) / abs(nist_1s) * 100:.2f}%)")
print(f"  计算时间: {results[best_idx]['time']:.2f} 秒")
print(f"  迭代次数: {results[best_idx]['iterations']}")
print()
