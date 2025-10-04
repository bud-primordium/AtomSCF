"""碳原子简单测试（变量变换方法，使用屏蔽库仑势）

测试变量变换方法在多电子原子上的表现。
使用简化的屏蔽库仑势 V(r) = -Z_eff/r 作为近似。
"""
import sys
sys.path.insert(0, '../src')
from atomscf.grid import radial_grid_exp_transformed
from atomscf.operator import solve_bound_states_transformed
import numpy as np

print('=== 碳原子简单测试（变量变换方法 + 屏蔽库仑势）===')
print()

# 生成指数变换网格
r, w, delta, Rp = radial_grid_exp_transformed(
    n=800, rmin=0.0, rmax=80.0, total_span=6.0
)

print(f'网格参数: n={len(r)}, rmax={r[-1]:.1f}, delta={delta:.6f}, Rp={Rp:.6f}')
print(f'r[0]={r[0]:.2e}, r[1]={r[1]:.2e}')
print()

# 碳原子 Z=6
Z = 6

# 使用不同的有效电荷测试（屏蔽效应）
# 1s: 几乎感受全部核电荷，Z_eff ~ 5.7
# 2s, 2p: 被内层屏蔽，Z_eff ~ 3.2

test_cases = [
    (0, 5.7, 1, '1s (Z_eff=5.7)'),
    (0, 3.2, 2, '2s (Z_eff=3.2)'),
    (1, 3.2, 1, '2p (Z_eff=3.2)'),
]

print('通道         Z_eff    计算能量(Ha)    氢样理论(Ha)    误差(Ha)')
print('-' * 70)

for l, Z_eff, k_target, label in test_cases:
    # 屏蔽库仑势
    v = -Z_eff / (r + 1e-30)

    # 求解
    eps, U = solve_bound_states_transformed(
        r, l=l, v_of_r=v, delta=delta, Rp=Rp, k=k_target, use_sparse=False
    )

    E_calc = eps[k_target-1]

    # 氢样理论值：E = -Z_eff^2 / (2n^2)
    n = l + k_target
    E_theory = -Z_eff**2 / (2 * n**2)

    error = abs(E_calc - E_theory)

    print(f'{label:16s} {Z_eff:5.1f}  {E_calc:15.6f}  {E_theory:15.6f}  {error:12.2e}')

print()
print('注：这是使用屏蔽库仑势的简化测试。')
print('   真实碳原子需要完整的 SCF 计算（Hartree + 交换关联）。')
print('   NIST 参考值：1s ≈ -11.33 Ha, 2s ≈ -0.71 Ha, 2p ≈ -0.44 Ha')
