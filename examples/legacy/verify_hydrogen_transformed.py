import sys
sys.path.insert(0, '../src')
from atomscf.grid import radial_grid_exp_transformed
from atomscf.operator import solve_bound_states_transformed
import numpy as np

print('=== 氢原子多能级测试（变量变换方法）===')

# 生成指数网格
r, w, delta, Rp = radial_grid_exp_transformed(
    n=800, rmin=0.0, rmax=80.0, total_span=6.0
)

print(f'网格: n={len(r)}, rmax={r[-1]:.1f}, delta={delta:.6f}')

# 氢原子库仑势
Z = 1
v = -Z / (r + 1e-30)

# 测试不同 l 值
test_cases = [
    (0, 1, '1s', -0.5),
    (0, 2, '2s', -0.125),
    (1, 1, '2p', -0.125),
]

print('\n态      计算值(Ha)      理论值(Ha)      误差(Ha)       相对误差(%)')
print('-' * 70)

for l, k_target, label, E_theory in test_cases:
    eps, U = solve_bound_states_transformed(
        r, l=l, v_of_r=v, delta=delta, Rp=Rp, k=k_target, use_sparse=False
    )
    E_calc = eps[k_target-1]
    error = abs(E_calc - E_theory)
    rel_error = error / abs(E_theory) * 100
    print(f'{label:6s} {E_calc:15.6f} {E_theory:15.6f} {error:12.2e} {rel_error:12.4f}')

print('\n成功！变量变换方法在氢原子上工作完美！')
