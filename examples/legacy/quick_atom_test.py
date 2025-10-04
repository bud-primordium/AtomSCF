"""通用原子测试脚本（变量变换方法）

快速测试任意原子的 LSDA 计算。

用法：
    python quick_atom_test.py [Z]

示例：
    python quick_atom_test.py 2   # 氦原子
    python quick_atom_test.py 10  # 氖原子
    python quick_atom_test.py 18  # 氩原子
"""
import sys
sys.path.insert(0, '../src')

import numpy as np
from atomscf.grid import radial_grid_exp_transformed
from atomscf.scf import SCFConfig, run_lsda_vwn
from atomscf.occupations import default_occupations

# 原子名称字典
ATOM_NAMES = {
    1: 'H',  2: 'He', 3: 'Li', 4: 'Be', 5: 'B',  6: 'C',  7: 'N',  8: 'O',  9: 'F',  10: 'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar',
    19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe',
    27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
}

def get_default_grid_params(Z):
    """根据原子序数返回推荐的网格参数"""
    if Z <= 2:
        return {'n': 800, 'rmax': 50.0, 'total_span': 5.0}
    elif Z <= 10:
        return {'n': 1000, 'rmax': 80.0, 'total_span': 6.0}
    elif Z <= 18:
        return {'n': 1200, 'rmax': 100.0, 'total_span': 6.0}
    else:
        return {'n': 1500, 'rmax': 120.0, 'total_span': 6.5}

def get_default_scf_params(Z):
    """根据原子序数返回推荐的 SCF 参数"""
    if Z <= 2:
        return {'mix_alpha': 0.4, 'maxiter': 100, 'eigs_per_l': 1, 'lmax': 1}
    elif Z <= 10:
        return {'mix_alpha': 0.3, 'maxiter': 150, 'eigs_per_l': 2, 'lmax': 2}
    elif Z <= 18:
        return {'mix_alpha': 0.25, 'maxiter': 200, 'eigs_per_l': 3, 'lmax': 3}
    else:
        return {'mix_alpha': 0.2, 'maxiter': 250, 'eigs_per_l': 3, 'lmax': 3}

def main():
    # 解析命令行参数
    if len(sys.argv) < 2:
        print("用法: python test_any_atom.py [Z]")
        print("示例: python test_any_atom.py 6  # 碳原子")
        sys.exit(1)

    Z = int(sys.argv[1])

    if Z < 1 or Z > 118:
        print(f"错误: Z = {Z} 超出范围 [1, 118]")
        sys.exit(1)

    atom_name = ATOM_NAMES.get(Z, f"Z={Z}")
    print('=' * 80)
    print(f'{atom_name} 原子 (Z={Z}) LSDA-VWN 计算（变量变换方法）')
    print('=' * 80)
    print()

    # 获取推荐参数
    grid_params = get_default_grid_params(Z)
    scf_params = get_default_scf_params(Z)

    # 生成网格
    print('网格参数:')
    print(f"  点数: n = {grid_params['n']}")
    print(f"  范围: rmax = {grid_params['rmax']:.1f} Bohr")
    print(f"  变换参数: total_span = {grid_params['total_span']:.1f}")

    r, w, delta, Rp = radial_grid_exp_transformed(
        n=grid_params['n'],
        rmin=0.0,
        rmax=grid_params['rmax'],
        total_span=grid_params['total_span']
    )

    print(f"  实际: δ = {delta:.6f}, R_p = {Rp:.6f}")
    print(f"  r[0] = {r[0]:.2e}, r[1] = {r[1]:.2e}")
    print()

    # SCF 配置
    print('SCF 参数:')
    print(f"  混合系数: α = {scf_params['mix_alpha']}")
    print(f"  最大迭代: {scf_params['maxiter']}")
    print(f"  每通道求解态数: {scf_params['eigs_per_l']}")
    print(f"  最大角动量: lmax = {scf_params['lmax']}")
    print()

    cfg = SCFConfig(
        Z=Z,
        r=r,
        w=w,
        mix_alpha=scf_params['mix_alpha'],
        tol=1e-6,
        maxiter=scf_params['maxiter'],
        eigs_per_l=scf_params['eigs_per_l'],
        lmax=scf_params['lmax'],
        compute_all_l=True,
        compute_all_l_mode="final",
        mix_kind="density",
        adapt_mixing=True,
        xc="VWN",
        eig_solver="transformed",
        delta=delta,
        Rp=Rp,
    )

    # 运行 SCF
    print('开始 SCF 迭代...')
    print('-' * 80)

    import time
    t_start = time.time()
    res = run_lsda_vwn(cfg, verbose=True, progress_every=20)
    t_elapsed = time.time() - t_start

    print('-' * 80)
    print()

    # 输出结果
    if res.converged:
        print('✓ SCF 收敛成功！')
    else:
        print('✗ SCF 未收敛，达到最大迭代次数')
    print(f'  总迭代次数: {res.iterations}')
    print(f'  计算时间: {t_elapsed:.2f} 秒')
    print()

    # 占据信息
    occ = default_occupations(Z)
    print('电子占据 (基态组态):')
    total_up = 0.0
    total_dn = 0.0
    for orb in occ:
        n_elec = orb.f_per_m * (2 * orb.l + 1)
        if orb.spin == "up":
            symbol = '↑'
            total_up += n_elec
        else:
            symbol = '↓'
            total_dn += n_elec
        print(f'  {orb.label:8s}: {symbol} {n_elec:.2f}')
    print(f'  总计: {total_up + total_dn:.0f} 电子 (↑{total_up:.0f} ↓{total_dn:.0f})')
    print()

    # 能级
    print('能级 (Ha):')
    print('  轨道       自旋向上       自旋向下      能级差')
    print('  ' + '-' * 55)

    # 提取所有已计算的能级
    displayed = set()
    for orb in occ:
        l = orb.l
        n_index = orb.n_index
        key = (l, n_index)

        if key in displayed:
            continue
        displayed.add(key)

        if (l, "up") in res.eps_by_l_sigma and (l, "down") in res.eps_by_l_sigma:
            if n_index < len(res.eps_by_l_sigma[(l, "up")]) and n_index < len(res.eps_by_l_sigma[(l, "down")]):
                e_up = res.eps_by_l_sigma[(l, "up")][n_index]
                e_dn = res.eps_by_l_sigma[(l, "down")][n_index]
                delta_e = e_up - e_dn

                print(f'  {orb.label:8s}   {e_up:14.6f}   {e_dn:14.6f}   {delta_e:10.6f}')

    print()

    # 总能量
    if res.energies:
        print('能量分量 (Ha):')
        for key, val in res.energies.items():
            print(f'  {key:12s}: {val:14.6f}')
        print()

    print('=' * 80)
    print('测试完成！')
    print('=' * 80)

if __name__ == '__main__':
    main()
