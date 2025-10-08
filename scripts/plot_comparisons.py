#!/usr/bin/env python3
"""
波函数对比可视化脚本
- H 原子：LSDA vs UHF 对比
- C 原子：LSDA vs RHF vs UHF 对比
- He, Li：LSDA 单方法
- Al：LSDA 3x3 子图布局
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 颜色方案
COLORS = {
    'LSDA': '#1f77b4',  # 蓝色
    'UHF': '#ff7f0e',   # 橙色
    'RHF': '#2ca02c',   # 绿色
}

LINESTYLES = {
    'LSDA': '-',
    'UHF': '--',
    'RHF': '-.',
}


def load_json(filepath):
    """加载 JSON 结果文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_orbitals(data, occupation_filter=None):
    """
    提取轨道数据

    Args:
        data: JSON 数据
        occupation_filter: 'occupied', 'unoccupied', 或 None (全部)

    Returns:
        [(orbital_key, orbital_data, wavefunction_u, wavefunction_r), ...]
    """
    levels = data['levels']
    wavefunctions = data.get('wavefunctions', {})

    orbitals = []
    for key, level in levels.items():
        if key not in wavefunctions:
            continue

        occ = level['occupation']
        if occupation_filter == 'occupied' and occ <= 0:
            continue
        if occupation_filter == 'unoccupied' and occ > 0:
            continue

        wf_dict = wavefunctions[key]
        orbitals.append((key, level, np.array(wf_dict['u']), np.array(wf_dict['r'])))

    return orbitals


def plot_h_comparison():
    """氢原子：LSDA vs UHF 对比（显示LSDA的up和down自旋）"""
    print("生成 H 原子对比图...")

    lsda = load_json('test_results/H_lsda.json')
    uhf = load_json('test_results/H_uhf.json')

    # 从第一个波函数中获取网格
    first_wf_key = list(lsda['wavefunctions'].keys())[0]
    r = np.array(lsda['wavefunctions'][first_wf_key]['r'])

    # 1s 占据态对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 子图1: 1s 占据态（LSDA显示up和down，UHF显示up）
    # LSDA 1s_up
    if '1s_up' in lsda['wavefunctions']:
        lsda_1s_up = lsda['levels']['1s_up']
        lsda_1s_up_wf = lsda['wavefunctions']['1s_up']
        ax1.plot(np.array(lsda_1s_up_wf['r']), np.array(lsda_1s_up_wf['u']),
                label=f'LSDA 1s↑ (ε={lsda_1s_up["value"]:.4f} Ha)',
                color=COLORS['LSDA'], linestyle='-', linewidth=2)

    # LSDA 1s_down
    if '1s_down' in lsda['wavefunctions']:
        lsda_1s_down = lsda['levels']['1s_down']
        lsda_1s_down_wf = lsda['wavefunctions']['1s_down']
        ax1.plot(np.array(lsda_1s_down_wf['r']), np.array(lsda_1s_down_wf['u']),
                label=f'LSDA 1s↓ (ε={lsda_1s_down["value"]:.4f} Ha)',
                color=COLORS['LSDA'], linestyle='--', linewidth=2, alpha=0.7)

    # UHF 1s_up
    if '1s_up' in uhf['wavefunctions']:
        uhf_1s_up = uhf['levels']['1s_up']
        uhf_1s_up_wf = uhf['wavefunctions']['1s_up']
        ax1.plot(np.array(uhf_1s_up_wf['r']), np.array(uhf_1s_up_wf['u']),
                label=f'UHF 1s↑ (ε={uhf_1s_up["value"]:.4f} Ha)',
                color=COLORS['UHF'], linestyle=LINESTYLES['UHF'], linewidth=2)

    ax1.set_xlabel('r (Bohr)', fontsize=12)
    ax1.set_ylabel('径向波函数 u(r)', fontsize=12)
    ax1.set_title('H 1s 占据态', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 8)

    # 子图2: 2s 未占据态（LSDA，展示 SIE）
    lsda_unocc = extract_orbitals(lsda, 'unoccupied')
    if lsda_unocc:
        unocc_key, unocc_level, unocc_u, unocc_r = lsda_unocc[0]
        ax2.plot(unocc_r, unocc_u, label=f'{unocc_key}: ε={unocc_level["value"]:.6f} Ha',
                 color='red', linewidth=2)
        ax2.set_xlabel('r (Bohr)', fontsize=12)
        ax2.set_ylabel('径向波函数 u(r)', fontsize=12)
        ax2.set_title('H 未占据态（LSDA，自相互作用误差导致正能量）', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 20)

    plt.tight_layout()
    plt.savefig('figures/H_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ 已保存: figures/H_comparison.png")
    plt.close()


def plot_c_comparison():
    """碳原子：LSDA vs RHF vs UHF 对比（占据态 + 未占据态，显示up/down自旋）"""
    print("生成 C 原子对比图...")

    lsda = load_json('test_results/C_lsda.json')
    rhf = load_json('test_results/C_rhf.json')
    uhf = load_json('test_results/C_uhf.json')

    # 提取未占据态（仅 LSDA，用于下排子图）
    lsda_unocc = extract_orbitals(lsda, 'unoccupied')

    # 定义要绘制的占据态轨道
    orbitals_to_plot = ['1s', '2s', '2p']
    n_orbs = len(orbitals_to_plot)

    # 创建子图（2行：上排占据态，下排未占据态）
    fig = plt.figure(figsize=(5*n_orbs, 10))
    gs = fig.add_gridspec(2, n_orbs, height_ratios=[1, 1])

    # 上排：占据态对比（1s, 2s, 2p）
    for idx, orb_name in enumerate(orbitals_to_plot):
        ax = fig.add_subplot(gs[0, idx])

        # LSDA: up and down
        lsda_up_key = f'{orb_name}_up'
        lsda_down_key = f'{orb_name}_down'

        if lsda_up_key in lsda['wavefunctions'] and lsda['levels'][lsda_up_key]['occupation'] > 0:
            level = lsda['levels'][lsda_up_key]
            wf = lsda['wavefunctions'][lsda_up_key]
            ax.plot(np.array(wf['r']), np.array(wf['u']),
                   label=f'LSDA ↑ (ε={level["value"]:.4f} Ha)',
                   color=COLORS['LSDA'], linestyle='-', linewidth=2)

        if lsda_down_key in lsda['wavefunctions'] and lsda['levels'][lsda_down_key]['occupation'] > 0:
            level = lsda['levels'][lsda_down_key]
            wf = lsda['wavefunctions'][lsda_down_key]
            ax.plot(np.array(wf['r']), np.array(wf['u']),
                   label=f'LSDA ↓ (ε={level["value"]:.4f} Ha)',
                   color=COLORS['LSDA'], linestyle='--', linewidth=2, alpha=0.7)

        # RHF: no spin (闭壳层)
        if orb_name in rhf['wavefunctions'] and rhf['levels'][orb_name]['occupation'] > 0:
            level = rhf['levels'][orb_name]
            wf = rhf['wavefunctions'][orb_name]
            ax.plot(np.array(wf['r']), np.array(wf['u']),
                   label=f'RHF (ε={level["value"]:.4f} Ha)',
                   color=COLORS['RHF'], linestyle=LINESTYLES['RHF'], linewidth=2)

        # UHF: up and down
        uhf_up_key = f'{orb_name}_up'
        uhf_down_key = f'{orb_name}_down'

        if uhf_up_key in uhf['wavefunctions'] and uhf['levels'][uhf_up_key]['occupation'] > 0:
            level = uhf['levels'][uhf_up_key]
            wf = uhf['wavefunctions'][uhf_up_key]
            ax.plot(np.array(wf['r']), np.array(wf['u']),
                   label=f'UHF ↑ (ε={level["value"]:.4f} Ha)',
                   color=COLORS['UHF'], linestyle='-', linewidth=2, alpha=0.8)

        if uhf_down_key in uhf['wavefunctions'] and uhf['levels'][uhf_down_key]['occupation'] > 0:
            level = uhf['levels'][uhf_down_key]
            wf = uhf['wavefunctions'][uhf_down_key]
            ax.plot(np.array(wf['r']), np.array(wf['u']),
                   label=f'UHF ↓ (ε={level["value"]:.4f} Ha)',
                   color=COLORS['UHF'], linestyle='--', linewidth=2, alpha=0.6)

        ax.set_xlabel('r (Bohr)', fontsize=12)
        ax.set_ylabel('径向波函数 u(r)', fontsize=12)
        ax.set_title(f'C {orb_name} 占据态', fontsize=14, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

        # 根据轨道调整 x 轴范围
        if '1s' in orb_name:
            ax.set_xlim(0, 3)
        elif '2' in orb_name:
            ax.set_xlim(0, 8)

    # 下排：未占据态（LSDA，仅显示 3s，排除 3p）
    if lsda_unocc:
        # 过滤出 3s 轨道（排除 3p）
        filtered_unocc = [(k, l, u, r) for k, l, u, r in lsda_unocc if '3s' in k]

        if filtered_unocc:
            # 将未占据态显示在下排的中间子图
            ax_unocc = fig.add_subplot(gs[1, :])

            for key, level, wf_u, wf_r in filtered_unocc:
                ax_unocc.plot(wf_r, wf_u,
                             label=f'{key}: ε={level["value"]:.6f} Ha',
                             linewidth=2)

            ax_unocc.set_xlabel('r (Bohr)', fontsize=12)
            ax_unocc.set_ylabel('径向波函数 u(r)', fontsize=12)
            ax_unocc.set_title('C 未占据态（LSDA，3s）', fontsize=14, fontweight='bold')
            ax_unocc.legend(fontsize=10)
            ax_unocc.grid(True, alpha=0.3)
            ax_unocc.set_xlim(0, 15)

    plt.tight_layout()
    plt.savefig('figures/C_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ 已保存: figures/C_comparison.png")
    plt.close()


def plot_single_atom(atom_name, json_file, xlim=None):
    """单原子 LSDA 结果（He, Li）"""
    print(f"生成 {atom_name} 原子图...")

    data = load_json(json_file)
    orbitals = extract_orbitals(data, 'occupied')

    n_orbs = len(orbitals)
    fig, axes = plt.subplots(n_orbs, 1, figsize=(8, 3*n_orbs))
    if n_orbs == 1:
        axes = [axes]

    for idx, (key, level, wf_u, wf_r) in enumerate(orbitals):
        ax = axes[idx]
        ax.plot(wf_r, wf_u, color=COLORS['LSDA'], linewidth=2)
        ax.set_xlabel('r (Bohr)', fontsize=12)
        ax.set_ylabel('径向波函数 u(r)', fontsize=12)
        ax.set_title(f'{atom_name} {key}: ε = {level["value"]:.6f} Ha, '
                    f'occ = {level["occupation"]:.1f}',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if xlim:
            ax.set_xlim(0, xlim)

    plt.tight_layout()
    plt.savefig(f'figures/{atom_name}_lsda.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ 已保存: figures/{atom_name}_lsda.png")
    plt.close()


def plot_al_3x3():
    """Al 原子：3x3 子图布局"""
    print("生成 Al 原子图（3x3 布局）...")

    data = load_json('test_results/Al_lsda.json')
    orbitals = extract_orbitals(data, 'occupied')

    n_orbs = len(orbitals)
    ncols = 3
    nrows = (n_orbs + ncols - 1) // ncols  # 向上取整

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3*nrows))
    axes = axes.flatten()

    for idx, (key, level, wf_u, wf_r) in enumerate(orbitals):
        ax = axes[idx]
        ax.plot(wf_r, wf_u, color=COLORS['LSDA'], linewidth=2)
        ax.set_xlabel('r (Bohr)', fontsize=10)
        ax.set_ylabel('u(r)', fontsize=10)
        ax.set_title(f'{key}: ε={level["value"]:.5f} Ha, occ={level["occupation"]:.1f}',
                    fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 根据轨道类型调整范围
        if '1s' in key:
            ax.set_xlim(0, 2)
        elif '2' in key:
            ax.set_xlim(0, 5)
        elif '3' in key:
            ax.set_xlim(0, 10)

    # 隐藏多余的子图
    for idx in range(n_orbs, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig('figures/Al_lsda.png', dpi=300, bbox_inches='tight')
    print("  ✓ 已保存: figures/Al_lsda.png")
    plt.close()


def main():
    """主函数"""
    print("=" * 50)
    print("波函数对比可视化")
    print("=" * 50)

    # 确保输出目录存在
    Path('figures').mkdir(exist_ok=True)

    # 生成所有图形
    plot_h_comparison()      # H: LSDA vs UHF
    plot_c_comparison()      # C: LSDA vs RHF vs UHF
    plot_single_atom('He', 'test_results/He_lsda.json', xlim=5)
    plot_single_atom('Li', 'test_results/Li_lsda.json', xlim=12)
    plot_al_3x3()            # Al: 3x3 布局

    print("=" * 50)
    print("✅ 所有波函数图生成完成！")
    print("=" * 50)
    print("\n生成的图形文件：")
    for fig_file in sorted(Path('figures').glob('*.png')):
        size_mb = fig_file.stat().st_size / 1024 / 1024
        print(f"  - {fig_file.name} ({size_mb:.2f} MB)")


if __name__ == '__main__':
    main()
