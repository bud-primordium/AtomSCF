#!/usr/bin/env python3
"""
提取所有 JSON 结果文件的关键数据，生成汇总表格
"""

import json
from pathlib import Path


def load_json(filepath):
    """加载 JSON 文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  无法加载 {filepath}: {e}")
        return None


def extract_key_data(data, atom_name, method):
    """提取关键数据"""
    if not data:
        return None

    result = {
        'atom': atom_name,
        'method': method,
        'total_energy': data['energies']['E_total'],
        'orbitals': {}
    }

    # 提取占据态能级（使用正确的键名：levels）
    for key, level in data['levels'].items():
        if level['occupation'] > 0:
            orb_name = key.split('_')[0] if '_' in key else key.rstrip('Du')
            result['orbitals'][key] = {
                'name': orb_name,
                'epsilon': level['value'],
                'occupation': level['occupation'],
                'spin': level.get('spin', '')
            }

    # 提取未占据态能级
    result['unoccupied'] = []
    for key, level in data['levels'].items():
        if level['occupation'] == 0:
            orb_name = key.split('_')[0] if '_' in key else key.rstrip('Du')
            result['unoccupied'].append({
                'name': orb_name,
                'key': key,
                'epsilon': level['value'],
                'spin': level.get('spin', '')
            })

    return result


def print_summary_table(results):
    """打印汇总表格"""
    print("=" * 80)
    print("计算结果汇总")
    print("=" * 80)

    for res in results:
        if not res:
            continue

        print(f"\n{'='*80}")
        print(f"原子: {res['atom']}, 方法: {res['method']}")
        print(f"{'='*80}")
        print(f"总能量: {res['total_energy']:.8f} Ha")
        print(f"\n占据态轨道能级:")
        print(f"  {'轨道':<15} {'能量 (Ha)':<20} {'占据':<10} {'自旋':<10}")
        print(f"  {'-'*60}")
        for key, orb in res['orbitals'].items():
            spin_str = orb['spin'] if orb['spin'] else '-'
            print(f"  {key:<15} {orb['epsilon']:<20.8f} {orb['occupation']:<10.1f} {spin_str:<10}")

        if res['unoccupied']:
            print(f"\n未占据态:")
            print(f"  {'轨道':<15} {'能量 (Ha)':<20} {'自旋':<10}")
            print(f"  {'-'*50}")
            for unocc in res['unoccupied'][:3]:  # 只显示前3个
                spin_str = unocc['spin'] if unocc['spin'] else '-'
                print(f"  {unocc['key']:<15} {unocc['epsilon']:<20.8f} {spin_str:<10}")


def generate_latex_tables(results):
    """生成 LaTeX 表格代码"""
    print("\n" + "=" * 80)
    print("LaTeX 表格代码")
    print("=" * 80)

    # 按原子分组
    atoms = {}
    for res in results:
        if not res:
            continue
        atom = res['atom']
        if atom not in atoms:
            atoms[atom] = []
        atoms[atom].append(res)

    # 生成每个原子的表格
    for atom, methods_data in sorted(atoms.items()):
        print(f"\n% ===== {atom} 原子 =====")

        # 提取所有方法
        methods = [r['method'] for r in methods_data]

        # 提取所有轨道（按第一个方法的顺序）
        if not methods_data:
            continue

        first_method = methods_data[0]
        orbitals_list = list(first_method['orbitals'].keys())

        # 表头
        print("\\begin{table}[H]")
        print("\\centering")
        print(f"\\caption{{{atom} 原子计算结果对比}}")

        # 列定义
        ncols = len(methods) + 1
        col_spec = "l" + "c" * len(methods)
        print(f"\\begin{{tabular}}{{{col_spec}}}")
        print("\\toprule")

        # 表头行
        header = "\\textbf{物理量}"
        for method in methods:
            header += f" & \\textbf{{{method}}}"
        header += " \\\\"
        print(header)
        print("\\midrule")

        # 总能量行
        energy_row = f"总能量 (Ha)"
        for r in methods_data:
            energy_row += f" & {r['total_energy']:.6f}"
        energy_row += " \\\\"
        print(energy_row)

        # 轨道能级行
        for orb_key in orbitals_list:
            orb_name = first_method['orbitals'][orb_key]['name']
            orb_row = f"{orb_name} 能级 (Ha)"

            for r in methods_data:
                if orb_key in r['orbitals']:
                    orb_row += f" & {r['orbitals'][orb_key]['epsilon']:.6f}"
                else:
                    # 尝试匹配相同的轨道（不同自旋标记）
                    found = False
                    for k, v in r['orbitals'].items():
                        if v['name'] == orb_name:
                            orb_row += f" & {v['epsilon']:.6f}"
                            found = True
                            break
                    if not found:
                        orb_row += " & -"
            orb_row += " \\\\"
            print(orb_row)

        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\end{table}")
        print()


def main():
    """主函数"""
    results_dir = Path('test_results')

    # 定义要提取的文件
    files_to_extract = [
        ('H_lsda.json', 'H', 'LSDA'),
        ('H_uhf.json', 'H', 'UHF'),
        ('C_lsda.json', 'C', 'LSDA'),
        ('C_rhf.json', 'C', 'RHF'),
        ('C_uhf.json', 'C', 'UHF'),
        ('He_lsda.json', 'He', 'LSDA'),
        ('Li_lsda.json', 'Li', 'LSDA'),
        ('Al_lsda.json', 'Al', 'LSDA'),
    ]

    results = []
    for filename, atom, method in files_to_extract:
        filepath = results_dir / filename
        if filepath.exists():
            data = load_json(filepath)
            res = extract_key_data(data, atom, method)
            results.append(res)
        else:
            print(f"⚠️  文件不存在: {filepath}")
            results.append(None)

    # 打印汇总表格
    print_summary_table(results)

    # 生成 LaTeX 表格
    generate_latex_tables(results)

    print("\n" + "=" * 80)
    print("✅ 数据提取完成")
    print("=" * 80)


if __name__ == '__main__':
    main()
