#!/usr/bin/env python
"""原子 DFT/HF 计算统一入口。

支持 LSDA/LDA 模式，多种网格与求解器，自动参考数据对比。
"""

import argparse
import sys
import json
from typing import Dict, Tuple
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from atomscf.grid import (
    radial_grid_linear,
    radial_grid_log,
    radial_grid_exp_transformed,
)
from atomscf.scf import SCFConfig, run_lsda_vwn, run_lsda_pz81, run_lsda_x_only
from atomscf.scf_hf import run_hf_scf, HFSCFGeneralConfig
from atomscf.refdata import load_nist_reference, load_nist_lsd
import numpy as np

from atomscf.io import export_for_ppgen
from atomscf.occupations import default_occupations


def build_grid(args):
    """根据参数构建径向网格。

    Returns
    -------
    r, w : np.ndarray
        网格点与积分权重
    params : dict | None
        网格参数（exp 网格需要 delta/Rp）
    """
    if args.grid == "linear":
        r, w = radial_grid_linear(n=args.n, rmin=args.rmin, rmax=args.rmax)
        return r, w, None
    elif args.grid == "log":
        r, w = radial_grid_log(n=args.n, rmin=args.rmin, rmax=args.rmax)
        return r, w, None
    elif args.grid == "exp":
        r, w, delta, Rp = radial_grid_exp_transformed(
            n=args.n,
            rmin=args.rmin,
            rmax=args.rmax,
            total_span=args.total_span,
        )
        return r, w, {"delta": delta, "Rp": Rp}
    else:
        raise ValueError(f"不支持的网格类型: {args.grid}")


def build_config(args, r, w, params=None):
    """根据参数构建配置对象（SCFConfig 或 HFSCFGeneralConfig）。"""
    if args.method == "HF":
        # HF 配置
        # 获取默认占据并转换为 HF 格式
        occ_list = default_occupations(args.Z)

        # 先按 (l, n_index) 分组累加占据数
        occ_by_l_n = {}
        for spec in occ_list:
            l = spec.l
            n = spec.n_index
            key = (l, n)
            if key not in occ_by_l_n:
                occ_by_l_n[key] = 0.0
            # f_per_m 是每个 m 的占据，乘以 (2l+1) 得到总占据
            occ_by_l_n[key] += spec.f_per_m * (2 * l + 1)

        # 转换为 occ_by_l（按 l 分组，按 n 排序）
        occ_by_l = {}
        for (l, n), f in sorted(occ_by_l_n.items()):
            if l not in occ_by_l:
                occ_by_l[l] = []
            occ_by_l[l].append(f)

        # 每个 l 求解的态数（根据占据态数量确定）
        eigs_per_l = {}
        for l, occ_list in occ_by_l.items():
            # 求解的态数至少要包含所有占据态
            eigs_per_l[l] = len(occ_list)

        cfg_dict = {
            "Z": args.Z,
            "r": r,
            "w": w,
            "occ_by_l": occ_by_l,
            "eigs_per_l": eigs_per_l,
            "spin_mode": args.hf_mode,
            "mix_alpha": args.mix_alpha,
            "tol": args.tol,
            "maxiter": args.maxiter,
        }

        # 添加 exp 网格参数（HF 用 delta/Rp 自动选择 transformed）
        if params is not None:
            cfg_dict["delta"] = params["delta"]
            cfg_dict["Rp"] = params["Rp"]

        return HFSCFGeneralConfig(**cfg_dict)
    else:
        # DFT 配置
        cfg_dict = {
            "Z": args.Z,
            "r": r,
            "w": w,
            "lmax": args.lmax,
            "eigs_per_l": args.eigs_per_l,
            "spin_mode": args.mode,
            "xc": args.xc,
            "eig_solver": args.solver,
            "mix_alpha": args.mix_alpha,
            "tol": args.tol,
            "maxiter": args.maxiter,
            "adapt_mixing": args.adapt,
            "mix_alpha_min": 0.05,
            "compute_all_l": True,
            "compute_all_l_mode": "final",
            "mix_kind": "density",
        }

        # 添加 exp 网格参数
        if params is not None:
            cfg_dict["delta"] = params["delta"]
            cfg_dict["Rp"] = params["Rp"]
        # Numerov 可选参数
        if args.solver == "numerov":
            if getattr(args, "numerov_samples", None) is not None:
                cfg_dict["numerov_samples"] = args.numerov_samples
            if getattr(args, "numerov_bisection_iter", None) is not None:
                cfg_dict["numerov_bisection_iter"] = args.numerov_bisection_iter

        return SCFConfig(**cfg_dict)


def run_scf(cfg, args):
    """运行 SCF 计算。"""
    if args.method == "HF":
        result = run_hf_scf(cfg)
    elif args.xc == "VWN":
        result = run_lsda_vwn(
            cfg, verbose=args.verbose, progress_every=args.progress_every
        )
    elif args.xc == "PZ81":
        result = run_lsda_pz81(
            cfg, verbose=args.verbose, progress_every=args.progress_every
        )
    elif args.xc == "X_ONLY":
        result = run_lsda_x_only(
            cfg, verbose=args.verbose, progress_every=args.progress_every
        )
    else:
        raise ValueError(f"不支持的 XC 泛函: {args.xc}")


    return result


def print_results(result, args, cfg=None):
    """格式化输出结果。"""
    print("\n" + "=" * 70)
    if args.method == "HF":
        print(f"原子计算结果 (Z={args.Z}, {args.hf_mode})")
    else:
        print(f"原子计算结果 (Z={args.Z}, {args.mode}-{args.xc})")
    print("=" * 70)

    # 收敛信息
    status = "✅ 收敛" if result.converged else "❌ 未收敛"
    print(f"\n状态: {status} ({result.iterations} 轮)")

    # 能量
    if args.method == "HF":
        # HF 结果
        print(f"\n总能量: {result.E_total:.8f} Ha")
        print(f"  动能: {result.E_kinetic:.8f} Ha")
        print(f"  核吸引能: {result.E_ext:.8f} Ha")
        print(f"  Hartree 能: {result.E_hartree:.8f} Ha")
        print(f"  交换能: {result.E_exchange:.8f} Ha")
    else:
        # DFT 结果
        print(f"\n总能量: {result.energies['E_total']:.8f} Ha")
        print(f"  外势能: {result.energies['E_ext']:.8f} Ha")
        print(f"  Hartree: {result.energies['E_H']:.8f} Ha")
        print(f"  交换能: {result.energies['E_x']:.8f} Ha")
        if "E_c" in result.energies:
            print(f"  关联能: {result.energies['E_c']:.8f} Ha")
        if "E_kin" in result.energies:
            print(f"  动能 T_s: {result.energies['E_kin']:.8f} Ha")

    # 能级（只显示束缚态，epsilon < 0）
    print("\n轨道能级（束缚态）:")

    if args.method == "HF":
        # HF 输出
        print(f"{'轨道':>6} {'占据':>6} {'epsilon (Ha)':>15}")
        print("-" * 32)

        l_symbols = "spdfgh"
        for ang_l in sorted(result.eigenvalues_by_l.keys()):
            eps_arr = result.eigenvalues_by_l[ang_l]
            occ_arr = cfg.occ_by_l[ang_l] if cfg else []
            for n_idx, eps in enumerate(eps_arr):
                if eps >= 0:
                    continue
                n_quantum = n_idx + ang_l + 1
                l_symbol = l_symbols[ang_l] if ang_l < len(l_symbols) else f"l={ang_l}"
                orbital = f"{n_quantum}{l_symbol}"
                occ = occ_arr[n_idx] if n_idx < len(occ_arr) else 0.0
                print(f"{orbital:>6} {occ:6.1f} {eps:15.6f}")
    else:
        # DFT 输出
        if args.mode == "LDA":
            print(f"{'轨道':>6} {'占据':>6} {'epsilon (Ha)':>15}")
            print("-" * 32)
        else:
            print(f"{'轨道':>6} {'自旋':>6} {'占据':>6} {'epsilon (Ha)':>15}")
            print("-" * 42)

        # 获取占据数信息
        occ_list = default_occupations(args.Z)
        occ_map = {}  # (n_quantum, l, spin) -> occupation
        for spec in occ_list:
            n_quantum = spec.n_index + spec.l + 1
            # 每个 (n, l) 轨道有 2l+1 个 m，每个 m 的占据为 f_per_m
            occ_per_spin = (2 * spec.l + 1) * spec.f_per_m
            # LSDA: 分 up/down 存储
            if spec.spin == "up":
                occ_map[(n_quantum, spec.l, "up")] = occ_per_spin
            elif spec.spin == "down":
                occ_map[(n_quantum, spec.l, "down")] = occ_per_spin
            else:  # "both" - 均分到 up 和 down
                occ_map[(n_quantum, spec.l, "up")] = occ_per_spin
                occ_map[(n_quantum, spec.l, "down")] = occ_per_spin

        # 收集所有 (l, sigma) 组合，按 l 排序
        all_keys = sorted(
            set((ang_l, sigma) for (ang_l, sigma) in result.eps_by_l_sigma.keys())
        )

        l_symbols = "spdfgh"

        for ang_l, sigma in all_keys:
            if (ang_l, sigma) in result.eps_by_l_sigma:
                eps_arr = result.eps_by_l_sigma[(ang_l, sigma)]
                for n_idx, eps in enumerate(eps_arr):
                    # 只显示束缚态
                    if eps >= 0:
                        continue

                    n_quantum = n_idx + ang_l + 1
                    l_symbol = l_symbols[ang_l] if ang_l < len(l_symbols) else f"l={ang_l}"
                    orbital = f"{n_quantum}{l_symbol}"

                    # 查找占据数
                    occupation = occ_map.get((n_quantum, ang_l, sigma), 0.0)

                    if args.mode == "LDA":
                        # LDA 模式：不显示自旋，占据数翻倍（因为 up=down）
                        if sigma == "up":  # 只显示一次
                            total_occ = occupation * 2  # up + down
                            print(f"{orbital:>6} {total_occ:6.1f} {eps:15.6f}")
                    else:
                        # LSDA 模式：显示自旋和占据数
                        spin_symbol = "↑" if sigma == "up" else "↓"
                        print(
                            f"{orbital:>6} {spin_symbol:>6} {occupation:6.1f} {eps:15.6f}"
                        )

    print()


def compare_with_ref(result, ref_path, args):
    """与参考数据对比。"""
    # 优先使用内嵌数据（Z<=18）
    nist_mode = "LDA" if args.mode == "LDA" else "LSD"

    try:
        if args.Z <= 18:
            # 使用内嵌 NIST 数据
            ref_data = load_nist_reference(args.Z, mode=nist_mode)
        else:
            # Z>18，尝试从路径加载或提示
            if ref_path:
                ref_data = load_nist_lsd(ref_path)
            else:
                print(f"\n⚠️  Z={args.Z} 超出内嵌数据范围（Z=1-18）")
                print(
                    f"   请访问 https://www.nist.gov/pml/atomic-reference-data-electronic-structure-calculations"
                )
                print(f"   获取参考数据，并使用 --compare-ref <path> 指定路径")
                return
    except Exception as e:
        print(f"\n⚠️  无法加载参考数据: {e}")
        return

    print("\n" + "=" * 70)
    print(f"与 NIST {nist_mode} 参考数据对比")
    print("=" * 70)

    # 对比总能
    if "Etot" in ref_data:
        ref_etot = ref_data["Etot"]
        calc_etot = result.energies["E_total"]
        err_etot = abs(calc_etot - ref_etot) / abs(ref_etot)
        print(f"\n总能:")
        print(f"  参考: {ref_etot:.8f} Ha")
        print(f"  计算: {calc_etot:.8f} Ha")
        print(f"  相对误差: {err_etot:.3e}")

    # 对比能级（简化版：只对比前几个主要轨道）
    print(f"\n能级对比:")
    print(f"{'轨道':>8} {'参考 (Ha)':>15} {'计算 (Ha)':>15} {'相对误差':>12}")
    print("-" * 55)

    # 常见轨道标签映射
    common_orbitals = {
        "1sD": (0, 0, "down"),  # (n_idx, l, spin)
        "1su": (0, 0, "up"),
        "2sD": (1, 0, "down"),
        "2su": (1, 0, "up"),
        "2pD": (0, 1, "down"),
        "2pu": (0, 1, "up"),
        "3sD": (2, 0, "down"),
        "3su": (2, 0, "up"),
        "3pD": (1, 1, "down"),
        "3pu": (1, 1, "up"),
    }

    for label, (n_idx, l, sigma) in common_orbitals.items():
        if label in ref_data:
            ref_eps = ref_data[label]
            if (l, sigma) in result.eps_by_l_sigma and n_idx < len(
                result.eps_by_l_sigma[(l, sigma)]
            ):
                calc_eps = result.eps_by_l_sigma[(l, sigma)][n_idx]
                err = abs(calc_eps - ref_eps) / abs(ref_eps)
                print(f"{label:>8} {ref_eps:15.8f} {calc_eps:15.8f} {err:12.3e}")


def export_results(result, export_path, args, cfg=None):
    """导出结果到 JSON（兼容 DFT/HF；增强波函数采样与自旋标注）。"""

    def _extract_hf_energies(hf_result):
        """从 HF 结果对象中提取能量字典（HFSCFGeneralResult 无 energies 属性）。"""
        energies = {}
        if hasattr(hf_result, 'E_total'):
            energies['E_total'] = float(hf_result.E_total)
        if hasattr(hf_result, 'E_kinetic'):
            energies['E_kin'] = float(hf_result.E_kinetic)
        if hasattr(hf_result, 'E_ext'):
            energies['E_ext'] = float(hf_result.E_ext)
        if hasattr(hf_result, 'E_hartree'):
            energies['E_H'] = float(hf_result.E_hartree)
        if hasattr(hf_result, 'E_exchange'):
            energies['E_x'] = float(hf_result.E_exchange)
        return energies

    def _build_occ_map(Z: int) -> Dict[Tuple[int, int, str], float]:
        """构建占据查找表: key=(n_quantum, l, spin)->(2l+1)f_per_m。"""
        occ_list = default_occupations(Z)
        m = {}
        for spec in occ_list:
            n_quantum = spec.n_index + spec.l + 1
            m[(n_quantum, spec.l, spec.spin)] = (2 * spec.l + 1) * spec.f_per_m
        return m

    def _determine_majority_spin(eps_by_l_sigma: Dict[Tuple[int, str], np.ndarray],
                                 l: int, n_idx: int, sigma: str,
                                 occ_map: Dict[Tuple[int, int, str], float]) -> str:
        """判定同一 (n,l) 的多数/少数自旋。

        返回 'majority'/'minority'/'unknown'。
        规则：
        - 若该 (n,l) 同时存在 up/down 能量，能量更负者为 majority
        - 若仅一自旋存在，按占据数较大者为 majority；占据数相等返回 unknown
        """
        other = 'down' if sigma == 'up' else 'up'
        key_this = (l, sigma)
        key_other = (l, other)
        n_quantum = n_idx + l + 1
        if key_this in eps_by_l_sigma and key_other in eps_by_l_sigma and \
           n_idx < len(eps_by_l_sigma[key_this]) and n_idx < len(eps_by_l_sigma[key_other]):
            e_this = float(eps_by_l_sigma[key_this][n_idx])
            e_other = float(eps_by_l_sigma[key_other][n_idx])
            if e_this < e_other:
                return 'majority'
            elif e_this > e_other:
                return 'minority'
            else:
                return 'unknown'
        # 回退：按占据数
        occ_this = occ_map.get((n_quantum, l, sigma), 0.0)
        occ_other = occ_map.get((n_quantum, l, other), 0.0)
        if occ_this > occ_other:
            return 'majority'
        if occ_this < occ_other:
            return 'minority'
        return 'unknown'

    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    # 元数据签名（网格/求解器/SCF 配置）
    grid_sig = {
        'type': args.grid,
        'n': args.n,
        'rmin': args.rmin,
        'rmax': args.rmax,
        'total_span': args.total_span if args.grid == 'exp' else None,
    }
    if args.method == 'DFT':
        solver_sig = {
            'type': args.solver,
            'eigs_per_l': args.eigs_per_l,
            'lmax': args.lmax,
        }
    else:
        # HF: 是否使用变量变换由 delta/Rp 是否存在决定
        # run_atom.py 构建 HF 配置时，exp 网格会注入 delta/Rp
        # 这里无法直接访问 cfg，只给出近似签名
        solver_sig = {
            'type': 'transformed' if args.grid == 'exp' else 'fd',
        }

    scf_cfg = {
        'tol': args.tol,
        'maxiter': args.maxiter,
        'mix_alpha': args.mix_alpha,
    }

    # 能量字典（DFT 直接使用；HF 需提取）
    energies = getattr(result, 'energies', None)
    if energies is None:
        energies = _extract_hf_energies(result)

    data = {
        'metadata': {
            'Z': args.Z,
            'method': args.method,
            'mode': (args.mode if args.method == 'DFT' else args.hf_mode),
            'xc': (args.xc if args.method == 'DFT' else None),
            'grid_signature': grid_sig,
            'solver_signature': solver_sig,
            'scf_config': scf_cfg,
            'converged': bool(getattr(result, 'converged', False)),
            'iterations': int(getattr(result, 'iterations', 0)),
        },
        'energies': energies,
        'levels': {},
        'wavefunctions': {},
    }

    l_symbols = 'spdfgh'
    occ_map = _build_occ_map(args.Z)

    # 能级导出 + 自旋标注
    if args.method == 'DFT':
        for (l, sigma), eps_arr in getattr(result, 'eps_by_l_sigma', {}).items():
            for n_idx, eps in enumerate(eps_arr):
                n_quantum = n_idx + l + 1
                key = f"{n_quantum}{l_symbols[l]}_{sigma}"
                spin_type = _determine_majority_spin(result.eps_by_l_sigma, l, n_idx, sigma, occ_map)
                nist_label = ('D' if spin_type == 'majority' else ('u' if spin_type == 'minority' else None))
                occ_val = float(occ_map.get((n_quantum, l, sigma), 0.0))
                data['levels'][key] = {
                    'value': float(eps),
                    'spin': sigma,
                    'spin_type': spin_type,
                    'occupation': occ_val,
                    'nist_label': nist_label,
                }
    else:
        # HF: RHF/UHF 分支
        if getattr(result, 'spin_mode', 'RHF') == 'UHF' and getattr(result, 'eigenvalues_by_l_spin', None) is not None:
            eps_dict = result.eigenvalues_by_l_spin
            # 构造类似结构以复用多数/少数判定
            for (l, sigma), eps_arr in eps_dict.items():
                for n_idx, eps in enumerate(eps_arr):
                    n_quantum = n_idx + l + 1
                    key = f"{n_quantum}{l_symbols[l]}_{sigma}"
                    spin_type = 'unknown'
                    # UHF 情形可以通过能量比较判定
                    try:
                        spin_type = _determine_majority_spin(eps_dict, l, n_idx, sigma, occ_map)
                    except Exception:
                        pass
                    data['levels'][key] = {
                        'value': float(eps),
                        'spin': sigma,
                        'spin_type': spin_type,
                        'occupation': float(occ_map.get((n_quantum, l, sigma), 0.0)),
                        'nist_label': ('D' if spin_type == 'majority' else ('u' if spin_type == 'minority' else None)),
                    }
        else:
            # RHF：无自旋标签
            for l, eps_arr in getattr(result, 'eigenvalues_by_l', {}).items():
                for n_idx, eps in enumerate(eps_arr):
                    n_quantum = n_idx + l + 1
                    key = f"{n_quantum}{l_symbols[l]}"
                    data['levels'][key] = {
                        'value': float(eps),
                        'spin': None,
                        'spin_type': None,
                        'occupation': float(occ_map.get((n_quantum, l, 'up'), 0.0) + occ_map.get((n_quantum, l, 'down'), 0.0)),
                        'nist_label': None,
                    }

    # 波函数导出（均匀稀疏采样约 200 点）
    # 注意：DFT 与 UHF 的存储结构为 (l, sigma) -> U[k, n_r]；RHF 为 l -> List[u]
    if cfg is not None and hasattr(cfg, 'r'):
        r = cfg.r
        n_points = len(r)
        sample_stride = max(1, n_points // 200)
        sample_idx = np.arange(0, n_points, sample_stride)
        if sample_idx[-1] != n_points - 1:
            sample_idx = np.append(sample_idx, n_points - 1)
        sampled_r = r[sample_idx]

        # DFT: u_by_l_sigma
        if args.method == 'DFT' and hasattr(result, 'u_by_l_sigma') and result.u_by_l_sigma:
            for (l, sigma), U in result.u_by_l_sigma.items():
                for n_idx in range(U.shape[0]):
                    u = U[n_idx]
                    label = f"{n_idx + l + 1}{l_symbols[l]}_{sigma}"
                    data['wavefunctions'][label] = {
                        'r': sampled_r.tolist(),
                        'u': u[sample_idx].tolist(),
                        'grid_type': args.grid,
                        'n_total': int(n_points),
                    }

        # HF-UHF: orbitals_by_l_spin
        if args.method == 'HF' and getattr(result, 'spin_mode', 'RHF') == 'UHF' and getattr(result, 'orbitals_by_l_spin', None) is not None:
            for (l, sigma), u_list in result.orbitals_by_l_spin.items():
                for n_idx, u in enumerate(u_list):
                    label = f"{n_idx + l + 1}{l_symbols[l]}_{sigma}"
                    data['wavefunctions'][label] = {
                        'r': sampled_r.tolist(),
                        'u': u[sample_idx].tolist(),
                        'grid_type': args.grid,
                        'n_total': int(n_points),
                    }

        # HF-RHF: orbitals_by_l
        if args.method == 'HF' and getattr(result, 'orbitals_by_l', None) is not None and getattr(result, 'spin_mode', 'RHF') != 'UHF':
            for l, u_list in result.orbitals_by_l.items():
                for n_idx, u in enumerate(u_list):
                    label = f"{n_idx + l + 1}{l_symbols[l]}"
                    data['wavefunctions'][label] = {
                        'r': sampled_r.tolist(),
                        'u': u[sample_idx].tolist(),
                        'grid_type': args.grid,
                        'n_total': int(n_points),
                    }

    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 结果已导出到: {export_path}")


def main():
    parser = argparse.ArgumentParser(
        description="原子 DFT/HF 计算统一入口",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 必选参数
    parser.add_argument("--Z", type=int, required=True, help="原子序数")

    # 物理参数
    parser.add_argument(
        "--method",
        type=str,
        default="DFT",
        choices=["DFT", "HF"],
        help="计算方法：DFT 或 HF"
    )
    parser.add_argument(
        "--mode", type=str, default="LSDA", choices=["LSDA", "LDA"], help="自旋模式（DFT）"
    )
    parser.add_argument(
        "--xc",
        type=str,
        default="VWN",
        choices=["VWN", "PZ81", "X_ONLY"],
        help="XC 泛函（DFT）",
    )
    parser.add_argument(
        "--hf-mode",
        type=str,
        default="RHF",
        choices=["RHF", "UHF"],
        help="HF 模式（仅当 --method HF 时使用）"
    )
    parser.add_argument("--lmax", type=int, default=3, help="最大角动量")
    parser.add_argument(
        "--eigs-per-l", type=int, default=3, help="每个 l 求解的本征态数"
    )

    # 数值参数
    parser.add_argument(
        "--solver",
        type=str,
        default="transformed",
        choices=["fd", "fd5", "fd5_aux", "transformed", "numerov"],
        help="本征值求解器（默认 transformed，推荐）",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default="exp",
        choices=["linear", "log", "exp"],
        help="径向网格类型（默认 exp，推荐）",
    )
    parser.add_argument(
        "--n", type=int, default=2000, help="网格点数（推荐 2000-3000）"
    )
    parser.add_argument("--rmin", type=float, default=1e-6, help="最小半径 (Bohr)")
    parser.add_argument("--rmax", type=float, default=150.0, help="最大半径 (Bohr)")
    parser.add_argument(
        "--total-span", type=float, default=7.0, help="指数网格参数（仅 exp 网格）"
    )

    # SCF 参数
    parser.add_argument("--mix-alpha", type=float, default=0.3, help="密度混合参数")
    parser.add_argument("--tol", type=float, default=1e-6, help="收敛阈值")
    parser.add_argument("--maxiter", type=int, default=200, help="最大迭代次数")
    parser.add_argument("--adapt", action="store_true", help="启用自适应混合")

    # 功能参数
    parser.add_argument(
        "--compare-ref",
        action="store_true",
        default=True,
        help="对比 NIST 参考数据（Z<=18 自动使用内嵌数据）",
    )
    parser.add_argument(
        "--ref-path",
        type=str,
        default=None,
        help="参考数据文件路径（Z>18 时需要）",
    )
    parser.add_argument("--export", type=str, default=None, help="导出结果到 JSON")
    parser.add_argument(
        "--export-ppgen",
        type=str,
        default=None,
        help="导出原子参考数据供伪势生成器使用（仅 LDA 模式）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="详细输出（默认启用，使用 --no-verbose 禁用）",
    )
    parser.add_argument(
        "--no-verbose",
        dest="verbose",
        action="store_false",
        help="禁用详细输出",
    )
    parser.add_argument("--progress-every", type=int, default=10, help="进度输出间隔")

    # Numerov 可选参数（仅 solver=numerov 时有效）
    parser.add_argument(
        "--numerov-samples",
        type=int,
        default=None,
        help="Numerov 能量扫描采样数（节点计数法）",
    )
    parser.add_argument(
        "--numerov-bisection-iter",
        type=int,
        default=None,
        help="Numerov 二分迭代次数（节点计数法）",
    )

    args = parser.parse_args()

    # 检查数值参数组合，给出精度警告
    if args.grid != "exp" or args.solver != "transformed":
        print("\n⚠️  警告：当前配置可能导致精度较低")
        print(f"   当前设置: --grid {args.grid} --solver {args.solver}")
        print(f"   推荐配置: --grid exp --solver transformed")

    # 构建网格
    r, w, params = build_grid(args)

    # 构建配置
    cfg = build_config(args, r, w, params)

    # 运行 SCF
    import time
    t_start = time.time()
    result = run_scf(cfg, args)
    t_elapsed = time.time() - t_start

    # 输出结果
    print_results(result, args, cfg)
    print(f"\n⏱️  总用时: {t_elapsed:.2f}s")

    # 对比参考（Z≤18 自动启用，或用户明确要求，DFT 模式）
    if args.method == "DFT" and (args.Z <= 18 or args.compare_ref):
        compare_with_ref(result, args.ref_path, args)

    # 导出（如果需要）
    if args.export:
        export_results(result, args.export, args, cfg)

    # 导出供伪势生成器使用
    if args.export_ppgen:
        if cfg.spin_mode != "LDA":
            print("\n⚠️  --export-ppgen 仅支持 LDA 模式，请添加 --mode LDA")
        else:
            export_for_ppgen(result, cfg, args.export_ppgen)
            print(f"\n✅ AtomPPGen 参考数据已导出到: {args.export_ppgen}")


if __name__ == "__main__":
    main()
