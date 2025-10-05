from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

from .occupations import OrbitalSpec, default_occupations

__all__ = [
    "export_levels_csv",
    "export_wavefunctions_csv",
    "export_energies_json",
    "export_for_ppgen",
]


def export_levels_csv(
    out_path: str | Path,
    eps_by_l_sigma: Dict[Tuple[int, str], np.ndarray],
    occ: List[OrbitalSpec] | None = None,
) -> None:
    """导出能级表为 CSV：列为 `l,spin,n_index,occ,eps(Ha)`。

    若提供占据表，则对应通道写入总占据数 `(2l+1)*f_per_m`，否则为 0。
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # 构建占据查找表
    occ_map: Dict[Tuple[int, str, int], float] = {}
    if occ is None:
        occ = []
    for spec in occ:
        occ_map[(spec.l, spec.spin, spec.n_index)] = (2 * spec.l + 1) * spec.f_per_m

    with p.open("w", encoding="utf-8") as f:
        f.write("l,spin,n_index,occ,eps(Ha)\n")
        for (l, spin), eps in sorted(eps_by_l_sigma.items()):
            for n_index, e in enumerate(eps):
                occ_val = occ_map.get((l, spin, n_index), 0.0)
                f.write(f"{l},{spin},{n_index},{occ_val:.6f},{float(e):.12f}\n")


def export_wavefunctions_csv(
    out_dir: str | Path,
    r: np.ndarray,
    u_by_l_sigma: Dict[Tuple[int, str], np.ndarray],
    which: List[Tuple[int, str, int]] | None = None,
) -> None:
    """导出指定通道的径向波函数至 CSV，列为 `r,u(r)`。

    参数 `which` 为三元组列表 (l, spin, n_index)。若为 None，则导出所有通道的所有已求解态。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if which is None:
        which = []
        for key, U in u_by_l_sigma.items():
            l, spin = key
            for n_index in range(U.shape[0]):
                which.append((l, spin, n_index))

    for l, spin, n_index in which:
        U = u_by_l_sigma[(l, spin)]
        if not (0 <= n_index < U.shape[0]):
            continue
        u = U[n_index]
        data = np.column_stack([r, u])
        fn = out / f"wavefunc_l{l}_{spin}_n{n_index}.csv"
        np.savetxt(fn, data, delimiter=",", header="r,u(r)")


def export_energies_json(out_path: str | Path, energies: dict) -> None:
    """导出总能与分能为 JSON。"""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(energies, f, indent=2, ensure_ascii=False)


def export_for_ppgen(
    result,  # SCFResult
    cfg,  # SCFConfig
    output_path: str | Path,
    rc_suggestions: Dict[int, float] | None = None
) -> None:
    """导出原子参考数据供伪势生成器使用。

    Parameters
    ----------
    result : SCFResult
        SCF 结果（包含能级、波函数、势等）
    cfg : SCFConfig
        SCF 配置（包含 r, w, Z, xc 等）
    output_path : str | Path
        输出 JSON 文件路径
    rc_suggestions : dict | None
        各角动量的建议截断半径（Bohr），例如 {0: 2.0, 1: 2.2, 2: 2.4}

    Notes
    -----
    - 仅支持 LDA 模式（spin_mode="LDA"），输出自旋无关势
    - JSON 格式专为 AtomPPGen 设计
    """
    if cfg.spin_mode != "LDA":
        raise ValueError("export_for_ppgen 仅支持 LDA 模式（spin_mode='LDA'）")

    # 元素符号映射
    Z_to_element = {
        1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
        9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
        16: "S", 17: "Cl", 18: "Ar",
    }
    element = Z_to_element.get(cfg.Z, f"Z{cfg.Z}")

    # 网格信息
    r = cfg.r
    dr = np.diff(r, prepend=0.0)  # 简化：dr[0] = 0

    # 势信息（LDA 模式下 up = down，取 up 即可）
    v_ext = -float(cfg.Z) / np.maximum(r, 1e-12)
    v_H = result.v_h
    v_xc = result.v_x_up + (result.v_c_up if result.v_c_up is not None else 0.0)
    v_eff = v_ext + v_H + v_xc

    # 占据信息（从 cfg.occ 提取）
    occ_list = cfg.occ or default_occupations(cfg.Z)
    occ_map: Dict[Tuple[int, str], float] = {}  # (n_quantum, l) -> total_occ
    for spec in occ_list:
        n_quantum = spec.n_index + spec.l + 1
        key = (n_quantum, spec.l)
        if key not in occ_map:
            occ_map[key] = 0.0
        occ_map[key] += (2 * spec.l + 1) * spec.f_per_m

    # 轨道信息（合并 up/down，LDA 模式下相同）
    orbitals = []
    processed_keys = set()

    for (l, sigma), eps_arr in sorted(result.eps_by_l_sigma.items()):
        if sigma != "up":
            continue  # LDA 模式下 up = down，只处理一次

        U_arr = result.u_by_l_sigma[(l, sigma)]

        for n_idx, (eps, u) in enumerate(zip(eps_arr, U_arr)):
            n_quantum = n_idx + l + 1
            key = (n_quantum, l)

            if key in processed_keys:
                continue
            processed_keys.add(key)

            occ_val = occ_map.get(key, 0.0)

            # 只导出有占据或低能级（前几个）的轨道
            if occ_val > 0 or n_idx < 3:
                orbitals.append({
                    "n": n_quantum,
                    "l": l,
                    "occupation": float(occ_val),
                    "epsilon": float(eps),
                    "u": u.tolist(),
                })

    # 构建输出字典
    data = {
        "metadata": {
            "Z": cfg.Z,
            "element": element,
            "xc": cfg.xc,
            "mode": cfg.spin_mode,
            "solver": cfg.eig_solver,
            "grid_type": "unknown",  # 暂不推断
            "n_points": len(r),
            "rmax": float(r[-1]),
        },
        "grid": {
            "r": r.tolist(),
            "dr": dr.tolist(),
        },
        "potential": {
            "v_eff": v_eff.tolist(),
            "v_ext": v_ext.tolist(),
            "v_H": v_H.tolist(),
            "v_xc": v_xc.tolist(),
        },
        "orbitals": orbitals,
    }

    if rc_suggestions is not None:
        data["rc_suggestions"] = {str(k): float(v) for k, v in rc_suggestions.items()}

    # 写入 JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


