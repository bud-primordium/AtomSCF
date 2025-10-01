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

