from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Spin = Literal["up", "down"]


@dataclass(frozen=True)
class OrbitalSpec:
    r"""轨道占据信息（径向通道）

    Attributes
    ----------
    l : int
        角动量量子数 :math:`\ell`。
    n_index : int
        同一 :math:`\ell` 通道内的径向量子数索引（0 表示最低能，即 1s/2p 等的"1"）。
    spin : {"up", "down"}
        自旋通道。
    f_per_m : float
        每个 :math:`m` 的分数占据 :math:`f_{nl\sigma}`；该通道总电子数为 :math:`(2\ell+1) f_{nl\sigma}`。
    label : str
        人类可读的标签（如 "1s_up"）。
    """

    l: int
    n_index: int
    spin: Spin
    f_per_m: float
    label: str


def default_occupations(Z: int) -> list[OrbitalSpec]:
    """返回原子基态的默认电子占据方案（球对称平均）。

    Parameters
    ----------
    Z : int
        原子序数 (1-18)。

    Returns
    -------
    list[OrbitalSpec]
        轨道占据列表（径向通道）。

    Notes
    -----
    - **仅支持 Z=1-18**，填充顺序为 1s → 2s → 2p → 3s → 3p
    - 闭壳层原子 (He, Be, Ne, Mg, Ar): 所有轨道自旋配对
    - 开壳层原子: 采用 Hund 规则（最大自旋多重度），球对称平均占据
    - **警告**: Z>18 需要考虑能级交叉（如 K: 4s 先于 3d），当前实现不支持
    - 示例:
      - H (Z=1): 1s¹ (自旋向上)
      - He (Z=2): 1s²
      - C (Z=6): 1s² 2s² 2p² (2p: ↑↑, m 平均)
      - Ne (Z=10): 1s² 2s² 2p⁶
      - Ar (Z=18): [Ne] 3s² 3p⁶
    """
    occ = []

    # 闭壳层填充辅助函数
    def add_closed_shell(l, n_index, label_prefix):
        occ.append(OrbitalSpec(l=l, n_index=n_index, spin="up", f_per_m=1.0, label=f"{label_prefix}_up"))
        occ.append(OrbitalSpec(l=l, n_index=n_index, spin="down", f_per_m=1.0, label=f"{label_prefix}_down"))

    # 部分填充辅助函数（用于开壳层，简单自旋极化）
    def add_partial_shell(l, n_index, label_prefix, n_electrons):
        max_per_shell = 2 * (2 * l + 1)
        if n_electrons <= 0:
            return
        elif n_electrons >= max_per_shell:
            add_closed_shell(l, n_index, label_prefix)
        else:
            # Hund 规则：先填自旋向上
            n_up = min(n_electrons, 2 * l + 1)
            n_down = n_electrons - n_up
            if n_up > 0:
                occ.append(OrbitalSpec(l=l, n_index=n_index, spin="up",
                                     f_per_m=n_up / (2 * l + 1), label=f"{label_prefix}_up"))
            if n_down > 0:
                occ.append(OrbitalSpec(l=l, n_index=n_index, spin="down",
                                     f_per_m=n_down / (2 * l + 1), label=f"{label_prefix}_down"))

    # 按原子序数填充（基态电子组态）
    remaining = Z

    # 1s (n=1, l=0)
    n_1s = min(remaining, 2)
    add_partial_shell(0, 0, "1s", n_1s)
    remaining -= n_1s
    if remaining == 0:
        return occ

    # 2s (n=2, l=0)
    n_2s = min(remaining, 2)
    add_partial_shell(0, 1, "2s", n_2s)
    remaining -= n_2s
    if remaining == 0:
        return occ

    # 2p (n=2, l=1)
    n_2p = min(remaining, 6)
    add_partial_shell(1, 0, "2p", n_2p)
    remaining -= n_2p
    if remaining == 0:
        return occ

    # 3s (n=3, l=0)
    n_3s = min(remaining, 2)
    add_partial_shell(0, 2, "3s", n_3s)
    remaining -= n_3s
    if remaining == 0:
        return occ

    # 3p (n=3, l=1)
    n_3p = min(remaining, 6)
    add_partial_shell(1, 1, "3p", n_3p)
    remaining -= n_3p
    if remaining == 0:
        return occ

    raise ValueError(f"Z={Z} 超出当前支持范围 (1-18)")

