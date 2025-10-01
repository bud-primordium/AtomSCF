from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Spin = Literal["up", "down"]


@dataclass(frozen=True)
class OrbitalSpec:
    """轨道占据信息（径向通道）

    Attributes
    ----------
    l : int
        角动量量子数 :math:`\ell`。
    n_index : int
        同一 :math:`\ell` 通道内的径向量子数索引（0 表示最低能，即 1s/2p 等的“1”）。
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
    """针对 H 与 C，返回球对称平均下的默认占据方案。

    Parameters
    ----------
    Z : int
        原子序数，目前支持 1（H）与 6（C）。

    Returns
    -------
    list[OrbitalSpec]
        轨道占据列表（径向通道）。

    Notes
    -----
    - H (Z=1): 1s 单占（默认自旋向上）。
    - C (Z=6): 1s^2 2s^2 2p^2；为保持球对称，2p 壳层采用 m 平均且自旋极化：
      2p_up :math:`f=2/3`，2p_down :math:`f=0`。
    - 若需其他元素/组态，可在后续扩展或提供手动占据接口。
    """
    if Z == 1:
        return [
            OrbitalSpec(l=0, n_index=0, spin="up", f_per_m=1.0, label="1s_up"),
        ]
    if Z == 6:
        return [
            # 1s^2
            OrbitalSpec(l=0, n_index=0, spin="up", f_per_m=1.0, label="1s_up"),
            OrbitalSpec(l=0, n_index=0, spin="down", f_per_m=1.0, label="1s_down"),
            # 2s^2（n_index=1 表示第二个 s 轨道，即 2s）
            OrbitalSpec(l=0, n_index=1, spin="up", f_per_m=1.0, label="2s_up"),
            OrbitalSpec(l=0, n_index=1, spin="down", f_per_m=1.0, label="2s_down"),
            # 2p^2：m 平均 + 自旋极化（Hund 规则），f_per_m=2/3，down=0
            OrbitalSpec(l=1, n_index=0, spin="up", f_per_m=2.0 / 3.0, label="2p_up"),
            # down 通道留空（0），不返回占据以免影响密度
        ]
    raise ValueError("当前仅支持 Z=1 (H) 与 Z=6 (C) 的默认占据方案")

