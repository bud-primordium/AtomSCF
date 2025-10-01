from __future__ import annotations

import numpy as np

__all__ = ["hf_exchange_action_s_only"]


def hf_exchange_action_s_only(u_occ: list[np.ndarray], r: np.ndarray) -> list[np.ndarray]:
    r"""最小版 HF 交换（仅 s 轨道，l=0），用于教学与 H 验证。

    对于 s 轨道，若体系仅有单电子（如 H），交换与 Hartree 完全相消，
    本函数返回与输入同形的全零作用（表示交换对 u 的作用结果）。

    Parameters
    ----------
    u_occ : list[numpy.ndarray]
        已占据的径向波函数列表（同一自旋、l=0）。
    r : numpy.ndarray
        径向网格。

    Returns
    -------
    list[numpy.ndarray]
        与 :data:`u_occ` 同形的交换作用结果（此处为全零）。

    Notes
    -----
    - 这是 HF 非局域交换的占位实现，完整实现需引入多极展开与 Slater 径向积分。
    - 在教学阶段，可先用于 H 的 HF 自洽：只剩 v_ext。随后逐步扩展到一般 l 与多电子。
    """
    return [np.zeros_like(u) for u in u_occ]

