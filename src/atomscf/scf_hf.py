from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .grid import trapezoid_weights
from .operator import solve_bound_states_fd
from .utils import trapz

__all__ = ["HFConfig", "HFResult", "run_hf_minimal"]


@dataclass
class HFConfig:
    r"""HF 最小实现配置（教学版，仅支持 H）。

    Attributes
    ----------
    Z : int
        原子序数（当前仅支持 1）。
    r : numpy.ndarray
        径向网格。
    w : numpy.ndarray
        梯形权重。
    tol : float
        收敛阈值。
    maxiter : int
        最大迭代数。
    """

    Z: int
    r: np.ndarray
    w: np.ndarray
    tol: float = 1e-7
    maxiter: int = 50


@dataclass
class HFResult:
    r"""HF 最小实现结果容器（H 验证）。"""

    converged: bool
    iterations: int
    eps_1s: float
    u_1s: np.ndarray


def run_hf_minimal(cfg: HFConfig) -> HFResult:
    r"""对 H 原子运行最小 HF 自洽：交换与 Hartree 相消，仅剩外势。

    该实现用于快速验证 HF 思想与数值骨架。对多电子（如 C）不适用。
    """
    if cfg.Z != 1:
        raise NotImplementedError("最小 HF 实现当前仅支持 Z=1（氢原子）")
    r = cfg.r
    v_ext = -1.0 / np.maximum(r, 1e-12)
    # 直接在外势下求解 1s 态（HF 下与单电子精确解一致）
    eps, U = solve_bound_states_fd(r, l=0, v_of_r=v_ext, k=1)
    return HFResult(converged=True, iterations=1, eps_1s=float(eps[0]), u_1s=U[0])

