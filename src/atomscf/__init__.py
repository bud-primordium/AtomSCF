"""atomscf 包
=================

面向教学的原子自旋极化 Hartree–Fock 与 LSDA-DFT 最小实现。

本包优先提供径向（一维）球对称问题的数值工具：

- 径向网格（线性/对数）与积分权重
- 径向二阶导数有限差分与 Hamiltonian 组装
- 基于有限差分的径向本征求解（MVP），用于氢样势验证

后续将扩展：Hartree 势、LSDA 交换关联、HF 非局域交换与完整 SCF 框架。

注：本项目所有文档与注释均使用中文，Docstring 采用 Sphinx + NumPy 风格，公式使用 ``:math:`` 标记。
"""

from atomscf.grid import radial_grid_linear, radial_grid_log, trapezoid_weights
from atomscf.operator import radial_hamiltonian_matrix, solve_bound_states_fd
from atomscf.utils import trapz, normalize_radial_u
from atomscf.shooting import shooting_refine_energy

__all__ = [
    "radial_grid_linear",
    "radial_grid_log",
    "trapezoid_weights",
    "radial_hamiltonian_matrix",
    "solve_bound_states_fd",
    "trapz",
    "normalize_radial_u",
    "shooting_refine_energy",
]

__version__ = "0.1.0"
