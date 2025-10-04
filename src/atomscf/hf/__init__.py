"""Hartree-Fock 非局域交换模块

本子模块实现原子 HF 计算所需的核心组件：

- **Slater 径向积分** (`slater.py`): 非局域交换核 R^k_{ij}(r)
- **角动量耦合** (`angular.py`): Wigner-3j 系数与 allowed_k 规则
- **交换算子** (`exchange.py`): K̂ 算子的实现与应用
- **HF SCF** (`scf_hf.py`): Hartree-Fock 自洽场主循环

技术路线
========

本项目采用**算子-作用式**实现精确 HF（而非线性化近似）：

1. 使用 `scipy.sparse.linalg.LinearOperator` 封装 Fock 算子
2. 通过迭代本征求解器（LOBPCG）计算低端本征态
3. Slater 积分在每次算子应用时实时计算（避免存储密集矩阵）

与 LSDA 的核心区别
==================

+-------------------+-------------------------+-------------------------+
| 方面              | LSDA                    | HF                      |
+===================+=========================+=========================+
| 交换处理          | 局域势 v_x^σ(r)         | 非局域算子 K̂           |
| 势能表示          | 纯函数                  | 积分算子                |
| 求解复杂度        | 低（矩阵对角化）        | 高（迭代本征）          |
| 代码行数          | ~800 行                 | ~1200 行（+400 行）     |
+-------------------+-------------------------+-------------------------+

"""

from .slater import (
    SlaterIntegralCache,
    slater_integral_k0,
    slater_integral_radial,
)

__all__ = [
    # Slater 积分
    "slater_integral_radial",
    "slater_integral_k0",
    "SlaterIntegralCache",
]
