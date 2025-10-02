"""XC 常量集中维护
====================

集中维护 LSDA 相关常量，便于校准与统一管理。

参考与核对来源（建议）：
- PZ81：Perdew & Zunger, Phys. Rev. B 23, 5048 (1981)
- VWN：Vosko, Wilk, Nusair, Can. J. Phys. 58, 1200 (1980)（VWN5/RPA 参数）
- Libxc 项目源码（对应 LDA_C_PZ、LDA_C_VWN 系列实现）

注意：不同文献/实现有细微差异（如 VWN 变体），需在文档与代码中标注所用变体。
"""

from __future__ import annotations

# PZ81 参数（非极化、全极化）：(A, B, C, D, gamma, beta1, beta2)
PZ81_PARAMS = {
    "unpolarized": (0.031091, -0.048, 0.0020, -0.0116, -0.1423, 1.0529, 0.3334),
    "polarized": (0.015545, -0.0269, 0.0007, -0.0048, -0.0843, 1.3981, 0.2611),
}

# VWN5/RPA 参数（常见取值）：(A, x0, b, c)
VWN5_RPA_PARAMS = {
    "unpolarized": (0.0310907, -0.10498, 3.72744, 12.9352),
    "polarized": (0.01554535, -0.32500, 7.06042, 18.0578),
}

