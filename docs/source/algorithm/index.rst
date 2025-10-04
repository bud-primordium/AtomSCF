算法原理
========

本节详细推导 AtomSCF 中实现的各种方法的数学公式和数值离散化方案。

.. toctree::
   :maxdepth: 2

   hartree_fock
   density_functional
   numerical_methods

概述
----

原子电子结构计算的核心是求解多电子 Schrödinger 方程。由于球对称性，问题可简化为径向一维：

.. math::

   \\left[-\\frac{1}{2}\\frac{d^2}{dr^2} + \\frac{\\ell(\\ell+1)}{2r^2} + v_{\\text{eff}}(r)\\right] u_{n\\ell}(r) = \\varepsilon_{n\\ell} u_{n\\ell}(r)

其中 $u_{n\\ell}(r) = r R_{n\\ell}(r)$ 为径向波函数，$v_{\\text{eff}}(r)$ 为有效势能。

自洽场方法
----------

HF 和 DFT 都采用自洽场 (SCF) 迭代框架：

1. **初始猜测**: 使用类氢轨道或 Slater 屏蔽估计
2. **构造有效势**: 计算 Hartree 势、交换势/泛函
3. **求解方程**: 对角化 Fock/Kohn-Sham 矩阵
4. **更新密度**: 用新轨道重建电子密度
5. **收敛检查**: 判断密度变化是否低于阈值
6. **密度混合**: $\\rho^{(n+1)} = \\alpha \\rho_{\\text{new}} + (1-\\alpha) \\rho^{(n)}$

方法对比
--------

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - 方法
     - 有效势 $v_{\\text{eff}}$
     - 优点
     - 缺点
   * - RHF
     - $v_H + \\hat{K}_{\\text{RHF}}$
     - 精确交换，闭壳层准确
     - 开壳层强制配对，无关联
   * - UHF
     - $v_H^{\\sigma} + \\hat{K}_{\\sigma}$
     - 自旋极化，开壳层改进
     - 自旋污染，无关联
   * - LSDA
     - $v_H + v_{xc}[n_{\\uparrow}, n_{\\downarrow}]$
     - 包含关联，速度快
     - 局域近似，低估带隙

符号约定
--------

- 原子单位: $\\hbar = m_e = e = 4\\pi\\epsilon_0 = 1$
- 能量单位: Hartree (Ha), 1 Ha = 27.211 eV
- 长度单位: Bohr (a₀), 1 a₀ = 0.529 Å
- 角动量量子数: $\\ell = 0$ (s), 1 (p), 2 (d), ...
- 自旋标记: $\\sigma = \\uparrow$ (up), $\\downarrow$ (down)
