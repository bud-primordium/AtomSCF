密度泛函理论
============

密度泛函理论 (DFT) 是处理多电子体系的强大方法，通过将问题重新表述为电子密度的泛函来避免多体波函数的复杂性。

Hohenberg-Kohn 定理
-------------------

定理 1：密度唯一性
~~~~~~~~~~~~~~~~~~

基态电子密度 $n(\\mathbf{r})$ 唯一确定外势 $v_{\\text{ext}}(\\mathbf{r})$（相差常数）。

**推论**：所有基态性质都是密度的泛函。

定理 2：变分原理
~~~~~~~~~~~~~~~~

基态能量泛函：

.. math::

   E[n] = T[n] + V_{\\text{ext}}[n] + V_{ee}[n]

在归一化约束 $\\int n(\\mathbf{r}) d^3\\mathbf{r} = N$ 下，真实基态密度使 $E[n]$ 最小。

Kohn-Sham 方法
--------------

核心思想
~~~~~~~~

引入辅助的 **非相互作用参考系统**，其基态密度与真实系统相同。

Kohn-Sham 方程
~~~~~~~~~~~~~~

.. math::

   \\left[ -\\frac{1}{2}\\nabla^2 + v_s(\\mathbf{r}) \\right] \\psi_i = \\varepsilon_i \\psi_i

有效势：

.. math::

   v_s(\\mathbf{r}) = v_{\\text{ext}}(\\mathbf{r}) + v_H(\\mathbf{r}) + v_{xc}(\\mathbf{r})

- $v_H$: Hartree 势（同 HF）
- $v_{xc}$: 交换-关联势（DFT 核心）

交换-关联泛函
~~~~~~~~~~~~~

.. math::

   v_{xc}(\\mathbf{r}) = \\frac{\\delta E_{xc}[n]}{\\delta n(\\mathbf{r})}

$E_{xc}[n]$ 包含所有量子多体效应。

局域密度近似 (LDA)
------------------

基本假设
~~~~~~~~

局域密度近似：系统在点 $\\mathbf{r}$ 处的交换-关联能密度等于均匀电子气在相同密度下的值：

.. math::

   E_{xc}^{\\text{LDA}}[n] = \\int n(\\mathbf{r}) \\varepsilon_{xc}(n(\\mathbf{r})) d^3\\mathbf{r}

其中 $\\varepsilon_{xc}(n)$ 为均匀电子气的单位体积交换-关联能。

交换部分
~~~~~~~~

Dirac 交换（精确解）：

.. math::

   \\varepsilon_x^{\\text{Dirac}}(n) = -C_x n^{1/3}

其中 $C_x = \\frac{3}{4}\\left(\\frac{3}{\\pi}\\right)^{1/3} \\approx 0.7386$。

关联部分
~~~~~~~~

**需要数值计算或参数化拟合**。常用泛函：

- **PZ81** (Perdew-Zunger 1981)
- **VWN** (Vosko-Wilk-Nusair 1980)

局域自旋密度近似 (LSDA)
-----------------------

自旋极化
~~~~~~~~

分别处理自旋 $\\uparrow$ 和 $\\downarrow$ 密度：

.. math::

   n(\\mathbf{r}) = n_{\\uparrow}(\\mathbf{r}) + n_{\\downarrow}(\\mathbf{r})

自旋极化度：

.. math::

   \\zeta(\\mathbf{r}) = \\frac{n_{\\uparrow} - n_{\\downarrow}}{n_{\\uparrow} + n_{\\downarrow}}

LSDA 能量
~~~~~~~~~

.. math::

   E_{xc}^{\\text{LSDA}}[n_{\\uparrow}, n_{\\downarrow}] =
   \\int n(\\mathbf{r}) \\varepsilon_{xc}(n_{\\uparrow}, n_{\\downarrow}) d^3\\mathbf{r}

分别求解自旋上/下的 Kohn-Sham 方程。

Perdew-Zunger 关联 (PZ81)
--------------------------

参数化形式
~~~~~~~~~~

基于 Ceperley-Alder 量子蒙特卡罗数据拟合：

**高密度区** ($r_s < 1$)：

.. math::

   \\varepsilon_c^{\\text{PZ}}(r_s, \\zeta) = A \\ln r_s + B + C r_s \\ln r_s + D r_s

**低密度区** ($r_s \\geq 1$)：

.. math::

   \\varepsilon_c^{\\text{PZ}}(r_s, \\zeta) = \\frac{\\gamma}{1 + \\beta_1 \\sqrt{r_s} + \\beta_2 r_s}

其中 $r_s = (3/(4\\pi n))^{1/3}$ 为 Wigner-Seitz 半径。

自旋内插
~~~~~~~~

.. math::

   \\varepsilon_c(n, \\zeta) = \\varepsilon_c(n, 0) + \\alpha_c(r_s) \\frac{f(\\zeta)}{f''(0)} (1 - \\zeta^4)

插值函数：

.. math::

   f(\\zeta) = \\frac{(1+\\zeta)^{4/3} + (1-\\zeta)^{4/3} - 2}{2^{4/3} - 2}

Vosko-Wilk-Nusair 关联 (VWN)
-----------------------------

RPA 拟合
~~~~~~~~

基于随机相位近似 (RPA) 的解析拟合：

.. math::

   \\varepsilon_c^{\\text{VWN}}(r_s) = \\frac{A}{2} \\left\\{
   \\ln\\frac{x^2}{X(x)} + \\frac{2b}{Q} \\tan^{-1}\\frac{Q}{2x+b}
   - \\frac{bx_0}{X(x_0)} \\left[ \\ln\\frac{(x-x_0)^2}{X(x)} + \\frac{2(b+2x_0)}{Q} \\tan^{-1}\\frac{Q}{2x+b} \\right]
   \\right\\}

其中：

.. math::

   x = \\sqrt{r_s}, \\quad X(x) = x^2 + bx + c, \\quad Q = \\sqrt{4c - b^2}

参数值
~~~~~~

- 顺磁态：$A = 0.0621814$, $b = 3.72744$, $c = 12.9352$, $x_0 = -0.10498$
- 铁磁态：$A = 0.0310907$, $b = 7.06042$, $c = 18.0578$, $x_0 = -0.32500$

自旋插值：与 PZ81 类似。

原子中的实现
------------

径向 Kohn-Sham 方程
~~~~~~~~~~~~~~~~~~~

.. math::

   \\left[ -\\frac{1}{2}\\frac{d^2}{dr^2} + \\frac{\\ell(\\ell+1)}{2r^2} + v_s^{\\sigma}(r) \\right] u_{n\\ell\\sigma} = \\varepsilon_{n\\ell\\sigma} u_{n\\ell\\sigma}

有效势（自旋分辨）：

.. math::

   v_s^{\\sigma}(r) = -\\frac{Z}{r} + v_H(r) + v_{xc}^{\\sigma}[n_{\\uparrow}, n_{\\downarrow}](r)

交换-关联势
~~~~~~~~~~~

.. math::

   v_{xc}^{\\sigma}(r) = \\frac{\\partial (n \\varepsilon_{xc})}{\\partial n_{\\sigma}}
   = \\varepsilon_{xc} + n \\frac{\\partial \\varepsilon_{xc}}{\\partial n_{\\sigma}}

需要对泛函求变分导数。

能量计算
~~~~~~~~

总能量：

.. math::

   E_{\\text{LSDA}} = T_s + V_{\\text{ext}} + E_H + E_{xc}

其中：
- $T_s = \\sum_{i\\sigma} n_i \\int u_i^{\\sigma} \\left(-\\frac{1}{2}\\frac{d^2}{dr^2} + \\frac{\\ell(\\ell+1)}{2r^2}\\right) u_i^{\\sigma} dr$
- $E_{xc} = \\int n(r) \\varepsilon_{xc}(n_{\\uparrow}, n_{\\downarrow}) 4\\pi r^2 dr$

DFT vs HF 对比
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - 方面
     - HF
     - DFT (LSDA)
   * - 基本变量
     - 多电子波函数 $\\Psi$
     - 电子密度 $n(\\mathbf{r})$
   * - 交换
     - 精确（非局域）
     - 近似（局域）
   * - 关联
     - 无
     - 包含（近似）
   * - 计算复杂度
     - $O(N^4)$
     - $O(N^3)$
   * - 精度（能量）
     - 闭壳层好，开壳层差
     - 一般好
   * - 精度（带隙）
     - 高估
     - 低估
   * - 多重态
     - RHF/ROHF 正确
     - LSDA 近似

应用示例
--------

碳原子 (1s² 2s² 2p²)
~~~~~~~~~~~~~~~~~~~~~

LSDA 自旋极化配置：
- $n_{\\uparrow}$: 1s¹ 2s¹ 2p²（4 个 $\\uparrow$ 电子）
- $n_{\\downarrow}$: 1s¹ 2s¹（2 个 $\\downarrow$ 电子）

总密度：$n = n_{\\uparrow} + n_{\\downarrow}$

自旋极化：$\\zeta = (4-2)/6 = 1/3$

收敛技巧
--------

1. **密度混合**：线性或 DIIS
2. **初始猜测**：原子密度叠加
3. **网格选择**：近核密集，远程稀疏
4. **占据数涂抹**：金属体系（Fermi-Dirac）

局限性
------

LDA/LSDA 已知问题：
- **自相互作用误差**：电子与自身 Hartree 势耦合
- **带隙低估**：半导体/绝缘体
- **弱相互作用**：范德华力缺失
- **强关联**：过渡金属 d 电子

解决方案：
- GGA (广义梯度近似)
- Meta-GGA (含动能密度)
- Hybrid 泛函 (混合 HF 交换)
- DFT+U (强关联修正)

参考文献
--------

1. Hohenberg, P. & Kohn, W. *Phys. Rev.* **136**, B864 (1964)
2. Kohn, W. & Sham, L. J. *Phys. Rev.* **140**, A1133 (1965)
3. Perdew, J. P. & Zunger, A. *Phys. Rev. B* **23**, 5048 (1981)
4. Vosko, S. H., Wilk, L. & Nusair, M. *Can. J. Phys.* **58**, 1200 (1980)
5. Parr, R. G. & Yang, W. *Density-Functional Theory of Atoms and Molecules* (1989)
