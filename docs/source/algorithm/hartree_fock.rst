Hartree-Fock 方法
=================

Hartree-Fock (HF) 方法是处理多电子体系的基础量子化学方法，通过变分原理将多电子问题化为有效单电子问题。

理论基础
--------

多电子 Hamiltonian
~~~~~~~~~~~~~~~~~~

原子中 $N$ 个电子的非相对论性 Hamiltonian：

.. math::

   \\hat{H} = \\sum_{i=1}^N \\left[ -\\frac{1}{2}\\nabla_i^2 - \\frac{Z}{r_i} \\right]
   + \\sum_{i<j} \\frac{1}{|\\mathbf{r}_i - \\mathbf{r}_j|}

- 第一项：电子动能
- 第二项：核-电子吸引（原子单位：$Z/r$）
- 第三项：电子-电子排斥

波函数 Ansatz
~~~~~~~~~~~~~

HF 方法采用 **Slater 行列式** 作为多电子波函数的近似：

.. math::

   \\Psi(\\mathbf{r}_1, \\dots, \\mathbf{r}_N) = \\frac{1}{\\sqrt{N!}}
   \\begin{vmatrix}
   \\psi_1(\\mathbf{r}_1) & \\psi_1(\\mathbf{r}_2) & \\cdots & \\psi_1(\\mathbf{r}_N) \\\\
   \\psi_2(\\mathbf{r}_1) & \\psi_2(\\mathbf{r}_2) & \\cdots & \\psi_2(\\mathbf{r}_N) \\\\
   \\vdots & \\vdots & \\ddots & \\vdots \\\\
   \\psi_N(\\mathbf{r}_1) & \\psi_N(\\mathbf{r}_2) & \\cdots & \\psi_N(\\mathbf{r}_N)
   \\end{vmatrix}

其中 $\\psi_i$ 为自旋轨道（spin-orbital），包含空间和自旋部分。

变分原理
~~~~~~~~

最小化能量泛函：

.. math::

   E[\\{\\psi_i\\}] = \\langle \\Psi | \\hat{H} | \\Psi \\rangle

在正交归一化约束 $\\langle \\psi_i | \\psi_j \\rangle = \\delta_{ij}$ 下，导出 **Hartree-Fock 方程**。

Hartree-Fock 方程
-----------------

Fock 算符
~~~~~~~~~

单电子 Fock 算符：

.. math::

   \\hat{f} = -\\frac{1}{2}\\nabla^2 - \\frac{Z}{r} + v_H(\\mathbf{r}) + \\hat{K}

- $v_H$: Hartree 势（经典电子排斥）
- $\\hat{K}$: 交换算符（量子效应）

自洽场方程
~~~~~~~~~~

.. math::

   \\hat{f} \\psi_i = \\varepsilon_i \\psi_i

$\\varepsilon_i$ 为轨道能，$\\psi_i$ 为自旋轨道。

Hartree 势
~~~~~~~~~~

.. math::

   v_H(\\mathbf{r}) = \\int \\frac{\\rho(\\mathbf{r}')}{|\\mathbf{r} - \\mathbf{r}'|} d^3\\mathbf{r}'

其中电子密度：

.. math::

   \\rho(\\mathbf{r}) = \\sum_{i=1}^N |\\psi_i(\\mathbf{r})|^2

交换算符
~~~~~~~~

非局域算符，作用在轨道 $\\psi_j$ 上：

.. math::

   \\hat{K} \\psi_j(\\mathbf{r}) = \\left[ \\sum_{i=1}^N \\int \\frac{\\psi_i^*(\\mathbf{r}') \\psi_j(\\mathbf{r}')}{|\\mathbf{r} - \\mathbf{r}'|} d^3\\mathbf{r}' \\right] \\psi_i(\\mathbf{r})

**注意**：仅同自旋轨道间有交换相互作用。

球对称原子的简化
----------------

径向方程
~~~~~~~~

利用球对称性，分离角向和径向部分：

.. math::

   \\psi_{n\\ell m}(\\mathbf{r}) = \\frac{u_{n\\ell}(r)}{r} Y_{\\ell}^m(\\theta, \\phi)

径向波函数 $u_{n\\ell}(r)$ 满足：

.. math::

   \\left[ -\\frac{1}{2}\\frac{d^2}{dr^2} + \\frac{\\ell(\\ell+1)}{2r^2} + v_{\\text{eff}}(r) \\right] u_{n\\ell} = \\varepsilon_{n\\ell} u_{n\\ell}

边界条件：$u(0) = u(\\infty) = 0$。

有效势
~~~~~~

.. math::

   v_{\\text{eff}}(r) = -\\frac{Z}{r} + v_H(r) + v_x(r)

- $v_H(r)$: 径向 Hartree 势
- $v_x(r)$: 交换势（需特殊处理）

径向 Hartree 势
~~~~~~~~~~~~~~~

.. math::

   v_H(r) = \\int_0^\\infty \\frac{n(r')}{\\max(r, r')} r'^2 dr'

其中径向密度：

.. math::

   n(r) = \\sum_{n\\ell} f_{n\\ell} u_{n\\ell}^2(r)

$f_{n\\ell}$ 为占据数（考虑自旋和磁量子数简并）。

自旋限制类型
------------

RHF (Restricted HF)
~~~~~~~~~~~~~~~~~~~

**适用**：闭壳层体系（所有电子配对）

**特点**：
- 自旋 $\\alpha$ 和 $\\beta$ 电子共享同一空间轨道
- 占据数 $f_{n\\ell} = 2(2\\ell + 1)$（满壳层）
- 交换算符包含所有占据轨道

**优点**：
- 保持自旋对称性
- 计算量小

**缺点**：
- 无法处理开壳层（如 Li: 1s² 2s¹）
- 强制电子配对，物理不合理

UHF (Unrestricted HF)
~~~~~~~~~~~~~~~~~~~~~

**适用**：开壳层体系（未配对电子）

**特点**：
- 自旋 $\\alpha$ 和 $\\beta$ 有独立空间轨道
- 占据数按自旋分离：$f_{n\\ell\\sigma}$
- 交换仅在同自旋间

**优点**：
- 自旋极化自由度
- 变分能量更低

**缺点**：
- 自旋污染（$\\langle \\hat{S}^2 \\rangle \\neq S(S+1)$）
- 破坏自旋对称性

ROHF (Restricted Open-shell HF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**适用**：开壳层，但保持自旋对称

**特点**：
- 闭壳层和开壳层分别处理
- 三种 Fock 算符（core, open, virtual）

**优点**：
- 保持自旋对称性
- 正确描述多重态（如碳 ³P）

**缺点**：
- 实现复杂
- 收敛困难

交换积分计算
------------

Slater 积分
~~~~~~~~~~~

径向交换积分展开为多极矩：

.. math::

   \\int \\frac{u_{n\\ell}(r) u_{n'\\ell'}(r')}{|\\mathbf{r} - \\mathbf{r}'|} d^3\\mathbf{r}'
   = \\sum_{k=0}^{\\infty} a_k(\\ell, \\ell') R^k(r)

Slater 径向积分：

.. math::

   R^k(r) = \\frac{1}{r} \\int_0^r u_{n\\ell}(r') u_{n'\\ell'}(r') r'^k dr'
   + \\int_r^\\infty u_{n\\ell}(r') u_{n'\\ell'}(r') r'^{k-1} dr'

角动量耦合系数
~~~~~~~~~~~~~~

.. math::

   a_k(\\ell, \\ell') = (2\\ell + 1)(2\\ell' + 1) \\sum_m \\sum_{m'}
   \\left[ C_{\\ell m \\ell' m'}^{k 0} \\right]^2

其中 $C$ 为 Clebsch-Gordan 系数。

**选择规则**：
- $|\\ell - \\ell'| \\leq k \\leq \\ell + \\ell'$
- $k + \\ell + \\ell'$ 为偶数

能量表达式
----------

总能量
~~~~~~

.. math::

   E_{\\text{HF}} = \\sum_i n_i \\varepsilon_i - \\frac{1}{2}(E_H + E_x)

其中：
- $\\varepsilon_i$: 轨道能
- $E_H$: Hartree 能（双计数校正）
- $E_x$: 交换能（双计数校正）

能量分解
~~~~~~~~

.. math::

   E_{\\text{HF}} = T + V_{\\text{ext}} + E_H + E_x

- $T$: 动能
- $V_{\\text{ext}}$: 核吸引能
- $E_H$: Hartree 能
- $E_x$: 交换能

数值实现要点
------------

1. **初始猜测**：类氢轨道或 Slater 屏蔽
2. **求解 Fock 方程**：有限差分离散化
3. **Hartree 势计算**：泊松方程求解
4. **交换积分**：Slater 积分两段累积法
5. **密度混合**：DIIS 或简单线性混合
6. **收敛判据**：密度变化 $< 10^{-6}$

参考文献
--------

1. Roothaan, C. C. J. *Rev. Mod. Phys.* **23**, 69 (1951)
2. Clementi, E. & Roetti, C. *Atomic Data and Nuclear Data Tables* **14**, 177 (1974)
3. Szabo, A. & Ostlund, N. S. *Modern Quantum Chemistry* (1996)
