数值方法
========

本节介绍 AtomSCF 中实现的数值离散化方法和求解技术。

径向网格生成
------------

线性网格
~~~~~~~~

均匀间距：

.. math::

   r_i = r_{\mathrm{min}} + i \cdot h, \quad i = 0, 1, \dots, N-1

其中 :math:`h = (r_{\mathrm{max}} - r_{\mathrm{min}}) / (N-1)`。

**优点**：
- 简单易实现
- 近核区域精细

**缺点**：
- 远程需要大量网格点
- 效率低

对数网格
~~~~~~~~

对数等间距：

.. math::

   r_i = r_0 \exp(i \delta), \quad i = 0, 1, \dots, N-1

或等价地 :math:`\ln r_i = \ln r_0 + i \delta`。

**优点**：
- 自适应：近核密集，远程稀疏
- 波函数衰减区域覆盖好

**缺点**：
- 不包含 :math:`r=0` 点
- 变换后的微分算子复杂

指数变换网格
~~~~~~~~~~~~

基于 [ExpGridTransform]_ 的方法：

.. math::

   r_j = R_p (e^{j\delta} - 1), \quad j = 0, 1, \dots, N-1

变量变换推导
^^^^^^^^^^^^

**原始径向 Schrödinger 方程**：

.. math::

   -\frac{1}{2}\frac{d^2 u}{dr^2} + \left[ V(r) + \frac{\ell(\ell+1)}{2r^2} \right] u = \varepsilon u

引入变量代换 :math:`u(r) = v(r) \cdot w(r)`，其中 :math:`w(r) = \exp(-r/(2R_p))`。

**一阶导数**：

.. math::

   \frac{du}{dr} = \frac{dv}{dr} w + v \frac{dw}{dr} = w \left( \frac{dv}{dr} - \frac{v}{2R_p} \right)

**二阶导数**：

.. math::

   \frac{d^2u}{dr^2} = w \left( \frac{d^2v}{dr^2} - \frac{1}{R_p}\frac{dv}{dr} + \frac{v}{4R_p^2} \right)

代入原方程并消去 :math:`w`：

.. math::

   -\frac{1}{2}\frac{d^2v}{dr^2} + \frac{1}{2R_p}\frac{dv}{dr} + \left[ V(r) + \frac{\ell(\ell+1)}{2r^2} - \frac{1}{8R_p^2} \right] v = \varepsilon v

**在指数网格上离散化**：

取 :math:`R_p = 1/\delta`，则：

.. math::

   -\frac{1}{2}\frac{d^2v}{dr^2} + \left[ V(r) + \frac{\ell(\ell+1)}{2r^2} - \frac{\delta^2}{8} \right] v = \varepsilon v

关键特性：**一阶导数项消失**，Hamiltonian 矩阵对称。

**数值实现**：

在网格点 :math:`r_j` 上：

.. math::

   u_j = v_j \exp\left(-\frac{r_j}{2R_p}\right) = v_j \exp\left(-\frac{j\delta}{2}\right)

反解：

.. math::

   v_j = u_j \exp\left(\frac{j\delta}{2}\right)

**优点**：
- 精度提升 ~7x（相比线性网格）
- 速度提升 ~3x
- 包含 :math:`r=0` 点（:math:`j=0 \Rightarrow r=0`）
- 对称矩阵可用 :code:`scipy.linalg.eigh()` 求解

**缺点**：
- 需要额外参数 :math:`(\delta, R_p)`
- 变换后有效势包含常数项 :math:`-\delta^2/8`

**参数选择**：

- :math:`\delta \approx 0.01 \sim 0.05`（控制网格密度）
- :math:`R_p \approx Z/4`（Z 为原子序数，优化波函数衰减匹配）

有限差分方法
------------

二阶中心差分 (FD2)
~~~~~~~~~~~~~~~~~~

二阶导数近似（非均匀网格）：

.. math::

   \frac{d^2 u}{dr^2}\bigg|_{r_i} \approx
   \frac{2}{\Delta r_{i-}(\Delta r_{i-} + \Delta r_{i+})} u_{i-1}
   - \frac{2}{\Delta r_{i-} \Delta r_{i+}} u_i
   + \frac{2}{\Delta r_{i+}(\Delta r_{i-} + \Delta r_{i+})} u_{i+1}

其中：
- :math:`\Delta r_{i-} = r_i - r_{i-1}`
- :math:`\Delta r_{i+} = r_{i+1} - r_i`

**精度**：:math:`O(h^2)`（均匀网格）

五阶中心差分 (FD5)
~~~~~~~~~~~~~~~~~~

等间距线性网格专用：

.. math::

   \frac{d^2 u}{dr^2}\bigg|_{r_i} \approx
   \frac{-u_{i+2} + 16u_{i+1} - 30u_i + 16u_{i-1} - u_{i-2}}{12h^2}

**精度**：:math:`O(h^4)`

**要求**：必须为等间距网格

Numerov 方法
~~~~~~~~~~~~

特别适用于形如 :math:`u'' + k^2(r) u = 0` 的方程（对数网格）。

递推公式：

.. math::

   u_{n+1} = \frac{(2 - \frac{10}{12}h^2 k_n^2) u_n - (1 + \frac{1}{12}h^2 k_{n-1}^2) u_{n-1}}{1 + \frac{1}{12}h^2 k_{n+1}^2}

其中 :math:`k^2(r) = 2[v(r) - E] - \ell(\ell+1)/r^2`。

**精度**：:math:`O(h^6)`（局域截断误差）

**应用**：边界值问题（打靶法 + 二分法）

Hamiltonian 矩阵构造
--------------------

标准 FD2 方法
~~~~~~~~~~~~~

三对角矩阵（内部点）：

.. math::

   H_{ij} = \begin{cases}
   -\frac{1}{2}\frac{2}{\Delta r_{i-} \Delta r_{i+}} + v(r_i) + \frac{\ell(\ell+1)}{2r_i^2}, & i = j \\
   -\frac{1}{2}\frac{2}{\Delta r_{i-}(\Delta r_{i-} + \Delta r_{i+})}, & j = i-1 \\
   -\frac{1}{2}\frac{2}{\Delta r_{i+}(\Delta r_{i-} + \Delta r_{i+})}, & j = i+1 \\
   0, & |i-j| > 1
   \end{cases}

边界条件：:math:`u_0 = u_{N-1} = 0`（Dirichlet）

变换 Hamiltonian
~~~~~~~~~~~~~~~~

指数变换网格的 :math:`v` 空间 Hamiltonian：

.. math::

   H_{jj'} = \int v_j(r) \left[ -\frac{1}{2}\frac{d^2}{dr^2} + V(r) \right] v_{j'}(r) w(r) dr

其中基函数：:math:`v_j(r) = \delta_{jj'} / \sqrt{w(j\delta)}`

权重：:math:`w(r) = \exp(-r/R_p)`

**优势**：消除一阶导数项，提高精度

本征值问题求解
--------------

标准对角化
~~~~~~~~~~

对称矩阵：

.. math::

   H \mathbf{v} = \varepsilon \mathbf{v}

使用 `numpy.linalg.eigh()` 或 `scipy.linalg.eigh()`

广义本征值问题
~~~~~~~~~~~~~~

变换网格需求解：

.. math::

   H \mathbf{v} = \varepsilon B \mathbf{v}

其中 :math:`B` 为重叠矩阵（非单位）

使用 `scipy.linalg.eigh(H, B)`

自洽场迭代
----------

SCF 循环框架
~~~~~~~~~~~~

1. **初始猜测**：类氢轨道或原子密度叠加
2. **迭代**：

   a. 构造有效势：:math:`v_{\mathrm{eff}} = v_{\mathrm{ext}} + v_H + v_{xc}`（DFT）或 :math:`v_{\mathrm{ext}} + v_H + \hat{K}`（HF）
   b. 求解 KS/Fock 方程
   c. 更新密度：:math:`n^{(k+1)} = \sum_i n_i |\psi_i^{(k+1)}|^2`
   d. 密度混合：:math:`n_{\mathrm{mix}} = \alpha n^{(k+1)} + (1-\alpha) n^{(k)}`
   e. 检查收敛：:math:`\|n^{(k+1)} - n^{(k)}\| < \epsilon`

3. **后处理**：计算总能量和其他性质

密度混合策略
~~~~~~~~~~~~

**线性混合**：

.. math::

   n_{\mathrm{mix}} = \alpha n_{\mathrm{new}} + (1-\alpha) n_{\mathrm{old}}

典型值：:math:`\alpha \in [0.1, 0.7]`

**DIIS (Direct Inversion in Iterative Subspace)**：

使用历史密度的线性组合，最小化残差：

.. math::

   R_i = n_{\mathrm{out},i} - n_{\mathrm{in},i}

求解最优系数：

.. math::

   \sum_j c_j \langle R_i | R_j \rangle + \lambda = 0, \quad \sum_j c_j = 1

收敛判据
~~~~~~~~

常用标准：

1. **密度变化**：:math:`\|\Delta n\|_2 < 10^{-6}`
2. **能量变化**：:math:`|\Delta E| < 10^{-8}` Ha
3. **轨道变化**：:math:`\sum_i \|\psi_i^{(k+1)} - \psi_i^{(k)}\|^2 < 10^{-6}`

Hartree 势计算
--------------

泊松方程求解
~~~~~~~~~~~~

.. math::

   \nabla^2 v_H = -4\pi n

径向形式（球对称）：

.. math::

   \frac{1}{r^2}\frac{d}{dr}\left(r^2 \frac{dv_H}{dr}\right) = -4\pi n

解析解（分段积分）：

.. math::

   v_H(r) = \int_0^r \frac{n(r')}{r} r'^2 dr' + \int_r^\infty n(r') r' dr'

数值实现（梯形积分）：

.. math::

   v_H(r_i) = \frac{1}{r_i} \sum_{j=0}^{i} n_j r_j^2 w_j + \sum_{j=i}^{N-1} n_j r_j w_j

其中 :math:`w_j` 为积分权重。

交换积分计算
------------

Slater 径向积分
~~~~~~~~~~~~~~~

定义：

.. math::

   R^k(r) = Y^k(r) + Z^k(r)

其中：

.. math::

   Y^k(r) &= \frac{1}{r} \int_0^r u_i(r') u_j(r') r'^k dr' \\
   Z^k(r) &= \int_r^\infty u_i(r') u_j(r') r'^{k-1} dr'

两段累积算法：

.. code-block:: python

   # 向外累积 Y^k
   Y[0] = 0
   for i in range(1, N):
       Y[i] = Y[i-1] + u_i[i] * u_j[i] * r[i]**k * w[i]
   Y /= r  # 除以 r

   # 向内累积 Z^k
   Z[N-1] = 0
   for i in range(N-2, -1, -1):
       Z[i] = Z[i+1] + u_i[i] * u_j[i] * r[i]**(k-1) * w[i]

数值积分
--------

梯形公式
~~~~~~~~

非均匀网格：

.. math::

   \int_a^b f(x) dx \approx \sum_{i=0}^{N-1} f(x_i) w_i

权重：

.. math::

   w_i = \begin{cases}
   (x_1 - x_0)/2, & i = 0 \
   (x_{i+1} - x_{i-1})/2, & 0 < i < N-1 \
   (x_{N-1} - x_{N-2})/2, & i = N-1
   \end{cases}

Simpson 公式
~~~~~~~~~~~~

等间距网格，:math:`N` 为奇数：

.. math::

   \int_a^b f(x) dx \approx \frac{h}{3} \left[ f(x_0) + 4\sum_{\mathrm{odd}} f(x_i) + 2\sum_{\mathrm{even}} f(x_i) + f(x_N) \right]

**精度**：:math:`O(h^4)`

边界条件处理
------------

Dirichlet 边界
~~~~~~~~~~~~~~

固定值：:math:`u(r_0) = u(r_N) = 0`

实现：
- 不求解边界点
- 内部矩阵维度 :math:`(N-2) \times (N-2)`

Neumann 边界
~~~~~~~~~~~~

固定导数：:math:`u'(r_0) = 0`（s 轨道在原点）

实现：镜像点法或单侧差分

性能优化
--------

缓存 Slater 积分
~~~~~~~~~~~~~~~~

Slater 积分仅依赖占据态对 :math:`(i, j)` 和多极指标 :math:`k`，可预计算并缓存：

.. code-block:: python

   cache = {}
   key = (i, j, k)
   if key not in cache:
       cache[key] = compute_slater_integral(u_i, u_j, k)
   return cache[key]

并行化
~~~~~~

多角动量通道可并行求解：

.. code-block:: python

   from multiprocessing import Pool

   with Pool(ncpu) as pool:
       results = pool.map(solve_channel, l_values)

数值稳定性检查
--------------

归一化
~~~~~~

每次迭代检查：

.. math::

   \int u_{n\ell}^2(r) dr = 1

电子数守恒
~~~~~~~~~~

.. math::

   N = \sum_{n\ell m \sigma} n_{n\ell m \sigma} = \int n(r) 4\pi r^2 dr

Virial 定理
~~~~~~~~~~~

HF 基态满足：

.. math::

   -\frac{T}{E_{\mathrm{total}}} = 1

DFT:

.. math::

   -\frac{T + E_{xc} - \int n v_{xc} dr}{E_{\mathrm{total}}} = 1

参考文献
--------

.. [ExpGridTransform] Computational Physics Fall 2024, Assignment 7, Problem 2
   https://github.com/bud-primordium/Computational-Physics-Fall-2024/

1. Press, W. H. et al. *Numerical Recipes* (2007)
2. Johnson, D. D. *Phys. Rev. B* **38**, 12807 (1988) [DIIS]
3. Lehtola, S. et al. *J. Chem. Theory Comput.* **15**, 1593 (2019) [SCF 算法综述]
