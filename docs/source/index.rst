AtomSCF 文档
==============

教学用原子自旋极化 HF 与 LSDA-DFT（径向一维）最小实现。

.. toctree::
   :maxdepth: 2
   :caption: 目录

   introduction
   algorithm/index
   api/index
   examples/index

简介
----

AtomSCF 是一个用于原子自洽场计算的 Python 库，实现了：

- **Hartree-Fock (HF)** 方法

  - Restricted HF (RHF): 闭壳层体系
  - Unrestricted HF (UHF): 自旋极化开壳层体系

- **密度泛函理论 (DFT)** 方法

  - LSDA (Local Spin Density Approximation)
  - 交换泛函: Dirac
  - 关联泛函: Perdew-Zunger 1981 (PZ81), Vosko-Wilk-Nusair (VWN)

- **数值方法**

  - 径向 Schrödinger 方程求解器（多种网格类型）
  - 有限差分离散化
  - 变量变换方法（指数网格）

快速开始
--------

安装
~~~~

.. code-block:: bash

   # 克隆仓库
   git clone <repository-url>
   cd AtomSCF

   # 安装依赖
   pip install -e .[dev,docs]

示例：氢原子 HF 计算
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from atomscf.grid import radial_grid_linear
   from atomscf.scf_hf import run_hf_minimal

   # 创建径向网格
   r, w = radial_grid_linear(n=800, rmin=1e-6, rmax=50.0)

   # 运行 HF 计算
   result = run_hf_minimal(Z=1, r=r, w=w)

   # 输出结果
   print(f"总能量: {result.E_total:.6f} Ha")
   print(f"1s 轨道能: {result.epsilon_1s:.6f} Ha")

支持的原子
----------

- **闭壳层**: He, Be, Ne, Mg, Ar, ...
- **开壳层**: H, Li, B, C, N, O, F, ...

文档结构
--------

- **算法原理**: 详细推导 HF 和 DFT 的数学公式和数值实现
- **API 参考**: 完整的函数和类接口文档
- **示例**: 常见原子的计算示例和结果对比

索引与搜索
----------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
