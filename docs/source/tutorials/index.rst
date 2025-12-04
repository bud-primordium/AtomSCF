教程
====

本教程系列从基础概念出发，逐步介绍原子自洽场计算的核心方法。

.. toctree::
   :maxdepth: 1
   :caption: 教程列表

   01_radial_schrodinger
   02_hartree_fock
   03_dft_lda

教程概览
--------

**教程 01：径向薛定谔方程基础**

- 球谐分解与一维径向方程
- 有效势的物理含义
- 径向网格类型与选择
- 有限差分离散化
- 氢原子解析解验证

**教程 02：自洽场方法**

- 多电子问题的困难
- Hartree 平均场近似
- 交换相互作用与 HF
- SCF 迭代原理
- 周期表原子计算

**教程 03：密度泛函理论**

- Hohenberg-Kohn 定理
- Kohn-Sham 方程
- 局域密度近似 (LDA)
- 自旋极化 LSDA
- HF vs DFT 对比

运行方式
--------

**方式一：Google Colab（推荐）**

点击教程顶部的 "在 Colab 中打开" 徽章，无需本地安装。

**方式二：本地运行**

.. code-block:: bash

   # 克隆仓库
   git clone https://github.com/bud-primordium/AtomSCF.git
   cd AtomSCF

   # 安装依赖
   pip install -e .[dev]

   # 启动 Jupyter
   jupyter notebook docs/source/tutorials/

先备知识
--------

- Python 编程基础
- 量子力学入门（薛定谔方程）
- 线性代数基础（矩阵本征值）

建议学习顺序
------------

1. 教程 01 → 02 → 03（循序渐进）
2. 可根据兴趣跳过部分章节
3. 配合 API 文档查阅函数细节
