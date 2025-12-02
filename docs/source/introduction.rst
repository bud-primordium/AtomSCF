项目介绍
========

AtomSCF 是一个为电子结构理论教学设计的原子自洽场计算工具，专注于径向一维问题的最小化实现。

设计理念
--------

本项目遵循以下设计原则：

1. **教学优先**: 代码清晰易读，注重物理直觉而非性能优化
2. **渐进式实现**: 从最简单的氢原子 HF 到多电子 LSDA，逐步增加复杂度
3. **数值稳定性**: 提供多种网格和求解器选择，适应不同数值场景
4. **文档完整**: 区分算法原理推导与 API 使用文档

适用场景
--------

**推荐用于**:

- 电子结构理论课程教学
- 理解 HF 和 DFT 方法的核心原理
- 测试新的数值方法或泛函
- 小规模原子基准计算

**不推荐用于**:

- 生产环境的高精度计算（建议使用 PySCF, Psi4 等）
- 分子或固体计算（本代码仅支持球对称原子）
- 性能关键场景（未进行深度优化）

核心功能
--------

Hartree-Fock 方法
~~~~~~~~~~~~~~~~~

- **RHF (Restricted HF)**:

  - 适用于闭壳层体系（He, Be, Ne, ...）
  - 自旋轨道共享同一空间波函数
  - 精度：氦原子 -2.86 Ha（实验值 -2.90 Ha）

- **UHF (Unrestricted HF)**:

  - 适用于开壳层体系（Li, C, N, O, ...）
  - 自旋上/下轨道独立优化
  - Li 原子能量比 RHF 降低 ~54 mHa

- **交换积分**:

  - Slater 积分径向计算（:math:`R^k` 两段累积法）
  - 角动量耦合系数（Wigner 3j 符号）
  - 多角动量通道耦合（s-s, s-p, p-p, ...）

密度泛函理论
~~~~~~~~~~~~

- **LSDA (Local Spin Density Approximation)**:

  - 自旋极化密度
  - 支持多种交换-关联泛函组合

- **交换泛函**:

  - Dirac 交换: :math:`E_x = -C_x \int n^{4/3}(r) dr`

- **关联泛函**:

  - **PZ81**: Perdew-Zunger 1981 参数化
  - **VWN**: Vosko-Wilk-Nusair (RPA 拟合)

数值方法
~~~~~~~~

- **径向网格类型**:

  - 线性网格: 适合近核区域精细计算
  - 对数网格: 适合长程衰减行为
  - 指数变换网格: 结合两者优势（精度提升 ~7x）
  - 混合网格: 分段不同间距

- **求解器**:

  - 2 阶有限差分 (FD2)
  - 5 阶有限差分 (FD5, 等距网格)
  - Numerov 方法（对数网格）
  - 变换 Hamiltonian（指数网格）

代码架构
--------

模块层次::

    atomscf/
    ├── grid.py          # 径向网格生成
    ├── operator.py      # 方程求解器
    ├── scf.py          # DFT SCF 框架
    ├── scf_hf.py       # HF SCF 框架
    ├── hartree.py      # Hartree 势计算
    ├── occupations.py  # 电子占据方案
    ├── utils.py        # 积分工具
    ├── xc/             # 交换-关联泛函
    │   ├── lda.py
    │   └── vwn.py
    └── hf/             # HF 交换积分
        ├── slater.py
        ├── angular.py
        └── exchange.py

依赖项
------

**核心依赖**:

- Python ≥ 3.9
- NumPy ≥ 1.20
- SciPy (用于 eigh, 3j 符号等)
- SymPy (用于符号计算 Wigner 系数)

**可选依赖**:

- pytest ≥ 7 (开发测试)
- Sphinx ≥ 7 (文档生成)

开发状态
--------

**已实现** (v0.1.0):

- ✅ RHF 多 l 通道（s, p, d）
- ✅ UHF 自旋极化
- ✅ LSDA (PZ81, VWN)
- ✅ 多种求解器与网格类型
- ✅ Slater 积分径向计算

**规划中**:

- ⏳ ROHF (Restricted Open-shell HF)
- ⏳ DIIS 收敛加速
- ⏳ GGA 泛函支持
- ⏳ 完整单元测试覆盖

许可协议
--------

本项目为教学用途，采用 MIT 协议开源。

参考文献
--------

1. Clementi, E. & Roetti, C. *Atomic Data and Nuclear Data Tables* **14**, 177 (1974)
2. Perdew, J. P. & Zunger, A. *Phys. Rev. B* **23**, 5048 (1981)
3. Vosko, S. H., Wilk, L. & Nusair, M. *Can. J. Phys.* **58**, 1200 (1980)
4. Martin, R. M. *Electronic Structure: Basic Theory and Practical Methods* (2004)
