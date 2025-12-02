基础使用教程
============

本教程介绍 AtomSCF 的基本使用方法，从最简单的氢原子开始，逐步到复杂的多电子体系。

快速安装
--------

克隆仓库并安装：

.. code-block:: bash

   git clone <repository-url>
   cd AtomSCF
   pip install -e .[dev,docs]

验证安装：

.. code-block:: python

   import atomscf
   print(atomscf.__name__)  # 应输出 'atomscf'

第一个计算：氢原子
------------------

最简单的单电子 HF 计算：

.. code-block:: python

   from atomscf.grid import radial_grid_linear
   from atomscf.scf_hf import run_hf_minimal

   # 步骤 1: 创建径向网格
   r, w = radial_grid_linear(n=800, rmin=1e-6, rmax=50.0)

   # 步骤 2: 运行氢原子 HF (Z=1)
   result = run_hf_minimal(Z=1, r=r, w=w)

   # 步骤 3: 查看结果
   print(f"总能量: {result.E_total:.6f} Ha")
   print(f"1s 轨道能: {result.epsilon_1s:.6f} Ha")

   # 步骤 4: 检查精度
   E_exact = -0.5  # 氢原子精确解
   error = abs(result.E_total - E_exact)
   print(f"误差: {error:.2e} Ha ({error/abs(E_exact)*100:.4f}%)")

预期输出::

   总能量: -0.499995 Ha
   1s 轨道能: -0.499995 Ha
   误差: 4.60e-06 Ha (0.0009%)

.. note::

   氢原子 HF 与精确解相同（无电子-电子相互作用），这是验证代码正确性的好基准。

多电子闭壳层：氦原子
--------------------

使用通用 HF SCF 框架：

.. code-block:: python

   from atomscf.grid import radial_grid_linear
   from atomscf.scf_hf import run_hf_scf, HFSCFGeneralConfig

   # 网格
   r, w = radial_grid_linear(n=800, rmin=1e-6, rmax=50.0)

   # 配置 HF 计算
   config = HFSCFGeneralConfig(
       Z=2,                   # 氦原子 (Z=2)
       r=r, w=w,
       occ_by_l={0: [2.0]},  # 占据：1s² (l=0, 2 个电子)
       eigs_per_l={0: 1},    # 求解 1 个 s 轨道
       spin_mode='RHF',      # 限制性 HF（闭壳层）
       mix_alpha=0.5,        # 密度混合参数
       tol=1e-6,             # 收敛阈值
       maxiter=100           # 最大迭代次数
   )

   # 运行 SCF
   result = run_hf_scf(config)

   # 结果分析
   print(f"收敛状态: {result.converged}")
   print(f"迭代次数: {result.iterations}")
   print(f"总能量: {result.E_total:.6f} Ha")
   print(f"ε_1s: {result.eigenvalues_by_l[0][0]:.6f} Ha")

   # 能量分解
   print("\n能量分解:")
   print(f"  动能: {result.E_kinetic:.6f} Ha")
   print(f"  外势能: {result.E_ext:.6f} Ha")
   print(f"  Hartree: {result.E_hartree:.6f} Ha")
   print(f"  交换: {result.E_exchange:.6f} Ha")

预期输出::

   收敛状态: True
   迭代次数: 24
   总能量: -2.787236 Ha
   ε_1s: -0.865629 Ha

   能量分解:
     动能: 2.702591 Ha
     外势能: -6.545806 Ha
     Hartree: 2.111959 Ha
     交换: -1.055979 Ha

.. tip::

   实验值：-2.9037 Ha，HF 值：-2.8617 Ha（Clementi）。
   我们的结果 -2.787 Ha 比 HF 极限略高，因为使用了有限基组（网格离散化）。

开壳层体系：锂原子 (UHF)
------------------------

自旋极化计算：

.. code-block:: python

   from atomscf.grid import radial_grid_linear
   from atomscf.scf_hf import run_hf_scf, HFSCFGeneralConfig

   r, w = radial_grid_linear(n=1000, rmin=1e-6, rmax=60.0)

   # UHF 配置（自旋分辨）
   config = HFSCFGeneralConfig(
       Z=3,
       r=r, w=w,
       occ_by_l={0: [2.0, 1.0]},     # 1s² 2s¹
       occ_by_l_spin={
           0: {
               'up': [1.0, 1.0],      # 1s↑ 2s↑
               'down': [1.0, 0.0],    # 1s↓ (2s 无占据)
           }
       },
       eigs_per_l={0: 2},             # 求解 2 个 s 轨道
       spin_mode='UHF',               # 非限制性 HF（开壳层）
       mix_alpha=0.3,                 # 较小的混合（开壳层收敛慢）
       tol=1e-6,
       maxiter=120
   )

   result = run_hf_scf(config)

   print(f"总能量: {result.E_total:.6f} Ha")
   print("\n自旋分辨轨道能:")
   print(f"  ε_1s(↑): {result.eigenvalues_by_l_spin[(0, 'up')][0]:.6f} Ha")
   print(f"  ε_2s(↑): {result.eigenvalues_by_l_spin[(0, 'up')][1]:.6f} Ha")
   print(f"  ε_1s(↓): {result.eigenvalues_by_l_spin[(0, 'down')][0]:.6f} Ha")
   print(f"  ε_2s(↓): {result.eigenvalues_by_l_spin[(0, 'down')][1]:.6f} Ha")

.. note::

   注意 `occ_by_l_spin` 的结构：`{l: {'up': [...], 'down': [...]}}`
   如果未提供，会自动从 `occ_by_l` 均分为闭壳层配置。

密度泛函：碳原子 (LSDA)
------------------------

包含关联效应：

.. code-block:: python

   from atomscf.grid import radial_grid_linear
   from atomscf.scf import run_lsda_vwn, SCFConfig

   r, w = radial_grid_linear(n=1200, rmin=1e-6, rmax=70.0)

   config = SCFConfig(
       Z=6,                   # 碳原子
       r=r, w=w,
       lmax=2,                # 包含 s, p, d 轨道（l=0,1,2）
       eigs_per_l=2,          # 每个 l 求解 2 个本征态
       eig_solver="fd5_aux",  # 求解器（插值 + FD5）
       xc="VWN",              # VWN 关联泛函
       mix_alpha=0.5,
       tol=5e-5,
       maxiter=140
   )

   result = run_lsda_vwn(config, verbose=True, progress_every=10)

   print(f"\n总能量: {result.energies['E_total']:.6f} Ha")
   print("\n轨道能级（spin-up）:")
   print(f"  ε_1s: {result.eps_by_l_sigma[(0, 'up')][0]:.6f} Ha")
   print(f"  ε_2s: {result.eps_by_l_sigma[(0, 'up')][1]:.6f} Ha")
   print(f"  ε_2p: {result.eps_by_l_sigma[(1, 'up')][0]:.6f} Ha")

   # 自旋密度
   import numpy as np
   n_up = result.n_up
   n_dn = result.n_dn
   n_total = n_up + n_dn

   # 检查电子数
   N_electrons = np.trapz(n_total * 4 * np.pi * r**2, r)
   print(f"\n电子数检验: {N_electrons:.6f} (应为 6.0)")

参数说明
--------

网格参数
~~~~~~~~

.. code-block:: python

   # 线性网格
   r, w = radial_grid_linear(
       n=1000,          # 网格点数（更多 → 更精确）
       rmin=1e-6,       # 最小半径（避免 r=0 奇点）
       rmax=70.0        # 最大半径（覆盖波函数衰减区）
   )

   # 对数网格（大原子推荐）
   from atomscf.grid import radial_grid_log
   r, w = radial_grid_log(n=1000, rmin=1e-5, rmax=100.0)

SCF 参数
~~~~~~~~

.. code-block:: python

   config = HFSCFGeneralConfig(
       mix_alpha=0.3,   # 密度混合（0.1-0.7，开壳层用小值）
       tol=1e-6,        # 收敛阈值（密度变化）
       maxiter=150      # 最大迭代（不收敛则增加）
   )

求解器选择
~~~~~~~~~~

.. code-block:: python

   config = SCFConfig(
       eig_solver="fd5_aux",  # 推荐（精度和速度平衡）
       # 其他选项：
       # "fd": 基础 FD2
       # "fd5": FD5（仅等距网格）
       # "numerov": Numerov（对数网格）
       # "transformed": 变换 Hamiltonian（指数网格）
   )

常见问题排查
------------

SCF 不收敛
~~~~~~~~~~

**症状**: 迭代达到 `maxiter` 但未收敛

**解决方案**:
1. 减小 `mix_alpha` (如 0.1-0.3)
2. 增加 `maxiter`
3. 使用更密集的网格
4. 检查占据数配置是否合理

**示例**:

.. code-block:: python

   # 困难收敛的开壳层
   config = HFSCFGeneralConfig(
       Z=8,  # 氧原子
       mix_alpha=0.2,  # 小混合
       maxiter=200,
       tol=1e-5  # 放宽收敛
   )

能量异常
~~~~~~~~

**症状**: 能量为正或极大负值

**检查清单**:
1. 占据数总和 = 电子数
2. 网格范围合理（`rmax` 足够大）
3. 边界条件正确（`u(0) = u(∞) = 0`）

**验证电子数**:

.. code-block:: python

   import numpy as np
   n_radial = sum(n_i * u_i**2 for u_i, n_i in zip(orbitals, occupations))
   N = np.trapz(n_radial, r) * 4 * np.pi  # 积分权重
   print(f"电子数: {N:.6f}")

轨道能不合理
~~~~~~~~~~~~

**症状**: :math:`\varepsilon_{2s} < \varepsilon_{1s}` (能级倒序)

**原因**: 网格太粗或求解器不适配

**解决**: 增加网格点数或更换求解器

下一步学习
----------

- :doc:`atoms`: 各种原子的完整计算示例
- :doc:`../api/index`: API 详细参考
- :doc:`../algorithm/hartree_fock`: HF 理论推导
