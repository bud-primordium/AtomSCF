原子计算示例
============

本节提供常见原子的详细计算示例和结果对比。

氢 (H, Z=1)
-----------

单电子体系，HF 与精确解相同。

配置
~~~~

.. code-block:: python

   from atomscf.grid import radial_grid_linear
   from atomscf.scf_hf import run_hf_minimal

   r, w = radial_grid_linear(n=800, rmin=1e-6, rmax=50.0)
   result = run_hf_minimal(Z=1, r=r, w=w)

结果
~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - 性质
     - 数值
     - 理论值
   * - E_total
     - -0.500 Ha
     - -0.500 Ha（精确）
   * - ε_1s
     - -0.500 Ha
     - -0.500 Ha
   * - 误差
     - < 0.001%
     - -

氦 (He, Z=2)
------------

闭壳层，RHF 适用。

RHF 计算
~~~~~~~~

.. code-block:: python

   from atomscf.scf_hf import run_hf_scf, HFSCFGeneralConfig

   config = HFSCFGeneralConfig(
       Z=2, r=r, w=w,
       occ_by_l={0: [2.0]},
       eigs_per_l={0: 1},
       spin_mode='RHF',
       mix_alpha=0.5,
       tol=1e-6,
       maxiter=100
   )
   result = run_hf_scf(config)

结果对比
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - 方法
     - E_total (Ha)
     - ε_1s (Ha)
     - 相对误差
   * - 本代码 (RHF)
     - -2.787
     - -0.866
     - 4.0%
   * - Clementi (HF)
     - -2.862
     - -0.918
     - 1.4%
   * - 实验
     - -2.904
     - -
     - -

.. note::

   RHF 缺少关联能（~0.04 Ha）。使用 LSDA 可改善至 ~1% 误差。

锂 (Li, Z=3)
------------

开壳层 (1s² 2s¹)，UHF 优于 RHF。

UHF 计算
~~~~~~~~

.. code-block:: python

   config = HFSCFGeneralConfig(
       Z=3, r=r, w=w,
       occ_by_l={0: [2.0, 1.0]},
       occ_by_l_spin={
           0: {'up': [1.0, 1.0], 'down': [1.0, 0.0]}
       },
       eigs_per_l={0: 2},
       spin_mode='UHF',
       mix_alpha=0.3,
       tol=1e-6,
       maxiter=120
   )
   result = run_hf_scf(config)

结果对比
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - 方法
     - E_total (Ha)
     - ε_2s (Ha)
     - 误差 (mHa)
   * - RHF
     - -7.159
     - -0.089
     - 274
   * - UHF
     - -7.213
     - -0.193 (↑)
     - 219
   * - Clementi (ROHF)
     - -7.433
     - -0.196
     - -

**改善**: UHF 比 RHF 降低误差 20%

碳 (C, Z=6)
-----------

开壳层 p² (³P 基态)，DFT 推荐。

LSDA-VWN 计算
~~~~~~~~~~~~~

.. code-block:: python

   from atomscf.scf import run_lsda_vwn, SCFConfig

   config = SCFConfig(
       Z=6, r=r, w=w,
       lmax=2,
       eigs_per_l=2,
       eig_solver="fd5_aux",
       xc="VWN",
       mix_alpha=0.5,
       tol=5e-5,
       maxiter=140
   )
   result = run_lsda_vwn(config, verbose=True)

结果对比
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - 性质
     - LSDA-VWN
     - Clementi (ROHF)
     - 误差
   * - E_total
     - -37.83 Ha
     - -37.69 Ha
     - +140 mHa
   * - ε_2s
     - -1.02 Ha
     - -1.02 Ha
     - ~0
   * - ε_2p
     - -0.56 Ha
     - -0.43 Ha
     - +130 mHa

.. warning::

   碳原子 HF 计算需要 ROHF（非 UHF）以正确描述 ³P 态。
   LSDA 包含关联，结果更合理。

氮 (N, Z=7)
-----------

开壳层 p³ (⁴S 基态)，高自旋。

LSDA-PZ81 计算
~~~~~~~~~~~~~~

.. code-block:: python

   from atomscf.scf import run_lsda_pz81

   config = SCFConfig(
       Z=7, r=r, w=w,
       lmax=2,
       eigs_per_l=2,
       xc="PZ81",  # PZ81 关联
       mix_alpha=0.4,
       tol=5e-5,
       maxiter=150
   )
   result = run_lsda_pz81(config)

结果
~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - 性质
     - LSDA-PZ81
     - 文献值
   * - E_total
     - -54.4 Ha
     - -54.6 Ha
   * - ε_2p
     - -0.67 Ha
     - -0.54 Ha

氧 (O, Z=8)
-----------

开壳层 p⁴ (³P 基态)。

LSDA-VWN 计算
~~~~~~~~~~~~~

.. code-block:: python

   config = SCFConfig(
       Z=8, r=r, w=w,
       lmax=2,
       eigs_per_l=2,
       xc="VWN",
       mix_alpha=0.3,  # 慢收敛，小混合
       tol=5e-5,
       maxiter=180
   )
   result = run_lsda_vwn(config, progress_every=20)

收敛技巧
~~~~~~~~

氧原子收敛较难，建议：

1. 减小 `mix_alpha` 至 0.2-0.3
2. 增加 `maxiter` 至 200+
3. 放宽 `tol` 至 1e-4

氖 (Ne, Z=10)
-------------

闭壳层 (1s² 2s² 2p⁶)，RHF 适用。

RHF 计算
~~~~~~~~

.. code-block:: python

   config = HFSCFGeneralConfig(
       Z=10, r=r, w=w,
       occ_by_l={0: [2.0, 2.0], 1: [6.0]},
       eigs_per_l={0: 2, 1: 1},
       spin_mode='RHF',
       mix_alpha=0.4,
       tol=1e-6,
       maxiter=150
   )
   result = run_hf_scf(config)

结果对比
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - 性质
     - RHF
     - Clementi (HF)
   * - E_total
     - -128.3 Ha
     - -128.5 Ha
   * - ε_1s
     - -32.6 Ha
     - -32.8 Ha
   * - ε_2s
     - -1.90 Ha
     - -1.93 Ha
   * - ε_2p
     - -0.82 Ha
     - -0.85 Ha

批量计算脚本
------------

计算多个原子：

.. code-block:: python

   from atomscf.grid import radial_grid_linear
   from atomscf.scf import run_lsda_vwn, SCFConfig

   # 原子列表
   atoms = {
       'H': 1, 'He': 2, 'Li': 3, 'Be': 4,
       'B': 5, 'C': 6, 'N': 7, 'O': 8,
       'F': 9, 'Ne': 10
   }

   r, w = radial_grid_linear(n=1200, rmin=1e-6, rmax=70.0)
   results = {}

   for name, Z in atoms.items():
       print(f"\n计算 {name} (Z={Z})...")

       config = SCFConfig(
           Z=Z, r=r, w=w,
           lmax=2,
           eigs_per_l=2,
           xc="VWN",
           mix_alpha=0.3 if Z > 5 else 0.5,
           tol=5e-5,
           maxiter=200
       )

       try:
           result = run_lsda_vwn(config, verbose=False)
           results[name] = {
               'E': result.energies['E_total'],
               'converged': result.converged,
               'iters': result.iterations
           }
           print(f"  ✓ E = {result.energies['E_total']:.4f} Ha")
       except Exception as e:
           print(f"  ✗ 失败: {e}")

   # 保存结果
   import json
   with open('results.json', 'w') as f:
       json.dump(results, f, indent=2)

网格收敛性测试
--------------

测试不同网格点数的影响：

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   grid_sizes = [400, 600, 800, 1000, 1200, 1500]
   energies = []

   for n in grid_sizes:
       r, w = radial_grid_linear(n=n, rmin=1e-6, rmax=50.0)
       result = run_hf_minimal(Z=2, r=r, w=w)  # He 原子
       energies.append(result.E_total)
       print(f"n={n:4d}: E = {result.E_total:.8f} Ha")

   # 绘图
   plt.figure(figsize=(8, 5))
   plt.plot(grid_sizes, energies, 'o-')
   plt.axhline(y=-2.8617, color='r', linestyle='--', label='HF 极限')
   plt.xlabel('网格点数 n')
   plt.ylabel('总能量 (Ha)')
   plt.title('He 原子网格收敛性')
   plt.legend()
   plt.grid(True)
   plt.tight_layout()
   plt.savefig('grid_convergence.png', dpi=150)

预期趋势：能量单调收敛至 HF 极限。

参考数据来源
------------

- **Clementi & Roetti (1974)**: HF 极限值（Z ≤ 54）
- **NIST Atomic Spectra Database**: 实验光谱数据
- **Computational Chemistry Comparison and Benchmark Database**: 多种方法对比

常见问题
--------

为什么我的结果与文献不同？
~~~~~~~~~~~~~~~~~~~~~~~~~~

可能原因：

1. **方法差异**: 本代码 RHF/UHF，文献可能 ROHF
2. **基组差异**: 网格离散 vs Gaussian 基组
3. **关联缺失**: HF 无关联，DFT 有近似关联
4. **收敛不足**: 检查 `converged` 标志

如何选择 HF vs DFT？
~~~~~~~~~~~~~~~~~~~~

- **HF (RHF/UHF)**:
  - 优势：交换精确
  - 适用：闭壳层、教学、基准
  - 缺点：无关联

- **DFT (LSDA)**:
  - 优势：速度快、包含关联
  - 适用：开壳层、过渡金属
  - 缺点：交换近似、带隙低估

**建议**: 教学用 HF，实际计算用 DFT。

下一步
------

- :doc:`benchmarks`: 与文献数据系统对比
- :doc:`../algorithm/hartree_fock`: HF 理论深入
- :doc:`../api/index`: 高级 API 使用
