使用示例
========

本节提供常见原子的计算示例和结果对比。

.. toctree::
   :maxdepth: 2

   basic_usage
   atoms
   benchmarks

快速开始
--------

氢原子最小 HF 计算
~~~~~~~~~~~~~~~~~~

最简单的单电子 HF 计算：

.. code-block:: python

   from atomscf.grid import radial_grid_linear
   from atomscf.scf_hf import run_hf_minimal

   # 创建线性网格
   r, w = radial_grid_linear(n=800, rmin=1e-6, rmax=50.0)

   # 氢原子 HF (Z=1)
   result = run_hf_minimal(Z=1, r=r, w=w)

   print(f"总能量: {result.E_total:.6f} Ha")
   print(f"1s 轨道能: {result.epsilon_1s:.6f} Ha")

预期输出::

   总能量: -0.499995 Ha
   1s 轨道能: -0.499995 Ha

氦原子 RHF 计算
~~~~~~~~~~~~~~~

闭壳层多电子体系：

.. code-block:: python

   from atomscf.grid import radial_grid_linear
   from atomscf.scf_hf import run_hf_scf, HFSCFGeneralConfig

   r, w = radial_grid_linear(n=800, rmin=1e-6, rmax=50.0)

   cfg = HFSCFGeneralConfig(
       Z=2,                   # 氦原子
       r=r, w=w,
       occ_by_l={0: [2.0]},  # 1s²
       eigs_per_l={0: 1},    # 求解 1 个 s 轨道
       spin_mode='RHF',
       mix_alpha=0.5,
       tol=1e-6,
       maxiter=100
   )

   result = run_hf_scf(cfg)

   print(f"总能量: {result.E_total:.6f} Ha")
   print(f"ε_1s: {result.eigenvalues_by_l[0][0]:.6f} Ha")

锂原子 UHF 计算
~~~~~~~~~~~~~~~

开壳层自旋极化：

.. code-block:: python

   from atomscf.grid import radial_grid_linear
   from atomscf.scf_hf import run_hf_scf, HFSCFGeneralConfig

   r, w = radial_grid_linear(n=1000, rmin=1e-6, rmax=60.0)

   cfg = HFSCFGeneralConfig(
       Z=3,                      # 锂原子
       r=r, w=w,
       occ_by_l={0: [2.0, 1.0]}, # 1s² 2s¹
       occ_by_l_spin={
           0: {
               'up': [1.0, 1.0],    # 1s↑ 2s↑
               'down': [1.0, 0.0],  # 1s↓ (2s 无占据)
           }
       },
       eigs_per_l={0: 2},        # 求解 2 个 s 轨道
       spin_mode='UHF',
       mix_alpha=0.3,
       tol=1e-6,
       maxiter=120
   )

   result = run_hf_scf(cfg)

   print(f"总能量: {result.E_total:.6f} Ha")
   print(f"ε_2s(↑): {result.eigenvalues_by_l_spin[(0, 'up')][1]:.6f} Ha")

碳原子 LSDA 计算
~~~~~~~~~~~~~~~~

密度泛函方法：

.. code-block:: python

   from atomscf.grid import radial_grid_linear
   from atomscf.scf import run_lsda_vwn, SCFConfig

   r, w = radial_grid_linear(n=1200, rmin=1e-6, rmax=70.0)

   cfg = SCFConfig(
       Z=6,              # 碳原子
       r=r, w=w,
       lmax=2,           # 最大角动量（包含 p 轨道）
       eigs_per_l=2,     # 每个 l 求解 2 个轨道
       eig_solver="fd5_aux",
       xc="VWN",
       mix_alpha=0.5,
       tol=5e-5,
       maxiter=140
   )

   result = run_lsda_vwn(cfg, verbose=True)

   print(f"总能量: {result.energies['E_total']:.6f} Ha")
   print(f"ε_2p(↑): {result.eps_by_l_sigma[(1, 'up')][0]:.6f} Ha")

更多示例
--------

完整示例代码位于 ``examples/`` 目录：

- ``run_h_1s_fd.py``: 氢原子（不同求解器对比）
- ``run_c_lsda_vwn.py``: 碳原子 LSDA
- ``test_uhf.py``: UHF vs RHF 对比
- ``benchmark_*.py``: 批量原子计算
