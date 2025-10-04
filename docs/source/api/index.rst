API 参考
=========

本节提供完整的模块、类和函数接口文档。

.. toctree::
   :maxdepth: 2

   modules

核心模块
--------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   atomscf.grid
   atomscf.operator
   atomscf.scf
   atomscf.scf_hf
   atomscf.hartree
   atomscf.occupations
   atomscf.utils

HF 子模块
---------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   atomscf.hf.slater
   atomscf.hf.angular
   atomscf.hf.exchange

交换-关联泛函
-------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   atomscf.xc.lda
   atomscf.xc.vwn
   atomscf.xc.constants

常用功能索引
------------

网格生成
~~~~~~~~

.. autosummary::

   atomscf.grid.radial_grid_linear
   atomscf.grid.radial_grid_log
   atomscf.grid.radial_grid_exp_transformed
   atomscf.grid.radial_grid_mixed

HF 计算
~~~~~~~

.. autosummary::

   atomscf.scf_hf.run_hf_minimal
   atomscf.scf_hf.run_hf_scf_s
   atomscf.scf_hf.run_hf_scf

DFT 计算
~~~~~~~~

.. autosummary::

   atomscf.scf.run_lsda_x_only
   atomscf.scf.run_lsda_pz81
   atomscf.scf.run_lsda_vwn

方程求解器
~~~~~~~~~~

.. autosummary::

   atomscf.operator.solve_bound_states_fd
   atomscf.operator.solve_bound_states_fd5
   atomscf.operator.solve_bound_states_numerov
   atomscf.operator.solve_bound_states_transformed

配置与结果类
------------

HF 配置
~~~~~~~

.. autoclass:: atomscf.scf_hf.HFConfig
   :members:
   :show-inheritance:

.. autoclass:: atomscf.scf_hf.HFSCFGeneralConfig
   :members:
   :show-inheritance:

HF 结果
~~~~~~~

.. autoclass:: atomscf.scf_hf.HFResult
   :members:
   :show-inheritance:

.. autoclass:: atomscf.scf_hf.HFSCFGeneralResult
   :members:
   :show-inheritance:

DFT 配置
~~~~~~~~

.. autoclass:: atomscf.scf.SCFConfig
   :members:
   :show-inheritance:

DFT 结果
~~~~~~~~

.. autoclass:: atomscf.scf.SCFResult
   :members:
   :show-inheritance:
