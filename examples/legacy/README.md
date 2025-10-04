# Legacy Examples

此目录包含历史版本的示例脚本，主要用于：
- 方法开发过程的记录
- 不同实现版本的对比
- 教学参考

**注意**：这些脚本可能使用过时的 API 或参数，建议优先使用主目录中的示例。

## 文件说明

### 碳原子 LSDA 计算（历史版本）

- **run_c_lsda_pz81_log.py** - PZ81 泛函 + 对数网格（已被 run_c_lsda_pz81.py 替代）
- **run_c_lsda_vwn_transformed.py** - VWN 泛函 + 变量变换方法（实验性版本，已整合到主线）

### 方法对比与验证

- **benchmark_methods_carbon.py** - 多种求解器方法对比测试（参考用）
- **compare_c_to_nist_lsd.py** - PZ81 与 NIST 数据对比（旧版）
- **compare_c_to_nist_lsd_vwn.py** - VWN 与 NIST 数据对比（旧版）

### 验证脚本

- **verify_carbon_screened_potential.py** - 碳原子屏蔽势验证（实验性）
- **verify_hydrogen_transformed.py** - 氢原子变量变换方法验证（实验性）
- **quick_atom_test.py** - Z=1-18 原子快速测试（实验性）

## 推荐的替代脚本

请参考主 examples 目录中的以下脚本：

- **run_c_lsda_vwn.py** - 推荐的 VWN 泛函计算脚本
- **run_c_lsda_vwn_log.py** - 推荐的对数网格 + FD5-aux 方法
- **run_c_lsda_pz81.py** - 推荐的 PZ81 泛函计算脚本
- **validate_numerov_frozen.py** - 推荐的冻结势验证脚本
