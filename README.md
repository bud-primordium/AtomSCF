# AtomSCF

教学用原子自旋极化 Hartree–Fock 与 LSDA-DFT（径向一维）最小实现。

- 包名：`atomscf`
- 目录：`src/`、`tests/`、`docs/`、`examples/`
- 特性：
  - 径向网格（线性/对数）、非均匀网格二阶导数离散
  - LSDA：Dirac 交换 + PZ81 关联（`run_lsda_pz81`），总能分解
  - HF 最小实现（H）：`run_hf_minimal`

## 安装与环境（建议）

- 使用 uv（可选）：
  ```bash
  cd AtomSCF
  uv venv
  source .venv/bin/activate
  uv pip install -e .[dev]
  uv run pytest -q
  ```
- 或使用 Python venv：
  ```bash
  cd AtomSCF
  python -m venv .venv
  source .venv/bin/activate
  pip install -e .[dev]
  pytest -q
  ```

## 运行示例

```bash
PYTHONPATH=src python examples/run_h_1s_fd.py
```

若需 SCF 迭代进度输出，调用 `run_lsda_x_only(..., verbose=True)` 或 `run_lsda_pz81(..., verbose=True)`。

## 文档与参考
- 中文文档见 `docs/`；参考文献见 `docs/REFERENCES.md`。
- 术语、数值与实现细节将以 Sphinx 形式整理于 `docs/`，对外保持统一表述。
