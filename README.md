# AtomSCF

教学用球对称原子 Hartree-Fock 与 LSDA-DFT（径向一维）量子化学计算工具。

## 重要说明

**AtomSCF 专门用于球对称体系的径向计算**，适用于单原子体系（H, He, Li, C, N, O 等），**不支持分子或固体**。

### 核心方法特点

- **径向一维离散化**：将三维 Schrödinger 方程约化为径向方程，仅沿半径方向数值离散
- **实空间方法**：有限差分（FD2/FD5）+ 变量变换方法（指数网格），**非基组展开**
- **变量变换技术**：基于指数网格 $r(j) = R_p(\exp(j\delta) - 1)$ 的 $u \to v$ 变换，消除一阶导数项，精度提升约 7 倍
- **交换-关联处理**：
  - **HF**：精确交换（Slater 积分 + Wigner 3-j），支持 RHF/UHF
  - **DFT-LSDA**：Dirac 交换 $\varepsilon_x = -(3/(4\pi))^{1/3} (3\rho)^{1/3}$ + PZ81/VWN 关联

详细推导见 [变量变换理论](docs/source/algorithm/numerical_methods.rst)，上游实现源自 [Computational Physics Assignment 7-2](https://github.com/bud-primordium/Computational-Physics-Fall-2024/tree/main/Assignment_7/Problem_2)。

---

## 项目结构

```
AtomSCF/
├── src/atomscf/          # 核心源码
│   ├── grid.py           # 径向网格生成（线性/对数/指数变换/混合）
│   ├── operator.py       # Schrödinger 方程求解器（FD2/FD5/变换/Numerov）
│   ├── scf.py            # DFT 自洽场（LSDA-PZ81/VWN）
│   ├── scf_hf.py         # HF 自洽场（RHF/UHF）
│   ├── hartree.py        # Hartree 势计算
│   ├── hf/               # HF 交换算符（Slater 积分、角动量耦合）
│   └── xc/               # 交换-关联泛函（LDA/VWN）
├── tests/                # 单元测试（pytest）
├── examples/             # 使用示例（H/He/Li/C 原子）
├── docs/                 # Sphinx 文档
│   ├── source/algorithm/ # 算法原理推导（HF/DFT/数值方法）
│   ├── source/examples/  # 使用教程
│   └── source/api/       # API 自动生成文档
└── pyproject.toml        # 项目配置
```

---

## 源代码结构

```
src/atomscf/
├── grid.py          # 径向网格生成
│   ├── radial_grid_linear           # 线性等距网格
│   ├── radial_grid_log              # 对数网格（ln(r) 等差）
│   ├── radial_grid_exp_transformed  # 指数变换网格（变量代换专用）
│   └── radial_grid_mixed            # 混合网格（近核对数 + 远程线性）
│
├── operator.py      # 径向 Schrödinger 方程求解器
│   ├── solve_bound_states_fd        # FD2（非均匀网格）
│   ├── solve_bound_states_fd5       # FD5（等距网格，O(h⁴)）
│   ├── solve_bound_states_transformed # 变量变换方法（指数网格）
│   └── solve_bound_states_fd5_auxlinear # FD5 + 插值辅助网格
│
├── scf.py           # DFT 自洽场框架
│   ├── SCFConfig, SCFResult         # 配置与结果容器
│   ├── run_lsda_x_only              # 仅交换（教学）
│   ├── run_lsda_pz81                # Dirac 交换 + PZ81 关联
│   └── run_lsda_vwn                 # Dirac 交换 + VWN 关联
│
├── scf_hf.py        # HF 自洽场框架
│   ├── run_hf_minimal               # 最小 HF（H 原子，教学）
│   ├── run_hf_scf_s                 # s 轨道 HF（He/Li）
│   └── run_hf_scf                   # 通用 HF（RHF/UHF，多 l 通道）
│
├── hartree.py       # Hartree 势计算（泊松方程求解）
├── numerov.py       # Numerov 方法（对数网格边界值问题）
├── occupations.py   # 电子占据方案（Z=1-18，Hund 规则）
├── utils.py         # 工具函数（积分、归一化、权重）
├── io.py            # 数据导出（CSV/JSON）
│
├── hf/              # HF 交换算符子模块
│   ├── slater.py    # Slater 径向积分（Y^k/Z^k 两段累积）
│   ├── angular.py   # Wigner 3-j 符号、角动量耦合因子
│   └── exchange.py  # 交换算符（s 轨道、通用多 l）
│
└── xc/              # 交换-关联泛函子模块
    ├── lda.py       # Dirac 交换 + PZ81 关联
    ├── vwn.py       # VWN 关联泛函（RPA 参数化）
    └── constants.py # 物理常数
```

**核心工作流**：
1. `grid.py` → 生成径向网格 `(r, w)`
2. `operator.py` → 构造 Hamiltonian 矩阵，求解本征值问题
3. `scf.py` / `scf_hf.py` → 自洽迭代（密度混合）
4. `io.py` → 导出结果（轨道、能量、密度）

---

## 关键技术细节

### 变量变换方法

**网格定义**（指数变换网格）：

$$
r(j) = R_p (\exp(j\delta) - 1) + r_{\min}, \quad j = 0, 1, \dots, N-1
$$

**变量代换**：

$$
u(j) = v(j) \cdot \exp(j\delta/2)
$$

**原始径向 Schrödinger 方程**：

$$
-\frac{1}{2}\frac{d^2 u}{dr^2} + \left[ V(r) + \frac{\ell(\ell+1)}{2r^2} \right] u = \varepsilon u
$$

引入 $w(r) = \exp(-r/(2R_p))$，则 $u(r) = v(r) \cdot w(r)$。

**一阶导数**：

$$
\frac{du}{dr} = w \left( \frac{dv}{dr} - \frac{v}{2R_p} \right)
$$

**二阶导数**：

$$
\frac{d^2u}{dr^2} = w \left( \frac{d^2v}{dr^2} - \frac{1}{R_p}\frac{dv}{dr} + \frac{v}{4R_p^2} \right)
$$

代入原方程并取 $R_p = 1/\delta$，得到**变换后的方程**（一阶导数项消失）：

$$
-\frac{1}{2}\frac{d^2v}{dr^2} + \left[ V(r) + \frac{\ell(\ell+1)}{2r^2} - \frac{\delta^2}{8} \right] v = \varepsilon v
$$

**离散方程**（索引空间 $j$）：

$$
v''(j) - \frac{\delta^2}{4}v(j) = 2R_p^2\delta^2 \exp(2j\delta) [\varepsilon - V_{\text{eff}}(r(j))] v(j)
$$

**关键特性**：
- 消除一阶导数项 → 对称 Hamiltonian 矩阵 → `eigh()` 求解
- 精度提升约 7 倍，速度提升约 3 倍
- 包含 $r=0$ 点（当 $j=0$ 时 $r=r_{\min}=0$）

**上游实现**：[Computational Physics Assignment 7-2](https://github.com/bud-primordium/Computational-Physics-Fall-2024/tree/main/Assignment_7/Problem_2)

详细推导见 [docs/source/algorithm/numerical_methods.rst](docs/source/algorithm/numerical_methods.rst)。

---

## 快速开始

### 安装

**使用 uv（推荐）**：
```bash
cd AtomSCF
uv venv
source .venv/bin/activate
uv pip install -e .[dev]
```

**使用标准 venv**：
```bash
cd AtomSCF
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### 运行示例

**推荐使用统一 CLI**（`examples/run_atom.py`）：

```bash
# 基本用法：碳原子 LSDA-VWN（默认配置）
python examples/run_atom.py --Z 6

# 氢原子
python examples/run_atom.py --Z 1 --n 2000 --total-span 8

# 铝原子 LDA 模式（用于赝势生成）
python examples/run_atom.py --Z 13 --mode LDA --n 2000

# 导出供赝势生成器使用
python examples/run_atom.py --Z 13 --mode LDA --export-ppgen data/al_ae_lda.json

# 完整参数示例
python examples/run_atom.py \
  --Z 13 \
  --grid exp \
  --solver transformed \
  --n 2000 \
  --total-span 8.0 \
  --rmax 150.0 \
  --tol 1e-6 \
  --no-verbose

# Hartree-Fock 计算
python examples/run_atom.py --Z 6 --method HF --hf-mode RHF --n 1000  # Carbon RHF
python examples/run_atom.py --Z 6 --method HF --hf-mode UHF --n 1000  # Carbon UHF
```

**Carbon HF 结果对比**（n=1000 vs ROHF 参考）：

| 方法 | E_total (Ha) | 1s (Ha) | 2s (Ha) | 2p (Ha) | 用时 |
|------|-------------|---------|---------|---------|------|
| **RHF** | -37.445 | -11.307 | -0.703 | -0.232 | 46.8s |
| **UHF** | -37.445 | -11.308 | -0.703 | -0.232 | 51.8s |
| **ROHF (参考)** | -37.689 | -11.326 | -0.706 | -0.433 | - |

**注**：Carbon 基态为 ³P（开壳层），RHF/UHF 与 ROHF 的 2p 能级差异较大属于预期。
```

**默认推荐配置**：
- `--grid exp`：指数网格（核附近密，远处疏）
- `--solver transformed`：变量变换方法（快速且稳定）
- `--n 2000`：网格点数（精度 ~1.5%）
- `--total-span 8.0`：网格跨度参数

**程序化使用（Python API）**：

```python
# LSDA-VWN 计算（自旋极化）
from atomscf.grid import radial_grid_exp_transformed
from atomscf.scf import run_lsda_vwn, SCFConfig

r, w, delta, Rp = radial_grid_exp_transformed(n=2000, rmin=0.0, rmax=150.0, total_span=8.0)
cfg = SCFConfig(
    Z=13, r=r, w=w,
    eig_solver="transformed",
    delta=delta, Rp=Rp,
    tol=1e-6
)
result = run_lsda_vwn(cfg, verbose=False)
print(f"总能量: {result.energies['E_total']:.6f} Ha")

# LDA 模式（赝势生成用，强制自旋对称）
cfg_lda = SCFConfig(
    Z=13, r=r, w=w,
    spin_mode="LDA",  # 强制 n_up = n_dn
    eig_solver="transformed",
    delta=delta, Rp=Rp
)
result_lda = run_lsda_vwn(cfg_lda)

# 导出供 AtomPPGen 使用
from atomscf.io import export_for_ppgen
export_for_ppgen(result_lda, cfg_lda, "data/al_ae_lda.json")
```

**Legacy 示例**（早期教学用，不推荐）：

```bash
# 氢原子（最简单）
PYTHONPATH=src python examples/run_h_1s_fd.py

# 碳原子 LSDA
PYTHONPATH=src python examples/run_c_lsda_vwn.py
```

---

## 精度与性能

**测试配置**：`--grid exp --solver transformed --n 2000 --total-span 8.0`

### 典型原子精度（LSDA-VWN vs NIST 非相对论参考）

| 原子 | Z | SCF 轮数 | 计算时间 | E_total (Ha) | NIST (Ha) | 相对误差 |
|------|---|---------|---------|--------------|-----------|----------|
| **H** | 1 | 33 | 46.5s | -0.42677 | -0.47867 | **10.84%** |
| **C** | 6 | 45 | 104.8s | -36.52579 | -37.47003 | **2.52%** |
| **Al** | 13 | 48 | 112.0s | -237.69169 | -241.32116 | **1.50%** |

---

### 测试

```bash
PYTHONPATH=src pytest -q
```

---

## 文档

**在线阅读**：[https://bud-primordium.github.io/AtomSCF/](https://bud-primordium.github.io/AtomSCF/)（推送到 main 分支后自动构建）

**本地构建**：
```bash
pip install -e .[docs]
cd docs
make html
make serve  # 访问 http://localhost:8000
```

**文档内容**：
- 算法原理推导（HF、DFT、数值方法）
- 基础使用教程（H/He/Li/C 原子示例）
- 完整 API 参考（自动生成）

---

## 开发规范

- **语言**：代码注释与文档使用中文
- **Docstring**：NumPy 风格
- **公式**：使用 LaTeX（`:math:` 或 `.. math::`）
- **测试**：pytest（`tests/` 目录）
- **Git 提交**：Conventional Commits 格式

---

## 依赖

- **运行时**：`numpy>=1.20`
- **开发**：`pytest>=7`
- **文档**：`sphinx>=7.0`, `sphinx-rtd-theme`, `myst-parser`

---

## 参考资料

- 变量变换方法：[Computational Physics Fall 2024, Assignment 7, Problem 2](https://github.com/bud-primordium/Computational-Physics-Fall-2024/tree/main/Assignment_7/Problem_2)
- 数值方法详解：`docs/source/algorithm/numerical_methods.rst`
- 文献列表：`docs/REFERENCES.md`

---

## 许可与致谢

本项目为教学课程作业，代码与文档供学习参考。核心变量变换技术源自上游计算物理作业实现。
