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
- 包含 $r=0$ 点（$j=0 \Rightarrow r=r_{\min}=0$）

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

**氢原子（最简单）**：
```bash
PYTHONPATH=src python examples/run_h_1s_fd.py
```

**碳原子 LSDA**：
```bash
PYTHONPATH=src python examples/run_c_lsda_vwn.py
```

**氦原子 RHF**：
```python
from atomscf.grid import radial_grid_linear
from atomscf.scf_hf import run_hf_scf, HFSCFGeneralConfig

r, w = radial_grid_linear(n=800, rmin=1e-6, rmax=50.0)
config = HFSCFGeneralConfig(
    Z=2, r=r, w=w,
    occ_by_l={0: [2.0]},  # 1s²
    eigs_per_l={0: 1},
    spin_mode='RHF',
    mix_alpha=0.5,
    tol=1e-6,
    maxiter=100
)
result = run_hf_scf(config)
print(f"总能量: {result.E_total:.6f} Ha")
```

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
