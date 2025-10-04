# AtomSCF 文档

本目录包含 AtomSCF 项目的 Sphinx 文档源文件。

## 重要说明

**AtomSCF 是一个专门用于球对称体系的径向量子化学计算工具**，请在使用前了解以下限制与特性。

### 适用范围

- **单原子体系**：H, He, Li, C, N, O 等独立原子
- **球对称问题**：具有中心对称势场的体系
- **不支持分子**：不适用于多中心问题（如 H₂, H₂O 等）
- **不支持固体**：不适用于周期性边界条件

### 核心方法特点

#### 1. 径向一维离散化

将三维 Schrödinger 方程约化为径向方程：

$$
\left[-\frac{1}{2}\frac{d^2}{dr^2} + \frac{\ell(\ell+1)}{2r^2} + V_{\text{eff}}(r)\right] u_{n\ell}(r) = \varepsilon_{n\ell} u_{n\ell}(r)
$$

其中：

- 径向波函数 $u_{n\ell}(r) = r \cdot R_{n\ell}(r)$ 满足边界条件 $u(0) = u(\infty) = 0$
- 角动量 $\ell$ 作为量子数，球谐函数 $Y_\ell^m$ 已知
- 仅沿半径方向进行数值离散

#### 2. 实空间方法（非基组展开）

本项目采用**实空间有限差分方法**，区别于传统量子化学软件的 Gaussian 基组或平面波展开。

**标准有限差分**：

- **FD2**：二阶中心差分，支持非均匀网格
- **FD5**：五阶中心差分，需等间距网格，精度 $O(h^4)$

**变量变换方法**（核心创新）：

基于指数网格的变量变换技术，源自 [Computational Physics Fall 2024, Assignment 7, Problem 2](https://github.com/bud-primordium/Computational-Physics-Fall-2024/tree/main/Assignment_7/Problem_2)。

**网格定义**：

$$
r(j) = R_p (\exp(j\delta) - 1) + r_{\min}, \quad j = 0, 1, \dots, N-1
$$

**变量代换**：

$$
u(j) = v(j) \cdot \exp\left(\frac{j\delta}{2}\right)
$$

**变换后的离散方程**（在索引空间 $j$ 中）：

$$
v''(j) - \frac{\delta^2}{4}v(j) = 2R_p^2\delta^2 \exp(2j\delta) \left[\varepsilon - V_{\text{eff}}(r(j))\right] v(j)
$$

变换的核心目的是**消除一阶导数项**，使得离散后的 Hamiltonian 矩阵对称，可用高效的 `eigh()` 求解广义特征值问题。

**关键优势**：

- 精度提升约 7 倍（相比线性网格）
- 速度提升约 3 倍（网格点数减少）
- 自然包含原点 $r_0 = 0$（物理正确）
- 对称矩阵 + 广义特征值问题数值稳定

**参数选择**：

- $\delta \approx 0.01 \sim 0.05$（控制网格疏密）
- $R_p \approx Z/4$（$Z$ 为原子序数，优化波函数衰减匹配）

#### 3. 交换-关联处理

**Hartree-Fock**：

- 精确交换通过 Slater 径向积分 $R^k(r) = Y^k(r) + Z^k(r)$
- 角动量耦合通过 Wigner 3-j 符号计算
- 支持 RHF（限制性，闭壳层）和 UHF（非限制性，开壳层）

**DFT-LSDA**：

- Dirac 交换：$\varepsilon_x = -\frac{3}{4}\left(\frac{3\rho}{\pi}\right)^{1/3}$
- 关联泛函：PZ81（Perdew-Zunger 1981）或 VWN（Vosko-Wilk-Nusair）

---

## 项目源码结构

```
src/atomscf/
├── grid.py          # 径向网格生成（线性/对数/指数变换/混合）
├── operator.py      # 径向 Schrödinger 方程求解器（FD2/FD5/变换/Numerov）
├── scf.py          # DFT 自洽场框架（LSDA-PZ81/VWN）
├── scf_hf.py       # HF 自洽场框架（RHF/UHF，s/p/d 轨道）
├── hartree.py      # Hartree 势计算（泊松方程求解）
├── occupations.py  # 原子电子占据方案（Z=1-18，Hund 规则）
├── numerov.py      # Numerov 方法（对数网格边界值问题）
├── utils.py        # 积分、归一化、梯形权重等工具
├── io.py          # CSV/JSON 数据导出
├── hf/
│   ├── slater.py   # Slater 径向积分（Y^k/Z^k 两段累积）
│   ├── angular.py  # Wigner 3-j 符号、角动量耦合因子
│   └── exchange.py # HF 交换算符（s 轨道、通用多 l 通道）
└── xc/
    ├── lda.py      # Dirac 交换 + PZ81 关联
    ├── vwn.py      # VWN 关联泛函
    └── constants.py
```

**核心工作流**：

1. `grid.py` 生成网格 → 2. `operator.py` 构造 Hamiltonian 矩阵 → 3. `scf.py` 或 `scf_hf.py` 自洽迭代 → 4. `io.py` 导出结果

---

## 文档构建

### 安装依赖

```bash
pip install -e .[docs]
```

### 本地构建

```bash
cd docs
make html
```

生成的 HTML 文档位于 `docs/build/index.html`。

### 本地预览

```bash
make serve
```

访问 http://localhost:8000

### 在线阅读

项目文档已部署到 GitHub Pages，推送到 `main` 分支后自动构建：

- 文档地址：`https://bud-primordium.github.io/AtomSCF/`（待部署后更新）

---

## 文档结构

```
docs/
├── source/
│   ├── index.rst              # 文档主页
│   ├── introduction.rst       # 项目介绍
│   ├── algorithm/             # 算法原理（HF/DFT/数值方法推导）
│   ├── api/                   # API 自动生成文档
│   │   ├── index.rst
│   │   └── generated/         # autosummary 生成的模块文档
│   ├── examples/              # 使用教程（H/He/Li/C 等原子示例）
│   └── conf.py                # Sphinx 配置
├── build/                     # 构建输出（git 忽略）
└── Makefile                   # 构建脚本
```

---

## 文档状态

**当前进度**：

- 算法原理推导（HF、DFT、数值方法）已完成
- 基础使用教程（H/He/Li/C 原子示例）已完成
- API 自动生成文档已完成
- 高级主题（收敛技巧、网格优化）待补充
- 基准测试与文献对比待完善

文档持续改进中，如遇问题欢迎提交 Issue 或 Pull Request。
