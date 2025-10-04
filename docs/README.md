# AtomSCF 文档

本目录包含 AtomSCF 项目的 Sphinx 文档源文件，面向**文档开发者和贡献者**。

用户文档请访问：[https://bud-primordium.github.io/AtomSCF/](https://bud-primordium.github.io/AtomSCF/)

---

## 快速构建

### 安装依赖

```bash
pip install -e .[docs]
```

### 本地构建

```bash
cd docs
make html       # 构建 HTML
make serve      # 启动本地服务器（http://localhost:8000）
make clean      # 清理构建产物
```

### 自动部署

推送到 `main` 分支后，GitHub Actions 自动构建并部署到 GitHub Pages。

---

## 文档结构

```
docs/
├── source/
│   ├── conf.py                # Sphinx 配置
│   ├── index.rst              # 文档主页
│   ├── introduction.rst       # 项目介绍
│   ├── algorithm/             # 算法原理推导
│   │   ├── index.rst
│   │   ├── hartree_fock.rst       # HF 方法（Slater 行列式、Fock 算符）
│   │   ├── density_functional.rst # DFT 方法（KS 方程、泛函）
│   │   └── numerical_methods.rst  # 数值方法（网格、FD、变量变换）
│   ├── api/                   # API 自动生成文档
│   │   ├── index.rst
│   │   └── generated/         # autosummary 生成的模块文档（git 忽略）
│   ├── examples/              # 使用教程
│   │   ├── index.rst
│   │   ├── basic_usage.rst    # 基础教程（H/He/Li）
│   │   └── atoms.rst          # 各种原子示例（H-Ne）
│   └── _static/               # 静态资源（主题、CSS）
├── build/                     # 构建输出（git 忽略）
└── Makefile                   # 构建脚本
```

---

## Sphinx 配置要点

**`conf.py` 关键配置**：

- **中文支持**：`language = 'zh_CN'`
- **扩展**：`autodoc`, `autosummary`, `napoleon`, `mathjax`, `myst_parser`
- **主题**：`sphinx_rtd_theme`（ReadTheDocs 风格）
- **自动生成**：`autosummary_generate = True`

**数学公式**：

- 块公式：`.. math::`
- 行内公式：`:math:\`...\``
- LaTeX 语法，双反斜杠转义（如 `\\frac`）

**API 文档**：

- `api/index.rst` 使用 `.. autosummary::` 自动生成模块文档
- 生成的文件位于 `api/generated/`（git 忽略）
- docstring 使用 NumPy 风格

---

## 文档编写规范

### reStructuredText 语法

**标题层级**：

```rst
一级标题
========

二级标题
--------

三级标题
~~~~~~~~
```

**代码块**：

```rst
.. code-block:: python

   from atomscf import run_hf_minimal
   result = run_hf_minimal(Z=1, r=r, w=w)
```

**数学公式**：

```rst
.. math::

   E = -\frac{Z^2}{2n^2}
```

**交叉引用**：

```rst
:ref:`算法原理 <algorithm-index>`
:func:`atomscf.scf_hf.run_hf_minimal`
:class:`atomscf.scf.SCFConfig`
```

### Docstring 规范

**NumPy 风格**：

```python
def function_name(param1: type1, param2: type2) -> return_type:
    r"""简短描述（一行）。

    详细描述（可选，多行）。

    Parameters
    ----------
    param1 : type1
        参数 1 的说明
    param2 : type2
        参数 2 的说明

    Returns
    -------
    return_type
        返回值说明

    Notes
    -----
    数学公式使用 LaTeX（注意原始字符串 r"""）：

    .. math::

        E = \sum_i n_i \epsilon_i

    Examples
    --------
    >>> result = function_name(1.0, 2.0)
    >>> print(result)
    3.0
    """
```

**关键点**：

- 使用 `r"""` 原始字符串（避免 Python 转义反斜杠）
- LaTeX 使用双反斜杠（`\\frac`, `\\sum`）
- 参数类型用 `:` 分隔（不是 Python 类型注解）

---

## 常见问题

### 构建警告过多

当前约 234 个警告，主要是：

- 重复对象描述（`duplicate object description`）
- 内联公式语法（`Inline interpreted text`）
- 引用未定义（`Unknown target name`）

**解决方向**：

- 减少 `autosummary` 和手动 `autoclass` 的重复
- 修正 docstring 中的 LaTeX 语法
- 统一引用标签（如 `[ExpGridTransform]`）

### 数学公式不显示

**原因**：Python 解释反斜杠

**解决**：docstring 使用 `r"""` 开头

### API 文档为空

**原因**：`autosummary_generate = True` 未生效或模块导入失败

**解决**：

1. 确保 `pip install -e .` 已安装包
2. 检查 `sys.path.insert(0, os.path.abspath("../../src"))`
3. 清理 `build/` 和 `api/generated/` 后重新构建

---

## 贡献指南

1. **修改源文件**：编辑 `source/*.rst` 文件
2. **本地验证**：`make html && make serve` 检查渲染效果
3. **提交前检查**：确保无新增 ERROR（WARNING 可接受）
4. **提交规范**：`docs: <具体修改>`（英文，Conventional Commits）

---

## 技术资源

- **Sphinx 官方文档**：https://www.sphinx-doc.org/
- **reStructuredText 参考**：https://docutils.sourceforge.io/rst.html
- **NumPy docstring 规范**：https://numpydoc.readthedocs.io/
- **ReadTheDocs 主题**：https://sphinx-rtd-theme.readthedocs.io/
