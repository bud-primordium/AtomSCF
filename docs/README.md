# AtomSCF 文档

本目录包含 AtomSCF 项目的 Sphinx 文档源文件。

## 构建文档

### 安装依赖

```bash
# 安装文档构建依赖
pip install -e .[docs]
```

### 构建 HTML 文档

```bash
cd docs
make html
```

生成的 HTML 文档位于 `docs/build/html/index.html`。

### 本地预览

```bash
make serve
```

然后在浏览器中访问 http://localhost:8000

### 清理构建

```bash
make clean
```

## 文档结构

```
docs/
├── source/
│   ├── index.rst              # 主页
│   ├── introduction.rst       # 项目介绍
│   ├── algorithm/             # 算法原理
│   │   ├── index.rst
│   │   ├── hartree_fock.rst   # HF 方法推导
│   │   ├── density_functional.rst  # DFT 方法
│   │   └── numerical_methods.rst   # 数值方法
│   ├── api/                   # API 参考
│   │   ├── index.rst
│   │   └── modules.rst        # 自动生成的模块文档
│   ├── examples/              # 使用示例
│   │   ├── index.rst
│   │   ├── basic_usage.rst
│   │   ├── atoms.rst
│   │   └── benchmarks.rst
│   └── conf.py                # Sphinx 配置
├── build/                     # 构建输出（git 忽略）
└── Makefile                   # 构建命令
```

## 文档编写规范

### reStructuredText 语法

Sphinx 使用 reStructuredText (RST) 格式，支持丰富的标记：

**标题**:
```rst
一级标题
========

二级标题
--------

三级标题
~~~~~~~~
```

**代码块**:
```rst
.. code-block:: python

   from atomscf import run_hf_minimal
   result = run_hf_minimal(Z=1, r=r, w=w)
```

**数学公式**:
```rst
.. math::

   E = -\frac{Z^2}{2n^2}
```

**交叉引用**:
```rst
参见 :ref:`算法原理 <algorithm-index>`
查看 :func:`atomscf.scf_hf.run_hf_minimal`
```

### Docstring 规范

代码中的 docstring 使用 NumPy 风格：

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
    数学公式使用 LaTeX:

    .. math::

        E = \sum_i n_i \epsilon_i

    Examples
    --------
    >>> result = function_name(1.0, 2.0)
    >>> print(result)
    3.0
    """
```

## 注意事项

1. **中文支持**: `conf.py` 已配置 `language = 'zh_CN'`
2. **数学公式**: 使用 MathJax 渲染，支持 LaTeX 语法
3. **API 自动生成**: 使用 `sphinx.ext.autodoc` 从代码提取文档
4. **Markdown 支持**: 通过 `myst-parser` 扩展支持 Markdown 文件

## 常见问题

### 构建警告

如果看到 "WARNING: document isn't included in any toctree"，检查 `index.rst` 的 `toctree` 指令是否包含该文件。

### 数学公式不显示

确保使用原始字符串 `r"""` 开头的 docstring，避免 Python 解释反斜杠。

### 中文显示问题

PDF 生成需要安装中文字体（参见 `conf.py` 的 `latex_elements` 配置）。
