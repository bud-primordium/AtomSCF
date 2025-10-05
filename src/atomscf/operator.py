from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from .grid import trapezoid_weights
from .utils import normalize_radial_u

"""径向薛定谔方程求解器模块。

本模块提供多种求解径向薛定谔方程束缚态的求解器，适用于不同的网格类型和精度需求。

求解器选择指南
==============

本模块提供以下四种求解器，按推荐优先级排序：

1. **solve_bound_states_transformed** (变量变换方法)
   - 适用网格：指数网格（需配合 radial_grid_exp_transformed 使用）
   - 精度：最高（相比 FD 方法提升 ~7 倍）
   - 速度：快（相比 FD5-aux 提升 ~4 倍）
   - 限制：仅适用于特定网格类型，需要提供 delta 和 Rp 参数
   - 推荐场景：生产环境计算，需要高精度和高效率时

2. **solve_bound_states_fd5_auxlinear** (插值到均匀网格 + 5点有限差分)
   - 适用网格：任意非均匀网格
   - 精度：高（5 点差分格式）
   - 速度：中等（需要插值开销）
   - 限制：插值会引入额外误差
   - 推荐场景：使用非均匀网格但需要较高精度时的通用方案

3. **solve_bound_states_fd5** (直接 5点有限差分)
   - 适用网格：线性等距网格
   - 精度：高（5 点差分格式）
   - 速度：快（无插值开销）
   - 限制：仅限均匀网格
   - 推荐场景：使用均匀网格时的高效选择

4. **solve_bound_states_fd** (基础有限差分)
   - 适用网格：任意网格（自动检测并适配）
   - 精度：基础（2 点差分格式）
   - 速度：快
   - 限制：精度较低
   - 推荐场景：快速原型验证，或作为其他方法的参考基准

使用示例
========

使用变量变换方法（推荐）::

    from atomscf.grid import radial_grid_exp_transformed
    from atomscf.operator import solve_bound_states_transformed

    # 生成指数网格
    r, w, delta, Rp = radial_grid_exp_transformed(n=800, rmin=1e-6, rmax=50.0)

    # 求解
    eps, U = solve_bound_states_transformed(r, l=0, v_of_r=V_eff,
                                            delta=delta, Rp=Rp, k=3)

使用通用 FD5-aux 方法::

    from atomscf.grid import radial_grid_log
    from atomscf.operator import solve_bound_states_fd5_auxlinear

    # 生成对数网格
    r, w = radial_grid_log(n=800, rmin=1e-6, rmax=50.0)

    # 求解（自动插值到均匀网格）
    eps, U = solve_bound_states_fd5_auxlinear(r, l=0, v_of_r=V_eff, k=3)

注意事项
========

- 所有求解器返回归一化的径向波函数 u(r)，满足 ∫ |u(r)|² dr = 1
- 能量按升序排列（束缚态从浅到深）
- 对于高激发态，可能需要调整网格参数或能量搜索范围
"""

__all__ = [
    "radial_hamiltonian_matrix",
    "solve_bound_states_fd",
    "radial_hamiltonian_matrix_linear_fd5",
    "solve_bound_states_fd5",
    "solve_bound_states_fd5_auxlinear",
    "solve_bound_states_transformed",
    "build_transformed_hamiltonian",  # 新增：HF 复用
]


def _second_derivative_tridiag_nonuniform(r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""构造非均匀网格上二阶导数算子的三对角表示（内部点）。

    对于非均匀一维网格 :math:`(r_0,\dots,r_{N-1})`，在内部点 :math:`i=1,\dots,N-2` 上，
    二阶导数可用以下一致格式近似：

    .. math::
        \left.\frac{d^2u}{dr^2}\right|_{r_i}
        \approx \frac{2}{h_{i-1}+h_i}\left[\frac{u_{i+1}-u_i}{h_i} - \frac{u_i-u_{i-1}}{h_{i-1}}\right],

    其中 :math:`h_{i-1}=r_i-r_{i-1},\ h_i=r_{i+1}-r_i`。

    本函数返回内部点上的三对角系数数组 :math:`(a, b, c)`，对应 :math:`u_{i-1}, u_i, u_{i+1}` 的系数，
    以及内部坐标数组 :math:`r_{\text{inner}}=(r_1,\dots,r_{N-2})`。

    Parameters
    ----------
    r : numpy.ndarray
        单调递增的一维网格坐标。

    Returns
    -------
    a : numpy.ndarray
        下对角系数数组，长度 :math:`N-2`，首元素对物理上无意义（可忽略或视为填充）。
    b : numpy.ndarray
        对角系数数组，长度 :math:`N-2`。
    c : numpy.ndarray
        上对角系数数组，长度 :math:`N-2`，末元素对物理上无意义（可忽略或视为填充）。
    r_inner : numpy.ndarray
        内部网格坐标 :math:`(r_1,\dots,r_{N-2})`。

    Notes
    -----
    - 该离散对应 Dirichlet 边界条件 :math:`u(r_0)=u(r_{N-1})=0`，因此仅对内部点建立方程。
    - 返回系数对应算子 :math:`\frac{d^2}{dr^2}`，后续 Hamiltonian 需乘以 :math:`-\tfrac{1}{2}`。
    """
    if r.ndim != 1:
        raise ValueError("r 必须是一维数组")
    if np.any(np.diff(r) <= 0):
        raise ValueError("r 必须严格单调递增")
    n = r.size
    if n < 3:
        raise ValueError("至少需要 3 个网格点以建立内部二阶导数")

    r_inner = r[1:-1]
    h_left = np.diff(r[:-1])  # r[i] - r[i-1], len = n-2 (for i=1..n-2)
    h_right = np.diff(r[1:])  # r[i+1] - r[i], len = n-2 (for i=1..n-2)

    a = np.empty(n - 2, dtype=float)
    b = np.empty(n - 2, dtype=float)
    c = np.empty(n - 2, dtype=float)

    denom = h_left + h_right
    a[:] = 2.0 / (h_left * denom)
    c[:] = 2.0 / (h_right * denom)
    b[:] = -2.0 / (h_left * h_right)

    return a, b, c, r_inner


def radial_hamiltonian_matrix(
    r: np.ndarray,
    l: int,
    v_of_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""构造径向 Hamiltonian（内部点）并返回矩阵与内部网格坐标。

    径向方程（原子单位）在 :math:`u(r)` 表示下为：

    .. math::
        \left[-\frac{1}{2}\frac{d^2}{dr^2}+\frac{\ell(\ell+1)}{2r^2}+v(r)\right]u(r)=\varepsilon\,u(r),

    本函数在非均匀网格上采用二阶差分对 :math:`\tfrac{d^2}{dr^2}` 离散，并以 Dirichlet 边界
    :math:`u(r_0)=u(r_{N-1})=0` 构造内部点的对称三对角矩阵。

    Parameters
    ----------
    r : numpy.ndarray
        单调递增的径向网格 :math:`(r_0,\dots,r_{N-1})`。
    l : int
        角动量量子数 :math:`\ell`。
    v_of_r : numpy.ndarray
        势能数组 :math:`v(r_i)`，长度与 :data:`r` 一致。

    Returns
    -------
    H : numpy.ndarray
        内部点上的 Hamiltonian 矩阵，形状 :math:`(N-2, N-2)`。
    r_inner : numpy.ndarray
        内部网格坐标 :math:`(r_1,\dots,r_{N-2})`。

    Notes
    -----
    - 矩阵是对称实矩阵，适用于标准本征求解器。
    - 若需得到完整长度的 :math:`u(r)`，可在内部解向量两端补零以满足 Dirichlet 边界。
    """
    if v_of_r.shape != r.shape:
        raise ValueError("v_of_r 的形状必须与 r 相同")
    if l < 0:
        raise ValueError("l 必须为非负整数")

    a, b, c, r_inner = _second_derivative_tridiag_nonuniform(r)
    n_in = r_inner.size

    # 组装二阶导数三对角矩阵 D2
    D2 = np.zeros((n_in, n_in), dtype=float)
    # 对角
    np.fill_diagonal(D2, b)
    # 上/下对角
    idx = np.arange(n_in - 1)
    D2[idx, idx + 1] = c[:-1]
    D2[idx + 1, idx] = a[1:]

    # 动能算子：T = -1/2 D2
    T = -0.5 * D2

    # 势能 + 离心项（内部点）
    v_inner = v_of_r[1:-1]
    lterm = 0.5 * l * (l + 1) / (r_inner**2)
    V = np.diag(v_inner + lterm)

    H = T + V
    return H, r_inner


def solve_bound_states_fd(
    r: np.ndarray,
    l: int,
    v_of_r: np.ndarray,
    k: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    r"""基于有限差分 Hamiltonian 的径向束缚态求解（取低端 :math:`k` 个本征对）。

    该方法将径向 Hamiltonian 离散为内部点的对称矩阵，调用密集线性代数本征求解。
    适合教学验证（例如氢样势下的 1s 能级），在网格较大时可能较慢。

    Parameters
    ----------
    r : numpy.ndarray
        单调递增的径向网格 :math:`(r_0,\dots,r_{N-1})`。
    l : int
        角动量量子数 :math:`\ell`。
    v_of_r : numpy.ndarray
        势能数组 :math:`v(r_i)`，长度与 :data:`r` 一致。
    k : int, optional
        返回最低能的本征态个数（默认 4）。

    Returns
    -------
    eps : numpy.ndarray
        低端 :math:`k` 个本征值（按升序）。
    U : numpy.ndarray
        对应的径向函数矩阵 :math:`U`，形状 :math:`(k, N)`，已在 :math:`[r_0,r_{N-1}]` 上按
        :math:`\int u^2\,dr=1` 归一，并在两端补零以满足 Dirichlet 边界。

    Notes
    -----
    - 实际内部计算在 :math:`(r_1,\dots,r_{N-2})` 上进行，端点边界条件 :math:`u=0`。
    - 若只需氢样势 1s 态，建议选择较大的 :math:`r_\max`（如 50–100）与足够细的网格以降低边界误差。
    - 使用 scipy.linalg.eigh 的 subset_by_index 只求前 k 个本征值以提升性能。
    """
    H, r_inner = radial_hamiltonian_matrix(r, l, v_of_r)
    k_actual = min(k, H.shape[0])
    # 只求前 k_actual 个最低本征值（参考 dftatom 和 tinydft）
    e_vals, U_inner = eigh(H, subset_by_index=(0, k_actual - 1))

    eps = e_vals
    U_out = np.zeros((k_actual, r.size), dtype=float)
    w = trapezoid_weights(r)

    for j in range(k_actual):
        u = np.zeros_like(r)
        u[1:-1] = U_inner[:, j]
        u, _ = normalize_radial_u(u, r, w)
        U_out[j] = u

    return eps, U_out


def _is_uniform_linear_grid(r: np.ndarray, tol: float = 1e-12) -> tuple[bool, float]:
    """判断是否为等间距线性网格，并返回步长。

    Parameters
    ----------
    r : numpy.ndarray
        径向网格。
    tol : float
        判断等间距的容差。

    Returns
    -------
    is_uniform : bool
        是否等间距。
    h : float
        网格步长（若非等间距则返回 0）。
    """
    dr = np.diff(r)
    if np.allclose(dr, dr[0], atol=tol, rtol=0):
        return True, float(dr[0])
    return False, 0.0


def radial_hamiltonian_matrix_linear_fd5(
    r: np.ndarray,
    l: int,
    v_of_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""在等间距线性网格上构造五点有限差分的径向 Hamiltonian。

    使用五点中心差分格式近似二阶导数：

    .. math::
        \frac{d^2 u}{dr^2}\bigg|_{r_i} \approx \frac{-u_{i+2} + 16u_{i+1} - 30u_i + 16u_{i-1} - u_{i-2}}{12h^2}.

    精度：:math:`\mathcal{O}(h^4)`，在等间距网格上比二点/三点差分精度更高。

    Parameters
    ----------
    r : numpy.ndarray
        **必须为等间距** 线性网格 :math:`(r_0,\dots,r_{N-1})`，否则抛出异常。
    l : int
        角动量量子数 :math:`\ell`。
    v_of_r : numpy.ndarray
        势能数组 :math:`v(r_i)`，长度与 :data:`r` 一致。

    Returns
    -------
    H : numpy.ndarray
        内部点上的 Hamiltonian 矩阵，形状 :math:`(N-4, N-4)`。由于五点模板需要两侧各两点，
        实际对应网格索引 :math:`i=2,\dots,N-3`，即去掉首尾各两个点。
    r_inner : numpy.ndarray
        内部网格坐标 :math:`(r_2,\dots,r_{N-3})`。
    """
    is_uniform, h = _is_uniform_linear_grid(r)
    if not is_uniform:
        raise ValueError("radial_hamiltonian_matrix_linear_fd5 要求 r 为等间距网格，请考虑插值或使用二阶 FD")

    if l < 0:
        raise ValueError("l 必须为非负整数")
    if v_of_r.shape != r.shape:
        raise ValueError("v_of_r 的形状必须与 r 相同")

    n = r.size
    if n < 5:
        raise ValueError("五点差分至少需要 5 个网格点")

    r_inner = r[2:-2]
    n_in = r_inner.size
    h2 = h * h

    # 五点差分系数（标准化到步长 h=1）
    # -u_{i-2} + 16*u_{i-1} - 30*u_i + 16*u_{i+1} - u_{i+2}
    # 对应二阶导数的 12 倍（除以 12h^2 后得到真实二阶导数）
    coef_main = -30.0 / (12.0 * h2)
    coef_near = 16.0 / (12.0 * h2)
    coef_far = -1.0 / (12.0 * h2)

    # 构造五对角矩阵（动能项）
    T = np.zeros((n_in, n_in), dtype=float)
    idx = np.arange(n_in)
    # 主对角
    np.fill_diagonal(T, coef_main)
    # ±1 对角
    T[idx[:-1], idx[:-1] + 1] = coef_near
    T[idx[:-1] + 1, idx[:-1]] = coef_near
    # ±2 对角
    T[idx[:-2], idx[:-2] + 2] = coef_far
    T[idx[:-2] + 2, idx[:-2]] = coef_far

    T *= -0.5

    # 势能 + 离心项（内部点）
    v_inner = v_of_r[2:-2]
    lterm = 0.5 * l * (l + 1) / (r_inner**2)
    V = np.diag(v_inner + lterm)

    H = T + V
    return H, r_inner


def solve_bound_states_fd5(
    r: np.ndarray,
    l: int,
    v_of_r: np.ndarray,
    k: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    r"""在等间距线性网格上使用五点有限差分求解径向束缚态（取低端 :math:`k` 个本征对）。

    相比二阶差分（`solve_bound_states_fd`），五点格式精度更高（:math:`\mathcal{O}(h^4)`），
    但要求网格**必须为等间距**，且去掉首尾各两个点以满足五点模板要求。

    Parameters
    ----------
    r : numpy.ndarray
        **必须为等间距** 线性网格 :math:`(r_0,\dots,r_{N-1})`。
    l : int
        角动量量子数 :math:`\ell`。
    v_of_r : numpy.ndarray
        势能数组 :math:`v(r_i)`，长度与 :data:`r` 一致。
    k : int, optional
        返回最低能的本征态个数（默认 4）。

    Returns
    -------
    eps : numpy.ndarray
        低端 :math:`k` 个本征值（按升序）。
    U : numpy.ndarray
        对应的径向函数矩阵 :math:`U`，形状 :math:`(k, N)`，已在 :math:`[r_0,r_{N-1}]` 上按
        :math:`\int u^2\,dr=1` 归一，并在两端补零（首尾各两个点为 0）。

    Notes
    -----
    - 实际内部计算在 :math:`(r_2,\dots,r_{N-3})` 上进行，首尾各两点作为边界条件设为 0。
    - 使用 scipy.linalg.eigh 的 subset_by_index 只求前 k 个本征值以提升性能。
    """
    H, r_inner = radial_hamiltonian_matrix_linear_fd5(r, l, v_of_r)
    k_actual = min(k, H.shape[0])
    # 只求前 k_actual 个最低本征值
    e_vals, U_inner = eigh(H, subset_by_index=(0, k_actual - 1))

    eps = e_vals
    U_out = np.zeros((k_actual, r.size), dtype=float)
    w = trapezoid_weights(r)

    for j in range(k_actual):
        u = np.zeros_like(r)
        u[2:-2] = U_inner[:, j]  # 首尾各两个点为 0
        u, _ = normalize_radial_u(u, r, w)
        U_out[j] = u

    return eps, U_out


def solve_bound_states_fd5_auxlinear(
    r: np.ndarray,
    l: int,
    v_of_r: np.ndarray,
    k: int = 4,
    n_aux: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""在非等间距网格上，使用辅助等间距线性网格（FD5）求解本征，再插值回原网格。

    步骤：
    1) 构造线性等间距网格 :math:`r_{\text{lin}} \in [r_0, r_{N-1}]`，点数 :math:`n_{\text{aux}}`（默认与原网格同级别但限制上限2500以平衡精度和速度）。
    2) 将势 :math:`v(r)` 插值到线性网格上；
    3) 用五点格式 `solve_bound_states_fd5` 在线性网格上求低端本征；
    4) 将解插值回原网格，并在原网格上归一化（`∫ u^2 dr = 1`）。

    性能优化：使用 scipy.linalg.eigh subset_by_index 只求前 k 个本征值，即使 n_aux 较大也能快速求解。
    """
    if n_aux is None:
        n_aux = min(max(len(r), 1000), 2500)  # 提高上限以减少插值误差
    rmin, rmax = float(r[0]), float(r[-1])
    r_lin = np.linspace(rmin, rmax, n_aux)
    v_lin = np.interp(r_lin, r, v_of_r)
    eps, U_lin = solve_bound_states_fd5(r_lin, l=l, v_of_r=v_lin, k=k)

    U_out = np.zeros((len(eps), r.size), dtype=float)
    w = trapezoid_weights(r)
    for j in range(len(eps)):
        u_lin = U_lin[j]
        u = np.interp(r, r_lin, u_lin)
        u, _ = normalize_radial_u(u, r, w)
        U_out[j] = u

    return eps, U_out


def solve_bound_states_transformed(
    r: np.ndarray,
    l: int,
    v_of_r: np.ndarray,
    delta: float,
    Rp: float,
    k: int = 4,
    use_sparse: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    r"""使用变量变换方法求解径向束缚态（适用于指数网格）。

    方法基于文献 [1]_：

    网格：
        :math:`r(j) = R_p(\exp(j\delta) - 1) + r_{\min}, \quad j=0,1,\ldots,j_{\max}`

    变量变换：
        :math:`u(j) = v(j) \exp(j\delta/2)`

    变换后的方程（没有一阶导数项）：
        :math:`v''(j) - \frac{\delta^2}{4}v(j) = 2R_p^2\delta^2 \exp(2j\delta)(E - V_{\text{eff}}(r(j)))v(j)`

    有限差分离散：
        :math:`v(j+1) - 2v(j) + v(j-1) - \frac{\delta^2}{4}v(j) = 2R_p^2\delta^2 \exp(2j\delta)(E - V_{\text{eff}}(j))v(j)`

    广义特征值问题：
        :math:`H v = E B v`

    其中：
        - :math:`H[i,i] = 2 + \frac{\delta^2}{4} + \text{exp\_factor} \cdot V_{\text{eff}}(i+1)`
        - :math:`H[i,i\pm1] = -1`
        - :math:`B[i,i] = 2\delta^2 R_p^2 \exp(2\delta(i+1))`

    **关键处理**：去掉 j=0 点避免奇异性（参考 [1]_ 第54行）

    Parameters
    ----------
    r : numpy.ndarray
        指数网格，由 `radial_grid_exp_transformed` 生成，满足 r[j] = Rp*(exp(j*delta)-1) + rmin。
    l : int
        角动量量子数。
    v_of_r : numpy.ndarray
        势能数组（包含核势能），长度与 r 一致。
    delta : float
        网格参数 δ（从 `radial_grid_exp_transformed` 返回）。
    Rp : float
        网格参数 R_p（从 `radial_grid_exp_transformed` 返回）。
    k : int, optional
        返回最低能的本征态个数（默认 4）。
    use_sparse : bool, optional
        是否使用稀疏矩阵求解器（默认 True，适用于大型矩阵）。

    Returns
    -------
    eps : numpy.ndarray
        低端 k 个本征值（按升序）。
    U : numpy.ndarray
        对应的径向波函数矩阵，形状 (k, N)，已归一化。

    Notes
    -----
    - 去掉 j=0 点后，实际求解的网格索引为 j=1,2,...,j_max
    - v[0] 在边界条件下为 0（不参与求解）
    - 求解后变换回 u = v * exp(j*delta/2) 并归一化

    References
    ----------
    .. [ExpGridTransform] 指数网格变量变换方法
       来源：Computational Physics Fall 2024, Assignment 7, Problem 2
       https://github.com/bud-primordium/Computational-Physics-Fall-2024/tree/main/Assignment_7/Problem_2
       problem_2.tex 第155-235行（矩阵元推导）
    """
    if v_of_r.shape != r.shape:
        raise ValueError("v_of_r 的形状必须与 r 相同")
    if l < 0:
        raise ValueError("l 必须为非负整数")

    n = r.size
    if n < 3:
        raise ValueError("至少需要 3 个网格点")

    # 去掉 j=0 点（近核奇异）
    # 实际求解的索引：j_actual = 1, 2, ..., n-1 对应矩阵索引 i = 0, 1, ..., n-2
    n_reduced = n - 1

    # 计算有效势能（包含离心项）
    V_eff = v_of_r + 0.5 * l * (l + 1) / (r**2 + 1e-30)

    # 构造 Hamiltonian 矩阵 H 和质量矩阵 B
    # 参考 problem_2.tex 第155-183行

    # 规模阈值：实际禁用稀疏（密集+白化已足够快）
    # 稀疏求解器（ARPACK eigsh）对该问题收敛极慢，即使白化后仍不如密集
    use_sparse_actual = use_sparse and (n_reduced > 10000)

    if use_sparse_actual:
        from scipy.sparse import diags
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix
        import numpy as _np
        import os
        import time as _time

        debug_solver = os.environ.get("ATOMSCF_DEBUG_SOLVER", "0") == "1"

        # 组装 H 的对角（三对角 Off=-1）
        diag_H = _np.zeros(n_reduced)
        for i in range(n_reduced):
            j_actual = i + 1  # 实际网格索引（跳过 j=0）
            exp_factor = 2.0 * delta**2 * Rp**2 * _np.exp(2.0 * delta * j_actual)
            diag_H[i] = 2.0 + delta**2 / 4.0 + exp_factor * V_eff[j_actual]

        # B 的对角
        diag_B = _np.zeros(n_reduced)
        for i in range(n_reduced):
            j_actual = i + 1
            diag_B[i] = 2.0 * delta**2 * Rp**2 * _np.exp(2.0 * delta * j_actual)

        # 标准化白化：C = B^{-1/2} H B^{-1/2}（保持三对角）
        inv_sqrt_B = 1.0 / _np.sqrt(_np.maximum(diag_B, 1e-300))
        # C 主对角
        diag_C = diag_H * (inv_sqrt_B ** 2)
        # C 上/下对角：-inv[i]*inv[i+1]
        off_C = -inv_sqrt_B[:-1] * inv_sqrt_B[1:]

        C = diags([off_C, diag_C, off_C], offsets=[-1, 0, 1], shape=(n_reduced, n_reduced), format="csr")

        # eigsh 参数（放宽以提高收敛性）
        # 对于 n>2000 的大网格，优先保证收敛而非极致精度
        k_actual = min(k, n_reduced)
        # ncv 需要足够大：至少 4k+10，对大矩阵用 min(8k, 100)
        ncv = min(max(8 * k_actual, 60), n_reduced - 1, 150)
        tol = 1e-7  # 放宽容差
        maxiter = 3000  # 允许更多迭代

        if debug_solver:
            print(f"    [DEBUG] l={l}, n_reduced={n_reduced}, k={k_actual}, ncv={ncv}, 开始 eigsh...")
            t0 = _time.time()

        try:
            e_vals, Y = eigsh(C, k=k_actual, which="SA", ncv=ncv, tol=tol, maxiter=maxiter)
            if debug_solver:
                print(f"    [DEBUG] eigsh 成功，用时 {_time.time()-t0:.4f}s")
        except Exception as e1:
            # 重试：进一步放宽参数
            if debug_solver:
                print(f"    [DEBUG] 首次 eigsh 失败: {e1}, 尝试重试...")
            try:
                ncv2 = min(max(10 * k_actual, 100), n_reduced - 1, 200)
                if debug_solver:
                    t1 = _time.time()
                e_vals, Y = eigsh(C, k=k_actual, which="SA", ncv=ncv2, tol=1e-6, maxiter=5000)
                if debug_solver:
                    print(f"    [DEBUG] 重试成功（ncv={ncv2}），用时 {_time.time()-t1:.4f}s")
            except Exception as e2:
                # 稀疏失败：回退到密集广义特征值
                if debug_solver:
                    print(f"    [DEBUG] 重试失败: {e2}, 回退到密集求解...")
                    t2 = _time.time()
                H = _np.zeros((n_reduced, n_reduced), dtype=float)
                B = _np.zeros((n_reduced, n_reduced), dtype=float)
                for i in range(n_reduced):
                    if i > 0:
                        H[i, i - 1] = -1.0
                    H[i, i] = diag_H[i]
                    if i < n_reduced - 1:
                        H[i, i + 1] = -1.0
                    B[i, i] = diag_B[i]
                from scipy.linalg import eigh as _eigh
                e_vals, V_inner = _eigh(H, B, subset_by_index=(0, k_actual - 1))
                if debug_solver:
                    print(f"    [DEBUG] 密集求解成功，用时 {_time.time()-t2:.4f}s")
                # 归一化到 v 空间（密集回退已经是 v）
                Y = V_inner

        # 将 y 变回 v：v = B^{-1/2} y
        if Y.ndim == 1:
            Y = Y[:, None]
        V_inner = (inv_sqrt_B[:, None] * Y)

    else:
        # 密集矩阵求解（也使用白化避免病态）
        import numpy as _np

        # 组装 diag_H 与 diag_B
        diag_H = _np.zeros(n_reduced)
        diag_B = _np.zeros(n_reduced)
        for i in range(n_reduced):
            j_actual = i + 1
            exp_factor = 2.0 * delta**2 * Rp**2 * _np.exp(2.0 * delta * j_actual)
            diag_H[i] = 2.0 + delta**2 / 4.0 + exp_factor * V_eff[j_actual]
            diag_B[i] = exp_factor

        # 白化：C = B^{-1/2} H B^{-1/2}（保持三对角）
        inv_sqrt_B = 1.0 / _np.sqrt(_np.maximum(diag_B, 1e-300))
        diag_C = diag_H * (inv_sqrt_B ** 2)
        off_C = -inv_sqrt_B[:-1] * inv_sqrt_B[1:]

        # 构建密集三对角矩阵 C
        C = _np.zeros((n_reduced, n_reduced), dtype=float)
        for i in range(n_reduced):
            if i > 0:
                C[i, i-1] = off_C[i-1]
            C[i, i] = diag_C[i]
            if i < n_reduced - 1:
                C[i, i+1] = off_C[i]

        # 求解标准特征值问题 C y = E y
        k_actual = min(k, n_reduced)
        from scipy.linalg import eigh as _eigh
        e_vals, Y = _eigh(C, subset_by_index=(0, k_actual - 1))

        # 将 y 变回 v：v = B^{-1/2} y
        if Y.ndim == 1:
            Y = Y[:, None]
        V_inner = (inv_sqrt_B[:, None] * Y)

    # 变换回 u = v * exp(j*delta/2)
    eps = e_vals
    U_out = np.zeros((k_actual, r.size), dtype=float)
    w = trapezoid_weights(r)

    for idx in range(k_actual):
        v_reduced = V_inner[:, idx]  # 长度 n_reduced

        # 还原完整的 v（j=0 处为 0）
        v_full = np.zeros(n)
        v_full[1:] = v_reduced  # v[0] = 0, v[1:] = v_reduced

        # 变换回 u = v * exp(j*delta/2)
        j = np.arange(n)
        u = v_full * np.exp(j * delta / 2.0)

        # 归一化
        u, _ = normalize_radial_u(u, r, w)
        U_out[idx] = u

    return eps, U_out


def build_transformed_hamiltonian(
    r: np.ndarray,
    l: int,
    v_of_r: np.ndarray,
    delta: float,
    Rp: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""构造变量变换的 Hamiltonian 和质量矩阵（HF 复用）。

    返回广义特征值问题的矩阵形式（但不求解）：

        H v = E B v

    其中 v 是变换后的波函数。

    Parameters
    ----------
    r : np.ndarray
        指数变换网格
    l : int
        角动量量子数
    v_of_r : np.ndarray
        势能（含外势 + Hartree）
    delta : float
        网格参数 δ
    Rp : float
        网格参数 R_p

    Returns
    -------
    H : np.ndarray
        Hamiltonian 矩阵（去掉 j=0 后的尺寸 n-1 × n-1）
    B : np.ndarray
        质量矩阵（对角，尺寸 n-1 × n-1）
    r_inner : np.ndarray
        内部网格点（去掉 j=0，长度 n-1）

    Notes
    -----
    该函数抽取自 solve_bound_states_transformed，
    用于 HF 中构造局域 Hamiltonian 部分。

    交换矩阵 K 仍在原始 u(r) 空间构造，
    需要在 v 和 u 之间转换：u(j) = v(j) * exp(j*delta/2)
    """
    if v_of_r.shape != r.shape:
        raise ValueError("v_of_r 的形状必须与 r 相同")
    if l < 0:
        raise ValueError("l 必须为非负整数")

    n = r.size
    if n < 3:
        raise ValueError("至少需要 3 个网格点")

    # 去掉 j=0 点（近核奇异）
    n_reduced = n - 1
    r_inner = r[1:]  # j=1, 2, ..., n-1

    # 有效势能
    V_eff = v_of_r + 0.5 * l * (l + 1) / (r**2 + 1e-30)

    # 构造 H 矩阵（三对角）
    H = np.zeros((n_reduced, n_reduced), dtype=float)
    B = np.zeros((n_reduced, n_reduced), dtype=float)

    for i in range(n_reduced):
        j_actual = i + 1  # 实际网格索引
        exp_factor = 2.0 * delta**2 * Rp**2 * np.exp(2.0 * delta * j_actual)

        # H 矩阵元
        if i > 0:
            H[i, i - 1] = -1.0
        H[i, i] = 2.0 + delta**2 / 4.0 + exp_factor * V_eff[j_actual]
        if i < n_reduced - 1:
            H[i, i + 1] = -1.0

        # B 矩阵（对角）
        B[i, i] = 2.0 * delta**2 * Rp**2 * np.exp(2.0 * delta * j_actual)

    return H, B, r_inner
