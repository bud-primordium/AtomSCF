from __future__ import annotations

"""轻量打靶细化（shooting）求解器

提供基于向内积分（从大 r 向小 r）与能量一维根寻找的能量细化工具，
用于在已获得较好初值（如 transformed+exp 求得的本征能）基础上，
对外层轨道（例如 2p）进行精化，改善边界条件满足度与能级精度。

基本思想：
- 径向方程（原始 r 变量）：
    u''(r) = [2 (V_eff(r) - E) + l(l+1)/r^2] u(r)
- 远端边界（r -> ∞）：u ~ exp(-κ r), κ = sqrt(-2E)
- 从 r_max 向 r_min 采用可变步长 RK4 积分；
- 以近原点的“对数导数匹配”作为目标：r * u'/u -> l + 1；
- 对能量做一维根寻找（优先二分法，必要时割线）以逼近匹配条件。

注意：
- 本模块仅做轻量精化，不替代主本征求解流程；
- 要求输入网格 r 为严格单调递增（从小到大），v_eff 与 r 等长；
- 仅依赖 numpy；归一化使用梯形权重。
"""

from typing import Tuple, Callable

import numpy as np

from .grid import trapezoid_weights

Array = np.ndarray


def _rhs_u2(r: float, u: float, du: float, l: int, E: float, v_eff_func: Callable[[float], float]) -> tuple[float, float]:
    """径向方程一阶化后的 RHS：

    y1 = u, y2 = du/dr
    y1' = y2
    y2' = Q(r; E) * y1, 其中 Q(r;E) = 2*(v_eff(r)-E) + l(l+1)/r^2
    """
    r_safe = r if r > 1e-12 else 1e-12
    Q = 2.0 * (float(v_eff_func(r_safe)) - E) + float(l * (l + 1)) / (r_safe * r_safe)
    return du, Q * u


def _integrate_inward_rk4(r: Array, l: int, v_eff: Array, E: float) -> tuple[Array, Array]:
    """从大 r 向小 r 的 RK4 积分（可变步长）。

    返回：u(r) 与 du/dr(r)（与 r 同向：从小到大）。
    """
    assert r.ndim == 1 and v_eff.shape == r.shape
    N = r.size

    # 采用线性插值构造 v_eff(r) 的连续表示（便于 RK4 在中点取值）
    v_func = lambda x: np.interp(x, r, v_eff)

    u = np.zeros(N, dtype=float)
    du = np.zeros(N, dtype=float)

    # 远端边界条件：选择幅度=1，导数=-κ
    # 说明：幅度仅影响整体缩放，不影响匹配函数符号与根寻找；
    # 选取 u(R)=1, u'(R)=-κ 数值上更稳健（避免过小幅度导致下溢）。
    R = float(r[-1])
    kappa = np.sqrt(max(-2.0 * E, 1e-16)) if E < 0 else 1.0
    uN = 1.0
    duN = -kappa
    u[-1] = uN
    du[-1] = duN

    # 反向步进：i = N-2 -> 0，每步长度 h = r[i]-r[i+1] (<0)
    for i in range(N - 2, -1, -1):
        r1 = float(r[i + 1])
        h = float(r[i] - r[i + 1])  # 负值

        y1 = u[i + 1]
        y2 = du[i + 1]

        # k1
        k1_u, k1_du = _rhs_u2(r1, y1, y2, l, E, v_func)
        # k2 at (r1 + h/2)
        k2_u, k2_du = _rhs_u2(r1 + 0.5 * h, y1 + 0.5 * h * k1_u, y2 + 0.5 * h * k1_du, l, E, v_func)
        # k3
        k3_u, k3_du = _rhs_u2(r1 + 0.5 * h, y1 + 0.5 * h * k2_u, y2 + 0.5 * h * k2_du, l, E, v_func)
        # k4 at (r1 + h)
        k4_u, k4_du = _rhs_u2(r1 + h, y1 + h * k3_u, y2 + h * k3_du, l, E, v_func)

        u[i] = y1 + (h / 6.0) * (k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u)
        du[i] = y2 + (h / 6.0) * (k1_du + 2.0 * k2_du + 2.0 * k3_du + k4_du)

    return u, du


def _logderiv_mismatch(u: Array, du: Array, r: Array, l: int) -> float:
    """计算近原点的对数导数匹配误差：mean_i [ r_i * (u'/u)_i ] - (l+1)。

    数值稳定性：
    - 从小 r 端起，选取前若干个 |u| 不太小的点（例如 |u|>阈值）；
    - 若不足，则退化为使用 u[r_min] 的符号作为替代指标（返回该值，供二分符号判据）。
    """
    N = r.size
    # 选择前 m 个点（最多 6 个），且 |u| 足够大
    idxs = []
    m = min(6, N)
    for i in range(0, m):
        if abs(u[i]) > 1e-12:
            idxs.append(i)
    if len(idxs) >= 2:
        vals = []
        for i in idxs:
            r_i = r[i] if r[i] > 1e-12 else 1e-12
            vals.append(r_i * (du[i] / u[i]))
        return float(np.median(vals) - (l + 1))
    else:
        # 回退指标：使用 u(r_min)（非严格物理，但可用于符号二分）
        return float(u[0])


def _find_bracket(fn: Callable[[float], float], E0: float, sign_only: bool = False) -> tuple[float, float, float, float] | None:
    """围绕 E0 扩展搜索，寻找 fn(E) 的异号括区间。

    若 sign_only=True，fn 仅用于符号（例如使用 u(r_min) 指标时）。
    """
    E_lo = E0 * 1.2 if E0 < 0 else -0.1
    E_hi = E0 * 0.8 if E0 < 0 else -1e-3
    E_hi = min(E_hi, -1e-6)  # 保持束缚态
    f_lo = fn(E_lo)
    f_hi = fn(E_hi)
    if np.sign(f_lo) != np.sign(f_hi) and np.isfinite(f_lo) and np.isfinite(f_hi):
        return E_lo, E_hi, f_lo, f_hi

    # 逐步扩展（最多 25 次）：向浅束缚和深束缚两端扩展
    E_left, f_left = E_lo, f_lo
    E_right, f_right = E_hi, f_hi
    for _ in range(25):
        # 向右（接近 0-）
        E_right = 0.5 * (E_right)
        if E_right >= -1e-8:
            E_right = -1e-8
        f_right = fn(E_right)
        if np.sign(f_left) != np.sign(f_right) and np.isfinite(f_left) and np.isfinite(f_right):
            return E_left, E_right, f_left, f_right
        # 向左（更深）
        E_left *= 1.5
        f_left = fn(E_left)
        if np.sign(f_left) != np.sign(f_right) and np.isfinite(f_left) and np.isfinite(f_right):
            return E_left, E_right, f_left, f_right

    return None


def _bisect_root(fn: Callable[[float], float], a: float, b: float, fa: float, fb: float, tol: float, max_iter: int) -> tuple[float, float]:
    """标准二分法（带初始函数值）。返回 (E_root, f(E_root))。"""
    left, right = a, b
    f_left, f_right = fa, fb
    if np.sign(f_left) == np.sign(f_right):
        raise ValueError("区间端点同号，无法二分")
    for _ in range(max_iter):
        c = 0.5 * (left + right)
        f_c = fn(c)
        if abs(right - left) < tol or abs(f_c) < 1e-9:
            return c, f_c
        if np.sign(f_c) == np.sign(f_left):
            left, f_left = c, f_c
        else:
            right, f_right = c, f_c
    return 0.5 * (left + right), fn(0.5 * (left + right))


def shooting_refine_energy(
    r: np.ndarray,
    l: int,
    v_eff: np.ndarray,
    E_initial: float,
    target_norm: float = 1.0,
    method: str = "rk4",
    E_tol: float = 1e-6,
    max_iter: int = 50,
) -> Tuple[float, np.ndarray]:
    """使用打靶法细化能量本征值（从大 r 向小 r 积分）。

    参数
    ----
    r : np.ndarray
        径向网格（从小到大，严格单调）。
    l : int
        角动量量子数。
    v_eff : np.ndarray
        有效势 V_eff(r)（不含离心项）。
    E_initial : float
        初始能量（推荐来自 transformed 求解器）。
    target_norm : float
        目标归一化（最终将按 ∫ u^2 dr = target_norm 归一）。
    method : str
        数值积分方法（目前支持 'rk4'）。
    E_tol : float
        能量收敛阈值（绝对值）。
    max_iter : int
        最大迭代次数。

    返回
    ----
    E_refined, u_refined : (float, np.ndarray)
        细化后的能量与对应归一化的径向函数 u(r)。

    算法说明
    --------
    1) 在给定 E 下，采用 RK4 从 r_max 向 r_min 积分；
    2) 以近原点的对数导数匹配为目标：r * u'/u -> l+1，构造 mismatch(E)；
    3) 围绕 E_initial 寻找 mismatch(E) 异号括区间，采用二分法细化；
    4) 若无法稳定使用对数导数匹配，则回退到 u(r_min) 作为符号指标；
    5) 最终对 u 进行归一化并返回。
    """
    if method != "rk4":
        raise NotImplementedError("当前仅支持 method='rk4'")
    if r.ndim != 1 or v_eff.shape != r.shape:
        raise ValueError("r 与 v_eff 形状不一致或维度不为 1")
    if E_initial >= 0:
        # 仅针对束缚态细化；若初值非负，稍作偏移
        E_initial = -abs(E_initial) if E_initial != 0 else -1e-3

    # 定义 mismatch(E)
    def mismatch(E: float) -> float:
        u_tmp, du_tmp = _integrate_inward_rk4(r, l, v_eff, float(E))
        return _logderiv_mismatch(u_tmp, du_tmp, r, l)

    # 先用对数导数匹配寻找括区间
    bracket = _find_bracket(mismatch, E_initial)
    use_u0 = False
    if bracket is None:
        # 回退：以 u(r_min) 作为符号指标
        def u0_indicator(E: float) -> float:
            u_tmp, _ = _integrate_inward_rk4(r, l, v_eff, float(E))
            return float(u_tmp[0])

        bracket = _find_bracket(u0_indicator, E_initial, sign_only=True)
        use_u0 = True

    if bracket is None:
        # 仍未能找到括区间：直接返回初值积分的 u（视为失败的保守回退）
        u0, _ = _integrate_inward_rk4(r, l, v_eff, float(E_initial))
        # 归一化
        w = trapezoid_weights(r)
        norm = float(np.sqrt(np.sum(w * (u0 * u0))))
        if norm > 0:
            u0 = u0 * (target_norm / norm)
        return float(E_initial), u0

    a, b, fa, fb = bracket
    fn = (lambda E: float(_integrate_inward_rk4(r, l, v_eff, float(E))[0][0])) if use_u0 else mismatch
    # 细化根
    E_star, f_star = _bisect_root(fn, a, b, fa if not use_u0 else fn(a), fb if not use_u0 else fn(b), E_tol, max_iter)

    # 最终再做一次积分并归一化
    u_star, _ = _integrate_inward_rk4(r, l, v_eff, float(E_star))
    w = trapezoid_weights(r)
    norm = float(np.sqrt(np.sum(w * (u_star * u_star))))
    if norm > 0:
        u_star = u_star * (target_norm / norm)

    return float(E_star), u_star

