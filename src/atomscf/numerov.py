from __future__ import annotations

import numpy as np


def _is_uniform_log_grid(r: np.ndarray, tol: float = 1e-10) -> tuple[bool, float, np.ndarray]:
    s = np.log(r)
    ds = np.diff(s)
    if np.allclose(ds, ds[0], atol=tol, rtol=0):
        return True, float(ds[0]), s
    return False, 0.0, s


def _build_Q(eps: float, r: np.ndarray, v_eff: np.ndarray, l: int) -> np.ndarray:
    # 在对数坐标变换下得到的等价方程 y'' = Q(s) y
    # Q(s) = 1/4 + l(l+1) + 2 r^2 v_eff - 2 eps r^2
    r2 = r * r
    return 0.25 + l * (l + 1.0) + 2.0 * r2 * (v_eff - eps)


def _numerov_forward(y0: float, y1: float, h: float, Q: np.ndarray) -> np.ndarray:
    n = Q.size
    y = np.zeros(n, dtype=float)
    y[0] = y0
    y[1] = y1
    c1 = 1.0 / 12.0
    for i in range(1, n - 1):
        Qi_1 = Q[i - 1]
        Qi = Q[i]
        Qi1 = Q[i + 1]
        # Numerov 三点公式：
        # y_{i+1} = [2 y_i (1 - 5 h^2 Qi / 12) - y_{i-1} (1 + h^2 Qi-1 / 12)] / (1 + h^2 Qi+1 / 12)
        a = 1.0 + c1 * h * h * Qi1
        b = 2.0 * (1.0 - 5.0 * c1 * h * h * Qi)
        c = 1.0 + c1 * h * h * Qi_1
        y[i + 1] = (b * y[i] - c * y[i - 1]) / a
    return y


def numerov_eigen_log_ground(r: np.ndarray, v_eff: np.ndarray, l: int, eps_bounds: tuple[float, float] = (-50.0, -1e-6), max_iter: int = 60, tol: float = 1e-8) -> tuple[float, np.ndarray]:
    """在对数网格上用 Numerov 寻找基态（给定 l）的本征对（粗略版）。

    .. deprecated::
        此函数已不推荐使用，建议改用 `numerov_find_k_log_by_nodes` 以获得更好的稳定性和多态支持。
        保留以支持教学型示例和向后兼容。

    仅用于教学与快速逼近：采用末端值符号变化作为二分根搜索指标，未做节点计数与匹配优化。
    若二分失败，抛出 RuntimeError。
    """
    import warnings
    warnings.warn(
        "numerov_eigen_log_ground 已被 numerov_find_k_log_by_nodes 替代，"
        "计划在未来版本中移除。请更新您的代码。",
        DeprecationWarning,
        stacklevel=2
    )
    ok, h, s = _is_uniform_log_grid(r)
    if not ok:
        raise ValueError("numerov_eigen_log_ground 要求 r 为对数等距网格（ln r 等差）")

    s0 = s[0]
    # y = e^{-s/2} u，近核 u ~ r^{l+1} → y ~ e^{(l+1/2) s}
    y0 = 0.0
    y1 = np.exp((l + 0.5) * (s[1] - s0))

    def shoot(eps: float) -> float:
        Q = _build_Q(eps, r, v_eff, l)
        y = _numerov_forward(y0, y1, h, Q)
        return y[-1]

    a, b = eps_bounds
    fa = shoot(a)
    fb = shoot(b)
    # 尝试扩大上界直到符号相异或达到限制
    tries = 0
    while fa * fb > 0.0 and tries < 10:
        b *= 0.5
        fb = shoot(b)
        tries += 1
    if fa * fb > 0.0:
        raise RuntimeError("Numerov 二分初始区间未能找到符号相异的端点")

    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc = shoot(c)
        if abs(fc) < 1e-12 or abs(b - a) < tol:
            eps = c
            break
        if fa * fc < 0.0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    else:
        raise RuntimeError("Numerov 二分未在迭代内收敛")

    # 回代求最终 y 并转回 u = e^{s/2} y
    Q = _build_Q(eps, r, v_eff, l)
    y = _numerov_forward(y0, y1, h, Q)
    u = np.exp(0.5 * (s - s0)) * y  # 相对尺度无影响，后续归一化
    return eps, u


def _shoot_yend_and_nodes(eps: float, r: np.ndarray, v_eff: np.ndarray, l: int, h: float, s: np.ndarray) -> tuple[float, int]:
    Q = _build_Q(eps, r, v_eff, l)
    # 初始条件（近核）
    s0 = s[0]
    y0 = 0.0
    y1 = np.exp((l + 0.5) * (s[1] - s0))
    y = _numerov_forward(y0, y1, h, Q)
    # 节点计数（忽略起点）
    nodes = 0
    prev = y[1]
    for val in y[2:]:
        if prev == 0.0:
            prev = val
            continue
        if val == 0.0 or (prev > 0 and val < 0) or (prev < 0 and val > 0):
            nodes += 1
        prev = val
    return y[-1], nodes


def numerov_find_k_log(
    r: np.ndarray,
    v_eff: np.ndarray,
    l: int,
    k: int,
    eps_min: float = -80.0,
    eps_max: float = -1e-6,
    samples: int = 200,
    bisection_iter: int = 60,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """在对数等距网格上，用 Numerov 扫描能量段并用二分逼近，寻找前 k 个本征态（粗略）。

    .. deprecated::
        此函数已不推荐使用，建议改用 `numerov_find_k_log_by_nodes` 以获得更精确的节点计数和根选择。
        保留以支持教学型示例和向后兼容。

    说明：
    - 仅用于原型与教学，未做匹配/外向积分等高级稳定化；
    - 通过扫描区间 [eps_min, eps_max]，记录 y_end 的符号变化来判别根，辅以节点计数判断态序；
    - 返回 (eps[:k], U[:k, :])，若不足 k 个根则返回找到的所有根。
    """
    import warnings
    warnings.warn(
        "numerov_find_k_log 已被 numerov_find_k_log_by_nodes 替代，"
        "计划在未来版本中移除。请更新您的代码。",
        DeprecationWarning,
        stacklevel=2
    )
    ok, h, s = _is_uniform_log_grid(r)
    if not ok:
        raise ValueError("numerov_find_k_log 要求 r 为对数等距网格（ln r 等差）")
    # 扫描
    es = np.linspace(eps_min, eps_max, samples)
    yend = []
    nodes = []
    for e in es:
        ye, nd = _shoot_yend_and_nodes(e, r, v_eff, l, h, s)
        yend.append(ye)
        nodes.append(nd)
    yend = np.array(yend)
    nodes = np.array(nodes)
    # 找到符号变化的区间
    roots = []
    for i in range(samples - 1):
        if yend[i] == 0.0:
            roots.append((es[i], es[i]))
        elif yend[i] * yend[i + 1] < 0:
            roots.append((es[i], es[i + 1]))
        if len(roots) >= k:
            break
    # 二分逼近每个区间，提取 k 个
    eps_list = []
    u_list = []
    for a, b in roots:
        fa, _ = _shoot_yend_and_nodes(a, r, v_eff, l, h, s)
        fb, _ = _shoot_yend_and_nodes(b, r, v_eff, l, h, s)
        # 二分
        for _ in range(bisection_iter):
            c = 0.5 * (a + b)
            fc, _ = _shoot_yend_and_nodes(c, r, v_eff, l, h, s)
            if abs(fc) < 1e-12 or abs(b - a) < tol:
                break
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        eps = 0.5 * (a + b)
        # 求对应的 u
        Q = _build_Q(eps, r, v_eff, l)
        s0 = s[0]
        y0 = 0.0
        y1 = np.exp((l + 0.5) * (s[1] - s0))
        y = _numerov_forward(y0, y1, h, Q)
        u = np.exp(0.5 * (s - s0)) * y
        eps_list.append(eps)
        u_list.append(u)
        if len(eps_list) >= k:
            break
    if not eps_list:
        raise RuntimeError("Numerov 未找到任何本征根，建议调整 eps_min/eps_max 或 samples")
    return np.array(eps_list), np.vstack(u_list)


# ----------------------------- 匹配法（推荐） -----------------------------

def _numerov_backward(yN: float, yNm1: float, h: float, Q: np.ndarray) -> np.ndarray:
    """从右端向左递推的 Numerov（与前向形式对称），返回与 Q.size 等长的解。"""
    n = Q.size
    y = np.zeros(n, dtype=float)
    y[-1] = yN
    y[-2] = yNm1
    c1 = 1.0 / 12.0
    for i in range(n - 2, 0, -1):
        Qi1 = Q[i + 1]
        Qi = Q[i]
        Qi_1 = Q[i - 1]
        a = 1.0 + c1 * h * h * Qi_1
        b = 2.0 * (1.0 - 5.0 * c1 * h * h * Qi)
        c = 1.0 + c1 * h * h * Qi1
        y[i - 1] = (b * y[i] - c * y[i + 1]) / a
    return y


def _log_derivative(y: np.ndarray, h: float, i: int) -> float:
    """在等距 s 网格上计算 y'/y 的近似（中心差分）。"""
    denom = y[i]
    if abs(denom) < 1e-20:
        return 1e20
    dy = (y[i + 1] - y[i - 1]) / (2.0 * h)
    return dy / denom


def numerov_find_k_log_matching(
    r: np.ndarray,
    v_eff: np.ndarray,
    l: int,
    k: int,
    eps_min: float = -80.0,
    eps_max: float = -1e-6,
    samples: int = 120,
    bisection_iter: int = 60,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """匹配法 Numerov：在对数等距网格上寻找前 k 个本征态（更稳定）。

    .. deprecated::
        此函数已不推荐使用，建议改用 `numerov_find_k_log_by_nodes` 作为生产环境的主力求解器。
        保留以支持教学型示例、方法对比研究和向后兼容。

    思路：
    - 选匹配点 i_m（默认中点），从左端与右端分别进行 Numerov 积分；
    - 计算匹配点的对数导数差 L_left - L_right；
    - 以该差号变为根判据，在能量段内用扫描 + 二分逼近根；
    - 通过节点计数辅助判断根的态序；
    - 合并左右解并按匹配点归一（保证函数连续），转换回 u(r)。
    """
    import warnings
    warnings.warn(
        "numerov_find_k_log_matching 已被 numerov_find_k_log_by_nodes 替代，"
        "计划在未来版本中移除。请更新您的代码。",
        DeprecationWarning,
        stacklevel=2
    )
    ok, h, s = _is_uniform_log_grid(r)
    if not ok:
        raise ValueError("numerov_find_k_log_matching 要求 r 为对数等距网格（ln r 等差）")

    n = r.size
    def _choose_match_index(eps: float) -> int:
        # 选择靠近转折点的位置作为匹配点：使 |V_eff_tot(r) - eps| 最小
        # V_eff_tot = V_eff + l(l+1)/(2 r^2)
        Vtot = v_eff + 0.5 * l * (l + 1.0) / np.maximum(r * r, 1e-30)
        idx = int(np.argmin(np.abs(Vtot - eps)))
        # 保证不在边界，留出中心差分空间
        idx = min(max(idx, 2), n - 3)
        return idx

    def mismatch(eps: float) -> tuple[float, np.ndarray, np.ndarray, int]:
        Q = _build_Q(eps, r, v_eff, l)
        # 左端初值（近核行为）
        s0 = s[0]
        y0L = 0.0
        y1L = np.exp((l + 0.5) * (s[1] - s0))
        yL = _numerov_forward(y0L, y1L, h, Q)
        # 右端初值（远端指数衰减，按局部 kappa 给两点初值）
        # kappa^2 = 2 (V_eff_tot - eps)（束缚态 E < V，渐近衰减）
        # 参考 codex reply_check_4.md 第 68 行
        Vtot = v_eff + 0.5 * l * (l + 1.0) / np.maximum(r * r, 1e-30)
        kappa2 = 2.0 * np.maximum(Vtot - eps, 0.0)  # 修正符号
        kappa = np.sqrt(kappa2 + 1e-30)
        # 近似 y ~ exp(-kappa r) → 在 s=ln r 坐标，步长对应 dr/r = ds，取一个温和衰减比例
        decay = np.exp(-kappa[-1] * (r[-1] - r[-2]))
        yN = 0.0
        yNm1 = decay
        yR = _numerov_backward(yN, yNm1, h, Q)
        # 计算匹配点的对数导数差
        im = _choose_match_index(eps)
        L_left = _log_derivative(yL, h, im)
        L_right = _log_derivative(yR, h, im)
        return (L_left - L_right), yL, yR, im

    # 扫描能量段，寻找 mismatch 符号改变的区间
    es = np.linspace(eps_min, eps_max, samples)
    mm = []
    yLs = []
    yRs = []
    ims = []
    for e in es:
        m, yL, yR, im = mismatch(e)
        mm.append(m)
        yLs.append(yL)
        yRs.append(yR)
        ims.append(im)
    mm = np.array(mm)
    ims = np.array(ims)

    brackets = []
    for i in range(samples - 1):
        if mm[i] == 0.0:
            brackets.append((es[i], es[i]))
        elif mm[i] * mm[i + 1] < 0:
            brackets.append((es[i], es[i + 1]))
        if len(brackets) >= k:
            break
    if not brackets:
        raise RuntimeError("Numerov 匹配法：未找到匹配区间，请调整 eps_min/eps_max 或 samples")

    eps_list = []
    u_list = []
    for a, b in brackets:
        ma, _, _, _ = mismatch(a)
        mb, _, _, _ = mismatch(b)
        # 二分逼近
        for _ in range(bisection_iter):
            c = 0.5 * (a + b)
            mc, yL, yR, im = mismatch(c)
            if abs(mc) < 1e-10 or abs(b - a) < tol:
                break
            if ma * mc < 0:
                b, mb = c, mc
            else:
                a, ma = c, mc
        eps = 0.5 * (a + b)
        # 最终 yL, yR 采用上一步的 yL, yR；按匹配点进行缩放合并
        # 使 yL[im] == yR[im]
        scale = 1.0
        if abs(yR[im]) > 1e-30:
            scale = yL[im] / yR[im]
        yR_scaled = yR * scale
        y = np.concatenate([yL[: im + 1], yR_scaled[im + 1 :]])
        # 转回 u 并归一
        s0 = s[0]
        u = np.exp(0.5 * (s - s0)) * y
        # 归一化留给外层（SCF 通常会统一归一化）——这里直接返回 u，SCF 会 normalize_radial_u
        eps_list.append(eps)
        u_list.append(u)
        if len(eps_list) >= k:
            break

    return np.array(eps_list), np.vstack(u_list)


def numerov_find_k_log_by_nodes(
    r: np.ndarray,
    v_eff: np.ndarray,
    l: int,
    k: int,
    eps_min: float = -80.0,
    eps_max: float = -1e-6,
    samples: int = 200,
    bisection_iter: int = 80,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """基于“节点计数”选根的 Numerov（对数等距网格）。

    思路：
    - 在 [eps_min, eps_max] 上粗扫，记录 y_end 与节点数 nodes(e)；
    - 对于目标节点数 n_tgt=0..k-1，寻找 nodes 从 >n_tgt 降到 ≤n_tgt 的区间作为括号；
    - 在括号内用“节点数优先”的二分收缩（并辅以 y_end 收缩），逼近 y_end≈0 的根；
    - 返回按节点数从小到大（即束缚程度从浅到深的顺序）的前 k 个态。
    """
    ok, h, s = _is_uniform_log_grid(r)
    if not ok:
        raise ValueError("numerov_find_k_log_by_nodes 要求 r 为对数等距网格（ln r 等差）")

    def shoot_info(eps: float) -> tuple[float, int]:
        return _shoot_yend_and_nodes(eps, r, v_eff, l, h, s)

    # 粗扫
    es = np.linspace(eps_min, eps_max, samples)
    yend = np.empty_like(es)
    nodes = np.empty_like(es, dtype=int)
    for i, e in enumerate(es):
        ye, nd = shoot_info(e)
        yend[i] = ye
        nodes[i] = nd

    eps_list = []
    u_list = []
    # 逐个目标节点数
    for n_tgt in range(k):
        # 寻找括号：nodes 左值 > n_tgt，右值 ≤ n_tgt
        a = None
        b = None
        for i in range(len(es) - 1):
            if nodes[i] > n_tgt and nodes[i + 1] <= n_tgt:
                a = es[i]
                b = es[i + 1]
                break
        if a is None:
            # 尝试加密采样一次
            es = np.linspace(eps_min, eps_max, samples * 2)
            yend = np.empty_like(es)
            nodes = np.empty_like(es, dtype=int)
            for i, e in enumerate(es):
                ye, nd = shoot_info(e)
                yend[i] = ye
                nodes[i] = nd
            for i in range(len(es) - 1):
                if nodes[i] > n_tgt and nodes[i + 1] <= n_tgt:
                    a = es[i]
                    b = es[i + 1]
                    break
        if a is None:
            raise RuntimeError("Numerov 节点法：未找到满足目标节点数的括号区间，建议调整能量范围/采样数")

        ya, na = shoot_info(a)
        yb, nb = shoot_info(b)
        # 二分：优先保持 nodes(a)>n_tgt >= nodes(b)
        for _ in range(bisection_iter):
            c = 0.5 * (a + b)
            yc, nc = shoot_info(c)
            # 收敛判据：能量间隔/端点函数值
            if abs(b - a) < tol or abs(yc) < 1e-12:
                a = b = c
                ya = yb = yc
                break
            if nc > n_tgt:
                a, ya, na = c, yc, nc
            else:
                b, yb, nb = c, yc, nc
        eps = 0.5 * (a + b)
        # 最终 u（用前向）
        Q = _build_Q(eps, r, v_eff, l)
        s0 = s[0]
        y0 = 0.0
        y1 = np.exp((l + 0.5) * (s[1] - s0))
        y = _numerov_forward(y0, y1, h, Q)
        u = np.exp(0.5 * (s - s0)) * y
        eps_list.append(eps)
        u_list.append(u)

    return np.array(eps_list), np.vstack(u_list)
