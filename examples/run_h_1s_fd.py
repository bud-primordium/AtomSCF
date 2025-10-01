"""氢原子 1s 能级与径向波函数（有限差分离散）

运行示例：

    python -m examples.run_h_1s_fd

将打印最低若干能级，并保存 1s 径向波函数到 CSV。
"""
from __future__ import annotations

import numpy as np

from atomscf.grid import radial_grid_linear, trapezoid_weights
from atomscf.operator import solve_bound_states_fd
from atomscf.utils import trapz


def main() -> None:
    Z = 1.0
    rmin, rmax, n = 1e-6, 80.0, 3000
    r, w = radial_grid_linear(n, rmin, rmax)
    v = -Z / np.maximum(r, 1e-12)

    eps, U = solve_bound_states_fd(r, l=0, v_of_r=v, k=4)
    print("最低 4 个本征值 (Ha):", eps)
    u1s = U[0]
    print("1s 能级 (Ha):", eps[0])
    print("∫u^2 dr ≈", trapz(u1s * u1s, r, w))

    # 保存 1s 径向波函数
    data = np.column_stack([r, u1s])
    np.savetxt("wavefunc_n1_l0_s0_fd.csv", data, delimiter=",", header="r,u(r)")
    print("已保存: wavefunc_n1_l0_s0_fd.csv")


if __name__ == "__main__":
    main()

