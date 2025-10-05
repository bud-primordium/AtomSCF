"""参考数据加载器。

提供 NIST LDA/LSD 等参考数据的统一加载接口。

内嵌数据范围：Z=1-18（非相对论）
- LDA: Local Density Approximation（自旋非极化）
- LSD: Local Spin Density（自旋极化，即 LSDA）

数据来源: https://www.nist.gov/pml/atomic-reference-data-electronic-structure-calculations
"""
import os
from pathlib import Path
from typing import Dict

from .nist_reference_data import NIST_REFERENCE_DATA

__all__ = ["load_nist_reference", "get_ref_data_path"]


def get_ref_data_path() -> Path | None:
    """获取参考数据根目录。

    优先级：
    1. 环境变量 ATOMSCF_REF_PATH
    2. 默认位置（相对于仓库根目录）

    Returns
    -------
    ref_path : Path | None
        参考数据根目录，不存在则返回 None
    """
    # 尝试环境变量
    env_path = os.environ.get("ATOMSCF_REF_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    # 尝试默认位置（相对于 src/atomscf）
    default_path = Path(__file__).parent.parent.parent / ".workenv" / "For_AtomSCF" / "ref_data"
    if default_path.exists():
        return default_path

    return None


def load_nist_reference(
    element_or_Z: str | int,
    mode: str = "LSD"
) -> Dict[str, float]:
    """加载 NIST 参考数据（内嵌版，Z=1-18）。

    Parameters
    ----------
    element_or_Z : str | int
        元素符号（如 "C", "Al"）或原子序数（如 6, 13）
    mode : str
        数据类型："LDA"（自旋非极化）或 "LSD"（自旋极化）

    Returns
    -------
    ref_data : dict
        包含能级与能量的字典，例如：
        LSD: {"Etot": -37.470031, "1sD": -9.940546, "1su": -9.905802, ...}
        LDA: {"Etot": -37.425749, "1s": -9.947718, "2s": -0.500866, ...}

    Raises
    ------
    ValueError
        不支持的元素或模式
    KeyError
        Z>18（需要外部数据源）

    Examples
    --------
    >>> # 通过元素符号加载
    >>> data = load_nist_reference("C", mode="LSD")
    >>> print(data["Etot"])
    -37.470031

    >>> # 通过原子序数加载
    >>> data = load_nist_reference(6, mode="LDA")
    >>> print(data["Etot"])
    -37.425749

    Notes
    -----
    - 仅支持 Z=1-18（非相对论数据）
    - Z>18 请访问 https://www.nist.gov/pml/atomic-reference-data-electronic-structure-calculations
    """
    if mode not in ("LDA", "LSD"):
        raise ValueError(f"mode 必须为 'LDA' 或 'LSD'，当前值: {mode}")

    # 转换为原子序数
    if isinstance(element_or_Z, str):
        element_to_Z = {
            "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
            "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
            "S": 16, "Cl": 17, "Ar": 18,
        }
        Z = element_to_Z.get(element_or_Z.capitalize())
        if Z is None:
            raise ValueError(f"不支持的元素符号: {element_or_Z}")
    else:
        Z = element_or_Z

    # 检查范围
    if Z < 1 or Z > 18:
        raise KeyError(
            f"Z={Z} 超出内嵌数据范围（Z=1-18）。\n"
            f"请访问 https://www.nist.gov/pml/atomic-reference-data-electronic-structure-calculations 获取完整数据"
        )

    # 从内嵌字典加载
    if Z not in NIST_REFERENCE_DATA[mode]:
        raise KeyError(f"Z={Z} 的 {mode} 数据不存在")

    return NIST_REFERENCE_DATA[mode][Z]


# 向后兼容别名
def load_nist_lsd(element_or_path: str | Path) -> Dict[str, float]:
    """向后兼容接口（已弃用）。

    推荐使用 load_nist_reference(element_or_Z, mode="LSD")
    """
    # 判断是元素符号还是路径
    if isinstance(element_or_path, str) and len(element_or_path) <= 2:
        # 元素符号
        return load_nist_reference(element_or_path, mode="LSD")
    else:
        # 路径，使用原始解析逻辑
        ref_path = Path(element_or_path)
        if not ref_path.exists():
            raise FileNotFoundError(f"参考数据文件不存在: {ref_path}")

        ref_data = {}
        with open(ref_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if "=" in line:
                    key, value = line.split("=")
                    ref_data[key.strip()] = float(value.strip())
                else:
                    parts = line.split()
                    if len(parts) == 2:
                        label, value = parts
                        ref_data[label] = float(value)

        return ref_data
