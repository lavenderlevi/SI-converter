"""
si_converter — Thư viện chuyển đổi đơn vị sang hệ SI
======================================================

Cài đặt:
    pip install si-converter          # (khi đã publish lên PyPI)
    pip install -e .                  # (dev / local)

Sử dụng cơ bản:
    from si_converter import convert

    convert("1 eV")          # → {'value_out': 1.602e-19, 'unit_out': 'joule', ...}
    convert("100 degC")      # → {'value_out': 373.15,    'unit_out': 'kelvin', ...}
    convert(60, "mph")       # → {'value_out': 26.8224,   'unit_out': 'meter / second', ...}
    convert(1, "eV", "J")   # → chỉ định đơn vị đích

Bán dẫn chuyên sâu:
    from si_converter import convert_semiconductor
    convert_semiconductor(1, "hartree")   # → joule
    convert_semiconductor(1, "bohr")      # → meter
    convert_semiconductor(2.5, "debye")   # → coulomb * meter

API đầy đủ:
    from si_converter import SIConverter
    cvt = SIConverter()
    cvt.convert_dataframe(df, "col", "kPa")
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Union

import numpy as np

# Re-export toàn bộ public API của core
from si_converter.core import (  # noqa: F401
    PHYSICAL_CONSTANTS,
    Q_,
    SIConverter,
    _parse_number,
    clean_numeric_input,
    ureg,
    validate_unit,
)
from si_converter.semiconductor import (  # noqa: F401
    SEMICONDUCTOR_CONSTANTS,
    convert_semiconductor,
)

__version__ = "1.0.0"
__author__ = "SI Converter"
__all__ = [
    "convert",
    "convert_semiconductor",
    "SIConverter",
    "validate_unit",
    "clean_numeric_input",
    "PHYSICAL_CONSTANTS",
    "SEMICONDUCTOR_CONSTANTS",
    "__version__",
]

# ── Singleton converter (lazy init) ──────────────────────────────────────────
_DEFAULT_CVT: Optional[SIConverter] = None


def _get_default_cvt() -> SIConverter:
    global _DEFAULT_CVT
    if _DEFAULT_CVT is None:
        _DEFAULT_CVT = SIConverter(verbose=False)
    return _DEFAULT_CVT


# ── Pattern: "1 eV", "100 degC", "1.5e-3 kgf/cm**2" ─────────────────────────
_STR_PATTERN = re.compile(r"^\s*([+-]?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?)\s+(.+)\s*$")


def convert(
    value: Union[str, float, int, np.ndarray, list],
    from_unit: Optional[str] = None,
    to_unit: Optional[str] = None,
    *,
    clean: bool = True,
) -> Dict[str, Any]:
    """
    Chuyển đổi ``value`` từ ``from_unit`` sang ``to_unit`` (mặc định → SI tự động).

    Cú pháp linh hoạt:
        convert("1 eV")              # chuỗi tự phân tích
        convert("100 degC")
        convert(60, "mph")           # số + đơn vị riêng
        convert(1, "eV", "joule")    # chỉ định đơn vị đích
        convert([1, 2, 3], "atm")    # mảng

    Parameters
    ----------
    value     : chuỗi dạng "số đơn_vị", scalar, hoặc array-like
    from_unit : đơn vị nguồn (bỏ qua nếu value là chuỗi tự đủ)
    to_unit   : đơn vị đích (None → SI tự động)
    clean     : loại bỏ phần tử không hợp lệ trong mảng

    Returns
    -------
    dict:
        value_in, unit_in, value_out, unit_out,
        factor, dimensionality, is_affine, quantity

    Examples
    --------
    >>> from si_converter import convert
    >>> result = convert("1 eV")
    >>> print(result["value_out"], result["unit_out"])
    1.602176634e-19 joule

    >>> result = convert("100 degC")
    >>> print(result["value_out"])
    373.15

    >>> result = convert(60, "mph", "m/s")
    >>> print(result["value_out"])
    26.8224
    """
    # ── Nếu value là chuỗi dạng "1 eV" hoặc "100 degC" ──────────────────────
    if isinstance(value, str) and from_unit is None:
        m = _STR_PATTERN.match(value)
        if m:
            num_str, unit_str = m.group(1), m.group(2).strip()
            num_str = num_str.replace(",", ".")
            parsed_val = float(num_str)
            return _get_default_cvt().convert(
                parsed_val, unit_str, to_unit, clean=clean
            )
        else:
            raise ValueError(
                f"Không thể phân tích chuỗi '{value}'.\n"
                "Định dạng hợp lệ: 'số đơn_vị'  ví dụ: '1 eV' hoặc '100 degC'.\n"
                "Nếu chỉ nhập đơn vị, hãy truyền from_unit riêng."
            )

    # ── Trường hợp thông thường ────────────────────────────────────────────────
    if from_unit is None:
        raise ValueError(
            "Cần truyền from_unit. Ví dụ:\n"
            "  convert(60, 'mph')         # số + đơn vị\n"
            "  convert('60 mph')          # chuỗi tự phân tích"
        )

    return _get_default_cvt().convert(value, from_unit, to_unit, clean=clean)
