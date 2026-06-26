"""
si_converter.semiconductor
==========================
Module chuyên sâu cho vật liệu bán dẫn — đơn vị nguyên tử và hạt nhân.

Đơn vị được hỗ trợ:
    hartree     — Năng lượng Hartree (năng lượng hệ nguyên tử)
    rydberg     — Năng lượng Rydberg
    bohr        — Bán kính Bohr (độ dài hệ nguyên tử)
    debye       — Đơn vị mômen lưỡng cực điện
    electron_mass / m_e  — Khối lượng electron
    amu / u     — Đơn vị khối lượng nguyên tử
    barn        — Đơn vị tiết diện hạt nhân (10⁻²⁸ m²)

Sử dụng:
    from si_converter import convert_semiconductor

    convert_semiconductor(1, "hartree")       # → joule
    convert_semiconductor(1, "bohr")          # → meter
    convert_semiconductor(2.5, "debye")       # → C·m
    convert_semiconductor(1, "rydberg")       # → joule
    convert_semiconductor(5, "barn")          # → m²
    convert_semiconductor(1, "amu")           # → kg
    convert_semiconductor(1, "electron_mass") # → kg
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import constants as _sc

# ──────────────────────────────────────────────────────────────────────────────
#  Hằng số bán dẫn / nguyên tử (tất cả theo SI)
# ──────────────────────────────────────────────────────────────────────────────

#: Hằng số bán dẫn và nguyên tử đặc biệt
SEMICONDUCTOR_CONSTANTS: Dict[str, Tuple[float, str, str]] = {
    # Năng lượng
    "hartree": (
        _sc.physical_constants["Hartree energy"][0],
        "J",
        "Năng lượng Hartree — đơn vị năng lượng hệ nguyên tử (Eₕ = 2 Ry = 27.211 eV)",
    ),
    "rydberg": (
        _sc.physical_constants["Rydberg constant times hc in J"][0],
        "J",
        "Năng lượng Rydberg — năng lượng ion hóa hydrogen (Ry = Eₕ/2 = 13.606 eV)",
    ),
    # Độ dài
    "bohr": (
        _sc.physical_constants["Bohr radius"][0],
        "m",
        "Bán kính Bohr — đơn vị độ dài hệ nguyên tử (a₀ ≈ 0.529 Å)",
    ),
    "barn": (
        1e-28,
        "m²",
        "Barn — đơn vị tiết diện hạt nhân (1 barn = 10⁻²⁸ m²)",
    ),
    # Mômen lưỡng cực
    "debye": (
        _sc.physical_constants["atomic unit of electric dipole mom."][0]
        * 0.393430307,  # 1 D = 0.393430307 a.u. → C·m
        "C·m",
        "Debye — đơn vị mômen lưỡng cực điện (1 D ≈ 3.336×10⁻³⁰ C·m)",
    ),
    # Khối lượng
    "electron_mass": (
        _sc.m_e,
        "kg",
        "Khối lượng electron (mₑ ≈ 9.109×10⁻³¹ kg)",
    ),
    "amu": (
        _sc.u,
        "kg",
        "Đơn vị khối lượng nguyên tử (u ≈ 1.661×10⁻²⁷ kg)",
    ),
}

# Bí danh (aliases) cho tiện dùng
_ALIASES: Dict[str, str] = {
    "ha": "hartree",
    "eh": "hartree",
    "eₕ": "hartree",
    "ry": "rydberg",
    "a0": "bohr",
    "a₀": "bohr",
    "bohr_radius": "bohr",
    "d": "debye",
    "me": "electron_mass",
    "m_e": "electron_mass",
    "electron mass": "electron_mass",
    "u": "amu",
    "dalton": "amu",
    "da": "amu",
    "atomic_mass_unit": "amu",
    "b": "barn",
    "mb": "barn",   # millibarn → handled specially below
    "nb": "barn",   # nanobarn → handled specially below
}

# Tiền tố số cho barn
_BARN_PREFIXES: Dict[str, float] = {
    "mb": 1e-3,   # millibarn
    "µb": 1e-6,   # microbarn
    "nb": 1e-9,   # nanobarn
    "pb": 1e-12,  # picobarn
    "fb": 1e-15,  # femtobarn
}


def _resolve_unit(unit_str: str) -> Tuple[str, float]:
    """
    Chuyển chuỗi đơn vị → (canonical_name, scale_factor_to_base).
    Trả về (tên_chuẩn, hệ_số).
    """
    key = unit_str.strip().lower()

    # Kiểm tra barn với tiền tố
    if key in _BARN_PREFIXES:
        return "barn", _BARN_PREFIXES[key]

    # Kiểm tra aliases
    if key in _ALIASES:
        key = _ALIASES[key]

    # Kiểm tra trực tiếp
    if key in SEMICONDUCTOR_CONSTANTS:
        return key, 1.0

    return "", 0.0


def convert_semiconductor(
    value: float,
    from_unit: str,
    to_unit: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Chuyển đổi đơn vị bán dẫn / nguyên tử đặc biệt sang SI.

    Parameters
    ----------
    value     : giá trị số đầu vào
    from_unit : đơn vị nguồn:
                'hartree', 'rydberg', 'bohr', 'debye',
                'electron_mass' / 'm_e' / 'me',
                'amu' / 'u' / 'dalton',
                'barn' / 'mb' / 'nb' / 'pb'
    to_unit   : đơn vị đích (hiện tại luôn là SI)

    Returns
    -------
    dict:
        value_in       — giá trị đầu vào
        unit_in        — đơn vị nguồn (canonical)
        value_out      — giá trị sau chuyển đổi (SI)
        unit_out       — đơn vị SI
        factor         — hệ số nhân (SI_value = factor × input)
        description    — mô tả đơn vị
        si_equivalent  — chuỗi mô tả đẹp
        category       — loại thứ nguyên

    Raises
    ------
    ValueError — đơn vị không được nhận diện

    Examples
    --------
    >>> from si_converter import convert_semiconductor
    >>> r = convert_semiconductor(1, "hartree")
    >>> print(f"{r['value_out']:.6e} {r['unit_out']}")
    4.359745e-18 J

    >>> r = convert_semiconductor(1, "bohr")
    >>> print(f"{r['value_out']:.6e} {r['unit_out']}")
    5.291772e-11 m

    >>> r = convert_semiconductor(2.5, "debye")
    >>> print(f"{r['value_out']:.6e} {r['unit_out']}")
    8.339508e-30 C·m
    """
    canonical, prefix_scale = _resolve_unit(from_unit)

    if not canonical:
        available = ", ".join(sorted(SEMICONDUCTOR_CONSTANTS.keys()))
        raise ValueError(
            f"Đơn vị bán dẫn không nhận diện được: '{from_unit}'\n"
            f"Đơn vị hỗ trợ: {available}\n"
            f"Bí danh: {', '.join(sorted(_ALIASES.keys()))}"
        )

    si_value_per_unit, si_unit, description = SEMICONDUCTOR_CONSTANTS[canonical]
    factor = si_value_per_unit * prefix_scale
    value_out = float(value) * factor

    # Xác định danh mục thứ nguyên
    _CATEGORIES = {
        "hartree": "Năng lượng",
        "rydberg": "Năng lượng",
        "bohr": "Độ dài",
        "barn": "Diện tích",
        "debye": "Mômen lưỡng cực",
        "electron_mass": "Khối lượng",
        "amu": "Khối lượng",
    }

    return {
        "value_in": float(value),
        "unit_in": canonical,
        "value_out": value_out,
        "unit_out": si_unit,
        "factor": factor,
        "description": description,
        "si_equivalent": f"1 {canonical} = {factor:.6e} {si_unit}",
        "category": _CATEGORIES.get(canonical, "—"),
    }


def list_semiconductor_units() -> List[Dict[str, str]]:
    """
    Trả về danh sách tất cả đơn vị bán dẫn được hỗ trợ.

    Returns
    -------
    List[dict] với các khóa: unit, si_unit, factor, description, category
    """
    _CATEGORIES = {
        "hartree": "Năng lượng",
        "rydberg": "Năng lượng",
        "bohr": "Độ dài",
        "barn": "Diện tích",
        "debye": "Mômen lưỡng cực",
        "electron_mass": "Khối lượng",
        "amu": "Khối lượng",
    }

    return [
        {
            "unit": name,
            "si_unit": info[1],
            "factor": f"{info[0]:.6e}",
            "description": info[2],
            "category": _CATEGORIES.get(name, "—"),
        }
        for name, info in SEMICONDUCTOR_CONSTANTS.items()
    ]
