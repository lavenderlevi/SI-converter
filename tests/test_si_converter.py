"""
tests/test_si_converter.py
==========================
Kiểm thử tự động cho si_converter package.

Chạy:
    pytest tests/ -v
    pytest tests/ -v --cov=si_converter --cov-report=term-missing
"""

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

# Đảm bảo import được từ thư mục dự án
sys.path.insert(0, str(Path(__file__).parent.parent))

from si_converter import (
    SIConverter,
    clean_numeric_input,
    convert,
    convert_semiconductor,
    validate_unit,
)
from si_converter.semiconductor import list_semiconductor_units

# ══════════════════════════════════════════════════════════════════════════════
#  Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def cvt():
    return SIConverter(verbose=False)


# ══════════════════════════════════════════════════════════════════════════════
#  1. convert() — public API
# ══════════════════════════════════════════════════════════════════════════════


class TestConvertPublicAPI:
    """Kiểm thử hàm convert() cấp cao nhất."""

    def test_string_ev(self):
        """convert('1 eV') → joule (pint có thể trả về 'J' hoặc 'joule')"""
        r = convert("1 eV")
        assert r["unit_out"].lower() in ("joule", "j")
        assert math.isclose(r["value_out"], 1.602176634e-19, rel_tol=1e-6)

    def test_string_degc(self):
        """convert('100 degC') → 373.15 K"""
        r = convert("100 degC")
        assert r["unit_out"].lower() in ("kelvin", "k")
        assert math.isclose(r["value_out"], 373.15, rel_tol=1e-5)

    def test_string_atm(self):
        """convert('1 atm') → pascal (pint có thể trả về 'Pa' hoặc 'pascal')"""
        r = convert("1 atm")
        assert r["unit_out"].lower() in ("pascal", "pa")
        assert math.isclose(r["value_out"], 101325.0, rel_tol=1e-5)

    def test_scalar_mph_to_ms(self):
        """convert(60, 'mph', 'm/s')"""
        r = convert(60, "mph", "m/s")
        assert math.isclose(r["value_out"], 26.8224, rel_tol=1e-4)

    def test_scalar_angstrom(self):
        """convert(1, 'angstrom') → 1e-10 m"""
        r = convert("1 angstrom")
        assert math.isclose(r["value_out"], 1e-10, rel_tol=1e-6)

    def test_scalar_atm_to_pa(self):
        """Kiểm thử ảnh yêu cầu: test_atm_to_pa"""
        r = convert(1, "atm")
        assert math.isclose(r["value_out"], 101325.0, rel_tol=1e-5)

    def test_degc_to_kelvin(self):
        """Kiểm thử ảnh yêu cầu: test_degc_to_kelvin"""
        r = convert(0, "degC")
        assert math.isclose(r["value_out"], 273.15, rel_tol=1e-5)
        r2 = convert(-273.15, "degC")
        assert math.isclose(r2["value_out"], 0.0, abs_tol=1e-6)

    def test_array_input(self):
        """Mảng số → chuyển đổi hàng loạt"""
        r = convert([1, 2, 3], "eV")
        vals = r["value_out"]
        assert len(vals) == 3
        assert math.isclose(vals[0], 1.602176634e-19, rel_tol=1e-6)
        assert math.isclose(vals[2], 3 * 1.602176634e-19, rel_tol=1e-6)

    def test_missing_from_unit_raises(self):
        """Thiếu from_unit → ValueError"""
        with pytest.raises(ValueError, match="from_unit"):
            convert(100)  # type: ignore

    def test_invalid_unit_string_raises(self):
        """Đơn vị không hợp lệ → ValueError"""
        with pytest.raises(Exception):
            convert(1, "cubits_xyz_invalid")

    def test_string_parse_scientific(self):
        """Chuỗi ký hiệu khoa học: '1.5e3 Pa'"""
        r = convert("1.5e3 Pa")
        assert math.isclose(r["value_out"], 1500.0, rel_tol=1e-5)

    def test_horsepower_to_watt(self):
        r = convert("1 horsepower")
        assert math.isclose(r["value_out"], 745.69987, rel_tol=1e-3)

    def test_kwh_to_joule(self):
        r = convert("1 kilowatt_hour")
        assert math.isclose(r["value_out"], 3.6e6, rel_tol=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
#  2. SIConverter — class methods
# ══════════════════════════════════════════════════════════════════════════════


class TestSIConverter:
    def test_ev_to_j(self, cvt):
        """Kiểm thử ảnh yêu cầu: test_ev_to_j"""
        r = cvt.convert(1, "eV")
        assert math.isclose(r["value_out"], 1.602176634e-19, rel_tol=1e-6)

    def test_atm_to_pa(self, cvt):
        r = cvt.convert(1, "atm")
        assert math.isclose(r["value_out"], 101325.0, rel_tol=1e-5)

    def test_degc_to_kelvin(self, cvt):
        r = cvt.convert(100, "degC")
        assert math.isclose(r["value_out"], 373.15, rel_tol=1e-5)

    def test_affine_is_flagged(self, cvt):
        """degC phải là is_affine=True và factor=None"""
        r = cvt.convert(100, "degC")
        assert r["is_affine"] is True
        assert r["factor"] is None

    def test_non_affine_has_factor(self, cvt):
        """eV phải có factor số học"""
        r = cvt.convert(1, "eV")
        assert r["is_affine"] is False
        assert r["factor"] is not None
        assert math.isclose(r["factor"], 1.602176634e-19, rel_tol=1e-6)

    def test_convert_series(self, cvt):
        """pandas Series chuyển đổi, ô không hợp lệ → NaN"""
        s = pd.Series([1.0, "bad", None, 2.0], dtype=object)
        result = cvt.convert_series(s, "eV")
        assert pd.notna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])
        assert pd.notna(result.iloc[3])

    def test_convert_dataframe(self, cvt):
        """DataFrame: thêm cột _SI"""
        df = pd.DataFrame({"P_kPa": [100.0, 200.0, "err", None]})
        df_out = cvt.convert_dataframe(df, "P_kPa", "kilopascal")
        assert "P_kPa_SI" in df_out.columns
        assert math.isclose(df_out["P_kPa_SI"].iloc[0], 100_000.0, rel_tol=1e-5)

    def test_convert_many(self, cvt):
        """Batch conversion trả về DataFrame đúng cột"""
        items = [
            {"label": "A", "value": 1.12, "from_unit": "eV"},
            {"label": "B", "value": 100, "from_unit": "degC"},
            {"label": "C", "value": 1, "from_unit": "INVALID_UNIT_XYZ"},
        ]
        df = cvt.convert_many(items)
        assert len(df) == 3
        assert df.loc[0, "status"] == "✓"
        assert df.loc[1, "status"] == "✓"
        assert df.loc[2, "status"].startswith("✗")

    def test_invalid_column_raises(self, cvt):
        df = pd.DataFrame({"A": [1, 2, 3]})
        with pytest.raises(KeyError):
            cvt.convert_dataframe(df, "B_nonexistent", "eV")


# ══════════════════════════════════════════════════════════════════════════════
#  3. validate_unit
# ══════════════════════════════════════════════════════════════════════════════


class TestValidateUnit:
    @pytest.mark.parametrize(
        "unit",
        [
            "eV",
            "meter",
            "degC",
            "mph",
            "angstrom",
            "kilopascal",
            "horsepower",
            "kilowatt_hour",
            "mmHg",
            "atm",
        ],
    )
    def test_valid_units(self, unit):
        assert validate_unit(unit) is True

    @pytest.mark.parametrize(
        "unit",
        [
            "cubits",
            "XYZ_INVALID",
            "fortnight_xyz",
        ],
    )
    def test_invalid_units(self, unit):
        assert validate_unit(unit) is False


# ══════════════════════════════════════════════════════════════════════════════
#  4. clean_numeric_input
# ══════════════════════════════════════════════════════════════════════════════


class TestCleanNumericInput:
    def test_removes_none(self):
        arr, info = clean_numeric_input([1.0, None, 2.0], report=False)
        assert len(arr) == 2
        assert info["removed_count"] == 1

    def test_removes_nan(self):
        arr, info = clean_numeric_input([1.0, float("nan"), 3.0], report=False)
        assert len(arr) == 2

    def test_removes_bool(self):
        arr, info = clean_numeric_input([1.0, True, False, 2.0], report=False)
        assert len(arr) == 2

    def test_removes_inf(self):
        arr, info = clean_numeric_input(
            [1.0, float("inf"), float("-inf"), 2.0], report=False
        )
        assert len(arr) == 2

    def test_removes_non_numeric_string(self):
        arr, info = clean_numeric_input(["hello", "1.5", "world"], report=False)
        assert len(arr) == 1
        assert math.isclose(arr[0], 1.5)

    def test_keeps_numeric_string_with_unit(self):
        """'1.5 kPa' → trích 1.5"""
        arr, info = clean_numeric_input(["1.5 kPa", "2.0"], report=False)
        assert len(arr) == 2

    def test_unicode_minus(self):
        """Dấu trừ Unicode (−) được chấp nhận"""
        arr, info = clean_numeric_input(["\u22121.5"], report=False)
        assert len(arr) == 1
        assert math.isclose(arr[0], -1.5)

    def test_comma_decimal(self):
        """Dấu phẩy thập phân '1,5' → 1.5"""
        arr, info = clean_numeric_input(["1,5"], report=False)
        assert len(arr) == 1
        assert math.isclose(arr[0], 1.5)

    def test_empty_array(self):
        arr, info = clean_numeric_input([], report=False)
        assert len(arr) == 0
        assert info["original_count"] == 0

    def test_all_invalid(self):
        arr, info = clean_numeric_input([None, "abc", True, float("nan")], report=False)
        assert len(arr) == 0
        assert info["removed_count"] == 4


# ══════════════════════════════════════════════════════════════════════════════
#  5. Semiconductor units
# ══════════════════════════════════════════════════════════════════════════════


class TestSemiconductor:
    def test_hartree_to_joule(self):
        r = convert_semiconductor(1, "hartree")
        # 1 Hartree ≈ 4.3597447e-18 J
        assert math.isclose(r["value_out"], 4.3597447e-18, rel_tol=1e-5)
        assert r["unit_out"] == "J"
        assert r["category"] == "Năng lượng"

    def test_rydberg_to_joule(self):
        r = convert_semiconductor(1, "rydberg")
        # 1 Ry ≈ 2.1799e-18 J (half Hartree)
        assert math.isclose(r["value_out"], 2.1799e-18, rel_tol=1e-3)
        assert r["unit_out"] == "J"

    def test_two_rydberg_equals_one_hartree(self):
        """2 Ry == 1 Hartree"""
        r_ry = convert_semiconductor(2, "rydberg")
        r_ha = convert_semiconductor(1, "hartree")
        assert math.isclose(r_ry["value_out"], r_ha["value_out"], rel_tol=1e-5)

    def test_bohr_to_meter(self):
        r = convert_semiconductor(1, "bohr")
        # 1 Bohr ≈ 5.29177e-11 m
        assert math.isclose(r["value_out"], 5.29177e-11, rel_tol=1e-4)
        assert r["unit_out"] == "m"
        assert r["category"] == "Độ dài"

    def test_debye_to_cm(self):
        r = convert_semiconductor(1, "debye")
        # 1 D ≈ 3.336e-30 C·m
        assert math.isclose(r["value_out"], 3.336e-30, rel_tol=1e-2)
        assert "C" in r["unit_out"]

    def test_barn_to_m2(self):
        r = convert_semiconductor(1, "barn")
        assert math.isclose(r["value_out"], 1e-28, rel_tol=1e-6)
        assert r["unit_out"] == "m²"

    def test_electron_mass_to_kg(self):
        r = convert_semiconductor(1, "electron_mass")
        assert math.isclose(r["value_out"], 9.10938e-31, rel_tol=1e-4)
        assert r["unit_out"] == "kg"

    def test_amu_to_kg(self):
        r = convert_semiconductor(1, "amu")
        # 1 u ≈ 1.66054e-27 kg
        assert math.isclose(r["value_out"], 1.66054e-27, rel_tol=1e-4)

    def test_alias_me(self):
        """Bí danh 'm_e' → electron_mass"""
        r = convert_semiconductor(1, "m_e")
        assert r["unit_in"] == "electron_mass"

    def test_alias_a0(self):
        """Bí danh 'a0' → bohr"""
        r = convert_semiconductor(1, "a0")
        assert r["unit_in"] == "bohr"

    def test_alias_dalton(self):
        """Bí danh 'dalton' → amu"""
        r = convert_semiconductor(1, "dalton")
        assert r["unit_in"] == "amu"

    def test_scalar_multiple(self):
        """5 hartree → 5 × 4.3597e-18 J"""
        r = convert_semiconductor(5, "hartree")
        r1 = convert_semiconductor(1, "hartree")
        assert math.isclose(r["value_out"], 5 * r1["value_out"], rel_tol=1e-9)

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError, match="không nhận diện"):
            convert_semiconductor(1, "furlong_per_fortnight")

    def test_list_units_completeness(self):
        units = list_semiconductor_units()
        names = {u["unit"] for u in units}
        assert "hartree" in names
        assert "bohr" in names
        assert "debye" in names
        assert "barn" in names
        assert "electron_mass" in names
        assert "amu" in names
        assert "rydberg" in names

    def test_factor_matches_value(self):
        """factor × value_in == value_out"""
        r = convert_semiconductor(3.5, "bohr")
        assert math.isclose(r["factor"] * r["value_in"], r["value_out"], rel_tol=1e-9)


# ══════════════════════════════════════════════════════════════════════════════
#  6. Edge cases & regression
# ══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_zero_celsius_to_kelvin(self):
        r = convert(0, "degC")
        assert math.isclose(r["value_out"], 273.15, rel_tol=1e-5)

    def test_absolute_zero_celsius(self):
        r = convert(-273.15, "degC")
        assert math.isclose(r["value_out"], 0.0, abs_tol=1e-4)

    def test_negative_value(self):
        r = convert(-10, "degC")
        assert math.isclose(r["value_out"], 263.15, rel_tol=1e-5)

    def test_large_value(self):
        r = convert(1e12, "eV")
        assert math.isclose(r["value_out"], 1e12 * 1.602176634e-19, rel_tol=1e-6)

    def test_very_small_value(self):
        r = convert(1e-15, "eV")
        assert math.isclose(r["value_out"], 1e-15 * 1.602176634e-19, rel_tol=1e-6)

    def test_dimensionality_string_present(self):
        r = convert(1, "eV")
        assert "dimensionality" in r
        assert r["dimensionality"]  # không rỗng

    def test_quantity_returned(self):
        r = convert(1, "eV")
        assert "quantity" in r

    def test_array_dirty_with_mph(self, cvt=None):
        """Mảng bẩn mph → m/s"""
        if cvt is None:
            cvt = SIConverter(verbose=False)
        r = cvt.convert([60, "N/A", None, 80, float("nan"), 100.0], "mph")
        assert r["clean_report"]["kept_count"] == 3
        assert r["clean_report"]["removed_count"] == 3
