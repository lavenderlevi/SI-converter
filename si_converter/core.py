#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           SI Unit Converter  ─  Công cụ Khoa học & Kỹ thuật               ║
║                                                                              ║
║  Thư viện : pint · numpy · pandas · scipy                                  ║
║  Tính năng:                                                                  ║
║    • Chuyển đổi scalar / mảng / Series / DataFrame → đơn vị SI             ║
║    • Nhận diện tự động đơn vị SI theo thứ nguyên vật lý (dimensionality)   ║
║    • Loại bỏ dữ liệu không phải số (chuỗi, NaN, None, bool, ∞, …)          ║
║    • Chuyển đổi hàng loạt (batch) trả về DataFrame                         ║
║    • Bảng hằng số vật lý (scipy.constants, đơn vị SI)                      ║
║    • Tra cứu thông tin đơn vị bất kỳ                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Cài đặt:
    pip install pint numpy pandas scipy

Ví dụ nhanh:
    >>> cvt = SIConverter()
    >>> cvt.convert(100, "degC")                     # 373.15 K
    >>> cvt.convert([60, "bad", None, 80], "mph")    # m/s (đã làm sạch)
    >>> cvt.convert_dataframe(df, "P_kPa", "kPa")   # thêm cột _SI vào df
    >>> SIConverter.show_constants()                  # bảng hằng số vật lý
"""

from __future__ import annotations

import logging
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pint
from scipy import constants as _sc

# ──────────────────────────────────────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-7s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  Pint unit registry
# ──────────────────────────────────────────────────────────────────────────────
ureg = pint.UnitRegistry()
ureg.default_format = "~P"  # ký hiệu ngắn (m, kg, s, …)
Q_ = ureg.Quantity  # constructor shorthand


# ──────────────────────────────────────────────────────────────────────────────
#  SI reference mapping
#  Dùng pint dimensionality equality (==) — không phụ thuộc thứ tự chuỗi
#  Ví dụ: [mass]*[length]/[time]² == [length]*[mass]/[time]²  → True
# ──────────────────────────────────────────────────────────────────────────────
_SI_REF_PAIRS: List[Tuple[str, str]] = [
    # (đơn vị tham chiếu, đơn vị SI ưu tiên)
    # ── 7 đơn vị cơ bản SI ───────────────────────────────────────────────────
    ("kilogram", "kilogram"),
    ("meter", "meter"),
    ("second", "second"),
    ("ampere", "ampere"),
    ("kelvin", "kelvin"),
    ("mole", "mole"),
    ("candela", "candela"),
    # ── Hình học ─────────────────────────────────────────────────────────────
    ("meter**2", "meter ** 2"),
    ("meter**3", "meter ** 3"),
    # ── Động học ─────────────────────────────────────────────────────────────
    ("kilogram/meter**3", "kilogram / meter ** 3"),
    ("meter/second", "meter / second"),
    ("meter/second**2", "meter / second ** 2"),
    # ── Cơ học ───────────────────────────────────────────────────────────────
    ("newton", "newton"),
    ("pascal", "pascal"),
    ("joule", "joule"),
    ("watt", "watt"),
    # ── Điện từ học ──────────────────────────────────────────────────────────
    ("hertz", "hertz"),
    ("coulomb", "coulomb"),
    ("volt", "volt"),
    ("ohm", "ohm"),
    ("farad", "farad"),
    ("weber", "weber"),
    ("tesla", "tesla"),
    ("henry", "henry"),
]

# Tính trước các đối tượng Quantity để so sánh nhanh
_SI_REFS_Q: List[Tuple[pint.Quantity, str]] = [
    (Q_(1, ref), si) for ref, si in _SI_REF_PAIRS
]


# ──────────────────────────────────────────────────────────────────────────────
#  Hằng số vật lý (scipy.constants, tất cả theo SI)
# ──────────────────────────────────────────────────────────────────────────────
PHYSICAL_CONSTANTS: Dict[str, Tuple[float, str, str]] = {
    "c": (_sc.c, "m/s", "Tốc độ ánh sáng trong chân không"),
    "h": (_sc.h, "J·s", "Hằng số Planck"),
    "hbar": (_sc.hbar, "J·s", "Hằng số Planck rút gọn (ħ = h/2π)"),
    "G": (_sc.G, "m³/(kg·s²)", "Hằng số hấp dẫn"),
    "e": (_sc.e, "C", "Điện tích nguyên tố"),
    "m_e": (_sc.m_e, "kg", "Khối lượng electron"),
    "m_p": (_sc.m_p, "kg", "Khối lượng proton"),
    "N_A": (_sc.N_A, "mol⁻¹", "Hằng số Avogadro"),
    "k_B": (_sc.k, "J/K", "Hằng số Boltzmann"),
    "eps0": (_sc.epsilon_0, "F/m", "Hằng số điện môi chân không (ε₀)"),
    "mu0": (_sc.mu_0, "N/A²", "Độ thấm từ chân không (μ₀)"),
    "R": (_sc.R, "J/(mol·K)", "Hằng số khí lý tưởng"),
    "sigma": (_sc.sigma, "W/(m²·K⁴)", "Hằng số Stefan–Boltzmann"),
    "eV": (_sc.eV, "J", "Electron-volt (đổi sang joule)"),
    "u": (_sc.u, "kg", "Đơn vị khối lượng nguyên tử (amu)"),
    "atm": (_sc.atm, "Pa", "Khí quyển tiêu chuẩn (đổi sang pascal)"),
}

# Đơn vị có biến đổi affine (offset) — không có hệ số nhân đơn
_AFFINE_UNITS: frozenset = frozenset(
    {
        "degc",
        "celsius",
        "degree_celsius",
        "°c",
        "degf",
        "fahrenheit",
        "degree_fahrenheit",
        "°f",
    }
)


# ══════════════════════════════════════════════════════════════════════════════
#  Hàm tiện ích cấp module (reusable, độc lập với class)
# ══════════════════════════════════════════════════════════════════════════════


def _parse_number(value: Any) -> Optional[float]:
    """
    Cố gắng ép kiểu ``value`` thành float64 thuần túy.

    Trả về ``None`` nếu không thể chuyển đổi:
        - None, NaN, ±inf
        - bool (True/False — không phải đại lượng vật lý)
        - chuỗi không chứa số hợp lệ

    Chuỗi có hậu tố đơn vị ("1.5 kPa", "3.14 m/s²") → trích số.
    Dấu trừ Unicode (−, U+2212) → được chấp nhận.

    Parameters
    ----------
    value : Any — giá trị cần kiểm tra

    Returns
    -------
    float hoặc None
    """
    if value is None:
        return None
    if isinstance(value, bool):  # bool ⊂ int — loại bỏ
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(value, str):
        s = value.strip()
        s = s.replace("\u2212", "-")  # Unicode minus → ASCII
        s = s.replace(",", ".")  # dấu phẩy thập phân
        s = re.sub(r"[^\d.\-+eE]", "", s)  # giữ ký tự số, dấu, e/E
        try:
            return float(s) if s else None
        except ValueError:
            return None
    # numpy scalar types (np.float32, np.int64, …)
    try:
        f = float(value)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def clean_numeric_input(
    data: Union[np.ndarray, pd.Series, List[Any]],
    report: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Loại bỏ phần tử không phải số khỏi ``data``.

    Các phần tử bị loại: None · NaN · ±inf · bool · chuỗi không hợp lệ.
    Chuỗi dạng "1.5 kPa" được giữ lại (trích phần số).

    Parameters
    ----------
    data   : array-like  — có thể chứa dữ liệu hỗn hợp
    report : in log tóm tắt khi True

    Returns
    -------
    clean : np.ndarray, dtype=float64 — chỉ chứa số hợp lệ
    info  : dict với các khóa:
            ``original_count`` | ``kept_count`` | ``removed_count``
            ``removed_values`` — list[(index_gốc, giá_trị_gốc)]

    Ví dụ
    -----
    >>> arr, info = clean_numeric_input([1.2, "lỗi", None, 3.4, float("nan")])
    >>> arr
    array([1.2, 3.4])
    >>> info["removed_count"]
    3
    """
    flat = list(np.asarray(data, dtype=object).flatten())
    items = [(i, _parse_number(v), v) for i, v in enumerate(flat)]

    good = [(i, num, orig) for i, num, orig in items if num is not None]
    bad = [(i, orig) for i, num, orig in items if num is None]

    clean = np.array([num for _, num, _ in good], dtype=np.float64)

    info: Dict[str, Any] = {
        "original_count": len(flat),
        "kept_count": len(good),
        "removed_count": len(bad),
        "removed_values": [(i, v) for i, v in bad],
    }

    if report:
        log.info(
            "clean_numeric_input → giữ %d/%d  |  loại bỏ %d phần tử",
            len(good),
            len(flat),
            len(bad),
        )
        if bad:
            sample = [repr(v) for _, v in bad[:6]]
            more = f"  … (+{len(bad) - 6} nữa)" if len(bad) > 6 else ""
            log.warning("Phần tử bị loại (index→value): %s%s", sample, more)

    return clean, info


def validate_unit(unit_str: str) -> bool:
    """
    Trả về True nếu ``unit_str`` được Pint nhận diện.

    Ví dụ
    -----
    >>> validate_unit("eV")       # True
    >>> validate_unit("cubits")   # False
    """
    try:
        Q_(1, unit_str)
        return True
    except pint.errors.UndefinedUnitError:
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  Lớp chuyển đổi chính
# ══════════════════════════════════════════════════════════════════════════════


class SIConverter:
    """
    Chuyển đổi scalar, mảng, Series, hoặc DataFrame từ đơn vị bất kỳ → SI.

    Quy trình nội bộ
    ----------------
    1. Làm sạch đầu vào (loại phần tử không phải số)
    2. Gắn đơn vị nguồn bằng pint
    3. Tra cứu đơn vị SI theo thứ nguyên vật lý (dimensionality equality)
    4. Chuyển đổi và trả về kết quả + metadata

    Parameters
    ----------
    verbose : in log chi tiết mỗi thao tác (mặc định: True)

    Ví dụ
    -----
    >>> cvt = SIConverter()
    >>> cvt.convert(100, "degC")
    >>> cvt.convert([60, "bad", None, 80], "mph")
    >>> cvt.convert_dataframe(df, "P_kPa", "kilopascal")
    >>> SIConverter.show_constants(filter_str="Boltzmann")
    """

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    # ── private helpers ───────────────────────────────────────────────────────

    def _log(self, msg: str, level: str = "info") -> None:
        if self.verbose:
            getattr(log, level)(msg)

    @staticmethod
    def _resolve_si_unit(q: pint.Quantity) -> Optional[str]:
        """
        Tra cứu đơn vị SI ưu tiên bằng so sánh thứ nguyên (dimensionality ==).
        Trả về None nếu không có trong bảng.
        """
        for ref_q, si_unit in _SI_REFS_Q:
            if q.dimensionality == ref_q.dimensionality:
                return si_unit
        return None

    @staticmethod
    def _is_affine(from_unit: str) -> bool:
        """True với đơn vị có biến đổi offset (°C, °F) — hệ số nhân không xác định."""
        return from_unit.strip().lower() in _AFFINE_UNITS

    # ── chuyển đổi chính ──────────────────────────────────────────────────────

    def convert(
        self,
        value: Union[float, int, np.ndarray, pd.Series, List[Any]],
        from_unit: str,
        to_unit: Optional[str] = None,
        *,
        clean: bool = True,
    ) -> Dict[str, Any]:
        """
        Chuyển đổi ``value`` từ ``from_unit`` sang ``to_unit`` (mặc định → SI).

        Parameters
        ----------
        value     : scalar hoặc array-like (có thể chứa dữ liệu bẩn)
        from_unit : chuỗi đơn vị nguồn được Pint nhận diện
                    Ví dụ: "eV", "degF", "mph", "kgf/cm**2", "angstrom"
        to_unit   : đơn vị đích; None → tự động phát hiện SI tương ứng
        clean     : loại bỏ phần tử không phải số trước khi chuyển đổi

        Returns
        -------
        dict với các khóa:
            value_in        — đầu vào số (scalar hoặc ndarray)
            unit_in         — đơn vị nguồn (Pint canonical)
            value_out       — giá trị sau chuyển đổi
            unit_out        — đơn vị SI (hoặc to_unit được chỉ định)
            factor          — hệ số nhân (None nếu đơn vị affine như °C/°F)
            dimensionality  — chuỗi thứ nguyên Pint
            is_affine       — True nếu là đơn vị offset
            clean_report    — báo cáo làm sạch (None nếu input là scalar)
            quantity        — pint.Quantity đầu ra (mang theo đơn vị)

        Raises
        ------
        ValueError  — đơn vị không hợp lệ, dữ liệu không chuyển đổi được
        """
        # ── 1. Chuẩn bị mảng số ───────────────────────────────────────────────
        clean_report: Optional[Dict] = None
        is_scalar = not isinstance(value, (list, np.ndarray, pd.Series))

        if is_scalar:
            v = _parse_number(value)
            if v is None:
                raise ValueError(
                    f"Đầu vào không phải số: {value!r}  (type={type(value).__name__})"
                )
            arr_in = np.array([v], dtype=np.float64)
        else:
            if clean:
                arr_in, clean_report = clean_numeric_input(value, report=self.verbose)
            else:
                arr_in = np.asarray(value, dtype=np.float64)

        if arr_in.size == 0:
            raise ValueError("Không còn giá trị số hợp lệ sau khi làm sạch.")

        # ── 2. Gắn đơn vị Pint ───────────────────────────────────────────────
        try:
            q_in = Q_(arr_in, from_unit)
        except pint.errors.UndefinedUnitError as exc:
            raise ValueError(
                f"Đơn vị không xác định: '{from_unit}'\n"
                f"  → Kiểm tra tại: https://pint.readthedocs.io/en/stable/user/units.html\n"
                f"  Chi tiết lỗi: {exc}"
            ) from exc

        # ── 3. Xác định đơn vị đích ───────────────────────────────────────────
        if to_unit is None:
            si_unit = self._resolve_si_unit(q_in)
            if si_unit is None:
                warnings.warn(
                    f"Chưa có mapping SI cho thứ nguyên {q_in.dimensionality}. "
                    "Dùng đơn vị cơ sở Pint (to_base_units).",
                    stacklevel=2,
                )
                q_out = q_in.to_base_units()
                to_unit = str(q_out.units)
            else:
                q_out = q_in.to(si_unit)
                to_unit = si_unit
        else:
            try:
                q_out = q_in.to(to_unit)
            except pint.errors.DimensionalityError as exc:
                raise ValueError(
                    f"Không thể chuyển đổi [{from_unit}] → [{to_unit}]\n"
                    f"  Lý do: thứ nguyên không tương thích\n"
                    f"  Chi tiết: {exc}"
                ) from exc

        # ── 4. Trích xuất kết quả ─────────────────────────────────────────────
        mag_out = q_out.magnitude
        is_affine = self._is_affine(from_unit)

        if is_scalar:
            v_in = float(arr_in[0])
            v_out = float(mag_out[0])
            factor = None if is_affine else ((v_out / v_in) if v_in != 0.0 else None)
        else:
            v_in = arr_in
            v_out = mag_out
            if is_affine:
                factor = None
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratios = np.where(arr_in != 0, mag_out / arr_in, np.nan)
                factor = float(np.nanmean(ratios))

        self._log(
            f"  {str(q_in.units):>20}  →  {str(q_out.units):<20}"
            f"  [{q_in.dimensionality}]  affine={is_affine}"
        )

        return {
            "value_in": v_in,
            "unit_in": str(q_in.units),
            "value_out": v_out,
            "unit_out": str(q_out.units),
            "factor": factor,
            "dimensionality": str(q_in.dimensionality),
            "is_affine": is_affine,
            "clean_report": clean_report,
            "quantity": q_out,
        }

    # ── pandas Series ─────────────────────────────────────────────────────────

    def convert_series(
        self,
        series: pd.Series,
        from_unit: str,
        to_unit: Optional[str] = None,
    ) -> pd.Series:
        """
        Chuyển đổi pandas Series → SI, giữ nguyên index.

        Ô không phải số → NaN trong kết quả.
        Series trả về có ``.attrs["unit"]`` chứa tên đơn vị SI.

        Parameters
        ----------
        series    : pd.Series gốc (có thể chứa giá trị hỗn hợp)
        from_unit : đơn vị của cột
        to_unit   : đơn vị đích (None → SI tự động)

        Returns
        -------
        pd.Series, dtype=float64
        """
        clean_arr, _ = clean_numeric_input(series, report=self.verbose)

        if clean_arr.size == 0:
            log.warning("Series '%s': tất cả phần tử đều không phải số.", series.name)
            out = pd.Series(np.nan, index=series.index, name=series.name)
            out.attrs["unit"] = to_unit or "unknown"
            return out

        result = self.convert(clean_arr, from_unit, to_unit, clean=False)
        numeric_mask = np.array(
            [_parse_number(v) is not None for v in series], dtype=bool
        )
        valid_idx = np.where(numeric_mask)[0]

        out = pd.Series(np.nan, index=series.index, name=series.name, dtype=float)
        out.iloc[valid_idx] = result["value_out"]
        out.attrs["unit"] = result["unit_out"]
        return out

    # ── pandas DataFrame ──────────────────────────────────────────────────────

    def convert_dataframe(
        self,
        df: pd.DataFrame,
        column: str,
        from_unit: str,
        to_unit: Optional[str] = None,
        *,
        suffix: str = "_SI",
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Thêm cột ``column + suffix`` chứa giá trị SI của ``column``.

        Ô không phải số → NaN trong cột mới.

        Parameters
        ----------
        df        : DataFrame nguồn
        column    : tên cột cần chuyển đổi
        from_unit : đơn vị của cột đó
        to_unit   : đơn vị đích (None → SI tự động)
        suffix    : hậu tố tên cột mới (mặc định ``"_SI"``)
        inplace   : sửa trực tiếp df nếu True (mặc định False)

        Returns
        -------
        pd.DataFrame — bản sao (hoặc chính df nếu inplace=True)
        """
        if column not in df.columns:
            raise KeyError(
                f"Cột '{column}' không tồn tại.\n"
                f"  Cột có trong DataFrame: {list(df.columns)}"
            )

        out_df = df if inplace else df.copy()
        si_col = column + suffix
        out_df[si_col] = self.convert_series(df[column], from_unit, to_unit)
        si_unit = out_df[si_col].attrs.get("unit", "SI")

        self._log(f"  DataFrame: '{column}' [{from_unit}] → '{si_col}' [{si_unit}]")
        return out_df

    # ── chuyển đổi hàng loạt ─────────────────────────────────────────────────

    def convert_many(
        self,
        items: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Chuyển đổi hàng loạt từ danh sách dict.

        Mỗi dict cần có:
            ``value``      : giá trị số đầu vào          (bắt buộc)
            ``from_unit``  : đơn vị nguồn                 (bắt buộc)
            ``to_unit``    : đơn vị đích — None → SI      (tùy chọn)
            ``label``      : nhãn mô tả                   (tùy chọn)

        Returns
        -------
        pd.DataFrame với các cột:
            label · value_in · unit_in · value_out · unit_out ·
            factor · dimensionality · status

        Ví dụ
        -----
        >>> cvt.convert_many([
        ...     {"label": "Band gap Si",  "value": 1.12, "from_unit": "eV"},
        ...     {"label": "Nóng chảy Al", "value": 660,  "from_unit": "degC"},
        ... ])
        """
        saved_verbose = self.verbose
        self.verbose = False  # tắt log chi tiết trong batch

        rows: List[Dict] = []
        for item in items:
            base: Dict[str, Any] = {
                "label": item.get("label", ""),
                "value_in": item.get("value"),
                "unit_in": item.get("from_unit", ""),
            }
            try:
                r = self.convert(
                    item["value"],
                    item["from_unit"],
                    item.get("to_unit"),
                )
                rows.append(
                    {
                        **base,
                        "value_out": r["value_out"],
                        "unit_out": r["unit_out"],
                        "factor": r["factor"],
                        "dimensionality": r["dimensionality"],
                        "status": "✓",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        **base,
                        "value_out": None,
                        "unit_out": item.get("to_unit", "—"),
                        "factor": None,
                        "dimensionality": "—",
                        "status": f"✗ {exc}",
                    }
                )

        self.verbose = saved_verbose
        return pd.DataFrame(
            rows,
            columns=[
                "label",
                "value_in",
                "unit_in",
                "value_out",
                "unit_out",
                "factor",
                "dimensionality",
                "status",
            ],
        )

    # ── tra cứu & hiển thị ───────────────────────────────────────────────────

    @staticmethod
    def show_constants(filter_str: str = "") -> None:
        """
        In bảng hằng số vật lý cơ bản (nguồn: scipy.constants, đơn vị SI).

        Parameters
        ----------
        filter_str : chuỗi lọc (tìm theo ký hiệu hoặc mô tả, không phân biệt hoa/thường)
                     Để trống → hiển thị tất cả

        Ví dụ
        -----
        >>> SIConverter.show_constants()
        >>> SIConverter.show_constants("Planck")
        >>> SIConverter.show_constants("nhiệt")
        """
        w = 88
        sep = "─" * w
        print(f"\n{'HẰNG SỐ VẬT LÝ CƠ BẢN  (scipy.constants | Đơn vị SI)':^{w}}")
        print(sep)
        print(f"  {'Ký hiệu':<8}  {'Giá trị':>18}  {'Đơn vị SI':<24}  Mô tả")
        print(sep)

        shown = 0
        for sym, (val, unit, desc) in PHYSICAL_CONSTANTS.items():
            if (
                not filter_str
                or filter_str.lower() in sym.lower()
                or filter_str.lower() in desc.lower()
            ):
                print(f"  {sym:<8}  {val:>18.6e}  {unit:<24}  {desc}")
                shown += 1

        if shown == 0:
            print(f"  (Không tìm thấy hằng số nào khớp với '{filter_str}')")
        print(sep + "\n")

    @staticmethod
    def unit_info(unit_str: str) -> None:
        """
        In thông tin thứ nguyên và đơn vị SI tương ứng của ``unit_str``.

        Ví dụ
        -----
        >>> SIConverter.unit_info("eV")
        >>> SIConverter.unit_info("kgf/cm**2")
        >>> SIConverter.unit_info("angstrom")
        """
        try:
            q = Q_(1, unit_str)
            si_tgt = SIConverter._resolve_si_unit(q)
            if si_tgt is None:
                si_tgt = str(q.to_base_units().units)
            print(f"  Đơn vị             : {unit_str}")
            print(f"  Pint canonical     : {q.units:~P}")
            print(f"  Thứ nguyên         : {q.dimensionality}")
            print(f"  Đơn vị SI tương ứng: {si_tgt}")
        except pint.errors.UndefinedUnitError:
            print(f"  ✗ Đơn vị '{unit_str}' không được Pint nhận diện.")


# ══════════════════════════════════════════════════════════════════════════════
#  Demo tổng hợp
# ══════════════════════════════════════════════════════════════════════════════


def _sep(title: str = "") -> None:
    w = 74
    if title:
        pad = max(0, (w - len(title) - 2) // 2)
        print(f"\n{'─' * pad} {title} {'─' * max(0, w - pad - len(title) - 2)}")
    else:
        print("─" * w)


def demo() -> None:
    """Chạy demo đầy đủ cho SIConverter."""
    print("\n" + "═" * 74)
    print("  SI Unit Converter  ─  Demo Tổng hợp")
    print("═" * 74)

    cvt = SIConverter(verbose=True)

    # ── 1. Scalar: các loại đơn vị phổ biến ─────────────────────────────────
    _sep("1 · Chuyển đổi scalar — nhiều lĩnh vực")

    scalar_cases: List[Tuple] = [
        # (giá_trị, đơn_vị_vào, đơn_vị_ra, nhãn)
        # Nhiệt độ (affine)
        (100, "degC", None, "°C → K"),
        (212, "degF", None, "°F → K"),
        (-40, "degF", None, "−40 °F → K"),
        # Chiều dài
        (1, "inch", None, "inch → m"),
        (1, "angstrom", None, "Å → m"),
        (1, "mile", None, "mile → m"),
        # Năng lượng / vật lý bán dẫn
        (1, "eV", None, "eV → J"),
        (1, "calorie", None, "cal → J"),
        (1, "kilowatt_hour", None, "kWh → J"),
        # Áp suất
        (1, "atm", None, "atm → Pa"),
        (1, "bar", None, "bar → Pa"),
        (1, "kgf/cm**2", "pascal", "kgf/cm² → Pa"),
        (760, "mmHg", None, "760 mmHg → Pa"),
        # Tốc độ
        (60, "mph", "m/s", "60 mph → m/s"),
        (1, "knot", None, "1 knot → m/s"),
        # Công suất
        (1, "horsepower", None, "1 hp → W"),
        # Điện từ
        (1, "farad", None, "1 F (đã SI)"),
        (1, "henry", None, "1 H (đã SI)"),
    ]

    rows = []
    for val, u_in, u_out, label in scalar_cases:
        r = cvt.convert(val, u_in, u_out)
        rows.append(
            {
                "Chuyển đổi": label,
                "Đầu vào": f"{val} {u_in}",
                "Kết quả SI": f"{r['value_out']:.6g} {r['unit_out']}",
                "Hệ số": f"{r['factor']:.6g}"
                if r["factor"] is not None
                else "(offset)",
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    # ── 2. Làm sạch dữ liệu không phải số ───────────────────────────────────
    _sep("2 · clean_numeric_input — loại bỏ dữ liệu không hợp lệ")

    dirty_data = [
        "1.5 kPa",  # chuỗi có đơn vị → trích số
        2.0,  # float hợp lệ
        None,  # None → loại
        "N/A",  # chuỗi chữ → loại
        float("nan"),  # NaN → loại
        "3.14",  # chuỗi thuần số → giữ
        True,  # bool → loại
        "\u22121.2",  # dấu trừ Unicode → giữ (= −1.2)
        np.inf,  # infinity → loại
        5.0,
        "sai số",  # chuỗi không có số → loại
        8.3,
        "1,5",  # dấu phẩy thập phân → giữ (= 1.5)
    ]

    print(f"\n  Đầu vào  ({len(dirty_data)} phần tử):")
    for i, v in enumerate(dirty_data):
        print(f"    [{i:2d}] {repr(v):>20}  →  {repr(_parse_number(v))}")
    clean_arr, info = clean_numeric_input(dirty_data, report=True)
    print(f"\n  Kết quả  : {clean_arr}")
    print(
        f"  Tóm tắt  : giữ {info['kept_count']}/{info['original_count']}, "
        f"loại {info['removed_count']} phần tử"
    )

    # ── 3. Mảng dữ liệu bẩn → chuyển đổi SI ─────────────────────────────────
    _sep("3 · Array bẩn → m/s (từ mph)")

    speeds_dirty = [
        60,
        "nhanh lắm",
        None,
        80,
        np.nan,
        "100 mph lol",
        True,
        120.5,
        np.inf,
        -0.0,
        "45.5 mph",
    ]
    print(f"  Đầu vào (mph): {speeds_dirty}")
    r = cvt.convert(speeds_dirty, "mph")
    print(f"  Kết quả (m/s): {np.round(r['value_out'], 4)}")
    print(
        f"  Hệ số: {r['factor']:.6f}  |  Loại bỏ: {r['clean_report']['removed_count']} phần tử"
    )

    # ── 4. pandas Series ──────────────────────────────────────────────────────
    _sep("4 · pandas Series — áp suất kPa → Pa")

    s_pressure = pd.Series(
        [101.325, "N/A", None, 200.0, np.nan, "50.5 kPa", 0.0, "-10"],
        name="P_kPa",
        dtype=object,
    )
    s_si = cvt.convert_series(s_pressure, "kilopascal")
    print(
        pd.DataFrame(
            {
                "P_kPa (gốc)": s_pressure,
                "P_Pa  (SI)  ": s_si,
            }
        ).to_string()
    )

    # ── 5. pandas DataFrame ───────────────────────────────────────────────────
    _sep("5 · pandas DataFrame — cột nhiệt độ °F → K")

    df_mat = pd.DataFrame(
        {
            "Vật liệu": ["Si", "GaAs", "GaN", "SiC", "Ge", "InP"],
            "T_nóng_chảy_°F": [2577, 2372, "err", None, 1720.3, 1830],
            "Band_gap_eV": [1.12, 1.42, 3.4, 3.26, 0.67, 1.35],
            "Ứng dụng": ["CPU", "LED", "Đèn UV", "SiC MOSFET", "Quang", "Laser"],
        }
    )
    print("  Dữ liệu gốc:")
    print(df_mat.to_string(index=False))

    df_si = cvt.convert_dataframe(df_mat, "T_nóng_chảy_°F", "degF", suffix="_K")
    df_si = cvt.convert_dataframe(df_si, "Band_gap_eV", "eV", suffix="_J")
    print("\n  Sau chuyển đổi (thêm cột _K và _J):")
    print(df_si.to_string(index=False))

    # ── 6. Batch conversion ───────────────────────────────────────────────────
    _sep("6 · Chuyển đổi hàng loạt (convert_many)")

    batch: List[Dict[str, Any]] = [
        {"label": "Band gap Si", "value": 1.12, "from_unit": "eV"},
        {"label": "Band gap GaN", "value": 3.4, "from_unit": "eV"},
        {"label": "Band gap SiC", "value": 3.26, "from_unit": "eV"},
        {"label": "Nóng chảy Al", "value": 660.3, "from_unit": "degC"},
        {"label": "Nóng chảy W", "value": 3422, "from_unit": "degC"},
        {"label": "Áp suất khí quyển", "value": 1013.25, "from_unit": "hectopascal"},
        {"label": "Tốc độ âm (không khí)", "value": 343, "from_unit": "m/s"},
        {"label": "Bước sóng Si (crit.)", "value": 1107, "from_unit": "nanometer"},
        {"label": "Khối lượng electron", "value": 9.109e-28, "from_unit": "gram"},
        {"label": "Đơn vị lỗi", "value": 1, "from_unit": "cubits"},
    ]
    summary = cvt.convert_many(batch)
    print(summary.to_string(index=False))

    # ── 7. Hằng số vật lý ────────────────────────────────────────────────────
    _sep("7 · Hằng số vật lý (scipy.constants)")
    SIConverter.show_constants()

    # ── 8. Tra cứu đơn vị ────────────────────────────────────────────────────
    _sep("8 · Tra cứu thông tin đơn vị (unit_info)")
    for u in [
        "eV",
        "kgf/cm**2",
        "degF",
        "knot",
        "angstrom",
        "horsepower",
        "mmHg",
        "nanometer",
    ]:
        print()
        SIConverter.unit_info(u)

    # ── 9. validate_unit ─────────────────────────────────────────────────────
    _sep("9 · Kiểm tra đơn vị hợp lệ (validate_unit)")
    test_units = [
        "eV",
        "meter",
        "degC",
        "mph",
        "cubits",
        "fortnight",
        "angstrom",
        "kilopascal",
        "XYZ_UNKNOWN",
    ]
    for u in test_units:
        valid = validate_unit(u)
        print(f"  {'✓' if valid else '✗'}  {u}")

    _sep()
    print("  ✓ Demo hoàn tất.\n")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo()
