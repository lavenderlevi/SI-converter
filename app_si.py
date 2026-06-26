#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app_si.py  —  Streamlit web app: SI Unit Converter
────────────────────────────────────────────────────
Chạy local:    streamlit run app_si.py
Deploy cloud:  streamlit.io  /  huggingface.co  /  railway.app

Yêu cầu cùng thư mục: si_converter.py
"""

import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from si_converter import convert_semiconductor
from si_converter.core import (
    _SI_REFS_Q,
    PHYSICAL_CONSTANTS,
    Q_,
    SIConverter,
    _parse_number,
    clean_numeric_input,
    validate_unit,
)
from si_converter.semiconductor import SEMICONDUCTOR_CONSTANTS, list_semiconductor_units

# Tắt DeprecationWarning từ pint (ureg.default_format cũ)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import module SI Converter
sys.path.insert(0, str(Path(__file__).parent))


# ══════════════════════════════════════════════════════════════════
#  Cấu hình trang
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SI Unit Converter",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
#  Custom CSS — giao diện phòng thí nghiệm / kỹ thuật
# ══════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
/* ── Layout ────────────────────────────────────────────────── */
.main .block-container { padding-top: 1.2rem; max-width: 1280px; }

/* ── Result card ────────────────────────────────────────────── */
.result-card {
    background: linear-gradient(135deg, #0d2137 0%, #0f3460 100%);
    border: 1px solid rgba(100, 180, 255, 0.2);
    border-radius: 14px;
    padding: 1.6rem 2rem;
    text-align: center;
    margin: 0.6rem 0 1rem;
}
.result-arrow  { font-size: 0.85rem; color: #7eb8f7; letter-spacing: .05em; margin-bottom: .3rem; }
.result-value  { font-size: 2.6rem; font-weight: 700; font-family: 'Courier New', monospace; color: #4dffc3; }
.result-unit   { font-size: 1.3rem; color: #a8d8f0; margin-left: .35rem; font-family: 'Courier New', monospace; }
.result-factor { font-size: .82rem; color: #8ab8d4; margin-top: .45rem; }
.result-dim    {
    display: inline-block;
    background: rgba(77, 255, 195, .12);
    border: 1px solid rgba(77, 255, 195, .25);
    border-radius: 20px;
    padding: 3px 14px;
    font-size: .75rem;
    color: #4dffc3;
    margin-top: .45rem;
    font-family: monospace;
}

/* ── Section label ──────────────────────────────────────────── */
.sec-label {
    font-size: .72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: #5580a0;
    border-bottom: 1px solid #dde5f0;
    padding-bottom: .25rem;
    margin: .8rem 0 .4rem;
}

/* ── Clean badges ───────────────────────────────────────────── */
.badge-ok  { background:#e6faf4; color:#059669; padding:2px 10px; border-radius:12px; font-size:.8rem; font-weight:600; }
.badge-bad { background:#fef2f2; color:#dc2626; padding:2px 10px; border-radius:12px; font-size:.8rem; font-weight:600; }

/* ── Sidebar ────────────────────────────────────────────────── */
section[data-testid="stSidebar"] { background: #0b1929; }
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] label { color: #c8dff0 !important; }
section[data-testid="stSidebar"] h2 { color: #4dffc3 !important; }
section[data-testid="stSidebar"] code { background: #1a3a5c; color: #7dcfff; }

/* ── Tab strip ───────────────────────────────────────────────── */
.stTabs [data-baseweb="tab"] { padding: 8px 18px; font-size: .87rem; border-radius: 8px 8px 0 0; }

/* ── Info note ───────────────────────────────────────────────── */
.info-note {
    background: #f0f7ff;
    border-left: 4px solid #3b82f6;
    padding: .7rem 1rem;
    border-radius: 0 8px 8px 0;
    font-size: .85rem;
    color: #1e3a5f;
    margin: .5rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════
#  Cached converter (không verbose để tránh log ra terminal)
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def _get_cvt() -> SIConverter:
    return SIConverter(verbose=False)


cvt = _get_cvt()

# ══════════════════════════════════════════════════════════════════
#  Danh mục đơn vị phổ biến (sidebar reference)
# ══════════════════════════════════════════════════════════════════
UNIT_CATALOG = {
    "📏 Chiều dài": "m  km  cm  mm  nm  angstrom  inch  ft  mile",
    "⚖️ Khối lượng": "kg  g  mg  lb  oz  u",
    "⏱️ Thời gian": "s  ms  minute  hour  day",
    "🌡️ Nhiệt độ": "degC  degF  kelvin  rankine",
    "💨 Áp suất": "Pa  kPa  MPa  bar  atm  mmHg  psi  kgf/cm**2",
    "⚡ Năng lượng": "J  kJ  MJ  eV  keV  calorie  kWh  BTU",
    "🔋 Công suất": "W  kW  MW  horsepower",
    "🚀 Tốc độ": "m/s  km/h  mph  knot",
    "🔌 Điện & Từ": "V  mV  kV  A  mA  ohm  kohm  F  nF  pF  H  mH",
    "🔬 Bán dẫn": "eV  nm  angstrom  cm**-3  S/m",
}

# ══════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚗️ SI Unit Converter")
    st.caption("Công cụ chuyển đổi đơn vị sang hệ SI — Khoa học & Kỹ thuật")
    st.markdown("---")

    st.markdown("**📚 Đơn vị phổ biến**")
    for cat, units in UNIT_CATALOG.items():
        with st.expander(cat, expanded=False):
            st.code(units)

    st.markdown("---")
    st.markdown("""
**⚙️ Thư viện**  
`pint` `numpy` `pandas` `scipy`  
`streamlit` `plotly`

**🌡️ Nhiệt độ** — Đơn vị có offset:  
`degC` và `degF` không có hệ số nhân đơn  
(°K = °C + 273.15)

**ℹ️ Cú pháp đơn vị phức hợp:**  
`kgf/cm**2` · `m/s**2` · `kg*m/s**2`
""")

# ══════════════════════════════════════════════════════════════════
#  Tiêu đề chính
# ══════════════════════════════════════════════════════════════════
st.markdown("# ⚗️ SI Unit Converter")
st.markdown(
    "Chuyển đổi đơn vị đo lường sang hệ SI — "
    "hỗ trợ làm sạch dữ liệu không hợp lệ, xử lý mảng, upload file CSV/Excel."
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════
#  Tabs
# ══════════════════════════════════════════════════════════════════
T = st.tabs(
    [
        "⚡ Scalar",
        "📊 Mảng dữ liệu",
        "📁 Upload File",
        "🔢 Batch",
        "🧲 Hằng số",
        "🔍 Tra cứu đơn vị",
        "💎 Bán dẫn chuyên sâu",
    ]
)
tab_scalar, tab_array, tab_file, tab_batch, tab_const, tab_lookup, tab_semi = T


# ──────────────────────────────────────────────────────────────────
# TAB 1 · SCALAR
# ──────────────────────────────────────────────────────────────────
with tab_scalar:
    st.subheader("Chuyển đổi giá trị đơn lẻ")

    col_form, col_result, col_examples = st.columns([5, 6, 4], gap="large")

    # ── Form ──────────────────────────────────────────────────────
    with col_form:
        st.markdown('<div class="sec-label">Đầu vào</div>', unsafe_allow_html=True)

        s_val = st.text_input(
            "Giá trị",
            value="100",
            placeholder="100  |  1.5e-10  |  -273.15",
            help="Số thực, ký hiệu khoa học, hoặc số âm",
        )
        s_from = st.text_input(
            "Đơn vị nguồn",
            value="degC",
            placeholder="eV · degC · mph · kgf/cm**2 · atm …",
            help="Xem sidebar để tham khảo danh sách đơn vị",
        )
        s_to = st.text_input(
            "Đơn vị đích (để trống → SI tự động)",
            value="",
            placeholder="kelvin · pascal · m/s …",
        )

        convert_btn = st.button(
            "🔄 Chuyển đổi", type="primary", use_container_width=True, key="btn_scalar"
        )

        # ── Quick examples ─────────────────────────────────────────
        st.markdown(
            '<div class="sec-label" style="margin-top:1.2rem">Ví dụ nhanh</div>',
            unsafe_allow_html=True,
        )
        EXAMPLES = [
            ("100", "degC", "", "100 °C → K"),
            ("1", "eV", "", "1 eV → J"),
            ("1", "atm", "", "1 atm → Pa"),
            ("60", "mph", "m/s", "60 mph → m/s"),
            ("1", "angstrom", "", "1 Å → m"),
            ("1", "horsepower", "", "1 hp → W"),
            ("1", "kgf/cm**2", "pascal", "1 kgf/cm² → Pa"),
            ("1", "kilowatt_hour", "", "1 kWh → J"),
        ]
        for raw_v, fu, tu, lbl in EXAMPLES:
            if st.button(lbl, use_container_width=True, key=f"ex_{fu}_{tu}"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        r = cvt.convert(float(raw_v), fu, tu or None)
                        st.session_state["sc_result"] = r
                        st.session_state["sc_input"] = (float(raw_v), fu)
                    except Exception as exc:
                        st.error(str(exc))

    # ── Trigger from main button ───────────────────────────────────
    if convert_btn:
        parsed = _parse_number(s_val)
        if parsed is None:
            st.error(f"❌ Giá trị `{s_val}` không phải số hợp lệ.")
        elif not validate_unit(s_from):
            st.error(
                f"❌ Đơn vị `{s_from}` không được Pint nhận diện.\n\n"
                "→ Kiểm tra sidebar hoặc docs.pint: https://pint.readthedocs.io"
            )
        elif s_to and not validate_unit(s_to):
            st.error(f"❌ Đơn vị đích `{s_to}` không hợp lệ.")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    r = cvt.convert(parsed, s_from, s_to or None)
                    st.session_state["sc_result"] = r
                    st.session_state["sc_input"] = (parsed, s_from)
                except Exception as exc:
                    st.error(f"❌ {exc}")

    # ── Result ─────────────────────────────────────────────────────
    with col_result:
        st.markdown('<div class="sec-label">Kết quả</div>', unsafe_allow_html=True)

        if "sc_result" in st.session_state:
            r = st.session_state["sc_result"]
            v_raw, u_raw = st.session_state["sc_input"]
            factor_txt = (
                f"× {r['factor']:.6g}"
                if r["factor"] is not None
                else "biến đổi offset (không có hệ số nhân đơn)"
            )
            st.markdown(
                f"""
            <div class="result-card">
                <div class="result-arrow">{v_raw} {u_raw} &nbsp;→</div>
                <div>
                    <span class="result-value">{r["value_out"]:.6g}</span>
                    <span class="result-unit">{r["unit_out"]}</span>
                </div>
                <div class="result-factor">Hệ số chuyển đổi: {factor_txt}</div>
                <div><span class="result-dim">{r["dimensionality"]}</span></div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            m1, m2 = st.columns(2)
            m1.metric("Đơn vị nguồn (Pint)", r["unit_in"])
            m2.metric("Đơn vị SI đích", r["unit_out"])
            if r["factor"]:
                st.metric("Hệ số nhân", f"{r['factor']:.8g}")
        else:
            st.markdown(
                """
            <div class="info-note">
            ← Nhập giá trị và đơn vị rồi nhấn <b>Chuyển đổi</b>,<br>
            hoặc chọn một ví dụ nhanh bên trái.
            </div>""",
                unsafe_allow_html=True,
            )

    # ── Bảng tham khảo nhanh ──────────────────────────────────────
    with col_examples:
        st.markdown(
            '<div class="sec-label">Bảng tham khảo</div>', unsafe_allow_html=True
        )
        REF = [
            ("inch", "in → m", "0.0254"),
            ("foot", "ft → m", "0.3048"),
            ("mile", "mi → m", "1609.34"),
            ("angstrom", "Å → m", "1×10⁻¹⁰"),
            ("pound", "lb → kg", "0.45359"),
            ("atm", "atm → Pa", "101325"),
            ("bar", "bar → Pa", "100000"),
            ("calorie", "cal → J", "4.184"),
            ("horsepower", "hp → W", "745.70"),
            ("mph", "mph → m/s", "0.44704"),
            ("knot", "kn → m/s", "0.51444"),
            ("kilowatt_hour", "kWh → J", "3.6×10⁶"),
            ("eV", "eV → J", "1.602×10⁻¹⁹"),
        ]
        ref_df = pd.DataFrame(REF, columns=["Đơn vị", "Chuyển đổi", "Hệ số"])
        st.dataframe(ref_df, use_container_width=True, hide_index=True, height=460)


# ──────────────────────────────────────────────────────────────────
# TAB 2 · MẢNG DỮ LIỆU
# ──────────────────────────────────────────────────────────────────
with tab_array:
    st.subheader("Chuyển đổi mảng — làm sạch dữ liệu tự động")
    st.markdown(
        """
    <div class="info-note">
    Dán dữ liệu thô (có thể lẫn chuỗi, N/A, ô trống).
    Module tự <b>phát hiện và loại bỏ</b> giá trị không phải số trước khi chuyển đổi.
    </div>""",
        unsafe_allow_html=True,
    )

    a_col1, a_col2 = st.columns([1, 1], gap="large")

    with a_col1:
        st.markdown('<div class="sec-label">Đầu vào</div>', unsafe_allow_html=True)
        arr_raw = st.text_area(
            "Dán dữ liệu (mỗi dòng một giá trị, hoặc phân cách bằng dấu phẩy)",
            value="60\n72.5\nN/A\n80\n\n100\nnull\n120.5\n−45\n1.5e2",
            height=230,
            help="Chấp nhận: số nguyên, số thực, ký hiệu khoa học.\n"
            "Loại bỏ: chuỗi chữ, None, NaN, True/False, ±∞",
        )
        aa1, aa2 = st.columns(2)
        with aa1:
            arr_from = st.text_input("Đơn vị nguồn", value="mph", key="arr_from")
        with aa2:
            arr_to = st.text_input(
                "Đơn vị đích", value="m/s", key="arr_to", help="Để trống → SI tự động"
            )

        arr_btn = st.button(
            "🔄 Chuyển đổi & Làm sạch", type="primary", use_container_width=True
        )

    # ── Kết quả ───────────────────────────────────────────────────
    with a_col2:
        st.markdown(
            '<div class="sec-label">Kết quả & Báo cáo làm sạch</div>',
            unsafe_allow_html=True,
        )

        if arr_btn:
            raw_list = [
                v.strip()
                for line in arr_raw.replace(",", "\n").splitlines()
                for v in [line.strip()]
                if v
            ]
            err_msg = ""
            if not raw_list:
                err_msg = "Chưa có dữ liệu đầu vào."
            elif not validate_unit(arr_from):
                err_msg = f"Đơn vị `{arr_from}` không hợp lệ."
            elif arr_to and not validate_unit(arr_to):
                err_msg = f"Đơn vị đích `{arr_to}` không hợp lệ."

            if err_msg:
                st.error(f"❌ {err_msg}")
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        r = cvt.convert(raw_list, arr_from, arr_to or None)
                        st.session_state["arr_result"] = (r, raw_list)
                    except Exception as exc:
                        st.error(f"❌ {exc}")

        if "arr_result" in st.session_state:
            r, raw_list = st.session_state["arr_result"]
            rpt = r["clean_report"]
            removed = rpt["removed_values"]

            # Badges
            b1, b2, b3 = st.columns(3)
            b1.metric("📥 Tổng đầu vào", rpt["original_count"])
            b2.metric("✅ Giữ lại", rpt["kept_count"])
            b3.metric(
                "🗑️ Đã loại bỏ",
                rpt["removed_count"],
                delta=f"-{rpt['removed_count']}" if rpt["removed_count"] else None,
                delta_color="inverse" if rpt["removed_count"] else "off",
            )

            if removed:
                with st.expander(f"📋 Chi tiết {len(removed)} phần tử bị loại"):
                    for idx, val in removed:
                        reason = (
                            "NaN/inf"
                            if isinstance(val, float)
                            else "None"
                            if val is None
                            else "bool"
                            if isinstance(val, bool)
                            else "chuỗi"
                        )
                        st.markdown(f"- `[{idx}]`  `{repr(val)}`  → _{reason}_")

            # Bảng kết quả
            clean_arr, _ = clean_numeric_input(raw_list, report=False)
            v_out = r["value_out"]
            df_arr = pd.DataFrame(
                {
                    f"Đầu vào ({arr_from})": np.round(clean_arr, 6),
                    f"Kết quả ({r['unit_out']})": np.round(v_out, 6),
                }
            )
            st.dataframe(df_arr, use_container_width=True, height=200)

            csv_arr = df_arr.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Tải CSV",
                csv_arr,
                "array_converted.csv",
                "text/csv",
                use_container_width=True,
            )

    # ── Biểu đồ ───────────────────────────────────────────────────
    if "arr_result" in st.session_state:
        r, _ = st.session_state["arr_result"]
        clean_arr, _ = clean_numeric_input(
            [
                v.strip()
                for line in arr_raw.replace(",", "\n").splitlines()
                for v in [line.strip()]
                if v
            ],
            report=False,
        )
        v_out = r["value_out"]
        idx = list(range(len(clean_arr)))

        st.markdown("---")
        st.markdown("**Biểu đồ so sánh trước / sau chuyển đổi**")
        ch1, ch2 = st.columns(2)

        with ch1:
            fig_in = px.bar(
                x=idx,
                y=clean_arr,
                labels={"x": "Index", "y": arr_from},
                title=f"Đầu vào [{arr_from}]",
                color_discrete_sequence=["#457b9d"],
            )
            fig_in.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig_in, use_container_width=True)

        with ch2:
            fig_out = px.bar(
                x=idx,
                y=v_out,
                labels={"x": "Index", "y": r["unit_out"]},
                title=f"Kết quả SI [{r['unit_out']}]",
                color_discrete_sequence=["#06d6a0"],
            )
            fig_out.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10))
            st.plotly_chart(fig_out, use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# TAB 3 · UPLOAD FILE
# ──────────────────────────────────────────────────────────────────
with tab_file:
    st.subheader("Chuyển đổi cột trong file CSV / Excel")

    uploaded = st.file_uploader(
        "Kéo thả hoặc chọn file",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )

    # ── Demo template nếu chưa upload ─────────────────────────────
    if uploaded is None:
        st.markdown(
            """
        <div class="info-note">
        Chưa có file. Tải file mẫu bên dưới để thử ngay:
        </div>""",
            unsafe_allow_html=True,
        )

        demo_df = pd.DataFrame(
            {
                "material": ["Si", "GaAs", "GaN", "SiC", "Ge", "InP", "AlN"],
                "band_gap_eV": [1.12, 1.42, 3.4, 3.26, 0.67, 1.35, 6.2],
                "melt_temp_degF": [2577, 2372, "N/A", 5036, 1720, 1830, 4532],
                "density_g_cm3": [2.33, 5.32, 6.15, 3.22, 5.32, 4.81, 3.26],
                "mobility_cm2_Vs": [1500, 8500, 1200, 900, 3900, 4600, 300],
            }
        )
        demo_csv = demo_df.to_csv(index=False).encode("utf-8")
        dc1, dc2 = st.columns([2, 5])
        dc1.download_button(
            "⬇️ Tải file mẫu CSV",
            demo_csv,
            "sample_semiconductors.csv",
            "text/csv",
        )
        st.dataframe(demo_df, use_container_width=True)

    else:
        # ── Đọc file ───────────────────────────────────────────────
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_raw = pd.read_csv(uploaded)
            else:
                df_raw = pd.read_excel(uploaded)

            st.success(
                f"✅ **{uploaded.name}** — "
                f"{len(df_raw):,} hàng × {len(df_raw.columns)} cột"
            )

            with st.expander("📋 Xem trước dữ liệu gốc", expanded=True):
                st.dataframe(df_raw.head(10), use_container_width=True)

            st.markdown("---")
            st.markdown("### ⚙️ Cấu hình chuyển đổi")

            fc1, fc2, fc3, fc4 = st.columns([3, 2, 2, 1])
            with fc1:
                sel_col = st.selectbox("Cột cần chuyển đổi", df_raw.columns.tolist())
            with fc2:
                file_from = st.text_input(
                    "Đơn vị của cột", placeholder="eV · degF · kPa · g/cm**3"
                )
            with fc3:
                file_to = st.text_input("Đơn vị đích (để trống → SI tự động)", value="")
            with fc4:
                file_sfx = st.text_input("Hậu tố cột", value="_SI")

            if st.button("🔄 Chuyển đổi cột", type="primary"):
                if not file_from.strip():
                    st.warning("Vui lòng nhập đơn vị của cột.")
                elif not validate_unit(file_from.strip()):
                    st.error(f"❌ Đơn vị `{file_from}` không hợp lệ.")
                elif file_to.strip() and not validate_unit(file_to.strip()):
                    st.error(f"❌ Đơn vị đích `{file_to}` không hợp lệ.")
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            df_out = cvt.convert_dataframe(
                                df_raw,
                                sel_col,
                                file_from.strip(),
                                file_to.strip() or None,
                                suffix=file_sfx,
                            )
                            si_col = sel_col + file_sfx
                            si_unit = df_out[si_col].attrs.get("unit", "SI")
                            n_ok = int(df_out[si_col].notna().sum())
                            n_nan = int(df_out[si_col].isna().sum())

                            fm1, fm2, fm3 = st.columns(3)
                            fm1.metric("✅ Chuyển đổi thành công", n_ok)
                            fm2.metric("⚠️ Không hợp lệ (NaN)", n_nan)
                            fm3.metric("Đơn vị SI đích", si_unit)

                            st.dataframe(df_out, use_container_width=True)

                            # Tải về
                            dl1, dl2 = st.columns(2)
                            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                            base_name = uploaded.name.rsplit(".", 1)[0]
                            dl1.download_button(
                                "⬇️ Tải CSV",
                                csv_bytes,
                                f"{base_name}_SI.csv",
                                "text/csv",
                                use_container_width=True,
                            )
                            xlsx_buf = io.BytesIO()
                            with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as w:
                                df_out.to_excel(w, index=False)
                            dl2.download_button(
                                "⬇️ Tải Excel",
                                xlsx_buf.getvalue(),
                                f"{base_name}_SI.xlsx",
                                use_container_width=True,
                            )

                        except Exception as exc:
                            st.error(f"❌ {exc}")

        except Exception as exc:
            st.error(f"❌ Không đọc được file: {exc}")


# ──────────────────────────────────────────────────────────────────
# TAB 4 · BATCH
# ──────────────────────────────────────────────────────────────────
with tab_batch:
    st.subheader("Chuyển đổi hàng loạt")
    st.markdown(
        "Nhập nhiều phép chuyển đổi khác nhau vào bảng bên dưới (có thể thêm/xóa hàng)."
    )

    if "batch_df" not in st.session_state:
        st.session_state["batch_df"] = pd.DataFrame(
            {
                "label": [
                    "Band gap Si",
                    "Band gap GaN",
                    "Band gap SiC",
                    "Nóng chảy Al",
                    "Áp suất chuẩn",
                    "Tốc độ âm (không khí)",
                    "Bước sóng hấp thụ Si",
                ],
                "value": [1.12, 3.4, 3.26, 660.3, 1013.25, 343.0, 1107.0],
                "from_unit": [
                    "eV",
                    "eV",
                    "eV",
                    "degC",
                    "hectopascal",
                    "m/s",
                    "nanometer",
                ],
                "to_unit": ["", "", "", "", "", "", "meter"],
            }
        )

    edited = st.data_editor(
        st.session_state["batch_df"],
        use_container_width=True,
        num_rows="dynamic",
        key="batch_editor",
        column_config={
            "label": st.column_config.TextColumn("Nhãn / Mô tả", width="large"),
            "value": st.column_config.NumberColumn("Giá trị", format="%.6g"),
            "from_unit": st.column_config.TextColumn("Đơn vị nguồn", width="medium"),
            "to_unit": st.column_config.TextColumn(
                "Đơn vị đích (trống→SI)", width="medium"
            ),
        },
    )
    st.session_state["batch_df"] = edited

    if st.button("🔄 Chuyển đổi tất cả", type="primary"):
        items = []
        for _, row in edited.iterrows():
            if pd.notna(row.get("value")) and str(row.get("from_unit", "")).strip():
                items.append(
                    {
                        "label": str(row.get("label", "")),
                        "value": float(row["value"]),
                        "from_unit": str(row["from_unit"]).strip(),
                        "to_unit": str(row.get("to_unit", "")).strip() or None,
                    }
                )

        if not items:
            st.warning("Không có hàng hợp lệ nào.")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res_df = cvt.convert_many(items)

            ok_df = res_df[res_df["status"] == "✓"]
            err_df = res_df[res_df["status"] != "✓"]

            bm1, bm2, bm3 = st.columns(3)
            bm1.metric("📋 Tổng", len(res_df))
            bm2.metric("✅ Thành công", len(ok_df))
            bm3.metric("❌ Lỗi", len(err_df))

            # Hiển thị kết quả với màu sắc
            def _style_row(row):
                color = "" if row["status"] == "✓" else "background-color:#fff0f0"
                return [color] * len(row)

            st.dataframe(
                res_df.style.apply(_style_row, axis=1).format(
                    {
                        "value_in": lambda x: f"{x:.6g}" if pd.notna(x) else "—",
                        "value_out": lambda x: f"{x:.6g}" if pd.notna(x) else "—",
                        "factor": lambda x: f"{x:.4g}" if pd.notna(x) else "—",
                    }
                ),
                use_container_width=True,
            )

            csv_b = res_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Tải kết quả CSV",
                csv_b,
                "batch_result.csv",
                "text/csv",
            )

            if len(ok_df) > 1:
                st.markdown("---")
                st.markdown("**Biểu đồ kết quả batch (bar chart)**")
                fig_b = px.bar(
                    ok_df.dropna(subset=["value_out"]),
                    x="label",
                    y="value_out",
                    color="unit_out",
                    hover_data=["value_in", "unit_in", "factor"],
                    title="Giá trị sau chuyển đổi SI",
                    labels={"value_out": "Giá trị SI", "label": ""},
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                )
                fig_b.update_layout(
                    height=360,
                    legend_title="Đơn vị SI",
                    margin=dict(t=50, b=60, l=10, r=10),
                )
                fig_b.update_xaxes(tickangle=-30)
                st.plotly_chart(fig_b, use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# TAB 5 · HẰNG SỐ VẬT LÝ
# ──────────────────────────────────────────────────────────────────
with tab_const:
    st.subheader("Hằng số vật lý cơ bản (scipy.constants | Đơn vị SI)")

    search_c = st.text_input(
        "🔍 Tìm kiếm",
        placeholder="Planck · Boltzmann · electron · ánh sáng …",
        label_visibility="collapsed",
    )

    rows_c = [
        {"Ký hiệu": sym, "Giá trị (SI)": val, "Đơn vị": unit, "Mô tả": desc}
        for sym, (val, unit, desc) in PHYSICAL_CONSTANTS.items()
        if not search_c
        or search_c.lower() in sym.lower()
        or search_c.lower() in desc.lower()
    ]

    if not rows_c:
        st.info(f"Không tìm thấy hằng số nào khớp với `{search_c}`.")
    else:
        const_df = pd.DataFrame(rows_c)
        st.dataframe(
            const_df.style.format({"Giá trị (SI)": "{:.8e}"}),
            use_container_width=True,
            height=480,
        )

        # Bảng dạng highlight
        if len(rows_c) > 1:
            st.markdown("---")
            st.markdown("**Trực quan (thang log) — thể hiện độ lớn tương đối**")
            fig_c = px.bar(
                const_df,
                x="Ký hiệu",
                y="Giá trị (SI)",
                hover_data=["Mô tả", "Đơn vị"],
                log_y=True,
                color="Giá trị (SI)",
                color_continuous_scale="Plasma",
                title="Giá trị hằng số vật lý (log scale)",
                labels={"Giá trị (SI)": "Giá trị (log SI)"},
            )
            fig_c.update_layout(
                height=380,
                coloraxis_showscale=False,
                margin=dict(t=50, b=50, l=10, r=10),
            )
            st.plotly_chart(fig_c, use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# TAB 6 · TRA CỨU ĐƠN VỊ
# ──────────────────────────────────────────────────────────────────
with tab_lookup:
    st.subheader("Tra cứu & Kiểm tra đơn vị")

    lu1, lu2 = st.columns([1, 1], gap="large")

    # ── Tra cứu chi tiết ─────────────────────────────────────────
    with lu1:
        st.markdown("#### 🔎 Tra cứu chi tiết")
        lookup_u = st.text_input(
            "Nhập chuỗi đơn vị",
            value="eV",
            placeholder="eV · kgf/cm**2 · degF · knot · angstrom",
        )
        if st.button("Tra cứu", type="primary", use_container_width=True):
            if not validate_unit(lookup_u):
                st.error(
                    f"❌ Đơn vị `{lookup_u}` **không** được Pint nhận diện.\n\n"
                    "Gợi ý: kiểm tra chính tả, thử tên tiếng Anh đầy đủ "
                    "(ví dụ `kilogram` thay vì `Kg`)."
                )
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        q = Q_(1.0, lookup_u)
                        si_tgt = next(
                            (
                                s
                                for rq, s in _SI_REFS_Q
                                if q.dimensionality == rq.dimensionality
                            ),
                            str(q.to_base_units().units),
                        )
                        r1 = cvt.convert(1.0, lookup_u)
                        r0 = cvt.convert(0.0, lookup_u)

                        st.success("✅ Đơn vị hợp lệ (được Pint nhận diện)")

                        li1, li2 = st.columns(2)
                        li1.metric("Pint canonical", str(q.units))
                        li2.metric("SI tương ứng", si_tgt)

                        st.info(f"**Thứ nguyên:** `{q.dimensionality}`")
                        st.info(
                            f"**1 {lookup_u}** = **{r1['value_out']:.6g} {r1['unit_out']}**"
                        )
                        st.info(
                            f"**0 {lookup_u}** = **{r0['value_out']:.6g} {r0['unit_out']}**"
                        )

                        if r1["factor"] is not None:
                            st.info(f"**Hệ số nhân:** `{r1['factor']:.8g}`")
                        else:
                            st.warning(
                                "⚠️ Đây là đơn vị **offset (affine)** — "
                                "không có hệ số nhân cố định.\n\n"
                                f"Ví dụ: 0 {lookup_u} ≠ 0 {r0['unit_out']} "
                                f"(thực ra = {r0['value_out']:.2f} {r0['unit_out']})"
                            )
                    except Exception as exc:
                        st.error(f"❌ {exc}")

        st.markdown("---")
        st.markdown("#### ✅ Kiểm tra hàng loạt")
        validate_bulk = st.text_area(
            "Nhập các đơn vị (mỗi dòng một đơn vị)",
            value="eV\nmph\ndegC\nangstrom\ncubits\nkgf/cm**2\nXYZ_INVALID\nkilopascal\nhorsepower\nfortnight",
            height=200,
        )
        if st.button("Kiểm tra tất cả", use_container_width=True):
            to_check = [u.strip() for u in validate_bulk.splitlines() if u.strip()]
            if to_check:
                check_rows = []
                for u in to_check:
                    valid = validate_unit(u)
                    canonical = "—"
                    dim = "—"
                    if valid:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            q = Q_(1.0, u)
                            canonical = str(q.units)
                            dim = str(q.dimensionality)
                    check_rows.append(
                        {
                            "Đơn vị nhập": u,
                            "Hợp lệ": "✅" if valid else "❌",
                            "Canonical": canonical,
                            "Thứ nguyên": dim,
                        }
                    )

                result_check = pd.DataFrame(check_rows)
                st.dataframe(
                    result_check.style.apply(
                        lambda row: (
                            [""] * len(row)
                            if row["Hợp lệ"] == "✅"
                            else ["background-color:#fff0f0"] * len(row)
                        ),
                        axis=1,
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    # ── Bảng tham khảo tổng hợp ──────────────────────────────────
    with lu2:
        st.markdown("#### 📖 Bảng quy đổi tham khảo (giá trị = 1)")

        ref_conversions = [
            # (giá_trị, đơn_vị, to, nhãn)
            (1, "inch", None, "Chiều dài"),
            (1, "foot", None, "Chiều dài"),
            (1, "yard", None, "Chiều dài"),
            (1, "mile", None, "Chiều dài"),
            (1, "angstrom", None, "Chiều dài"),
            (1, "nanometer", None, "Chiều dài"),
            (1, "pound", None, "Khối lượng"),
            (1, "ounce", None, "Khối lượng"),
            (1, "u", None, "Khối lượng"),
            (1, "atm", None, "Áp suất"),
            (1, "bar", None, "Áp suất"),
            (1, "psi", None, "Áp suất"),
            (1, "mmHg", None, "Áp suất"),
            (1, "eV", None, "Năng lượng"),
            (1, "calorie", None, "Năng lượng"),
            (1, "kilowatt_hour", None, "Năng lượng"),
            (1, "horsepower", None, "Công suất"),
            (1, "mph", "m/s", "Tốc độ"),
            (1, "knot", None, "Tốc độ"),
            (0, "degC", None, "Nhiệt độ"),
            (100, "degC", None, "Nhiệt độ"),
            (0, "degF", None, "Nhiệt độ"),
        ]

        ref_rows_lu = []
        for val, fu, tu, cat in ref_conversions:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    r = cvt.convert(float(val), fu, tu)
                    ref_rows_lu.append(
                        {
                            "Danh mục": cat,
                            "Biểu thức": f"{val} {fu}",
                            "= Giá trị": f"{r['value_out']:.6g}",
                            "Đơn vị SI": r["unit_out"],
                        }
                    )
                except Exception:
                    pass

        st.dataframe(
            pd.DataFrame(ref_rows_lu),
            use_container_width=True,
            height=580,
            hide_index=True,
        )


# ──────────────────────────────────────────────────────────────────
# TAB 7 · BÁN DẪN CHUYÊN SÂU
# ──────────────────────────────────────────────────────────────────
with tab_semi:
    st.subheader("💎 Vật liệu bán dẫn chuyên sâu — Đơn vị nguyên tử & hạt nhân")
    st.markdown(
        """
    <div class="info-note">
    Module chuyên biệt cho vật lý bán dẫn và hóa lý lượng tử: chuyển đổi các đơn vị
    <b>nguyên tử</b> (Hartree, Rydberg, Bohr) và <b>hạt nhân</b> (barn) sang SI.
    </div>""",
        unsafe_allow_html=True,
    )

    # ── Bảng đơn vị tham khảo ──────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Đơn vị được hỗ trợ")

    units_data = list_semiconductor_units()
    units_df = pd.DataFrame(units_data)
    units_df.columns = ["Đơn vị", "Đơn vị SI", "Hệ số (SI)", "Mô tả", "Danh mục"]

    # Nhóm theo danh mục
    for cat in ["Năng lượng", "Độ dài", "Mômen lưỡng cực", "Diện tích", "Khối lượng"]:
        sub = units_df[units_df["Danh mục"] == cat]
        if not sub.empty:
            st.markdown(f"**{cat}**")
            st.dataframe(
                sub[["Đơn vị", "Đơn vị SI", "Hệ số (SI)", "Mô tả"]],
                use_container_width=True,
                hide_index=True,
                height=min(80 + len(sub) * 45, 250),
            )

    st.markdown("---")
    st.markdown("### ⚙️ Chuyển đổi")

    sc1, sc2 = st.columns([1, 1], gap="large")

    with sc1:
        st.markdown('<div class="sec-label">Đầu vào</div>', unsafe_allow_html=True)

        # Chọn đơn vị từ dropdown
        SEMI_UNIT_OPTIONS = {
            "Hartree (Eₕ) — Năng lượng nguyên tử": "hartree",
            "Rydberg (Ry) — Năng lượng ion hóa": "rydberg",
            "Bohr radius (a₀) — Bán kính nguyên tử": "bohr",
            "Debye (D) — Mômen lưỡng cực": "debye",
            "Electron mass (mₑ)": "electron_mass",
            "Atomic mass unit (u / Da)": "amu",
            "Barn (b) — Tiết diện hạt nhân": "barn",
        }

        semi_unit_label = st.selectbox(
            "Chọn đơn vị nguồn",
            list(SEMI_UNIT_OPTIONS.keys()),
            index=0,
        )
        semi_unit = SEMI_UNIT_OPTIONS[semi_unit_label]

        semi_val_str = st.text_input(
            "Giá trị",
            value="1.0",
            placeholder="1.0  |  2.5  |  13.6",
        )

        semi_btn = st.button(
            "🔄 Chuyển đổi", type="primary", use_container_width=True, key="btn_semi"
        )

        # Ví dụ nhanh
        st.markdown(
            '<div class="sec-label" style="margin-top:1rem">Ví dụ nhanh</div>',
            unsafe_allow_html=True,
        )
        SEMI_EXAMPLES = [
            (1.0, "hartree", "1 Hartree → J   (≈ 27.211 eV)"),
            (1.0, "rydberg", "1 Rydberg → J   (≈ 13.606 eV)"),
            (1.0, "bohr", "1 Bohr → m      (≈ 0.529 Å)"),
            (2.5, "debye", "2.5 Debye → C·m"),
            (1.0, "electron_mass", "1 mₑ → kg"),
            (12.0, "amu", "12 u → kg        (¹²C)"),
            (100.0, "barn", "100 barn → m²   (tiết diện lớn)"),
        ]
        for ex_val, ex_unit, ex_lbl in SEMI_EXAMPLES:
            if st.button(
                ex_lbl, use_container_width=True, key=f"semi_ex_{ex_unit}_{ex_val}"
            ):
                try:
                    r_ex = convert_semiconductor(ex_val, ex_unit)
                    st.session_state["semi_result"] = r_ex
                except Exception as e:
                    st.error(str(e))

    with sc2:
        st.markdown('<div class="sec-label">Kết quả</div>', unsafe_allow_html=True)

        # Trigger từ nút chính
        if semi_btn:
            try:
                val_f = float(semi_val_str.replace(",", "."))
                r_semi = convert_semiconductor(val_f, semi_unit)
                st.session_state["semi_result"] = r_semi
            except ValueError as e:
                st.error(f"❌ {e}")

        if "semi_result" in st.session_state:
            r = st.session_state["semi_result"]

            # Card kết quả
            st.markdown(
                f"""
            <div class="result-card">
                <div class="result-arrow">{r["value_in"]} {r["unit_in"]} &nbsp;→</div>
                <div>
                    <span class="result-value">{r["value_out"]:.6e}</span>
                    <span class="result-unit">{r["unit_out"]}</span>
                </div>
                <div class="result-factor">Hệ số: {r["factor"]:.6e}</div>
                <div><span class="result-dim">{r["category"]}</span></div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Metrics
            m1, m2 = st.columns(2)
            m1.metric("Đơn vị nguồn", r["unit_in"])
            m2.metric("Đơn vị SI", r["unit_out"])
            st.metric("Hệ số chuyển đổi", f"{r['factor']:.8e}")

            st.info(f"**Tương đương SI:** `{r['si_equivalent']}`")
            st.caption(f"📖 {r['description']}")

        else:
            st.markdown(
                """
            <div class="info-note">
            ← Chọn đơn vị và nhập giá trị rồi nhấn <b>Chuyển đổi</b>,<br>
            hoặc chọn một ví dụ nhanh bên trái.
            </div>""",
                unsafe_allow_html=True,
            )

    # ── So sánh trực quan ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 So sánh trực quan — Độ lớn hệ số chuyển đổi (log scale)")

    chart_data = []
    for u_info in list_semiconductor_units():
        chart_data.append(
            {
                "Đơn vị": u_info["unit"],
                "Hệ số SI": float(u_info["factor"]),
                "Danh mục": u_info["category"],
                "Mô tả": u_info["description"],
            }
        )
    chart_df = pd.DataFrame(chart_data)

    fig_semi = px.bar(
        chart_df,
        x="Đơn vị",
        y="Hệ số SI",
        color="Danh mục",
        log_y=True,
        hover_data=["Mô tả"],
        title="Hệ số chuyển đổi sang SI (thang logarithm)",
        labels={"Hệ số SI": "Hệ số (SI) — log scale"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_semi.update_layout(
        height=380,
        margin=dict(t=50, b=40, l=10, r=10),
        legend_title="Danh mục",
    )
    st.plotly_chart(fig_semi, use_container_width=True)

    # ── Bảng tham chiếu vật lý bán dẫn ──────────────────────────
    st.markdown("---")
    st.markdown("### 🔬 Bảng tra nhanh — Năng lượng vật liệu bán dẫn thông dụng")

    mat_data = [
        ("Si", 1.12, "eV", "hartree", "CPU, pin mặt trời"),
        ("GaAs", 1.42, "eV", "hartree", "LED đỏ, laser"),
        ("GaN", 3.4, "eV", "hartree", "LED xanh, UV"),
        ("SiC", 3.26, "eV", "hartree", "MOSFET công suất cao"),
        ("Ge", 0.67, "eV", "hartree", "Transistor cổ điển"),
        ("InP", 1.35, "eV", "hartree", "Laser viễn thông"),
        ("AlN", 6.2, "eV", "hartree", "UV sâu"),
        ("ZnO", 3.37, "eV", "hartree", "Cảm biến khí"),
    ]

    mat_rows = []
    for mat, eg_ev, unit_in, unit_out_name, app in mat_data:
        try:
            # eV → J
            r_j = cvt.convert(eg_ev, "eV")
            # eV → Hartree (chia cho hệ số Hartree)
            ha_factor = SEMICONDUCTOR_CONSTANTS["hartree"][0]
            eg_ha = (eg_ev * 1.602176634e-19) / ha_factor
            mat_rows.append(
                {
                    "Vật liệu": mat,
                    "Band gap (eV)": eg_ev,
                    "Band gap (J)": f"{r_j['value_out']:.4e}",
                    "Band gap (Hartree)": f"{eg_ha:.4f}",
                    "Ứng dụng": app,
                }
            )
        except Exception:
            pass

    st.dataframe(
        pd.DataFrame(mat_rows),
        use_container_width=True,
        hide_index=True,
    )
