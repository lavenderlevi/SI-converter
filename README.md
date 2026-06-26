# ⚗️ SI Unit Converter

Thư viện Python chuyển đổi đơn vị sang hệ SI — Khoa học, Kỹ thuật & Bán dẫn.

[![CI](https://github.com/your-repo/si-converter/actions/workflows/ci.yml/badge.svg)](https://github.com/your-repo/si-converter/actions)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📦 Cài đặt
pip install -r requirements.txt or Example: pip install -r E:\si_converter_project\requirements.txt
```bash
# Cơ bản (thư viện Python)
pip install -e .

# Với REST API
pip install -e ".[api]"

# Với Streamlit web app
pip install -e ".[app]"

# Toàn bộ (dev + test)
pip install -e ".[dev,api,app]"
```

---

## 🚀 Sử dụng nhanh

### 1. Python Package

```python
from si_converter import convert

# Cú pháp chuỗi tự phân tích
convert("1 eV")          # → 1.602e-19 J
convert("100 degC")      # → 373.15 K
convert("1 atm")         # → 101325 Pa

# Cú pháp tường minh
convert(60, "mph")            # → 26.82 m/s
convert(1, "eV", "joule")     # chỉ định đơn vị đích
convert([1, 2, 3], "atm")     # mảng giá trị
```

### 2. Đơn vị Bán dẫn chuyên sâu

```python
from si_converter import convert_semiconductor

convert_semiconductor(1, "hartree")       # 4.360e-18 J  (≈ 27.211 eV)
convert_semiconductor(1, "rydberg")       # 2.180e-18 J  (≈ 13.606 eV)
convert_semiconductor(1, "bohr")          # 5.292e-11 m  (≈ 0.529 Å)
convert_semiconductor(2.5, "debye")       # 8.340e-30 C·m
convert_semiconductor(1, "electron_mass") # 9.109e-31 kg
convert_semiconductor(1, "amu")           # 1.661e-27 kg
convert_semiconductor(1, "barn")          # 1.000e-28 m²
```

Đơn vị hỗ trợ và bí danh:

| Đơn vị | Bí danh | Đơn vị SI | Mô tả |
|--------|---------|-----------|-------|
| `hartree` | `eh`, `ha` | J | Năng lượng Hartree (Eₕ) |
| `rydberg` | `ry` | J | Năng lượng Rydberg |
| `bohr` | `a0`, `bohr_radius` | m | Bán kính Bohr (a₀) |
| `debye` | `d` | C·m | Mômen lưỡng cực |
| `electron_mass` | `m_e`, `me` | kg | Khối lượng electron |
| `amu` | `u`, `dalton`, `da` | kg | Đơn vị khối lượng nguyên tử |
| `barn` | `mb`, `nb`, `pb` | m² | Tiết diện hạt nhân |

### 3. SIConverter class (API đầy đủ)

```python
from si_converter import SIConverter
import pandas as pd

cvt = SIConverter(verbose=False)

# Chuyển đổi scalar
r = cvt.convert(100, "degC")
print(r["value_out"], r["unit_out"])   # 373.15 kelvin

# Chuyển đổi DataFrame
df = pd.DataFrame({"P_kPa": [101.325, 200.0, "N/A", None]})
df_si = cvt.convert_dataframe(df, "P_kPa", "kilopascal")
# → Thêm cột P_kPa_SI (Pa)

# Batch conversion
results = cvt.convert_many([
    {"label": "Band gap Si",  "value": 1.12, "from_unit": "eV"},
    {"label": "Nhiệt độ Al",  "value": 660,  "from_unit": "degC"},
])
```

---

## 🌐 REST API

```bash
# Khởi động server
uvicorn api:app --reload --port 8000
```

Swagger UI: http://localhost:8000/docs

### Endpoints

| Method | Path | Mô tả |
|--------|------|-------|
| `GET` | `/health` | Health check |
| `POST` | `/convert` | Chuyển đổi scalar / mảng |
| `POST` | `/convert/semiconductor` | Đơn vị bán dẫn |
| `POST` | `/convert/batch` | Hàng loạt (tối đa 500) |
| `GET` | `/units/validate` | Kiểm tra đơn vị hợp lệ |
| `GET` | `/units/semiconductor` | Danh sách đơn vị bán dẫn |
| `GET` | `/constants` | Hằng số vật lý |

### Ví dụ cURL

```bash
# Chuyển đổi 1 eV → joule
curl -X POST http://localhost:8000/convert \
  -H "Content-Type: application/json" \
  -d '{"value": 1, "unit": "eV"}'

# Chuyển đổi 100 degC → kelvin
curl -X POST http://localhost:8000/convert \
  -H "Content-Type: application/json" \
  -d '{"value": 100, "unit": "degC"}'

# Đơn vị bán dẫn: 1 Hartree → J
curl -X POST http://localhost:8000/convert/semiconductor \
  -H "Content-Type: application/json" \
  -d '{"value": 1, "unit": "hartree"}'

# Batch
curl -X POST http://localhost:8000/convert/batch \
  -H "Content-Type: application/json" \
  -d '{"items": [
    {"label": "Band gap Si", "value": 1.12, "unit": "eV"},
    {"label": "Áp suất",     "value": 1,    "unit": "atm"}
  ]}'
```

---

## 🖥️ Streamlit Web App

```bash
streamlit run app_si.py
```

Tabs:
- ⚡ Scalar — chuyển đổi giá trị đơn lẻ
- 📊 Mảng dữ liệu — làm sạch và chuyển đổi hàng loạt
- 📁 Upload File — CSV/Excel
- 🔢 Batch — bảng chỉnh sửa
- 🧲 Hằng số — bảng hằng số vật lý
- 🔍 Tra cứu đơn vị
- 💎 **Bán dẫn chuyên sâu** ← tab mới

---

## 🧪 Kiểm thử

```bash
# Chạy tất cả tests
pytest tests/ -v

# Với coverage
pytest tests/ -v --cov=si_converter --cov-report=term-missing

# Chỉ test một nhóm
pytest tests/ -v -k "TestSemiconductor"
pytest tests/ -v -k "TestConvertPublicAPI"
```

### Cấu trúc tests

```
tests/
└── test_si_converter.py
    ├── TestConvertPublicAPI     — hàm convert() public
    ├── TestSIConverter          — SIConverter class
    ├── TestValidateUnit         — validate_unit()
    ├── TestCleanNumericInput    — clean_numeric_input()
    ├── TestSemiconductor        — đơn vị bán dẫn (15 tests)
    └── TestEdgeCases            — edge cases & regression
```

### GitHub Actions CI

Mỗi lần push code:
- ✅ Chạy test trên Python 3.9, 3.10, 3.11, 3.12
- 📊 Kiểm tra coverage ≥ 85%
- 🔍 Lint với Ruff
- 🚀 API smoke test tự động

---

## 📁 Cấu trúc dự án

```
si_converter_project/
├── si_converter/              # Package chính
│   ├── __init__.py            # Public API: convert(), convert_semiconductor()
│   ├── core.py                # SIConverter class (corefile.py gốc)
│   └── semiconductor.py       # Module bán dẫn chuyên sâu
├── tests/
│   └── test_si_converter.py   # ~60 test cases
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI
├── api.py                     # FastAPI REST API
├── app_si.py                  # Streamlit web app
├── pyproject.toml             # Package config
└── README.md
```

---

## 📚 Phụ thuộc

| Thư viện | Mục đích |
|---------|---------|
| `pint` | Phân tích và chuyển đổi đơn vị |
| `numpy` | Xử lý mảng |
| `pandas` | DataFrame |
| `scipy` | Hằng số vật lý chính xác |
| `fastapi` | REST API (optional) |
| `streamlit` | Web app (optional) |
