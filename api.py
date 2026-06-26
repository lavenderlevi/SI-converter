"""
api.py — REST API cho SI Unit Converter
========================================
Chạy local:
    pip install fastapi uvicorn[standard]
    uvicorn api:app --reload --port 8000

Docs tự động:
    http://localhost:8000/docs     (Swagger UI)
    http://localhost:8000/redoc    (ReDoc)

Endpoints:
    POST /convert              — chuyển đổi scalar / array
    POST /convert/semiconductor — đơn vị bán dẫn chuyên sâu
    POST /convert/batch        — chuyển đổi hàng loạt
    GET  /units/validate        — kiểm tra đơn vị hợp lệ
    GET  /units/semiconductor   — danh sách đơn vị bán dẫn
    GET  /constants             — hằng số vật lý
    GET  /health                — health check
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).parent))

from si_converter import (
    PHYSICAL_CONSTANTS,
    SIConverter,
    convert_semiconductor,
    validate_unit,
)
from si_converter.semiconductor import list_semiconductor_units

# ──────────────────────────────────────────────────────────────────────────────
#  App setup
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SI Unit Converter API",
    description=(
        "API chuyển đổi đơn vị sang hệ SI.\n\n"
        "Hỗ trợ scalar, mảng, batch, đơn vị bán dẫn chuyên sâu "
        "(Hartree, Rydberg, Bohr, Debye, barn, electron mass, AMU)."
    ),
    version="1.0.0",
    contact={
        "name": "SI Converter",
        "url": "https://github.com/your-repo/si-converter",
    },
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cvt = SIConverter(verbose=False)

# ──────────────────────────────────────────────────────────────────────────────
#  Pydantic schemas
# ──────────────────────────────────────────────────────────────────────────────


class ConvertRequest(BaseModel):
    """Request body cho POST /convert"""

    value: Union[float, List[Any]] = Field(
        ...,
        description="Giá trị hoặc mảng giá trị cần chuyển đổi",
        examples=[1, [1, 2, 3], 100],
    )
    unit: str = Field(
        ...,
        description="Đơn vị nguồn (Pint format)",
        examples=["eV", "degC", "mph", "atm", "kgf/cm**2"],
    )
    to_unit: Optional[str] = Field(
        None,
        description="Đơn vị đích (để trống → SI tự động)",
        examples=["joule", "kelvin", "m/s"],
    )

    @field_validator("unit")
    @classmethod
    def validate_from_unit(cls, v: str) -> str:
        if not validate_unit(v):
            raise ValueError(
                f"Đơn vị '{v}' không được Pint nhận diện. "
                "Tham khảo: https://pint.readthedocs.io/en/stable/user/units.html"
            )
        return v

    @field_validator("to_unit")
    @classmethod
    def validate_to_unit(cls, v: Optional[str]) -> Optional[str]:
        if v and not validate_unit(v):
            raise ValueError(f"Đơn vị đích '{v}' không hợp lệ.")
        return v


class ConvertResponse(BaseModel):
    """Response body cho POST /convert"""

    value_in: Union[float, List[float]]
    unit_in: str
    value_out: Union[float, List[float]]
    unit_out: str
    factor: Optional[float]
    dimensionality: str
    is_affine: bool
    si_equivalent: str


class SemiconductorRequest(BaseModel):
    """Request body cho POST /convert/semiconductor"""

    value: float = Field(..., description="Giá trị số", examples=[1.0, 2.5, 13.6])
    unit: str = Field(
        ...,
        description="Đơn vị bán dẫn: hartree, rydberg, bohr, debye, electron_mass, amu, barn",
        examples=[
            "hartree",
            "bohr",
            "debye",
            "barn",
            "amu",
            "electron_mass",
            "rydberg",
        ],
    )


class SemiconductorResponse(BaseModel):
    value_in: float
    unit_in: str
    value_out: float
    unit_out: str
    factor: float
    description: str
    si_equivalent: str
    category: str


class BatchItem(BaseModel):
    label: Optional[str] = Field(None, description="Nhãn mô tả")
    value: float = Field(..., description="Giá trị đầu vào")
    unit: str = Field(..., description="Đơn vị nguồn")
    to_unit: Optional[str] = Field(None, description="Đơn vị đích")


class BatchRequest(BaseModel):
    items: List[BatchItem] = Field(..., min_length=1, max_length=500)


class BatchResultItem(BaseModel):
    label: Optional[str]
    value_in: float
    unit_in: str
    value_out: Optional[float]
    unit_out: str
    factor: Optional[float]
    dimensionality: str
    status: str  # "✓" hoặc "✗ <lỗi>"


# ──────────────────────────────────────────────────────────────────────────────
#  Endpoints
# ──────────────────────────────────────────────────────────────────────────────


@app.get("/health", tags=["System"])
def health_check():
    """Kiểm tra trạng thái API."""
    return {"status": "ok", "version": "1.0.0", "service": "SI Unit Converter"}


# ── POST /convert ─────────────────────────────────────────────────────────────
@app.post(
    "/convert",
    response_model=ConvertResponse,
    tags=["Chuyển đổi"],
    summary="Chuyển đổi scalar hoặc mảng sang SI",
)
def api_convert(req: ConvertRequest):
    """
    Chuyển đổi giá trị từ đơn vị nguồn sang SI (hoặc đơn vị đích chỉ định).

    **Ví dụ scalar:**
    ```json
    { "value": 1, "unit": "eV" }
    ```
    → `{ "value_out": 1.602176634e-19, "unit_out": "joule" }`

    **Ví dụ mảng:**
    ```json
    { "value": [60, 80, 100], "unit": "mph", "to_unit": "m/s" }
    ```

    **Nhiệt độ (offset):**
    ```json
    { "value": 100, "unit": "degC" }
    ```
    → `{ "value_out": 373.15, "unit_out": "kelvin" }`
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = _cvt.convert(req.value, req.unit, req.to_unit)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    v_out = r["value_out"]
    v_in = r["value_in"]

    # Chuyển numpy array → list cho JSON serialization
    if hasattr(v_out, "tolist"):
        v_out = v_out.tolist()
    if hasattr(v_in, "tolist"):
        v_in = v_in.tolist()

    factor = r["factor"]
    if factor is not None and hasattr(factor, "item"):
        factor = float(factor)

    return ConvertResponse(
        value_in=v_in,
        unit_in=r["unit_in"],
        value_out=v_out,
        unit_out=r["unit_out"],
        factor=factor,
        dimensionality=r["dimensionality"],
        is_affine=r["is_affine"],
        si_equivalent=(
            f"1 {r['unit_in']} = {factor:.6e} {r['unit_out']}"
            if factor is not None
            else f"offset unit: 0 {r['unit_in']} = {_cvt.convert(0, req.unit, req.to_unit)['value_out']:.4g} {r['unit_out']}"
        ),
    )


# ── POST /convert/semiconductor ───────────────────────────────────────────────
@app.post(
    "/convert/semiconductor",
    response_model=SemiconductorResponse,
    tags=["Bán dẫn chuyên sâu"],
    summary="Chuyển đổi đơn vị bán dẫn / nguyên tử",
)
def api_convert_semiconductor(req: SemiconductorRequest):
    """
    Chuyển đổi các đơn vị bán dẫn và nguyên tử đặc biệt sang SI.

    Đơn vị hỗ trợ:
    - `hartree` — Năng lượng Hartree (Eₕ ≈ 27.211 eV)
    - `rydberg` — Năng lượng Rydberg (Ry ≈ 13.606 eV)
    - `bohr` — Bán kính Bohr (a₀ ≈ 0.529 Å)
    - `debye` — Mômen lưỡng cực (1 D ≈ 3.336×10⁻³⁰ C·m)
    - `electron_mass` / `m_e` — Khối lượng electron
    - `amu` / `u` / `dalton` — Đơn vị khối lượng nguyên tử
    - `barn` / `mb` / `nb` — Tiết diện hạt nhân

    **Ví dụ:**
    ```json
    { "value": 1, "unit": "hartree" }
    ```
    → `{ "value_out": 4.359745e-18, "unit_out": "J" }`
    """
    try:
        r = convert_semiconductor(req.value, req.unit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return SemiconductorResponse(**r)


# ── POST /convert/batch ────────────────────────────────────────────────────────
@app.post(
    "/convert/batch",
    response_model=List[BatchResultItem],
    tags=["Chuyển đổi"],
    summary="Chuyển đổi hàng loạt (tối đa 500 phép)",
)
def api_batch(req: BatchRequest):
    """
    Chuyển đổi nhiều phép cùng lúc.

    **Ví dụ:**
    ```json
    {
      "items": [
        { "label": "Band gap Si", "value": 1.12, "unit": "eV" },
        { "label": "Nhiệt độ nóng chảy Al", "value": 660, "unit": "degC" },
        { "label": "Áp suất", "value": 1, "unit": "atm" }
      ]
    }
    ```
    """
    items_dicts = [
        {
            "label": item.label or "",
            "value": item.value,
            "from_unit": item.unit,
            "to_unit": item.to_unit,
        }
        for item in req.items
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _cvt.convert_many(items_dicts)

    results = []
    for _, row in df.iterrows():
        v_out = row["value_out"]
        factor = row["factor"]
        results.append(
            BatchResultItem(
                label=row["label"] or None,
                value_in=float(row["value_in"]),
                unit_in=str(row["unit_in"]),
                value_out=float(v_out)
                if v_out is not None and str(v_out) != "None"
                else None,
                unit_out=str(row["unit_out"]),
                factor=float(factor)
                if factor is not None and str(factor) != "None"
                else None,
                dimensionality=str(row["dimensionality"]),
                status=str(row["status"]),
            )
        )
    return results


# ── GET /units/validate ────────────────────────────────────────────────────────
@app.get(
    "/units/validate",
    tags=["Đơn vị"],
    summary="Kiểm tra đơn vị có hợp lệ không",
)
def api_validate_unit(
    unit: str = Query(..., description="Chuỗi đơn vị cần kiểm tra", examples=["eV"]),
):
    """
    Kiểm tra xem chuỗi đơn vị có được Pint nhận diện không.

    Ví dụ: `GET /units/validate?unit=eV` → `{"unit": "eV", "valid": true}`
    """
    valid = validate_unit(unit)
    result: Dict[str, Any] = {"unit": unit, "valid": valid}

    if valid:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from si_converter.core import _SI_REFS_Q, Q_

            q = Q_(1.0, unit)
            si_tgt = next(
                (s for rq, s in _SI_REFS_Q if q.dimensionality == rq.dimensionality),
                str(q.to_base_units().units),
            )
            result["canonical"] = str(q.units)
            result["dimensionality"] = str(q.dimensionality)
            result["si_target"] = si_tgt
    else:
        result["message"] = (
            f"Đơn vị '{unit}' không được nhận diện. "
            "Tham khảo: https://pint.readthedocs.io/en/stable/user/units.html"
        )

    return result


# ── GET /units/semiconductor ───────────────────────────────────────────────────
@app.get(
    "/units/semiconductor",
    tags=["Bán dẫn chuyên sâu"],
    summary="Danh sách đơn vị bán dẫn được hỗ trợ",
)
def api_semiconductor_units():
    """Trả về danh sách đầy đủ các đơn vị bán dẫn / nguyên tử và hệ số chuyển đổi SI."""
    return list_semiconductor_units()


# ── GET /constants ─────────────────────────────────────────────────────────────
@app.get(
    "/constants",
    tags=["Hằng số"],
    summary="Hằng số vật lý cơ bản (scipy.constants | SI)",
)
def api_constants(
    search: Optional[str] = Query(
        None,
        description="Lọc theo ký hiệu hoặc mô tả (không phân biệt hoa/thường)",
        examples=["Planck", "electron"],
    ),
):
    """
    Trả về bảng hằng số vật lý cơ bản.

    - Không có `search` → tất cả hằng số
    - `search=Planck` → lọc theo từ khóa
    """
    results = []
    for sym, (val, unit, desc) in PHYSICAL_CONSTANTS.items():
        if (
            not search
            or search.lower() in sym.lower()
            or search.lower() in desc.lower()
        ):
            results.append(
                {
                    "symbol": sym,
                    "value": val,
                    "unit": unit,
                    "description": desc,
                }
            )

    if not results and search:
        return JSONResponse(
            status_code=404,
            content={"message": f"Không tìm thấy hằng số nào khớp với '{search}'."},
        )
    return results


# ──────────────────────────────────────────────────────────────────────────────
#  Run trực tiếp
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
