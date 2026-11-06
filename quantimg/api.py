"""FastAPI backend for QuantImg.

Receives a base64 image + a JSON circuit spec, returns the processed image.
The front end is a separate web app, so CORS is open and images travel as
base64 data URLs (debounce requests while dragging gates).

Run:  uvicorn quantimg.api:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import base64
import io
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from quantimg import presets
from quantimg.circuit import PHASE_SOURCES, SUPPORTED_GATES, to_qiskit_code
from quantimg.engine import WIRE_INDEX
from quantimg.runner import run_spec

app = FastAPI(title="QuantImg", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class FilterRequest(BaseModel):
    image: str = Field(..., description="base64 image (optionally a data: URL)")
    spec: dict = Field(default_factory=dict, description="JSON circuit spec")
    max_width: int | None = Field(
        512, description="downscale wider images before processing (None = full)"
    )


class ExportRequest(BaseModel):
    spec: dict = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# base64 <-> image
# --------------------------------------------------------------------------- #
def _decode(b64: str) -> Image.Image:
    data = b64.strip()
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001 - surface any decode failure as 400
        raise HTTPException(400, f"could not decode image: {exc}") from exc


def _encode(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}


@app.get("/gates")
def gates():
    """Everything the front end needs to build the palette."""
    return {
        "gates": SUPPORTED_GATES,
        "wires": list(WIRE_INDEX),
        "phase_sources": PHASE_SOURCES,
        "modes": ["exact", "shots"],
        "engines": ["perpixel", "quanvolution"],
    }


@app.get("/presets")
def list_presets():
    return presets.list_presets()


@app.get("/presets/{name}")
def get_preset(name: str):
    try:
        return presets.get_preset(name)
    except KeyError as exc:
        raise HTTPException(404, f"no preset named {name!r}") from exc


@app.post("/filter")
def filter_image(req: FilterRequest):
    img = _decode(req.image)
    if req.max_width and img.width > req.max_width:
        new_h = max(1, round(req.max_width * img.height / img.width))
        img = img.resize((req.max_width, new_h), Image.LANCZOS)
    arr = np.asarray(img)

    start = time.perf_counter()
    try:
        out = run_spec(arr, req.spec)
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

    return {
        "image": _encode(out),
        "width": int(out.shape[1]),
        "height": int(out.shape[0]),
        "ms": elapsed_ms,
    }


@app.post("/export")
def export_code(req: ExportRequest):
    """Return standalone Qiskit Python code for a spec's circuit."""
    engine = str(req.spec.get("engine", "perpixel")).lower()
    n_qubits = 4 if engine in {"quanvolution", "quanv", "conv", "convolution"} else 5
    try:
        code = to_qiskit_code(req.spec.get("gates", []), n_qubits)
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"code": code, "n_qubits": n_qubits}
