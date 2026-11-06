"""Tests for the FastAPI backend (via TestClient)."""

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from quantimg.api import app

client = TestClient(app)


def _png_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _decode(data_url):
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_gates_and_presets_listed():
    g = client.get("/gates").json()
    assert "H" in g["gates"] and "CX" in g["gates"]
    assert g["wires"] == ["R", "G", "B", "X", "Y"]

    names = [p["name"] for p in client.get("/presets").json()]
    assert "invert" in names and "quanv-edge" in names


def test_filter_invert_roundtrip():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(6, 8, 3), dtype=np.uint8)
    spec = client.get("/presets/invert").json()
    r = client.post("/filter", json={"image": _png_b64(img), "spec": spec,
                                     "max_width": None})
    assert r.status_code == 200
    out = _decode(r.json()["image"])
    # X on every colour wire inverts each channel (exact mode round-trips).
    np.testing.assert_array_equal(out, 255 - img)


def test_filter_max_width_downscales():
    img = np.zeros((100, 400, 3), dtype=np.uint8)
    r = client.post("/filter", json={"image": _png_b64(img), "spec": {"gates": []},
                                     "max_width": 128})
    body = r.json()
    assert body["width"] == 128
    assert body["height"] == 32


def test_filter_bad_gate_is_400():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    r = client.post("/filter", json={"image": _png_b64(img),
                                     "spec": {"gates": [{"type": "NOPE", "targets": [0]}]}})
    assert r.status_code == 400


def test_export_code():
    spec = {"gates": [{"type": "H", "targets": ["X"]},
                      {"type": "CX", "controls": ["X"], "targets": ["R"]}]}
    r = client.post("/export", json={"spec": spec})
    assert r.status_code == 200
    code = r.json()["code"]
    assert "QuantumCircuit(5)" in code and "qc.cx(3, 0)" in code
