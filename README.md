# QuantImg

Learn quantum gates by seeing them, one pixel at a time.

A drag-and-drop quantum image editor: you build a small circuit on five wires,
and **every pixel** of your image is run through it (like a shader). The output
probabilities become the pixel's new colour. There's no speedup — the point is
creative and educational: interference and entanglement produce effects ordinary
colour filters can't, and you get a visual window into what gates actually do.

Built with Qiskit + NumPy for IBM Qiskit Fall Fest 2026 (UW Quantum Club).
By Juyeon Lee and Aarnav Thite.

## The five wires

| Wire | Starts as | Role |
|------|-----------|------|
| R, G, B | each colour channel (RY tilt, so P(measure 1) = value) | read out as the new colour |
| X, Y | normalized pixel position | input-only pattern source |

Each pixel is encoded (`RY` tilt + optional `RZ` phase), the user's gates are
applied, and the new R/G/B come from the per-qubit probability of measuring 1.
Phase hidden with `RZ` stays invisible until an `H` turns it into interference.

## Backend

```
quantimg/
  engine.py         vectorized per-pixel engine (exact + shots)
  circuit.py        JSON circuit format <-> Qiskit QuantumCircuit
  quanvolution.py   2x2 quantum convolution (neighbour-aware effects)
  presets.py        named demo filters
  runner.py         dispatch a JSON spec to the right engine
  api.py            FastAPI service
tests/              pytest suite (ordering pinned vs Qiskit Statevector)
scripts/            generate_samples.py
```

The engine never builds a Qiskit circuit per pixel: it compiles the user's
circuit once to a 32×32 matrix and applies it to every pixel's starting state
in batched NumPy. Qubit ordering is little-endian (qubit 0 = R = lowest bit) and
is pinned by a unit test against a single-pixel `Statevector`.

### Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux
```

### Run the API

```bash
uvicorn quantimg.api:app --reload
```

Interactive docs at <http://127.0.0.1:8000/docs>.

| Route | Purpose |
|-------|---------|
| `GET /health` | liveness |
| `GET /gates` | supported gates, wire labels, phase sources, modes |
| `GET /presets`, `GET /presets/{name}` | demo filters |
| `POST /filter` | `{image, spec, max_width}` → processed image (base64 PNG) |
| `POST /export` | `{spec}` → standalone Qiskit Python code |

### JSON circuit format

```json
{
  "engine": "perpixel",
  "gates": [
    {"type": "H",   "targets": ["X"]},
    {"type": "CX",  "controls": ["X"], "targets": ["R"]},
    {"type": "CRY", "controls": ["Y"], "targets": ["G"], "angle": 0.9},
    {"type": "SWAP", "targets": ["R", "B"]}
  ],
  "phase": {"R": {"source": "x", "multiplier": 4.0}},
  "mode": "exact",
  "shots": 1024,
  "position_repeat": [1, 1]
}
```

Wires may be names (`R,G,B,X,Y`) or indices. `mode: "shots"` samples outcomes —
entangled wires then produce *correlated* noise, making entanglement visible.
Set `"engine": "quanvolution"` for the 2×2 neighbour mode (4 wires, indices 0–3).

### Tests & samples

```bash
.venv/Scripts/python -m pytest -q
.venv/Scripts/python scripts/generate_samples.py        # -> samples/*.png
```

## Legacy demo

`Quantum_Image_Manipulation.ipynb` is an earlier Gradio/Colab app using a
2-qubit neighbouring-pixel approach. It still runs, but the per-pixel engine
above supersedes it.
