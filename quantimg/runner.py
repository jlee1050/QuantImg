"""Dispatch a JSON spec to the right engine and run it on an image."""

from __future__ import annotations

from quantimg.circuit import build_circuit, parse_phase, parse_request, resolve_wire
from quantimg.engine import process_image
from quantimg.quanvolution import quanvolve

_QUANV_ALIASES = {"quanvolution", "quanv", "conv", "convolution"}


def run_spec(image, spec):
    """Run a request spec against an (H, W, 3) image, returning (H, W, 3) uint8."""
    engine = str(spec.get("engine", "perpixel")).lower()

    if engine in _QUANV_ALIASES:
        circuit = build_circuit(spec.get("gates", []), n_qubits=4)
        return quanvolve(
            image,
            circuit,
            readout_wire=resolve_wire(spec.get("readout_wire", 0), 4),
            mode=spec.get("mode", "exact"),
            shots=int(spec.get("shots", 1024)),
            phase=parse_phase(spec.get("phase")),
            seed=spec.get("seed"),
        )

    req = parse_request(spec, n_qubits=5)
    circuit = req.pop("circuit")
    return process_image(image, circuit, **req)
