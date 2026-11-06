"""JSON circuit format <-> Qiskit QuantumCircuit.

The front end and backend agree on one JSON shape (handoff: "agree early on a
JSON circuit format"). A request looks like::

    {
      "engine": "perpixel",            # or "quanvolution"
      "gates": [
        {"type": "H",   "targets": ["X"]},
        {"type": "CX",  "controls": ["X"], "targets": ["R"]},
        {"type": "CRY", "controls": ["Y"], "targets": ["G"], "angle": 0.9},
        {"type": "SWAP", "targets": ["R", "B"]}
      ],
      "phase": {"R": {"source": "x", "multiplier": 4.0}},
      "mode": "exact",                 # or "shots"
      "shots": 1024,
      "position_repeat": [1, 1],
      "seed": null
    }

Wires may be named ("R","G","B","X","Y") or given as integer indices. A gate
may list "controls"/"targets" (or singular "control"/"target"), or a single
flat "qubits" list which is split controls-then-targets by gate arity.
"""

from __future__ import annotations

from qiskit import QuantumCircuit

from quantimg.engine import WIRE_INDEX

# name -> (n_controls, n_targets, needs_angle, qiskit method)
_GATES = {
    "I": (0, 1, False, "id"),
    "X": (0, 1, False, "x"),
    "Y": (0, 1, False, "y"),
    "Z": (0, 1, False, "z"),
    "H": (0, 1, False, "h"),
    "S": (0, 1, False, "s"),
    "SDG": (0, 1, False, "sdg"),
    "T": (0, 1, False, "t"),
    "TDG": (0, 1, False, "tdg"),
    "SX": (0, 1, False, "sx"),
    "RX": (0, 1, True, "rx"),
    "RY": (0, 1, True, "ry"),
    "RZ": (0, 1, True, "rz"),
    "P": (0, 1, True, "p"),
    "SWAP": (0, 2, False, "swap"),
    "CX": (1, 1, False, "cx"),
    "CNOT": (1, 1, False, "cx"),
    "CY": (1, 1, False, "cy"),
    "CZ": (1, 1, False, "cz"),
    "CH": (1, 1, False, "ch"),
    "CRX": (1, 1, True, "crx"),
    "CRY": (1, 1, True, "cry"),
    "CRZ": (1, 1, True, "crz"),
    "CP": (1, 1, True, "cp"),
    "CCX": (2, 1, False, "ccx"),
    "TOFFOLI": (2, 1, False, "ccx"),
    "CSWAP": (1, 2, False, "cswap"),
    "FREDKIN": (1, 2, False, "cswap"),
}

SUPPORTED_GATES = sorted(_GATES)
PHASE_SOURCES = ["none", "x", "y", "r", "g", "b", "brightness", "hue"]


def resolve_wire(w, n_qubits):
    """Map a wire name ("R") or index to a validated qubit index."""
    if isinstance(w, str):
        key = w.strip().upper()
        idx = WIRE_INDEX[key] if key in WIRE_INDEX else int(key)
    else:
        idx = int(w)
    if not 0 <= idx < n_qubits:
        raise ValueError(f"wire {w!r} out of range for {n_qubits}-qubit circuit")
    return idx


def _as_list(v):
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def build_circuit(gates, n_qubits=5):
    """Build a QuantumCircuit from a list of gate dicts."""
    qc = QuantumCircuit(n_qubits)
    for i, g in enumerate(gates):
        if "type" not in g:
            raise ValueError(f"gate {i} is missing 'type'")
        gtype = str(g["type"]).strip().upper()
        if gtype not in _GATES:
            raise ValueError(f"unknown gate type {g['type']!r}")
        nc, nt, needs_angle, method = _GATES[gtype]

        controls = _as_list(g.get("controls", g.get("control")))
        targets = _as_list(g.get("targets", g.get("target")))
        if not controls and not targets and "qubits" in g:
            allq = _as_list(g["qubits"])
            controls, targets = allq[:nc], allq[nc:]

        if len(controls) != nc or len(targets) != nt:
            raise ValueError(
                f"gate {i} ({gtype}) expects {nc} control(s) and {nt} target(s), "
                f"got {len(controls)} and {len(targets)}"
            )
        controls = [resolve_wire(w, n_qubits) for w in controls]
        targets = [resolve_wire(w, n_qubits) for w in targets]
        if len(set(controls + targets)) != nc + nt:
            raise ValueError(f"gate {i} ({gtype}) has repeated wires")

        args = []
        if needs_angle:
            if "angle" not in g:
                raise ValueError(f"gate {i} ({gtype}) requires an 'angle'")
            args.append(float(g["angle"]))
        args += controls + targets
        getattr(qc, method)(*args)
    return qc


def parse_phase(phase_spec):
    """Normalize a phase spec into {wire: (source, multiplier)} for the engine."""
    if not phase_spec:
        return None
    out = {}
    for wire, spec in phase_spec.items():
        if isinstance(spec, dict):
            source = spec.get("source", "none")
            mult = spec.get("multiplier", 1.0)
        else:
            source, mult = spec
        out[wire] = (source, float(mult))
    return out


def parse_request(spec, n_qubits=5):
    """Parse a request dict into process_image() keyword arguments."""
    pr = spec.get("position_repeat", (1.0, 1.0))
    return {
        "circuit": build_circuit(spec.get("gates", []), n_qubits),
        "mode": spec.get("mode", "exact"),
        "shots": int(spec.get("shots", 1024)),
        "position_repeat": (float(pr[0]), float(pr[1])),
        "phase": parse_phase(spec.get("phase")),
        "seed": spec.get("seed"),
    }


def to_qiskit_code(gates, n_qubits=5):
    """Emit standalone Qiskit Python code for a gate list (handoff UI export)."""
    lines = ["from qiskit import QuantumCircuit", "", f"qc = QuantumCircuit({n_qubits})"]
    for g in gates:
        gtype = str(g["type"]).strip().upper()
        nc, nt, needs_angle, method = _GATES[gtype]
        controls = _as_list(g.get("controls", g.get("control")))
        targets = _as_list(g.get("targets", g.get("target")))
        if not controls and not targets and "qubits" in g:
            allq = _as_list(g["qubits"])
            controls, targets = allq[:nc], allq[nc:]
        controls = [resolve_wire(w, n_qubits) for w in controls]
        targets = [resolve_wire(w, n_qubits) for w in targets]
        args = ([repr(float(g["angle"]))] if needs_angle else []) + [
            str(q) for q in controls + targets
        ]
        lines.append(f"qc.{method}({', '.join(args)})")
    return "\n".join(lines)
