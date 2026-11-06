"""Tests for the JSON circuit format parser."""

import numpy as np
import pytest
from qiskit import QuantumCircuit

from quantimg.circuit import (
    build_circuit,
    parse_phase,
    parse_request,
    resolve_wire,
    to_qiskit_code,
)
from quantimg.engine import to_unitary


def test_resolve_wire_names_and_indices():
    assert resolve_wire("R", 5) == 0
    assert resolve_wire("Y", 5) == 4
    assert resolve_wire("b", 5) == 2          # case-insensitive
    assert resolve_wire(3, 5) == 3
    assert resolve_wire("3", 5) == 3          # numeric string
    with pytest.raises(ValueError):
        resolve_wire("Z", 5)                  # not a wire name
    with pytest.raises(ValueError):
        resolve_wire(7, 5)                    # out of range


def test_build_circuit_matches_manual():
    gates = [
        {"type": "H", "targets": ["X"]},
        {"type": "CX", "controls": ["X"], "targets": ["R"]},
        {"type": "CRY", "controls": ["Y"], "targets": ["G"], "angle": 0.9},
        {"type": "SWAP", "targets": ["R", "B"]},
        {"type": "CCX", "qubits": ["X", "Y", "B"]},
    ]
    got = build_circuit(gates)

    want = QuantumCircuit(5)
    want.h(3)
    want.cx(3, 0)
    want.cry(0.9, 4, 1)
    want.swap(0, 2)
    want.ccx(3, 4, 2)

    np.testing.assert_allclose(to_unitary(got), to_unitary(want), atol=1e-12)


def test_build_circuit_validation():
    with pytest.raises(ValueError):
        build_circuit([{"type": "NOPE", "targets": [0]}])
    with pytest.raises(ValueError):
        build_circuit([{"type": "RY", "targets": [0]}])          # missing angle
    with pytest.raises(ValueError):
        build_circuit([{"type": "CX", "targets": [0]}])          # missing control
    with pytest.raises(ValueError):
        build_circuit([{"type": "CX", "controls": [0], "targets": [0]}])  # repeat


def test_parse_phase_forms():
    assert parse_phase(None) is None
    p = parse_phase({"R": {"source": "x", "multiplier": 4.0}, "G": ("y", 2)})
    assert p == {"R": ("x", 4.0), "G": ("y", 2.0)}


def test_parse_request_defaults():
    req = parse_request({"gates": []})
    assert req["mode"] == "exact"
    assert req["shots"] == 1024
    assert req["position_repeat"] == (1.0, 1.0)
    assert req["phase"] is None
    assert isinstance(req["circuit"], QuantumCircuit)


def test_to_qiskit_code_roundtrips():
    gates = [
        {"type": "H", "targets": ["X"]},
        {"type": "CRY", "controls": ["Y"], "targets": ["G"], "angle": 0.5},
    ]
    code = to_qiskit_code(gates)
    ns = {}
    exec(code, ns)                             # noqa: S102 - trusted, generated
    np.testing.assert_allclose(
        to_unitary(ns["qc"]), to_unitary(build_circuit(gates)), atol=1e-12
    )
