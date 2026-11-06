"""Engine tests.

The load-bearing test is `test_marginals_match_single_pixel_statevector`: it
pins the batched NumPy pipeline (and its Qiskit little-endian qubit ordering)
against a per-pixel Qiskit Statevector computed the slow, obvious way.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from quantimg.engine import (
    N_QUBITS,
    apply_unitary,
    batched_statevector,
    process_image,
    readout_exact,
    single_qubit_states,
    to_unitary,
)


# --------------------------------------------------------------------------- #
# Ground-truth helpers (deliberately un-vectorized)
# --------------------------------------------------------------------------- #
def reference_pixel(values, phases, user_circuit):
    """Single-pixel ground truth: build the whole circuit, measure marginals."""
    qc = QuantumCircuit(N_QUBITS)
    theta = 2.0 * np.arcsin(np.sqrt(np.clip(values, 0.0, 1.0)))
    for q in range(N_QUBITS):
        qc.ry(theta[q], q)
        if phases is not None and phases[q] != 0.0:
            qc.rz(phases[q], q)
    qc.compose(user_circuit, inplace=True)
    sv = Statevector(qc)
    return np.array([sv.probabilities([q])[1] for q in range(3)])


def sample_user_circuit():
    """A mix of real and complex-phase, position- and color-controlled gates."""
    qc = QuantumCircuit(N_QUBITS)
    qc.h(3)            # X -> superposition
    qc.cx(3, 0)        # X controls Red
    qc.cry(0.7, 4, 1)  # Y rotates Green
    qc.swap(0, 2)      # Red <-> Blue crosstalk
    qc.t(2)            # complex phase on Blue
    qc.h(2)            # reveal that phase (interference)
    return qc


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_marginals_match_single_pixel_statevector():
    rng = np.random.default_rng(0)
    m = 200
    values = rng.random((m, N_QUBITS))
    phases = rng.uniform(-np.pi, np.pi, size=(m, N_QUBITS))
    circuit = sample_user_circuit()

    unitary = to_unitary(circuit)
    states = single_qubit_states(values, phases)
    psi = apply_unitary(batched_statevector(states), unitary)
    got = readout_exact(psi)

    want = np.stack(
        [reference_pixel(values[i], phases[i], circuit) for i in range(m)]
    )
    np.testing.assert_allclose(got, want, atol=1e-10)


def test_marginals_match_without_phase():
    """Real-only path (no RZ) must also match the Statevector ground truth."""
    rng = np.random.default_rng(1)
    m = 64
    values = rng.random((m, N_QUBITS))
    circuit = QuantumCircuit(N_QUBITS)
    circuit.x(1)          # flip Green
    circuit.h(0)          # mix Red toward 0.5
    circuit.cx(3, 2)      # X controls Blue

    unitary = to_unitary(circuit)
    states = single_qubit_states(values, None)
    psi = apply_unitary(batched_statevector(states), unitary)
    got = readout_exact(psi)

    want = np.stack([reference_pixel(values[i], None, circuit) for i in range(m)])
    np.testing.assert_allclose(got, want, atol=1e-10)


def test_qubit_ordering_is_little_endian():
    """X on exactly one color wire must move only that output channel."""
    values = np.full((1, N_QUBITS), 0.3)
    for target, name in [(0, "R"), (1, "G"), (2, "B")]:
        qc = QuantumCircuit(N_QUBITS)
        qc.x(target)
        psi = apply_unitary(
            batched_statevector(single_qubit_states(values)), to_unitary(qc)
        )
        out = readout_exact(psi)[0]
        expected = np.array([0.3, 0.3, 0.3])
        expected[target] = 0.7                     # flipped: P(1) -> 1 - 0.3
        np.testing.assert_allclose(out, expected, atol=1e-12)


def test_identity_roundtrips_image():
    """Empty circuit + exact mode returns the original pixels (up to rounding)."""
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, size=(12, 16, 3), dtype=np.uint8)
    out = process_image(img, QuantumCircuit(N_QUBITS), mode="exact")
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, img)


def test_shots_converge_to_exact():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    circuit = sample_user_circuit()
    exact = process_image(img, circuit, mode="exact").astype(float)
    sampled = process_image(
        img, circuit, mode="shots", shots=20000, seed=7
    ).astype(float)
    # Mean absolute error over all channels should be small at high shot count.
    assert np.abs(exact - sampled).mean() < 6.0


def test_shots_make_entanglement_correlated():
    """A Bell-type link between Red and Blue yields correlated sampled noise."""
    # Start from black so R and B begin in |0>; H then gives a clean Bell pair.
    img = np.zeros((1, 400, 3), dtype=np.uint8)
    qc = QuantumCircuit(N_QUBITS)
    qc.h(0)          # Red -> superposition (|0>+|1>)/sqrt(2)
    qc.cx(0, 2)      # entangle Blue with Red  => they always measure equal

    ent = process_image(img, qc, mode="shots", shots=1, seed=11).astype(float)
    red, blue = ent[0, :, 0], ent[0, :, 2]
    # With 1 shot each pixel is 0 or 255; entanglement forces red == blue.
    np.testing.assert_array_equal(red, blue)

    # A product state (no CX) must NOT be perfectly correlated.
    qc2 = QuantumCircuit(N_QUBITS)
    qc2.h(0)
    qc2.h(2)
    indep = process_image(img, qc2, mode="shots", shots=1, seed=11).astype(float)
    assert not np.array_equal(indep[0, :, 0], indep[0, :, 2])


def test_position_ramp_tiles():
    """Position wires produce a horizontal/vertical ramp from X/Y controls."""
    img = np.zeros((4, 8, 3), dtype=np.uint8)
    qc = QuantumCircuit(N_QUBITS)
    qc.cx(3, 0)   # X controls Red -> red increases left to right
    out = process_image(img, qc, mode="exact")
    row = out[0, :, 0].astype(float)
    assert row[0] < row[-1]                       # ramp goes up across the row
