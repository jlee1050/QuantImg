"""Vectorized per-pixel quantum image filter engine.

Every pixel is run through the SAME small circuit (like a shader). Five wires:

    index  wire  meaning
    0      R     red channel      -> read out as new red
    1      G     green channel    -> read out as new green
    2      B     blue channel     -> read out as new blue
    3      X     horizontal pos   -> input-only pattern source
    4      Y     vertical pos     -> input-only pattern source

Qiskit is little-endian: qubit 0 (R) is the least-significant bit of a basis
state label, so a 5-qubit statevector is ordered kron(q4, q3, q2, q1, q0).

Pipeline per pixel:
  1. normalize channels + position to [0, 1]
  2. tilt each wire   RY(2*arcsin(sqrt(v)))           -> P(measure 1) = v
  3. optional phase   RZ(2*pi * multiplier * source)  -> hidden until an H
  4. apply the user's gate circuit (as a 32x32 unitary)
  5. read out new R/G/B from per-qubit marginal P(qubit = 1)

The engine never builds a Qiskit circuit per pixel: it builds the user's
circuit once (into a 32x32 matrix) and applies it to every pixel's starting
state in batched NumPy. Correctness of the qubit ordering is pinned by a unit
test that compares these marginals against a single-pixel Qiskit Statevector.
"""

from __future__ import annotations

import numpy as np

N_QUBITS = 5
WIRE_INDEX = {"R": 0, "G": 1, "B": 2, "X": 3, "Y": 4}
COLOR_QUBITS = (0, 1, 2)
_LUMA = np.array([0.299, 0.587, 0.114])

# Default number of pixels processed per batch. 65536 pixels * 32 amplitudes *
# 16 bytes (complex128) ~= 33 MB of statevector, comfortably in memory.
DEFAULT_PIXEL_CHUNK = 65_536


# --------------------------------------------------------------------------- #
# Core primitives (ordering-sensitive — exercised directly by the tests)
# --------------------------------------------------------------------------- #
def single_qubit_states(values, phases=None):
    """Build per-wire single-qubit states.

    Parameters
    ----------
    values : array (..., n_wires), real in [0, 1]
        Tilt target: RY(2*arcsin(sqrt(v))) so P(measure 1) == v.
    phases : array (..., n_wires), real radians, optional
        RZ twist applied after the tilt.

    Returns
    -------
    states : complex array (..., n_wires, 2)
        states[..., q, :] == [amplitude of |0>, amplitude of |1>] for wire q.
    """
    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    theta = 2.0 * np.arcsin(np.sqrt(values))          # RY angle
    a = np.cos(theta / 2.0)                            # <0|psi>
    b = np.sin(theta / 2.0)                            # <1|psi>
    states = np.stack([a, b], axis=-1).astype(np.complex128)
    if phases is not None:
        phases = np.asarray(phases, dtype=float)
        # RZ(phi) = diag(e^{-i phi/2}, e^{+i phi/2}); global phase is irrelevant.
        states[..., 0] *= np.exp(-0.5j * phases)
        states[..., 1] *= np.exp(0.5j * phases)
    return states


def batched_statevector(states):
    """Tensor single-qubit states into full statevectors.

    Parameters
    ----------
    states : complex array (N, n_wires, 2)

    Returns
    -------
    psi : complex array (N, 2**n_wires) in Qiskit little-endian order
        basis index i has bit (i >> q) & 1 for wire q.
    """
    n = states.shape[1]
    psi = states[:, 0, :]                              # qubit 0 = lowest bit
    for q in range(1, n):
        s = states[:, q, :]                            # qubit q = new high bit
        psi = (s[:, :, None] * psi[:, None, :]).reshape(psi.shape[0], -1)
    return psi


def apply_unitary(psi, unitary):
    """Apply a (D, D) unitary to a stack of row statevectors (N, D)."""
    return psi @ unitary.T


def marginal_p1(probs, qubit):
    """P(qubit == 1) from a (N, 2**n) probability array."""
    idx = np.arange(probs.shape[1])
    mask = ((idx >> qubit) & 1).astype(bool)
    return probs[:, mask].sum(axis=1)


def readout_exact(psi, qubits=COLOR_QUBITS):
    """Exact marginal P(qubit == 1) per requested qubit -> (N, len(qubits))."""
    probs = np.abs(psi) ** 2
    return np.stack([marginal_p1(probs, q) for q in qubits], axis=1)


def readout_shots(psi, shots, qubits=COLOR_QUBITS, rng=None):
    """Sampled marginals that preserve cross-wire correlations.

    We only read the color wires, so the full 2**n distribution is first
    marginalized onto the joint distribution of `qubits` (<=3 of them -> <=8
    categories). Sampling that *joint* — rather than each channel
    independently — is what makes entanglement visible: entangled wires then
    produce correlated noise (e.g. red and blue always matching), which
    per-channel binomial sampling cannot show.

    The joint counts are drawn with a chain of conditional binomials, each
    vectorized over every pixel at once (NumPy's binomial broadcasts over its
    n and p arrays), so there is no per-shot intermediate array.
    """
    rng = np.random.default_rng() if rng is None else rng
    qubits = tuple(qubits)
    k = len(qubits)
    probs = np.abs(psi) ** 2
    n, d = probs.shape

    # Joint distribution over `qubits`: category c has bit j == value of qubit j.
    idx = np.arange(d)
    cat = np.zeros(d, dtype=np.intp)
    for j, q in enumerate(qubits):
        cat += (((idx >> q) & 1) << j).astype(np.intp)
    ncat = 1 << k
    joint = np.stack([probs[:, cat == c].sum(axis=1) for c in range(ncat)], axis=1)
    joint /= joint.sum(axis=1, keepdims=True)          # kill tiny numeric drift

    # Multinomial(shots) per pixel via sequential conditional binomials.
    counts = np.empty((n, ncat), dtype=np.int64)
    remaining_n = np.full(n, shots, dtype=np.int64)
    remaining_p = np.ones(n)
    for c in range(ncat - 1):
        with np.errstate(invalid="ignore", divide="ignore"):
            p_cond = np.where(remaining_p > 0, joint[:, c] / remaining_p, 0.0)
        np.clip(p_cond, 0.0, 1.0, out=p_cond)
        drawn = rng.binomial(remaining_n, p_cond)
        counts[:, c] = drawn
        remaining_n -= drawn
        remaining_p -= joint[:, c]
    counts[:, ncat - 1] = remaining_n

    # Per output wire: fraction of shots whose category has that bit set.
    out = np.empty((n, k))
    cats = np.arange(ncat)
    for j in range(k):
        bitmask = ((cats >> j) & 1).astype(bool)
        out[:, j] = counts[:, bitmask].sum(axis=1) / shots
    return out


# --------------------------------------------------------------------------- #
# Phase sources
# --------------------------------------------------------------------------- #
def _hue(rgb):
    """Hue in [0, 1) from an (N, 3) RGB array in [0, 1]."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx = rgb.max(axis=1)
    mn = rgb.min(axis=1)
    delta = mx - mn
    h = np.zeros_like(mx)
    nz = delta > 1e-12
    # Piecewise hue by which channel is the max.
    rmax = nz & (mx == r)
    gmax = nz & (mx == g) & ~rmax
    bmax = nz & (mx == b) & ~rmax & ~gmax
    h[rmax] = ((g[rmax] - b[rmax]) / delta[rmax]) % 6.0
    h[gmax] = (b[gmax] - r[gmax]) / delta[gmax] + 2.0
    h[bmax] = (r[bmax] - g[bmax]) / delta[bmax] + 4.0
    return (h / 6.0) % 1.0


def _phase_source_value(name, rgb_flat, x, y):
    """Normalized [0, 1] driver array for a phase source, or None."""
    name = (name or "none").lower()
    if name in ("none", ""):
        return None
    if name == "x":
        return x
    if name == "y":
        return y
    if name in ("r", "red"):
        return rgb_flat[:, 0]
    if name in ("g", "green"):
        return rgb_flat[:, 1]
    if name in ("b", "blue"):
        return rgb_flat[:, 2]
    if name in ("brightness", "luma", "value"):
        return rgb_flat @ _LUMA
    if name == "hue":
        return _hue(rgb_flat)
    raise ValueError(f"unknown phase source: {name!r}")


def _resolve_phases(phase, rgb_flat, x, y, n):
    """Turn a {wire: (source, multiplier)} spec into an (N, 5) radian array."""
    if not phase:
        return None
    phases = np.zeros((n, N_QUBITS))
    any_set = False
    for wire, spec in phase.items():
        q = WIRE_INDEX[wire] if isinstance(wire, str) else int(wire)
        source, mult = spec
        val = _phase_source_value(source, rgb_flat, x, y)
        if val is None or mult == 0:
            continue
        phases[:, q] = 2.0 * np.pi * float(mult) * val
        any_set = True
    return phases if any_set else None


# --------------------------------------------------------------------------- #
# Circuit -> matrix
# --------------------------------------------------------------------------- #
def to_unitary(circuit):
    """Accept a QuantumCircuit, Operator, or raw ndarray; return a matrix."""
    if isinstance(circuit, np.ndarray):
        return circuit
    data = getattr(circuit, "data", None)
    if isinstance(data, np.ndarray):          # qiskit Operator
        return data
    from qiskit.quantum_info import Operator
    return Operator(circuit).data


# --------------------------------------------------------------------------- #
# High-level image processing
# --------------------------------------------------------------------------- #
def process_image(
    rgb,
    circuit,
    *,
    mode="exact",
    shots=1024,
    position_repeat=(1.0, 1.0),
    phase=None,
    seed=None,
    pixel_chunk=DEFAULT_PIXEL_CHUNK,
):
    """Run every pixel of an image through one quantum circuit.

    Parameters
    ----------
    rgb : array (H, W, 3), uint8 (0-255) or float (0-1)
    circuit : QuantumCircuit | Operator | (32, 32) ndarray
        The user's 5-qubit gate circuit.
    mode : "exact" | "shots"
        Exact statevector marginals, or sampled (correlated) marginals.
    shots : int
        Samples per pixel when mode == "shots".
    position_repeat : (rx, ry)
        Cycles of the X / Y position ramp across the image. Integer values that
        divide the image evenly tile seamlessly.
    phase : {wire: (source, multiplier)} | None
        Optional RZ phase twist per wire. wire in {"R","G","B","X","Y"} or an
        index; source in {x, y, r, g, b, brightness, hue}.
    seed : int | None
        RNG seed for shots mode (reproducible demos).

    Returns
    -------
    out : array (H, W, 3) uint8
    """
    rgb = np.asarray(rgb, dtype=float)
    if rgb.max(initial=0.0) > 1.0:
        rgb = rgb / 255.0
    h, w = rgb.shape[:2]
    n = h * w
    flat = rgb.reshape(n, 3)

    rx, ry = position_repeat
    xs = (np.arange(w) * (rx / w)) % 1.0
    ys = (np.arange(h) * (ry / h)) % 1.0
    x = np.broadcast_to(xs[None, :], (h, w)).reshape(n)
    y = np.broadcast_to(ys[:, None], (h, w)).reshape(n)

    values = np.empty((n, N_QUBITS))
    values[:, 0:3] = flat
    values[:, 3] = x
    values[:, 4] = y

    phases = _resolve_phases(phase, flat, x, y, n)
    unitary = to_unitary(circuit)
    rng = np.random.default_rng(seed) if mode == "shots" else None

    new_rgb = np.empty((n, 3))
    for start in range(0, n, pixel_chunk):
        sl = slice(start, start + pixel_chunk)
        ph = None if phases is None else phases[sl]
        states = single_qubit_states(values[sl], ph)
        psi = apply_unitary(batched_statevector(states), unitary)
        if mode == "shots":
            new_rgb[sl] = readout_shots(psi, shots, rng=rng)
        else:
            new_rgb[sl] = readout_exact(psi)

    out = (np.clip(new_rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return out.reshape(h, w, 3)
