"""2x2 quantum convolution ("quanvolution", Henderson et al. 2019).

Slide a 2x2 window over the image. For each colour channel separately, load the
four pixels of the window onto four wires (RY tilt), run the user's 4-qubit
circuit, and read one wire back out as the new pixel. Linking neighbour wires
with entangling gates + H turns "how similar are my neighbours" into
edge / glow / emboss-like effects.

Wire layout of the window (indices 0..3):

    0 = top-left     1 = top-right
    2 = bottom-left  3 = bottom-right

Reflect padding keeps the output the same size as the input. We deliberately
avoid 3x3-full-colour (27 qubits); 2x2 one-channel-at-a-time is 4 qubits.
"""

from __future__ import annotations

import numpy as np

from quantimg.engine import (
    apply_unitary,
    batched_statevector,
    marginal_p1,
    single_qubit_states,
    to_unitary,
)

N_WINDOW = 4


def _xy_phases(phase, h, w):
    """Build (N, 4) RZ phases from a {wire: (source, mult)} spec (x/y only)."""
    if not phase:
        return None
    n = h * w
    xs = np.arange(w) / w
    ys = np.arange(h) / h
    x = np.broadcast_to(xs[None, :], (h, w)).reshape(n)
    y = np.broadcast_to(ys[:, None], (h, w)).reshape(n)
    phases = np.zeros((n, N_WINDOW))
    any_set = False
    for wire, (source, mult) in phase.items():
        q = int(wire)
        source = (source or "none").lower()
        val = {"x": x, "y": y}.get(source)
        if val is None or mult == 0:
            continue
        phases[:, q] = 2.0 * np.pi * float(mult) * val
        any_set = True
    return phases if any_set else None


def quanvolve(
    rgb,
    circuit,
    *,
    readout_wire=0,
    mode="exact",
    shots=1024,
    phase=None,
    seed=None,
):
    """Apply a 2x2 quanvolution filter.

    Parameters
    ----------
    rgb : array (H, W, 3), uint8 or float
    circuit : QuantumCircuit | Operator | (16, 16) ndarray  (4 qubits)
    readout_wire : int in 0..3
        Which window wire becomes the output pixel.
    mode, shots, seed : as in engine.process_image
    phase : {wire_index: (source, multiplier)} | None
        Optional RZ twist per window wire; sources "x"/"y" (output position).

    Returns
    -------
    out : array (H, W, 3) uint8
    """
    rgb = np.asarray(rgb, dtype=float)
    if rgb.max(initial=0.0) > 1.0:
        rgb = rgb / 255.0
    h, w = rgb.shape[:2]
    n = h * w

    # Reflect-pad one row/col so every pixel has a full 2x2 window below-right.
    pad = np.pad(rgb, ((0, 1), (0, 1), (0, 0)), mode="reflect")
    windows = [
        pad[0:h, 0:w],        # top-left  (wire 0 == the pixel itself)
        pad[0:h, 1:w + 1],    # top-right
        pad[1:h + 1, 0:w],    # bottom-left
        pad[1:h + 1, 1:w + 1],  # bottom-right
    ]

    unitary = to_unitary(circuit)
    phases = _xy_phases(phase, h, w)
    rng = np.random.default_rng(seed) if mode == "shots" else None

    out = np.empty((h, w, 3))
    for c in range(3):
        values = np.stack([win[:, :, c].reshape(n) for win in windows], axis=1)
        states = single_qubit_states(values, phases)
        psi = apply_unitary(batched_statevector(states), unitary)
        p1 = marginal_p1(np.abs(psi) ** 2, readout_wire)
        if mode == "shots":
            p1 = rng.binomial(shots, np.clip(p1, 0.0, 1.0)) / shots
        out[:, :, c] = p1.reshape(h, w)

    return (np.clip(out, 0.0, 1.0) * 255.0).round().astype(np.uint8)
