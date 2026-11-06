"""Tests for the 2x2 quanvolution mode."""

import numpy as np
from qiskit import QuantumCircuit

from quantimg.quanvolution import quanvolve


def test_identity_reads_back_the_pixel():
    """Empty circuit + readout wire 0 (top-left) returns the original image."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(10, 14, 3), dtype=np.uint8)
    out = quanvolve(img, QuantumCircuit(4), readout_wire=0, mode="exact")
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, img)


def test_constant_image_stays_constant():
    """If all four neighbours are equal, any circuit yields a flat output."""
    img = np.full((8, 8, 3), 90, dtype=np.uint8)
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.h(0)
    out = quanvolve(img, qc, readout_wire=0, mode="exact")
    assert out.shape == img.shape
    assert np.ptp(out) == 0                    # perfectly uniform


def test_edge_filter_reacts_to_neighbours():
    """Neighbour coupling + H must turn a gradient into a non-trivial response."""
    h = w = 24
    ramp = (np.linspace(0, 255, w)[None, :] + np.linspace(0, 255, h)[:, None]) / 2
    img = np.stack([ramp.astype(np.uint8)] * 3, axis=-1)
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(1, 0)                                # neighbours control the pixel wire
    qc.cx(2, 0)
    qc.cx(3, 0)
    qc.h(0)                                    # reveal their interference
    out = quanvolve(img, qc, readout_wire=0, mode="exact")
    # Reacts to neighbours: many distinct values spanning the range.
    assert len(np.unique(out[:, :, 0])) > 8
    assert np.ptp(out[:, :, 0]) > 100
