"""QuantImg — a per-pixel quantum image filter engine (Qiskit + NumPy)."""

from quantimg.engine import (
    N_QUBITS,
    WIRE_INDEX,
    COLOR_QUBITS,
    single_qubit_states,
    batched_statevector,
    apply_unitary,
    marginal_p1,
    readout_exact,
    readout_shots,
    to_unitary,
    process_image,
)
from quantimg.circuit import (
    SUPPORTED_GATES,
    PHASE_SOURCES,
    build_circuit,
    parse_request,
    to_qiskit_code,
)
from quantimg.quanvolution import quanvolve
from quantimg.runner import run_spec
from quantimg import presets

__all__ = [
    "N_QUBITS",
    "WIRE_INDEX",
    "COLOR_QUBITS",
    "single_qubit_states",
    "batched_statevector",
    "apply_unitary",
    "marginal_p1",
    "readout_exact",
    "readout_shots",
    "to_unitary",
    "process_image",
    "SUPPORTED_GATES",
    "PHASE_SOURCES",
    "build_circuit",
    "parse_request",
    "to_qiskit_code",
    "quanvolve",
    "run_spec",
    "presets",
]
