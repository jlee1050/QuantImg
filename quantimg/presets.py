"""Demo presets — named JSON specs, like PHOSIMP's numbered filters.

Each entry is a ready-to-run request spec for the /filter endpoint. They are
chosen to show the span from classical-ish recolouring to genuinely quantum
interference and entanglement effects.
"""

from __future__ import annotations

PRESETS = {
    "invert": {
        "description": "Classic colour inversion — X on every colour wire.",
        "spec": {
            "gates": [
                {"type": "X", "targets": ["R"]},
                {"type": "X", "targets": ["G"]},
                {"type": "X", "targets": ["B"]},
            ]
        },
    },
    "channel-swap": {
        "description": "Swap red and blue — SWAP between the R and B wires.",
        "spec": {"gates": [{"type": "SWAP", "targets": ["R", "B"]}]},
    },
    "red-ramp": {
        "description": "Horizontal red gradient — X position controls the red wire.",
        "spec": {"gates": [{"type": "CX", "controls": ["X"], "targets": ["R"]}]},
    },
    "plaid": {
        "description": "Position-driven plaid — X into red, Y into green, both into blue.",
        "spec": {
            "gates": [
                {"type": "CX", "controls": ["X"], "targets": ["R"]},
                {"type": "CRY", "controls": ["Y"], "targets": ["G"], "angle": 3.14159},
                {"type": "CX", "controls": ["Y"], "targets": ["B"]},
            ],
            "position_repeat": [6, 6],
        },
    },
    "ripples": {
        "description": "Interference ripples — hide position in phase, reveal with H.",
        "spec": {
            "gates": [
                {"type": "H", "targets": ["R"]},
                {"type": "H", "targets": ["B"]},
            ],
            "phase": {
                "R": {"source": "x", "multiplier": 6.0},
                "B": {"source": "y", "multiplier": 6.0},
            },
        },
    },
    "plasma": {
        "description": "Plasma field — all colour wires carry x+y phase, revealed by H.",
        "spec": {
            "gates": [
                {"type": "H", "targets": ["R"]},
                {"type": "H", "targets": ["G"]},
                {"type": "H", "targets": ["B"]},
            ],
            "phase": {
                "R": {"source": "x", "multiplier": 5.0},
                "G": {"source": "y", "multiplier": 5.0},
                "B": {"source": "x", "multiplier": 3.0},
            },
        },
    },
    "hue-fringe": {
        "description": "Colour fringing — hue drives phase, revealed by H (interference).",
        "spec": {
            "gates": [
                {"type": "H", "targets": ["R"]},
                {"type": "H", "targets": ["G"]},
                {"type": "H", "targets": ["B"]},
            ],
            "phase": {
                "R": {"source": "hue", "multiplier": 2.0},
                "G": {"source": "hue", "multiplier": 4.0},
                "B": {"source": "hue", "multiplier": 6.0},
            },
        },
    },
    "entangled-grain": {
        "description": "Correlated sparkle — entangle the colour wires, then sample shots.",
        "spec": {
            "gates": [
                {"type": "H", "targets": ["R"]},
                {"type": "CX", "controls": ["R"], "targets": ["G"]},
                {"type": "CX", "controls": ["R"], "targets": ["B"]},
            ],
            "mode": "shots",
            "shots": 64,
            "seed": 7,
        },
    },
    "quanv-edge": {
        "description": "2x2 quanvolution edge/glow — entangle neighbours, reveal with H.",
        "spec": {
            "engine": "quanvolution",
            "gates": [
                {"type": "H", "targets": [0]},
                {"type": "CX", "controls": [1], "targets": [0]},
                {"type": "CX", "controls": [2], "targets": [0]},
                {"type": "CX", "controls": [3], "targets": [0]},
                {"type": "H", "targets": [0]},
            ],
            "readout_wire": 0,
        },
    },
}


def list_presets():
    """Return [{name, description, engine}] for the gate palette / menu."""
    return [
        {
            "name": name,
            "description": entry["description"],
            "engine": entry["spec"].get("engine", "perpixel"),
        }
        for name, entry in PRESETS.items()
    ]


def get_preset(name):
    """Return the request spec for a preset name (raises KeyError if missing)."""
    return PRESETS[name]["spec"]
