"""Surgery circuit layer (split from the former circuit.py / y_circuit.py).

See docs/superpowers/specs/2026-06-30-surgery-layered-reorg-design.md.

Public API: the four PPM circuit builders plus the two public helpers. These are
exactly the names ``surgery/__init__.py`` re-exports; internal consumers import
the specific submodules (``.engine``, ``.support``, ``.PPM_XZ``, ``.PPM_joint``,
``.PPM_Y``) directly.
"""

from __future__ import annotations

from .PPM_joint import build_joint_ppm_circuit
from .PPM_XZ import build_single_ppm_circuit
from .PPM_Y import build_single_y_ppm_circuit
from .support import keep_only_observable, logical_state_init

__all__ = [
    "build_single_ppm_circuit",
    "build_joint_ppm_circuit",
    "build_single_y_ppm_circuit",
    "keep_only_observable",
    "logical_state_init",
]
