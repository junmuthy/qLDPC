"""Surgery construction package (simplified — see
docs/superpowers/specs/2026-06-07-surgery-simplification-design.md).

Public API:
    build_gadget, build_bridge,
    build_single_ppm_circuit, build_joint_ppm_circuit,
    keep_only_observable,
    boost_gadget, cheeger_constant
"""

from __future__ import annotations

from .circuit import (
    build_joint_ppm_circuit,
    build_single_ppm_circuit,
    build_single_y_ppm_circuit,
    keep_only_observable,
    logical_state_init,
)
from .hmatrix.cheeger import boost_gadget, cheeger_constant
from .hmatrix.logical_basis import (
    logical_distance,
    low_weight_logicals,
    symplectic_logical_basis,
)
from .hmatrix.PPM_joint import Bridge, build_bridge
from .hmatrix.PPM_X_Z import GadgetLayout, build_gadget, minimize_z_checks
from .hmatrix.PPM_Y import YGadgetLayout, build_y_gadget

__all__ = [
    "GadgetLayout",
    "YGadgetLayout",
    "Bridge",
    "build_gadget",
    "minimize_z_checks",
    "build_y_gadget",
    "build_bridge",
    "build_single_ppm_circuit",
    "build_single_y_ppm_circuit",
    "build_joint_ppm_circuit",
    "keep_only_observable",
    "logical_state_init",
    "boost_gadget",
    "cheeger_constant",
    "low_weight_logicals",
    "logical_distance",
    "symplectic_logical_basis",
]
