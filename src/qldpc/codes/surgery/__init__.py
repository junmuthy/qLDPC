"""Surgery construction package (simplified — see
docs/superpowers/specs/2026-06-07-surgery-simplification-design.md).

Public API:
    build_gadget, build_bridge,
    build_single_ppm_circuit, build_joint_ppm_circuit,
    boost_gadget
"""

from __future__ import annotations

from .gadget import GadgetLayout, build_gadget, load_webster_seed_set
from .bridge import Bridge, build_bridge
from .circuit import build_single_ppm_circuit, build_joint_ppm_circuit
from .cheeger import boost_gadget

__all__ = [
    "GadgetLayout",
    "Bridge",
    "build_gadget",
    "build_bridge",
    "build_single_ppm_circuit",
    "build_joint_ppm_circuit",
    "boost_gadget",
    "load_webster_seed_set",
]
