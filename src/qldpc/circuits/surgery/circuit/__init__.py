"""Surgery circuit layer (split from the former circuit.py / y_circuit.py).

See docs/superpowers/specs/2026-06-30-surgery-layered-reorg-design.md.
NOTE: this re-export surface is transitional (Piece D Task 1) so existing test
imports keep resolving during the source move; Task 2 narrows it to the public API.
"""

from __future__ import annotations

from .engine import (
    _reliable_checks,
    _surgery_detach_and_readout,
    _surgery_final_detectors,
    _surgery_observable,
    _surgery_qec_cycle,
    _surgery_state_prep,
)
from .PPM_joint import (
    _build_joint_ppm_circuit_same_basis,
    _expand_joint_data_init,
    _stitch_to_joint_code,
    build_joint_ppm_circuit,
)
from .PPM_XZ import build_single_ppm_circuit
from .PPM_Y import build_single_y_ppm_circuit
from .support import (
    QubitIDs,
    _block_observable_targets,
    _check_lane_index_map,
    _commuting_logical_basis,
    _gadget_merged_csscode,
    _gf2_solve,
    _surgery_qubit_coordinates,
    keep_only_observable,
    logical_state_init,
)

__all__ = [
    # public builders + helpers
    "build_single_ppm_circuit",
    "build_joint_ppm_circuit",
    "build_single_y_ppm_circuit",
    "keep_only_observable",
    "logical_state_init",
    # transitional private re-exports (tests import these from this path)
    "QubitIDs",
    "_gf2_solve",
    "_commuting_logical_basis",
    "_block_observable_targets",
    "_gadget_merged_csscode",
    "_surgery_qubit_coordinates",
    "_check_lane_index_map",
    "_reliable_checks",
    "_surgery_state_prep",
    "_surgery_qec_cycle",
    "_surgery_observable",
    "_surgery_final_detectors",
    "_surgery_detach_and_readout",
    "_stitch_to_joint_code",
    "_expand_joint_data_init",
    "_build_joint_ppm_circuit_same_basis",
]
