"""Surgery construction package.

Webster L-layer single-logical (build_layered_surgery_code) +
SkipTree-based joint measurement (build_joint_measurement_code, v3
introduced in later tasks of the plan at
docs/superpowers/plans/2026-06-07-skiptree-bridge-v3.md).

Public API is re-exported for backwards compatibility with
``from qldpc.codes.surgery import ...`` callers.
"""

from __future__ import annotations

from .cellulation import _cellulate_long_cycles  # noqa: F401
from .skiptree import _skip_tree, _skip_tree_hr  # noqa: F401
from .cheeger import (  # noqa: F401
    BoostResult,
    DistanceBoostResult,
    boost_gadget_cheeger,
    boost_gadget_cheeger_combinatorial,
    boost_gadget_distance,
    _spectral_cheeger_lower_bound,
    _exact_boundary_cheeger,
)
from .layered import (  # noqa: F401
    SurgeryLayout,
    build_layered_surgery_code,
    load_webster_seed_set,
    _restrict_to_logical_support,
    _compute_gauge_fix,
    _build_layered_blocks,
    _assemble_merged_HX,
    _assemble_merged_HZ,
    _build_generalised_bicycle_code,
)
from .joint import (  # noqa: F401
    JointSurgeryLayout,
    build_joint_measurement_code,
    build_joint_measurement_code_intercode,
    _validate_joint_logical_ops,
    _BridgeSpec,
    _build_bridge_via_skiptree,
    _stitch_gadgets_with_bridge,
    _find_bridge_z_stab_data_logical,
    _solve_gf2_system,
    _build_auxiliary_graph,
    _label_inverse,
    canonical_HR,
    _running_xor_b_c,
    _chi_z_compatibility_check,
    _solve_chi_z_bridge_choices,
    _extend_chi_rows_with_bridge,
)
