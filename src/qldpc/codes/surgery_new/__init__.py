"""Surgery construction package (v3, replaces flat surgery.py).

Public API is re-exported here for backwards compatibility with
``from qldpc.codes.surgery import ...`` callers.
"""

# Will be populated as modules are split out.
from qldpc.codes.surgery import (
    # public
    SurgeryLayout,
    JointSurgeryLayout,
    BoostResult,
    DistanceBoostResult,
    build_layered_surgery_code,
    build_joint_measurement_code,
    boost_gadget_cheeger,
    boost_gadget_cheeger_combinatorial,
    boost_gadget_distance,
    load_webster_seed_set,
    # internals used by tests
    _restrict_to_logical_support,
    _compute_gauge_fix,
    _build_layered_blocks,
    _assemble_merged_HX,
    _assemble_merged_HZ,
    _build_generalised_bicycle_code,
    _skip_tree,
    _spectral_cheeger_lower_bound,
)

from .cellulation import _cellulate_long_cycles  # noqa: F401
