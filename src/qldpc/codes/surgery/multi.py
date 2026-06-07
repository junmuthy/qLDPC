"""Multi-PPM (Pauli product measurement) surgery construction.

Measures t commuting Pauli logicals simultaneously via a single Webster
gadget on V_0 = union of supports. Each logical's chi-sum subset becomes
an HX-row-span stabilizer, consuming one logical DOF per measurement.

Overlapping supports are handled via SetValuedPort (Ide §VII C, Cain
Processor mode).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import galois
import numpy as np
import numpy.typing as npt

from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli

from .layered import SurgeryLayout, build_layered_surgery_code
from .port import SetValuedPort


@dataclasses.dataclass(frozen=True, eq=False)
class MultiSurgeryLayout:
    """Layout for a multi-PPM Webster gadget.

    Attributes:
        base_layout: SurgeryLayout from the Webster gadget on V_0_union.
        logical_ops: tuple of original binary support vectors, length t.
        set_valued_port: SetValuedPort mapping qubit -> list of logical indices.
        chi_group_per_logical: tuple of length t; chi_group_per_logical[i] is
            the tuple of chi row indices in merged.matrix_x whose sum modulo 2
            equals logical_ops[i].
    """

    base_layout: SurgeryLayout
    logical_ops: tuple[np.ndarray, ...]
    set_valued_port: SetValuedPort
    chi_group_per_logical: tuple[tuple[int, ...], ...]


def build_multi_target_surgery_code(
    data_code: CSSCode,
    logical_ops: Sequence[npt.ArrayLike],
    *,
    num_layers: int = 1,
    validate: bool = True,
) -> tuple[CSSCode, MultiSurgeryLayout]:
    """Webster gadget measuring t commuting Pauli logicals simultaneously.

    All logical_ops must commute pairwise (same Pauli type). V_0 = union of
    supports. After construction, each X_bar_i = sum of chi rows over supp(op_i)
    is in HX row span, so each consumes one logical DOF: k_joint = k_data - t.

    Args:
        data_code: stabilizer CSSCode with dimension >= len(logical_ops).
        logical_ops: t binary support vectors of length data_code.num_qubits.
        num_layers: Webster L (odd, >= 1).
        validate: if True, check inputs.

    Returns:
        (merged_code, MultiSurgeryLayout).

    Raises:
        ValueError: empty logical_ops, support out of range, or commutation
            failure when validate=True.
    """
    ops_arr = tuple(np.asarray(op).astype(np.int_) for op in logical_ops)
    if len(ops_arr) == 0:
        raise ValueError("logical_ops must contain at least one operator")
    n_data = data_code.num_qubits
    for i, op in enumerate(ops_arr):
        if op.shape != (n_data,):
            raise ValueError(
                f"logical_ops[{i}] has shape {op.shape}, expected ({n_data},)"
            )
        if not np.all((op == 0) | (op == 1)):
            raise ValueError(f"logical_ops[{i}] must be binary")

    if validate:
        # All ops commute with H_Z (Z-type) OR all commute with H_X (X-type).
        field = data_code.field
        z_like = []
        x_like = []
        for op in ops_arr:
            op_gf = field(op)
            commutes_with_hz = bool(np.all((data_code.matrix_z @ op_gf) == 0))
            commutes_with_hx = bool(np.all((data_code.matrix_x @ op_gf) == 0))
            z_like.append(commutes_with_hz)
            x_like.append(commutes_with_hx)
        if not (all(z_like) or all(x_like)):
            raise ValueError(
                "logical_ops must all be the same Pauli type "
                "(all X-type or all Z-type)."
            )

    # V_0_union: binary OR of all support vectors.
    v0_union = np.zeros(n_data, dtype=np.int_)
    for op in ops_arr:
        v0_union = v0_union | op

    set_valued_port = SetValuedPort.from_supports(list(ops_arr))

    merged, base_layout = build_layered_surgery_code(
        data_code, v0_union, num_layers=num_layers,
        validate_logical_op=False, validate_commutation=False,
    )

    # chi rows are indexed by V_0_union vertices; v0_indices[k] = data qubit
    # index for chi row k. For each logical i, chi_group_per_logical[i] is
    # the list of chi row indices whose corresponding V_0 vertex is in
    # supp(op_i).
    v0_indices = np.asarray(base_layout.v0_indices)
    n_x_data = int(np.sum(base_layout.hx_row_kind == "data"))
    chi_groups: list[tuple[int, ...]] = []
    for op in ops_arr:
        # row indices for chi rows that touch a V_0 vertex in supp(op).
        mask = op[v0_indices].astype(bool)
        chi_row_positions = np.flatnonzero(mask) + n_x_data
        chi_groups.append(tuple(int(r) for r in chi_row_positions))

    layout = MultiSurgeryLayout(
        base_layout=base_layout,
        logical_ops=ops_arr,
        set_valued_port=set_valued_port,
        chi_group_per_logical=tuple(chi_groups),
    )
    return merged, layout
