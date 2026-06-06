"""Gadget construction for QLDPC lattice surgery.

Primary reference: Webster, Smith, Cohen, arXiv:2511.15989 §II.A Steps 1-3,
an explicit pedagogically clean 3-step recipe for building a logical-X
measurement gadget on any CSS code. The default ``num_layers=1`` mode
implements Webster's 3 steps verbatim; ``num_layers > 1`` activates the
multi-layer fallback of Cross et al. 2024 (arXiv:2407.18393 §III) for codes
whose induced Tanner graph has insufficient boundary Cheeger constant.

The two formulations produce the same merged code at L=1: Webster's "gadget
qubit kappa_j for each adjacent Z-check S_j" = Cross's "C_1 ancilla qubit
at the same index as the C_0 Z-check"; Webster's "X-check chi_i wired to
kappa_j iff q_i in S_j" = Cross's `[Pi_V_0, F^T]` row pattern.

See docs/superpowers/specs/2026-06-05-cross-layered-ancilla-design.md for
the full paper traceability and design rationale.

Copyright 2026 The qLDPC Authors.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import dataclasses

import galois
import numpy as np
import numpy.typing as npt

from .common import CSSCode


@dataclasses.dataclass(frozen=True, eq=False)
class SurgeryLayout:
    """Provenance of qubits and checks in a merged surgery code.

    Returned by ``build_layered_surgery_code`` alongside the merged ``CSSCode``.
    Downstream pipelines (circuit synthesis, decoder configuration, plotting)
    can use the layout to distinguish data qubits from ancilla and identify
    which check rows are gauge-fixing.

    Attributes:
        num_data_qubits: Number of qubits in the original data code.
        num_ancilla_qubits: Total ancilla qubits across all L layers.
        num_layers: L. Always odd, >= 1.
        qubit_layer: Length (num_data + num_ancilla) array. Value 0 marks a
            data qubit; values 1..L mark the layer index of an ancilla qubit.
        v0_indices: Indices (within data qubits) of supp(X̄_M) = V_0.
        c0_indices: Row indices (within H_Z of data code) of Z-checks adjacent
            to V_0 = C_0.
        F: Step-1 restriction matrix; shape (|C_0|, |V_0|), equal to
            ``data_code.matrix_z[c0_indices][:, v0_indices]``.
        G: Step-4 gauge-fix basis; rows span the left null space of F (i.e.
            ``G @ F == 0``); shape (rank(left_null(F)), |C_0|).
        hx_row_kind: Length (num_x_checks_merged) string array. Values:
            "data" for old X-checks, "ancilla_L{i}" for new X-checks added by
            odd layer i in {1, 3, ..., L}.
        hz_row_kind: Length (num_z_checks_merged) string array. Values:
            "data" for old Z-checks, "ancilla_L{i}" for new Z-checks added by
            even layer i in {2, 4, ..., L-1}, "gauge_fix" for U_L rows.
    """

    num_data_qubits: int
    num_ancilla_qubits: int
    num_layers: int
    qubit_layer: npt.NDArray[np.int_]
    v0_indices: npt.NDArray[np.int_]
    c0_indices: npt.NDArray[np.int_]
    F: galois.FieldArray
    G: galois.FieldArray
    hx_row_kind: npt.NDArray
    hz_row_kind: npt.NDArray


def _restrict_to_logical_support(
    data_code: CSSCode,
    logical_op: npt.ArrayLike,
    num_layers: int,
    validate_logical_op: bool,
) -> tuple[np.ndarray, np.ndarray, galois.FieldArray]:
    """Compute V_0, C_0, F per Cross 2024 §III Step 1, with input validation.

    See spec §5 for the validation contract. Returns the indices V_0 (qubit
    columns) and C_0 (Z-check rows) into the data code, plus the restriction
    matrix F = H_Z[C_0, V_0] as a GF(2) ``galois.FieldArray``.

    The expensive row-span check (rejecting stabilizers as logical operators)
    is gated by ``validate_logical_op`` — see Task 3 / spec §5 item 6.
    """
    if data_code.is_subsystem_code:
        raise ValueError(
            "build_layered_surgery_code requires a stabilizer CSSCode, not a "
            "subsystem code."
        )
    if num_layers < 1 or num_layers % 2 != 1:
        raise ValueError(f"num_layers must be odd and >= 1, got {num_layers}.")

    field = data_code.field
    logical_op_arr = np.asarray(logical_op)
    n_data = data_code.num_qubits

    if logical_op_arr.shape != (n_data,):
        raise ValueError(
            f"logical_op has shape {logical_op_arr.shape}, expected ({n_data},)."
        )
    int_view = logical_op_arr.astype(np.int_, copy=False)
    if not np.all((int_view == 0) | (int_view == 1)):
        raise ValueError("logical_op must be binary (values in {0, 1}).")

    v0_indices = np.flatnonzero(int_view)
    if v0_indices.size == 0:
        raise ValueError("logical_op support V_0 is empty (logical_op is the zero vector).")

    logical_op_gf = field(int_view)
    hz = data_code.matrix_z
    # commutation with Z-stabilizers: H_Z @ X̄^T == 0 over GF(2)
    if np.any(hz @ logical_op_gf != 0):
        raise ValueError(
            "logical_op does not commute with Z-stabilizers (H_Z @ logical_op != 0)."
        )

    if validate_logical_op:
        hx = data_code.matrix_x
        # rank over GF(2): count nonzero rows of row-reduced form
        rank_hx = int(np.sum(np.any(hx.row_reduce() != 0, axis=1)))
        augmented = field(np.vstack([np.asarray(hx), logical_op_gf.reshape(1, -1)]))
        rank_aug = int(np.sum(np.any(augmented.row_reduce() != 0, axis=1)))
        if rank_aug == rank_hx:
            raise ValueError(
                "logical_op lies in the row span of H_X — it is a stabilizer, "
                "not a logical operator. Pass validate_logical_op=False to skip "
                "this check."
            )

    # Identify C_0: Z-check rows whose support intersects V_0.
    c0_mask = np.any(hz[:, v0_indices] != 0, axis=1)
    c0_indices = np.flatnonzero(c0_mask)
    if c0_indices.size == 0:
        raise ValueError(
            "No Z-checks of the data code touch V_0; the ancilla system cannot "
            "be constructed (degenerate logical operator)."
        )

    F = hz[c0_indices][:, v0_indices]
    return v0_indices, c0_indices, F


def _compute_gauge_fix(F: galois.FieldArray) -> galois.FieldArray:
    """Compute G whose rows form a basis of the left null space of F.

    Cross 2024 §III Step 4: ``null(F) = {c : c @ F == 0}``. We promote the
    CKBB gauge operators to stabilizers by introducing ``rank(null(F))`` new
    Z-checks U_L connected via G. Returns G with shape (rank, |C_0|).
    """
    return F.left_null_space()
