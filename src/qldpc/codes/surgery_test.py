"""Unit tests for surgery.py — Cross et al. 2024 layered ancilla construction.

Copyright 2026 The qLDPC Authors.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import dataclasses

import galois
import numpy as np
import pytest

from qldpc import codes
from qldpc.codes.surgery import SurgeryLayout
from qldpc.objects import Pauli


def test_surgery_layout_construction() -> None:
    """SurgeryLayout is a frozen dataclass with the documented fields."""
    F = galois.GF(2)([[1, 0, 1], [0, 1, 1]])
    G = galois.GF(2).Zeros((0, 2))
    layout = SurgeryLayout(
        num_data_qubits=7,
        num_ancilla_qubits=2,
        num_layers=1,
        qubit_layer=np.array([0] * 7 + [1] * 2, dtype=np.int_),
        v0_indices=np.array([0, 3, 4], dtype=np.int_),
        c0_indices=np.array([0, 2], dtype=np.int_),
        F=F,
        G=G,
        hx_row_kind=np.array(["data"] * 3 + ["ancilla_L1"] * 3, dtype=object),
        hz_row_kind=np.array(["data"] * 3, dtype=object),
    )

    assert layout.num_data_qubits == 7
    assert layout.num_ancilla_qubits == 2
    assert layout.num_layers == 1
    assert np.array_equal(layout.F, F)
    assert layout.G.shape == (0, 2)
    assert dataclasses.is_dataclass(layout) and layout.__dataclass_params__.frozen


from qldpc.codes.surgery import _restrict_to_logical_support, _compute_gauge_fix


def _steane_logical_x() -> tuple[codes.SteaneCode, galois.FieldArray]:
    """Return Steane code and one of its logical-X representatives."""
    code = codes.SteaneCode()
    logical_x = code.get_logical_ops(Pauli.X)[0]
    return code, logical_x


def test_restrict_returns_F_equal_to_HZ_restriction() -> None:
    """F = H_Z[C_0, V_0] elementwise."""
    code, logical_x = _steane_logical_x()
    v0, c0, F = _restrict_to_logical_support(
        code, np.asarray(logical_x).astype(np.int_), num_layers=1, validate_logical_op=False
    )
    expected = code.matrix_z[c0][:, v0]
    assert np.array_equal(F, expected)
    assert v0.size > 0 and c0.size > 0
    assert F.shape == (c0.size, v0.size)


def test_restrict_rejects_wrong_shape() -> None:
    code, _ = _steane_logical_x()
    with pytest.raises(ValueError, match="shape"):
        _restrict_to_logical_support(code, np.zeros(5, dtype=np.int_), 1, False)


def test_restrict_rejects_non_binary() -> None:
    code, _ = _steane_logical_x()
    bad = np.zeros(code.num_qubits, dtype=np.int_)
    bad[0] = 2
    with pytest.raises(ValueError, match="binary"):
        _restrict_to_logical_support(code, bad, 1, False)


def test_restrict_rejects_zero_vector() -> None:
    code, _ = _steane_logical_x()
    with pytest.raises(ValueError, match="empty"):
        _restrict_to_logical_support(code, np.zeros(code.num_qubits, dtype=np.int_), 1, False)


def test_restrict_rejects_even_num_layers() -> None:
    code, logical_x = _steane_logical_x()
    arr = np.asarray(logical_x).astype(np.int_)
    with pytest.raises(ValueError, match="odd"):
        _restrict_to_logical_support(code, arr, 2, False)
    with pytest.raises(ValueError, match="odd"):
        _restrict_to_logical_support(code, arr, 0, False)


def test_restrict_rejects_non_commuting_op() -> None:
    """A vector that anticommutes with H_Z must be rejected."""
    code, _ = _steane_logical_x()
    single = np.zeros(code.num_qubits, dtype=np.int_)
    single[0] = 1
    with pytest.raises(ValueError, match="commute"):
        _restrict_to_logical_support(code, single, 1, False)


def test_restrict_rejects_stabilizer_when_validating() -> None:
    """A row of H_X is a stabilizer, not a logical operator."""
    code, _ = _steane_logical_x()
    stabilizer_row = np.asarray(code.matrix_x[0]).astype(np.int_)
    with pytest.raises(ValueError, match="stabilizer"):
        _restrict_to_logical_support(code, stabilizer_row, 1, validate_logical_op=True)


def test_restrict_accepts_stabilizer_when_skipping_validation() -> None:
    """With validate_logical_op=False, the row-span check is skipped."""
    code, _ = _steane_logical_x()
    stabilizer_row = np.asarray(code.matrix_x[0]).astype(np.int_)
    v0, c0, F = _restrict_to_logical_support(
        code, stabilizer_row, 1, validate_logical_op=False
    )
    assert v0.size > 0
    assert F.shape == (c0.size, v0.size)


def test_compute_gauge_fix_left_nulls_F() -> None:
    """G satisfies G @ F == 0 with shape (rank(left_null(F)), |C_0|)."""
    code, logical_x = _steane_logical_x()
    _, _, F = _restrict_to_logical_support(
        code, np.asarray(logical_x).astype(np.int_), 1, False
    )
    G = _compute_gauge_fix(F)
    assert G.shape[1] == F.shape[0]
    if G.shape[0] > 0:
        assert np.all(G @ F == 0)
    rank_F = int(np.sum(np.any(F.row_reduce() != 0, axis=1)))
    assert G.shape[0] == F.shape[0] - rank_F


def test_compute_gauge_fix_handles_full_rank_F() -> None:
    """When F has full row rank, G is empty (0 × |C_0|)."""
    field = galois.GF(2)
    F = field([[1, 0, 1], [0, 1, 1]])  # rank 2, |C_0| = 2 → G is 0 × 2
    G = _compute_gauge_fix(F)
    assert G.shape == (0, 2)


from qldpc.codes.surgery import _build_layered_blocks


def test_layered_blocks_L1_sizes() -> None:
    field = galois.GF(2)
    F = field([[1, 0, 1], [0, 1, 1]])  # |C_0|=2, |V_0|=3
    blocks = _build_layered_blocks(F, num_layers=1)
    assert blocks.n_v0 == 3
    assert blocks.n_c0 == 2
    assert blocks.ancilla_layer_sizes == [2]  # C_1 only
    assert blocks.total_ancilla == 2
    assert blocks.ancilla_col_slice(1) == slice(0, 2)


def test_layered_blocks_L3_sizes_and_slices() -> None:
    field = galois.GF(2)
    F = field([[1, 0, 1], [0, 1, 1]])
    blocks = _build_layered_blocks(F, num_layers=3)
    # L=3: layers 1 (C, |C_0|=2), 2 (V, |V_0|=3), 3 (C, |C_0|=2)
    assert blocks.ancilla_layer_sizes == [2, 3, 2]
    assert blocks.total_ancilla == 7
    assert blocks.ancilla_col_slice(1) == slice(0, 2)
    assert blocks.ancilla_col_slice(2) == slice(2, 5)
    assert blocks.ancilla_col_slice(3) == slice(5, 7)
    assert np.array_equal(blocks.F_T, F.T)


from qldpc.codes.surgery import _assemble_merged_HX


def test_assemble_HX_steane_L1_shape_and_structure() -> None:
    """For Steane L=1, H_X^merged has correct shape and per-row support pattern."""
    code, logical_x = _steane_logical_x()
    arr = np.asarray(logical_x).astype(np.int_)
    v0, _, F = _restrict_to_logical_support(code, arr, 1, False)
    blocks = _build_layered_blocks(F, 1)

    HX = _assemble_merged_HX(code, blocks, v0)

    n_x_data = code.matrix_x.shape[0]
    n_ancilla = blocks.total_ancilla  # = |C_0|
    expected_rows = n_x_data + blocks.n_v0
    expected_cols = code.num_qubits + n_ancilla
    assert HX.shape == (expected_rows, expected_cols)

    # Old data X-checks: zero on ancilla columns.
    assert np.all(HX[:n_x_data, code.num_qubits:] == 0)
    assert np.array_equal(HX[:n_x_data, :code.num_qubits], code.matrix_x)

    # V_1 X-check rows: Π_V_0 on data columns (1s at v0_indices, rows = identity)
    v1_rows = HX[n_x_data:]
    data_block = v1_rows[:, :code.num_qubits]
    # row v of data_block should have exactly a 1 at column v0[v]
    assert np.all(np.sum(data_block, axis=1) == 1)
    for v in range(blocks.n_v0):
        assert data_block[v, v0[v]] == 1

    # V_1 X-check rows: F^T on C_1 ancilla columns
    c1_block = v1_rows[:, code.num_qubits:]
    assert np.array_equal(c1_block, F.T)


from qldpc.codes.surgery import _assemble_merged_HZ


def test_assemble_HZ_steane_L1_shape_and_structure() -> None:
    """For Steane L=1: data Z-checks with C_0 extension + (possibly empty) gauge-fix."""
    code, logical_x = _steane_logical_x()
    arr = np.asarray(logical_x).astype(np.int_)
    v0, c0, F = _restrict_to_logical_support(code, arr, 1, False)
    G = _compute_gauge_fix(F)
    blocks = _build_layered_blocks(F, 1)

    HZ = _assemble_merged_HZ(code, blocks, G, c0)

    n_z_data = code.matrix_z.shape[0]
    n_ancilla = blocks.total_ancilla
    expected_rows = n_z_data + G.shape[0]  # no even ancilla layers for L=1
    expected_cols = code.num_qubits + n_ancilla
    assert HZ.shape == (expected_rows, expected_cols)

    # Old data Z-checks: original H_Z on data columns
    assert np.array_equal(HZ[:n_z_data, :code.num_qubits], code.matrix_z)

    # C_0 rows have identity entries on the corresponding C_1 ancilla columns.
    c1_slice = blocks.ancilla_col_slice(1)
    ancilla_block_z = HZ[:n_z_data, code.num_qubits:]
    for j, c_idx in enumerate(c0):
        assert ancilla_block_z[c_idx, c1_slice.start + j] == 1
    # Non-C_0 rows have zero on all ancilla columns.
    non_c0 = np.setdiff1d(np.arange(n_z_data), c0)
    assert np.all(ancilla_block_z[non_c0] == 0)

    # Gauge-fix rows (if any): zero on data, G on C_1.
    if G.shape[0] > 0:
        gauge_rows = HZ[n_z_data:]
        assert np.all(gauge_rows[:, :code.num_qubits] == 0)
        assert np.array_equal(gauge_rows[:, code.num_qubits:], G)
