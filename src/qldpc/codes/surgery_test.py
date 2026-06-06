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
from qldpc.codes.surgery import SurgeryLayout, load_webster_seed_set
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


from qldpc.codes.surgery import build_layered_surgery_code


def _assert_css_and_logical_count(
    merged: codes.CSSCode,
    data: codes.CSSCode,
) -> None:
    """Merged code satisfies CSS commutation and has dimension k_data − 1."""
    assert merged.is_subsystem_code is False
    assert np.all((merged.matrix_x @ merged.matrix_z.T) == 0)
    assert merged.dimension == data.dimension - 1


def test_build_surgery_steane_L1_integration() -> None:
    """Steane L=1: merged code is CSS, has k_merged = 0, layout is consistent."""
    code, logical_x = _steane_logical_x()
    merged, layout = build_layered_surgery_code(
        code, np.asarray(logical_x).astype(np.int_), num_layers=1
    )

    _assert_css_and_logical_count(merged, code)
    assert layout.num_data_qubits == code.num_qubits
    assert layout.num_layers == 1
    assert layout.num_ancilla_qubits == layout.qubit_layer.size - code.num_qubits
    assert merged.num_qubits == code.num_qubits + layout.num_ancilla_qubits

    # Data qubits marked layer 0, ancilla marked layer 1.
    assert np.all(layout.qubit_layer[: code.num_qubits] == 0)
    assert np.all(layout.qubit_layer[code.num_qubits :] == 1)


def test_build_surgery_steane_L3_integration() -> None:
    """Steane L=3 exercises the loop body for >= 1 odd and >= 1 even ancilla layer."""
    code, logical_x = _steane_logical_x()
    merged, layout = build_layered_surgery_code(
        code, np.asarray(logical_x).astype(np.int_), num_layers=3
    )
    _assert_css_and_logical_count(merged, code)
    assert layout.num_layers == 3

    # Qubit-layer labels appear in {0, 1, 2, 3}.
    assert set(np.unique(layout.qubit_layer).tolist()) <= {0, 1, 2, 3}

    # Layout row-kind labels match expected counts.
    n_x_data = code.matrix_x.shape[0]
    n_z_data = code.matrix_z.shape[0]
    assert int(np.sum(layout.hx_row_kind == "data")) == n_x_data
    assert int(np.sum(layout.hz_row_kind == "data")) == n_z_data
    assert "ancilla_L1" in set(layout.hx_row_kind.tolist())
    assert "ancilla_L3" in set(layout.hx_row_kind.tolist())
    assert "ancilla_L2" in set(layout.hz_row_kind.tolist())


def test_build_surgery_layout_row_counts_match_matrices() -> None:
    """hx_row_kind / hz_row_kind lengths == merged check counts."""
    code, logical_x = _steane_logical_x()
    merged, layout = build_layered_surgery_code(
        code, np.asarray(logical_x).astype(np.int_), num_layers=3
    )
    assert layout.hx_row_kind.size == merged.matrix_x.shape[0]
    assert layout.hz_row_kind.size == merged.matrix_z.shape[0]


def test_build_surgery_small_hgp_L1() -> None:
    """Cross-code coverage on a small HGPCode."""
    seed = 0
    classical = codes.ClassicalCode.random(4, 2, seed=seed)
    hgp = codes.HGPCode(classical)
    logical_x = hgp.get_logical_ops(Pauli.X)[0]
    merged, layout = build_layered_surgery_code(
        hgp, np.asarray(logical_x).astype(np.int_), num_layers=1
    )
    _assert_css_and_logical_count(merged, hgp)
    assert layout.num_layers == 1
    assert layout.num_data_qubits == hgp.num_qubits


def test_webster_observable_equals_logical_x_on_data() -> None:
    """Webster Eq. (1) algebraic identity for the noise-free observable.

    Claim: with gadget qubits kappa_j initialized to |0>, measuring the merged
    code's stabilizers and taking the product of the chi_i (new X-check)
    outcomes equals the X_M eigenvalue. The proof is purely algebraic:

        Pi_i chi_i = (Pi_{i in V_0} X_{q_i}) * Pi_j X_{kappa_j}^{|S_j cap supp(L)| mod 2}

    and |S_j cap supp(L)| == 0 mod 2 for every Z-stabilizer S_j of the data
    code (because Z-stabilizers commute with the logical X X_M). So the
    second factor is identity and the first factor is X_M on data qubits.

    Equivalently, the XOR of all chi_i rows of merged.matrix_x, restricted
    to data columns, equals logical_op, and restricted to ancilla columns
    equals 0. This is a pure GF(2) identity and is the noise-free core
    that the Section 7 notebook's logical observable definition relies on.
    """
    code, logical_x = _steane_logical_x()
    arr = np.asarray(logical_x).astype(np.int_)
    merged, layout = build_layered_surgery_code(code, arr, num_layers=1)

    chi_mask = layout.hx_row_kind == "ancilla_L1"
    chi_rows = np.asarray(merged.matrix_x[chi_mask]).astype(np.int_)
    product = chi_rows.sum(axis=0) % 2  # XOR of all chi_i rows

    n_data = layout.num_data_qubits
    assert np.array_equal(product[:n_data], arr), (
        "Webster Eq. (1): XOR of chi_i restricted to data should equal logical_op"
    )
    assert np.all(product[n_data:] == 0), (
        "Webster Eq. (1): XOR of chi_i restricted to ancilla should be zero "
        "(every Z-check of the data code touches V_0 in an even number of qubits)"
    )


def test_surgery_reexport_from_qldpc_codes() -> None:
    """``build_layered_surgery_code`` and ``SurgeryLayout`` are re-exported."""
    from qldpc import codes as codes_module

    assert hasattr(codes_module, "build_layered_surgery_code")
    assert hasattr(codes_module, "SurgeryLayout")
    assert "build_layered_surgery_code" in codes_module.__all__
    assert "SurgeryLayout" in codes_module.__all__


def test_load_webster_seed_set_returns_4_codes() -> None:
    """Each call to load_webster_seed_set with code_index in 0..3 returns a dict
    with the expected schema."""
    for code_index in range(4):
        data = load_webster_seed_set(code_index)
        assert data["l"] in (31, 63, 127, 255)
        assert isinstance(data["A"], list)
        assert isinstance(data["B"], list)
        assert len(data["seeds"]) == 4
        for seed in data["seeds"]:
            assert seed["name"] in ("X_bar_1", "Z_bar_1", "X_bar_k2p1", "Z_bar_k2p1")
            assert seed["pauli_type"] in ("X", "Z")
            assert isinstance(seed["L_support"], list)
            assert isinstance(seed["R_support"], list)


def test_load_webster_seed_set_out_of_range_raises() -> None:
    with pytest.raises(IndexError):
        load_webster_seed_set(4)
    with pytest.raises(IndexError):
        load_webster_seed_set(-1)


from qldpc.codes.surgery import _build_generalised_bicycle_code


def test_build_generalised_bicycle_code_dimension_and_shape() -> None:
    """For l=31, A={0,6,15}, B={0,5,7} (Webster code 0), the constructed code
    has 62 data qubits and dimension 10."""
    code = _build_generalised_bicycle_code(l=31, A_set=[0, 6, 15], B_set=[0, 5, 7])
    assert code.num_qubits == 62
    assert code.dimension == 10
    assert code.is_subsystem_code is False
    # CSS commutation
    assert np.all((code.matrix_x @ code.matrix_z.T) == 0)


def test_build_generalised_bicycle_code_l3_smoke() -> None:
    """Tiny l=3 case: A={0,1}, B={0,1} → known small bicycle code."""
    code = _build_generalised_bicycle_code(l=3, A_set=[0, 1], B_set=[0, 1])
    assert code.num_qubits == 6
    assert code.is_subsystem_code is False
    assert np.all((code.matrix_x @ code.matrix_z.T) == 0)
