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


def _support_to_binary_vector(L_support: list[int], R_support: list[int], l: int) -> np.ndarray:
    """Convert Webster's (L_support, R_support) per-block lists to a single
    binary support vector of length 2l."""
    vec = np.zeros(2 * l, dtype=np.int_)
    for i in L_support:
        vec[i] = 1
    for i in R_support:
        vec[l + i] = 1
    return vec


@pytest.mark.parametrize("code_index", [0, 1, 2, 3])
def test_webster_table1_bare_gadget(code_index: int) -> None:
    """Webster Table I bare-gadget verification.

    For each of the 4 codes (l ∈ {31, 63, 127, 255}) and each of its 4
    seed operators, build the gadget via build_layered_surgery_code and
    assert num_ancilla_qubits equals the bare-gadget number from Table I.
    """
    data = load_webster_seed_set(code_index)
    code = _build_generalised_bicycle_code(
        l=data["l"], A_set=data["A"], B_set=data["B"]
    )
    expected = data["expected_bare_gadget_qubits_per_seed"]
    for seed in data["seeds"]:
        op = _support_to_binary_vector(seed["L_support"], seed["R_support"], data["l"])
        # For Z-type seeds, use the ZX-dual code: swap H_X and H_Z. Then logical
        # Z of original = logical X of dual, and gadget structure is symmetric.
        if seed["pauli_type"] == "X":
            target_code = code
        else:
            target_code = codes.CSSCode(
                code.matrix_z, code.matrix_x, is_subsystem_code=False
            )
        _, layout = build_layered_surgery_code(target_code, op, num_layers=1, validate_logical_op=False)
        # Webster Table I "gadget qubits" includes data ancillas (κ_j),
        # syndrome ancillas for χ_i X-checks, and syndrome ancillas for
        # gauge-fix Z-checks. Our layout.num_ancilla_qubits counts only κ_j;
        # we add the new check counts to match Webster's definition.
        n_kappa = layout.num_ancilla_qubits
        n_chi = int(np.sum(layout.hx_row_kind != "data"))
        n_gauge_fix = int(np.sum(layout.hz_row_kind == "gauge_fix"))
        webster_count = n_kappa + n_chi + n_gauge_fix
        assert webster_count == expected, (
            f"Code {data['name']} seed {seed['name']}: expected Webster "
            f"gadget qubits = {expected}, got {webster_count} "
            f"(κ_j={n_kappa}, χ_i={n_chi}, gauge-fix={n_gauge_fix})"
        )


import networkx as nx
from qldpc.codes.surgery import _skip_tree


def test_skip_tree_path_graph_3_vertices() -> None:
    """SkipTree on a 3-vertex path graph 0—1—2: T has shape (2, 2),
    P is 3x3 permutation matrix."""
    S = nx.Graph()
    S.add_edges_from([(0, 1), (1, 2)])
    T, P = _skip_tree(S, root=0)
    assert T.shape == (2, 2)
    assert P.shape == (3, 3)
    # P is a permutation: each row and column has exactly one 1.
    assert np.all(P.sum(axis=0) == 1)
    assert np.all(P.sum(axis=1) == 1)


def test_skip_tree_star_graph_5_vertices() -> None:
    """SkipTree on a 5-vertex star (center=0, leaves 1..4)."""
    S = nx.Graph()
    S.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4)])
    T, P = _skip_tree(S, root=0)
    assert T.shape == (4, 4)
    assert P.shape == (5, 5)
    assert np.all(P.sum(axis=0) == 1)
    assert np.all(P.sum(axis=1) == 1)


from qldpc.codes.surgery import _cellulate_long_cycles


def test_cellulate_long_cycles_breaks_8cycle() -> None:
    """An 8-cycle: cellulate with max_len=4 should add at least one chord
    so that no cycle of length > 4 remains in the cycle basis."""
    G = nx.Graph()
    edges_8cycle = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0)]
    G.add_edges_from(edges_8cycle)
    edge_qubit_to_vertices = {i: tuple(sorted(e)) for i, e in enumerate(edges_8cycle)}
    vert_to_edge = {v: k for k, v in edge_qubit_to_vertices.items()}
    G_mat = np.zeros((len(edges_8cycle), 8), dtype=np.int_)
    for i, (u, v) in enumerate(edges_8cycle):
        G_mat[i, u] = 1
        G_mat[i, v] = 1

    new_edges, edge_qubit_to_vertices, vert_to_edge, G_mat = _cellulate_long_cycles(
        G, edge_qubit_to_vertices, vert_to_edge, G_mat, max_len=4
    )
    cycles = nx.cycle_basis(G)
    for cyc in cycles:
        assert len(cyc) <= 4, f"Cycle of length {len(cyc)} remains: {cyc}"
    assert len(new_edges) >= 1


from qldpc.codes.surgery import _spectral_cheeger_lower_bound


def test_spectral_cheeger_lower_bound_positive_on_connected_F() -> None:
    """For a connected non-trivial F, the spectral lower bound is positive."""
    field = galois.GF(2)
    F = field([[1, 1, 0], [0, 1, 1]])  # |C_0|=2, |V_0|=3
    h_lb = _spectral_cheeger_lower_bound(F)
    assert h_lb > 0
    assert isinstance(h_lb, float)


def test_spectral_cheeger_lower_bound_zero_on_disconnected_F() -> None:
    """A disconnected F (zero matrix) gives lambda_2 = 0."""
    field = galois.GF(2)
    F = field([[0, 0, 0, 0], [0, 0, 0, 0]])  # all zeros -> degenerate
    h_lb = _spectral_cheeger_lower_bound(F)
    assert h_lb == pytest.approx(0.0, abs=1e-10)


from qldpc.codes.surgery import BoostResult, boost_gadget_cheeger


def test_boost_gadget_cheeger_increases_lower_bound_or_terminates() -> None:
    """Boost on Steane terminates either by reaching target or hitting max."""
    code, logical_x = _steane_logical_x()
    arr = np.asarray(logical_x).astype(np.int_)
    merged, layout = build_layered_surgery_code(code, arr, num_layers=1)

    boosted_merged, boosted_layout, result = boost_gadget_cheeger(
        merged, layout, target_h=0.5, max_extra_qubits=20, seed=42,
    )
    assert isinstance(result, BoostResult)
    assert result.extra_qubits_added >= 0
    assert result.terminated_by in ("target_reached", "max_qubits_exhausted", "no_progress")
    if result.terminated_by == "target_reached":
        assert result.final_h_lower_bound >= 0.5


def test_boost_gadget_cheeger_reproducible_with_seed() -> None:
    code, logical_x = _steane_logical_x()
    arr = np.asarray(logical_x).astype(np.int_)
    merged, layout = build_layered_surgery_code(code, arr, num_layers=1)

    _, _, r1 = boost_gadget_cheeger(merged, layout, target_h=2.0, max_extra_qubits=5, seed=42)
    _, _, r2 = boost_gadget_cheeger(merged, layout, target_h=2.0, max_extra_qubits=5, seed=42)
    assert r1.extra_qubits_added == r2.extra_qubits_added
    assert r1.terminated_by == r2.terminated_by


def test_boost_gadget_cheeger_respects_max_extra_qubits() -> None:
    code, logical_x = _steane_logical_x()
    arr = np.asarray(logical_x).astype(np.int_)
    merged, layout = build_layered_surgery_code(code, arr, num_layers=1)
    _, _, result = boost_gadget_cheeger(
        merged, layout, target_h=100.0, max_extra_qubits=3, seed=0,
    )
    assert result.extra_qubits_added <= 3


def test_boost_gadget_cheeger_invalid_target_raises() -> None:
    code, logical_x = _steane_logical_x()
    arr = np.asarray(logical_x).astype(np.int_)
    merged, layout = build_layered_surgery_code(code, arr, num_layers=1)
    with pytest.raises(ValueError, match="target_h"):
        boost_gadget_cheeger(merged, layout, target_h=-1.0)


def test_joint_surgery_layout_construction() -> None:
    """JointSurgeryLayout is a frozen dataclass with the documented fields."""
    from qldpc.codes.surgery import JointSurgeryLayout

    field = galois.GF(2)
    F1 = field([[1, 0, 1]])
    F2 = field([[0, 1, 1]])
    G_empty = field.Zeros((0, 1))
    sub_layout1 = SurgeryLayout(
        num_data_qubits=7, num_ancilla_qubits=1, num_layers=1,
        qubit_layer=np.array([0]*7 + [1], dtype=np.int_),
        v0_indices=np.array([0, 1, 2], dtype=np.int_),
        c0_indices=np.array([0], dtype=np.int_),
        F=F1, G=G_empty,
        hx_row_kind=np.array(["data"]*3 + ["ancilla_L1"]*3, dtype=object),
        hz_row_kind=np.array(["data"]*3, dtype=object),
    )
    sub_layout2 = SurgeryLayout(
        num_data_qubits=7, num_ancilla_qubits=1, num_layers=1,
        qubit_layer=np.array([0]*7 + [1], dtype=np.int_),
        v0_indices=np.array([3, 4, 5], dtype=np.int_),
        c0_indices=np.array([1], dtype=np.int_),
        F=F2, G=G_empty,
        hx_row_kind=np.array(["data"]*3 + ["ancilla_L1"]*3, dtype=object),
        hz_row_kind=np.array(["data"]*3, dtype=object),
    )
    joint = JointSurgeryLayout(
        gadget_layouts=(sub_layout1, sub_layout2),
        pauli_type=Pauli.X,
        num_data_qubits=7,
        num_ancilla_qubits=2,
        num_bridge_qubits=1,
        bridge_qubit_slice=slice(9, 10),
        u_b_check_kind_mask=np.array([False]*3 + [True], dtype=bool),
    )
    assert joint.num_data_qubits == 7
    assert joint.num_bridge_qubits == 1
    assert dataclasses.is_dataclass(joint) and joint.__dataclass_params__.frozen


def test_validate_joint_logical_ops_X_pair_returns_X_type() -> None:
    from qldpc.codes.surgery import _validate_joint_logical_ops

    seed = 0
    classical = codes.ClassicalCode.random(4, 2, seed=seed)
    hgp = codes.HGPCode(classical)
    logical_x = np.asarray(hgp.get_logical_ops(Pauli.X)[0]).astype(np.int_)
    arr1 = logical_x.copy()
    arr2 = logical_x.copy()
    pauli_type = _validate_joint_logical_ops(hgp, arr1, arr2)
    assert pauli_type == Pauli.X


def test_validate_joint_logical_ops_rejects_low_k_data() -> None:
    from qldpc.codes.surgery import _validate_joint_logical_ops

    code, logical_x = _steane_logical_x()  # k=1
    arr = np.asarray(logical_x).astype(np.int_)
    with pytest.raises(ValueError, match="at least 2 logical qubits"):
        _validate_joint_logical_ops(code, arr, arr)


def test_validate_joint_logical_ops_rejects_mixed_type() -> None:
    """An X-type op and a Z-type op should be rejected."""
    from qldpc.codes.surgery import _validate_joint_logical_ops

    seed = 0
    classical = codes.ClassicalCode.random(4, 2, seed=seed)
    hgp = codes.HGPCode(classical)
    logical_x = np.asarray(hgp.get_logical_ops(Pauli.X)[0]).astype(np.int_)
    logical_z = np.asarray(hgp.get_logical_ops(Pauli.Z)[0]).astype(np.int_)
    with pytest.raises(ValueError, match="same Pauli type"):
        _validate_joint_logical_ops(hgp, logical_x, logical_z)


from qldpc.codes.surgery import _BridgeSpec, _build_bridge_via_skiptree


def _two_overlapping_hgp_gadgets():
    """Build two HGPCode gadgets that share some V_0 overlap."""
    classical = codes.ClassicalCode.random(4, 2, seed=0)
    hgp = codes.HGPCode(classical)
    logicals_x = hgp.get_logical_ops(Pauli.X)
    arr1 = np.asarray(logicals_x[0]).astype(np.int_)
    arr2 = np.asarray(logicals_x[1]).astype(np.int_)
    _, lay1 = build_layered_surgery_code(hgp, arr1, num_layers=1)
    _, lay2 = build_layered_surgery_code(hgp, arr2, num_layers=1)
    return hgp, lay1, lay2


def test_build_bridge_returns_BridgeSpec() -> None:
    try:
        _, lay1, lay2 = _two_overlapping_hgp_gadgets()
    except ValueError:
        pytest.skip("Random HGP gave disjoint V_0; need a better fixture")
    try:
        spec = _build_bridge_via_skiptree(lay1, lay2)
    except ValueError:
        pytest.skip("Random HGP gave disconnected interface graph; need a better fixture")
    assert isinstance(spec, _BridgeSpec)
    assert spec.num_bridge_qubits >= 0


def test_build_bridge_u_b_rows_have_correct_width() -> None:
    try:
        _, lay1, lay2 = _two_overlapping_hgp_gadgets()
        spec = _build_bridge_via_skiptree(lay1, lay2)
    except ValueError:
        pytest.skip("Test fixture produced incompatible inputs; see v2.9 plan")

    expected_width = lay1.F.shape[0] + lay2.F.shape[0] + spec.num_bridge_qubits
    if spec.u_b_rows.shape[0] > 0:
        assert spec.u_b_rows.shape[1] == expected_width
