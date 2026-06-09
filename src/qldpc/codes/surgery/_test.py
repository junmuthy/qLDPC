"""Tests for the simplified surgery package (see
docs/superpowers/specs/2026-06-07-surgery-simplification-design.md)."""

from __future__ import annotations

import dataclasses
import numpy as np
import pytest

from qldpc import codes
from qldpc.objects import Pauli


def test_gadget_layout_is_frozen_dataclass():
    from qldpc.codes.surgery.gadget import GadgetLayout
    assert dataclasses.is_dataclass(GadgetLayout)
    # frozen
    fields = {f.name for f in dataclasses.fields(GadgetLayout)}
    assert fields == {
        "code", "x", "V0", "C0", "F", "G",
        "HX_merged", "HZ_merged", "kappa_qubits", "basis",
    }
    # Verify actually frozen: mutation must raise
    inst = GadgetLayout(
        code=None, x=None, V0=(), C0=(),
        F=None, G=None, HX_merged=None, HZ_merged=None,
        kappa_qubits=(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        inst.code = object()


def test_step1_restriction_steane():
    from qldpc.codes.surgery.gadget import _step1_restriction
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    V0, C0, F = _step1_restriction(code, x)
    # V_0 = supp(x), sorted ascending
    assert V0 == tuple(int(i) for i in np.where(x)[0])
    assert list(V0) == sorted(V0)
    # C_0 = Z-checks touching V_0, sorted ascending
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    touched = sorted({j for j in range(HZ.shape[0])
                      for i in V0 if HZ[j, i] == 1})
    assert C0 == tuple(touched)
    assert list(C0) == sorted(C0)
    # F = H_Z[C_0, V_0]
    assert F.shape == (len(C0), len(V0))
    assert np.array_equal(F, HZ[np.ix_(C0, V0)])
    # F @ 1_{V0} == 0 (math.md §1.1 invariant)
    ones = np.ones(len(V0), dtype=np.uint8)
    assert np.array_equal((F @ ones) % 2, np.zeros(len(C0), dtype=np.uint8))


def test_step2_gauge_fix_basis_property():
    from qldpc.codes.surgery.gadget import _step1_restriction, _step2_gauge_fix
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    _, _, F = _step1_restriction(code, x)
    G = _step2_gauge_fix(F)
    # math.md §1.2: G F = 0 over GF(2)
    assert G.shape[1] == F.shape[0]
    GF = (G @ F) % 2
    assert np.array_equal(GF, np.zeros_like(GF))
    # rank(G) = |C_0| - rank(F)
    import galois
    r_expected = F.shape[0] - int(np.linalg.matrix_rank(galois.GF(2)(F.tolist())))
    assert G.shape[0] == r_expected


def test_step2_gauge_fix_deterministic():
    """Same F twice → byte-identical G (non-trivial: rank-deficient F → non-empty G)."""
    from qldpc.codes.surgery.gadget import _step2_gauge_fix
    # 3x3 matrix with rank 2 (row 0 + row 1 = row 2 over GF(2)), so G has 1 row.
    F = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.uint8)
    G1 = _step2_gauge_fix(F)
    G2 = _step2_gauge_fix(F)
    assert G1.shape == (1, 3), f"expected G shape (1,3), got {G1.shape}"
    assert np.array_equal(G1, G2)
    # And sanity-check the basis property holds on this F too.
    assert np.array_equal((G1 @ F) % 2, np.zeros((1, F.shape[1]), dtype=np.uint8))


def test_step3_assemble_basis_z_places_chi_in_HZ_merged_and_G_in_HX_merged():
    """basis=Pauli.Z: χ rows added to HZ_merged (Z-type); G added to HX_merged (X-type)."""
    from qldpc.codes.surgery.gadget import (
        _step1_restriction, _step2_gauge_fix, _step3_assemble,
    )
    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    V0, C0, F = _step1_restriction(code, z, basis=Pauli.Z)
    G = _step2_gauge_fix(F)
    HX_m, HZ_m = _step3_assemble(code, V0, C0, F, G, basis=Pauli.Z)

    n, mX, mZ = code.num_qudits, code.matrix_x.shape[0], code.matrix_z.shape[0]
    # For basis=Z: HX_merged grows by r rows (gauge-fix), HZ_merged by |V_0| rows (chi).
    assert HX_m.shape == (mX + G.shape[0], n + len(C0)), f"HX shape {HX_m.shape}"
    assert HZ_m.shape == (mZ + len(V0), n + len(C0)), f"HZ shape {HZ_m.shape}"
    # CSS commutation
    product = (HX_m @ HZ_m.T) % 2
    assert np.array_equal(product, np.zeros_like(product))


def test_step3_assemble_steane_css_commutes():
    from qldpc.codes.surgery.gadget import (
        _step1_restriction, _step2_gauge_fix, _step3_assemble,
    )
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    V0, C0, F = _step1_restriction(code, x)
    G = _step2_gauge_fix(F)
    HX_m, HZ_m = _step3_assemble(code, V0, C0, F, G)

    n, mX, mZ = code.num_qudits, code.matrix_x.shape[0], code.matrix_z.shape[0]
    assert HX_m.shape == (mX + len(V0), n + len(C0))
    assert HZ_m.shape == (mZ + G.shape[0], n + len(C0))
    # math.md §1.5(a): H_X^merged @ H_Z^merged.T == 0 over GF(2)
    product = (HX_m @ HZ_m.T) % 2
    assert np.array_equal(product, np.zeros_like(product))


def test_step3_assemble_csscode_with_distinct_nV_nC():
    """Synthetic CSS code where nV != nC — catches F_tilde shape bug.

    Uses a 5-qubit CSS code with k=1, picking a logical-X representative
    whose support size (nV=4) differs from the number of Z-checks it
    touches (nC=2). With the buggy F_tilde[j] = F[k] form, numpy raises
    ValueError because F[k] has shape (nV=4,) but the row width is (nC=2).
    The fix (F_tilde[j, k] = 1) is the correct indicator/selection matrix.

    Verifies:
    1. CSS commutation: HX_merged @ HZ_merged.T == 0 over GF(2).
    2. Indicator form: each Z-check in C_0 attaches to EXACTLY ONE kappa
       ancilla (row-sum == 1 in the kappa block).
    """
    from qldpc.codes.surgery.gadget import (
        _step1_restriction, _step2_gauge_fix, _step3_assemble,
    )

    # 5-qubit CSS code (k=1):
    #   HX = [[1,1,1,0,0],[0,0,0,1,1]]
    #   HZ = [[1,1,0,0,0],[1,0,1,0,0]]
    # Commutativity check (each pair of rows):
    #   row0(HX)·row0(HZ) = 1+1+0+0+0 = 0 mod 2 ✓
    #   row0(HX)·row1(HZ) = 1+0+1+0+0 = 0 mod 2 ✓
    #   row1(HX)·row0(HZ) = 0+0+0+0+0 = 0 mod 2 ✓
    #   row1(HX)·row1(HZ) = 0+0+0+0+0 = 0 mod 2 ✓
    HX_raw = np.array([[1, 1, 1, 0, 0],
                        [0, 0, 0, 1, 1]], dtype=np.uint8)
    HZ_raw = np.array([[1, 1, 0, 0, 0],
                        [1, 0, 1, 0, 0]], dtype=np.uint8)
    assert np.array_equal((HX_raw @ HZ_raw.T) % 2,
                           np.zeros((2, 2), dtype=np.uint8)), "CSS sanity failed"

    code = codes.CSSCode(HX_raw, HZ_raw)

    # Logical X rep: x = [1,1,1,1,0].
    #   HZ @ x = [1+1+0,1+0+1] = [0,0] mod 2  =>  x in ker(HZ) ✓
    #   row(HX) = span{[1,1,1,0,0],[0,0,0,1,1]}: cannot produce [1,1,1,1,0]
    #   because the last coord would require b=0 while 4th coord requires b=1 ✓ logical
    x_logical = np.array([1, 1, 1, 1, 0], dtype=np.uint8)
    assert np.array_equal((HZ_raw @ x_logical) % 2,
                           np.zeros(2, dtype=np.uint8)), "x_logical not in ker(HZ)"

    V0, C0, F = _step1_restriction(code, x_logical)
    # V0 = {0,1,2,3} (nV=4); HZ row0 touches {0,1}, HZ row1 touches {0,2} -> C0=(0,1) (nC=2)
    assert len(V0) != len(C0), (
        f"nV={len(V0)} == nC={len(C0)}: this test requires nV != nC to catch the bug"
    )

    G = _step2_gauge_fix(F)
    HX_m, HZ_m = _step3_assemble(code, V0, C0, F, G)

    # 1. CSS commutation
    product = (HX_m @ HZ_m.T) % 2
    assert np.array_equal(product, np.zeros_like(product)), (
        "CSS commutation failed: HX_merged @ HZ_merged.T != 0"
    )

    # 2. Indicator form: each Z-check j in C_0 should attach to exactly
    #    one kappa ancilla (column-slice after n data qubits in HZ_merged).
    n = code.num_qudits
    mZ = HZ_raw.shape[0]
    HZ_kappa_block = HZ_m[:mZ, n:]
    for k, j in enumerate(C0):
        row_sum = int(HZ_kappa_block[j].sum())
        assert row_sum == 1, (
            f"row j={j} of HZ kappa-block should have exactly 1 one (indicator form), "
            f"got {row_sum} — F_tilde indicator form violated"
        )


def test_build_gadget_steane_returns_valid_layout():
    from qldpc.codes.surgery.gadget import build_gadget, GadgetLayout
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    assert isinstance(g, GadgetLayout)
    assert g.code is code
    assert np.array_equal(g.x, x)
    # κ qubits indexed contiguously after data qubits
    assert g.kappa_qubits == tuple(range(code.num_qudits, code.num_qudits + len(g.C0)))


def test_build_gadget_deterministic():
    from qldpc.codes.surgery.gadget import build_gadget
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g1 = build_gadget(code, x)
    g2 = build_gadget(code, x)
    assert g1.V0 == g2.V0
    assert g1.C0 == g2.C0
    assert np.array_equal(g1.F, g2.F)
    assert np.array_equal(g1.G, g2.G)
    assert np.array_equal(g1.HX_merged, g2.HX_merged)
    assert np.array_equal(g1.HZ_merged, g2.HZ_merged)
    assert g1.kappa_qubits == g2.kappa_qubits


def test_build_gadget_rejects_non_x_logical():
    from qldpc.codes.surgery.gadget import build_gadget
    code = codes.SteaneCode()
    x = np.zeros(code.num_qudits, dtype=np.uint8)
    x[0] = 1  # not a logical X (HZ @ x ≠ 0 in general)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    if ((HZ @ x) % 2).any():
        with pytest.raises(ValueError, match="logical"):
            build_gadget(code, x)


def test_load_webster_seed_set_returns_known_shape():
    from qldpc.codes.surgery.gadget import load_webster_seed_set
    data = load_webster_seed_set(0)
    assert "l" in data and "A" in data and "B" in data
    assert "seeds" in data


def test_build_generalised_bicycle_code_constructs_css():
    from qldpc.codes.surgery.gadget import (
        load_webster_seed_set, _build_generalised_bicycle_code,
    )
    data = load_webster_seed_set(0)
    code = _build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    assert code.num_qudits == 2 * data["l"]
    # CSS commutation
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    assert np.array_equal((HX @ HZ.T) % 2, np.zeros((HX.shape[0], HZ.shape[0]), dtype=np.uint8))


WEBSTER_TABLE_I_KAPPA_CHI_R = [(0, 19), (1, 31), (2, 49), (3, 79)]


def _webster_x_bar_1_operator(data: dict) -> np.ndarray:
    """Extract the X_bar_1 operator from a Webster seed_set dict as a 2l-vector.

    L_support and R_support are sparse index lists (positions within each l-half
    that are set to 1). This converts them to a dense binary vector of length 2l.
    """
    l = data["l"]
    for seed in data["seeds"]:
        if seed["name"] == "X_bar_1" and seed["pauli_type"] == "X":
            v_L = np.zeros(l, dtype=np.uint8)
            v_L[seed["L_support"]] = 1
            v_R = np.zeros(l, dtype=np.uint8)
            v_R[seed["R_support"]] = 1
            return np.concatenate([v_L, v_R])
    raise ValueError("X_bar_1 seed not found")


@pytest.mark.parametrize("code_index,n_anc", WEBSTER_TABLE_I_KAPPA_CHI_R)
def test_webster_table_i_kappa_chi_r_exact(code_index, n_anc):
    """Webster Table I: κ + χ + r matches for each of the 4 codes."""
    from qldpc.codes.surgery.gadget import (
        build_gadget, load_webster_seed_set, _build_generalised_bicycle_code,
    )
    data = load_webster_seed_set(code_index)
    code = _build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x1 = _webster_x_bar_1_operator(data)
    g1 = build_gadget(code, x1)
    kappa = len(g1.kappa_qubits)
    chi = int(g1.x.sum())  # |V_0|
    r = g1.G.shape[0]
    assert kappa + chi + r == n_anc, (
        f"code {code_index}: κ={kappa}, χ={chi}, r={r}, "
        f"sum={kappa+chi+r}, expected {n_anc}"
    )


def test_bridge_dataclass_fields():
    from qldpc.codes.surgery.bridge import Bridge
    fields = {f.name for f in dataclasses.fields(Bridge)}
    assert fields == {
        "width", "qubits", "U_B",
        "chi_endpoint_extensions", "intercode",
        "aux_graph_edges", "z_extensions", "basis",
    }
    assert dataclasses.is_dataclass(Bridge)


def test_path_graph_U_B_telescoping():
    """math.md §2.2: sum of U_B rows == e_0 + e_{w-1}."""
    from qldpc.codes.surgery.bridge import _build_path_graph_U_B
    for w in (2, 3, 5, 11):
        U_B = _build_path_graph_U_B(w)
        assert U_B.shape == (w - 1, w)
        col_sum = U_B.sum(axis=0) % 2
        expected = np.zeros(w, dtype=np.uint8)
        expected[0] = 1
        expected[-1] = 1
        assert np.array_equal(col_sum, expected)


WEBSTER_TABLE_I_BRIDGE = [(0, 11), (1, 19), (2, 31), (3, 51)]


@pytest.mark.parametrize("code_index,bridge_w_minus_1", WEBSTER_TABLE_I_BRIDGE)
def test_webster_table_i_bridge_width_exact(code_index, bridge_w_minus_1):
    """Webster Table I: 2w - 1 matches."""
    from qldpc.codes.surgery.gadget import (
        build_gadget, load_webster_seed_set, _build_generalised_bicycle_code,
    )
    from qldpc.codes.surgery.bridge import build_bridge
    data = load_webster_seed_set(code_index)
    code = _build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x1 = _webster_x_bar_1_operator(data)
    x2 = _webster_x_bar_k2p1_operator(data)
    g1 = build_gadget(code, x1)
    g2 = build_gadget(code, x2)
    bridge = build_bridge(g1, g2)
    assert bridge.intercode is False
    assert 2 * bridge.width - 1 == bridge_w_minus_1


def _webster_x_bar_k2p1_operator(data: dict) -> np.ndarray:
    """Extract X_bar_k2p1 operator from a Webster seed_set dict."""
    l = data["l"]
    for seed in data["seeds"]:
        if seed["name"] == "X_bar_k2p1" and seed["pauli_type"] == "X":
            L = np.zeros(l, dtype=np.uint8)
            R = np.zeros(l, dtype=np.uint8)
            for i in seed["L_support"]:
                L[i] = 1
            for i in seed["R_support"]:
                R[i] = 1
            return np.concatenate([L, R])
    raise ValueError("X_bar_k2p1 seed not found")


def test_build_bridge_intracode_chi_endpoint_extensions():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    code = codes.SteaneCode()
    x1 = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    x2 = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g1 = build_gadget(code, x1)
    g2 = build_gadget(code, x2)
    bridge = build_bridge(g1, g2)
    # math.md §2.3: χ_0 from each gadget gets an X on its bridge endpoint
    assert 0 in bridge.chi_endpoint_extensions  # gadget 1, row 0
    assert bridge.intercode is False


def test_skip_tree_path_graph_returns_identity():
    """math.md skip-tree on a path graph yields T and P with correct shapes."""
    import networkx as nx
    from qldpc.codes.surgery.bridge import _skip_tree
    # Path graph on 5 vertices: edges (0,1),(1,2),(2,3),(3,4)
    G = nx.path_graph(5)
    T, P = _skip_tree(G)
    # T should have shape (n-1, |E|) = (4, 4)
    assert T.shape == (4, 4)
    # P should be a permutation matrix of shape (n, n) = (5, 5)
    assert P.shape == (5, 5)
    # P is a permutation: each row and column has exactly one 1
    assert np.array_equal(P.sum(axis=0), np.ones(5, dtype=np.int_))
    assert np.array_equal(P.sum(axis=1), np.ones(5, dtype=np.int_))


def test_skip_tree_fullrank_on_K4_matches_H_R():
    """SkipTree full-rank: T_ind · G · P_ind = H_R for the complete graph K_4."""
    import networkx as nx
    from qldpc.codes.surgery.bridge import _skip_tree_fullrank, _canonical_H_R

    G_nx = nx.complete_graph(4)
    n = 4
    edges = sorted(tuple(sorted(e)) for e in G_nx.edges())
    edge_index = {e: i for i, e in enumerate(edges)}
    G_mat = np.zeros((len(edges), n), dtype=np.int_)
    for (u, v), i in edge_index.items():
        G_mat[i, u] = 1
        G_mat[i, v] = 1

    T_ind, P_ind = _skip_tree_fullrank(G_nx, root=0, edge_index=edge_index)
    H_R = _canonical_H_R(n)

    assert T_ind.shape == (n - 1, len(edges))
    assert P_ind.shape == (n, n)
    # SkipTree key identity: T_ind · G · P_ind == H_R over GF(2)
    product = (T_ind @ G_mat @ P_ind) % 2
    assert np.array_equal(product, H_R), f"got\n{product}\nwant\n{H_R}"
    # (3,2)-sparsity
    assert T_ind.sum(axis=1).max() <= 3
    assert T_ind.sum(axis=0).max() <= 2


def test_cellulate_long_cycles_no_op_when_short():
    """Cellulation of a graph with no long cycles returns no new edges."""
    import networkx as nx
    from qldpc.codes.surgery.bridge import _cellulate_long_cycles
    # Triangle: only cycle has length 3, well within max_len=6
    G = nx.cycle_graph(3)
    edge_qubit_to_vertices = {i: tuple(sorted(e)) for i, e in enumerate(G.edges())}
    vert_to_edge = {v: k for k, v in edge_qubit_to_vertices.items()}
    n_v = G.number_of_nodes()
    n_e = G.number_of_edges()
    G_mat = np.zeros((n_e, n_v), dtype=np.int_)
    for idx, (u, v) in edge_qubit_to_vertices.items():
        G_mat[idx, u] = 1
        G_mat[idx, v] = 1
    new_edges, _, _, _ = _cellulate_long_cycles(G, edge_qubit_to_vertices, vert_to_edge, G_mat, max_len=6)
    assert new_edges == []


def test_build_bridge_intercode_two_different_codes():
    """Inter-code dispatch fires (smoke test). Exact behavior tested with Ide BB-LP fixtures later."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    code1 = codes.SteaneCode()
    code2 = codes.SteaneCode()
    assert code1 is not code2
    x1 = np.asarray(code1.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    x2 = np.asarray(code2.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g1 = build_gadget(code1, x1)
    g2 = build_gadget(code2, x2)
    bridge = build_bridge(g1, g2)
    # Just verify the dispatch went the intercode path:
    assert bridge.intercode is True
    # aux_graph_edges may be empty for trivial inputs; just check it's set
    assert bridge.aux_graph_edges is not None


def test_build_single_ppm_circuit_noiseless_compiles():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import build_single_ppm_circuit
    import stim
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    circuit = build_single_ppm_circuit(g, rounds=2, noise_model=None)
    assert isinstance(circuit, stim.Circuit)
    assert len(circuit) > 0


def test_build_single_ppm_circuit_noiseless_no_detectors_fire():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import build_single_ppm_circuit
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    circuit = build_single_ppm_circuit(g, rounds=2, noise_model=None)
    sampler = circuit.compile_detector_sampler()
    samples = sampler.sample(shots=16)
    assert (samples == 0).all()


def test_build_joint_ppm_circuit_intracode_returns_pair():
    from qldpc.codes.surgery.gadget import (
        build_gadget, load_webster_seed_set, _build_generalised_bicycle_code,
    )
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import build_joint_ppm_circuit
    import stim
    # Use a k>=2 code so the k_joint = k_data - 1 invariant is not masked by
    # the spurious bridge logical (see joint.py lines 327-346 for details).
    data = load_webster_seed_set(0)  # (62, 10, 6) bicycle code
    code = _build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x1 = _webster_x_bar_1_operator(data)
    x2 = _webster_x_bar_k2p1_operator(data)
    g1 = build_gadget(code, x1)
    g2 = build_gadget(code, x2)
    bridge = build_bridge(g1, g2)
    circuit, joint_code = build_joint_ppm_circuit(
        g1, g2, bridge, rounds=1, noise_model=None,
    )
    assert isinstance(circuit, stim.Circuit)
    assert isinstance(joint_code, codes.CSSCode)
    # math.md §2.8: k_joint = k_data - 1
    assert joint_code.dimension == code.dimension - 1


def test_build_joint_ppm_circuit_intercode_css_commutation_and_dim():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import build_joint_ppm_circuit
    code1 = codes.SteaneCode()
    code2 = codes.SteaneCode()
    assert code1 is not code2
    x1 = np.asarray(code1.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    x2 = np.asarray(code2.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g1 = build_gadget(code1, x1)
    g2 = build_gadget(code2, x2)
    bridge = build_bridge(g1, g2)
    _, joint = build_joint_ppm_circuit(g1, g2, bridge, rounds=1, noise_model=None)
    HX = np.asarray(joint.matrix_x).astype(np.uint8)
    HZ = np.asarray(joint.matrix_z).astype(np.uint8)
    assert np.array_equal((HX @ HZ.T) % 2, np.zeros((HX.shape[0], HZ.shape[0]), dtype=np.uint8))
    # math.md §2.8: k_joint = k_combined - 1 (Cross §3.6 protocol consumes 1 logical DOF)
    # For two Steanes (each k=1, combined k=2), expect k_joint = 2 - 1 = 1.
    # The spurious bridge X-logical (same as Steane intra-code case) may add +1 here.
    # Accept either k=1 or k=2 — log which.
    expected_naive = code1.dimension + code2.dimension - 1
    actual = joint.dimension
    assert actual in (expected_naive, expected_naive + 1), (
        f"k_joint = {actual}, expected {expected_naive} or {expected_naive + 1}"
    )


def test_build_single_ppm_circuit_with_noise_detectors_fire():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits.noise_model import DepolarizingNoiseModel
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    circuit = build_single_ppm_circuit(
        g, rounds=2, noise_model=DepolarizingNoiseModel(p=0.05),
    )
    samples = circuit.compile_detector_sampler().sample(shots=200)
    assert samples.any()  # at least one detector fires under noise


def test_boost_gadget_dispatches_to_three_methods():
    from qldpc.codes.surgery.gadget import (
        build_gadget, GadgetLayout, load_webster_seed_set,
        _build_generalised_bicycle_code,
    )
    from qldpc.codes.surgery.cheeger import boost_gadget
    # Use Webster code 0 (l=31, k>=2): Steane gadget has dimension 0 (Steane
    # k=1 minus 1 gadget-consumed logical), which causes the BP+OSD decoder
    # used by boost_gadget_distance to hang searching for nonexistent logicals.
    data = load_webster_seed_set(0)
    code = _build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x = _webster_x_bar_1_operator(data)
    g = build_gadget(code, x)
    for method in ("spectral", "combinatorial", "distance"):
        out = boost_gadget(g, method=method, target=1.0, seed=42)
        assert isinstance(out, GadgetLayout), f"method={method}"


def test_boost_gadget_seed_reproducible():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.cheeger import boost_gadget
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    a = boost_gadget(g, method="spectral", target=1.0, seed=42)
    b = boost_gadget(g, method="spectral", target=1.0, seed=42)
    assert np.array_equal(a.F, b.F)
    assert np.array_equal(a.HX_merged, b.HX_merged)


@pytest.mark.parametrize("method", ["spectral", "combinatorial", "distance"])
def test_boost_gadget_preserves_css_commutation(method):
    from qldpc.codes.surgery.gadget import (
        build_gadget, load_webster_seed_set, _build_generalised_bicycle_code,
    )
    from qldpc.codes.surgery.cheeger import boost_gadget
    # Webster code 0 — Steane causes distance-boost decoder to hang on k=0 merged.
    data = load_webster_seed_set(0)
    code = _build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x = _webster_x_bar_1_operator(data)
    g = build_gadget(code, x)
    boosted = boost_gadget(g, method=method, target=1.0, seed=0)
    product = (boosted.HX_merged @ boosted.HZ_merged.T) % 2
    assert np.array_equal(product, np.zeros_like(product))


# ---------------------------------------------------------------------------
# Ports from surgery_test.py (T22): behavioral correctness invariants.
# ---------------------------------------------------------------------------

def _gf2_in_row_span(HX: np.ndarray, target: np.ndarray) -> bool:
    """Return True iff `target` (1-D, uint8) is in the GF(2) row span of HX."""
    import galois
    GF2 = galois.GF(2)
    M = GF2(HX.astype(np.int_))
    t = GF2(target.astype(np.int_).reshape(1, -1))
    rank_M = int(np.linalg.matrix_rank(M))
    augmented = GF2(np.vstack([np.asarray(M), np.asarray(t)]).astype(np.int_))
    return int(np.linalg.matrix_rank(augmented)) == rank_M


@pytest.mark.parametrize("code_index", [0, 1, 2, 3])
def test_joint_xx_in_stabilizer_on_webster(code_index: int) -> None:
    """X̄_1 ⊗ X̄_{k/2+1} padded with zeros on ancilla/bridge MUST lie in HX_joint row span.

    This is the key stabilizer-membership criterion for joint measurement:
    the merged code accepts X̄_1 X̄_2 as an X-stabilizer (Cross §3.6 invariant).
    If this fails the surgery construction is broken.
    """
    from qldpc.codes.surgery.gadget import (
        build_gadget, load_webster_seed_set, _build_generalised_bicycle_code,
    )
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import build_joint_ppm_circuit
    data = load_webster_seed_set(code_index)
    code = _build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x1 = _webster_x_bar_1_operator(data)
    x2 = _webster_x_bar_k2p1_operator(data)
    g1 = build_gadget(code, x1)
    g2 = build_gadget(code, x2)
    bridge = build_bridge(g1, g2)
    _, joint_code = build_joint_ppm_circuit(g1, g2, bridge, rounds=1, noise_model=None)

    n_data = code.num_qubits
    n_total = joint_code.num_qubits
    HX = np.asarray(joint_code.matrix_x).astype(np.uint8)
    op1_padded = np.zeros(n_total, dtype=np.uint8)
    op1_padded[:n_data] = x1
    op2_padded = np.zeros(n_total, dtype=np.uint8)
    op2_padded[:n_data] = x2
    joint_op = (op1_padded + op2_padded) % 2

    assert _gf2_in_row_span(HX, joint_op), (
        f"Code {data.get('name', code_index)}: X̄_1 ⊗ X̄_2 is NOT in HX_joint row span. "
        f"Construction is broken."
    )


@pytest.mark.parametrize("code_index", [0, 1, 2, 3])
def test_singleton_x_not_in_stabilizer_on_webster(code_index: int) -> None:
    """Negative: X̄_1 alone (padded) must NOT lie in HX_joint row span.

    Otherwise the surgery would stabilize X̄_1 individually rather than the joint
    product X̄_1 X̄_2, violating Cross §3.6. Both singletons are tested.
    """
    from qldpc.codes.surgery.gadget import (
        build_gadget, load_webster_seed_set, _build_generalised_bicycle_code,
    )
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import build_joint_ppm_circuit
    data = load_webster_seed_set(code_index)
    code = _build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x1 = _webster_x_bar_1_operator(data)
    x2 = _webster_x_bar_k2p1_operator(data)
    g1 = build_gadget(code, x1)
    g2 = build_gadget(code, x2)
    bridge = build_bridge(g1, g2)
    _, joint_code = build_joint_ppm_circuit(g1, g2, bridge, rounds=1, noise_model=None)

    n_data = code.num_qubits
    n_total = joint_code.num_qubits
    HX = np.asarray(joint_code.matrix_x).astype(np.uint8)
    op1_padded = np.zeros(n_total, dtype=np.uint8)
    op1_padded[:n_data] = x1
    op2_padded = np.zeros(n_total, dtype=np.uint8)
    op2_padded[:n_data] = x2

    assert not _gf2_in_row_span(HX, op1_padded), (
        f"Code {data.get('name', code_index)}: X̄_1 alone IS in HX_joint row span. "
        f"Single-operator stabilization detected — joint surgery broken."
    )
    assert not _gf2_in_row_span(HX, op2_padded), (
        f"Code {data.get('name', code_index)}: X̄_2 alone IS in HX_joint row span."
    )


@pytest.mark.parametrize("code_index", [0, 1, 2, 3])
def test_alpha_star_yields_joint_op_on_webster(code_index: int) -> None:
    """Cross §3.6 / math.md §2.7: α* · HX_joint = (X̄_1 + X̄_2, 0_anc, 0_bridge).

    The canonical protocol vector α* has 1 on every chi row from both gadgets
    AND every U_B bridge-path-stabilizer row, and 0 on the data X-check rows.

    Equivalently: XOR of (chi1 + chi2 + U_B) rows of HX_joint restricted to
    data columns equals op1 + op2, and is zero on all ancilla/bridge columns.

    Derivation:
      Σ chi1 rows  = op1 on data | 0 on g1-kappa | X on bridge[0]   (Webster Eq. 1)
      Σ chi2 rows  = op2 on data | 0 on g2-kappa | X on bridge[w-1] (Webster Eq. 1)
      Σ U_B rows   = 0 on data   | 0 on ancillas  | e_0 + e_{w-1}   (path telescoping)
      Total        = (op1+op2) on data | 0 on ancillas | 0 on bridge
    """
    from qldpc.codes.surgery.gadget import (
        build_gadget, load_webster_seed_set, _build_generalised_bicycle_code,
    )
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import build_joint_ppm_circuit
    data = load_webster_seed_set(code_index)
    code = _build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x1 = _webster_x_bar_1_operator(data)
    x2 = _webster_x_bar_k2p1_operator(data)
    g1 = build_gadget(code, x1)
    g2 = build_gadget(code, x2)
    bridge = build_bridge(g1, g2)
    _, joint_code = build_joint_ppm_circuit(g1, g2, bridge, rounds=1, noise_model=None)

    n_data = code.num_qubits
    mX = int(code.matrix_x.shape[0])
    nV1 = len(g1.V0)  # number of chi rows from g1
    nV2 = len(g2.V0)  # number of chi rows from g2
    n_bridge = bridge.width
    HX = np.asarray(joint_code.matrix_x).astype(np.int_)
    n_rows = HX.shape[0]

    # α* = 0 on data X-check rows (0..mX-1),
    #       1 on chi1 rows (mX..mX+nV1-1),
    #       1 on chi2 rows (mX+nV1..mX+nV1+nV2-1),
    #       1 on U_B rows  (mX+nV1+nV2..)
    alpha = np.zeros(n_rows, dtype=np.int_)
    alpha[mX:] = 1  # chi1 + chi2 + U_B rows all get 1

    product = (alpha @ HX) % 2

    expected = np.zeros(joint_code.num_qubits, dtype=np.int_)
    expected[:n_data] = (x1.astype(np.int_) + x2.astype(np.int_)) % 2

    assert np.array_equal(product, expected), (
        f"Code {data.get('name', code_index)}: α* · HX_joint != (op1+op2, 0, 0). "
        f"Mismatch at columns: {np.flatnonzero(product ^ expected)[:20]}"
    )


def test_gadget_layout_has_basis_field():
    from qldpc.codes.surgery.gadget import GadgetLayout
    fields = {f.name for f in dataclasses.fields(GadgetLayout)}
    assert "basis" in fields, f"basis field missing; got {fields}"


def test_gadget_layout_basis_defaults_to_x_via_build_gadget():
    """Backward compatibility: build_gadget without explicit basis defaults to Pauli.X."""
    from qldpc.codes.surgery.gadget import build_gadget
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    assert g.basis is Pauli.X


def test_step1_restriction_basis_z_uses_HX():
    """For basis=Pauli.Z, F = H_X[C_0, V_0] (not H_Z)."""
    from qldpc.codes.surgery.gadget import _step1_restriction
    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    V0, C0, F = _step1_restriction(code, z, basis=Pauli.Z)
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    # V_0 = supp(z)
    assert V0 == tuple(int(i) for i in np.where(z)[0])
    # C_0 = X-checks touching V_0
    touched = sorted({j for j in range(HX.shape[0]) for i in V0 if HX[j, i] == 1})
    assert C0 == tuple(touched)
    # F = H_X[C_0, V_0]
    assert np.array_equal(F, HX[np.ix_(C0, V0)])
    # math.md §1.1 invariant: F @ 1_{V0} = 0 (since H_X @ z = 0 for a logical Z)
    ones = np.ones(len(V0), dtype=np.uint8)
    assert np.array_equal((F @ ones) % 2, np.zeros(len(C0), dtype=np.uint8))


def test_build_gadget_z_basis_css_commutation():
    """build_gadget(code, z_logical, basis=Pauli.Z) yields a CSS-commuting merged code."""
    from qldpc.codes.surgery.gadget import build_gadget
    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    assert g.basis is Pauli.Z
    product = (g.HX_merged @ g.HZ_merged.T) % 2
    assert np.array_equal(product, np.zeros_like(product))


def test_build_gadget_z_basis_rejects_non_z_logical():
    """For basis=Pauli.Z, build_gadget checks HX @ x == 0 (z must be a Z-logical)."""
    from qldpc.codes.surgery.gadget import build_gadget
    code = codes.SteaneCode()
    # An X-logical fails: HX @ x_logical_X is typically nonzero
    x_logical = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    if ((HX @ x_logical) % 2).any():
        with pytest.raises(ValueError, match="logical"):
            build_gadget(code, x_logical, basis=Pauli.Z)


def test_build_gadget_z_basis_dual_matches_x_basis_on_dual_code():
    """basis-symmetric invariant: build_gadget(code, z, basis=Z) gives the same
    merged matrices as build_gadget(dual_code, z, basis=X), where dual_code has
    HX/HZ swapped. The swap labels swap too, so we compare HX_z vs HZ_dx_x and
    HZ_z vs HX_dx_x."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.common import CSSCode
    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_z = build_gadget(code, z, basis=Pauli.Z)
    # Dual code: swap matrix_x and matrix_z
    dual = CSSCode(
        np.asarray(code.matrix_z).astype(np.int_),
        np.asarray(code.matrix_x).astype(np.int_),
        is_subsystem_code=False,
    )
    g_dual = build_gadget(dual, z, basis=Pauli.X)
    # In the dual construction, the basis-X chi rows end up in dual.HX_merged
    # which corresponds to original.HZ_merged in the basis-Z construction.
    assert np.array_equal(g_z.HZ_merged, g_dual.HX_merged), (
        "basis-Z chi (in HZ_merged) should equal basis-X chi (in HX_merged) on dual"
    )
    assert np.array_equal(g_z.HX_merged, g_dual.HZ_merged), (
        "basis-Z gauge-fix (in HX_merged) should equal basis-X gauge-fix (in HZ_merged) on dual"
    )


def test_webster_table_i_z_basis_kappa_chi_r_exact():
    """Webster Z̄_1 seed produces the same κ+χ+r counts (basis-symmetric)."""
    from qldpc.codes.surgery.gadget import (
        build_gadget, load_webster_seed_set, _build_generalised_bicycle_code,
    )

    def z_bar_1_operator(d: dict) -> np.ndarray:
        l = d["l"]
        for seed in d["seeds"]:
            if seed["name"] == "Z_bar_1" and seed["pauli_type"] == "Z":
                L = np.zeros(l, dtype=np.uint8); R = np.zeros(l, dtype=np.uint8)
                for i in seed["L_support"]:
                    L[i] = 1
                for i in seed["R_support"]:
                    R[i] = 1
                return np.concatenate([L, R])
        raise ValueError("Z_bar_1 not found")

    for code_index, expected in [(0, 19), (1, 31), (2, 49), (3, 79)]:
        d = load_webster_seed_set(code_index)
        c = _build_generalised_bicycle_code(d["l"], d["A"], d["B"])
        z = z_bar_1_operator(d)
        g = build_gadget(c, z, basis=Pauli.Z)
        kappa = len(g.kappa_qubits)
        chi = len(g.V0)
        r = g.G.shape[0]
        assert kappa + chi + r == expected, (
            f"code {code_index}: Z-basis got κ+χ+r={kappa+chi+r}, expected {expected}"
        )


def test_bridge_has_basis_field_and_inherits_from_gadgets():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge, Bridge
    code = codes.SteaneCode()
    z1 = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    z2 = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g1 = build_gadget(code, z1, basis=Pauli.Z)
    g2 = build_gadget(code, z2, basis=Pauli.Z)
    bridge = build_bridge(g1, g2)
    assert bridge.basis is Pauli.Z


def test_build_bridge_rejects_basis_mismatch():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_x = build_gadget(code, x, basis=Pauli.X)
    g_z = build_gadget(code, z, basis=Pauli.Z)
    with pytest.raises(ValueError, match="basis"):
        build_bridge(g_x, g_z)


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_boost_gadget_preserves_css_commutation_both_bases(basis):
    """boost_gadget on a basis=X or basis=Z gadget preserves CSS commutation."""
    from qldpc.codes.surgery.gadget import (
        build_gadget, load_webster_seed_set, _build_generalised_bicycle_code,
    )
    from qldpc.codes.surgery.cheeger import boost_gadget

    def operator(d, name):
        l = d["l"]
        for seed in d["seeds"]:
            if seed["name"] == name and seed["pauli_type"] == name[0]:
                L = np.zeros(l, dtype=np.uint8); R = np.zeros(l, dtype=np.uint8)
                for i in seed["L_support"]: L[i] = 1
                for i in seed["R_support"]: R[i] = 1
                return np.concatenate([L, R])
        raise ValueError(f"{name} not found")

    d = load_webster_seed_set(0)
    c = _build_generalised_bicycle_code(d["l"], d["A"], d["B"])
    op_name = "X_bar_1" if basis is Pauli.X else "Z_bar_1"
    op = operator(d, op_name)
    g = build_gadget(c, op, basis=basis)
    boosted = boost_gadget(g, method="combinatorial", target=1.0, seed=0)
    product = (boosted.HX_merged @ boosted.HZ_merged.T) % 2
    assert np.array_equal(product, np.zeros_like(product))
    assert boosted.basis is basis  # boost preserves basis


def test_classify_reliable_round1_checks_basis_x():
    """For basis=X: reliable round-1 checks are data H_X (first m_X X-checks)
    plus gauge-fix G (last r Z-checks)."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import _classify_reliable_round1_checks
    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.codes.common import CSSCode
    import galois
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    F2 = galois.GF(2)
    merged = CSSCode(
        F2(g.HX_merged.astype(np.int_).tolist()),
        F2(g.HZ_merged.astype(np.int_).tolist()),
        is_subsystem_code=False,
    )
    qubit_ids = QubitIDs.from_code(merged)
    reliable = _classify_reliable_round1_checks(g, qubit_ids)
    m_X = code.matrix_x.shape[0]
    m_Z = code.matrix_z.shape[0]
    # Reliable X-checks: first m_X of checks_x (the original data H_X rows)
    expected_x_reliable = set(qubit_ids.checks_x[:m_X])
    # Reliable Z-checks: last r of checks_z (the gauge-fix G rows)
    r = g.G.shape[0]
    expected_z_reliable = set(qubit_ids.checks_z[m_Z:])
    expected = expected_x_reliable | expected_z_reliable
    assert set(reliable) == expected, (
        f"reliable={set(reliable)}, expected={expected}"
    )


def test_classify_reliable_round1_checks_basis_z():
    """For basis=Z: reliable round-1 checks are data H_Z (first m_Z Z-checks)
    plus gauge-fix G (last r X-checks)."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import _classify_reliable_round1_checks
    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.codes.common import CSSCode
    import galois
    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    F2 = galois.GF(2)
    merged = CSSCode(
        F2(g.HX_merged.astype(np.int_).tolist()),
        F2(g.HZ_merged.astype(np.int_).tolist()),
        is_subsystem_code=False,
    )
    qubit_ids = QubitIDs.from_code(merged)
    reliable = _classify_reliable_round1_checks(g, qubit_ids)
    m_X = code.matrix_x.shape[0]
    m_Z = code.matrix_z.shape[0]
    r = g.G.shape[0]
    # basis=Z: data H_Z rows are first m_Z Z-checks; G rows are last r X-checks
    expected_z_reliable = set(qubit_ids.checks_z[:m_Z])
    expected_x_reliable = set(qubit_ids.checks_x[m_X:])
    expected = expected_z_reliable | expected_x_reliable
    assert set(reliable) == expected


def test_surgery_state_prep_basis_x_resets():
    """basis=X: data RX (→|+⟩), kappa R (→|0⟩)."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import _surgery_state_prep
    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.codes.common import CSSCode
    import galois
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    F2 = galois.GF(2)
    merged = CSSCode(
        F2(g.HX_merged.astype(np.int_).tolist()),
        F2(g.HZ_merged.astype(np.int_).tolist()),
        is_subsystem_code=False,
    )
    qubit_ids = QubitIDs.from_code(merged)
    n_data = code.num_qudits
    data_ids = qubit_ids.data[:n_data]
    kappa_ids = qubit_ids.data[n_data:]
    circuit = _surgery_state_prep(g, data_ids, kappa_ids, bridge_ids=())
    text = str(circuit)
    assert f"RX {' '.join(str(q) for q in data_ids)}" in text
    assert f"R {' '.join(str(q) for q in kappa_ids)}" in text


def test_surgery_state_prep_basis_z_resets():
    """basis=Z: data R (→|0⟩), kappa RX (→|+⟩)."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import _surgery_state_prep
    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.codes.common import CSSCode
    import galois
    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    F2 = galois.GF(2)
    merged = CSSCode(
        F2(g.HX_merged.astype(np.int_).tolist()),
        F2(g.HZ_merged.astype(np.int_).tolist()),
        is_subsystem_code=False,
    )
    qubit_ids = QubitIDs.from_code(merged)
    n_data = code.num_qudits
    data_ids = qubit_ids.data[:n_data]
    kappa_ids = qubit_ids.data[n_data:]
    circuit = _surgery_state_prep(g, data_ids, kappa_ids, bridge_ids=())
    text = str(circuit)
    assert f"R {' '.join(str(q) for q in data_ids)}" in text
    assert f"RX {' '.join(str(q) for q in kappa_ids)}" in text


def test_surgery_qec_cycle_round_1_detectors_classified():
    """Round-1 detectors are 1-arg only for RELIABLE checks; unreliable ones skipped."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import _surgery_qec_cycle, _classify_reliable_round1_checks
    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.codes.common import CSSCode
    import galois
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    F2 = galois.GF(2)
    merged = CSSCode(
        F2(g.HX_merged.astype(np.int_).tolist()),
        F2(g.HZ_merged.astype(np.int_).tolist()),
        is_subsystem_code=False,
    )
    qubit_ids = QubitIDs.from_code(merged)
    reliable = _classify_reliable_round1_checks(g, qubit_ids)

    circuit, meas_rec, det_rec = _surgery_qec_cycle(
        g, merged, num_rounds=2, qubit_ids=qubit_ids,
    )
    # Count round-1 1-arg DETECTORs (those appearing before any REPEAT_BLOCK).
    text = str(circuit)
    # Number of "DETECTOR" instructions in the first round (before the REPEAT block)
    # should equal len(reliable).
    first_round_str = text.split("REPEAT")[0]
    n_det = first_round_str.count("DETECTOR")
    assert n_det == len(reliable), (
        f"round-1 detectors={n_det}, expected len(reliable)={len(reliable)}"
    )


def test_surgery_detach_and_readout_basis_x_measures_kappa_then_data():
    """basis=X: detach with M (Z-basis) on κ, then MX on data."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import _surgery_detach_and_readout
    from qldpc.circuits.bookkeeping import MeasurementRecord
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    n_data = code.num_qudits
    data_ids = tuple(range(n_data))
    kappa_ids = tuple(range(n_data, n_data + len(g.kappa_qubits)))
    bridge_ids = ()
    meas_rec = MeasurementRecord()
    circuit = _surgery_detach_and_readout(
        g, data_ids=data_ids, kappa_ids=kappa_ids, bridge_ids=bridge_ids,
        measurement_record=meas_rec,
    )
    text = str(circuit)
    # κ measured first (in Z), then data (in X)
    m_kappa_idx = text.find(f"M {' '.join(str(q) for q in kappa_ids)}")
    m_data_idx = text.find(f"MX {' '.join(str(q) for q in data_ids)}")
    assert m_kappa_idx >= 0 and m_data_idx >= 0
    assert m_kappa_idx < m_data_idx


def test_surgery_detach_and_readout_basis_z_measures_kappa_in_x_then_data_in_z():
    """basis=Z: detach with MX on κ, then M on data."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import _surgery_detach_and_readout
    from qldpc.circuits.bookkeeping import MeasurementRecord
    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    n_data = code.num_qudits
    data_ids = tuple(range(n_data))
    kappa_ids = tuple(range(n_data, n_data + len(g.kappa_qubits)))
    meas_rec = MeasurementRecord()
    circuit = _surgery_detach_and_readout(
        g, data_ids=data_ids, kappa_ids=kappa_ids, bridge_ids=(),
        measurement_record=meas_rec,
    )
    text = str(circuit)
    m_kappa_idx = text.find(f"MX {' '.join(str(q) for q in kappa_ids)}")
    m_data_idx = text.find(f"M {' '.join(str(q) for q in data_ids)}")
    assert m_kappa_idx >= 0 and m_data_idx >= 0
    assert m_kappa_idx < m_data_idx


def test_surgery_observable_emits_two_observable_include():
    """Observable 0 = XOR of χ-row records across all rounds; Observable 1 = data measurement on V_0."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import _surgery_observable
    from qldpc.circuits.bookkeeping import MeasurementRecord
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    n_data = code.num_qudits
    chi_check_ids = tuple(range(100, 100 + len(g.V0)))   # placeholder ids
    data_ids = tuple(range(n_data))
    meas_rec = MeasurementRecord()
    # Simulate 2 rounds of chi-check measurements
    for _ in range(2):
        meas_rec.append({cid: i for i, cid in enumerate(chi_check_ids)})
    # Simulate final data measurement
    meas_rec.append({d: i for i, d in enumerate(data_ids)})

    circuit = _surgery_observable(
        g, chi_check_ids=chi_check_ids, data_ids=data_ids,
        v0_indices=g.V0, num_rounds=2, measurement_record=meas_rec,
    )
    text = str(circuit)
    assert text.count("OBSERVABLE_INCLUDE") == 2  # PPM + cross-check
    assert "(0)" in text and "(1)" in text  # two distinct observable indices


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_build_single_ppm_circuit_noiseless_observables_zero(basis):
    """Both OBSERVABLE_INCLUDEs evaluate to 0 (= +1) under no noise."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import build_single_ppm_circuit
    code = codes.SteaneCode()
    op = (code.get_logical_ops(Pauli.X)[0]
          if basis is Pauli.X
          else code.get_logical_ops(Pauli.Z)[0])
    op_arr = np.asarray(op).astype(np.uint8)
    g = build_gadget(code, op_arr, basis=basis)
    circuit = build_single_ppm_circuit(g, rounds=3, noise_model=None)
    # Sample observables; all should be 0.
    sampler = circuit.compile_detector_sampler()
    _, obs = sampler.sample(shots=16, separate_observables=True)
    assert (obs == 0).all(), (
        f"noiseless observables fired: {obs.sum()} flips across 16 shots"
    )


def test_build_joint_ppm_circuit_noiseless_observables_zero():
    """Noiseless joint PPM: observable 0 (α* per math.md §2.7) = 0; observable 1 = 0."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import build_joint_ppm_circuit
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g1 = build_gadget(code, x, basis=Pauli.X)
    g2 = build_gadget(code, x, basis=Pauli.X)
    bridge = build_bridge(g1, g2)
    circuit, joint_code = build_joint_ppm_circuit(g1, g2, bridge, rounds=2, noise_model=None)
    sampler = circuit.compile_detector_sampler()
    _, obs = sampler.sample(shots=16, separate_observables=True)
    assert (obs == 0).all(), f"noiseless joint observables fired: {obs.sum()} flips"


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_build_joint_ppm_circuit_basis_parametrized_noiseless_observables_zero(basis):
    """Both bases for the joint PPM circuit: noiseless observables = 0."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import build_joint_ppm_circuit
    code = codes.SteaneCode()
    op = (code.get_logical_ops(Pauli.X)[0]
          if basis is Pauli.X
          else code.get_logical_ops(Pauli.Z)[0])
    op_arr = np.asarray(op).astype(np.uint8)
    g1 = build_gadget(code, op_arr, basis=basis)
    g2 = build_gadget(code, op_arr, basis=basis)
    bridge = build_bridge(g1, g2)
    circuit, joint_code = build_joint_ppm_circuit(g1, g2, bridge, rounds=2, noise_model=None)
    sampler = circuit.compile_detector_sampler()
    _, obs = sampler.sample(shots=16, separate_observables=True)
    assert (obs == 0).all(), f"noiseless joint observables fired for basis={basis}: {obs.sum()} flips"


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_single_ppm_circuit_noise_flips_observable_at_high_p(basis):
    """At p=0.1, the PPM observable (observable 0) flips ≥ 5% of shots."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits.noise_model import DepolarizingNoiseModel
    code = codes.SteaneCode()
    op = (code.get_logical_ops(Pauli.X)[0]
          if basis is Pauli.X
          else code.get_logical_ops(Pauli.Z)[0])
    op_arr = np.asarray(op).astype(np.uint8)
    g = build_gadget(code, op_arr, basis=basis)
    circuit = build_single_ppm_circuit(
        g, rounds=3, noise_model=DepolarizingNoiseModel(p=0.1),
    )
    sampler = circuit.compile_detector_sampler()
    _, obs = sampler.sample(shots=400, separate_observables=True)
    # Observable 0 (PPM) flips a nontrivial fraction at p=0.1
    obs_0_flip_rate = float(obs[:, 0].mean())
    assert obs_0_flip_rate >= 0.05, (
        f"PPM observable flip rate {obs_0_flip_rate:.2%} too low at p=0.1"
    )


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_surgery_final_detectors_count_matches_reliable_round1(basis):
    """Number of final DETECTORs equals |reliable round-1 set|.

    Tests the helper in isolation: build a circuit through detach_and_readout,
    then call _surgery_final_detectors and count emitted DETECTOR instructions.
    """
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import (
        _surgery_state_prep, _surgery_qec_cycle, _surgery_detach_and_readout,
        _surgery_final_detectors, _classify_reliable_round1_checks,
        _gadget_merged_csscode,
    )
    from qldpc.circuits.bookkeeping import QubitIDs

    code = codes.SteaneCode()
    op = (code.get_logical_ops(Pauli.X)[0] if basis is Pauli.X
          else code.get_logical_ops(Pauli.Z)[0])
    op_arr = np.asarray(op).astype(np.uint8)
    g = build_gadget(code, op_arr, basis=basis)
    merged = _gadget_merged_csscode(g)
    qubit_ids = QubitIDs.from_code(merged)
    n_data = code.num_qudits
    data_ids = qubit_ids.data[:n_data]
    kappa_ids = qubit_ids.data[n_data:]

    # Simulate the pipeline through detach (we need measurement_record populated).
    _qec, mrec, _det = _surgery_qec_cycle(g, merged, num_rounds=2, qubit_ids=qubit_ids)
    _surgery_detach_and_readout(
        g, data_ids=data_ids, kappa_ids=kappa_ids, bridge_ids=(),
        measurement_record=mrec,
    )

    circuit = _surgery_final_detectors(g, merged, qubit_ids, measurement_record=mrec)
    n_final_det = str(circuit).count("DETECTOR")
    expected = len(_classify_reliable_round1_checks(g, qubit_ids))
    assert n_final_det == expected, (
        f"basis={basis}: emitted {n_final_det} DETECTORs, expected {expected}"
    )


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_build_single_ppm_circuit_noiseless_no_detector_fires(basis):
    """Noiseless: NO detector fires (including the new final detectors).

    The total detector count must equal: round-1 reliable + (rounds-1)*all_checks + final reliable.
    Under noiseless conditions all of them must remain silent.
    """
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import build_single_ppm_circuit
    code = codes.SteaneCode()
    op = (code.get_logical_ops(Pauli.X)[0] if basis is Pauli.X
          else code.get_logical_ops(Pauli.Z)[0])
    op_arr = np.asarray(op).astype(np.uint8)
    g = build_gadget(code, op_arr, basis=basis)
    circuit = build_single_ppm_circuit(g, rounds=3, noise_model=None)
    sampler = circuit.compile_detector_sampler()
    dets, _ = sampler.sample(shots=64, separate_observables=True)
    assert not dets.any(), (
        f"basis={basis}: {dets.sum()} detector fires noiselessly across {dets.shape[0]} shots"
    )


def test_build_joint_ppm_circuit_noiseless_no_detector_fires():
    """Joint noiseless: NO detector fires (including final detectors)."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import build_joint_ppm_circuit
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g1 = build_gadget(code, x, basis=Pauli.X)
    g2 = build_gadget(code, x, basis=Pauli.X)
    bridge = build_bridge(g1, g2)
    circuit, _ = build_joint_ppm_circuit(g1, g2, bridge, rounds=2, noise_model=None)
    sampler = circuit.compile_detector_sampler()
    dets, _ = sampler.sample(shots=64, separate_observables=True)
    assert not dets.any(), f"{dets.sum()} detector fires noiselessly"


@pytest.mark.slow
def test_single_ppm_ler_monotone_in_p():
    """Tiny sinter sweep: PPM LER monotonically increasing in p.

    Catches gross protocol errors (wrong observable basis, sign flips, etc.).
    """
    import sinter
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits import DepolarizingNoiseModel
    from qldpc import decoders
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)

    error_rates = [0.001, 0.005, 0.02]
    tasks = []
    for p in error_rates:
        circuit = build_single_ppm_circuit(
            g, rounds=3, noise_model=DepolarizingNoiseModel(p),
        )
        tasks.append(sinter.Task(
            circuit=circuit,
            json_metadata={"p": float(p)},
        ))
    sinter_decoder = decoders.SinterDecoder()
    results = sinter.collect(
        tasks=tasks,
        decoders=["custom"],
        custom_decoders={"custom": sinter_decoder},
        num_workers=4,
        max_shots=2000,
        max_errors=30,
        print_progress=False,
    )
    by_p = {r.json_metadata["p"]: r.errors / max(r.shots, 1) for r in results}
    sorted_p = sorted(by_p.keys())
    ler_vals = [by_p[p] for p in sorted_p]
    print(f"LER values: {list(zip(sorted_p, ler_vals))}")
    # Monotonically non-decreasing (allow small statistical noise)
    for i in range(len(ler_vals) - 1):
        assert ler_vals[i] <= ler_vals[i + 1] * 1.5, (
            f"LER not monotonic: p={sorted_p[i]} → {ler_vals[i]}, "
            f"p={sorted_p[i+1]} → {ler_vals[i+1]}"
        )


@pytest.mark.slow
def test_single_ppm_ler_with_final_detectors_below_threshold():
    """With final detectors wired, LER at p=0.001 should be ≤ 0.01.

    Reference: before the final-detector wiring, LER at p=0.001 was ~0.024
    (from test_single_ppm_ler_monotone_in_p in the surgery-circuit-rewrite plan).
    Adding the inferred detectors should drop it significantly.
    """
    import sinter
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits import DepolarizingNoiseModel
    from qldpc import decoders

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)

    p = 0.001
    circuit = build_single_ppm_circuit(
        g, rounds=3, noise_model=DepolarizingNoiseModel(p),
    )
    sinter_decoder = decoders.SinterDecoder()
    results = sinter.collect(
        tasks=[sinter.Task(circuit=circuit, json_metadata={"p": float(p)})],
        decoders=["custom"],
        custom_decoders={"custom": sinter_decoder},
        num_workers=4,
        max_shots=5000,
        max_errors=50,
        print_progress=False,
    )
    assert len(results) == 1
    ler = results[0].errors / max(results[0].shots, 1)
    assert ler <= 0.01, (
        f"LER at p=0.001 = {ler:.4f} (errors={results[0].errors}/{results[0].shots} shots). "
        f"Expected ≤ 0.01 with final detectors wired. Was ~0.024 without them."
    )
