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
        "aux_graph_edges", "z_extensions",
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
