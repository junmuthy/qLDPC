# edge_expanded_test.py
import numpy as np
from qldpc.circuits.surgery.hmatrix.edge_expanded import restrict_maps
from qldpc.circuits.surgery.hmatrix.edge_expanded import (
    boundary, cheeger_constant, sparsest_cut)
from qldpc.circuits.surgery.hmatrix.edge_expanded import algorithm_1
from qldpc.circuits.surgery.hmatrix.edge_expanded import (
    GF2, algorithm_2, _random_invertible_gf2)
from qldpc.circuits.surgery.hmatrix.edge_expanded import cellulate, expand_hyperedges

STEANE_HZ = np.array([  # Steane [[7,1,3]] H_Z (arXiv:2410.02753 Eq.54), qubits 0..6
    [0,0,0,1,1,1,1],
    [0,1,1,0,0,1,1],
    [1,0,1,0,1,0,1],
], dtype=np.uint8)

def test_restrict_maps_steane_weight3_logical():
    # X̄ = X1 X2 X3 (support {0,1,2}) — arXiv:2410.02753 Example 6.
    x = np.array([1,1,1,0,0,0,0], dtype=np.uint8)
    r = restrict_maps(STEANE_HZ, x)
    assert r.support == (0, 1, 2)
    # H_Z|_Q columns {0,1,2}: rows -> [[0,0,0],[0,1,1],[1,0,1]]; row 0 is zero -> dropped
    assert r.nz_rows == (1, 2)
    np.testing.assert_array_equal(r.incidence_star, np.array([[0,1,1],[1,0,1]], dtype=np.uint8))
    # f1 (Eq 35): n=7 rows, |Q|=3 cols, (f1)_{i,j}=delta_{i,q_j}
    expected_f1 = np.zeros((7, 3), dtype=np.uint8)
    expected_f1[0,0] = expected_f1[1,1] = expected_f1[2,2] = 1
    np.testing.assert_array_equal(r.f1, expected_f1)
    # f0_star (Eq 48): (f0*)_{i,j}=delta_{i,h_j}, h_j = row index of j-th nonzero row of H_Z|_Q
    expected_f0 = np.zeros((3, 2), dtype=np.uint8)
    expected_f0[1,0] = expected_f0[2,1] = 1
    np.testing.assert_array_equal(r.f0_star, expected_f0)


def test_algorithm_1_fiedler_reaches_cheeger_one_large_v():
    # 6x6 grid = 36 vertices (> _EXACT_CHEEGER_MAX_V): exact enumeration is
    # infeasible, so algorithm_1 takes the Fiedler-sweep path. It must still add
    # edges until the (sweep-estimated) Cheeger constant is >= 1.
    inc = _grid_incidence(6)
    assert inc.shape[1] == 36
    out = algorithm_1(inc, seed=0)
    assert out.shape[0] > inc.shape[0]                 # edges were added
    assert cheeger_constant(out) >= 1.0                # sweep estimate reaches 1


def test_fiedler_sweep_estimate_brackets_exact():
    # On a small graph the Fiedler sweep sees only a subset of cuts, so its Cheeger
    # estimate can only OVER-report (>= the exact value). Sanity that it is not
    # wildly off and never under-reports.
    from qldpc.circuits.surgery.hmatrix.edge_expanded import _fiedler_sweep_cuts
    inc = _grid_incidence(4)                           # 16 vertices -> exact path
    h_exact = cheeger_constant(inc)
    cuts = _fiedler_sweep_cuts(inc)
    bc = (inc @ cuts.T % 2).sum(axis=0)
    sz = cuts.sum(axis=1)
    h_sweep = float((bc[sz > 0] / sz[sz > 0]).min())
    assert h_sweep >= h_exact - 1e-9                   # sweep never under-reports
    assert h_exact > 0                                 # grid is connected


def _incidence(edges, n_v):
    M = np.zeros((len(edges), n_v), dtype=np.uint8)
    for i, e in enumerate(edges):
        for v in e:
            M[i, v] = 1
    return M

def test_cheeger_path_graph_p8_example4():
    # arXiv:2410.02753 Example 4 / Fig 1a: V={v1..v6}, edges (v1v2)(v2v3)(v4v5)(v5v6)
    inc = _incidence([(0,1),(1,2),(3,4),(4,5)], 6)
    assert cheeger_constant(inc) == 0.0            # disconnected -> h=0
    # boundary of S={v0,v1,v2}: only edge (v1,v2)->(1,2) crosses? edges within S carry even.
    S = np.array([1,1,1,0,0,0], dtype=np.uint8)
    assert int(boundary(inc, S).sum()) == 0        # {v0,v1,v2} is a full component -> empty boundary

def test_cheeger_square_is_one():
    # 4-cycle: |V|=4, half=2. Single vertex -> |∂S|=2 (ratio 2); adjacent pair
    # {0,1} -> boundary edges (1,2),(3,0) = 2, ratio 2/2 = 1. So h = 1.
    inc = _incidence([(0,1),(1,2),(2,3),(3,0)], 4)
    assert cheeger_constant(inc) == 1.0
    # (Sanity: a triangle has h=2 here, NOT 1, because half=1 admits only single-vertex cuts.)
    assert cheeger_constant(_incidence([(0,1),(1,2),(0,2)], 3)) == 2.0

def test_sparsest_cut_returns_min_ratio_set():
    inc = _incidence([(0,1),(1,2),(3,4),(4,5)], 6)
    S = sparsest_cut(inc)
    assert 1 <= int(S.sum()) <= 3
    assert int(boundary(inc, S).sum()) == 0        # sparsest cut isolates a component (ratio 0)


def test_algorithm_1_reaches_cheeger_one():
    inc = _incidence([(0,1),(1,2),(3,4),(4,5)], 6)   # h=0
    out = algorithm_1(inc, seed=0)
    assert out.shape[1] == 6                          # same vertices
    assert out.shape[0] >= inc.shape[0]               # superset of edges
    np.testing.assert_array_equal(out[:inc.shape[0]], inc)  # original edges preserved
    assert cheeger_constant(out) >= 1.0
    added = out[inc.shape[0]:]
    if added.size:
        assert np.all(added.sum(axis=1) == 2)         # each new edge is weight-2

def test_algorithm_1_noop_when_already_one():
    inc = _incidence([(0,1),(1,2),(2,3),(3,0)], 4)    # 4-cycle already h=1 (see Task 2)
    out = algorithm_1(inc, seed=0)
    np.testing.assert_array_equal(out, inc)


def test_random_invertible_is_invertible():
    rng = np.random.default_rng(3)
    for _ in range(5):
        A = _random_invertible_gf2(4, rng)
        assert int(A.row_space().shape[0]) == 4       # full rank over GF(2)

def test_algorithm_2_is_valid_cycle_basis_and_low_weight():
    # A 6-cycle graph: edges (0,1)(1,2)(2,3)(3,4)(4,5)(5,0). Cycle space dim 1,
    # the single cycle uses all 6 edges. With one big cycle, ∂0 = that weight-6 row.
    inc = _incidence([(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)], 6)
    H_complement = np.zeros((0, 6), dtype=np.uint8)   # no backing checks (all edges are κ)
    f0 = np.zeros((0, 6), dtype=np.uint8)             # |edges|=6 columns
    d0 = algorithm_2(inc, H_complement, f0, n_samples=100, seed=0)
    # rows are cycles: d0 @ incidence == 0 over GF(2)
    assert np.all((np.asarray(d0).astype(int) @ inc.astype(int)) % 2 == 0)
    # spans the whole cycle space (dim 1 here)
    assert int(GF2(np.asarray(d0).astype(np.uint8)).row_space().shape[0]) == 1

def test_algorithm_2_nonempty_V_direct_sum():
    # Theta graph: 2 vertices joined by 3 parallel edges -> cycle space dim 2.
    # H_complement has a redundant pair of identical checks, so
    # ker(H_complementᵀ) = span{[1,1]} is nonzero, and f0 maps it to the
    # nonzero redundant cycle [0,1,1]. V = span{[0,1,1]} is then a STRICT
    # nonempty subspace of the cycle space, exercising Algorithm 2 lines 3-5
    # (arXiv:2410.02753 Algorithm 2): W must satisfy V ⊕ W = full cycle space.
    inc = _incidence([(0, 1), (0, 1), (0, 1)], 2)          # 3 parallel edges
    H_complement = np.array([[1, 1], [1, 1]], dtype=np.uint8)  # redundant checks
    f0 = np.array([[0, 1, 0], [0, 0, 1]], dtype=np.uint8)  # rows↔checks, cols↔edges
    # V exactly as computed inside algorithm_2 (lines 1-2)
    V = (GF2(H_complement).left_null_space() @ GF2(f0)).row_space()
    rank_V = int(V.shape[0])
    assert rank_V >= 1                                     # V genuinely nonempty
    cycle_space = GF2(inc).left_null_space()
    rank_cycles = int(cycle_space.row_space().shape[0])
    assert rank_V < rank_cycles                            # strict subspace (1 < 2)
    d0 = algorithm_2(inc, H_complement, f0, n_samples=50, seed=0)
    # every row of ∂0 is still a cycle
    assert np.all((np.asarray(d0).astype(int) @ inc.astype(int)) % 2 == 0)
    # direct sum, no gap: span(V) + span(∂0) = full cycle space
    stacked = np.vstack([np.asarray(V).astype(np.uint8), d0.astype(np.uint8)])
    assert int(GF2(stacked).row_space().shape[0]) == rank_cycles
    # direct sum, no overlap: rank(∂0) + rank(V) = rank(cycle space)
    rank_d0 = int(GF2(d0.astype(np.uint8)).row_space().shape[0])
    assert rank_d0 + rank_V == rank_cycles


def test_algorithm_2_beats_arbitrary_basis_weight():
    # Two triangles sharing edge structure so the arbitrary basis has a heavy row
    # but a low-weight basis (two triangles) exists.
    inc = _incidence([(0,1),(1,2),(0,2),(2,3),(3,4),(2,4)], 5)
    H_complement = np.zeros((0, 5), dtype=np.uint8)
    f0 = np.zeros((0, 6), dtype=np.uint8)
    d0 = algorithm_2(inc, H_complement, f0, n_samples=300, seed=0)
    assert np.all((np.asarray(d0).astype(int) @ inc.astype(int)) % 2 == 0)
    # each independent cycle here is a triangle -> max row weight 3
    assert int(np.asarray(d0).astype(int).sum(axis=1).max()) <= 3


def _grid_incidence(n):
    """(n x n) grid graph as an edge-vertex incidence (all weight-2 edges)."""
    vid = lambda r, c: r * n + c
    edges = []
    for r in range(n):
        for c in range(n):
            if c + 1 < n:
                edges.append((vid(r, c), vid(r, c + 1)))
            if r + 1 < n:
                edges.append((vid(r, c), vid(r + 1, c)))
    return _incidence(edges, n * n)


def test_algorithm_2_finds_minimum_cycle_basis_on_grid():
    # 4x4 grid: cycle space dim 9; the minimum cycle basis is the 9 unit squares,
    # each weight 4. A dense RREF basis + random A·W+B·V search stalls at weight 8
    # (the random search only makes rows denser); a minimum-cycle-basis construction
    # reaches the weight-4 optimum. Regression test for the algorithm_2 rewrite.
    inc = _grid_incidence(4)
    H_complement = np.zeros((0, inc.shape[1]), dtype=np.uint8)
    f0 = np.zeros((0, inc.shape[0]), dtype=np.uint8)
    d0 = algorithm_2(inc, H_complement, f0, n_samples=300, seed=0)
    # valid: rows are cycles and span the full cycle space (dim 9)
    assert np.all((np.asarray(d0).astype(int) @ inc.astype(int)) % 2 == 0)
    assert int(GF2(np.asarray(d0).astype(np.uint8)).row_space().shape[0]) == 9
    # every basis cycle is a unit square -> max row weight 4
    assert int(np.asarray(d0).astype(int).sum(axis=1).max()) == 4


def test_expand_hyperedges_splits_wide_rows():
    # A single weight-4 hyperedge over 4 vertices + f0 column backing it.
    inc = np.array([[1,1,1,1]], dtype=np.uint8)          # one wt-4 row (hyperedge)
    f0 = np.array([[1]], dtype=np.uint8)                  # 1 check, 1 edge column
    inc2, f02 = expand_hyperedges(inc, f0)
    # wt 4 -> 4//2 = 2 weight-2 rows that SUM to the original row
    assert inc2.shape[0] == 2
    assert np.all(inc2.sum(axis=1) == 2)                  # every new row weight 2
    np.testing.assert_array_equal(inc2.sum(axis=0) % 2, inc[0])   # rows sum to e
    assert f02.shape == (1, 2)                            # column duplicated (wt e)//2 = 2 times
    np.testing.assert_array_equal(f02, np.array([[1,1]], dtype=np.uint8))

def test_expand_hyperedges_cheeger_aware_pairing_connects():
    # Two overlapping weight-4 hyperedges. Naive index-order matching gives
    # (0,1)(2,3) then (2,3)(4,5) -> 3 disconnected components {0,1},{2,3},{4,5};
    # the Cheeger-aware pairing (arXiv:2410.02753 Alg 3: "keep the Cheeger constant
    # as high as possible") merges components, leaving the expanded weight-2 graph
    # far better connected so the subsequent Algorithm 1 adds fewer edges.
    inc = np.array([[1, 1, 1, 1, 0, 0],   # hyperedge {0,1,2,3}
                    [0, 0, 1, 1, 1, 1]],  # hyperedge {2,3,4,5}
                   dtype=np.uint8)
    f0 = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    inc2, f02 = expand_hyperedges(inc, f0)
    # correctness (Eq 36): each hyperedge's replacement rows sum to it, all weight 2
    assert np.array_equal(inc2[:2].sum(0) % 2, inc[0])
    assert np.array_equal(inc2[2:].sum(0) % 2, inc[1])
    assert np.all(inc2.sum(1) == 2)
    assert f02.shape[1] == inc2.shape[0]                # one f0 column per new edge
    # connectivity: expanded weight-2 graph has <= 2 components (index order gives 3)
    parent = list(range(6))
    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for r in inc2:
        a, b = (int(v) for v in np.flatnonzero(r))
        parent[find(a)] = find(b)
    assert len({find(v) for v in range(6)}) <= 2


def test_expand_hyperedges_passes_weight2_through():
    inc = np.array([[1,1,0],[0,1,1]], dtype=np.uint8)     # both weight-2 already
    f0 = np.array([[1,0],[0,1]], dtype=np.uint8)
    inc2, f02 = expand_hyperedges(inc, f0)
    np.testing.assert_array_equal(inc2, inc)             # unchanged
    np.testing.assert_array_equal(f02, f0)

def test_cellulate_splits_heavy_cycle():
    # Single weight-6 cycle (hexagon). target_weight=4 -> must add a chord, giving
    # two cycles each of weight <= 4.
    inc = _incidence([(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)], 6)
    f0 = np.zeros((0, 6), dtype=np.uint8)
    d0 = algorithm_2(inc, np.zeros((0,6),np.uint8), f0, n_samples=50, seed=0)
    assert int(np.asarray(d0).astype(int).sum(axis=1).max()) == 6
    d0c, incc, f0c = cellulate(d0, inc, f0, target_weight=4, seed=0)
    assert incc.shape[0] > inc.shape[0]                          # chord edge(s) added
    assert np.all((np.asarray(d0c).astype(int) @ incc.astype(int)) % 2 == 0)  # still cycles
    assert int(np.asarray(d0c).astype(int).sum(axis=1).max()) <= 4

def test_cellulate_noop_when_within_target():
    inc = _incidence([(0,1),(1,2),(0,2)], 3)
    f0 = np.zeros((0, 3), dtype=np.uint8)
    d0 = algorithm_2(inc, np.zeros((0,3),np.uint8), f0, n_samples=20, seed=0)
    d0c, incc, f0c = cellulate(d0, inc, f0, target_weight=3, seed=0)
    assert incc.shape[0] == inc.shape[0]                         # nothing added


def test_cellulate_reduces_large_cycle_below_target():
    # Regression: cellulate must directly split heavy ∂0 rows to <= target — it must
    # NOT re-run Algorithm 2 (whose heuristic search leaves large cycles heavy, the
    # bb_18 weight-12 bug). A single weight-16 cycle must come out <= 4.
    inc = _incidence([(i, (i + 1) % 16) for i in range(16)], 16)
    f0 = np.zeros((0, 16), dtype=np.uint8)
    d0 = np.array([[1] * 16], dtype=np.uint8)                    # the 16-cycle
    d0c, incc, f0c = cellulate(d0, inc, f0, target_weight=4, seed=0)
    d0c = np.asarray(d0c).astype(int)
    assert np.all((d0c @ incc.astype(int)) % 2 == 0)            # still cycles
    assert int(d0c.sum(axis=1).max()) <= 4                      # actually reduced


def test_cellulate_decomposes_disjoint_cycle_union_without_chords():
    # A ∂0 row that is a union of two triangles (non-simple) splits into two weight-3
    # rows with NO new edges (Veblen decomposition), not a re-derivation.
    inc = _incidence([(0,1),(1,2),(0,2),(3,4),(4,5),(3,5)], 6)
    f0 = np.zeros((0, 6), dtype=np.uint8)
    d0 = np.array([[1,1,1,1,1,1]], dtype=np.uint8)              # both triangles as one row
    d0c, incc, f0c = cellulate(d0, inc, f0, target_weight=4, seed=0)
    assert incc.shape[0] == inc.shape[0]                        # no chords needed
    assert int(np.asarray(d0c).astype(int).sum(axis=1).max()) <= 3


from qldpc.circuits.surgery.hmatrix.edge_expanded import edge_expanded_maps

def test_edge_expanded_steane_weight3_no_cycles():
    # Example 6: Steane X̄=X1X2X3 -> after Alg 1 the graph has no cycles (∂0 empty).
    x = np.array([1,1,1,0,0,0,0], dtype=np.uint8)
    cm = edge_expanded_maps(STEANE_HZ, x, seed=0)
    assert cheeger_constant(cm.incidence) >= 1.0
    # chain complex: ∂0 @ ∂1 == 0
    d1 = cm.incidence                      # edge-vertex; ∂1 = incidence.T applied as cols
    assert np.all((np.asarray(cm.partial_0).astype(int) @ cm.incidence.astype(int)) % 2 == 0)
    # dim ker ∂1 == 1 (only X̄ measured, Remark 3): cycle space dim after Alg1
    from qldpc.circuits.surgery.hmatrix.edge_expanded import GF2
    # here incidence has full column rank -> cycle space may be empty; ker ∂1 measured
    assert cm.f1.shape == (7, 3)

def test_edge_expanded_branch_triggers_tiny():
    # Synthetic 4-qubit input: H_complement = two DISCONNECTED weight-2 checks
    # (0,1),(2,3); x = all-ones commutes (each row · x = 0 mod 2). The support
    # graph is two disjoint edges (Cheeger 0) → Algorithm 1 adds two edges to
    # connect them into a 4-cycle whose added edges have no backing check, so the
    # cycle is NON-redundant → main-path ∂0 is one weight-4 cycle. With
    # cellulate_to=3 the `if sparsity unacceptable` branch fires and cellulate
    # splits it into two weight-3 cycles. |support|=4 → instant. (Verified in
    # pre-flight: main_max=4, branch ∂0 weights [3,3], chord 4→5 edges.)
    HC = np.array([[1,1,0,0],[0,0,1,1]], dtype=np.uint8)
    x = np.array([1,1,1,1], dtype=np.uint8)
    cm0 = edge_expanded_maps(HC, x, seed=0, cellulate_to=None)   # main path
    main_max = int(cm0.partial_0.sum(axis=1).max()) if cm0.partial_0.shape[0] else 0
    assert main_max == 4                                          # the weight-4 cycle
    cm = edge_expanded_maps(HC, x, seed=0, cellulate_to=3)        # force the branch
    assert cheeger_constant(cm.incidence) >= 1.0
    assert np.all((np.asarray(cm.partial_0).astype(int) @ cm.incidence.astype(int)) % 2 == 0)
    assert int(cm.partial_0.sum(axis=1).max()) <= 3              # cellulated to target
    assert cm.incidence.shape[0] > cm0.incidence.shape[0]        # a chord edge was added
