# edge_expanded_test.py
import numpy as np
from qldpc.circuits.surgery.hmatrix.edge_expanded import restrict_maps
from qldpc.circuits.surgery.hmatrix.edge_expanded import (
    boundary, cheeger_constant, sparsest_cut)

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
