# edge_expanded_test.py
import numpy as np
from qldpc.circuits.surgery.hmatrix.edge_expanded import restrict_maps

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
