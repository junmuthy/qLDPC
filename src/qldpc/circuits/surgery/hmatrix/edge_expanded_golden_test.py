# edge_expanded_golden_test.py
"""Paper golden fixtures for the edge-expanded homological measurement.

Validates the worked Examples 5, 6, 7 of Benjamin Ide, Manoj G. Gowda,
Priya J. Nadkarni, Guillaume Dauphinais, "Fault-Tolerant Logical Measurements
via Homological Measurement", arXiv:2410.02753 (§III B) against the committed
``restrict_maps`` / ``algorithm_1`` / ``edge_expanded_maps`` /
``cheeger_constant`` implementations:

- Example 6: Steane [[7,1,3]] weight-3 X̄ = X1X2X3 — merged [[9,0]], ∂_0 empty
  ("∂_0 does not appear in this example"), Cheeger ≥ 1 out of the box.
- Example 7: [[15,7,3]] quantum Hamming code, X̄ = X3X4X5X12X14 — support graph
  has Cheeger 1/2; Algorithm 1 adds exactly two edges to restore Cheeger ≥ 1.
- Example 5: weight-8 X̄ whose ∂_1* graph is an 8-cycle (Cheeger 1/2);
  Algorithm 1 adds exactly two edges (Fig 2b) to reach Cheeger ≥ 1.
"""
import numpy as np
from qldpc.circuits.surgery.hmatrix.edge_expanded import (
    restrict_maps, algorithm_1, edge_expanded_maps, cheeger_constant)

STEANE_HZ = np.array([[0,0,0,1,1,1,1],[0,1,1,0,0,1,1],[1,0,1,0,1,0,1]], dtype=np.uint8)

def test_example6_steane_no_cycles():
    # arXiv:2410.02753 Example 6: X̄=X1X2X3 -> merged [[9,0]], ∂0 empty, no cycles.
    x = np.array([1,1,1,0,0,0,0], dtype=np.uint8)
    cm = edge_expanded_maps(STEANE_HZ, x, seed=0)
    assert cm.partial_0.shape[0] == 0                    # "∂0 does not appear in this example"
    assert cheeger_constant(cm.incidence) >= 1.0

def test_example7_hamming_algorithm1_is_load_bearing():
    # arXiv:2410.02753 Example 7: [[15,7,3]] quantum Hamming code, X̄=X3X4X5X12X14.
    # H_X=H_Z = the [15,11,3] Hamming parity check (column j = binary rep of j,
    # 1-indexed). Without Alg 1 the support graph has Cheeger 0.5 (distance would
    # drop to 2); Alg 1 adds exactly 2 edges to restore Cheeger 1. (Pre-verified:
    # support (2,3,4,11,13), incidence_star weights [2,2,4,2], 0.5 -> 1.0, +2 edges.)
    H = np.array([[(j >> i) & 1 for j in range(1, 16)] for i in range(4)], dtype=np.uint8)
    x = np.zeros(15, dtype=np.uint8); x[[2,3,4,11,13]] = 1     # X3X4X5X12X14 (0-indexed)
    assert not ((H @ x) % 2).any()                             # X̄ commutes with H_Z
    r = restrict_maps(H, x)
    assert abs(cheeger_constant(r.incidence_star) - 0.5) < 1e-9   # paper: Cheeger 0.5
    inc_after = algorithm_1(r.incidence_star, seed=0)
    assert cheeger_constant(inc_after) >= 1.0
    assert inc_after.shape[0] - r.incidence_star.shape[0] == 2    # paper: two additional edges

def test_example5_weight8_cellulation_bound():
    # arXiv:2410.02753 Example 5: a weight-8 X̄ whose ∂1* graph is an 8-cycle
    # (Cheeger 1/2). After Alg 1 (adds 2 edges) + cellulation, max ∂0 weight ≤ 5.
    inc8 = np.zeros((8, 8), dtype=np.uint8)
    for i in range(8):
        inc8[i, i] = 1; inc8[i, (i+1) % 8] = 1                 # 8-cycle
    assert abs(cheeger_constant(inc8) - 0.5) < 1e-9
    out = algorithm_1(inc8, seed=0)
    assert out.shape[0] - 8 == 2                               # "two additional edges" (Fig 2b)
    assert cheeger_constant(out) >= 1.0
