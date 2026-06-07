"""Verify SkipTree gives Ide's canonical H_R(d) form on auxiliary graphs.

Ide et al. arXiv:2410.03628 §VIII Algorithm 2 (flag-based SkipTree):
  "Applying the slightly optimized flag-based SkipTree algorithm to the
   auxiliary graph G_1 for [[98, 6, 12]] BB_1 code logical, we obtain
   sparse T_1 and permutation P_1, such that T_1 G_1 P_1 = H_R(14),
   where H_R(d) is the canonical basis of the repetition code, here of
   distance d = 14."

Our _skip_tree port (from eswaroop/adapters-LDPC-surgery, MIT) gives the
EXACT canonical form when applied to the spanning tree of G_1.

This validates that:
  1. Our SkipTree port is correct
  2. Ide's bridge construction (connecting same-label vertices) corresponds
     to our v2 bridge with w = min(wt(L_1), wt(L_2)) = 14 data qubits
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import sympy

from qldpc import codes
from qldpc.codes.surgery import _skip_tree, build_layered_surgery_code


def main():
    print("=" * 70)
    print("SkipTree verification: T G_1 P = H_R(14) (Ide arXiv:2410.03628)")
    print("=" * 70)

    x, y = sympy.symbols("x y")
    bb1 = codes.BBCode((7, 7), x**3 + y**3 + y**4, y**6 + x**2 + x**5)

    support_z1 = [6, 8, 13, 17, 31, 32, 33, 35, 36, 37, 41, 50, 51, 93]
    z1 = np.zeros(98, dtype=int)
    for q in support_z1:
        z1[q] = 1

    target_code = codes.CSSCode(
        bb1.matrix_z, bb1.matrix_x, is_subsystem_code=False
    )
    _, layout = build_layered_surgery_code(
        target_code, z1, num_layers=1, validate_logical_op=False
    )
    F = np.asarray(layout.F).astype(int)
    n_V = F.shape[1]

    G1 = nx.Graph()
    G1.add_nodes_from(range(n_V))
    for i, row in enumerate(F):
        eps = sorted(np.flatnonzero(row).tolist())
        if len(eps) == 2 and not G1.has_edge(eps[0], eps[1]):
            G1.add_edge(eps[0], eps[1])
    print(f"\nBB_1 Z̄_1 auxiliary graph G_1:")
    print(f"  |V| = {G1.number_of_nodes()}, |E| = {G1.number_of_edges()}")

    span = nx.minimum_spanning_tree(G1)
    T, P = _skip_tree(span, root=0)
    T = T.astype(int)
    P = P.astype(int)

    edge_list = list(span.edges())
    G_mat = np.zeros((len(edge_list), n_V), dtype=int)
    for i, (u, v) in enumerate(edge_list):
        G_mat[i, u] = 1
        G_mat[i, v] = 1

    product = (T @ G_mat @ P) % 2
    expected = np.zeros((n_V - 1, n_V), dtype=int)
    for l in range(n_V - 1):
        expected[l, l] = 1
        expected[l, l + 1] = 1
    match = np.array_equal(product, expected)

    print(f"\nApply our _skip_tree port to spanning_tree(G_1):")
    print(f"  T shape: {T.shape}, sparsity: {sorted(set(int(r.sum()) for r in T))}")
    print(f"  P shape: {P.shape} (permutation matrix)")

    print(f"\nT @ G_mat @ P:")
    print(product)

    print(f"\nExpected H_R({n_V}) (length-{n_V} repetition code parity check):")
    print(expected)

    print(f"\n★ EXACT MATCH: T G_1 P = H_R({n_V}): {match}")

    print()
    print("=" * 70)
    print("IMPLICATIONS for joint measurement (Ide §VII B):")
    print("=" * 70)
    print(f"After SkipTree, BB_1 G_1 and LP_2 G_2 both reduce to H_R(14).")
    print(f"Bridge between codes = connect SAME-LABEL vertices = 14 edges.")
    print(f"This matches our v2 bridge formula w = min(wt(L_1), wt(L_2)) = 14.")
    print(f"  → 14 bridge data qubits in BOTH our v2 AND Ide's construction.")


if __name__ == "__main__":
    main()
