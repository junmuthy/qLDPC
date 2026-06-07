"""DEPRECATED: this script depended on the legacy surgery API
(_build_layered_blocks, layout.c0_indices, layout.num_layers,
layout.hx_row_kind, layout.hz_row_kind, layout.num_ancilla_qubits)
which was removed in feat/surgery-construction.

# TODO: port to new API (build_gadget + GadgetLayout.{V0,C0,F,G})
Will be ported as needed; for now skip execution.

Original docstring preserved below:

Prototype Ide Lemma 10 adapter (full SkipTree bridge for Z̄_1 Z̄_2 measurement).

Ide arXiv:2410.03628 Lemma 10: Adapter X-cycle-check matrix N has rows:

    N_l = (T_1[l, :] on E_1) + (H_C[l, :] on A) + (T_2[l, :] on E_2)

Where T_s G_s P_s = H_C (canonical repetition code parity check).

The adapter X-stabs are CYCLES in the adapted graph G_l ∼_A G_r and
automatically commute with vertex Z-stabs via the cycle property (each
vertex has EVEN incidence in any cycle).
"""

from __future__ import annotations

import sys
sys.exit(0)

import galois
import networkx as nx
import numpy as np
import sympy

from qldpc import codes
from qldpc.codes.surgery.bridge import _skip_tree
from qldpc.codes.surgery import build_gadget
from qldpc.objects import Pauli


def build_bb1():
    x, y = sympy.symbols("x y")
    return codes.BBCode((7, 7), x**3 + y**3 + y**4, y**6 + x**2 + x**5)


def build_aux_graph_from_F(F: np.ndarray) -> tuple[nx.Graph, np.ndarray]:
    """Build aux graph G from F, returning (G, edge_incidence_matrix).

    G has |V_0| vertices and rows of F as edges (each row = e_u + e_v).
    Returns G and a (|E|, |V|) incidence matrix.
    """
    n_V = F.shape[1]
    G = nx.Graph()
    G.add_nodes_from(range(n_V))
    incidence_rows = []
    for row in F:
        eps = np.flatnonzero(row).tolist()
        if len(eps) == 2:
            u, v = int(eps[0]), int(eps[1])
            G.add_edge(u, v)
            incidence_rows.append(row.copy())
    return G, np.array(incidence_rows, dtype=np.int_)


def apply_skiptree_to_F(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply SkipTree to spanning tree of F's graph. Returns (T, P)."""
    G, _ = build_aux_graph_from_F(F)
    span = nx.minimum_spanning_tree(G)
    T, P = _skip_tree(span, root=0)
    return T.astype(np.int_), P.astype(np.int_)


def build_joint_with_ide_adapter():
    bb1 = build_bb1()
    # Z̄_1 from Table I
    support_z1 = [6, 8, 13, 17, 31, 32, 33, 35, 36, 37, 41, 50, 51, 93]
    z1 = np.zeros(98, dtype=np.int_)
    for q in support_z1:
        z1[q] = 1

    # For demonstration, use BB_1 for both gadgets (same code, different logical)
    # Use Z̄_3 as the second logical
    support_z3 = [10, 17, 35, 39, 42, 43, 53, 55, 61, 70, 84, 89]
    z3 = np.zeros(98, dtype=np.int_)
    for q in support_z3:
        z3[q] = 1

    # Build single-gadget codes
    target_code = codes.CSSCode(
        bb1.matrix_z, bb1.matrix_x, is_subsystem_code=False
    )
    merged1, layout1 = build_layered_surgery_code(
        target_code, z1, num_layers=1, validate_logical_op=False
    )
    merged2, layout2 = build_layered_surgery_code(
        target_code, z3, num_layers=1, validate_logical_op=False
    )

    F1 = np.asarray(layout1.F).astype(np.int_)
    F2 = np.asarray(layout2.F).astype(np.int_)
    print(f"F_1 shape: {F1.shape} (|E_1|={F1.shape[0]}, |V_1|={F1.shape[1]})")
    print(f"F_2 shape: {F2.shape} (|E_2|={F2.shape[0]}, |V_2|={F2.shape[1]})")

    # Apply SkipTree to spanning trees
    G1, _ = build_aux_graph_from_F(F1)
    G2, _ = build_aux_graph_from_F(F2)
    span1 = nx.minimum_spanning_tree(G1)
    span2 = nx.minimum_spanning_tree(G2)
    T1, P1 = _skip_tree(span1, root=0)
    T2, P2 = _skip_tree(span2, root=0)
    T1 = T1.astype(np.int_)
    P1 = P1.astype(np.int_)
    T2 = T2.astype(np.int_)
    P2 = P2.astype(np.int_)
    print(f"T_1 shape: {T1.shape}, T_2 shape: {T2.shape}")

    # T_s acts on SPANNING TREE edges only. We need to map T_s indices to
    # which row of F_s corresponds to each spanning tree edge.
    tree1_edges = list(span1.edges())
    tree2_edges = list(span2.edges())
    # For each F row (edge), find its spanning tree index (or None if not in tree)
    def edge_to_tree_idx(F: np.ndarray, tree_edges: list[tuple[int, int]]) -> dict[int, int]:
        # Maps F row idx → spanning tree edge idx (in tree_edges order)
        edge_to_idx = {tuple(sorted([int(u), int(v)])): i for i, (u, v) in enumerate(tree_edges)}
        result = {}
        for i, row in enumerate(F):
            eps = sorted(np.flatnonzero(row).tolist())
            if len(eps) == 2:
                key = (eps[0], eps[1])
                if key in edge_to_idx:
                    result[i] = edge_to_idx[key]
        return result

    F1_to_tree1 = edge_to_tree_idx(F1, tree1_edges)
    F2_to_tree1 = edge_to_tree_idx(F2, tree2_edges)
    print(f"F_1 → spanning tree edges: {len(F1_to_tree1)}/{F1.shape[0]} edges in tree")

    # Bridge: w qubits
    w = min(F1.shape[1], F2.shape[1])
    print(f"Bridge w = {w}")

    # Adapter X-cycle-checks (w-1 of them):
    #   N[l, :] support:
    #     on κ_1 (F_1 rows, indexed by tree edges): T_1[l, :] expanded
    #     on bridge: e_l + e_{l+1}
    #     on κ_2 (F_2 rows, indexed by tree edges): T_2[l, :] expanded

    # Build the new joint code's HX matrix:
    # Layout: [data (98) | κ_1 (21) | κ_2 (?) | bridge (w)]
    n_data = bb1.num_qubits
    n_k1 = layout1.num_ancilla_qubits
    n_k2 = layout2.num_ancilla_qubits
    n_total = n_data + n_k1 + n_k2 + w

    # Stack: HX_BB1 + chi rows of gadget 1 + chi rows of gadget 2 + adapter
    HX1 = np.asarray(merged1.matrix_x).astype(np.int_)
    HX2 = np.asarray(merged2.matrix_x).astype(np.int_)

    # Pad merged1's matrix to full size
    def pad_hx(HX, k_block_start, k_block_size):
        out = np.zeros((HX.shape[0], n_total), dtype=np.int_)
        out[:, :n_data] = HX[:, :n_data]
        out[:, k_block_start : k_block_start + k_block_size] = HX[:, n_data:]
        return out

    HX1_padded = pad_hx(HX1, n_data, n_k1)
    HX2_padded = pad_hx(HX2, n_data + n_k1, n_k2)

    # EXTEND each chi row of gadget 1 with X on bridge qubit at its SkipTree label.
    # chi rows are at indices where layout1.hx_row_kind != "data", in V_0 order.
    # For vertex v (= chi row), label is determined by P_1.
    # label_inv[v] = l means vertex v has SkipTree label l.
    n_x_data_1 = int(np.sum(layout1.hx_row_kind == "data"))
    n_x_data_2 = int(np.sum(layout2.hx_row_kind == "data"))
    bridge_start = n_data + n_k1 + n_k2

    label_inv_1 = np.zeros(F1.shape[1], dtype=np.int_)
    for l in range(P1.shape[1]):
        v = int(np.argmax(P1[:, l]))
        label_inv_1[v] = l
    label_inv_2 = np.zeros(F2.shape[1], dtype=np.int_)
    for l in range(P2.shape[1]):
        v = int(np.argmax(P2[:, l]))
        label_inv_2[v] = l

    # Extend chi rows of gadget 1 (rows n_x_data_1 ... in HX1_padded)
    for v in range(F1.shape[1]):
        l = label_inv_1[v]
        if l < w:
            HX1_padded[n_x_data_1 + v, bridge_start + l] = 1
    for v in range(F2.shape[1]):
        l = label_inv_2[v]
        if l < w:
            HX2_padded[n_x_data_2 + v, bridge_start + l] = 1

    # Remove duplicate data rows from HX2
    is_data2 = layout2.hx_row_kind == "data"
    HX2_nondata = HX2_padded[~is_data2]

    # Build adapter X-stabs (w-1 of them, rows of N[1:])
    # T_1 has shape (n_V_1, n_V_1 - 1) = (14, 13) in our test
    # Each ROW of T_1 corresponds to a "path" (canonical label pair)
    # Each COLUMN of T_1 corresponds to a spanning tree edge

    # For adapter row l: support on κ_1 = sum over tree_edge_idx e of T_1[l, e] * F_row_for_tree_edge_e
    # T_1 and T_2 may have different sizes (w_1-1, w_2-1) if V_0 sizes differ.
    # Use w-1 = min(T1.shape[0], T2.shape[0]) - 1 ... actually we need T to act
    # on each gadget independently. For now, use the smaller dimension.
    n_adapter = min(T1.shape[0], T2.shape[0]) - 1  # = w - 1
    adapter_rows = np.zeros((n_adapter, n_total), dtype=np.int_)
    bridge_start = n_data + n_k1 + n_k2

    for l in range(n_adapter):
        # κ_1 part: T_1[l, :] selects spanning tree edges
        for tree_edge_idx in range(T1.shape[1]):
            if T1[l, tree_edge_idx] == 1:
                # Find F_1 row corresponding to this tree edge
                f1_row_idx = None
                for f_idx, t_idx in F1_to_tree1.items():
                    if t_idx == tree_edge_idx:
                        f1_row_idx = f_idx
                        break
                if f1_row_idx is not None:
                    # κ_1 column for F_1 row f1_row_idx is at offset f1_row_idx in κ_1 block
                    adapter_rows[l, n_data + f1_row_idx] ^= 1
        # κ_2 part: T_2[l, :]
        for tree_edge_idx in range(T2.shape[1]):
            if T2[l, tree_edge_idx] == 1:
                f2_row_idx = None
                for f_idx, t_idx in F2_to_tree1.items():
                    if t_idx == tree_edge_idx:
                        f2_row_idx = f_idx
                        break
                if f2_row_idx is not None:
                    adapter_rows[l, n_data + n_k1 + f2_row_idx] ^= 1
        # Bridge: e_l + e_{l+1}
        adapter_rows[l, bridge_start + l] = 1
        adapter_rows[l, bridge_start + l + 1] = 1

    # Combine all X-stabs
    HX_joint = np.vstack([HX1_padded, HX2_nondata, adapter_rows])

    print(f"\nHX_joint shape: {HX_joint.shape}")

    # HZ: simpler — just data Z-stabs + gauge-fix of both gadgets
    # For prototype, skip detailed κ-splicing and just use merged1's HZ extended
    HZ1 = np.asarray(merged1.matrix_z).astype(np.int_)
    HZ2 = np.asarray(merged2.matrix_z).astype(np.int_)

    def pad_hz(HZ, k_block_start, k_block_size):
        out = np.zeros((HZ.shape[0], n_total), dtype=np.int_)
        out[:, :n_data] = HZ[:, :n_data]
        out[:, k_block_start : k_block_start + k_block_size] = HZ[:, n_data:]
        return out

    HZ1_padded = pad_hz(HZ1, n_data, n_k1)
    HZ2_padded = pad_hz(HZ2, n_data + n_k1, n_k2)

    # κ_2 splice: gadget1's data Z rows must extend with κ_2 column for each
    # j in layout2.c0_indices, so that gadget2's chi rows commute with them.
    is_data_z1 = layout1.hz_row_kind == "data"
    data_hz1_indices = np.flatnonzero(is_data_z1)
    from qldpc.codes.surgery import _build_layered_blocks
    blocks2 = _build_layered_blocks(layout2.F, layout2.num_layers)
    c1_slice_2 = blocks2.ancilla_col_slice(1)
    kappa2_col_start = n_data + n_k1 + c1_slice_2.start
    for k_idx, j in enumerate(layout2.c0_indices):
        row_idx = int(data_hz1_indices[j])
        HZ1_padded[row_idx, kappa2_col_start + k_idx] = 1

    is_data_z2 = layout2.hz_row_kind == "data"
    HZ2_nondata = HZ2_padded[~is_data_z2]
    HZ_joint = np.vstack([HZ1_padded, HZ2_nondata])

    print(f"HZ_joint shape: {HZ_joint.shape}")

    # Test CSS commutation
    css_ok = np.all((HX_joint @ HZ_joint.T) % 2 == 0)
    print(f"CSS commutation: {css_ok}")

    if not css_ok:
        product = (HX_joint @ HZ_joint.T) % 2
        bad = np.argwhere(product != 0)
        print(f"  Bad commutations: {len(bad)} pairs")
        if len(bad) > 0:
            print(f"  Example: HX row {bad[0][0]}, HZ row {bad[0][1]}")
            print(f"  This row is {'data' if bad[0][0] < HX1.shape[0] else 'chi/adapter'}")

    # Dimension count
    GF2 = galois.GF(2)
    rank_HX = int(np.linalg.matrix_rank(GF2(HX_joint)))
    rank_HZ = int(np.linalg.matrix_rank(GF2(HZ_joint)))
    k_joint = n_total - rank_HX - rank_HZ
    print(f"n_total = {n_total}, rank(HX) = {rank_HX}, rank(HZ) = {rank_HZ}")
    print(f"k_joint = {k_joint}, expected k_data - 1 = {bb1.dimension - 1}")


if __name__ == "__main__":
    build_joint_with_ide_adapter()
