"""Lemma 14 cellulation: break long cycles in the auxiliary graph.

Direct port of cellulate_long_cycles() in
https://github.com/eswaroop/adapters-LDPC-surgery (MIT, 2025).
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def _cellulate_long_cycles(
    G: nx.Graph,
    edge_qubit_to_vertices: dict[int, tuple[int, int]],
    vert_to_edge: dict[tuple[int, int], int],
    G_mat: np.ndarray,
    max_len: int = 6,
) -> tuple[list[tuple[int, int]], dict[int, tuple[int, int]], dict[tuple[int, int], int], np.ndarray]:
    """Cellulation: break cycles longer than max_len by adding chord edges.

    Direct port of cellulate_long_cycles() in
    https://github.com/eswaroop/adapters-LDPC-surgery cellulation.py
    (MIT, 2025). Implements Lemma 14 of Swaroop et al. arXiv:2410.03628.

    For each cycle of length > max_len in nx.cycle_basis(G), adds a chord
    edge between vertex 0 and vertex n//2 of the cycle, then recomputes
    the cycle basis. Mutates G, edge_qubit_to_vertices, vert_to_edge, and
    G_mat in place.

    Args:
        G: graph to mutate.
        edge_qubit_to_vertices: dict mapping edge-qubit index -> vertex pair.
        vert_to_edge: inverse mapping.
        G_mat: edge-vertex incidence matrix (shape: |E| x |V|).
        max_len: maximum allowed cycle length. Default 6.

    Returns:
        (new_edges_added, edge_qubit_to_vertices, vert_to_edge, G_mat).
    """
    new_edges = []
    next_edge_index = (max(edge_qubit_to_vertices.keys()) + 1) if edge_qubit_to_vertices else 0

    while True:
        cycles = nx.cycle_basis(G)
        long_cycles = [c for c in cycles if len(c) > max_len]
        if not long_cycles:
            break
        cycle = long_cycles[0]
        n = len(cycle)
        u = cycle[0]
        v = cycle[(n // 2) % n]
        u, v = sorted((u, v))

        if not G.has_edge(u, v):
            G.add_edge(u, v)
            new_edges.append((u, v))
            edge_qubit_to_vertices[next_edge_index] = (u, v)
            vert_to_edge[(u, v)] = next_edge_index
            n_vertices = G_mat.shape[1]
            new_row = np.zeros((1, n_vertices), dtype=np.int_)
            new_row[0, u] = 1
            new_row[0, v] = 1
            G_mat = np.vstack([G_mat, new_row])
            next_edge_index += 1

    return new_edges, edge_qubit_to_vertices, vert_to_edge, G_mat
