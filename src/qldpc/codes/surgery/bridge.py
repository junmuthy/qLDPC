"""Standalone bridge adapter for two-PPM joint surgery (math.md §2).

Handles both intra-code (g1.code is g2.code) and inter-code joints. SkipTree
and cellulation helpers are private to this module.
"""

from __future__ import annotations

import dataclasses

import galois
import networkx as nx
import numpy as np

from .gadget import GadgetLayout

GF2 = galois.GF(2)


@dataclasses.dataclass(frozen=True, eq=False)
class Bridge:
    width: int
    qubits: tuple[int, ...]
    U_B: np.ndarray
    chi_endpoint_extensions: dict[int, np.ndarray]
    intercode: bool
    aux_graph_edges: tuple[tuple[int, int], ...] | None
    z_extensions: dict[int, np.ndarray] | None


def _build_path_graph_U_B(w: int) -> np.ndarray:
    """math.md §2.2 — path-graph X-stabilizers on w bridge qubits."""
    if w < 2:
        raise ValueError(f"bridge width must be >= 2, got {w}")
    U_B = np.zeros((w - 1, w), dtype=np.uint8)
    for i in range(w - 1):
        U_B[i, i] = 1
        U_B[i, i + 1] = 1
    return U_B


def _skip_tree(
    S: nx.Graph,
    root: int = 0,
    edge_index_verts: dict[tuple[int, int], int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """SkipTree basis transformation (Swaroop et al. arXiv:2410.03628 §III).

    Direct port of skipTree() in https://github.com/eswaroop/adapters-LDPC-surgery
    (MIT, 2025) skip_tree_algorithm.py with attribution. The qldpc project
    is Apache 2.0; MIT and Apache 2.0 are compatible for redistribution.

    Args:
        S: connected simple graph.
        root: vertex to start the labelling at.
        edge_index_verts: optional override mapping each edge ``tuple(sorted)``
            to a column index in T. If None, columns are indexed by
            ``S.edges()`` order.

    Returns:
        T: shape (n-1, |E|) edge-incidence matrix. T[l, e] = 1 iff edge e
            lies on the shortest path from vertex labeled l to vertex
            labeled (l+1) mod n.
        P: shape (n, n) permutation matrix. P[v, l] = 1 iff vertex v has
            label l.
    """
    n = S.number_of_nodes()
    index = 0
    label = [0] * n
    visited: set[int] = set()

    def label_first(v: int, skip: bool) -> None:
        nonlocal index
        visited.add(v)
        label[index] = v
        index = index + 1

        children = [nbr for nbr in S.neighbors(v) if nbr not in visited]
        for child_idx, child in enumerate(children):
            last_in_gen = child_idx == len(children) - 1
            if last_in_gen and not skip:
                label_first(child, skip=False)
            else:
                label_last(child)

    def label_last(v: int) -> None:
        nonlocal index
        visited.add(v)
        for child in S.neighbors(v):
            if child not in visited:
                label_first(child, skip=True)
        label[index] = v
        index = index + 1

    label_first(root, skip=False)

    P = np.zeros((n, n), dtype=np.int_)
    for l_idx, v in enumerate(label):
        P[v, l_idx] = 1

    if not edge_index_verts:
        edge_index_verts = {tuple(sorted(e)): i for i, e in enumerate(S.edges())}

    T = np.zeros((n - 1, len(edge_index_verts)), dtype=np.int_)

    for l_idx in range(n - 1):
        path = nx.shortest_path(S, source=label[l_idx], target=label[(l_idx + 1) % n])
        for u, v in zip(path[:-1], path[1:]):
            e = tuple(sorted((u, v)))
            T[l_idx, edge_index_verts[e]] = 1
    return T, P


def _skip_tree_hr(
    S: nx.Graph,
    root: int = 0,
    edge_index_verts: dict[tuple[int, int], int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """SkipTree returning T G P = H_R (open-path canonical basis).

    Ide et al. arXiv:2410.03628 Appendix VIII Algorithm 2.

    On spanning-tree inputs this is observationally equivalent to
    ``_skip_tree`` (Algorithm 1), because the existing Algorithm 1
    implementation iterates ``range(n - 1)`` in its T-construction loop
    and therefore never produces the cyclic-closing row of H_C. This
    function is provided for paper traceability and as a hook for future
    divergence on non-tree inputs (where Algorithm 2's flag-based
    skipping logic would yield strictly sparser T than Algorithm 1).

    Args:
        S: connected simple graph (typically a spanning tree).
        root: vertex to start labelling at.
        edge_index_verts: optional override mapping each edge
            ``tuple(sorted)`` to a column index in T. If None, columns
            are indexed by ``S.edges()`` order.

    Returns:
        T: shape (n-1, |E|) edge-incidence matrix. T[l, e] = 1 iff
            edge e lies on the shortest path from vertex labeled l to
            vertex labeled (l+1).
        P: shape (n, n) permutation matrix. P[v, l] = 1 iff vertex v
            has label l.
    """
    n = S.number_of_nodes()
    index = 0
    label = [0] * n
    visited: set[int] = set()

    def label_first(v: int, skip: bool) -> None:
        nonlocal index
        visited.add(v)
        label[index] = v
        index = index + 1

        children = [nbr for nbr in S.neighbors(v) if nbr not in visited]
        for child_idx, child in enumerate(children):
            youngest = child_idx == len(children) - 1
            if youngest and not skip:
                label_first(child, skip=False)
            else:
                label_last(child)

    def label_last(v: int) -> None:
        nonlocal index
        visited.add(v)
        for child in S.neighbors(v):
            if child not in visited:
                label_first(child, skip=True)
        label[index] = v
        index = index + 1

    label_first(root, skip=False)

    P = np.zeros((n, n), dtype=np.int_)
    for l_idx, v in enumerate(label):
        P[v, l_idx] = 1

    if not edge_index_verts:
        edge_index_verts = {tuple(sorted(e)): i for i, e in enumerate(S.edges())}

    T = np.zeros((n - 1, len(edge_index_verts)), dtype=np.int_)
    # Open path: only labels l → l+1, no cyclic close-the-loop.
    for l_idx in range(n - 1):
        path = nx.shortest_path(S, source=label[l_idx], target=label[l_idx + 1])
        for u, v in zip(path[:-1], path[1:]):
            e = tuple(sorted((u, v)))
            T[l_idx, edge_index_verts[e]] = 1
    return T, P


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


def build_bridge(g1: GadgetLayout, g2: GadgetLayout) -> "Bridge":
    """Two-PPM bridge between gadgets. Auto-dispatches intra vs inter-code.

    math.md §2: bridge data qubits + path-graph U_B + chi endpoint extensions.
    """
    intercode = g1.code is not g2.code
    w = min(len(g1.V0), len(g2.V0))
    if w < 2:
        raise ValueError(f"bridge width must be >= 2, got {w}")

    qubits = tuple(range(w))  # relative offsets; circuit.py rebases.

    U_B = _build_path_graph_U_B(w)

    # math.md §2.3 χ-extension
    # gadget 1's χ_0 row → X on bridge[0]
    # gadget 2's χ_0 row → X on bridge[w-1]
    chi_endpoint_extensions: dict[int, np.ndarray] = {
        0: np.array([0], dtype=np.uint8),
    }

    if not intercode:
        return Bridge(
            width=w, qubits=qubits, U_B=U_B,
            chi_endpoint_extensions=chi_endpoint_extensions,
            intercode=False,
            aux_graph_edges=None,
            z_extensions=None,
        )

    # Inter-code path added in Task 12.
    raise NotImplementedError("inter-code bridge added in Task 12")
