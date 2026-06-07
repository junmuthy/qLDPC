"""SkipTree algorithm (Ide / Swaroop et al. arXiv:2410.03628 §III).

Algorithm 1 returns T G P = H_C (cyclic repetition basis).
Algorithm 2 (added in Task 7) returns T G P = H_R (open path basis).
"""

from __future__ import annotations

import networkx as nx
import numpy as np


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
