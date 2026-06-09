"""Standalone bridge adapter for two-PPM joint surgery (arXiv:2410.03628 §IV / §VII).

Handles both intra-code (g1.code is g2.code) and inter-code joints.
"""

from __future__ import annotations

import dataclasses

import networkx as nx
import numpy as np

from qldpc.objects import PauliXZ

from .gadget import GadgetLayout


@dataclasses.dataclass(frozen=True, eq=False)
class Bridge:
    """Universal adapter between two GadgetLayouts (arXiv:2410.03628 §IV / §VII).

    Attributes match docs/superpowers/specs/2026-06-09-joint-ppm-bridge-design.md §1.
    """
    width: int                                  # w = |𝒜| (adapter qubits)
    basis: PauliXZ                              # X or Z (symmetric dual)
    port_l: tuple[int, ...]                     # 𝒫_l* ⊆ V_0^(l), length w
    port_r: tuple[int, ...]                     # 𝒫_r* ⊆ V_0^(r), length w
    label_l: tuple[int, ...]                    # label_l[i] = SkipTree label of V_0^(l)[i]; -1 if i ∉ 𝒫_l*
    label_r: tuple[int, ...]
    extra_kappa_l: np.ndarray                   # (e_l, |V_0^(l)|) F_2; weight-2 rows added
    extra_kappa_r: np.ndarray
    T_l: np.ndarray                             # (w-1, |C_0^(l)| + e_l) F_2 (3,2)-sparse
    T_r: np.ndarray
    H_R: np.ndarray                             # (w-1, w) canonical rep code parity
    g_l_aug: GadgetLayout                       # gadget rebuilt over F_aug^(l)
    g_r_aug: GadgetLayout


def _skip_tree(
    S: nx.Graph,
    root: int = 0,
    edge_index_verts: dict[tuple[int, int], int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """SkipTree basis transform (arXiv:2410.03628 §III). Returns T, P."""
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


def _canonical_H_R(w: int) -> np.ndarray:
    """Full-rank canonical rep-code parity check matrix, shape (w-1) × w.

    Row i has 1s in columns i and i+1. rank == w-1; column 0 and column w-1 have
    weight 1, other columns weight 2.
    """
    if w < 2:
        raise ValueError(f"H_R requires w >= 2, got {w}")
    H = np.zeros((w - 1, w), dtype=np.int_)
    for i in range(w - 1):
        H[i, i] = 1
        H[i, i + 1] = 1
    return H


def _label_inverse(P: np.ndarray) -> list[int]:
    """Return inv[l] = v such that P[v, l] = 1."""
    n = P.shape[0]
    inv = [-1] * n
    for v in range(n):
        for l in range(n):
            if P[v, l] == 1:
                inv[l] = v
                break
    return inv


def _skip_tree_fullrank(
    S: nx.Graph,
    root: int = 0,
    edge_index_verts: dict[tuple[int, int], int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SkipTree (T, P) satisfying T · G · P == H_R (full-rank rep code).

    Uses a spanning tree of S for the DFS vertex labeling (paper Algorithm 1
    is defined on a tree), then expresses each T row as the XOR of shortest-
    path edges in the full graph S. This lets S be any connected graph; the
    direct _skip_tree call would IndexError on cyclic inputs.

    Sparsity (paper Theorem 7): row weight ≤ 3, column weight ≤ 2.

    Returns (T, P) of shapes (n-1, |E|) and (n, n).
    """
    n = S.number_of_nodes()
    span = nx.minimum_spanning_tree(S)
    _, P = _skip_tree(span, root=root, edge_index_verts=None)
    label = _label_inverse(P)

    if edge_index_verts is None:
        edge_index_verts = {tuple(sorted(e)): i for i, e in enumerate(S.edges())}

    T = np.zeros((n - 1, len(edge_index_verts)), dtype=np.int_)
    for l_idx in range(n - 1):
        path = nx.shortest_path(S, source=label[l_idx], target=label[l_idx + 1])
        for u, v in zip(path[:-1], path[1:]):
            e = tuple(sorted((u, v)))
            T[l_idx, edge_index_verts[e]] ^= 1   # XOR cancels back-and-forth
    return T.astype(np.int_), P.astype(np.int_)


def _cellulate_strict(
    G_aux: nx.Graph,
    ports: tuple[int, ...],
    *,
    max_len: int = 6,
) -> list[tuple[int, int]]:
    """Break cycles longer than ``max_len`` by adding chord edges.

    Mutates G_aux. Only adds chords whose endpoints lie in ``ports`` so the
    added edges remain valid weight-2 rows of the augmented F matrix.

    Returns the list of added edges in insertion order. Idempotent once all
    basis cycles fit under the cap.
    """
    ports_set = set(ports)
    added: list[tuple[int, int]] = []
    while True:
        long_cycles = [c for c in nx.cycle_basis(G_aux) if len(c) > max_len]
        if not long_cycles:
            return added
        cycle = long_cycles[0]
        n = len(cycle)
        # Scan linearly from cycle[0] for the first port-port chord we can add.
        for offset in range(1, n):
            u, v = cycle[0], cycle[offset]
            if u not in ports_set or v not in ports_set:
                continue
            u, v = sorted((u, v))
            if G_aux.has_edge(u, v):
                continue
            G_aux.add_edge(u, v)
            added.append((u, v))
            break
        else:
            raise RuntimeError(
                f"No port-port chord found to cellulate cycle of length {n}; "
                f"ports={ports!r}, cycle={cycle!r}"
            )


def _build_aux_graph_strict(F: np.ndarray) -> tuple[nx.Graph, dict[tuple[int, int], int]]:
    """Build auxiliary graph from F; raise on hyperedges.

    Vertices: range(|V_0|) = range(F.shape[1]).
    Edges: one per weight-2 row of F, between the two columns where the row has 1s.

    Raises:
        ValueError: if any row of F has weight 1 (would-be self-loop / dangling edge).
        NotImplementedError: if any row of F has weight >= 3 (hyperedge), pointing to
        paper §II.C decomposition.
    """
    F_arr = np.asarray(F).astype(int)
    G = nx.Graph()
    G.add_nodes_from(range(F_arr.shape[1]))
    edge_index: dict[tuple[int, int], int] = {}
    for i, row in enumerate(F_arr):
        eps = np.flatnonzero(row).tolist()
        if len(eps) == 0:
            continue
        if len(eps) == 1:
            raise ValueError(
                f"F row {i} has weight 1 (column {eps[0]}). "
                f"Auxiliary-graph edges require exactly 2 endpoints."
            )
        if len(eps) >= 3:
            raise NotImplementedError(
                f"F row {i} has weight {len(eps)} (hyperedge). "
                f"Universal-adapter construction here requires weight-2 rows. "
                f"To handle hyperedges, decompose them per arXiv:2410.03628 §II.C."
            )
        u, v = sorted(eps)
        if (u, v) not in edge_index:
            edge_index[(u, v)] = len(edge_index)
            G.add_edge(u, v)
    return G, edge_index


def _connect_induced_subgraph(
    G_aux: nx.Graph,
    ports: tuple[int, ...],
) -> list[tuple[int, int]]:
    """Add edges to G_aux so that G_aux.subgraph(ports) is connected.

    Mutates G_aux. Each added edge has both endpoints in ``ports`` so it
    contributes a weight-2 row to the augmented F matrix downstream.

    Loop invariant: u and v are drawn from different components of
    G_aux.subgraph(ports), so G_aux cannot already have a (u, v) edge —
    such an edge would put them in the same component.

    Returns the list of added edges in insertion order.
    """
    added: list[tuple[int, int]] = []
    while True:
        comps = list(nx.connected_components(G_aux.subgraph(ports)))
        if len(comps) <= 1:
            return added
        u, v = sorted((min(comps[0]), min(comps[1])))
        G_aux.add_edge(u, v)
        added.append((u, v))


def build_bridge(g1: GadgetLayout, g2: GadgetLayout) -> "Bridge":
    """Two-PPM bridge between gadgets. arXiv:2410.03628 §IV / §VII."""
    raise NotImplementedError("rewritten in Task 7")
