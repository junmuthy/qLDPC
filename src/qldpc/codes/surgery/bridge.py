"""Standalone bridge adapter for two-PPM joint surgery (math.md §2).

Handles both intra-code (g1.code is g2.code) and inter-code joints.
"""

from __future__ import annotations

import dataclasses

import galois
import networkx as nx
import numpy as np

from qldpc.objects import Pauli, PauliXZ

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
    basis: PauliXZ = dataclasses.field(default=Pauli.X)


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


def _cellulate_long_cycles(
    G: nx.Graph,
    edge_qubit_to_vertices: dict[int, tuple[int, int]],
    vert_to_edge: dict[tuple[int, int], int],
    G_mat: np.ndarray,
    max_len: int = 6,
) -> tuple[list[tuple[int, int]], dict[int, tuple[int, int]], dict[tuple[int, int], int], np.ndarray]:
    """Cellulation: break cycles longer than max_len. math.md §2 / arXiv:2410.03628 Lemma 14."""
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


def _build_auxiliary_graph_from_F(
    F: np.ndarray,
) -> tuple[nx.Graph, dict[int, tuple[int, int]]]:
    """Build aux graph G_s from F matrix. Vertices = cols(F); edges = weight-2 rows."""
    F_arr = np.asarray(F).astype(int)
    n_V = F_arr.shape[1]
    G = nx.Graph()
    G.add_nodes_from(range(n_V))
    edge_qubit_to_vertices: dict[int, tuple[int, int]] = {}
    for i, row in enumerate(F_arr):
        eps = sorted(np.flatnonzero(row).tolist())
        if len(eps) == 2:
            u, v = eps[0], eps[1]
            edge_qubit_to_vertices[i] = (u, v)
            if not G.has_edge(u, v):
                G.add_edge(u, v)
    return G, edge_qubit_to_vertices


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

    Returns the list of added edges in insertion order.
    """
    ports_set = set(ports)
    added: list[tuple[int, int]] = []
    while True:
        sub = G_aux.subgraph(ports)
        comps = list(nx.connected_components(sub))
        if len(comps) <= 1:
            return added
        # Pick lowest-indexed vertex of first component and lowest of second
        c0 = sorted(comps[0])
        c1 = sorted(comps[1])
        u, v = sorted((c0[0], c1[0]))
        assert u in ports_set and v in ports_set
        if not G_aux.has_edge(u, v):
            G_aux.add_edge(u, v)
            added.append((u, v))


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


def _running_xor_b_c(T_col: np.ndarray) -> np.ndarray:
    """Solve H_R @ b = T_col with b[0]=0 via running XOR."""
    w = T_col.shape[0] + 1
    b = np.zeros(w, dtype=np.int_)
    for l in range(1, w):
        b[l] = (b[l - 1] + int(T_col[l - 1])) % 2
    return b


def _solve_chi_z_bridge_choices(
    T_s: np.ndarray,
    label_inv: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Solve (gamma, delta) F_2-linear system for chi-Z bridge compatibility. math.md §2."""
    w = T_s.shape[0] + 1
    n_E = T_s.shape[1]
    n_v0 = len(label_inv)

    if w % 2 != 0:
        raise ValueError(
            f"Joint (gamma, delta) system requires w even (got w={w}); "
            f"odd w introduces a gamma_v*delta_c cross-term that is not F_2-linear."
        )

    canonical_B = np.zeros((w, n_E), dtype=np.int_)
    for c in range(n_E):
        canonical_B[:, c] = _running_xor_b_c(T_s[:, c])

    sigma = canonical_B.sum(axis=0) % 2

    label = [0] * n_v0
    for l, v in enumerate(label_inv):
        label[v] = l

    a = np.zeros((n_v0, n_E), dtype=np.int_)
    for v in range(n_v0):
        a[v, :] = canonical_B[label[v], :]

    n_eq = n_v0 * n_E
    n_var = n_v0 + n_E
    A = np.zeros((n_eq, n_var), dtype=np.int_)
    rhs = np.zeros(n_eq, dtype=np.int_)
    for v in range(n_v0):
        for c in range(n_E):
            row = v * n_E + c
            A[row, v] = sigma[c]
            A[row, n_v0 + c] = 1
            rhs[row] = a[v, c]

    aug = GF2(np.hstack([A, rhs.reshape(-1, 1)]))
    rref = np.asarray(aug.row_reduce())
    x = np.zeros(n_var, dtype=np.int_)
    for r in range(rref.shape[0]):
        nz = np.flatnonzero(rref[r, :n_var])
        if nz.size == 0:
            if rref[r, n_var] == 1:
                raise ValueError(
                    "chi-Z joint bridge system infeasible: "
                    "(gamma, delta) F_2 linear system has no solution."
                )
            continue
        x[int(nz[0])] = int(rref[r, n_var])

    gamma = x[:n_v0]
    delta = x[n_v0:]
    return gamma, delta


def build_bridge(g1: GadgetLayout, g2: GadgetLayout) -> "Bridge":
    """Two-PPM bridge between gadgets. math.md §2."""
    if g1.basis is not g2.basis:
        raise ValueError(
            f"build_bridge requires g1.basis == g2.basis, got {g1.basis!r} vs {g2.basis!r}"
        )
    basis = g1.basis
    intercode = g1.code is not g2.code
    w = min(len(g1.V0), len(g2.V0))
    if w < 2:
        raise ValueError(f"bridge width must be >= 2, got {w}")

    qubits = tuple(range(w))
    U_B = _build_path_graph_U_B(w)
    chi_endpoint_extensions: dict[int, np.ndarray] = {
        0: np.array([0], dtype=np.uint8),
    }

    if not intercode:
        return Bridge(
            width=w, qubits=qubits, U_B=U_B,
            chi_endpoint_extensions=chi_endpoint_extensions,
            intercode=False, aux_graph_edges=None, z_extensions=None,
            basis=basis,
        )

    # Inter-code path (Ide §VII C)
    G1_aux, edge_q_to_v_1 = _build_auxiliary_graph_from_F(g1.F)
    vert_to_edge_1 = {uv: k for k, uv in edge_q_to_v_1.items()}
    F1_mat = np.asarray(g1.F).astype(np.int_)
    _, edge_q_to_v_1, vert_to_edge_1, _ = _cellulate_long_cycles(
        G1_aux, dict(edge_q_to_v_1), dict(vert_to_edge_1), F1_mat,
    )
    aux_graph_edges = tuple(tuple(sorted(e)) for e in G1_aux.edges())

    z_extensions: dict[int, np.ndarray] | None = None
    if G1_aux.number_of_nodes() >= 2 and nx.is_connected(G1_aux):
        edge_index_verts = {tuple(sorted(uv)): k for k, uv in edge_q_to_v_1.items()}
        try:
            T_s, P = _skip_tree(G1_aux, root=0, edge_index_verts=edge_index_verts)
            label_inv = _label_inverse(P)
            gamma, delta = _solve_chi_z_bridge_choices(T_s, label_inv)
            z_extensions = {
                int(c): np.asarray(delta, dtype=np.uint8).copy()
                for c in edge_q_to_v_1
            }
        except (ValueError, IndexError, RecursionError):
            z_extensions = None

    return Bridge(
        width=w, qubits=qubits, U_B=U_B,
        chi_endpoint_extensions=chi_endpoint_extensions,
        intercode=True,
        aux_graph_edges=aux_graph_edges,
        z_extensions=z_extensions,
        basis=basis,
    )
