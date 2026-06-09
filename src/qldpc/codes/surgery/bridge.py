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
        # Scan all (i, j) cycle-vertex pairs for the first port-port chord we
        # can add. Anchoring on cycle[0] alone would fail spuriously when
        # cycle[0] ∉ ports_set, even though valid chords exist elsewhere.
        chord_found = False
        for i in range(n):
            if chord_found:
                break
            u_raw = cycle[i]
            if u_raw not in ports_set:
                continue
            for j in range(i + 1, n):
                v_raw = cycle[j]
                if v_raw not in ports_set:
                    continue
                u, v = sorted((u_raw, v_raw))
                if G_aux.has_edge(u, v):
                    continue
                G_aux.add_edge(u, v)
                added.append((u, v))
                chord_found = True
                break
        if not chord_found:
            raise RuntimeError(
                f"No port-port chord found to cellulate cycle of length {n}; "
                f"ports={ports!r}, cycle={cycle!r}"
            )


def _build_aux_graph_strict(F: np.ndarray) -> tuple[nx.Graph, dict[tuple[int, int], int]]:
    """Build auxiliary graph from F; weight-2 rows become edges, hyperedges are skipped.

    Vertices: range(|V_0|) = range(F.shape[1]).
    Edges: one per weight-2 row of F, between the two columns where the row has 1s.

    Weight-≥3 rows (hyperedges) are silently ignored — they remain in F_aug so
    the gadget structure (G_aug = ker(F_aug^T), deformed check c → c · X(κ_r))
    is unchanged, but T_s assigns them zero columns (existing skip at
    _run_skiptree_on_port_subgraph). CSS commutation, κ-cancellation, joint
    observable, and dimension counting all hold by direct calculation;
    see docs/superpowers/specs/2026-06-09-bridge-hyperedge-and-cellulate-fix-design.md
    §correctness-proof. Paper Eq. 9's perfect-matching decomposition (§II.C)
    is not applied; structural Theorem 12 distance argument is substituted
    by empirical LER smoke tests.

    Raises:
        ValueError: if any row of F has weight 1 (defensive — F · 1 = 0 mod 2
        forbids odd weights for a valid logical).
    """
    F_arr = np.asarray(F).astype(int)
    G = nx.Graph()
    G.add_nodes_from(range(F_arr.shape[1]))
    edge_index: dict[tuple[int, int], int] = {}
    for i, row in enumerate(F_arr):
        eps = np.flatnonzero(row).tolist()
        if len(eps) == 0 or len(eps) >= 3:
            continue
        if len(eps) == 1:
            raise ValueError(
                f"F row {i} has weight 1 (column {eps[0]}). "
                f"Auxiliary-graph edges require exactly 2 endpoints "
                f"(F · 1 = 0 mod 2 forbids odd weights — invalid logical?)."
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


def _edges_to_F_extra(edges: list[tuple[int, int]], n_V0: int) -> np.ndarray:
    """Convert a list of weight-2 (u, v) edges into a (|edges|, n_V0) F_2 matrix."""
    out = np.zeros((len(edges), n_V0), dtype=np.uint8)
    for r, (u, v) in enumerate(edges):
        out[r, u] = 1
        out[r, v] = 1
    return out


def _run_skiptree_on_port_subgraph(
    G_aux_full: nx.Graph,
    port: tuple[int, ...],
    root_port_idx: int,
    F_aug: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """Run SkipTree on the induced port subgraph; embed result back onto F_aug rows.

    The induced subgraph's vertex IDs are relabeled to [0, |port|) so the n×n P
    allocation inside ``_skip_tree`` is square. The output T is then re-expressed
    onto the original F_aug edge ordering (rows of F_aug index the κ qubits =
    edges of G_aux_full). ``root_port_idx`` selects which entry of ``port`` is
    the SkipTree root.

    Returns (T_full, labels) where T_full has shape (w-1, F_aug.shape[0]) and
    labels[orig_v] = k iff orig_v ∈ port and got SkipTree label k (else -1).
    """
    sub_orig = G_aux_full.subgraph(port).copy()
    port_sorted = sorted(port)
    new_of_orig = {orig: new for new, orig in enumerate(port_sorted)}
    orig_of_new = {new: orig for orig, new in new_of_orig.items()}
    sub_relab = nx.relabel_nodes(sub_orig, new_of_orig, copy=True)
    # Take a spanning tree (Algorithm 1 of paper expects a tree input). MST is
    # deterministic; for unweighted graphs nx returns a BFS-like tree.
    sub_tree = nx.minimum_spanning_tree(sub_relab)
    tree_edges = sorted(tuple(sorted(e)) for e in sub_tree.edges())
    edge_idx_tree = {e: i for i, e in enumerate(tree_edges)}
    root_orig = port[root_port_idx]
    root_relab = new_of_orig[root_orig]
    T_relab, P_relab = _skip_tree_fullrank(
        sub_tree, root=root_relab, edge_index_verts=edge_idx_tree,
    )
    labels = [-1] * F_aug.shape[1]
    for new_v in range(len(port)):
        orig_v = orig_of_new[new_v]
        nz = np.flatnonzero(P_relab[new_v])
        assert len(nz) == 1, f"vertex {orig_v} (relab {new_v}) has {len(nz)} labels"
        labels[orig_v] = int(nz[0])
    T_full = np.zeros((T_relab.shape[0], F_aug.shape[0]), dtype=np.int_)
    for r in range(F_aug.shape[0]):
        cols = np.flatnonzero(F_aug[r])
        if len(cols) != 2:
            continue
        u_orig, v_orig = sorted(int(x) for x in cols)
        if u_orig not in new_of_orig or v_orig not in new_of_orig:
            continue
        e_relab = tuple(sorted((new_of_orig[u_orig], new_of_orig[v_orig])))
        if e_relab in edge_idx_tree:
            T_full[:, r] = T_relab[:, edge_idx_tree[e_relab]]
    return T_full.astype(np.int_), labels


def build_bridge(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    *,
    port_subset_l: tuple[int, ...] | None = None,
    port_subset_r: tuple[int, ...] | None = None,
    spanning_tree_root_l: int = 0,
    spanning_tree_root_r: int = 0,
    cellulate_max_len: int = 6,
) -> Bridge:
    """Universal-adapter bridge between two gadgets (arXiv:2410.03628 §IV).

    See docs/superpowers/specs/2026-06-09-joint-ppm-bridge-design.md §2 for the
    7-step recipe. ``spanning_tree_root_s`` is the index INTO the port tuple of
    the SkipTree root vertex on side s.
    """
    if g_l.basis is not g_r.basis:
        raise ValueError(
            f"build_bridge requires g_l.basis == g_r.basis, "
            f"got {g_l.basis!r} vs {g_r.basis!r}"
        )
    basis = g_l.basis

    # Step 1: auxiliary graphs
    G_l_aux, _ = _build_aux_graph_strict(g_l.F)
    G_r_aux, _ = _build_aux_graph_strict(g_r.F)

    # Step 2: port subsets + width
    port_l_all = tuple(port_subset_l) if port_subset_l is not None else tuple(range(len(g_l.V0)))
    port_r_all = tuple(port_subset_r) if port_subset_r is not None else tuple(range(len(g_r.V0)))
    width = min(len(port_l_all), len(port_r_all))
    if width < 2:
        raise ValueError(f"bridge width must be >= 2, got {width}")
    port_l = port_l_all[:width]
    port_r = port_r_all[:width]
    if not (0 <= spanning_tree_root_l < width):
        raise ValueError(
            f"spanning_tree_root_l={spanning_tree_root_l} out of [0, {width})"
        )
    if not (0 <= spanning_tree_root_r < width):
        raise ValueError(
            f"spanning_tree_root_r={spanning_tree_root_r} out of [0, {width})"
        )

    # Step 3: induced-subgraph connectivity augmentation
    extras_l_conn = _connect_induced_subgraph(G_l_aux, port_l)
    extras_r_conn = _connect_induced_subgraph(G_r_aux, port_r)

    # Step 4: cellulation
    extras_l_cell = _cellulate_strict(G_l_aux, port_l, max_len=cellulate_max_len)
    extras_r_cell = _cellulate_strict(G_r_aux, port_r, max_len=cellulate_max_len)

    extras_l_edges = extras_l_conn + extras_l_cell
    extras_r_edges = extras_r_conn + extras_r_cell
    extra_kappa_l = _edges_to_F_extra(extras_l_edges, len(g_l.V0))
    extra_kappa_r = _edges_to_F_extra(extras_r_edges, len(g_r.V0))

    # Spec §2 lists step 7 last, but rebuilding the augmented gadgets here lets
    # us thread g_l_aug.F as the column space for SkipTree (step 5). Reordering
    # is safe: F_aug.shape[0] is determined by extra_kappa_*, not by SkipTree.
    from .gadget import build_gadget_augmented
    g_l_aug = build_gadget_augmented(g_l.code, g_l.x, extra_kappa_l, basis=basis)
    g_r_aug = build_gadget_augmented(g_r.code, g_r.x, extra_kappa_r, basis=basis)

    # Step 5: SkipTree on induced port subgraph; embed back into full F_aug rows
    T_l, label_l = _run_skiptree_on_port_subgraph(
        G_l_aux, port_l, spanning_tree_root_l, g_l_aug.F,
    )
    T_r, label_r = _run_skiptree_on_port_subgraph(
        G_r_aux, port_r, spanning_tree_root_r, g_r_aug.F,
    )

    return Bridge(
        width=width,
        basis=basis,
        port_l=port_l,
        port_r=port_r,
        label_l=tuple(label_l),
        label_r=tuple(label_r),
        extra_kappa_l=extra_kappa_l.astype(np.uint8),
        extra_kappa_r=extra_kappa_r.astype(np.uint8),
        T_l=T_l,
        T_r=T_r,
        H_R=_canonical_H_R(width).astype(np.int_),
        g_l_aug=g_l_aug,
        g_r_aug=g_r_aug,
    )
