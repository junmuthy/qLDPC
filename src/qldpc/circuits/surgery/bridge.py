"""Standalone bridge adapter for two-PPM joint surgery
(Swaroop et al. arXiv:2410.03628 §IV / §VII).

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
    """Universal adapter between two GadgetLayouts.

    Implements the universal adapter construction of Swaroop et al.
    (Swaroop, Jochym-O'Connor, Yoder) arXiv:2410.03628 §III.

    For each side s ∈ {l, r}, the SkipTree (Swaroop et al. arXiv:2410.03628 §III)
    produces a matrix T_s and a permutation P_{σ_s} acting on the port 𝒫_s.
    The port-label block π_{𝒫_s}^T P_{σ_s} selects and reorders the port columns
    of the augmented incidence matrix H_X'^{s,aug}, satisfying the SkipTree identity:

        T_s (H_X'^{s,aug})^T π_{𝒫_s}^T = H_R P_{σ_s}^T

    where H_R is the canonical repetition-code parity-check matrix stored in
    ``Bridge.H_R``.  This identity guarantees CSS commutation between the adapter
    ancillas and the merged-code gauge generators.

    Cheeger-distance preservation (same-basis) is guaranteed by
    Cross et al. arXiv:2407.18393 Thm 6.  Port-subgraph cellulation (to cap
    basis cycle length) follows Williamson & Yoder arXiv:2410.02213.

    Same-basis fields (``width``, ``port_l/r``, ``label_l/r``, ``T_l/r``,
    ``H_R``, ``g_l/r_aug``) are populated for both same-basis and mixed-basis
    bridges.  Mixed-basis fields (``Y_stab``, ``merge_qubits``, ``obs0_xor_map``,
    ``x_leftover_indices``, ``z_leftover_indices``) implement the
    Webster–Smith–Cohen arXiv:2511.15989 §II.B.2 cross-merge for joint
    Pauli-product measurement of different-basis logicals (e.g. Z̄_l ⊗ X̄_r).
    They default to None / () for same-basis bridges and are populated only by
    the mixed-basis dispatch path in ``_stitch_to_joint_code``.
    """

    width: int  # w = |𝒜| (adapter qubits)
    basis_l: PauliXZ  # X or Z on the left gadget (Webster–Smith–Cohen mixed-basis)
    basis_r: PauliXZ  # X or Z on the right gadget
    port_l: tuple[int, ...]  # 𝒫_l* ⊆ V_0^(l), length w
    port_r: tuple[int, ...]  # 𝒫_r* ⊆ V_0^(r), length w
    label_l: tuple[int, ...]  # label_l[i] = SkipTree label of V_0^(l)[i]; -1 if i ∉ 𝒫_l*
    label_r: tuple[int, ...]
    extra_ancilla_l: np.ndarray  # (e_l, |support^(l)|) F_2; weight-2 rows added
    extra_ancilla_r: np.ndarray
    T_l: np.ndarray  # (w-1, |C_0^(l)| + e_l) F_2 (3,2)-sparse
    T_r: np.ndarray
    H_R: np.ndarray  # (w-1, w) canonical rep code parity
    g_l_aug: GadgetLayout  # gadget rebuilt over F_aug^(l)
    g_r_aug: GadgetLayout
    # Mixed-basis fields (Webster–Smith–Cohen arXiv:2511.15989 §II.B.2).
    # None / empty for same-basis bridges.
    Y_stab: np.ndarray | None = None  # (n_Y, 2*n_merged) symplectic Y-rows
    merge_qubits: tuple[int, ...] = ()  # bridge qubit indices touched by cross-merge
    obs0_xor_map: tuple[int, ...] = ()  # Y_stab row indices XORed into obs0
    x_leftover_indices: tuple[int, ...] = ()  # X-cycle row indices not cross-merged
    z_leftover_indices: tuple[int, ...] = ()  # Z-cycle row indices not cross-merged

    @property
    def basis(self) -> PauliXZ:
        """Backward-compat single-basis accessor.

        Returns the shared basis when basis_l == basis_r. Raises AttributeError
        for mixed-basis bridges — callers must explicitly use basis_l / basis_r.
        """
        if self.basis_l is not self.basis_r:
            raise AttributeError(
                "mixed-basis Bridge has no single .basis attribute; "
                f"use bridge.basis_l ({self.basis_l!r}) / bridge.basis_r ({self.basis_r!r})"
            )
        return self.basis_l


def _skip_tree(
    S: nx.Graph,
    root: int = 0,
    edge_index_verts: dict[tuple[int, int], int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """SkipTree basis transform (Swaroop et al. (Swaroop, Jochym-O'Connor, Yoder) arXiv:2410.03628 §III).

    Returns (T, P) where T has shape (n-1, |E|) and P is an n×n permutation
    matrix encoding the SkipTree vertex labeling σ.  Together they satisfy the
    SkipTree identity:

        T · G · P = H_R

    where G is the graph incidence matrix and H_R is the canonical rep-code
    parity-check matrix.  The port-label block π_{𝒫_s}^T P_{σ_s} selects
    port columns of the augmented incidence H_X'^{s,aug} and reorders them to
    match H_R's column convention.
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
    # P is a permutation matrix from _skip_tree; recover label[ell] = v such that P[v, ell] = 1.
    label = [-1] * n
    for v in range(n):
        for ell in range(n):
            if P[v, ell] == 1:
                label[ell] = v
                break

    if edge_index_verts is None:
        edge_index_verts = {tuple(sorted(e)): i for i, e in enumerate(S.edges())}

    T = np.zeros((n - 1, len(edge_index_verts)), dtype=np.int_)
    for l_idx in range(n - 1):
        path = nx.shortest_path(S, source=label[l_idx], target=label[l_idx + 1])
        for u, v in zip(path[:-1], path[1:]):
            e = tuple(sorted((u, v)))
            T[l_idx, edge_index_verts[e]] ^= 1  # XOR cancels back-and-forth
    return T.astype(np.int_), P.astype(np.int_)


def _cellulate_port_subgraph(
    G_aux: nx.Graph,
    ports: tuple[int, ...],
    *,
    max_len: int = 6,
) -> list[tuple[int, int]]:
    """Break port-subgraph cycles longer than ``max_len`` by adding chords.

    SkipTree runs on G_aux.subgraph(ports); cycles entirely outside the
    port subgraph never enter T_s, so we cellulate only there.

    Theorem 7 (Swaroop et al. arXiv:2410.03628) already bounds T_s row weight at ≤ 3
    regardless of cycle length, so this step is not load-bearing for
    correctness — it tightens the structural distance argument (Theorem 12)
    by capping basis cycle lengths.

    The full-graph version (the previous _cellulate_strict) failed spuriously
    on real BB codes when |V_0| > w and a long cycle threaded through non-port
    vertices: no available port-port chord existed despite the port subgraph
    being fine on its own.

    Chords are added to ``G_aux`` (the full graph). For port-subgraph cycles,
    chord endpoints are necessarily ports (cycle vertices = port vertices),
    so no port-membership filter is needed in the chord-search loop.

    Returns the list of added (u, v) edges in insertion order. Idempotent
    once all port-subgraph basis cycles fit under the cap.
    """
    added: list[tuple[int, int]] = []
    while True:
        sub = G_aux.subgraph(ports)
        long_cycles = [c for c in nx.cycle_basis(sub) if len(c) > max_len]
        if not long_cycles:
            return added
        cycle = long_cycles[0]
        n = len(cycle)
        chord_found = False
        for i in range(n):
            if chord_found:
                break
            for j in range(i + 2, n):
                u, v = sorted((cycle[i], cycle[j]))
                if G_aux.has_edge(u, v):
                    continue
                G_aux.add_edge(u, v)
                added.append((u, v))
                chord_found = True
                break
        if not chord_found:
            raise RuntimeError(
                f"No chord found to cellulate port-subgraph cycle of length {n}; "
                f"ports={ports!r}, cycle={cycle!r}"
            )


def _build_aux_graph_strict(incidence: np.ndarray) -> tuple[nx.Graph, dict[tuple[int, int], int]]:
    """Build auxiliary graph from F; weight-2 rows become edges, hyperedges are skipped.

    Vertices: range(|V_0|) = range(F.shape[1]).
    Edges: one per weight-2 row of F, between the two columns where the row has 1s.

    Why skipping hyperedges is safe — paired with the guard at
    `_run_skiptree_on_port_subgraph` (search for `if len(cols) != 2: continue`),
    which assigns T_s zero columns on those same rows. So
        (T_s · F_aug)[c, v] = Σ_{k: weight-2 row} T_s[c,k] · F_aug[k, v]
                            = H_R[c, label(v)] · [v ∈ port]      (SkipTree identity)
    and the hyperedge rows contribute 0 regardless of F_aug[r, v]. χ_v · cycle_c
    on the κ side cancels the adapter side, CSS commutation holds. The hyperedge
    κ qubit itself stays in F_aug, so the gadget (G_aug = ker(F_aug^T), deformed
    check c → c · X(κ_r), χ_v) is untouched. Paper Eq. 9's perfect-matching
    decomposition (§II.C) is not applied; structural Theorem 12 distance
    argument is replaced by empirical LER smoke tests.

    Raises:
        ValueError: if any row of F has weight 1 (defensive — F · 1_{V_0} = 0
        mod 2 forbids odd weights for any valid logical x with H · x = 0).
    """
    incidence_arr = np.asarray(incidence).astype(int)
    G = nx.Graph()
    G.add_nodes_from(range(incidence_arr.shape[1]))
    edge_index: dict[tuple[int, int], int] = {}
    for i, row in enumerate(incidence_arr):
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


def _edges_to_incidence_extra(edges: list[tuple[int, int]], n_V0: int) -> np.ndarray:
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
    incidence_aug: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """Run SkipTree on the induced port subgraph; embed result back onto F_aug rows.

    Implements the port-subgraph restriction step of Swaroop et al.
    (Swaroop, Jochym-O'Connor, Yoder) arXiv:2410.03628 §III.

    The induced subgraph's vertex IDs are relabeled to [0, |port|) so the n×n P
    allocation inside ``_skip_tree`` is square.  The output T is then
    re-expressed onto the original F_aug edge ordering (rows of F_aug index the
    κ qubits = edges of G_aux_full).  ``root_port_idx`` selects which entry of
    ``port`` is the SkipTree root.

    The result satisfies the SkipTree identity on the port subgraph:

        T_s (H_X'^{s,aug})^T π_{𝒫_s}^T = H_R P_{σ_s}^T

    where π_{𝒫_s} is the port-selection matrix (rows = port vertices of
    H_X'^{s,aug}) and P_{σ_s} is the permutation returned by ``_skip_tree``
    encoding the SkipTree label assignment σ_s.

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
        sub_tree,
        root=root_relab,
        edge_index_verts=edge_idx_tree,
    )
    labels = [-1] * incidence_aug.shape[1]
    for new_v in range(len(port)):
        orig_v = orig_of_new[new_v]
        nz = np.flatnonzero(P_relab[new_v])
        assert len(nz) == 1, f"vertex {orig_v} (relab {new_v}) has {len(nz)} labels"
        labels[orig_v] = int(nz[0])
    T_full = np.zeros((T_relab.shape[0], incidence_aug.shape[0]), dtype=np.int_)
    # Duplicate-edge guard: when two κ rows of F_aug share the same (u, v)
    # support (parallel edges in the *strict* aux graph), _build_aux_graph_strict
    # dedups them to one G_aux edge. Without this guard, both duplicate rows
    # would receive the same T_relab column → their contributions to T·F_aug
    # cancel mod 2 → SkipTree identity fails on codes like BB [[36, 8]] whose
    # restricted incidence has duplicate weight-2 rows. Assigning T only to the
    # FIRST matching row preserves T·F_aug = H_R (duplicate κ qubits remain in
    # the gauge group, untouched by the cycle).
    assigned_edges: set[tuple[int, int]] = set()
    for r in range(incidence_aug.shape[0]):
        cols = np.flatnonzero(incidence_aug[r])
        # Load-bearing skip: T_s gets zero columns on hyperedge rows (weight ≥ 3)
        # and on rows whose endpoints are outside the port subgraph. Paired with
        # _build_aux_graph_strict's matching skip; together they make hyperedge κ
        # qubits invisible to (T_s · F_aug), so χ_v · cycle_c commutation reduces
        # to the weight-2 sub-incidence SkipTree identity. See bridge.py docstring
        # for _build_aux_graph_strict for the proof sketch.
        if len(cols) != 2:
            continue
        u_orig, v_orig = sorted(int(x) for x in cols)
        if u_orig not in new_of_orig or v_orig not in new_of_orig:
            continue
        e_relab = tuple(sorted((new_of_orig[u_orig], new_of_orig[v_orig])))
        if e_relab in edge_idx_tree and e_relab not in assigned_edges:
            T_full[:, r] = T_relab[:, edge_idx_tree[e_relab]]
            assigned_edges.add(e_relab)
    return T_full.astype(np.int_), labels


def _max_basis_stabilizer_weight(code, basis: PauliXZ) -> int:
    """Max row weight of the underlying code's stabilizers on the gadget's basis side."""
    from qldpc.objects import Pauli

    H = code.matrix_z if basis is Pauli.Z else code.matrix_x
    return int(np.asarray(H).astype(int).sum(axis=1).max())


def build_bridge(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    *,
    port_subset_l: tuple[int, ...] | None = None,
    port_subset_r: tuple[int, ...] | None = None,
    spanning_tree_root_l: int = 0,
    spanning_tree_root_r: int = 0,
    cellulate_max_len: int | None = None,
) -> Bridge:
    """Build a universal-adapter bridge between two gadgets.

    Implements the construction of Swaroop et al. (Swaroop, Jochym-O'Connor,
    Yoder) arXiv:2410.03628 §III.  For each side s ∈ {l, r}:

      1. Build the auxiliary graph G_aux^(s) from weight-2 rows of F^(s)
         (the gadget incidence matrix, notation from arXiv:2410.03628 §III).
      2. Augment G_aux^(s) so that the induced subgraph on port 𝒫_s is
         connected (connectivity edges → extra_ancilla_s).
      3. Cellulate the port subgraph to cap basis cycle length, following
         Williamson & Yoder arXiv:2410.02213.
         Cheeger-distance preservation is guaranteed by Cross et al.
         arXiv:2407.18393 Thm 6.
      4. Run SkipTree (Swaroop et al. arXiv:2410.03628 §III) on the spanning
         tree of the port subgraph, yielding T_s and permutation P_{σ_s}.
         The port-label block π_{𝒫_s}^T P_{σ_s} selects and reorders port
         columns of H_X'^{s,aug}, satisfying the SkipTree identity:

             T_s (H_X'^{s,aug})^T π_{𝒫_s}^T = H_R P_{σ_s}^T

      5. Rebuild augmented gadgets g_l_aug / g_r_aug from the combined extras.

    ``spanning_tree_root_s`` is the index INTO the port tuple of the SkipTree
    root vertex on side s.

    When ``g_l.basis == g_r.basis``, returns a same-basis CSS bridge.
    When ``g_l.basis != g_r.basis``, returns a mixed-basis bridge (basis_l ≠
    basis_r) with mixed-basis fields (Y_stab, obs0_xor_map, ...) left
    UNPOPULATED — the Webster–Smith–Cohen (arXiv:2511.15989 §II.B.2)
    cross-merge populates them during ``_stitch_to_joint_code``.

    ``cellulate_max_len`` caps port-subgraph basis cycle length.  When ``None``
    (default), it is set to the maximum basis-side stabilizer row weight across
    both sides.
    """
    basis_l = g_l.basis
    basis_r = g_r.basis
    # Cellulation cap defaults to the worse-case basis-stabilizer weight across
    # both sides, irrespective of mixed-basis. The cap shapes the rep-code cycle
    # length in the merged code and is basis-agnostic for the structural distance
    # argument (Swaroop Theorem 12). Use each side's native basis.
    if cellulate_max_len is None:
        cellulate_max_len = max(
            _max_basis_stabilizer_weight(g_l.code, basis_l),
            _max_basis_stabilizer_weight(g_r.code, basis_r),
        )

    # Step 1: auxiliary graphs
    G_l_aux, _ = _build_aux_graph_strict(g_l.incidence)
    G_r_aux, _ = _build_aux_graph_strict(g_r.incidence)

    # Step 2: port subsets + width
    port_l_all = (
        tuple(port_subset_l) if port_subset_l is not None else tuple(range(len(g_l.support)))
    )
    port_r_all = (
        tuple(port_subset_r) if port_subset_r is not None else tuple(range(len(g_r.support)))
    )
    width = min(len(port_l_all), len(port_r_all))
    if width < 2:
        raise ValueError(f"bridge width must be >= 2, got {width}")
    port_l = port_l_all[:width]
    port_r = port_r_all[:width]
    if not (0 <= spanning_tree_root_l < width):
        raise ValueError(f"spanning_tree_root_l={spanning_tree_root_l} out of [0, {width})")
    if not (0 <= spanning_tree_root_r < width):
        raise ValueError(f"spanning_tree_root_r={spanning_tree_root_r} out of [0, {width})")

    # Step 3: induced-subgraph connectivity augmentation
    extras_l_conn = _connect_induced_subgraph(G_l_aux, port_l)
    extras_r_conn = _connect_induced_subgraph(G_r_aux, port_r)

    # Step 4: cellulation
    extras_l_cell = _cellulate_port_subgraph(G_l_aux, port_l, max_len=cellulate_max_len)
    extras_r_cell = _cellulate_port_subgraph(G_r_aux, port_r, max_len=cellulate_max_len)

    extras_l_edges = extras_l_conn + extras_l_cell
    extras_r_edges = extras_r_conn + extras_r_cell
    extra_ancilla_l = _edges_to_incidence_extra(extras_l_edges, len(g_l.support))
    extra_ancilla_r = _edges_to_incidence_extra(extras_r_edges, len(g_r.support))

    from .gadget import _step1_restriction, build_gadget_augmented

    # boost_gadget appends weight-2 κ' rows to g_l.incidence beyond the original
    # _step1_restriction output. These rows must be preserved when assembling
    # g_l_aug — SkipTree runs against G_aux (built from boosted g_l.incidence)
    # but T_full is embedded into g_l_aug.incidence; dropping boost rows leaves
    # tree edges through boost-κ' silently zeroed and breaks the invariant
    # T_s · F_aug · P_s = H_R.
    _, _, _orig_inc_l = _step1_restriction(g_l.code, g_l.x, basis=basis_l)
    _, _, _orig_inc_r = _step1_restriction(g_r.code, g_r.x, basis=basis_r)
    boost_extras_l = g_l.incidence[_orig_inc_l.shape[0] :].astype(np.uint8)
    boost_extras_r = g_r.incidence[_orig_inc_r.shape[0] :].astype(np.uint8)
    combined_extras_l = np.vstack([boost_extras_l, extra_ancilla_l.astype(np.uint8)])
    combined_extras_r = np.vstack([boost_extras_r, extra_ancilla_r.astype(np.uint8)])

    g_l_aug = build_gadget_augmented(g_l.code, g_l.x, combined_extras_l, basis=basis_l)
    g_r_aug = build_gadget_augmented(g_r.code, g_r.x, combined_extras_r, basis=basis_r)

    # Step 5: SkipTree on induced port subgraph; embed back into full F_aug rows
    T_l, label_l = _run_skiptree_on_port_subgraph(
        G_l_aux,
        port_l,
        spanning_tree_root_l,
        g_l_aug.incidence,
    )
    T_r, label_r = _run_skiptree_on_port_subgraph(
        G_r_aux,
        port_r,
        spanning_tree_root_r,
        g_r_aug.incidence,
    )

    return Bridge(
        width=width,
        basis_l=basis_l,
        basis_r=basis_r,
        port_l=port_l,
        port_r=port_r,
        label_l=tuple(label_l),
        label_r=tuple(label_r),
        extra_ancilla_l=extra_ancilla_l.astype(np.uint8),
        extra_ancilla_r=extra_ancilla_r.astype(np.uint8),
        T_l=T_l,
        T_r=T_r,
        H_R=_canonical_H_R(width).astype(np.int_),
        g_l_aug=g_l_aug,
        g_r_aug=g_r_aug,
        # Mixed-basis fields: populated by _stitch_to_joint_code when basis_l != basis_r.
        # Left as defaults (None / ()) here for both same-basis and mixed-basis bridges.
    )
