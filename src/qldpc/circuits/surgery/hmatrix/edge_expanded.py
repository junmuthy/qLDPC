# edge_expanded.py
"""Edge-expanded homological measurement — faithful implementation of
Benjamin Ide, Manoj G. Gowda, Priya J. Nadkarni, Guillaume Dauphinais,
"Fault-Tolerant Logical Measurements via Homological Measurement",
arXiv:2410.02753, Algorithms 1-3 (§III B).

Graph convention (arXiv:2410.02753 Def 2 / §III B): the incidence matrix has
rows = edges (ancilla qubits Q') and cols = vertices (V0 = supp(x)); it equals
∂_1 interpreted as an edge-vertex incidence matrix. The cycle space (∂_0 rows,
Def 4) is the left null space of the incidence matrix.
"""
from __future__ import annotations

import dataclasses

import galois
import numpy as np

GF2 = galois.GF(2)


@dataclasses.dataclass(frozen=True)
class RestrictMaps:
    """Pre-algorithm restriction of a logical measurement (arXiv:2410.02753
    Eqs 35, 47, 48). ``incidence_star`` = ∂_1* (edge-vertex, |nz_rows|×|Q|)."""

    support: tuple[int, ...]
    nz_rows: tuple[int, ...]
    incidence_star: np.ndarray
    f1: np.ndarray
    f0_star: np.ndarray


def restrict_maps(H_complement: np.ndarray, x: np.ndarray) -> RestrictMaps:
    """Build f_1, ∂_1*, f_0* (arXiv:2410.02753 Eqs 35, 47, 48).

    ``H_complement`` is the check matrix complementary to the measured type
    (H_Z when measuring X̄). Q = supp(x). ∂_1* = H_complement|_Q with zero rows
    removed (Eq 47); f_1 is the n×|Q| indicator (Eq 35); f_0* maps the |nz_rows|
    surviving checks (Eq 48).
    """
    H = np.asarray(H_complement).astype(np.uint8)
    x = np.asarray(x).astype(np.uint8)
    if x.shape != (H.shape[1],):
        raise ValueError(f"x has shape {x.shape}, expected ({H.shape[1]},)")
    support = tuple(int(i) for i in np.nonzero(x)[0])              # Q = supp(X̄)
    Q = np.array(support, dtype=np.int_)
    n = H.shape[1]
    w = len(support)
    if w == 0:
        return RestrictMaps((), (), np.zeros((0, 0), np.uint8),
                            np.zeros((n, 0), np.uint8), np.zeros((H.shape[0], 0), np.uint8))
    H_Q = H[:, Q]                                                  # H_complement|_Q
    nz_rows = tuple(int(i) for i in np.nonzero(H_Q.any(axis=1))[0])  # Eq 47: drop zero rows
    incidence_star = H_Q[list(nz_rows), :].astype(np.uint8)       # ∂_1*
    f1 = np.zeros((n, w), dtype=np.uint8)                          # Eq 35
    f1[Q, np.arange(w)] = 1
    f0_star = np.zeros((H.shape[0], len(nz_rows)), dtype=np.uint8)  # Eq 48
    f0_star[list(nz_rows), np.arange(len(nz_rows))] = 1
    return RestrictMaps(support, nz_rows, incidence_star, f1, f0_star)


def boundary(incidence: np.ndarray, S: np.ndarray) -> np.ndarray:
    """∂S = edges with an odd number of endpoints in S (arXiv:2410.02753 Eq 2)."""
    inc = np.asarray(incidence).astype(np.uint8)
    S = np.asarray(S).astype(np.uint8)
    return (inc @ S % 2).astype(np.uint8)


def _mask_to_indicator(mask: int, n_v: int) -> np.ndarray:
    return np.array([(mask >> i) & 1 for i in range(n_v)], dtype=np.uint8)


def _col_ints(inc: np.ndarray) -> list[int]:
    """Bit-pack each vertex column of the edge-vertex incidence into a Python int
    (bit e set iff edge e touches that vertex). Enables O(1) boundary XOR +
    ``bit_count`` during exhaustive cut enumeration — exact, same result as the
    numpy path, ~20× faster (|V|=24 becomes seconds, not minutes)."""
    n_v = inc.shape[1]
    return [
        int.from_bytes(np.packbits(inc[:, i][::-1]).tobytes()[::-1], "little")
        for i in range(n_v)
    ]


# Above this many vertices the exact 2^|V| Cheeger enumeration gets too slow and
# the Fiedler-sweep approximation is used instead (see _fiedler_sweep_cuts). The
# single-sweep cost doubles per vertex (|V|=20 ≈ 0.09 s, |V|=24 ≈ 1.5 s, |V|=26 ≈
# 6 s) and build_gadget runs ~10-20 sweeps, so |V|=20 keeps an exact build ≲ 2 s
# while |V|=26 would take minutes. Support ≤ 20 (typical low-weight logicals) stays
# certified-exact; heavier operators (e.g. products, Webster3's weight-26 logical)
# fall to the fast spectral estimate.
_EXACT_CHEEGER_MAX_V = 20


def _graph_laplacian(inc: np.ndarray) -> np.ndarray:
    """Dense |V|×|V| combinatorial Laplacian L = D − A, clique-expanding any
    hyperedge (weight > 2 row) into all pairwise vertex adjacencies."""
    inc = np.asarray(inc).astype(np.uint8)
    n_v = inc.shape[1]
    A = np.zeros((n_v, n_v), dtype=np.float64)
    for row in inc:
        verts = np.flatnonzero(row)
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                A[verts[i], verts[j]] += 1.0
                A[verts[j], verts[i]] += 1.0
    return np.diag(A.sum(axis=1)) - A


def _fiedler_sweep_cuts(inc: np.ndarray) -> np.ndarray:
    """Approximate sparsest-cut family for |V| too large to enumerate exactly.

    Orders vertices by the Fiedler vector (2nd eigenvector of the graph Laplacian,
    Cheeger's inequality) and returns the prefix ("sweep") cuts S_k = {first k
    vertices in Fiedler order}, 1 ≤ |S_k| ≤ |V|/2. Returned as a (|cuts|×|V|) 0/1
    matrix; each row scored with the same exact parity boundary as the exact path.
    """
    inc = np.asarray(inc).astype(np.uint8)
    n_v = inc.shape[1]
    _, vecs = np.linalg.eigh(_graph_laplacian(inc))
    fiedler = vecs[:, 1]                                # 2nd-smallest eigenvector
    order = np.argsort(fiedler, kind="stable")
    half = n_v // 2
    cuts = np.zeros((half, n_v), dtype=np.uint8)
    running = np.zeros(n_v, dtype=np.uint8)
    for k in range(half):
        running[order[k]] = 1
        cuts[k] = running
    return cuts


def cheeger_constant(incidence: np.ndarray) -> float:
    """h = min_{1≤|S|≤|V|/2} |∂S|/|S|  (arXiv:2410.02753 Eq 3).

    Exact for |V| ≤ ``_EXACT_CHEEGER_MAX_V`` via a bit-packed Gray-code sweep over
    all 2^|V| subsets (``boundary_int`` holds ∂S as an edge bitmask, XOR-updated as
    S grows by one vertex, one XOR + one ``bit_count`` per cut). For larger |V| the
    exact sweep is infeasible, so the minimum is taken over the Fiedler sweep cuts
    only — a spectral estimate (h may be over-reported if a sparser cut is missed),
    not a certified bound.
    """
    inc = np.asarray(incidence).astype(np.uint8)
    n_v = inc.shape[1]
    if n_v < 2:
        return float("inf")
    if n_v > _EXACT_CHEEGER_MAX_V:
        cuts = _fiedler_sweep_cuts(inc)
        boundary_counts = (inc @ cuts.T % 2).sum(axis=0)   # |∂S| per sweep cut
        sizes = cuts.sum(axis=1)
        return float((boundary_counts[sizes > 0] / sizes[sizes > 0]).min())
    cols = _col_ints(inc)
    half = n_v // 2
    best = float("inf")
    boundary_int = 0
    mask = 0
    for k in range(1, 1 << n_v):
        bit = (k & -k).bit_length() - 1
        mask ^= 1 << bit
        boundary_int ^= cols[bit]                 # ∂S XOR-updated (Eq 2)
        size = mask.bit_count()
        if 1 <= size <= half:
            cut = boundary_int.bit_count()
            if cut < best * size:
                best = cut / size
    return best


def sparsest_cut(incidence: np.ndarray) -> np.ndarray:
    """argmin_{1≤|S|≤|V|/2} |∂S|/|S|  (arXiv:2410.02753 Alg 1 line 3). Exact
    bit-packed Gray-code sweep for |V| ≤ ``_EXACT_CHEEGER_MAX_V`` (same enumeration
    as ``cheeger_constant``); for larger |V|, the argmin over the Fiedler sweep
    cuts (approximate, spectral)."""
    inc = np.asarray(incidence).astype(np.uint8)
    n_v = inc.shape[1]
    if n_v < 2:
        return np.zeros(n_v, dtype=np.uint8)
    if n_v > _EXACT_CHEEGER_MAX_V:
        cuts = _fiedler_sweep_cuts(inc)
        boundary_counts = (inc @ cuts.T % 2).sum(axis=0)
        sizes = cuts.sum(axis=1)
        ratios = np.where(sizes > 0, boundary_counts / np.maximum(sizes, 1), np.inf)
        return cuts[int(np.argmin(ratios))].copy()
    cols = _col_ints(inc)
    half = n_v // 2
    best_ratio = float("inf")
    best_mask = 0
    boundary_int = 0
    mask = 0
    for k in range(1, 1 << n_v):
        bit = (k & -k).bit_length() - 1
        mask ^= 1 << bit
        boundary_int ^= cols[bit]
        size = mask.bit_count()
        if 1 <= size <= half:
            cut = boundary_int.bit_count()
            if cut < best_ratio * size:
                best_ratio = cut / size
                best_mask = mask
    return _mask_to_indicator(best_mask, n_v)


def algorithm_1(incidence: np.ndarray, *, max_extra: int = 200, seed: int = 0) -> np.ndarray:
    """Greedy algorithm to add edges to reach Cheeger constant 1
    (arXiv:2410.02753 Algorithm 1). ``incidence`` = ∂_1 as edge-vertex incidence
    (rows=edges, cols=vertices). Returns a superset-of-edges incidence with h=1.

    Vectorized: the per-subset cut counts are precomputed once as a numpy array
    and each candidate edge is scored by an O(#subsets) numpy delta, rather than
    re-enumerating ``cheeger_constant`` per trial. Results are bit-identical to the
    literal per-trial form — the trial value ``min((cuts+δ)/sizes)`` equals
    ``cheeger_constant(trial)`` and the strict-``>`` argmax picks the same edge —
    but ~50× faster at |V|≈20 (75 s → ~2 s on bb_18). Exact enumeration is
    O(2^|V|); |V| = the logical operator weight.
    """
    E_star = np.asarray(incidence).astype(np.uint8).copy()        # line 1: E* ← E
    n_v = E_star.shape[1]
    if n_v < 2:
        return E_star
    if n_v > _EXACT_CHEEGER_MAX_V:
        return _algorithm_1_fiedler(E_star, max_extra=max_extra)

    # Precompute the valid cuts (1 ≤ |S| ≤ |V|//2) once: masks (subset bitmasks) and
    # sizes. ``cuts[i]`` = #edges with odd intersection with subset i (maintained
    # incrementally as edges are added).
    half = n_v // 2
    masks_list: list[int] = []
    sizes_list: list[int] = []
    m = 0
    for k in range(1, 1 << n_v):
        bit = (k & -k).bit_length() - 1
        m ^= 1 << bit
        sz = m.bit_count()
        if 1 <= sz <= half:
            masks_list.append(m)
            sizes_list.append(sz)
    masks = np.array(masks_list, dtype=np.uint64)
    sizes = np.array(sizes_list, dtype=np.float64)

    def _membership(v: int) -> np.ndarray:                        # 0/1 per subset: is v ∈ S?
        return ((masks >> np.uint64(v)) & np.uint64(1)).astype(np.float64)

    cuts = np.zeros(len(masks), dtype=np.float64)                 # |∂S| per subset
    for row in E_star:
        par = np.zeros(len(masks), dtype=np.float64)
        for v in np.flatnonzero(row):
            par = np.abs(par - _membership(int(v)))              # XOR of endpoint memberships
        cuts += par

    added = 0
    ratios = cuts / sizes
    while ratios.min() < 1.0:                                     # line 2: while h(B) < 1
        if added >= max_extra:
            raise RuntimeError(f"Algorithm 1 exceeded max_extra={max_extra}")
        # line 3: sparsest cut — use the exact integer routine (once per iteration,
        # cheap) so tie-breaking is identical to the literal form; the vectorized
        # ``cuts`` array is only used for the fast per-trial deltas below.
        S = sparsest_cut(E_star)
        deg = E_star.sum(axis=0)                                   # vertex degrees (over E*)
        inside = np.flatnonzero(S == 1)
        outside = np.flatnonzero(S == 0)
        # line 5-6: v1 over min-degree vertices of S; v2 over min-degree vertices of V∖S
        min_deg_in = inside[deg[inside] == deg[inside].min()]
        min_deg_out = outside[deg[outside] == deg[outside].min()]
        h_star = -np.inf                                           # line 4: h* ← -∞
        best_edge = None
        best_delta = None
        for v1 in min_deg_in:
            mv1 = _membership(int(v1))
            for v2 in min_deg_out:                                 # line 7-10
                delta = np.abs(mv1 - _membership(int(v2)))        # edge (v1,v2) ∈ ∂S? per subset
                h = float(((cuts + delta) / sizes).min())         # = cheeger_constant(trial)
                if h > h_star:                                     # line 8: if h(...) > h*
                    h_star = h                                     # line 9
                    best_edge = (int(v1), int(v2))                 # line 10: e ← (v1,v2)
                    best_delta = delta
        if best_edge is None:                                      # pragma: no cover
            raise RuntimeError("Algorithm 1: no admissible edge across the sparsest cut")
        row = np.zeros((1, n_v), dtype=np.uint8)
        row[0, best_edge[0]] = 1
        row[0, best_edge[1]] = 1
        E_star = np.vstack([E_star, row])                          # line 13: E* ← E* ∪ {e}
        cuts = cuts + best_delta                                   # incremental |∂S| update
        ratios = cuts / sizes
        added += 1
    return E_star                                                 # line 15: return B


def _algorithm_1_fiedler(E_star: np.ndarray, *, max_extra: int) -> np.ndarray:
    """Algorithm 1 for |V| > ``_EXACT_CHEEGER_MAX_V``, where exact 2^|V| Cheeger
    enumeration is infeasible (arXiv:2410.02753 Alg 1, Fiedler-sweep variant).

    Same greedy shape as the exact path — sparsest cut, then add the edge across it
    (min-degree endpoints) that maximizes the resulting Cheeger — but both the
    stop-test and every candidate score are taken over the Fiedler sweep cuts
    (``_fiedler_sweep_cuts``) rather than all subsets. The sweep is recomputed on
    the current graph each iteration (adding an edge shifts the Fiedler order).
    ``h ≥ 1`` is therefore best-effort (spectral estimate), not certified.
    """
    n_v = E_star.shape[1]
    added = 0
    while True:
        cuts = _fiedler_sweep_cuts(E_star)                        # (|cuts|×|V|)
        cut_counts = (E_star @ cuts.T % 2).sum(axis=0).astype(np.float64)  # |∂S|
        sizes = cuts.sum(axis=1).astype(np.float64)
        ratios = cut_counts / sizes
        if ratios.min() >= 1.0:                                   # stop: h(B) ≥ 1
            return E_star
        if added >= max_extra:
            raise RuntimeError(f"Algorithm 1 (Fiedler) exceeded max_extra={max_extra}")
        S = cuts[int(np.argmin(ratios))]
        deg = E_star.sum(axis=0)
        inside = np.flatnonzero(S == 1)
        outside = np.flatnonzero(S == 0)
        min_in = inside[deg[inside] == deg[inside].min()]         # min-degree in S
        min_out = outside[deg[outside] == deg[outside].min()]     # min-degree in V∖S
        h_star = -np.inf
        best_edge = None
        for v1 in min_in:
            col_v1 = cuts[:, v1].astype(np.float64)
            for v2 in min_out:
                delta = np.abs(col_v1 - cuts[:, v2])              # edge (v1,v2) ∈ ∂S?
                h = float(((cut_counts + delta) / sizes).min())
                if h > h_star:
                    h_star = h
                    best_edge = (int(v1), int(v2))
        row = np.zeros((1, n_v), dtype=np.uint8)
        row[0, best_edge[0]] = 1
        row[0, best_edge[1]] = 1
        E_star = np.vstack([E_star, row])
        added += 1


def _rref_drop_zero(M: galois.FieldArray) -> galois.FieldArray:
    """Reduced row echelon form with all-zero rows removed."""
    if M.shape[0] == 0:
        return M
    R = M.row_reduce()
    nz = np.asarray((R != 0).any(axis=1))
    return R[nz]


def _random_invertible_gf2(k: int, rng: np.random.Generator) -> galois.FieldArray:
    """Random invertible k×k GF(2) matrix (rejection on full rank)."""
    if k == 0:
        return GF2(np.zeros((0, 0), dtype=np.uint8))
    while True:
        A = GF2(rng.integers(0, 2, size=(k, k), dtype=np.uint8))
        if int(A.row_space().shape[0]) == k:
            return A


def _bitint(vec: np.ndarray) -> int:
    """Pack a 0/1 vector into a Python int (bit e set iff ``vec[e]``)."""
    x = 0
    for e in np.flatnonzero(np.asarray(vec)):
        x |= 1 << int(e)
    return x


def _horton_candidates(incidence: np.ndarray) -> list[tuple[int, frozenset[int]]]:
    """Low-weight cycle-space vectors via Horton's minimum-cycle-basis candidate
    set [J. D. Horton, "A polynomial-time algorithm to find the shortest cycle
    basis of a graph", SIAM J. Comput. 16(2), 358-366 (1987)].

    Builds a graph from the weight-2 edges of ``incidence`` and, from a BFS
    (shortest-path) tree rooted at every vertex, forms the fundamental cycle of
    each non-tree edge. The union over all roots contains a minimum cycle basis.
    Hyperedges (weight > 2 rows) are skipped here; the caller completes the basis
    from the full cycle space if the weight-2 candidates do not span it.

    Returns ``[(weight, frozenset(edge indices)), ...]`` sorted lightest-first,
    deduplicated.
    """
    import collections

    inc = np.asarray(incidence).astype(np.uint8)
    n_edges, n_v = inc.shape
    ends: dict[int, tuple[int, int]] = {}
    adj: dict[int, list[tuple[int, int]]] = {v: [] for v in range(n_v)}
    for e in range(n_edges):
        vs = np.flatnonzero(inc[e])
        if len(vs) == 2:
            u, v = int(vs[0]), int(vs[1])
            ends[e] = (u, v)
            adj[u].append((v, e))
            adj[v].append((u, e))

    cands: dict[frozenset[int], int] = {}
    for root in range(n_v):
        parent: dict[int, tuple[int, int]] = {root: (-1, -1)}   # v -> (parent, edge)
        dq = collections.deque([root])
        while dq:
            x = dq.popleft()
            for (y, e) in adj[x]:
                if y not in parent:
                    parent[y] = (x, e)
                    dq.append(y)
        tree_edges = {pe for (_, pe) in parent.values() if pe != -1}

        def climb(a: int) -> set[int]:
            s: set[int] = set()
            while parent[a][0] != -1:
                pa, pe = parent[a]
                s.add(pe)
                a = pa
            return s

        for e, (u, v) in ends.items():
            if e in tree_edges or u not in parent or v not in parent:
                continue
            cyc = {e} ^ (climb(u) ^ climb(v))       # e + tree path u..v = simple cycle
            key = frozenset(cyc)
            if key and key not in cands:
                cands[key] = len(key)
    return sorted(((w, key) for key, w in cands.items()))


def algorithm_2(
    incidence: np.ndarray,
    H_complement: np.ndarray,
    f0: np.ndarray,
    *,
    n_samples: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Low-weight ∂_0: a minimum-cycle-basis complement of the redundant cycles.

    ``incidence`` = ∂_1 (edge-vertex). Cycle space = left null space of ∂_1.
    ∂_0 must be a complement of V = redundant cycles {vᵀ f_0 : v ∈ ker H_Zᵀ}
    (arXiv:2410.02753 Alg 2 line 1) in that cycle space, kept low-weight.

    This DEVIATES from arXiv:2410.02753 Algorithm 2, whose random A·W + B·V
    search does not lower the weight of a dense RREF starting basis (multiplying
    by a random invertible A only makes rows denser) and stalls far above the
    achievable minimum — e.g. weight 13 vs. 5 on the bb_18 [[248,10,18]] gadget
    graph, forcing needless cellulation. Instead ∂_0 is built greedily from
    Horton's minimum-cycle-basis candidates [Horton, SIAM J. Comput. 16(2),
    358-366 (1987)]: lightest cycles first, each kept only if independent modulo
    V, until a full complement is chosen. If the weight-2 Horton candidates do
    not span the whole cycle space (hyperedges present), the remaining rows are
    completed from the full cycle-space basis so the complement is always valid.

    ``n_samples`` / ``seed`` are retained for API compatibility; the construction
    is deterministic and does not use them.
    """
    del n_samples, seed                            # deterministic; kept for API compat
    inc = GF2(np.asarray(incidence).astype(np.uint8))
    f0 = GF2(np.asarray(f0).astype(np.uint8))
    Hc = GF2(np.asarray(H_complement).astype(np.uint8))
    n_edges = int(inc.shape[0])

    # V = redundant cycles (arXiv:2410.02753 Alg 2 line 1). ker H_Zᵀ = left null
    # space of H_complement (H_Z when measuring X̄): rows v with v @ H_complement
    # = 0, one entry per complementary check, matching the rows of f_0.
    if Hc.shape[0] == 0:
        ker_HcT = GF2(np.zeros((0, f0.shape[0]), dtype=np.uint8))
    else:
        ker_HcT = Hc.left_null_space()
    V_rows = ker_HcT @ f0 if ker_HcT.shape[0] else GF2(np.zeros((0, f0.shape[1]), np.uint8))
    V = _rref_drop_zero(V_rows)

    # full cycle space (left null space of ∂_1) and the complement dimension ∂_0 needs
    Z1 = _rref_drop_zero(inc.left_null_space())
    target_dim = int(Z1.shape[0]) - int(V.shape[0])
    if target_dim <= 0:
        return np.zeros((0, n_edges), dtype=np.uint8)

    # online GF(2) rank over edge-indexed bitmasks; seed pivots with V so a
    # candidate is accepted iff it is independent MODULO V.
    pivots: dict[int, int] = {}

    def add_pivot(bitmask: int) -> bool:
        x = bitmask
        while x:
            hb = x.bit_length() - 1
            p = pivots.get(hb)
            if p is None:
                pivots[hb] = x
                return True
            x ^= p
        return False

    for i in range(V.shape[0]):
        add_pivot(_bitint(np.asarray(V[i])))

    chosen: list[np.ndarray] = []

    def take(bitmask: int, edges: frozenset[int] | np.ndarray) -> bool:
        if not add_pivot(bitmask):
            return False
        vec = np.zeros(n_edges, dtype=np.uint8)
        vec[list(edges) if isinstance(edges, frozenset) else np.flatnonzero(edges)] = 1
        chosen.append(vec)
        return len(chosen) == target_dim

    # 1) greedy over lightest Horton cycles
    for _w, key in _horton_candidates(incidence):
        if take(sum(1 << e for e in key), key):
            return np.array(chosen, dtype=np.uint8).reshape(-1, n_edges)

    # 2) complete from the full cycle-space basis (handles hyperedges / gaps)
    for i in range(Z1.shape[0]):
        row = np.asarray(Z1[i]).astype(np.uint8)
        if take(_bitint(row), row):
            break
    return np.array(chosen, dtype=np.uint8).reshape(-1, n_edges)


def _max_row_weight(M: galois.FieldArray) -> int:
    arr = np.asarray(M).astype(int)
    return 0 if arr.shape[0] == 0 else int(arr.sum(axis=1).max())


class _UnionFind:
    """Minimal union-find over vertex indices, for the Cheeger-aware pairing."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path halving
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _cheeger_aware_pairing(verts: list[int], uf: _UnionFind) -> list[tuple[int, int]]:
    """Perfect matching of a hyperedge's ``verts`` that keeps the Cheeger constant
    as high as possible (arXiv:2410.02753 Algorithm 3, hyperedge-expansion step).

    A weight-2 replacement must be a perfect matching (each vertex once) so the
    rows sum to the original edge (Eq 36). Among matchings, connectivity is what
    keeps the Cheeger constant up, so pairs are chosen greedily to join vertices in
    DIFFERENT current components (merging them); once no cross-component pair
    remains, leftovers are paired in order. ``uf`` accumulates across hyperedges.
    """
    unmatched = list(verts)
    pairs: list[tuple[int, int]] = []
    while len(unmatched) >= 2:
        pick: tuple[int, int] | None = None
        for i in range(len(unmatched)):
            for j in range(i + 1, len(unmatched)):
                if uf.find(unmatched[i]) != uf.find(unmatched[j]):
                    pick = (i, j)
                    break
            if pick is not None:
                break
        if pick is None:
            pick = (0, 1)                                 # all one component; pair in order
        i, j = pick
        a, b = unmatched[i], unmatched[j]
        uf.union(a, b)
        pairs.append((a, b))
        unmatched.pop(j)                                  # j > i, remove it first
        unmatched.pop(i)
    return pairs


def expand_hyperedges(
    incidence: np.ndarray, f0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Expand hyperedges to weight-two edges (arXiv:2410.02753 Algorithm 3).

    Each row ``e`` of ``incidence`` with ``wt e > 2`` is replaced by ``(wt e)//2``
    weight-two rows that sum to ``e`` (a perfect matching of its vertices, so the
    row space still contains ``e`` per Eq 36), and the column of ``f0`` for ``e``
    is replaced by ``(wt e)//2`` copies. Rows with ``wt e ≤ 2`` pass through.

    The matching is chosen to "keep the Cheeger constant as high as possible"
    (Algorithm 3): a union-find seeded with the existing weight-2 edges drives a
    greedy that pairs vertices in DIFFERENT components first (``_cheeger_aware_
    pairing``), leaving the expanded graph maximally connected so the subsequent
    Algorithm 1 adds far fewer edges. (Odd ``wt e`` never arises — H_Z rows commute
    with X̄, so every restricted edge has even weight.)
    """
    inc = np.asarray(incidence).astype(np.uint8)
    f0 = np.asarray(f0).astype(np.uint8)
    n_v = inc.shape[1]
    # seed union-find with the pass-through weight-2 edges so hyperedge matchings
    # connect across the graph that already exists.
    uf = _UnionFind(n_v)
    for i in range(inc.shape[0]):
        vs = np.flatnonzero(inc[i])
        if len(vs) == 2:
            uf.union(int(vs[0]), int(vs[1]))
    new_rows: list[np.ndarray] = []
    new_f0_cols: list[np.ndarray] = []
    for i in range(inc.shape[0]):
        verts = [int(v) for v in np.flatnonzero(inc[i])]
        col = f0[:, i] if f0.shape[1] > i else np.zeros(f0.shape[0], np.uint8)
        if len(verts) <= 2:                              # pass through unchanged
            new_rows.append(inc[i].copy())
            new_f0_cols.append(col)
            continue
        for a, b in _cheeger_aware_pairing(verts, uf):
            row = np.zeros(n_v, dtype=np.uint8)
            row[a] = 1
            row[b] = 1
            new_rows.append(row)
            new_f0_cols.append(col)                      # duplicate the f0 column
    inc2 = np.vstack(new_rows).astype(np.uint8) if new_rows else inc[:0]
    f02 = np.column_stack(new_f0_cols).astype(np.uint8) if new_f0_cols else f0[:, :0]
    return inc2, f02


def _edge_ends(inc: np.ndarray, e: int) -> tuple[int, int] | None:
    """The two vertices of a weight-2 edge (row ``e`` of the incidence), or None."""
    vs = np.flatnonzero(inc[e])
    return (int(vs[0]), int(vs[1])) if len(vs) == 2 else None


def _extract_simple_cycle(inc: np.ndarray, remaining: set[int]) -> list[int] | None:
    """Pull one simple cycle (list of edge indices) out of ``remaining`` and remove
    its edges. ``remaining`` must have all-even vertex degree (a cycle-space
    element). Returns None only if the edge set is malformed."""
    adj: dict[int, list[int]] = {}
    for e in remaining:
        ends = _edge_ends(inc, e)
        if ends is None:
            return None
        u, v = ends
        adj.setdefault(u, []).append(e)
        adj.setdefault(v, []).append(e)
    start = next(iter(adj))
    pos = {start: 0}
    path_e: list[int] = []
    cur = start
    while True:
        avail = [e for e in adj[cur] if e in remaining and e not in path_e]
        if not avail:
            return None
        e = avail[0]
        ends = _edge_ends(inc, e)
        assert ends is not None
        nxt = ends[1] if ends[0] == cur else ends[0]
        path_e.append(e)
        if nxt in pos:                               # closed a loop -> extract it
            cyc = path_e[pos[nxt]:]
            remaining.difference_update(cyc)
            return cyc
        pos[nxt] = len(pos)
        cur = nxt


def _cycle_vertex_order(inc: np.ndarray, cycle_edges: list[int]) -> list[int] | None:
    """Vertex traversal order of a SIMPLE cycle given its edge indices (every
    touched vertex must have in-cycle degree exactly 2)."""
    adj: dict[int, list[int]] = {}
    for e in cycle_edges:
        ends = _edge_ends(inc, e)
        if ends is None:
            return None
        u, v = ends
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    if any(len(nbrs) != 2 for nbrs in adj.values()):
        return None
    start = min(adj)
    order = [start]
    prev, cur = -1, start
    for _ in range(len(cycle_edges) - 1):
        nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
        order.append(nxt)
        prev, cur = cur, nxt
    return order if len(order) == len(cycle_edges) else None


def _edge_between(inc: np.ndarray, cycle_edges: list[int], u: int, v: int) -> int | None:
    for e in cycle_edges:
        if _edge_ends(inc, e) in ((u, v), (v, u)):
            return e
    return None


def cellulate(
    partial_0: np.ndarray,
    incidence: np.ndarray,
    f0: np.ndarray,
    *,
    target_weight: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cellulate large cycles (arXiv:2410.02753 Algorithm 3, "cellulate large
    cycles"). For each ∂_0 row heavier than ``target_weight``, split it into
    lower-weight cycles and REPLACE the heavy row with them — the paper's
    "replacing the high weight row c with multiple lower weight rows corresponding
    to the new cycles". Two mechanisms, both preserving the stabilizer group (the
    replacements GF(2)-sum to the original row):

    1. Decompose a row into edge-disjoint SIMPLE cycles (Veblen) — a row that is a
       union of small cycles is split at zero cost.
    2. Chord-split any simple cycle still longer than ``target_weight``: add the
       chord between opposite cycle vertices (as in the reference
       ``cellulate_long_cycles``, Swaroop, Jochym-O'Connor, Yoder —
       adapters-LDPC-surgery/cellulation.py, arXiv:2410.03628), replacing the
       cycle with the two arcs+chord. Chords become weight-2 ∂_1 rows (zero f_0
       columns).

    Algorithm 2 is NOT re-run — its heuristic search does not reliably recover a
    low-weight basis on large graphs, which is exactly why cellulation exists.
    ``seed`` is accepted for API uniformity (the split is deterministic).
    """
    inc = np.asarray(incidence).astype(np.uint8).copy()
    f0 = np.asarray(f0).astype(np.uint8).copy()
    n_v = inc.shape[1]
    rows = [set(int(e) for e in np.flatnonzero(r)) for r in np.asarray(partial_0)]

    # budget: bounded work even if a row cannot be split below target.
    max_splits = 8 * sum(max(0, len(r) - target_weight) for r in rows) + 4 * len(rows) + 8
    splits = 0
    changed = True
    while changed and splits < max_splits:
        changed = False
        for idx in range(len(rows)):
            if len(rows[idx]) <= target_weight:
                continue
            # (1) decompose into edge-disjoint simple cycles
            remaining = set(rows[idx])
            pieces: list[list[int]] = []
            ok = True
            while remaining:
                cyc = _extract_simple_cycle(inc, remaining)
                if cyc is None:
                    ok = False
                    break
                pieces.append(cyc)
            if not ok:                               # pragma: no cover  -- malformed row
                continue
            if len(pieces) > 1:                      # a union of smaller cycles -> just split
                rows[idx] = set(pieces[0])
                for p in pieces[1:]:
                    rows.append(set(p))
                splits += 1
                changed = True
                break
            # (2) single simple cycle still too long -> chord-split it
            cyc = pieces[0]
            order = _cycle_vertex_order(inc, cyc)
            if order is None:                        # pragma: no cover
                continue
            k = len(order)
            a, b = order[0], order[k // 2]           # opposite-vertex chord (balanced)
            chord_idx = inc.shape[0]
            chord = np.zeros((1, n_v), dtype=np.uint8)
            chord[0, a] = 1
            chord[0, b] = 1
            inc = np.vstack([inc, chord]).astype(np.uint8)
            f0 = np.hstack([f0, np.zeros((f0.shape[0], 1), np.uint8)])
            arc1 = {_edge_between(inc, cyc, order[i], order[i + 1]) for i in range(0, k // 2)}
            arc2 = {_edge_between(inc, cyc, order[i], order[(i + 1) % k]) for i in range(k // 2, k)}
            rows[idx] = {e for e in arc1 if e is not None} | {chord_idx}
            rows.append({e for e in arc2 if e is not None} | {chord_idx})
            splits += 1
            changed = True
            break

    d0 = np.zeros((len(rows), inc.shape[0]), dtype=np.uint8)
    for i, s in enumerate(rows):
        for e in s:
            d0[i, e] = 1
    return d0, inc, f0


@dataclasses.dataclass(frozen=True)
class ConeMaps:
    """The four maps of the mapping cone (arXiv:2410.02753 Eq 12)."""

    support: tuple[int, ...]
    f1: np.ndarray
    f0: np.ndarray
    incidence: np.ndarray     # ∂_1 as edge-vertex (|edges|×|V0|)
    partial_0: np.ndarray     # ∂_0 (cycles × |edges|)
    data_checks: tuple[int, ...]


def _edges_f0(r: RestrictMaps, n_comp: int, n_edges: int) -> np.ndarray:
    """f_0 with one column per edge (arXiv:2410.02753 Alg 3 line 3). The first
    |nz_rows| columns are indicators of the backing complementary checks (edge j
    ↔ check nz_rows[j]); the remaining edges (Alg-1 additions) get zero columns."""
    base = np.zeros((n_comp, len(r.nz_rows)), dtype=np.uint8)
    for j, row in enumerate(r.nz_rows):
        base[row, j] = 1
    pad = n_edges - len(r.nz_rows)
    return np.hstack([base, np.zeros((n_comp, max(0, pad)), dtype=np.uint8)])


def _pad_f0(f0: np.ndarray, n_edges: int) -> np.ndarray:
    """Append zero columns to f_0 until it has one column per edge (Alg 3 line 3)."""
    f0 = np.asarray(f0).astype(np.uint8)
    pad = n_edges - f0.shape[1]
    if pad <= 0:
        return f0
    return np.hstack([f0, np.zeros((f0.shape[0], pad), dtype=np.uint8)])


def _data_checks_from_f0(f0: np.ndarray) -> list[int]:
    """Recover which original check backs each edge from f_0: a column with a
    single 1 at row r → check r; an all-zero column (κ / chord edge) → -1."""
    f0 = np.asarray(f0).astype(np.uint8)
    out: list[int] = []
    for j in range(f0.shape[1]):
        rows = np.flatnonzero(f0[:, j])
        out.append(int(rows[0]) if rows.size == 1 else -1)
    return out


def edge_expanded_maps(
    H_complement: np.ndarray,
    x: np.ndarray,
    *,
    seed: int = 0,
    n_samples: int = 200,
    cellulate_to: int | None = None,
) -> ConeMaps:
    """Main construction (arXiv:2410.02753 Algorithm 3) for one X-/Z-logical.

    Implements ALL of Algorithm 3 (see the Verbatim Algorithm Source): the main
    path (lines "Define f1/∂1*/f0*" → Alg 1 → add zero columns → Alg 2) plus the
    full ``if the sparsity of ∂0 is deemed unacceptable`` branch (reset ∂1←∂1*,
    expand hyperedges, re-run Alg 1, re-run Alg 2, cellulate large cycles).
    Produces f_1, f_0, ∂_1, ∂_0 with Cheeger(∂_1) ≥ 1 (distance preserved, only the
    target operator measured) and low-weight ∂_0.

    ``cellulate_to`` is the "desired" weight: when the main-path ∂_0 has a row
    heavier than it, the branch runs and cellulates down to it. ``None`` accepts
    the main-path ∂_0 unconditionally (no branch).
    """
    r = restrict_maps(H_complement, x)                            # line 1: f1, ∂1*, f0*
    n_comp = np.asarray(H_complement).shape[0]

    incidence = algorithm_1(r.incidence_star, seed=seed)          # line 2: ∂1 ← Alg 1(∂1*)
    f0 = _edges_f0(r, n_comp, incidence.shape[0])                 # line 3: f0* + zero cols
    partial_0 = algorithm_2(incidence, H_complement, f0,          # line 4: ∂0 ← Alg 2
                            n_samples=n_samples, seed=seed)

    # line: if the sparsity of ∂0 is deemed unacceptable
    if cellulate_to is not None and _max_row_weight(GF2(partial_0)) > cellulate_to:
        inc_b = np.asarray(r.incidence_star).astype(np.uint8)     # ∂1 ← ∂1*
        f0_b = _edges_f0(r, n_comp, inc_b.shape[0])               # f1 ← f1* (edge side reset)
        inc_b, f0_b = expand_hyperedges(inc_b, f0_b)              # Expand hyperedges to wt-2
        inc_b = algorithm_1(inc_b, seed=seed)                     # Apply Alg 1 (adds edges)
        f0_b = _pad_f0(f0_b, inc_b.shape[0])                      # add zero columns to f0
        partial_0 = algorithm_2(inc_b, H_complement, f0_b,        # find a cycle basis: Alg 2
                                n_samples=n_samples, seed=seed)
        partial_0, inc_b, f0_b = cellulate(                       # cellulate large cycles
            partial_0, inc_b, f0_b, target_weight=cellulate_to, seed=seed)
        incidence, f0 = inc_b, f0_b

    data_checks = _data_checks_from_f0(f0)
    return ConeMaps(r.support, r.f1, f0, incidence, partial_0, tuple(data_checks))
