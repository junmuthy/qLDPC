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


def _all_cuts(n_v: int):
    """Yield (subset_mask, size) for 1 ≤ size ≤ n_v//2 via Gray-code order."""
    half = n_v // 2
    mask = 0
    for k in range(1, 1 << n_v):
        bit = (k & -k).bit_length() - 1
        mask ^= 1 << bit
        size = mask.bit_count()
        if 1 <= size <= half:
            yield mask, size


def _mask_to_indicator(mask: int, n_v: int) -> np.ndarray:
    return np.array([(mask >> i) & 1 for i in range(n_v)], dtype=np.uint8)


def cheeger_constant(incidence: np.ndarray) -> float:
    """h = min_{1≤|S|≤|V|/2} |∂S|/|S|  (arXiv:2410.02753 Eq 3), exact enumeration."""
    inc = np.asarray(incidence).astype(np.uint8)
    n_v = inc.shape[1]
    if n_v < 2:
        return float("inf")
    best = float("inf")
    for mask, size in _all_cuts(n_v):
        S = _mask_to_indicator(mask, n_v)
        cut = int(boundary(inc, S).sum())
        if cut < best * size:
            best = cut / size
    return best


def sparsest_cut(incidence: np.ndarray) -> np.ndarray:
    """argmin_{1≤|S|≤|V|/2} |∂S|/|S|  (arXiv:2410.02753 Alg 1 line 3)."""
    inc = np.asarray(incidence).astype(np.uint8)
    n_v = inc.shape[1]
    best_ratio = float("inf")
    best_mask = 0
    for mask, size in _all_cuts(n_v):
        S = _mask_to_indicator(mask, n_v)
        cut = int(boundary(inc, S).sum())
        if cut < best_ratio * size:
            best_ratio = cut / size
            best_mask = mask
    return _mask_to_indicator(best_mask, n_v)


def algorithm_1(incidence: np.ndarray, *, max_extra: int = 200, seed: int = 0) -> np.ndarray:
    """Greedy algorithm to add edges to reach Cheeger constant 1
    (arXiv:2410.02753 Algorithm 1). ``incidence`` = ∂_1 as edge-vertex incidence
    (rows=edges, cols=vertices). Returns a superset-of-edges incidence with h=1.
    """
    rng = np.random.default_rng(seed)
    E_star = np.asarray(incidence).astype(np.uint8).copy()        # line 1: E* ← E
    n_v = E_star.shape[1]
    if n_v < 2:
        return E_star
    added = 0
    while cheeger_constant(E_star) < 1.0:                         # line 2: while h(B)<1
        if added >= max_extra:
            raise RuntimeError(f"Algorithm 1 exceeded max_extra={max_extra}")
        S = sparsest_cut(E_star)                                   # line 3: sparsest cut
        deg = E_star.sum(axis=0)                                   # vertex degrees (over E*)
        inside = np.flatnonzero(S == 1)
        outside = np.flatnonzero(S == 0)
        # line 5-6: v1 over min-degree vertices of S; v2 over min-degree vertices of V∖S
        min_deg_in = inside[deg[inside] == deg[inside].min()]
        min_deg_out = outside[deg[outside] == deg[outside].min()]
        h_star = -np.inf                                           # line 4: h* ← -∞
        best_edge = None
        for v1 in min_deg_in:
            for v2 in min_deg_out:                                 # line 7-10
                trial_row = np.zeros((1, n_v), dtype=np.uint8)
                trial_row[0, v1] = 1
                trial_row[0, v2] = 1
                trial = np.vstack([E_star, trial_row])
                h = cheeger_constant(trial)
                if h > h_star:                                     # line 8: if h(...) > h*
                    h_star = h                                     # line 9
                    best_edge = (int(v1), int(v2))                 # line 10: e ← (v1,v2)
        if best_edge is None:                                      # pragma: no cover
            raise RuntimeError("Algorithm 1: no admissible edge across the sparsest cut")
        row = np.zeros((1, n_v), dtype=np.uint8)
        row[0, best_edge[0]] = 1
        row[0, best_edge[1]] = 1
        E_star = np.vstack([E_star, row])                          # line 13: E* ← E* ∪ {e}
        added += 1
    return E_star                                                 # line 15: return B
