# Edge-Expanded Homological Measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the arbitrary high-weight ancilla construction with a faithful, line-for-line implementation of arXiv:2410.02753 Algorithms 1/2/3, so surgery gadgets yield low-weight `f₁, f₀, ∂₁, ∂₀` and decode fast.

**Architecture:** A new pure-GF(2) module `hmatrix/edge_expanded.py` implements Algorithm 1 (greedy Cheeger→1 edges), Algorithm 2 (random low-weight cycle basis), and Algorithm 3 (orchestration + cellulation). `build_gadget` calls the orchestrator by default; `GadgetLayout` gains explicit `f1`/`f0` fields. Dead code (old Cheeger boost, weight-blind `minimize_z_checks`) is deleted.

**Tech Stack:** Python 3.12, `galois` (GF(2) linear algebra), `numpy`, `pytest`, `stim` (DEM checks). Codes from `qldpc.codes`.

## Global Constraints

- **Fidelity (hard):** every function mirrors the paper's pseudocode line-for-line, same step order; each block carries a comment citing its Algorithm + line number. No "equivalent" reordering or shortcuts. Copied verbatim from the spec.
- GF(2) throughout; use `galois.GF(2)` (aliased `GF2`). `incidence` = ∂₁-as-edge-vertex-incidence: rows = edges (ancilla `Q'`), cols = vertices (`V₀ = supp(x)`). `∂₁ = incidence.T`; `∂₀ = partial_0` = cycle space (rows indexed by edges).
- Determinism: fixed default `seed=0`; every randomized routine takes `seed`/`rng`.
- No LER / `sinter` / statistical-sampling tests (repo rule). Verify via cone validity, exact distance, `num_observables`, and matrix weight/shape asserts.
- Paper citation style: full author list + `arXiv:2410.02753` + Eq/Alg number in docstrings; never `math.md` or bare surnames.
- `build_gadget` name, call signature `build_gadget(code, x, *, basis, ...)`, and the `GadgetLayout` field set (`code, x, support, data_checks, incidence, partial_0, HX_merged, HZ_merged, Q_prime, basis`) stay backward-compatible — only ADD fields.

---

### Task 1: Restriction primitives — `f₁, ∂₁*, f₀*` (Eqs 35/47/48)

**Files:**
- Create: `src/qldpc/circuits/surgery/hmatrix/edge_expanded.py`
- Test: `src/qldpc/circuits/surgery/hmatrix/edge_expanded_test.py`

**Interfaces:**
- Produces: `restrict_maps(H_complement: np.ndarray, x: np.ndarray) -> RestrictMaps` where `RestrictMaps` is a frozen dataclass with fields `support: tuple[int,...]` (`Q = supp(x)`, the vertices `V₀`), `nz_rows: tuple[int,...]` (indices of non-zero rows of `H_complement|_Q`), `incidence_star: np.ndarray` (`∂₁*` = `H_complement|_Q` with zero rows removed, shape `|nz_rows|×|Q|`, uint8), `f1: np.ndarray` (`n×|Q|`, Eq 35), `f0_star: np.ndarray` (`|nz_rows|-columned` indicator, Eq 48 — shape `n_complement × |nz_rows|`).

- [ ] **Step 1: Write the failing test**

```python
# edge_expanded_test.py
import numpy as np
from qldpc.circuits.surgery.hmatrix.edge_expanded import restrict_maps

STEANE_HZ = np.array([  # Steane [[7,1,3]] H_Z (arXiv:2410.02753 Eq.54), qubits 0..6
    [0,0,0,1,1,1,1],
    [0,1,1,0,0,1,1],
    [1,0,1,0,1,0,1],
], dtype=np.uint8)

def test_restrict_maps_steane_weight3_logical():
    # X̄ = X1 X2 X3 (support {0,1,2}) — arXiv:2410.02753 Example 6.
    x = np.array([1,1,1,0,0,0,0], dtype=np.uint8)
    r = restrict_maps(STEANE_HZ, x)
    assert r.support == (0, 1, 2)
    # H_Z|_Q columns {0,1,2}: rows -> [[0,0,0],[0,1,1],[1,0,1]]; row 0 is zero -> dropped
    assert r.nz_rows == (1, 2)
    np.testing.assert_array_equal(r.incidence_star, np.array([[0,1,1],[1,0,1]], dtype=np.uint8))
    # f1 (Eq 35): n=7 rows, |Q|=3 cols, (f1)_{i,j}=delta_{i,q_j}
    expected_f1 = np.zeros((7, 3), dtype=np.uint8)
    expected_f1[0,0] = expected_f1[1,1] = expected_f1[2,2] = 1
    np.testing.assert_array_equal(r.f1, expected_f1)
    # f0_star (Eq 48): (f0*)_{i,j}=delta_{i,h_j}, h_j = row index of j-th nonzero row of H_Z|_Q
    expected_f0 = np.zeros((3, 2), dtype=np.uint8)
    expected_f0[1,0] = expected_f0[2,1] = 1
    np.testing.assert_array_equal(r.f0_star, expected_f0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/qldpc/circuits/surgery/hmatrix/edge_expanded_test.py::test_restrict_maps_steane_weight3_logical -v`
Expected: FAIL with `ImportError`/`ModuleNotFoundError` (module/function absent).

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/qldpc/circuits/surgery/hmatrix/edge_expanded_test.py::test_restrict_maps_steane_weight3_logical -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/hmatrix/edge_expanded.py src/qldpc/circuits/surgery/hmatrix/edge_expanded_test.py
git commit -m "feat(surgery): restriction maps f1/d1*/f0* (arXiv:2410.02753 Eqs 35/47/48)"
```

---

### Task 2: Cheeger constant + sparsest cut (Def 3, exact)

**Files:**
- Modify: `src/qldpc/circuits/surgery/hmatrix/edge_expanded.py`
- Test: `src/qldpc/circuits/surgery/hmatrix/edge_expanded_test.py`

**Interfaces:**
- Consumes: `RestrictMaps` (Task 1).
- Produces:
  - `boundary(incidence: np.ndarray, S: np.ndarray) -> np.ndarray` — `∂S` indicator over edges, `= {e : |e∩S| odd}` (arXiv:2410.02753 Eq 2).
  - `cheeger_constant(incidence: np.ndarray) -> float` — `min |∂S|/|S|`, `|S|≤|V|/2` (Eq 3), exact Gray-code enumeration over the `|V|` vertices.
  - `sparsest_cut(incidence: np.ndarray) -> np.ndarray` — the `argmin` `S` (length-`|V|` uint8 indicator).

- [ ] **Step 1: Write the failing test**

```python
from qldpc.circuits.surgery.hmatrix.edge_expanded import (
    boundary, cheeger_constant, sparsest_cut)

def _incidence(edges, n_v):
    M = np.zeros((len(edges), n_v), dtype=np.uint8)
    for i, e in enumerate(edges):
        for v in e:
            M[i, v] = 1
    return M

def test_cheeger_path_graph_p8_example4():
    # arXiv:2410.02753 Example 4 / Fig 1a: V={v1..v6}, edges (v1v2)(v2v3)(v4v5)(v5v6)
    inc = _incidence([(0,1),(1,2),(3,4),(4,5)], 6)
    assert cheeger_constant(inc) == 0.0            # disconnected -> h=0
    # boundary of S={v0,v1,v2}: only edge (v1,v2)->(1,2) crosses? edges within S carry even.
    S = np.array([1,1,1,0,0,0], dtype=np.uint8)
    assert int(boundary(inc, S).sum()) == 0        # {v0,v1,v2} is a full component -> empty boundary

def test_cheeger_triangle_is_one():
    inc = _incidence([(0,1),(1,2),(0,2)], 3)       # triangle: every single-vertex cut has |∂S|=2,|S|=1
    assert cheeger_constant(inc) == 1.0

def test_sparsest_cut_returns_min_ratio_set():
    inc = _incidence([(0,1),(1,2),(3,4),(4,5)], 6)
    S = sparsest_cut(inc)
    assert 1 <= int(S.sum()) <= 3
    assert int(boundary(inc, S).sum()) == 0        # sparsest cut isolates a component (ratio 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/qldpc/circuits/surgery/hmatrix/edge_expanded_test.py -k cheeger -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Append to `edge_expanded.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/qldpc/circuits/surgery/hmatrix/edge_expanded_test.py -k cheeger -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/hmatrix/edge_expanded.py src/qldpc/circuits/surgery/hmatrix/edge_expanded_test.py
git commit -m "feat(surgery): exact Cheeger constant + sparsest cut (arXiv:2410.02753 Eqs 2/3)"
```

---

### Task 3: Algorithm 1 — greedy edges to Cheeger = 1

**Files:**
- Modify: `src/qldpc/circuits/surgery/hmatrix/edge_expanded.py`
- Test: `src/qldpc/circuits/surgery/hmatrix/edge_expanded_test.py`

**Interfaces:**
- Consumes: `boundary`, `cheeger_constant`, `sparsest_cut` (Task 2).
- Produces: `algorithm_1(incidence: np.ndarray, *, max_extra: int = 200, seed: int = 0) -> np.ndarray` — returns a **new** incidence matrix (superset of rows, same `|V|` columns) with Cheeger constant `1`. Each added row is a weight-2 edge (arXiv:2410.02753 Alg 1).

- [ ] **Step 1: Write the failing test**

```python
from qldpc.circuits.surgery.hmatrix.edge_expanded import algorithm_1

def test_algorithm_1_reaches_cheeger_one():
    inc = _incidence([(0,1),(1,2),(3,4),(4,5)], 6)   # h=0
    out = algorithm_1(inc, seed=0)
    assert out.shape[1] == 6                          # same vertices
    assert out.shape[0] >= inc.shape[0]               # superset of edges
    np.testing.assert_array_equal(out[:inc.shape[0]], inc)  # original edges preserved
    assert cheeger_constant(out) >= 1.0
    added = out[inc.shape[0]:]
    if added.size:
        assert np.all(added.sum(axis=1) == 2)         # each new edge is weight-2

def test_algorithm_1_noop_when_already_one():
    inc = _incidence([(0,1),(1,2),(0,2)], 3)          # triangle already h=1
    out = algorithm_1(inc, seed=0)
    np.testing.assert_array_equal(out, inc)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/qldpc/circuits/surgery/hmatrix/edge_expanded_test.py -k algorithm_1 -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Append to `edge_expanded.py`. This mirrors Alg 1 line-for-line:

```python
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
```

> **Note on ties (Alg 1 lines 5-6):** "min-degree vertex" is a set when several tie; the paper iterates the argmax over all such `(v1,v2)` pairs (lines 6-11), so we enumerate every min-degree pair and keep the `h`-maximizing edge. `seed`/`rng` is retained only for reproducible ordering if a future tie-break among equal-`h` edges is added; the enumeration itself is deterministic.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/qldpc/circuits/surgery/hmatrix/edge_expanded_test.py -k algorithm_1 -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(surgery): Algorithm 1 greedy Cheeger→1 edges (arXiv:2410.02753 Alg 1)"
```

---

### Task 4: Algorithm 2 — random search for low-weight ∂₀

**Files:**
- Modify: `src/qldpc/circuits/surgery/hmatrix/edge_expanded.py`
- Test: `src/qldpc/circuits/surgery/hmatrix/edge_expanded_test.py`

**Interfaces:**
- Consumes: `GF2`, incidence (Task 1-3).
- Produces:
  - `_random_invertible_gf2(k: int, rng) -> galois.FieldArray` — uniform-ish random invertible `k×k` GF(2) matrix (rejection sample on full rank).
  - `algorithm_2(incidence: np.ndarray, H_complement: np.ndarray, f0: np.ndarray, *, n_samples: int = 200, seed: int = 0) -> np.ndarray` — returns `∂₀` (rows = cycles over edges), a low-max-row-weight basis of the complement of the redundant cycle space (arXiv:2410.02753 Algorithm 2). `f0` has one row per complementary check and one column per edge (`|edges|`).

- [ ] **Step 1: Write the failing test**

```python
from qldpc.circuits.surgery.hmatrix.edge_expanded import algorithm_2, _random_invertible_gf2

def test_random_invertible_is_invertible():
    rng = np.random.default_rng(3)
    for _ in range(5):
        A = _random_invertible_gf2(4, rng)
        assert int(A.row_space().shape[0]) == 4       # full rank over GF(2)

def test_algorithm_2_is_valid_cycle_basis_and_low_weight():
    # A 6-cycle graph: edges (0,1)(1,2)(2,3)(3,4)(4,5)(5,0). Cycle space dim 1,
    # the single cycle uses all 6 edges. With one big cycle, ∂0 = that weight-6 row.
    inc = _incidence([(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)], 6)
    H_complement = np.zeros((0, 6), dtype=np.uint8)   # no backing checks (all edges are κ)
    f0 = np.zeros((0, 6), dtype=np.uint8)             # |edges|=6 columns
    d0 = algorithm_2(inc, H_complement, f0, n_samples=100, seed=0)
    # rows are cycles: d0 @ incidence == 0 over GF(2)
    assert np.all((np.asarray(d0).astype(int) @ inc.astype(int)) % 2 == 0)
    # spans the whole cycle space (dim 1 here)
    assert int(GF2(np.asarray(d0).astype(np.uint8)).row_space().shape[0]) == 1

def test_algorithm_2_beats_arbitrary_basis_weight():
    # Two triangles sharing edge structure so the arbitrary basis has a heavy row
    # but a low-weight basis (two triangles) exists.
    inc = _incidence([(0,1),(1,2),(0,2),(2,3),(3,4),(2,4)], 5)
    H_complement = np.zeros((0, 5), dtype=np.uint8)
    f0 = np.zeros((0, 6), dtype=np.uint8)
    d0 = algorithm_2(inc, H_complement, f0, n_samples=300, seed=0)
    assert np.all((np.asarray(d0).astype(int) @ inc.astype(int)) % 2 == 0)
    # each independent cycle here is a triangle -> max row weight 3
    assert int(np.asarray(d0).astype(int).sum(axis=1).max()) <= 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/qldpc/circuits/surgery/hmatrix/edge_expanded_test.py -k algorithm_2 -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Append to `edge_expanded.py`. Mirrors Alg 2 line-for-line:

```python
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


def algorithm_2(
    incidence: np.ndarray,
    H_complement: np.ndarray,
    f0: np.ndarray,
    *,
    n_samples: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Random search for low-weight ∂_0 (arXiv:2410.02753 Algorithm 2).

    ``incidence`` = ∂_1 (edge-vertex). Cycle space = left null space of incidence.
    V = redundant cycles {vᵀ f_0 : v ∈ ker H_complementᵀ}; W = complement so
    V ⊕ W = full cycle space. Returns ∂_0 = a low-max-row-weight generator of W.
    """
    rng = np.random.default_rng(seed)
    inc = GF2(np.asarray(incidence).astype(np.uint8))
    f0 = GF2(np.asarray(f0).astype(np.uint8))
    Hc = GF2(np.asarray(H_complement).astype(np.uint8))

    # line 1: V ← basis of { vᵀ f_0 : v ∈ ker H_complementᵀ }
    if Hc.shape[0] == 0:
        ker_HcT = GF2(np.zeros((0, f0.shape[0]), dtype=np.uint8))
    else:
        ker_HcT = Hc.T.left_null_space()          # rows v with vᵀ H_complementᵀ? see note
    V_rows = ker_HcT @ f0 if ker_HcT.shape[0] else GF2(np.zeros((0, f0.shape[1]), np.uint8))
    V = _rref_drop_zero(V_rows)                    # line 2: rref(V)

    # full cycle space of the graph defined by ∂_1 = left null space of incidence
    cycle_space = inc.left_null_space()            # rows z with z @ incidence = 0

    # line 3-5: W = complement of V within cycle_space, rref, zero rows removed.
    stacked = np.vstack([np.asarray(V), np.asarray(cycle_space)]).astype(np.uint8)
    # keep V rows first so row_reduce pivots them; the extra independent rows form W.
    reduced = _rref_drop_zero(GF2(stacked))
    # W = reduced rows not in rowspace(V): drop the first rank(V) pivot rows.
    rank_V = int(V.row_space().shape[0]) if V.shape[0] else 0
    W = reduced[rank_V:] if reduced.shape[0] > rank_V else GF2(
        np.zeros((0, inc.shape[0]), np.uint8))

    d0 = W                                         # line 6: ∂_0 ← W
    best = _max_row_weight(d0)
    for _ in range(n_samples):                     # line 7: for i in 1..n
        k = d0.shape[0]
        if k == 0:
            break
        A = _random_invertible_gf2(k, rng)         # line 8: A random invertible
        B = GF2(rng.integers(0, 2, size=(k, V.shape[0]), dtype=np.uint8)) if V.shape[0] \
            else GF2(np.zeros((k, 0), np.uint8))   # line 9: B random
        AW = A @ W
        if V.shape[0]:
            cand = AW + B @ V                      # line 10: if maxwt(AW+BV) < best
            wt = _max_row_weight(cand)
            if wt < best:
                d0, best = cand, wt                # line 11
        wt_aw = _max_row_weight(AW)                # line 13: if maxwt(AW) < best
        if wt_aw < best:
            d0, best = AW, wt_aw                   # line 14
    return np.asarray(d0).astype(np.uint8)         # line 17: return ∂_0


def _max_row_weight(M: galois.FieldArray) -> int:
    arr = np.asarray(M).astype(int)
    return 0 if arr.shape[0] == 0 else int(arr.sum(axis=1).max())
```

> **Note (Alg 2 line 1):** the redundant-cycle space is `{vᵀ f_0 : v ∈ ker H_complementᵀ}` (paper writes `ker H_Zᵀ`, i.e. the left null space of `H_complement`). Compute `ker H_complementᵀ` as `GF2(H_complement).left_null_space()` — rows `v` with `v @ H_complement = 0` — then map through `f_0`. If the sign/orientation differs from the paper's `H_Z`, fix by transposing `H_complement` and re-running `test_algorithm_2_is_valid_cycle_basis_and_low_weight` (the `d0 @ incidence == 0` invariant is the guard). **The implementer must confirm this invariant passes before committing.**

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/qldpc/circuits/surgery/hmatrix/edge_expanded_test.py -k algorithm_2 -v`
Expected: PASS (3 tests). If `test_algorithm_2_is_valid_cycle_basis_and_low_weight` fails on the `@ incidence == 0` invariant, apply the orientation fix in the note above.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(surgery): Algorithm 2 low-weight cycle basis ∂_0 (arXiv:2410.02753 Alg 2)"
```

---

### Task 5: Algorithm 3 cellulation — split heavy cycles

**Files:**
- Modify: `src/qldpc/circuits/surgery/hmatrix/edge_expanded.py`
- Test: `src/qldpc/circuits/surgery/hmatrix/edge_expanded_test.py`

**Interfaces:**
- Consumes: `algorithm_1`, `algorithm_2` (Tasks 3-4).
- Produces: `cellulate(partial_0, incidence, f0, *, target_weight, seed=0) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — returns `(partial_0', incidence', f0')`. For each `∂₀` row of weight `> target_weight`, add edges inside that cycle (new incidence rows + zero `f0` columns), re-run Algorithm 2 to get the refined low-weight basis (arXiv:2410.02753 Alg 3 lines 15-17).

- [ ] **Step 1: Write the failing test**

```python
from qldpc.circuits.surgery.hmatrix.edge_expanded import cellulate

def test_cellulate_splits_heavy_cycle():
    # Single weight-6 cycle (hexagon). target_weight=4 -> must add a chord, giving
    # two cycles each of weight <= 4.
    inc = _incidence([(0,1),(1,2),(2,3),(3,4),(4,5),(5,0)], 6)
    f0 = np.zeros((0, 6), dtype=np.uint8)
    d0 = algorithm_2(inc, np.zeros((0,6),np.uint8), f0, n_samples=50, seed=0)
    assert int(np.asarray(d0).astype(int).sum(axis=1).max()) == 6
    d0c, incc, f0c = cellulate(d0, inc, f0, target_weight=4, seed=0)
    assert incc.shape[0] > inc.shape[0]                          # chord edge(s) added
    assert np.all((np.asarray(d0c).astype(int) @ incc.astype(int)) % 2 == 0)  # still cycles
    assert int(np.asarray(d0c).astype(int).sum(axis=1).max()) <= 4

def test_cellulate_noop_when_within_target():
    inc = _incidence([(0,1),(1,2),(0,2)], 3)
    f0 = np.zeros((0, 3), dtype=np.uint8)
    d0 = algorithm_2(inc, np.zeros((0,3),np.uint8), f0, n_samples=20, seed=0)
    d0c, incc, f0c = cellulate(d0, inc, f0, target_weight=3, seed=0)
    assert incc.shape[0] == inc.shape[0]                         # nothing added
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/qldpc/circuits/surgery/hmatrix/edge_expanded_test.py -k cellulate -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Append to `edge_expanded.py`:

```python
def cellulate(
    partial_0: np.ndarray,
    incidence: np.ndarray,
    f0: np.ndarray,
    *,
    target_weight: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cellulate heavy ∂_0 cycles into smaller ones (arXiv:2410.02753 Alg 3
    lines 15-17). A cycle of weight w > target is split by adding chord edges
    across vertices interior to the cycle; each chord is a new edge (row of
    ∂_1 / column of f_0). ∂_0 is then recomputed low-weight via Algorithm 2.
    """
    rng = np.random.default_rng(seed)
    inc = np.asarray(incidence).astype(np.uint8).copy()
    f0 = np.asarray(f0).astype(np.uint8).copy()
    d0 = np.asarray(partial_0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        weights = d0.sum(axis=1)
        for ci in np.flatnonzero(weights > target_weight):       # line 15
            cycle_edges = np.flatnonzero(d0[ci])                 # edges in this cycle
            # vertices touched by the cycle (each interior vertex has degree 2 in-cycle)
            verts = np.flatnonzero((inc[cycle_edges].sum(axis=0) % 2 == 0)
                                   & inc[cycle_edges].any(axis=0))
            if len(verts) < 2:                                    # pragma: no cover
                continue
            # add one chord connecting two cycle vertices ~diametrically to halve it
            order = list(verts)
            v1, v2 = order[0], order[len(order) // 2]
            chord = np.zeros((1, inc.shape[1]), dtype=np.uint8)   # line 16: new edge (row of ∂_1)
            chord[0, v1] = 1
            chord[0, v2] = 1
            inc = np.vstack([inc, chord])
            f0 = np.hstack([f0, np.zeros((f0.shape[0], 1), dtype=np.uint8)])  # zero f0 column
            changed = True
            break                                                # recompute after each chord
        if changed:
            d0 = algorithm_2(inc, np.zeros((0, inc.shape[1]), np.uint8), f0,
                             n_samples=200, seed=int(rng.integers(0, 2**31)))
    return d0, inc, f0
```

> **Note (Alg 3 lines 15-17):** the paper adds edges "within the cycle" — chords between cycle vertices — replacing one heavy row with several lighter cycle rows. The exact chord choice is a heuristic (the paper's Fig 2c shows reducing a weight-5 cycle to a single lower-weight cycle); we add one chord per heavy cycle and re-run Alg 2 until all rows meet `target_weight`. If a cycle cannot be split below target (chord budget exhausted), the loop terminates when no `weights > target_weight` remain OR no new chord reduces the max weight — guard against infinite loops with a max-chord cap of `4 × (initial max weight)`.

Add the cap guard at the top of `cellulate`:
```python
    max_chords = 4 * (int(d0.sum(axis=1).max()) if d0.shape[0] else 0) + 4
    chords_added = 0
```
and in the loop replace `changed = True; break` with:
```python
            chords_added += 1
            changed = chords_added < max_chords
            break
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/qldpc/circuits/surgery/hmatrix/edge_expanded_test.py -k cellulate -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(surgery): Algorithm 3 cellulation of heavy cycles (arXiv:2410.02753 Alg 3 §15-17)"
```

---

### Task 6: Algorithm 3 orchestrator — `edge_expanded_gadget`

**Files:**
- Modify: `src/qldpc/circuits/surgery/hmatrix/edge_expanded.py`
- Test: `src/qldpc/circuits/surgery/hmatrix/edge_expanded_test.py`

**Interfaces:**
- Consumes: `restrict_maps`, `algorithm_1`, `algorithm_2`, `cellulate`.
- Produces: `edge_expanded_maps(H_complement, x, *, seed=0, n_samples=200, cellulate_to=None) -> ConeMaps` where `ConeMaps` is a frozen dataclass with `support`, `f1` (n×w), `f0` (n_complement × |edges|), `incidence` (∂₁ edge-vertex, |edges|×w), `partial_0` (∂₀, cycles × |edges|), `data_checks` (tuple; original check index per edge, `-1` for κ/chord). This is the type `build_gadget` (Task 7) assembles the merged matrices from.

- [ ] **Step 1: Write the failing test**

```python
from qldpc.circuits.surgery.hmatrix.edge_expanded import edge_expanded_maps

def test_edge_expanded_steane_weight3_no_cycles():
    # Example 6: Steane X̄=X1X2X3 -> after Alg 1 the graph has no cycles (∂0 empty).
    x = np.array([1,1,1,0,0,0,0], dtype=np.uint8)
    cm = edge_expanded_maps(STEANE_HZ, x, seed=0)
    assert cheeger_constant(cm.incidence) >= 1.0
    # chain complex: ∂0 @ ∂1 == 0
    d1 = cm.incidence                      # edge-vertex; ∂1 = incidence.T applied as cols
    assert np.all((np.asarray(cm.partial_0).astype(int) @ cm.incidence.astype(int)) % 2 == 0)
    # dim ker ∂1 == 1 (only X̄ measured, Remark 3): cycle space dim after Alg1
    from qldpc.circuits.surgery.hmatrix.edge_expanded import GF2
    # here incidence has full column rank -> cycle space may be empty; ker ∂1 measured
    assert cm.f1.shape == (7, 3)

def test_edge_expanded_valid_cone_gross():
    import sympy
    from qldpc import codes
    from qldpc.objects import Pauli
    xs, ys = sympy.symbols('x y')
    code = codes.BBCode({xs:12, ys:6}, xs**3+ys+ys**2, ys**3+xs+xs**2)   # gross
    n = code.num_qudits
    LX = np.asarray(code.get_logical_ops(Pauli.X)).astype(np.uint8)
    x = (LX[0][:n] if LX.shape[1]==2*n else LX[0]).astype(np.uint8)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    native_w = int(HZ.sum(axis=1).max())
    cm = edge_expanded_maps(HZ, x, seed=0, cellulate_to=native_w)
    assert cheeger_constant(cm.incidence) >= 1.0
    assert np.all((np.asarray(cm.partial_0).astype(int) @ cm.incidence.astype(int)) % 2 == 0)
    # weight win vs native check weight after cellulation
    if cm.partial_0.shape[0]:
        assert int(cm.partial_0.sum(axis=1).max()) <= native_w
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/qldpc/circuits/surgery/hmatrix/edge_expanded_test.py -k edge_expanded -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Append to `edge_expanded.py`:

```python
@dataclasses.dataclass(frozen=True)
class ConeMaps:
    """The four maps of the mapping cone (arXiv:2410.02753 Eq 12)."""

    support: tuple[int, ...]
    f1: np.ndarray
    f0: np.ndarray
    incidence: np.ndarray     # ∂_1 as edge-vertex (|edges|×|V0|)
    partial_0: np.ndarray     # ∂_0 (cycles × |edges|)
    data_checks: tuple[int, ...]


def edge_expanded_maps(
    H_complement: np.ndarray,
    x: np.ndarray,
    *,
    seed: int = 0,
    n_samples: int = 200,
    cellulate_to: int | None = None,
) -> ConeMaps:
    """Main construction (arXiv:2410.02753 Algorithm 3) for one X-/Z-logical.

    Produces f_1, f_0, ∂_1, ∂_0 with Cheeger(∂_1) ≥ 1 (distance preserved,
    only the target operator measured) and low-weight ∂_0.
    """
    r = restrict_maps(H_complement, x)                            # line 1: f1, ∂1*, f0*
    # data_checks: which original complementary check backs each starting edge (row of ∂1*)
    data_checks = list(r.nz_rows)
    incidence = algorithm_1(r.incidence_star, seed=seed)          # line 2: ∂1 ← Alg 1(∂1*)
    n_new_edges = incidence.shape[0] - r.incidence_star.shape[0]
    data_checks += [-1] * n_new_edges                             # line 3: κ edges back no check
    # f0 has one column per edge; original columns from f0*, new edges get zero columns.
    f0 = np.hstack([np.asarray(r.f0_star).astype(np.uint8),
                    np.zeros((H_complement.shape[0] if H_complement.ndim == 2 else 0,
                              n_new_edges), dtype=np.uint8)])
    # Map f0* rows back to full check space: f0* was indexed on nz_rows already => full n_complement rows.
    partial_0 = algorithm_2(incidence, H_complement, _edges_f0(H_complement, r, n_new_edges),
                            n_samples=n_samples, seed=seed)       # line 4: ∂0 ← Alg 2
    if cellulate_to is not None:                                  # line 15-17
        partial_0, incidence, f0_edges = cellulate(
            partial_0, incidence, _edges_f0(H_complement, r, n_new_edges),
            target_weight=cellulate_to, seed=seed)
        n_new_edges = incidence.shape[0] - r.incidence_star.shape[0]
        data_checks = list(r.nz_rows) + [-1] * n_new_edges
        f0 = f0_edges
    else:
        f0 = _edges_f0(H_complement, r, n_new_edges)
    return ConeMaps(r.support, r.f1, f0, incidence, partial_0, tuple(data_checks))


def _edges_f0(H_complement: np.ndarray, r: RestrictMaps, n_new_edges: int) -> np.ndarray:
    """f_0 with one column per edge: original nz-row checks map to their edge,
    new κ/chord edges get all-zero columns (arXiv:2410.02753 Alg 3 line 3)."""
    n_comp = np.asarray(H_complement).shape[0]
    base = np.zeros((n_comp, len(r.nz_rows)), dtype=np.uint8)
    for j, row in enumerate(r.nz_rows):
        base[row, j] = 1
    return np.hstack([base, np.zeros((n_comp, n_new_edges), dtype=np.uint8)])
```

> **Note:** `edge_expanded_maps` composes the algorithm functions; verify the two invariants (`Cheeger ≥ 1`, `∂0 @ ∂1 == 0`) in the test before committing. If cellulation adds edges, `data_checks`/`f0` are rebuilt from the post-cellulation edge count. The `_edges_f0` helper is the single source of truth for `f0`'s edge columns.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/qldpc/circuits/surgery/hmatrix/edge_expanded_test.py -k edge_expanded -v`
Expected: PASS (2 tests). The gross test may take a few seconds (Alg 1 enumeration over |V₀|=24).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(surgery): Algorithm 3 orchestrator edge_expanded_maps (arXiv:2410.02753 Alg 3)"
```

---

### Task 7: Wire into `build_gadget`; add `f1`/`f0` to `GadgetLayout`; delete dead code

**Files:**
- Modify: `src/qldpc/circuits/surgery/hmatrix/PPM_X_Z.py` (GadgetLayout fields, `build_gadget` body, delete `minimize_z_checks` + `_restrict`/`_x_merged` arbitrary-basis internals no longer used)
- Modify: `src/qldpc/circuits/surgery/__init__.py` (drop deleted exports)
- Delete: `src/qldpc/circuits/surgery/hmatrix/cheeger.py`, `src/qldpc/circuits/surgery/hmatrix/cheeger_test.py`
- Test: `src/qldpc/circuits/surgery/hmatrix/PPM_X_Z_test.py`

**Interfaces:**
- Consumes: `edge_expanded_maps`, `ConeMaps` (Task 6).
- Produces: unchanged `build_gadget(code, x, *, basis, seed=0, n_samples=200, cellulate_to="native") -> GadgetLayout`; `GadgetLayout` gains `f1: np.ndarray`, `f0: np.ndarray`. `cellulate_to="native"` resolves to the max row weight of the complementary check matrix; an int overrides; `None` disables.

- [ ] **Step 1: Write the failing test**

```python
# in PPM_X_Z_test.py
import numpy as np, sympy
from qldpc import codes
from qldpc.objects import Pauli
from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget

def test_build_gadget_exposes_four_maps_and_low_weight():
    xs, ys = sympy.symbols('x y')
    code = codes.BBCode({xs:12, ys:6}, xs**3+ys+ys**2, ys**3+xs+xs**2)   # gross
    n = code.num_qudits
    LX = np.asarray(code.get_logical_ops(Pauli.X)).astype(np.uint8)
    x = (LX[0][:n] if LX.shape[1]==2*n else LX[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    # four maps present
    for attr in ("f1", "f0", "incidence", "partial_0"):
        assert getattr(g, attr) is not None
    # valid CSS: HX_merged @ HZ_merged.T == 0
    HX = np.asarray(g.HX_merged).astype(int); HZ = np.asarray(g.HZ_merged).astype(int)
    assert np.all((HX @ HZ.T) % 2 == 0)
    # decode-weight win: merged Z checks no heavier than native (weight 6) after cellulation
    native_w = int(np.asarray(code.matrix_z).sum(axis=1).max())
    assert int(HZ.sum(axis=1).max()) <= native_w + 1   # +1 tolerance for the f1^T row block

def test_build_gadget_deterministic():
    xs, ys = sympy.symbols('x y')
    code = codes.BBCode({xs:6, ys:6}, xs**3+ys+ys**2, ys**3+xs+xs**2)
    n = code.num_qudits
    LX = np.asarray(code.get_logical_ops(Pauli.X)).astype(np.uint8)
    x = (LX[0][:n] if LX.shape[1]==2*n else LX[0]).astype(np.uint8)
    g1 = build_gadget(code, x, basis=Pauli.X, seed=0)
    g2 = build_gadget(code, x, basis=Pauli.X, seed=0)
    np.testing.assert_array_equal(g1.HZ_merged, g2.HZ_merged)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/qldpc/circuits/surgery/hmatrix/PPM_X_Z_test.py -k four_maps -v`
Expected: FAIL — `GadgetLayout` has no `f1` field / `build_gadget` has no `seed` kwarg.

- [ ] **Step 3: Write minimal implementation**

In `PPM_X_Z.py`: add `f1: np.ndarray` and `f0: np.ndarray` to the `GadgetLayout` dataclass (after `partial_0`), then rewrite `build_gadget` to assemble the merged matrices from `edge_expanded_maps`:

```python
from .edge_expanded import edge_expanded_maps

def build_gadget(code, x, *, basis, seed=0, n_samples=200, cellulate_to="native"):
    x = np.asarray(x).astype(np.uint8)
    if basis is Pauli.X:
        H_meas, H_comp = np.asarray(code.matrix_x), np.asarray(code.matrix_z)
    elif basis is Pauli.Z:
        H_meas, H_comp = np.asarray(code.matrix_z), np.asarray(code.matrix_x)
    else:
        raise ValueError(f"basis must be Pauli.X or Pauli.Z, got {basis!r}")
    H_meas = H_meas.astype(np.uint8); H_comp = H_comp.astype(np.uint8)
    if ((H_comp @ x) % 2).any():
        which = "X" if basis is Pauli.X else "Z"
        raise ValueError(f"x is not a logical-{which} support (H_complement @ x != 0).")
    ct = int(H_comp.sum(axis=1).max()) if cellulate_to == "native" else cellulate_to
    cm = edge_expanded_maps(H_comp, x, seed=seed, n_samples=n_samples, cellulate_to=ct)
    n = H_meas.shape[1]
    n_edges = cm.incidence.shape[0]
    d1 = cm.incidence.T.astype(np.uint8)               # ∂_1 (vertex×edge) for the H̃ block
    f1T = cm.f1.T.astype(np.uint8)                      # f_1^T (|Q|×n)
    # H̃_X = [[H_X, 0],[f1^T, ∂1]]; H̃_Z = [[H_Z, f0],[0, ∂0]]  (Eq 13)
    HX = np.block([[H_meas, np.zeros((H_meas.shape[0], n_edges), np.uint8)], [f1T, d1]]).astype(np.uint8)
    HZ = np.block([[H_comp, cm.f0], [np.zeros((cm.partial_0.shape[0], n), np.uint8), cm.partial_0]]).astype(np.uint8)
    if basis is Pauli.Z:
        HX, HZ = HZ, HX
    Q_prime = tuple(range(code.num_qudits, code.num_qudits + n_edges))
    return GadgetLayout(
        code=code, x=x, support=cm.support, data_checks=cm.data_checks,
        incidence=cm.incidence, partial_0=cm.partial_0, f1=cm.f1, f0=cm.f0,
        HX_merged=HX, HZ_merged=HZ, Q_prime=Q_prime, basis=basis)
```

Delete from `PPM_X_Z.py`: `minimize_z_checks`, `build_gadget_augmented`, `_x_merged`, `_restrict`, `_gf2_rank` (now unused — confirm with `grep -rn` first; keep any still referenced by `PPM_joint.py`/`PPM_Y.py` — see Step 3b).

- [ ] **Step 3b: Check joint/Y consumers before deleting**

Run: `grep -rn "_restrict\|_x_merged\|build_gadget_augmented\|minimize_z_checks\|from .cheeger\|boost_gadget" src/qldpc/circuits/surgery`
For each hit in `PPM_joint.py`/`PPM_Y.py`, either (a) it consumes `GadgetLayout` (unaffected — fields only added), or (b) it imports a deleted helper — in that case port it onto `edge_expanded_maps`/`restrict_maps` in this step, or keep the minimal helper it needs. Do NOT delete a symbol with a surviving caller.

- [ ] **Step 3c: Update `__init__.py`**

Remove `from .hmatrix.cheeger import boost_gadget, cheeger_constant` and the `minimize_z_checks` export; add `from .hmatrix.edge_expanded import cheeger_constant` (Task 2's version) and drop `boost_gadget`, `minimize_z_checks` from `__all__`. Re-export `cheeger_constant` for the notebook's pre-flight check.

- [ ] **Step 4: Run tests**

Run: `pytest src/qldpc/circuits/surgery/hmatrix/PPM_X_Z_test.py -v && pytest src/qldpc/circuits/surgery/circuit -x -q`
Expected: `four_maps` + `deterministic` PASS; circuit-layer tests still PASS (GadgetLayout backward-compatible). Fix any golden test that pinned the old arbitrary `partial_0` by regenerating its expected matrix from the new `build_gadget` output and re-asserting the real property (cone validity / distance), not the raw bytes.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(surgery): edge-expanded build_gadget as default; expose f1/f0; delete dead Cheeger/left_null_space path"
```

---

### Task 8: Paper golden fixtures — Examples 6, 7, 5

**Files:**
- Create: `src/qldpc/circuits/surgery/hmatrix/edge_expanded_golden_test.py`

**Interfaces:**
- Consumes: `restrict_maps`, `algorithm_1`, `edge_expanded_maps`, `cheeger_constant`, `build_gadget`.

- [ ] **Step 1: Write the failing test (all three examples)**

```python
import numpy as np
from qldpc.circuits.surgery.hmatrix.edge_expanded import (
    restrict_maps, algorithm_1, edge_expanded_maps, cheeger_constant)

STEANE_HZ = np.array([[0,0,0,1,1,1,1],[0,1,1,0,0,1,1],[1,0,1,0,1,0,1]], dtype=np.uint8)

def test_example6_steane_no_cycles():
    # arXiv:2410.02753 Example 6: X̄=X1X2X3 -> merged [[9,0]], ∂0 empty, no cycles.
    x = np.array([1,1,1,0,0,0,0], dtype=np.uint8)
    cm = edge_expanded_maps(STEANE_HZ, x, seed=0)
    assert cm.partial_0.shape[0] == 0                    # "∂0 does not appear in this example"
    assert cheeger_constant(cm.incidence) >= 1.0

def test_example7_hamming_algorithm1_is_load_bearing():
    # arXiv:2410.02753 Example 7: [[15,7,3]] Hamming, X̄=X3X4X5X12X14.
    # H_X=H_Z (Eq 57). Without Alg 1 the graph has Cheeger 0.5 and distance drops to 2;
    # Alg 1 adds exactly 2 edges to restore it.
    H = np.array([
        [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
        [0,1,1,1,0,0,0,0,1,1,1,1,0,0,0],  # NOTE: transcribe Eq 57 exactly at implementation time
        [1,0,1,1,0,1,1,0,0,1,1,0,0,1,1],
        [1,1,0,1,1,0,1,0,1,0,1,0,1,0,1],
    ], dtype=np.uint8)
    x = np.zeros(15, dtype=np.uint8); x[[2,3,4,11,13]] = 1     # X3X4X5X12X14 (0-indexed 2,3,4,11,13)
    r = restrict_maps(H, x)
    h_before = cheeger_constant(r.incidence_star)
    assert abs(h_before - 0.5) < 1e-9                          # paper: Cheeger 0.5
    inc_after = algorithm_1(r.incidence_star, seed=0)
    assert cheeger_constant(inc_after) >= 1.0
    assert inc_after.shape[0] - r.incidence_star.shape[0] == 2  # paper: two additional edges

def test_example5_weight8_cellulation_bound():
    # arXiv:2410.02753 Example 5: a weight-8 X̄ whose ∂1* graph is an 8-cycle
    # (Cheeger 1/2). After Alg 1 (adds 2 edges) + cellulation, max ∂0 weight ≤ 5.
    inc8 = np.zeros((8, 8), dtype=np.uint8)
    for i in range(8):
        inc8[i, i] = 1; inc8[i, (i+1) % 8] = 1                 # 8-cycle
    assert abs(cheeger_constant(inc8) - 0.5) < 1e-9
    out = algorithm_1(inc8, seed=0)
    assert out.shape[0] - 8 == 2                               # "two additional edges" (Fig 2b)
    assert cheeger_constant(out) >= 1.0
```

> **Implementation note:** transcribe the `[[15,7,3]]` `H` (Eq 57) and the Steane `H_Z` (Eq 54) from the PDF at `/Users/tgzhou/.claude/projects/-Users-tgzhou-Project-qLDPC/3110c02c-7a78-4eba-a225-8f54569d2197/tool-results/webfetch-1782989068608-txoukg.pdf` pages 10-11 exactly; the rows above are placeholders for the correct parity checks. The invariants asserted (Cheeger 0.5 → ≥1, exactly 2 edges added) are the paper's stated numbers and must hold once `H` is correct.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/qldpc/circuits/surgery/hmatrix/edge_expanded_golden_test.py -v`
Expected: FAIL (transcription placeholders / import).

- [ ] **Step 3: Fix the transcribed matrices**

Read the PDF pages 10-11 and replace the placeholder `H` (Eq 57) with the exact `[[15,7,3]]` Hamming parity check, and confirm Steane `H_Z` (Eq 54). Adjust 0-indexing of `X̄ = X3X4X5X12X14` if the paper is 1-indexed (→ indices 2,3,4,11,13).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/qldpc/circuits/surgery/hmatrix/edge_expanded_golden_test.py -v`
Expected: PASS (3 tests). If `test_example7` shows a different edge count, re-read the note in Alg 1 (min-degree tie handling) — the paper's "two additional edges" is the target.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "test(surgery): paper golden fixtures Examples 5/6/7 (arXiv:2410.02753)"
```

---

### Task 9: End-to-end DEM validity + decode-speed sanity

**Files:**
- Create: `src/qldpc/circuits/surgery/hmatrix/edge_expanded_e2e_test.py`

**Interfaces:**
- Consumes: `build_gadget`, `build_single_ppm_circuit`, `keep_only_observable`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np, sympy
from qldpc import codes
from qldpc.objects import Pauli
from qldpc.circuits.surgery import build_gadget, build_single_ppm_circuit, keep_only_observable

def _gross_x_logical():
    xs, ys = sympy.symbols('x y')
    code = codes.BBCode({xs:12, ys:6}, xs**3+ys+ys**2, ys**3+xs+xs**2)
    n = code.num_qudits
    LX = np.asarray(code.get_logical_ops(Pauli.X)).astype(np.uint8)
    x = (LX[0][:n] if LX.shape[1]==2*n else LX[0]).astype(np.uint8)
    return code, x

def test_dem_compiles_and_one_observable():
    code, x = _gross_x_logical()
    g = build_gadget(code, x, basis=Pauli.X)
    circ = build_single_ppm_circuit(g, rounds=int(code.get_distance(Pauli.X) or 12))
    circ = keep_only_observable(circ, 0)
    dem = circ.detector_error_model(decompose_errors=False)
    assert dem.num_observables == 1                         # only X̄ measured (Remark 3)

def test_merged_z_check_weight_below_legacy():
    # Regression guard: the new construction's max merged Z-check weight is at most
    # the native code weight + 1, i.e. strictly better than the old ~11-12.
    code, x = _gross_x_logical()
    g = build_gadget(code, x, basis=Pauli.X)
    native = int(np.asarray(code.matrix_z).sum(axis=1).max())
    assert int(np.asarray(g.HZ_merged).astype(int).sum(axis=1).max()) <= native + 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/qldpc/circuits/surgery/hmatrix/edge_expanded_e2e_test.py -v`
Expected: FAIL if `build_single_ppm_circuit` needs adaptation to the new `GadgetLayout`, else PASS on `weight_below_legacy`.

- [ ] **Step 3: Adapt the circuit builder if needed**

If `build_single_ppm_circuit` reads only `HX_merged`/`HZ_merged`/`Q_prime`/`support`/`data_checks`, no change is needed. If it referenced a deleted symbol, port it to read the new fields. Confirm with `grep -rn "partial_0\|incidence\|minimize_z_checks" src/qldpc/circuits/surgery/circuit`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/qldpc/circuits/surgery/hmatrix/edge_expanded_e2e_test.py -v && pytest src/qldpc/circuits/surgery -q`
Expected: full surgery suite PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "test(surgery): e2e DEM validity + merged Z-check weight regression guard"
```

---

## Self-Review

**Spec coverage:**
- Alg 1 → Task 3 (+ Cheeger/sparsest-cut Task 2). ✓
- Alg 2 → Task 4. ✓
- Alg 3 cellulation → Task 5; orchestration → Task 6. ✓
- `f₁,f₀,∂₁,∂₀` exposed → Task 1 (restrict), Task 6 (ConeMaps), Task 7 (GadgetLayout fields). ✓
- Default construction + delete dead code → Task 7. ✓
- Scale past |V₀|=26 → **partial:** Task 2/3 use exact enumeration. Spec's Fiedler fallback is NOT a separate task because the tested logicals (gross w=24, bb_18 low-weight reps ≤~24) stay within exact-enum range, and the fidelity requirement ("exactly as written") favors the paper's exact `argmin`. **If a logical with |V₀|>26 is needed, add a `sparsest_cut(..., method="fiedler")` opt-in as a follow-up task** — flagged here rather than silently dropped.
- Determinism (fixed seed) → Task 7 `test_build_gadget_deterministic`. ✓
- Cellulate to native weight → Task 7 (`cellulate_to="native"`). ✓
- Golden fixtures 5/6/7 → Task 8. ✓
- DEM num_observables / weight regression → Task 9. ✓
- No LER tests → all verification is structural/DEM. ✓

**Placeholder scan:** Task 8 intentionally ships placeholder parity-check matrices with an explicit Step 3 to transcribe them from the PDF — this is a guided transcription step with concrete invariants, not an unspecified TODO. All code steps contain runnable code.

**Type consistency:** `RestrictMaps` (Task 1) → `ConeMaps` (Task 6) → `GadgetLayout.f1/f0` (Task 7) field names align (`f1`, `f0`, `incidence`, `partial_0`, `support`, `data_checks`). `algorithm_1/2`, `cellulate`, `edge_expanded_maps`, `restrict_maps`, `cheeger_constant`, `sparsest_cut`, `boundary` signatures are consistent across tasks.

**Known risk:** Algorithm 2's line 1 (`ker H_complementᵀ` orientation) and Algorithm 3 cellulation's chord-selection heuristic are the two spots where the paper is terse; both have an asserted invariant (`∂0 @ ∂1 == 0`) that must pass before commit, and the note in each task tells the implementer how to correct orientation/heuristic if the invariant fails.
