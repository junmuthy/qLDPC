# Edge-Expanded Homological Measurement (Algorithms 1/2/3)

**Date:** 2026-07-02
**Branch:** `feat/edge-expanded-homological`
**Paper:** Ide, Gowda, Nadkarni, Dauphinais, *Fault-Tolerant Logical Measurements
via Homological Measurement*, arXiv:2410.02753 (§III B, Algorithms 1–3).

## Problem

After adding surgery, decoding is slow. Root cause: the ancilla Z-checks
`partial_0` (= ∂₀ = a basis of the cycle space `ker ∂₁`) are computed with
galois `left_null_space()` — an **arbitrary row-echelon basis** whose rows can be
high weight. Measured on the gross `[[144,12,12]]` code, a weight-24 X̄ gadget
produces **49 ancilla and ∂₀ rows of weight up to 12** (native checks are weight
6). These dense, high-weight Z-stabilizers become wide DEM hyperedges → dense
Tanner graph → slow matching/BP. Additionally, the current Cheeger boost
(`boost_gadget_cheeger_combinatorial`) uses exact subset enumeration limited to
`|support| ≤ 26`, so it cannot handle wider logicals (gross weight 24, bb_18
weight up to ~39).

The paper's Algorithms 1/2/3 (*edge-expanded homological measurement*) are the
targeted fix: they produce a low-weight, low-connectivity `f₁, f₀, ∂₁, ∂₀`.

## Mapping to the existing code

The current `incidence` matrix (`|C₀|×|V₀|`, in `hmatrix/PPM_X_Z.py::_restrict`)
already **is** the paper's ∂₁-as-edge-vertex-incidence (Def 2):
- **vertices** `V = V₀ = supp(x)` (support qubits),
- **edges** `E = Q'` = ancilla, one per complementary check `C₀` plus boost-added κ,
- `∂₁` (edge→vertex incidence) = `incidence`; `∂₀` = cycle space = left-null-space
  of `incidence` (each row is a cycle = a set of edges, Def 4).

So all three algorithms slot in exactly where the code already computes
`incidence` and `partial_0`. The defect is only *which* cycle basis is chosen and
*whether* edges are expanded.

## Algorithms (verbatim intent from the paper)

### Algorithm 1 — greedy edges to Cheeger constant = 1 (p8)
Input: hypergraph `A = G(V,E)` with `h(A) < 1`. Output: superset `B` with
`h(B) = 1`, same vertices.
```
E* ← E ; B ← G(V, E*)
while h(B) < 1:
    S ← argmin_{S⊂V, |S|≤|V|/2} |∂S| / |S|        # sparsest cut, ∂S via E*
    h* ← -∞ ; e ← ⊥
    for v1 in min-degree vertices of S:
        for v2 in min-degree vertices of V∖S:
            if h(G(V, E* ∪ {(v1,v2)})) > h*:
                h* ← that value ; e ← (v1, v2)
    E* ← E* ∪ {e} ; B ← G(V, E*)
return B
```
`∂S = {e ∈ E : |e ∩ S| odd}`; `h(G) = min |∂S|/|S|` over `|S| ≤ |V|/2`.
A positive Cheeger constant ⇒ single connected component ⇒ `dim ker ∂₁ = 1`, so by
Lemma 1 only the desired operator is measured (Remark 3). `h ≥ 1` ⇒ distance
preserved (Theorem 2). **Must scale past |V₀| = 26.**

### Algorithm 2 — random search for low-weight ∂₀ (p9)
Input: `∂₁, H_Z, f₀, n` (sample count). Output: `∂₀`.
```
V ← basis of redundant cycles { vᵀ f₀ : v ∈ ker H_Zᵀ }   ; V ← rref(V)
W ← basis s.t. V ⊕ W = full cycle space (ker of edge-vertex ∂₁)
W ← rref(W with V's pivot columns zeroed, drop zero rows)
∂₀ ← W
for i in 1..n:
    A ← random invertible (GF2) ; B ← random (GF2)
    if maxrowweight(A W + B V) < maxrowweight(∂₀): ∂₀ ← A W + B V
    if maxrowweight(A W)       < maxrowweight(∂₀): ∂₀ ← A W
return ∂₀
```
`V` = cycles already represented in `H̃_Z` regardless of `∂₀` (redundant); `∂₀`
generates the complement. Adding `V` rows or invertibly recombining `W` rows keeps
the same coset ⇒ same stabilizer group, lower weight. This is the weight-aware
generalization of the current weight-blind `minimize_z_checks`.

### Algorithm 3 — full construction + cellulation (p10)
Input: `C = CSS(H_X, H_Z)`, X-logical `X̄`. Output: `C̃` with distance ≥ d, `X̄` in
the stabilizer group, all other logicals unharmed.
```
1  f₁, ∂₁*, f₀*  ← Eqs (35), (47), (48)
2  ∂₁ ← Algorithm 1 on ∂₁*
3  f₀ ← f₀* with zero columns added for the new edges
4  ∂₀ ← Algorithm 2 on (∂₁, H_Z, f₀, n)
5  if sparsity of ∂₀ unacceptable:            # hyperedge expansion path
6-13  expand hyperedges (wt>2) to weight-2 edges; re-run Alg 1 & Alg 2
15-17 cellulate: for each ∂₀ row heavier than target, add edges inside that
      cycle to split it into smaller cycles (adds a few ancilla, lowers weight)
19 C̃ ← mapping cone of f (Eq 12): H̃_X = [[H_X,0],[f₁ᵀ,∂₁]], H̃_Z = [[H_Z,f₀],[0,∂₀]]
20 return C̃
```
Eq (35): `(f₁)_{i,j} = δ_{i,q_j}` (n×w indicator on `Q = supp X̄`).
Eq (47): `∂₁* = H_Z|_Q` restricted to columns `Q`, zero rows removed.
Eq (48): `(f₀*)_{i,j} = δ_{i,h_j}`, `h_j` = row index of the j-th non-zero row of `H_Z|_Q`.

## Architecture

New module **`src/qldpc/circuits/surgery/hmatrix/edge_expanded.py`** — four pure
GF(2) functions plus one orchestrator:

| function | algorithm | replaces |
|---|---|---|
| `greedy_cheeger_edges(incidence, *, seed)` | Alg 1 | `boost_gadget_cheeger_combinatorial` |
| `low_weight_cycle_basis(incidence, H_complement, f0, *, n_samples, seed)` | Alg 2 | `left_null_space` + weight-blind `minimize_z_checks` |
| `cellulate(partial_0, incidence, f0, *, target_weight, seed)` | Alg 3 §15-17 | (new) |
| `edge_expanded_gadget(code, x, *, basis, seed, n_samples, cellulate_to)` | Alg 3 orchestration | core of `build_gadget` |

**Sparsest cut** (Alg 1 line 3) is NP-hard in general. `|V₀|` = logical weight
(≤ ~26 for many cases, but 24–39 for gross/bb_18). Strategy:
- exact Gray-code subset enumeration for `|V₀| ≤ 24` (reuse the bit-packed
  routine already in `cheeger.py::_exact_boundary_cheeger`),
- **Fiedler-vector sweep** (spectral sparsest-cut heuristic) for `|V₀| > 24`.

This removes the current `|V₀| ≤ 26` hard limit.

## Integration

- `build_gadget(code, x, *, basis, seed=0, n_samples=..., cellulate_to=...)` runs
  the new edge-expanded path (the only path), producing all four maps.
- `GadgetLayout` gains explicit `f1` and `f0` fields (currently reconstructed
  inline in `_x_merged`), so `f₁, f₀, ∂₁ (=incidence.T), ∂₀ (=partial_0)` are all
  first-class on the returned layout.
- **Fidelity requirement (hard):** each of `greedy_cheeger_edges`,
  `low_weight_cycle_basis`, `cellulate`, and `edge_expanded_gadget` mirrors the
  paper's pseudocode **line-for-line**, in the same step order, with a comment on
  each block citing the exact Algorithm/line it implements. No "equivalent"
  shortcuts or reordering.
- **Delete dead code:** the old arbitrary-basis path is *replaced*, not kept
  behind a flag. Once the new construction is the default, the now-unused pieces
  are deleted on this branch (git-reversible): `cheeger.py`'s
  `boost_gadget_cheeger_combinatorial` / `boost_gadget_distance` /
  `_augment_incidence_with_random_edges` / spectral+exact enum helpers that no
  longer have a caller, and the weight-blind `minimize_z_checks` (superseded by
  Algorithm 2). Tests that only covered the deleted code are removed; tests that
  assert real properties (cone validity, distance, num_observables) are
  regenerated against the new lower-weight matrices. The Cain et al. Table III
  count check is re-expressed against the new construction (or dropped if it only
  pinned the old arbitrary basis).
- `basis=Pauli.Z` reuses the identical primitives via the X↔Z dual (swap
  `H_X`/`H_Z` in, swap merged matrices out), matching the current dispatch.
- **Determinism:** a fixed default `seed=0` makes `build_gadget(code, x, basis)`
  reproducible run-to-run; `seed` is exposed for sweeps.
- **Cellulation default:** cellulate ∂₀ rows down toward the **native code's max
  check weight** (`cellulate_to = max row weight of H_Z`), per the "fast decode"
  goal. Exposed as a parameter; `None` disables.

The joint / mixed-Pauli path (`PPM_joint.py`, `PPM_Y.py`, §III.D Eq 66/67/68)
reuses the same primitives: Alg 1 is applied separately to X̄ and Z̄, stabilizers
merged, then Alg 2 finds a cycle basis on the union graph (Eq 66) with per-edge
cellulation choice (X-graph vs Z-graph). This spec covers the single X-/Z-operator
path; wiring the joint/mixed callers onto `edge_expanded` is a follow-up that does
not change their existing constraints (see repo memory on 2407.18393 / SJOY /
homological-measurement mixed-check rules).

## Verification (deterministic only — no LER / statistical sampling)

1. **Valid cone:** `∂₀ @ ∂₁ = 0 (mod 2)` and `H̃_X H̃_Zᵀ = 0`.
2. **No extra logicals:** `dim ker ∂₁ = 1` (Remark 3), cross-checked via DEM
   `num_observables == expected` on a compiled circuit.
3. **Distance preserved:** exact merged distance ≥ d on Steane and `[[36,8,4]]`;
   Cheeger(∂₁) ≥ 1 as the structural guarantee (Theorem 2) on gross / bb_18.
4. **Weight win:** max row weight of ∂₀ and of `HZ_merged`, and the ancilla count,
   are each **≤** the current construction (assert strict improvement on the gross
   case: current max weight 12 → target ≤ 6).
5. **Paper golden fixtures:**
   - **Example 6** — Steane `[[7,1,3]]`, `X̄ = X₁X₂X₃`: reproduce `f₁, ∂₁, f₀`
     (Eq 55/56); `∂₀` empty (no cycles); merged `[[9,0]]`.
   - **Example 7** — `[[15,7,3]]` Hamming, `X̄ = X₃X₄X₅X₁₂X₁₄`: **without** Alg 1 →
     `[[19,6,2]]` (distance 3→2, weight-2 logicals appear); **with** Alg 1 → exactly
     2 extra ancilla, distance 3 held, Cheeger 0.5→1 (Eq 58/59). Proves Alg 1 is
     load-bearing.
   - **Example 5** — weight-8 X̄: 10 ancilla without cellulation, 12 with cellulation
     at weight ≤ 5 (Eq 50-53).
6. **Decode-time sanity** (structural, not LER): DEM compile + a wall-clock decode
   timing on the notebook gross/bb_18 case, showing the new construction's DEM is
   sparser and decodes faster than the current one.

## Out of scope

- LER / sinter sweeps (repo preference: no statistical-sampling tests).
- Rewiring the joint/Ȳ/mixed callers (follow-up; primitives are shared).
- The `L > 1` / repetition-code-adapter constructions (Appendix A).
