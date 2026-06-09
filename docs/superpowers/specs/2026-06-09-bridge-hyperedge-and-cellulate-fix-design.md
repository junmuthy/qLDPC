# Bridge fix: hyperedge skip + cellulate on port subgraph

**Date:** 2026-06-09
**Branch:** `refactor/cheeger-l1-collapse`
**Paper:** [arXiv:2410.03628v4](https://arxiv.org/abs/2410.03628) — *Universal adapters between quantum LDPC codes*

## Why this exists

`build_bridge` (`src/qldpc/codes/surgery/bridge.py`) currently fails on real BB
codes like `bb_18` (Cain et al. 2024, Ref. [138]:
`a = 1 + x⁶y + x²⁷`, `b = y² + x¹⁵y³ + x²⁴`, l=31, m=4, [[248,10,≤18]]).
Two independent bugs surface as we try to bridge logicals on such codes:

- **Bug 1 — Hyperedges.** `_build_aux_graph_strict` (`bridge.py:200`) raises
  `NotImplementedError` whenever an F row has weight ≥ 3, pointing to "paper
  §II.C decomposition". For `bb_18` with `Z̄ = code.get_logical_ops(Pauli.Z)[0]`,
  F has exactly one weight-4 row (94 weight-2 rows).
- **Bug 2 — Cellulation on the wrong graph.** `_cellulate_strict`
  (`bridge.py:150`) finds long cycles in the **full** auxiliary graph and
  demands chords whose **both endpoints lie in the port subset**. SkipTree
  actually runs on the *induced port subgraph*, so cycles that thread through
  non-port V_0 vertices are irrelevant to T_s row weight (Theorem 7 bounds
  it at 3 unconditionally). The full-graph restriction makes cellulation fail
  spuriously: a cycle like `[1, 22, 31, 48, 60, 53, 43, 57, 47, 30, 11]`
  contains only two port vertices and no available port-port chord, so the
  function raises even though the port subgraph itself is fine.

The user explicitly verified that swapping in a different Z-logical
representative (whose F has all weight-2 rows) still trips Bug 2. The two bugs
are independent.

## Scope

- **In:** silent skip of hyperedge rows in the auxiliary graph; cellulation
  restricted to the port subgraph with any-V_0-vertex chords; minimal updates
  to existing tests; one new end-to-end test on `bb_18`.
- **Out:** paper-faithful §II.C decomposition (matching pairs routed to the
  original check via F̃, dropping `κ_r`). This was considered (Option B in
  brainstorming) and rejected: it requires a `build_gadget_augmented` API
  change for ~150 LOC, and Option A gives provably correct CSS commutation
  and joint observable for any F — the only thing forfeited is paper
  Theorem 12's structural distance argument, which finite-size LER smoke tests
  already substitute for in current practice.
- **Out:** changing which logical the demo picks. The fix must work for the
  user's chosen `code.get_logical_ops(Pauli.Z)[0]` and any other
  representative.

## Correctness proof — why "skip" is safe

`_run_skiptree_on_port_subgraph` (`bridge.py:271-320`) already gates SkipTree
column embedding on `if len(cols) != 2: continue` — hyperedge rows of F_aug
get **zero columns in T_s** regardless of what `_build_aux_graph_strict`
does. The only thing currently preventing hyperedge bridges is the explicit
`raise` in `_build_aux_graph_strict`. Removing the raise (Option A) leaves
the math intact:

### 1. CSS commutation `χ_v · cycle_c` (the only χ–cycle pairing)

For basis=Z (the user's case), χ_v rows are Z-type and cycle adapter rows
are X-type. The κ-side product is `(T_s · F_aug)[c, v]`. For hyperedge row
r_0: `T_s[c, r_0] = 0`, so its contribution is `0 · F_aug[r_0, v] = 0`
regardless of v. The weight-2 rows give `(T_s · F_aug_w2)[c, v]`, which the
SkipTree key identity `T_s · F_aug_w2 · P_s = H_R` makes equal to
`H_R[c, label(v)] · [v ∈ port]`. The adapter side contributes the same
quantity, so the two cancel mod 2.

### 2. CSS commutation `χ_v · c_deformed` (the hyperedge's original check)

For c = C_0[r_0] with hyperedge support `{v_a, v_b, v_c, v_d}`:
F̃[c, r_0] = 1 (unchanged). For v ∈ {v_a..v_d}: data anti-commute on v (1)
+ κ anti-commute on κ_r_0 (1) = 0 ✓. For v ∈ V_0 \ {v_a..v_d}: both
contributions 0 ✓.

### 3. κ-cancellation

`Σ χ on κ = F_aug^T · 1_{V_0}`. Hyperedge rows have even weight (forced by
`H_complement · x = 0`), so the row sum is 0 mod 2. Joint observable's κ
contribution is 0 ✓.

### 4. Merged code dimension

`G_aug = ker(F_aug^T)` is computed on the **full** F_aug (with hyperedge
rows). The standard counting still holds:

```
k_merged = qubits − Z_stab_rank − X_stab_rank = k_l + k_r − 1
```

The `−1` comes from `Σ χ_l + Σ χ_r = (x_l + x_r, 0, 0, 0)` placing the joint
logical into the stabilizer group. Hyperedge presence doesn't change this
relation because κ-cancellation holds.

### 5. LDPC weight bounds

χ_v row weight = 1 (data) + (F column-v weight). Column weights of F come
from the underlying code (number of complementary checks touching v),
which is O(1) for LDPC. Hyperedge presence is a row-weight phenomenon, not
column-weight, so χ row weights are unaffected.

The only thing **not** preserved by Option A is paper Theorem 12's
structural distance argument (which assumes the aux graph is a true
graph). For finite-size codes the demo's `obs0 == obs1` cross-check plus
LER monotone smoke tests are the operational substitute, matching the
paper's own remark at the end of §IV:

> we believe that to perform specific join logical measurements in
> practice, the construction in the proof of Theorem 11 will typically be
> used without the explicit guarantees that the original auxiliary graphs
> have sufficient relative expansion. One should then simply verify
> directly that the deformed code defining the joint logical measurement
> has code distance d.

## Logical-representative invariance

The user's question — what happens if we swap `Z̄_l` for an equivalent
representative `Z̄_l ⊕ s_l` — has a clean answer:

| Quantity | Varies with representative? |
|---|---|
| V_0, C_0, F, G, κ qubit count, merged total qubits | yes |
| Joint operator equivalence class `[Z̄_l ⊗ Z̄_r]` | **no** |
| Merged code dimension `k_l + k_r − 1` | **no** |
| Physical PPM eigenvalue distribution | **no** |
| LER / distance | possibly (different structure, different noise resilience) |

The demo's three asserts (`rate0 ≈ 50%`, `rate1 ≈ 50%`, `obs0 == obs1 per
shot`) are representative-invariant physical properties. Switching
logicals to side-step hyperedges is therefore a valid escape hatch, but
the bridge needs to handle hyperedges anyway because the canonical
`get_logical_ops()[0]` returns whatever it returns and the user should
not have to hunt for a "lucky" representative.

## §1. Bug 1 fix — hyperedge skip

`_build_aux_graph_strict` (`bridge.py:200-234`): replace the raise on
`len(eps) >= 3` with a silent continue. Keep the weight-1 raise (a weight-1
row violates `F · 1 = 0 mod 2`; if it ever appears it signals an invalid
logical, not a hyperedge).

```python
def _build_aux_graph_strict(F: np.ndarray) -> tuple[nx.Graph, dict[tuple[int, int], int]]:
    """Build auxiliary graph from F; weight-2 rows become edges; hyperedges skipped.

    Weight-≥3 rows are silently ignored — they remain in F_aug so the gadget,
    G_aug = ker(F_aug^T), and the deformed check c → c · X(κ_r) are unchanged,
    but T_s assigns them zero columns (already-existing skip at
    _run_skiptree_on_port_subgraph). CSS commutation, κ-cancellation, joint
    observable, and dimension counting all hold; see design doc §correctness-proof.

    Weight-0 rows (no V_0 overlap) cannot enter C_0 in the first place but
    are skipped defensively.

    Raises:
        ValueError: if any F row has weight 1 (defensive — F · 1 = 0 mod 2
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
```

`_run_skiptree_on_port_subgraph` is **untouched**: its existing
`if len(cols) != 2: continue` guard at `bridge.py:312` already handles
hyperedge rows of F_aug. The aux-graph skip and the T_s zero-column behavior
agree by construction.

## §2. Bug 2 fix — cellulate on port subgraph

Rename `_cellulate_strict` → `_cellulate_port_subgraph` and change its
contract:

- **Operates on** `G_aux.subgraph(ports)`, not the full G_aux. Cycles outside
  the port subgraph are SkipTree-irrelevant and ignored.
- **Chord endpoints** may be any V_0 vertex (not restricted to ports). The
  resulting weight-2 row is a valid extra_κ regardless of whether its endpoints
  are ports — gadget assembly is endpoint-blind.
- **Adds the chord** to `G_aux` (the full graph). After each addition, refresh
  the port-subgraph view and re-test.

```python
def _cellulate_port_subgraph(
    G_aux: nx.Graph,
    ports: tuple[int, ...],
    *,
    max_len: int = 6,
) -> list[tuple[int, int]]:
    """Break port-subgraph cycles longer than max_len by adding chords.

    SkipTree runs on G_aux.subgraph(ports); cycles entirely outside the
    port subgraph never enter T_s, so we cellulate only there. Mutates
    G_aux by adding weight-2 chords (any V_0 endpoints — not restricted
    to ports).

    Returns the list of added (u, v) edges in insertion order.
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
                f"cycle={cycle!r}"
            )
```

The `j >= i + 2` lower bound on the inner loop skips immediate cycle
neighbors (their edge is already in the cycle).

The error message no longer mentions ports — chord endpoints are any V_0
vertex.

## §3. `build_bridge` integration

`build_bridge` (`bridge.py:323-409`) Step 4 changes one call:

```python
# Step 4: cellulation (port subgraph only)
extras_l_cell = _cellulate_port_subgraph(G_l_aux, port_l, max_len=cellulate_max_len)
extras_r_cell = _cellulate_port_subgraph(G_r_aux, port_r, max_len=cellulate_max_len)
```

Step 1 (build aux graph) and Step 7 (build augmented gadget) are unchanged
in code — the hyperedge fix is internal to `_build_aux_graph_strict`.

## §4. Test changes

### Existing tests to update

| `_test.py` location | change |
|---|---|
| `test_build_aux_graph_rejects_hyperedge` (line 995) | Rename to `test_build_aux_graph_filters_hyperedges`. Assert the function returns a graph with the hyperedge row absent (not in `edge_index`), instead of expecting `NotImplementedError`. |
| `test_cellulate_raises_when_no_port_chord_available` (line 1059) | Rewrite. New `_cellulate_port_subgraph` raises only if no non-edge chord exists in the port-subgraph cycle, not if ports are too few. Test must use a port subgraph that's already a complete graph on its vertices. |
| `test_cellulate_caps_cycle_length` (line 1037) | Pass; new function operates on port subgraph. Adjust ports to make the cycle a port-subgraph cycle so the test still exercises cellulation. |
| `test_cellulate_no_op_when_already_short` (line 1050) | Pass with no change. |
| `test_cellulation_caps_aug_aux_cycle_length_on_webster` (line 1434) | Inspect the augmented aux graph's *port subgraph*, not the full graph. |

### New tests

| test | invariant |
|---|---|
| `test_build_aux_graph_skips_hyperedge_row` | Weight-4 row in F → corresponding edge absent from `G_aux.edges()`, weight-2 rows still present. |
| `test_cellulate_port_subgraph_accepts_nonport_chord` | A cycle in the port subgraph with no port-port chord available is cellulated by a chord between non-port endpoints (new edge in G_aux, new row in extras). |
| `test_cellulate_skips_non_port_cycles` | A long cycle entirely on non-port vertices is NOT touched; no edges added. |
| `test_build_bridge_bb18_hyperedge_and_long_cycle` | End-to-end smoke: `code = BBCode({x:31, y:4}, 1+x⁶y+x²⁷, y²+x¹⁵y³+x²⁴)`, `z = code.get_logical_ops(Pauli.Z)[0]`, `g = build_gadget(code, z, basis=Pauli.Z)`, `build_bridge(g, g, ...)` returns without raising. Assert `T_s · F_aug_w2 · P_s == H_R` (SkipTree identity holds on weight-2 sub-incidence) and CSS commutation `H_X^merged · H_Z^merged.T == 0`. |
| `test_build_bridge_bbcode_k_reduces_by_one` | The user-flagged dimension question: the merged code from the BB pair satisfies `merged.dimension == g_l.code.dimension + g_r.code.dimension - 1` (inter-code) — verifying hyperedge presence doesn't break the −1 invariant. |

### Demo verification

`examples/scripts/joint_ppm_z_with_superposition_demo.py`: run as-is after
the fix. Expected: all three asserts pass (`rate0`, `rate1` near 50%;
`obs0 == obs1` on every shot).

## §5. Documentation updates

### `docs/superpowers/math.md` §2.2

Append a paragraph after the existing "Auxiliary graph augmentation" section:

> When `F` has rows of weight ≥ 4 (hyperedges), they are kept in `F_aug` so
> the gadget structure is unchanged but skipped in the auxiliary graph 𝒢_s.
> SkipTree assigns `T_s` zero columns to hyperedge rows, so the SkipTree key
> identity reduces to the weight-2 sub-incidence. Paper Eq. 9's perfect-matching
> decomposition (§II.C) is *not* applied; CSS commutation, κ-cancellation,
> and the joint observable identity all hold by direct calculation, but the
> structural distance argument of Theorem 12 does not apply — code distance
> is verified empirically via LER smoke tests.

### `docs/superpowers/specs/2026-06-09-joint-ppm-bridge-design.md` §3 (Hyperedge handling)

Replace the existing "raises NotImplementedError" note with a pointer to
this design doc.

## §6. Migration / blast radius

| File | LOC delta | Risk |
|---|---|---|
| `src/qldpc/codes/surgery/bridge.py` | +5 / −15 | Low — only two functions touched; behavior change is "skip instead of raise" + "operate on subgraph". |
| `src/qldpc/codes/surgery/_test.py` | +60 / −15 | Low — rename + 4 new tests + 1 invariant test. |
| `docs/superpowers/math.md` | +10 | Doc-only. |
| `docs/superpowers/specs/2026-06-09-joint-ppm-bridge-design.md` | +2 / −5 | Doc-only. |
| Demo scripts (`joint_ppm_z_with_superposition_demo.py`) | 0 | No source change; should run as-is. |
| `examples/logical_error_rates/_9_mylattice_surgery_test.ipynb`, `examples/scripts/single_ppm_vs_memory_ler_webster.py` | 0 | Leave WIP files untouched per session-start instruction. |

No changes to `gadget.py`, `circuit.py`, or `cheeger.py`. The `Bridge`
dataclass and its consumers are not affected.

## §7. Out of scope (deferred — not part of this fix)

- Paper-faithful §II.C hyperedge decomposition (Option B). Documented for
  future revisit if structural Theorem 12 distance guarantees become
  necessary.
- Decongestion (paper §VII.A Lemma 6) for sparse cycle-basis multiplicity.
  Theorem 7 already bounds T_s row weight ≤ 3 regardless.
- Hamiltonian-path SkipTree optimization (Remark 20). Default BFS spanning
  tree is fine.
- Logical-representative reduction (search for an equivalent representative
  whose F is all weight-2). Useful as a separate optimization pass but not
  required for correctness.
