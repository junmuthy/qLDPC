# Joint PPM bridge design — Universal Adapter (Swaroop–Jochym-O'Connor–Yoder 2024)

**Date:** 2026-06-09
**Branch:** `feat/surgery-construction`
**Paper:** [arXiv:2410.03628v4](https://arxiv.org/abs/2410.03628) — *Universal adapters between quantum LDPC codes*

## Why this exists

`src/qldpc/codes/surgery/bridge.py` currently implements a path-graph bridge between two
`GadgetLayout` instances: two endpoint `χ`-extensions plus a chain of `U_B` X-stabilizers. That
construction is the Cain *et al.* (Ref. [21] of the Swaroop paper) "bridge system" — a special case
that does not preserve code distance for `d > 1` because Lemma 9's relative-expansion premise
`|𝒜| ≥ d` is violated when `|𝒜| = 1`. Demo scripts (`joint_ppm_z_with_superposition_demo.py`,
`_9_mylattice_surgery_test.ipynb`) hit this exact failure mode.

The paper's §IV–§V universal adapter is what we want: a bijection-based adapter on `w ≥ d` port
pairs, with new Z-type cycle-check rows produced by the **SkipTree** algorithm guaranteeing the
merged code is LDPC and `d_merged ≥ min(d_l, d_r)` (Theorem 12).

## Scope

- **Construction:** Theorem 11 + Theorem 12 practical recipe (paper §VII.B / §VII.C). SkipTree
  driven, no relative-expansion certificates, no decongestion automation. Cellulation is included
  because the current implementation already has it and the paper's §VII.A relies on it.
- **Coverage:** inter-code (e.g. BB-LP §VII.B) and intra-code (e.g. BB-BB §VII.C, with the
  sparsely-overlapping `V_0` two-vertex-check trick from Appendix VIII).
- **Verification:** structural (CSS commutation, joint logical in row span, singletons excluded,
  dimension reduces by 1, adapter check weight ≤ 8, SkipTree `T·G·P = H_C`) **plus** LER-sampling
  smoke tests. No BP-OSD / CPLEX integration.
- **Basis:** symmetric in `Pauli.X` (joint X̄_l X̄_r) and `Pauli.Z` (joint Z̄_l Z̄_r); dual
  construction across `H_X ↔ H_Z`.

## Notation summary

| Symbol | Meaning |
|---|---|
| `g_s` | `GadgetLayout` for side s ∈ {l, r} from existing `build_gadget(code_s, x_s, basis)` |
| `V_0^(s)` | qubit support of measured logical = `g_s.V0` |
| `C_0^(s)` | complementary-type checks touching `V_0^(s)` = `g_s.C0` |
| `F^(s)` | restriction matrix `H_complement[C_0, V_0]` from `g_s.F`; shape `|C_0| × |V_0|` |
| `G^(s)` | gauge-fix basis `ker(F^T)` from `g_s.G` |
| `𝒢_s = (V_s, E_s)` | auxiliary graph: vertices = `V_0^(s)`, edges = weight-2 rows of F^(s) |
| `𝒫_s* ⊆ V_0^(s)` | selected port subset |
| `𝒜` | adapter edge bijection between `𝒫_l*` and `𝒫_r*` |
| `w = |𝒜|` | adapter width |
| `T_s`, `P_s` | SkipTree output for `𝒢_s[𝒫_s*]` (after augmentation/cellulation) |
| `H_R` | full-rank canonical rep-code parity matrix, `(w-1) × w` |

## §1. Architecture

```
GadgetLayout g_l ─┐
                 ├──── build_bridge(g_l, g_r, *,
GadgetLayout g_r ─┘        port_subset_l=None,
                            port_subset_r=None,
                            spanning_tree_root_l=0,
                            spanning_tree_root_r=0,
                            cellulate_max_len=6,
                         ) ──→ Bridge

Bridge:
    width: int                                # w = |𝒜|
    basis: Pauli                              # X or Z (symmetric dual)
    port_l, port_r: tuple[int, ...]           # 𝒫_l*, 𝒫_r* (length w)
    label_l, label_r: tuple[int, ...]         # label_s[port_s[k]] = k (SkipTree labelling)
    extra_kappa_l, extra_kappa_r: np.ndarray  # added κ-rows to make 𝒢_s[𝒫_s*] connected + cellulated
                                              # shape (e_s, |V_0^(s)|) over GF(2)
    T_l, T_r: np.ndarray                      # (w-1, |C_0^(s)| + e_s)
    H_R: np.ndarray                           # (w-1, w) full-rank canonical rep-code parity
    g_l_aug, g_r_aug: GadgetLayout            # gadgets rebuilt with F_aug, G_aug, HX/HZ_merged_aug
```

Merged register columns:

```
inter-code (basis=X): [ data_l | data_r | κ_l_aug | κ_r_aug | adapter ]
intra-code (basis=X): [ shared_data       | κ_l_aug | κ_r_aug | adapter ]
```

with `κ_s_aug = κ_s ∪ extra_κ_s`, widths `n_s, |C_0^(s)|+e_s, w` respectively. For `basis=Z` swap
X ↔ Z everywhere.

`width = min(|port_l|, |port_r|)`. Default `port_subset_s = g_s.V0` gives `width ≥ min(d_l, d_r)`
since logical weight ≥ d.

## §2. `Bridge` dataclass + `build_bridge` algorithm

`build_bridge(g_l, g_r, ...)` runs 6 steps; the auxiliary graph machinery mirrors paper §IV proof
plus §V.A "add edges to make induced subgraph connected" trick.

```
1. Build NetworkX auxiliary graph 𝒢_s for each side:
   𝒢_s.nodes = range(|V_0^(s)|)
   for each row j ∈ C_0^(s) of F^(s):
       if F^(s)[j, :] has weight 2: add edge (a, b)
       else: raise NotImplementedError(hyperedge requires §II.C decomposition)

2. Select port subset 𝒫_s* = port_subset_s or V_0^(s):
   width = min(|𝒫_l*|, |𝒫_r*|)
   trim each side to first `width` ports in V_0 ordering (or as overridden)

3. Augment 𝒢_s on 𝒫_s* to be connected (§V.A trick):
   while not nx.is_connected(𝒢_s[𝒫_s*]):
       pick (u, v) from two different components, add edge (u, v) to 𝒢_s
       append row [1@u, 1@v] to extra_kappa_s

4. Cellulate long cycles (re-use _cellulate_long_cycles logic with new accounting):
   while any cycle longer than cellulate_max_len:
       add chord (u, v), append row to extra_kappa_s

5. Run SkipTree on the induced subgraph:
   T_s_ind, P_s_ind = _skip_tree_fullrank(𝒢_s[𝒫_s*], root=spanning_tree_root_s)
   # full-rank version: drop last row of cyclic T to get (w-1) × |E(𝒢_s[𝒫_s*])|
   # invariant on induced subgraph: T_s_ind · G_s_ind · P_s_ind = H_R
   # where G_s_ind is the |E_ind| × w incidence matrix of 𝒢_s[𝒫_s*]
   # and P_s_ind is the w × w SkipTree permutation
   # embed T_s_ind to T_s (with zero columns on edges outside 𝒢_s[𝒫_s*]):
   T_s ∈ F_2^{(w-1) × (|C_0^(s)| + e_s)}
   # embed P_s_ind to P_s (with zero rows for vertices outside 𝒫_s*):
   P_s ∈ F_2^{|V_0^(s)| × w}

6. Compute label_s[v] for v ∈ 𝒫_s*:
   label_s[v] = k such that P_s[v, k] == 1
   # for v ∉ 𝒫_s*, label_s[v] undefined (P_s row is zero)

7. Rebuild augmented gadget:
   F_s_aug = vstack([F_s, extra_kappa_s])      # shape (|C_0|+e, |V_0|)
   G_s_aug = ker(F_s_aug.T)                    # use _step2_gauge_fix on augmented matrix
   HX_aug, HZ_aug = _step3_assemble(code_s, V_0^(s), C_0_aug, F_s_aug, G_s_aug, basis)
   g_s_aug = GadgetLayout(..., F=F_s_aug, G=G_s_aug, HX_merged=HX_aug, HZ_merged=HZ_aug, ...)
```

SkipTree weight bound: each `T_s` row has weight ≤ 3 (Theorem 7's (3,2)-sparsity). Combined with
`H_R` rows of weight 2, adapter cycle-check rows are bounded above by `3 + 2 + 3 = 8`, matching
paper §VII.B (Hamiltonian-path optimization would give ≤ 4 but is not implemented; Remark 20).

`build_gadget_augmented(code, x, F_extra, basis)` is added to `gadget.py` as a thin wrapper around
`_step2_gauge_fix` + `_step3_assemble` over an augmented F.

## §3. `_stitch_to_joint_csscode` block structure

**Key correction over my initial mental model:** the new adapter cycle-checks are **Z-type** (not
X-type) for `basis=X`. The paper's §IV uses Z̄ measurement so vertex checks `A_v` are Z-type and
cycle checks `B_c` are X-type; our `basis=X` (X̄ measurement) is the symmetric dual, so cycle
checks land in `H_Z^merged`.

### Inter-code, basis=X

H_X^merged rows (count `m_X^(l) + m_X^(r) + |V_0^(l)| + |V_0^(r)|`):

| block | rows | data_l | data_r | κ_l_aug | κ_r_aug | adapter |
|---|---|---|---|---|---|---|
| data H_X^(l) | m_X^(l) | H_X^(l) | 0 | 0 | 0 | 0 |
| data H_X^(r) | m_X^(r) | 0 | H_X^(r) | 0 | 0 | 0 |
| χ^(l) | \|V_0^(l)\| | E_V0_l^T | 0 | F_aug^(l)T | 0 | Π_l |
| χ^(r) | \|V_0^(r)\| | 0 | E_V0_r^T | 0 | F_aug^(r)T | Π_r |

`Π_s` is `|V_0^(s)| × w` with `Π_s[v, k] = 1` iff `v ∈ 𝒫_s*` and `label_s[v] == k`.

H_Z^merged rows (count `m_Z^(l) + m_Z^(r) + r_l_aug + r_r_aug + (w-1)`):

| block | rows | data_l | data_r | κ_l_aug | κ_r_aug | adapter |
|---|---|---|---|---|---|---|
| data H_Z^(l) ext | m_Z^(l) | H_Z^(l) | 0 | tilde_F_aug^(l) | 0 | 0 |
| data H_Z^(r) ext | m_Z^(r) | 0 | H_Z^(r) | 0 | tilde_F_aug^(r) | 0 |
| G_aug^(l) | r_l_aug | 0 | 0 | G_aug^(l) | 0 | 0 |
| G_aug^(r) | r_r_aug | 0 | 0 | 0 | G_aug^(r) | 0 |
| **new cycle Z-checks** | w-1 | 0 | 0 | T_l | T_r | H_R |

`tilde_F_aug^(s)`: shape `m_Z^(s) × (|C_0^(s)| + e_s)`. Column `k < |C_0^(s)|` embeds the
corresponding original Z-check (one-hot); columns ≥ |C_0^(s)| (extra κ) are all zero (no original
Z-check is associated with new edges).

### Intra-code, basis=X

Replace `[data_l | data_r]` with a single `shared_data` block of width n. `χ^(s)` rows extend onto
`shared_data` at `V_0^(s)` positions. When `q ∈ V_0^(l) ∩ V_0^(r)`, the rows `χ_q^(l)` and
`χ_q^(r)` are **distinct rows of H_X^merged**, both with X on `shared_data[q]`. Summing them in
α* cancels the X on `q`, giving `x_l ⊕ x_r` on data — the (XOR-support) joint X̄_l X̄_r.

### CSS commutation invariants

| pair | proof |
|---|---|
| data H_X^(s) · data H_Z^(s)^T | original CSS code |
| data H_X^(s) · G_aug^(s)^T | disjoint support (data vs κ) |
| χ^(s) · data H_Z^(s)^T | gadget invariant §1.5 (b), extended to F_aug because `tilde_F_aug` is zero on extra columns and F_aug embeds H_Z[j,v] consistently |
| χ^(s) · G_aug^(s)^T | G_aug = ker(F_aug^T) by construction |
| χ^(s) · cycle-Z^T | the **SkipTree key identity**: `(T_s · F_aug^(s))[c, v] + H_R[c, label_s(v)] · [v ∈ 𝒫_s*] = 0` mod 2, which is exactly `T_s · G_s_aug · P_s = H_R` (where G_s_aug = F_aug^(s) here is the auxiliary-graph incidence matrix) |
| cycle-Z · cycle-Z^T (different sides) | both Z-type — always commute |
| data H_X^(s) · cycle-Z^T | disjoint support (data vs κ + adapter) |

### Hyperedge handling

The build raises `NotImplementedError("hyperedge requires §II.C decomposition")` if any F row has
weight ≥ 3. For the curated BB / LP code + logical pairs used in `examples/`, this never triggers
(paper §VII.A confirms BB_1 [[98,6,12]] and LP_2 [[200,20,10]] both have only weight-2 F rows on
the chosen logicals).

## §4. Joint observable + α* selection

**α* picks `χ^(l) ∪ χ^(r)` only.** No `U_B` analog because every adapter qubit `a_k` is touched
by exactly two χ rows (one from each side via the bijection), so the adapter contribution to
`Σ_α χ` is automatically zero.

`Σ_α χ = (x_l + x_r, 0_{κ_l_aug}, 0_{κ_r_aug}, 0_{adapter})` on the merged register.

- data side: `x_l + x_r` (XOR support).
- κ side: `F_aug^(s)^T · 1_{V_0^(s)} = 0` (κ-cancellation, inherited from gadget invariant).
- adapter side: each qubit hit twice → zero.

New cycle Z-checks are **not** in α* — they're Z-type stabilizers serving LDPC structure, not part
of the X-type joint observable.

### stim circuit changes (`circuit.py`)

```python
# current (path-graph):
chi_check_ids = chi1_ids + chi2_ids + ub_ids

# new (universal adapter):
chi_check_ids = chi1_ids + chi2_ids
```

`chi1_ids` and `chi2_ids` row-offset calculation follows the new H_X^merged ordering (data H_X
blocks, then χ blocks). `_surgery_observable` itself is unchanged: it accepts any `chi_check_ids`
collection and emits `OBSERVABLE_INCLUDE 0` from per-round XORs. Observable 1 (final M-X readout
on `V_0^(l) ∪ V_0^(r)`) is unchanged — intra-code overlap of `V_0` sets is auto-handled because
each appearance of a qubit gets XOR'd once.

## §5. Testing strategy

### Structural unit tests (added to `surgery/_test.py`)

| test | invariant |
|---|---|
| `test_skiptree_invariant` | `T·G·P == H_R` on K_4, cycle_10, BB-aux subgraph; row-weight ≤ 3; col-weight ≤ 2 |
| `test_bridge_induced_connected` | `nx.is_connected(g_l_aug.aux[port_l])` after augmentation |
| `test_joint_css_commutation` | `(H_X @ H_Z.T) % 2 == 0` for the stitched merged code |
| `test_joint_dim_reduces_by_one` | `code.dimension == g_l.code.dimension + g_r.code.dimension - 1` (inter-code) or `code.dimension == g_l.code.dimension - 1` (intra-code) |
| `test_joint_logical_in_stabilizer` | rank check: row span of H_X includes the (x_l, x_r, 0, 0, 0) joint logical vector |
| `test_singletons_excluded` | rank strictly increases when (x_l, 0, 0, 0, 0) alone is appended |
| `test_adapter_cycle_check_weight_bounded` | every row of the last `w-1` Z-stabilizers has weight ≤ 8 |
| `test_cellulation_caps_cycle_length` | every basis cycle of `g_s_aug.aux` has length ≤ `cellulate_max_len` |

### LER smoke tests (`pytest.mark.slow`)

| test | check |
|---|---|
| `test_joint_ppm_noiseless` | `detectors.sum() == 0` over 100 shots, no noise; obs[0] matches expected x_l ⊕ x_r initial value |
| `test_joint_ppm_ler_monotone` | LER non-increasing across `p ∈ [1e-4, 3e-4, 1e-3]` for (Steane, Steane) and (BB_18, BB_18) — tolerance 1.3× to absorb sampling noise |

### Removed regressions

Tests assuming path-graph behavior (`U_B` row count, endpoint-only χ extension) are deleted, not
adapted, to prevent silent re-introduction of the broken construction.

## §6. Migration impact

### New / rewritten files

| file | change |
|---|---|
| `src/qldpc/codes/surgery/bridge.py` | full rewrite (~250 lines). Delete `_build_path_graph_U_B`, `_solve_chi_z_bridge_choices`, `chi_endpoint_extensions`, the old `Bridge` fields. Keep `_skip_tree` but add `_skip_tree_fullrank` wrapper. New `build_bridge` implements §2 steps 1–7. |
| `src/qldpc/codes/surgery/circuit.py` | rewrite `_stitch_to_joint_csscode` per §3 block tables. Update `build_joint_ppm_circuit` to (a) consume new Bridge fields (`g_l_aug`, `g_r_aug`, `T_l`, `T_r`, `H_R`, port labels) and (b) build `chi_check_ids` without `ub_ids`. |
| `src/qldpc/codes/surgery/gadget.py` | add `build_gadget_augmented(code, x, F_extra, basis)` thin wrapper over `_step2_gauge_fix` + `_step3_assemble` for the augmented F path. |
| `src/qldpc/codes/surgery/_test.py` | add §5 tests; delete path-graph-specific tests. |
| `docs/superpowers/math.md` | replace §2.2–2.7 with universal-adapter math (block forms from §3, SkipTree key identity, α* derivation). Keep old §2 as historical footnote. |

### Demo migrations

| script | change |
|---|---|
| `examples/scripts/joint_ppm_z_with_superposition_demo.py` | new Bridge fields; observable formula simplifies (no U_B); expected LER range updated |
| `examples/scripts/single_ppm_vs_memory_ler_webster.py` | no `build_bridge` calls; unaffected |
| `examples/logical_error_rates/_9_mylattice_surgery_test.ipynb` | same diff as `joint_ppm_z_with_superposition_demo.py` |
| `examples/scripts/*_demo.py` (others) | `git grep build_bridge` → update each |

### Out of scope (deferred)

- Hyperedge decomposition for weight ≥ 3 F rows (paper §II.C). Build raises clearly until a code
  pair needing it shows up.
- Hamiltonian-path SkipTree optimization (Remark 20). Default to NetworkX BFS spanning tree;
  adapter check weight is ≤ 8 rather than ≤ 4.
- Decongestion / thickening (paper §V.B). Cellulation alone suffices for the curated examples.
- Relative-expansion certificates (Lemma 9 quantitative form). Distance is verified empirically by
  LER sampling, not proved structurally.
- BP-OSD / CPLEX distance verification. Not integrated; tests rely on structural invariants +
  LER monotonicity.

### LOC estimate

- `bridge.py`: +~250 / -~210 = net +40
- `circuit.py`: +~80 / -~50 = net +30
- `gadget.py`: +~30
- `_test.py`: +~150 / -~60 = net +90
- `math.md`: +~80 / -~80 = net 0
- Demo scripts: ~5 scripts × ~15 lines each = ~75 lines edited
