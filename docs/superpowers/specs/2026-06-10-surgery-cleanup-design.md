# Surgery Module Cleanup — Design

**Status**: Draft (2026-06-10)
**Scope**: `src/qldpc/circuits/surgery/` (cheeger.py, circuit.py, _test.py)
**Author**: tgzhou (with Claude)

## Background

The surgery module is in good shape after the recent moves (`codes/surgery → circuits/surgery`, parallel to `circuits/memory`) and the universal-adapter Bridge rewrite. Public API is minimal (7 names: `build_gadget`, `build_bridge`, `build_single_ppm_circuit`, `build_joint_ppm_circuit`, `keep_only_observable`, `boost_gadget`, `cheeger_constant`), file boundaries are crisp (`gadget.py` 239, `bridge.py` 426, `circuit.py` 729, `cheeger.py` 811, `_test.py` 1665), and math.md / paper §-references are inlined throughout.

What remains is debt accumulated through legitimate evolution:

1. **`cheeger.py` legacy adapter** — three boost implementations still operate on the obsolete `(merged: CSSCode, layout: SurgeryLayout)` pair, even though every external caller produces and consumes `GadgetLayout`. The dispatcher `boost_gadget` translates GadgetLayout→legacy at the entrance and legacy→GadgetLayout at the exit (`_gadget_to_legacy_layout` / `_legacy_to_gadget`, ~110 LOC of glue). The `basis=Pauli.Z` path additionally swaps `HX_merged ↔ HZ_merged` on entry and back on exit, because the legacy boost was written under the basis=X assumption "χ rows live in HX_merged."
2. **Unused 'spectral' boost strategy** — `boost_gadget(g, method='spectral', ...)` exists but no script, notebook, or example calls it; only tests parametrize over it. Webster's recommendation and every production call site use `method='combinatorial'`, which gives a deterministic Cross Thm 6 distance guarantee. The `'distance'` strategy (BP+OSD verifier) has no current external user either, but is retained as a verification fallback.
3. **`_test.py` 1665 LOC single file** — 77 tests across four logical groups (gadget / bridge / circuit / cheeger) all in one file with mixed top-of-file fixtures. Hard to navigate; touching any feature requires opening the whole file.
4. **Four `_stitch_*` helpers in `circuit.py`** — `_stitch_to_joint_csscode` (inter-code, basis=X), `_stitch_intracode_joint_csscode` (intra-code, basis=X), `_stitch_intercode_joint_csscode_basis_z` (inter-code, basis=Z), `_stitch_intracode_joint_csscode_basis_z` (intra-code, basis=Z) form a 2×2 dispatch matrix. The `basis=X / basis=Z` axis is a pure duality (χ rows go to `HX_merged` vs `HZ_merged`); the `intercode / intracode` axis is a real structural difference (intracode shares data check rows and data columns).

## Goal

Pay down all four debts in a single PR with four atomic commits, ordered so that each commit can be reviewed and tested independently and each later commit benefits from the earlier ones.

## Non-goals

- **No public API change.** The 7-name public surface stays bit-identical: same parameter names, same defaults, same return types. `boost_gadget(g, method=..., target=..., seed=..., **kw) -> GadgetLayout` keeps its current contract.
- **No algorithm change.** The Cheeger combinatorial boost still uses `_exact_boundary_cheeger` + greedy worst-cut augmentation; the distance boost still uses BP+OSD verification with the same `num_trials_per_step` / `decoder_trials` defaults. Cheeger constant computation is unchanged.
- **No new tests for behavior already covered.** Existing tests carry the safety net; we only relocate and adjust them mechanically (test-split + the spectral parametrize removal).
- **No removal of the `'distance'` boost strategy.** It is kept as the BP+OSD-verified fallback, even though no external user currently calls it.
- **No change to `bridge.py` or `gadget.py` semantics.** `build_gadget_augmented` (already in `gadget.py`) is reused by commit 3 but not modified.

## Target state

| File | Before | After |
|---|---|---|
| `cheeger.py` | 811 | ~520 (-290) |
| `circuit.py` | 729 | ~653 (-76) |
| `_test.py` | 1665 | (removed; split below) |
| `_test_helpers.py` | — | ~30 (new) |
| `_test_gadget.py` | — | ~280 (new) |
| `_test_bridge.py` | — | ~440 (new) |
| `_test_circuit.py` | — | ~900 (new) |
| `_test_cheeger.py` | — | ~80 (new, narrowed) |

Breakdown of the cheeger.py reduction:

- Commit 2 (drop spectral): −85 (`boost_gadget_cheeger` body + dispatcher branch).
- Commit 3 (native GadgetLayout): −205 = `SurgeryLayout` (18) + `_compute_gauge_fix` (3) + `_build_layout` (22) + `_reassemble_gadget_with_new_F` (52) + `_gadget_to_legacy_layout` (44) + `_legacy_to_gadget` (25) + `BoostResult` (12) + `DistanceBoostResult` (12) + net body cleanup inside the two surviving boost functions (~17).

Symbols removed entirely: `SurgeryLayout`, `BoostResult`, `DistanceBoostResult`, `_build_layout`, `_compute_gauge_fix`, `_reassemble_gadget_with_new_F`, `_gadget_to_legacy_layout`, `_legacy_to_gadget`, `boost_gadget_cheeger` (the 'spectral' implementation), `_stitch_intracode_joint_csscode_basis_z`, `_stitch_intercode_joint_csscode_basis_z`. The existing `_spectral_cheeger_lower_bound` and `_exact_boundary_cheeger` helpers stay (consumed by `cheeger_constant`).

## Success criteria

- `pytest src/qldpc/circuits/surgery/` collects the same set of test function names before and after (modulo: `test_boost_gadget_dispatches_to_three_methods` → `_two_methods`; the `method='spectral'` parametrize value disappears).
- `python examples/scripts/cain_bb18_resource_exact_match.py` still prints `(Qubits=39, X-checks=20, Z-checks=20)`.
- `python examples/scripts/single_ppm_vs_memory_ler.py` still runs and produces output unchanged within sampling noise.
- After commit 2: `grep -n "spectral" src/qldpc/circuits/surgery/cheeger.py` returns only the `_spectral_cheeger_lower_bound` definition + the one comment line in `cheeger_constant` describing its fallback role.
- After commit 3: `grep -n "SurgeryLayout\|_gadget_to_legacy_layout\|_legacy_to_gadget" src/qldpc/circuits/surgery/` returns empty.
- After commit 4: `grep -n "basis_z" src/qldpc/circuits/surgery/circuit.py` returns empty (the `*_basis_z` function names are gone; basis dispatch is in-function).

## Approach: four atomic commits

Each commit is independently testable. Before each commit lands, `pytest src/qldpc/circuits/surgery/ -v` must pass.

### Commit 1 — `refactor(surgery/tests): split _test.py by feature`

**Risk: very low** (pure position move, no logic change).

Split `_test.py` (1665 LOC, 77 tests) into:

- `_test_helpers.py` (~30 LOC): the `sys.path.insert(...)` Webster fixture loader from line 17, plus the three Webster operator helpers `_webster_x_bar_operator`, `_webster_z_bar_operator`, `_webster_x_bar_1_operator` (currently lines 260-294 of `_test.py`). All other test files `from ._test_helpers import ...`.
- `_test_gadget.py` (~13 tests, ~280 LOC): all tests that exercise `gadget.py` — `test_gadget_layout_*`, `test_step1_*`, `test_step2_*`, `test_step3_*`, `test_build_gadget_*`, `test_build_gadget_augmented_*`, `test_webster_table_i_kappa_chi_r_exact`, `test_webster_table_i_z_basis_*`, basis=Z step/build tests.
- `_test_bridge.py` (~17 tests, ~440 LOC): `test_build_aux_graph_*`, `test_connect_induced_subgraph_*`, `test_cellulate_*`, `test_cellulate_port_subgraph_*`, `test_skip_tree_fullrank_*`, `test_bridge_dataclass_*`, `test_build_bridge_*`, `test_adapter_cycle_check_weight_bounded`, `test_cellulation_caps_aug_aux_cycle_length_on_webster`.
- `_test_circuit.py` (~37 tests, ~900 LOC): `test_build_single_ppm_circuit_*`, `test_classify_reliable_round1_checks_*`, `test_surgery_state_prep_*`, `test_surgery_qec_cycle_*`, `test_surgery_detach_and_readout_*`, `test_surgery_observable_*`, `test_surgery_final_detectors_*`, `test_stitch_inter*_*`, `test_stitch_intra*_*`, `test_build_joint_ppm_circuit_*`, `test_single_ppm_ler_*`, `test_joint_ppm_ler_*`, `test_joint_xx_in_stabilizer_*`.
- `_test_cheeger.py` (~5 tests, ~80 LOC): `test_cheeger_constant_matches_boost_target`, `test_boost_gadget_dispatches_to_three_methods`, `test_boost_gadget_seed_reproducible`, `test_boost_gadget_preserves_css_commutation`, `test_boost_gadget_preserves_css_commutation_both_bases`.

**Verification:** `pytest src/qldpc/circuits/surgery/ --collect-only -q` produces the same set of test IDs before/after (set-equal modulo file path).

### Commit 2 — `refactor(surgery/cheeger): drop unused 'spectral' boost strategy`

**Risk: low** (no external user; helper retained for `cheeger_constant`).

- Delete `boost_gadget_cheeger` (~80 LOC, the 'spectral' implementation).
- Delete the `if method == "spectral":` branch in `boost_gadget`.
- Update the dispatcher error message: `unknown method: {method!r}. Allowed: 'combinatorial', 'distance'.`
- **Keep** `_spectral_cheeger_lower_bound` — it remains the `|V_0| > 26` fallback inside the public `cheeger_constant(g)`.
- **Keep** `_exact_boundary_cheeger` — consumed by `cheeger_constant` and by `boost_gadget_cheeger_combinatorial`.

Test changes in `_test_cheeger.py`:

- `test_boost_gadget_dispatches_to_three_methods` → `test_boost_gadget_dispatches_to_two_methods`; loop `for method in ("combinatorial", "distance")`.
- `test_boost_gadget_seed_reproducible`: replace `method="spectral"` with `method="combinatorial"` (`combinatorial` is also deterministic given seed).
- `test_boost_gadget_preserves_css_commutation`: change parametrize from `["spectral", "combinatorial", "distance"]` to `["combinatorial", "distance"]`.
- `test_boost_gadget_preserves_css_commutation_both_bases` already only runs combinatorial; unchanged.

### Commit 3 — `refactor(surgery/cheeger): boost paths consume GadgetLayout natively`

**Risk: medium** (most substantive change; full test suite + Cain script must pass).

Strategy: each remaining boost function computes `F_extra` (the new weight-2 rows being added) and hands off to `build_gadget_augmented(g.code, g.x, F_extra, basis=g.basis)`, which already exists in `gadget.py:188` and already handles basis=X/Z duality, the "new κ' carry no data-Z extension" sentinel rule, and the `G_aug = ker(F_aug^T)` recomputation.

**New signatures:**

```python
def boost_gadget_combinatorial(
    g: GadgetLayout, *,
    target_h: float = 1.0,
    max_extra_qubits: int = 50,
    seed: int | None = None,
) -> GadgetLayout: ...

def boost_gadget_distance(
    g: GadgetLayout, *,
    target_distance: int,
    max_extra_qubits: int = 30,
    num_trials_per_step: int = 20,
    decoder_trials: int = 10,
    seed: int | None = None,
) -> GadgetLayout: ...
```

**`boost_gadget` dispatcher becomes:**

```python
def boost_gadget(g, *, method, target, seed=None, **kw):
    if method == "combinatorial":
        return boost_gadget_combinatorial(g, target_h=target, seed=seed, **kw)
    if method == "distance":
        return boost_gadget_distance(g, target_distance=int(target), seed=seed, **kw)
    raise ValueError(
        f"unknown method: {method!r}. Allowed: 'combinatorial', 'distance'."
    )
```

**Symbols removed (cheeger.py):** `SurgeryLayout` (class), `BoostResult` (class), `DistanceBoostResult` (class), `_compute_gauge_fix`, `_build_layout`, `_reassemble_gadget_with_new_F`, `_gadget_to_legacy_layout`, `_legacy_to_gadget`, and all `layout.F` / `layout.num_data_x_checks` / `field = layout.F.__class__` accesses inside the boost bodies (replaced with `g.F` / `galois.GF(2)`).

**basis=Z handling disappears entirely.** `F` is basis-agnostic (rows = κ qubits, columns = V_0 vertices); the χ-vs-G placement decision lives in `_step3_assemble` (called via `build_gadget_augmented`) and is keyed off `g.basis`.

**Distance boost detail.** The current `boost_gadget_distance` tries random augmentations and rejects via BP+OSD. The new version constructs each candidate via `build_gadget_augmented(g.code, g.x, F_extra_candidate, basis=g.basis)` and feeds `.HX_merged` / `.HZ_merged` to the existing decoder helpers. This is a few percent slower per trial than the legacy "patch in place" path, but `'distance'` has no external user (no production hot path) and code clarity wins.

**New test (added to `_test_cheeger.py`)**: after the boost, the returned `GadgetLayout` for a basis=Z input has `basis is Pauli.Z` and χ rows live in `HZ_merged` (not `HX_merged`). One assertion per boost method. This is the safety net for the removed HX/HZ swap.

**Verification:**

- `pytest src/qldpc/circuits/surgery/`
- `python examples/scripts/cain_bb18_resource_exact_match.py` → `(39, 20, 20)` reproduces.

**Rollback:** atomic commit; `git revert` if it goes sideways. Commits 1 and 2 unaffected.

### Commit 4 — `refactor(surgery/circuit): consolidate 4 stitch helpers via (intercode, basis) dispatch`

**Risk: medium** (basis duality must be preserved exactly; safety net is the existing parametrized stitch + joint-PPM tests).

The 2×2 = 4 stitch helpers collapse to 2 + dispatcher. Strategy: the `intercode` / `intracode` axis is structural (column layout differs) so it stays as two functions; the `basis=X` / `basis=Z` axis is pure duality and folds into a "χ-carrier vs co-carrier" abstraction inside each function.

**New shape:**

```python
def _stitch_intercode(g_l, g_r, bridge): ...   # handles basis=X and basis=Z internally
def _stitch_intracode(g_l, g_r, bridge): ...   # handles basis=X and basis=Z internally

def _stitch_to_joint_csscode(g_l, g_r, bridge):
    if g_l.code is g_r.code:
        return _stitch_intracode(g_l, g_r, bridge)
    return _stitch_intercode(g_l, g_r, bridge)
```

**The χ-carrier abstraction (inside each new function):**

```python
if bridge.basis is Pauli.X:
    M_chi_l, M_co_l = g_l_aug.HX_merged, g_l_aug.HZ_merged
    M_chi_r, M_co_r = g_r_aug.HX_merged, g_r_aug.HZ_merged
    m_chi_l_data = g_l.code.matrix_x.shape[0]
    m_chi_r_data = g_r.code.matrix_x.shape[0]
    m_co_l_data  = g_l.code.matrix_z.shape[0]
    m_co_r_data  = g_r.code.matrix_z.shape[0]
else:
    M_chi_l, M_co_l = g_l_aug.HZ_merged, g_l_aug.HX_merged
    M_chi_r, M_co_r = g_r_aug.HZ_merged, g_r_aug.HX_merged
    m_chi_l_data = g_l.code.matrix_z.shape[0]
    m_chi_r_data = g_r.code.matrix_z.shape[0]
    m_co_l_data  = g_l.code.matrix_x.shape[0]
    m_co_r_data  = g_r.code.matrix_x.shape[0]

# ... assemble M_chi (χ rows + adapter Π) and M_co (cycle Z-checks + G_aug) ...

if bridge.basis is Pauli.X:
    return CSSCode(field(M_chi), field(M_co), is_subsystem_code=False)
return CSSCode(field(M_co), field(M_chi), is_subsystem_code=False)
```

The middle 30-60 lines of block matrix assembly write to `M_chi` / `M_co` once, regardless of basis. Only the χ-carrier selection (top) and the final CSSCode construction (bottom) branch on basis.

**Symbols removed:** `_stitch_intracode_joint_csscode`, `_stitch_intercode_joint_csscode_basis_z`, `_stitch_intracode_joint_csscode_basis_z`. The current `_stitch_to_joint_csscode` (which does both inter-X assembly and dispatch) is split: dispatch logic to the new ~4-line shell, inter-X assembly merged into `_stitch_intercode`.

**Safety net (existing tests, not modified):**

- `test_stitch_intercode_both_bases_commute_and_singletons_excluded[basis]` (parametrized over X, Z)
- `test_stitch_intracode_both_bases_commute[basis]`
- `test_build_joint_ppm_circuit_intercode_noiseless_observables_zero` (basis=X end-to-end)
- `test_build_joint_ppm_circuit_intracode_noiseless_observables_zero` (basis=X end-to-end)
- `test_joint_ppm_ler_monotone_steane_intercode` (basis=X LER monotone)
- `test_joint_xx_in_stabilizer_on_webster_intracode` (basis=X joint logical in stabilizer)
- `test_adapter_cycle_check_weight_bounded` (Webster BB invariant)

If basis-Z is broken by a wrong χ-carrier selection, `*_both_bases_*` parametrize values fail. If basis-X is broken, the joint-PPM end-to-end tests fail. The combined coverage protects both axes.

**Verification:** `pytest src/qldpc/circuits/surgery/` + the two example scripts.

## Order and PR shape

One branch `refactor/surgery-cleanup`, four commits in the order above. Single PR. `pytest src/qldpc/circuits/surgery/` must be green between commits during local development.

Order rationale:

1. **Commit 1 (test-split) first** — narrows the test-file blast radius for each later commit: commit 2 only touches `_test_cheeger.py`, commit 4 only touches `_test_circuit.py`.
2. **Commit 2 (drop spectral) next** — pure deletion. Smaller cheeger.py going into commit 3 means less to convert.
3. **Commit 3 (native GadgetLayout boost)** — the substantive cheeger.py cleanup. Removes the SurgeryLayout adapter layer entirely.
4. **Commit 4 (stitch dedup) last** — independent of commits 2-3, but ordering last means a regression here doesn't muddy the cheeger.py history.

## Verification matrix

| Commit | Local tests | Example scripts |
|---|---|---|
| 1 | `pytest src/qldpc/circuits/surgery/` collects same set of test IDs | — |
| 2 | `pytest src/qldpc/circuits/surgery/_test_cheeger.py` | — |
| 3 | `pytest src/qldpc/circuits/surgery/` | `cain_bb18_resource_exact_match.py` → (39, 20, 20); `single_ppm_vs_memory_ler.py` runs |
| 4 | `pytest src/qldpc/circuits/surgery/` | both joint-PPM example paths still produce baseline LER |

## Open questions

None — all four items have a single recommended approach above. If commit 3's "build candidate via `build_gadget_augmented` per BP+OSD trial" turns out measurably slow on a realistic distance-boost run, add a private fast-reassemble path inside `boost_gadget_distance` as a follow-up; do not block the cleanup on it.
