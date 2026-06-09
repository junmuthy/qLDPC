# Cheeger.py L=1 Collapse — Design

**Status**: Draft (2026-06-09)
**Scope**: `src/qldpc/codes/surgery/cheeger.py` + small extraction into `gadget.py`
**Author**: tgzhou (with Claude)

## Background

The current `surgery/` module follows a "Webster L=1 + Cheeger boost" recipe end-to-end:

- `gadget.py:build_gadget` produces a `GadgetLayout` with `HX_merged`, `HZ_merged` from the 3-step Webster construction (§II.A of arXiv:2511.15989).
- `cheeger.py:boost_gadget` augments the gadget's restriction matrix `F` by adding edges until the Cheeger constant `h(F) ≥ target` (per Cain et al. arXiv:2603.28627 §III).

`cheeger.py` (953 LOC) contains two intermingled layers:

1. **The real boost machinery** (~700 LOC): Cheeger constant computation (exact + spectral lower bound), three boost algorithms (spectral / combinatorial / distance), the `boost_gadget` dispatcher.
2. **An L≥1 "general layered" wrapper** (~200 LOC): `SurgeryLayout`, `_LayeredBlocks`, `_assemble_merged_HX/HZ`, `_build_layout`, plus odd/even-layer loops in the boost-internal rebuild path.

Every call site of the layered infrastructure pins `num_layers = 1` (the one assignment site is `cheeger.py:877`). The L≥1 wrapper is dead concept: it exists "in case L>1 is needed later," but Cain's protocol — the only protocol this repo implements end-to-end — uses L=1 + Cheeger boost exclusively. Going to L≥d would cost ~8× more ancillas than the Cheeger boost path (verified empirically: bb_18 boosted to 39 ancillas vs. L=d ≈ 324).

## Goal

Collapse `cheeger.py` so that "L=1 + Cheeger boost" is the **only** path in the code — no `num_layers` parameter, no `_LayeredBlocks`, no odd/even-layer loops, no row-kind string arrays that exist only to identify "which layer."

## Non-goals

- No change to the public API (`build_gadget`, `boost_gadget`, `build_bridge`, `build_single_ppm_circuit`, `build_joint_ppm_circuit`, `load_webster_seed_set`).
- No change to the boost algorithms themselves (spectral, combinatorial, distance).
- No removal of `SurgeryLayout` (still needed by the gadget↔legacy bridge); it gets slimmer.
- No change to the basis=Z HX/HZ swap handling (`_gadget_to_legacy_layout`'s isolated hack stays).
- No change to test behavior — all existing surgery tests pass unchanged.

## Target state

- `cheeger.py`: 953 → ~795 LOC (−158).
- `gadget.py`: +10 LOC (one extracted helper).
- `SurgeryLayout` fields: 10 → 7 (drop `num_layers`, `qubit_layer`, `hx_row_kind`, `hz_row_kind`; add `num_data_x_checks: int`, `num_data_z_checks: int`).
- `_LayeredBlocks`, `_build_layered_blocks`, `_assemble_merged_HX`, `_assemble_merged_HZ` deleted.
- `_assemble_HX_L1` extracted to `gadget.py` and used by both `_step3_assemble` and `_reassemble_gadget_with_new_F`.
- One stale doc comment in `examples/logical_error_rates/_9_lattice_surgery_cain_fig1b_source.py:92-93` deleted.

## Success criteria

- `pytest src/qldpc/codes/surgery/_test.py` passes unmodified.
- `python examples/scripts/cain_bb18_resource_exact_match.py` still prints `(Qubits=39, X-checks=20, Z-checks=20)`.
- `python examples/scripts/cain_fig1b_full_protocol.py` (short run, 1-2 noise points) produces LER values within numerical noise of the baseline.
- After Step 1, the symbol `num_layers` does not appear in `cheeger.py` source.
- After Step 2, `_assemble_merged_HX` and `_assemble_merged_HZ` are not defined anywhere.

## Approach: Step-by-step

The refactor is split into two commits so risk is isolated. Step 1 is mechanical (output bit-identical); Step 2 has a single semantic judgment (sharing the HX assembly across two consumers).

### Step 1 — Mechanical collapse (cheeger.py only)

**Pure mechanical** — substitute `num_layers = 1` into all loops and delete dead branches. Output matrices are bit-identical to the pre-refactor versions for any (code, x) input.

1. **`SurgeryLayout` slim-down** (lines 19-51)
   - Delete fields: `num_layers`, `qubit_layer`, `hx_row_kind`, `hz_row_kind`.
   - Add fields: `num_data_x_checks: int`, `num_data_z_checks: int` (replace the `row_kind == "data"` filter with simple integer counts).
   - Keep: `num_data_qubits`, `num_ancilla_qubits`, `v0_indices`, `c0_indices`, `F`, `G`.

2. **Delete `_LayeredBlocks` + `_build_layered_blocks`** (lines 59-90)
   - At L=1 the class reduces to `(n_v0, n_c0, F, F_T)`; inline these into call sites.
   - `ancilla_col_slice(1)` becomes `slice(0, n_c0)` directly.

3. **`_assemble_merged_HX` collapse** (lines 93-121, ~29 → ~12 LOC)
   - Odd-layer loop `for i in range(1, num_layers + 1, 2)` executes only `i=1`; inline.
   - Drop the `if i == 1 / else` branch (only i=1 survives).
   - Drop the `if i + 1 <= num_layers` branch (false at L=1).

4. **`_assemble_merged_HZ` collapse** (lines 124-157, ~34 → ~10 LOC)
   - Even-layer loop `for i in range(2, num_layers, 2)` has empty range at L=1; delete entirely.
   - Resulting function: data block + F̃ embedding on c0_indices + G gauge rows.

5. **`_build_layout` collapse** (lines 160-197, ~38 → ~8 LOC)
   - Delete the `qubit_layer` filling loop.
   - Delete the `for i in range(1, num_layers + 1, 2)` row_kind loop.
   - Compute `num_data_x_checks = data_code.matrix_x.shape[0]`, `num_data_z_checks = data_code.matrix_z.shape[0]` directly.

6. **`_reassemble_gadget_with_new_F` collapse** (lines 426-498)
   - `layout.hx_row_kind == "data"` boolean mask → `slice(0, layout.num_data_x_checks)`.
   - Even-layer loop (lines 473-483) → delete (empty range at L=1).
   - `blocks.ancilla_col_slice(1)` → `slice(0, n_kappa_orig)`.

7. **`_gadget_to_legacy_layout` adjust** (lines 830-887)
   - Drop `num_layers=1`, `qubit_layer=...` from the `SurgeryLayout(...)` construction.
   - Compute `num_data_x_checks`, `num_data_z_checks` from the original code's check matrices (or from `g.HX_merged.shape[0] - len(g.V0)` for basis=X, with the basis=Z analogue).
   - basis=Z HX/HZ swap logic preserved verbatim.

**Verification for Step 1**
- `pytest src/qldpc/codes/surgery/_test.py -x` → all green.
- One-off sanity script (not committed): for each test fixture (code, x), assert `np.array_equal(HX_merged_old, HX_merged_new)` and `np.array_equal(HZ_merged_old, HZ_merged_new)`.

### Step 2 — Assembly dedupe (cheeger.py + gadget.py)

**One semantic judgment**: the L=1 HX assembly is identical between `gadget.py:_step3_assemble` and `cheeger.py:_assemble_merged_HX`. Extract once, use twice. The HZ paths differ (the boost-rebuild treats new gauge qubits with no data-Z extension), so HZ assembly is NOT shared.

1. **Add `_assemble_HX_L1` to `gadget.py`** (~10 LOC, pure function over numpy arrays):
   ```python
   def _assemble_HX_L1(
       HX_data: np.ndarray,
       v0_indices: np.ndarray,
       F: np.ndarray,
   ) -> np.ndarray:
       """L=1 HX assembly: [[HX_data, 0], [E_V0, F^T]] over GF(2)."""
       mX, n = HX_data.shape
       n_v0, n_c0 = F.shape[1], F.shape[0]
       n_merged = n + n_c0
       top = np.hstack([HX_data, np.zeros((mX, n_c0), dtype=np.uint8)])
       bot = np.zeros((n_v0, n_merged), dtype=np.uint8)
       bot[np.arange(n_v0), v0_indices] = 1
       bot[:, n:] = F.T
       return np.vstack([top, bot]).astype(np.uint8)
   ```

2. **`_step3_assemble` calls `_assemble_HX_L1`**
   - basis=X: `HX_merged = _assemble_HX_L1(HX_data, np.array(V0), F)`.
   - basis=Z (dual): `HZ_merged = _assemble_HX_L1(HZ_data, np.array(V0), F)` (the dual symmetry swaps the role of HX and HZ).
   - The F̃ + G block in the other matrix is built inline as today (small, basis-dependent).

3. **`cheeger.py` cleanup**
   - Delete `_assemble_merged_HX` (lines 93-121, post-Step-1 form).
   - Delete `_assemble_merged_HZ` (lines 124-157, post-Step-1 form) — confirmed dead, zero call sites (comment at line 461 already says "Manually build HZ_new instead of _assemble_merged_HZ").
   - `_reassemble_gadget_with_new_F` calls `from .gadget import _assemble_HX_L1` and uses it.

**Verification for Step 2**
- `pytest src/qldpc/codes/surgery/_test.py -k "basis_z or boost"` (focused on basis=Z dual and boost paths).
- `python examples/scripts/cain_bb18_resource_exact_match.py` → still `(39, 20, 20)`.
- `python examples/scripts/cain_fig1b_full_protocol.py` short run → LER matches baseline within noise.

### Step 3 — Doc cleanup

Delete the stale comment in `examples/logical_error_rates/_9_lattice_surgery_cain_fig1b_source.py:92-93` referencing `layout.hx_row_kind` and `layout.qubit_layer` (these attributes no longer exist).

## Commit plan

```
Commit 1: refactor: collapse cheeger.py L≥1 wrapper to L=1 (mechanical)
Commit 2: refactor: dedupe HX assembly between gadget.py and cheeger.py
Commit 3: docs: drop stale qubit_layer / hx_row_kind comment in cain_fig1b source
```

Each commit runs `pytest src/qldpc/codes/surgery/_test.py` and must be all-green before the next.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| basis=Z swap path broken in Step 1 | Swap logic untouched in Step 1 (only the `SurgeryLayout` field shape changes); Step 2 introduces `_assemble_HX_L1` with explicit basis=X/Z dual test in `test_basis_z_dual_equivalence`. |
| Row-kind → integer-count translation misses an edge case | Single source of truth: `data_code.matrix_x.shape[0]` and `.matrix_z.shape[0]` give the counts; sanity script in Step 1 verifies bit-identical HX_merged/HZ_merged for every test fixture. |
| External example breaks due to `SurgeryLayout` field change | Grep verified: no external import of `SurgeryLayout`. Only mention outside `cheeger.py` is a stale comment (handled by Step 3). Cain scripts read `GadgetLayout`, not `SurgeryLayout`. |
| `_reassemble_gadget_with_new_F` even-layer-loop deletion changes augmented HZ | At L=1, `range(2, 1, 2)` is empty; the loop never appended rows. Deletion is provably a no-op. |

## What we are explicitly NOT doing

- **Not** dropping `SurgeryLayout` entirely (the radical "drop legacy" option). The gadget↔legacy bridge stays; this refactor only slims the bridge.
- **Not** changing the basis=Z HX/HZ swap into "native basis-aware boost." That would be the next refactor, not this one.
- **Not** routing the boost through `GadgetLayout` directly. Same reason.
- **Not** adding new tests. Existing tests already cover the success criteria; the refactor's output is bit-identical to the pre-refactor for Step 1 and equivalent for Step 2.

## Open questions

None at this stage. All scoping questions resolved during brainstorming.
