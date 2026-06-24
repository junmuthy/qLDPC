# Ȳ-emitter decomposition + H̃-faithful construction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Ȳ merged-code construction read directly as the paper's
check matrix H̃ (formula-faithful names, explicit per-block construction,
formula-order rows), and split the 635-line Ȳ emitter into named phases in a
new `y_circuit.py`, pruning untested obs1/benchmark_y scaffolding.

**Architecture:** Two units. (1) `y_gadget.py` `build_y_gadget` assembles
`H_sym` block-by-block in H̃ order (§0 of the design spec) from formula-named
variables. (2) A new `y_circuit.py` holds the Ȳ emitter, decomposed into
`_y_state_prep / _y_qec_cycle / _y_detach_and_readout / _y_final_detectors /
_y_emit_obs0 / _y_emit_survivor_memory`, orchestrated by a thin
`build_single_y_ppm_circuit`. The CSS path (`circuit.py`) is untouched.

**Tech Stack:** Python 3.12, numpy, galois (GF2), stim; pytest; `uv`/`.venv`.

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-06-25-y-emitter-decomposition-design.md` (§0 has the H̃ block + block→name→construction table — copy names verbatim).
- Behaviour-preserving at the **DEM / observable** level. CSS circuits (X̄/Z̄/X̄X̄/Z̄Z̄) stay **byte-identical** — do not touch `circuit.py`'s CSS code.
- Check IDs / detector coordinates **may renumber** (formula-order assembly); logical/DEM/observable assertions must stay green.
- Citations in docstrings/comments use full form (authors + arXiv:ID + §), never `math.md` or bare surnames.
- `gadget.py` internals are **mapped** to formula symbols at use sites, **never renamed**.
- Run tests with `.venv/bin/pytest`. Fast surgery run: `.venv/bin/pytest src/qldpc/circuits/surgery/ -q -p no:cacheprovider`.
- Commit after every task. Branch: `feat/latticesurgery-mixedjoint` (do not create a new branch).

---

## File Structure

- `src/qldpc/circuits/surgery/y_gadget.py` — **modify** `build_y_gadget` assembly (lines ~676-738) + its docstring. No change to `apply_mixed_basis_merge` / `_partial0_symplectic_rows` logic.
- `src/qldpc/circuits/surgery/y_circuit.py` — **create**. Receives the whole Ȳ emitter subsystem from `circuit.py`.
- `src/qldpc/circuits/surgery/circuit.py` — **modify**: delete the moved Ȳ subsystem (build_single_y_ppm_circuit + 5 private helpers + `_steane_logical_y_eigenstate_prep` + nested `_row_paulis`), drop now-unused imports.
- `src/qldpc/circuits/surgery/__init__.py` — **modify**: re-export `build_single_y_ppm_circuit` from `.y_circuit`.
- `src/qldpc/circuits/surgery/y_circuit_test.py` — **create**: tests asserting the H̃ block structure + phase behaviour (some moved from `circuit_single_y_test.py`).
- `src/qldpc/circuits/surgery/circuit_single_y_test.py` — **modify**: update import paths (`.circuit` → `.y_circuit`); update any exact check-ID assertions.
- `src/qldpc/circuits/surgery/y_gadget_test.py` — **modify**: add the H̃ block-order test; update any exact check-ID assertions.

---

## Task 1: H̃-faithful construction in `y_gadget.py`

Rebuild `H_sym` from formula-named blocks in formula order. Behaviour change:
row order goes grouped `[X|Z|Y|cycle]` → H̃ order `1,2,3,4,5,6`, renumbering
check IDs.

**Files:**
- Modify: `src/qldpc/circuits/surgery/y_gadget.py:676-738` (the assembly) + `build_y_gadget` docstring
- Test: `src/qldpc/circuits/surgery/y_gadget_test.py`

**Interfaces:**
- Consumes: `apply_mixed_basis_merge(HX_all, HZ_all, merge_qubits=W, adapter_cols) -> (HX_out, HZ_out, Y_stab, _, _, _)`; `_partial0_symplectic_rows(g_x, g_z, x, z, n, k_x, k_z) -> partial0` (both unchanged).
- Produces: `H_sym` (np.int_, shape `(rows, 2*n_merged)`) with rows in H̃ block order 1..6; everything downstream (`merged_code`, `_ybar_obs0_rows`) derives families from `H_sym` and adapts. `YGadgetLayout` fields unchanged.

- [ ] **Step 1: Write the failing test pinning H̃ block order**

In `y_gadget_test.py`:

```python
def test_h_sym_rows_in_h_tilde_block_order() -> None:
    """H_sym rows follow the H̃ formula order: block1 H_X(X), block2 χ_X(X),
    block3 Y(mixed), block4 H_Z(Z), block5 χ_Z(Z), block6 cycles
    (Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.D)."""
    import numpy as np
    from qldpc.circuits.surgery.y_gadget import _steane_y_pair, build_y_gadget

    code, x, z = _steane_y_pair()
    yg = build_y_gadget(code, x=x, z=z)
    H = np.asarray(yg.H_sym).astype(int)
    n = yg.merged_code.num_qudits
    xparts = (H[:, :n] != 0).any(axis=1)
    zparts = (H[:, n:] != 0).any(axis=1)
    m_x = code.matrix_x.shape[0]
    m_z = code.matrix_z.shape[0]
    n_w = len(yg.W)
    # block 1 (H_X) + block 2 (χ_X on V_X) are pure-X and come first
    assert xparts[: m_x].all() and not zparts[: m_x].any(), "block 1 not pure-X-first"
    # block 3 (Y on W) is mixed and follows the pure-X blocks
    y0 = m_x + (yg.g_x.HX_merged.shape[0] - m_x - n_w)  # m_x + |V_X|
    assert (xparts[y0 : y0 + n_w] & zparts[y0 : y0 + n_w]).all(), "block 3 not mixed"
```

- [ ] **Step 2: Run it — expect FAIL** (current grouped order puts Y_stab last)

Run: `.venv/bin/pytest src/qldpc/circuits/surgery/y_gadget_test.py::test_h_sym_rows_in_h_tilde_block_order -v`
Expected: FAIL (block 3 mixed rows are at the end, not after the χ_X block).

- [ ] **Step 3: Rewrite the assembly in formula order with named blocks**

Replace `y_gadget.py:725-738` (the `rows_sym` loop + `H_sym = …`) with the
H̃-ordered, named-block assembly. The named sub-blocks per design §0:

```python
    # --- Assemble H̃ block-by-block in formula order (Ide, Gowda, Nadkarni,
    # Dauphinais arXiv:2410.02753 §III.D; design spec §0). Column layout per
    # symplectic half: [ data (n) | κ_x (k_x) | κ_z (k_z) ].
    m_x, m_z = code.matrix_x.shape[0], code.matrix_z.shape[0]

    def _sym_x(rows: np.ndarray) -> np.ndarray:  # X-only rows → [X | 0]
        return np.hstack([rows, np.zeros_like(rows)]).astype(np.int_)

    def _sym_z(rows: np.ndarray) -> np.ndarray:  # Z-only rows → [0 | Z]
        return np.hstack([np.zeros_like(rows), rows]).astype(np.int_)

    # apply_mixed_basis_merge removed the χ@W rows into Y_stab, so:
    Xcheck_rows = HX_out[:m_x]   # block 1: [H_X | 0 | π_{C₀^Z}^T]
    chiX_rows   = HX_out[m_x:]   # block 2: [π_{V_X} | ∂₁ˣ|_{V_X} | 0]
    Ymix_rows   = Y_stab         # block 3: [π_W|∂₁ˣ|_W|0 ‖ π_W|0|∂₁ᶻ|_W]
    Zcheck_rows = HZ_out[:m_z]   # block 4: [H_Z | π_{C₀^X}^T | 0]
    chiZ_rows   = HZ_out[m_z:]   # block 5: [π_{V_Z} | 0 | ∂₁ᶻ|_{V_Z}]
    cycle_rows  = partial0       # block 6: [0|0|∂₀^Z ‖ 0|∂₀^X|0]

    blocks = [
        _sym_x(Xcheck_rows),
        _sym_x(chiX_rows),
        Ymix_rows.astype(np.int_),
        _sym_z(Zcheck_rows),
        _sym_z(chiZ_rows),
        cycle_rows.astype(np.int_),
    ]
    H_sym = (
        np.vstack([b for b in blocks if b.shape[0]])
        if any(b.shape[0] for b in blocks)
        else np.zeros((0, 2 * n_merged), dtype=np.int_)
    )
```

Delete the old `rows_sym = []` loop (the four `for r in …` blocks) it replaces.

- [ ] **Step 4: Run the new test + the full y_gadget suite**

Run: `.venv/bin/pytest src/qldpc/circuits/surgery/y_gadget_test.py -q -p no:cacheprovider`
Expected: the new test PASSES. If any existing test asserts an **exact check
index / row index** that shifted, update the expected number to the H̃-order
value (the code is equivalent; only the index moved). Do NOT weaken logical
assertions.

- [ ] **Step 5: Run the Ȳ circuit + full surgery suite**

Run: `.venv/bin/pytest src/qldpc/circuits/surgery/circuit_single_y_test.py src/qldpc/circuits/surgery/y_gadget_test.py -q -p no:cacheprovider`
Expected: PASS. Fix any exact-check-ID assertion in `circuit_single_y_test.py`
the same way (update the index; keep DEM/observable assertions intact).

- [ ] **Step 6: Update `build_y_gadget` docstring**

Add the §0 H̃ block diagram and the block→name→construction table to the
`build_y_gadget` docstring (copy from the design spec §0). Map each named block
to its source (`Xcheck_rows = HX_out[:m_x]  # H_X ext onto κ_z = π_{C₀^Z}^T`).

- [ ] **Step 7: Commit**

```bash
git add src/qldpc/circuits/surgery/y_gadget.py src/qldpc/circuits/surgery/y_gadget_test.py src/qldpc/circuits/surgery/circuit_single_y_test.py
git commit -m "refactor(surgery): assemble Ȳ H_sym in H̃ formula order with named blocks

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Extract the Ȳ subsystem to `y_circuit.py` (verbatim move)

Pure move — no logic change. Establishes the module before decomposition.

**Files:**
- Create: `src/qldpc/circuits/surgery/y_circuit.py`
- Modify: `src/qldpc/circuits/surgery/circuit.py` (delete moved defs + dead imports)
- Modify: `src/qldpc/circuits/surgery/__init__.py`
- Modify: `src/qldpc/circuits/surgery/circuit_single_y_test.py` (import paths)

**Interfaces:**
- Produces: `y_circuit.build_single_y_ppm_circuit(yg, *, rounds, noise_model=None, data_init=None, memory_logical=None, force_obs0=False, benchmark_y=False) -> stim.Circuit` (signature unchanged this task; `benchmark_y` removed in Task 4). Also re-exports the 5 private helpers + `_steane_logical_y_eigenstate_prep`.

- [ ] **Step 1: Identify the exact line ranges to move**

Run: `grep -nE "^def (build_single_y_ppm_circuit|_steane_logical_y_eigenstate_prep|_split_quditcode_into_virtual_cssc|_mixed_basis_qubit_coords|_compute_stabilizer_center_mask|_observable_is_deterministic)" src/qldpc/circuits/surgery/circuit.py`
Record the start line of each and its `end_lineno` (next `def`/EOF).

- [ ] **Step 2: Create `y_circuit.py` with the moved defs**

Create `src/qldpc/circuits/surgery/y_circuit.py`. Header + imports (only what
the moved code uses):

```python
"""Single logical-Ȳ (Ȳ = iX̄Z̄) measurement circuit — non-CSS homological
surgery (Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.C/§III.D;
docs/superpowers/docs/main.tex §4). Emits the split X/Z/Y syndrome schedule
over the merged code H̃ (see y_gadget.build_y_gadget for the H̃ block layout)."""

from __future__ import annotations

import numpy as np
import stim

from qldpc.circuits.bookkeeping import DetectorRecord, MeasurementRecord, QubitIDs
from qldpc.circuits.memory.syndrome_measurement import EdgeColoring
from qldpc.circuits.noise_model import NoiseModel
from qldpc.codes.common import CSSCode, QuditCode
from qldpc.objects import Pauli

from .y_gadget import YGadgetLayout
```

Move `build_single_y_ppm_circuit`, `_steane_logical_y_eigenstate_prep`,
`_split_quditcode_into_virtual_cssc`, `_mixed_basis_qubit_coords`,
`_compute_stabilizer_center_mask`, `_observable_is_deterministic` verbatim from
`circuit.py` into this file.

- [ ] **Step 3: Delete the moved defs from `circuit.py`**

Remove those 6 functions from `circuit.py`. Then prune imports in `circuit.py`
that only the moved code used (verify with the next step's import check).

- [ ] **Step 4: Update `__init__.py` re-export**

In `surgery/__init__.py`, change `build_single_y_ppm_circuit` import source:

```python
from .circuit import (
    build_joint_ppm_circuit,
    build_single_ppm_circuit,
    keep_only_observable,
    logical_state_init,
)
from .y_circuit import build_single_y_ppm_circuit
```

- [ ] **Step 5: Update test imports**

In `circuit_single_y_test.py`, change any
`from qldpc.circuits.surgery.circuit import <moved helper>` to
`from qldpc.circuits.surgery.y_circuit import <moved helper>`.
(`from qldpc.circuits.surgery import build_single_y_ppm_circuit` is unchanged.)

- [ ] **Step 6: Import + lint check**

Run: `.venv/bin/python -c "import qldpc.circuits.surgery.circuit, qldpc.circuits.surgery.y_circuit; from qldpc.circuits.surgery import build_single_y_ppm_circuit; print('OK')"`
Run: `.venv/bin/ruff check src/qldpc/circuits/surgery/circuit.py src/qldpc/circuits/surgery/y_circuit.py`
Expected: `OK` + `All checks passed!` (fix any unused-import in `circuit.py`).

- [ ] **Step 7: Run the full surgery suite**

Run: `.venv/bin/pytest src/qldpc/circuits/surgery/ -q -p no:cacheprovider`
Expected: PASS (same count as before the move).

- [ ] **Step 8: Commit**

```bash
git add src/qldpc/circuits/surgery/y_circuit.py src/qldpc/circuits/surgery/circuit.py src/qldpc/circuits/surgery/__init__.py src/qldpc/circuits/surgery/circuit_single_y_test.py
git commit -m "refactor(surgery): extract Ȳ subsystem to y_circuit.py (verbatim move)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Decompose `build_single_y_ppm_circuit` into named phases

Pure extraction — the resulting circuit is byte-identical. Extract each inline
phase (delimited by the existing `# --- … ---` section comments) into a named
function; the body is the existing code, with the cross-phase state passed as
explicit parameters instead of closures.

**Files:**
- Modify: `src/qldpc/circuits/surgery/y_circuit.py`

**Interfaces (produced — exact signatures the orchestrator calls):**
- `_y_state_prep(yg, *, data_init) -> tuple[stim.Circuit, _YCtx]` — emits coords + init; returns the circuit and a context object holding the shared state (merged_code, QubitIDs, the id arrays `real_data_ids/kx_ids/kz_ids/y_ancilla_ids`, virtual_cssc_X/Z, `n_code/k_x/k_z/n_q`, `center_mask`).
- `_y_qec_cycle(ctx, *, rounds) -> tuple[stim.Circuit, MeasurementRecord]` — split X/Z/Y schedule + round-1 reliable detectors + repeat block.
- `_y_detach_and_readout(ctx, *, measurement_record) -> stim.Circuit` — mixed-basis MX/M/MY destructive readout.
- `_y_final_detectors(ctx, *, measurement_record) -> stim.Circuit` — stabilizer-center rows from final readouts.
- `_y_emit_obs0(ctx, circuit, *, data_init, force_obs0, measurement_record) -> None` — appends OBSERVABLE_INCLUDE(idx 0) per the obs0 gate.
- `_y_emit_survivor_memory(ctx, circuit, *, memory_logical, data_init, measurement_record) -> None` — appends the survivor-Z̄ observable.

Define `_YCtx` as a small `@dataclasses.dataclass` holding the shared fields
above (this replaces the current closure variables).

- [ ] **Step 1: Add a characterization test (lock current Ȳ circuit text)**

In `y_circuit_test.py` (create it):

```python
def test_y_circuit_text_stable_under_decomposition() -> None:
    """Pin the exact Ȳ circuit so the phase extraction stays byte-identical."""
    from qldpc.circuits.surgery import build_single_y_ppm_circuit
    from qldpc.circuits.surgery.y_gadget import _steane_y_pair, build_y_gadget

    code, x, z = _steane_y_pair()
    yg = build_y_gadget(code, x=x, z=z)
    circ = build_single_y_ppm_circuit(yg, rounds=3, data_init="Y+")
    # snapshot — capture the current text before extracting phases
    assert circ.detector_error_model().num_detectors > 0
    globals().setdefault("_Y_SNAPSHOT", str(circ))
    assert str(circ) == globals()["_Y_SNAPSHOT"]
```

Better: capture the snapshot to a module-level constant. Run the emitter once,
paste `str(circ)` into a `_EXPECTED_Y_CIRCUIT` triple-quoted constant in the
test, and assert equality. This makes the extraction verifiably byte-identical.

- [ ] **Step 2: Run it — expect PASS** (baseline)

Run: `.venv/bin/pytest src/qldpc/circuits/surgery/y_circuit_test.py::test_y_circuit_text_stable_under_decomposition -v`
Expected: PASS (records the current circuit).

- [ ] **Step 3: Extract `_y_state_prep` + define `_YCtx`**

Add `import dataclasses` and define `_YCtx`. Move the setup + state-prep section
(from the function top through the `# --- State prep ---` block, i.e. the
`_split_quditcode_into_virtual_cssc` / `_mixed_basis_qubit_coords` /
`_steane_logical_y_eigenstate_prep` lines) into `_y_state_prep`, returning
`(circuit, ctx)`. Replace that region in `build_single_y_ppm_circuit` with
`circuit, ctx = _y_state_prep(yg, data_init=data_init)`.

- [ ] **Step 4: Run the snapshot test** — Expected: PASS (byte-identical).
Run: `.venv/bin/pytest src/qldpc/circuits/surgery/y_circuit_test.py -q -p no:cacheprovider`

- [ ] **Step 5: Extract `_y_qec_cycle`**

Move the `# --- Build the split X / Z / Y per-round circuit ---` through the
round-emission block (incl. nested `_row_paulis`, the `EdgeColoring` X/Z phase
circuits, the Y-phase, the round-1 reliable detectors + repeat block) into
`_y_qec_cycle(ctx, *, rounds)`, returning `(circuit, measurement_record)`.
Annotate: `# X-phase → H̃ blocks 1,2; Z-phase → 4,5; Y-phase → block 3`.

- [ ] **Step 6: Run the snapshot test** — Expected: PASS.

- [ ] **Step 7: Extract `_y_detach_and_readout` + `_y_final_detectors`**

Move the `# --- Detach + destructive readout ---` block into
`_y_detach_and_readout(ctx, *, measurement_record)`, and the
`# --- Final detectors ---` block into `_y_final_detectors(ctx, *, measurement_record)`.

- [ ] **Step 8: Run the snapshot test** — Expected: PASS.

- [ ] **Step 9: Extract `_y_emit_obs0` + `_y_emit_survivor_memory`**

Move the `# --- obs0 ---` block into `_y_emit_obs0(...)` and the
`# --- survivor-memory observable ---` block into `_y_emit_survivor_memory(...)`.
The orchestrator now reads as the §3 phase list. (`obs1`/`benchmark_y` blocks
stay for now — removed in Task 4.)

- [ ] **Step 10: Run the full surgery suite** — Expected: PASS.
Run: `.venv/bin/pytest src/qldpc/circuits/surgery/ -q -p no:cacheprovider`

- [ ] **Step 11: Commit**

```bash
git add src/qldpc/circuits/surgery/y_circuit.py src/qldpc/circuits/surgery/y_circuit_test.py
git commit -m "refactor(surgery): decompose Ȳ emitter into named H̃-annotated phases

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Prune `obs1` + `benchmark_y`

Remove the two untested bring-up observables and the `benchmark_y` parameter.

**Files:**
- Modify: `src/qldpc/circuits/surgery/y_circuit.py`
- Modify: `src/qldpc/circuits/surgery/y_circuit_test.py` (refresh the snapshot constant)

**Interfaces:**
- Produces: `build_single_y_ppm_circuit(yg, *, rounds, noise_model=None, data_init=None, memory_logical=None, force_obs0=False) -> stim.Circuit` (no `benchmark_y`).

- [ ] **Step 1: Confirm nothing references the pruned surface**

Run: `grep -rn "benchmark_y\|obs1" src/qldpc/circuits/surgery/`
Expected: only the `y_circuit.py` definitions + design docs. If a test asserts
on Ȳ `obs1`/`benchmark_y`, stop and report (spec says none exist).

- [ ] **Step 2: Remove `benchmark_y` param + block and the `obs1` block**

In `y_circuit.py`: delete the `benchmark_y: bool = False` parameter, the
`# --- benchmark_y: … ---` block, and the `# --- obs1: destructive cross-check
… ---` block. Update the docstring to list only obs0 + survivor-memory.

- [ ] **Step 3: Refresh the snapshot test constant**

The `data_init="Y+"` circuit loses its obs1 OBSERVABLE_INCLUDE(1) line. Re-run
the emitter, update `_EXPECTED_Y_CIRCUIT` in `y_circuit_test.py` to the new
text, and confirm `dem.num_observables` matches the kept observables.

- [ ] **Step 4: Run the full surgery suite**

Run: `.venv/bin/pytest src/qldpc/circuits/surgery/ -q -p no:cacheprovider`
Expected: PASS. Fix any test asserting the old obs1 presence (none expected).

- [ ] **Step 5: Lint + final import check**

Run: `.venv/bin/ruff check src/qldpc/circuits/surgery/`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/qldpc/circuits/surgery/y_circuit.py src/qldpc/circuits/surgery/y_circuit_test.py
git commit -m "refactor(surgery): prune untested Ȳ obs1 + benchmark_y scaffolding

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- §0 H̃ reference + naming → Task 1 (assembly + docstring). ✓
- §1 H̃-faithful construction / formula-order → Task 1. ✓
- §2 module structure (`y_circuit.py`) → Task 2. ✓
- §3 phase decomposition → Task 3. ✓
- §4 prune obs1/benchmark_y → Task 4. ✓
- §5 success criteria (CSS untouched, check-ID renumber tolerated, ruff clean) → embedded in each task's test steps + Global Constraints. ✓

**Placeholder scan:** Task 3 Step 1 leaves the snapshot mechanism as "paste
`str(circ)` into a constant" — this is the standard characterization-test
pattern (the exact text is machine-generated at implement time, not knowable in
the plan); acceptable. No other TBD/TODO.

**Type consistency:** `_YCtx` is produced by `_y_state_prep` (Task 3 Step 3) and
consumed by every later phase (`ctx` parameter) — names consistent. The Task 2
signature still has `benchmark_y`; Task 4 removes it and updates the snapshot —
consistent across tasks.
