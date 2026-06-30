# Surgery Reorg — Piece D: circuit/ package split (pure motion) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `circuit.py` (1221) and `y_circuit.py` (1275) into a `circuit/` subpackage — `circuit/{engine,support,PPM_XZ,PPM_joint,PPM_Y,PPM_Y_prep,PPM_Y_qec}.py` — pure motion, public API byte-identical.

**Architecture:** Piece D (final) of the layered reorg spec. The circuit layer is the last monolith. **Python forbids `circuit.py` and `circuit/` coexisting**, so the source relocation (all 7 submodules + delete both old files + rewire) is ONE atomic commit (Task 1). To avoid churning the ~60 private-import sites in `circuit_test.py` during that risky move, Task 1's `circuit/__init__.py` **transitionally re-exports every symbol** the old `circuit.py`/`y_circuit.py` exposed, so all existing test/`__init__` imports keep resolving unchanged. Task 2 then splits the test files into mirrored `circuit/*_test.py` (importing the specific submodules) and tightens `circuit/__init__.py` to the public API.

**THE COMPLETE function→file assignment, internal call-graph import wiring, external-consumer list, and test-split boundaries are in `/Users/tgzhou/Project/qLDPC/.superpowers/sdd/piece-d-map.md` (§1–§6). Follow that map exactly — it has every function's line range and target file.**

**Tech Stack:** Python 3, numpy, galois, stim, pytest. Runner: `.venv/bin/pytest`.

## Global Constraints

- **Pure motion:** moved functions are verbatim; only import lines change. No logic edits.
- **Public API byte-identical:** `surgery/__init__.py __all__` unchanged; external `from qldpc.circuits.surgery import …` (and the `examples/` notebooks) unaffected.
- **Symbol names preserved:** `build_single_ppm_circuit`, `build_joint_ppm_circuit`, `build_single_y_ppm_circuit`, `keep_only_observable`, `logical_state_init`, and every `_surgery_*`/`_y_*`/algebra helper keep their names; only their module changes.
- **Layer direction:** `circuit/` imports from `..hmatrix` (one extra dot vs today's `.hmatrix`) and within `circuit/`; never the reverse. Module-import graph must be ACYCLIC: `{PPM_XZ, PPM_joint} → engine → support` and `PPM_Y → PPM_Y_qec → PPM_Y_prep`. The one back-edge (`support.logical_state_init` → the builders) uses a **function-local deferred import** (map §3 / §6.3).
- **File-size cap:** no file > ~500 lines. All 7 source submodules clear it (map §1–§2, recommended Y split: `_split_leading_reset`/`_split_trailing_measure` go to `PPM_Y_qec.py`). Test files are sub-split in Task 2 to clear ~500 too.
- **`hmatrix` basename clash is cosmetic** (map §6.2): `circuit/PPM_XZ.py` ≠ `hmatrix/PPM_XZ.py` — distinct packages. Inside `circuit/`, `from .PPM_XZ import` is the circuit one; the hmatrix layout needs `from ..hmatrix.PPM_XZ import`.
- **ruff clean on every changed file** (run ruff on the FULL changed set before each commit; `--fix` `I001`).
- **No LER/sinter tests.** Pure motion → existing suite is the regression, no new golden.
- **Scoped commit**, never `git add -A`; unrelated working-tree files untouched. Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Baseline:** `.venv/bin/pytest src/qldpc/circuits/surgery/ -q` → `245 passed`.

---

## File Structure After Piece D

```
surgery/
  __init__.py              (modified: y_circuit re-export → .circuit)
  circuit/
    __init__.py            (NEW: Task 1 comprehensive re-export → Task 2 public-only)
    engine.py    ~430  _reliable_checks, _surgery_{state_prep,qec_cycle,observable,final_detectors,detach_and_readout}
    support.py   ~420  gf2/observable algebra, keep_only_observable, logical_state_init, coords, lane map, QubitIDs re-import
    PPM_XZ.py    ~140  build_single_ppm_circuit
    PPM_joint.py ~300  _stitch_to_joint_code, _expand_joint_data_init, build_joint_ppm_circuit, _build_joint_ppm_circuit_same_basis
    PPM_Y_prep.py ~485 _YCtx, _steane_logical_y_eigenstate_prep, _y_state_prep, _split_quditcode_into_virtual_cssc, _mixed_basis_qubit_coords, _compute_stabilizer_center_mask
    PPM_Y_qec.py ~400  _y_qec_cycle(+_row_paulis), _y_detach_and_readout, _split_leading_reset, _split_trailing_measure
    PPM_Y.py     ~460  _y_final_detectors, _y_emit_obs0, _y_emit_survivor_memory, build_single_y_ppm_circuit, _observable_is_deterministic
    + Task 2 test files: support_test, engine_test, PPM_XZ_test(+_e2e), PPM_joint_test(+_data_init), PPM_Y_test
DELETED: circuit.py, y_circuit.py, circuit_test.py, circuit_single_y_test.py
```

---

### Task 1: Create `circuit/` package — atomic source relocation

Split `circuit.py` and `y_circuit.py` into the 7 submodules, delete both, add a **transitional** `circuit/__init__.py` re-exporting every old symbol so nothing outside `circuit/` needs to change yet (except one `surgery/__init__.py` line). Test files stay put and unchanged this task — they ride the re-export.

**Files:**
- Create: `circuit/__init__.py`, `circuit/engine.py`, `circuit/support.py`, `circuit/PPM_XZ.py`, `circuit/PPM_joint.py`, `circuit/PPM_Y_prep.py`, `circuit/PPM_Y_qec.py`, `circuit/PPM_Y.py` (all under `src/qldpc/circuits/surgery/`)
- Delete: `circuit.py`, `y_circuit.py`
- Modify: `surgery/__init__.py`

**Interfaces:**
- Consumes: `..hmatrix.PPM_XZ` (`GadgetLayout`), `..hmatrix.PPM_joint` (`Bridge`, `_joint_merged_dispatch`), `..hmatrix.PPM_Y` (`YGadgetLayout`), `qldpc.circuits.bookkeeping` (`QubitIDs`, …), etc.
- Produces: the 7 submodules with the function homes in map §1–§2; `circuit/__init__.py` re-exporting (this task) **all** names previously importable from `circuit.py`/`y_circuit.py`.

- [ ] **Step 1: Create the 4 `circuit.py`-derived submodules.** Per map §1, move verbatim:
  - `circuit/support.py` ← `_gf2_solve`, `_commuting_logical_basis`, `_block_observable_targets`, `_gadget_merged_csscode`, `keep_only_observable`, `logical_state_init`, `_surgery_qubit_coordinates`, `_check_lane_index_map`. Add module-level `GF2 = galois.GF(2)` and the imports these use (incl. `from qldpc.circuits.bookkeeping import QubitIDs` — needed for type hints AND so the test re-export resolves, map §6.7). **`logical_state_init`** uses **function-local** `from .PPM_XZ import build_single_ppm_circuit` / `from .PPM_joint import build_joint_ppm_circuit` (deferred, to keep the graph acyclic — map §3/§6.3; if its body does not actually call them, omit).
  - `circuit/engine.py` ← `_reliable_checks`, `_surgery_state_prep`, `_surgery_qec_cycle`, `_surgery_observable`, `_surgery_final_detectors`, `_surgery_detach_and_readout`. Imports per map §3: `from .support import _check_lane_index_map, _block_observable_targets, _commuting_logical_basis` (whichever each uses).
  - `circuit/PPM_XZ.py` ← `build_single_ppm_circuit`. Imports: `from .support import _gadget_merged_csscode, _surgery_qubit_coordinates`, `from .engine import _surgery_state_prep, _surgery_qec_cycle, _surgery_observable, _surgery_final_detectors, _surgery_detach_and_readout`, `from ..hmatrix.PPM_XZ import GadgetLayout`.
  - `circuit/PPM_joint.py` ← `_stitch_to_joint_code`, `_expand_joint_data_init`, `build_joint_ppm_circuit`, `_build_joint_ppm_circuit_same_basis`. Imports: `from ..hmatrix.PPM_joint import Bridge, _joint_merged_dispatch`, `from ..hmatrix.PPM_XZ import GadgetLayout`, `from .support import _surgery_qubit_coordinates`, `from .engine import _surgery_state_prep, _surgery_qec_cycle, _surgery_observable, _surgery_final_detectors, _surgery_detach_and_readout`.
  - For each: replicate only the third-party/`qldpc` imports it uses (`stim`, `numpy`, `galois`, `CSSCode`/`QuditCode`, `Pauli`/`PauliXZ`, `NoiseModel`, `EdgeColoring`, bookkeeping records); ruff F401 will flag extras. Today's `from .hmatrix.X import …` become `from ..hmatrix.X import …` (one extra dot).

- [ ] **Step 2: Create the 3 `y_circuit.py`-derived submodules** per map §2 (recommended split):
  - `circuit/PPM_Y_prep.py` ← `_YCtx`, `_steane_logical_y_eigenstate_prep`, `_y_state_prep`, `_split_quditcode_into_virtual_cssc`, `_mixed_basis_qubit_coords`, `_compute_stabilizer_center_mask`. Imports: `from ..hmatrix.PPM_Y import YGadgetLayout` + its third-party imports.
  - `circuit/PPM_Y_qec.py` ← `_y_qec_cycle` (with its nested `_row_paulis`), `_y_detach_and_readout`, `_split_leading_reset`, `_split_trailing_measure`. Imports: `from .PPM_Y_prep import _YCtx`.
  - `circuit/PPM_Y.py` ← `_y_final_detectors`, `_y_emit_obs0`, `_y_emit_survivor_memory`, `build_single_y_ppm_circuit`, `_observable_is_deterministic`. Imports: `from .PPM_Y_prep import _YCtx, _y_state_prep`, `from .PPM_Y_qec import _y_qec_cycle, _y_detach_and_readout`.

- [ ] **Step 3: Write `circuit/__init__.py` as a comprehensive transitional re-export.** Re-export **every** name that was importable from the old `circuit.py` and `y_circuit.py` — public AND the privates the tests use — so `from qldpc.circuits.surgery.circuit import <anything>` keeps resolving:
  ```python
  """Surgery circuit layer (split from the former circuit.py / y_circuit.py).

  See docs/superpowers/specs/2026-06-30-surgery-layered-reorg-design.md.
  NOTE: this re-export surface is transitional (Piece D Task 1) so existing test
  imports keep resolving during the source move; Task 2 narrows it to the public API.
  """
  from .support import (_gf2_solve, _commuting_logical_basis, _block_observable_targets,
      _gadget_merged_csscode, keep_only_observable, logical_state_init,
      _surgery_qubit_coordinates, _check_lane_index_map, QubitIDs)
  from .engine import (_reliable_checks, _surgery_state_prep, _surgery_qec_cycle,
      _surgery_observable, _surgery_final_detectors, _surgery_detach_and_readout)
  from .PPM_XZ import build_single_ppm_circuit
  from .PPM_joint import (_stitch_to_joint_code, _expand_joint_data_init,
      build_joint_ppm_circuit, _build_joint_ppm_circuit_same_basis)
  from .PPM_Y import build_single_y_ppm_circuit
  ```
  (Cross-check against `git show HEAD:.../circuit.py` and `y_circuit.py` for the exact set of names `circuit_test.py`/`circuit_single_y_test.py`/`hmatrix/PPM_joint_test.py` import — add any missed one. `QubitIDs` is re-exported here because `circuit_test.py:1306` imports it from this path, map §6.7.)

- [ ] **Step 4: Delete the old modules and rewire `surgery/__init__.py`.**
  ```bash
  git rm src/qldpc/circuits/surgery/circuit.py src/qldpc/circuits/surgery/y_circuit.py
  ```
  In `surgery/__init__.py`: the existing `from .circuit import (build_joint_ppm_circuit, build_single_ppm_circuit, keep_only_observable, logical_state_init)` now resolves through `circuit/__init__.py` — leave it. Change `from .y_circuit import build_single_y_ppm_circuit` → `from .circuit import build_single_y_ppm_circuit`. `__all__` unchanged.

- [ ] **Step 5: Verify acyclicity, lint, and the suite.**
  - Import check: `.venv/bin/python -c "import qldpc.circuits.surgery"` → no ImportError (catches cycles).
  - `grep -rn "from \.circuit import\|from \.y_circuit import\|surgery\.y_circuit" src/qldpc/circuits/surgery/*.py` → only the `surgery/__init__.py` `from .circuit import …` lines remain; no `.y_circuit`.
  - `grep -n "\.hmatrix\." src/qldpc/circuits/surgery/circuit/*.py` → empty (all use `..hmatrix`).
  - `.venv/bin/ruff check src/qldpc/circuits/surgery/circuit/ src/qldpc/circuits/surgery/__init__.py` → clean (`--fix` `I001`).
  - `.venv/bin/pytest src/qldpc/circuits/surgery/ -q` → `245 passed` (circuit_test.py + circuit_single_y_test.py unchanged, riding the re-export).
  - `wc -l src/qldpc/circuits/surgery/circuit/*.py` → every submodule ≤ ~500.

- [ ] **Step 6: Commit.**
  ```bash
  git add src/qldpc/circuits/surgery/circuit/ src/qldpc/circuits/surgery/circuit.py \
    src/qldpc/circuits/surgery/y_circuit.py src/qldpc/circuits/surgery/__init__.py
  git commit -m "refactor(surgery): split circuit.py + y_circuit.py into circuit/ package

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 2: Split the circuit tests into the mirrored `circuit/*_test.py` and tighten `__init__`

Move `circuit_test.py` (2261) and `circuit_single_y_test.py` (314) into `circuit/`, split by which submodule each test exercises (map §5), repoint imports to the specific submodules, and narrow `circuit/__init__.py` to the public API.

**Files:**
- Create: `circuit/support_test.py`, `circuit/engine_test.py`, `circuit/PPM_XZ_test.py`, `circuit/PPM_XZ_e2e_test.py`, `circuit/PPM_joint_test.py`, `circuit/PPM_joint_data_init_test.py`, `circuit/PPM_Y_test.py`, and (if shared fixtures need a home) `circuit/conftest.py`
- Delete: `circuit_test.py`, `circuit_single_y_test.py`
- Modify: `circuit/__init__.py` (narrow to public), `hmatrix/PPM_joint_test.py:450` (repoint to specific submodule)

- [ ] **Step 1: Split `circuit_test.py` into the mirrored files** per map §5 (sub-split the two over-cap files to clear ~500):
  - `circuit/support_test.py` (~221) — the 16 `support`-only tests.
  - `circuit/engine_test.py` (~414) — the 7 engine + 6 engine/support tests.
  - `circuit/PPM_XZ_test.py` (~500) — the single-PPM unit/structural tests; `circuit/PPM_XZ_e2e_test.py` (~350) — the heavy e2e/truth-table/x-error-locality tests (map §5 option 2 boundary).
  - `circuit/PPM_joint_test.py` (~420) — merge/dimension/coords/observable + the 3 `_joint_merged_*` "none" tests; `circuit/PPM_joint_data_init_test.py` (~280) — the data_init truth-table/tuple tests.
  - Shared fixtures (`_data_measured`, `_bb_36_8_code`, `_steane_joint_fixture`) + `test_memory_experiment_*` baseline → `circuit/conftest.py` (or duplicate into the using files). In each new file, repoint imports to the SPECIFIC submodule: `_surgery_*`/`_reliable_checks` → `from qldpc.circuits.surgery.circuit.engine import …`; `_gf2_solve`/`_commuting_logical_basis`/`_block_observable_targets`/`_gadget_merged_csscode`/`keep_only_observable`/`logical_state_init`/`QubitIDs` → `…circuit.support`; `build_single_ppm_circuit` → `…circuit.PPM_XZ`; `build_joint_ppm_circuit`/`_expand_joint_data_init` → `…circuit.PPM_joint`. `_joint_merged_dispatch` stays `from …hmatrix.PPM_joint import …`.
  ```bash
  git rm src/qldpc/circuits/surgery/circuit_test.py
  ```

- [ ] **Step 2: Relocate `circuit_single_y_test.py` → `circuit/PPM_Y_test.py`** (map §5: all 14 tests use public `build_single_y_ppm_circuit` + `conftest._steane_y_pair` + `hmatrix.PPM_Y.build_y_gadget` — no symbol-path rewire, just the file move; keep its absolute imports).
  ```bash
  git mv src/qldpc/circuits/surgery/circuit_single_y_test.py src/qldpc/circuits/surgery/circuit/PPM_Y_test.py
  ```

- [ ] **Step 3: Repoint the one remaining external private consumer.** `hmatrix/PPM_joint_test.py:450` `from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit, keep_only_observable` → `from qldpc.circuits.surgery.circuit.PPM_joint import build_joint_ppm_circuit` + `from qldpc.circuits.surgery.circuit.support import keep_only_observable` (or leave via the package `__init__` public re-export — both work; prefer the specific submodule for layer clarity).

- [ ] **Step 4: Narrow `circuit/__init__.py` to the public API.** Replace the transitional comprehensive re-export with only:
  ```python
  from .PPM_XZ import build_single_ppm_circuit
  from .PPM_joint import build_joint_ppm_circuit
  from .PPM_Y import build_single_y_ppm_circuit
  from .support import keep_only_observable, logical_state_init
  ```
  (These are exactly the names `surgery/__init__.py` re-exports.) Confirm no test still imports a private symbol via the package `__init__` (they now hit the specific submodules from Steps 1–3).

- [ ] **Step 5: Verify lint + the suite + caps.**
  - `grep -rn "from \.circuit import\|surgery\.circuit import" src/qldpc/` → only public names via `surgery/__init__.py` (and any deliberately-kept `__init__` import); no private `_surgery_*`/`_gf2_*` via the package root.
  - `.venv/bin/ruff check src/qldpc/circuits/surgery/circuit/` → clean.
  - `.venv/bin/pytest src/qldpc/circuits/surgery/ -q` → `245 passed`.
  - `wc -l src/qldpc/circuits/surgery/circuit/*_test.py` → each ≤ ~500.

- [ ] **Step 6: Commit.**
  ```bash
  git add src/qldpc/circuits/surgery/circuit/ src/qldpc/circuits/surgery/circuit_test.py \
    src/qldpc/circuits/surgery/circuit_single_y_test.py src/qldpc/circuits/surgery/hmatrix/PPM_joint_test.py
  git commit -m "refactor(surgery): split circuit tests into circuit/*_test, narrow circuit/__init__

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

## Plan Self-Review

**Spec coverage (Piece D row of §7):** `circuit.py` → `circuit/{engine,support,PPM_XZ,PPM_joint}` (Task 1 Step 1) ✓; `y_circuit.py` → `circuit/{PPM_Y,PPM_Y_prep,PPM_Y_qec}` (Task 1 Step 2) ✓; `circuit/__init__.py` (Task 1 Step 3 → Task 2 Step 4) ✓; rewire `surgery/__init__.py` (Task 1 Step 4) ✓; mirror the 2261-line `circuit_test.py` (Task 2) ✓; existing tests + golden hashes + public API unchanged; no source file > ~500 ✓.

**Placeholder scan:** "follow the map exactly" + "ruff F401 will flag extras" + "if its body does not actually call them, omit" are concrete procedures for a verbatim relocation whose detailed tables live in `piece-d-map.md` (a committed plan-prep artifact the implementer reads), not logic placeholders. The sub-split boundaries (Task 2 Step 1) cite the map's exact test groupings.

**Type/name consistency:** symbol names unchanged throughout; the acyclic import direction (`PPM_* → engine → support`; `PPM_Y → PPM_Y_qec → PPM_Y_prep`; deferred import in `logical_state_init`) is consistent between Task 1's submodule creation and the map's call graph. `circuit/__init__.py` goes comprehensive (Task 1) → public-only (Task 2), and `surgery/__init__.py __all__` is unchanged in both.
