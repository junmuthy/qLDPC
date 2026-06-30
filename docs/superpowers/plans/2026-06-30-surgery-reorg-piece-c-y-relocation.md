# Surgery Reorg — Piece C: Y relocation (pure motion) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Relocate `y_gadget.py` into the `hmatrix/` layer, split into `hmatrix/PPM_Y.py` (Y H-matrix construction) and `hmatrix/PPM_Y_obs0.py` (the Ȳ readout planner) — pure motion, no logic change.

**Architecture:** Piece C of the layered reorg spec (`docs/superpowers/specs/2026-06-30-surgery-layered-reorg-design.md`). **No flatten:** `build_y_gadget`'s assembly is *already* closed-form/Eq.68 (named blocks `Xcheck_rows`/`SXprime_rows`/`Ymix_rows`/… + `_sym_x`/`_sym_z` + `np.vstack`), so it relocates verbatim. The `Obs0` readout planner (`Obs0Row`, `Obs0ReadoutPlan`, `_ybar_obs0_rows`) is self-contained (calls none of the Y core helpers), so `PPM_Y → PPM_Y_obs0` is the only cross-edge and the split is acyclic. The Y *circuit* (`y_circuit.py`) stays flat (it moves in Piece D); only its `YGadgetLayout` import changes.

**Tech Stack:** Python 3, numpy, galois (GF(2)), stim, pytest. Runner: `.venv/bin/pytest`.

## Global Constraints

- **Pure motion:** moved functions are verbatim; only import lines change. No logic edits.
- **Public API byte-identical:** `surgery/__init__.py __all__` unchanged; it re-exports `YGadgetLayout`, `build_y_gadget` from the new module. External `from qldpc.circuits.surgery import …` unaffected.
- **Symbol names preserved:** `YGadgetLayout`, `build_y_gadget`, `Obs0Row`, `Obs0ReadoutPlan`, `_ybar_obs0_rows`, `_overlap_size`, `_locate_overlap(s)`, `_in_rowspace_gf2`, `_merged_incidence`, `_partial0_symplectic_rows`.
- **Layer direction:** `hmatrix/PPM_Y*.py` import only `qldpc.*`/numpy/galois and `hmatrix/` siblings (`.PPM_XZ`, `.merge`, `.cheeger`, `.PPM_Y_obs0`) — never `circuit`/`y_circuit`. `PPM_Y → PPM_Y_obs0` is one-directional (PPM_Y_obs0 imports nothing from PPM_Y).
- **Citations** in moved docstrings stay intact and full (Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.C/§III.D; Webster, Smith, Cohen arXiv:2511.15989 §II.A). Never `main.tex`/bare surnames.
- **File-size cap:** no source file > ~500 lines (`PPM_Y.py` ~460, `PPM_Y_obs0.py` ~222 — both under).
- **ruff clean on every changed file** (run ruff on the FULL changed set before committing; `--fix` any `I001`).
- **No LER/sinter tests.** Pure motion → verified by the existing suite, no new golden.
- **Scoped commit**, never `git add -A`; unrelated working-tree files (`.gitignore`, `main.*`, notebooks) untouched. Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Baseline:** `.venv/bin/pytest src/qldpc/circuits/surgery/ -q` → `245 passed`.

---

## File Structure After Piece C

```
surgery/
  __init__.py        (modified: YGadgetLayout/build_y_gadget re-export from .hmatrix.PPM_Y)
  conftest.py        (modified: _overlap_size import → hmatrix.PPM_Y)
  y_circuit.py       (modified: YGadgetLayout import → hmatrix.PPM_Y — stays flat)
  circuit_single_y_test.py (modified: build_y_gadget import → hmatrix.PPM_Y — stays flat)
  hmatrix/
    PPM_Y.py         (NEW ~460) _in_rowspace_gf2, _locate_overlap, _locate_overlaps,
                                _overlap_size, _merged_incidence, _partial0_symplectic_rows,
                                YGadgetLayout, build_y_gadget
    PPM_Y_obs0.py    (NEW ~222) Obs0Row, Obs0ReadoutPlan, _ybar_obs0_rows
    PPM_Y_test.py    (NEW ~213) ← y_gadget_test.py
DELETED: y_gadget.py, y_gadget_test.py
```

---

### Task 1: Split `y_gadget.py` → `hmatrix/PPM_Y.py` + `hmatrix/PPM_Y_obs0.py`

One atomic pure-motion relocation: split the source, rewire every importer, move the test. Deleting `y_gadget.py` breaks `y_gadget_test.py`'s imports immediately, so the source split, all rewires, and the test move land in one commit.

**Files:**
- Create: `src/qldpc/circuits/surgery/hmatrix/PPM_Y.py`, `src/qldpc/circuits/surgery/hmatrix/PPM_Y_obs0.py`, `src/qldpc/circuits/surgery/hmatrix/PPM_Y_test.py`
- Delete: `src/qldpc/circuits/surgery/y_gadget.py`, `src/qldpc/circuits/surgery/y_gadget_test.py`
- Modify: `src/qldpc/circuits/surgery/__init__.py`, `src/qldpc/circuits/surgery/y_circuit.py`, `src/qldpc/circuits/surgery/conftest.py`, `src/qldpc/circuits/surgery/circuit_single_y_test.py`

**Interfaces:**
- Produces: `qldpc.circuits.surgery.hmatrix.PPM_Y` (`build_y_gadget`, `YGadgetLayout`, `_overlap_size`, `_locate_overlap(s)`, `_in_rowspace_gf2`, `_merged_incidence`, `_partial0_symplectic_rows`); `qldpc.circuits.surgery.hmatrix.PPM_Y_obs0` (`Obs0Row`, `Obs0ReadoutPlan`, `_ybar_obs0_rows`).

- [ ] **Step 1: Create `hmatrix/PPM_Y_obs0.py`.** Move **verbatim** from `y_gadget.py`: `Obs0Row` (class, ~193), `Obs0ReadoutPlan` (class, ~224), `_ybar_obs0_rows` (~278–415). It is self-contained — it calls none of the Y core helpers. Module header: `from __future__ import annotations`, `import dataclasses`, `import numpy as np`, and whatever else these three actually use (`import galois`, `from qldpc.codes.common import CSSCode`, `from qldpc.objects import Pauli` — keep only the ones referenced; ruff F401 will flag extras). Docstring: cite Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.D.

- [ ] **Step 2: Create `hmatrix/PPM_Y.py`.** Move **verbatim** the rest of `y_gadget.py`: `_in_rowspace_gf2` (~31), `_locate_overlap` (~49), `_locate_overlaps` (~100), `_overlap_size` (~122), `_merged_incidence` (~127), `_partial0_symplectic_rows` (~160), `YGadgetLayout` (class, ~416), `build_y_gadget` (~514–702, including its `_embed`/`_sym_x`/`_sym_z` nested helpers and the already-Eq.68 block assembly — DO NOT alter the assembly). Fix imports (these were `.hmatrix.X` in `y_gadget.py`; now siblings inside `hmatrix/`):
  - `from .hmatrix.merge import apply_mixed_basis_merge` → `from .merge import apply_mixed_basis_merge`
  - `from .hmatrix.PPM_XZ import GadgetLayout, build_gadget` → `from .PPM_XZ import GadgetLayout, build_gadget`
  - the in-`build_y_gadget` import `from .hmatrix.cheeger import boost_gadget, cheeger_constant` → `from .cheeger import boost_gadget, cheeger_constant`
  - ADD `from .PPM_Y_obs0 import Obs0Row, Obs0ReadoutPlan, _ybar_obs0_rows` (the `YGadgetLayout` field types `Obs0Row`/`Obs0ReadoutPlan` and `build_y_gadget`'s call to `_ybar_obs0_rows`).
  Keep `import dataclasses`, `import galois`, `import numpy as np`, `from qldpc.codes.common import CSSCode, QuditCode`, `from qldpc.objects import Pauli`. After writing, `grep -n "\.hmatrix\." src/qldpc/circuits/surgery/hmatrix/PPM_Y.py` must be empty (sibling form everywhere).

- [ ] **Step 3: Delete `y_gadget.py` and rewire source consumers.**
  ```bash
  git rm src/qldpc/circuits/surgery/y_gadget.py
  ```
  - `__init__.py:23` `from .y_gadget import YGadgetLayout, build_y_gadget` → `from .hmatrix.PPM_Y import YGadgetLayout, build_y_gadget`
  - `y_circuit.py:19` `from .y_gadget import YGadgetLayout` → `from .hmatrix.PPM_Y import YGadgetLayout`
  - `conftest.py:29` `from qldpc.circuits.surgery.y_gadget import _overlap_size` → `from qldpc.circuits.surgery.hmatrix.PPM_Y import _overlap_size`

- [ ] **Step 4: Move the test and rewire test consumers.**
  ```bash
  git mv src/qldpc/circuits/surgery/y_gadget_test.py src/qldpc/circuits/surgery/hmatrix/PPM_Y_test.py
  ```
  - In `hmatrix/PPM_Y_test.py`: replace every `from qldpc.circuits.surgery.y_gadget import …` → `from qldpc.circuits.surgery.hmatrix.PPM_Y import …` (all imported names — `build_y_gadget`, `_locate_overlaps`, `_in_rowspace_gf2`, `_merged_incidence`, `_locate_overlap`, etc. — are Y core helpers now in `PPM_Y`).
  - In `circuit_single_y_test.py:16`: `from qldpc.circuits.surgery.y_gadget import build_y_gadget` → `from qldpc.circuits.surgery.hmatrix.PPM_Y import build_y_gadget`.

- [ ] **Step 5: Verify grep gates, lint, and the suite.**
  - `grep -rn "from \.y_gadget import\|surgery\.y_gadget import" src/qldpc/circuits/surgery/` → empty.
  - `grep -n "\.hmatrix\." src/qldpc/circuits/surgery/hmatrix/PPM_Y.py src/qldpc/circuits/surgery/hmatrix/PPM_Y_obs0.py` → empty.
  - `.venv/bin/ruff check <all created/changed files>` → "All checks passed" (`--fix` any `I001`).
  - `.venv/bin/pytest src/qldpc/circuits/surgery/ -q` → `245 passed`.
  - Confirm `wc -l` on `PPM_Y.py` (~460) and `PPM_Y_obs0.py` (~222) are both ≤ ~500.

- [ ] **Step 6: Commit.**
  ```bash
  git add src/qldpc/circuits/surgery/hmatrix/PPM_Y.py src/qldpc/circuits/surgery/hmatrix/PPM_Y_obs0.py \
    src/qldpc/circuits/surgery/hmatrix/PPM_Y_test.py \
    src/qldpc/circuits/surgery/y_gadget.py src/qldpc/circuits/surgery/y_gadget_test.py \
    src/qldpc/circuits/surgery/__init__.py src/qldpc/circuits/surgery/y_circuit.py \
    src/qldpc/circuits/surgery/conftest.py src/qldpc/circuits/surgery/circuit_single_y_test.py
  git commit -m "refactor(surgery): relocate y_gadget.py -> hmatrix/PPM_Y + PPM_Y_obs0 (pure motion)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

## Plan Self-Review

**Spec coverage (Piece C row of §7, as amended to pure-motion relocation):**
- `y_gadget.py` → `hmatrix/PPM_Y.py` (core + `YGadgetLayout` + `build_y_gadget` verbatim) — Steps 2 ✓
- `hmatrix/PPM_Y_obs0.py` (`Obs0Row`/`Obs0ReadoutPlan`/`_ybar_obs0_rows`, self-contained) — Step 1 ✓
- delete `y_gadget.py`; rewire `__init__`/`y_circuit.py`/`conftest.py` — Step 3 ✓
- mirror `y_gadget_test.py` — Step 4 ✓
- existing tests pass, no new golden (pure motion) — Step 5 ✓

**Placeholder scan:** "whatever else these three actually use … keep only the ones referenced; ruff F401 will flag extras" (Step 1) is a concrete import-pruning procedure for a verbatim move, not a logic placeholder. No `TBD`/unspecified-code steps.

**Type/name consistency:** Symbol names unchanged across all steps. The acyclic direction (`PPM_Y` imports from `PPM_Y_obs0`, never the reverse) is consistent with the verified coupling (the `Obs0` region calls no Y core helper). `circuit_single_y_test.py` stays flat (moves to `circuit/` in Piece D), only its `build_y_gadget` import repointed.
