# Surgery Reorg — Piece A: hmatrix relocation + fixtures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Relocate the already-closed-form / algorithmic H-matrix modules
(`gadget.py`, `cheeger.py`, `merge.py`) into a new `hmatrix/` subpackage and
consolidate all test fixtures into `surgery/conftest.py` — pure motion, public
API and all 244 tests byte-identical.

**Architecture:** This is Piece A of the layered reorg spec
(`docs/superpowers/specs/2026-06-30-surgery-layered-reorg-design.md`). It is a
**pure-motion refactor**: no function logic changes. "Move a module" = `git mv` +
rewire every import + run the full suite green. `circuit.py`, `bridge.py`,
`y_gadget.py`, `y_circuit.py` stay flat at `surgery/`; only their *import paths*
to the relocated modules change. Pieces B/C (joint/Y flatten) and D (circuit-
package split) follow in separate plans.

**Tech Stack:** Python 3, numpy, galois (GF(2)), stim, pytest. Test runner:
`.venv/bin/pytest`.

## Global Constraints

- **Public API byte-identical.** `surgery/__init__.py` `__all__` is unchanged;
  it simply re-exports the same symbols from their new modules. External
  `from qldpc.circuits.surgery import …` is unaffected.
- **Symbol names preserved.** `build_gadget`, `GadgetLayout`,
  `build_gadget_augmented`, `_restrict`, `_x_merged`, `boost_gadget`,
  `cheeger_constant`, `apply_mixed_basis_merge` keep their names; only the file
  they live in changes.
- **No logic changes.** Moved code is moved verbatim. The only edits to moved
  code are (a) its own relative-import lines and (b) `load_webster_seed_set`
  reading a dict literal instead of a JSON file.
- **Layer direction.** `hmatrix/` modules import only from `qldpc.*` and each
  other (siblings), never from `circuit.py`/`y_*.py`.
- **No `.json`, no standalone `_*.py` fixture module.** `_webster_fixture.py`,
  `_webster_app_a.json`, `_gadget_golden.json` are all deleted; their data lives
  as dict literals in `conftest.py` / the golden test.
- **No LER / sinter tests.** Verification is the existing deterministic suite +
  golden hashes only.
- **Commit hygiene.** The working tree carries unrelated uncommitted files; every
  commit is scoped via explicit `git add <paths>`, **never** `git add -A`. Commit
  trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Per-task verification (every task):** `.venv/bin/pytest src/qldpc/circuits/surgery/ -q`
  → `244 passed`. The golden test is one of the 244, so an unchanged count proves
  the golden hashes still match.

---

## File Structure After Piece A

```
surgery/
  __init__.py            (modified: re-exports now from .hmatrix.*)
  conftest.py            (NEW: Webster builders + seed dict + _steane_y_pair/_bb_y_pair)
  hmatrix/
    __init__.py          (NEW: docstring only, no re-exports)
    PPM_XZ.py            (was gadget.py)
    PPM_XZ_test.py       (was gadget_test.py)
    PPM_XZ_golden_test.py(was gadget_golden_test.py; json embedded as dict)
    cheeger.py           (moved)
    cheeger_test.py      (moved)
    merge.py             (moved)
    merge_test.py        (moved)
  bridge.py              (modified imports only — stays flat)
  circuit.py             (modified imports only — stays flat)
  y_gadget.py            (modified: imports + _steane_y_pair/_bb_y_pair removed)
  y_circuit.py           (unchanged)
  bridge_test.py, circuit_test.py, y_gadget_test.py,
  circuit_single_y_test.py  (modified imports only — stay flat)
DELETED: _webster_fixture.py, _webster_app_a.json, _gadget_golden.json
```

---

### Task 1: Consolidate fixtures into `surgery/conftest.py`

Create `conftest.py`, move the Webster fixtures (with the JSON inlined as a dict
literal) and the `_steane_y_pair`/`_bb_y_pair` builders into it, delete the
standalone fixture files, and rewire every importer. No module relocation yet.

**Files:**
- Create: `src/qldpc/circuits/surgery/conftest.py`
- Delete: `src/qldpc/circuits/surgery/_webster_fixture.py`,
  `src/qldpc/circuits/surgery/_webster_app_a.json`
- Modify: `src/qldpc/circuits/surgery/y_gadget.py` (remove `_steane_y_pair`,
  `_bb_y_pair`; fix docstring referencing them)
- Modify (import rewiring): `bridge_test.py`, `cheeger_test.py`,
  `circuit_test.py`, `gadget_test.py`, `gadget_golden_test.py`,
  `circuit_single_y_test.py`, `y_gadget_test.py`

**Interfaces:**
- Produces (importable from `qldpc.circuits.surgery.conftest`):
  `load_webster_seed_set(code_index: int) -> dict[str, Any]`,
  `build_generalised_bicycle_code(ell: int, A_set: list[int], B_set: list[int]) -> CSSCode`,
  `_webster_x_bar_operator(...)`, `_webster_z_bar_operator(data, name="Z_bar_1")`,
  `_steane_y_pair() -> tuple[CSSCode, np.ndarray, np.ndarray]`,
  `_bb_y_pair(overlap: int = 1) -> tuple[CSSCode, np.ndarray, np.ndarray]`.

- [ ] **Step 1: Create `conftest.py` with the Webster fixtures + seed dict.**
  Copy `load_webster_seed_set`, `build_generalised_bicycle_code`,
  `_webster_x_bar_operator`, `_webster_z_bar_operator` **verbatim** from
  `_webster_fixture.py` into a new `src/qldpc/circuits/surgery/conftest.py`
  (keep the same module imports: `from __future__ import annotations`, `numpy`,
  `typing.Any`, `CSSCode`). Replace the JSON file read in `load_webster_seed_set`
  — the lines

  ```python
  _WEBSTER_APP_A_PATH = Path(__file__).resolve().parent / "_webster_app_a.json"
  ...
      with _WEBSTER_APP_A_PATH.open() as fh:
          data = json.load(fh)
  ```

  with a module-level dict literal and an in-memory lookup:

  ```python
  # Inlined verbatim from the former _webster_app_a.json (Webster App. A seeds).
  _WEBSTER_APP_A: dict[str, Any] = {  # <paste exact parsed contents of the JSON>
      ...
  }

  def load_webster_seed_set(code_index: int) -> dict[str, Any]:
      data = _WEBSTER_APP_A
      ...  # keep the remaining body identical (whatever indexing/return it did)
  ```

  Drop the now-unused `import json` / `from pathlib import Path`.

- [ ] **Step 2: Move `_steane_y_pair` / `_bb_y_pair` into `conftest.py`.**
  Cut both functions **verbatim** from `y_gadget.py` and paste them into
  `conftest.py`. If either references other `y_gadget` names (e.g. a helper or
  `build_generalised_bicycle_code`), satisfy it inside `conftest.py`:
  `build_generalised_bicycle_code` is now local; any genuine `y_gadget` helper is
  imported with `from qldpc.circuits.surgery.y_gadget import <name>`.

- [ ] **Step 3: Delete the standalone fixture files and fix the y_gadget docstring.**

  ```bash
  git rm src/qldpc/circuits/surgery/_webster_fixture.py src/qldpc/circuits/surgery/_webster_app_a.json
  ```

  In `y_gadget.py`, remove the module-docstring bullet that documents
  `_steane_y_pair` (around line 13) so the docstring matches the moved code.

- [ ] **Step 4: Rewire `_webster_fixture` importers to `conftest`.**
  Replace, in `bridge_test.py:11`, `cheeger_test.py:11`, `circuit_test.py:12`,
  `gadget_test.py:13`:
  `from ._webster_fixture import (` → `from qldpc.circuits.surgery.conftest import (`.
  Replace, in `cheeger_test.py:70`, `circuit_test.py:683`, `gadget_test.py:268`:
  `from ._webster_fixture import _webster_z_bar_operator`
  → `from qldpc.circuits.surgery.conftest import _webster_z_bar_operator`.
  Replace, in `gadget_golden_test.py:21`:
  `from qldpc.circuits.surgery._webster_fixture import (`
  → `from qldpc.circuits.surgery.conftest import (`.

- [ ] **Step 5: Rewire the bundled Y-pair imports (split them).**
  These import statements mix a moved fixture with a still-in-`y_gadget` name, so
  split each into two lines:
  - `circuit_single_y_test.py:15`
    `from qldpc.circuits.surgery.y_gadget import _steane_y_pair, build_y_gadget`
    →
    ```python
    from qldpc.circuits.surgery.conftest import _steane_y_pair
    from qldpc.circuits.surgery.y_gadget import build_y_gadget
    ```
  - `circuit_single_y_test.py:169,246,259,277,293`
    `from qldpc.circuits.surgery.y_gadget import _bb_y_pair`
    → `from qldpc.circuits.surgery.conftest import _bb_y_pair`.
  - `y_gadget_test.py:118`
    `from qldpc.circuits.surgery.y_gadget import _bb_y_pair, _locate_overlaps`
    →
    ```python
    from qldpc.circuits.surgery.conftest import _bb_y_pair
    from qldpc.circuits.surgery.y_gadget import _locate_overlaps
    ```
  - `y_gadget_test.py:170`
    `from qldpc.circuits.surgery.y_gadget import _bb_y_pair, _merged_incidence`
    →
    ```python
    from qldpc.circuits.surgery.conftest import _bb_y_pair
    from qldpc.circuits.surgery.y_gadget import _merged_incidence
    ```

- [ ] **Step 6: Run the full surgery suite.**
  Run: `.venv/bin/pytest src/qldpc/circuits/surgery/ -q`
  Expected: `244 passed`.

- [ ] **Step 7: Commit.**

  ```bash
  git add src/qldpc/circuits/surgery/conftest.py src/qldpc/circuits/surgery/y_gadget.py \
    src/qldpc/circuits/surgery/bridge_test.py src/qldpc/circuits/surgery/cheeger_test.py \
    src/qldpc/circuits/surgery/circuit_test.py src/qldpc/circuits/surgery/gadget_test.py \
    src/qldpc/circuits/surgery/gadget_golden_test.py \
    src/qldpc/circuits/surgery/circuit_single_y_test.py src/qldpc/circuits/surgery/y_gadget_test.py \
    src/qldpc/circuits/surgery/_webster_fixture.py src/qldpc/circuits/surgery/_webster_app_a.json
  git commit -m "refactor(surgery): consolidate test fixtures into conftest.py, inline seed dict

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 2: Relocate `gadget.py` → `hmatrix/PPM_XZ.py`

Create the `hmatrix/` package, move the X/Z H-matrix module and its unit test,
and rewire every importer. (The golden test's *import* is rewired here but the
file move + JSON embed is deferred to Task 3, because its `Path(__file__)` JSON
lookup must change in the same step it moves.)

**Files:**
- Create: `src/qldpc/circuits/surgery/hmatrix/__init__.py`
- Move: `gadget.py` → `hmatrix/PPM_XZ.py`; `gadget_test.py` → `hmatrix/PPM_XZ_test.py`
- Modify (import rewiring): `bridge.py`, `circuit.py`, `cheeger.py`,
  `y_gadget.py`, `__init__.py`, `bridge_test.py`, `cheeger_test.py`,
  `circuit_test.py`, `y_gadget_test.py`, `gadget_golden_test.py`,
  and the moved `hmatrix/PPM_XZ_test.py`

**Interfaces:**
- Consumes: fixtures from `qldpc.circuits.surgery.conftest` (Task 1).
- Produces: `qldpc.circuits.surgery.hmatrix.PPM_XZ` exporting `GadgetLayout`,
  `build_gadget`, `build_gadget_augmented`, `_restrict`, `_x_merged`.

- [ ] **Step 1: Create the package and move the module + unit test.**

  ```bash
  mkdir -p src/qldpc/circuits/surgery/hmatrix
  git mv src/qldpc/circuits/surgery/gadget.py src/qldpc/circuits/surgery/hmatrix/PPM_XZ.py
  git mv src/qldpc/circuits/surgery/gadget_test.py src/qldpc/circuits/surgery/hmatrix/PPM_XZ_test.py
  ```

  Create `src/qldpc/circuits/surgery/hmatrix/__init__.py` containing only a
  docstring (no re-exports):

  ```python
  """H-matrix construction layer (merged check matrices H̃_X / H̃_Z).

  See docs/superpowers/specs/2026-06-30-surgery-layered-reorg-design.md.
  Consumers import the concrete module, e.g. ``from .hmatrix.PPM_XZ import build_gadget``.
  """
  ```

- [ ] **Step 2: Rewire source relative imports of `.gadget` → `.hmatrix.PPM_XZ`.**
  - `bridge.py:16` `from .gadget import GadgetLayout` → `from .hmatrix.PPM_XZ import GadgetLayout`
  - `bridge.py:519` `from .gadget import _restrict, build_gadget_augmented` → `from .hmatrix.PPM_XZ import _restrict, build_gadget_augmented`
  - `circuit.py:21` `from .gadget import GadgetLayout` → `from .hmatrix.PPM_XZ import GadgetLayout`
  - `cheeger.py:21` `from .gadget import GadgetLayout` → `from .hmatrix.PPM_XZ import GadgetLayout`
  - `cheeger.py:222` and `cheeger.py:373` `from .gadget import build_gadget_augmented` → `from .hmatrix.PPM_XZ import build_gadget_augmented`
  - `y_gadget.py:30` `from .gadget import GadgetLayout, build_gadget` → `from .hmatrix.PPM_XZ import GadgetLayout, build_gadget`
  - `__init__.py:21` `from .gadget import GadgetLayout, build_gadget` → `from .hmatrix.PPM_XZ import GadgetLayout, build_gadget`

- [ ] **Step 3: Rewire all test absolute imports.**
  Replace every occurrence of the string `qldpc.circuits.surgery.gadget import`
  with `qldpc.circuits.surgery.hmatrix.PPM_XZ import` across
  `hmatrix/PPM_XZ_test.py`, `bridge_test.py`, `cheeger_test.py`,
  `circuit_test.py`, `y_gadget_test.py`, `gadget_golden_test.py`. This is a safe
  literal substitution — `surgery.y_gadget` does **not** contain the substring
  `surgery.gadget`, so it is untouched. Verify with:

  ```bash
  grep -rn "surgery\.gadget import" src/qldpc/circuits/surgery/   # expect: no matches
  ```

- [ ] **Step 4: Run the full surgery suite.**
  Run: `.venv/bin/pytest src/qldpc/circuits/surgery/ -q`
  Expected: `244 passed`. (The golden test still reads `_gadget_golden.json` from
  the package root and recomputes hashes from the relocated `build_gadget`; pure
  motion ⇒ hashes identical ⇒ pass.)

- [ ] **Step 5: Commit.**

  ```bash
  git add src/qldpc/circuits/surgery/hmatrix/__init__.py \
    src/qldpc/circuits/surgery/hmatrix/PPM_XZ.py src/qldpc/circuits/surgery/hmatrix/PPM_XZ_test.py \
    src/qldpc/circuits/surgery/gadget.py src/qldpc/circuits/surgery/gadget_test.py \
    src/qldpc/circuits/surgery/bridge.py src/qldpc/circuits/surgery/circuit.py \
    src/qldpc/circuits/surgery/cheeger.py src/qldpc/circuits/surgery/y_gadget.py \
    src/qldpc/circuits/surgery/__init__.py src/qldpc/circuits/surgery/bridge_test.py \
    src/qldpc/circuits/surgery/cheeger_test.py src/qldpc/circuits/surgery/circuit_test.py \
    src/qldpc/circuits/surgery/y_gadget_test.py src/qldpc/circuits/surgery/gadget_golden_test.py
  git commit -m "refactor(surgery): relocate gadget.py -> hmatrix/PPM_XZ.py

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 3: Embed the gadget golden as a dict + move the golden test

Inline `_gadget_golden.json` as a Python dict literal in the golden test, give it
a print-based regenerate mode, delete the JSON, and move the test into
`hmatrix/`.

**Files:**
- Modify then move: `gadget_golden_test.py` → `hmatrix/PPM_XZ_golden_test.py`
- Delete: `_gadget_golden.json`

**Interfaces:**
- Consumes: `qldpc.circuits.surgery.hmatrix.PPM_XZ` (build_gadget*),
  `qldpc.circuits.surgery.conftest` (load_webster_seed_set) — both from A1/A2.

- [ ] **Step 1: Inline the golden hashes as a dict literal.**
  In `gadget_golden_test.py`, replace the file-backed baseline

  ```python
  _GOLDEN = pathlib.Path(__file__).with_name("_gadget_golden.json")
  ...
      expected = json.loads(_GOLDEN.read_text())
  ```

  with a module-level dict literal holding the exact current contents of
  `_gadget_golden.json` (read the file, paste its parsed value), and compare
  against it directly:

  ```python
  _GOLDEN: dict[str, str] = {  # <paste exact parsed contents of _gadget_golden.json>
      ...
  }
  ...
      expected = _GOLDEN
  ```

  Drop the now-unused `import json` / `import pathlib` if nothing else uses them.

- [ ] **Step 2: Convert `_regenerate_golden()` to print the dict.**
  Replace the file-writing body

  ```python
      _GOLDEN.write_text(json.dumps(_hashes(), indent=2, sort_keys=True) + "\n")
  ```

  with a printer the maintainer pastes back into `_GOLDEN`:

  ```python
      import pprint
      print("_GOLDEN = " + pprint.pformat(_hashes(), sort_dicts=True))
  ```

  Update the module docstring line that says "Regenerate `_gadget_golden.json`
  only via `_regenerate_golden()`" to "Regenerate the `_GOLDEN` dict by pasting
  the output of `_regenerate_golden()`."

- [ ] **Step 3: Delete the JSON and move the test.**

  ```bash
  git rm src/qldpc/circuits/surgery/_gadget_golden.json
  git mv src/qldpc/circuits/surgery/gadget_golden_test.py \
    src/qldpc/circuits/surgery/hmatrix/PPM_XZ_golden_test.py
  ```

  (The test no longer references `__file__`, so the move is path-safe.)

- [ ] **Step 4: Run the golden test, then the full suite.**
  Run: `.venv/bin/pytest src/qldpc/circuits/surgery/hmatrix/PPM_XZ_golden_test.py -q`
  Expected: PASS (hashes match the inlined dict).
  Run: `.venv/bin/pytest src/qldpc/circuits/surgery/ -q`
  Expected: `244 passed`.

- [ ] **Step 5: Commit.**

  ```bash
  git add src/qldpc/circuits/surgery/hmatrix/PPM_XZ_golden_test.py \
    src/qldpc/circuits/surgery/gadget_golden_test.py src/qldpc/circuits/surgery/_gadget_golden.json
  git commit -m "refactor(surgery): inline gadget golden as dict literal, move under hmatrix/

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task 4: Relocate `cheeger.py` and `merge.py` → `hmatrix/`

Move the boost and mixed-basis-merge modules (and their tests) into `hmatrix/`,
and fix imports: their now-sibling reference to `PPM_XZ`, and their consumers'
references to them.

**Files:**
- Move: `cheeger.py` → `hmatrix/cheeger.py`; `cheeger_test.py` → `hmatrix/cheeger_test.py`;
  `merge.py` → `hmatrix/merge.py`; `merge_test.py` → `hmatrix/merge_test.py`
- Modify (import rewiring): `__init__.py`, `y_gadget.py`, the moved
  `hmatrix/cheeger.py`, the moved `hmatrix/cheeger_test.py`, `hmatrix/merge_test.py`

**Interfaces:**
- Produces: `qldpc.circuits.surgery.hmatrix.cheeger` (`boost_gadget`,
  `cheeger_constant`), `qldpc.circuits.surgery.hmatrix.merge`
  (`apply_mixed_basis_merge`).

- [ ] **Step 1: Move the modules and their tests.**

  ```bash
  git mv src/qldpc/circuits/surgery/cheeger.py src/qldpc/circuits/surgery/hmatrix/cheeger.py
  git mv src/qldpc/circuits/surgery/cheeger_test.py src/qldpc/circuits/surgery/hmatrix/cheeger_test.py
  git mv src/qldpc/circuits/surgery/merge.py src/qldpc/circuits/surgery/hmatrix/merge.py
  git mv src/qldpc/circuits/surgery/merge_test.py src/qldpc/circuits/surgery/hmatrix/merge_test.py
  ```

- [ ] **Step 2: Fix `cheeger.py`'s now-sibling import of `PPM_XZ`.**
  In `hmatrix/cheeger.py`, the imports set to `.hmatrix.PPM_XZ` in Task 2 are now
  one level too deep (cheeger lives *inside* `hmatrix/`). Change all three:
  - `from .hmatrix.PPM_XZ import GadgetLayout` → `from .PPM_XZ import GadgetLayout`
  - `from .hmatrix.PPM_XZ import build_gadget_augmented` → `from .PPM_XZ import build_gadget_augmented` (both occurrences)

  Check `hmatrix/merge.py` for any `from .` sibling imports of surgery modules and
  apply the same one-level fix if present (grep `^from \.` in the moved file).

- [ ] **Step 3: Rewire consumers of `.cheeger` / `.merge`.**
  - `__init__.py:14` `from .cheeger import boost_gadget, cheeger_constant` → `from .hmatrix.cheeger import boost_gadget, cheeger_constant`
  - `y_gadget.py:31` `from .merge import apply_mixed_basis_merge` → `from .hmatrix.merge import apply_mixed_basis_merge`
  - `y_gadget.py:678` `from .cheeger import boost_gadget, cheeger_constant` → `from .hmatrix.cheeger import boost_gadget, cheeger_constant`

- [ ] **Step 4: Fix the moved tests' imports.**
  In `hmatrix/cheeger_test.py` and `hmatrix/merge_test.py`, make any
  `from qldpc.circuits.surgery.cheeger import …` / `…surgery.merge import …`
  absolute-path the new location: `qldpc.circuits.surgery.hmatrix.cheeger` /
  `qldpc.circuits.surgery.hmatrix.merge`. Their `surgery.hmatrix.PPM_XZ` and
  `surgery.conftest` imports (set in A1/A2) are already correct. Confirm no stale
  references remain:

  ```bash
  grep -rn "surgery\.cheeger import\|surgery\.merge import\|from \.cheeger\|from \.merge" src/qldpc/circuits/surgery/   # expect: no matches
  ```

- [ ] **Step 5: Run the full surgery suite.**
  Run: `.venv/bin/pytest src/qldpc/circuits/surgery/ -q`
  Expected: `244 passed`.

- [ ] **Step 6: Commit.**

  ```bash
  git add src/qldpc/circuits/surgery/hmatrix/cheeger.py src/qldpc/circuits/surgery/hmatrix/cheeger_test.py \
    src/qldpc/circuits/surgery/hmatrix/merge.py src/qldpc/circuits/surgery/hmatrix/merge_test.py \
    src/qldpc/circuits/surgery/cheeger.py src/qldpc/circuits/surgery/cheeger_test.py \
    src/qldpc/circuits/surgery/merge.py src/qldpc/circuits/surgery/merge_test.py \
    src/qldpc/circuits/surgery/__init__.py src/qldpc/circuits/surgery/y_gadget.py
  git commit -m "refactor(surgery): relocate cheeger.py + merge.py into hmatrix/

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

## Plan Self-Review

**Spec coverage (Piece A row of §7):**
- `gadget.py` → `hmatrix/PPM_XZ.py` — Task 2 ✓
- `cheeger.py`, `merge.py` → `hmatrix/` — Task 4 ✓
- `conftest.py` absorbing Webster + `_steane_y_pair`/`_bb_y_pair`, seed as dict — Task 1 ✓
- gadget golden hashes as dict literal — Task 3 ✓
- delete `_webster_fixture.py`, `_webster_app_a.json`, `_gadget_golden.json` — A1, A3 ✓
- rewire every importer (source + ~110 deep test imports) — A1–A4 ✓
- `circuit.py`/`bridge.py`/`y_*.py` stay flat, rewired — ✓ (only import lines change)
- verify existing 244 + golden identical — every task Step "Run the full surgery suite" ✓

**Placeholder scan:** The two `# <paste exact parsed contents …>` markers (A1 seed
dict, A3 golden dict) are deliberate data-inlining instructions, not logic
placeholders — the implementer pastes the existing file's parsed contents
verbatim and the golden/seed verification proves correctness. No `TBD`/`handle
edge cases`/unspecified-code steps remain.

**Type/name consistency:** Symbol names are unchanged across all tasks
(`build_gadget`, `GadgetLayout`, `build_gadget_augmented`, `_restrict`,
`_x_merged`, `boost_gadget`, `cheeger_constant`, `apply_mixed_basis_merge`,
`load_webster_seed_set`, `build_generalised_bicycle_code`, `_steane_y_pair`,
`_bb_y_pair`); only module paths change. Task 4's sibling-import fix correctly
undoes Task 2's `.hmatrix.PPM_XZ` once `cheeger.py` moves inside `hmatrix/`.

**Ordering:** A1 (conftest, absolute imports) precedes the module moves so moved
test files carry stable absolute fixture imports; A2 (gadget) precedes A4
(cheeger imports gadget as a sibling); A3 embeds the golden so the golden test's
`__file__` JSON lookup is gone before it moves.
