# Surgery Module Layered Reorganization — Design

**Date:** 2026-06-30
**Author:** Tiangang Zhou (with Claude)
**Status:** Approved (pending user review)

## 1. Goal & Motivation

The `src/qldpc/circuits/surgery/` package has grown into a few monoliths
(`circuit.py` 1426, `y_circuit.py` 1275, `y_gadget.py` 799, `bridge.py` 570;
`circuit_test.py` 2254) that mix two distinct concerns and three measurement
types in the same files. This makes review hard and obscures the one axis that
future research actually moves along.

Two observations drive the reorganization:

1. **There are exactly two layers.** *H-matrix construction* (the GF(2) algebra
   that builds merged check matrices H̃_X / H̃_Z) and *surgery implementation*
   (the `stim` circuits: state prep, QEC cycle, detectors, observables). Future
   research changes the **H-matrix construction**; the circuit layer should be
   untouched by it.

2. **There are exactly three measurement types** — single X/Z PPM, single Y (and
   mixed) PPM, and joint ZZ/XX PPM. They are largely independent and should be
   separable columns.

The reorganization makes both axes explicit in the directory tree, holds every
file to a small size, and — per the same principle that produced the
already-merged single-gadget refactor
(`2026-06-29-gadget-closed-form-refactor-design.md`) — ensures every H-matrix
file is written in **closed form** matching `docs/superpowers/docs/main.tex`,
with **no legacy construction code transiting** through the new files.

## 2. Constraints (Global)

These bind every piece of the implementation:

- **Public API is byte-identical.** `surgery/__init__.py` re-exports the same
  symbols (`build_gadget`, `GadgetLayout`, `YGadgetLayout`, `Bridge`,
  `build_bridge`, `build_single_ppm_circuit`, `build_joint_ppm_circuit`,
  `build_single_y_ppm_circuit`, `keep_only_observable`, `logical_state_init`,
  `boost_gadget`, `cheeger_constant`). External `from qldpc.circuits.surgery
  import …` is unaffected.
- **Symbol names are preserved.** Only the *file a symbol lives in* changes
  (`build_gadget` stays `build_gadget` even though its file becomes
  `hmatrix/PPM_XZ.py`). Renaming public symbols is out of scope.
- **File-size cap.** Target ≤ 400 lines/source file; hard cap ~500. Each test
  file mirrors its source file's split. A single long function stays whole — the
  cap is per file, not per function.
- **Layer direction.** `circuit/` imports from `hmatrix/`, **never the reverse**,
  and `hmatrix/` modules do not import from `circuit/`.
- **Citations.** Every H-matrix docstring/comment cites papers fully (authors +
  arXiv:ID + §), never `main.tex` or bare surnames. Verified IDs: single &
  mixed/Y gadget — Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 (Eq. 62 /
  66 / 68); joint adapter & SkipTree — Swaroop, Jochym-O'Connor, Yoder
  arXiv:2410.03628 (§III, Thm 7); cellulation — Williamson & Yoder
  arXiv:2410.02213; Cheeger-distance — Cross et al. arXiv:2407.18393 (Thm 6);
  mixed-basis cross-merge — Webster, Smith, Cohen arXiv:2511.15989 (§II.B.2).
- **No LER / statistical-sampling tests.** Verification is deterministic
  (DEM-compile, `num_observables`, truth tables, structural matrix properties,
  golden hashes), never `sinter.collect`.
- **Commit hygiene.** The working tree carries unrelated uncommitted files;
  every commit is scoped via explicit `git add <paths>`, never `git add -A`.

## 3. Target Structure

```
surgery/
  __init__.py              public API re-exports — external import path unchanged
  conftest.py              shared test fixtures: Webster builders (+ seed dict),
                           _steane_y_pair, _bb_y_pair

  hmatrix/                 ← H-matrix construction ("future research edits here")
    __init__.py
    PPM_XZ.py        ~250  GadgetLayout, build_gadget(_augmented), _restrict, _x_merged
                           (already closed-form; relocated as-is)
    PPM_joint.py     ~440  Bridge, build_bridge, closed-form np.block H̃_X/H̃_Z^joint
                           (replaces _stitch_*)
    PPM_joint_cellulation.py ~330  cellulation + SkipTree (T_s, σ_s) — algorithmic
    PPM_Y.py         ~484  YGadgetLayout, build_y_gadget closed-form (§4 / Eq. 68),
                           _merged_incidence, _partial0_*
    PPM_Y_obs0.py    ~223  Obs0Row, Obs0ReadoutPlan, _ybar_obs0_rows (Ȳ readout planner)
    cheeger.py       ~470  boost_gadget*, cheeger_constant (relocated as-is)
    merge.py         ~106  apply_mixed_basis_merge (relocated as-is)

  circuit/                 ← stim circuit implementation (depends on hmatrix only)
    __init__.py
    engine.py        ~440  _surgery_{state_prep,qec_cycle,observable,
                           final_detectors,detach_and_readout}, _reliable_checks
    support.py       ~400  _surgery_qubit_coordinates, _check_lane_index_map,
                           gf2/observable algebra, keep_only_observable, logical_state_init
    PPM_XZ.py        ~150  build_single_ppm_circuit
    PPM_joint.py     ~290  build_joint_ppm_circuit, _build_*_same_basis,
                           _expand_joint_data_init
    PPM_Y.py         ~436  build_single_y_ppm_circuit, _y_final_detectors, _y_emit_*
    PPM_Y_prep.py    ~489  _YCtx, eigenstate/state prep, split/quditcode/mixed utils
    PPM_Y_qec.py     ~327  _y_qec_cycle (+_row_paulis), _y_detach_and_readout

  + one mirrored *_test.py per source file (2254-line circuit_test.py splits to match)
```

**File count per type:** X/Z = 2 · joint = 3 · Y = 5 · shared infra = 5
(`engine`, `support`, `cheeger`, `merge`, `conftest`). Y's 5 files are
proportional to its ~2074 source lines (2.7× X/Z), not special-casing: 2074
lines cannot fit a ≤500 cap in fewer than ~5 files. Y collapses toward parity
only when its circuit layer stops duplicating `engine.py` — a logic change
deferred to the Y-flatten work, **not** this reorg.

## 4. The Layer Contract

`circuit/` consumes exactly four interface objects from `hmatrix/`:
`GadgetLayout` (X/Z), `YGadgetLayout` (Y), `Bridge` (joint adapter), and the
merged joint `CSSCode`. "Future research = change H-matrix construction" means:
edit a `hmatrix/PPM_*.py`, keep its layout object's field shapes, and `circuit/`
needs no change. The three PPM types are independent columns; only
`circuit/engine.py` + `circuit/support.py` are shared infrastructure.

## 5. The Closed-Form Principle

Each H-matrix file is **born in closed form**; no legacy construction code is
relocated and then rewritten. Concretely:

- **X/Z (`PPM_XZ.py`).** Already closed-form from the merged
  `2026-06-29` refactor. Pure relocation of `gadget.py`.

- **Joint (`PPM_joint.py`).** Today's `_stitch_intercode` / `_stitch_intracode`
  (in `circuit.py`) are *already* the §3 block construction, but written as a
  manual `np.zeros` buffer with slice-assignment and a port-label `for` loop,
  re-slicing each side's `HX_merged`/`HZ_merged` after the fact. They are
  replaced by a direct `np.block([...])` reading 1:1 with arXiv:2410.03628 §III
  (main.tex Eq. 192/200), consuming the per-side single-gadget closed form
  (`_x_merged`) plus the bridge's `T_s`, `H_R`, and port-label block
  `π_{𝒫_s}^T P_{σ_s}`. The X↔Z basis is handled by the same dual-swap pattern as
  the single gadget.

  **Caveat — the bridge itself stays algorithmic.** `T_s`, the port labels
  `σ_s`, and the augmented gadgets `g_*_aug` come from **cellulation + SkipTree**
  (arXiv:2410.03628 §III.2 "Subroutines": SkipTree = Thm 7; cellulation =
  Williamson & Yoder arXiv:2410.02213). These are graph algorithms, not closed
  forms; `build_bridge` and that machinery relocate **as-is** to
  `PPM_joint.py` / `PPM_joint_cellulation.py`. Only the H̃^joint *assembly*
  becomes closed form — a formula *in terms of* the bridge's algorithmic outputs.

- **Y (`PPM_Y.py`).** `build_y_gadget` is rewritten in closed form per
  arXiv:2410.02753 Eq. 68 (main.tex §4), mirroring the single-gadget approach.
  The `Obs0` symplectic readout planner splits to `PPM_Y_obs0.py` (forced by the
  707-line post-fixture total; the readout planner is the clean seam — a distinct
  concern from merged-H-matrix construction, and it cannot move to `circuit/`
  because `build_y_gadget` builds the `Obs0ReadoutPlan` into the layout).

## 6. Fixtures

No standalone `_*.py` fixture module and no `.json` data files remain in the
package:

- **`surgery/conftest.py`** absorbs the genuinely-shared fixtures: the Webster
  generalised-bicycle builders (`build_generalised_bicycle_code`,
  `load_webster_seed_set`) with the seed-set data as a **module-level dict
  literal** (replaces `_webster_app_a.json`), plus `_steane_y_pair` /
  `_bb_y_pair` (used by 2 Y test files each). `conftest.py` is auto-discovered by
  pytest — no test module imports another.
- **Golden hashes → Python dict literal** inside `gadget_golden_test.py` (and the
  new joint / Y golden test files), replacing `_gadget_golden.json`. The golden
  test gains a "print regenerated dict" mode so regenerating the baseline stays
  one command.
- **Single-use fixtures** are inlined into their one consuming test file.

Rationale: the two shared fixtures have several consumers each (Webster → 5 test
files, Y-pairs → 2), so "put it in the corresponding test file" is undefined for
them; `conftest.py` is the DRY, pytest-native shared home. Embedding the seed and
golden data as dict literals removes the loose `.json` files without
duplicating code.

## 7. Implementation Decomposition

Four independently-verifiable pieces against the one target structure. **A
relocates the H-matrix algebra that is already closed-form or algorithmic, plus
the fixtures (pure motion); B and C each flatten one measurement type's H-matrix
end-to-end (closed-form + golden), replacing legacy in place so nothing transits;
D splits the circuit layer into the `circuit/` package (pure motion).**

D is last by necessity: the `circuit/` package cannot be created until B and C
remove the joint/Y H-matrix from `circuit.py` — Python forbids `circuit.py` and
`circuit/` coexisting, so all of `circuit.py` must move out at once, and its
joint half (`build_joint_ppm_circuit` + `_stitch_*`) both calls the shared infra
and contains the joint H-matrix that B flattens. Doing the circuit split in A
would force `_stitch_*` to transit as legacy. So through A–C the circuit/bridge/Y
modules stay flat (rewired to new `hmatrix/` import paths); D distributes them
into `circuit/` only once no H-matrix remains in them.

| Piece | Scope | Verified by |
|---|---|---|
| **A. hmatrix relocation + fixtures** | Pure motion. Create `hmatrix/` package. Relocate `gadget.py` → `hmatrix/PPM_XZ.py`, `cheeger.py` → `hmatrix/cheeger.py`, `merge.py` → `hmatrix/merge.py`; mirror their tests. Create `surgery/conftest.py` absorbing the Webster builders (seed data as dict literal) + `_steane_y_pair`/`_bb_y_pair`; embed the gadget golden hashes as a dict literal. Delete `_webster_fixture.py`, `_webster_app_a.json`, `_gadget_golden.json`. Rewire every importer (source + ~110 deep test imports). `circuit.py`/`bridge.py`/`y_*.py` stay flat, rewired to new paths. | Existing 244 tests pass unchanged; golden hashes identical; public API identical |
| **B. Joint, end-to-end** | `hmatrix/PPM_joint.py` = `Bridge` + `build_bridge` (relocated as-is) **+ closed-form `np.block` H̃_X/H̃_Z^joint replacing `_stitch_*`**; `hmatrix/PPM_joint_cellulation.py` = cellulation + SkipTree (relocated as-is); rewire `circuit.py`'s joint builder to import from `hmatrix/`. Delete `bridge.py`; mirror tests. | **New joint golden** (intra + inter, X & Z) proving byte-identity to today's `_stitch_*` output; existing tests pass |
| **C. Y, end-to-end** | `hmatrix/PPM_Y.py` + `PPM_Y_obs0.py` = closed-form `build_y_gadget` per Eq. 68 (replaces `y_gadget.py`); rewire `y_circuit.py` to import from `hmatrix/`. Delete `y_gadget.py`; mirror tests. | **New Y golden** proving byte-identity to today's `build_y_gadget` output; existing tests pass |
| **D. Circuit-package split** | Pure motion. Split `circuit.py` → `circuit/{engine,support,PPM_XZ,PPM_joint}` and `y_circuit.py` → `circuit/{PPM_Y,PPM_Y_prep,PPM_Y_qec}`; create `circuit/__init__.py`; rewire `surgery/__init__.py`; mirror the 2254-line `circuit_test.py` and `y_circuit` tests to match. | Existing tests pass unchanged; golden hashes identical; public API identical; no file > ~500 lines |

Each piece becomes its own implementation plan under this spec. Pieces B and C
mirror the structure of the already-completed single-gadget closed-form plan
(golden basket → additive closed form → swap callers → delete legacy + rewrite
tests).

## 8. Verification & Acceptance

A piece is done when:

- All surgery tests pass (`pytest src/qldpc/circuits/surgery/ -q`), count
  unchanged or increased (mirror splits add no/duplicate tests; goldens add
  cases).
- Existing and new golden hashes are identical to the pre-change baseline —
  proves the closed-form H-matrices and the pure-motion reorg changed nothing.
- Public API (`surgery/__init__.py` `__all__`) is byte-identical.
- No source file exceeds ~500 lines.
- `circuit/` modules import algebra only from `hmatrix/`; `hmatrix/` modules
  import nothing from `circuit/`.

## 9. Non-Goals

- Renaming public symbols (`build_gadget`, `GadgetLayout`, …).
- De-duplicating `circuit/PPM_Y*` against `circuit/engine.py` (Y collapses to
  parity only after this; it is a separate logic change, not this reorg).
- Any change to mixed-basis (`merge.py`) or boost (`cheeger.py`) internals beyond
  relocation.
- Touching the unrelated uncommitted working-tree files (`main.tex`,
  `examples/` notebooks).

## 10. Risks

- **Import rewiring breadth.** Deep imports of internal modules
  (`surgery.gadget`, `surgery.circuit`, …) from elsewhere in the repo must all be
  found and rewired. The plan enumerates every site; CI import resolution + the
  full test run is the net.
- **Golden faithfulness for B/C.** The current `_stitch_*` / `build_y_gadget` are
  the source of truth; the golden basket must be captured from them *before* the
  closed-form rewrite, across enough cases (codes × bases × intra/inter) to pin
  the byte-identity.
- **Y file-size pressure.** `PPM_Y_prep.py` (~489) sits near the cap; if it grows
  during the closed-form rewrite, re-split at the prep/utils seam.
