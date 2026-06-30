# Surgery Reorg — Piece B: joint closed-form (end-to-end) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the verbose `_stitch_intercode`/`_stitch_intracode` joint-PPM merged-check assembly (currently in `circuit.py`) with a closed-form `np.block` that reads 1:1 with `main.tex` §3 Eq. (192)/(200), relocate the whole bridge subsystem into `hmatrix/`, and prove byte-identity with a new joint golden.

**Architecture:** Piece B of the layered reorg spec (`docs/superpowers/specs/2026-06-30-surgery-layered-reorg-design.md`). The bridge **algorithm** (`Bridge`, `build_bridge`, cellulation + SkipTree) is pure-motion relocated; the joint **H-matrix assembly** is rewritten in closed form (the `_stitch_*` indirection is replaced, not transited). The replacement is **already verified byte-identical** (6 joint cases: intra Steane X/Z, intra BB[[36,8]] Z, inter Steane×Steane X/Z, inter Surface×Steane X — `np.array_equal` on `matrix_x` and `matrix_z`; `CSSCode` stores matrices verbatim so this is true byte-identity). The circuit-side joint *builders* (`build_joint_ppm_circuit` etc.) stay in `circuit.py` (they move in Piece D); only their import of the joint H-matrix changes.

**Tech Stack:** Python 3, numpy, galois (GF(2)), networkx, stim, pytest. Runner: `.venv/bin/pytest`.

## Global Constraints

- **Public API byte-identical.** `surgery/__init__.py __all__` unchanged; it re-exports `Bridge`, `build_bridge` from their new module. External `from qldpc.circuits.surgery import …` unaffected.
- **No legacy transit.** `_stitch_intercode`/`_stitch_intracode`/`_stitch_to_joint_csscode` are *deleted* from `circuit.py` and replaced by closed-form `np.block` functions born in `hmatrix/PPM_joint.py` — they are NOT moved verbatim.
- **The closed form must be byte-identical** to today's `_stitch_*` output (the joint golden, Task 1, is the regression proof).
- **The bridge algorithm relocates verbatim** — `Bridge`, `build_bridge`, `_max_basis_stabilizer_weight`, and the cellulation/SkipTree helpers are moved with no logic change (only import lines).
- **Layer direction.** `hmatrix/PPM_joint*.py` import only from `qldpc.*`, `numpy`/`networkx`, and `hmatrix/` siblings (`.PPM_XZ`) — never from `circuit`/`y_*`.
- **Citations.** New/moved H-matrix docstrings cite fully: joint adapter & SkipTree — Swaroop, Jochym-O'Connor, Yoder arXiv:2410.03628 (§III, Thm 7); cellulation — Williamson & Yoder arXiv:2410.02213; single-gadget primitives — Webster, Smith, Cohen arXiv:2511.15989 §II.A. Never `main.tex`/bare surnames.
- **File-size cap.** No source file > ~500 lines; test files mirror their source split.
- **ruff clean on every changed file** (`line-length=100`, default+`I`; E501 not enforced). Run ruff on the FULL changed set before each commit.
- **No LER/sinter tests.** Verification is deterministic: existing suite + the new joint golden hashes.
- **Scoped commits**, never `git add -A`; unrelated working-tree files (`.gitignore`, `main.*`, notebooks) untouched. Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Baseline:** `.venv/bin/pytest src/qldpc/circuits/surgery/ -q` → `244 passed` (becomes 245 after Task 1 adds the joint golden test).

---

## File Structure After Piece B

```
surgery/
  __init__.py                          (modified: Bridge/build_bridge re-export from .hmatrix.PPM_joint)
  circuit.py                           (modified: _stitch_* deleted; joint H-matrix imported from hmatrix)
  hmatrix/
    PPM_joint.py             (NEW ~430) Bridge, build_bridge, _max_basis_stabilizer_weight
                                        + closed-form _joint_merged_inter/intra/dispatch (+helpers)
    PPM_joint_cellulation.py (NEW ~330) _skip_tree, _canonical_H_R, _skip_tree_fullrank,
                                        _cellulate_port_subgraph, _build_aux_graph_strict,
                                        _connect_induced_subgraph, _edges_to_incidence_extra,
                                        _run_skiptree_on_port_subgraph
    PPM_joint_test.py        (NEW)      Bridge / build_bridge / joint-merge tests (from bridge_test.py)
    PPM_joint_cellulation_test.py (NEW) cellulation / skiptree tests (from bridge_test.py)
    PPM_joint_golden_test.py (NEW)      joint merged-matrix golden (6-case basket, dict literal)
DELETED: bridge.py, bridge_test.py
```

---

### Task 1: Joint golden regression baseline

Create a golden test that pins TODAY's joint merged-check matrices across the verified 6-case basket, capturing hashes from the **current** `circuit._stitch_to_joint_csscode`. This is the byte-identity regression that Task 3's rewrite must satisfy.

**Files:**
- Create: `src/qldpc/circuits/surgery/hmatrix/PPM_joint_golden_test.py`

**Interfaces:**
- Consumes (current): `from qldpc.circuits.surgery.circuit import _stitch_to_joint_csscode`; `from qldpc.circuits.surgery import build_bridge`; `from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget`. (Task 3 rewires the dispatch import.)

- [ ] **Step 1: Write the golden test module.** Create the file with the 6-case basket (identical construction to the verified proof) and a shape-aware hash, comparing recomputed hashes against an inlined `_GOLDEN` dict literal:

```python
"""Golden regression for the joint-PPM merged check matrices.

Pins CSSCode.matrix_x / matrix_z (shape-aware SHA-256) for a 6-case basket
(intra/inter, X/Z bases, mixed codes) against an inlined baseline. This proves
the closed-form np.block joint assembly (hmatrix/PPM_joint.py) is byte-identical
to the prior `_stitch_*` output. Regenerate `_GOLDEN` by pasting the output of
`_regenerate_golden()` after an INTENTIONAL change.

Joint construction: Swaroop, Jochym-O'Connor, Yoder arXiv:2410.03628 §III.
"""

from __future__ import annotations

import hashlib

import numpy as np
import sympy

from qldpc import codes
from qldpc.circuits.surgery import build_bridge
from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget
from qldpc.objects import Pauli

# Task 3 rewires this import to the closed form:
#   from qldpc.circuits.surgery.hmatrix.PPM_joint import _joint_merged_dispatch as _joint_csscode
from qldpc.circuits.surgery.circuit import _stitch_to_joint_csscode as _joint_csscode


def _canon(arr: np.ndarray) -> str:
    """Shape-aware digest: distinguishes same-bytes/different-shape matrices."""
    a = np.asarray(arr).astype(np.int_)
    h = hashlib.sha256()
    h.update(a.tobytes())
    h.update(repr(a.shape).encode())
    return h.hexdigest()


def _cases() -> list[tuple[str, object, object, object]]:
    """(name, g_l, g_r, bridge) for the verified basket."""
    out = []
    steane = codes.SteaneCode()
    xs = np.asarray(steane.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    zs = np.asarray(steane.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    out.append(("intra Steane X", steane, steane, xs, xs, Pauli.X))
    out.append(("intra Steane Z", steane, steane, zs, zs, Pauli.Z))

    xsym, ysym = sympy.symbols("x y")
    bb36 = codes.BBCode({xsym: 3, ysym: 6}, xsym**3 + ysym + ysym**2, ysym**3 + xsym + xsym**2)
    z_ops = bb36.get_logical_ops(Pauli.Z)
    z0 = np.asarray(z_ops[0]).astype(np.uint8)
    z1 = np.asarray(z_ops[1]).astype(np.uint8)
    out.append(("intra BB36 Z", bb36, bb36, z0, z1, Pauli.Z))

    c1, c2 = codes.SteaneCode(), codes.SteaneCode()
    x1 = np.asarray(c1.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    x2 = np.asarray(c2.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    out.append(("inter Steane X", c1, c2, x1, x2, Pauli.X))

    c3, c4 = codes.SteaneCode(), codes.SteaneCode()
    z3 = np.asarray(c3.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    z4 = np.asarray(c4.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    out.append(("inter Steane Z", c3, c4, z3, z4, Pauli.Z))

    sc, st = codes.SurfaceCode(3), codes.SteaneCode()
    xsc = np.asarray(sc.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    xst = np.asarray(st.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    out.append(("inter Surface3xSteane X", sc, st, xsc, xst, Pauli.X))

    built = []
    for name, cl, cr, xl, xr, basis in out:
        g_l = build_gadget(cl, xl, basis=basis)
        g_r = build_gadget(cr, xr, basis=basis)
        built.append((name, g_l, g_r, build_bridge(g_l, g_r)))
    return built


def _hashes() -> dict[str, dict[str, str]]:
    result = {}
    for name, g_l, g_r, bridge in _cases():
        merged = _joint_csscode(g_l, g_r, bridge)
        result[name] = {"matrix_x": _canon(merged.matrix_x), "matrix_z": _canon(merged.matrix_z)}
    return result


_GOLDEN: dict[str, dict[str, str]] = {}  # <-- Step 2 fills this


def test_joint_merged_matrices_match_golden() -> None:
    assert _hashes() == _GOLDEN


def _regenerate_golden() -> None:  # pragma: no cover - maintenance helper
    import pprint

    print("_GOLDEN: dict[str, dict[str, str]] = " + pprint.pformat(_hashes(), sort_dicts=True, width=100))
```

- [ ] **Step 2: Capture the baseline `_GOLDEN`.** Generate it from the CURRENT code and paste it in:

```bash
.venv/bin/python -c "import qldpc.circuits.surgery.hmatrix.PPM_joint_golden_test as g; g._regenerate_golden()"
```

Paste the printed `_GOLDEN: dict[str, dict[str, str]] = {...}` over the empty literal in Step 1.

- [ ] **Step 3: Verify the golden passes and the suite grows by one.**
  Run: `.venv/bin/pytest src/qldpc/circuits/surgery/hmatrix/PPM_joint_golden_test.py -q` → `1 passed`.
  Run: `.venv/bin/pytest src/qldpc/circuits/surgery/ -q` → `245 passed`.
  Run: `.venv/bin/ruff check src/qldpc/circuits/surgery/hmatrix/PPM_joint_golden_test.py` → clean.

- [ ] **Step 4: Commit.**

```bash
git add src/qldpc/circuits/surgery/hmatrix/PPM_joint_golden_test.py
git commit -m "test(surgery): joint merged-matrix golden baseline (6-case basket)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Relocate the bridge subsystem into `hmatrix/` (pure motion)

Split `bridge.py` into `hmatrix/PPM_joint.py` (the `Bridge` dataclass + `build_bridge` + `_max_basis_stabilizer_weight`) and `hmatrix/PPM_joint_cellulation.py` (the graph algorithms), delete `bridge.py`, rewire every importer, and split `bridge_test.py` to mirror. No logic changes. `_stitch_*` stays in `circuit.py` for now (Task 3 replaces it), importing `Bridge` from its new home.

**Files:**
- Create: `src/qldpc/circuits/surgery/hmatrix/PPM_joint.py`, `src/qldpc/circuits/surgery/hmatrix/PPM_joint_cellulation.py`, `src/qldpc/circuits/surgery/hmatrix/PPM_joint_test.py`, `src/qldpc/circuits/surgery/hmatrix/PPM_joint_cellulation_test.py`
- Delete: `src/qldpc/circuits/surgery/bridge.py`, `src/qldpc/circuits/surgery/bridge_test.py`
- Modify: `src/qldpc/circuits/surgery/__init__.py`, `src/qldpc/circuits/surgery/circuit.py`, and the joint golden test from Task 1 (its `build_bridge` import is public, so no change needed there)

**Interfaces:**
- Produces: `qldpc.circuits.surgery.hmatrix.PPM_joint` exporting `Bridge`, `build_bridge`, `_max_basis_stabilizer_weight`; `qldpc.circuits.surgery.hmatrix.PPM_joint_cellulation` exporting the eight graph helpers.

- [ ] **Step 1: Create `PPM_joint_cellulation.py`** with these functions moved **verbatim** from `bridge.py` (current line ranges): `_skip_tree` (88–150), `_canonical_H_R` (151–165), `_skip_tree_fullrank` (166–204), `_cellulate_port_subgraph` (205–259), `_build_aux_graph_strict` (260–302), `_connect_induced_subgraph` (303–327), `_edges_to_incidence_extra` (328–336), `_run_skiptree_on_port_subgraph` (337–417). Module header: `from __future__ import annotations`, `import networkx as nx`, `import numpy as np`, and a docstring citing cellulation = Williamson & Yoder arXiv:2410.02213 and SkipTree = Swaroop et al. arXiv:2410.03628 Thm 7. If any of these helpers call each other, the calls resolve in-module (same file) — no import needed.

- [ ] **Step 2: Create `PPM_joint.py`** with `class Bridge` (20–87), `_max_basis_stabilizer_weight` (418–425), and `build_bridge` (426–end) moved **verbatim** from `bridge.py`. Header imports:
  - `from __future__ import annotations`, `import dataclasses`, `import numpy as np`, `from qldpc.objects import PauliXZ`
  - `from .PPM_XZ import GadgetLayout` (sibling — was `.hmatrix.PPM_XZ` in `bridge.py`; one level shallower now). Also move `bridge.py`'s in-function `from .hmatrix.PPM_XZ import _restrict, build_gadget_augmented` (line ~519) to `from .PPM_XZ import _restrict, build_gadget_augmented`.
  - Import the cellulation helpers `build_bridge`/`_max_basis_stabilizer_weight` actually use from the new sibling: `from .PPM_joint_cellulation import (...)` — determine the exact set by what they call (run the suite; a `NameError` names any missing one).
  - Keep the module/`Bridge` docstring and its citations intact.

- [ ] **Step 3: Delete `bridge.py` and rewire source consumers.**
  ```bash
  git rm src/qldpc/circuits/surgery/bridge.py
  ```
  - `__init__.py:13` `from .bridge import Bridge, build_bridge` → `from .hmatrix.PPM_joint import Bridge, build_bridge`.
  - `circuit.py:20` `from .bridge import Bridge` → `from .hmatrix.PPM_joint import Bridge`.

- [ ] **Step 4: Split `bridge_test.py` into the two mirrored test files.** Move each test to the file matching its target:
  - **`PPM_joint_cellulation_test.py`** ← tests importing/exercising `_canonical_H_R`, `_skip_tree_fullrank`, `_build_aux_graph_strict`, `_connect_induced_subgraph`, `_cellulate_port_subgraph` (the graph helpers). Rewrite their imports to `from qldpc.circuits.surgery.hmatrix.PPM_joint_cellulation import …`.
  - **`PPM_joint_test.py`** ← tests of `Bridge`, `build_bridge`, and the joint merge (`_stitch_to_joint_csscode`). Rewrite `Bridge`/`build_bridge` imports to `from qldpc.circuits.surgery.hmatrix.PPM_joint import …`; leave `from qldpc.circuits.surgery.circuit import _stitch_to_joint_csscode` AS-IS (Task 3 rewires it). Keep the shared header imports (numpy, pytest, `from qldpc import codes`, `from qldpc.circuits.surgery.conftest import …`, `from qldpc.objects import Pauli`) in both files as each needs.
  ```bash
  git rm src/qldpc/circuits/surgery/bridge_test.py
  ```

- [ ] **Step 5: Verify motion + grep gates.**
  - `grep -rn "from \.bridge import\|surgery\.bridge import" src/qldpc/circuits/surgery/` → empty.
  - `grep -rn "\.hmatrix\.PPM_XZ" src/qldpc/circuits/surgery/hmatrix/PPM_joint.py` → empty (sibling form used).
  - `.venv/bin/ruff check <all changed/created files>` → clean (`--fix` any `I001`).
  - `.venv/bin/pytest src/qldpc/circuits/surgery/ -q` → `245 passed` (incl. the joint golden, still green — `_stitch_*` output unchanged).

- [ ] **Step 6: Commit.**
```bash
git add src/qldpc/circuits/surgery/hmatrix/PPM_joint.py \
  src/qldpc/circuits/surgery/hmatrix/PPM_joint_cellulation.py \
  src/qldpc/circuits/surgery/hmatrix/PPM_joint_test.py \
  src/qldpc/circuits/surgery/hmatrix/PPM_joint_cellulation_test.py \
  src/qldpc/circuits/surgery/bridge.py src/qldpc/circuits/surgery/bridge_test.py \
  src/qldpc/circuits/surgery/__init__.py src/qldpc/circuits/surgery/circuit.py
git commit -m "refactor(surgery): relocate bridge subsystem into hmatrix/PPM_joint*

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Closed-form `np.block` joint assembly — replace `_stitch_*`

Add the verified closed-form joint assembly to `hmatrix/PPM_joint.py`, delete `_stitch_intercode`/`_stitch_intracode`/`_stitch_to_joint_csscode` from `circuit.py`, and rewire every consumer to the closed form. The joint golden (Task 1) must stay byte-identical — that is the proof the rewrite changed nothing.

**Files:**
- Modify: `src/qldpc/circuits/surgery/hmatrix/PPM_joint.py` (append the closed-form functions), `src/qldpc/circuits/surgery/circuit.py` (delete `_stitch_*`, rewire `_stitch_to_joint_code`), `src/qldpc/circuits/surgery/hmatrix/PPM_joint_test.py` (rewire the merge import/call), `src/qldpc/circuits/surgery/hmatrix/PPM_joint_golden_test.py` (rewire the dispatch import)

**Interfaces:**
- Consumes: `Bridge` (local to `PPM_joint.py`), `GadgetLayout` (`from .PPM_XZ`), `CSSCode`, `Pauli`.
- Produces: `_joint_merged_dispatch(g_l, g_r, bridge) -> CSSCode` (mirrors the old `_stitch_to_joint_csscode`), plus `_joint_merged_intercode`, `_joint_merged_intracode`, `_select_meas_comp`, `_port_label_block`.

- [ ] **Step 1: Append the verified closed-form functions to `hmatrix/PPM_joint.py`.** Add `CSSCode` and `Pauli` to the module imports (`from qldpc.codes.common import CSSCode`, `from qldpc.objects import Pauli` — `PauliXZ` is already imported; add `Pauli`). Then add these functions exactly as below (`Bridge` and `GadgetLayout` are already in-module / imported, so do NOT re-import them):

```python
def _select_meas_comp(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int, int]:
    """Basis-dispatched (M_meas, M_comp) sources + data-row counts.

    M_meas holds the measured-basis merged check rows, M_comp the complementary;
    for basis=X these are (HX_merged, HZ_merged), swapped for basis=Z. Mirrors the
    abstraction of the prior ``_stitch_*`` helpers (Swaroop et al. arXiv:2410.03628 §III).
    """
    g_l_aug, g_r_aug = bridge.g_l_aug, bridge.g_r_aug
    if bridge.basis is Pauli.X:
        M_meas_l_src, M_comp_l_src = g_l_aug.HX_merged, g_l_aug.HZ_merged
        M_meas_r_src, M_comp_r_src = g_r_aug.HX_merged, g_r_aug.HZ_merged
        m_meas_l_data = g_l.code.matrix_x.shape[0]
        m_meas_r_data = g_r.code.matrix_x.shape[0]
        m_comp_l_data = g_l.code.matrix_z.shape[0]
        m_comp_r_data = g_r.code.matrix_z.shape[0]
    else:
        M_meas_l_src, M_comp_l_src = g_l_aug.HZ_merged, g_l_aug.HX_merged
        M_meas_r_src, M_comp_r_src = g_r_aug.HZ_merged, g_r_aug.HX_merged
        m_meas_l_data = g_l.code.matrix_z.shape[0]
        m_meas_r_data = g_r.code.matrix_z.shape[0]
        m_comp_l_data = g_l.code.matrix_x.shape[0]
        m_comp_r_data = g_r.code.matrix_x.shape[0]
    return (
        np.asarray(M_meas_l_src).astype(np.int_),
        np.asarray(M_meas_r_src).astype(np.int_),
        np.asarray(M_comp_l_src).astype(np.int_),
        np.asarray(M_comp_r_src).astype(np.int_),
        m_meas_l_data,
        m_meas_r_data,
        m_comp_l_data,
        m_comp_r_data,
    )


def _port_label_block(label: tuple[int, ...], width: int) -> np.ndarray:
    """Π_s = π_{𝒫_s}^T P_{σ_s} ∈ F_2^{|label|×w}: row i has a 1 at column
    ``label[i]`` when ``label[i] >= 0`` (port vertex), else all-zero."""
    blk = np.zeros((len(label), width), dtype=np.int_)
    for v_idx, lab in enumerate(label):
        if lab >= 0:
            blk[v_idx, lab] = 1
    return blk


def _joint_merged_intercode(g_l: GadgetLayout, g_r: GadgetLayout, bridge: Bridge) -> CSSCode:
    """Inter-code joint merge (g_l.code is not g_r.code), closed form of
    Swaroop et al. arXiv:2410.03628 §III Eq. (H̃_X/H̃_Z^joint). Columns
    (Q_l | Q_r | Q'_l | Q'_r | 𝒜) with separate data blocks Q_l, Q_r."""
    assert g_l.code is not g_r.code
    field = g_l.code.field
    (M_meas_l, M_meas_r, M_comp_l, M_comp_r,
     m_meas_l, m_meas_r, m_comp_l, m_comp_r) = _select_meas_comp(g_l, g_r, bridge)

    n_l, n_r = g_l.code.num_qudits, g_r.code.num_qudits
    k_l = bridge.g_l_aug.incidence.shape[0]
    k_r = bridge.g_r_aug.incidence.shape[0]
    w = bridge.width

    HX_l = M_meas_l[:m_meas_l, :n_l]
    HX_r = M_meas_r[:m_meas_r, :n_r]
    f1T_l = M_meas_l[m_meas_l:, :n_l]
    d1_l = M_meas_l[m_meas_l:, n_l:]
    f1T_r = M_meas_r[m_meas_r:, :n_r]
    d1_r = M_meas_r[m_meas_r:, n_r:]

    HZ_l = M_comp_l[:m_comp_l, :n_l]
    f0_l = M_comp_l[:m_comp_l, n_l:]
    d0_l = M_comp_l[m_comp_l:, n_l:]
    HZ_r = M_comp_r[:m_comp_r, :n_r]
    f0_r = M_comp_r[:m_comp_r, n_r:]
    d0_r = M_comp_r[m_comp_r:, n_r:]

    Pi_l = _port_label_block(bridge.label_l, w)
    Pi_r = _port_label_block(bridge.label_r, w)
    T_l = np.asarray(bridge.T_l).astype(np.int_)
    T_r = np.asarray(bridge.T_r).astype(np.int_)
    H_R = np.asarray(bridge.H_R).astype(np.int_)
    sup_l, sup_r = len(bridge.label_l), len(bridge.label_r)
    r_l, r_r = d0_l.shape[0], d0_r.shape[0]

    def Z(rows: int, cols: int) -> np.ndarray:
        return np.zeros((rows, cols), dtype=np.int_)

    M_meas = np.block([
        [HX_l,         Z(m_meas_l, n_r), Z(m_meas_l, k_l), Z(m_meas_l, k_r), Z(m_meas_l, w)],
        [Z(m_meas_r, n_l), HX_r,         Z(m_meas_r, k_l), Z(m_meas_r, k_r), Z(m_meas_r, w)],
        [f1T_l,        Z(sup_l, n_r),    d1_l,             Z(sup_l, k_r),    Pi_l],
        [Z(sup_r, n_l), f1T_r,           Z(sup_r, k_l),    d1_r,             Pi_r],
    ]).astype(np.int_)

    M_comp = np.block([
        [HZ_l,         Z(m_comp_l, n_r), f0_l,             Z(m_comp_l, k_r), Z(m_comp_l, w)],
        [Z(m_comp_r, n_l), HZ_r,         Z(m_comp_r, k_l), f0_r,             Z(m_comp_r, w)],
        [Z(r_l, n_l),  Z(r_l, n_r),      d0_l,             Z(r_l, k_r),      Z(r_l, w)],
        [Z(r_r, n_l),  Z(r_r, n_r),      Z(r_r, k_l),      d0_r,             Z(r_r, w)],
        [Z(w - 1, n_l), Z(w - 1, n_r),   T_l,              T_r,              H_R],
    ]).astype(np.int_)

    if bridge.basis is Pauli.X:
        return CSSCode(field(M_meas), field(M_comp), is_subsystem_code=False)
    return CSSCode(field(M_comp), field(M_meas), is_subsystem_code=False)


def _joint_merged_intracode(g_l: GadgetLayout, g_r: GadgetLayout, bridge: Bridge) -> CSSCode:
    """Intra-code joint merge (g_l.code is g_r.code), closed form of Swaroop et al.
    arXiv:2410.03628 §III. Columns (Q | Q'_l | Q'_r | 𝒜): the two sides SHARE the
    single data column block Q and the single H_X/H_Z data-row block (written once)."""
    assert g_l.code is g_r.code
    field = g_l.code.field
    (M_meas_l, M_meas_r, M_comp_l, M_comp_r,
     m_meas_data, _mr, m_comp_data, _cr) = _select_meas_comp(g_l, g_r, bridge)

    n = g_l.code.num_qudits
    k_l = bridge.g_l_aug.incidence.shape[0]
    k_r = bridge.g_r_aug.incidence.shape[0]
    w = bridge.width

    HX = M_meas_l[:m_meas_data, :n]
    f1T_l = M_meas_l[m_meas_data:, :n]
    d1_l = M_meas_l[m_meas_data:, n:]
    f1T_r = M_meas_r[m_meas_data:, :n]
    d1_r = M_meas_r[m_meas_data:, n:]

    HZ = M_comp_l[:m_comp_data, :n]
    f0_l = M_comp_l[:m_comp_data, n:]
    f0_r = M_comp_r[:m_comp_data, n:]
    d0_l = M_comp_l[m_comp_data:, n:]
    d0_r = M_comp_r[m_comp_data:, n:]

    Pi_l = _port_label_block(bridge.label_l, w)
    Pi_r = _port_label_block(bridge.label_r, w)
    T_l = np.asarray(bridge.T_l).astype(np.int_)
    T_r = np.asarray(bridge.T_r).astype(np.int_)
    H_R = np.asarray(bridge.H_R).astype(np.int_)
    sup_l, sup_r = len(bridge.label_l), len(bridge.label_r)
    r_l, r_r = d0_l.shape[0], d0_r.shape[0]

    def Z(rows: int, cols: int) -> np.ndarray:
        return np.zeros((rows, cols), dtype=np.int_)

    M_meas = np.block([
        [HX,           Z(m_meas_data, k_l), Z(m_meas_data, k_r), Z(m_meas_data, w)],
        [f1T_l,        d1_l,                Z(sup_l, k_r),       Pi_l],
        [f1T_r,        Z(sup_r, k_l),       d1_r,                Pi_r],
    ]).astype(np.int_)

    M_comp = np.block([
        [HZ,           f0_l,                f0_r,                Z(m_comp_data, w)],
        [Z(r_l, n),    d0_l,                Z(r_l, k_r),         Z(r_l, w)],
        [Z(r_r, n),    Z(r_r, k_l),         d0_r,                Z(r_r, w)],
        [Z(w - 1, n),  T_l,                 T_r,                 H_R],
    ]).astype(np.int_)

    if bridge.basis is Pauli.X:
        return CSSCode(field(M_meas), field(M_comp), is_subsystem_code=False)
    return CSSCode(field(M_comp), field(M_meas), is_subsystem_code=False)


def _joint_merged_dispatch(g_l: GadgetLayout, g_r: GadgetLayout, bridge: Bridge) -> CSSCode:
    """Assemble the merged joint CSSCode; intra (shared data) vs inter dispatch."""
    if g_l.code is g_r.code:
        return _joint_merged_intracode(g_l, g_r, bridge)
    return _joint_merged_intercode(g_l, g_r, bridge)
```

- [ ] **Step 2: Delete `_stitch_*` from `circuit.py` and rewire the wrapper.** Remove `_stitch_intercode`, `_stitch_intracode`, and `_stitch_to_joint_csscode` entirely. In `_stitch_to_joint_code` (the wrapper that returns `(QuditCode, Bridge)`), replace the body call `_stitch_to_joint_csscode(g_l, g_r, bridge)` with `_joint_merged_dispatch(g_l, g_r, bridge)`, importing it at the top of `circuit.py`: `from .hmatrix.PPM_joint import _joint_merged_dispatch` (next to the existing `from .hmatrix.PPM_joint import Bridge` — combine into one line). After deletion, prune any `circuit.py` import left unused (e.g. `Pauli`, `CSSCode`) **only if** ruff F401 flags it.

- [ ] **Step 3: Rewire the two test consumers to the closed form.**
  - `hmatrix/PPM_joint_test.py`: `from qldpc.circuits.surgery.circuit import _stitch_to_joint_csscode` → `from qldpc.circuits.surgery.hmatrix.PPM_joint import _joint_merged_dispatch`; and each `_stitch_to_joint_csscode(...)` call → `_joint_merged_dispatch(...)`.
  - `hmatrix/PPM_joint_golden_test.py`: change the dispatch import to `from qldpc.circuits.surgery.hmatrix.PPM_joint import _joint_merged_dispatch as _joint_csscode` (replacing the `circuit import _stitch_to_joint_csscode as _joint_csscode` line). The `_GOLDEN` literal stays untouched.

- [ ] **Step 4: Verify byte-identity + clean.**
  - `grep -rn "_stitch_to_joint_csscode\|_stitch_intercode\|_stitch_intracode" src/qldpc/circuits/surgery/` → empty.
  - `.venv/bin/pytest src/qldpc/circuits/surgery/hmatrix/PPM_joint_golden_test.py -q` → `1 passed` (**hashes IDENTICAL — the byte-identity proof**).
  - `.venv/bin/ruff check <all changed files>` → clean.
  - `.venv/bin/pytest src/qldpc/circuits/surgery/ -q` → `245 passed`.

- [ ] **Step 5: Commit.**
```bash
git add src/qldpc/circuits/surgery/hmatrix/PPM_joint.py src/qldpc/circuits/surgery/circuit.py \
  src/qldpc/circuits/surgery/hmatrix/PPM_joint_test.py \
  src/qldpc/circuits/surgery/hmatrix/PPM_joint_golden_test.py
git commit -m "feat(surgery): closed-form np.block joint merge replaces _stitch_* (byte-identical)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Plan Self-Review

**Spec coverage (Piece B row of §7):**
- `hmatrix/PPM_joint.py` = Bridge + build_bridge (relocated) + closed-form `np.block` replacing `_stitch_*` — Tasks 2 (relocate) + 3 (closed form) ✓
- `hmatrix/PPM_joint_cellulation.py` = cellulation + SkipTree relocated — Task 2 ✓
- rewire `circuit.py`'s joint builder to import from `hmatrix/` — Task 3 Step 2 ✓
- delete `bridge.py`; mirror tests — Task 2 ✓
- new joint golden (intra + inter, X & Z) proving byte-identity — Task 1 (baseline) + Task 3 Step 4 (identical) ✓; basket = intra Steane X/Z, intra BB Z, inter Steane×Steane X/Z, inter Surface×Steane X (covers intra+inter, both bases, mixed codes)

**Placeholder scan:** The Task 1 `_GOLDEN = {}` is a deliberate fill-in-Step-2 marker (the implementer captures the baseline from the current code). Task 2's "determine the exact set by what they call / run the suite; a NameError names any missing one" is a concrete import-wiring procedure for a verbatim relocation, not a logic placeholder. No `TBD`/unspecified-code steps.

**Type/name consistency:** The closed-form names (`_joint_merged_intercode/intracode/dispatch`, `_select_meas_comp`, `_port_label_block`) are used identically in Task 1's golden import comment, Task 3's definitions, and Task 3's rewires. `_joint_merged_dispatch` replaces `_stitch_to_joint_csscode` at every call site (circuit wrapper, `PPM_joint_test`, golden test). The closed-form code is the subagent-verified candidate verbatim (byte-identical proof), only `Bridge`/`GadgetLayout` imports adjusted to their in-`hmatrix` homes.

**Ordering:** Task 1 captures the baseline from the *current* `_stitch_*` before any rewrite; Task 2 relocates the bridge algorithm (golden stays green — output unchanged); Task 3 swaps in the closed form and the golden's unchanged hashes are the regression proof. `build_bridge` is imported publicly in the golden test, so Task 2's relocation doesn't disturb it.
