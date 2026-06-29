# gadget.py closed-form flattening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `gadget.py`'s single-gadget construction as the closed-form `H̃_X`/`H̃_Z` of main.tex §2.1, flattening away `_step1/2/3` + `_projection` + `_assemble_HX_L1`, while keeping every downstream contract byte-identical.

**Architecture:** One unified shared kernel `_restrict` (V0/C0/incidence/gauge) + one X-frame closed-form core `_x_merged`; `build_gadget`/`build_gadget_augmented` dispatch basis via the X↔Z dual (swap `H_X↔H_Z`, swap merged outputs). `GadgetLayout` (10 fields) and both public builders stay byte-identical, guarded by a golden snapshot regression. Only one other file changes: `bridge.py` (3 lines, `_step1_restriction`→`_restrict`).

**Tech Stack:** Python 3.12, numpy, galois (`GF2 = galois.GF(2)`), stim (downstream only), pytest, uv (`.venv/bin/python`, `.venv/bin/pytest`).

**Spec:** `docs/superpowers/specs/2026-06-29-gadget-closed-form-refactor-design.md` (commits `64d4205`, `ff9eabe`).

## Global Constraints

- Docstrings/comments cite papers fully: authors + `arXiv:ID` + §. NEVER cite `main.tex` or bare surnames. Canonical citations: Webster, Smith, Cohen `arXiv:2511.15989` §II.A; Cain et al. `arXiv:2603.28627` §B.1; Ide, Gowda, Nadkarni, Dauphinais `arXiv:2410.02753` Eq.(62)/(68).
- No LER / `sinter` / statistical-sampling tests. Verify via byte-identity, structural matrix properties, `num_observables`, truth tables.
- `GadgetLayout` fields, names, types: UNCHANGED (`code, x, support, data_checks, incidence, partial_0, HX_merged, HZ_merged, Q_prime, basis`).
- Every merged matrix returned as `np.uint8`. Tuples (`support, data_checks, Q_prime`) are `tuple[int, ...]`; `data_checks` carries `-1` sentinels for boost-added κ (augmented only).
- Run tests with `.venv/bin/pytest` from repo root `/Users/tgzhou/Project/qLDPC`.

---

### Task 1: Golden byte-identity regression (lands on CURRENT code)

A frozen snapshot of the current `build_gadget` / `build_gadget_augmented` output
across the basket. Stored as compact SHA-256 hashes (not raw arrays). Must pass
now and stay green through every later task.

**Files:**
- Create: `src/qldpc/circuits/surgery/_gadget_golden.json` (generated, committed)
- Create: `src/qldpc/circuits/surgery/gadget_golden_test.py`

**Interfaces:**
- Consumes: `build_gadget`, `build_gadget_augmented` (current).
- Produces: `_golden_cases()` (basket iterator) and `_canon(value)` (canonical
  SHA-256), reused by the generator and the test.

- [ ] **Step 1: Write the test module (basket + hashing + comparison)**

Create `src/qldpc/circuits/surgery/gadget_golden_test.py`:

```python
"""Golden byte-identity regression for the single-gadget builders.

Freezes build_gadget / build_gadget_augmented output (Cain et al.
arXiv:2603.28627 §B.1) across a fixed basket as SHA-256 hashes, so the
closed-form refactor of gadget.py is proven byte-identical to the pre-refactor
implementation. Regenerate _gadget_golden.json only via _regenerate_golden()
against a known-good tree.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

from qldpc import codes
from qldpc.objects import Pauli
from qldpc.circuits.surgery._webster_fixture import (
    _webster_x_bar_operator,
    build_generalised_bicycle_code,
    load_webster_seed_set,
)
from qldpc.circuits.surgery.gadget import build_gadget, build_gadget_augmented

_GOLDEN = pathlib.Path(__file__).with_name("_gadget_golden.json")
_FIELDS = (
    "support", "data_checks", "incidence", "partial_0",
    "HX_merged", "HZ_merged", "Q_prime",
)


def _canon(value) -> str:
    """Shape-aware canonical SHA-256 of an array/tuple field (int64 bytes)."""
    arr = np.ascontiguousarray(np.asarray(value, dtype=np.int64))
    return hashlib.sha256(arr.tobytes() + repr(arr.shape).encode()).hexdigest()


def _golden_extra(support_len: int) -> np.ndarray:
    """Deterministic weight-2 incidence_extra: rows (0,1) and (1,2) mod L."""
    rows = []
    for j in (0, 1):
        r = np.zeros(support_len, dtype=np.uint8)
        r[j % support_len] = 1
        r[(j + 1) % support_len] = 1
        rows.append(r)
    return np.array(rows, dtype=np.uint8)


def _golden_cases():
    """Yield (tag, code, x, basis, incidence_extra | None)."""
    entries = [("Steane", codes.SteaneCode())]
    for ci in range(4):
        d = load_webster_seed_set(ci)
        entries.append((f"Webster{ci}", build_generalised_bicycle_code(d["l"], d["A"], d["B"])))
    for name, code in entries:
        for basis in (Pauli.X, Pauli.Z):
            x = np.asarray(code.get_logical_ops(basis)[0]).astype(np.uint8)
            yield (f"{name}|{basis.name}|plain", code, x, basis, None)
            extra = _golden_extra(int(np.count_nonzero(x)))
            yield (f"{name}|{basis.name}|aug", code, x, basis, extra)


def _layout_for(code, x, basis, extra):
    if extra is None:
        return build_gadget(code, x, basis=basis)
    return build_gadget_augmented(code, x, extra, basis=basis)


def _hashes() -> dict:
    out: dict[str, dict[str, str]] = {}
    for tag, code, x, basis, extra in _golden_cases():
        g = _layout_for(code, x, basis, extra)
        out[tag] = {f: _canon(getattr(g, f)) for f in _FIELDS}
    return out


def test_gadget_builders_byte_identical_to_golden() -> None:
    expected = json.loads(_GOLDEN.read_text())
    actual = _hashes()
    assert actual.keys() == expected.keys(), (
        f"case set drift: extra={set(actual) - set(expected)}, "
        f"missing={set(expected) - set(actual)}"
    )
    for tag in expected:
        for field in _FIELDS:
            assert actual[tag][field] == expected[tag][field], (
                f"byte mismatch at {tag}.{field}"
            )


def _regenerate_golden() -> None:  # pragma: no cover - manual, run on good tree
    _GOLDEN.write_text(json.dumps(_hashes(), indent=2, sort_keys=True) + "\n")
```

- [ ] **Step 2: Generate the golden fixture from current code**

Run:
```bash
cd /Users/tgzhou/Project/qLDPC
.venv/bin/python -c "from qldpc.circuits.surgery.gadget_golden_test import _regenerate_golden; _regenerate_golden()"
```
Expected: creates `src/qldpc/circuits/surgery/_gadget_golden.json` with 20 case
keys (`Steane|X|plain` … `Webster3|Z|aug`), each mapping 7 field hashes.

- [ ] **Step 3: Run the regression test (passes on current code)**

Run: `.venv/bin/pytest src/qldpc/circuits/surgery/gadget_golden_test.py -v`
Expected: PASS (`test_gadget_builders_byte_identical_to_golden`).

- [ ] **Step 4: Commit**

```bash
git add src/qldpc/circuits/surgery/gadget_golden_test.py src/qldpc/circuits/surgery/_gadget_golden.json
git commit -m "test(surgery): golden byte-identity regression for gadget builders

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add unified `_restrict` + `_x_merged` (additive, proven equal)

Introduce the new kernel + core ABOVE the existing functions. Nothing is wired
or deleted yet; `build_gadget` still runs the old `_step*` path. A unit test
proves the new functions reproduce the current outputs.

**Files:**
- Modify: `src/qldpc/circuits/surgery/gadget.py` (add two functions after `_projection`/before `_assemble_HX_L1`, leaving existing code intact)
- Modify: `src/qldpc/circuits/surgery/gadget_test.py` (append unit tests)

**Interfaces:**
- Consumes: `GF2`, `numpy`.
- Produces:
  - `_restrict(H_complement, x) -> tuple[tuple[int,...], tuple[int,...], np.ndarray, np.ndarray]`
    returning `(support, data_checks, incidence, partial_0)`.
  - `_x_merged(H_X, H_Z, x, incidence_extra=None) -> tuple[tuple[int,...], tuple[int,...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]`
    returning `(support, data_checks, incidence, partial_0, HX, HZ)` in the X frame.

- [ ] **Step 1: Write the failing unit test**

Append to `src/qldpc/circuits/surgery/gadget_test.py`:

```python
def test_restrict_matches_legacy_step1_and_gauge() -> None:
    from qldpc.circuits.surgery.gadget import (
        _restrict,
        _step1_restriction,
        _step2_gauge_fix,
    )

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    support, data_checks, incidence, partial_0 = _restrict(code.matrix_z, x)
    leg_support, leg_dc, leg_inc = _step1_restriction(code, x, basis=Pauli.X)
    assert support == leg_support
    assert data_checks == leg_dc
    assert np.array_equal(incidence, leg_inc)
    assert np.array_equal(partial_0, _step2_gauge_fix(leg_inc))


def test_x_merged_matches_legacy_build_gadget_x_frame() -> None:
    from qldpc.circuits.surgery.gadget import _x_merged, build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    sup, dc, inc, p0, HX, HZ = _x_merged(code.matrix_x, code.matrix_z, x)
    g = build_gadget(code, x, basis=Pauli.X)
    assert sup == g.support and dc == g.data_checks
    assert np.array_equal(inc, g.incidence)
    assert np.array_equal(p0, g.partial_0)
    assert np.array_equal(HX, g.HX_merged)
    assert np.array_equal(HZ, g.HZ_merged)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest src/qldpc/circuits/surgery/gadget_test.py::test_restrict_matches_legacy_step1_and_gauge src/qldpc/circuits/surgery/gadget_test.py::test_x_merged_matches_legacy_build_gadget_x_frame -v`
Expected: FAIL — `ImportError: cannot import name '_restrict'` (and `_x_merged`).

- [ ] **Step 3: Add `_restrict` and `_x_merged` to gadget.py**

Insert into `src/qldpc/circuits/surgery/gadget.py` immediately after the
`_projection` function definition:

```python
def _restrict(
    H_complement: np.ndarray,
    x: np.ndarray,
) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray, np.ndarray]:
    """Unified single-gadget kernel — the useful quantities in one place
    (Webster, Smith, Cohen arXiv:2511.15989 §II.A; Cain et al. arXiv:2603.28627 §B.1).

    V₀ = support = supp(x); C₀ = data_checks = complementary-basis checks touching V₀;
    incidence = ∂_1^T = H_complement[C₀, V₀]  (|C₀|×|V₀|, edge×vertex);
    partial_0 = ∂_0 = ker(∂_1) = GF2(incidence).left_null_space() (row-reduced, deterministic).

    ``H_complement`` is the complementary check matrix to the measured logical type
    (H_Z when measuring X̄, H_X when measuring Z̄). This is the single primitive the
    joint-PPM and Y/mixed-PPM constructions also consume.
    """
    H_complement = np.asarray(H_complement).astype(np.uint8)
    x = np.asarray(x).astype(np.uint8)
    if x.shape != (H_complement.shape[1],):
        raise ValueError(f"x has shape {x.shape}, expected ({H_complement.shape[1]},)")
    support = tuple(int(i) for i in np.nonzero(x)[0])
    if support:
        V0 = np.array(support, dtype=np.int_)
        C0 = np.nonzero(H_complement[:, V0].any(axis=1))[0]
        incidence = H_complement[np.ix_(C0, V0)].astype(np.uint8)
    else:
        C0 = np.zeros(0, dtype=np.int_)
        incidence = np.zeros((0, 0), dtype=np.uint8)
    data_checks = tuple(int(j) for j in C0)
    if incidence.size:
        partial_0 = np.asarray(GF2(incidence).left_null_space()).astype(np.uint8)
    else:
        partial_0 = np.zeros((0, incidence.shape[0]), dtype=np.uint8)
    return support, data_checks, incidence, partial_0


def _x_merged(
    H_X: np.ndarray,
    H_Z: np.ndarray,
    x: np.ndarray,
    incidence_extra: np.ndarray | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form merged matrices for measuring X̄ (Webster, Smith, Cohen
    arXiv:2511.15989 §II.A; Cain et al. arXiv:2603.28627 §B.1; Ide, Gowda, Nadkarni,
    Dauphinais arXiv:2410.02753 Eq.(62)):

        H̃_X = [[H_X,   0 ],     H̃_Z = [[H_Z,   f_0 ],
               [f_1^T, ∂_1]]            [ 0,   ∂_0 ]]

    f_1^T = π_{V₀} (indicator on support data qubits); ∂_1 = incidence.T (vertex×edge,
    entered untransposed); f_0 = π_{C₀}^T (extends complementary checks onto Q'),
    with all-zero columns for boost-added κ that back no original check; ∂_0 = ker(∂_1).

    ``incidence_extra`` (weight-2 rows, |·|×|V₀|) stacks new κ onto ∂_1^T for the
    boost / joint path; ∂_0 is re-derived on the stacked incidence.
    """
    H_X = np.asarray(H_X).astype(np.uint8)
    H_Z = np.asarray(H_Z).astype(np.uint8)
    support, data_checks, incidence, partial_0 = _restrict(H_Z, x)
    n = H_X.shape[1]
    n_C0 = len(data_checks)
    if incidence_extra is not None:
        incidence_extra = np.asarray(incidence_extra).astype(np.uint8)
        n_extra = incidence_extra.shape[0]
        incidence = np.vstack([incidence, incidence_extra]).astype(np.uint8)
        if incidence.size:
            partial_0 = np.asarray(GF2(incidence).left_null_space()).astype(np.uint8)
        else:
            partial_0 = np.zeros((0, incidence.shape[0]), dtype=np.uint8)
        data_checks = tuple(data_checks) + tuple([-1] * n_extra)
    n_V0 = len(support)
    n_cols = incidence.shape[0]  # |C₀| + n_extra
    d1 = incidence.T.astype(np.uint8)  # ∂_1, |V₀|×n_cols
    f1T = np.zeros((n_V0, n), dtype=np.uint8)
    if n_V0:
        f1T[np.arange(n_V0), np.array(support, dtype=np.int_)] = 1
    f0 = np.zeros((H_Z.shape[0], n_cols), dtype=np.uint8)
    if n_C0:
        f0[np.array(data_checks[:n_C0], dtype=np.int_), np.arange(n_C0)] = 1
    HX = np.block(
        [[H_X, np.zeros((H_X.shape[0], n_cols), dtype=np.uint8)], [f1T, d1]]
    ).astype(np.uint8)
    HZ = np.block(
        [[H_Z, f0], [np.zeros((partial_0.shape[0], n), dtype=np.uint8), partial_0]]
    ).astype(np.uint8)
    return support, data_checks, incidence, partial_0, HX, HZ
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/pytest src/qldpc/circuits/surgery/gadget_test.py::test_restrict_matches_legacy_step1_and_gauge src/qldpc/circuits/surgery/gadget_test.py::test_x_merged_matches_legacy_build_gadget_x_frame -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/gadget.py src/qldpc/circuits/surgery/gadget_test.py
git commit -m "feat(surgery): add unified _restrict kernel + _x_merged closed form

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Rewire `build_gadget` / `build_gadget_augmented` to the X↔Z dual

Replace the bodies of the two public builders with the dual dispatch over
`_x_merged`. The old `_step*` / `_projection` / `_assemble_HX_L1` stay in the
file (now used only by their direct tests and `bridge.py`) — deleted in Task 4.
The golden regression (Task 1) guards byte-identity.

**Files:**
- Modify: `src/qldpc/circuits/surgery/gadget.py` (`build_gadget` body lines ~206-246; `build_gadget_augmented` body lines ~249-309)

**Interfaces:**
- Consumes: `_x_merged` (Task 2), `GadgetLayout`, `Pauli`.
- Produces: unchanged signatures `build_gadget(code, x, *, basis) -> GadgetLayout`
  and `build_gadget_augmented(code, x, incidence_extra, *, basis) -> GadgetLayout`.

- [ ] **Step 1: Replace `build_gadget` body**

In `src/qldpc/circuits/surgery/gadget.py`, replace the body of `build_gadget`
(keep the existing signature + docstring; swap the implementation after the
docstring) with:

```python
    x = np.asarray(x).astype(np.uint8)
    if basis is Pauli.X:
        H_check = np.asarray(code.matrix_z).astype(np.uint8)
    elif basis is Pauli.Z:
        H_check = np.asarray(code.matrix_x).astype(np.uint8)
    else:
        raise ValueError(f"basis must be Pauli.X or Pauli.Z, got {basis!r}")
    if ((H_check @ x) % 2).any():
        which = "X" if basis is Pauli.X else "Z"
        comp = "H_Z" if basis is Pauli.X else "H_X"
        raise ValueError(f"x is not a logical-{which} support ({comp} @ x != 0).")

    # basis=X: X frame directly; basis=Z: X↔Z dual (swap H_X/H_Z in, swap merged out)
    # (Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III D — the dual of §2).
    if basis is Pauli.X:
        support, data_checks, incidence, partial_0, HX_m, HZ_m = _x_merged(
            code.matrix_x, code.matrix_z, x
        )
    else:
        support, data_checks, incidence, partial_0, HZ_m, HX_m = _x_merged(
            code.matrix_z, code.matrix_x, x
        )
    Q_prime = tuple(range(code.num_qudits, code.num_qudits + len(data_checks)))
    return GadgetLayout(
        code=code,
        x=x,
        support=support,
        data_checks=data_checks,
        incidence=incidence,
        partial_0=partial_0,
        HX_merged=HX_m,
        HZ_merged=HZ_m,
        Q_prime=Q_prime,
        basis=basis,
    )
```

- [ ] **Step 2: Replace `build_gadget_augmented` body**

Replace the body of `build_gadget_augmented` (keep signature + docstring) with:

```python
    x = np.asarray(x).astype(np.uint8)
    incidence_extra = np.asarray(incidence_extra).astype(np.uint8)
    support_len = int(np.count_nonzero(x))
    if incidence_extra.shape[1] != support_len:
        raise ValueError(
            f"incidence_extra has {incidence_extra.shape[1]} columns; "
            f"expected {support_len} (= |support|)"
        )
    if incidence_extra.size and not np.all(incidence_extra.sum(axis=1) == 2):
        bad = np.flatnonzero(incidence_extra.sum(axis=1) != 2).tolist()
        raise ValueError(f"incidence_extra rows {bad} have weight != 2; required weight 2.")
    if basis is Pauli.X:
        support, data_checks, incidence, partial_0, HX_m, HZ_m = _x_merged(
            code.matrix_x, code.matrix_z, x, incidence_extra
        )
    elif basis is Pauli.Z:
        support, data_checks, incidence, partial_0, HZ_m, HX_m = _x_merged(
            code.matrix_z, code.matrix_x, x, incidence_extra
        )
    else:
        raise ValueError(f"basis must be Pauli.X or Pauli.Z, got {basis!r}")
    Q_prime = tuple(range(code.num_qudits, code.num_qudits + len(data_checks)))
    return GadgetLayout(
        code=code,
        x=x,
        support=support,
        data_checks=data_checks,
        incidence=incidence,
        partial_0=partial_0,
        HX_merged=HX_m,
        HZ_merged=HZ_m,
        Q_prime=Q_prime,
        basis=basis,
    )
```

- [ ] **Step 3: Run the golden regression + the full gadget test module**

Run: `.venv/bin/pytest src/qldpc/circuits/surgery/gadget_golden_test.py src/qldpc/circuits/surgery/gadget_test.py -v`
Expected: PASS (golden byte-identical; all existing `_step*` tests still pass —
those helpers are untouched in this task).

- [ ] **Step 4: Run the broader surgery suite (circuit/bridge/cheeger build on these)**

Run: `.venv/bin/pytest src/qldpc/circuits/surgery/ -q`
Expected: PASS — no regressions in `circuit_test`, `bridge_test`, `cheeger_test`,
`y_gadget_test`, `circuit_single_y_test` (all consume `GadgetLayout` /
`build_gadget(_augmented)`, whose outputs are byte-identical).

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/gadget.py
git commit -m "refactor(surgery): build_gadget(_augmented) via X-core + X<->Z dual

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Migrate `bridge.py`, rewrite step-tests, delete dead helpers

Point the one external consumer (`bridge.py`) at `_restrict`, convert the
helper-level `gadget_test.py` tests to public-API equivalence/property tests,
then delete `_projection`, `_step1_restriction`, `_step2_gauge_fix`,
`_step3_assemble`, `_assemble_HX_L1`.

**Files:**
- Modify: `src/qldpc/circuits/surgery/bridge.py:517-526`
- Modify: `src/qldpc/circuits/surgery/gadget_test.py` (remove/retarget helper tests)
- Modify: `src/qldpc/circuits/surgery/gadget.py` (delete the 5 dead helpers)

**Interfaces:**
- Consumes: `_restrict` (Task 2).
- Produces: no public surface change; `gadget.py` no longer exports
  `_projection`, `_step1_restriction`, `_step2_gauge_fix`, `_step3_assemble`,
  `_assemble_HX_L1`.

- [ ] **Step 1: Migrate the bridge call site**

In `src/qldpc/circuits/surgery/bridge.py`, replace the import line at 517 and the
two `_step1_restriction` calls at 525-526:

Replace:
```python
    from .gadget import _step1_restriction, build_gadget_augmented
```
with:
```python
    from qldpc.objects import Pauli

    from .gadget import _restrict, build_gadget_augmented
```

Replace:
```python
    _, _, _orig_inc_l = _step1_restriction(g_l.code, g_l.x, basis=basis_l)
    _, _, _orig_inc_r = _step1_restriction(g_r.code, g_r.x, basis=basis_r)
```
with:
```python
    _Hc_l = g_l.code.matrix_z if basis_l is Pauli.X else g_l.code.matrix_x
    _Hc_r = g_r.code.matrix_z if basis_r is Pauli.X else g_r.code.matrix_x
    _orig_inc_l = _restrict(_Hc_l, g_l.x)[2]
    _orig_inc_r = _restrict(_Hc_r, g_r.x)[2]
```

- [ ] **Step 2: Run the bridge + joint suite (still green, helpers still present)**

Run: `.venv/bin/pytest src/qldpc/circuits/surgery/bridge_test.py src/qldpc/circuits/surgery/circuit_test.py -q`
Expected: PASS — bridge now uses `_restrict`; `_step1_restriction` is still in
`gadget.py` (deleted in Step 5) so nothing else breaks yet.

- [ ] **Step 3: Rewrite the helper-level gadget tests as public-API tests**

In `src/qldpc/circuits/surgery/gadget_test.py`:

(a) DELETE these tests (they import soon-deleted helpers; their assertions move to
the property tests below): `test_step2_gauge_fix_basis_property`,
`test_step2_gauge_fix_deterministic`,
`test_step3_assemble_basis_z_places_chi_in_HZ_merged_and_G_in_HX_merged`,
`test_step3_assemble_steane_css_commutes`,
`test_step3_assemble_csscode_with_distinct_nV_nC`,
`test_step2_gauge_fix_rows_linearly_independent`,
`test_step2_gauge_fix_empty_incidence_returns_zero_rows`,
`test_projection_basic_selection`, `test_projection_identity_pi_M_piT_is_submatrix`,
`test_projection_empty_indices`, `test_projection_negative_sentinel_is_zero_row`.

(b) RETARGET the `_step1_restriction` tests to `_restrict`. Replace
`test_step1_restriction_steane`, `test_step1_restriction_basis_z_uses_HX`,
`test_step1_restriction_rejects_x_shape_mismatch` with:

```python
def test_restrict_steane_x_frame() -> None:
    from qldpc.circuits.surgery.gadget import _restrict

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    support, data_checks, incidence, _ = _restrict(code.matrix_z, x)
    assert support == tuple(int(i) for i in np.where(x)[0])
    assert list(support) == sorted(support)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    touched = sorted({j for j in range(HZ.shape[0]) for i in support if HZ[j, i] == 1})
    assert data_checks == tuple(touched)
    assert incidence.shape == (len(data_checks), len(support))
    assert np.array_equal(incidence, HZ[np.ix_(data_checks, support)])
    ones = np.ones(len(support), dtype=np.uint8)
    assert np.array_equal((incidence @ ones) % 2, np.zeros(len(data_checks), dtype=np.uint8))


def test_restrict_basis_z_uses_HX() -> None:
    from qldpc.circuits.surgery.gadget import _restrict

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    support, data_checks, incidence, _ = _restrict(code.matrix_x, z)
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    touched = sorted({j for j in range(HX.shape[0]) for i in support if HX[j, i] == 1})
    assert data_checks == tuple(touched)
    assert np.array_equal(incidence, HX[np.ix_(data_checks, support)])


def test_restrict_rejects_x_shape_mismatch() -> None:
    from qldpc.circuits.surgery.gadget import _restrict

    code = codes.SteaneCode()
    bad_x = np.ones(code.num_qudits + 1, dtype=np.uint8)
    with pytest.raises(ValueError):
        _restrict(code.matrix_z, bad_x)
```

(c) ADD public-API property tests carrying the assertions from the deleted
helper tests (insert near the other `build_gadget` tests):

```python
def test_build_gadget_css_commutes_and_gauge_kernel() -> None:
    """H̃_X H̃_Z^T = 0; ∂_1 ∂_0^T = 0; rank(∂_0) = |C0| - rank(∂_1)."""
    import galois

    GF = galois.GF(2)
    for basis in (Pauli.X, Pauli.Z):
        code = codes.SteaneCode()
        x = np.asarray(code.get_logical_ops(basis)[0]).astype(np.uint8)
        g = build_gadget(code, x, basis=basis)
        HX = np.asarray(g.HX_merged).astype(np.uint8)
        HZ = np.asarray(g.HZ_merged).astype(np.uint8)
        assert np.array_equal((HX @ HZ.T) % 2, np.zeros((HX.shape[0], HZ.shape[0]), np.uint8))
        inc = np.asarray(g.incidence).astype(np.uint8)  # ∂_1^T
        p0 = np.asarray(g.partial_0).astype(np.uint8)
        assert np.array_equal((p0 @ inc) % 2, np.zeros((p0.shape[0], inc.shape[1]), np.uint8))
        r_expected = inc.shape[0] - int(np.linalg.matrix_rank(GF(inc.tolist())))
        assert p0.shape[0] == r_expected


def test_build_gadget_basis_z_places_new_x_checks_in_HZ_merged() -> None:
    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    m_Z = code.matrix_z.shape[0]
    n = code.num_qudits
    # the new S' rows live below H_Z in HZ_merged with f_1^T = π_{V0} on data
    new_rows = np.asarray(g.HZ_merged).astype(np.uint8)[m_Z:, :n]
    f1T = np.zeros((len(g.support), n), np.uint8)
    f1T[np.arange(len(g.support)), np.array(g.support, np.int_)] = 1
    assert np.array_equal(new_rows, f1T)


def test_build_gadget_product_of_new_x_checks_is_logical() -> None:
    """∏ rows(S_X') = X̄ on data, identity on Q' (Cain et al. arXiv:2603.28627 §B.1)."""
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    n = code.num_qudits
    m_X = code.matrix_x.shape[0]
    new_x_rows = np.asarray(g.HX_merged).astype(np.uint8)[m_X:]
    prod = new_x_rows.sum(axis=0) % 2
    assert np.array_equal(prod[:n], x)
    assert not prod[n:].any()
```

- [ ] **Step 4: Delete the dead helpers from gadget.py**

In `src/qldpc/circuits/surgery/gadget.py`, delete the functions `_projection`,
`_step1_restriction`, `_step2_gauge_fix`, `_assemble_HX_L1`, `_step3_assemble`
(all now unreferenced — `build_gadget`/`build_gadget_augmented` use
`_restrict`/`_x_merged`; `bridge.py` uses `_restrict`; tests retargeted).

- [ ] **Step 5: Verify no lingering references, then run the full surgery suite**

Run:
```bash
grep -rn -E "_projection|_step1_restriction|_step2_gauge_fix|_step3_assemble|_assemble_HX_L1" src/qldpc/circuits/surgery/
```
Expected: no matches (empty output).

Run: `.venv/bin/pytest src/qldpc/circuits/surgery/ -q`
Expected: PASS (entire surgery suite green, including the golden regression).

- [ ] **Step 6: Commit**

```bash
git add src/qldpc/circuits/surgery/gadget.py src/qldpc/circuits/surgery/gadget_test.py src/qldpc/circuits/surgery/bridge.py
git commit -m "refactor(surgery): drop _step*/_projection; bridge + tests on _restrict

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- §4 target structure (`_restrict`, `_x_merged`, dual dispatch) → Tasks 2-3. ✓
- §2.1 unified `_restrict` (returns 4) → Task 2 Step 3. ✓
- §5 bridge `_restrict` migration → Task 4 Step 1. ✓
- §6 step-tests → public-API equivalence/property tests → Task 4 Step 3. ✓
- §7 golden byte-identical regression (Steane + 4 Webster × {X,Z} + augmented) →
  Task 1 (basket has plain + aug per code per basis = 20 cases). ✓
- §8 acceptance: golden green (T1/T3/T4), full suite green (T3 S4, T4 S5), helpers
  gone (T4 S5 grep), only bridge changed (T4), full citations (Global Constraints). ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every run
step shows the command + expected output. ✓

**Type consistency:** `_restrict` returns the 4-tuple everywhere it is called
(`_x_merged` Step 3, bridge `[2]` indexing Task 4, tests unpack 4); `_x_merged`
returns the 6-tuple, unpacked as `...HX_m, HZ_m` (X) / `...HZ_m, HX_m` (Z)
consistently in both builders. `GadgetLayout(...)` constructed with all 10 keyword
fields in both builders. ✓
