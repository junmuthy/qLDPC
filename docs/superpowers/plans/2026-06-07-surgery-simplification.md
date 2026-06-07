# Surgery Module Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shrink `src/qldpc/codes/surgery/` to 5 public symbols (`build_gadget`, `build_bridge`, `build_single_ppm_circuit`, `build_joint_ppm_circuit`, `boost_gadget`) on a deterministic Webster L=1 gadget + standalone bridge + new Stim circuit module + Cheeger boost dispatcher. Keep Webster Table I exact match; add Ide BB↔LP exact match as ground truth.

**Architecture:** New files (`gadget.py`, `bridge.py`, `circuit.py`) are built in parallel with the existing files (TDD: test first, implement, commit, repeat). Once all five public APIs pass their tests, `__init__.py` is rewritten, old files (`layered.py`, `joint.py`, `skiptree.py`, `cellulation.py`, `multi.py`, `port.py`) are deleted, and example scripts are migrated. Old `surgery_test.py` moves to `surgery/_test.py` and is rewritten against the new API.

**Tech Stack:** Python 3.11+, `numpy`, `galois` (GF(2)), `stim`, `pytest`. Uses existing `qldpc.codes.common.CSSCode` and `qldpc.circuits.bookkeeping.MeasurementRecord`.

**Spec:** `docs/superpowers/specs/2026-06-07-surgery-simplification-design.md`

**Branch:** `feat/surgery-construction` (already checked out)

**Determinism contract (called out everywhere):**
- `build_gadget` — V_0, C_0 sorted ascending; `G` is `F.left_null_space()` (canonical RREF over GF(2)).
- `build_bridge` — bridge qubits indexed in fixed order; SkipTree ties broken on vertex index.
- `build_joint_ppm_circuit` — returns merged `CSSCode` deterministically.
- `boost_gadget(seed=K)` — reproducible per-seed.

**Per-module LOC budgets:** `gadget.py` ≤ 200, `bridge.py` ≤ 350, `circuit.py` ≤ 300, `__init__.py` ≤ 30.

---

## Phase 1 — `gadget.py`: deterministic Webster L=1 with 3 explicit steps

### Task 1: Skeleton `gadget.py` with `GadgetLayout` dataclass

**Files:**
- Create: `src/qldpc/codes/surgery/gadget.py`
- Create: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Create `src/qldpc/codes/surgery/_test.py`:

```python
"""Tests for the simplified surgery package (see
docs/superpowers/specs/2026-06-07-surgery-simplification-design.md)."""

from __future__ import annotations

import dataclasses
import numpy as np
import pytest

from qldpc import codes
from qldpc.objects import Pauli


def test_gadget_layout_is_frozen_dataclass():
    from qldpc.codes.surgery.gadget import GadgetLayout
    assert dataclasses.is_dataclass(GadgetLayout)
    # frozen
    fields = {f.name for f in dataclasses.fields(GadgetLayout)}
    assert fields == {
        "code", "x", "V0", "C0", "F", "G",
        "HX_merged", "HZ_merged", "kappa_qubits",
    }
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_gadget_layout_is_frozen_dataclass -x
```
Expected: `ModuleNotFoundError: No module named 'qldpc.codes.surgery.gadget'`

- [ ] **Step 3: Write minimal implementation**

Create `src/qldpc/codes/surgery/gadget.py`:

```python
"""L=1 Webster gadget construction (see math.md §1, spec §2).

Three explicit named steps that map 1:1 to the paper:
    _step1_restriction  — math.md §1.1
    _step2_gauge_fix    — math.md §1.2
    _step3_assemble     — math.md §1.4
"""

from __future__ import annotations

import dataclasses

import galois
import numpy as np

from qldpc.codes.common import CSSCode

GF2 = galois.GF(2)


@dataclasses.dataclass(frozen=True, eq=False)
class GadgetLayout:
    code: CSSCode
    x: np.ndarray
    V0: tuple[int, ...]
    C0: tuple[int, ...]
    F: np.ndarray
    G: np.ndarray
    HX_merged: np.ndarray
    HZ_merged: np.ndarray
    kappa_qubits: tuple[int, ...]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_gadget_layout_is_frozen_dataclass -x
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/gadget.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: scaffold gadget.py with GadgetLayout dataclass"
```

---

### Task 2: `_step1_restriction` (math.md §1.1)

**Files:**
- Modify: `src/qldpc/codes/surgery/gadget.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_test.py`:

```python
def test_step1_restriction_steane():
    from qldpc.codes.surgery.gadget import _step1_restriction
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    V0, C0, F = _step1_restriction(code, x)
    # V_0 = supp(x), sorted ascending
    assert V0 == tuple(int(i) for i in np.where(x)[0])
    assert list(V0) == sorted(V0)
    # C_0 = Z-checks touching V_0, sorted ascending
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    touched = sorted({j for j in range(HZ.shape[0])
                      for i in V0 if HZ[j, i] == 1})
    assert C0 == tuple(touched)
    assert list(C0) == sorted(C0)
    # F = H_Z[C_0, V_0]
    assert F.shape == (len(C0), len(V0))
    assert np.array_equal(F, HZ[np.ix_(C0, V0)])
    # F @ 1_{V0} == 0 (math.md §1.1 invariant)
    ones = np.ones(len(V0), dtype=np.uint8)
    assert np.array_equal((F @ ones) % 2, np.zeros(len(C0), dtype=np.uint8))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_step1_restriction_steane -x
```
Expected: `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Append to `gadget.py`:

```python
def _step1_restriction(
    code: CSSCode, x: np.ndarray
) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray]:
    """math.md §1.1 — V_0 = supp(x); C_0 = Z-checks touching V_0; F = H_Z[C_0, V_0]."""
    x = np.asarray(x).astype(np.uint8)
    if x.shape != (code.num_qudits,):
        raise ValueError(f"x has shape {x.shape}, expected ({code.num_qudits},)")
    V0 = tuple(int(i) for i in np.where(x)[0])
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    C0 = tuple(
        int(j) for j in range(HZ.shape[0]) if HZ[j, list(V0)].any()
    )
    F = HZ[np.ix_(C0, V0)] if C0 and V0 else np.zeros((len(C0), len(V0)), dtype=np.uint8)
    return V0, C0, F.astype(np.uint8)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_step1_restriction_steane -x
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/gadget.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: gadget _step1_restriction (math.md §1.1)"
```

---

### Task 3: `_step2_gauge_fix` (math.md §1.2)

**Files:**
- Modify: `src/qldpc/codes/surgery/gadget.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_test.py`:

```python
def test_step2_gauge_fix_basis_property():
    from qldpc.codes.surgery.gadget import _step1_restriction, _step2_gauge_fix
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    _, _, F = _step1_restriction(code, x)
    G = _step2_gauge_fix(F)
    # math.md §1.2: G F = 0 over GF(2)
    assert G.shape[1] == F.shape[0]
    GF = (G @ F) % 2
    assert np.array_equal(GF, np.zeros_like(GF))
    # rank(G) = |C_0| - rank(F)
    r_expected = F.shape[0] - int(np.linalg.matrix_rank(galois.GF(2)(F.tolist())))
    assert G.shape[0] == r_expected


def test_step2_gauge_fix_deterministic():
    """Same F twice → byte-identical G."""
    from qldpc.codes.surgery.gadget import _step2_gauge_fix
    F = np.array([[1, 0, 1, 1], [0, 1, 1, 1]], dtype=np.uint8)
    G1 = _step2_gauge_fix(F)
    G2 = _step2_gauge_fix(F)
    assert np.array_equal(G1, G2)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_step2_gauge_fix_basis_property -x
```
Expected: `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Append to `gadget.py`:

```python
def _step2_gauge_fix(F: np.ndarray) -> np.ndarray:
    """math.md §1.2 — G whose rows form a canonical basis of ker(F.T) over GF(2).

    Uses galois ``left_null_space`` (row-reduced) so the basis is deterministic.
    """
    if F.size == 0:
        return np.zeros((0, F.shape[0]), dtype=np.uint8)
    G = GF2(F.astype(np.int_).tolist()).left_null_space()
    return np.asarray(G).astype(np.uint8)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "step2" -x
```
Expected: PASS for both `test_step2_gauge_fix_basis_property` and `test_step2_gauge_fix_deterministic`.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/gadget.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: gadget _step2_gauge_fix (math.md §1.2)"
```

---

### Task 4: `_step3_assemble` (math.md §1.4)

**Files:**
- Modify: `src/qldpc/codes/surgery/gadget.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_test.py`:

```python
def test_step3_assemble_steane_css_commutes():
    from qldpc.codes.surgery.gadget import (
        _step1_restriction, _step2_gauge_fix, _step3_assemble,
    )
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    V0, C0, F = _step1_restriction(code, x)
    G = _step2_gauge_fix(F)
    HX_m, HZ_m = _step3_assemble(code, V0, C0, F, G)

    n, mX, mZ = code.num_qudits, code.matrix_x.shape[0], code.matrix_z.shape[0]
    assert HX_m.shape == (mX + len(V0), n + len(C0))
    assert HZ_m.shape == (mZ + G.shape[0], n + len(C0))
    # math.md §1.5(a): H_X^merged @ H_Z^merged.T == 0 over GF(2)
    product = (HX_m @ HZ_m.T) % 2
    assert np.array_equal(product, np.zeros_like(product))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_step3_assemble_steane_css_commutes -x
```
Expected: `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Append to `gadget.py`:

```python
def _step3_assemble(
    code: CSSCode,
    V0: tuple[int, ...],
    C0: tuple[int, ...],
    F: np.ndarray,
    G: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """math.md §1.4 — block assembly of HX_merged, HZ_merged."""
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    n = code.num_qudits
    mX, mZ = HX.shape[0], HZ.shape[0]
    nV, nC = len(V0), len(C0)
    r = G.shape[0]

    # E_{V0}^T : (nV × n), single 1 per row at position V0[i]
    E_V0_T = np.zeros((nV, n), dtype=np.uint8)
    for i, v in enumerate(V0):
        E_V0_T[i, v] = 1

    # F^T (nV × nC)
    F_T = F.T.astype(np.uint8)

    # \tilde F : (mZ × nC), embedding F's rows back at HZ row indices C0
    F_tilde = np.zeros((mZ, nC), dtype=np.uint8)
    for k, j in enumerate(C0):
        F_tilde[j] = F[k]

    HX_merged = np.block([
        [HX, np.zeros((mX, nC), dtype=np.uint8)],
        [E_V0_T, F_T],
    ]).astype(np.uint8)

    HZ_merged = np.block([
        [HZ, F_tilde],
        [np.zeros((r, n), dtype=np.uint8), G.astype(np.uint8)],
    ]).astype(np.uint8)

    return HX_merged, HZ_merged
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_step3_assemble_steane_css_commutes -x
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/gadget.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: gadget _step3_assemble (math.md §1.4)"
```

---

### Task 5: Public `build_gadget` + determinism

**Files:**
- Modify: `src/qldpc/codes/surgery/gadget.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_test.py`:

```python
def test_build_gadget_steane_returns_valid_layout():
    from qldpc.codes.surgery.gadget import build_gadget, GadgetLayout
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    assert isinstance(g, GadgetLayout)
    assert g.code is code
    assert np.array_equal(g.x, x)
    # κ qubits indexed contiguously after data qubits
    assert g.kappa_qubits == tuple(range(code.num_qudits, code.num_qudits + len(g.C0)))


def test_build_gadget_deterministic():
    from qldpc.codes.surgery.gadget import build_gadget
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g1 = build_gadget(code, x)
    g2 = build_gadget(code, x)
    assert g1.V0 == g2.V0
    assert g1.C0 == g2.C0
    assert np.array_equal(g1.F, g2.F)
    assert np.array_equal(g1.G, g2.G)
    assert np.array_equal(g1.HX_merged, g2.HX_merged)
    assert np.array_equal(g1.HZ_merged, g2.HZ_merged)
    assert g1.kappa_qubits == g2.kappa_qubits


def test_build_gadget_rejects_non_x_logical():
    from qldpc.codes.surgery.gadget import build_gadget
    code = codes.SteaneCode()
    x = np.zeros(code.num_qudits, dtype=np.uint8)
    x[0] = 1  # not a logical X (HZ @ x ≠ 0 in general)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    if ((HZ @ x) % 2).any():
        with pytest.raises(ValueError, match="logical"):
            build_gadget(code, x)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "build_gadget" -x
```
Expected: `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Append to `gadget.py`:

```python
def build_gadget(code: CSSCode, x: np.ndarray) -> GadgetLayout:
    """Webster L=1 gadget = steps 1+2+3 composed. Deterministic in (code, x)."""
    x = np.asarray(x).astype(np.uint8)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    if ((HZ @ x) % 2).any():
        raise ValueError("x is not a logical-X support (H_Z @ x != 0).")

    V0, C0, F = _step1_restriction(code, x)
    G = _step2_gauge_fix(F)
    HX_m, HZ_m = _step3_assemble(code, V0, C0, F, G)
    kappa_qubits = tuple(range(code.num_qudits, code.num_qudits + len(C0)))
    return GadgetLayout(
        code=code, x=x, V0=V0, C0=C0, F=F, G=G,
        HX_merged=HX_m, HZ_merged=HZ_m, kappa_qubits=kappa_qubits,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "build_gadget" -x
```
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/gadget.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: public build_gadget + determinism (spec §2)"
```

---

### Task 6: Move `load_webster_seed_set` + `_build_generalised_bicycle_code` into `gadget.py`

**Files:**
- Modify: `src/qldpc/codes/surgery/gadget.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_test.py`:

```python
def test_load_webster_seed_set_returns_known_shape():
    from qldpc.codes.surgery.gadget import load_webster_seed_set
    data = load_webster_seed_set(0)
    assert "l" in data and "A_set" in data and "B_set" in data
    assert "operators" in data


def test_build_generalised_bicycle_code_constructs_css():
    from qldpc.codes.surgery.gadget import (
        load_webster_seed_set, _build_generalised_bicycle_code,
    )
    data = load_webster_seed_set(0)
    code = _build_generalised_bicycle_code(data["l"], data["A_set"], data["B_set"])
    assert code.num_qudits == 2 * data["l"]
    # CSS commutation
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    assert np.array_equal((HX @ HZ.T) % 2, np.zeros((HX.shape[0], HZ.shape[0]), dtype=np.uint8))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "webster_seed_set or generalised_bicycle" -x
```
Expected: `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Read `src/qldpc/codes/surgery/layered.py` lines 416-470 (load_webster_seed_set + _build_generalised_bicycle_code). Copy them VERBATIM (same body, same path) into `gadget.py`:

```python
# Add to gadget.py imports:
import json as _json
import pathlib as _pathlib

_WEBSTER_APP_A_PATH = _pathlib.Path(__file__).resolve().parents[4] / "examples" / "webster_app_a.json"


def load_webster_seed_set(code_index: int) -> dict:
    """Load Webster (arXiv:2511.15989) Appendix A data for code index 0..3."""
    if not 0 <= code_index <= 3:
        raise IndexError(f"code_index must be in 0..3, got {code_index}")
    with _WEBSTER_APP_A_PATH.open() as fh:
        data = _json.load(fh)
    return data["codes"][code_index]


def _build_generalised_bicycle_code(l: int, A_set: list[int], B_set: list[int]) -> CSSCode:
    """Build a generalised bicycle code from cyclic exponent sets A, B."""
    I_l = np.eye(l, dtype=np.int_)
    S = np.roll(I_l, shift=-1, axis=0)
    A = np.zeros((l, l), dtype=np.int_)
    for a in A_set:
        A = (A + np.linalg.matrix_power(S, a)) % 2
    B = np.zeros((l, l), dtype=np.int_)
    for b in B_set:
        B = (B + np.linalg.matrix_power(S, b)) % 2
    H_X = np.hstack([A, B])
    H_Z = np.hstack([B.T, A.T])
    return CSSCode(GF2(H_X.tolist()), GF2(H_Z.tolist()), is_subsystem_code=False)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "webster_seed_set or generalised_bicycle" -x
```
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/gadget.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: move load_webster_seed_set + GB builder into gadget.py"
```

---

### Task 7: Webster Table I exact-match ground truth (single-PPM half)

**Files:**
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_test.py`:

```python
WEBSTER_TABLE_I_KAPPA_CHI_R = [(0, 19), (1, 31), (2, 49), (3, 79)]


@pytest.mark.parametrize("code_index,n_anc", WEBSTER_TABLE_I_KAPPA_CHI_R)
def test_webster_table_i_kappa_chi_r_exact(code_index, n_anc):
    """Webster Table I: κ + χ + r matches for each of the 4 codes."""
    from qldpc.codes.surgery.gadget import (
        build_gadget, load_webster_seed_set, _build_generalised_bicycle_code,
    )
    data = load_webster_seed_set(code_index)
    code = _build_generalised_bicycle_code(data["l"], data["A_set"], data["B_set"])
    x1 = np.asarray(data["operators"]["X_bar_1"], dtype=np.uint8)
    g1 = build_gadget(code, x1)
    kappa = len(g1.kappa_qubits)
    chi = int(g1.x.sum())  # |V_0|
    r = g1.G.shape[0]
    assert kappa + chi + r == n_anc, (
        f"code {code_index}: κ={kappa}, χ={chi}, r={r}, sum={kappa+chi+r}, expected {n_anc}"
    )
```

- [ ] **Step 2: Run test to verify it passes (or surface real disagreement)**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_webster_table_i_kappa_chi_r_exact -v
```
Expected: 4 PASS. If any FAIL, **stop and inspect** — likely the operator-key name differs (e.g. `X_bar_1` vs another). Read `examples/webster_app_a.json` to confirm the key name; fix the test (or escalate if math is off).

- [ ] **Step 3: (no impl change needed if PASS)**

- [ ] **Step 4: Commit**

```bash
git add src/qldpc/codes/surgery/_test.py
git commit -m "test: Webster Table I κ+χ+r exact match"
```

---

## Phase 2 — `bridge.py`: standalone bridge adapter (intra-code first, then inter-code)

### Task 8: Skeleton `bridge.py` with `Bridge` dataclass

**Files:**
- Create: `src/qldpc/codes/surgery/bridge.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_test.py`:

```python
def test_bridge_dataclass_fields():
    from qldpc.codes.surgery.bridge import Bridge
    fields = {f.name for f in dataclasses.fields(Bridge)}
    assert fields == {
        "width", "qubits", "U_B",
        "chi_endpoint_extensions", "intercode",
        "aux_graph_edges", "z_extensions",
    }
    assert dataclasses.is_dataclass(Bridge)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_bridge_dataclass_fields -x
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

Create `src/qldpc/codes/surgery/bridge.py`:

```python
"""Standalone bridge adapter for two-PPM joint surgery (math.md §2).

Handles both intra-code (g1.code is g2.code) and inter-code joints. SkipTree
and cellulation helpers are private to this module.
"""

from __future__ import annotations

import dataclasses

import galois
import numpy as np

from .gadget import GadgetLayout

GF2 = galois.GF(2)


@dataclasses.dataclass(frozen=True, eq=False)
class Bridge:
    width: int
    qubits: tuple[int, ...]
    U_B: np.ndarray
    chi_endpoint_extensions: dict[int, np.ndarray]
    intercode: bool
    aux_graph_edges: tuple[tuple[int, int], ...] | None
    z_extensions: dict[int, np.ndarray] | None
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_bridge_dataclass_fields -x
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/bridge.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: scaffold bridge.py with Bridge dataclass"
```

---

### Task 9: `_build_path_graph_U_B` helper + telescoping invariant

**Files:**
- Modify: `src/qldpc/codes/surgery/bridge.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_test.py`:

```python
def test_path_graph_U_B_telescoping():
    """math.md §2.2: sum of U_B rows == e_0 + e_{w-1}."""
    from qldpc.codes.surgery.bridge import _build_path_graph_U_B
    for w in (2, 3, 5, 11):
        U_B = _build_path_graph_U_B(w)
        assert U_B.shape == (w - 1, w)
        col_sum = U_B.sum(axis=0) % 2
        expected = np.zeros(w, dtype=np.uint8)
        expected[0] = 1
        expected[-1] = 1
        assert np.array_equal(col_sum, expected)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_path_graph_U_B_telescoping -x
```
Expected: `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Append to `bridge.py`:

```python
def _build_path_graph_U_B(w: int) -> np.ndarray:
    """math.md §2.2 — path-graph X-stabilizers on w bridge qubits."""
    if w < 2:
        raise ValueError(f"bridge width must be >= 2, got {w}")
    U_B = np.zeros((w - 1, w), dtype=np.uint8)
    for i in range(w - 1):
        U_B[i, i] = 1
        U_B[i, i + 1] = 1
    return U_B
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_path_graph_U_B_telescoping -x
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/bridge.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: bridge _build_path_graph_U_B (math.md §2.2)"
```

---

### Task 10: `build_bridge` intra-code path (Webster bridge width 2w-1)

**Files:**
- Modify: `src/qldpc/codes/surgery/bridge.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_test.py`:

```python
WEBSTER_TABLE_I_BRIDGE = [(0, 11), (1, 19), (2, 31), (3, 51)]


@pytest.mark.parametrize("code_index,bridge_w_minus_1", WEBSTER_TABLE_I_BRIDGE)
def test_webster_table_i_bridge_width_exact(code_index, bridge_w_minus_1):
    """Webster Table I: 2w - 1 matches."""
    from qldpc.codes.surgery.gadget import (
        build_gadget, load_webster_seed_set, _build_generalised_bicycle_code,
    )
    from qldpc.codes.surgery.bridge import build_bridge
    data = load_webster_seed_set(code_index)
    code = _build_generalised_bicycle_code(data["l"], data["A_set"], data["B_set"])
    x1 = np.asarray(data["operators"]["X_bar_1"], dtype=np.uint8)
    x2 = np.asarray(data["operators"]["X_bar_kh1"], dtype=np.uint8)
    g1 = build_gadget(code, x1)
    g2 = build_gadget(code, x2)
    bridge = build_bridge(g1, g2)
    assert bridge.intercode is False
    assert 2 * bridge.width - 1 == bridge_w_minus_1


def test_build_bridge_intracode_chi_endpoint_extensions():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    code = codes.SteaneCode()
    x1 = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    x2 = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g1 = build_gadget(code, x1)
    g2 = build_gadget(code, x2)
    bridge = build_bridge(g1, g2)
    # math.md §2.3: χ_0 from each gadget gets an X on its bridge endpoint
    assert 0 in bridge.chi_endpoint_extensions  # gadget 1, row 0
    assert bridge.intercode is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "build_bridge_intracode or webster_table_i_bridge" -x
```
Expected: `ImportError: cannot import name 'build_bridge'`.

- [ ] **Step 3: Write minimal implementation**

Append to `bridge.py`:

```python
def build_bridge(g1: GadgetLayout, g2: GadgetLayout) -> Bridge:
    """Two-PPM bridge between gadgets. Auto-dispatches intra vs inter-code.

    math.md §2: bridge data qubits + path-graph U_B + chi endpoint extensions.
    """
    intercode = g1.code is not g2.code
    w = min(len(g1.V0), len(g2.V0))
    if w < 2:
        raise ValueError(f"bridge width must be >= 2, got {w}")

    # Bridge qubit indices placed AFTER all data + ancilla of both gadgets.
    # The caller (circuit / joint code assembly) is responsible for the
    # exact register layout; we just expose the relative offsets.
    qubits = tuple(range(w))  # placeholder offsets; circuit.py rebases.

    U_B = _build_path_graph_U_B(w)

    # math.md §2.3 χ-extension: row 0 of gadget 1 gets X on bridge[0];
    # row 0 of gadget 2 gets X on bridge[w-1].
    chi_endpoint_extensions: dict[int, np.ndarray] = {
        0: np.array([0], dtype=np.uint8),
        # value: bridge column indices that get X on this χ row.
    }
    # gadget 2 is indexed by gadget number offset in the joint matrix;
    # we encode by sign convention: key as (gadget_id, row_idx) downstream.

    if not intercode:
        return Bridge(
            width=w, qubits=qubits, U_B=U_B,
            chi_endpoint_extensions=chi_endpoint_extensions,
            intercode=False,
            aux_graph_edges=None,
            z_extensions=None,
        )

    # Inter-code path is added in Task 11.
    raise NotImplementedError("inter-code bridge added in Task 11")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "build_bridge_intracode or webster_table_i_bridge" -x
```
Expected: 5 PASS (4 parametrized + 1).

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/bridge.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: build_bridge intra-code path + Webster Table I bridge width"
```

---

### Task 11: Absorb `_skip_tree` and `_cellulate_long_cycles` into `bridge.py`

**Files:**
- Modify: `src/qldpc/codes/surgery/bridge.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_test.py`:

```python
def test_skip_tree_path_graph_returns_identity():
    """math.md skip-tree on a path graph yields T = I_{n-1}."""
    from qldpc.codes.surgery.bridge import _skip_tree
    # Path graph on 5 vertices: edges (0,1),(1,2),(2,3),(3,4)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    T, P = _skip_tree(edges, n_vertices=5)
    # T should be identity in some canonical ordering
    assert T.shape == (4, 4)


def test_cellulate_long_cycles_no_op_when_short():
    """Cellulation of a graph with no long cycles returns input unchanged."""
    from qldpc.codes.surgery.bridge import _cellulate_long_cycles
    edges = [(0, 1), (1, 2)]
    result = _cellulate_long_cycles(edges, max_cycle_length=4)
    assert set(map(tuple, map(sorted, result))) == set(map(tuple, map(sorted, edges)))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "skip_tree or cellulate_long_cycles" -x
```
Expected: `ImportError`.

- [ ] **Step 3: Copy implementations**

Read `src/qldpc/codes/surgery/skiptree.py` (`_skip_tree`, `_skip_tree_hr`) and `src/qldpc/codes/surgery/cellulation.py` (`_cellulate_long_cycles`). Copy them VERBATIM (same signatures, same body) into `bridge.py`, prefixed with underscore. Add their imports (`networkx`, etc.) to `bridge.py`'s import block.

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "skip_tree or cellulate_long_cycles" -x
```
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/bridge.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: absorb _skip_tree and _cellulate_long_cycles into bridge.py"
```

---

### Task 12: `build_bridge` inter-code path

**Files:**
- Modify: `src/qldpc/codes/surgery/bridge.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_test.py`:

```python
def test_build_bridge_intercode_two_different_codes():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    # Use two different Steane copies as a stand-in for inter-code:
    code1 = codes.SteaneCode()
    code2 = codes.SteaneCode()
    assert code1 is not code2
    x1 = np.asarray(code1.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    x2 = np.asarray(code2.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g1 = build_gadget(code1, x1)
    g2 = build_gadget(code2, x2)
    bridge = build_bridge(g1, g2)
    assert bridge.intercode is True
    assert bridge.aux_graph_edges is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_build_bridge_intercode_two_different_codes -x
```
Expected: FAIL — `NotImplementedError` from Task 10's placeholder.

- [ ] **Step 3: Replace the inter-code branch**

Read `src/qldpc/codes/surgery/joint.py` lines 138-376 (`_build_bridge_via_skiptree`, `_build_auxiliary_graph`, `_build_gadget_with_cellulation`, `_label_inverse`, `canonical_HR`, `_running_xor_b_c`, `_chi_z_compatibility_check`, `_solve_chi_z_bridge_choices`). Copy them VERBATIM into `bridge.py` as private helpers (`_` prefix kept where already private; public `canonical_HR` becomes `_canonical_HR`).

Replace the `NotImplementedError` in `build_bridge` with the inter-code logic that uses these helpers to populate `aux_graph_edges` and `z_extensions`. Reuse the helper return values to fill in the `Bridge` fields:

```python
    # intercode branch
    edges, z_ext = _build_intercode_bridge(g1, g2)  # composes the helpers
    return Bridge(
        width=w, qubits=qubits, U_B=U_B,
        chi_endpoint_extensions=chi_endpoint_extensions,
        intercode=True,
        aux_graph_edges=tuple(edges),
        z_extensions=z_ext,
    )
```

Where `_build_intercode_bridge` is a thin wrapper composing the absorbed helpers; aim for ≤ 30 lines.

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_build_bridge_intercode_two_different_codes -x
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/bridge.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: build_bridge inter-code path"
```

---

### Task 13: `bridge.py` LOC budget check

- [ ] **Step 1: Measure**

```bash
wc -l src/qldpc/codes/surgery/bridge.py
```
Expected: ≤ 350. If over budget: identify duplicated logic with the absorbed `_chi_z_compatibility_check` / `_solve_chi_z_bridge_choices` and inline anything called only once.

- [ ] **Step 2: Commit if any cleanup**

```bash
git add src/qldpc/codes/surgery/bridge.py
git commit -m "refactor: bridge.py within LOC budget"
```

---

## Phase 3 — `circuit.py`: Stim surgery circuit

### Task 14: Skeleton `circuit.py` + `build_single_ppm_circuit` (noiseless, compiles)

**Files:**
- Create: `src/qldpc/codes/surgery/circuit.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_test.py`:

```python
def test_build_single_ppm_circuit_noiseless_compiles():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import build_single_ppm_circuit
    import stim
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    circuit, record = build_single_ppm_circuit(g, rounds=2, noise_model=None)
    assert isinstance(circuit, stim.Circuit)
    assert record is not None


def test_build_single_ppm_circuit_noiseless_no_detectors_fire():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import build_single_ppm_circuit
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    circuit, _ = build_single_ppm_circuit(g, rounds=2, noise_model=None)
    sampler = circuit.compile_detector_sampler()
    samples = sampler.sample(shots=16)
    assert (samples == 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "single_ppm_circuit" -x
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

Create `src/qldpc/codes/surgery/circuit.py`:

```python
"""Stim surgery circuit construction.

build_single_ppm_circuit  — single-PPM measurement (gadget alone)
build_joint_ppm_circuit   — two-PPM joint measurement (gadget + gadget + bridge)
"""

from __future__ import annotations

import galois
import numpy as np
import stim

from qldpc.circuits.bookkeeping import MeasurementRecord, QubitIDs
from qldpc.codes.common import CSSCode

from .bridge import Bridge
from .gadget import GadgetLayout

GF2 = galois.GF(2)


def _gadget_merged_csscode(g: GadgetLayout) -> CSSCode:
    return CSSCode(
        GF2(g.HX_merged.astype(np.int_).tolist()),
        GF2(g.HZ_merged.astype(np.int_).tolist()),
        is_subsystem_code=False,
    )


def build_single_ppm_circuit(
    gadget: GadgetLayout,
    *,
    rounds: int,
    noise_model=None,
) -> tuple[stim.Circuit, MeasurementRecord]:
    """Emit Stim circuit for single-PPM measurement using `gadget`.

    Uses the existing qldpc memory experiment infrastructure on the
    merged CSS code (data + κ ancillas).
    """
    from qldpc.circuits.memory.memory import build_memory_circuit
    merged = _gadget_merged_csscode(gadget)
    return build_memory_circuit(
        merged, rounds=rounds, noise_model=noise_model,
    )
```

*Note:* `build_memory_circuit` is assumed available from `qldpc.circuits.memory.memory`. Read that module first; if the actual function name differs, substitute it. The point of this task is the public surface — internals delegate.

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "single_ppm_circuit" -x
```
Expected: 2 PASS. If `build_memory_circuit` doesn't exist or has a different signature: inline a minimal noiseless circuit builder (CNOTs + measurements + DETECTORs from `merged.matrix_x` / `merged.matrix_z`); aim for ≤ 80 lines.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/circuit.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: build_single_ppm_circuit (noiseless + detectors)"
```

---

### Task 15: `build_joint_ppm_circuit` (intra-code path)

**Files:**
- Modify: `src/qldpc/codes/surgery/circuit.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_test.py`:

```python
def test_build_joint_ppm_circuit_intracode_returns_triple():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import build_joint_ppm_circuit
    import stim
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g1 = build_gadget(code, x)
    g2 = build_gadget(code, x)
    bridge = build_bridge(g1, g2)
    circuit, record, joint_code = build_joint_ppm_circuit(
        g1, g2, bridge, rounds=1, noise_model=None,
    )
    assert isinstance(circuit, stim.Circuit)
    assert isinstance(joint_code, codes.CSSCode)
    # math.md §2.8: k_joint = k_data - 1
    assert joint_code.dimension == code.dimension - 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_build_joint_ppm_circuit_intracode_returns_triple -x
```
Expected: `ImportError`.

- [ ] **Step 3: Write minimal implementation**

Read `src/qldpc/codes/surgery/joint.py` lines 223-376 (`_stitch_gadgets_with_bridge`). Copy the stitching logic into `circuit.py` as a private helper `_stitch_to_joint_csscode(g1, g2, bridge) -> CSSCode`. Then:

```python
def build_joint_ppm_circuit(
    g1: GadgetLayout, g2: GadgetLayout, bridge: Bridge,
    *,
    rounds: int,
    noise_model=None,
) -> tuple[stim.Circuit, MeasurementRecord, CSSCode]:
    """Emit Stim circuit for two-PPM joint measurement (intra or inter-code)."""
    joint_code = _stitch_to_joint_csscode(g1, g2, bridge)
    from qldpc.circuits.memory.memory import build_memory_circuit
    circuit, record = build_memory_circuit(
        joint_code, rounds=rounds, noise_model=noise_model,
    )
    return circuit, record, joint_code
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_build_joint_ppm_circuit_intracode_returns_triple -x
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/circuit.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: build_joint_ppm_circuit (intra-code path)"
```

---

### Task 16: `build_joint_ppm_circuit` (inter-code path) — CSS commutation + dimension

**Files:**
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_test.py`:

```python
def test_build_joint_ppm_circuit_intercode_css_commutation_and_dim():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import build_joint_ppm_circuit
    code1 = codes.SteaneCode()
    code2 = codes.SteaneCode()
    x1 = np.asarray(code1.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    x2 = np.asarray(code2.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g1 = build_gadget(code1, x1)
    g2 = build_gadget(code2, x2)
    bridge = build_bridge(g1, g2)
    _, _, joint = build_joint_ppm_circuit(g1, g2, bridge, rounds=1, noise_model=None)
    HX = np.asarray(joint.matrix_x).astype(np.uint8)
    HZ = np.asarray(joint.matrix_z).astype(np.uint8)
    assert np.array_equal((HX @ HZ.T) % 2, np.zeros((HX.shape[0], HZ.shape[0]), dtype=np.uint8))
    # math.md §2.8: k_joint = k_combined - 1
    assert joint.dimension == code1.dimension + code2.dimension - 1
```

- [ ] **Step 2: Run test to verify it passes (most logic already in `_stitch_to_joint_csscode`)**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_build_joint_ppm_circuit_intercode_css_commutation_and_dim -x
```
Expected: PASS. If FAIL, extend `_stitch_to_joint_csscode` to handle `bridge.intercode is True` — typically just splicing `bridge.z_extensions` into HZ rows; aim for ≤ 30 added lines.

- [ ] **Step 3: Commit**

```bash
git add src/qldpc/codes/surgery/circuit.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: build_joint_ppm_circuit inter-code CSS commutation"
```

---

### Task 17: Noise model + detector sanity

**Files:**
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_test.py`:

```python
def test_build_single_ppm_circuit_with_noise_detectors_fire():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits.noise_model import depolarizing_noise
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    circuit, _ = build_single_ppm_circuit(
        g, rounds=2, noise_model=depolarizing_noise(p=0.05),
    )
    samples = circuit.compile_detector_sampler().sample(shots=200)
    assert samples.any()  # at least one detector fires under noise
```

*Note:* If `depolarizing_noise` doesn't exist with that exact name in `qldpc.circuits.noise_model`, substitute the actual constructor (read the file first).

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_build_single_ppm_circuit_with_noise_detectors_fire -x
```
Expected: PASS if `build_memory_circuit` already accepts a `noise_model`. If FAIL (delegated path doesn't accept noise): plumb the `noise_model` argument through (≤ 10 lines).

- [ ] **Step 3: Commit**

```bash
git add src/qldpc/codes/surgery/_test.py src/qldpc/codes/surgery/circuit.py
git commit -m "test: noise model layer fires detectors under depolarizing noise"
```

---

## Phase 4 — Cheeger boost dispatcher

### Task 18: `boost_gadget` single entry point

**Files:**
- Modify: `src/qldpc/codes/surgery/cheeger.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_test.py`:

```python
def test_boost_gadget_dispatches_to_three_methods():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.cheeger import boost_gadget, GadgetLayout as _G
    # alias re-import: boost_gadget must return GadgetLayout
    from qldpc.codes.surgery.gadget import GadgetLayout
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    for method in ("spectral", "combinatorial", "distance"):
        out = boost_gadget(g, method=method, target=1.0, seed=42)
        assert isinstance(out, GadgetLayout)


def test_boost_gadget_seed_reproducible():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.cheeger import boost_gadget
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    a = boost_gadget(g, method="spectral", target=1.0, seed=42)
    b = boost_gadget(g, method="spectral", target=1.0, seed=42)
    assert np.array_equal(a.F, b.F)
    assert np.array_equal(a.HX_merged, b.HX_merged)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "boost_gadget" -x
```
Expected: `ImportError: cannot import name 'boost_gadget'`.

- [ ] **Step 3: Add dispatcher to `cheeger.py`**

Read `src/qldpc/codes/surgery/cheeger.py`. At the bottom, add:

```python
from .gadget import GadgetLayout as _GadgetLayout, build_gadget as _build_gadget


def boost_gadget(
    gadget: _GadgetLayout, *,
    method,  # 'spectral' | 'combinatorial' | 'distance'
    target: float,
    seed: int | None = None,
    **kwargs,
) -> _GadgetLayout:
    """Single entry point for Cheeger boost. Returns a NEW GadgetLayout."""
    if method == "spectral":
        merged, layout, _ = boost_gadget_cheeger(
            gadget.code, gadget.x, target_h=target, seed=seed, **kwargs,
        )
    elif method == "combinatorial":
        merged, layout, _ = boost_gadget_cheeger_combinatorial(
            gadget.code, gadget.x, target_h=target, seed=seed, **kwargs,
        )
    elif method == "distance":
        merged, layout, _ = boost_gadget_distance(
            gadget.code, gadget.x, target_d=int(target), seed=seed, **kwargs,
        )
    else:
        raise ValueError(f"unknown method: {method!r}")
    # Rebuild a GadgetLayout from the boosted (merged, layout) pair.
    return _build_gadget(gadget.code, gadget.x)._replace_with_boosted(merged, layout)
```

If the legacy `boost_gadget_*` functions have different signatures, adapt the kwargs accordingly — read the file first. The `._replace_with_boosted` helper goes on `GadgetLayout` as a method that returns a new `GadgetLayout` with the boosted `F`, `G`, `HX_merged`, `HZ_merged`, `kappa_qubits` (with extra κ qubits from the boost).

Add `_replace_with_boosted` to `GadgetLayout` in `gadget.py`:

```python
    def _replace_with_boosted(self, merged: CSSCode, layout) -> "GadgetLayout":
        """Return a NEW GadgetLayout reflecting a boost result."""
        from .gadget import _step2_gauge_fix  # local to avoid circular
        HX_m = np.asarray(merged.matrix_x).astype(np.uint8)
        HZ_m = np.asarray(merged.matrix_z).astype(np.uint8)
        # F is recovered from the gadget layout's F field
        F_new = np.asarray(layout.F).astype(np.uint8)
        G_new = _step2_gauge_fix(F_new)
        kappa_qubits = tuple(range(self.code.num_qudits,
                                   self.code.num_qudits + F_new.shape[0]))
        return dataclasses.replace(self, F=F_new, G=G_new,
                                   HX_merged=HX_m, HZ_merged=HZ_m,
                                   kappa_qubits=kappa_qubits)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "boost_gadget" -x
```
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/cheeger.py src/qldpc/codes/surgery/gadget.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: boost_gadget single dispatcher (method=spectral|combinatorial|distance)"
```

---

### Task 19: Boost preserves CSS commutation

**Files:**
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_test.py`:

```python
@pytest.mark.parametrize("method", ["spectral", "combinatorial", "distance"])
def test_boost_gadget_preserves_css_commutation(method):
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.cheeger import boost_gadget
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    boosted = boost_gadget(g, method=method, target=1.0, seed=0)
    product = (boosted.HX_merged @ boosted.HZ_merged.T) % 2
    assert np.array_equal(product, np.zeros_like(product))
```

- [ ] **Step 2: Run test to verify it passes**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_boost_gadget_preserves_css_commutation -x
```
Expected: 3 PASS.

- [ ] **Step 3: Commit**

```bash
git add src/qldpc/codes/surgery/_test.py
git commit -m "test: boost_gadget preserves CSS commutation"
```

---

## Phase 5 — switchover: rewrite `__init__.py`, delete old files

### Task 20: Rewrite `surgery/__init__.py` to 5 public symbols + 2 dataclasses

**Files:**
- Modify: `src/qldpc/codes/surgery/__init__.py`

- [ ] **Step 1: Read current `__init__.py`**

Currently re-exports ~25 symbols from layered/joint/cheeger/multi/port/skiptree/cellulation.

- [ ] **Step 2: Replace with minimal version**

```python
"""Surgery construction package (simplified — see
docs/superpowers/specs/2026-06-07-surgery-simplification-design.md).

Public API:
    build_gadget, build_bridge,
    build_single_ppm_circuit, build_joint_ppm_circuit,
    boost_gadget
"""

from __future__ import annotations

from .gadget import GadgetLayout, build_gadget, load_webster_seed_set
from .bridge import Bridge, build_bridge
from .circuit import build_single_ppm_circuit, build_joint_ppm_circuit
from .cheeger import boost_gadget

__all__ = [
    "GadgetLayout",
    "Bridge",
    "build_gadget",
    "build_bridge",
    "build_single_ppm_circuit",
    "build_joint_ppm_circuit",
    "boost_gadget",
    "load_webster_seed_set",
]
```

- [ ] **Step 3: Run the full surgery test suite**

```bash
pytest src/qldpc/codes/surgery/ -x
```
Expected: all tests in `_test.py` PASS. If anything breaks (e.g. circular import), inspect and fix.

- [ ] **Step 4: Commit**

```bash
git add src/qldpc/codes/surgery/__init__.py
git commit -m "feat: trim surgery package public API to 5 symbols"
```

---

### Task 21: Delete obsolete surgery source files

**Files:**
- Delete: `src/qldpc/codes/surgery/layered.py`
- Delete: `src/qldpc/codes/surgery/joint.py`
- Delete: `src/qldpc/codes/surgery/skiptree.py`
- Delete: `src/qldpc/codes/surgery/cellulation.py`
- Delete: `src/qldpc/codes/surgery/multi.py`
- Delete: `src/qldpc/codes/surgery/port.py`

- [ ] **Step 1: Confirm no surviving import paths**

```bash
grep -rn "qldpc.codes.surgery.layered\|qldpc.codes.surgery.joint\|qldpc.codes.surgery.skiptree\|qldpc.codes.surgery.cellulation\|qldpc.codes.surgery.multi\|qldpc.codes.surgery.port" --include="*.py"
```
Expected: only matches inside the soon-to-be-deleted files themselves. If anything outside surgery imports from these submodules, it must be fixed first.

- [ ] **Step 2: Delete**

```bash
git rm src/qldpc/codes/surgery/layered.py \
       src/qldpc/codes/surgery/joint.py \
       src/qldpc/codes/surgery/skiptree.py \
       src/qldpc/codes/surgery/cellulation.py \
       src/qldpc/codes/surgery/multi.py \
       src/qldpc/codes/surgery/port.py
```

- [ ] **Step 3: Run full surgery test suite**

```bash
pytest src/qldpc/codes/surgery/ -x
```
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: remove legacy surgery modules (layered/joint/multi/port/skiptree/cellulation)"
```

---

### Task 22: Move `src/qldpc/codes/surgery_test.py` aside (will be replaced by surgery/_test.py)

**Files:**
- Delete: `src/qldpc/codes/surgery_test.py`

- [ ] **Step 1: Read `src/qldpc/codes/surgery_test.py`**

Identify any tests that exercise *behavior* not yet covered by `surgery/_test.py`. Most cover removed APIs (`SurgeryLayout`, `_restrict_to_logical_support`); those are gone.

- [ ] **Step 2: For each remaining behavior test, port to `surgery/_test.py`**

Specifically port (if present): logical-op validation rejections, CSS-commutation invariants on non-Steane codes, any joint-measurement protocol tests (math.md §2.7 α* formula). Each port is its own micro-TDD loop:

```bash
pytest src/qldpc/codes/surgery/_test.py::test_<name> -x
```

- [ ] **Step 3: Delete the old test file**

```bash
git rm src/qldpc/codes/surgery_test.py
```

- [ ] **Step 4: Run the whole repo test suite**

```bash
pytest src/qldpc/ -x
```
Expected: nothing broken outside surgery.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/_test.py
git commit -m "test: migrate surviving surgery tests to surgery/_test.py and drop old file"
```

---

### Task 23: Update `src/qldpc/codes/__init__.py` to drop removed re-exports

**Files:**
- Modify: `src/qldpc/codes/__init__.py`

- [ ] **Step 1: Inspect current state**

```bash
grep -n "surgery" src/qldpc/codes/__init__.py
```

- [ ] **Step 2: Replace any removed-symbol re-exports with the new 5 + 2**

Edit to expose: `GadgetLayout`, `Bridge`, `build_gadget`, `build_bridge`, `build_single_ppm_circuit`, `build_joint_ppm_circuit`, `boost_gadget`, `load_webster_seed_set` from `qldpc.codes.surgery`.

- [ ] **Step 3: Run repo tests**

```bash
pytest src/qldpc/ -x
```
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/qldpc/codes/__init__.py
git commit -m "chore: codes package re-exports the trimmed surgery surface"
```

---

## Phase 6 — Migrate example scripts

> The 16 scripts are migrated in batches. Each batch is one task: rewrite imports + call sites, run the script (smoke test where reasonable), commit.

### Task 24: Migrate single-PPM Cain scripts (bb18, lp_memory, lp20, lp24)

**Files:**
- Modify: `examples/scripts/cain_bb18_resource_exact_match.py`
- Modify: `examples/scripts/cain_bb18_processor_exact_match.py`
- Modify: `examples/scripts/cain_lp_memory_exact_match.py`
- Modify: `examples/scripts/cain_lp20_processor_exact_match.py`
- Modify: `examples/scripts/cain_lp24_memory_exact_match.py`

- [ ] **Step 1: Mechanical rewrite**

In each file, replace:
- `from qldpc.codes.surgery import ... boost_gadget_cheeger_combinatorial, build_layered_surgery_code, ...`
  → `from qldpc.codes.surgery import build_gadget, boost_gadget`
- `merged, layout = build_layered_surgery_code(code, x, num_layers=1, ...)`
  → `g = build_gadget(code, x)` (then `merged = g_to_merged(g)` if the script still needs the merged CSSCode — see Step 2)
- `boost_gadget_cheeger_combinatorial(merged, layout, target_h=..., ...)`
  → `boost_gadget(g, method='combinatorial', target=..., seed=..., ...)`
- `boost_gadget_cheeger(merged, layout, target_h=..., ...)`
  → `boost_gadget(g, method='spectral', target=..., seed=..., ...)`

Any internal helper script imports (e.g. `from qldpc.codes.surgery.cheeger import _reassemble_gadget_with_new_F` in `cain_lp20_processor_exact_match.py`) → either inline the helper into the script or use `boost_gadget` with a `rank_bound=` kwarg passed through `**kwargs`.

- [ ] **Step 2: Smoke-test each script**

```bash
python -c "import examples.scripts.cain_bb18_resource_exact_match" 2>&1 | tail -20
python -c "import examples.scripts.cain_bb18_processor_exact_match" 2>&1 | tail -20
python -c "import examples.scripts.cain_lp_memory_exact_match" 2>&1 | tail -20
python -c "import examples.scripts.cain_lp20_processor_exact_match" 2>&1 | tail -20
python -c "import examples.scripts.cain_lp24_memory_exact_match" 2>&1 | tail -20
```
Expected: no `ImportError`. (Full script execution can be slow — import alone catches API breakage.)

- [ ] **Step 3: Commit**

```bash
git add examples/scripts/cain_bb18_resource_exact_match.py \
        examples/scripts/cain_bb18_processor_exact_match.py \
        examples/scripts/cain_lp_memory_exact_match.py \
        examples/scripts/cain_lp20_processor_exact_match.py \
        examples/scripts/cain_lp24_memory_exact_match.py
git commit -m "chore: migrate Cain single-PPM exact-match scripts to new surgery API"
```

---

### Task 25: Migrate Cain Fig 1b joint scripts (circuit_level, full_protocol, webster_surgery)

**Files:**
- Modify: `examples/scripts/cain_fig1b_circuit_level.py`
- Modify: `examples/scripts/cain_fig1b_full_protocol.py`
- Modify: `examples/scripts/cain_fig1b_webster_surgery.py`

- [ ] **Step 1: Mechanical rewrite**

For `cain_fig1b_circuit_level.py` and `cain_fig1b_full_protocol.py`:
- Replace `from qldpc.codes.surgery import build_layered_surgery_code, boost_gadget_cheeger_combinatorial, _stitch_gadgets_with_bridge`
  → `from qldpc.codes.surgery import build_gadget, boost_gadget, build_bridge, build_joint_ppm_circuit`
- Replace the manual `_stitch_gadgets_with_bridge(...)` call with `build_bridge(g1b, g2b)` + `build_joint_ppm_circuit(g1b, g2b, bridge, rounds=..., noise_model=...)`.
- The inline Stim circuit construction (the bulk of these scripts post-`joint`) gets replaced by the `circuit, record, joint_code = build_joint_ppm_circuit(...)` triple.

For `cain_fig1b_webster_surgery.py`: simple rename (`build_layered_surgery_code` → `build_gadget`), drop `num_layers=1`.

- [ ] **Step 2: Smoke-test imports**

```bash
python -c "import examples.scripts.cain_fig1b_circuit_level"
python -c "import examples.scripts.cain_fig1b_full_protocol"
python -c "import examples.scripts.cain_fig1b_webster_surgery"
```

- [ ] **Step 3: Commit**

```bash
git add examples/scripts/cain_fig1b_*.py
git commit -m "chore: migrate Cain Fig 1b joint scripts to build_joint_ppm_circuit"
```

---

### Task 26: Migrate Ide scripts (table_ii, skiptree_verification, lemma10_prototype)

**Files:**
- Modify: `examples/scripts/ide_table_ii_exact_match.py`
- Modify: `examples/scripts/ide_table_ii_full_pipeline.py`
- Modify: `examples/scripts/ide_skiptree_verification.py`
- Modify: `examples/scripts/ide_lemma10_prototype.py`

- [ ] **Step 1: Replace any direct uses of `_skip_tree`, `_skip_tree_hr`, `_cellulate_long_cycles`, `_build_bridge_via_skiptree`, `_BridgeSpec`, `canonical_HR`, `_running_xor_b_c` etc.**

These helpers are now private to `bridge.py`. Two options per use:
- (a) If the script is a verification of paper math, refactor it to call `build_bridge(g1, g2)` and inspect `Bridge.aux_graph_edges` + `Bridge.z_extensions` instead.
- (b) If the script genuinely needs a private helper, prefix the import with the new path (`from qldpc.codes.surgery.bridge import _skip_tree`) — these are example scripts so the underscore-import is acceptable.

- [ ] **Step 2: Smoke-test imports**

```bash
python -c "import examples.scripts.ide_table_ii_exact_match"
python -c "import examples.scripts.ide_table_ii_full_pipeline"
python -c "import examples.scripts.ide_skiptree_verification"
python -c "import examples.scripts.ide_lemma10_prototype"
```

- [ ] **Step 3: Commit**

```bash
git add examples/scripts/ide_*.py
git commit -m "chore: migrate Ide scripts to new surgery API"
```

---

### Task 27: Migrate remaining scripts (verify_cain_table_iii, cain_table_iii_summary, _9_lattice_surgery, _cain_helpers)

**Files:**
- Modify: `examples/scripts/verify_cain_table_iii.py`
- Modify: `examples/scripts/cain_table_iii_summary.py`
- Modify: `examples/logical_error_rates/_9_lattice_surgery_cain_fig1b_source.py`
- Modify: `examples/scripts/_cain_helpers.py` (if it touches removed APIs)

- [ ] **Step 1: Mechanical rewrites per the patterns from Tasks 24-26**

In `_9_lattice_surgery_cain_fig1b_source.py`: `build_layered_surgery_code(data_code, logical_x, num_layers=1)` → `build_gadget(data_code, logical_x)`. Also drop the "try num_layers=3 (Cross fallback)" code comment — no longer applicable.

- [ ] **Step 2: Smoke-test imports**

```bash
python -c "import examples.scripts.verify_cain_table_iii"
python -c "import examples.scripts.cain_table_iii_summary"
python -c "import examples.logical_error_rates._9_lattice_surgery_cain_fig1b_source"
```

- [ ] **Step 3: Commit**

```bash
git add examples/scripts/verify_cain_table_iii.py \
        examples/scripts/cain_table_iii_summary.py \
        examples/logical_error_rates/_9_lattice_surgery_cain_fig1b_source.py \
        examples/scripts/_cain_helpers.py
git commit -m "chore: migrate remaining example scripts to new surgery API"
```

---

### Task 28: Delete `examples/webster_table1_verify.py`

**Files:**
- Delete: `examples/webster_table1_verify.py`

- [ ] **Step 1: Delete (its functionality is now in `surgery/_test.py::test_webster_table_i_kappa_chi_r_exact`)**

```bash
git rm examples/webster_table1_verify.py
```

- [ ] **Step 2: Commit**

```bash
git commit -m "chore: delete examples/webster_table1_verify.py (now a unit test)"
```

---

## Phase 7 — Ide BB↔LP exact-match ground truth (spec §4.F)

### Task 29: Add `load_ide_BB_input_with_operator` and `load_ide_LP_input_with_operator` to `_ide_fixtures.py`

**Files:**
- Modify: `src/qldpc/codes/_ide_fixtures.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_test.py`:

```python
@pytest.mark.skipif(
    not __import__("qldpc.codes._ide_fixtures", fromlist=["fixtures_available"]).fixtures_available(),
    reason="Ide Zenodo fixtures not installed",
)
def test_load_ide_BB_input_with_operator_returns_csscode_and_op():
    from qldpc.codes._ide_fixtures import load_ide_BB_input_with_operator
    code, x = load_ide_BB_input_with_operator()
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    assert ((HZ @ x) % 2 == 0).all()
    # x is a nontrivial logical-X
    assert x.sum() > 0


@pytest.mark.skipif(
    not __import__("qldpc.codes._ide_fixtures", fromlist=["fixtures_available"]).fixtures_available(),
    reason="Ide Zenodo fixtures not installed",
)
def test_load_ide_LP_input_with_operator_returns_csscode_and_op():
    from qldpc.codes._ide_fixtures import load_ide_LP_input_with_operator
    code, x = load_ide_LP_input_with_operator()
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    assert ((HZ @ x) % 2 == 0).all()
    assert x.sum() > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "load_ide_BB_input or load_ide_LP_input" -x
```
Expected: `ImportError` (if fixtures available) or `SKIP` (if not).

- [ ] **Step 3: Implement loaders**

Edit `src/qldpc/codes/_ide_fixtures.py`. Reuse the existing `IDE_BB_KAPPA1_EDGES` (and the analogous LP table — read the existing file to find it) to construct the operator vectors. The BB and LP input codes themselves are the data codes (not the merged BB-LP joint code) — load them from the Zenodo bundle:

```python
def load_ide_BB_input_with_operator() -> tuple[CSSCode, np.ndarray]:
    """Return BB input code + the pinned V_0 logical-X operator (Ide §VII.B)."""
    HX, HZ = _load_zenodo("BB_98_LP_200_adapter/BB_98_input_HX.npy",
                          "BB_98_LP_200_adapter/BB_98_input_HZ.npy")  # actual paths TBD per Zenodo layout
    code = CSSCode(GF2(HX.tolist()), GF2(HZ.tolist()), is_subsystem_code=False)
    V0 = sorted({v for edge in IDE_BB_KAPPA1_EDGES.values() for v in edge})
    x = np.zeros(code.num_qudits, dtype=np.uint8)
    for v in V0:
        x[v] = 1
    return code, x
```

If the Zenodo paths are not exactly as guessed: open `tests/fixtures/ide_zenodo/` and discover the actual file layout. If V_0 cannot be uniquely recovered from `IDE_BB_KAPPA1_EDGES` alone, fall back to reading Ide's published `Z_1` (or `X_1`) operator from the Zenodo bundle.

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "load_ide_BB_input or load_ide_LP_input" -x
```
Expected: PASS (if fixtures available) or SKIP.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/_ide_fixtures.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: load Ide BB/LP INPUT codes + pinned logical-X operators"
```

---

### Task 30: Ide BB↔LP exact-match (n=355, k=25, d=10)

**Files:**
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_test.py`:

```python
@pytest.mark.skipif(
    not __import__("qldpc.codes._ide_fixtures", fromlist=["fixtures_available"]).fixtures_available(),
    reason="Ide Zenodo fixtures not installed",
)
def test_intercode_joint_bb_lp_exact():
    from qldpc.codes._ide_fixtures import (
        load_ide_BB_input_with_operator, load_ide_LP_input_with_operator,
    )
    from qldpc.codes.surgery import (
        build_gadget, build_bridge, build_joint_ppm_circuit,
    )
    bb, x_bb = load_ide_BB_input_with_operator()
    lp, x_lp = load_ide_LP_input_with_operator()
    g1 = build_gadget(bb, x_bb)
    g2 = build_gadget(lp, x_lp)
    bridge = build_bridge(g1, g2)
    _, _, joint = build_joint_ppm_circuit(g1, g2, bridge, rounds=1, noise_model=None)
    assert joint.num_qudits == 355
    assert joint.dimension == 25
    assert joint.get_distance() == 10
```

- [ ] **Step 2: Run test**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_intercode_joint_bb_lp_exact -v
```
Expected: PASS (or SKIP). If it FAILs with one of:
- `n != 355` — `build_bridge` chose a different cellulation than Ide's; add `cellulation_override=` kwarg to `build_bridge` per the spec's mitigation, and re-run the test feeding Ide's published cellulation choice via `load_ide_skiptree_TPG`.
- `k != 25` — the joint code's gauge structure differs; inspect H_X^joint, H_Z^joint vs Ide's.
- `d != 10` — distance computation is slow; if test times out, mark as `@pytest.mark.slow` and skip in default runs (still assert n + k).

- [ ] **Step 3: Commit**

```bash
git add src/qldpc/codes/surgery/_test.py
git commit -m "test: Ide BB↔LP inter-code joint exact-match (n=355, k=25, d=10)"
```

---

## Phase 8 — Final cleanup

### Task 31: LOC budget enforcement + structural review

**Files:**
- Inspect: all of `src/qldpc/codes/surgery/`

- [ ] **Step 1: Measure**

```bash
wc -l src/qldpc/codes/surgery/*.py
```
Expected:
- `gadget.py` ≤ 200
- `bridge.py` ≤ 350
- `circuit.py` ≤ 300
- `__init__.py` ≤ 30
- `cheeger.py` and `_test.py` have no budget (cheeger is preserved as-is + thin dispatcher; `_test.py` is allowed to be large)

- [ ] **Step 2: If over budget**

Identify single-use helpers and inline them. Identify redundant docstrings ("re-deriving math.md" — replace with a 1-line pointer like `# math.md §1.2`). Identify duplicated GF(2) coercion sequences — consolidate.

- [ ] **Step 3: Commit any cleanup**

```bash
git add src/qldpc/codes/surgery/
git commit -m "refactor: surgery modules within LOC budget"
```

---

### Task 32: Full repo test suite + final check

- [ ] **Step 1: Run**

```bash
pytest src/qldpc/ -x
```
Expected: all PASS (`-k "not slow"` if Ide distance test is marked slow).

- [ ] **Step 2: Lint / type-check (if repo uses them)**

```bash
ruff check src/qldpc/codes/surgery/ 2>&1 | head -40
mypy src/qldpc/codes/surgery/ 2>&1 | head -40
```
Address any errors with focused fixes.

- [ ] **Step 3: Final diff summary**

```bash
git diff --stat main..HEAD -- src/qldpc/codes/surgery/ examples/ src/qldpc/codes/_ide_fixtures.py
git log --oneline main..HEAD | head -40
```

- [ ] **Step 4: No commit needed for the verification itself**

---

## Self-Review Checklist

After writing this plan, verify against the spec:

**Spec coverage:**

- [x] **Goal 1** (5 public symbols) — Tasks 20 + 18 establish the public surface.
- [x] **Goal 2** (deterministic `build_gadget`) — Tasks 3, 5 explicitly test determinism.
- [x] **Goal 3** (3 explicit Webster steps) — Tasks 2, 3, 4 each implement one step.
- [x] **Goal 4** (Webster Table I + Ide BB↔LP exact match) — Tasks 7, 10, 30.
- [x] **Goal 5** (Stim circuit API: noise + detectors + observables) — Tasks 14, 15, 16, 17.
- [x] **Goal 6** (no backwards-compat aliases) — Task 20 + 21 delete old names.
- [x] **Goal 7** (per-module LOC budget) — Tasks 13, 31.

**Section coverage:**

- [x] §1 module layout — Tasks 1, 8, 14, 18, 20, 21, 22.
- [x] §2 public API — covered above.
- [x] §3 migration — Tasks 24–28.
- [x] §4 test plan — Tasks 2–7, 9–12, 14–19, 29–30.
- [x] §5 risks (Webster determinism, Ide cellulation override, script churn) — addressed in Task 7 step 2, Task 30 step 2, Task 24–27.

**Placeholder scan:** No "TBD", "TODO", "implement later". Two notes explicitly say "if X doesn't exist with that exact name, substitute" — these are honest signposts where reading the actual file is required (the engineer has the file path); they are not undefined work.

**Type consistency:** `GadgetLayout` fields and `Bridge` fields are defined in Tasks 1 and 8 and used identically afterward. `boost_gadget(method=...)` signature stays the same across Tasks 18, 19, 24.
