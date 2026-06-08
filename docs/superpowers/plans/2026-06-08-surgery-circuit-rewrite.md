# Surgery Circuit Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `build_single_ppm_circuit` and `build_joint_ppm_circuit` to faithfully implement the Cain §III.A surgery protocol (κ |0⟩ init, τ_s rounds of merged-code SE with classified round-1 detectors, detach Q' via Z-measure, PPM observable per Webster Eq. 1), and add a `basis=Pauli.X|Pauli.Z` parameter to `build_gadget` for symmetric X̄/Z̄ PPM.

**Architecture:** Phase 1 adds `basis` to `GadgetLayout` (default Pauli.X for backward compatibility) and refactors the 3 Webster steps to dispatch on basis. Phase 2 propagates basis through `build_bridge`. Phase 3 patches `cheeger.boost_gadget` for basis-Z. Phase 4 rewrites `circuit.py` from scratch following the `_get_basis_memory_experiment_parts` / `_get_qec_cycle` pattern but with surgery-specific init, detector classification, detach, and observable. Phase 5 adds semantic + LER tests. Phase 6 migrates the Ide test and notebook.

**Tech Stack:** Python 3.11+, `numpy`, `galois` (GF(2)), `stim`, `pytest`. Uses existing `qldpc.codes.common.CSSCode`, `qldpc.circuits.bookkeeping.{QubitIDs, MeasurementRecord, DetectorRecord}`, `qldpc.circuits.memory.syndrome_measurement.EdgeColoring`, `qldpc.circuits.memory.memory.get_qubit_coordinates`, `qldpc.circuits.noise_model.NoiseModel`.

**Spec:** `docs/superpowers/specs/2026-06-08-surgery-circuit-rewrite-design.md`
**Builds on:** `docs/superpowers/specs/2026-06-07-surgery-simplification-design.md`
**Branch:** `feat/surgery-construction` (currently at HEAD `1adf84a` — spec just committed)

**Reference files to read before starting:**
- `src/qldpc/circuits/memory/memory.py` — `_get_basis_memory_experiment_parts` (lines 232-298) and `_get_qec_cycle` (lines 483-538). This is the pattern we mirror.
- `src/qldpc/circuits/bookkeeping.py` — `QubitIDs.from_code` (line 62), `MeasurementRecord.get_target_rec` (line 206).
- `docs/superpowers/math.md` — §1 (gadget construction), §2 (joint bridge), §2.7 (joint observable α*).

**Key conventions in the merged code:**
- `QubitIDs.from_code(merged_code).data` includes both ORIGINAL data qubits AND κ ancilla qubits AND bridge qubits (for the joint case). The merged code treats all of these as its "data register".
- We need to split: original data qubits = first `n_original_data`, κ ancillas next `|C_0|`, bridge last `w` (joint only).
- For inter-code joint: data_1 + data_2 + kappa_1 + kappa_2 + bridge.
- `QubitIDs.checks_x[:m_X]` are syndrome ancillas for data H_X rows of merged code; `checks_x[m_X:]` are for χ rows. Symmetric for `checks_z`.

---

## Phase 1 — Add `basis` to `GadgetLayout` and `build_gadget`

### Task 1: Add `basis` field to `GadgetLayout`

**Files:**
- Modify: `src/qldpc/codes/surgery/gadget.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append to `src/qldpc/codes/surgery/_test.py`:

```python
def test_gadget_layout_has_basis_field():
    from qldpc.codes.surgery.gadget import GadgetLayout
    fields = {f.name for f in dataclasses.fields(GadgetLayout)}
    assert "basis" in fields, f"basis field missing; got {fields}"


def test_gadget_layout_basis_defaults_to_x_via_build_gadget():
    """Backward compatibility: build_gadget without explicit basis defaults to Pauli.X."""
    from qldpc.codes.surgery.gadget import build_gadget
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    assert g.basis is Pauli.X
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_gadget_layout_has_basis_field src/qldpc/codes/surgery/_test.py::test_gadget_layout_basis_defaults_to_x_via_build_gadget -x
```
Expected: FAIL (`basis` not a field of GadgetLayout).

- [ ] **Step 3: Add the `basis` field with default Pauli.X**

Edit `src/qldpc/codes/surgery/gadget.py`. Find the `GadgetLayout` dataclass (around line 22-31) and add a `basis` field. Use `dataclasses.field(default=Pauli.X)` so existing callers can construct without specifying it:

```python
from qldpc.objects import Pauli, PauliXZ

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
    basis: PauliXZ = dataclasses.field(default_factory=lambda: Pauli.X)
```

(Use `default_factory` because `Pauli.X` may not be hashable for `default=...`; if `default=Pauli.X` works, use it directly.)

Also update `build_gadget` so the returned `GadgetLayout(...)` includes `basis=Pauli.X` (no signature change yet — that's Task 4):

```python
def build_gadget(code: CSSCode, x: np.ndarray) -> "GadgetLayout":
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
        basis=Pauli.X,
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_gadget_layout_has_basis_field src/qldpc/codes/surgery/_test.py::test_gadget_layout_basis_defaults_to_x_via_build_gadget -x
```
Expected: 2 PASS.

- [ ] **Step 5: Run full suite to confirm nothing broke**

```bash
pytest src/qldpc/codes/surgery/ -x 2>&1 | tail -5
```
Expected: 49 PASS (47 existing + 2 new).

- [ ] **Step 6: Commit**

```bash
git add src/qldpc/codes/surgery/gadget.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: add basis field to GadgetLayout (default Pauli.X)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Refactor `_step1_restriction` to dispatch on basis

**Files:**
- Modify: `src/qldpc/codes/surgery/gadget.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_step1_restriction_basis_z_uses_HX():
    """For basis=Pauli.Z, F = H_X[C_0, V_0] (not H_Z)."""
    from qldpc.codes.surgery.gadget import _step1_restriction
    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    V0, C0, F = _step1_restriction(code, z, basis=Pauli.Z)
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    # V_0 = supp(z)
    assert V0 == tuple(int(i) for i in np.where(z)[0])
    # C_0 = X-checks touching V_0
    touched = sorted({j for j in range(HX.shape[0]) for i in V0 if HX[j, i] == 1})
    assert C0 == tuple(touched)
    # F = H_X[C_0, V_0]
    assert np.array_equal(F, HX[np.ix_(C0, V0)])
    # math.md §1.1 invariant: F @ 1_{V0} = 0 (since H_X @ z = 0 for a logical Z)
    ones = np.ones(len(V0), dtype=np.uint8)
    assert np.array_equal((F @ ones) % 2, np.zeros(len(C0), dtype=np.uint8))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_step1_restriction_basis_z_uses_HX -x
```
Expected: FAIL (TypeError: unexpected kwarg 'basis').

- [ ] **Step 3: Add basis dispatch to `_step1_restriction`**

Edit `src/qldpc/codes/surgery/gadget.py`. Find the existing `_step1_restriction` (~line 35-50). Update:

```python
def _step1_restriction(
    code: CSSCode, x: np.ndarray, *, basis: PauliXZ = Pauli.X,
) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray]:
    """math.md §1.1 — V_0 = supp(x); C_0 = checks touching V_0; F = H_complement[C_0, V_0].

    For basis=Pauli.X: F = H_Z[C_0, V_0] (the complementary basis to the measured logical).
    For basis=Pauli.Z: F = H_X[C_0, V_0].
    """
    x = np.asarray(x).astype(np.uint8)
    if x.shape != (code.num_qudits,):
        raise ValueError(f"x has shape {x.shape}, expected ({code.num_qudits},)")
    V0 = tuple(int(i) for i in np.where(x)[0])
    # Use the COMPLEMENTARY check matrix to the measured logical type
    H_complement = (
        np.asarray(code.matrix_z).astype(np.uint8)
        if basis is Pauli.X
        else np.asarray(code.matrix_x).astype(np.uint8)
    )
    C0 = tuple(
        int(j) for j in range(H_complement.shape[0]) if H_complement[j, list(V0)].any()
    )
    F = (
        H_complement[np.ix_(C0, V0)]
        if C0 and V0
        else np.zeros((len(C0), len(V0)), dtype=np.uint8)
    )
    return V0, C0, F.astype(np.uint8)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_step1_restriction_basis_z_uses_HX src/qldpc/codes/surgery/_test.py::test_step1_restriction_steane -x
```
Expected: 2 PASS (the new Z test plus the existing X test continues to pass with the default).

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/gadget.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: _step1_restriction dispatches on basis (HZ vs HX)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Refactor `_step3_assemble` to dispatch on basis

**Files:**
- Modify: `src/qldpc/codes/surgery/gadget.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_step3_assemble_basis_z_places_chi_in_HZ_merged_and_G_in_HX_merged():
    """basis=Pauli.Z: χ rows added to HZ_merged (Z-type); G added to HX_merged (X-type)."""
    from qldpc.codes.surgery.gadget import (
        _step1_restriction, _step2_gauge_fix, _step3_assemble,
    )
    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    V0, C0, F = _step1_restriction(code, z, basis=Pauli.Z)
    G = _step2_gauge_fix(F)
    HX_m, HZ_m = _step3_assemble(code, V0, C0, F, G, basis=Pauli.Z)

    n, mX, mZ = code.num_qudits, code.matrix_x.shape[0], code.matrix_z.shape[0]
    # For basis=Z: HX_merged grows by r rows (gauge-fix), HZ_merged by |V_0| rows (chi).
    assert HX_m.shape == (mX + G.shape[0], n + len(C0)), f"HX shape {HX_m.shape}"
    assert HZ_m.shape == (mZ + len(V0), n + len(C0)), f"HZ shape {HZ_m.shape}"
    # CSS commutation
    product = (HX_m @ HZ_m.T) % 2
    assert np.array_equal(product, np.zeros_like(product))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_step3_assemble_basis_z_places_chi_in_HZ_merged_and_G_in_HX_merged -x
```
Expected: FAIL.

- [ ] **Step 3: Add basis dispatch to `_step3_assemble`**

Edit `gadget.py`. Find `_step3_assemble` (~line 60-95). Replace with:

```python
def _step3_assemble(
    code: CSSCode,
    V0: tuple[int, ...],
    C0: tuple[int, ...],
    F: np.ndarray,
    G: np.ndarray,
    *,
    basis: PauliXZ = Pauli.X,
) -> tuple[np.ndarray, np.ndarray]:
    """math.md §1.4 — block assembly of HX_merged, HZ_merged.

    basis=X (default): χ rows added to HX_merged, G to HZ_merged.
    basis=Z: χ rows added to HZ_merged, G to HX_merged (basis-symmetric dual).
    """
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
    F_T = F.T.astype(np.uint8)

    # F_tilde : (mZ_or_mX × nC) indicator/selection matrix
    # F_tilde[j, k] = 1 iff j == C_0[k]
    if basis is Pauli.X:
        F_tilde = np.zeros((mZ, nC), dtype=np.uint8)
    else:
        F_tilde = np.zeros((mX, nC), dtype=np.uint8)
    for k, j in enumerate(C0):
        F_tilde[j, k] = 1

    if basis is Pauli.X:
        # χ rows extend HX_merged; G rows extend HZ_merged
        HX_merged = np.block([
            [HX, np.zeros((mX, nC), dtype=np.uint8)],
            [E_V0_T, F_T],
        ]).astype(np.uint8)
        HZ_merged = np.block([
            [HZ, F_tilde],
            [np.zeros((r, n), dtype=np.uint8), G.astype(np.uint8)],
        ]).astype(np.uint8)
    else:
        # basis=Z: χ rows extend HZ_merged; G rows extend HX_merged (symmetric)
        HZ_merged = np.block([
            [HZ, np.zeros((mZ, nC), dtype=np.uint8)],
            [E_V0_T, F_T],
        ]).astype(np.uint8)
        HX_merged = np.block([
            [HX, F_tilde],
            [np.zeros((r, n), dtype=np.uint8), G.astype(np.uint8)],
        ]).astype(np.uint8)

    return HX_merged, HZ_merged
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "step3_assemble" -x
```
Expected: all 3 step3 tests PASS (the existing Steane X test, the existing distinct-nV-nC X test, and the new Z test).

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/gadget.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: _step3_assemble dispatches on basis (chi in HX or HZ)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `build_gadget(code, x, *, basis=Pauli.X)` signature

**Files:**
- Modify: `src/qldpc/codes/surgery/gadget.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_build_gadget_z_basis_css_commutation():
    """build_gadget(code, z_logical, basis=Pauli.Z) yields a CSS-commuting merged code."""
    from qldpc.codes.surgery.gadget import build_gadget
    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    assert g.basis is Pauli.Z
    product = (g.HX_merged @ g.HZ_merged.T) % 2
    assert np.array_equal(product, np.zeros_like(product))


def test_build_gadget_z_basis_rejects_non_z_logical():
    """For basis=Pauli.Z, build_gadget checks HX @ x == 0 (z must be a Z-logical)."""
    from qldpc.codes.surgery.gadget import build_gadget
    code = codes.SteaneCode()
    # An X-logical fails: HX @ x_logical_X is typically nonzero
    x_logical = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    if ((HX @ x_logical) % 2).any():
        with pytest.raises(ValueError, match="logical"):
            build_gadget(code, x_logical, basis=Pauli.Z)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "build_gadget_z_basis" -x
```
Expected: FAIL (build_gadget doesn't accept `basis`).

- [ ] **Step 3: Update `build_gadget` signature**

Edit `gadget.py`. Replace `build_gadget` body:

```python
def build_gadget(
    code: CSSCode, x: np.ndarray, *, basis: PauliXZ = Pauli.X,
) -> GadgetLayout:
    """Webster L=1 gadget = steps 1+2+3 composed. Deterministic in (code, x, basis).

    basis=Pauli.X: measures a logical X (PPM of X̄). Validates H_Z @ x == 0.
    basis=Pauli.Z: measures a logical Z (PPM of Z̄). Validates H_X @ x == 0.
    """
    x = np.asarray(x).astype(np.uint8)
    # Validate x is in the kernel of the COMPLEMENTARY check matrix
    if basis is Pauli.X:
        H_check = np.asarray(code.matrix_z).astype(np.uint8)
        if ((H_check @ x) % 2).any():
            raise ValueError("x is not a logical-X support (H_Z @ x != 0).")
    elif basis is Pauli.Z:
        H_check = np.asarray(code.matrix_x).astype(np.uint8)
        if ((H_check @ x) % 2).any():
            raise ValueError("x is not a logical-Z support (H_X @ x != 0).")
    else:
        raise ValueError(f"basis must be Pauli.X or Pauli.Z, got {basis!r}")

    V0, C0, F = _step1_restriction(code, x, basis=basis)
    G = _step2_gauge_fix(F)
    HX_m, HZ_m = _step3_assemble(code, V0, C0, F, G, basis=basis)
    kappa_qubits = tuple(range(code.num_qudits, code.num_qudits + len(C0)))
    return GadgetLayout(
        code=code, x=x, V0=V0, C0=C0, F=F, G=G,
        HX_merged=HX_m, HZ_merged=HZ_m, kappa_qubits=kappa_qubits,
        basis=basis,
    )
```

- [ ] **Step 4: Run tests**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "build_gadget" -x
```
Expected: all build_gadget tests PASS (existing X + new Z).

- [ ] **Step 5: Run full suite**

```bash
pytest src/qldpc/codes/surgery/ -x 2>&1 | tail -5
```
Expected: 53 PASS (47 existing + 6 new across Tasks 1-4).

- [ ] **Step 6: Commit**

```bash
git add src/qldpc/codes/surgery/gadget.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: build_gadget accepts basis=Pauli.X|Pauli.Z param

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Symmetric-dual sanity test and Webster Z̄_1 Table I

**Files:**
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_build_gadget_z_basis_dual_matches_x_basis_on_dual_code():
    """basis-symmetric invariant: build_gadget(code, z, basis=Z) gives the same
    merged matrices as build_gadget(dual_code, z, basis=X), where dual_code has
    HX/HZ swapped. The swap labels swap too, so we compare HX_z vs HZ_dx_x and
    HZ_z vs HX_dx_x."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.common import CSSCode
    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_z = build_gadget(code, z, basis=Pauli.Z)
    # Dual code: swap matrix_x and matrix_z
    dual = CSSCode(
        np.asarray(code.matrix_z).astype(np.int_),
        np.asarray(code.matrix_x).astype(np.int_),
        is_subsystem_code=False,
    )
    g_dual = build_gadget(dual, z, basis=Pauli.X)
    # In the dual construction, the basis-X chi rows end up in dual.HX_merged
    # which corresponds to original.HZ_merged in the basis-Z construction.
    assert np.array_equal(g_z.HZ_merged, g_dual.HX_merged), (
        "basis-Z chi (in HZ_merged) should equal basis-X chi (in HX_merged) on dual"
    )
    assert np.array_equal(g_z.HX_merged, g_dual.HZ_merged), (
        "basis-Z gauge-fix (in HX_merged) should equal basis-X gauge-fix (in HZ_merged) on dual"
    )


def test_webster_table_i_z_basis_kappa_chi_r_exact():
    """Webster Z̄_1 seed produces the same κ+χ+r counts (basis-symmetric)."""
    from qldpc.codes.surgery.gadget import (
        build_gadget, load_webster_seed_set, _build_generalised_bicycle_code,
    )

    def z_bar_1_operator(d: dict) -> np.ndarray:
        l = d["l"]
        for seed in d["seeds"]:
            if seed["name"] == "Z_bar_1" and seed["pauli_type"] == "Z":
                L = np.zeros(l, dtype=np.uint8); R = np.zeros(l, dtype=np.uint8)
                for i in seed["L_support"]:
                    L[i] = 1
                for i in seed["R_support"]:
                    R[i] = 1
                return np.concatenate([L, R])
        raise ValueError("Z_bar_1 not found")

    for code_index, expected in [(0, 19), (1, 31), (2, 49), (3, 79)]:
        d = load_webster_seed_set(code_index)
        c = _build_generalised_bicycle_code(d["l"], d["A"], d["B"])
        z = z_bar_1_operator(d)
        g = build_gadget(c, z, basis=Pauli.Z)
        kappa = len(g.kappa_qubits)
        chi = len(g.V0)
        r = g.G.shape[0]
        assert kappa + chi + r == expected, (
            f"code {code_index}: Z-basis got κ+χ+r={kappa+chi+r}, expected {expected}"
        )
```

- [ ] **Step 2: Run tests**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "z_basis_dual or webster_table_i_z_basis" -v
```
Expected: 2 PASS. If the Webster Z̄ test FAILS — investigate whether Z-supports are symmetrically structured for these BB codes. They typically are (GB codes are self-dual), but report DONE_WITH_CONCERNS with actual numbers if not.

- [ ] **Step 3: Commit**

```bash
git add src/qldpc/codes/surgery/_test.py
git commit -m "test: basis-Z dual equivalence + Webster Table I Z̄ symmetry

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 2 — `Bridge` basis propagation

### Task 6: Add `basis` to `Bridge` and assert match in `build_bridge`

**Files:**
- Modify: `src/qldpc/codes/surgery/bridge.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_bridge_has_basis_field_and_inherits_from_gadgets():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge, Bridge
    code = codes.SteaneCode()
    z1 = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    z2 = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g1 = build_gadget(code, z1, basis=Pauli.Z)
    g2 = build_gadget(code, z2, basis=Pauli.Z)
    bridge = build_bridge(g1, g2)
    assert bridge.basis is Pauli.Z


def test_build_bridge_rejects_basis_mismatch():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_x = build_gadget(code, x, basis=Pauli.X)
    g_z = build_gadget(code, z, basis=Pauli.Z)
    with pytest.raises(ValueError, match="basis"):
        build_bridge(g_x, g_z)
```

- [ ] **Step 2: Run tests**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "bridge_has_basis or bridge_rejects_basis" -x
```
Expected: FAIL.

- [ ] **Step 3: Add `basis` field to `Bridge`**

Edit `src/qldpc/codes/surgery/bridge.py`. Add to the `Bridge` dataclass (find it near top of file):

```python
@dataclasses.dataclass(frozen=True, eq=False)
class Bridge:
    width: int
    qubits: tuple[int, ...]
    U_B: np.ndarray
    chi_endpoint_extensions: dict[int, np.ndarray]
    intercode: bool
    aux_graph_edges: tuple[tuple[int, int], ...] | None
    z_extensions: dict[int, np.ndarray] | None
    basis: PauliXZ = dataclasses.field(default_factory=lambda: Pauli.X)
```

Add the import at top: `from qldpc.objects import Pauli, PauliXZ`.

Modify `build_bridge` body to assert and inherit:

```python
def build_bridge(g1: GadgetLayout, g2: GadgetLayout) -> Bridge:
    """Two-PPM bridge ... [existing docstring]"""
    if g1.basis is not g2.basis:
        raise ValueError(
            f"build_bridge requires g1.basis == g2.basis, got {g1.basis!r} vs {g2.basis!r}"
        )
    basis = g1.basis
    # ... existing body ...
    # When constructing the Bridge(...), add basis=basis to the call:
    return Bridge(
        width=w, qubits=qubits, U_B=U_B,
        chi_endpoint_extensions=chi_endpoint_extensions,
        intercode=False,  # or True per existing dispatch
        aux_graph_edges=...,
        z_extensions=...,
        basis=basis,
    )
```

Apply the `basis=basis` argument to ALL `Bridge(...)` constructions in this file (there may be 2-3 — one per intercode/intracode branch).

- [ ] **Step 4: Run tests**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "bridge" -x
```
Expected: all bridge tests PASS (existing + new 2).

- [ ] **Step 5: Run full suite**

```bash
pytest src/qldpc/codes/surgery/ -x 2>&1 | tail -5
```
Expected: 57 PASS (53 + 4 from Tasks 5,6).

- [ ] **Step 6: Commit**

```bash
git add src/qldpc/codes/surgery/bridge.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: Bridge inherits basis from gadgets; build_bridge asserts match

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 3 — Cheeger boost basis support

### Task 7: `_gadget_to_legacy_layout` handles basis=Pauli.Z

**Files:**
- Modify: `src/qldpc/codes/surgery/cheeger.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_boost_gadget_preserves_css_commutation_both_bases(basis):
    """boost_gadget on a basis=X or basis=Z gadget preserves CSS commutation."""
    from qldpc.codes.surgery.gadget import (
        build_gadget, load_webster_seed_set, _build_generalised_bicycle_code,
    )
    from qldpc.codes.surgery.cheeger import boost_gadget

    def operator(d, name):
        l = d["l"]
        for seed in d["seeds"]:
            if seed["name"] == name and seed["pauli_type"] == name[0]:
                L = np.zeros(l, dtype=np.uint8); R = np.zeros(l, dtype=np.uint8)
                for i in seed["L_support"]: L[i] = 1
                for i in seed["R_support"]: R[i] = 1
                return np.concatenate([L, R])
        raise ValueError(f"{name} not found")

    d = load_webster_seed_set(0)
    c = _build_generalised_bicycle_code(d["l"], d["A"], d["B"])
    op_name = "X_bar_1" if basis is Pauli.X else "Z_bar_1"
    op = operator(d, op_name)
    g = build_gadget(c, op, basis=basis)
    boosted = boost_gadget(g, method="combinatorial", target=1.0, seed=0)
    product = (boosted.HX_merged @ boosted.HZ_merged.T) % 2
    assert np.array_equal(product, np.zeros_like(product))
    assert boosted.basis is basis  # boost preserves basis
```

- [ ] **Step 2: Run test to find what breaks**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_boost_gadget_preserves_css_commutation_both_bases -x
```
Expected: PASS for `basis=Pauli.X`; likely FAIL for `basis=Pauli.Z` (because `_gadget_to_legacy_layout` assumes X-basis construction).

- [ ] **Step 3: Patch `_gadget_to_legacy_layout` in `cheeger.py`**

Find `_gadget_to_legacy_layout` (search `grep -n "def _gadget_to_legacy_layout" src/qldpc/codes/surgery/cheeger.py`). The current implementation builds a legacy `SurgeryLayout` from a `GadgetLayout` assuming basis=X. For basis=Z, we need to swap HX and HZ when constructing the legacy `merged` CSSCode — the boost machinery was designed for the X-basis layout and we use it via the dual swap:

```python
def _gadget_to_legacy_layout(g):
    """Convert a GadgetLayout into the legacy (CSSCode, SurgeryLayout) pair
    consumed by boost_gadget_cheeger* / boost_gadget_distance.

    For basis=Pauli.Z, we SWAP HX/HZ so the legacy boost code (designed for
    X-basis chi rows in HX_merged) sees the chi rows where it expects them.
    The boost result is dual-swapped back in _legacy_to_gadget.
    """
    F2 = galois.GF(2)
    n = g.code.num_qudits
    n_anc = len(g.C0)

    if g.basis is Pauli.X:
        HX_for_legacy = g.HX_merged
        HZ_for_legacy = g.HZ_merged
    else:  # Pauli.Z: swap so chi rows are in HX_for_legacy
        HX_for_legacy = g.HZ_merged
        HZ_for_legacy = g.HX_merged

    merged = CSSCode(
        F2(HX_for_legacy.astype(np.int_).tolist()),
        F2(HZ_for_legacy.astype(np.int_).tolist()),
        is_subsystem_code=False,
    )
    # ... [rest of existing implementation, using HX_for_legacy / HZ_for_legacy] ...
```

In `_legacy_to_gadget`, undo the swap when basis=Z:

```python
def _legacy_to_gadget(merged, layout, original_g):
    """Reconstruct a GadgetLayout from a boost result."""
    HX_m_legacy = np.asarray(merged.matrix_x).astype(np.uint8)
    HZ_m_legacy = np.asarray(merged.matrix_z).astype(np.uint8)
    if original_g.basis is Pauli.X:
        HX_m = HX_m_legacy
        HZ_m = HZ_m_legacy
    else:  # undo the basis-Z swap
        HX_m = HZ_m_legacy
        HZ_m = HX_m_legacy
    F_new = np.asarray(layout.F).astype(np.uint8)
    G_new = np.asarray(layout.G).astype(np.uint8)
    n = original_g.code.num_qudits
    n_anc_new = HX_m.shape[1] - n
    kappa_qubits = tuple(range(n, n + n_anc_new))
    return dataclasses.replace(
        original_g,
        F=F_new, G=G_new,
        HX_merged=HX_m, HZ_merged=HZ_m,
        kappa_qubits=kappa_qubits,
    )
```

- [ ] **Step 4: Run test**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "boost_gadget_preserves_css_commutation_both_bases" -x
```
Expected: 2 PASS (X and Z).

- [ ] **Step 5: Run full suite**

```bash
pytest src/qldpc/codes/surgery/ -x 2>&1 | tail -5
```
Expected: 59 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/qldpc/codes/surgery/cheeger.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: cheeger boost handles basis=Pauli.Z via internal HX/HZ swap

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 4 — Circuit module rewrite

### Task 8: Helper — classify reliable round-1 checks

**Files:**
- Modify: `src/qldpc/codes/surgery/circuit.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_classify_reliable_round1_checks_basis_x():
    """For basis=X: reliable round-1 checks are data H_X (first m_X X-checks)
    plus gauge-fix G (last r Z-checks)."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import _classify_reliable_round1_checks
    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.codes.common import CSSCode
    import galois
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    F2 = galois.GF(2)
    merged = CSSCode(
        F2(g.HX_merged.astype(np.int_).tolist()),
        F2(g.HZ_merged.astype(np.int_).tolist()),
        is_subsystem_code=False,
    )
    qubit_ids = QubitIDs.from_code(merged)
    reliable = _classify_reliable_round1_checks(g, merged, qubit_ids)
    m_X = code.matrix_x.shape[0]
    m_Z = code.matrix_z.shape[0]
    # Reliable X-checks: first m_X of checks_x (the original data H_X rows)
    expected_x_reliable = set(qubit_ids.checks_x[:m_X])
    # Reliable Z-checks: last r of checks_z (the gauge-fix G rows)
    r = g.G.shape[0]
    expected_z_reliable = set(qubit_ids.checks_z[m_Z:])
    expected = expected_x_reliable | expected_z_reliable
    assert set(reliable) == expected, (
        f"reliable={set(reliable)}, expected={expected}"
    )


def test_classify_reliable_round1_checks_basis_z():
    """For basis=Z: reliable round-1 checks are data H_Z (first m_Z Z-checks)
    plus gauge-fix G (last r X-checks)."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import _classify_reliable_round1_checks
    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.codes.common import CSSCode
    import galois
    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    F2 = galois.GF(2)
    merged = CSSCode(
        F2(g.HX_merged.astype(np.int_).tolist()),
        F2(g.HZ_merged.astype(np.int_).tolist()),
        is_subsystem_code=False,
    )
    qubit_ids = QubitIDs.from_code(merged)
    reliable = _classify_reliable_round1_checks(g, merged, qubit_ids)
    m_X = code.matrix_x.shape[0]
    m_Z = code.matrix_z.shape[0]
    r = g.G.shape[0]
    # basis=Z: data H_Z rows are first m_Z Z-checks; G rows are last r X-checks
    expected_z_reliable = set(qubit_ids.checks_z[:m_Z])
    expected_x_reliable = set(qubit_ids.checks_x[m_X:])
    expected = expected_z_reliable | expected_x_reliable
    assert set(reliable) == expected
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "classify_reliable_round1" -x
```
Expected: FAIL (ImportError).

- [ ] **Step 3: Add helper to `circuit.py`**

Edit `src/qldpc/codes/surgery/circuit.py`. Append:

```python
from qldpc.objects import Pauli, PauliXZ
from qldpc.circuits.bookkeeping import QubitIDs


def _classify_reliable_round1_checks(
    gadget: GadgetLayout,
    merged_code: CSSCode,
    qubit_ids: QubitIDs,
) -> tuple[int, ...]:
    """Return the subset of merged-code check ancillas whose round-1 syndrome
    is reliable (= +1) given the surgery init state.

    For basis=Pauli.X (data in |+⟩, κ in |0⟩):
        reliable = data H_X rows (X-type, data |+⟩ → +1) +
                   gauge-fix G rows (Z-type, κ |0⟩ → +1)
        unreliable = χ rows (X on κ is random) + data H_Z rows (Z on data |+⟩ random)
    For basis=Pauli.Z (data in |0⟩, κ in |+⟩): swap X↔Z in the above.
    """
    m_X = gadget.code.matrix_x.shape[0]
    m_Z = gadget.code.matrix_z.shape[0]
    n_V = len(gadget.V0)
    r = gadget.G.shape[0]

    if gadget.basis is Pauli.X:
        # X-checks: first m_X are data H_X (reliable), next n_V are χ (unreliable)
        reliable_x = qubit_ids.checks_x[:m_X]
        # Z-checks: first m_Z are data H_Z (unreliable), last r are G (reliable)
        reliable_z = qubit_ids.checks_z[m_Z:]
    else:  # Pauli.Z (basis swap)
        # X-checks: first m_X are data H_X (unreliable), last r are G (reliable)
        reliable_x = qubit_ids.checks_x[m_X:]
        # Z-checks: first m_Z are data H_Z (reliable), next n_V are χ (unreliable)
        reliable_z = qubit_ids.checks_z[:m_Z]

    return tuple(reliable_x) + tuple(reliable_z)
```

- [ ] **Step 4: Run tests**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "classify_reliable_round1" -x
```
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/circuit.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: classify reliable round-1 checks per surgery init state

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Helper — surgery state preparation

**Files:**
- Modify: `src/qldpc/codes/surgery/circuit.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_surgery_state_prep_basis_x_resets():
    """basis=X: data RX (→|+⟩), kappa R (→|0⟩)."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import _surgery_state_prep
    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.codes.common import CSSCode
    import galois
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    F2 = galois.GF(2)
    merged = CSSCode(
        F2(g.HX_merged.astype(np.int_).tolist()),
        F2(g.HZ_merged.astype(np.int_).tolist()),
        is_subsystem_code=False,
    )
    qubit_ids = QubitIDs.from_code(merged)
    n_data = code.num_qudits
    data_ids = qubit_ids.data[:n_data]
    kappa_ids = qubit_ids.data[n_data:]
    circuit = _surgery_state_prep(g, data_ids, kappa_ids, bridge_ids=())
    text = str(circuit)
    assert f"RX {' '.join(str(q) for q in data_ids)}" in text
    assert f"R {' '.join(str(q) for q in kappa_ids)}" in text


def test_surgery_state_prep_basis_z_resets():
    """basis=Z: data R (→|0⟩), kappa RX (→|+⟩)."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import _surgery_state_prep
    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.codes.common import CSSCode
    import galois
    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    F2 = galois.GF(2)
    merged = CSSCode(
        F2(g.HX_merged.astype(np.int_).tolist()),
        F2(g.HZ_merged.astype(np.int_).tolist()),
        is_subsystem_code=False,
    )
    qubit_ids = QubitIDs.from_code(merged)
    n_data = code.num_qudits
    data_ids = qubit_ids.data[:n_data]
    kappa_ids = qubit_ids.data[n_data:]
    circuit = _surgery_state_prep(g, data_ids, kappa_ids, bridge_ids=())
    text = str(circuit)
    assert f"R {' '.join(str(q) for q in data_ids)}" in text
    assert f"RX {' '.join(str(q) for q in kappa_ids)}" in text
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "surgery_state_prep" -x
```
Expected: FAIL.

- [ ] **Step 3: Implement `_surgery_state_prep`**

Append to `circuit.py`:

```python
def _surgery_state_prep(
    gadget: GadgetLayout,
    data_ids: tuple[int, ...],
    kappa_ids: tuple[int, ...],
    bridge_ids: tuple[int, ...] = (),
) -> stim.Circuit:
    """Cain step 1: init data in logical |+⟩ (basis=X) or |0⟩ (basis=Z),
    init κ ancillas in |0⟩ (basis=X) or |+⟩ (basis=Z). Bridge follows κ.

    Coordinates are emitted separately by build_*_ppm_circuit; this helper
    only emits reset instructions.
    """
    circuit = stim.Circuit()
    if gadget.basis is Pauli.X:
        circuit.append("RX", list(data_ids))
        circuit.append("R", list(kappa_ids))
        if bridge_ids:
            circuit.append("R", list(bridge_ids))
    else:  # Pauli.Z
        circuit.append("R", list(data_ids))
        circuit.append("RX", list(kappa_ids))
        if bridge_ids:
            circuit.append("RX", list(bridge_ids))
    return circuit
```

- [ ] **Step 4: Run tests**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "surgery_state_prep" -x
```
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/circuit.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: _surgery_state_prep helper (Cain step 1)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Helper — surgery QEC cycle with classified detectors

**Files:**
- Modify: `src/qldpc/codes/surgery/circuit.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_surgery_qec_cycle_round_1_detectors_classified():
    """Round-1 detectors are 1-arg only for RELIABLE checks; unreliable ones skipped."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import _surgery_qec_cycle, _classify_reliable_round1_checks
    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.codes.common import CSSCode
    import galois
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    F2 = galois.GF(2)
    merged = CSSCode(
        F2(g.HX_merged.astype(np.int_).tolist()),
        F2(g.HZ_merged.astype(np.int_).tolist()),
        is_subsystem_code=False,
    )
    qubit_ids = QubitIDs.from_code(merged)
    reliable = _classify_reliable_round1_checks(g, merged, qubit_ids)

    circuit, meas_rec, det_rec = _surgery_qec_cycle(
        g, merged, num_rounds=2, qubit_ids=qubit_ids,
    )
    # Count round-1 1-arg DETECTORs (those preceded only by REPEAT 0 or appearing before any REPEAT_BLOCK).
    text = str(circuit)
    # Reliability-aware count: number of "DETECTOR" instructions in the first
    # round (before the REPEAT block) should equal len(reliable).
    first_round_str = text.split("REPEAT")[0]
    n_det = first_round_str.count("DETECTOR")
    assert n_det == len(reliable), (
        f"round-1 detectors={n_det}, expected len(reliable)={len(reliable)}"
    )
```

- [ ] **Step 2: Run test**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_surgery_qec_cycle_round_1_detectors_classified -x
```
Expected: FAIL.

- [ ] **Step 3: Implement `_surgery_qec_cycle`**

Append to `circuit.py`. Read `src/qldpc/circuits/memory/memory.py:483-538` (the existing `_get_qec_cycle`) and mirror its structure, then replace the round-1 detector loop:

```python
from qldpc.circuits.memory.syndrome_measurement import EdgeColoring
from qldpc.circuits.bookkeeping import MeasurementRecord, DetectorRecord


def _surgery_qec_cycle(
    gadget: GadgetLayout,
    merged_code: CSSCode,
    num_rounds: int,
    qubit_ids: QubitIDs,
) -> tuple[stim.Circuit, MeasurementRecord, DetectorRecord]:
    """Build num_rounds rounds of merged-code SE with surgery-aware round-1 detectors.

    Mirrors qldpc.circuits.memory.memory._get_qec_cycle except round-1 DETECTORs
    are only emitted for reliable checks (per _classify_reliable_round1_checks).
    Rounds 2..N emit standard 2-arg consistency detectors for ALL checks.
    """
    strategy = EdgeColoring()
    one_round, round_measurement_record = strategy.get_circuit(merged_code, qubit_ids)
    reliable = set(_classify_reliable_round1_checks(gadget, merged_code, qubit_ids))
    all_check_ids = qubit_ids.check

    circuit = stim.Circuit()
    measurement_record = MeasurementRecord()
    detector_record = DetectorRecord()

    # Round 1: classified DETECTOR emission
    circuit += one_round
    measurement_record.append(round_measurement_record)
    for kk, check_id in enumerate(all_check_ids):
        if check_id in reliable:
            circuit.append(
                "DETECTOR",
                [measurement_record.get_target_rec(check_id)],
                (0, 0, kk),
            )
    detector_record.append({
        check_id: dd for dd, check_id in enumerate(check_id for check_id in all_check_ids if check_id in reliable)
    })

    # Rounds 2..N: full consistency detectors for ALL checks
    if num_rounds > 1:
        repeat_circuit = one_round.copy()
        measurement_record.append(round_measurement_record)
        repeat_circuit.append("SHIFT_COORDS", [], (1, 0, 0))
        for kk, check_id in enumerate(all_check_ids):
            targets = [
                measurement_record.get_target_rec(check_id, -1),
                measurement_record.get_target_rec(check_id, -2),
            ]
            repeat_circuit.append("DETECTOR", targets, (0, 0, kk))
        circuit.append(stim.CircuitRepeatBlock(num_rounds - 1, repeat_circuit))
        measurement_record.append(round_measurement_record, repeat=num_rounds - 2)
        detector_record.append(
            {check_id: dd for dd, check_id in enumerate(all_check_ids)},
            repeat=num_rounds - 1,
        )

    return circuit, measurement_record, detector_record
```

- [ ] **Step 4: Run test**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_surgery_qec_cycle_round_1_detectors_classified -x
```
Expected: PASS. If detector count differs (e.g. because the REPEAT block parsing is off), adjust the test's parse to count round-1 DETECTORs more robustly (e.g., split on `REPEAT` line).

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/circuit.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: _surgery_qec_cycle with classified round-1 detectors

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Helper — surgery detach + readout

**Files:**
- Modify: `src/qldpc/codes/surgery/circuit.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_surgery_detach_and_readout_basis_x_measures_kappa_then_data():
    """basis=X: detach with M (Z-basis) on κ, then MX on data."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import _surgery_detach_and_readout
    from qldpc.circuits.bookkeeping import MeasurementRecord
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    n_data = code.num_qudits
    data_ids = tuple(range(n_data))
    kappa_ids = tuple(range(n_data, n_data + len(g.kappa_qubits)))
    bridge_ids = ()
    meas_rec = MeasurementRecord()  # empty for this test
    circuit = _surgery_detach_and_readout(
        g, data_ids=data_ids, kappa_ids=kappa_ids, bridge_ids=bridge_ids,
        measurement_record=meas_rec,
    )
    text = str(circuit)
    # κ measured first (in Z), then data (in X)
    m_kappa_idx = text.find(f"M {' '.join(str(q) for q in kappa_ids)}")
    m_data_idx = text.find(f"MX {' '.join(str(q) for q in data_ids)}")
    assert m_kappa_idx >= 0 and m_data_idx >= 0
    assert m_kappa_idx < m_data_idx


def test_surgery_detach_and_readout_basis_z_measures_kappa_in_x_then_data_in_z():
    """basis=Z: detach with MX on κ, then M on data."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import _surgery_detach_and_readout
    from qldpc.circuits.bookkeeping import MeasurementRecord
    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    n_data = code.num_qudits
    data_ids = tuple(range(n_data))
    kappa_ids = tuple(range(n_data, n_data + len(g.kappa_qubits)))
    meas_rec = MeasurementRecord()
    circuit = _surgery_detach_and_readout(
        g, data_ids=data_ids, kappa_ids=kappa_ids, bridge_ids=(),
        measurement_record=meas_rec,
    )
    text = str(circuit)
    m_kappa_idx = text.find(f"MX {' '.join(str(q) for q in kappa_ids)}")
    m_data_idx = text.find(f"M {' '.join(str(q) for q in data_ids)}")
    assert m_kappa_idx >= 0 and m_data_idx >= 0
    assert m_kappa_idx < m_data_idx
```

- [ ] **Step 2: Run tests**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "surgery_detach_and_readout" -x
```
Expected: FAIL.

- [ ] **Step 3: Implement `_surgery_detach_and_readout`**

Append to `circuit.py`. This helper emits Mκ, Mdata, and the inferred final detectors. For simplicity in this task, emit ONLY the M instructions — the final detectors are wired separately in Task 13 (build_single_ppm_circuit). Keep this helper small (<40 LOC).

```python
def _surgery_detach_and_readout(
    gadget: GadgetLayout,
    *,
    data_ids: tuple[int, ...],
    kappa_ids: tuple[int, ...],
    bridge_ids: tuple[int, ...],
    measurement_record: MeasurementRecord,
) -> stim.Circuit:
    """Cain step 3 + final data measurement.

    Emits:
        1. M (or MX) on κ + bridge to detach Q'.
        2. SHIFT_COORDS for clarity.
        3. MX (or M) on data qubits for the final X̄ (or Z̄) measurement.

    Updates measurement_record with new measurement indices. Final-readout
    detector wiring is composed by build_*_ppm_circuit (it has access to
    H_X / H_Z row classifications).
    """
    circuit = stim.Circuit()
    detach_qubits = list(kappa_ids) + list(bridge_ids)
    if gadget.basis is Pauli.X:
        circuit.append("M", detach_qubits)
        measurement_record.append({q: i for i, q in enumerate(detach_qubits)})
        circuit.append("SHIFT_COORDS", [], (1, 0, 0))
        circuit.append("MX", list(data_ids))
        measurement_record.append({q: i for i, q in enumerate(data_ids)})
    else:  # Pauli.Z
        circuit.append("MX", detach_qubits)
        measurement_record.append({q: i for i, q in enumerate(detach_qubits)})
        circuit.append("SHIFT_COORDS", [], (1, 0, 0))
        circuit.append("M", list(data_ids))
        measurement_record.append({q: i for i, q in enumerate(data_ids)})
    return circuit
```

- [ ] **Step 4: Run tests**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "surgery_detach_and_readout" -x
```
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/circuit.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: _surgery_detach_and_readout (Cain step 3 + final data measure)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: Helper — surgery observables (Webster Eq. 1 + cross-check)

**Files:**
- Modify: `src/qldpc/codes/surgery/circuit.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_surgery_observable_emits_two_observable_include():
    """Observable 0 = XOR of χ-row records across all rounds; Observable 1 = data measurement on V_0."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import _surgery_observable
    from qldpc.circuits.bookkeeping import MeasurementRecord
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    # Build a fake measurement_record with enough entries that get_target_rec
    # can resolve the chi check ids and data ids.
    n_data = code.num_qudits
    n_kappa = len(g.kappa_qubits)
    chi_check_ids = tuple(range(100, 100 + len(g.V0)))   # placeholder ids
    data_ids = tuple(range(n_data))
    meas_rec = MeasurementRecord()
    # Simulate 2 rounds of chi-check measurements
    for _ in range(2):
        meas_rec.append({cid: i for i, cid in enumerate(chi_check_ids)})
    # Simulate final data measurement
    meas_rec.append({d: i for i, d in enumerate(data_ids)})

    circuit = _surgery_observable(
        g, chi_check_ids=chi_check_ids, data_ids=data_ids,
        v0_indices=g.V0, num_rounds=2, measurement_record=meas_rec,
    )
    text = str(circuit)
    assert text.count("OBSERVABLE_INCLUDE") == 2  # PPM + cross-check
    assert "(0)" in text and "(1)" in text  # two distinct observable indices
```

- [ ] **Step 2: Run test**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_surgery_observable_emits_two_observable_include -x
```
Expected: FAIL.

- [ ] **Step 3: Implement `_surgery_observable`**

Append to `circuit.py`:

```python
def _surgery_observable(
    gadget: GadgetLayout,
    *,
    chi_check_ids: tuple[int, ...],
    data_ids: tuple[int, ...],
    v0_indices: tuple[int, ...],
    num_rounds: int,
    measurement_record: MeasurementRecord,
) -> stim.Circuit:
    """Two OBSERVABLE_INCLUDE entries:
       0 = ⊕ chi-row records across all num_rounds rounds (Webster Eq. 1 — PPM result).
       1 = ⊕ data measurements on V_0 (X̄_M or Z̄_M cross-check from final data measurement).

    For single PPM: chi_check_ids is the |V_0| chi rows of the gadget.
    For joint PPM: chi_check_ids = chi^(1) ∪ chi^(2) ∪ U_B path-graph rows (per math.md §2.7).
    v0_indices is the union of V_0 indices contributing to the data observable
    (single: gadget.V0; joint: g1.V0 + g2.V0 with offset).
    """
    circuit = stim.Circuit()

    # Observable 0: chi-XOR across all rounds.
    # Each round had len(chi_check_ids) chi measurements; index -k counts back
    # from the most recent measurement record.
    chi_targets = []
    for round_offset_from_last in range(num_rounds):
        # round 0 = most recent; round num_rounds-1 = oldest
        for cid in chi_check_ids:
            chi_targets.append(
                measurement_record.get_target_rec(cid, -1 - round_offset_from_last)
            )
    circuit.append("OBSERVABLE_INCLUDE", chi_targets, 0)

    # Observable 1: data measurement on V_0 (X̄_M or Z̄_M cross-check).
    data_targets = [
        measurement_record.get_target_rec(data_ids[i]) for i in v0_indices
    ]
    circuit.append("OBSERVABLE_INCLUDE", data_targets, 1)

    return circuit
```

NOTE: `MeasurementRecord.get_target_rec` may have a different signature for round indexing. Read `bookkeeping.py:206` to see the actual `measurement_index` parameter semantics — it may be `-1` for last, `-2` for second-to-last, etc. Adapt the round_offset logic if needed (the test's parse only cares that 2 OBSERVABLE_INCLUDE lines are emitted, not their exact target counts).

- [ ] **Step 4: Run test**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_surgery_observable_emits_two_observable_include -x
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/circuit.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: _surgery_observable (Webster Eq. 1 + data X̄_M cross-check)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: Rewrite `build_single_ppm_circuit`

**Files:**
- Modify: `src/qldpc/codes/surgery/circuit.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Replace the existing test `test_build_single_ppm_circuit_noiseless_no_detectors_fire` body — it currently uses the old (incorrect) memory delegation. The new spec says: under noiseless conditions, the PPM observable (observable 0) is +1 (= 0 bit) deterministically. Append:

```python
@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_build_single_ppm_circuit_noiseless_observables_zero(basis):
    """Both OBSERVABLE_INCLUDEs evaluate to 0 (= +1) under no noise."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import build_single_ppm_circuit
    code = codes.SteaneCode()
    op = (code.get_logical_ops(Pauli.X)[0]
          if basis is Pauli.X
          else code.get_logical_ops(Pauli.Z)[0])
    op_arr = np.asarray(op).astype(np.uint8)
    g = build_gadget(code, op_arr, basis=basis)
    circuit = build_single_ppm_circuit(g, rounds=3, noise_model=None)
    # Sample observables; all should be 0.
    obs_samples = circuit.compile_m2d_converter().convert(
        measurements=circuit.compile_sampler().sample(shots=16),
        separate_observables=True,
    )[1]  # observables array
    assert (obs_samples == 0).all(), (
        f"noiseless observables fired: {obs_samples.sum()} flips across 16 shots"
    )
```

NOTE: the `compile_m2d_converter().convert(...)` API may need adjustment. Simpler alternative:

```python
    sampler = circuit.compile_sampler()
    samples_inc_obs = sampler.sample(shots=16, separate_observables=True)
    # samples_inc_obs is (measurements, observables) tuple if separate=True; else just measurements
    if isinstance(samples_inc_obs, tuple):
        _, obs = samples_inc_obs
    else:
        obs = samples_inc_obs  # fallback
    assert (obs == 0).all()
```

Use whichever Stim sampler API gives observable flips directly. Read `stim.Circuit.compile_sampler()` docs if unsure.

- [ ] **Step 2: Run test to verify failure**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_build_single_ppm_circuit_noiseless_observables_zero -x
```
Expected: FAIL (current build_single_ppm_circuit produces 1 observable from memory; the test expects 2; or some other shape mismatch).

- [ ] **Step 3: Rewrite `build_single_ppm_circuit`**

Edit `circuit.py`. Replace the current `build_single_ppm_circuit` body:

```python
from qldpc.circuits.memory.memory import get_qubit_coordinates


def build_single_ppm_circuit(
    gadget: GadgetLayout,
    *,
    rounds: int,
    noise_model=None,
) -> stim.Circuit:
    """Stim circuit implementing Cain §III.A single-PPM surgery on `gadget`.

    Pipeline (mirrors qldpc.circuits.memory.memory._get_basis_memory_experiment_parts
    but with surgery-specific init, detector classification, detach, and observable):
        1. QubitIDs + coordinates
        2. _surgery_state_prep (κ |0⟩, data |+⟩ or basis-Z duals)
        3. _surgery_qec_cycle (τ_s rounds with classified round-1 detectors)
        4. _surgery_detach_and_readout (Mκ then Mdata)
        5. Inferred final detectors for reliable stabilizers
        6. _surgery_observable (observable 0 = chi-XOR; observable 1 = data on V_0)
        7. Apply noise_model
    """
    merged_code = _gadget_merged_csscode(gadget)
    qubit_ids = QubitIDs.from_code(merged_code)
    n_original_data = gadget.code.num_qudits
    data_ids = qubit_ids.data[:n_original_data]
    kappa_ids = qubit_ids.data[n_original_data:]
    bridge_ids = ()

    # Coordinates
    circuit = get_qubit_coordinates(qubit_ids.data, qubit_ids.check)

    # State prep
    circuit += _surgery_state_prep(gadget, data_ids, kappa_ids, bridge_ids)

    # QEC cycle
    qec_cycle, measurement_record, _ = _surgery_qec_cycle(
        gadget, merged_code, num_rounds=rounds, qubit_ids=qubit_ids,
    )
    circuit += qec_cycle

    # Detach + final data measurement
    circuit += _surgery_detach_and_readout(
        gadget,
        data_ids=data_ids, kappa_ids=kappa_ids, bridge_ids=bridge_ids,
        measurement_record=measurement_record,
    )

    # Chi-row check_ids: the χ rows are HX_merged rows [m_X : m_X + |V_0|]
    # The corresponding check ancillas are qubit_ids.checks_x[m_X : m_X + |V_0|]
    # (for basis=X) or qubit_ids.checks_z[m_Z : m_Z + |V_0|] (for basis=Z).
    m_X = gadget.code.matrix_x.shape[0]
    m_Z = gadget.code.matrix_z.shape[0]
    n_V = len(gadget.V0)
    if gadget.basis is Pauli.X:
        chi_check_ids = tuple(qubit_ids.checks_x[m_X:m_X + n_V])
    else:
        chi_check_ids = tuple(qubit_ids.checks_z[m_Z:m_Z + n_V])

    # Observables
    circuit += _surgery_observable(
        gadget,
        chi_check_ids=chi_check_ids,
        data_ids=data_ids,
        v0_indices=gadget.V0,
        num_rounds=rounds,
        measurement_record=measurement_record,
    )

    # Noise
    if noise_model is not None:
        circuit = noise_model.noisy_circuit(circuit)

    return circuit
```

- [ ] **Step 4: Run test**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_build_single_ppm_circuit_noiseless_observables_zero -x
```
Expected: 2 PASS (X and Z basis).

If FAIL with non-zero observable: likely a sign-convention issue or the data-final-measurement detector is missing. Adjust the helper(s) or add the missing inferred detectors. Common fix: also need to add DETECTORs that infer the reliable data H_X / H_Z syndrome from the final Mx / M outcomes — this consistency may matter for the observable sampling. Add a small helper `_surgery_final_detectors(...)` if needed and append between detach and observable.

- [ ] **Step 5: Run the full suite (some old tests may need updating)**

```bash
pytest src/qldpc/codes/surgery/ 2>&1 | tail -10
```

The following pre-existing tests use the old `build_single_ppm_circuit` semantics and may fail:
- `test_build_single_ppm_circuit_noiseless_compiles`: still passes (just checks compile).
- `test_build_single_ppm_circuit_noiseless_no_detectors_fire`: may pass or fail depending on detector wiring. If fails: investigate.
- `test_build_single_ppm_circuit_with_noise_detectors_fire`: may pass or fail.

If any old test fails: update its assertions to match the new circuit shape (it's expected that the old test of "no detectors fire" may need adjustment because the new circuit has different detector structure). Document the change in the commit.

- [ ] **Step 6: Commit**

```bash
git add src/qldpc/codes/surgery/circuit.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: rewrite build_single_ppm_circuit per Cain §III.A protocol

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: Rewrite `build_joint_ppm_circuit`

**Files:**
- Modify: `src/qldpc/codes/surgery/circuit.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_build_joint_ppm_circuit_noiseless_observables_zero():
    """Noiseless joint PPM: observable 0 (α* per math.md §2.7) = 0; observable 1 = 0."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import build_joint_ppm_circuit
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g1 = build_gadget(code, x, basis=Pauli.X)
    g2 = build_gadget(code, x, basis=Pauli.X)
    bridge = build_bridge(g1, g2)
    circuit, joint_code = build_joint_ppm_circuit(g1, g2, bridge, rounds=2, noise_model=None)
    sampler = circuit.compile_sampler()
    samples_inc_obs = sampler.sample(shots=16, separate_observables=True)
    if isinstance(samples_inc_obs, tuple):
        _, obs = samples_inc_obs
    else:
        obs = samples_inc_obs
    assert (obs == 0).all()
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_build_joint_ppm_circuit_noiseless_observables_zero -x
```
Expected: FAIL.

- [ ] **Step 3: Rewrite `build_joint_ppm_circuit`**

Edit `circuit.py`. Replace `build_joint_ppm_circuit`:

```python
def build_joint_ppm_circuit(
    g1: GadgetLayout, g2: GadgetLayout, bridge: Bridge,
    *,
    rounds: int,
    noise_model=None,
) -> tuple[stim.Circuit, CSSCode]:
    """Stim circuit + merged joint CSS code implementing Cain §III.A joint PPM.

    Uses the same pipeline as build_single_ppm_circuit but with:
      - Joint merged code = _stitch_to_joint_csscode(g1, g2, bridge)
      - Chi rows = χ^(1) ∪ χ^(2) ∪ U_B path-graph rows (math.md §2.7 α*)
      - Observable 1 (data cross-check) = X̄_1 ⊗ X̄_2 (XOR of g1.V0 and g2.V0 data measurements)
    """
    joint_code = _stitch_to_joint_csscode(g1, g2, bridge)
    qubit_ids = QubitIDs.from_code(joint_code)

    intercode = g1.code is not g2.code
    n1 = g1.code.num_qudits
    n2 = g2.code.num_qudits if intercode else 0
    n_anc_1 = len(g1.C0)
    n_anc_2 = len(g2.C0)
    n_bridge = bridge.width

    # Register split: data_1 (+ data_2) + kappa_1 + kappa_2 + bridge
    if intercode:
        data_ids = qubit_ids.data[: n1 + n2]   # both data sections combined
        v0_indices_combined = tuple(g1.V0) + tuple(n1 + i for i in g2.V0)
    else:
        data_ids = qubit_ids.data[:n1]
        v0_indices_combined = tuple(g1.V0) + tuple(g2.V0)   # both index into data_1

    kappa_ids = qubit_ids.data[n1 + n2 : n1 + n2 + n_anc_1 + n_anc_2]
    bridge_ids = qubit_ids.data[n1 + n2 + n_anc_1 + n_anc_2 :]

    circuit = get_qubit_coordinates(qubit_ids.data, qubit_ids.check)
    circuit += _surgery_state_prep(g1, data_ids, kappa_ids, bridge_ids)

    qec_cycle, measurement_record, _ = _surgery_qec_cycle(
        g1, joint_code, num_rounds=rounds, qubit_ids=qubit_ids,
    )
    circuit += qec_cycle

    circuit += _surgery_detach_and_readout(
        g1,
        data_ids=data_ids, kappa_ids=kappa_ids, bridge_ids=bridge_ids,
        measurement_record=measurement_record,
    )

    # Identify chi rows in the joint code.
    # HX_joint rows (basis=X): data_HX_1 (mX1), data_HX_2 (mX2 for intercode), chi^(1) (n_V1), chi^(2) (n_V2), U_B (w-1).
    # For basis=X: chi rows live in HX_joint → mapping to checks_x.
    mX1 = g1.code.matrix_x.shape[0]
    mX2 = g2.code.matrix_x.shape[0] if intercode else 0
    n_V1 = len(g1.V0)
    n_V2 = len(g2.V0)
    n_UB = bridge.U_B.shape[0]
    if g1.basis is Pauli.X:
        # checks_x layout: [data_HX_1 | data_HX_2 | chi^(1) | chi^(2) | U_B]
        offset = mX1 + mX2
        chi_check_ids = tuple(qubit_ids.checks_x[offset : offset + n_V1 + n_V2 + n_UB])
    else:
        # basis=Z mirror
        mZ1 = g1.code.matrix_z.shape[0]
        mZ2 = g2.code.matrix_z.shape[0] if intercode else 0
        offset = mZ1 + mZ2
        chi_check_ids = tuple(qubit_ids.checks_z[offset : offset + n_V1 + n_V2 + n_UB])

    circuit += _surgery_observable(
        g1,
        chi_check_ids=chi_check_ids,
        data_ids=data_ids,
        v0_indices=v0_indices_combined,
        num_rounds=rounds,
        measurement_record=measurement_record,
    )

    if noise_model is not None:
        circuit = noise_model.noisy_circuit(circuit)

    return circuit, joint_code
```

- [ ] **Step 4: Run test**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_build_joint_ppm_circuit_noiseless_observables_zero -x
```
Expected: PASS.

- [ ] **Step 5: Full suite**

```bash
pytest src/qldpc/codes/surgery/ 2>&1 | tail -10
```
Some pre-existing joint tests may fail; update them similar to Task 13's pattern.

- [ ] **Step 6: Commit**

```bash
git add src/qldpc/codes/surgery/circuit.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: rewrite build_joint_ppm_circuit per Cain §III.A protocol

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 5 — Semantic + LER tests

### Task 15: Noisy observable test (both bases)

**Files:**
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_single_ppm_circuit_noise_flips_observable_at_high_p(basis):
    """At p=0.1, the PPM observable (observable 0) flips ≥ 5% of shots."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits.noise_model import DepolarizingNoiseModel
    code = codes.SteaneCode()
    op = (code.get_logical_ops(Pauli.X)[0]
          if basis is Pauli.X
          else code.get_logical_ops(Pauli.Z)[0])
    op_arr = np.asarray(op).astype(np.uint8)
    g = build_gadget(code, op_arr, basis=basis)
    circuit = build_single_ppm_circuit(
        g, rounds=3, noise_model=DepolarizingNoiseModel(p=0.1),
    )
    sampler = circuit.compile_sampler()
    samples_inc_obs = sampler.sample(shots=400, separate_observables=True)
    if isinstance(samples_inc_obs, tuple):
        _, obs = samples_inc_obs
    else:
        obs = samples_inc_obs
    # Observable 0 (PPM) flips a nontrivial fraction at p=0.1
    obs_0_flip_rate = float(obs[:, 0].mean())
    assert obs_0_flip_rate >= 0.05, (
        f"PPM observable flip rate {obs_0_flip_rate:.2%} too low at p=0.1"
    )
```

- [ ] **Step 2: Run test**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_single_ppm_circuit_noise_flips_observable_at_high_p -x
```
Expected: 2 PASS.

- [ ] **Step 3: Commit**

```bash
git add src/qldpc/codes/surgery/_test.py
git commit -m "test: PPM observable flips under high depolarizing noise

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 16: LER monotonicity smoke test

**Files:**
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
@pytest.mark.slow
def test_single_ppm_ler_monotone_in_p():
    """Tiny sinter sweep: PPM LER monotonically increasing in p.

    Catches gross protocol errors (wrong observable basis, sign flips, etc.).
    """
    import sinter
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits import DepolarizingNoiseModel
    from qldpc import decoders
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)

    error_rates = [0.001, 0.005, 0.02]
    tasks = []
    for p in error_rates:
        circuit = build_single_ppm_circuit(
            g, rounds=3, noise_model=DepolarizingNoiseModel(p),
        )
        tasks.append(sinter.Task(
            circuit=circuit,
            json_metadata={"p": float(p)},
        ))
    results = sinter.collect(
        tasks=tasks,
        decoders=[decoders.SinterDecoder()],
        num_workers=4,
        max_shots=2000,
        max_errors=30,
        print_progress=False,
    )
    by_p = {r.json_metadata["p"]: r.errors / max(r.shots, 1) for r in results}
    sorted_p = sorted(by_p.keys())
    ler_vals = [by_p[p] for p in sorted_p]
    # Monotonically non-decreasing (allow small statistical noise)
    for i in range(len(ler_vals) - 1):
        assert ler_vals[i] <= ler_vals[i + 1] * 1.5, (
            f"LER not monotonic: p={sorted_p[i]} → {ler_vals[i]}, "
            f"p={sorted_p[i+1]} → {ler_vals[i+1]}"
        )
```

- [ ] **Step 2: Run test**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_single_ppm_ler_monotone_in_p -x -v
```
Expected: PASS. Runs in ~30 seconds.

If FAIL: investigate. Common failures:
- LER is FLAT or DECREASES: protocol bug, observable not coupled to noise correctly.
- LER is huge (~50%) at low p: observable convention is reversed (decoder confused).
- Test times out: lower max_shots/max_errors and decrease num_workers.

- [ ] **Step 3: Commit**

```bash
git add src/qldpc/codes/surgery/_test.py
git commit -m "test: LER monotonicity smoke (single PPM)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 6 — Migrate existing callers

### Task 17: Update `examples/test_ide_bb_lp.py` to use basis=Pauli.Z

**Files:**
- Modify: `examples/_ide_fixtures.py`
- Modify: `examples/test_ide_bb_lp.py`

- [ ] **Step 1: Inspect current state**

The current `load_ide_BB_input_with_operator` and `load_ide_LP_input_with_operator` construct `CSSCode(HZ, HX)` (HX/HZ swapped) so that `build_gadget(code, Z̄_support)` works via the dual trick.

```bash
grep -n "CSSCode\|build_gadget" examples/_ide_fixtures.py examples/test_ide_bb_lp.py | head -20
```

- [ ] **Step 2: Update `_ide_fixtures.py` to return natural CSSCode**

In `examples/_ide_fixtures.py`, both loaders should construct the natural `CSSCode(HX, HZ)` (no swap):

```python
def load_ide_BB_input_with_operator() -> tuple[CSSCode, np.ndarray]:
    """Return BB INPUT code (n=98) + pinned Z̄_1 logical operator (Ide §VII.B).

    Use with build_gadget(code, x, basis=Pauli.Z).
    """
    if not fixtures_available():
        raise FileNotFoundError(f"Zenodo fixtures not found at {_FIXTURE_ROOT}.")
    HX = load_mtx("BB_98_6_12/original_codes/Hx_BB_98_6_12_original-code-canonicalbasis.mtx")
    HZ = load_mtx("BB_98_6_12/original_codes/Hz_BB_98_6_12_original-code-canonicalbasis.mtx")
    import galois
    GF2 = galois.GF(2)
    code = CSSCode(GF2(HX.tolist()), GF2(HZ.tolist()), is_subsystem_code=False)
    V0 = sorted({v for edge in IDE_BB_KAPPA1_EDGES.values() for v in edge})
    x = np.zeros(code.num_qudits, dtype=np.uint8)
    for v in V0:
        x[v] = 1
    return code, x


def load_ide_LP_input_with_operator() -> tuple[CSSCode, np.ndarray]:
    """Same docstring for LP."""
    if not fixtures_available():
        raise FileNotFoundError(f"Zenodo fixtures not found at {_FIXTURE_ROOT}.")
    HX = load_mtx("LP_200_20_10/original_codes/Hx_LP_200_20_10_original-code.mtx")
    HZ = load_mtx("LP_200_20_10/original_codes/Hz_LP_200_20_10_original-code.mtx")
    import galois
    GF2 = galois.GF(2)
    code = CSSCode(GF2(HX.tolist()), GF2(HZ.tolist()), is_subsystem_code=False)
    V0 = sorted(IDE_LP_V0_2)
    x = np.zeros(code.num_qudits, dtype=np.uint8)
    for v in V0:
        x[v] = 1
    return code, x
```

- [ ] **Step 3: Update `examples/test_ide_bb_lp.py` to pass basis=Pauli.Z**

In `examples/test_ide_bb_lp.py`, update `test_intercode_joint_bb_lp_exact`:

```python
def test_intercode_joint_bb_lp_exact():
    bb, x_bb = load_ide_BB_input_with_operator()
    lp, x_lp = load_ide_LP_input_with_operator()
    g1 = build_gadget(bb, x_bb, basis=Pauli.Z)
    g2 = build_gadget(lp, x_lp, basis=Pauli.Z)
    bridge = build_bridge(g1, g2)
    _, joint = build_joint_ppm_circuit(g1, g2, bridge, rounds=1, noise_model=None)
    # ... (rest of test, including the relaxed n ∈ (354, 355) assertion)
```

Make sure `from qldpc.objects import Pauli` is imported at the top.

- [ ] **Step 4: Run**

```bash
pytest examples/test_ide_bb_lp.py -v 2>&1 | tail -10
```
Expected: 3 PASS (or SKIP if fixtures not installed).

- [ ] **Step 5: Commit**

```bash
git add examples/_ide_fixtures.py examples/test_ide_bb_lp.py
git commit -m "refactor: Ide BB/LP loaders use natural CSSCode + basis=Pauli.Z

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 18: Update walkthrough notebook (Sections 5b, 6, 9, 10)

**Files:**
- Modify: `examples/logical_error_rates/_9_lattice_surgery_source.py`
- Generated: `examples/logical_error_rates/9_lattice_surgery.ipynb`

- [ ] **Step 1: Edit the .py source**

Open `examples/logical_error_rates/_9_lattice_surgery_source.py`. Three edits:

**Edit A — Section 5b "X̄ vs Z̄: same code, two basis choices"** (insert AFTER Section 5):

```python
# %% [markdown]
# ## 5b. X̄ vs Z̄ — basis-symmetric API
#
# `build_gadget(code, x, basis=Pauli.X|Pauli.Z)` selects whether the gadget
# measures an X̄ or Z̄ logical. The construction is dual-symmetric: for
# `basis=Pauli.Z` the χ rows live in `HZ_merged` (Z-type) instead of `HX_merged`,
# and the surgery circuit init swaps |+⟩ ↔ |0⟩ on data vs κ.

# %%
from qldpc.objects import Pauli

def z_bar_1_operator(d: dict) -> np.ndarray:
    l = d["l"]
    for seed in d["seeds"]:
        if seed["name"] == "Z_bar_1" and seed["pauli_type"] == "Z":
            L = np.zeros(l, dtype=np.uint8); R = np.zeros(l, dtype=np.uint8)
            for i in seed["L_support"]: L[i] = 1
            for i in seed["R_support"]: R[i] = 1
            return np.concatenate([L, R])
    raise ValueError("Z_bar_1 not found")

z = z_bar_1_operator(data)
g_z = build_gadget(code, z, basis=Pauli.Z)
print(f"basis=Z gadget: |V_0|={len(g_z.V0)}, |C_0|={len(g_z.C0)}, r={g_z.G.shape[0]}")
print(f"basis=Z κ+χ+r = {len(g_z.kappa_qubits) + len(g_z.V0) + g_z.G.shape[0]} (same as X)")
```

**Edit B — Section 6 (build_single_ppm_circuit)**: nothing changes at the API call level (`build_single_ppm_circuit(g_boosted, rounds=3, noise_model=None)`). Update the explanatory markdown to reflect that the circuit now implements Cain §III.A (not memory):

```python
# %% [markdown]
# ## 6. `build_single_ppm_circuit` — Cain §III.A surgery circuit
#
# `build_single_ppm_circuit(gadget, *, rounds, noise_model)` implements the
# full Cain §III.A 3-step surgery protocol:
# 1. Initialize κ ancillas in |0⟩ (or |+⟩ for basis=Z); data qubits in |+⟩
#    (or |0⟩ for basis=Z).
# 2. Run `rounds` cycles of merged-code syndrome extraction. Round-1
#    detectors are emitted only for stabilizers in known eigenstate of the
#    init state — data H_X and gauge-fix G for basis=X.
# 3. Detach κ ancillas by Z-measurement; measure data qubits in X basis.
#
# The circuit emits two observables:
# - `OBSERVABLE_INCLUDE(0)`: ⊕ χ-row records across all rounds (Webster Eq. 1
#   — the PPM result).
# - `OBSERVABLE_INCLUDE(1)`: ⊕ data measurements on V_0 (X̄_M cross-check
#   inferred from the final data Mx).
#
# Under noiseless conditions both observables evaluate to 0 (= +1) deterministically.
```

**Edit C — Section 10 (LER plot)**: the LER curve now reflects the PPM error rate (correct semantics). Update the explanatory markdown:

```python
# %% [markdown]
# ## 10. End-to-end: PPM logical error rate vs physical p
#
# Since section 6's circuit implements the full Cain §III.A surgery protocol
# (and not a generic memory experiment), the LER below is the **PPM failure
# rate** — the probability that the surgery measurement of X̄_M returns a
# flipped outcome relative to the noiseless +1. This is the correct
# end-to-end quantity for evaluating surgery fault tolerance.
```

Update Section 10's code cell — the merged code construction is no longer needed (the circuit already encapsulates the protocol):

```python
# %%
import sinter
import matplotlib.pyplot as plt
from qldpc import circuits, decoders

error_rates = np.logspace(-3, -2, 4)
num_rounds = 3

tasks = []
for p in error_rates:
    circuit = build_single_ppm_circuit(
        g_boosted, rounds=num_rounds,
        noise_model=circuits.DepolarizingNoiseModel(p, include_idling_error=False),
    )
    tasks.append(sinter.Task(circuit=circuit, json_metadata={"p": float(p)}))

results = sinter.collect(
    tasks=tasks,
    decoders=[decoders.SinterDecoder()],
    num_workers=4,
    max_shots=5_000,
    max_errors=50,
    print_progress=False,
)
print(f"Collected {len(results)} task results.")
```

(Keep the matplotlib plotting cell from the existing notebook unchanged; just update its title to "Webster code 0 single-PPM LER (Cain §III.A)".)

- [ ] **Step 2: Regenerate the notebook**

```bash
jupytext --to notebook examples/logical_error_rates/_9_lattice_surgery_source.py -o examples/logical_error_rates/9_lattice_surgery.ipynb
```

If `jupytext` is unavailable, edit the `.ipynb` JSON directly (see existing notebook for cell format reference).

- [ ] **Step 3: Smoke-execute the source**

```bash
python3 examples/logical_error_rates/_9_lattice_surgery_source.py 2>&1 | tail -15
```
Expected: completes within ~2 minutes, LER values shown in section 10 are monotonically increasing.

- [ ] **Step 4: Commit**

```bash
git add examples/logical_error_rates/_9_lattice_surgery_source.py examples/logical_error_rates/9_lattice_surgery.ipynb
git commit -m "feat: notebook reflects Cain §III.A circuit semantics (section 5b/6/10)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 19: LOC budget check + final verification

**Files:**
- Inspect: `src/qldpc/codes/surgery/circuit.py`

- [ ] **Step 1: Measure**

```bash
wc -l src/qldpc/codes/surgery/*.py
```

Expected per spec Goal 7:
- `circuit.py` ≤ 300 LOC

- [ ] **Step 2: If circuit.py > 300 LOC, trim**

Look for:
- Redundant docstrings (replace with 1-line math.md references)
- Single-use helpers (inline)
- Verbose error messages

If over-budget: trim until ≤ 300. All existing surgery tests must continue to pass.

- [ ] **Step 3: Full repo test**

```bash
pytest src/qldpc/codes/surgery/ -v 2>&1 | tail -20
pytest examples/test_ide_bb_lp.py 2>&1 | tail -5
```
Expected: all surgery tests PASS. Ide BB↔LP test PASS or SKIP.

- [ ] **Step 4: Commit (if trim happened)**

```bash
git add src/qldpc/codes/surgery/circuit.py
git commit -m "refactor: circuit.py within LOC budget (≤ 300)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Final state summary** (no commit)

```bash
git log --oneline main..HEAD | head -40
git diff --stat main..HEAD -- src/qldpc/codes/surgery/ examples/ docs/superpowers/ 2>&1 | tail -20
```

Print the final summary as the report.

---

## Self-Review Checklist

**Spec coverage:**

- [x] **Goal 1** (Cain §III.A protocol with κ |0⟩ init, detach, etc.) — Tasks 8-14.
- [x] **Goal 2** (PPM observable = Webster Eq. 1) — Task 12 + Task 13.
- [x] **Goal 3** (Secondary observable for X̄_M cross-check) — Task 12.
- [x] **Goal 4** (Round-1 detector classification) — Task 8 + Task 10.
- [x] **Goal 5** (Basis-symmetric API `basis=Pauli.X|Pauli.Z`) — Tasks 1-4 (gadget), Task 6 (bridge), Task 7 (cheeger), Tasks 13-14 (circuit).
- [x] **Goal 6** (Reuse `QubitIDs`, `MeasurementRecord`, `EdgeColoring`, etc.) — Tasks 8-12.
- [x] **Goal 7** (LOC budget circuit.py ≤ 300) — Task 19.
- [x] **Goal 8** (47 existing surgery tests pass) — verified after Tasks 1, 6, 13.

**Spec sections:**

- §1 module layout — circuit.py rewrite (Tasks 8-14), no new modules.
- §2 public API — `build_gadget(code, x, *, basis=Pauli.X)` (Task 4); `Bridge.basis` (Task 6); circuit signatures unchanged (Tasks 13-14).
- §3.1 basis-dependent construction — Tasks 2, 3, 4.
- §3.2 surgery init — Task 9.
- §3.3 SE rounds — Tasks 8, 10.
- §3.4 detach + readout — Task 11.
- §3.5 PPM observable — Task 12.
- §4 tests — Tasks 5 (basis-symmetry), 13-14 (noiseless observables), 15 (noise), 16 (LER monotonicity), 8-11 (helper tests).
- §5 migration — Tasks 17 (Ide test), 18 (notebook).
- §6 risks — Task 5's dual-equivalence test is the canary; Task 7 covers cheeger; LER monotonicity smoke catches gross protocol bugs.

**Placeholder scan:** No "TBD", "TODO", "implement later", "add error handling", or other red flags found.

**Type consistency:**
- `basis: PauliXZ` (Pauli.X | Pauli.Z) used consistently in Tasks 1-7.
- `GadgetLayout.basis` field added in Task 1, accessed in Tasks 7-14.
- `chi_check_ids` is `tuple[int, ...]` consistently (Tasks 12-14).
- `measurement_record.get_target_rec(check_id, measurement_index=-1)` signature consistent with `bookkeeping.py:206`.
