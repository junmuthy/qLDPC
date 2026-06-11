# Cain-convention rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Webster's κ/χ/r/V_0/C_0/F/G vocabulary throughout `src/qldpc/circuits/surgery/` with Cain-style `ancilla_qubits / support / data_checks / incidence / gauge` plus basis-symmetric `n_meas_checks / n_comp_checks` counters, with the math unchanged.

**Architecture:** Sequence of 9 atomic renames. Each rename touches all relevant files at once (so the module stays internally consistent), runs `uv run pytest src/qldpc/circuits/surgery/ -q`, and commits independently. After each commit the test suite is green.

**Tech Stack:** Python (numpy / galois / stim), pytest, no new dependencies.

---

## Files Touched

| File | Role |
|---|---|
| `src/qldpc/circuits/surgery/gadget.py` | `GadgetLayout` dataclass + step helpers + `build_gadget*`. |
| `src/qldpc/circuits/surgery/bridge.py` | `Bridge` dataclass + `build_bridge`. |
| `src/qldpc/circuits/surgery/cheeger.py` | `boost_gadget*` consumers. |
| `src/qldpc/circuits/surgery/circuit.py` | Stim circuit builders (largest surface). |
| `src/qldpc/circuits/surgery/_test_gadget.py` | Gadget tests. |
| `src/qldpc/circuits/surgery/_test_bridge.py` | Bridge tests. |
| `src/qldpc/circuits/surgery/_test_cheeger.py` | Cheeger tests. |
| `src/qldpc/circuits/surgery/_test_circuit.py` | Circuit tests. |

Every task below lists exact line numbers and `Edit` tool patterns to mechanically apply.

## Pre-flight (before Task 1)

- [ ] **Step P.1: Establish baseline**

Run: `uv run pytest src/qldpc/circuits/surgery/ -q 2>&1 | tail -3`
Expected: `147 passed` (or close; record the exact number). All subsequent task commits MUST keep this count.

---

### Task 1: V0 → support

**Files (rename `V0` → `support` for the `GadgetLayout` field and all references):**
- Modify: `src/qldpc/circuits/surgery/gadget.py`
- Modify: `src/qldpc/circuits/surgery/bridge.py`
- Modify: `src/qldpc/circuits/surgery/circuit.py`
- Modify: `src/qldpc/circuits/surgery/_test_gadget.py`
- Modify: `src/qldpc/circuits/surgery/_test_bridge.py`
- Modify: `src/qldpc/circuits/surgery/_test_circuit.py`
- Modify: `src/qldpc/circuits/surgery/_test_cheeger.py`

- [ ] **Step 1.1: Rename `GadgetLayout` field `V0` → `support` in gadget.py**

In `src/qldpc/circuits/surgery/gadget.py` line 26:

```python
# Before
    V0: tuple[int, ...]
# After
    support: tuple[int, ...]
```

- [ ] **Step 1.2: Rename function-local `V0` → `support` and parameter `V0` → `support` in gadget.py**

In `_step1_restriction` (lines 38–62), `_step3_assemble` (lines 106–154), `build_gadget` (lines 177–185), `build_gadget_augmented` (lines 210–237): every local variable named `V0` and the keyword parameter `V0` becomes `support`. The `np.where(x)[0]` line stays. Example diff for `build_gadget`:

```python
# Before
    V0, C0, F = _step1_restriction(code, x, basis=basis)
    G = _step2_gauge_fix(F)
    HX_m, HZ_m = _step3_assemble(code, V0, C0, F, G, basis=basis)
    kappa_qubits = tuple(range(code.num_qudits, code.num_qudits + len(C0)))
    return GadgetLayout(
        code=code, x=x, V0=V0, C0=C0, F=F, G=G,
        HX_merged=HX_m, HZ_merged=HZ_m, kappa_qubits=kappa_qubits,
        basis=basis,
    )

# After
    support, C0, F = _step1_restriction(code, x, basis=basis)
    G = _step2_gauge_fix(F)
    HX_m, HZ_m = _step3_assemble(code, support, C0, F, G, basis=basis)
    kappa_qubits = tuple(range(code.num_qudits, code.num_qudits + len(C0)))
    return GadgetLayout(
        code=code, x=x, support=support, C0=C0, F=F, G=G,
        HX_merged=HX_m, HZ_merged=HZ_m, kappa_qubits=kappa_qubits,
        basis=basis,
    )
```

Apply the same pattern to `_step1_restriction` (`return V0, C0, F.astype(np.uint8)` → `return support, C0, F.astype(np.uint8)`), `_step3_assemble` (signature `V0` → `support`, body `nV = len(V0)` → `nV = len(support)`, `v0_arr = np.asarray(V0, ...)` → `support_arr = np.asarray(support, ...)`), `build_gadget_augmented` (same pattern).

Rename the local `v0_arr` → `support_arr` and `v0_indices` (parameter of `_assemble_HX_L1`, line 78) → `support_indices` in gadget.py.

- [ ] **Step 1.3: Rename `v0_indices` parameter and update `_assemble_HX_L1` body**

In `_assemble_HX_L1` (lines 76–103):

```python
# Before
def _assemble_HX_L1(
    HX_data: np.ndarray,
    v0_indices: np.ndarray,
    F: np.ndarray,
) -> np.ndarray:
    ...
    n_v0, n_c0 = int(F.shape[1]), int(F.shape[0])
    ...
    bot = np.zeros((n_v0, n_merged), dtype=np.uint8)
    bot[np.arange(n_v0), np.asarray(v0_indices)] = 1

# After
def _assemble_HX_L1(
    HX_data: np.ndarray,
    support_indices: np.ndarray,
    F: np.ndarray,
) -> np.ndarray:
    ...
    n_v0, n_c0 = int(F.shape[1]), int(F.shape[0])
    ...
    bot = np.zeros((n_v0, n_merged), dtype=np.uint8)
    bot[np.arange(n_v0), np.asarray(support_indices)] = 1
```

Note: `n_v0` (the local count variable) stays as-is in this task — it's not a Cain-named entity; we touch it in Task 8 if needed.

- [ ] **Step 1.4: Update bridge.py references**

In `src/qldpc/circuits/surgery/bridge.py` lines 363, 364, 389, 390:

```python
# Before
    port_l_all = tuple(port_subset_l) if port_subset_l is not None else tuple(range(len(g_l.V0)))
    port_r_all = tuple(port_subset_r) if port_subset_r is not None else tuple(range(len(g_r.V0)))
    ...
    extra_kappa_l = _edges_to_F_extra(extras_l_edges, len(g_l.V0))
    extra_kappa_r = _edges_to_F_extra(extras_r_edges, len(g_r.V0))

# After
    port_l_all = tuple(port_subset_l) if port_subset_l is not None else tuple(range(len(g_l.support)))
    port_r_all = tuple(port_subset_r) if port_subset_r is not None else tuple(range(len(g_r.support)))
    ...
    extra_kappa_l = _edges_to_F_extra(extras_l_edges, len(g_l.support))
    extra_kappa_r = _edges_to_F_extra(extras_r_edges, len(g_r.support))
```

- [ ] **Step 1.5: Update circuit.py references**

In `src/qldpc/circuits/surgery/circuit.py` — replace every `.V0` attribute access with `.support`. There are 9 occurrences at lines 165, 174, 179, 282, 291, 374, 384, 436, 444, 445, 446, 447, 453, 518, 524, 525, 526, 527, 533, 673, 676, 713, 714. Examples:

```python
# Before (line 165)
    chi_l = len(g_l.V0)
# After
    chi_l = len(g_l.support)
```

```python
# Before (line 384)
        v0_indices=gadget.V0,
# After
        v0_indices=gadget.support,
```

Note: `v0_indices` here is the keyword arg name of `_surgery_observable`. Rename it to `support_indices` in this same task — both the function-signature (line 1017) and the caller (lines 384, 725). Inside the function body it appears at line 1049 (`for i in v0_indices`).

```python
# Before (line 1014–1051)
def _surgery_observable(
    gadget: GadgetLayout,
    *,
    chi_check_ids: tuple[int, ...],
    data_ids: tuple[int, ...],
    v0_indices: tuple[int, ...],
    ...
):
    ...
    data_targets = [
        measurement_record.get_target_rec(data_ids[i]) for i in v0_indices
    ]
# After
def _surgery_observable(
    gadget: GadgetLayout,
    *,
    chi_check_ids: tuple[int, ...],
    data_ids: tuple[int, ...],
    support_indices: tuple[int, ...],
    ...
):
    ...
    data_targets = [
        measurement_record.get_target_rec(data_ids[i]) for i in support_indices
    ]
```

Also update the `v0_combined` local (lines 673, 676, 725) which is a stitched support across the joint:

```python
# Before
        v0_combined = tuple(g_l.V0) + tuple(n_l + i for i in g_r.V0)
# After
        support_combined = tuple(g_l.support) + tuple(n_l + i for i in g_r.support)
```

And the call site `v0_indices=v0_combined` → `support_indices=support_combined`.

- [ ] **Step 1.6: Update `_test_gadget.py` references**

In `src/qldpc/circuits/surgery/_test_gadget.py`:

Line 29 (`fields == {...}` assertion):

```python
# Before
    assert fields == {
        "code", "x", "V0", "C0", "F", "G",
        "HX_merged", "HZ_merged", "kappa_qubits", "basis",
    }
# After
    assert fields == {
        "code", "x", "support", "C0", "F", "G",
        "HX_merged", "HZ_merged", "kappa_qubits", "basis",
    }
```

Line 34–36 (`GadgetLayout(...)` constructor):

```python
# Before
    inst = GadgetLayout(
        code=None, x=None, V0=(), C0=(),
        F=None, G=None, HX_merged=None, HZ_merged=None,
        kappa_qubits=(), basis=Pauli.X,
    )
# After
    inst = GadgetLayout(
        code=None, x=None, support=(), C0=(),
        F=None, G=None, HX_merged=None, HZ_merged=None,
        kappa_qubits=(), basis=Pauli.X,
    )
```

Every `V0` local-variable in test bodies (lines 46–48, 56, 60, 79, 100–107, 119, 124, 174–181, 220, 379, 393–396) — rename to `support`. And every `g.V0` → `g.support`. Use Edit with each chunk pulled from the existing file.

- [ ] **Step 1.7: Update `_test_bridge.py` references**

Lines 228, 299:

```python
# Before
    assert bridge.width == min(len(g_l.V0), len(g_r.V0))
    ...
    row_weights = np.asarray(g_l.F.sum(axis=1)).ravel().astype(int).tolist()
# After
    assert bridge.width == min(len(g_l.support), len(g_r.support))
    ...
    row_weights = np.asarray(g_l.F.sum(axis=1)).ravel().astype(int).tolist()  # .F left for Task 4
```

- [ ] **Step 1.8: Update `_test_circuit.py` references**

Line 260, 271:

```python
# Before
    chi_check_ids = tuple(range(100, 100 + len(g.V0)))
    ...
    circuit = _surgery_observable(
        g, chi_check_ids=chi_check_ids, data_ids=data_ids,
        v0_indices=g.V0, num_rounds=2, measurement_record=meas_rec,
    )
# After
    chi_check_ids = tuple(range(100, 100 + len(g.support)))
    ...
    circuit = _surgery_observable(
        g, chi_check_ids=chi_check_ids, data_ids=data_ids,
        support_indices=g.support, num_rounds=2, measurement_record=meas_rec,
    )
```

- [ ] **Step 1.9: Update `_test_cheeger.py` references**

Line 133:

```python
# Before
    n_chi = len(boosted.V0)
# After
    n_chi = len(boosted.support)
```

- [ ] **Step 1.10: Run tests**

Run: `uv run pytest src/qldpc/circuits/surgery/ -q 2>&1 | tail -3`
Expected: `147 passed`.

- [ ] **Step 1.11: Sanity-grep**

Run: `git grep -nE '\.V0\b' src/qldpc/circuits/surgery/`
Expected: zero hits.

Run: `git grep -nE 'v0_indices|v0_combined|v0_arr' src/qldpc/circuits/surgery/`
Expected: zero hits.

- [ ] **Step 1.12: Commit**

```bash
git add src/qldpc/circuits/surgery/
git commit -m "$(cat <<'EOF'
refactor(surgery): rename V0 → support (Cain convention)

GadgetLayout.V0 / v0_indices / v0_combined / v0_arr → support /
support_indices / support_combined / support_arr. Math unchanged;
attribute and parameter rename only.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: C0 → data_checks

**Files:**
- Modify: `src/qldpc/circuits/surgery/gadget.py`
- Modify: `src/qldpc/circuits/surgery/_test_gadget.py`

(C0 only leaks to attribute callers — bridge.py, cheeger.py, circuit.py do NOT reference `.C0`. Verified by `grep "\.C0" src/qldpc/circuits/surgery/`.)

- [ ] **Step 2.1: Rename `GadgetLayout` field `C0` → `data_checks`**

In `src/qldpc/circuits/surgery/gadget.py` line 27:

```python
# Before
    C0: tuple[int, ...]
# After
    data_checks: tuple[int, ...]
```

- [ ] **Step 2.2: Rename local `C0` everywhere in gadget.py**

In `_step1_restriction` (lines 38–62), `_step3_assemble` (lines 106–154 — parameter `C0` and body `nV, nC = len(support), len(C0)` and the `for k, j in enumerate(C0):` loop), `build_gadget` (lines 177–185), `build_gadget_augmented` (lines 210–237 — including `C0_aug`):

Every local `C0` → `data_checks`, every `C0_aug` → `data_checks_aug`.

Example diff for `_step1_restriction`:

```python
# Before
    C0 = tuple(
        int(j) for j in range(H_complement.shape[0]) if H_complement[j, list(support)].any()
    )
    F = (
        H_complement[np.ix_(C0, support)]
        if C0 and support
        else np.zeros((len(C0), len(support)), dtype=np.uint8)
    )
    return support, C0, F.astype(np.uint8)
# After
    data_checks = tuple(
        int(j) for j in range(H_complement.shape[0]) if H_complement[j, list(support)].any()
    )
    F = (
        H_complement[np.ix_(data_checks, support)]
        if data_checks and support
        else np.zeros((len(data_checks), len(support)), dtype=np.uint8)
    )
    return support, data_checks, F.astype(np.uint8)
```

Apply the same to `build_gadget_augmented`'s `C0_aug` → `data_checks_aug`.

- [ ] **Step 2.3: Update `_test_gadget.py` references**

Lines 29, 34 (`fields == {...}` and `GadgetLayout` constructor):

```python
# Before
        "code", "x", "support", "C0", "F", "G",
        ...
        code=None, x=None, support=(), C0=(),
# After
        "code", "x", "support", "data_checks", "F", "G",
        ...
        code=None, x=None, support=(), data_checks=(),
```

Lines 46–62, 100–107, 119–124, 174–199, 211, 221, 291–299 — every local `C0` becomes `data_checks` and every `g.C0` becomes `g.data_checks`. Pull exact lines from current file using Read.

- [ ] **Step 2.4: Run tests**

Run: `uv run pytest src/qldpc/circuits/surgery/ -q 2>&1 | tail -3`
Expected: `147 passed`.

- [ ] **Step 2.5: Sanity-grep**

Run: `git grep -nE '\.C0\b|\bC0\b' src/qldpc/circuits/surgery/`
Expected: zero hits.

- [ ] **Step 2.6: Commit**

```bash
git add src/qldpc/circuits/surgery/
git commit -m "$(cat <<'EOF'
refactor(surgery): rename C0 → data_checks (Cain convention)

GadgetLayout.C0 → data_checks. The Webster paper's C_0 = data checks
in S_X or S_Z that touch the seed support.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: kappa* → ancilla*

This is a comprehensive prefix rename: `kappa_qubits` → `ancilla_qubits`, `kappa_ids` → `ancilla_ids`, `kappa_op` → `ancilla_op`, `cl_kappa` → `cl_ancilla`, `cr_kappa` → `cr_ancilla`, `kappa_qubits_aug` → `ancilla_qubits_aug`, `HZ_kappa_block` → `HZ_ancilla_block`. (Local count variables `k_l / k_r` stay — they are scalars, not Cain-named entities; we keep their `k_` prefix.)

**Files:**
- Modify: `src/qldpc/circuits/surgery/gadget.py`
- Modify: `src/qldpc/circuits/surgery/circuit.py`
- Modify: `src/qldpc/circuits/surgery/cheeger.py` (only docstring mention)
- Modify: `src/qldpc/circuits/surgery/_test_gadget.py`
- Modify: `src/qldpc/circuits/surgery/_test_circuit.py`

- [ ] **Step 3.1: Rename `GadgetLayout` field `kappa_qubits` → `ancilla_qubits`**

In `src/qldpc/circuits/surgery/gadget.py` line 32:

```python
# Before
    kappa_qubits: tuple[int, ...]
# After
    ancilla_qubits: tuple[int, ...]
```

In `build_gadget` (line 180): `kappa_qubits = tuple(range(...))` → `ancilla_qubits = tuple(range(...))` and the `kappa_qubits=kappa_qubits` constructor keyword → `ancilla_qubits=ancilla_qubits`.

In `build_gadget_augmented` (line 232–235): `kappa_qubits_aug` → `ancilla_qubits_aug` and the constructor keyword.

- [ ] **Step 3.2: Update circuit.py references**

In `src/qldpc/circuits/surgery/circuit.py`:

Line 135 (lane y=1 docstring): `κ ancillas` → `ancilla qubits (Q')`. Line 137: leave the `χ ancillas` mention untouched (will be addressed in a later task if at all — `χ ancillas` is prose).

Line 167 (`k_l = len(g_l.kappa_qubits)`) → `k_l = len(g_l.ancilla_qubits)`.

Lines 354, 359, 366 (`kappa_ids = qubit_ids.data[n_data:]` and call sites): rename to `ancilla_ids`. Example:

```python
# Before (lines 354–366)
    n_data = gadget.code.num_qudits
    data_ids = qubit_ids.data[:n_data]
    kappa_ids = qubit_ids.data[n_data:]
    bridge_ids: tuple[int, ...] = ()

    circuit = _surgery_qubit_coordinates(gadget, qubit_ids)
    circuit += _surgery_state_prep(
        gadget, data_ids, kappa_ids, bridge_ids, data_init=data_init,
    )
    ...
    circuit += _surgery_detach_and_readout(
        gadget, data_ids=data_ids, kappa_ids=kappa_ids, bridge_ids=bridge_ids,
        ...
    )
# After
    n_data = gadget.code.num_qudits
    data_ids = qubit_ids.data[:n_data]
    ancilla_ids = qubit_ids.data[n_data:]
    bridge_ids: tuple[int, ...] = ()

    circuit = _surgery_qubit_coordinates(gadget, qubit_ids)
    circuit += _surgery_state_prep(
        gadget, data_ids, ancilla_ids, bridge_ids, data_init=data_init,
    )
    ...
    circuit += _surgery_detach_and_readout(
        gadget, data_ids=data_ids, ancilla_ids=ancilla_ids, bridge_ids=bridge_ids,
        ...
    )
```

Lines 430–431, 512–513 (`cl_kappa`, `cr_kappa` slice locals in `_stitch_intercode` / `_stitch_intracode`): rename to `cl_ancilla`, `cr_ancilla`. The downstream `M_chi[..., cl_kappa] = ...` lines (445, 447, 461, 463, 465, 466, 468, 469, 525, 527, 542, 543, 544, 545, 547, 548) all become `cl_ancilla` / `cr_ancilla`.

Lines 677, 694 (`kappa_ids` in `build_joint_ppm_circuit`): rename to `ancilla_ids`.

Line 887–960 — `_surgery_state_prep` signature, `kappa_ids` parameter → `ancilla_ids`; body `anc_ids = list(kappa_ids) + ...` → `anc_ids = list(ancilla_ids) + ...`.

Line 1097–1116 — `_surgery_detach_and_readout` signature, `kappa_ids: tuple[int, ...]` parameter → `ancilla_ids: tuple[int, ...]`; body `detach_qubits = list(kappa_ids) + ...` → `list(ancilla_ids) + ...`; `kappa_op = "M" ...` → `ancilla_op = "M" ...`; `circuit.append(kappa_op, detach_qubits)` → `circuit.append(ancilla_op, detach_qubits)`.

- [ ] **Step 3.3: Update cheeger.py docstring**

In `src/qldpc/circuits/surgery/cheeger.py` line 428 (in `boost_gadget` docstring):

```python
# Before
        kappa_qubits.
# After
        ancilla_qubits.
```

- [ ] **Step 3.4: Update `_test_gadget.py` references**

Line 30: `"kappa_qubits"` → `"ancilla_qubits"` in `fields == {...}` set.
Line 36: `kappa_qubits=()` → `ancilla_qubits=()` in GadgetLayout constructor.
Lines 142–143, 190 (comments mentioning "kappa ancilla"): leave as prose (Cain says "ancilla"; replace "kappa ancilla" with "ancilla" — drop the redundant "kappa").
Line 193, 195, 197: `HZ_kappa_block` → `HZ_ancilla_block`.
Line 211: `g.kappa_qubits == tuple(range(...))` → `g.ancilla_qubits == tuple(range(...))`. Also update `len(g.C0)` → `len(g.data_checks)` here (it's already renamed by Task 2 if Task 2 ran first; if processing this task in isolation, confirm).
Line 226: `g1.kappa_qubits == g2.kappa_qubits` → `g1.ancilla_qubits == g2.ancilla_qubits`.
Line 266, 378: `kappa = len(g1.kappa_qubits)` → keep `kappa` local for now (this is the Webster Table I counter, renamed in Task 8). Just change attribute access: `len(g.kappa_qubits)` → `len(g.ancilla_qubits)`.

- [ ] **Step 3.5: Update `_test_circuit.py` references**

Lines 137, 138, 141, 163, 164, 167, 213, 214, 217, 218, 222, 237, 238, 241, 245 — `kappa_ids` local → `ancilla_ids`, and the assertion strings `f"R {' '.join(str(q) for q in kappa_ids)}"` etc. → `f"R {' '.join(str(q) for q in ancilla_ids)}"`.

Lines 213, 238 (`len(g.kappa_qubits)`) → `len(g.ancilla_qubits)`.

- [ ] **Step 3.6: Run tests**

Run: `uv run pytest src/qldpc/circuits/surgery/ -q 2>&1 | tail -3`
Expected: `147 passed`.

- [ ] **Step 3.7: Sanity-grep**

Run: `git grep -nE 'kappa_(qubits|ids|op|block)|cl_kappa|cr_kappa|HZ_kappa' src/qldpc/circuits/surgery/`
Expected: zero hits (extra_kappa_l/r still present — those rename in Task 6).

- [ ] **Step 3.8: Commit**

```bash
git add src/qldpc/circuits/surgery/
git commit -m "$(cat <<'EOF'
refactor(surgery): rename kappa_* → ancilla_* (Cain convention)

GadgetLayout.kappa_qubits → ancilla_qubits; kappa_ids parameter → ancilla_ids;
cl_kappa / cr_kappa slice locals → cl_ancilla / cr_ancilla; kappa_op
selector → ancilla_op; HZ_kappa_block → HZ_ancilla_block. The Cain
ancilla system A(Q', S'_X, S'_Z) Q' is what we now consistently call
"ancilla qubits". extra_kappa_l/r in Bridge dataclass renamed in Task 6.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: F → incidence

This rename touches every file. `\.F\b` is the safe regex (doesn't match `.F_CONTIGUOUS` etc.).

**Files:**
- Modify: `src/qldpc/circuits/surgery/gadget.py`
- Modify: `src/qldpc/circuits/surgery/bridge.py`
- Modify: `src/qldpc/circuits/surgery/cheeger.py`
- Modify: `src/qldpc/circuits/surgery/circuit.py`
- Modify: `src/qldpc/circuits/surgery/_test_gadget.py`
- Modify: `src/qldpc/circuits/surgery/_test_bridge.py`
- Modify: `src/qldpc/circuits/surgery/_test_cheeger.py`

- [ ] **Step 4.1: Rename `GadgetLayout` field `F` → `incidence`**

In `src/qldpc/circuits/surgery/gadget.py` line 28:

```python
# Before
    F: np.ndarray
# After
    incidence: np.ndarray
```

- [ ] **Step 4.2: Rename `F` everywhere in gadget.py**

Every `F` local variable, parameter, and `F_aug` / `F_extra` / `F_tilde` becomes `incidence` / `incidence_aug` / `incidence_extra` / `incidence_tilde`. Examples:

`_step1_restriction` return: `return support, data_checks, F.astype(np.uint8)` → `return support, data_checks, incidence.astype(np.uint8)`. (And rename local `F = H_complement[...]` → `incidence = H_complement[...]`.)

`_step2_gauge_fix(F: np.ndarray)` → `_step2_gauge_fix(incidence: np.ndarray)`. Inside: every `F` → `incidence`. The local `G = GF2(F.astype(...))...` becomes `G = GF2(incidence.astype(...))...`.

`_step3_assemble(code, support, data_checks, F, G, ...)` → `_step3_assemble(code, support, data_checks, incidence, gauge, ...)` — note we also rename `G` → `gauge` here. (Combined with Task 5; for review clarity, you may split Task 4 and Task 5; doing them together is simpler.) Actually wait — the spec assigns Task 5 to G; keep these separate. So in Task 4 we only rename `F` → `incidence` and the parameter `F: np.ndarray` of `_step3_assemble` becomes `incidence: np.ndarray` while `G` stays.

`_assemble_HX_L1(HX_data, support_indices, F)` → `(HX_data, support_indices, incidence)`. Body: `n_v0, n_c0 = int(F.shape[1]), int(F.shape[0])` → `int(incidence.shape[1]), int(incidence.shape[0])`. `bot[:, n:] = F.T` → `bot[:, n:] = incidence.T`.

`build_gadget` body line 177: `support, data_checks, F = _step1_restriction(...)` → `support, data_checks, incidence = _step1_restriction(...)`. `G = _step2_gauge_fix(F)` → `G = _step2_gauge_fix(incidence)`. The GadgetLayout constructor: `F=F` → `incidence=incidence`.

`build_gadget_augmented`: rename `F_extra` parameter to `incidence_extra`; rename `F` (output of `_step1_restriction`) → `incidence`; `F_aug = np.vstack([F, F_extra])` → `incidence_aug = np.vstack([incidence, incidence_extra])`; `G_aug = _step2_gauge_fix(F_aug)` → `G_aug = _step2_gauge_fix(incidence_aug)`; constructor `F=F_aug` → `incidence=incidence_aug`. The error message `"F_extra has {F_extra.shape[1]} columns; expected {len(support)} (= |V_0|)"` → `"incidence_extra has {incidence_extra.shape[1]} columns; expected {len(support)} (= |support|)"`.

Lastly: the `F_tilde` matrix inside `_step3_assemble` (lines 127–135) is NOT the incidence matrix — it's a SELECTION matrix. Per Cain naming, this is `selector_tilde` or just leave as `F_tilde` since it's an internal computation aid. Decision: rename `F_tilde` → `incidence_tilde` for naming consistency.

- [ ] **Step 4.3: Update bridge.py references**

Replace every `.F` attribute access in bridge.py with `.incidence`. Specifically lines 359, 360, 401, 404. The local `F_aug` references in `_run_skiptree_on_port_subgraph` (parameter and body, lines 282–331): rename parameter `F_aug` → `incidence_aug` and update all references including `F_aug.shape`, `F_aug[r]`.

The local helper function `_edges_to_F_extra` (line 268) — function name itself contains `F_extra`. Rename function name to `_edges_to_incidence_extra` and update callers in `build_bridge` (lines 389, 390).

The internal variable in `_build_aux_graph_strict` has a parameter `F` (line 200): `def _build_aux_graph_strict(F: np.ndarray) -> ...`. Rename to `incidence`. The function name `_build_aux_graph_strict` keeps "F"-agnostic naming; no change. But the body's `F_arr = np.asarray(F).astype(int)` → `incidence_arr = np.asarray(incidence).astype(int)` and downstream `F_arr.shape[1]`, etc.

- [ ] **Step 4.4: Update cheeger.py references**

In `src/qldpc/circuits/surgery/cheeger.py`:

`_exact_boundary_cheeger(F: galois.FieldArray)` → `(incidence: galois.FieldArray)`. Body: `F_arr` → `incidence_arr`, `F_col_ints` → `incidence_col_ints`.

`_spectral_cheeger_lower_bound(F: galois.FieldArray)` → `(incidence: galois.FieldArray)`. Body: `F_float = np.asarray(F).astype(np.float64)` → `incidence_float = np.asarray(incidence).astype(np.float64)`.

`cheeger_constant(g)` body: `F = galois.GF(2)(np.asarray(g.F).astype(int))` → `incidence = galois.GF(2)(np.asarray(g.incidence).astype(int))`. The downstream `F.shape[1]` etc. → `incidence.shape[1]`.

`_augment_F_with_random_edges(F_base, n_new_edges, rng)` → `_augment_incidence_with_random_edges(incidence_base, n_new_edges, rng)`. Inside: every `F` → `incidence`. The callers in `boost_gadget_distance` (line 389) → `_augment_incidence_with_random_edges`.

`boost_gadget_cheeger_combinatorial` body lines 219–315: every `F = np.asarray(g.F)...` → `incidence = np.asarray(g.incidence)...`; `F_col_ints` → `incidence_col_ints`; `F = np.vstack([F, new_row])` → `incidence = np.vstack([incidence, new_row])`; the return `F_extra = F[n_orig_rows:]...` → `incidence_extra = incidence[n_orig_rows:]...`; `build_gadget_augmented(g.code, g.x, F_extra, ...)` → `build_gadget_augmented(g.code, g.x, incidence_extra, ...)`.

`boost_gadget_distance` body lines 364–402: `F_base = np.asarray(g.F)...` → `incidence_base = np.asarray(g.incidence)...`. `bare = build_gadget_augmented(g.code, g.x, np.zeros((0, n_V), dtype=np.uint8), basis=g.basis)` — the `n_V = F_base.shape[1]` line → `n_V = incidence_base.shape[1]`. The `F_extra = _augment_F_with_random_edges(F_base, n_extra, rng)` → `incidence_extra = _augment_incidence_with_random_edges(incidence_base, n_extra, rng)`. The next two lines `F_extra_rows = np.asarray(F_extra[F_base.shape[0]:])...` → `incidence_extra_rows = np.asarray(incidence_extra[incidence_base.shape[0]:])...` and `build_gadget_augmented(g.code, g.x, F_extra_rows, ...)` → `build_gadget_augmented(g.code, g.x, incidence_extra_rows, ...)`.

The docstring "Returns: A NEW GadgetLayout with boosted F, G, HX_merged, HZ_merged, ..." → "boosted incidence, gauge, HX_merged, HZ_merged, ancilla_qubits."

- [ ] **Step 4.5: Update circuit.py references**

In `src/qldpc/circuits/surgery/circuit.py`:

Lines 190, 191: `k_l = bridge.g_l_aug.F.shape[0]` → `k_l = bridge.g_l_aug.incidence.shape[0]`; same for `g_r_aug`.
Lines 423, 426: `k_l, k_r = g_l_aug.F.shape[0], g_r_aug.F.shape[0]` → `k_l, k_r = g_l_aug.incidence.shape[0], g_r_aug.incidence.shape[0]`; `r_l, r_r = g_l_aug.G.shape[0], g_r_aug.G.shape[0]` stays (G → gauge in Task 5).
Lines 506, 509: same pattern for `_stitch_intracode`.
Lines 667, 668: `k_l = g_l_aug.F.shape[0]` etc. → `g_l_aug.incidence.shape[0]`.

- [ ] **Step 4.6: Update test files**

In `_test_gadget.py`: lines 29, 35 (`"F"` in fields set and `F=None` in constructor) → `"incidence"` and `incidence=None`. Lines 46–62, 100–107, 121, 174–199, 222, 269, 380, 393–404 — every local `F`, every `g.F`, every `g1.F`, `g2.F`, `g_aug.F` → corresponding `incidence` form. Also `extra_F` (local variable in `test_build_gadget_augmented_extends_F_and_recomputes_G`, line 394) → `extra_incidence` (it's the F_extra fed into `build_gadget_augmented`). Also the test name `test_build_gadget_augmented_extends_F_and_recomputes_G` → `test_build_gadget_augmented_extends_incidence_and_recomputes_gauge` (rename Webster letters to Cain in test names).

Lines 412–483 (`test_step2_gauge_fix_rows_linearly_independent`): `F_mat = g.F.astype(np.uint8)` → `incidence_mat = g.incidence.astype(np.uint8)`. The docstring mentioning "F" → "incidence".

In `_test_bridge.py`: lines 232, 233, 252, 299, 359, 401, 404. Examples:

```python
# Before line 232–233
    assert bridge.T_l.shape == (bridge.width - 1, bridge.g_l_aug.F.shape[0])
    assert bridge.T_r.shape == (bridge.width - 1, bridge.g_r_aug.F.shape[0])
# After
    assert bridge.T_l.shape == (bridge.width - 1, bridge.g_l_aug.incidence.shape[0])
    assert bridge.T_r.shape == (bridge.width - 1, bridge.g_r_aug.incidence.shape[0])
```

In `_test_cheeger.py`: lines 32, 60 — every `.F` → `.incidence`.

- [ ] **Step 4.7: Run tests**

Run: `uv run pytest src/qldpc/circuits/surgery/ -q 2>&1 | tail -3`
Expected: `147 passed`.

- [ ] **Step 4.8: Sanity-grep**

Run: `git grep -nE '\.F\b' src/qldpc/circuits/surgery/`
Expected: zero hits (modulo possible Webster-citation comments like `F = H_complement[C_0, V_0]` quoting paper math notation — those should remain, but be sure they are clearly inside `# Webster ...` comments).

Run: `git grep -nE 'F_aug|F_extra|F_tilde|F_arr|F_col_ints|F_base|F_float' src/qldpc/circuits/surgery/`
Expected: zero hits.

- [ ] **Step 4.9: Commit**

```bash
git add src/qldpc/circuits/surgery/
git commit -m "$(cat <<'EOF'
refactor(surgery): rename F → incidence (Cain convention)

GadgetLayout.F / F_aug / F_extra / F_tilde / F_base / F_arr / F_col_ints
/ F_float and the local F variables in all surgery files become incidence
/ incidence_aug / incidence_extra / incidence_tilde / etc. The cheeger
helper _augment_F_with_random_edges is also renamed to
_augment_incidence_with_random_edges; _edges_to_F_extra in bridge.py is
renamed to _edges_to_incidence_extra. Matrix shape and semantics unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: G → gauge

`\.G\b` regex is safe (doesn't match `.GF` or `.GG...`).

**Files:**
- Modify: `src/qldpc/circuits/surgery/gadget.py`
- Modify: `src/qldpc/circuits/surgery/bridge.py`
- Modify: `src/qldpc/circuits/surgery/cheeger.py`
- Modify: `src/qldpc/circuits/surgery/circuit.py`
- Modify: `src/qldpc/circuits/surgery/_test_gadget.py`
- Modify: `src/qldpc/circuits/surgery/_test_bridge.py`
- Modify: `src/qldpc/circuits/surgery/_test_circuit.py`

- [ ] **Step 5.1: Rename `GadgetLayout` field `G` → `gauge`**

In `src/qldpc/circuits/surgery/gadget.py` line 29:

```python
# Before
    G: np.ndarray
# After
    gauge: np.ndarray
```

- [ ] **Step 5.2: Rename `G` everywhere in gadget.py**

`_step2_gauge_fix`: return value `G = GF2(...).left_null_space()` → `gauge = GF2(...).left_null_space()`; `return np.asarray(G).astype(np.uint8)` → `return np.asarray(gauge).astype(np.uint8)`.

`_step3_assemble(code, support, data_checks, incidence, G, ...)` → `(code, support, data_checks, incidence, gauge, ...)`. Body: `r = G.shape[0]` → `r = gauge.shape[0]`; `G.astype(np.uint8)` in `np.block([[...]])` → `gauge.astype(np.uint8)`.

`build_gadget` body: `G = _step2_gauge_fix(incidence)` → `gauge = _step2_gauge_fix(incidence)`; `_step3_assemble(code, support, data_checks, incidence, G, ...)` → `(code, support, data_checks, incidence, gauge, ...)`; constructor `G=G` → `gauge=gauge`.

`build_gadget_augmented` body: `G_aug = _step2_gauge_fix(incidence_aug)` → `gauge_aug = _step2_gauge_fix(incidence_aug)`; the `_step3_assemble(code, support, data_checks_aug, incidence_aug, G_aug, basis=basis)` → `(... , gauge_aug, basis=basis)`; constructor `G=G_aug` → `gauge=gauge_aug`.

- [ ] **Step 5.3: Update circuit.py references**

Lines 166, 175, 180, 283, 292, 426, 509: every `g.G`, `g_l.G`, `g_r.G`, `g_l_aug.G`, `g_r_aug.G` → corresponding `.gauge`. Examples:

```python
# Before (line 166)
    G_l = g_l.G.shape[0]
# After
    G_l = g_l.gauge.shape[0]
```

Local variables `G_l`, `G_r`, `G_total` (lines 166, 175, 180, 216, 228, 238, 283, 292, 426, 509): these are integer counts and should be renamed to `n_gauge_l`, `n_gauge_r`, `n_gauge_total` for consistency with Cain. Apply this rename in the same task. Update all downstream uses (lines 216, 228, 238 in the X-check ancilla loop; line 283, 292 in `_check_lane_index_map`; line 426 in stitch; line 509 in intracode stitch).

- [ ] **Step 5.4: Update cheeger.py references**

Line 427 docstring (already updated in Task 4 to mention "gauge"): verify.

- [ ] **Step 5.5: Update bridge.py references**

Line 252: `G_aug = g_aug.F.astype(np.int_)` — this is a LOCAL variable named G_aug but the comment says "G_aug = F_aug (incidence: rows = edges = κ qubits, cols = V_0 vertices)". So `G_aug` is being used as a different symbol (graph-theoretic adjacency, not Webster's G). Rename to `adjacency` and update the comment to reflect the Cain naming:

```python
# Before
        # G_aug = F_aug (incidence: rows = edges = κ qubits, cols = V_0 vertices)
        G_aug = g_aug.F.astype(np.int_)
# After
        # adjacency = incidence_aug (rows = edges = ancilla qubits, cols = support vertices)
        adjacency = g_aug.incidence.astype(np.int_)
```

The downstream `lhs = (T @ G_aug @ P) % 2` → `lhs = (T @ adjacency @ P) % 2`.

(Note: `G_aux` — the networkx graph variable in bridge.py — is NOT Webster's G; it's "G_auxiliary". Leave as-is.)

- [ ] **Step 5.6: Update test files**

`_test_gadget.py`:
- Line 29: `"G"` → `"gauge"` in fields set.
- Line 35: `G=None` → `gauge=None`.
- Lines 65–90 (`test_step2_gauge_fix_basis_property` and `test_step2_gauge_fix_deterministic`): every local `G = _step2_gauge_fix(F)` (already `incidence` after Task 4) → `gauge = _step2_gauge_fix(incidence)`. The downstream `G.shape[1]` → `gauge.shape[1]`; `(G @ F)` → `(gauge @ incidence)`; etc.
- Lines 100–125: same pattern (build, body, assertions).
- Lines 175–199: same.
- Line 223: `np.array_equal(g1.G, g2.G)` → `np.array_equal(g1.gauge, g2.gauge)`.
- Lines 268, 380: `r = g1.G.shape[0]` / `r = g.G.shape[0]` → `r = g1.gauge.shape[0]` / `r = g.gauge.shape[0]`. (Local `r` count stays for now — renamed in Task 8.)
- Lines 412–483 (`test_step2_gauge_fix_rows_linearly_independent`): `G = g.G` → `gauge = g.gauge`; rest of body's `G.shape[0]`, `G.astype(np.uint8)` → `gauge.shape[0]`, `gauge.astype(np.uint8)`.

`_test_bridge.py`: line 252 `G_aug = g_aug.F.astype(np.int_)` — see Step 5.5 (already handled).

`_test_circuit.py`: lines 81, 110 `r = g.G.shape[0]` → `r = g.gauge.shape[0]`.

- [ ] **Step 5.7: Run tests**

Run: `uv run pytest src/qldpc/circuits/surgery/ -q 2>&1 | tail -3`
Expected: `147 passed`.

- [ ] **Step 5.8: Sanity-grep**

Run: `git grep -nE '\.G\b' src/qldpc/circuits/surgery/`
Expected: zero hits (excluding `G_aux` networkx variable, which lacks the `.G` form).

Run: `git grep -nE '\bG_aug\b|\bG_l\b|\bG_r\b|\bG_total\b' src/qldpc/circuits/surgery/`
Expected: zero hits (renamed to `gauge_aug`, `n_gauge_l`, `n_gauge_r`, `n_gauge_total`).

- [ ] **Step 5.9: Commit**

```bash
git add src/qldpc/circuits/surgery/
git commit -m "$(cat <<'EOF'
refactor(surgery): rename G → gauge (Cain convention)

GadgetLayout.G → gauge. Local count variables G_l / G_r / G_total
become n_gauge_l / n_gauge_r / n_gauge_total. Bridge test's G_aug
(graph-theoretic adjacency, not Webster G) renamed to adjacency.
networkx auxiliary graph G_aux retains its name.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: extra_kappa_l / extra_kappa_r → extra_ancilla_l / extra_ancilla_r

**Files:**
- Modify: `src/qldpc/circuits/surgery/bridge.py`
- Modify: `src/qldpc/circuits/surgery/_test_bridge.py`

- [ ] **Step 6.1: Rename `Bridge` dataclass fields**

In `src/qldpc/circuits/surgery/bridge.py` lines 31–32:

```python
# Before
    extra_kappa_l: np.ndarray                   # (e_l, |V_0^(l)|) F_2; weight-2 rows added
    extra_kappa_r: np.ndarray
# After
    extra_ancilla_l: np.ndarray                 # (e_l, |support^(l)|) F_2; weight-2 rows added
    extra_ancilla_r: np.ndarray
```

- [ ] **Step 6.2: Rename local variables and constructor kwargs in `build_bridge`**

Lines 389, 390, 414, 415:

```python
# Before
    extra_kappa_l = _edges_to_incidence_extra(extras_l_edges, len(g_l.support))
    extra_kappa_r = _edges_to_incidence_extra(extras_r_edges, len(g_r.support))
    ...
    return Bridge(
        ...
        extra_kappa_l=extra_kappa_l.astype(np.uint8),
        extra_kappa_r=extra_kappa_r.astype(np.uint8),
        ...
    )
# After
    extra_ancilla_l = _edges_to_incidence_extra(extras_l_edges, len(g_l.support))
    extra_ancilla_r = _edges_to_incidence_extra(extras_r_edges, len(g_r.support))
    ...
    return Bridge(
        ...
        extra_ancilla_l=extra_ancilla_l.astype(np.uint8),
        extra_ancilla_r=extra_ancilla_r.astype(np.uint8),
        ...
    )
```

Also lines 396, 397 (consumer):

```python
# Before
    g_l_aug = build_gadget_augmented(g_l.code, g_l.x, extra_kappa_l, basis=basis)
    g_r_aug = build_gadget_augmented(g_r.code, g_r.x, extra_kappa_r, basis=basis)
# After
    g_l_aug = build_gadget_augmented(g_l.code, g_l.x, extra_ancilla_l, basis=basis)
    g_r_aug = build_gadget_augmented(g_r.code, g_r.x, extra_ancilla_r, basis=basis)
```

- [ ] **Step 6.3: Update _test_bridge.py field-set assertion**

Lines 211–212:

```python
# Before
        "extra_kappa_l", "extra_kappa_r",
# After
        "extra_ancilla_l", "extra_ancilla_r",
```

- [ ] **Step 6.4: Run tests**

Run: `uv run pytest src/qldpc/circuits/surgery/ -q 2>&1 | tail -3`
Expected: `147 passed`.

- [ ] **Step 6.5: Sanity-grep**

Run: `git grep -nE 'extra_kappa' src/qldpc/circuits/surgery/`
Expected: zero hits.

- [ ] **Step 6.6: Commit**

```bash
git add src/qldpc/circuits/surgery/
git commit -m "$(cat <<'EOF'
refactor(surgery): rename extra_kappa_l/r → extra_ancilla_l/r

Bridge.extra_kappa_l / extra_kappa_r → extra_ancilla_l / extra_ancilla_r.
Each row of these matrices corresponds to a new ancilla qubit added
during cellulation, so the Cain naming is more direct.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: chi_l / chi_r / chi_total in circuit.py → n_meas_l / n_meas_r / n_meas_total

These are circuit-construction LOCAL variables in `_surgery_qubit_coordinates`, `_check_lane_index_map`, `_stitch_intercode`, `_stitch_intracode`, and `build_joint_ppm_circuit`. They count `|S'_meas|` (the new measured-basis check rows) per side and total.

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py`

- [ ] **Step 7.1: Rename in `_surgery_qubit_coordinates`**

Lines 165, 174, 179, 215–224, 243–246:

```python
# Before
    chi_l = len(g_l.support)
    G_l = g_l.gauge.shape[0]
    ...
    if joint is not None and intercode:
        ...
        chi_r = len(g_r.support)
        G_r = g_r.gauge.shape[0]
    elif joint is not None:
        ...
        chi_r = len(g_r.support)
        G_r = g_r.gauge.shape[0]
    else:
        ...
        chi_r = G_r = k_r = 0
    ...
    chi_total = chi_l + chi_r
    G_total = G_l + G_r

    for i in range(m_X_total):
        circuit.append("QUBIT_COORDS", qubit_ids.checks_x[i], (i, 2))
    if is_basis_x:
        for i in range(chi_total):
            circuit.append(...)
    ...
# After
    n_meas_l = len(g_l.support)
    n_gauge_l = g_l.gauge.shape[0]
    ...
    if joint is not None and intercode:
        ...
        n_meas_r = len(g_r.support)
        n_gauge_r = g_r.gauge.shape[0]
    elif joint is not None:
        ...
        n_meas_r = len(g_r.support)
        n_gauge_r = g_r.gauge.shape[0]
    else:
        ...
        n_meas_r = n_gauge_r = k_r = 0
    ...
    n_meas_total = n_meas_l + n_meas_r
    n_gauge_total = n_gauge_l + n_gauge_r

    for i in range(m_X_total):
        circuit.append("QUBIT_COORDS", qubit_ids.checks_x[i], (i, 2))
    if is_basis_x:
        for i in range(n_meas_total):
            circuit.append(...)
```

(Task 5 already renamed `G_l / G_r / G_total` → `n_gauge_l / n_gauge_r / n_gauge_total`; this task only handles `chi_*` → `n_meas_*`.)

- [ ] **Step 7.2: Rename in `_check_lane_index_map`**

Lines 282, 283, 291, 292, 305–314:

```python
# Before
        chi_total = len(gadget.support)
        G_total = gadget.gauge.shape[0]
    else:
        ...
        chi_total = len(gadget.support) + len(g_r.support)
        G_total = gadget.gauge.shape[0] + g_r.gauge.shape[0]
    ...
    if is_basis_x:
        for i in range(chi_total):
            result[qubit_ids.checks_x[m_X_total + i]] = (3, i)
        for i in range(G_total):
            result[qubit_ids.checks_z[m_Z_total + i]] = (5, i)
# After
        n_meas_total = len(gadget.support)
        n_gauge_total = gadget.gauge.shape[0]
    else:
        ...
        n_meas_total = len(gadget.support) + len(g_r.support)
        n_gauge_total = gadget.gauge.shape[0] + g_r.gauge.shape[0]
    ...
    if is_basis_x:
        for i in range(n_meas_total):
            result[qubit_ids.checks_x[m_X_total + i]] = (3, i)
        for i in range(n_gauge_total):
            result[qubit_ids.checks_z[m_Z_total + i]] = (5, i)
```

The cycle-check IDs lines 319–323 use `m_Z_total + G_total` — rename to `m_Z_total + n_gauge_total`.

- [ ] **Step 7.3: Rename `M_chi`-construction locals in `_stitch_intercode` and `_stitch_intracode`**

Lines 441–447, 522–533: the local arrays `chi_l_rows`, `chi_r_rows`, `chi_start` are about the `M_chi` matrix (the "measured-basis check carrier"). The name `M_chi` refers to χ-carrier. Per the spec, χ rows are the new measured-basis checks, so rename:

- `M_chi` → `M_meas` (the matrix of all measured-basis check rows across the joint code).
- `chi_l_rows` → `meas_l_rows` (the χ rows from the left gadget).
- `chi_r_rows` → `meas_r_rows`.
- `chi_start` → `meas_start`.
- `M_chi_l`, `M_chi_r`, `M_chi_l_src`, `M_chi_r_src` → `M_meas_l`, `M_meas_r`, `M_meas_l_src`, `M_meas_r_src`.
- `M_co`, `M_co_l`, `M_co_r`, `M_co_l_src`, `M_co_r_src` → `M_comp`, `M_comp_l`, `M_comp_r`, `M_comp_l_src`, `M_comp_r_src`.
- `m_chi_l_data`, `m_chi_r_data`, `m_chi_data` → `m_meas_l_data`, `m_meas_r_data`, `m_meas_data`.
- `m_co_l_data`, `m_co_r_data`, `m_co_data` → `m_comp_l_data`, `m_comp_r_data`, `m_comp_data`.

The comment at line 401 `# χ-carrier abstraction: M_chi holds χ rows; M_co holds the dual cycle rows.` → `# measured-basis abstraction: M_meas holds the new meas-basis check rows; M_comp holds the dual cycle/comp-basis rows.`

Apply identical renames in `_stitch_intracode` (lines 477–553).

- [ ] **Step 7.4: Rename chi locals in `build_joint_ppm_circuit`**

Lines 715–719:

```python
# Before
    chi_l_offset = m_l + m_r
    chi_r_offset = chi_l_offset + n_V_l
    chi_l_ids = tuple(check_ids[chi_l_offset : chi_l_offset + n_V_l])
    chi_r_ids = tuple(check_ids[chi_r_offset : chi_r_offset + n_V_r])
    chi_check_ids = chi_l_ids + chi_r_ids   # NO U_B / no adapter cycle-check ids
# After
    meas_l_offset = m_l + m_r
    meas_r_offset = meas_l_offset + n_V_l
    meas_l_ids = tuple(check_ids[meas_l_offset : meas_l_offset + n_V_l])
    meas_r_ids = tuple(check_ids[meas_r_offset : meas_r_offset + n_V_r])
    meas_check_ids = meas_l_ids + meas_r_ids   # NO U_B / no adapter cycle-check ids
```

And `chi_check_ids=chi_check_ids` at line 723 → `meas_check_ids=meas_check_ids`. The `_surgery_observable` parameter `chi_check_ids: tuple[int, ...]` (line 1015) → `meas_check_ids: tuple[int, ...]`. Inside `_surgery_observable` (line 1042–1047): `chi_targets = [... for cid in chi_check_ids]` → `meas_targets = [... for cid in meas_check_ids]`.

Also rename the local `chi_check_ids` in `build_single_ppm_circuit` (lines 374–386):

```python
# Before
    m_X, m_Z, n_V = gadget.code.matrix_x.shape[0], gadget.code.matrix_z.shape[0], len(gadget.support)
    if gadget.basis is Pauli.X:
        chi_check_ids = tuple(qubit_ids.checks_x[m_X : m_X + n_V])
    else:
        chi_check_ids = tuple(qubit_ids.checks_z[m_Z : m_Z + n_V])

    circuit += _surgery_observable(
        gadget,
        chi_check_ids=chi_check_ids,
        ...
    )
# After
    m_X, m_Z, n_V = gadget.code.matrix_x.shape[0], gadget.code.matrix_z.shape[0], len(gadget.support)
    if gadget.basis is Pauli.X:
        meas_check_ids = tuple(qubit_ids.checks_x[m_X : m_X + n_V])
    else:
        meas_check_ids = tuple(qubit_ids.checks_z[m_Z : m_Z + n_V])

    circuit += _surgery_observable(
        gadget,
        meas_check_ids=meas_check_ids,
        ...
    )
```

- [ ] **Step 7.5: Update _test_circuit.py callers of `_surgery_observable`**

Line 260, 270:

```python
# Before
    chi_check_ids = tuple(range(100, 100 + len(g.support)))
    ...
    circuit = _surgery_observable(
        g, chi_check_ids=chi_check_ids, data_ids=data_ids,
        support_indices=g.support, num_rounds=2, measurement_record=meas_rec,
    )
# After
    meas_check_ids = tuple(range(100, 100 + len(g.support)))
    ...
    circuit = _surgery_observable(
        g, meas_check_ids=meas_check_ids, data_ids=data_ids,
        support_indices=g.support, num_rounds=2, measurement_record=meas_rec,
    )
```

Lines 264–265 in the local meas_rec setup also rename `chi_check_ids` → `meas_check_ids`.

- [ ] **Step 7.6: Run tests**

Run: `uv run pytest src/qldpc/circuits/surgery/ -q 2>&1 | tail -3`
Expected: `147 passed`.

- [ ] **Step 7.7: Sanity-grep**

Run: `git grep -nE 'chi_(l|r|total|start|check_ids|l_offset|r_offset|l_rows|r_rows|l_ids|r_ids)\b' src/qldpc/circuits/surgery/`
Expected: zero hits.

Run: `git grep -nE 'M_chi|M_co|m_chi|m_co' src/qldpc/circuits/surgery/`
Expected: zero hits.

- [ ] **Step 7.8: Commit**

```bash
git add src/qldpc/circuits/surgery/
git commit -m "$(cat <<'EOF'
refactor(surgery): rename chi/co locals to meas/comp in circuit.py

Stitch and lane-layout locals named chi_l/chi_r/chi_total/M_chi/M_co
become n_meas_l/n_meas_r/n_meas_total/M_meas/M_comp. The
_surgery_observable parameter chi_check_ids becomes meas_check_ids.
Pure local-variable rename; no semantics or order-of-operations change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Test-local kappa/chi/r counters → n_ancilla/n_meas_checks/n_comp_checks; test fn rename

**Files:**
- Modify: `src/qldpc/circuits/surgery/_test_gadget.py`
- Modify: `src/qldpc/circuits/surgery/_test_circuit.py`
- Modify: `src/qldpc/circuits/surgery/_test_cheeger.py`

- [ ] **Step 8.1: Update `WEBSTER_TABLE_I_KAPPA_CHI_R` constant name**

In `_test_gadget.py` line 20:

```python
# Before
WEBSTER_TABLE_I_KAPPA_CHI_R = [(0, 19), (1, 31), (2, 49), (3, 79)]
# After
WEBSTER_TABLE_I_ANCILLA_MEAS_COMP = [(0, 19), (1, 31), (2, 49), (3, 79)]
```

And the `@pytest.mark.parametrize("code_index,n_anc", WEBSTER_TABLE_I_KAPPA_CHI_R)` decorator on line 256 → `... WEBSTER_TABLE_I_ANCILLA_MEAS_COMP`.

- [ ] **Step 8.2: Rename `test_webster_table_i_kappa_chi_r_exact` → `test_webster_table_i_ancilla_meas_comp_exact`**

In `_test_gadget.py` line 257, 355:

```python
# Before
def test_webster_table_i_kappa_chi_r_exact(code_index, n_anc):
    """Webster Table I: κ + χ + r matches for each of the 4 codes."""
    ...
    kappa = len(g1.ancilla_qubits)
    chi = int(g1.x.sum())  # |V_0|
    r = g1.gauge.shape[0]
    assert kappa + chi + r == n_anc, (
        f"code {code_index}: κ={kappa}, χ={chi}, r={r}, "
        f"sum={kappa+chi+r}, expected {n_anc}"
    )

def test_webster_table_i_z_basis_kappa_chi_r_exact():
    """Webster Z̄_1 seed produces the same κ+χ+r counts (basis-symmetric)."""
    ...
    for code_index, expected in [(0, 19), (1, 31), (2, 49), (3, 79)]:
        ...
        kappa = len(g.ancilla_qubits)
        chi = len(g.support)
        r = g.gauge.shape[0]
        assert kappa + chi + r == expected, (
            f"code {code_index}: Z-basis got κ+χ+r={kappa+chi+r}, expected {expected}"
        )
# After
def test_webster_table_i_ancilla_meas_comp_exact(code_index, n_anc):
    """Webster Table I in Cain notation: |Q'| + |S'_meas| + |S'_comp| matches
    each of the 4 generalised-bicycle codes. Reproduces Webster Table I exactly."""
    ...
    n_ancilla = len(g1.ancilla_qubits)
    n_meas_checks = int(g1.x.sum())  # |support|
    n_comp_checks = g1.gauge.shape[0]
    assert n_ancilla + n_meas_checks + n_comp_checks == n_anc, (
        f"code {code_index}: |Q'|={n_ancilla}, |S'_meas|={n_meas_checks}, |S'_comp|={n_comp_checks}, "
        f"sum={n_ancilla+n_meas_checks+n_comp_checks}, expected {n_anc}"
    )

def test_webster_table_i_z_basis_ancilla_meas_comp_exact():
    """Webster Z̄_1 seed in Cain notation: |Q'| + |S'_meas| + |S'_comp| matches
    (basis-symmetric dual; reproduces Webster Table I)."""
    ...
    for code_index, expected in [(0, 19), (1, 31), (2, 49), (3, 79)]:
        ...
        n_ancilla = len(g.ancilla_qubits)
        n_meas_checks = len(g.support)
        n_comp_checks = g.gauge.shape[0]
        assert n_ancilla + n_meas_checks + n_comp_checks == expected, (
            f"code {code_index}: Z-basis got |Q'|+|S'_meas|+|S'_comp|={n_ancilla+n_meas_checks+n_comp_checks}, expected {expected}"
        )
```

- [ ] **Step 8.3: Rename other local counters in tests**

In `_test_circuit.py` lines 81, 110 (local `r = g.gauge.shape[0]`): rename to `n_comp_checks` and update downstream usage (these are just length-comparison locals).

Actually grep first: only used in immediate scope for assertions like `assert n_det == len(reliable)`. Verify no downstream usage of `r` survives.

In `_test_cheeger.py` line 133 `n_chi = len(boosted.support)`: rename `n_chi` → `n_meas_checks` for consistency.

- [ ] **Step 8.4: Update comment-only Greek-letter mentions**

In `_test_gadget.py` lines 270–272 `f"code {code_index}: κ={kappa}, χ={chi}, r={r}, ..."`: already updated to use `|Q'|=...` form in Step 8.2.

In `_test_gadget.py` lines 142–143 (kappa-ancilla comment), 190 (kappa ancilla comment): already updated to drop "kappa" prefix in Task 3.

- [ ] **Step 8.5: Run tests**

Run: `uv run pytest src/qldpc/circuits/surgery/ -q 2>&1 | tail -3`
Expected: `147 passed`.

- [ ] **Step 8.6: Sanity-grep**

Run: `git grep -nE '\bkappa\b|\bchi\b|\bn_chi\b' src/qldpc/circuits/surgery/_test_*.py`
Expected: zero hits.

Run: `git grep -n 'kappa_chi_r' src/qldpc/circuits/surgery/`
Expected: zero hits.

- [ ] **Step 8.7: Commit**

```bash
git add src/qldpc/circuits/surgery/
git commit -m "$(cat <<'EOF'
refactor(surgery): rename test counters to Cain (Q'/S'_meas/S'_comp)

Webster Table I tests now use n_ancilla / n_meas_checks / n_comp_checks
locals and |Q'| / |S'_meas| / |S'_comp| in error messages. Test
function names updated: test_webster_table_i_kappa_chi_r_exact →
test_webster_table_i_ancilla_meas_comp_exact (with Z-basis dual).
Constant WEBSTER_TABLE_I_KAPPA_CHI_R → WEBSTER_TABLE_I_ANCILLA_MEAS_COMP.
Docstrings continue to cite Webster Table I as the source of truth.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Webster citation comments — add Cain mapping lines; final verification

**Files:** all surgery source files

- [ ] **Step 9.1: Add Cain-mapping lines under Webster citation comments**

For each Webster paper-citation comment in gadget.py / bridge.py / cheeger.py / circuit.py, add a one-line Cain translation directly below.

In `src/qldpc/circuits/surgery/gadget.py`:

```python
# Before (line 39–43)
    """Webster §II.A step 1 — V_0 = supp(x); C_0 = checks touching V_0; F = H_complement[C_0, V_0].

    For basis=Pauli.X: F = H_Z[C_0, V_0] (the complementary basis to the measured logical).
    For basis=Pauli.Z: F = H_X[C_0, V_0].
    """
# After
    """Webster §II.A step 1 — V_0 = supp(x); C_0 = checks touching V_0; F = H_complement[C_0, V_0].

    Cain mapping: V_0 → support; C_0 → data_checks; F → incidence.

    For basis=Pauli.X: incidence = H_Z[data_checks, support] (the complementary
    basis to the measured logical). For basis=Pauli.Z: incidence = H_X[data_checks, support].
    """
```

Apply the same pattern at:
- `_step2_gauge_fix` docstring (lines 66–69)
- `_assemble_HX_L1` docstring (lines 81–95)
- `_step3_assemble` docstring (lines 115–119)
- `build_gadget` docstring (lines 160–164)
- `build_gadget_augmented` docstring (lines 195–207)

In `src/qldpc/circuits/surgery/bridge.py`:
- `Bridge` dataclass docstring (lines 21–24)
- `build_bridge` docstring (lines 344–349)

In `src/qldpc/circuits/surgery/cheeger.py`:
- `_exact_boundary_cheeger` docstring (lines 22–48)
- `_spectral_cheeger_lower_bound` docstring (lines 91–104)
- `cheeger_constant` docstring (lines 114–125)
- `boost_gadget_cheeger_combinatorial` docstring (lines 179–227)
- `boost_gadget_distance` docstring (lines 318–355)
- `boost_gadget` docstring (lines 416–428)

In `src/qldpc/circuits/surgery/circuit.py`:
- The lane docstring (lines 130–148): rename `κ ancillas` → `ancilla qubits (Q')`, `χ ancillas` → `S'_meas ancillas`, `G ancillas` → `S'_comp ancillas`. Add Cain mapping summary at the top.
- `_classify_reliable_round1_checks` (lines 874–883): `# data H_X rows (det. +1)` → `# data S_X rows`; `# gauge-fix G rows` → `# gauge rows (= S'_comp)`. Same for basis=Z.

For each change, the goal is: a developer reading the comment can verify the Cain symbol from the Webster paper symbol on the same screen.

- [ ] **Step 9.2: Sanity-check no production attribute still uses Webster letters**

Run:

```bash
git grep -nE '\.(V0|C0|kappa_qubits|F|G)\b' src/qldpc/circuits/surgery/*.py
```

Expected: hits ONLY in Webster citation comments (e.g. `# Webster §II.A step 1 — V_0 = supp(x); ...`); no hits in production code lines (attribute access, method calls).

Run:

```bash
git grep -nE 'extra_kappa|kappa_ids|kappa_op|cl_kappa|cr_kappa|HZ_kappa_block|chi_total|chi_l|chi_r|G_aug|G_l\b|G_r\b|G_total' src/qldpc/circuits/surgery/*.py
```

Expected: zero hits.

- [ ] **Step 9.3: Notebook-drift check**

Run:

```bash
grep -rnE '\.(V0|C0|kappa_qubits|F|G)\b|\bkappa\b|\bchi\b' docs/notebooks/ 2>/dev/null || true
```

If hits are found in notebook code cells (not markdown), update them using the same rename. Notebook markdown text may continue to use Webster notation since they're prose explanations.

- [ ] **Step 9.4: Final run of full test suite**

Run: `uv run pytest src/qldpc/circuits/surgery/ -q 2>&1 | tail -3`
Expected: `147 passed`.

Run: `uv run pytest 2>&1 | tail -10`
Expected: full project pass count unchanged from baseline (rename does not affect other modules).

- [ ] **Step 9.5: Commit**

```bash
git add src/qldpc/circuits/surgery/
git add -u docs/notebooks/  # only if notebooks were touched
git commit -m "$(cat <<'EOF'
docs(surgery): annotate Webster citations with Cain mapping

Each Webster §II.A citation comment now carries a Cain-mapping line
below it (V_0 → support; C_0 → data_checks; F → incidence; G → gauge;
κ → ancilla_qubits) so future readers can verify the paper symbol
against the code symbol on the same screen. Lane labels in circuit.py
updated similarly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Success Criteria (already in spec; restated here)

* `uv run pytest src/qldpc/circuits/surgery/ -q` reports `147 passed` at the end and after every intermediate commit.
* `git grep -nE '\.(V0|C0|kappa_qubits|F|G)\b' src/qldpc/circuits/surgery/` hits ONLY Webster citation comments (no live attribute access).
* `git grep -nE 'extra_kappa|chi_total|chi_l|chi_r|M_chi|M_co' src/qldpc/circuits/surgery/` returns zero.
* Each commit in the series leaves the test suite green.
* Webster paper citations remain verifiable (V_0/C_0/F/G appear in comments next to the Cain mapping).

## Risks (already in spec; restated)

* Regex `\.F\b` / `\.G\b` could theoretically catch unrelated attributes. Mitigation: scoped to `surgery/` and audited per-file before commit.
* Local variables `chi_l/chi_r/chi_total` are NOT GadgetLayout attributes but are circuit-construction counters; they get a parallel rename in Task 7.
* If notebooks reference old attribute names, they break at runtime. Task 9 step 9.3 catches this.
