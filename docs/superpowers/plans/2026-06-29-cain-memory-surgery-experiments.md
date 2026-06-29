# Cain-faithful memory & surgery experiments — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `build_single_ppm_circuit` / `build_joint_ppm_circuit` emit the Cain et al. arXiv:2603.28627 Appendix D surgery-experiment observable sets (`k+1` match-basis, `k−1` opposite-basis, `t=1`), with first-cycle `L` readout and Pauli-frame-corrected block observables; reuse `get_memory_experiment` for the memory experiment.

**Architecture:** Data-qubit init/readout basis (`experiment_basis`) is decoupled from `gadget.basis`. Block (space-like) observables are bare-code logicals of `experiment_basis`, frame-corrected onto the `Q'`/bridge split records via a GF(2) solve so they are valid merged-code logicals; the time-like `L` observable (match-basis only) is the XOR of the first-cycle `S'_meas` outcomes. The obs0/obs1/`block_observables` path is removed.

**Tech Stack:** Python, `numpy`, `galois` (GF(2) linear algebra), `stim`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-06-29-cain-memory-surgery-experiments-design.md`

## Global Constraints

- GF(2) linear algebra via `galois.GF(2)` (already imported as `GF2` in `gadget.py`; re-create locally in `circuit.py`).
- Citations in docstrings/comments use full arXiv form (authors + arXiv:ID + §); never `math.md` or bare surnames.
- No LER / sinter / statistical tests. Verify via DEM-compile, observable counts, noiseless determinism, structural matrix properties, truth tables.
- Y / mixed-basis surgery is out of scope (do not touch `y_circuit.py`, `y_gadget.py`, `build_single_y_ppm_circuit`).
- `experiment_basis: PauliXZ` defaults to the measured basis (`gadget.basis` for single, `bridge.basis` for joint) → the natural `k+1` experiment.
- `t = 1` always (L=1 gadget; joint measures one product).
- Observable index order: block observables `0..(num_block−1)`, then the time-like `L` at index `num_block` (match-basis only).

---

### Task 1: GF(2) particular-solution solver

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py` (add `GF2` import + `_gf2_solve`)
- Test: `src/qldpc/circuits/surgery/circuit_test.py`

**Interfaces:**
- Produces: `_gf2_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray | None` — a particular solution `x` (uint8 vector, length `A.shape[1]`) to `A x = b` over GF(2), with free variables set to 0; `None` if inconsistent.

- [ ] **Step 1: Write the failing test**

```python
# in circuit_test.py
import numpy as np
from qldpc.circuits.surgery.circuit import _gf2_solve


def test_gf2_solve_consistent_returns_particular_solution():
    A = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    b = np.array([1, 0], dtype=np.uint8)
    x = _gf2_solve(A, b)
    assert x is not None
    assert np.array_equal((A @ x) % 2, b)


def test_gf2_solve_inconsistent_returns_none():
    A = np.array([[1, 0], [1, 0], [0, 0]], dtype=np.uint8)
    b = np.array([1, 0, 0], dtype=np.uint8)  # rows 0,1 demand x0=1 and x0=0
    assert _gf2_solve(A, b) is None


def test_gf2_solve_zero_rhs_returns_zero_vector():
    A = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    b = np.array([0, 0], dtype=np.uint8)
    x = _gf2_solve(A, b)
    assert x is not None
    assert np.array_equal(x, np.zeros(2, dtype=np.uint8))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/qldpc/circuits/surgery/circuit_test.py -k gf2_solve -q`
Expected: FAIL with `ImportError` / `cannot import name '_gf2_solve'`.

- [ ] **Step 3: Write minimal implementation**

```python
# near the top of circuit.py, after the existing imports
import galois

GF2 = galois.GF(2)


def _gf2_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """Particular solution to ``A x = b`` over GF(2), or None if inconsistent.

    Free variables are set to 0. Uses row reduction of the augmented matrix
    [A | b] via galois; a pivot-free row with nonzero RHS means inconsistency.
    """
    A = np.asarray(A).astype(np.int_) % 2
    b = np.asarray(b).astype(np.int_).reshape(-1) % 2
    m, n = A.shape
    if m == 0:
        return np.zeros(n, dtype=np.uint8)
    aug = GF2(np.hstack([A, b.reshape(-1, 1)]))
    rref = np.asarray(aug.row_reduce()).astype(np.int_)
    x = np.zeros(n, dtype=np.uint8)
    for row in rref:
        nz = np.nonzero(row[:n])[0]
        if nz.size == 0:
            if row[n]:
                return None  # 0 == 1 : inconsistent
            continue
        x[nz[0]] = row[n]
    return x
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/qldpc/circuits/surgery/circuit_test.py -k gf2_solve -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/circuit.py src/qldpc/circuits/surgery/circuit_test.py
git commit -m "feat(surgery): GF(2) particular-solution solver for frame correction"
```

---

### Task 2: Commuting-logical basis builder

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py` (add `_commuting_logical_basis`)
- Test: `src/qldpc/circuits/surgery/circuit_test.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `_commuting_logical_basis(logical_ops: np.ndarray, L_support: np.ndarray) -> np.ndarray` — given the `k×n` bare-code logical supports (`logical_ops`, of the readout type) and the `n`-vector `L_support` (the measured operator, opposite Pauli type), returns a `(k or k−1)×n` GF(2) basis of the logicals commuting with `L`. Returns all `k` rows when none anticommute (same-type / match-basis); otherwise `k−1` rows.

- [ ] **Step 1: Write the failing test**

```python
from qldpc.circuits.surgery.circuit import _commuting_logical_basis


def test_commuting_basis_all_commute_returns_all():
    logical_ops = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint8)
    L = np.array([0, 0, 0], dtype=np.uint8)  # symplectic product 0 with everything
    basis = _commuting_logical_basis(logical_ops, L)
    assert basis.shape == (2, 3)
    assert np.array_equal(basis, logical_ops)


def test_commuting_basis_drops_one_when_one_anticommutes():
    logical_ops = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint8)
    L = np.array([1, 0, 0], dtype=np.uint8)  # anticommutes only with row 0
    basis = _commuting_logical_basis(logical_ops, L)
    assert basis.shape == (1, 3)
    assert ((basis @ L) % 2 == 0).all()  # all commute with L
    assert np.array_equal(basis[0], np.array([0, 1, 0], dtype=np.uint8))


def test_commuting_basis_general_L_combines_multiple_anticommuters():
    # L overlaps rows 0 AND 1 (both anticommute); result must be k-1 = 1, commuting.
    logical_ops = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.uint8)
    L = np.array([1, 0, 0], dtype=np.uint8)  # dot row0=1, row1=1 -> both anticommute
    basis = _commuting_logical_basis(logical_ops, L)
    assert basis.shape == (1, 3)
    assert ((basis @ L) % 2 == 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/qldpc/circuits/surgery/circuit_test.py -k commuting_basis -q`
Expected: FAIL with `cannot import name '_commuting_logical_basis'`.

- [ ] **Step 3: Write minimal implementation**

```python
def _commuting_logical_basis(logical_ops: np.ndarray, L_support: np.ndarray) -> np.ndarray:
    """Basis of the bare-code logicals (rows of ``logical_ops``) commuting with L.

    The symplectic functional a_i = (L_support . logical_ops[i]) mod 2 is a linear
    functional on the k-dim logical space; its kernel (dim k or k-1) is the
    commuting subspace. When some a_i == 1, pick a pivot p with a_p == 1 and return
    {ops[i] : a_i == 0} ∪ {ops[i] ⊕ ops[p] : a_i == 1, i != p}. When all a_i == 0
    (same Pauli type / match-basis) return all k rows unchanged.

    Construction mirrors the gauge-fix logic of Webster, Smith, Cohen
    arXiv:2511.15989 §II.A used by build_gadget; here it selects the k-t readout
    observables of Cain et al. arXiv:2603.28627 Appendix D (t=1).
    """
    logical_ops = np.asarray(logical_ops).astype(np.uint8)
    L_support = np.asarray(L_support).astype(np.uint8).reshape(-1)
    a = (logical_ops @ L_support) % 2
    ones = np.nonzero(a)[0]
    if ones.size == 0:
        return logical_ops.copy()
    p = int(ones[0])
    rows = [logical_ops[i] for i in range(logical_ops.shape[0]) if a[i] == 0]
    rows += [(logical_ops[i] ^ logical_ops[p]) for i in ones[1:]]
    return np.array(rows, dtype=np.uint8).reshape(-1, logical_ops.shape[1])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/qldpc/circuits/surgery/circuit_test.py -k commuting_basis -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/circuit.py src/qldpc/circuits/surgery/circuit_test.py
git commit -m "feat(surgery): commuting-logical basis builder (k-t selector)"
```

---

### Task 3: Frame-corrected block-observable builder

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py` (add `_block_observable_targets`)
- Test: `src/qldpc/circuits/surgery/circuit_test.py`

**Interfaces:**
- Consumes: `_gf2_solve` (Task 1).
- Produces: `_block_observable_targets(merged_code, experiment_basis, w, n_data, col_record) -> list[stim.target_rec]` — for a single commuting logical support `w` (length `n_data`, `experiment_basis` type), returns the end-of-circuit measurement targets for its frame-corrected merged-code representative. `col_record` is a `dict[int, stim.target_rec]` mapping every merged column (data + `Q'` + bridge) to its end measurement record. Raises `AssertionError` if `w` does not commute with `L` (unsolvable) — callers pass only commuting `w`.

- [ ] **Step 1: Write the failing test**

```python
import stim
from qldpc.codes import CSSCode
from qldpc.objects import Pauli
from qldpc.circuits.surgery.circuit import _block_observable_targets


def test_block_observable_targets_no_deformation_when_data_only_valid():
    # Merged code = a code where a data-only Z logical already commutes with all X.
    # Use a 2-qubit code with HX empty, HZ empty (1 logical), Q' = none.
    merged = CSSCode(
        matrix_x=np.zeros((0, 1), dtype=int),
        matrix_z=np.zeros((0, 1), dtype=int),
        is_subsystem_code=False,
    )
    col_record = {0: stim.target_rec(-1)}
    w = np.array([1], dtype=np.uint8)  # Z on the single data qubit
    targets = _block_observable_targets(merged, Pauli.Z, w, n_data=1, col_record=col_record)
    assert targets == [stim.target_rec(-1)]


def test_block_observable_targets_adds_qprime_records_for_deformation():
    # merged X-check forces a Z logical to deform onto the Q' column.
    # cols: 0 = data, 1 = Q'.  HX_merged = [[1,1]] (one X-check on data0 & Q').
    merged = CSSCode(
        matrix_x=np.array([[1, 1]], dtype=int),
        matrix_z=np.zeros((0, 2), dtype=int),
        is_subsystem_code=False,
    )
    col_record = {0: stim.target_rec(-2), 1: stim.target_rec(-1)}
    w = np.array([1, 0], dtype=np.uint8)  # data-only Z on col 0 anticommutes with the X-check
    targets = _block_observable_targets(merged, Pauli.Z, w, n_data=1, col_record=col_record)
    # deformed rep must add the Q' column (col 1) so it commutes with the X-check
    assert set(targets) == {stim.target_rec(-2), stim.target_rec(-1)}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/qldpc/circuits/surgery/circuit_test.py -k block_observable_targets -q`
Expected: FAIL with `cannot import name '_block_observable_targets'`.

- [ ] **Step 3: Write minimal implementation**

```python
def _block_observable_targets(
    merged_code: CSSCode,
    experiment_basis: Pauli,
    w: np.ndarray,
    n_data: int,
    col_record: dict[int, stim.target_rec],
) -> list[stim.target_rec]:
    """End-of-circuit targets for the frame-corrected merged-code rep of logical ``w``.

    ``w`` is the bare-code logical support (length n_data, ``experiment_basis`` type).
    The deformed representative v = w ⊕ c commutes with every opposite-type merged
    check, where c lives on the non-data columns (Q' + bridge) and is read from the
    split. Pauli-frame correction folded into the observable (no physical gates),
    per Cain et al. arXiv:2603.28627 Appendix D.
    """
    # opposite-type checks: an experiment_basis logical must commute with them.
    M_opp = (
        np.asarray(merged_code.matrix_z).astype(np.uint8)
        if experiment_basis is Pauli.X
        else np.asarray(merged_code.matrix_x).astype(np.uint8)
    )
    n_merged = merged_code.num_qudits
    corr_cols = np.arange(n_data, n_merged)  # Q' + bridge
    w = np.asarray(w).astype(np.uint8).reshape(-1)
    if M_opp.shape[0] == 0:
        c = np.zeros(corr_cols.size, dtype=np.uint8)
    else:
        syndrome = (M_opp[:, :n_data] @ w) % 2
        c = _gf2_solve(M_opp[:, corr_cols], syndrome)
        assert c is not None, "logical w does not commute with L (unsolvable deformation)"
    support_cols = list(np.nonzero(w)[0]) + [int(corr_cols[j]) for j in np.nonzero(c)[0]]
    return [col_record[col] for col in support_cols]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/qldpc/circuits/surgery/circuit_test.py -k block_observable_targets -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/circuit.py src/qldpc/circuits/surgery/circuit_test.py
git commit -m "feat(surgery): frame-corrected block-observable target builder"
```

---

### Task 4: Decouple data init/readout from gadget.basis

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py` (`_surgery_state_prep`, `_surgery_detach_and_readout`)
- Test: `src/qldpc/circuits/surgery/circuit_test.py`

**Interfaces:**
- Produces:
  - `_surgery_state_prep(gadget, data_ids, ancilla_ids, bridge_ids=(), *, experiment_basis, data_init=None)` — data default init from `experiment_basis` (`RX`/`|+⟩` for X, `R`/`|0⟩` for Z); ancilla+bridge init unchanged (complement of `gadget.basis`).
  - `_surgery_detach_and_readout(gadget, *, data_ids, ancilla_ids, bridge_ids, measurement_record, experiment_basis, destructive_measure_data=True)` — data readout op from `experiment_basis` (`MX` for X, `M` for Z); ancilla/bridge detach op unchanged (`M` for X-gadget, `MX` for Z-gadget).

- [ ] **Step 1: Write the failing test**

```python
from qldpc.codes import HammingCode  # small CSS code for tests
from qldpc.circuits.surgery.circuit import _surgery_state_prep, _surgery_detach_and_readout
from qldpc.circuits.bookkeeping import MeasurementRecord
from qldpc.circuits.surgery.gadget import build_gadget


def _x_gadget():
    code = CSSCode.from_code(HammingCode(3), HammingCode(3))  # [[7,1,3]] Steane
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    return build_gadget(code, x, basis=Pauli.X)


def test_state_prep_z_experiment_on_x_gadget_inits_data_in_z():
    g = _x_gadget()
    data_ids = tuple(range(g.code.num_qudits))
    anc_ids = tuple(range(g.code.num_qudits, g.code.num_qudits + len(g.Q_prime)))
    circ = _surgery_state_prep(g, data_ids, anc_ids, experiment_basis=Pauli.Z)
    text = str(circ)
    # data in Z -> R on data; ancilla (X-gadget complement = Z) -> R on ancilla; no RX
    assert "RX" not in text
    assert "R " in text or text.strip().startswith("R")


def test_detach_readout_z_experiment_on_x_gadget_measures_data_in_z():
    g = _x_gadget()
    data_ids = tuple(range(g.code.num_qudits))
    anc_ids = tuple(range(g.code.num_qudits, g.code.num_qudits + len(g.Q_prime)))
    rec = MeasurementRecord()
    circ = _surgery_detach_and_readout(
        g, data_ids=data_ids, ancilla_ids=anc_ids, bridge_ids=(),
        measurement_record=rec, experiment_basis=Pauli.Z,
    )
    text = str(circ)
    assert "MX" not in text  # data measured with M (Z), ancilla measured with M (Z)
    assert "M " in text or "\nM" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/qldpc/circuits/surgery/circuit_test.py -k "state_prep_z_experiment or detach_readout_z_experiment" -q`
Expected: FAIL with `TypeError` (`unexpected keyword argument 'experiment_basis'`).

- [ ] **Step 3: Write minimal implementation**

Replace the signature/body of `_surgery_state_prep` so the data default char comes from `experiment_basis`:

```python
def _surgery_state_prep(
    gadget: GadgetLayout,
    data_ids: tuple[int, ...],
    ancilla_ids: tuple[int, ...],
    bridge_ids: tuple[int, ...] = (),
    *,
    experiment_basis: PauliXZ,
    data_init: str | None = None,
) -> stim.Circuit:
    """Init data/ancilla/bridge qubits at the start of a surgery PPM circuit.

    Data default init follows ``experiment_basis`` (decoupled from gadget.basis,
    per Cain et al. arXiv:2603.28627 Appendix D): X -> |+> (RX), Z -> |0> (R).
    ancilla + bridge init follows the complement of ``gadget.basis`` (the merge
    mechanics): basis=X -> |0> (R), basis=Z -> |+> (RX). ``data_init`` overrides
    the per-data-qubit state (chars '0','1','+','-'); see body for the mapping.
    """
    if data_init is None:
        default_char = "+" if experiment_basis is Pauli.X else "0"
        per_qubit = default_char * len(data_ids)
    else:
        if len(data_init) == 1:
            data_init = data_init * len(data_ids)
        if len(data_init) != len(data_ids):
            raise ValueError(
                f"data_init length {len(data_init)} does not match num data "
                f"qubits {len(data_ids)}; pass a length-1 string to broadcast"
            )
        invalid = sorted(set(data_init) - set("01+-"))
        if invalid:
            raise ValueError(
                f"data_init must contain only '0', '1', '+', '-'; got invalid chars {invalid}"
            )
        per_qubit = data_init
    # ... (keep the existing r_data/rx_data/x_after/z_after loop unchanged) ...
    # ... (keep the existing ancilla init block: anc_init = "R" if gadget.basis is Pauli.X else "RX")
```

Replace `_surgery_detach_and_readout` so the data op follows `experiment_basis`:

```python
def _surgery_detach_and_readout(
    gadget: GadgetLayout,
    *,
    data_ids: tuple[int, ...],
    ancilla_ids: tuple[int, ...],
    bridge_ids: tuple[int, ...],
    measurement_record: MeasurementRecord,
    experiment_basis: PauliXZ,
    destructive_measure_data: bool = True,
) -> stim.Circuit:
    """Detach the κ/bridge ancillas; optionally destructively measure the data.

    ancilla/bridge detach op follows complement of ``gadget.basis`` (the split);
    data readout op follows ``experiment_basis`` (Cain et al. arXiv:2603.28627
    Appendix D): X -> MX, Z -> M.
    """
    circuit = stim.Circuit()
    detach_qubits = list(ancilla_ids) + list(bridge_ids)
    ancilla_op = "M" if gadget.basis is Pauli.X else "MX"
    data_op = "MX" if experiment_basis is Pauli.X else "M"
    circuit.append(ancilla_op, detach_qubits)
    measurement_record.append({q: i for i, q in enumerate(detach_qubits)})
    if not destructive_measure_data:
        return circuit
    circuit.append("SHIFT_COORDS", [], (0, 0, 1))
    circuit.append(data_op, list(data_ids))
    measurement_record.append({q: i for i, q in enumerate(data_ids)})
    return circuit
```

Update the two call sites (`build_single_ppm_circuit`, `_build_joint_ppm_circuit_same_basis`) to pass `experiment_basis=` (placeholder until Tasks 6/7 thread the param — for now pass `gadget.basis` / `g_l.basis` so existing behavior is preserved):

```python
    circuit += _surgery_state_prep(
        gadget, data_ids, Q_prime_ids, bridge_ids,
        experiment_basis=gadget.basis, data_init=data_init,
    )
    ...
    circuit += _surgery_detach_and_readout(
        gadget, data_ids=data_ids, ancilla_ids=Q_prime_ids, bridge_ids=bridge_ids,
        measurement_record=measurement_record, experiment_basis=gadget.basis,
        destructive_measure_data=destructive_measure_data,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/qldpc/circuits/surgery/circuit_test.py -k "state_prep_z_experiment or detach_readout_z_experiment" -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/circuit.py src/qldpc/circuits/surgery/circuit_test.py
git commit -m "feat(surgery): decouple data init/readout basis from gadget.basis"
```

---

### Task 5: Unify round-1 reliability and final detectors on experiment_basis

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py` (`_classify_reliable_round1_checks`, `_surgery_qec_cycle`, `_surgery_final_detectors`)
- Test: `src/qldpc/circuits/surgery/circuit_test.py`

**Interfaces:**
- Produces: `_reliable_checks(gadget, merged_code, qubit_ids, *, experiment_basis, n_data, joint=None) -> tuple[int, ...]` — the merged checks whose support lies entirely within matching-basis qubits (data in `experiment_basis`, `Q'`/bridge in complement of `gadget.basis`). Used for both round-1 detectors and final-readout reconstructable detectors. Replaces `_classify_reliable_round1_checks`.

**Note:** the rule is computed directly from the merged check matrices + the per-qubit init basis, so it is correct for single and joint without index-slicing.

- [ ] **Step 1: Write the failing test**

```python
from qldpc.circuits.surgery.circuit import _reliable_checks, _gadget_merged_csscode
from qldpc.circuits.bookkeeping import QubitIDs


def test_reliable_checks_match_basis_x_gadget_reproduces_hx_and_gauge():
    g = _x_gadget()
    merged = _gadget_merged_csscode(g)
    qids = QubitIDs.from_code(merged)
    n_data = g.code.num_qudits
    rel = set(_reliable_checks(g, merged, qids, experiment_basis=Pauli.X, n_data=n_data))
    m_X = g.code.matrix_x.shape[0]
    m_Z = g.code.matrix_z.shape[0]
    # original H_X rows are reliable; original H_Z rows (touch X-init data) are NOT
    assert set(qids.checks_x[:m_X]) <= rel
    assert not (set(qids.checks_z[:m_Z]) & rel)


def test_reliable_checks_opposite_basis_z_experiment_makes_all_z_checks_reliable():
    g = _x_gadget()
    merged = _gadget_merged_csscode(g)
    qids = QubitIDs.from_code(merged)
    n_data = g.code.num_qudits
    rel = set(_reliable_checks(g, merged, qids, experiment_basis=Pauli.Z, n_data=n_data))
    m_Z = g.code.matrix_z.shape[0]
    # data |0> + Q' |0>  => every Z-type merged check is reliable; no X-type check is
    assert set(qids.checks_z) <= rel
    assert not (set(qids.checks_x) & rel)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/qldpc/circuits/surgery/circuit_test.py -k reliable_checks -q`
Expected: FAIL with `cannot import name '_reliable_checks'`.

- [ ] **Step 3: Write minimal implementation**

```python
def _reliable_checks(
    gadget: GadgetLayout,
    merged_code: CSSCode,
    qubit_ids: QubitIDs,
    *,
    experiment_basis: PauliXZ,
    n_data: int,
    joint: tuple[GadgetLayout, Bridge, bool] | None = None,
) -> tuple[int, ...]:
    """Merged checks deterministic at round 1 == reconstructable from final readout.

    A CSS check is deterministic iff every qubit in its support is initialized in
    the basis matching the check's Pauli type. Data qubits are in
    ``experiment_basis``; Q'/bridge qubits are in complement(gadget.basis). This is
    the unifying rule of the design (Cain et al. arXiv:2603.28627 Appendix D).
    """
    HX = np.asarray(merged_code.matrix_x).astype(np.uint8)
    HZ = np.asarray(merged_code.matrix_z).astype(np.uint8)
    n_merged = merged_code.num_qudits
    # per-qubit init basis: True = X-basis init, False = Z-basis init
    anc_is_x = gadget.basis is Pauli.Z  # X-gadget -> Q' in Z (False); Z-gadget -> Q' in X (True)
    data_is_x = experiment_basis is Pauli.X
    x_init = np.zeros(n_merged, dtype=bool)
    x_init[:n_data] = data_is_x
    x_init[n_data:] = anc_is_x

    reliable: list[int] = []
    # X-type checks: deterministic iff support ⊆ X-init qubits
    for r in range(HX.shape[0]):
        supp = np.nonzero(HX[r])[0]
        if supp.size and x_init[supp].all():
            reliable.append(qubit_ids.checks_x[r])
    # Z-type checks: deterministic iff support ⊆ Z-init qubits
    for r in range(HZ.shape[0]):
        supp = np.nonzero(HZ[r])[0]
        if supp.size and (~x_init[supp]).all():
            reliable.append(qubit_ids.checks_z[r])
    return tuple(reliable)
```

Update `_surgery_qec_cycle` to accept `experiment_basis` + `n_data` and call `_reliable_checks` instead of `_classify_reliable_round1_checks`:

```python
def _surgery_qec_cycle(
    gadget, merged_code, num_rounds, qubit_ids, *,
    experiment_basis: PauliXZ, n_data: int,
    joint=None, single_sector=False,
):
    ...
    reliable = set(
        _reliable_checks(
            gadget, merged_code, qubit_ids,
            experiment_basis=experiment_basis, n_data=n_data, joint=joint,
        )
    )
    lane_idx = _check_lane_index_map(gadget, qubit_ids, joint=joint)
    ...  # (rest of the round loop unchanged)
```

Update `_surgery_final_detectors` to emit detectors for exactly the `_reliable_checks` set (each XOR'd against the final readouts that reconstruct it):

```python
def _surgery_final_detectors(
    gadget, merged_code, qubit_ids, *,
    measurement_record, experiment_basis: PauliXZ, n_data: int,
    joint=None, single_sector=False,
):
    """DETECTORs for checks reconstructable from the final readouts.

    Reconstructable set == _reliable_checks (same per-qubit basis rule). Each
    DETECTOR XORs ⊕(final M-records on the stab support) ⊕ last-round syndrome.
    """
    HX = np.asarray(merged_code.matrix_x).astype(np.uint8)
    HZ = np.asarray(merged_code.matrix_z).astype(np.uint8)
    reliable = set(
        _reliable_checks(
            gadget, merged_code, qubit_ids,
            experiment_basis=experiment_basis, n_data=n_data, joint=joint,
        )
    )
    circuit = stim.Circuit()
    lane_idx = _check_lane_index_map(gadget, qubit_ids, joint=joint)

    def _emit(stab_row, check_id):
        supp = np.nonzero(stab_row)[0]
        targets = [measurement_record.get_target_rec(qubit_ids.data[q]) for q in supp]
        targets.append(measurement_record.get_target_rec(check_id, -1))
        lane, idx = lane_idx[check_id]
        circuit.append("DETECTOR", targets, (idx, lane, 0))

    for r in range(HX.shape[0]):
        cid = qubit_ids.checks_x[r]
        if cid in reliable:
            _emit(HX[r], cid)
    for r in range(HZ.shape[0]):
        cid = qubit_ids.checks_z[r]
        if cid in reliable:
            _emit(HZ[r], cid)
    return circuit
```

(Note: the `data` index used in `_emit` is the merged column; for both single and joint the reconstructable checks are supported only on qubits measured in their matching basis at the end, so `qubit_ids.data[q]` resolves to the correct final/split record because `QubitIDs.data` covers data+`Q'`+bridge columns in order.)

Update the two builder call sites to pass `experiment_basis=gadget.basis` (placeholder) + `n_data` to `_surgery_qec_cycle` and `_surgery_final_detectors`, and delete `_classify_reliable_round1_checks`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/qldpc/circuits/surgery/circuit_test.py -k reliable_checks -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/circuit.py src/qldpc/circuits/surgery/circuit_test.py
git commit -m "feat(surgery): unify round-1 reliability + final detectors on experiment_basis"
```

---

### Task 6: experiment_basis observables in build_single_ppm_circuit

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py` (`_surgery_observable`, `build_single_ppm_circuit`)
- Test: `src/qldpc/circuits/surgery/circuit_test.py`

**Interfaces:**
- Consumes: `_commuting_logical_basis` (T2), `_block_observable_targets` (T3), `_reliable_checks` (T5).
- Produces:
  - `_surgery_observable(gadget, *, experiment_basis, merged_code, meas_check_ids, logical_ops, L_support, n_data, data_ids, qprime_ids, bridge_ids, measurement_record) -> stim.Circuit` — emits the block observables (frame-corrected, indices `0..m−1`) and, when `experiment_basis is gadget.basis`, the time-like `L` observable at index `m` (XOR of first-cycle `meas_check_ids`).
  - `build_single_ppm_circuit(gadget, *, rounds, experiment_basis=None, noise_model=None, data_init=None, destructive_measure_data=True, single_sector=False) -> stim.Circuit` — `experiment_basis=None` defaults to `gadget.basis`. `block_observables` param removed.

- [ ] **Step 1: Write the failing test**

```python
def test_single_ppm_match_basis_emits_k_plus_1_observables():
    g = _x_gadget()  # Steane: k=1
    circ = build_single_ppm_circuit(g, rounds=3, experiment_basis=Pauli.X)
    assert circ.num_observables == g.code.dimension + 1  # k + t, t=1


def test_single_ppm_opposite_basis_emits_k_minus_1_observables():
    # Use a k>=2 code so k-1 >= 1. BBCode [[18,...]] or a 2-logical CSS.
    code = CSSCode.from_code(HammingCode(3), HammingCode(3))  # k=1 -> k-1=0
    # For k-1>0 pick a 2-logical code:
    from qldpc.codes import BBCode
    bb = BBCode({"x": 3, "y": 3}, [("x", 1), ("y", 1)], [("y", 2), ("x", 1)])  # example k>=2
    xop = np.asarray(bb.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    gbb = build_gadget(bb, xop, basis=Pauli.X)
    circ = build_single_ppm_circuit(gbb, rounds=3, experiment_basis=Pauli.Z)
    assert circ.num_observables == bb.dimension - 1


def test_single_ppm_observables_deterministic_noiseless():
    code = CSSCode.from_code(HammingCode(3), HammingCode(3))
    xop = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, xop, basis=Pauli.X)
    circ = build_single_ppm_circuit(g, rounds=3, experiment_basis=Pauli.X)
    sampler = circ.compile_detector_sampler()
    _, obs = sampler.sample(shots=64, separate_observables=True)
    assert not obs.any()  # every observable deterministic (=0) with no noise
```

(If the `BBCode` constructor signature differs, substitute any in-repo CSS code with `dimension >= 2`; the assertion is `num_observables == dimension - 1`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/qldpc/circuits/surgery/circuit_test.py -k "single_ppm_match_basis or single_ppm_opposite_basis or single_ppm_observables_deterministic" -q`
Expected: FAIL (`block_observables` removed / `experiment_basis` not accepted / wrong observable count).

- [ ] **Step 3: Write minimal implementation**

Rewrite `_surgery_observable`:

```python
def _surgery_observable(
    gadget: GadgetLayout,
    *,
    experiment_basis: PauliXZ,
    merged_code: CSSCode,
    meas_check_ids: tuple[int, ...],
    logical_ops: np.ndarray,
    L_support: np.ndarray,
    n_data: int,
    data_ids: tuple[int, ...],
    qprime_ids: tuple[int, ...],
    bridge_ids: tuple[int, ...],
    measurement_record: MeasurementRecord,
) -> stim.Circuit:
    """Emit the Cain et al. arXiv:2603.28627 Appendix D surgery observable set.

    Block (space-like) observables: the commuting basis of experiment_basis
    logicals (k if match-basis, k-1 if opposite), each frame-corrected onto the
    Q'/bridge split records. Time-like L observable (match-basis only): XOR of the
    first-cycle S'_meas outcomes (get_target_rec(cid, 0)).
    """
    # column -> end measurement record (data final M; Q'/bridge split M)
    col_record: dict[int, stim.target_rec] = {}
    for col in range(n_data):
        col_record[col] = measurement_record.get_target_rec(data_ids[col])
    non_data_ids = tuple(qprime_ids) + tuple(bridge_ids)
    for j, qid in enumerate(non_data_ids):
        col_record[n_data + j] = measurement_record.get_target_rec(qid)

    circuit = stim.Circuit()
    basis_ops = _commuting_logical_basis(logical_ops, L_support)
    idx = 0
    for w in basis_ops:
        targets = _block_observable_targets(merged_code, experiment_basis, w, n_data, col_record)
        circuit.append("OBSERVABLE_INCLUDE", targets, idx)
        idx += 1
    if experiment_basis is gadget.basis:
        L_targets = [measurement_record.get_target_rec(cid, 0) for cid in meas_check_ids]
        circuit.append("OBSERVABLE_INCLUDE", L_targets, idx)
    return circuit
```

Rewrite `build_single_ppm_circuit` to thread `experiment_basis` and call the new observable builder (drop `block_observables`, obs0/obs1):

```python
def build_single_ppm_circuit(
    gadget: GadgetLayout,
    *,
    rounds: int,
    experiment_basis: PauliXZ | None = None,
    noise_model: NoiseModel | None = None,
    data_init: str | None = None,
    destructive_measure_data: bool = True,
    single_sector: bool = False,
) -> stim.Circuit:
    """Cain et al. arXiv:2603.28627 Appendix D single-PPM surgery experiment.

    experiment_basis (default gadget.basis): the data init/readout basis. Match
    basis -> k+1 observables (k block + time-like L); opposite -> k-1 (block
    commuting with L). See the design spec for full semantics.
    """
    if experiment_basis is None:
        experiment_basis = gadget.basis
    merged_code = _gadget_merged_csscode(gadget)
    qubit_ids = QubitIDs.from_code(merged_code)
    n_data = gadget.code.num_qudits
    data_ids = qubit_ids.data[:n_data]
    Q_prime_ids = qubit_ids.data[n_data:]
    bridge_ids: tuple[int, ...] = ()

    circuit = _surgery_qubit_coordinates(gadget, qubit_ids)
    circuit += _surgery_state_prep(
        gadget, data_ids, Q_prime_ids, bridge_ids,
        experiment_basis=experiment_basis, data_init=data_init,
    )
    qec_cycle, measurement_record, _ = _surgery_qec_cycle(
        gadget, merged_code, num_rounds=rounds, qubit_ids=qubit_ids,
        experiment_basis=experiment_basis, n_data=n_data, single_sector=single_sector,
    )
    circuit += qec_cycle
    circuit += _surgery_detach_and_readout(
        gadget, data_ids=data_ids, ancilla_ids=Q_prime_ids, bridge_ids=bridge_ids,
        measurement_record=measurement_record, experiment_basis=experiment_basis,
        destructive_measure_data=destructive_measure_data,
    )
    if destructive_measure_data:
        circuit += _surgery_final_detectors(
            gadget, merged_code, qubit_ids, measurement_record=measurement_record,
            experiment_basis=experiment_basis, n_data=n_data, single_sector=single_sector,
        )

    m_X, m_Z, n_V = (
        gadget.code.matrix_x.shape[0], gadget.code.matrix_z.shape[0], len(gadget.support),
    )
    if gadget.basis is Pauli.X:
        meas_check_ids = tuple(qubit_ids.checks_x[m_X : m_X + n_V])
    else:
        meas_check_ids = tuple(qubit_ids.checks_z[m_Z : m_Z + n_V])

    if destructive_measure_data:
        logical_ops = np.asarray(gadget.code.get_logical_ops(experiment_basis)).astype(np.uint8)
        circuit += _surgery_observable(
            gadget, experiment_basis=experiment_basis, merged_code=merged_code,
            meas_check_ids=meas_check_ids, logical_ops=logical_ops,
            L_support=np.asarray(gadget.x).astype(np.uint8), n_data=n_data,
            data_ids=data_ids, qprime_ids=Q_prime_ids, bridge_ids=bridge_ids,
            measurement_record=measurement_record,
        )

    if noise_model is not None:
        circuit = noise_model.noisy_circuit(circuit)
    return circuit
```

(`get_logical_ops(experiment_basis)` returns the `k×n` support matrix of the readout-type logicals; `gadget.x` is the `L` support.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/qldpc/circuits/surgery/circuit_test.py -k "single_ppm_match_basis or single_ppm_opposite_basis or single_ppm_observables_deterministic" -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/circuit.py src/qldpc/circuits/surgery/circuit_test.py
git commit -m "feat(surgery): single-PPM experiment_basis observables (k+1 / k-1)"
```

---

### Task 7: experiment_basis observables in build_joint_ppm_circuit

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py` (`build_joint_ppm_circuit`, `_build_joint_ppm_circuit_same_basis`)
- Test: `src/qldpc/circuits/surgery/circuit_test.py`

**Interfaces:**
- Consumes: `_surgery_observable` (T6) and all helpers.
- Produces: `build_joint_ppm_circuit(g_l, g_r, bridge, *, rounds, experiment_basis=None, noise_model=None, data_init=None, destructive_measure_data=True) -> tuple[stim.Circuit, QuditCode]` — `experiment_basis=None` defaults to `bridge.basis`. Combined data code is `c_l ⊕ c_r` (intercode) or shared `c` (intracode); `L_support` is the combined measured operator.

- [ ] **Step 1: Write the failing test**

```python
def test_joint_ppm_match_basis_emits_kl_plus_kr_plus_1(_steane_joint_fixture):
    g_l, g_r, bridge = _steane_joint_fixture  # two [[7,1,3]] patches, basis=X
    circ, joint_code = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=3, experiment_basis=Pauli.X)
    assert circ.num_observables == g_l.code.dimension + g_r.code.dimension + 1


def test_joint_ppm_opposite_basis_emits_kl_plus_kr_minus_1(_steane_joint_fixture):
    g_l, g_r, bridge = _steane_joint_fixture
    circ, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=3, experiment_basis=Pauli.Z)
    assert circ.num_observables == g_l.code.dimension + g_r.code.dimension - 1


def test_joint_ppm_observables_deterministic_noiseless(_steane_joint_fixture):
    g_l, g_r, bridge = _steane_joint_fixture
    circ, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=3, experiment_basis=Pauli.X)
    _, obs = circ.compile_detector_sampler().sample(shots=64, separate_observables=True)
    assert not obs.any()
```

(Reuse the existing joint-PPM fixture in `circuit_test.py`; if it constructs the gadgets/bridge inline, factor it into a `pytest.fixture` named `_steane_joint_fixture` returning `(g_l, g_r, bridge)`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/qldpc/circuits/surgery/circuit_test.py -k "joint_ppm_match_basis or joint_ppm_opposite_basis or joint_ppm_observables_deterministic" -q`
Expected: FAIL (`experiment_basis` not accepted / wrong count).

- [ ] **Step 3: Write minimal implementation**

Thread `experiment_basis` through `build_joint_ppm_circuit` → `_build_joint_ppm_circuit_same_basis`. In `_build_joint_ppm_circuit_same_basis`, default it to `bridge.basis`, pass to prep/qec/detach/final-detectors (as in Task 6), and build the combined `logical_ops` + `L_support`:

```python
    if experiment_basis is None:
        experiment_basis = bridge.basis
    ...
    # combined data-code logicals (block-diagonal over the data columns) + L
    if intercode:
        lx_l = np.asarray(g_l.code.get_logical_ops(experiment_basis)).astype(np.uint8)  # k_l×n_l
        lx_r = np.asarray(g_r.code.get_logical_ops(experiment_basis)).astype(np.uint8)  # k_r×n_r
        logical_ops = np.zeros((lx_l.shape[0] + lx_r.shape[0], n_l + n_r), dtype=np.uint8)
        logical_ops[: lx_l.shape[0], :n_l] = lx_l
        logical_ops[lx_l.shape[0] :, n_l:] = lx_r
        L_support = np.zeros(n_l + n_r, dtype=np.uint8)
        L_support[: n_l] = np.asarray(g_l.x).astype(np.uint8)
        L_support[n_l:] = np.asarray(g_r.x).astype(np.uint8)
        n_data = n_l + n_r
    else:  # intracode: shared data code
        logical_ops = np.asarray(g_l.code.get_logical_ops(experiment_basis)).astype(np.uint8)
        L_support = (np.asarray(g_l.x).astype(np.uint8) ^ np.asarray(g_r.x).astype(np.uint8))
        n_data = n_l
    ...
    if destructive_measure_data:
        circuit += _surgery_observable(
            g_l, experiment_basis=experiment_basis, merged_code=joint_code,
            meas_check_ids=meas_check_ids, logical_ops=logical_ops, L_support=L_support,
            n_data=n_data, data_ids=data_ids, qprime_ids=Q_prime_ids, bridge_ids=bridge_ids,
            measurement_record=measurement_record,
        )
```

The match-basis check inside `_surgery_observable` uses `gadget.basis`; pass `g_l` (whose `.basis == bridge.basis`) so `experiment_basis is g_l.basis` correctly gates the time-like `L`.

Update `build_joint_ppm_circuit` signature to accept and forward `experiment_basis`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/qldpc/circuits/surgery/circuit_test.py -k "joint_ppm_match_basis or joint_ppm_opposite_basis or joint_ppm_observables_deterministic" -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/circuit.py src/qldpc/circuits/surgery/circuit_test.py
git commit -m "feat(surgery): joint-PPM experiment_basis observables (k+1 / k-1)"
```

---

### Task 8: Folded cross-check + memory-experiment determinism test

**Files:**
- Test: `src/qldpc/circuits/surgery/circuit_test.py`
- (No production change; confirms `get_memory_experiment` usage + the §3.4 folded cross-check.)

**Interfaces:**
- Consumes: `build_single_ppm_circuit` (T6), `qldpc.circuits.get_memory_experiment`.

- [ ] **Step 1: Write the failing test**

```python
from qldpc.circuits import get_memory_experiment


def test_memory_experiment_emits_k_logical_x_observables():
    code = CSSCode.from_code(HammingCode(3), HammingCode(3))  # k=1
    circ = get_memory_experiment(code, basis=Pauli.X, num_rounds=2)
    assert circ.num_observables == code.dimension


def test_frame_correction_is_load_bearing_opposite_basis():
    # An opposite-basis block observable WITHOUT its Q'-split records is non-deterministic.
    from qldpc.codes import BBCode
    bb = BBCode({"x": 3, "y": 3}, [("x", 1), ("y", 1)], [("y", 2), ("x", 1)])
    g = build_gadget(bb, np.asarray(bb.get_logical_ops(Pauli.X)[0]).astype(np.uint8), basis=Pauli.X)
    circ = build_single_ppm_circuit(g, rounds=3, experiment_basis=Pauli.Z)
    _, obs = circ.compile_detector_sampler().sample(shots=128, separate_observables=True)
    assert not obs.any()  # corrected observables ARE deterministic
```

(The "load-bearing" assertion is positive: corrected observables are deterministic. The negative case—omitting the records yields randomness—is implicitly covered by Task 3's deformation test; do not build a circuit with deliberately-wrong observables here.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/qldpc/circuits/surgery/circuit_test.py -k "memory_experiment_emits or frame_correction_is_load_bearing" -q`
Expected: FAIL if `get_memory_experiment` import path differs or counts are wrong; otherwise confirms behavior.

- [ ] **Step 3: Write minimal implementation**

No production code. If `test_memory_experiment_emits_k_logical_x_observables` fails on the import, fix the import to the public path (`from qldpc.circuits import get_memory_experiment`). Otherwise both pass once Tasks 1–7 are in.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/qldpc/circuits/surgery/circuit_test.py -k "memory_experiment_emits or frame_correction_is_load_bearing" -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/circuit_test.py
git commit -m "test(surgery): memory k-logical-X + opposite-basis frame-correction determinism"
```

---

### Task 9: Migrate existing obs0/obs1 tests

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit_test.py` (and `circuit_single_y_test.py` only if it imports removed symbols)

**Interfaces:** none new.

- [ ] **Step 1: Inventory the breakage**

Run: `pytest src/qldpc/circuits/surgery/circuit_test.py -q 2>&1 | tail -40`
Expected: failures/errors in tests referencing removed names: `block_observables=`, `keep_only_observable`-on-memory, obs0/obs1 truth-table helpers (`raw_observables`), and any call to `build_single_ppm_circuit` / `build_joint_ppm_circuit` asserting 2 observables.

- [ ] **Step 2: Apply the migration patterns**

For each failing test, rewrite per these patterns (no LER/sinter):

1. **obs0==obs1 truth-table sign check** → folded cross-check: assert the time-like `L` observable (last index, match-basis) equals the GF(2) combination of the block observables that reproduces `L`. Concretely, for a match-basis run, sample raw observables and assert determinism plus the known eigenvalue from `data_init` (use the existing `raw_observables` helper against the new index layout: block at `0..k-1`, `L` at `k`).

```python
def test_match_basis_L_equals_init_eigenvalue():
    code = CSSCode.from_code(HammingCode(3), HammingCode(3))
    xop = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, xop, basis=Pauli.X)
    # init |-> on the logical => L (=X̄) eigenvalue -1 (bit 1)
    init = logical_state_init(code, "-", log_idx=0)  # note: "-" sets Z̄; use "+"/"-" appropriately
    circ = build_single_ppm_circuit(g, rounds=3, experiment_basis=Pauli.X, data_init=init)
    _, obs = circ.compile_detector_sampler().sample(shots=16, separate_observables=True)
    # deterministic; the L observable (index k) matches the prepared eigenvalue
    assert (obs[:, code.dimension] == obs[0, code.dimension]).all()
```

2. **`block_observables=True` count check** → drop the kwarg; assert `num_observables == dimension` (match) or `dimension - 1` (opposite).
3. **`keep_only_observable` on the memory baseline** → delete; memory uses `get_memory_experiment` directly (Task 8).
4. **2-observable (obs0/obs1) assertions** → replace with the `k+1` / `k−1` counts.
5. Tests that asserted detector counts tied to the old reliability slicing → recompute against `_reliable_checks` (the set is the same for match-basis X-gadget, so most should still pass; update only the ones that changed).

Delete tests that only validated the removed obs1 cross-check mechanism (now covered by Task 8 determinism). Keep all DEM-compile, coordinate, and structural-matrix tests.

- [ ] **Step 3: Run the full surgery suite**

Run: `pytest src/qldpc/circuits/surgery/circuit_test.py src/qldpc/circuits/surgery/circuit_single_y_test.py -q`
Expected: PASS (all). If `circuit_single_y_test.py` imports `keep_only_observable`/`block_observables`, leave `keep_only_observable` defined (still used by Y path) and only remove the `block_observables` references.

- [ ] **Step 4: Run the broader surgery module + a smoke import**

Run: `pytest src/qldpc/circuits/surgery -q && python -c "import qldpc.circuits.surgery.circuit"`
Expected: PASS; clean import.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/circuit_test.py src/qldpc/circuits/surgery/circuit_single_y_test.py
git commit -m "test(surgery): migrate obs0/obs1 tests to k+1 / k-1 experiment observables"
```

---

## Self-Review

**Spec coverage:**
- §3.1 match-basis k+1 → Task 6 (single), Task 7 (joint) + first-cycle `L` in `_surgery_observable`.
- §3.2 opposite-basis k−1 + frame correction → Tasks 2, 3, 6, 7.
- §3.3 memory → Task 8 (reuse `get_memory_experiment`, no `keep_only_observable`).
- §3.4 folded cross-check → Task 8 + Task 9 pattern 1.
- §4 GF(2) construction → Tasks 1, 3 (and the solvability⇔commutes refinement is realized via Task 2's kernel basis, so every `w` passed to Task 3 is solvable).
- §5 API (experiment_basis default, remove obs0/obs1/block_observables) → Tasks 6, 7, 9.
- §6 refactor (state_prep/detach/reliability/final-detectors/observable) → Tasks 4, 5, 6.
- §7 scope (no Y/LER) → Global Constraints; Task 9 leaves `keep_only_observable` for the Y path.
- §8 tests → Tasks 6, 7, 8, 9.

**Placeholder scan:** none — every code step has complete code; the only conditional substitution is "pick a k≥2 in-repo CSS code" in Tasks 6/8, with the exact assertion stated.

**Type consistency:** `_surgery_observable` signature in Task 6 matches its calls in Tasks 6 (single) and 7 (joint); `_reliable_checks` signature in Task 5 matches calls in `_surgery_qec_cycle`/`_surgery_final_detectors` and Tasks 6/7; `experiment_basis: PauliXZ` used consistently (import `from qldpc.objects import Pauli, PauliXZ` already present in `circuit.py`). `_gf2_solve` return `np.ndarray | None` consumed by `_block_observable_targets` with the `assert c is not None` guard backed by Task 2 ensuring solvability.
