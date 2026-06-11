# obs0 last-round fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `_surgery_observable`'s `obs0` encode the logical Pauli parity for any number of QEC rounds, by pointing it at the last-round meas-check outcomes instead of XOR-ing across all rounds.

**Architecture:** Single-function body rewrite inside `src/qldpc/circuits/surgery/circuit.py:1012`. Caller signature updates in `build_single_ppm_circuit` (≈ line 380) and `build_joint_ppm_circuit` (≈ line 721) to drop the now-unused `num_rounds=rounds` kwarg. Test additions and one test strengthening in `_test_circuit.py`. No detector changes, no public API changes, no new modules.

**Tech Stack:** Python 3.12, `stim`, `pytest`, `uv`. Tests run via `uv run pytest src/qldpc/circuits/surgery/ -q`.

**Spec:** `docs/superpowers/specs/2026-06-11-obs0-last-round-fix-design.md`

**Naming note:** The Cain-convention rename (`docs/superpowers/specs/2026-06-11-cain-convention-rename-design.md`) is in flight. As of the time this plan was written, `circuit.py` still uses the legacy `chi_check_ids` identifier (already grepped). This plan uses the legacy names throughout to keep the diff mechanical; the cain rename will pick them up later. **If you find `chi_check_ids` has already been renamed to `meas_check_ids` in main when you start, substitute throughout.**

---

## File Structure

| File | Role | Edited by this plan |
|---|---|---|
| `src/qldpc/circuits/surgery/circuit.py` | `_surgery_observable` body + two caller sites + obs0 docstrings | Tasks 4, 5 |
| `src/qldpc/circuits/surgery/_test_circuit.py` | Add 2 new tests; strengthen 1 existing test | Tasks 1, 2, 3 |
| `examples/` (read-only scan) | Verify no external code references break | Task 6 |

No new files. No module split. Same test file holds both new and modified tests.

---

## Task 1: Add failing joint-PPM even-rounds truth-table test

**Files:**
- Modify: `src/qldpc/circuits/surgery/_test_circuit.py` (append new test at end of file, or after the `test_joint_ppm_data_init_*` group ≈ line 945)

- [ ] **Step 1: Write the failing test**

Append after the last `test_joint_ppm_data_init_*` test:

```python
def test_joint_ppm_even_rounds_truth_table():
    """obs0 must encode logical X̄_l X̄_r parity correctly at EVEN rounds.

    Regression test for the bug where _surgery_observable XOR'd χ syndromes
    across all rounds (R · m_v ≡ 0 mod 2 for even R) instead of using a
    single round's product (Webster L2255: Z̄ = ∏_v A_v).
    """
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(codes.SteaneCode(), x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    n = code.num_qudits
    # basis=X, so we sweep ("+", "+"), ("-", "+"), ("+", "-"), ("-", "-").
    # In basis=X, "-" on data flips X̄ to -1; X̄_l X̄_r = product.
    cases = [
        (("+", "+"), 0),
        (("-", "+"), 1),
        (("+", "-"), 1),
        (("-", "-"), 0),
    ]
    for data_init, expected in cases:
        circuit, _ = build_joint_ppm_circuit(
            g_l, g_r, bridge, rounds=2, noise_model=None,
            data_init=data_init,
        )
        sampler = circuit.compile_detector_sampler()
        _, obs = sampler.sample(shots=16, separate_observables=True)
        # obs0 must encode the joint X̄_l X̄_r eigenvalue per shot.
        assert (obs[:, 0] == expected).all(), (
            f"data_init={data_init!r}: obs0 has {(obs[:, 0] != expected).sum()}/"
            f"16 shots disagreeing with expected parity bit {expected}"
        )
        # obs1 (destructive cross-check) must agree with obs0 per shot.
        assert (obs[:, 0] == obs[:, 1]).all(), (
            f"data_init={data_init!r}: obs0 != obs1 in noiseless run"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/qldpc/circuits/surgery/_test_circuit.py::test_joint_ppm_even_rounds_truth_table -v`

Expected: **FAIL** on the `(-,+)` or `(+,-)` case with old code (obs0 is identically 0 for even rounds, but `expected == 1`). The assertion message should mention "16/16 shots disagreeing with expected parity bit 1".

- [ ] **Step 3: Commit the failing test**

```bash
git add src/qldpc/circuits/surgery/_test_circuit.py
git commit -m "test(surgery): joint-PPM even-rounds truth-table (failing, regression for obs0 bug)"
```

---

## Task 2: Add failing single-PPM even-rounds truth-table test

**Files:**
- Modify: `src/qldpc/circuits/surgery/_test_circuit.py` (append new test directly after Task 1's test)

- [ ] **Step 1: Write the failing test**

Append immediately after `test_joint_ppm_even_rounds_truth_table`:

```python
def test_single_ppm_even_rounds_truth_table():
    """obs0 must encode single-patch X̄ (or Z̄) parity at EVEN rounds.

    Same regression as test_joint_ppm_even_rounds_truth_table but for the
    single-patch PPM construction. Sweeps "+" and "-" data inits in basis=X
    and "0", "1" in basis=Z to expose the cumulative-XOR bug at even rounds.
    """
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit, logical_state_init
    code = codes.SteaneCode()
    for basis, cases in [
        (Pauli.X, [("+", 0), ("-", 1)]),
        (Pauli.Z, [("0", 0), ("1", 1)]),
    ]:
        op = (code.get_logical_ops(Pauli.X)[0] if basis is Pauli.X
              else code.get_logical_ops(Pauli.Z)[0])
        op_arr = np.asarray(op).astype(np.uint8)
        g = build_gadget(code, op_arr, basis=basis)
        for state, expected in cases:
            data_init = logical_state_init(code, state=state, log_idx=0)
            circuit = build_single_ppm_circuit(
                g, rounds=2, noise_model=None, data_init=data_init,
            )
            sampler = circuit.compile_detector_sampler()
            _, obs = sampler.sample(shots=16, separate_observables=True)
            assert (obs[:, 0] == expected).all(), (
                f"basis={basis!r} state={state!r}: obs0 has "
                f"{(obs[:, 0] != expected).sum()}/16 shots disagreeing with "
                f"expected parity bit {expected}"
            )
            assert (obs[:, 0] == obs[:, 1]).all(), (
                f"basis={basis!r} state={state!r}: obs0 != obs1 in noiseless run"
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/qldpc/circuits/surgery/_test_circuit.py::test_single_ppm_even_rounds_truth_table -v`

Expected: **FAIL** on `state="-"` (basis=X) and `state="1"` (basis=Z) cases.

- [ ] **Step 3: Commit the failing test**

```bash
git add src/qldpc/circuits/surgery/_test_circuit.py
git commit -m "test(surgery): single-PPM even-rounds truth-table (failing, regression for obs0 bug)"
```

---

## Task 3: Strengthen existing vacuous test into truth-table cross-check

**Files:**
- Modify: `src/qldpc/circuits/surgery/_test_circuit.py:690-705` (`test_build_joint_ppm_circuit_intercode_noiseless_observables_zero`)

- [ ] **Step 1: Replace test body**

Replace lines 690-705 with:

```python
def test_build_joint_ppm_circuit_intercode_noiseless_observables_zero():
    """Cross-check obs0 == obs1 per shot across all 4 parity inits.

    Previously asserted only ``obs.sum() == 0`` for a single |+⟩^n init,
    which was vacuous: parity = +1 trivially gave 0 even with the broken
    XOR-over-rounds obs0. Now sweeps non-trivial parity inits so the bug
    is exposed if it ever returns.
    """
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(codes.SteaneCode(), x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    for data_init in [("+", "+"), ("-", "+"), ("+", "-"), ("-", "-")]:
        circuit, _ = build_joint_ppm_circuit(
            g_l, g_r, bridge, rounds=2, data_init=data_init,
        )
        sampler = circuit.compile_detector_sampler()
        _, obs = sampler.sample(shots=8, separate_observables=True)
        assert (obs[:, 0] == obs[:, 1]).all(), (
            f"data_init={data_init!r}: obs0 disagrees with obs1 on "
            f"{(obs[:, 0] != obs[:, 1]).sum()}/8 noiseless shots"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/qldpc/circuits/surgery/_test_circuit.py::test_build_joint_ppm_circuit_intercode_noiseless_observables_zero -v`

Expected: **FAIL** on `("-", "+")` or `("+", "-")` case because obs0 (old code) is 0 but obs1 is 1 (cross-check correctly reads the -1 eigenvalue).

- [ ] **Step 3: Commit the strengthened test**

```bash
git add src/qldpc/circuits/surgery/_test_circuit.py
git commit -m "test(surgery): strengthen joint-PPM noiseless test to truth-table cross-check"
```

---

## Task 4: Rewrite `_surgery_observable` and drop `num_rounds` from callers

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py:1012-1052` (`_surgery_observable` body + signature)
- Modify: `src/qldpc/circuits/surgery/circuit.py:380-387` (single-PPM caller)
- Modify: `src/qldpc/circuits/surgery/circuit.py:721-728` (joint-PPM caller)

- [ ] **Step 1: Rewrite `_surgery_observable` body**

Replace lines 1012-1052 with:

```python
def _surgery_observable(
    gadget: GadgetLayout,
    *,
    chi_check_ids: tuple[int, ...],
    data_ids: tuple[int, ...],
    support_indices: tuple[int, ...],
    measurement_record: MeasurementRecord,
) -> stim.Circuit:
    """Emit two OBSERVABLE_INCLUDE entries (obs0, obs1) for the surgery PPM.

    obs0 — physical readout of the logical Pauli. The merged stabilizer group
        satisfies the single-round identity Z̄ = ∏_{v ∈ support} A_v (Webster
        et al. arXiv:2410.03628 §II.A and L2255 of the v4 PDF). We point
        ``OBSERVABLE_INCLUDE`` at the **last** QEC round's meas-check outcomes
        — their XOR is the eigenvalue bit of Z̄ (or X̄ for basis=X). Detectors
        carry the FT load via round-to-round consistency; following Cohen et
        al. arXiv:2407.18393 §3.5 the final round is the natural readout point.

    obs1 — Direct stim measurement of the data qubits on ``support``. NOT a
        physical protocol — destructively projects the data — but a useful
        noiseless cross-check: in any noiseless shot ``obs0 == obs1``.

    For LER sweeps and any noisy run, keep ONLY obs0 via
    ``keep_only_observable(circuit, keep_idx=0)``.
    """
    # Precondition: every meas-check ancilla must have been measured during
    # the QEC cycle. Detach/readout only touches data + κ + bridge, never the
    # meas-check ancillas, so ``get_target_rec(cid)`` (default -1) is
    # guaranteed to resolve to the last QEC round. Fail loudly if a future
    # refactor breaks this.
    for cid in chi_check_ids:
        assert measurement_record[cid], (
            f"meas-check {cid} has no measurement record; "
            f"_surgery_observable expects the QEC cycle to have run first."
        )
    circuit = stim.Circuit()
    chi_targets = [
        measurement_record.get_target_rec(cid) for cid in chi_check_ids
    ]
    circuit.append("OBSERVABLE_INCLUDE", chi_targets, 0)
    data_targets = [
        measurement_record.get_target_rec(data_ids[i]) for i in support_indices
    ]
    circuit.append("OBSERVABLE_INCLUDE", data_targets, 1)
    return circuit
```

- [ ] **Step 2: Drop `num_rounds=rounds` from single-PPM caller**

In `build_single_ppm_circuit`, modify the call at line 380-387. Locate this block:

```python
    circuit += _surgery_observable(
        gadget,
        chi_check_ids=chi_check_ids,
        data_ids=data_ids,
        support_indices=gadget.support,
        num_rounds=rounds,
        measurement_record=measurement_record,
    )
```

Replace with:

```python
    circuit += _surgery_observable(
        gadget,
        chi_check_ids=chi_check_ids,
        data_ids=data_ids,
        support_indices=gadget.support,
        measurement_record=measurement_record,
    )
```

- [ ] **Step 3: Drop `num_rounds=rounds` from joint-PPM caller**

In `build_joint_ppm_circuit`, modify the call at line 721-728. Locate:

```python
    circuit += _surgery_observable(
        g_l,
        chi_check_ids=chi_check_ids,
        data_ids=data_ids,
        support_indices=support_combined,
        num_rounds=rounds,
        measurement_record=measurement_record,
    )
```

Replace with:

```python
    circuit += _surgery_observable(
        g_l,
        chi_check_ids=chi_check_ids,
        data_ids=data_ids,
        support_indices=support_combined,
        measurement_record=measurement_record,
    )
```

- [ ] **Step 4: Run new + strengthened tests to verify they now pass**

Run: `uv run pytest src/qldpc/circuits/surgery/_test_circuit.py::test_joint_ppm_even_rounds_truth_table src/qldpc/circuits/surgery/_test_circuit.py::test_single_ppm_even_rounds_truth_table src/qldpc/circuits/surgery/_test_circuit.py::test_build_joint_ppm_circuit_intercode_noiseless_observables_zero -v`

Expected: **3 PASSED**.

- [ ] **Step 5: Run full surgery test suite to verify no regressions**

Run: `uv run pytest src/qldpc/circuits/surgery/ -q`

Expected: all tests pass (count = N existing + 2 new, where N is the pre-fix passing count).

- [ ] **Step 6: Commit the implementation**

```bash
git add src/qldpc/circuits/surgery/circuit.py
git commit -m "fix(surgery): obs0 reads last-round meas-checks (single-round Z̄=∏A_v identity)

The cumulative XOR over rounds was equivalent to R·m_v mod 2: correct for
odd R but identically 0 for even R, silently breaking parity readout on
common BB-code distances d∈{6,10,12,18}. Use the single-round Webster
identity (L2255 of arXiv:2410.03628) and Cohen's final-round convention
(arXiv:2407.18393 §3.5) instead. Detectors are unchanged so FT distance
is preserved."
```

---

## Task 5: Update obs0 description in `build_*_ppm_circuit` docstrings

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py:337-348` (`build_single_ppm_circuit` docstring obs0 lines)
- Modify: `src/qldpc/circuits/surgery/circuit.py:640-650` (`build_joint_ppm_circuit` docstring obs0 lines)

- [ ] **Step 1: Find current obs0 description in single-PPM docstring**

Run: `grep -n "obs0\|Webster Eq" src/qldpc/circuits/surgery/circuit.py | head -10`

You should see lines roughly matching:

```
337:    Emits two OBSERVABLE_INCLUDE entries (see ``_surgery_observable`` for
640:    Emits two OBSERVABLE_INCLUDE entries (see ``_surgery_observable`` for
643:      * obs0 — Webster Eq. 1, the physical syndrome-based readout of
```

Confirm exact line numbers in your tree.

- [ ] **Step 2: Update single-PPM `build_single_ppm_circuit` docstring**

Inside `build_single_ppm_circuit` (around line 337), the docstring currently says (verify by reading the block):

> "Emits two OBSERVABLE_INCLUDE entries (see ``_surgery_observable`` for full semantics): * obs0 — Webster Eq. 1 ..."

Replace any phrasing along the lines of "Webster Eq. 1, the physical syndrome-based readout ... from intermediate-round syndromes only" with:

```
      * obs0 — Single-round Z̄ = ∏_{v ∈ support} A_v readout (Webster et al.
        arXiv:2410.03628 L2255 of v4). XOR of the **last** QEC round's
        meas-check outcomes. The repeated rounds give FT distance via the
        detector layer; following Cohen et al. arXiv:2407.18393 §3.5 we
        read the logical eigenvalue from the final round.
      * obs1 — Direct destructive M on ``support`` qubits; noiseless
        cross-check, not a physical protocol.
```

(Match the surrounding docstring style — bullet markers, indentation, and any preceding context lines stay as they are.)

- [ ] **Step 3: Update joint-PPM `build_joint_ppm_circuit` docstring**

Inside `build_joint_ppm_circuit` (around line 640), apply the same replacement. The current text mentions "obs0 — Webster Eq. 1, the physical syndrome-based readout of X̄_l ⊗ X̄_r (or Z̄_l ⊗ Z̄_r for basis=Z)". Rewrite to:

```
      * obs0 — Single-round joint readout via Webster's identity
        ∏_{v ∈ support_l ∪ support_r} A_v = X̄_l ⊗ X̄_r (or Z̄_l ⊗ Z̄_r for
        basis=Z). See Webster et al. arXiv:2410.03628 L2255 of v4. XOR of
        the **last** QEC round's meas-check outcomes on both patches.
        Detectors carry the FT load; following Cohen et al.
        arXiv:2407.18393 §3.5 the final round is the readout point.
      * obs1 — Direct destructive M on ``support_l ∪ support_r``; noiseless
        cross-check, not a physical protocol.
```

- [ ] **Step 4: Run full surgery test suite to confirm docstring-only edit didn't break anything**

Run: `uv run pytest src/qldpc/circuits/surgery/ -q`

Expected: same pass count as Task 4.

- [ ] **Step 5: Commit the docstring updates**

```bash
git add src/qldpc/circuits/surgery/circuit.py
git commit -m "docs(surgery): obs0 docstrings reflect single-round Webster identity"
```

---

## Task 6: Smoke-check external scripts and notebooks

**Files:**
- Read-only scan: `examples/scripts/`, `examples/logical_error_rates/`

- [ ] **Step 1: Grep for any code that manipulates `OBSERVABLE_INCLUDE` by raw index**

Run: `git grep -nE "OBSERVABLE_INCLUDE|keep_only_observable|separate_observables" examples/`

Expected output: only `keep_only_observable(circuit, keep_idx=0)` and `separate_observables=True` invocations. No direct `OBSERVABLE_INCLUDE` writes from external scripts.

If you see anything else, read that file and confirm it still makes sense after the obs0 semantics change (it should — `keep_idx=0` continues to be the joint parity readout, just now correct for any rounds parity).

- [ ] **Step 2: Run one LER script end-to-end as smoke check**

Pick a fast script (under a minute). Run:

```bash
uv run python examples/scripts/single_ppm_vs_memory_ler.py --help 2>/dev/null || \
  uv run python examples/scripts/single_ppm_vs_memory_ler.py
```

(If the script has no `--help` it just runs; abort with Ctrl+C after one or two LER points come through.)

Expected: script completes (or starts producing LER values) without error. No regression in the LER output (compare to a known-good prior run only if you have one — otherwise just confirm it doesn't error out).

- [ ] **Step 3: Verify success criteria from spec**

Run these checks one by one:

```bash
# Success criterion 1: full suite is green
uv run pytest src/qldpc/circuits/surgery/ -q

# Success criterion 2: no cumulative-XOR loop survives inside _surgery_observable
grep -nE 'for r in range\(num_rounds\)' src/qldpc/circuits/surgery/circuit.py
# Expected: NO output (loop is gone)

# Success criterion 3: num_rounds argument removed from _surgery_observable
grep -n 'def _surgery_observable' src/qldpc/circuits/surgery/circuit.py
# Then read the signature lines below — confirm num_rounds is NOT a parameter

# Success criterion 4: new tests are in the file
grep -n 'def test_joint_ppm_even_rounds_truth_table\|def test_single_ppm_even_rounds_truth_table' \
  src/qldpc/circuits/surgery/_test_circuit.py
# Expected: two matches
```

- [ ] **Step 4: Final summary (no commit needed for this task)**

No code changes in this task — it's a verification gate. The implementation is complete after Task 5's commit. Report back:

* total commits added by this plan (expected 5: Task 1, Task 2, Task 3, Task 4, Task 5)
* total tests added (2 new) and strengthened (1)
* whether external scripts still work

---

## Risks and rollback

* **If Task 4 fails an existing test other than the three modified in Tasks 1-3**, do NOT continue. The most likely candidates are tests that previously relied on the XOR-over-rounds being identically 0 (vacuous passes), or tests that pass a `num_rounds=` kwarg through to `_surgery_observable` directly (none expected — it's private). Read the failing test, check whether it was vacuously passing or genuinely correct, then either strengthen it (if vacuous, add to Task 3) or revert and investigate.
* **If the precondition assertion fires** (`measurement_record[cid]` falsy), the circuit assembly order has changed since this plan was written. Locate where `_surgery_qec_cycle` / `_surgery_qec_cycle_joint` is invoked relative to `_surgery_observable` and confirm the meas-check ancillas are measured in between.
* **Rollback**: each task is a single commit. `git revert <sha>` undoes that task only. Tasks 1-3 (test additions) can stand alone if Task 4 is reverted — they then act as failing-test documentation of the bug until the fix returns.
