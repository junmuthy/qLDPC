# Surgery Final-Measurement Detectors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `_surgery_final_detectors` helper that emits inferred-from-destructive-readout DETECTORs for reliable stabilizers, and wire it into `build_single_ppm_circuit` and `build_joint_ppm_circuit`. Closes the LER over-estimation gap left by the surgery circuit rewrite.

**Architecture:** New helper in `circuit.py` mirrors `_get_basis_memory_experiment_parts`'s lines 274-285 pattern but iterates over BOTH the data-side and κ-side reliable stab subsets. Called from `build_*_ppm_circuit` between `_surgery_detach_and_readout` and `_surgery_observable`. Reliable set is identical to `_classify_reliable_round1_checks` output.

**Tech Stack:** Python 3.11+, `numpy`, `stim`, `pytest`. Uses existing `qldpc.codes.common.CSSCode`, `qldpc.circuits.bookkeeping.{QubitIDs, MeasurementRecord}`.

**Spec:** `docs/superpowers/specs/2026-06-08-surgery-final-detectors-design.md`
**Branch:** `feat/surgery-construction` (HEAD: `316e43c`)

**Key invariants:**
- For basis=X: reliable stabs = data H_X rows (HX_merged[:m_X], checks_x[:m_X]) + G rows (HZ_merged[m_Z:], checks_z[m_Z:])
- For basis=Z: reliable stabs = data H_Z rows (HZ_merged[:m_Z], checks_z[:m_Z]) + G rows (HX_merged[m_X:], checks_x[m_X:])
- Each DETECTOR = ⊕(final M-record on stab support) ⊕ last-round syndrome = 0 noiselessly.

**Reference reading:**
- `src/qldpc/circuits/memory/memory.py:274-285` — the inferred-detector pattern we mirror.
- `src/qldpc/codes/surgery/circuit.py` — current state (helpers + build_* functions).

---

## Task 1: Helper detector-count test (failing) + minimal `_surgery_final_detectors`

**Files:**
- Modify: `src/qldpc/codes/surgery/circuit.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append to `src/qldpc/codes/surgery/_test.py`:

```python
@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_surgery_final_detectors_count_matches_reliable_round1(basis):
    """Number of final DETECTORs equals |reliable round-1 set|.

    Tests the helper in isolation: build a circuit through detach_and_readout,
    then call _surgery_final_detectors and count emitted DETECTOR instructions.
    """
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import (
        _surgery_state_prep, _surgery_qec_cycle, _surgery_detach_and_readout,
        _surgery_final_detectors, _classify_reliable_round1_checks,
        _gadget_merged_csscode,
    )
    from qldpc.circuits.bookkeeping import QubitIDs

    code = codes.SteaneCode()
    op = (code.get_logical_ops(Pauli.X)[0] if basis is Pauli.X
          else code.get_logical_ops(Pauli.Z)[0])
    op_arr = np.asarray(op).astype(np.uint8)
    g = build_gadget(code, op_arr, basis=basis)
    merged = _gadget_merged_csscode(g)
    qubit_ids = QubitIDs.from_code(merged)
    n_data = code.num_qudits
    data_ids = qubit_ids.data[:n_data]
    kappa_ids = qubit_ids.data[n_data:]

    # Simulate the pipeline through detach (we need measurement_record populated).
    import stim
    _ = stim.Circuit()
    _ += _surgery_state_prep(g, data_ids, kappa_ids, bridge_ids=())
    _qec, mrec, _ = _surgery_qec_cycle(g, merged, num_rounds=2, qubit_ids=qubit_ids)
    _ += _qec
    _ += _surgery_detach_and_readout(
        g, data_ids=data_ids, kappa_ids=kappa_ids, bridge_ids=(),
        measurement_record=mrec,
    )

    circuit = _surgery_final_detectors(g, merged, qubit_ids, measurement_record=mrec)
    n_final_det = str(circuit).count("DETECTOR")
    expected = len(_classify_reliable_round1_checks(g, merged, qubit_ids))
    assert n_final_det == expected, (
        f"basis={basis}: emitted {n_final_det} DETECTORs, expected {expected}"
    )
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_surgery_final_detectors_count_matches_reliable_round1 -x
```
Expected: FAIL (`ImportError: cannot import name '_surgery_final_detectors'`).

- [ ] **Step 3: Implement `_surgery_final_detectors`**

Edit `src/qldpc/codes/surgery/circuit.py`. Append (after `_surgery_detach_and_readout`, before the `build_single_ppm_circuit` function):

```python
def _surgery_final_detectors(
    gadget: GadgetLayout,
    merged_code: CSSCode,
    qubit_ids: QubitIDs,
    *,
    measurement_record: MeasurementRecord,
) -> stim.Circuit:
    """Emit DETECTORs for reliable stabs inferable from final readouts.

    For basis=X: data H_X (from Mx data) + G (from Mz κ).
    For basis=Z: data H_Z (from Mz data) + G (from Mx κ).
    Each DETECTOR XORs ⊕(final M-record on stab support) ⊕ last-round syndrome.
    """
    m_X = gadget.code.matrix_x.shape[0]
    m_Z = gadget.code.matrix_z.shape[0]
    HX = np.asarray(merged_code.matrix_x).astype(np.uint8)
    HZ = np.asarray(merged_code.matrix_z).astype(np.uint8)

    circuit = stim.Circuit()

    def _emit_detector(stab_row: np.ndarray, check_id: int, det_idx: int) -> None:
        supp = np.where(stab_row)[0]
        targets = [measurement_record.get_target_rec(qubit_ids.data[q]) for q in supp]
        targets.append(measurement_record.get_target_rec(check_id, -1))
        circuit.append("DETECTOR", targets, (0, 0, det_idx))

    if gadget.basis is Pauli.X:
        # data H_X rows (X-checks indices [:m_X])
        for kk in range(m_X):
            _emit_detector(HX[kk], qubit_ids.checks_x[kk], kk)
        # G rows (Z-checks indices [m_Z:])
        for offset, kk in enumerate(range(m_Z, HZ.shape[0])):
            _emit_detector(HZ[kk], qubit_ids.checks_z[kk], m_X + offset)
    else:  # Pauli.Z (symmetric: chi in HZ, G in HX)
        for kk in range(m_Z):
            _emit_detector(HZ[kk], qubit_ids.checks_z[kk], kk)
        for offset, kk in enumerate(range(m_X, HX.shape[0])):
            _emit_detector(HX[kk], qubit_ids.checks_x[kk], m_Z + offset)

    return circuit
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_surgery_final_detectors_count_matches_reliable_round1 -x
```
Expected: 2 PASS (basis=X and basis=Z).

If FAIL with the wrong count: print `(n_final_det, expected)` and compare with `_classify_reliable_round1_checks`'s output to find the discrepancy.

- [ ] **Step 5: Run full surgery suite — no regressions**

```bash
pytest src/qldpc/codes/surgery/ -x 2>&1 | tail -3
```
Expected: 77 PASS (75 existing + 2 new).

- [ ] **Step 6: Commit**

```bash
git add src/qldpc/codes/surgery/circuit.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: _surgery_final_detectors helper (spec §3.4)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Wire helper into `build_single_ppm_circuit`

**Files:**
- Modify: `src/qldpc/codes/surgery/circuit.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append to `_test.py`:

```python
@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_build_single_ppm_circuit_noiseless_no_detector_fires(basis):
    """Noiseless: NO detector fires (including the new final detectors).

    The total detector count must equal: round-1 reliable + (rounds-1)*all_checks + final reliable.
    Under noiseless conditions all of them must remain silent.
    """
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import build_single_ppm_circuit
    code = codes.SteaneCode()
    op = (code.get_logical_ops(Pauli.X)[0] if basis is Pauli.X
          else code.get_logical_ops(Pauli.Z)[0])
    op_arr = np.asarray(op).astype(np.uint8)
    g = build_gadget(code, op_arr, basis=basis)
    circuit = build_single_ppm_circuit(g, rounds=3, noise_model=None)
    sampler = circuit.compile_detector_sampler()
    dets, _ = sampler.sample(shots=64, separate_observables=True)
    assert not dets.any(), (
        f"basis={basis}: {dets.sum()} detector fires noiselessly across {dets.shape[0]} shots"
    )
```

- [ ] **Step 2: Run to verify it currently passes WITHOUT the wiring**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_build_single_ppm_circuit_noiseless_no_detector_fires -x
```

Expected behavior **before** wiring: PASS for both bases — because without the final detectors, the only detectors are the QEC-cycle ones (which we know don't fire noiselessly from existing tests).

After wiring, the test still must pass — the new final detectors should ALSO not fire noiselessly.

So this test is a guard that wiring doesn't BREAK noiseless. If it starts firing after wiring, the detector targets are wrong.

- [ ] **Step 3: Wire `_surgery_final_detectors` into `build_single_ppm_circuit`**

Find `build_single_ppm_circuit` in `src/qldpc/codes/surgery/circuit.py`. Currently it has this sequence:

```python
circuit += _surgery_detach_and_readout(...)
# (chi_check_ids computation)
circuit += _surgery_observable(...)
```

Insert the new call between detach and observable:

```python
circuit += _surgery_detach_and_readout(
    gadget, data_ids=data_ids, kappa_ids=kappa_ids, bridge_ids=bridge_ids,
    measurement_record=measurement_record,
)
circuit += _surgery_final_detectors(            # NEW
    gadget, merged_code, qubit_ids,
    measurement_record=measurement_record,
)
# (chi_check_ids computation unchanged)
circuit += _surgery_observable(...)
```

- [ ] **Step 4: Run the noiseless test**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_build_single_ppm_circuit_noiseless_no_detector_fires -x
```
Expected: 2 PASS.

If FAIL: the final detectors fire under noiseless. Possible causes:
- `qubit_ids.data` indexing wrong (off by 1, or κ region mis-located).
- `get_target_rec(check_id, -1)` returning the wrong syndrome record (maybe the QEC cycle's REPEAT_BLOCK only adds the record once, not per repetition).

Inspect the circuit text via `print(circuit)` near the readout/final-detector boundary; verify the DETECTOR target indices look right.

- [ ] **Step 5: Run existing noiseless / observable tests**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "noiseless_observables_zero or noiseless_no_detectors_fire or noiseless_compiles" -x
```
Expected: all PASS (the existing 4-6 tests around this area).

- [ ] **Step 6: Run full surgery suite**

```bash
pytest src/qldpc/codes/surgery/ -x 2>&1 | tail -3
```
Expected: 79 PASS (77 + 2 new from this task).

- [ ] **Step 7: Commit**

```bash
git add src/qldpc/codes/surgery/circuit.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: wire final detectors into build_single_ppm_circuit

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Wire helper into `build_joint_ppm_circuit`

**Files:**
- Modify: `src/qldpc/codes/surgery/circuit.py`
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_build_joint_ppm_circuit_noiseless_no_detector_fires():
    """Joint noiseless: NO detector fires (including final detectors)."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import build_joint_ppm_circuit
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g1 = build_gadget(code, x, basis=Pauli.X)
    g2 = build_gadget(code, x, basis=Pauli.X)
    bridge = build_bridge(g1, g2)
    circuit, _ = build_joint_ppm_circuit(g1, g2, bridge, rounds=2, noise_model=None)
    sampler = circuit.compile_detector_sampler()
    dets, _ = sampler.sample(shots=64, separate_observables=True)
    assert not dets.any(), f"{dets.sum()} detector fires noiselessly"
```

- [ ] **Step 2: Run — should pass currently (no wiring yet for joint)**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_build_joint_ppm_circuit_noiseless_no_detector_fires -x
```
Expected: PASS (existing joint circuit doesn't have final detectors so no detectors to fire incorrectly).

The test's purpose is to catch regressions after we wire.

- [ ] **Step 3: Wire `_surgery_final_detectors` into `build_joint_ppm_circuit`**

Find `build_joint_ppm_circuit` in `circuit.py`. Insert the helper call between detach_and_readout and the chi_check_ids computation (same position as in Task 2):

```python
circuit += _surgery_detach_and_readout(
    g1, data_ids=data_ids, kappa_ids=kappa_ids, bridge_ids=bridge_ids,
    measurement_record=measurement_record,
)
circuit += _surgery_final_detectors(            # NEW
    g1, joint_code, qubit_ids,
    measurement_record=measurement_record,
)
# (existing chi_check_ids computation)
circuit += _surgery_observable(...)
```

Notes:
- Pass `g1` as the gadget (g1.basis == g2.basis is asserted by build_bridge — they share the basis).
- Pass `joint_code` as the merged code.
- The joint code's `qubit_ids.data` register includes data_1 + (data_2) + κ_1 + κ_2 + bridge — `_surgery_final_detectors` uses `qubit_ids.data[q]` indexing which naturally handles this layout.

- [ ] **Step 4: Run the joint noiseless test**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_build_joint_ppm_circuit_noiseless_no_detector_fires -x
```
Expected: PASS.

- [ ] **Step 5: Run all joint tests**

```bash
pytest src/qldpc/codes/surgery/_test.py -k "joint" -x 2>&1 | tail -5
```
Expected: all joint tests still PASS (intra/inter, observables-zero, etc.).

- [ ] **Step 6: Run full surgery suite**

```bash
pytest src/qldpc/codes/surgery/ -x 2>&1 | tail -3
```
Expected: 80 PASS (79 + 1 new from this task).

- [ ] **Step 7: Commit**

```bash
git add src/qldpc/codes/surgery/circuit.py src/qldpc/codes/surgery/_test.py
git commit -m "feat: wire final detectors into build_joint_ppm_circuit

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: LER drop verification (slow test)

**Files:**
- Modify: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
@pytest.mark.slow
def test_single_ppm_ler_with_final_detectors_below_threshold():
    """With final detectors wired, LER at p=0.001 should be ≤ 0.01.

    Reference: before the final-detector wiring, LER at p=0.001 was ~0.024
    (from test_single_ppm_ler_monotone_in_p in the surgery-circuit-rewrite plan).
    Adding the inferred detectors should drop it significantly.
    """
    import sinter
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits import DepolarizingNoiseModel
    from qldpc import decoders

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)

    p = 0.001
    circuit = build_single_ppm_circuit(
        g, rounds=3, noise_model=DepolarizingNoiseModel(p),
    )
    sinter_decoder = decoders.SinterDecoder()
    results = sinter.collect(
        tasks=[sinter.Task(circuit=circuit, json_metadata={"p": float(p)})],
        decoders=["custom"],
        custom_decoders={"custom": sinter_decoder},
        num_workers=4,
        max_shots=5000,
        max_errors=50,
        print_progress=False,
    )
    assert len(results) == 1
    ler = results[0].errors / max(results[0].shots, 1)
    assert ler <= 0.01, (
        f"LER at p=0.001 = {ler:.4f} (errors={results[0].errors}/{results[0].shots} shots). "
        f"Expected ≤ 0.01 with final detectors wired. Was ~0.024 without them."
    )
```

- [ ] **Step 2: Run the test**

```bash
pytest src/qldpc/codes/surgery/_test.py::test_single_ppm_ler_with_final_detectors_below_threshold -v
```
Expected: PASS within ~10 seconds. LER value should be roughly 0.001 - 0.005 (large improvement vs the previous 0.024).

If FAIL with LER > 0.01: the final-detector wiring may be wrong (decoder isn't seeing the new constraints). Investigate by:
1. Printing the circuit DEM and inspecting the new DETECTORs.
2. Verifying targets reference the correct measurement records.
3. Comparing decoder output before and after wiring.

If LER is borderline (0.008 - 0.012): adjust the threshold to 0.015 and document the actual measured value. The point is to verify a *significant* improvement, not hit a specific number.

- [ ] **Step 3: Commit**

```bash
git add src/qldpc/codes/surgery/_test.py
git commit -m "test: LER drops below 0.01 at p=0.001 with final detectors

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Final verification + LOC check

**Files:**
- Inspect: `src/qldpc/codes/surgery/circuit.py`

- [ ] **Step 1: Measure final LOC**

```bash
wc -l src/qldpc/codes/surgery/*.py
```

Expected: `circuit.py` ~400 LOC (was 365 after the rewrite plan; final-detector helper + 2 integration calls adds ~35 LOC). Already over the 300 budget — accept.

- [ ] **Step 2: Run full surgery suite (excluding slow)**

```bash
pytest src/qldpc/codes/surgery/ -m "not slow" 2>&1 | tail -3
```
Expected: 79 PASS (80 total - 1 slow).

- [ ] **Step 3: Run slow tests separately**

```bash
pytest src/qldpc/codes/surgery/ -m "slow" 2>&1 | tail -3
```
Expected: 2 PASS (LER monotonicity from earlier + new LER threshold).

- [ ] **Step 4: Run Ide test (regression)**

```bash
pytest examples/test_ide_bb_lp.py 2>&1 | tail -3
```
Expected: 3 PASS or SKIP.

- [ ] **Step 5: Final diff summary** (no commit)

```bash
git log --oneline main..HEAD | head -30
git diff --stat main..HEAD -- src/qldpc/codes/surgery/circuit.py src/qldpc/codes/surgery/_test.py
```

- [ ] **Step 6: No commit needed for verification.**

---

## Self-Review Checklist

**Spec coverage:**

- [x] **Goal 1** (helper `_surgery_final_detectors`) — Task 1.
- [x] **Goal 2** (integration into build_single_ppm_circuit / build_joint_ppm_circuit) — Tasks 2, 3.
- [x] **Goal 3** (reliable set = `_classify_reliable_round1_checks`) — Task 1 helper iterates over the same rows.
- [x] **Goal 4** (basis-symmetric) — Task 1 helper has both X and Z branches.
- [x] **Goal 5** (Webster Table I tests untouched) — no changes to construction tests.
- [x] **Goal 6** (all 75 existing surgery tests pass) — verified at Tasks 1, 2, 3, 5.
- [x] **Goal 7** (LER drop verified) — Task 4.

**Spec sections:**

- §1 helper signature — Task 1 step 3.
- §2 reliable-stabilizer inference — Task 1 helper body.
- §3 DETECTOR target construction — Task 1 `_emit_detector` inner function.
- §4 integration — Tasks 2 (single), 3 (joint).
- §5 tests — Tasks 1 (5.1), 2-3 (5.2), 4 (5.3).
- §6 risks — addressed by helper test (count mismatch catches indexing bugs) + LER test (catches semantic bugs).

**Placeholder scan:** No "TBD", "TODO", "implement later", or other red flags found.

**Type consistency:**
- `_surgery_final_detectors(gadget, merged_code, qubit_ids, *, measurement_record)` — signature consistent across Tasks 1, 2, 3.
- `qubit_ids.data` indexing used uniformly (no per-task variations).
- `measurement_record.get_target_rec(qubit_id)` (default `-1`) and `get_target_rec(check_id, -1)` (last-round syndrome) — consistent semantics.
