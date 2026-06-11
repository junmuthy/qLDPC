# Surgery math-invariants test hardening

**Date:** 2026-06-11
**Goal:** Add 4 targeted pytest cases that close the highest-risk math-layer gaps in the surgery module, so any future regression (in `circuit.py` Stim wiring, `circuit.py` joint-code stitching, or `gadget.py` algebraic helpers) is caught loudly instead of silently.

## Motivation

After the recent `logical_state_init` work, the surgery module's interface is clean and the existing 127 pytest cases cover the bulk of important invariants:

* CSS commutation `HX @ HZ.T = 0`
* Webster Table I `(κ, χ, r)` exact reproduction on 4 GB codes
* Cain Table III bb_18 `(κ, χ, G, degree)` reproduction
* Cross Thm 6 distance preservation on `[[72, 12, 6]]` (notebook §3.3)
* Webster Eq.1 ≡ direct X̄/Z̄_M (noiseless `obs0 == obs1`)
* `logical_state_init` correctness on even-wt Z̄ codes
* Bridge basis-mismatch rejection, cellulation cycle cap, etc.

A code review of the four math-bearing files (`gadget.py`, `bridge.py`, `cheeger.py`, `circuit.py`) plus their test suites identifies four math-correctness claims that are NOT yet tested directly:

1. **Multi-round invariance.** The notebook truth tables and PPM integration tests all run at `rounds=3`. The `_surgery_qec_cycle` loop emits DETECTORs and `SHIFT_COORDS (0,0,1)` per round, and the obs0 XOR pulls χ-check measurements across all rounds. A miscount in the round index (off-by-one in `SHIFT_COORDS` timing, wrong base for `MeasurementRecord.get_target_rec(...,-1-r)`, etc.) would slip past every existing test because all of them use the same `rounds=3` budget. The protocol is supposed to work for any odd `rounds`; we should verify it does, and also that even `rounds` gives the documented identity outcome.

2. **Stim circuit error response.** Existing noiseless tests verify "no detectors fire under noiseless evolution". They do NOT verify that a specific physical error triggers the correct detector pattern. A swap of two CX qubits, an off-by-one in the EdgeColoring schedule, or a missing measurement basis flip would produce a noiseless circuit that still passes all existing tests (because the abstract gadget protocol is symmetric and the Pauli frame absorbs many wiring errors). Such a bug would only surface as a degraded LER curve — i.e. silently break error correction. A direct "inject X on qubit i, assert only Z-stab j ∋ i fires" test catches this class of bug.

3. **Joint code dimension.** `_stitch_intercode` and `_stitch_intracode` build the joint CSS code by stacking χ-rows, co-carrier rows, G rows, and adapter cycle rows. Many tests check CSS commutation on the result, but none assert the joint code's logical dimension matches the algebraic prediction (`k_l + k_r − 1` for intercode Z̄_l ⊗ Z̄_r measurement; `k − 1` for intracode self-measurement). A missing or duplicated stabilizer row in the stitching code would change `joint_code.dimension` while still satisfying CSS commutation.

4. **Gauge-fix row independence.** `_step2_gauge_fix(F)` computes `G` as a basis for `ker(F^T)`. Webster §II.A step 3 requires `|S_L| − wt(L) + 1` independent constraints. The existing test verifies `F @ G^T == 0` but not that the G rows are linearly independent. A degenerate F could let the gauge fix return redundant rows, inflating the gadget's reported `G.shape[0]` without changing the actual gauge structure — exactly the kind of bug that would silently break the Cain Table III bb_18 `G=20` reproduction in a subtle way.

These four gaps are not academic. They are concrete bug patterns that the existing test surface cannot detect.

## Scope

**In scope:**

* 4 pytest cases (see Section "Test cases" below) added to existing test files
* All four cover algebraic / Stim-wiring claims, not statistical decoder behavior
* No source code changes — the tests assert claims that the production code already promises

**Out of scope:**

* `boost_gadget` distance preservation post-boost (Test 5 in the brainstorming session — already covered by the BP+OSD-verified `boost_gadget_distance` strategy and partially by the bb_18 reproduction in notebook §3.2)
* Property-based / hypothesis-style randomized invariant testing (would be slow, not enough payoff for current goal)
* External paper-fixture reproduction (Swaroop arXiv:2410.03628 BB↔LP exact-match) — useful for publishable confidence, not production confidence
* `build_gadget_augmented` commutation hardening — `boost_gadget` already exercises this through its augmentation path; an explicit test is nice-to-have but not load-bearing
* Decoder LER curve validation — orthogonal to surgery protocol correctness

## Test cases

### Test 1: Multi-round invariance on Steane

**File:** `src/qldpc/circuits/surgery/_test_circuit.py`

**Name:** `test_multi_round_invariance_steane_basis_z`

**Spec:**
* Build Steane gadget for `Z̄_0` with `basis=Pauli.Z`.
* Parametrize over `rounds ∈ {1, 2, 3, 5, 10}` and `state ∈ {"0", "1"}` (Z-eigenstates, deterministic outcome).
* For each combination, build the single-PPM circuit with that round count and `data_init=logical_state_init(steane, state, log_idx=0)`.
* Use the raw-measurement-XOR pattern (see `test_logical_state_init_end_to_end_steane_basis_z`) to compute obs0.
* Assert obs0 deterministically equals `int(state)` for ALL combinations.

**What it catches:** A round-indexing bug in `_surgery_qec_cycle`, `_surgery_observable`, or `MeasurementRecord.get_target_rec(...,-1-r)` that drifts as the round count changes.

**Why this and not a noisy LER sweep:** Noiseless determinism is a stronger signal — any mismatch is a wiring bug, no statistical interpretation needed.

### Test 2: Single-qubit X error triggers exactly the Z-checks on its support

**File:** `src/qldpc/circuits/surgery/_test_circuit.py`

**Name:** `test_single_data_x_error_triggers_only_neighboring_z_checks_steane`

**Spec:**
* Build Steane gadget with `basis=Pauli.Z`, single-PPM circuit, `rounds=3`, `noise_model=None`.
* For each data qubit `i ∈ {0, ..., 6}`:
  * Construct a modified circuit by inserting `X_ERROR(1.0) data[i]` immediately after the first state-prep block (before any QEC round).
  * Use `compile_detector_sampler().sample(shots=1, separate_observables=True)` to draw one shot.
  * Compute the expected detector set: any Z-stabilizer row of the Steane code whose support contains `i`, in round 1, should fire. Other detectors (X-stabs, χ-rows, G-rows, later rounds without further errors) should NOT fire.
  * Assert the sampled `detection_events` mask matches this expected pattern exactly.

**What it catches:** A swap in CX target/control, wrong measurement basis on a check ancilla, EdgeColoring scheduling bug that delays/dispatches the syndrome to the wrong round, missing measurement reset, etc.

**Why Steane × all 7 qubits:** Steane is the smallest CSS code with full logical structure, and 7 is small enough to exhaustively iterate. Catching the bug class needs at least one error per qubit; one fixed qubit could pass by accident.

### Test 3: Joint code dimension matches algebraic formula

**File:** `src/qldpc/circuits/surgery/_test_circuit.py`

**Name:** `test_joint_code_dimension_matches_kl_plus_kr_minus_one`

**Spec:**
* Construct two test cases:
  * **Intercode Steane × Steane** (basis=Z, measuring Z̄_l ⊗ Z̄_r): assert `joint_code.dimension == 1` (`k_l + k_r − 1 = 1 + 1 − 1`).
  * **Intercode Webster GB code 0 × Steane**: `k_l = 10`, `k_r = 1`, expect `joint_code.dimension == 10`.
* Use the existing `_test_webster_fixture.py` helpers to construct the Webster code; use `_seed_op` (or copy the inline `_webster_x_bar_operator` already imported in `_test_circuit.py`) to extract the seed Z̄_1.
* Build gadgets, bridge, `joint_code = build_joint_ppm_circuit(...)[1]`; assert dimension.

**What it catches:** A stabilizer row dropped or duplicated in `_stitch_intercode`'s M_chi/M_co block construction. CSS commutation still holds (already tested), but the codespace dimension would shift.

**Why these two codes:** Steane × Steane is the simplest sanity check; Webster × Steane forces a non-trivial k_l > 1 case where the formula's "−1" reduction is observable (the bug would surface as dim==9 instead of 10).

### Test 4: Gauge-fix G rows are linearly independent

**File:** `src/qldpc/circuits/surgery/_test_gadget.py`

**Name:** `test_step2_gauge_fix_rows_linearly_independent`

**Spec:**
* For each test code in `[SteaneCode, Webster GB code 0, bb_18]`:
  * Build gadget with `basis=Pauli.X` and the published seed operator.
  * Extract `G = g.G`.
  * If `G.shape[0] > 0`: assert `np.linalg.matrix_rank(galois.GF(2)(G.astype(np.uint8))) == G.shape[0]` (full row rank over GF(2)).
  * Re-assert the existing `F @ G^T == 0` invariant alongside, since the two together are what Webster step 3 actually requires.

**What it catches:** `_step2_gauge_fix(F)` returning redundant rows on a degenerate F. The current `ker(F^T)` computation uses scipy / numpy linear algebra; certain F structures could return a non-minimal basis. The Cain bb_18 G=20 reproduction would catch the FINAL count but not the underlying rank degeneracy that might still be present in different F structures.

**Why bb_18 explicitly:** It's the only code in the suite where `G.shape[0] > 0` is non-trivially exercised (20 rows). Steane has G=0; Webster GB has small G. A rank-degeneracy bug might only appear at larger scale.

## Architecture / file organization

All four tests are appended to existing test files (`_test_circuit.py`, `_test_gadget.py`). No new files. Existing test patterns are followed:

* Local imports inside each test function (matches established convention)
* Parametrize over the dimensions that matter (rounds × state, qubit index `i`)
* Use the existing `raw-measurement-XOR pattern` (codified in `test_logical_state_init_end_to_end_steane_basis_z`) for raw observable extraction

`_test_circuit.py` is already 1358 lines; adding ~120 lines does not materially worsen file size and keeps related concerns colocated.

## Testing

Run `uv run pytest src/qldpc/circuits/surgery/ -q` after each test lands. Expect:
* Existing 127 tests continue to pass
* 4 new tests pass on the first try (the production code already satisfies the invariants — this is "test the math we already trust")

If any new test fails on first run, that IS the bug discovery scenario — the corresponding bug surface from Section "Motivation" has materialized. Handle by debugging the production code, not by softening the test.

## Risks and edge cases

* **Test 2 (X_ERROR injection)** depends on stim's ability to inject deterministic errors at specific circuit locations. The test must insert the X_ERROR at the right position — immediately after `_surgery_state_prep` and before the first QEC round. If stim's `compile_detector_sampler` does its detector regrouping in a way that hides single-shot patterns, fall back to manual decoder error model parsing. This is unlikely; stim supports deterministic single-shot debugging well.

* **Test 3 (Webster × Steane intercode dimension)** assumes the Webster gadget's `Z̄_1` and Steane's `Z̄_0` can be bridged without basis or weight conflicts. The existing `build_bridge` supports this; smoke-tested by `test_build_bridge_smoke_steane_intracode`. If the construction fails for a different reason (e.g. bridge cellulation can't find a chord), substitute another small intercode pair (two distinct BB codes).

* **Test 4 (rank check on bb_18 G)** runs the bb_18 build, which is somewhat slow (~10s in §3.2). Limit the test to one bb_18 case (use the cached `Z_BAR_SUPPORT` from the notebook helper or compute fresh — both are acceptable). If runtime becomes a concern, mark as `@pytest.mark.slow`.

## Success criteria

* Full surgery test suite `uv run pytest src/qldpc/circuits/surgery/ -q` reports 131 passed (127 existing + 4 new)
* Notebook §1, §2, §3 still produce the same outputs (no source code change → no behavior change)
* Each new test's failure message immediately points at one of the four bug surfaces (round indexing / Stim wiring / joint stitching / gauge-fix rank), so future regressions self-diagnose
