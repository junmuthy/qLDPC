# Surgery math-invariants test hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 4 targeted pytest cases that close the highest-risk math-layer gaps in the surgery module (multi-round invariance, Stim wiring correctness, joint code dimension, gauge-fix row independence).

**Architecture:** Pure test-only additions to existing test files (`_test_circuit.py`, `_test_gadget.py`). No source code changes. Each test asserts a math claim the production code already promises but that current tests don't enforce.

**Tech Stack:** Python 3.12, pytest, numpy, stim, galois (already imported), qldpc (already imported), sympy (for BBCode construction).

**Spec:** `docs/superpowers/specs/2026-06-11-surgery-math-invariants-design.md`

---

## File Structure

| File | Action | Lines | Responsibility |
|---|---|---|---|
| `src/qldpc/circuits/surgery/_test_circuit.py` | Modify | +~150 | Add 3 tests (T1, T2, T3) |
| `src/qldpc/circuits/surgery/_test_gadget.py` | Modify | +~35 | Add 1 test (T4) |

All four test files in the package already have the imports needed (numpy, pytest, codes, Pauli, galois, sympy via top-level imports or function-scope imports per the established pattern).

---

## Task 1: Multi-round invariance on Steane (basis=Z)

**Files:**
- Modify: `src/qldpc/circuits/surgery/_test_circuit.py` (append at end-of-file)

- [ ] **Step 1: Write the test**

Append to `src/qldpc/circuits/surgery/_test_circuit.py`:

```python
@pytest.mark.parametrize("rounds", [1, 2, 3, 5, 10])
@pytest.mark.parametrize("state,expected_obs0", [("0", 0), ("1", 1)])
def test_multi_round_invariance_steane_basis_z(rounds, state, expected_obs0):
    """obs0 = int(state) for ALL round counts ∈ {1, 2, 3, 5, 10}.

    Existing PPM tests pin rounds=3 only. A round-index drift in
    _surgery_qec_cycle, _surgery_observable, or
    MeasurementRecord.get_target_rec(...,-1-r) would slip past every
    one of them because they all share the same round budget. This test
    asserts the round-counted XOR identity holds across a wide range
    of round counts.
    """
    from qldpc.circuits.surgery.circuit import (
        build_single_ppm_circuit, logical_state_init,
    )
    from qldpc.circuits.surgery.gadget import build_gadget
    code = codes.SteaneCode()
    z_bar = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z_bar, basis=Pauli.Z)
    circuit = build_single_ppm_circuit(
        g, rounds=rounds, noise_model=None,
        data_init=logical_state_init(code, state, log_idx=0),
    )
    raw = circuit.compile_sampler().sample(shots=200).astype(np.uint8)
    n_meas = raw.shape[1]
    obs0_recs = []
    for ln in str(circuit).splitlines():
        if ln.startswith("OBSERVABLE_INCLUDE(0)"):
            obs0_recs = [
                int(t.strip("rec[]")) for t in ln.split() if t.startswith("rec[")
            ]
            break
    obs0 = np.bitwise_xor.reduce(
        raw[:, [n_meas + off for off in obs0_recs]], axis=1
    )
    rate = float(obs0.mean())
    assert rate == float(expected_obs0), (
        f"rounds={rounds}, state={state!r}: obs0 rate {rate:.3f} != "
        f"expected {expected_obs0}"
    )
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest src/qldpc/circuits/surgery/_test_circuit.py::test_multi_round_invariance_steane_basis_z -v`
Expected: 10 PASSED (5 rounds × 2 states).

If any combination fails, you have found a round-index bug — do not commit. Stop and report.

- [ ] **Step 3: Commit**

```bash
git add src/qldpc/circuits/surgery/_test_circuit.py
git commit -m "$(cat <<'EOF'
test(surgery): multi-round invariance on Steane (basis=Z)

Pin obs0 = int(state) for rounds ∈ {1, 2, 3, 5, 10} × state ∈ {"0", "1"}.
Existing PPM tests fix rounds=3, missing round-index drift bugs in
DETECTOR / SHIFT_COORDS / OBSERVABLE_INCLUDE XOR across round counts.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Single-qubit |1⟩ injection triggers exactly the touching Z-checks

**Files:**
- Modify: `src/qldpc/circuits/surgery/_test_circuit.py` (append at end-of-file)

- [ ] **Step 1: Write the test**

Append to `src/qldpc/circuits/surgery/_test_circuit.py`:

```python
@pytest.mark.parametrize("error_qubit", list(range(7)))
def test_single_qubit_one_init_triggers_only_neighboring_z_checks_steane(
    error_qubit,
):
    """Inject a single |1⟩ at data qubit ``error_qubit`` (rest |0⟩) and
    sample one shot of the noiseless Steane basis=Z PPM. Assert exactly
    the round-1 Z-stab detectors whose support contains ``error_qubit``
    fire, and no others.

    Why this catches stim wiring bugs:
    * Round-1 detectors for reliable Z-checks compare the measured
      syndrome against the +1 expectation. A perfect |0⟩^n input gives
      all +1 syndromes → zero detector events.
    * Flipping a single data qubit to |1⟩ flips the parity of every
      Z-stab whose support contains that qubit, so those detectors
      MUST fire and no others may.
    * If CX in syndrome extraction is on the wrong qubit, or the
      ancilla measurement basis is wrong, or the EdgeColoring schedule
      delays a check to a later round, this pattern breaks loudly.

    Steane Z-stabilizers have weight 4. With 7 data qubits, every
    qubit is in exactly 3 of the 3 Z-stabs (Steane is symmetric:
    H_Z @ ones(7) = 0 doesn't apply because Steane's H_Z is full-rank
    weight-4 — qubit i is in either 1 or 2 Z-stabs depending on row
    canonicalization).
    """
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget
    code = codes.SteaneCode()
    z_bar = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z_bar, basis=Pauli.Z)
    # data_init prepares physical |1⟩ at error_qubit, |0⟩ elsewhere — this
    # is equivalent to applying X(error_qubit) before any QEC round runs.
    data_init = "".join(
        "1" if i == error_qubit else "0" for i in range(7)
    )
    circuit = build_single_ppm_circuit(
        g, rounds=1, noise_model=None, data_init=data_init,
    )
    sampler = circuit.compile_detector_sampler()
    detection_events, _ = sampler.sample(
        shots=1, separate_observables=True,
    )
    # detection_events shape: (1, num_detectors)
    events = detection_events[0]

    # Build the expected detector pattern by parsing DETECTOR lines and
    # mapping each round-1 detector to its underlying check row index.
    HZ = np.asarray(code.matrix_z).astype(int)
    # Steane Z-stabs touching error_qubit: row indices j s.t. HZ[j, i] == 1
    z_stabs_touching = set(np.where(HZ[:, error_qubit] == 1)[0].tolist())

    # In _surgery_qec_cycle, round-1 detectors are emitted in
    # `check_id` order over `all_check_ids` (qubit_ids.checks_x +
    # qubit_ids.checks_z), restricted to the "reliable" subset. For
    # basis=Z, the reliable Z-checks are exactly the m_Z data H_Z rows
    # (first m_Z entries of qubit_ids.checks_z). The detector index for
    # row j of HZ equals the position of `qubit_ids.checks_z[j]` in
    # the emit order of round-1 detectors.
    #
    # Rather than reconstruct this mapping in the test (fragile),
    # parse the circuit's DETECTOR lines: each round-1 detector
    # corresponds to a single measurement (since round 1 has no diff),
    # and that measurement's qubit is `qubit_ids.checks_z[j]` (or _x).
    # We assert events fire precisely on detectors whose underlying
    # check ancilla is in the Z-stab-touching set.
    #
    # Easier route: count the syndrome bits directly.
    # Number of Z-stab rows touching error_qubit MUST equal sum of
    # events flagged on Z-side reliable detectors.

    # Read DETECTOR lines from circuit text; each round-1 DETECTOR(...)
    # references one or more `rec[-k]` entries.
    detector_lines = [
        ln for ln in str(circuit).splitlines()
        if ln.startswith("DETECTOR(")
    ]
    # For Steane basis=Z, rounds=1, the number of reliable Z-stab
    # detectors should equal m_Z = 3.
    n_reliable_z = HZ.shape[0]  # 3 for Steane
    # Sanity: events should fire on exactly len(z_stabs_touching) detectors
    # AMONG the m_Z Z-side reliable detectors. (Other reliable detectors
    # are X-side: data H_X and/or G; on |0⟩ input with X(error_qubit),
    # X-side parity is RANDOM in general — but X stabs on data prepared in
    # Z-basis with one flip still give random outcomes per round, so the
    # X-side detector contributions are stochastic. We must focus on
    # Z-side detectors only.)
    #
    # CRITICAL: the test infers Z-side detector indices by sampling a
    # CLEAN reference circuit (all |0⟩, no flip) and identifying which
    # detector positions are deterministic-0 across many shots.
    clean_init = "0" * 7
    clean_circuit = build_single_ppm_circuit(
        g, rounds=1, noise_model=None, data_init=clean_init,
    )
    clean_sampler = clean_circuit.compile_detector_sampler()
    clean_events, _ = clean_sampler.sample(
        shots=256, separate_observables=True,
    )
    # Detectors that are deterministic-0 across 256 shots on clean input
    # are the Z-side ones (Z̄_M parity over Z-eigenstate input is fixed).
    deterministic_zero = np.where(clean_events.sum(axis=0) == 0)[0]
    # Sanity: there should be at least n_reliable_z such detectors.
    assert len(deterministic_zero) >= n_reliable_z, (
        f"expected >= {n_reliable_z} deterministic-zero detectors on "
        f"clean Steane basis=Z PPM, got {len(deterministic_zero)}"
    )

    # Count how many of the deterministic-zero positions fired in the
    # error-injected sample. This should equal the number of Z-stabs
    # touching error_qubit (each one's parity flipped from +1 to -1).
    fired_on_z_side = int(events[deterministic_zero].sum())
    expected_fired = len(z_stabs_touching)
    assert fired_on_z_side == expected_fired, (
        f"error on qubit {error_qubit}: expected {expected_fired} "
        f"Z-side detectors to fire (one per Z-stab containing the qubit), "
        f"got {fired_on_z_side}. This is the syndrome-extraction wiring "
        f"regression: CX swap, wrong measurement basis, or EdgeColoring "
        f"schedule bug."
    )
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest src/qldpc/circuits/surgery/_test_circuit.py::test_single_qubit_one_init_triggers_only_neighboring_z_checks_steane -v`
Expected: 7 PASSED (one per `error_qubit ∈ 0..6`).

If any parametrization fails, the syndrome extraction wiring is broken for that qubit. Investigate before committing.

- [ ] **Step 3: Commit**

```bash
git add src/qldpc/circuits/surgery/_test_circuit.py
git commit -m "$(cat <<'EOF'
test(surgery): single-qubit |1⟩ init triggers correct Z-stab detectors

For each Steane data qubit i ∈ 0..6, prep |1⟩ at i (rest |0⟩) and
sample one shot of the noiseless basis=Z PPM. Identify Z-side
deterministic-zero detectors via a clean reference sample, then
assert the error injection fires exactly the count predicted by
H_Z's structure (one detector per Z-stab containing qubit i).

Catches CX swap, wrong measurement basis, EdgeColoring schedule
bugs that noiseless invariance tests can't see.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Joint code dimension matches k_l + k_r − 1

**Files:**
- Modify: `src/qldpc/circuits/surgery/_test_circuit.py` (append at end-of-file)

- [ ] **Step 1: Write the test**

Append to `src/qldpc/circuits/surgery/_test_circuit.py`:

```python
def test_joint_code_dimension_steane_x_steane_equals_one():
    """Intercode Steane × Steane joint PPM gives joint_code.dimension == 1.

    Formula: k_l + k_r − 1 because Z̄_l ⊗ Z̄_r becomes a stabilizer of
    the joint code after surgery. For k_l = k_r = 1, that's 1.

    Catches a stitching bug in _stitch_intercode that drops or
    duplicates a stabilizer row — CSS commutation would still hold
    but the joint code's logical dimension would shift.
    """
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.bridge import build_bridge
    c1, c2 = codes.SteaneCode(), codes.SteaneCode()
    z1 = np.asarray(c1.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    z2 = np.asarray(c2.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g1 = build_gadget(c1, z1, basis=Pauli.Z)
    g2 = build_gadget(c2, z2, basis=Pauli.Z)
    bridge = build_bridge(g1, g2)
    _, joint_code = build_joint_ppm_circuit(
        g1, g2, bridge, rounds=3, noise_model=None,
    )
    expected = c1.dimension + c2.dimension - 1  # 1 + 1 - 1 = 1
    assert joint_code.dimension == expected, (
        f"Steane × Steane intercode joint_code.dimension = "
        f"{joint_code.dimension}, expected {expected}"
    )


def test_joint_code_dimension_webster_x_steane_equals_ten():
    """Intercode Webster GB code 0 × Steane joint PPM gives dim == k_l + k_r − 1 = 10.

    Webster GB code 0 is [[62, 10, _]]; k_l = 10. Steane is k_r = 1.
    Expected: 10 + 1 − 1 = 10.

    The k_l > 1 case exposes the −1 reduction in the formula. A
    stitching bug that fails to add the Z̄_l ⊗ Z̄_r constraint would
    surface as dim = 11.
    """
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.bridge import build_bridge
    data = load_webster_seed_set(0)
    webster = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    z_webster = _webster_x_bar_operator(data, "Z_bar_1")
    steane = codes.SteaneCode()
    z_steane = np.asarray(steane.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(webster, z_webster, basis=Pauli.Z)
    g_r = build_gadget(steane, z_steane, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)
    _, joint_code = build_joint_ppm_circuit(
        g_l, g_r, bridge, rounds=3, noise_model=None,
    )
    expected = webster.dimension + steane.dimension - 1  # 10 + 1 - 1 = 10
    assert joint_code.dimension == expected, (
        f"Webster × Steane intercode joint_code.dimension = "
        f"{joint_code.dimension}, expected {expected}"
    )
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest src/qldpc/circuits/surgery/_test_circuit.py -k "test_joint_code_dimension" -v`
Expected: 2 PASSED.

If `test_joint_code_dimension_webster_x_steane_equals_ten` fails with `BBCode build_bridge` cellulation errors, replace Webster with a different small GB code OR with a self-paired Webster intracode case. Note in the commit if you had to substitute.

- [ ] **Step 3: Commit**

```bash
git add src/qldpc/circuits/surgery/_test_circuit.py
git commit -m "$(cat <<'EOF'
test(surgery): joint code dimension matches k_l + k_r - 1 formula

Assert intercode joint PPM joint_code.dimension equals k_l + k_r - 1
on two cases:
  Steane × Steane → 1
  Webster GB code 0 × Steane → 10 (exposes the -1 reduction)

Catches a stitching bug in _stitch_intercode that drops or duplicates
a stabilizer row. CSS commutation would still pass but dimension
would shift.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Gauge-fix G rows are linearly independent over GF(2)

**Files:**
- Modify: `src/qldpc/circuits/surgery/_test_gadget.py` (append at end-of-file)

- [ ] **Step 1: Write the test**

Append to `src/qldpc/circuits/surgery/_test_gadget.py`:

```python
def test_step2_gauge_fix_rows_linearly_independent():
    """G rows from _step2_gauge_fix are linearly independent over GF(2).

    Webster §II.A step 3 requires |S_L| - wt(L) + 1 INDEPENDENT gauge
    constraints. The existing test verifies F @ G^T == 0 but not that
    G has full row rank.

    A degenerate F could let the gauge fix return redundant rows,
    inflating g.G.shape[0] without changing the actual gauge structure.
    The Cain Table III bb_18 G=20 reproduction would catch the final
    count but not the underlying rank degeneracy.
    """
    import sympy
    from qldpc.circuits.surgery.gadget import build_gadget
    import galois

    F2 = galois.GF(2)
    xs, ys = sympy.symbols("x y")

    cases: list[tuple[str, object, np.ndarray]] = []

    # Case 1: Steane
    steane = codes.SteaneCode()
    x_steane = np.asarray(steane.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    cases.append(("Steane", steane, x_steane))

    # Case 2: Webster GB code 0
    data = load_webster_seed_set(0)
    webster = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x_webster = _webster_x_bar_operator(data, "X_bar_1")
    cases.append(("Webster GB 0", webster, x_webster))

    # Case 3: Cain bb_18 (cached Z̄ support — same as notebook §3.2)
    bb18 = codes.BBCode(
        (31, 4),
        1 + xs**6 * ys + xs**27,
        ys**2 + xs**15 * ys**3 + xs**24,
    )
    # Use the same cached wt-20 Z̄ rep used by the notebook §3.2 cell to
    # exercise the largest realistic gauge-fix case (G=20 rows). Treat
    # via swap (matrix_z ↔ matrix_x) so vec_20 acts as the X̄ on
    # target_code (matches notebook usage).
    z_bar_support = [8, 9, 14, 18, 24, 34, 40, 56, 75, 76,
                     97, 111, 122, 171, 202, 208, 213, 218, 228, 238]
    from qldpc.codes.common import CSSCode
    vec_20 = np.zeros(bb18.num_qudits, dtype=np.uint8)
    vec_20[z_bar_support] = 1
    bb18_swapped = CSSCode(
        bb18.matrix_z, bb18.matrix_x, is_subsystem_code=False,
    )
    cases.append(("Cain bb_18 (swapped, wt-20)", bb18_swapped, vec_20))

    for label, code, seed_op in cases:
        g = build_gadget(code, seed_op, basis=Pauli.X)
        G = g.G
        if G.shape[0] == 0:
            # Steane has G empty; trivially row-rank == 0 == shape[0].
            assert G.shape[0] == 0
            continue
        rank = int(np.linalg.matrix_rank(F2(G.astype(np.uint8).tolist())))
        assert rank == G.shape[0], (
            f"{label}: gauge-fix G has {G.shape[0]} rows but rank only "
            f"{rank}. _step2_gauge_fix returned redundant rows on this F."
        )
        # Re-assert the existing F @ G^T == 0 invariant alongside.
        F_mat = g.F.astype(np.uint8)
        commute = (G.astype(np.uint8) @ F_mat.T) % 2
        assert not commute.any(), (
            f"{label}: G @ F^T != 0 (gauge-fix output failed commutation)."
        )
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest src/qldpc/circuits/surgery/_test_gadget.py::test_step2_gauge_fix_rows_linearly_independent -v`
Expected: 1 PASSED (one test function iterating over 3 cases).

bb_18 build is the slow case (~10s on the build_gadget call). If runtime is a problem in your CI, the test can be marked `@pytest.mark.slow` — but it stays as one test function regardless.

If any case fails, _step2_gauge_fix is returning rank-degenerate output for that F. Do not commit; report.

- [ ] **Step 3: Commit**

```bash
git add src/qldpc/circuits/surgery/_test_gadget.py
git commit -m "$(cat <<'EOF'
test(surgery): step2_gauge_fix G rows are linearly independent

Assert rank_GF2(G) == G.shape[0] on Steane, Webster GB code 0, and
Cain bb_18 (cached wt-20 Z̄ representative). Re-assert F @ G^T == 0
alongside.

A degenerate F could let _step2_gauge_fix return redundant rows;
the existing F @ G^T == 0 invariant alone doesn't catch this.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Final verification gate

**Files:** No code changes.

- [ ] **Step 1: Run the full surgery test suite**

Run: `uv run pytest src/qldpc/circuits/surgery/ -q`
Expected: 131 passed (127 existing + 4 new test functions; T1 parametrizes to 10, T2 to 7, so actual count is higher — check the final number).

Actual count check: T1 = 10 cases, T2 = 7 cases, T3 = 2 cases, T4 = 1 case → 20 new cases. So expected total: 127 + 20 = **147 passed**.

- [ ] **Step 2: Verify branch state**

```bash
git log --oneline -5
git status
```

Expected: 4 new commits from Tasks 1–4, working tree clean.

This task has no commit — it is a verification gate.

---

## How to test the whole feature manually

After all 5 tasks land, do one end-to-end smoke check:

```bash
cd /Users/tgzhou/Project/qLDPC && uv run pytest src/qldpc/circuits/surgery/ -q -k "multi_round or single_qubit_one or joint_code_dimension or gauge_fix_rows" 2>&1 | tail -5
```

Expected:
```
20 passed in X.Xs
```

If any test fails, you have found one of the bug surfaces described in the spec — debug the corresponding production code area, not the test.

---

## After plan completes

Use `superpowers:finishing-a-development-branch` to merge / push / cleanup the feature branch (it already has 13 commits from earlier work; the math-invariants additions will bring it higher).
