# obs0 last-round fix for surgery PPM observables

**Date:** 2026-06-11
**Goal:** Make `obs0` in `build_single_ppm_circuit` and `build_joint_ppm_circuit` encode the logical Pauli measurement outcome correctly for any number of rounds, by pointing it at the **last QEC round's** meas-check outcomes instead of XOR-ing across all rounds.

## Motivation

The current `_surgery_observable` (`src/qldpc/circuits/surgery/circuit.py:1012-1052`) constructs `obs0` as the XOR of every meas-check syndrome across every QEC round:

```python
chi_targets = [
    measurement_record.get_target_rec(cid, -1 - r)
    for r in range(num_rounds)
    for cid in chi_check_ids
]
circuit.append("OBSERVABLE_INCLUDE", chi_targets, 0)
```

In a noiseless run each meas check produces the same outcome `m_v` every round (after the round-0 projection), so

$$
\mathrm{obs}_0 \;=\; \bigoplus_{r=0}^{R-1} \bigoplus_v m_v \;=\; R \cdot \bigoplus_v m_v \pmod 2 \;=\;
\begin{cases}
\bigoplus_v m_v & R \text{ odd} \\
0 & R \text{ even}
\end{cases}
$$

For odd $R$ this happens to equal the logical parity bit. For even $R$ the observable is identically 0 regardless of the underlying logical state — parity readout is **silently broken**.

This is masked by the existing tests:

* `test_build_joint_ppm_circuit_intercode_noiseless_observables_zero` uses `rounds=2` (even) but the data init has logical parity $+1$, so `obs0 = 0` is consistent with both the broken even-round formula and the correct readout. The test passes vacuously.
* `test_joint_ppm_ler_monotone_steane_intercode` uses `rounds=3` (odd), so the protocol happens to work and LER monotonicity is observed.

The bug is not "noise accumulates over rounds" (the cumulative-XOR construction is in fact insensitive to where individual faults occur, by symmetry). The bug is that the XOR-over-rounds construction is **not the observable from any paper**:

* **Webster, Jochym-O'Connor, Yoder (arXiv:2410.03628)** state the gauging identity as $Z = \prod_{v\in V} A_v$ (one round, line 2255 of the v4 PDF). The repeated-measurement step is for FT detectors, not for the observable.
* **Cohen, Kim, Bartlett, Brown (arXiv:2407.18393)** use the standard convention "final round is free of measurement errors" (line 830) — the logical measurement outcome is read from the last, perfect round.
* **Stim's `surface_code:rotated_memory_z` generator** places `OBSERVABLE_INCLUDE` on the final destructive data measurement, never on a XOR across stabilizer rounds.

We replace the cumulative XOR with a single-round product on the last QEC round — paper-faithful, decouples from rounds-parity, and unchanged FT distance.

## Scope

**In scope:**

* `src/qldpc/circuits/surgery/circuit.py`: rewrite `_surgery_observable`'s `obs0` construction; drop the now-unused `num_rounds` parameter; update both call sites (`build_single_ppm_circuit` ≈ line 380, `build_joint_ppm_circuit` ≈ line 721); rewrite docstrings (this function plus the two `build_*_ppm_circuit` entries) to describe the corrected semantics and cite Webster/Cohen for the single-round identity.
* `src/qldpc/circuits/surgery/_test_circuit.py`: strengthen the existing noiseless-observable test to be a truth-table assertion (`obs0 == obs1` per shot across multiple inits) and add two new even-rounds parity-readout tests.

**Out of scope:**

* Detector emission. The round-to-round meas-check consistency detectors stay exactly as they are; the decoder behaviour and spacetime fault distance are unchanged.
* `obs1` (the destructive `support`-qubit measurement on V₀ ∪ V₀, used as a noiseless cross-check). It already reads the correct logical parity and stays as-is.
* Public API renames. The two `build_*_ppm_circuit` functions keep their names and signatures (except that `_surgery_observable`'s internal `num_rounds` argument goes away, transparent to outside callers).
* External notebooks and benchmark scripts. They consume `obs0` via `keep_only_observable(circuit, keep_idx=0)`, which still resolves to the joint logical parity. No script changes required.

## Naming convention

This spec assumes the [Cain-convention rename](2026-06-11-cain-convention-rename-design.md) has applied the `chi_*` → `meas_*` rename to `circuit.py`'s circuit-construction locals (`chi_check_ids` → `meas_check_ids`, `chi_l` → `n_meas_l`, `chi_r` → `n_meas_r`, `chi_total` → `n_meas_total`, and the corresponding matrix names `M_chi` / `M_co` → `M_meas` / `M_comp` with `m_chi_*_data` → `m_meas_*_data`). All identifier names in this spec use the Cain post-rename form. If this spec lands before the rename completes, the implementer substitutes the legacy `chi_*` names mechanically — the only function body it edits is `_surgery_observable`, so the overlap is one regex.

The spec discusses meas checks as "measured-basis ancilla checks on the support" rather than `χ_v`. Webster's paper-symbol identity $\bar L = \prod_v A_v$ is quoted verbatim in code comments (with a Cain translation line below it) so the proof reference stays readable.

## Design

### Code change in `_surgery_observable`

Replace the cumulative XOR with a single round's product:

```python
# circuit.py, in _surgery_observable
meas_targets = [
    measurement_record.get_target_rec(cid)   # latest record = last QEC round
    for cid in meas_check_ids
]
circuit.append("OBSERVABLE_INCLUDE", meas_targets, 0)
```

`get_target_rec(cid)` defaults `measurement_index=-1`, returning the most recent measurement of `cid` in the record (see `MeasurementRecord.get_target_rec` in `src/qldpc/circuits/bookkeeping.py:206`). Since `_surgery_detach_and_readout` measures only `ancilla_qubits` + `bridge_ids` (never the meas-check syndrome ancillas), the most-recent measurement of each `meas_check_ids[i]` is its outcome in the final QEC round. We add a precondition to make a future reorder fail loudly rather than silently:

```python
# Precondition: every meas-check ID must have been measured during the
# QEC cycle. Detach/readout never touches meas-check ancillas, so the
# most-recent record is the last QEC round.
for cid in meas_check_ids:
    assert measurement_record[cid], (
        f"meas-check {cid} has no measurement record; "
        f"_surgery_observable expects the QEC cycle to have run first."
    )
```

`MeasurementRecord` inherits `__getitem__` from `Record`, returning `[]` for unmeasured keys, so the truthiness check is correct (`Record.__getitem__` at `bookkeeping.py:150`).

`obs1` construction is unchanged:

```python
data_targets = [
    measurement_record.get_target_rec(data_ids[i]) for i in support_indices
]
circuit.append("OBSERVABLE_INCLUDE", data_targets, 1)
```

### Signature change

`_surgery_observable`'s `num_rounds` keyword is removed (it was only consumed by the XOR loop). The two callers:

```python
# build_single_ppm_circuit (circuit.py ≈380)
circuit += _surgery_observable(
    gadget,
    meas_check_ids=meas_check_ids,
    data_ids=data_ids,
    support_indices=gadget.support,
    measurement_record=measurement_record,
)

# build_joint_ppm_circuit (circuit.py ≈721)
circuit += _surgery_observable(
    g_l,
    meas_check_ids=meas_check_ids,
    data_ids=data_ids,
    support_indices=support_combined,
    measurement_record=measurement_record,
)
```

drop the `num_rounds=rounds` line. `_surgery_observable` is module-private (`_` prefix), so no external API surface changes.

### Why this is fault-tolerant

* **Single-round identity** (Webster L2255). In the merged stabilizer group, $\bar L_X = \prod_{v \in \text{support}} A_v$ where $A_v$ is the meas check on support vertex $v$. The product of the last-round meas-check outcomes therefore equals the eigenvalue of $\bar L_X$ on the encoded state.
* **Noiseless determinism.** Initialisation projects $\bar L_X$ onto an eigenstate; the product is the eigenvalue, fixed across all rounds. Individual `m_v` values are random across rounds 0+, but their product is deterministic, so `obs0` is well-defined for Stim.
* **FT distance $d$.** Detectors still XOR meas-check syndromes across consecutive rounds, building the spacetime code of distance $d$ (one factor of $d$ from $R \ge d$ rounds, the other from the spatial distance). Any single fault that flips `obs0` (a flip on a last-round meas-check ancilla, or a propagating data error reaching it) is witnessed by at least one detector. The decoder predicts the observable from detector outcomes exactly as for a Stim memory experiment with destructive final readout.
* **Cohen alignment** (arXiv:2407.18393 L830-831). Cohen's "final round is free of measurement errors" model places the logical readout on a single perfect round. Our construction is the noisy-final-round analogue: same single-round product, with the round assumed measurable rather than perfect.

### Tests

| Test | Status | Action |
|------|--------|--------|
| `test_build_joint_ppm_circuit_intercode_noiseless_observables_zero` (`_test_circuit.py:690`) | exists, vacuous on parity | strengthen: keep `rounds=2`, sweep `data_init ∈ {("+","+"), ("-","+"), ("+","-"), ("-","-")}`, assert `obs[:,0] == obs[:,1]` per shot for every init. The `(-,+)` and `(+,-)` cases give parity $-1$, so `obs0` must be 1 — exactly the behaviour the old code breaks |
| `test_build_joint_ppm_circuit_meas_check_ids_no_UB` (`_test_circuit.py:673`, renamed in cain) | unchanged | nothing — this exercises detectors, not observables |
| `test_joint_ppm_ler_monotone_steane_intercode` (`_test_circuit.py:709`) | unchanged | nothing — `rounds=3` was already correct; new code keeps it correct |
| `test_joint_ppm_data_init_truth_table` (`_test_circuit.py:823`) | unchanged | already covers parity readout, but uses an unspecified `rounds`; verify rounds and ensure it includes at least one even-round case after this spec lands |
| `test_joint_ppm_even_rounds_truth_table` | new | `rounds=2`, sweep inits as above, assert `obs0` matches expected logical parity sign on every shot in a noiseless run. Documents the bug-fix explicitly. |
| `test_single_ppm_even_rounds_truth_table` | new | same idea for `build_single_ppm_circuit`, sweep `data_init ∈ {"0","1","+","-"}`. |

`obs[:, 0] == obs[:, 1]` is the strongest cross-check we can do — `obs1` is the destructive data measurement on `support` (computes parity directly) so it is the ground truth for `obs0` in a noiseless run.

### Documentation

* `_surgery_observable` docstring: rewrite to describe the single-round identity, cite Webster L2255 (`Z = ∏_{v∈V} A_v`) and Cohen §3.5 ("final round assumed perfect"), and note that detectors carry the FT load.
* `build_single_ppm_circuit` and `build_joint_ppm_circuit` docstrings: the lines describing "Webster Eq. 1, the physical syndrome-based readout of $\bar X_l \otimes \bar X_r$" stay; the implicit "XOR across all rounds" claim is replaced with "product of the last QEC round's meas-check outcomes".
* Any inline comment that says "χ XOR over rounds" → "product of last-round meas checks". Webster paper-citation comments keep paper-symbol identities (`Z = ∏_v A_v`) with a Cain-translation line below.

## File touch summary

| File | Edits |
|---|---|
| `src/qldpc/circuits/surgery/circuit.py` | rewrite `_surgery_observable` body + docstring (~25 lines); drop `num_rounds=rounds` in two callers; update obs0 description in two `build_*_ppm_circuit` docstrings |
| `src/qldpc/circuits/surgery/_test_circuit.py` | strengthen one existing test; add two new tests (~50 lines total) |

No other files touched. No new modules.

## Implementation strategy

One commit, since the change is small and tightly coupled.

1. Apply the `_surgery_observable` rewrite + caller updates in a single edit pass.
2. Update test file in the same commit.
3. Run `uv run pytest src/qldpc/circuits/surgery/ -q` and verify all existing tests still pass, plus the two new ones.
4. Spot-check one LER notebook (`examples/logical_error_rates/*.ipynb`) with the new circuit to confirm `keep_only_observable(circuit, keep_idx=0)` still produces a meaningful LER curve.

If we discover during step 1 that the cain rename hasn't fully landed in `circuit.py`, we either (a) wait until it does, or (b) apply this fix using the still-legacy `chi_*` identifiers and let the cain rename pick them up later. Either is fine — there is no ordering hazard, only a naming-consistency consideration.

## Risks and edge cases

* **Detach reordering.** If a future refactor moves meas-check measurement into the detach/readout block (e.g. measures meas-check ancillas a second time), `get_target_rec(cid)` will return the wrong record. The assertion in `_surgery_observable` is meant to catch this; if `MeasurementRecord` lacks a `has_record` helper, encode the check by stashing the QEC-cycle measurement-record snapshot before detach runs and reading from that snapshot.
* **`obs1` cross-check still requires odd rounds in noisy runs?** No. `obs1` reads `support` data destructively at the end and has always been correct on any number of rounds. The truth-table tests use noiseless samples, where `obs0 == obs1` is a strict per-shot identity regardless of rounds parity.
* **Backward compatibility.** This is a bug fix, not a semantic change: for odd `rounds` (the only case where the old code was right), the new and old `obs0` are equal modulo XOR ordering. For even `rounds`, the old code was silently wrong; users who happened to feed parity-$+1$ states will see no change, users who fed parity-$-1$ states will start seeing `obs0 = 1` (the correct value, was 0 before). Downstream LER scripts that aggregate `obs0 != 0` will see correct LERs in the previously-broken regime.
* **`docs/notebooks/*.ipynb` drift.** Notebooks are not edited by this spec, but we grep them during implementation for any direct `OBSERVABLE_INCLUDE` index manipulation. None expected (notebooks use the public `build_*_ppm_circuit` API only), but we verify.

## Success criteria

* `uv run pytest src/qldpc/circuits/surgery/ -q` reports **(N existing) + 2 new even-rounds truth-table tests passed**, with no existing tests broken. N is whatever the count is at the time this lands (cain rename is in flight, so the absolute count is moving).
* `test_build_joint_ppm_circuit_intercode_noiseless_observables_zero` is now a non-vacuous truth-table check: it asserts `obs0 == obs1` for inits with both parity-$+1$ and parity-$-1$ logical states.
* `git grep -nE 'for r in range\(num_rounds\)' src/qldpc/circuits/surgery/circuit.py` returns no hit inside `_surgery_observable`.
* `_surgery_observable` no longer accepts a `num_rounds` argument.
* No external code (notebooks, benchmark scripts) needs modification to keep working.
