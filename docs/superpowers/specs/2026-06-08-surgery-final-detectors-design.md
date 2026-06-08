# Surgery Final-Measurement Detectors — Design Spec

**Date:** 2026-06-08
**Status:** approved (brainstorming) → awaiting writing-plans
**Branch:** `feat/surgery-construction`
**Builds on:** `docs/superpowers/specs/2026-06-08-surgery-circuit-rewrite-design.md`

## Motivation

The surgery circuit rewrite (spec `2026-06-08-surgery-circuit-rewrite-design.md` §3.4) called for emitting DETECTORs that XOR the final M/Mx outcomes against the last QEC round's syndrome, for stabilizers that are reliably inferable from the destructive measurement bases. These detectors are present in `_get_basis_memory_experiment_parts` (lines 274-285) but were inadvertently omitted from `build_single_ppm_circuit` and `build_joint_ppm_circuit` during the surgery circuit rewrite.

Without them, the decoder's matching graph is missing edges connecting the final destructive readout to the last syndrome round. The PPM observable (Webster Eq. 1) and the protocol semantics remain correct, but the decoder over-counts logical errors because it cannot use the destructive-readout / syndrome correlation. Observed LER values are therefore **inflated** relative to what the same protocol would yield with a complete decoder graph.

This spec adds the missing final-measurement detectors.

## Goals

1. New helper `_surgery_final_detectors(gadget, merged_code, qubit_ids, *, measurement_record)` emits one DETECTOR per reliable round-1 stabilizer, each XORing the final M-record on the stabilizer's support against the last-round syndrome.
2. `build_single_ppm_circuit` and `build_joint_ppm_circuit` call the helper between `_surgery_detach_and_readout` and `_surgery_observable`.
3. The set of reliable stabilizers is identical to `_classify_reliable_round1_checks` output (the same physics that makes round-1 syndromes reliable makes the final destructive readouts inferable).
4. Basis-symmetric: works for `basis=Pauli.X` and `basis=Pauli.Z`.
5. **Webster Table I tests untouched.** The 8 exact-match tests (4 codes × 2 bases) only test construction, not circuit semantics.
6. **All existing 75 surgery tests continue to pass.** No regressions.
7. **LER drop verified.** New slow test asserts LER at p=0.001 ≤ 0.01 (was ~0.024 in the previous LER monotonicity sweep).

## Non-goals

- Cross §3.2 D_0 chi-product detector (the observation that round-1 ⊕χ rows is reliable even though individual χ rows aren't). Marginal decoder graph improvement, more math; explicitly skipped to keep this spec small.
- Adaptive Pauli FEEDBACK from κ measurements. Still absorbed into the observable algebraically.
- Tighter LER monotonicity than what's needed to detect "detectors didn't help."

## 1. Helper signature and module placement

```python
# circuit.py — appended after _surgery_detach_and_readout, before build_single_ppm_circuit

def _surgery_final_detectors(
    gadget: GadgetLayout,
    merged_code: CSSCode,
    qubit_ids: QubitIDs,
    *,
    measurement_record: MeasurementRecord,
) -> stim.Circuit:
    """Emit DETECTORs for reliable stabs inferable from final readouts.

    Reliable stabs (= the same set as round-1 reliable):
      basis=X: data H_X (HX_merged rows [:m_X], from Mx data)
             + gauge-fix G (HZ_merged rows [m_Z:], from Mz κ)
      basis=Z: data H_Z (HZ_merged rows [:m_Z], from Mz data)
             + gauge-fix G (HX_merged rows [m_X:], from Mx κ)

    Each DETECTOR XORs ⊕(final M-record on stab support) ⊕ last-round syndrome.
    """
```

## 2. Reliable-stabilizer inference

A merged-code stabilizer row is **inferable from final readouts** iff its support, restricted to each register, falls in the basis we measured that register in:

| basis=X | Z-measured κ | X-measured data |
|---|---|---|
| Z-type (data H_Z) | ⊥ (Z on data) — but data measured in X | ✗ |
| Z-type (G) | Z on κ — match | ✓ |
| X-type (data H_X) | X on data — match | (no κ support) | ✓ |
| X-type (χ) | X on κ — but κ measured in Z | ✗ |

For basis=Z swap rows/columns. The inferable set exactly matches `_classify_reliable_round1_checks` — this is because the **same** initial state (|+⟩_data ⊗ |0⟩_κ) that makes a stab's eigenvalue deterministic at round 1 also makes its support's measurements (in the prep basis) deterministic at readout.

## 3. DETECTOR target construction

For row index `kk` in the merged check matrix (HX or HZ depending on row class):

```python
stab_row = HX[kk] if X_type else HZ[kk]   # uint8 binary support
supp = np.where(stab_row)[0]               # merged-code data-register indices

targets = [
    measurement_record.get_target_rec(qubit_ids.data[q])   # final M-record on qubit
    for q in supp
]
targets.append(
    measurement_record.get_target_rec(check_id_for_kk, -1)  # last QEC round
)
circuit.append("DETECTOR", targets, (0, 0, det_idx))
```

`qubit_ids.data` is the merged-code data register, which includes both original data qubits AND κ ancillas (and bridge qubits in the joint case). The relevant measurement is whichever was applied to that qubit by `_surgery_detach_and_readout`:
- For κ qubit: Mz (basis=X) or Mx (basis=Z) — applied first
- For original data qubit: Mx (basis=X) or Mz (basis=Z) — applied second

`measurement_record.get_target_rec(qubit_id)` defaults to `-1`, picking the most recent measurement of that qubit. Both registers have exactly one final measurement per qubit, so this is unambiguous.

## 4. Integration into build_* functions

```python
# build_single_ppm_circuit:
circuit += _surgery_detach_and_readout(...)
circuit += _surgery_final_detectors(           # NEW
    gadget, merged_code, qubit_ids,
    measurement_record=measurement_record,
)
circuit += _surgery_observable(...)

# build_joint_ppm_circuit: same pattern with merged_code=joint_code, gadget=g1 (g1.basis == g2.basis)
```

LOC impact: +30 LOC helper + 4 LOC integration (2 per build_* function). `circuit.py` grows from 365 → ~400. Already over the 300 budget; accept.

## 5. Tests

### 5.1 Helper test — detector count

```python
@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_surgery_final_detectors_count_matches_reliable_round1(basis):
    """Number of final DETECTORs equals |reliable round-1 set|."""
    ...
    circuit = _surgery_final_detectors(g, merged, qubit_ids,
                                        measurement_record=mrec_after_readout)
    n_final_det = str(circuit).count("DETECTOR")
    expected = len(_classify_reliable_round1_checks(g, merged, qubit_ids))
    assert n_final_det == expected
```

### 5.2 Noiseless regression

```python
@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_build_single_ppm_circuit_noiseless_no_detector_fires(basis):
    """Noiseless: NO detector fires (including the new final detectors)."""
    ...
    circuit = build_single_ppm_circuit(g, rounds=3, noise_model=None)
    sampler = circuit.compile_detector_sampler()
    dets, _ = sampler.sample(shots=64, separate_observables=True)
    assert not dets.any(), f"{dets.sum()} detector fires noiselessly"
```

### 5.3 LER drop verification

```python
@pytest.mark.slow
def test_single_ppm_ler_with_final_detectors_below_threshold():
    """With final detectors wired, LER at p=0.001 should be ≤ 0.01 (was ~0.024)."""
    ... single-PPM sinter sweep with p ∈ {0.001} ...
    ler_low = next(r for r in results if r.json_metadata["p"] == 0.001).errors / shots
    assert ler_low <= 0.01
```

### 5.4 Webster Table I unchanged

The 8 exact-match tests (κ+χ+r and bridge width on 4 codes × 2 bases) continue to pass — they only verify construction, not circuit semantics.

## 6. Risks

- **`qubit_ids.data` covering κ/bridge assumption.** Verified in `QubitIDs.from_code(merged_code)`: the entire data register is enumerated. Mitigation: helper test asserts detector count, which would fail loudly if the indexing is wrong.
- **Joint code's data register layout.** For inter-code joint the order is data_1 + data_2 + κ_1 + κ_2 + bridge. `_stitch_to_joint_csscode` constructs the matrices in this order, and `QubitIDs.from_code` follows the matrix indexing, so they stay consistent.
- **LER drop fails to materialize.** If `test_single_ppm_ler_with_final_detectors_below_threshold` fails at LER > 0.01, the detector targets are wrong (likely the support-indexing or last-round syndrome target). Helper test catches obvious shape errors; LER test catches semantic errors.

## 7. Out of scope

- Cross §3.2 D_0 chi-product detector. Could be added in follow-up; not needed for "matches Webster + correct LER trend."
- Adaptive Pauli FEEDBACK.
- Tighter LER comparison vs published Cain Fig 1b numbers.
