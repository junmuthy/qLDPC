# Surgery Circuit Rewrite — Design Spec

**Date:** 2026-06-08
**Status:** approved (brainstorming) → awaiting writing-plans
**Branch:** `feat/surgery-construction`
**Builds on:** `docs/superpowers/specs/2026-06-07-surgery-simplification-design.md`

## Motivation

The current `build_single_ppm_circuit` and `build_joint_ppm_circuit` delegate to `qldpc.circuits.memory.get_memory_experiment(merged_code, basis=Pauli.X)`. This is **wrong** for surgery: the memory experiment initializes all data qubits in |+⟩ and tracks the logical-X observable of the **merged code**, whereas surgery requires:

- κ ancilla qubits initialized in **|0⟩** (Cain §III.A step 1), not |+⟩.
- The PPM observable is the **XOR of χ-row syndrome outcomes across all τ_s rounds** (Webster Eq. 1), not the merged code's logical X.
- Final readout: Z-measure κ + (X-measure data for cross-check).

The current LER curve produced by the walkthrough notebook reflects "merged-code memory error rate", not "PPM failure rate". They are different quantities; the absolute LER and threshold can differ.

This spec rewrites the circuit module to implement the full Cain §III.A surgery protocol while reusing the proven syndrome-extraction machinery from `qldpc.circuits.memory`.

## Goals

1. `build_single_ppm_circuit` and `build_joint_ppm_circuit` produce circuits that faithfully implement Cain §III.A's 3-step protocol: |0⟩-init on κ, τ_s rounds of full-merged-code SE, detach Q' by Z-measurement, X-measure data.
2. PPM observable (Webster Eq. 1) is `OBSERVABLE_INCLUDE(0) = ⊕_{r,i} χ_i^{(r)}`, the XOR of all χ-row syndrome bits across all rounds. For joint PPM, the χ set is `χ^{(1)} ∪ χ^{(2)} ∪ U_B` per math.md §2.7.
3. Secondary observable `OBSERVABLE_INCLUDE(1)` tracks X̄_M (or Z̄_M) from the final data measurement as a cross-check.
4. Round-1 detector classification — reliable stabilizers (data H_X, gauge-fix G) get 1-arg DETECTORs asserting +1; unreliable stabilizers (χ rows, data H_Z) skip the round-1 detector and rely on round-2 consistency (Cross §3.2 D_0 omission).
5. **Basis-symmetric API**: `build_gadget(code, x, *, basis=Pauli.X|Pauli.Z)` and the circuit builders read `gadget.basis` to flip init/detach/observable bases. Default basis is X for backwards compatibility.
6. Reuse infrastructure from `qldpc.circuits.memory`: `QubitIDs`, `MeasurementRecord`, `DetectorRecord`, `SyndromeMeasurementStrategy.EdgeColoring()`, `NoiseModel.noisy_circuit(...)`. **Pattern is borrowed from `_get_basis_memory_experiment_parts` and `_get_qec_cycle`** — we do not call them directly, but the helper structure mirrors them.
7. Per-module LOC budget: `circuit.py` ≤ 300 LOC (currently 184; target ~280).
8. All 47 existing surgery tests continue to pass without modification.

## Non-goals

- Mixed-basis joint PPM (X̄_1 ⊗ Z̄_2). Both gadgets share basis.
- Full Cross §3.2 D_0 projective subspace bookkeeping. Round-1 detectors are either present (reliable) or absent (unreliable); no partial-rank handling.
- Adaptive Pauli corrections applied as Stim instructions. The Cain step-3 corrections are absorbed into the OBSERVABLE_INCLUDE math.
- `basis=None` combined-mode experiments. Single fixed basis per gadget.
- LER threshold extraction from the LER plot. The notebook only verifies LER decreases monotonically with p.

## 1. Module layout

```
src/qldpc/codes/surgery/
  gadget.py          # Edit: add basis to GadgetLayout + build_gadget;
                     #       _step1/3 dispatch on basis.
  bridge.py          # Edit: build_bridge asserts g1.basis == g2.basis;
                     #       Bridge gains `basis` field (or derives from g1).
  circuit.py         # Full rewrite — Cain §III.A protocol.
  cheeger.py         # Small edit: _gadget_to_legacy_layout handles basis swap.
  _test.py           # Add basis tests, observable tests, LER smoke.
  __init__.py        # Unchanged.
```

No new files. `circuit.py` rewrite stays within the 300-LOC budget.

## 2. Public API (unchanged surface, basis added)

```python
def build_gadget(
    code: CSSCode,
    x: np.ndarray,
    *,
    basis: PauliXZ = Pauli.X,
) -> GadgetLayout: ...

@dataclass(frozen=True, eq=False)
class GadgetLayout:
    code: CSSCode
    x: np.ndarray
    basis: PauliXZ                   # NEW
    V0: tuple[int, ...]
    C0: tuple[int, ...]
    F: np.ndarray
    G: np.ndarray
    HX_merged: np.ndarray
    HZ_merged: np.ndarray
    kappa_qubits: tuple[int, ...]

def build_bridge(g1: GadgetLayout, g2: GadgetLayout) -> Bridge: ...
    # Asserts g1.basis == g2.basis; raises ValueError otherwise.

def build_single_ppm_circuit(
    gadget: GadgetLayout,
    *,
    rounds: int,
    noise_model: NoiseModel | None = None,
) -> stim.Circuit: ...

def build_joint_ppm_circuit(
    g1: GadgetLayout, g2: GadgetLayout, bridge: Bridge,
    *,
    rounds: int,
    noise_model: NoiseModel | None = None,
) -> tuple[stim.Circuit, CSSCode]: ...

def boost_gadget(gadget, *, method, target, seed=None, **kw) -> GadgetLayout: ...
    # Already handles GadgetLayout; needs internal swap for basis=Pauli.Z.
```

Backward compatibility: default `basis=Pauli.X` means existing callers (Webster Table I tests, notebook sections 1–5, all 47 existing surgery tests) work unchanged.

## 3. Protocol details

### 3.1 Basis-dependent construction (gadget.py)

| Field | basis=Pauli.X | basis=Pauli.Z |
|---|---|---|
| F (in `_step1_restriction`) | `HZ[C_0, V_0]` | `HX[C_0, V_0]` |
| χ rows added to | `HX_merged` | `HZ_merged` |
| Gauge-fix G rows added to | `HZ_merged` | `HX_merged` |
| `H_Z @ x == 0` check | check `H_Z @ x == 0` | check `H_X @ x == 0` |
| CSS commutation invariant | unchanged (block-equivalent) | unchanged |

### 3.2 Surgery init (Cain step 1)

```python
def _surgery_state_prep(qubit_ids, gadget, bridge_ids=()):
    circuit = stim.Circuit()
    circuit += get_qubit_coordinates(data_ids, check_ids)
    if gadget.basis is Pauli.X:
        circuit.append("RX", data_ids)      # data → |+⟩
        circuit.append("R",  kappa_ids)     # κ → |0⟩
        circuit.append("R",  bridge_ids)    # bridge → |0⟩ (joint only)
    else:  # Pauli.Z
        circuit.append("R",  data_ids)      # data → |0⟩
        circuit.append("RX", kappa_ids)     # κ → |+⟩
        circuit.append("RX", bridge_ids)
    return circuit
```

### 3.3 SE rounds with classified detectors (Cain step 2)

```python
def _surgery_qec_cycle(merged_code, num_rounds, qubit_ids, gadget):
    """Modeled on _get_qec_cycle. Round-1 detectors classified by reliability."""
    one_round, round_record = EdgeColoring().get_circuit(merged_code, qubit_ids)

    reliable_round1 = _classify_reliable_round1_checks(gadget)   # data H_X∪G if X; data H_Z∪G if Z

    circuit = stim.Circuit()
    measurement_record = MeasurementRecord()
    detector_record = DetectorRecord()

    # Round 1
    circuit += one_round
    measurement_record.append(round_record)
    for kk, check_id in enumerate(merged_code.check_ids):
        if check_id in reliable_round1:
            circuit.append("DETECTOR",
                           [measurement_record.get_target_rec(check_id)], (0, 0, kk))
        # else: skip round-1 detector (unreliable)
    detector_record.append({...reliable subset...})

    # Rounds 2..N: full consistency detectors
    if num_rounds > 1:
        repeat = one_round.copy()
        ...standard 2-arg detector wiring per _get_qec_cycle...
        circuit.append(stim.CircuitRepeatBlock(num_rounds - 1, repeat))

    return circuit, measurement_record, detector_record
```

Classification table (basis=Pauli.X):

| row class | row count | reliable in round 1? | reason |
|---|---|---|---|
| data H_X (X-type, data-only) | `m_X` | ✓ | data in |+⟩^⊗n is +1 of all data X-stabs |
| χ rows (X-type, data + κ X-support) | |V_0| | ✗ | X on κ in |0⟩ is uniform random |
| data H_Z (Z-type, data + κ Z-support) | `m_Z` | ✗ | Z on data in |+⟩^⊗n is uniform random |
| gauge-fix G (Z-type, κ-only) | `r` | ✓ | Z on κ in |0⟩^⊗m is +1 |

For basis=Pauli.Z swap all X↔Z above.

### 3.4 Detach + readout (Cain step 3 + user "+final data measurement")

```python
def _surgery_detach_and_readout(qubit_ids, gadget, measurement_record):
    readout = stim.Circuit()
    # Detach Q' (κ + bridge)
    if gadget.basis is Pauli.X:
        readout.append("M",  kappa_ids + bridge_ids)   # Z-measure κ
    else:
        readout.append("MX", kappa_ids + bridge_ids)   # X-measure κ

    # Final detectors for G rows (κ-only): ⊕ Mz/Mx outcomes on G support ⊕ last G syndrome
    readout.append("SHIFT_COORDS", [], (1, 0, 0))
    for k_row in gauge_fix_row_indices:
        targets = [measurement_record.get_target_rec(kappa_ids[j]) for j in nonzero_κ_cols(k_row)]
        targets.append(measurement_record.get_target_rec(gauge_fix_check_id_for_row(k_row), -1))
        readout.append("DETECTOR", targets, (0, 0, ...))

    # Measure data qubits
    if gadget.basis is Pauli.X:
        readout.append("MX", data_ids)
    else:
        readout.append("M",  data_ids)

    # Final detectors for data H_X (or data H_Z) rows: ⊕ M outcomes ⊕ last syndrome
    for hx_row in data_HX_or_HZ_rows:
        targets = [measurement_record.get_target_rec(data_ids[i]) for i in nonzero_data_cols(hx_row)]
        targets.append(measurement_record.get_target_rec(data_check_id_for_row(hx_row), -1))
        readout.append("DETECTOR", targets, (0, 0, ...))

    return readout
```

### 3.5 PPM observable (Webster Eq. 1 / math.md §2.7)

```python
def _surgery_observable(measurement_record, gadget, bridge, num_rounds):
    """Two OBSERVABLE_INCLUDE entries:
       0 = ⊕ χ-row records across all rounds (PPM result, Webster Eq. 1).
       1 = ⊕ data measurements on V_0 (X̄_M / Z̄_M cross-check).
    """
    # Single PPM: χ rows are the |V_0| chi rows of one gadget.
    # Joint PPM: χ rows are χ^{(1)} ∪ χ^{(2)} ∪ U_B rows (math.md §2.7).
    chi_check_ids = ...
    targets = [
        measurement_record.get_target_rec(cid, round_offset=r - num_rounds + 1)
        for r in range(num_rounds)
        for cid in chi_check_ids
    ]
    circuit = stim.Circuit()
    circuit.append("OBSERVABLE_INCLUDE", targets, 0)

    # Cross-check: X̄_M / Z̄_M from final data measurement.
    data_targets = [measurement_record.get_target_rec(data_ids[i]) for i in gadget.V0]
    circuit.append("OBSERVABLE_INCLUDE", data_targets, 1)

    return circuit
```

For joint PPM, `chi_check_ids` includes χ rows from both gadgets plus the U_B path-graph rows; the second observable becomes the XOR of `gadget1.V0` data measurements and `gadget2.V0` data measurements (representing X̄_1 ⊗ X̄_2).

## 4. Tests

### 4.1 Basis-symmetry

```python
@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_build_gadget_css_commutation_both_bases(basis): ...

def test_build_gadget_z_basis_dual_matches_x_basis_on_dual_code():
    """build_gadget(code, z, basis=Z) gives same merged matrices as
    build_gadget(CSSCode(matrix_z, matrix_x), z, basis=X)."""
    ...

def test_webster_table_i_z_basis_kappa_chi_r_exact():
    """Re-run Webster Table I with Z̄_1 seed → 19, 31, 49, 79."""
    ...
```

### 4.2 Surgery circuit semantic tests

```python
@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_single_ppm_circuit_noiseless_observable_is_plus_one(basis):
    """Both OBSERVABLE_INCLUDEs evaluate to 0 (= +1) under no noise."""
    ...

@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_single_ppm_circuit_noise_breaks_observable_at_high_p(basis):
    """At p=0.1, PPM observable flips ≥ 5% of shots."""
    ...

def test_single_ppm_circuit_round_1_detector_classification():
    """Inspect emitted circuit text: round 1 has 1-arg DETECTORs only for
    data H_X (basis=X) or data H_Z (basis=Z) plus G rows."""
    ...

def test_joint_ppm_observable_alpha_star():
    """Joint PPM observable = ⊕ (χ^(1) ∪ χ^(2) ∪ U_B) records = X̄_1 ⊗ X̄_2.
    Noiseless: 0."""
    ...
```

### 4.3 LER smoke

```python
def test_single_ppm_ler_monotone_in_p():
    """3 p values, sinter sweep, LER monotonically increasing.
    ≤ 30 seconds total via BP+OSD."""
    ...
```

### 4.4 Existing tests

The 47 existing surgery tests continue to pass unchanged because `basis=Pauli.X` is the default.

## 5. Migration

### 5.1 Existing callers

| Caller | Change |
|---|---|
| `_test.py` Webster Table I tests | None (default basis). |
| `_test.py` build_gadget tests | None (default basis). |
| `_test.py` joint PPM tests | None — `build_joint_ppm_circuit` still returns `(stim.Circuit, CSSCode)`; the joint code structure unchanged. |
| `examples/test_ide_bb_lp.py` | Replace `CSSCode(HZ, HX)` dual hack with `basis=Pauli.Z` param. |
| `examples/logical_error_rates/_9_lattice_surgery_source.py` | Sections 6, 9, 10 rewritten to use new circuit semantics. Add brief Section 5b on basis. Regenerate `.ipynb`. |
| `examples/scripts/cain_*.py`, `find_bbcode_layouts.py` | None — these only use `build_gadget` + `boost_gadget`, both backward compatible. |

### 5.2 cheeger.py

`_gadget_to_legacy_layout` already operates on a frozen merged code structure. When `gadget.basis is Pauli.Z`, the legacy `SurgeryLayout` needs the dual interpretation (F = HX[C_0, V_0] instead of HZ). One-line conditional inside `_gadget_to_legacy_layout`; estimated ≤ 30 LOC change.

## 6. Risks

- **Basis-symmetry refactor may break a hidden invariant.** Mitigation: the dual-equivalence test (test_build_gadget_z_basis_dual_matches_x_basis_on_dual_code) is a strong canary.
- **Cheeger boost on basis=Z gadgets is currently untested.** Mitigation: parametrize the existing `test_boost_gadget_preserves_css_commutation` over basis.
- **Round-1 detector classification might be incorrect for codes with non-trivial H_X / H_Z structure.** Mitigation: the LER monotonicity smoke test catches gross protocol errors.
- **Notebook regeneration via jupytext** requires `jupyter_client`. If the env doesn't have it, fall back to writing `.ipynb` JSON directly.
- **Joint PPM observable on inter-code joints** depends on `bridge.z_extensions` which may be `None` for trivial inputs (the Steane-Steane smoke test exercises this case). The observable computation must handle `z_extensions is None` gracefully.

## 7. Open questions for follow-up

- Cross §3.2 D_0 projective subspace handling: should round-1 detectors for χ-rows use the sum-of-chi syndrome instead of skipping? This would give a slightly better decoder graph but is more bookkeeping. Deferred.
- Per-gadget noise model differentiation (different p on data vs κ qubits): deferred; current single `NoiseModel` applies uniformly.
- Cain step-3 adaptive Pauli corrections as Stim FEEDBACK operations: deferred. Current observable definition absorbs the corrections algebraically.
