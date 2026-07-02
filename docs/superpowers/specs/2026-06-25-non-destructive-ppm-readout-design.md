# Non-destructive (detach-only) readout mode for surgery PPM builders

**Date:** 2026-06-25
**Status:** Design approved

## Problem

The four lattice-surgery measurement builders always end with a **destructive
single-qubit readout** of the data qubits:

- `build_single_ppm_circuit` (X / Z single PPM)
- `build_joint_ppm_circuit` (ZZ / same-basis joint PPM)
- `build_single_y_ppm_circuit` (Y single PPM)

That destructive readout is a **benchmark / memory-experiment** convenience, not
part of the surgery protocol. The logical Pauli result (`obs0`) is the product
of the **in-circuit last-QEC-round meas-check ancilla outcomes**, fully
determined *before* any detach or data measurement (verified this session:
obs0's records are all QEC-ancilla records, strictly before the first detach
measurement). The destructive readout only supplies the `obs1` cross-check and
the final boundary detectors.

For real use — and for cleaner notebook demos — we want a **non-destructive**
mode: run the merge rounds, detach the gadget ancillas, and leave the data
qubits encoded as the post-measurement logical state. This is exactly how
surface-code surgery works (measure the merge ancillas, then split; never
measure the data).

This is especially valuable for **Y / mixed-basis** measurements, where the
destructive all-Y readout requires the contorted "X·Z = Y" null-space
combination machinery in `_y_final_detectors`. The non-destructive mode
sidesteps that entirely.

## API

Add one keyword to each of the three builders:

```python
destructive_measure_data: bool = True
```

- `True` (default): current behavior, **byte-identical** to today.
- `False`: detach-only / non-destructive — the final round does **not** measure
  the data qubits.

The same name and semantics across all three builders.

## Behavior matrix

| phase | `destructive_measure_data=True` (today) | `=False` (new) |
|-------|------------------------------------------|----------------|
| state prep + QEC rounds | unchanged | **unchanged** |
| detach (measure κ + bridge ancillas) | yes | **yes** — restores the bare code |
| destructive data readout (M/MX/MY on real data) | yes | **skipped** |
| final destructive detectors | emitted (reconstruct stabilizers from data+κ) | **skipped** (depend on the data readout) |
| `obs0` (logical Pauli result) | emitted from in-circuit last-round meas-checks | **emitted** — identical, it is pre-detach |
| `obs1` (X/Z destructive cross-check) | emitted | **skipped** (no data to cross-check) |
| in-circuit round-to-round detectors | emitted | **unchanged** |

Net effect of `=False`: the data qubits are left encoded, and the circuit has
**much fewer detectors** (no destructive boundary detectors). The in-circuit
round detectors + `obs0` remain, so the circuit still compiles to a DEM.

## Correctness invariants

1. **Same logical result.** `obs0` reads only in-circuit meas-check records, so
   its value is identical between the two modes on any prepared state.
2. **DEM compiles.** `obs0` and the in-circuit detectors are deterministic on a
   prepared eigenstate, independent of the data readout.
3. **Detach kept.** The κ / bridge ancillas are still measured out (the split),
   which is the part that returns the bare code.

## Guards

- **Y survivor-memory** (`memory_logical`) reads a surviving logical Z̄ from the
  destructive data readout, so it is incompatible with the non-destructive mode:
  `destructive_measure_data=False` together with `memory_logical is not None`
  raises a clear `ValueError`.
- `obs1` gating for X/Z and the `obs0` determinism gate for Y are unchanged in
  the `=True` path.

## Implementation sketch

Thread `destructive_measure_data` into the detach/readout/observable/final-detector
helpers; each gets one guard:

- `_surgery_detach_and_readout` — keep the κ+bridge detach measurement; skip the
  `data_op data_ids` line when `destructive_measure_data=False`.
- `_surgery_final_detectors` — return an empty fragment (or is not called) when
  `destructive_measure_data=False`.
- `_surgery_observable` — emit `obs0` only; skip the `obs1` (data) entry when
  `destructive_measure_data=False`.
- Y equivalents: `_y_detach_and_readout` (skip the data MY), skip
  `_y_final_detectors`, keep `_y_emit_obs0`.

Builders pass the flag through; `build_joint_ppm_circuit` preserves its
`(circuit, ...)` return signature.

## Tests (TDD, per builder: X, Z, ZZ, Y)

For `destructive_measure_data=False`:

1. **No data readout** — no destructive measurement targets any real-data qubit.
2. **Detach kept** — the κ (and bridge) ancillas are still measured.
3. **obs0 emitted** — exactly one (the logical-Pauli) observable for X/Z/ZZ; for
   Y, obs0 emitted on a Y-eigenstate prep (existing determinism gate).
4. **DEM compiles** — `circuit.detector_error_model()` succeeds.
5. **Same logical value** — obs0 on a prepared eigenstate matches the
   `destructive_measure_data=True` build.
6. **Fewer detectors** — `num_detectors(False) < num_detectors(True)`.
7. **Default unchanged** — `destructive_measure_data=True` is byte-identical to
   the pre-change build (regression guard).
8. **Guard** — Y with `memory_logical` set and `destructive_measure_data=False`
   raises `ValueError`.
