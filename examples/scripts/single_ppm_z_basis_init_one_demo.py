"""Z̄ PPM on logical |1⟩ — verifies protocol detects -1 deterministically.

Symmetric to "X̄ PPM on |+⟩ = +1" but rotated 90°:
- Build basis=Pauli.Z gadget (measures Z̄_M).
- Default data init is |0⟩ via R.
- We mutate: append X after R on data → data in qubit-level |1⟩^⊗n.
- For Steane (and many CSS codes), |1⟩^⊗n is logical |1⟩_L:
    X^⊗n on |0⟩^⊗n = X̄ · (stab) |0⟩^⊗n ∝ |1⟩_L (modulo stabilizer).
- Z̄_M |1⟩_L = -1 |1⟩_L → PPM outcome = -1 (= bit 1).

Use rounds=3 (odd) so Webster Eq.1 = 3·Z̄ mod 2 = Z̄ = bit 1.

IMPORTANT: we use compile_sampler() to read RAW M-record bits and compute
the observables manually. `compile_detector_sampler()` returns deviations
from stim's deterministic noiseless trace, not raw bit values — for a
state where the noiseless observable is 1, the deviation reads as 0%
even though the bit is always 1.

Run: python examples/scripts/single_ppm_z_basis_init_one_demo.py
"""

from __future__ import annotations

import numpy as np
import stim

from qldpc import codes
from qldpc.circuits.surgery import build_gadget, build_single_ppm_circuit
from qldpc.objects import Pauli


def _flip_data_to_one(circuit: stim.Circuit, data_ids: list[int]) -> stim.Circuit:
    """Insert X(data_ids) after R(data_ids) → data prepared in |1⟩^⊗n."""
    data_set = set(data_ids)
    out = stim.Circuit()
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            out.append(stim.CircuitRepeatBlock(
                op.repeat_count,
                _flip_data_to_one(op.body_copy(), data_ids),
            ))
            continue
        if op.name == "R":
            targets = [t.value for t in op.targets_copy()]
            data_targets = [t for t in targets if t in data_set]
            other_targets = [t for t in targets if t not in data_set]
            if other_targets:
                out.append("R", other_targets)
            if data_targets:
                out.append("R", data_targets)
                out.append("X", data_targets)
        else:
            out.append(op)
    return out


def raw_observables(circuit: stim.Circuit, shots: int) -> np.ndarray:
    """Sample raw M-record bits and compute each OBSERVABLE_INCLUDE bit manually.

    Returns array of shape (shots, n_observables) of raw bit values.
    """
    sampler = circuit.compile_sampler()
    raw = sampler.sample(shots=shots).astype(np.uint8)
    n_meas = raw.shape[1]

    obs_lines = [line for line in str(circuit).splitlines() if line.startswith("OBSERVABLE_INCLUDE")]
    cols = []
    for line in obs_lines:
        tokens = [t for t in line.split() if t.startswith("rec[")]
        offsets = [int(t.strip("rec[]")) for t in tokens]  # negative ints
        meas_idx = [n_meas + off for off in offsets]
        cols.append(np.bitwise_xor.reduce(raw[:, meas_idx], axis=1))
    return np.stack(cols, axis=1)


def main() -> None:
    print("Z̄ PPM on logical |1⟩ — Steane")
    print("=" * 65)

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    print(f"  code  : Steane [[7, 1, 3]]")
    print(f"  Z̄_M  : Z on qubits {tuple(int(i) for i in np.where(z)[0])}")
    print(f"  V_0   : {g.V0}  (same — the support of Z̄_M)")
    print(f"  basis : Pauli.Z (measure Z̄_M)")

    rounds = 3
    shots = 2000
    data_ids = list(range(code.num_qudits))

    # Baseline: |0⟩^⊗n (built-in init). Z̄ on |0⟩_L = +1 → bit 0.
    print(f"\n  --- Baseline: data init |0⟩^⊗n (built-in) ---")
    circuit_zero = build_single_ppm_circuit(g, rounds=rounds, noise_model=None)
    raw = raw_observables(circuit_zero, shots)
    print(f"  obs0 (Webster Eq.1) bit=1 rate : {raw[:, 0].mean():.2%}  (expected: 0%, Z̄=+1)")
    print(f"  obs1 (Mz on V_0)    bit=1 rate : {raw[:, 1].mean():.2%}  (expected: 0%, Z̄=+1)")
    print(f"  obs0 == obs1 per shot          : {(raw[:, 0] == raw[:, 1]).mean():.2%}")
    assert raw[:, 0].sum() == 0 and raw[:, 1].sum() == 0, "|0⟩ baseline should give bit 0"

    # Mutate: |1⟩^⊗n. Z̄ on |1⟩_L = -1 → bit 1.
    print(f"\n  --- After mutation: data init |1⟩^⊗n (R then X) ---")
    circuit_one = _flip_data_to_one(circuit_zero, data_ids)
    raw = raw_observables(circuit_one, shots)
    print(f"  obs0 (Webster Eq.1) bit=1 rate : {raw[:, 0].mean():.2%}  (expected: 100%, Z̄=-1)")
    print(f"  obs1 (Mz on V_0)    bit=1 rate : {raw[:, 1].mean():.2%}  (expected: 100%, Z̄=-1)")
    print(f"  obs0 == obs1 per shot          : {(raw[:, 0] == raw[:, 1]).mean():.2%}")
    assert raw[:, 0].sum() == shots and raw[:, 1].sum() == shots, \
        "|1⟩ init should give bit 1 deterministically"
    assert (raw[:, 0] == raw[:, 1]).all(), "observables disagree on some shot"

    print("\n" + "=" * 65)
    print("✓ Z̄ PPM correctly distinguishes |0⟩_L vs |1⟩_L:")
    print("  - |0⟩^⊗n → Z̄ = +1 → both obs bit 0 (deterministic, 0/2000 flips)")
    print("  - |1⟩^⊗n → Z̄ = -1 → both obs bit 1 (deterministic, 2000/2000 flips)")
    print("  Webster Eq.1 (rounds=3) and final-Mz cross-check agree on every shot.")


if __name__ == "__main__":
    main()
