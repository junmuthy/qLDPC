"""Single-PPM with data in logical |0⟩: PPM must return random ±1, and
both observables (Webster Eq. 1 + X̄_M cross-check) must AGREE per shot.

This is a strictly stronger noiseless correctness test than the |+⟩-init
demo. With data in |+⟩_logical, X̄_M = +1 deterministically so both
observables are always 0 — they could trivially "agree" by both being
stuck. With data in |0⟩_logical, X̄_M is genuinely 50/50 random (QM:
measuring X on a Z-eigenstate). The two observables agree on every shot
ONLY IF they really measure the same X̄_M operator.

We use rounds=3 (odd) so Webster Eq. 1 = ⊕_{r,i} χ_i^(r) reduces to X̄_M
(over GF(2): 3 × X̄_M = X̄_M).

Note: we mutate the circuit by swapping RX(data) → R(data) because the
public API doesn't currently expose data-init basis. Some detectors will
fire because round-1 detector classification assumed data in |+⟩ — that's
expected and is NOT a correctness issue for the PPM observable.

Run: python examples/scripts/single_ppm_correctness_z_init_demo.py
"""

from __future__ import annotations

import numpy as np
import stim

from qldpc import codes
from qldpc.codes.surgery import build_gadget, build_single_ppm_circuit
from qldpc.objects import Pauli


def _swap_data_init_to_zero(circuit: stim.Circuit, data_ids: list[int]) -> stim.Circuit:
    """Return a new circuit with RX(data_ids) replaced by R(data_ids).

    This changes the data initial state from |+⟩ (X-eigenstate) to |0⟩
    (Z-eigenstate). Other RX instructions (on κ, bridge) are preserved.
    """
    data_set = set(data_ids)
    out = stim.Circuit()
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            out.append(stim.CircuitRepeatBlock(
                op.repeat_count,
                _swap_data_init_to_zero(op.body_copy(), data_ids),
            ))
            continue
        if op.name == "RX":
            targets = [t.value for t in op.targets_copy()]
            data_targets = [t for t in targets if t in data_set]
            other_targets = [t for t in targets if t not in data_set]
            if data_targets:
                out.append("R", data_targets)
            if other_targets:
                out.append("RX", other_targets)
        else:
            out.append(op)
    return out


def main() -> None:
    print("Single-PPM with data init |0⟩_logical (X̄ PPM on Z-eigenstate)")
    print("=" * 65)

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    rounds = 3  # odd → Webster Eq.1 = X̄_M
    circuit_plus = build_single_ppm_circuit(g, rounds=rounds, noise_model=None)

    # Mutate data init |+⟩ → |0⟩
    n_data = code.num_qudits
    data_ids = list(range(n_data))
    circuit_zero = _swap_data_init_to_zero(circuit_plus, data_ids)

    print(f"  code                 : Steane [[7, 1, 3]]")
    print(f"  data init            : |0⟩^⊗n (X-stabs random → projects to |0⟩_L or |1⟩_L)")
    print(f"  rounds               : {rounds} (odd → Webster Eq.1 ≡ X̄_M)")
    print(f"  expected PPM outcome : random ±1 (X̄_M on Z-eigenstate)")

    shots = 4000
    sampler = circuit_zero.compile_detector_sampler()
    _detectors, observables = sampler.sample(shots=shots, separate_observables=True)

    obs0 = observables[:, 0]
    obs1 = observables[:, 1]
    obs0_rate = float(obs0.mean())
    obs1_rate = float(obs1.mean())
    agree_rate = float((obs0 == obs1).mean())

    print(f"\n  shots                              : {shots}")
    print(f"  observable 0 (Webster Eq.1) flip rate : {obs0_rate:.2%} (expected ~50%)")
    print(f"  observable 1 (X̄_M cross-check)  flip rate : {obs1_rate:.2%} (expected ~50%)")
    print(f"  obs0 == obs1 per shot                : {agree_rate:.2%} (expected 100%)")

    # Correctness assertions:
    # 1. Genuine quantum randomness — observable is not stuck at 0 or 1.
    assert 0.4 < obs0_rate < 0.6, f"obs0 rate {obs0_rate:.2%} not near 50% — observable broken"
    assert 0.4 < obs1_rate < 0.6, f"obs1 rate {obs1_rate:.2%} not near 50%"
    # 2. CRITICAL: the two observables must agree on every shot (they measure the same X̄_M).
    assert agree_rate == 1.0, (
        f"obs0 and obs1 disagree on {int((1 - agree_rate) * shots)} shots "
        f"out of {shots} — they don't measure the same X̄_M!"
    )

    print()
    print("  ✓ Genuine 50/50 quantum randomness (X̄ on |0⟩_L)")
    print("  ✓ Webster Eq.1 and X̄_M cross-check agree on EVERY shot")
    print("    → both observables really measure the same X̄_M operator")


if __name__ == "__main__":
    main()
