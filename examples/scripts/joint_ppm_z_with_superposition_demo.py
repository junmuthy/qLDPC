"""Joint Z̄_1 ⊗ Z̄_2 PPM with code 2 in |+⟩_L — verifies quantum randomness.

Setup: two independent Steane copies, inter-code joint, basis=Pauli.Z.
- code 1 data: |0⟩^⊗n (built-in R) → logical |0⟩_L → Z̄_1 = +1 deterministic
- code 2 data: |+⟩^⊗n (mutated R → RX) → logical |+⟩_L → Z̄_2 = random ±1

Joint observable obs0 = Webster Eq.1 = 3·(Z̄_1 ⊗ Z̄_2) mod 2 = Z̄_1 ⊗ Z̄_2.
Since Z̄_1 = +1 and Z̄_2 is random ±1, obs0 should be random 50/50.

Critical consistency: obs0 (intermediate-syndrome path) and obs1 (final
Mz on V_0 path) must agree on every shot — they measure the same Z̄_1⊗Z̄_2.

This is strictly stronger than the deterministic ZZ-truth-table demo: a
broken protocol could fake deterministic agreement (both stuck at 0),
but two independent paths agreeing on a random bit cannot be faked.

Run: python examples/scripts/joint_ppm_z_with_superposition_demo.py
"""

from __future__ import annotations

import numpy as np
import stim
import sympy

from qldpc import codes
from qldpc.circuits.surgery import build_gadget, build_bridge, build_joint_ppm_circuit
from qldpc.objects import Pauli


def _switch_init_basis(circuit: stim.Circuit, plus_ids: list[int]) -> stim.Circuit:
    """Move `plus_ids` from the R group into a separate RX group → |+⟩^⊗ init."""
    if not plus_ids:
        return circuit
    plus_set = set(plus_ids)
    out = stim.Circuit()
    applied = False
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            out.append(stim.CircuitRepeatBlock(
                op.repeat_count, _switch_init_basis(op.body_copy(), plus_ids),
            ))
            continue
        if not applied and op.name == "R":
            targets = [t.value for t in op.targets_copy()]
            remain = [q for q in targets if q not in plus_set]
            move   = [q for q in targets if q in plus_set]
            if remain:
                out.append("R", remain)
            if move:
                out.append("RX", move)
                applied = True
            continue
        out.append(op)
    return out


def raw_observables(circuit: stim.Circuit, shots: int) -> np.ndarray:
    sampler = circuit.compile_sampler()
    raw = sampler.sample(shots=shots).astype(np.uint8)
    n_meas = raw.shape[1]
    obs_lines = [l for l in str(circuit).splitlines() if l.startswith("OBSERVABLE_INCLUDE")]
    cols = []
    for line in obs_lines:
        tokens = [t for t in line.split() if t.startswith("rec[")]
        offsets = [int(t.strip("rec[]")) for t in tokens]
        meas_idx = [n_meas + off for off in offsets]
        cols.append(np.bitwise_xor.reduce(raw[:, meas_idx], axis=1))
    return np.stack(cols, axis=1)


def main() -> None:
    print("Joint Z̄_1 ⊗ Z̄_2 PPM with code 2 in |+⟩_L")
    print("=" * 65)

    x, y = sympy.symbols("x y")
    code1 = codes.BBCode({x: 6, y: 12}, x**3 + y + y**2, y**3 + x + x**2)
    code2 = codes.BBCode({x: 31, y: 4}, 1 + x**6 * y + x ** 27, y**2 + x**15*y**3 + x**24)
    z1 = np.asarray(code1.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    z2 = np.asarray(code2.get_logical_ops(Pauli.Z)[4]).astype(np.uint8)
    g1 = build_gadget(code1, z1, basis=Pauli.Z)
    g2 = build_gadget(code2, z2, basis=Pauli.Z)
    bridge = build_bridge(g1, g2)
    rounds = 3

    circuit_z, joint = build_joint_ppm_circuit(g1, g2, bridge, rounds=rounds, noise_model=None)

    n1, n2 = code1.num_qudits, code2.num_qudits
    data2_ids = list(range(n1, n1 + n2))   # code 2 data qubits
    circuit_mixed = _switch_init_basis(circuit_z, data2_ids)

    print(f"  code1 = {code1}")
    print(f"  code2 = {code2}")
    print(f"  joint code: [[{joint.num_qudits}, {joint.dimension}]]")
    print(f"  basis = Pauli.Z (joint Z̄_1 ⊗ Z̄_2), rounds = {rounds}")
    print(f"  init: data 1 → |0⟩^⊗{n1} (R)   data 2 → |+⟩^⊗{n2} (R → RX mutation)")
    print(f"  expected: Z̄_1 = +1 deterministic, Z̄_2 = random ±1")
    print(f"            obs0 = Z̄_1 ⊗ Z̄_2 = random 50/50")
    print(f"            obs1 = same Z̄_1 ⊗ Z̄_2, MUST agree with obs0 per shot")

    shots = 4000
    obs = raw_observables(circuit_mixed, shots)
    obs0 = obs[:, 0]
    obs1 = obs[:, 1]
    rate0 = float(obs0.mean())
    rate1 = float(obs1.mean())
    agree = float((obs0 == obs1).mean())

    print(f"\n  shots                : {shots}")
    print(f"  obs0 bit=1 rate      : {rate0:.2%}  (expected ~50%)")
    print(f"  obs1 bit=1 rate      : {rate1:.2%}  (expected ~50%)")
    print(f"  obs0 == obs1 per shot: {agree:.2%}  (expected 100%)")

    assert 0.40 < rate0 < 0.60, f"obs0 rate {rate0:.2%} not near 50% — not random?"
    assert 0.40 < rate1 < 0.60, f"obs1 rate {rate1:.2%} not near 50%"
    assert agree == 1.0, (
        f"obs0 vs obs1 disagree on {int((1 - agree) * shots)} of {shots} shots — "
        f"they don't measure the same Z̄_1⊗Z̄_2!"
    )

    print()
    print("  ✓ Genuine quantum randomness (Z̄_2 on |+⟩_L collapses 50/50)")
    print("  ✓ Webster Eq.1 obs0 and final-Mz cross-check obs1 agree on EVERY shot")
    print("    → both paths really measure the same Z̄_1 ⊗ Z̄_2 operator")


if __name__ == "__main__":
    main()
