"""Joint Z̄_1 ⊗ Z̄_2 PPM on |a⟩_L ⊗ |b⟩_L — verifies joint PPM is correct.

The joint PPM observable (Webster Eq.1 across the joint chi set) should
equal Z̄_1 ⊗ Z̄_2 eigenvalue:
  |0⟩_L ⊗ |0⟩_L  →  +1·+1 = +1  →  bit 0
  |0⟩_L ⊗ |1⟩_L  →  +1·-1 = -1  →  bit 1
  |1⟩_L ⊗ |0⟩_L  →  -1·+1 = -1  →  bit 1
  |1⟩_L ⊗ |1⟩_L  →  -1·-1 = +1  →  bit 0

Setup: two independent Steane copies (inter-code joint). Each code's data
is initialized to qubit-level |0⟩^⊗n or |1⟩^⊗n via R + (optional X).

For Steane, |1⟩^⊗n is logical |1⟩_L deterministically.

Uses compile_sampler() to read raw OBSERVABLE_INCLUDE bits (avoids the
"deviation from noiseless" confusion of compile_detector_sampler).

Run: python examples/scripts/joint_ppm_correctness_z_demo.py
"""

from __future__ import annotations

import numpy as np
import stim

from qldpc import codes
from qldpc.circuits.surgery import build_gadget, build_bridge, build_joint_ppm_circuit
from qldpc.objects import Pauli


def _mutate_init(circuit: stim.Circuit, x_data_ids: list[int]) -> stim.Circuit:
    """Append X on the listed data ids right after R(data) → those qubits → |1⟩."""
    if not x_data_ids:
        return circuit
    x_set = set(x_data_ids)
    out = stim.Circuit()
    applied = False
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            out.append(stim.CircuitRepeatBlock(
                op.repeat_count, _mutate_init(op.body_copy(), x_data_ids),
            ))
            continue
        out.append(op)
        if not applied and op.name == "R":
            r_targets = [t.value for t in op.targets_copy()]
            x_targets = [q for q in r_targets if q in x_set]
            if x_targets:
                out.append("X", x_targets)
                applied = True
    return out


def raw_observables(circuit: stim.Circuit, shots: int) -> np.ndarray:
    """Sample raw bits and compute each OBSERVABLE_INCLUDE manually."""
    sampler = circuit.compile_sampler()
    raw = sampler.sample(shots=shots).astype(np.uint8)
    n_meas = raw.shape[1]
    obs_lines = [line for line in str(circuit).splitlines() if line.startswith("OBSERVABLE_INCLUDE")]
    cols = []
    for line in obs_lines:
        tokens = [t for t in line.split() if t.startswith("rec[")]
        offsets = [int(t.strip("rec[]")) for t in tokens]
        meas_idx = [n_meas + off for off in offsets]
        cols.append(np.bitwise_xor.reduce(raw[:, meas_idx], axis=1))
    return np.stack(cols, axis=1)


def main() -> None:
    print("Joint Z̄_1 ⊗ Z̄_2 PPM — two Steane copies, inter-code")
    print("=" * 65)

    # Two independent Steane copies (inter-code joint)
    code1 = codes.SteaneCode()
    code2 = codes.SteaneCode()
    assert code1 is not code2

    z1 = np.asarray(code1.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    z2 = np.asarray(code2.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g1 = build_gadget(code1, z1, basis=Pauli.Z)
    g2 = build_gadget(code2, z2, basis=Pauli.Z)
    bridge = build_bridge(g1, g2)
    rounds = 3  # odd so Webster Eq.1 ≡ Z̄_1 ⊗ Z̄_2

    print(f"  code1 = code2 = Steane [[7, 1, 3]]")
    print(f"  basis = Pauli.Z  (joint Z̄_1 ⊗ Z̄_2)")
    print(f"  rounds = {rounds}")
    print(f"  bridge.width = {bridge.width}  (universal adapter, arXiv:2410.03628 §IV)")
    print(f"  extra_κ_l: {bridge.extra_kappa_l.shape[0]} qubits, extra_κ_r: {bridge.extra_kappa_r.shape[0]} qubits")
    print(f"  T_l shape  : {bridge.T_l.shape}  (SkipTree, (3,2)-sparse)")
    print(f"  H_R shape  : {bridge.H_R.shape}  (canonical rep-code parity)")

    circuit, joint_code = build_joint_ppm_circuit(g1, g2, bridge, rounds=rounds, noise_model=None)
    print(f"  joint code: [[{joint_code.num_qudits}, {joint_code.dimension}]]")

    # Inter-code register layout: data_1 + data_2 + kappa_1 + kappa_2 + bridge
    n1 = code1.num_qudits  # 7
    n2 = code2.num_qudits  # 7
    data1_ids = list(range(n1))
    data2_ids = list(range(n1, n1 + n2))

    shots = 1000
    print(f"\n  4 init states × {shots} shots each:")
    print(f"  {'state':>14} | {'Z̄_1':>4} {'Z̄_2':>4} | {'expected obs0':>14} | {'measured obs0 (bit=1 frac)':>30}")
    print("  " + "-" * 80)

    test_cases = [
        ("|0⟩_L ⊗ |0⟩_L", [],                       0, 0, 0),
        ("|0⟩_L ⊗ |1⟩_L", data2_ids,                0, 1, 1),
        ("|1⟩_L ⊗ |0⟩_L", data1_ids,                1, 0, 1),
        ("|1⟩_L ⊗ |1⟩_L", data1_ids + data2_ids,    1, 1, 0),
    ]

    all_pass = True
    for label, flip_ids, z1_bit, z2_bit, expected_obs0 in test_cases:
        mutated = _mutate_init(circuit, flip_ids)
        obs = raw_observables(mutated, shots)
        obs0_rate = float(obs[:, 0].mean())
        flag = "✓" if obs0_rate == float(expected_obs0) else "✗"
        z1s = "-1" if z1_bit else "+1"
        z2s = "-1" if z2_bit else "+1"
        print(
            f"  {label:>14} | {z1s:>4} {z2s:>4} | {expected_obs0:>14} | "
            f"{obs0_rate:>10.2%} ({int(obs[:, 0].sum()):>4}/{shots})  {flag}"
        )
        if obs0_rate != float(expected_obs0):
            all_pass = False

    print()
    if all_pass:
        print("  ✓ All 4 init states give the correct Z̄_1 ⊗ Z̄_2 eigenvalue deterministically.")
        print("  Webster Eq.1 joint observable matches the expected parity in every shot.")
    else:
        print("  ✗ At least one init state failed. Inspect mismatched rows above.")


if __name__ == "__main__":
    main()
