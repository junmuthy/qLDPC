"""Circuit-level Cain §IV.C X-basis surgery experiment for Webster code 0.

Replaces noiseless MPP from cain_fig1b_full_protocol.py with explicit
ancilla-based syndrome extraction (CNOTs from data ↔ syndrome ancilla),
then applies DepolarizingNoiseModel which puts depolarizing noise on every
1q/2q gate, plus init/measure flip noise. This is what Cain uses.

Protocol per Cain §IV.C:
  - Data Q: RX init (|+⟩), MX final
  - Ancilla Q' (κ + bridge): RZ init (|0⟩), MZ final
  - τ_s rounds of joint code stabilizer measurement
  - k + t = 11 observables:
      * 1 target observable = XOR of first-round X-stabs touching ancilla
      * 10 data X̄ observables = XOR of final MX on supp(X̄_i)

Syndrome ancillas (for stab extraction) live in qubit indices beyond joint
code's n=86. Each X-stab needs 1 ancilla (init |+⟩, CX to data, M_X), each
Z-stab needs 1 ancilla (init |0⟩, CX from data, M_Z).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sinter
import stim

from qldpc import circuits, codes, decoders
from qldpc.codes.surgery import (
    boost_gadget,
    build_bridge,
    build_gadget,
    load_webster_seed_set,
)
from qldpc.codes.surgery.circuit import _stitch_to_joint_csscode
from qldpc.codes.surgery.gadget import _build_generalised_bicycle_code
from qldpc.objects import Pauli


def _stitch_compat(g1_boosted, g2_boosted):
    """Compat shim: returns (joint_csscode, None) like old _stitch_gadgets_with_bridge."""
    bridge = build_bridge(g1_boosted, g2_boosted)
    joint = _stitch_to_joint_csscode(g1_boosted, g2_boosted, bridge)
    return joint, None


def _seed_to_vec(seed: dict, l: int) -> np.ndarray:
    v = np.zeros(2 * l, dtype=np.int_)
    for i in seed["L_support"]:
        v[i] = 1
    for i in seed["R_support"]:
        v[l + i] = 1
    return v


def build_setup():
    data = load_webster_seed_set(0)
    data_code = _build_generalised_bicycle_code(
        l=data["l"], A_set=data["A"], B_set=data["B"]
    )
    x_seeds = [s for s in data["seeds"] if s["pauli_type"] == "X"]
    op1 = _seed_to_vec(x_seeds[0], data["l"])
    op2 = _seed_to_vec(x_seeds[1], data["l"])
    g1 = build_gadget(data_code, op1)
    g2 = build_gadget(data_code, op2)
    g1_boosted = boost_gadget(g1, method='combinatorial', target=1.0, max_extra_qubits=10, seed=42)
    g2_boosted = boost_gadget(g2, method='combinatorial', target=1.0, max_extra_qubits=10, seed=42)
    joint, _ = _stitch_compat(g1_boosted, g2_boosted)
    return data_code, joint, data_code.num_qubits


def build_circuit_level_cain_x_basis(
    joint_code: codes.CSSCode,
    data_code: codes.CSSCode,
    n_data: int,
    num_rounds: int,
) -> stim.Circuit:
    """Build a noiseless circuit; noise applied later by DepolarizingNoiseModel.

    Qubit register layout:
      - 0 .. n_data-1                          : data qubits (RX init)
      - n_data .. n_total-1                    : code ancilla κ + bridge (RZ init)
      - n_total .. n_total + n_X_stabs - 1     : X-stab syndrome ancillas
      - n_total + n_X_stabs ..                 : Z-stab syndrome ancillas
    """
    n_total = joint_code.num_qubits
    HX = np.asarray(joint_code.matrix_x).astype(np.int_)
    HZ = np.asarray(joint_code.matrix_z).astype(np.int_)
    n_x_stabs = HX.shape[0]
    n_z_stabs = HZ.shape[0]

    data_qubits = list(range(n_data))
    code_anc_qubits = list(range(n_data, n_total))
    x_synd_qubits = list(range(n_total, n_total + n_x_stabs))
    z_synd_qubits = list(range(n_total + n_x_stabs, n_total + n_x_stabs + n_z_stabs))

    x_pure_data_mask = ~np.any(HX[:, n_data:] != 0, axis=1)
    x_target_mask = ~x_pure_data_mask  # chi and U_B rows

    circ = stim.Circuit()

    # Initial state preparation.
    circ.append("RX", data_qubits)
    if code_anc_qubits:
        circ.append("RZ", code_anc_qubits)
    circ.append("TICK")

    # Track first-round and previous-round syndrome ancilla measurement records.
    x_records_per_round: list[list[int]] = []
    z_records_per_round: list[list[int]] = []

    for round_idx in range(num_rounds):
        # X-stab measurement: init ancilla |+⟩, CX from ancilla to data,
        # then MX ancilla.
        circ.append("RX", x_synd_qubits)
        circ.append("RZ", z_synd_qubits)
        circ.append("TICK")

        # Build CX gate list for X-stab extraction. We schedule all CXs in
        # a few layers based on edge coloring would be ideal; for simplicity
        # we do them sequentially (still circuit-level noise applies).
        for stab_idx, stab in enumerate(HX):
            data_targets = np.flatnonzero(stab)
            for q in data_targets:
                # CX from synd ancilla (control) to data qubit (target).
                circ.append("CX", [x_synd_qubits[stab_idx], int(q)])
        circ.append("TICK")

        # Z-stab measurement: CX from data to ancilla.
        for stab_idx, stab in enumerate(HZ):
            data_targets = np.flatnonzero(stab)
            for q in data_targets:
                circ.append("CX", [int(q), z_synd_qubits[stab_idx]])
        circ.append("TICK")

        # Measure syndrome ancillas.
        x_recs_this_round = []
        for stab_idx, q in enumerate(x_synd_qubits):
            circ.append("MX", [q])
            x_recs_this_round.append(circ.num_measurements - 1)

        z_recs_this_round = []
        for stab_idx, q in enumerate(z_synd_qubits):
            circ.append("MZ", [q])
            z_recs_this_round.append(circ.num_measurements - 1)

        x_records_per_round.append(x_recs_this_round)
        z_records_per_round.append(z_recs_this_round)

        # Detectors.
        if round_idx == 0:
            # Only pure-data X-stabs are +1 deterministic under |+⟩^n_data.
            for stab_idx, rec in enumerate(x_recs_this_round):
                if x_pure_data_mask[stab_idx]:
                    offset = rec - circ.num_measurements
                    circ.append("DETECTOR", [stim.target_rec(offset)])
        else:
            # Compare X-stabs round-to-round.
            prev_x = x_records_per_round[round_idx - 1]
            for cur_rec, prev_rec in zip(x_recs_this_round, prev_x):
                off_c = cur_rec - circ.num_measurements
                off_p = prev_rec - circ.num_measurements
                circ.append(
                    "DETECTOR", [stim.target_rec(off_c), stim.target_rec(off_p)],
                )
            prev_z = z_records_per_round[round_idx - 1]
            for cur_rec, prev_rec in zip(z_recs_this_round, prev_z):
                off_c = cur_rec - circ.num_measurements
                off_p = prev_rec - circ.num_measurements
                circ.append(
                    "DETECTOR", [stim.target_rec(off_c), stim.target_rec(off_p)],
                )

        circ.append("TICK")

    # Final measurements: MX on data, MZ on code ancilla.
    final_data_mx_base = circ.num_measurements
    circ.append("MX", data_qubits)
    if code_anc_qubits:
        circ.append("MZ", code_anc_qubits)

    # Final-round detectors using M_X on data:
    # For pure-data X-stabs, XOR of final MX on data support should equal
    # round-τ_s X-stab outcome.
    final_round_x = x_records_per_round[-1]
    for stab_idx, rec in enumerate(final_round_x):
        if not x_pure_data_mask[stab_idx]:
            continue
        data_support = np.flatnonzero(HX[stab_idx, :n_data])
        targets = [
            stim.target_rec(final_data_mx_base + int(q) - circ.num_measurements)
            for q in data_support
        ]
        targets.append(stim.target_rec(rec - circ.num_measurements))
        circ.append("DETECTOR", targets)

    # Observables.
    # Target (Cain t=1): XOR of first-round X-stabs touching ancilla.
    first_round_x = x_records_per_round[0]
    target_records = []
    for stab_idx, rec in enumerate(first_round_x):
        if x_target_mask[stab_idx]:
            target_records.append(stim.target_rec(rec - circ.num_measurements))
    circ.append("OBSERVABLE_INCLUDE", target_records, [0])

    # k data X̄ observables: XOR of final MX on supp(X̄_i).
    data_x_logicals = np.asarray(data_code.get_logical_ops(Pauli.X)).astype(np.int_)
    for i, lop in enumerate(data_x_logicals):
        targets = [
            stim.target_rec(final_data_mx_base + int(q) - circ.num_measurements)
            for q in np.flatnonzero(lop)
        ]
        circ.append("OBSERVABLE_INCLUDE", targets, [i + 1])

    return circ


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--num-rounds", type=int, default=3)
    args = parser.parse_args()

    print("Building Webster code 0 + joint surgery code...")
    data_code, joint, n_data = build_setup()
    print(f"  data: [[{data_code.num_qubits}, {data_code.dimension}]]")
    print(f"  joint: [[{joint.num_qubits}, {joint.dimension}]]\n")

    noiseless = build_circuit_level_cain_x_basis(
        joint, data_code, n_data, num_rounds=args.num_rounds,
    )
    print(f"Noiseless circuit: {noiseless.num_qubits} qubits, "
          f"{noiseless.num_measurements} measurements, "
          f"{noiseless.num_detectors} detectors, "
          f"{noiseless.num_observables} observables")
    try:
        noiseless.detector_error_model(decompose_errors=False)
        print("  ✓ DEM accepts noiseless circuit\n")
    except Exception as e:
        print(f"  ✗ DEM rejects: {e}\n")
        return

    if args.quick:
        error_rates = [1e-3, 3e-3, 1e-2]
        max_shots = 5_000
        max_errors = 50
    else:
        error_rates = list(np.logspace(-3.5, -1.5, 7))
        max_shots = 50_000
        max_errors = 100

    tasks = []
    for p in error_rates:
        noise = circuits.DepolarizingNoiseModel(p, include_idling_error=False)
        noisy = noise.noisy_circuit(noiseless)
        tasks.append(sinter.Task(circuit=noisy, json_metadata={"code": "joint", "p": p}))

        data_noise = circuits.DepolarizingNoiseModel(p, include_idling_error=False)
        data_circ = circuits.get_memory_experiment(
            data_code, basis=Pauli.X, num_rounds=args.num_rounds, noise_model=data_noise,
        )
        # Keep only 1 observable for data baseline.
        keep_one = stim.Circuit()
        kept = False
        for inst in data_circ.flattened():
            if inst.name == "OBSERVABLE_INCLUDE":
                if kept:
                    continue
                keep_one.append("OBSERVABLE_INCLUDE", inst.targets_copy(), [0])
                kept = True
            else:
                keep_one.append(inst.name, inst.targets_copy(), inst.gate_args_copy())
        tasks.append(sinter.Task(circuit=keep_one, json_metadata={"code": "data", "p": p}))

    num_workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"Running sinter, {num_workers} workers, {len(tasks)} tasks...")
    t0 = time.time()
    stats = sinter.collect(
        tasks=tasks,
        decoders=["bp_osd"],
        custom_decoders={"bp_osd": decoders.SinterDecoder(with_BP_OSD=True)},
        num_workers=num_workers,
        max_shots=max_shots,
        max_errors=max_errors,
        print_progress=True,
    )
    print(f"\nSweep complete in {time.time() - t0:.1f}s\n")

    by_code: dict[str, list[tuple[float, float, int, int]]] = {}
    for s in stats:
        label = s.json_metadata["code"]
        p = float(s.json_metadata["p"])
        if s.shots == 0:
            continue
        by_code.setdefault(label, []).append((p, s.errors / s.shots, s.shots, s.errors))

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"data": "tab:blue", "joint": "tab:red"}
    markers = {"data": "o", "joint": "s"}
    labels = {
        "data": "data [[62, 10, 6]] X memory (1 obs)",
        "joint": "joint surgery [[86, 9]] Cain X-basis (k+t obs)",
    }
    for code_label, data in by_code.items():
        data.sort()
        ps = [d[0] for d in data]
        lers = [d[1] for d in data]
        ax.loglog(
            ps, lers, marker=markers[code_label], color=colors[code_label],
            label=labels[code_label], linewidth=1.5, markersize=7,
        )
        print(f"{code_label}:")
        for p, ler, sh, err in data:
            print(f"  p={p:.4f}, LER={ler:.5f}, errors={err}/{sh}")

    ax.set_xlabel("Physical error rate $p$")
    ax.set_ylabel("Logical error rate per cycle")
    ax.set_title(
        f"Webster code 0 Cain §IV.C surgery (τ_s={args.num_rounds}, circuit-level noise)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = Path(__file__).parent / "cain_fig1b_circuit_level.png"
    fig.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
