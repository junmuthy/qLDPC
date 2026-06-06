"""Full Cain et al. (arXiv:2503.10390) §IV.C surgery experiment for Webster code 0.

X-basis experiment:
  - Data qubits Q: init |+⟩ (RX), measure X (MX)
  - Ancilla qubits Q' (κ + bridge): init |0⟩ (RZ), measure Z (MZ)
  - Run τ_s rounds of merged-code stabilizer measurement (via MPP)
  - k + t logical observables:
      * k observables = final M_X parities on supp(X̄_i) for each data X-logical
      * t observables = "target operators" = first-cycle parities of merged-code
        X-checks (chi rows + U_B path stabs). For our t=1 surgery, this is
        XOR of all first-round chi^(1) + chi^(2) + U_B X-stab outcomes,
        which by Cross §3.6 formula equals the X̄_1 X̄_2 eigenvalue.

Detectors:
  - First round X-stabs that touch ONLY data qubits: +1 deterministic given
    |+⟩^n_data init → 1-rec detector.
  - All other first-round X-stabs (chi, U_B): outcomes are random (ancilla |0⟩
    is not X-eigenstate) → NO first-round detector.
  - Round 2..R X-stabs: 2-rec detector comparing to previous round.
  - First round Z-stabs: random outcomes (|+⟩^n_data is not Z-eigenstate),
    no detector. Round 2+ Z-stabs: 2-rec comparison.

Noise model: per-gate depolarizing applied via DepolarizingNoiseModel(p).
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
    _build_bridge_via_skiptree,
    _build_generalised_bicycle_code,
    _stitch_gadgets_with_bridge,
    boost_gadget_cheeger_combinatorial,
    build_layered_surgery_code,
    load_webster_seed_set,
)
from qldpc.objects import Pauli


def _seed_to_vec(seed: dict, l: int) -> np.ndarray:
    v = np.zeros(2 * l, dtype=np.int_)
    for i in seed["L_support"]:
        v[i] = 1
    for i in seed["R_support"]:
        v[l + i] = 1
    return v


def build_webster_setup() -> tuple[codes.CSSCode, codes.CSSCode, np.ndarray, np.ndarray, int]:
    """Build Webster code 0 + joint surgery code. Returns
    (data_code, joint_code, op1, op2, n_data).
    """
    data = load_webster_seed_set(0)
    data_code = _build_generalised_bicycle_code(
        l=data["l"], A_set=data["A"], B_set=data["B"]
    )
    x_seeds = [s for s in data["seeds"] if s["pauli_type"] == "X"]
    op1 = _seed_to_vec(x_seeds[0], data["l"])
    op2 = _seed_to_vec(x_seeds[1], data["l"])
    m1, l1 = build_layered_surgery_code(
        data_code, op1, num_layers=1, validate_logical_op=False
    )
    m2, l2 = build_layered_surgery_code(
        data_code, op2, num_layers=1, validate_logical_op=False
    )
    m1b, l1b, _ = boost_gadget_cheeger_combinatorial(
        m1, l1, target_h=1.0, max_extra_qubits=10, seed=42
    )
    m2b, l2b, _ = boost_gadget_cheeger_combinatorial(
        m2, l2, target_h=1.0, max_extra_qubits=10, seed=42
    )
    bridge = _build_bridge_via_skiptree(l1b, l2b)
    joint, _ = _stitch_gadgets_with_bridge(
        data_code, m1b, l1b, m2b, l2b, bridge, pauli_type=Pauli.X
    )
    return data_code, joint, op1, op2, data_code.num_qubits


def _build_mpp_targets(stab_row: np.ndarray, basis: str) -> list[stim.GateTarget]:
    """Build Stim MPP targets for a stabilizer row in given basis."""
    targets = []
    nonzero = np.flatnonzero(stab_row)
    for k, q in enumerate(nonzero):
        if basis == "X":
            targets.append(stim.target_x(int(q)))
        else:
            targets.append(stim.target_z(int(q)))
        if k < len(nonzero) - 1:
            targets.append(stim.target_combiner())
    return targets


def build_cain_x_basis_circuit(
    joint_code: codes.CSSCode,
    data_code: codes.CSSCode,
    n_data: int,
    num_rounds: int,
    noise_p: float,
) -> stim.Circuit:
    """Build full Cain X-basis surgery experiment circuit.

    Layout:
      - Qubits 0..n_data-1 = data (RX init, MX final)
      - Qubits n_data..n_total-1 = ancilla κ + bridge (RZ init, MZ final)

    Stabilizer measurements via MPP gates (noiseless extraction). Noise
    is injected via `DEPOLARIZE1(p)` on every qubit between rounds (idle
    noise), plus `Z_ERROR(p)` after RX/RZ and before MX/MZ (init/measure
    flip noise).
    """
    n_total = joint_code.num_qubits
    n_anc = n_total - n_data

    HX = np.asarray(joint_code.matrix_x).astype(np.int_)
    HZ = np.asarray(joint_code.matrix_z).astype(np.int_)

    # Identify which X-stabs are "pure data" (no ancilla support).
    # These have deterministic +1 outcome under |+⟩^n_data init.
    x_pure_data_mask = ~np.any(HX[:, n_data:] != 0, axis=1)

    # Target operator extraction: per Cross §3.6 formula α* = (0 on
    # data X-stabs, 1 on chi rows + U_B). For our joint code, the chi
    # rows are HX rows with X support on ancilla but NOT on bridge-only.
    # Simplest: target = XOR of ALL non-pure-data X-stabs = XOR of HX
    # rows that touch ancilla or bridge.
    # By Cross §3.6 formula, this XOR equals X̄_1 X̄_2 eigenvalue of data.
    x_target_mask = ~x_pure_data_mask

    data_qubits = list(range(n_data))
    anc_qubits = list(range(n_data, n_total))

    circ = stim.Circuit()

    # Initialization: data in |+⟩, ancilla in |0⟩.
    circ.append("RX", data_qubits)
    if anc_qubits:
        circ.append("RZ", anc_qubits)
    # Init noise (Z error on |+⟩ = bit flip in X basis; X error on |0⟩ = bit flip in Z basis).
    if noise_p > 0:
        circ.append("Z_ERROR", data_qubits, noise_p)
        if anc_qubits:
            circ.append("X_ERROR", anc_qubits, noise_p)

    # Track measurement record indices for stabs across rounds.
    x_stab_records: list[list[int]] = []  # [round][stab_idx]
    z_stab_records: list[list[int]] = []

    for round_idx in range(num_rounds):
        # Idle noise before stab measurements (one round of decoherence).
        if noise_p > 0:
            circ.append("DEPOLARIZE1", list(range(n_total)), noise_p)

        # Measure X-stabs.
        cur_x = []
        for stab_idx, stab in enumerate(HX):
            targets = _build_mpp_targets(stab, "X")
            if not targets:
                continue
            circ.append("MPP", targets)
            cur_x.append(circ.num_measurements - 1)
        x_stab_records.append(cur_x)

        # Measure Z-stabs.
        cur_z = []
        for stab in HZ:
            targets = _build_mpp_targets(stab, "Z")
            if not targets:
                continue
            circ.append("MPP", targets)
            cur_z.append(circ.num_measurements - 1)
        z_stab_records.append(cur_z)

        # Detectors.
        if round_idx == 0:
            # First round: only pure-data X-stabs have deterministic
            # +1 outcome under |+⟩^n_data init.
            for stab_idx, rec_idx in enumerate(cur_x):
                if x_pure_data_mask[stab_idx]:
                    offset = rec_idx - circ.num_measurements
                    circ.append(
                        "DETECTOR",
                        [stim.target_rec(offset)],
                    )
        else:
            # Subsequent rounds: compare each X-stab and Z-stab to prev.
            prev_x = x_stab_records[round_idx - 1]
            for cur_rec, prev_rec in zip(cur_x, prev_x):
                off_cur = cur_rec - circ.num_measurements
                off_prev = prev_rec - circ.num_measurements
                circ.append(
                    "DETECTOR",
                    [stim.target_rec(off_cur), stim.target_rec(off_prev)],
                )
            prev_z = z_stab_records[round_idx - 1]
            for cur_rec, prev_rec in zip(cur_z, prev_z):
                off_cur = cur_rec - circ.num_measurements
                off_prev = prev_rec - circ.num_measurements
                circ.append(
                    "DETECTOR",
                    [stim.target_rec(off_cur), stim.target_rec(off_prev)],
                )

    # Final measurements: data in X, ancilla in Z.
    # Measurement noise: flip with prob p.
    if noise_p > 0:
        circ.append("Z_ERROR", data_qubits, noise_p)
        if anc_qubits:
            circ.append("X_ERROR", anc_qubits, noise_p)
    data_mx_record_base = circ.num_measurements
    circ.append("MX", data_qubits)
    if anc_qubits:
        circ.append("MZ", anc_qubits)

    # Final-round detectors using M_X on data:
    # Final M_X on a data qubit = qubit's X eigenvalue at end. The XOR
    # over supp(X-stab) on data should equal the round-R X-stab outcome
    # (for X-stabs that only touch data, since ancilla part has no
    # support).  Skip for chi/path stabs that touch ancilla.
    final_round_x = x_stab_records[-1]
    for stab_idx, rec_idx in enumerate(final_round_x):
        if not x_pure_data_mask[stab_idx]:
            continue
        # XOR of final M_X records on this stab's support equals
        # round-R measurement of this X-stab.
        data_support = np.flatnonzero(HX[stab_idx, :n_data])
        targets = [
            stim.target_rec(data_mx_record_base + int(q) - circ.num_measurements)
            for q in data_support
        ]
        targets.append(
            stim.target_rec(rec_idx - circ.num_measurements)
        )
        circ.append("DETECTOR", targets)

    # Target observable (t=1): XOR of first-round X-stabs that touch ancilla.
    # Per Cross §3.6 formula α*, this XOR = X̄_1 X̄_2 eigenvalue.
    first_round_x = x_stab_records[0]
    target_records = []
    for stab_idx, rec_idx in enumerate(first_round_x):
        if x_target_mask[stab_idx]:
            target_records.append(
                stim.target_rec(rec_idx - circ.num_measurements)
            )
    circ.append("OBSERVABLE_INCLUDE", target_records, [0])

    # k data X̄ observables: XOR of final M_X on supp(X̄_i) for each data X-logical.
    data_x_logicals = np.asarray(data_code.get_logical_ops(Pauli.X)).astype(np.int_)
    for i, lop in enumerate(data_x_logicals):
        targets = [
            stim.target_rec(data_mx_record_base + int(q) - circ.num_measurements)
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
    data_code, joint, op1, op2, n_data = build_webster_setup()
    print(f"  data: [[{data_code.num_qubits}, {data_code.dimension}]]")
    print(f"  joint: [[{joint.num_qubits}, {joint.dimension}]]\n")

    # Verify circuit at p=0 is deterministic (sanity check).
    print("Sanity: verifying noiseless circuit is detector-deterministic...")
    test_circ = build_cain_x_basis_circuit(
        joint, data_code, n_data, num_rounds=args.num_rounds, noise_p=0.0,
    )
    try:
        test_circ.detector_error_model(decompose_errors=False)
        print("  ✓ Noiseless circuit accepted by Stim DEM\n")
    except Exception as e:
        print(f"  ✗ Noiseless circuit FAILED DEM: {e}\n")
        return

    if args.quick:
        error_rates = [1e-3, 3e-3, 1e-2]
        max_shots = 5_000
        max_errors = 50
    else:
        error_rates = list(np.logspace(-3.5, -1.5, 7))
        max_shots = 50_000
        max_errors = 100

    # Run Cain X-basis experiment for joint code at each p.
    # For data baseline, use qldpc's standard X-memory experiment with 1 observable.
    tasks = []
    for p in error_rates:
        circ = build_cain_x_basis_circuit(
            joint, data_code, n_data, num_rounds=args.num_rounds, noise_p=p,
        )
        tasks.append(sinter.Task(circuit=circ, json_metadata={"code": "joint", "p": p}))

        noise = circuits.DepolarizingNoiseModel(p, include_idling_error=False)
        data_circ = circuits.get_memory_experiment(
            data_code, basis=Pauli.X, num_rounds=args.num_rounds, noise_model=noise,
        )
        # Strip to 1 observable for data baseline.
        new_circ = stim.Circuit()
        kept = False
        for inst in data_circ.flattened():
            if inst.name == "OBSERVABLE_INCLUDE":
                if kept:
                    continue
                new_circ.append("OBSERVABLE_INCLUDE", inst.targets_copy(), [0])
                kept = True
            else:
                new_circ.append(inst.name, inst.targets_copy(), inst.gate_args_copy())
        tasks.append(sinter.Task(circuit=new_circ, json_metadata={"code": "data", "p": p}))

    num_workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"Running sinter with {num_workers} workers, {len(tasks)} tasks...")
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
        ler = s.errors / s.shots
        by_code.setdefault(label, []).append((p, ler, s.shots, s.errors))

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"data": "tab:blue", "joint": "tab:red"}
    markers = {"data": "o", "joint": "s"}
    labels = {
        "data": "data [[62, 10, 6]] X memory (1 observable)",
        "joint": "joint surgery [[86, 9]] Cain X-basis (k+t obs)",
    }
    for code_label, data in by_code.items():
        data.sort()
        ps = [d[0] for d in data]
        lers = [d[1] for d in data]
        ax.loglog(
            ps, lers,
            marker=markers[code_label],
            color=colors[code_label],
            label=labels[code_label],
            linewidth=1.5,
            markersize=7,
        )
        print(f"{code_label}:")
        for p, ler, sh, err in data:
            print(f"  p={p:.4f}, LER={ler:.5f}, errors={err}/{sh}")

    ax.set_xlabel("Physical error rate $p$")
    ax.set_ylabel("Logical error rate per cycle")
    ax.set_title(
        f"Webster code 0 Cain §IV.C X-basis surgery experiment "
        f"(τ_s={args.num_rounds}, MPP+depol noise)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = Path(__file__).parent / "cain_fig1b_full_protocol.png"
    fig.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
