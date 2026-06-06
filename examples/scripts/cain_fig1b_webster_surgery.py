"""LER vs noise rate sweep for Webster code 0 joint X̄_1 X̄_2 surgery.

Produces a Cain et al. (arXiv:2503.10390) Fig 1b-style plot: log-log LER
curves for the data code memory experiment vs the joint surgery code
(data + 2 boosted gadgets + bridge), decoded with BP+OSD.

Usage:
    python examples/scripts/cain_fig1b_webster_surgery.py [--quick]

Output:
    examples/scripts/cain_fig1b_webster_surgery.png
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


def _support_to_vec(seed: dict, l: int) -> np.ndarray:
    v = np.zeros(2 * l, dtype=int)
    for i in seed["L_support"]:
        v[i] = 1
    for i in seed["R_support"]:
        v[l + i] = 1
    return v


def build_webster_code_0() -> tuple[codes.CSSCode, codes.CSSCode]:
    """Return (data_code, joint_surgery_code) for Webster code 0."""
    data = load_webster_seed_set(0)
    data_code = _build_generalised_bicycle_code(
        l=data["l"], A_set=data["A"], B_set=data["B"]
    )
    x_seeds = [s for s in data["seeds"] if s["pauli_type"] == "X"]
    op1 = _support_to_vec(x_seeds[0], data["l"])
    op2 = _support_to_vec(x_seeds[1], data["l"])

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
    return data_code, joint


def _keep_only_observables(circuit: stim.Circuit, keep_indices: set[int]) -> stim.Circuit:
    """Strip all OBSERVABLE_INCLUDE instructions except those in keep_indices,
    renumbered consecutively starting from 0.
    """
    new = stim.Circuit()
    index_map: dict[int, int] = {}
    for inst in circuit.flattened():
        if inst.name == "OBSERVABLE_INCLUDE":
            old_idx = int(inst.gate_args_copy()[0])
            if old_idx not in keep_indices:
                continue
            if old_idx not in index_map:
                index_map[old_idx] = len(index_map)
            new.append("OBSERVABLE_INCLUDE", inst.targets_copy(), [index_map[old_idx]])
        else:
            new.append(inst.name, inst.targets_copy(), inst.gate_args_copy())
    return new


def run_sweep(
    data_code: codes.CSSCode,
    joint_code: codes.CSSCode,
    joint_observable_idx: int,
    data_observable_idx: int,
    error_rates: list[float],
    num_rounds: int,
    max_shots: int,
    max_errors: int,
    num_workers: int,
) -> list[sinter.TaskStats]:
    """Run sinter sweep, tracking ONE observable per code.

    Per Cain et al. §IV.C surgery experiment: for joint code we track the
    joint X̄_1 X̄_2 target observable (weight-1 representative X̄_{joint_observable_idx}
    = X_{b_0}). For data code baseline we track ONE data X-logical
    (X̄_{data_observable_idx}). This isolates the surgery readout error from
    other logical-preservation errors, giving a clean Cain Fig 1b-style
    LER vs p curve.
    """
    tasks = []
    for code, label, keep_idx in [
        (data_code, "data X̄_0 memory", data_observable_idx),
        (joint_code, "joint X̄_1 X̄_2 readout (= X_{b_0})", joint_observable_idx),
    ]:
        for p in error_rates:
            noise = circuits.DepolarizingNoiseModel(p, include_idling_error=False)
            circuit = circuits.get_memory_experiment(
                code, basis=Pauli.X, num_rounds=num_rounds, noise_model=noise,
            )
            circuit = _keep_only_observables(circuit, {keep_idx})
            tasks.append(
                sinter.Task(
                    circuit=circuit,
                    json_metadata={"code": label, "p": p, "n": code.num_qubits, "k": code.dimension},
                )
            )
    decoder = decoders.SinterDecoder(with_BP_OSD=True)
    return sinter.collect(
        tasks=tasks,
        decoders=["bp_osd"],
        custom_decoders={"bp_osd": decoder},
        num_workers=num_workers,
        max_shots=max_shots,
        max_errors=max_errors,
        print_progress=True,
    )


def plot_results(stats: list[sinter.TaskStats], output_path: Path, num_rounds: int) -> None:
    by_code: dict[str, list[tuple[float, float, int, int]]] = {}
    for s in stats:
        meta = s.json_metadata
        label = meta["code"]
        p = float(meta["p"])
        if s.shots == 0:
            continue
        ler = s.errors / s.shots
        by_code.setdefault(label, []).append((p, ler, s.shots, s.errors))

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"data X̄_0 memory": "tab:blue", "joint X̄_1 X̄_2 readout (= X_{b_0})": "tab:red"}
    markers = {"data X̄_0 memory": "o", "joint X̄_1 X̄_2 readout (= X_{b_0})": "s"}
    labels = colors  # for label lookup
    for code_label, data in by_code.items():
        if not data:
            continue
        data.sort()
        ps = [d[0] for d in data]
        lers = [d[1] for d in data]
        shots = [d[2] for d in data]
        errors = [d[3] for d in data]
        ax.loglog(
            ps, lers, marker=markers[code_label], color=colors[code_label],
            label=code_label, linewidth=1.5, markersize=7,
        )
        for p, ler, sh, err in data:
            print(f"  {code_label}: p={p:.4f}, LER={ler:.5f}, errors={err}/{sh}")

    ax.set_xlabel("Physical error rate $p$ (depolarizing per gate)")
    ax.set_ylabel("Logical error rate per QEC cycle")
    ax.set_title(
        f"Webster code 0 surgery LER (n_rounds={num_rounds}, BP+OSD decoder)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", action="store_true",
        help="Smaller sweep (3 points, 5000 shots) for quick smoke test."
    )
    parser.add_argument(
        "--num-rounds", type=int, default=3,
        help="QEC rounds per shot. Default 3."
    )
    args = parser.parse_args()

    print("Building Webster code 0 + joint surgery code via end-to-end pipeline...")
    t0 = time.time()
    data_code, joint = build_webster_code_0()
    print(f"  data code: [[{data_code.num_qubits}, {data_code.dimension}]]")
    print(f"  joint code: [[{joint.num_qubits}, {joint.dimension}]]")
    print(f"  build time: {time.time() - t0:.1f}s\n")

    if args.quick:
        error_rates = [1e-3, 3e-3, 1e-2]
        max_shots = 5_000
        max_errors = 50
    else:
        error_rates = list(np.logspace(-3.5, -1.5, 7))
        max_shots = 50_000
        max_errors = 100

    # Per Cain §IV.C: track the target operator. For joint code, the
    # weight-1 representative X_{b_0} is the joint X̄_1 X̄_2 observable
    # (qldpc returns this as X-logical index 8 — verified empirically).
    joint_obs_idx = joint.dimension - 1  # last X-logical = joint observable
    data_obs_idx = 0  # any data X-logical; pick index 0 for fair comparison
    print(f"Tracking joint observable index {joint_obs_idx} (= X_{{b_0}}, weight 1)")
    print(f"Tracking data observable index {data_obs_idx} (one X̄ of data code)")

    num_workers = max(1, (os.cpu_count() or 4) - 1)
    print(
        f"Running sinter sweep: {len(error_rates)} noise rates × 2 codes × "
        f"≤{max_shots} shots, {num_workers} workers..."
    )
    t0 = time.time()
    stats = run_sweep(
        data_code, joint,
        joint_observable_idx=joint_obs_idx,
        data_observable_idx=data_obs_idx,
        error_rates=error_rates,
        num_rounds=args.num_rounds,
        max_shots=max_shots,
        max_errors=max_errors,
        num_workers=num_workers,
    )
    print(f"\nSweep complete in {time.time() - t0:.1f}s. Plotting...")

    output_path = Path(__file__).parent / "cain_fig1b_webster_surgery.png"
    plot_results(stats, output_path, num_rounds=args.num_rounds)


if __name__ == "__main__":
    main()
