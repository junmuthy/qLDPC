"""Surgery PPM vs Idling LER sweep on Webster code 0 [[62, 10, 6]].

Cluster-friendly:
- Headless matplotlib (Agg backend)
- sinter.collect with save_resume_filepath → fault-tolerant to restart
- All knobs via argparse, sensible defaults
- Streams progress to stdout (line-buffered)

Outputs:
- {out_dir}/results.csv             — sinter resume file
- {out_dir}/summary.txt             — human-readable LER table
- {out_dir}/ler_curve.png           — log-log plot

Usage:
    python run_webster0_ler_sweep.py --out-dir results/run1 --workers 32 \
        --max-shots 200000 --max-errors 200

Resume after interruption:
    Re-run the SAME command. sinter.collect reads results.csv and skips
    already-finished work.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import sinter
import stim

import sys

from qldpc import circuits, decoders
from qldpc.codes.surgery import build_gadget, build_single_ppm_circuit, keep_only_observable
from qldpc.circuits.noise_model import DepolarizingNoiseModel
from qldpc.objects import Pauli

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _webster_seed_set import (  # noqa: E402
    build_generalised_bicycle_code,
    load_webster_seed_set,
)


def _x_bar_1_operator(d: dict) -> np.ndarray:
    l = d["l"]
    for seed in d["seeds"]:
        if seed["name"] == "X_bar_1" and seed["pauli_type"] == "X":
            L = np.zeros(l, dtype=np.uint8)
            R = np.zeros(l, dtype=np.uint8)
            for i in seed["L_support"]:
                L[i] = 1
            for i in seed["R_support"]:
                R[i] = 1
            return np.concatenate([L, R])
    raise ValueError("X_bar_1 not found")


def build_tasks(rounds: int, p_values: list[float]) -> list[sinter.Task]:
    d0 = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(d0["l"], d0["A"], d0["B"])
    x = _x_bar_1_operator(d0)
    g = build_gadget(code, x)

    tasks = []
    for p in p_values:
        noise = DepolarizingNoiseModel(p, include_idling_error=False)
        surg = build_single_ppm_circuit(g, rounds=rounds, noise_model=noise)
        surg_obs0 = keep_only_observable(surg, keep_idx=0)
        tasks.append(sinter.Task(
            circuit=surg_obs0,
            json_metadata={"p": float(p), "kind": "surgery", "rounds": rounds},
        ))
        mem = circuits.get_memory_experiment(
            code, basis=Pauli.X, num_rounds=rounds, noise_model=noise,
        )
        tasks.append(sinter.Task(
            circuit=mem,
            json_metadata={"p": float(p), "kind": "idling", "rounds": rounds},
        ))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-shots", type=int, default=200_000)
    ap.add_argument("--max-errors", type=int, default=200)
    ap.add_argument(
        "--p-values", nargs="+", type=float,
        default=[0.001, 0.002, 0.004, 0.008, 0.015, 0.025],
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"

    print(f"[config] out_dir   = {out_dir.resolve()}")
    print(f"[config] workers   = {args.workers}")
    print(f"[config] rounds    = {args.rounds}")
    print(f"[config] max_shots = {args.max_shots}")
    print(f"[config] max_errs  = {args.max_errors}")
    print(f"[config] p_values  = {args.p_values}")
    print(f"[config] csv_path  = {csv_path}")

    tasks = build_tasks(args.rounds, args.p_values)
    print(f"[setup] built {len(tasks)} tasks")

    decoder = decoders.SinterDecoder()
    sinter.collect(
        tasks=tasks,
        decoders=["custom"],
        custom_decoders={"custom": decoder},
        num_workers=args.workers,
        max_shots=args.max_shots,
        max_errors=args.max_errors,
        print_progress=True,
        save_resume_filepath=str(csv_path),
    )

    # Load results back from CSV (works even after resume)
    results = sinter.read_stats_from_csv_files(csv_path)
    print(f"[results] loaded {len(results)} task stats from {csv_path}")

    surgery_lers = {}
    idling_lers = {}
    for r in results:
        p = r.json_metadata["p"]
        kind = r.json_metadata["kind"]
        ler = r.errors / max(r.shots, 1)
        target = surgery_lers if kind == "surgery" else idling_lers
        target[p] = (ler, r.errors, r.shots)

    # Summary table
    lines = []
    header = f"{'p':>10} | {'surgery PPM LER':>26} | {'idling LER':>26}"
    lines.append(header)
    lines.append("-" * len(header))
    for p in sorted(args.p_values):
        s = surgery_lers.get(p, (np.nan, 0, 0))
        m = idling_lers.get(p, (np.nan, 0, 0))
        lines.append(
            f"{p:>10.5f} | {s[0]:>10.5g} ({s[1]:>5}/{s[2]:>8})"
            f" | {m[0]:>10.5g} ({m[1]:>5}/{m[2]:>8})"
        )
    summary_text = "\n".join(lines)
    print("\n" + summary_text)
    (out_dir / "summary.txt").write_text(summary_text + "\n")
    print(f"\n[results] summary → {out_dir / 'summary.txt'}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ps_sorted = sorted(args.p_values)
    if all(p in surgery_lers for p in ps_sorted):
        ax.loglog(
            ps_sorted, [surgery_lers[p][0] for p in ps_sorted],
            "o--", label="Surgery PPM (obs0 = Webster Eq.1)",
            markersize=8, linewidth=2,
        )
    if all(p in idling_lers for p in ps_sorted):
        ax.loglog(
            ps_sorted, [idling_lers[p][0] for p in ps_sorted],
            "s-", label=f"Idling (memory, {args.rounds} rounds)",
            markersize=8, linewidth=2,
        )
    ax.set_xlabel("physical error rate $p$")
    ax.set_ylabel("logical error rate")
    ax.set_title(
        f"Webster code 0 [[62, 10, 6]] — "
        f"Surgery PPM vs Idling ({args.rounds} rounds, BP+OSD)"
    )
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plot_path = out_dir / "ler_curve.png"
    plt.savefig(plot_path, dpi=120)
    print(f"[results] plot    → {plot_path}")


if __name__ == "__main__":
    main()
