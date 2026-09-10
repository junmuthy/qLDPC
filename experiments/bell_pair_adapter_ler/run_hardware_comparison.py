#!/usr/bin/env python3
"""Small Surface/Steane comparison and paired detector-reduction benchmark.

The two hardware representations decode identical measurement samples. Their
rows are paired validation data and must never be pooled as independent shots.
The full-surgery noise experiment uses independent samples. This pilot is not
resumable; use run_experiment.py for large resumable collections.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import time
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sinter

from qldpc import decoders
from qldpc.circuits.surgery import AdapterNoiseTargets
from qldpc.objects import Pauli

from .run_experiment import BuiltExperiment, build_experiment, write_artifacts, write_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--preparation-rounds", type=int, default=3)
    parser.add_argument("--physical-error-rate", type=float, default=0.001)
    parser.add_argument("--bell-pair-infidelity", type=float, default=0.0)
    args = parser.parse_args()
    if args.shots < 1:
        parser.error("--shots must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("comparison output directory must be empty; choose a new directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    config.update(
        noise_scope="intrinsic",
        include_idling_error=False,
        versions={name: version(name) for name in ("stim", "sinter", "ldpc")},
        paired_hardware_samples=True,
        detector_error_model="undecomposed_exact",
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    experiments: list[BuiltExperiment] = []
    stats: list[sinter.TaskStats] = []
    timings: list[dict[str, object]] = []
    variants: tuple[tuple[AdapterNoiseTargets, bool], ...] = (
        ("full_surgery", False),
        ("adapter_hardware", False),
        ("adapter_hardware", True),
    )
    for basis_index, basis in enumerate((Pauli.X, Pauli.Z)):
        hardware_records = hardware_predictions = None
        for targets, reduce in variants:
            started = time.perf_counter()
            experiment = build_experiment(
                basis=basis,
                rounds=args.rounds,
                preparation_rounds=args.preparation_rounds,
                physical_error_rate=args.physical_error_rate,
                bell_pair_infidelity=args.bell_pair_infidelity,
                adapter_noise_targets=targets,
                reduce_detectors=reduce,
            )
            build_seconds = time.perf_counter() - started
            # Record the sampling method and seed in the task identity too.
            sample_seed = args.seed + 2 * basis_index + int(targets == "adapter_hardware")
            experiment = dataclasses.replace(
                experiment,
                metadata=dict(
                    experiment.metadata, sampling_method="paired_pilot", seed=sample_seed
                ),
            )
            experiments.append(experiment)
            started = time.perf_counter()
            if reduce:
                assert hardware_records is not None
                records = hardware_records
            else:
                records = experiment.circuit.compile_sampler(seed=sample_seed).sample(args.shots)
            detectors, observables = experiment.circuit.compile_m2d_converter().convert(
                measurements=records, separate_observables=True
            )
            sample_seconds = time.perf_counter() - started
            started = time.perf_counter()
            decoder = decoders.SinterDecoder().compile_decoder_for_dem(
                experiment.detector_error_model
            )
            compile_seconds = time.perf_counter() - started
            started = time.perf_counter()
            predictions = decoder.decode_shots(detectors)
            decode_seconds = time.perf_counter() - started
            errors = int(np.count_nonzero(np.any(predictions != observables, axis=1)))
            disagreements: int | str = ""
            if targets == "adapter_hardware":
                if reduce:
                    assert hardware_predictions is not None
                    disagreements = int(
                        np.count_nonzero(np.any(predictions != hardware_predictions, axis=1))
                    )
                else:
                    hardware_records, hardware_predictions = records, predictions
            task = sinter.Task(
                circuit=experiment.circuit,
                detector_error_model=experiment.detector_error_model,
                decoder="qldpc_bp_osd",
                json_metadata=experiment.metadata,
            )
            stats.append(
                sinter.TaskStats(
                    strong_id=task.strong_id(),
                    decoder="qldpc_bp_osd",
                    json_metadata=experiment.metadata,
                    shots=args.shots,
                    errors=errors,
                    seconds=sample_seconds + decode_seconds,
                )
            )
            timings.append(
                {
                    "logical_basis": basis.name,
                    "adapter_noise_targets": targets,
                    "detector_reduction": "inactive_rows" if reduce else "none",
                    "shots": args.shots,
                    "num_qubits": experiment.circuit.num_qubits,
                    "num_detectors": experiment.circuit.num_detectors,
                    "num_decoder_error_variables": decoder.dem_arrays.num_errors,
                    "build_seconds": build_seconds,
                    "sampling_seconds": sample_seconds,
                    "decoder_compile_seconds": compile_seconds,
                    "decode_seconds": decode_seconds,
                    "paired_prediction_disagreements": disagreements,
                }
            )
            write_summary(stats, args.output_dir)
            print(
                f"{basis}{basis} {targets}, reduced={reduce}: {errors}/{args.shots} errors, "
                f"{experiment.circuit.num_detectors} detectors, decoding {decode_seconds:.2f}s, "
                f"paired disagreements={disagreements}",
                flush=True,
            )

    write_artifacts(experiments, args.output_dir)
    with (args.output_dir / "timing.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(timings[0]))
        writer.writeheader()
        writer.writerows(timings)
    plot_comparison(args.output_dir)


def plot_comparison(output_dir: Path) -> None:
    with (output_dir / "summary.csv").open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    figure, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    for axis, basis in zip(axes, ("X", "Z")):
        selected = [
            row
            for row in rows
            if row["logical_basis"] == basis
            and (
                row["adapter_noise_targets"] == "full_surgery"
                or row["detector_reduction"] == "inactive_rows"
            )
        ]
        selected.sort(key=lambda row: row["adapter_noise_targets"] != "full_surgery")
        y = np.array([float(row["logical_error_rate"]) for row in selected]) * 100
        low = np.array([float(row["ler_95ci_low"]) for row in selected]) * 100
        high = np.array([float(row["ler_95ci_high"]) for row in selected]) * 100
        axis.errorbar([0, 1], y, yerr=np.maximum([y - low, high - y], 0), fmt="o", capsize=5)
        axis.set_xticks([0, 1], ["Full surgery", "Adapter hardware"])
        axis.set_xlim(-0.4, 1.4)
        axis.set_title(f"Logical {basis}{basis}")
        axis.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Adapter LER (%) — 95% Wilson intervals")
    axes[0].set_ylim(bottom=0)
    first = rows[0]
    figure.suptitle(
        f"Surface–Steane intrinsic pilot: {first['surgery_rounds']} adapter rounds, "
        f"p={first['physical_error_rate']}, Bell F={first['bell_pair_fidelity']}\n"
        f"{first['shots']} shots per condition"
    )
    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(output_dir / f"adapter_hardware_comparison.{extension}", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
