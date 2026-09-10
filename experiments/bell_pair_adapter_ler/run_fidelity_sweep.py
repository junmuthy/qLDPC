#!/usr/bin/env python3
"""Run a resumable Surface/Steane adapter sweep over Bell-pair fidelity."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from experiments.bell_pair_adapter_ler.analyze_fidelity_sweep import write_fidelity_analysis
from experiments.bell_pair_adapter_ler.run_experiment import (
    DEFAULT_FIDELITY_SWEEP_INFIDELITIES,
    DEFAULT_NOISE_SCOPE,
    DEFAULT_PHYSICAL_ERROR_RATE,
    build_fidelity_sweep_matrix,
    default_output_dir,
    sample_experiments,
    write_artifacts,
    write_summary,
)
from qldpc.objects import Pauli

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "outputs" / "intrinsic_prep3_surg4_bell_fidelity_sweep_pilot"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--bell-pair-infidelities",
        type=float,
        nargs="+",
        default=list(DEFAULT_FIDELITY_SWEEP_INFIDELITIES),
    )
    parser.add_argument("--rounds", type=int, nargs="+", default=[4])
    parser.add_argument("--preparation-rounds", type=int, default=3)
    parser.add_argument("--bases", choices=("X", "Z"), nargs="+", default=["X", "Z"])
    parser.add_argument("--physical-error-rate", type=float, default=DEFAULT_PHYSICAL_ERROR_RATE)
    parser.add_argument("--include-idling-error", action="store_true")
    parser.add_argument(
        "--adapter-noise-targets",
        choices=("full_surgery", "adapter_hardware"),
        default="full_surgery",
    )
    parser.add_argument("--reduce-detectors", action="store_true")
    parser.add_argument(
        "--noise-scope",
        choices=("intrinsic", "operational", "end-to-end"),
        default=DEFAULT_NOISE_SCOPE,
    )
    parser.add_argument("--max-shots", type=int, default=20_000)
    parser.add_argument("--max-errors", type=int, default=400)
    parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "--existing-data",
        type=Path,
        nargs="*",
        default=[],
        help="Sinter CSV files whose matching samples should seed this run",
    )
    parser.add_argument("--relative-overheads", type=float, nargs="+", default=[0.1, 0.25])
    parser.add_argument("--build-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output_dir is None:
        args.output_dir = (
            default_output_dir(args.adapter_noise_targets, args.reduce_detectors) / "fidelity_sweep"
            if args.adapter_noise_targets != "full_surgery" or args.reduce_detectors
            else DEFAULT_OUTPUT_DIR
        )
    missing_existing_data = [path for path in args.existing_data if not path.is_file()]
    if missing_existing_data:
        raise FileNotFoundError(f"existing Sinter data not found: {missing_existing_data}")

    bases = tuple(Pauli[basis] for basis in args.bases)
    rounds = tuple(args.rounds)
    infidelities = tuple(args.bell_pair_infidelities)
    experiments = build_fidelity_sweep_matrix(
        bell_pair_infidelities=infidelities,
        bases=bases,
        rounds=rounds,
        physical_error_rate=args.physical_error_rate,
        include_idling_error=args.include_idling_error,
        preparation_rounds=args.preparation_rounds,
        noise_scope=args.noise_scope,
        adapter_noise_targets=args.adapter_noise_targets,
        reduce_detectors=args.reduce_detectors,
    )
    write_artifacts(experiments, args.output_dir)
    run_config = {
        "bases": args.bases,
        "adapter_noise_targets": args.adapter_noise_targets,
        "reduce_detectors": args.reduce_detectors,
        "bell_pair_infidelities": args.bell_pair_infidelities,
        "bell_pair_fidelities": [1 - value for value in args.bell_pair_infidelities],
        "existing_data_filepaths": [
            os.path.relpath(path, args.output_dir) for path in args.existing_data
        ],
        "include_idling_error": args.include_idling_error,
        "max_errors": args.max_errors,
        "max_shots": args.max_shots,
        "noise_scope": args.noise_scope,
        "num_workers": args.num_workers,
        "physical_error_rate": args.physical_error_rate,
        "preparation_rounds": args.preparation_rounds,
        "relative_ler_overheads": args.relative_overheads,
        "surgery_rounds": args.rounds,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for experiment in experiments:
        metadata = experiment.metadata
        print(
            f"built {experiment.name}: "
            f"F_B={metadata['bell_pair_fidelity']}, "
            f"qubits={metadata['num_qubits']}, detectors={metadata['num_detectors']}, "
            f"dem_errors={metadata['num_dem_errors']}"
        )

    if args.build_only:
        print(f"artifacts written to {args.output_dir}")
        return

    stats = sample_experiments(
        experiments,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        max_shots=args.max_shots,
        max_errors=args.max_errors,
        existing_data_filepaths=tuple(args.existing_data),
    )
    summary_path = args.output_dir / "summary.csv"
    write_summary(stats, args.output_dir)
    write_fidelity_analysis(
        summary_path,
        args.output_dir,
        relative_overheads=args.relative_overheads,
    )
    print(f"results and fidelity analysis written to {args.output_dir}")


if __name__ == "__main__":
    main()
