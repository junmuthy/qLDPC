#!/usr/bin/env python3
"""Resumable hardware-only counterpart to the archived m=3 HGP/surface experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import stim

from qldpc import codes
from qldpc.objects import Pauli

from ..common.paths import OUTPUT_ROOT
from ..common.run_experiment import (
    BuiltExperiment,
    _wilson_interval,
    build_experiment_for_codes,
    sample_experiments,
    write_artifacts,
    write_summary,
)

OUTPUTS = OUTPUT_ROOT
BASELINE = OUTPUTS / "hgp_simplex3_surface_unrotated_d4_intrinsic_prep4_surg4_bell0_10k"
DEFAULT_OUTPUT = OUTPUTS / "hgp_simplex3_surface_unrotated_d4_hardware_prep4_surg4_bell0_10k"


def build_hgp_experiments() -> tuple[list[BuiltExperiment], list[dict[str, Any]]]:
    """Audit the archived circuit/model and construct the two hardware variants."""
    experiments, audit = [], []
    for basis in (Pauli.X, Pauli.Z):
        kwargs: dict[str, Any] = dict(
            left_code=codes.HGPCode(codes.SimplexCode(3)),
            right_code=codes.SurfaceCode(4, rotated=False),
            left_code_name="hgp_simplex3_98_18_4",
            right_code_name="surface_unrotated_d4",
            left_logical_index=0,
            right_logical_index=0,
            basis=basis,
            preparation_rounds=4,
            surgery_rounds=4,
            physical_error_rate=0.001,
            bell_pair_infidelity=0.0,
            include_idling_error=False,
            noise_scope="intrinsic",
            compute_exact_distances=False,
        )
        full = build_experiment_for_codes(**kwargs)
        archived = stim.Circuit.from_file(BASELINE / "circuits" / f"{full.name}.stim")
        # Stim's canonical text has rounded numerical arguments in saved artifacts.
        circuits_match = str(full.circuit) == str(archived)
        try:
            legacy_dem = archived.detector_error_model(
                decompose_errors=True, approximate_disjoint_errors=True
            )
            legacy_analysis = "decomposed"
        except ValueError:
            legacy_dem = archived.detector_error_model(approximate_disjoint_errors=True)
            legacy_analysis = "undecomposed"
        models_match = legacy_dem == full.detector_error_model
        if not circuits_match or not models_match:
            raise ValueError(f"Archived {basis}{basis} circuit or decoder model is not matched")
        hardware = build_experiment_for_codes(
            **kwargs, adapter_noise_targets="adapter_hardware", reduce_detectors=True
        )
        experiments.append(hardware)
        audit.append(
            {
                "logical_basis": basis.name,
                "baseline_name": full.name,
                "canonical_circuit_matches_archive": circuits_match,
                "legacy_dem_analysis": legacy_analysis,
                "legacy_dem_matches_explicit_undecomposed_model": models_match,
                "num_qubits": hardware.circuit.num_qubits,
                "num_adapter_hardware_qubits": hardware.metadata["num_adapter_hardware_qubits"],
                "full_num_detectors": full.circuit.num_detectors,
                "hardware_num_detectors": hardware.circuit.num_detectors,
                "full_num_dem_errors": full.detector_error_model.num_errors,
                "hardware_num_dem_errors": hardware.detector_error_model.num_errors,
            }
        )
    return experiments, audit


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def write_comparison(output_dir: Path) -> list[dict[str, Any]]:
    """Compare independent production estimates; historical times are not speed benchmarks."""
    rows: list[dict[str, Any]] = []
    row: dict[str, Any]
    for policy, directory in (("full_surgery", BASELINE), ("adapter_hardware", output_dir)):
        for row in read_csv(directory / "summary.csv"):
            if not (
                row["name"].startswith("hgp_simplex3_98_18_4_surface_unrotated_d4_joint_")
                and row["noise_scope"] == "intrinsic"
                and row["decoder"] == "qldpc_bp_osd"
                and row.get("adapter_noise_targets", "full_surgery") == policy
            ):
                raise ValueError("Mismatched HGP comparison identity or noise policy")
            shots, discards, errors = (int(row[key]) for key in ("shots", "discards", "errors"))
            accepted = shots - discards
            lower, upper = _wilson_interval(errors, accepted)
            rows.append(
                {
                    "logical_basis": row["logical_basis"],
                    "policy": policy,
                    "preparation_rounds": int(row["preparation_rounds"]),
                    "surgery_rounds": int(row["surgery_rounds"]),
                    "physical_error_rate": float(row["physical_error_rate"]),
                    "bell_pair_fidelity": float(row["bell_pair_fidelity"]),
                    "shots": shots,
                    "discards": discards,
                    "errors": errors,
                    "logical_error_rate": errors / accepted if accepted else 0.0,
                    "ler_95ci_low": lower,
                    "ler_95ci_high": upper,
                    "worker_seconds": float(row["seconds"]),
                    "worker_seconds_per_10000_shots": 10000 * float(row["seconds"]) / shots
                    if shots
                    else 0.0,
                    "source_summary": str(directory / "summary.csv"),
                }
            )
    observed = {(r["logical_basis"], r["policy"]) for r in rows}
    expected = {(b, p) for b in ("X", "Z") for p in ("full_surgery", "adapter_hardware")}
    if observed != expected or len(rows) != 4:
        raise ValueError("Need one completed result for each basis and noise policy")
    for row in rows:
        if not (
            row["preparation_rounds"] == row["surgery_rounds"] == 4
            and row["physical_error_rate"] == 0.001
            and row["bell_pair_fidelity"] == 1.0
            and row["shots"] > 0
        ):
            raise ValueError("Mismatched HGP comparison settings")
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
    for axis, basis in zip(axes, ("X", "Z")):
        selected = [r for r in rows if r["logical_basis"] == basis]
        y, low, high = [
            np.array([r[key] for r in selected]) * 100
            for key in ("logical_error_rate", "ler_95ci_low", "ler_95ci_high")
        ]
        axis.errorbar([0, 1], y, yerr=np.maximum([y - low, high - y], 0), fmt="o", capsize=5)
        axis.set_xticks([0, 1], ["Full surgery\n(archived)", "Adapter hardware\n(new)"])
        axis.set_xlim(-0.4, 1.4)
        axis.set_title(f"Logical {basis}{basis}")
        axis.grid(axis="y", alpha=0.25)
        for index, row in enumerate(selected):
            axis.annotate(
                f"{row['errors']}/{row['shots']} failures",
                (index, high[index]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
    axes[0].set_ylabel("Adapter LER (%) — 95% Wilson intervals")
    axes[0].set_ylim(0, 130 * max(row["ler_95ci_high"] for row in rows))
    figure.suptitle(
        "HGP simplex m=3 – unrotated surface d=4\n"
        "4 ideal preparation + 4 adapter rounds; p=0.001, Bell F=1; matched DEM analysis"
    )
    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(output_dir / f"hgp_adapter_hardware_comparison.{extension}", dpi=180)
    plt.close(figure)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-shots", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.max_shots < 1 or args.num_workers < 1:
        parser.error("Shots and workers must be positive")
    if args.analyze_only:
        write_comparison(args.output_dir)
        return
    started = time.perf_counter()
    experiments, audit = build_hgp_experiments()
    write_artifacts(experiments, args.output_dir)
    config = {
        "max_shots": args.max_shots,
        "max_errors": args.max_shots,
        "num_workers": args.num_workers,
        "baseline_directory": str(BASELINE),
        "preparation_rounds": 4,
        "surgery_rounds": 4,
        "physical_error_rate": 0.001,
        "bell_pair_infidelity": 0.0,
        "include_idling_error": False,
        "noise_scope": "intrinsic",
        "adapter_noise_targets": "adapter_hardware",
        "detector_reduction": "inactive_rows",
        "dem_analysis": "undecomposed_exact",
        "compute_exact_distances": False,
        "versions": {name: version(name) for name in ("stim", "sinter", "ldpc", "numpy")},
        "threads": {
            name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
    }
    for name, content in (("run_config.json", config), ("baseline_audit.json", audit)):
        (args.output_dir / name).write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
    for row in audit:
        print(
            f"{row['logical_basis']}: detectors {row['full_num_detectors']} -> "
            f"{row['hardware_num_detectors']}; DEM errors {row['full_num_dem_errors']} -> "
            f"{row['hardware_num_dem_errors']}; archive/model matched",
            flush=True,
        )
    if args.build_only:
        return
    stats = sample_experiments(
        experiments,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        max_shots=args.max_shots,
        max_errors=args.max_shots,
    )
    write_summary(stats, args.output_dir)
    write_comparison(args.output_dir)
    with (args.output_dir / "invocations.jsonl").open("a", encoding="utf-8") as log:
        log.write(
            json.dumps(
                {
                    "finished_unix_time": time.time(),
                    "elapsed_wall_seconds": time.perf_counter() - started,
                    "max_shots": args.max_shots,
                    "num_workers": args.num_workers,
                }
            )
            + "\n"
        )
    print(f"Completed; results and comparison saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
