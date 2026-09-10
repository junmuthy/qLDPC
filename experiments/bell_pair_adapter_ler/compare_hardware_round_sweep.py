#!/usr/bin/env python3
"""Compare matched round sweeps and benchmark adapter noise/detector policies.

LER estimates come only from the production summaries. Timing uses separate,
small samples; the two hardware representations decode identical records.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import stim

from qldpc import decoders
from qldpc.objects import Pauli

from .run_experiment import (
    _wilson_interval,
    build_experiment,
    build_fidelity_sweep_matrix,
    sample_experiments,
    write_artifacts,
    write_summary,
)

OUTPUTS = Path(__file__).resolve().parent / "outputs"
DEFAULT_OUTPUT = OUTPUTS / "adapter_hardware_prep3_round_sweep_1_5_comparison"
SETTINGS = (
    (0.0, "bell0", "intrinsic_prep3_round_sweep_1_5"),
    (0.001, "bell0p001", "intrinsic_prep3_round_sweep_1_5_bell_p0p001"),
)
POLICIES = ("full_surgery", "hardware_full", "hardware_reduced")
LABELS = ("Full surgery noise", "Hardware noise, all detectors", "Hardware noise, reduced")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def comparison_rows(output_dir: Path) -> list[dict[str, Any]]:
    """Read matched production datasets, rejecting mismatched or incomplete runs."""
    result = []
    for infidelity, slug, baseline in SETTINGS:
        for policy, directory in (
            ("archived_full_surgery", OUTPUTS / baseline),
            ("full_surgery", output_dir / "full_surgery_exact"),
            ("hardware_reduced", output_dir / slug),
        ):
            rows = read_csv(directory / "summary.csv")
            if policy == "full_surgery":
                rows = [r for r in rows if float(r["bell_pair_infidelity"]) == infidelity]
            manifest = {
                row["name"]: row for row in json.loads((directory / "manifest.json").read_text())
            }
            expected = {(basis, rounds) for basis in ("X", "Z") for rounds in range(1, 6)}
            observed = [(row["logical_basis"], int(row["surgery_rounds"])) for row in rows]
            if set(observed) != expected or len(observed) != len(expected):
                raise ValueError(f"Incomplete or duplicate round sweep: {directory}")
            for row in rows:
                metadata = manifest[row["name"]]
                targets = "adapter_hardware" if policy == "hardware_reduced" else "full_surgery"
                reduction = "inactive_rows" if policy == "hardware_reduced" else "none"
                if not (
                    int(row["preparation_rounds"]) == 3
                    and float(row["physical_error_rate"]) == 0.001
                    and float(row["bell_pair_infidelity"]) == infidelity
                    and row["noise_scope"] == "intrinsic"
                    and row["decoder"] == "qldpc_bp_osd"
                    and metadata["left_code"] == "rotated_surface_d3"
                    and metadata["right_code"] == "steane_d3"
                    and metadata["left_logical_index"] == metadata["right_logical_index"] == 0
                    and not metadata["include_idling_error"]
                    and metadata.get("adapter_noise_targets", "full_surgery") == targets
                    and metadata.get("detector_reduction", "none") == reduction
                    and (
                        policy != "full_surgery"
                        or metadata.get("dem_analysis") == "undecomposed_exact"
                    )
                ):
                    raise ValueError(f"Mismatched experiment: {row['name']}")
                shots, errors, discards = (int(row[key]) for key in ("shots", "errors", "discards"))
                if shots < 10000 and errors < 200:
                    raise ValueError(f"Unfinished production task: {row['name']}")
                low, high = _wilson_interval(errors, shots - discards)
                result.append(
                    {
                        "logical_basis": row["logical_basis"],
                        "surgery_rounds": int(row["surgery_rounds"]),
                        "bell_pair_infidelity": infidelity,
                        "bell_pair_fidelity": 1 - infidelity,
                        "policy": policy,
                        "dem_analysis": "sinter_auto"
                        if policy == "archived_full_surgery"
                        else "undecomposed_exact",
                        "shots": shots,
                        "discards": discards,
                        "errors": errors,
                        "logical_error_rate": errors / (shots - discards),
                        "ler_95ci_low": low,
                        "ler_95ci_high": high,
                        "worker_seconds": float(row["seconds"]),
                        "worker_seconds_per_10000_shots": 10000 * float(row["seconds"]) / shots,
                        "num_detectors": metadata["num_detectors"],
                        "source_summary": str(directory / "summary.csv"),
                    }
                )
    return result


def sample_matched_baseline(output_dir: Path, *, num_workers: int) -> None:
    """Resumable fresh full-noise reference with the hardware suite's exact DEM setting."""
    directory = output_dir / "full_surgery_exact"
    experiments = []
    for experiment in build_fidelity_sweep_matrix(
        bases=(Pauli.X, Pauli.Z),
        rounds=tuple(range(1, 6)),
        bell_pair_infidelities=(0.0, 0.001),
        physical_error_rate=0.001,
        preparation_rounds=3,
        noise_scope="intrinsic",
    ):
        name = experiment.name + "_exact_dem"
        experiments.append(
            dataclasses.replace(
                experiment,
                name=name,
                metadata=dict(
                    experiment.metadata,
                    name=name,
                    adapter_noise_targets="full_surgery",
                    detector_reduction="none",
                    dem_analysis="undecomposed_exact",
                ),
            )
        )
    write_artifacts(experiments, directory)
    config = {
        "max_shots": 10000,
        "max_errors": 200,
        "num_workers": num_workers,
        "preparation_rounds": 3,
        "surgery_rounds": list(range(1, 6)),
        "bell_pair_infidelities": [0.0, 0.001],
        "physical_error_rate": 0.001,
        "noise_scope": "intrinsic",
        "include_idling_error": False,
        "adapter_noise_targets": "full_surgery",
        "dem_analysis": "undecomposed_exact",
        "versions": {name: version(name) for name in ("stim", "sinter", "ldpc")},
    }
    (directory / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter()
    stats = sample_experiments(
        experiments, output_dir=directory, num_workers=num_workers, max_shots=10000, max_errors=200
    )
    write_summary(stats, directory)
    print(
        f"Matched full-surgery collection finished in {time.perf_counter() - started:.1f}s",
        flush=True,
    )


def plot_ler(rows: list[dict[str, Any]], output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for index, (infidelity, _, _) in enumerate(SETTINGS):
        for axis, basis in zip(axes[index], ("X", "Z")):
            for policy, label, color, marker in (
                ("archived_full_surgery", "Archived full surgery (auto DEM)", "0.55", "^"),
                ("full_surgery", "Full surgery (matched DEM)", "C0", "o"),
                ("hardware_reduced", "Adapter hardware (matched DEM)", "C1", "s"),
            ):
                selected = sorted(
                    (
                        r
                        for r in rows
                        if r["policy"] == policy
                        and r["logical_basis"] == basis
                        and r["bell_pair_infidelity"] == infidelity
                    ),
                    key=lambda r: r["surgery_rounds"],
                )
                x = np.array([r["surgery_rounds"] for r in selected])
                y, low, high = [
                    np.array([r[key] for r in selected]) * 100
                    for key in ("logical_error_rate", "ler_95ci_low", "ler_95ci_high")
                ]
                nonzero = y > 0
                axis.errorbar(
                    x[nonzero],
                    y[nonzero],
                    yerr=[y[nonzero] - low[nonzero], high[nonzero] - y[nonzero]],
                    color=color,
                    marker=marker,
                    capsize=4,
                    label=label,
                )
                if np.any(~nonzero):
                    axis.errorbar(
                        x[~nonzero],
                        high[~nonzero],
                        yerr=high[~nonzero] * 0.3,
                        uplims=True,
                        color=color,
                        fmt=marker,
                    )
            axis.set_title(f"{basis}{basis} — Bell fidelity {1 - infidelity:g}")
            axis.set_yscale("log")
            axis.set_xticks(range(1, 6))
            axis.grid(alpha=0.25, which="both")
    axes[0, 0].legend(fontsize=9)
    for axis in axes[:, 0]:
        axis.set_ylabel("Adapter LER (%) — 95% Wilson intervals")
    for axis in axes[1]:
        axis.set_xlabel("Adapter syndrome rounds")
    figure.suptitle(
        "Surface-3–Steane: three ideal preparation rounds, p=0.001\n"
        "Up to 10,000 shots or 200 failures per point; ideal closure, no idle noise"
    )
    save_figure(figure, output_dir / "adapter_ler_vs_rounds_comparison")


def save_figure(figure: Any, path: Path) -> None:
    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(path.with_suffix(f".{extension}"), dpi=180)
    plt.close(figure)


def benchmark(output_dir: Path, *, shots: int, repeats: int, seed: int) -> None:
    """Serial timing, rotating policy order and checkpointing each condition.

    Run after production sampling has finished. Build/compile costs are recorded
    separately; decode measurements exclude them and one untimed warmup shot.
    Hardware models share records, including their measured sampling time.
    """
    directory = output_dir / "benchmark"
    directory.mkdir(exist_ok=True)
    config = {
        "shots_per_repeat": shots,
        "repeats": repeats,
        "seed": seed,
        "rounds": list(range(1, 6)),
        "bell_pair_infidelities": [0.0, 0.001],
        "preparation_rounds": 3,
        "physical_error_rate": 0.001,
        "noise_scope": "intrinsic",
        "include_idling_error": False,
        "versions": {name: version(name) for name in ("stim", "sinter", "ldpc", "numpy")},
        "python": platform.python_version(),
        "machine": platform.machine(),
        "threads": {
            name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "hardware_records_paired": True,
        "timing_shots_excluded_from_production_ler": True,
        "decoder": "qldpc_bp_osd",
        "dem": "undecomposed_exact",
    }
    config_path = directory / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Benchmark configuration changed; choose a different output directory")
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    for setting_index, (infidelity, slug, baseline) in enumerate(SETTINGS):
        for basis_index, basis in enumerate((Pauli.X, Pauli.Z)):
            for rounds in range(1, 6):
                checkpoint = directory / f"{slug}_{basis.name}_rounds{rounds}.csv"
                if checkpoint.exists():
                    saved = read_csv(checkpoint)
                    expected = {
                        (policy, repeat) for policy in POLICIES for repeat in range(repeats)
                    }
                    actual = {(r["policy"], int(r["repeat"])) for r in saved}
                    if len(saved) != len(expected) or actual != expected:
                        raise ValueError(f"Incomplete timing checkpoint: {checkpoint}")
                    continue
                experiments, compilers, converters, build_times, compile_times = {}, {}, {}, {}, {}
                for policy in POLICIES:
                    started = time.perf_counter()
                    experiment = build_experiment(
                        basis=basis,
                        rounds=rounds,
                        preparation_rounds=3,
                        physical_error_rate=0.001,
                        bell_pair_infidelity=infidelity,
                        adapter_noise_targets=(
                            "full_surgery" if policy == "full_surgery" else "adapter_hardware"
                        ),
                        reduce_detectors=policy == "hardware_reduced",
                    )
                    build_times[policy] = time.perf_counter() - started
                    if policy == "full_surgery":
                        archived = OUTPUTS / baseline / "circuits" / f"{experiment.name}.stim"
                        # Text artifacts round floating arguments on serialization.
                        if str(experiment.circuit) != str(stim.Circuit.from_file(archived)):
                            raise ValueError(f"Baseline circuit changed: {archived}")
                    experiments[policy] = experiment
                    converters[policy] = experiment.circuit.compile_m2d_converter()
                    started = time.perf_counter()
                    compilers[policy] = decoders.SinterDecoder().compile_decoder_for_dem(
                        experiment.detector_error_model
                    )
                    compile_times[policy] = time.perf_counter() - started
                    compilers[policy].decode_shots(
                        np.zeros((1, experiment.circuit.num_detectors), dtype=np.uint8)
                    )
                rows = []
                for repeat in range(repeats):
                    sample_seed = (
                        seed + 10000 * setting_index + 1000 * basis_index + 10 * rounds + 2 * repeat
                    )
                    records, sampling_times = {}, {}
                    for offset, policy in enumerate(POLICIES[:2]):
                        started = time.perf_counter()
                        records[policy] = (
                            experiments[policy]
                            .circuit.compile_sampler(seed=sample_seed + offset)
                            .sample(shots)
                        )
                        sampling_times[policy] = time.perf_counter() - started
                    records["hardware_reduced"] = records["hardware_full"]
                    sampling_times["hardware_reduced"] = sampling_times["hardware_full"]
                    predictions, detectors, observables = {}, {}, {}
                    repeat_rows = []
                    for policy in POLICIES[repeat % 3 :] + POLICIES[: repeat % 3]:
                        started = time.perf_counter()
                        detectors[policy], observables[policy] = converters[policy].convert(
                            measurements=records[policy], separate_observables=True
                        )
                        conversion_seconds = time.perf_counter() - started
                        wall_start, cpu_start = time.perf_counter(), time.process_time()
                        predictions[policy] = compilers[policy].decode_shots(detectors[policy])
                        cpu_seconds = time.process_time() - cpu_start
                        wall_seconds = time.perf_counter() - wall_start
                        repeat_rows.append(
                            {
                                "logical_basis": basis.name,
                                "surgery_rounds": rounds,
                                "bell_pair_infidelity": infidelity,
                                "policy": policy,
                                "repeat": repeat,
                                "shots": shots,
                                "seed": sample_seed + int(policy != "full_surgery"),
                                "num_qubits": experiments[policy].circuit.num_qubits,
                                "num_detectors": experiments[policy].circuit.num_detectors,
                                "num_decoder_error_variables": compilers[
                                    policy
                                ].dem_arrays.num_errors,
                                "build_seconds": build_times[policy],
                                "decoder_compile_seconds": compile_times[policy],
                                "record_sampling_seconds": sampling_times[policy],
                                "record_sampling_shared": policy == "hardware_reduced",
                                "conversion_seconds": conversion_seconds,
                                "decode_wall_seconds": wall_seconds,
                                "decode_cpu_seconds": cpu_seconds,
                                "decode_ms_per_shot": 1000 * wall_seconds / shots,
                                "failures": int(
                                    np.count_nonzero(
                                        np.any(predictions[policy] != observables[policy], axis=1)
                                    )
                                ),
                                "paired_prediction_disagreements": "",
                            }
                        )
                    mapping = experiments["hardware_reduced"].metadata["retained_detector_ids"]
                    if not np.array_equal(
                        detectors["hardware_reduced"], detectors["hardware_full"][:, mapping]
                    ):
                        raise ValueError("Detector projection changed the shared samples")
                    if not np.array_equal(
                        observables["hardware_reduced"], observables["hardware_full"]
                    ):
                        raise ValueError("Detector reduction changed the observable")
                    disagreements = int(
                        np.count_nonzero(
                            np.any(
                                predictions["hardware_reduced"] != predictions["hardware_full"],
                                axis=1,
                            )
                        )
                    )
                    for row in repeat_rows:
                        if row["policy"] == "hardware_reduced":
                            row["paired_prediction_disagreements"] = disagreements
                    rows.extend(repeat_rows)
                write_csv(checkpoint, rows)
                print(
                    f"Benchmarked {basis}{basis}, rounds={rounds}, F={1 - infidelity:g}; "
                    f"paired disagreements={sum(int(r['paired_prediction_disagreements'] or 0) for r in rows)}",
                    flush=True,
                )
    all_rows = [
        row for path in sorted(directory.glob("bell*_rounds*.csv")) for row in read_csv(path)
    ]
    write_csv(output_dir / "timing_samples.csv", all_rows)


def summarize_timings(output_dir: Path) -> list[dict[str, Any]]:
    samples = read_csv(output_dir / "timing_samples.csv")
    rows: list[dict[str, Any]] = []
    for infidelity, _, _ in SETTINGS:
        for basis in ("X", "Z"):
            for rounds in range(1, 6):
                condition = [
                    r
                    for r in samples
                    if float(r["bell_pair_infidelity"]) == infidelity
                    and r["logical_basis"] == basis
                    and int(r["surgery_rounds"]) == rounds
                ]
                medians = {
                    policy: float(
                        np.median(
                            [
                                float(r["decode_ms_per_shot"])
                                for r in condition
                                if r["policy"] == policy
                            ]
                        )
                    )
                    for policy in POLICIES
                }
                for policy in POLICIES:
                    selected = [r for r in condition if r["policy"] == policy]
                    if not selected:
                        raise ValueError("Incomplete timing benchmark")
                    times = [float(r["decode_ms_per_shot"]) for r in selected]
                    rows.append(
                        {
                            "logical_basis": basis,
                            "surgery_rounds": rounds,
                            "bell_pair_infidelity": infidelity,
                            "policy": policy,
                            "shots_per_repeat": int(selected[0]["shots"]),
                            "repeats": len(selected),
                            "median_decode_ms_per_shot": medians[policy],
                            "min_decode_ms_per_shot": min(times),
                            "max_decode_ms_per_shot": max(times),
                            "speedup_vs_full_surgery": medians["full_surgery"] / medians[policy],
                            "reduction_only_speedup": medians["hardware_full"]
                            / medians["hardware_reduced"],
                            "num_detectors": int(selected[0]["num_detectors"]),
                            "num_decoder_error_variables": int(
                                selected[0]["num_decoder_error_variables"]
                            ),
                            "paired_prediction_disagreements": sum(
                                int(r["paired_prediction_disagreements"] or 0) for r in selected
                            ),
                        }
                    )
    write_csv(output_dir / "timing_summary.csv", rows)
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for index, (infidelity, _, _) in enumerate(SETTINGS):
        for axis, basis in zip(axes[index], ("X", "Z")):
            for policy, label in zip(POLICIES, LABELS):
                selected = [
                    r
                    for r in rows
                    if r["policy"] == policy
                    and r["logical_basis"] == basis
                    and r["bell_pair_infidelity"] == infidelity
                ]
                y, low, high = [
                    np.array([r[key] for r in selected])
                    for key in (
                        "median_decode_ms_per_shot",
                        "min_decode_ms_per_shot",
                        "max_decode_ms_per_shot",
                    )
                ]
                axis.errorbar(
                    range(1, 6), y, yerr=[y - low, high - y], marker="o", capsize=3, label=label
                )
            axis.set_title(f"{basis}{basis} — Bell fidelity {1 - infidelity:g}")
            axis.set_yscale("log")
            axis.set_xticks(range(1, 6))
            axis.grid(alpha=0.25, which="both")
    axes[0, 0].legend(fontsize=8)
    for axis in axes[:, 0]:
        axis.set_ylabel("Decoder wall time (ms/shot)")
    for axis in axes[1]:
        axis.set_xlabel("Adapter syndrome rounds")
    first = rows[0]
    figure.suptitle(
        f"Fresh serial BP+OSD benchmark — median and min–max range\n"
        f"{first['repeats']} batches × {first['shots_per_repeat']} shots; hardware models use paired samples"
    )
    save_figure(figure, output_dir / "adapter_decode_time_vs_rounds")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--sample-matched-baseline", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--benchmark-shots", type=int, default=256)
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    if args.benchmark_shots < 1 or args.benchmark_repeats < 1:
        parser.error("Benchmark shots and repeats must be positive")
    if args.sample_matched_baseline:
        sample_matched_baseline(args.output_dir, num_workers=args.num_workers)
    rows = comparison_rows(args.output_dir)
    write_csv(args.output_dir / "comparison.csv", rows)
    plot_ler(rows, args.output_dir)
    if args.benchmark:
        benchmark(
            args.output_dir,
            shots=args.benchmark_shots,
            repeats=args.benchmark_repeats,
            seed=args.seed,
        )
    if (args.output_dir / "timing_samples.csv").exists():
        summarize_timings(args.output_dir)
    print(f"Comparison saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
