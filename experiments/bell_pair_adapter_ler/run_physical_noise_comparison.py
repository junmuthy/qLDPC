"""Checkpointed XX-only HGP/surface distance/rounds comparison at lower physical p.

Six conditions: (m=3, r=4), (m=3, r=8), (m=4, r=8) at p=3e-4 and 1e-4.
The default target is 1,000 cumulative shots per condition, never an automatic
extension to 10,000. One worker pool runs at a time; higher p is collected first.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import platform
import time
import uuid
from pathlib import Path
from typing import Any

import sinter
import stim
from scipy.stats import beta

from qldpc import decoders
from qldpc.objects import Pauli

from .collection_profile import CollectionProfile
from .run_experiment import (
    PPM_BASIS_FIRST_VERSION,
    BuiltExperiment,
    build_experiment_for_codes,
    sample_experiments,
    write_artifacts,
    write_summary,
)
from .run_ppm_basis_first import code_pair, totals
from .run_ppm_fidelity_sweep import (
    check_output,
    digest,
    runtime_identity,
    saved_stats,
    task_for,
    write_json,
)

CONFIGURATIONS = ((3, 4, 4), (3, 4, 8), (4, 8, 8))  # simplex m, surface d, rounds
OUTPUTS = Path(__file__).resolve().parent / "outputs"
REFERENCES = (
    (3, 4, OUTPUTS / "ppm_basis_first_hgp_m3_d4_prep4_surg4_bell_fidelity"),
    (4, 8, OUTPUTS / "ppm_basis_first_hgp_m4_d8_prep4_surg8_10000" / "samples"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-error-rates", type=float, nargs="+", default=[3e-4, 1e-4])
    parser.add_argument("--max-shots", type=int, default=1000, help="Cumulative shots/condition")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-batch-seconds", type=int, default=5)
    parser.add_argument("--verbose-progress", action="store_true")
    parser.add_argument("--without-archived-references", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args(argv)
    if any(not 0 < p < 1 for p in args.physical_error_rates):
        parser.error("Physical probabilities must be finite and strictly between zero and one")
    if len(set(args.physical_error_rates)) != len(args.physical_error_rates):
        parser.error("Duplicate physical error rates")
    if min(args.max_shots, args.num_workers, args.max_batch_size, args.max_batch_seconds) < 1:
        parser.error("Shot, worker, and batch limits must be positive")
    args.physical_error_rates = sorted(args.physical_error_rates, reverse=True)
    return args


def scaled_noise_signature(circuit: stim.Circuit, p: float) -> str:
    """Hash all operations/targets with physical noise arguments divided by p.

    Unlike without_noise(), this also checks where the noisy gates occur and which
    qubits they act on. No Bell-source channel is permitted in this F=1 experiment.
    """
    instructions = []
    for op in circuit.flattened():
        if op.name.startswith("PAULI_CHANNEL"):
            raise ValueError("Unexpected Bell-source/Pauli channel in ideal-source experiment")
        args = op.gate_args_copy()
        if stim.gate_data(op.name).is_noisy_gate:
            args = [round(value / p, 12) for value in args]
        instructions.append((op.name, [str(t) for t in op.targets_copy()], args))
    return digest(json.dumps(instructions))


def build_sweep(
    args: argparse.Namespace,
    *,
    configurations: tuple[tuple[int, int, int], ...] = CONFIGURATIONS,
) -> tuple[list[BuiltExperiment], list[dict[str, Any]]]:
    pairs = {
        m: code_pair("hgp-simplex-surface", simplex_m=m)
        for m in sorted({m for m, _, _ in configurations})
    }
    experiments, audits = [], []
    templates: dict[tuple[int, int], tuple[str, list[int]]] = {}
    for p in args.physical_error_rates:
        for m, distance, rounds in configurations:
            started = time.perf_counter()
            left, right, left_name, right_name = pairs[m]
            experiment = build_experiment_for_codes(
                left_code=left,
                right_code=right,
                left_code_name=left_name,
                right_code_name=right_name,
                left_logical_index=0,
                right_logical_index=0,
                basis=Pauli.X,
                preparation_rounds=4,
                surgery_rounds=rounds,
                physical_error_rate=p,
                bell_pair_infidelity=0.0,
                noise_scope="intrinsic",
                include_idling_error=False,
                adapter_noise_targets="adapter_hardware",
                reduce_detectors=True,
                compute_exact_distances=False,
                schedule_policy="ppm_basis_first",
            )
            meta = experiment.metadata
            if meta["surgery_schedule"] != dict(
                local_order=["X", "Z"], bell_order=["X", "Z"], local_layers_sha256=None
            ):
                raise ValueError("Unexpected XX extraction schedule")
            signature = scaled_noise_signature(experiment.circuit, p)
            template = (signature, meta["adapter_hardware_qubits"])
            if templates.setdefault((m, rounds), template) != template:
                raise ValueError("Physical-p sweep changed operations, noise locations, or mask")
            detectors, observables = (
                experiment.circuit.without_noise()
                .compile_detector_sampler(seed=731)
                .sample(shots=32, separate_observables=True)
            )
            if detectors.any() or observables.any() or experiment.circuit.num_observables != 1:
                raise ValueError("Noiseless detector/XX-observable validation failed")
            compiled = decoders.SinterDecoder().compile_decoder_for_dem(
                experiment.detector_error_model
            )
            settings = {
                key: getattr(compiled.decoder, key)
                for key in ("osd_method", "osd_order", "bp_method", "schedule", "max_iter")
            }
            if settings["osd_order"] != 0:
                raise ValueError("Production decoder is no longer order zero")
            audits.append(
                dict(
                    name=experiment.name,
                    simplex_m=m,
                    nominal_distance=distance,
                    surgery_rounds=rounds,
                    physical_error_rate=p,
                    scaled_noise_sha256=signature,
                    circuit_sha256=digest(str(experiment.circuit)),
                    dem_sha256=digest(str(experiment.detector_error_model)),
                    noiseless_validation_shots=32,
                    decoder_settings=settings,
                    num_detectors=meta["num_detectors"],
                    num_dem_errors=meta["num_dem_errors"],
                    build_wall_seconds=time.perf_counter() - started,
                )
            )
            experiments.append(experiment)
            print(
                f"Validated XX: m={m}, d={distance}, rounds={rounds}, p={p:g}; "
                f"{meta['num_detectors']} detectors; {audits[-1]['build_wall_seconds']:.2f}s",
                flush=True,
            )
    return experiments, audits


def exact_interval(errors: int, shots: int) -> tuple[float, float]:
    """Two-sided 95% Clopper-Pearson interval, including zero-failure bounds."""
    if not 0 <= errors <= shots or shots < 1:
        raise ValueError("Invalid failure/shot counts")
    return (
        float(beta.ppf(0.025, errors, shots - errors + 1)) if errors else 0.0,
        float(beta.ppf(0.975, errors + 1, shots - errors)) if errors < shots else 1.0,
    )


def result_row(stat: sinter.TaskStats, source: str) -> dict[str, Any]:
    meta = stat.json_metadata
    low, high = exact_interval(stat.errors, stat.shots - stat.discards)
    return dict(
        simplex_m=int(meta["left_code"].removeprefix("hgp_simplex_m")),
        nominal_distance=int(meta["right_code"].removeprefix("surface_unrotated_d")),
        surgery_rounds=meta["surgery_rounds"],
        logical_basis=meta["logical_basis"],
        physical_error_rate=meta["physical_error_rate"],
        shots=stat.shots,
        errors=stat.errors,
        discards=stat.discards,
        logical_error_rate=stat.errors / (stat.shots - stat.discards),
        ler_95ci_low=low,
        ler_95ci_high=high,
        interval="two_sided_95pct_Clopper_Pearson",
        worker_seconds=stat.seconds,
        worker_seconds_per_shot=stat.seconds / stat.shots,
        strong_id=stat.strong_id,
        source=source,
    )


def archived_references(experiments: list[BuiltExperiment]) -> list[dict[str, Any]]:
    """Read only: validate archived p=1e-3 controls; never import their samples."""
    rows = []
    identity = runtime_identity()
    for m, rounds, directory in REFERENCES:
        current = next(
            e
            for e in experiments
            if e.metadata["left_code"] == f"hgp_simplex_m{m}"
            and e.metadata["surgery_rounds"] == rounds
        )
        config = json.loads((directory / "run_config.json").read_text())
        prior_identity = config.get("runtime_identity")
        if prior_identity is not None:
            if prior_identity != identity:
                raise ValueError(f"Archived decoder/dependency identity changed: {directory}")
        elif config["versions"] != identity["versions"]:
            raise ValueError(f"Archived dependency versions differ: {directory}")
        candidates = [
            meta
            for meta in json.loads((directory / "manifest.json").read_text())
            if meta["logical_basis"] == "X" and meta["bell_pair_infidelity"] == 0
        ]
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous archived XX control: {directory}")
        archived = candidates[0]
        if archived["physical_error_rate"] != 0.001:
            raise ValueError("Expected an archived p=0.001 control")
        metadata = {k: v for k, v in archived.items() if k not in {"circuit_path", "dem_path"}}
        for key in metadata.keys() | current.metadata.keys():
            if key not in {"name", "physical_error_rate"}:
                if metadata.get(key) != current.metadata.get(key):
                    raise ValueError(f"Archived control differs in {key}: {directory}")
        circuit = stim.Circuit.from_file(directory / archived["circuit_path"])
        if scaled_noise_signature(
            circuit, metadata["physical_error_rate"]
        ) != scaled_noise_signature(current.circuit, current.metadata["physical_error_rate"]):
            raise ValueError(f"Archived operations/noise mask differ: {directory}")
        dem = stim.DetectorErrorModel.from_file(directory / archived["dem_path"])
        experiment = BuiltExperiment(archived["name"], circuit, dem, metadata)
        expected_id = task_for(experiment).strong_id()
        stats = [
            s
            for s in sinter.read_stats_from_csv_files(directory / "sinter_progress.csv")
            if s.strong_id == expected_id
        ]
        if len(stats) != 1 or stats[0].json_metadata != metadata:
            raise ValueError(f"Missing archived checkpoint identity: {directory}")
        row = result_row(stats[0], str(directory))
        row["decoder_provenance"] = (
            "source hashes and versions verified"
            if prior_identity
            else "versions/default decoder verified; archive has no decoder-source hashes"
        )
        rows.append(row)
    return rows


def write_analysis(
    output: Path, experiments: list[BuiltExperiment], target: int, references: list[dict[str, Any]]
) -> None:
    import matplotlib.pyplot as plt

    stats = saved_stats(output, experiments)
    if len(stats) != len(experiments) or any(s.shots != target for s in stats):
        raise ValueError("Analysis requires the target count for every condition")
    stage = output / f"analysis_{target}"
    stage.mkdir(exist_ok=True)
    rows = [result_row(s, "new_sweep") for s in stats] + references
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (stage / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    fig, axis = plt.subplots(figsize=(8, 5.5))
    for index, (m, distance, rounds) in enumerate(CONFIGURATIONS):
        selected = sorted(
            [r for r in rows if r["simplex_m"] == m and r["surgery_rounds"] == rounds],
            key=lambda row: row["physical_error_rate"],
        )
        color = f"C{index}"
        axis.plot([], [], "o", color=color, label=f"d={distance}, {rounds} adapter rounds")
        for row in selected:
            x, y = row["physical_error_rate"], row["logical_error_rate"]
            x *= (0.94, 1.0, 1.06)[index]  # Separate coincident zero-failure bounds.
            low, high = row["ler_95ci_low"], row["ler_95ci_high"]
            if row["errors"] == 0:
                axis.errorbar(x, high, yerr=high * 0.4, uplims=True, fmt="_", color=color)
            else:
                axis.errorbar(
                    x,
                    y,
                    yerr=[[y - low], [high - y]],
                    fmt="o",
                    capsize=4,
                    color=color,
                    mfc=color if row["source"] == "new_sweep" else "white",
                )
        # Do not draw a line through upper limits as though they were measured rates.
    axis.set(
        xscale="log",
        yscale="log",
        xlabel="Physical error probability p",
        ylabel="XX adapter LER per complete PPM",
        title="HGP simplex / unrotated surface — physical-noise comparison",
    )
    axis.grid(which="both", alpha=0.2)
    ticks = sorted({r["physical_error_rate"] for r in rows})
    axis.set_xticks(ticks, [f"{p:g}" for p in ticks])
    axis.legend()
    fig.text(
        0.5,
        0.025,
        f"{target:,} new shots/point; four ideal preparation rounds; Bell-source fidelity 1.\n"
        "Hardware-only; matching-basis; production BP+OSD. Exact 95% intervals.\n"
        "Arrows: zero-failure upper bounds. Open circles: archived controls.\n"
        "Small horizontal offsets separate overlapping points; actual p values are at ticks.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    for suffix in ("pdf", "png"):
        fig.savefig(stage / f"adapter_ler_vs_physical_error_rate.{suffix}", dpi=180)
    plt.close(fig)
    events = [json.loads(line) for line in (output / "invocations.jsonl").read_text().splitlines()]
    completed = [e for e in events if e["status"] == "completed" and e.get("new_shots", 0)]
    timing = dict(
        per_condition=[result_row(s, "new_sweep") for s in stats],
        completed_invocations=completed,
        caveat="Worker seconds include sampling/decoding, not pure CPU. Elapsed times include startup/tails. Hard-stop elapsed time is unavailable.",
    )
    if completed:
        latest = completed[-1]
        remaining = sum(max(0, 10000 - s.shots) for s in stats)
        timing["remaining_shots_to_10000"] = remaining
        timing["rough_extension_wall_seconds_at_latest_throughput"] = (
            remaining * latest["sampling_decode_wall_seconds"] / latest["new_shots"]
        )
    write_json(stage / "timing.json", timing)
    print(f"Analysis saved: {stage}", flush=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".collection.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another writer owns this comparison directory") from error
        if not (output / "manifest.json").exists() and any(
            p.name != ".collection.lock" for p in output.iterdir()
        ):
            raise ValueError("Nonempty output directory without a matching manifest")
        started = time.perf_counter()
        experiments, audits = build_sweep(args)
        check_output(output, experiments)
        identity = runtime_identity()
        config_path = output / "run_config.json"
        if (
            config_path.exists()
            and json.loads(config_path.read_text())["runtime_identity"] != identity
        ):
            raise ValueError("Decoder/dependency identity changed; use a new output directory")
        references = [] if args.without_archived_references else archived_references(experiments)
        if args.analyze_only:
            write_analysis(output, experiments, args.max_shots, references)
            return {"status": "analyzed"}
        before = saved_stats(output, experiments)
        if any(s.shots > args.max_shots for s in before):
            raise ValueError("Shot target is below saved counts; refusing to discard data")
        write_artifacts(experiments, output)
        write_json(output / "preflight.json", audits)
        write_json(output / "archived_references.json", references)
        config = {
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "runtime_identity": identity,
            "configurations": CONFIGURATIONS,
            "basis": "X",
            "preparation_rounds": 4,
            "bell_pair_infidelity": 0.0,
            "noise_scope": "intrinsic",
            "adapter_noise_targets": "adapter_hardware",
            "schedule_policy": PPM_BASIS_FIRST_VERSION,
            "compute_exact_distances": False,
            "include_idling_error": False,
            "decoder_kwargs": {},
            "python": platform.python_version(),
            "threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        }
        write_json(config_path, config)
        event: dict[str, Any] = dict(
            invocation_id=uuid.uuid4().hex,
            config=config,
            before=totals(before),
            started_unix_time=time.time(),
            build_wall_seconds=time.perf_counter() - started,
        )

        def log(status: str) -> None:
            with (output / "invocations.jsonl").open("a") as stream:
                stream.write(json.dumps({**event, "status": status}, sort_keys=True) + "\n")

        log("built" if args.build_only else "sampling_started")
        if args.build_only:
            return {**event, "status": "built"}
        profile = CollectionProfile(output, event["invocation_id"])
        counts = {s.strong_id: s.shots for s in before}
        labels = {task_for(e).strong_id(): e.name for e in experiments}
        last_print = 0.0

        def progress(update: sinter.Progress) -> None:
            nonlocal last_print
            profile.progress(update)
            for stat in update.new_stats:
                counts[stat.strong_id] = counts.get(stat.strong_id, 0) + stat.shots
            if time.perf_counter() - last_print >= 30:
                for strong_id, label in labels.items():
                    print(
                        f"Checkpointed {counts.get(strong_id, 0)}/{args.max_shots}: {label}",
                        flush=True,
                    )
                last_print = time.perf_counter()

        status = "failed"
        sampling_started = time.perf_counter()
        try:
            profile.start()
            for p in args.physical_error_rates:
                print(f"Collecting p={p:g} with {args.num_workers} workers TOTAL", flush=True)
                sample_experiments(
                    [e for e in experiments if e.metadata["physical_error_rate"] == p],
                    output_dir=output,
                    num_workers=args.num_workers,
                    max_shots=args.max_shots,
                    max_errors=args.max_shots,
                    print_progress=args.verbose_progress,
                    progress_callback=progress,
                    max_batch_size=args.max_batch_size,
                    max_batch_seconds=args.max_batch_seconds,
                )
            status = "completed"
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        finally:
            saved = saved_stats(output, experiments)
            if saved:
                write_summary(saved, output)
            after = totals(saved)
            event.update(
                after=after,
                new_shots=after["shots"] - event["before"]["shots"],
                sampling_decode_wall_seconds=time.perf_counter() - sampling_started,
                finished_unix_time=time.time(),
                profile=profile.finish(),
            )
            log(status)
        write_analysis(output, experiments, args.max_shots, references)
        print(f"Finished {args.max_shots} shots per condition; pilot target reached.", flush=True)
        return {**event, "status": status}


if __name__ == "__main__":
    run(parse_args())
