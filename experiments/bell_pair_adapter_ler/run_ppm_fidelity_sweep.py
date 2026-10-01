"""Resumable matching-basis hardware-only Bell-fidelity sweeps for CSS code presets.

One shared worker pool; no exact-distance search or imports of archived shot counts.
Default collection stops at a 1,000-shot-per-condition pilot. Resume explicitly
with --max-shots 20000 for the main sweep. Completed analysis stages are preserved.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import platform
import time
import uuid
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from typing import Any

import sinter
import stim

from qldpc.objects import Pauli

from .analyze_fidelity_sweep import write_fidelity_analysis
from .collection_profile import CollectionProfile
from .run_experiment import (
    DEFAULT_FIDELITY_SWEEP_INFIDELITIES,
    PPM_BASIS_FIRST_VERSION,
    BuiltExperiment,
    build_experiment_for_codes,
    sample_experiments,
    write_artifacts,
    write_summary,
)
from .run_ppm_basis_first import code_pair, totals


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--code-pair",
        choices=("hgp-simplex-surface", "surface-steane"),
        default="hgp-simplex-surface",
    )
    parser.add_argument("--simplex-m", type=int, default=3)
    parser.add_argument("--surface-distance", type=int)
    parser.add_argument("--left-logical-index", type=int, default=0)
    parser.add_argument("--right-logical-index", type=int, default=0)
    parser.add_argument("--preparation-rounds", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--bases", choices=("X", "Z"), nargs="+", default=["X", "Z"])
    parser.add_argument("--physical-error-rate", type=float, default=0.001)
    grid = parser.add_mutually_exclusive_group()
    grid.add_argument("--bell-pair-fidelities", type=float, nargs="+")
    grid.add_argument("--bell-pair-infidelities", type=float, nargs="+")
    parser.add_argument(
        "--max-shots",
        type=int,
        default=1000,
        help="Cumulative shots per condition; pilot shots are retained on resume",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batch-size", type=int, default=128)
    parser.add_argument("--max-batch-seconds", type=int, default=5)
    parser.add_argument("--verbose-progress", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--build-only", action="store_true")
    modes.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args(argv)
    supplied = (
        args.bell_pair_fidelities
        if args.bell_pair_fidelities is not None
        else (
            args.bell_pair_infidelities
            if args.bell_pair_infidelities is not None
            else DEFAULT_FIDELITY_SWEEP_INFIDELITIES
        )
    )
    if not all(0 <= p <= 1 for p in (*supplied, args.physical_error_rate)):
        parser.error("All probabilities must be finite and between zero and one")
    # Decimal subtraction preserves canonical 0.001 instead of 0.0010000000000000009.
    infidelities = (
        [float(Decimal(1) - Decimal(str(f))) for f in supplied]
        if args.bell_pair_fidelities is not None
        else list(supplied)
    )
    if len(infidelities) != len(set(infidelities)):
        parser.error("Duplicate fidelities are not allowed")
    if len(infidelities) < 2 or 0.0 not in infidelities:
        parser.error("Include fidelity 1 and at least one nonideal fidelity")
    args.bell_pair_infidelities = sorted(infidelities)
    args.bell_pair_fidelities = [1 - p for p in args.bell_pair_infidelities]
    if len(set(args.bases)) != len(args.bases):
        parser.error("Duplicate bases are not allowed")
    if (
        min(
            args.preparation_rounds,
            args.rounds,
            args.max_shots,
            args.num_workers,
            args.max_batch_size,
            args.max_batch_seconds,
        )
        < 1
    ):
        parser.error("Round counts, shots, workers and batch limits must be positive")
    if args.simplex_m < 2 or (args.surface_distance is not None and args.surface_distance < 2):
        parser.error("Simplex m and surface distance must be at least two")
    if min(args.left_logical_index, args.right_logical_index) < 0:
        parser.error("Logical indices must be nonnegative")
    if args.code_pair == "surface-steane" and (
        args.simplex_m != 3 or args.surface_distance is not None
    ):
        parser.error("Simplex m/surface distance apply only to HGP-simplex/surface")
    return args


def digest(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()


def task_for(experiment: BuiltExperiment) -> sinter.Task:
    return sinter.Task(
        circuit=experiment.circuit,
        detector_error_model=experiment.detector_error_model,
        decoder="qldpc_bp_osd",
        json_metadata=experiment.metadata,
    )


def build_sweep(args: argparse.Namespace) -> tuple[list[BuiltExperiment], list[dict[str, Any]]]:
    left, right, left_name, right_name = code_pair(
        args.code_pair, simplex_m=args.simplex_m, surface_distance=args.surface_distance
    )
    experiments, audit = [], []
    for basis in args.bases:
        baseline = None
        source_template = None
        for infidelity in args.bell_pair_infidelities:
            started = time.perf_counter()
            experiment = build_experiment_for_codes(
                left_code=left,
                right_code=right,
                left_code_name=left_name,
                right_code_name=right_name,
                left_logical_index=args.left_logical_index,
                right_logical_index=args.right_logical_index,
                basis=Pauli[basis],
                preparation_rounds=args.preparation_rounds,
                surgery_rounds=args.rounds,
                physical_error_rate=args.physical_error_rate,
                bell_pair_infidelity=infidelity,
                noise_scope="intrinsic",
                include_idling_error=False,
                adapter_noise_targets="adapter_hardware",
                reduce_detectors=True,
                compute_exact_distances=False,
                schedule_policy="ppm_basis_first",
            )
            meta = experiment.metadata
            expected_schedule = dict(
                local_order=["X", "Z"] if basis == "X" else ["Z", "X"],
                bell_order=["X", "Z"],
                local_layers_sha256=None,
            )
            if meta["surgery_schedule"] != expected_schedule:
                raise ValueError("Unexpected extraction schedule")
            # Detector reduction may change row selection when the noise support changes.
            # Compare all physical operations after removing detector declarations and ONLY
            # the explicit Bell-source channel. Other physical noise must be identical.
            physical = stim.Circuit()
            source_channels = []
            source_locations = []
            for instruction in experiment.circuit.flattened():
                if instruction.name == "DETECTOR":
                    continue
                if instruction.name == "PAULI_CHANNEL_1":
                    source_channels.append(instruction)
                    source_locations.append((len(physical), tuple(instruction.targets_copy())))
                    if any(abs(p - infidelity / 3) > 1e-14 for p in instruction.gate_args_copy()):
                        raise ValueError("Unexpected Bell-source probability")
                    continue
                # An explicit source channel introduces an empty noise-only moment.
                # After removing it, collapse consecutive TICK markers on BOTH sides.
                # Idling is disabled; no physical operation/noise channel is removed.
                if instruction.name == "TICK" and physical and physical[-1].name == "TICK":
                    continue
                physical.append(instruction)
            channel_count = sum(len(i.targets_copy()) for i in source_channels)
            expected_count = (
                args.rounds * meta["num_bell_pair_checks_per_round"] if infidelity else 0
            )
            if channel_count != expected_count:
                raise ValueError("Bell-source channel count disagrees with Bell consumption")
            signature = (str(physical), meta["adapter_hardware_qubits"])
            if baseline is None:
                baseline = signature
            elif signature != baseline:
                raise ValueError("Fidelity sweep changed non-Bell operations or the hardware mask")
            if source_channels:
                if source_template is None:
                    source_template = source_locations
                elif source_template != source_locations:
                    raise ValueError(
                        "Bell-source channel targets/locations changed across fidelities"
                    )
            detectors, observables = (
                experiment.circuit.without_noise()
                .compile_detector_sampler(seed=731)
                .sample(shots=32, separate_observables=True)
            )
            if detectors.any() or observables.any():
                raise ValueError("Noiseless detector/observable validation failed")
            experiments.append(experiment)
            audit.append(
                dict(
                    name=experiment.name,
                    logical_basis=basis,
                    bell_pair_fidelity=1 - infidelity,
                    surgery_schedule=expected_schedule,
                    bell_pairs_consumed=expected_count
                    if infidelity
                    else args.rounds * meta["num_bell_pair_checks_per_round"],
                    num_bell_source_channels=channel_count,
                    noiseless_validation_shots=32,
                    circuit_sha256=digest(str(experiment.circuit)),
                    dem_sha256=digest(str(experiment.detector_error_model)),
                    non_bell_operations_sha256=digest(str(physical)),
                    num_ticks=experiment.circuit.num_ticks,
                    tick_normalization="Collapse empty consecutive TICKs after removing source noise; idling disabled",
                    num_detectors=meta["num_detectors"],
                    num_dem_errors=meta["num_dem_errors"],
                    build_wall_seconds=time.perf_counter() - started,
                )
            )
            print(
                f"Built {basis}{basis}, F={1 - infidelity:.8g}: "
                f"{meta['num_detectors']} detectors, {meta['num_dem_errors']} DEM errors",
                flush=True,
            )
    if len({e.name for e in experiments}) != len(experiments):
        raise ValueError("Fidelity grid produced colliding artifact names")
    return experiments, audit


def saved_stats(output: Path, experiments: list[BuiltExperiment]) -> list[sinter.TaskStats]:
    if not (output / "sinter_progress.csv").exists():
        return []
    stats = sinter.read_stats_from_csv_files(output / "sinter_progress.csv")
    expected = {task_for(e).strong_id(): e.metadata for e in experiments}
    if any(
        s.strong_id not in expected
        or s.json_metadata != expected[s.strong_id]
        or s.decoder != "qldpc_bp_osd"
        for s in stats
    ):
        raise ValueError("Checkpoint contains a foreign task or changed decoder identity")
    return stats


def check_output(output: Path, experiments: list[BuiltExperiment]) -> None:
    manifest = output / "manifest.json"
    if not manifest.exists():
        if any(p.name != ".collection.lock" for p in output.iterdir()):
            raise ValueError("Nonempty output directory without a matching manifest")
        return
    previous = json.loads(manifest.read_text())
    expected = {e.name: e for e in experiments}
    if len(previous) != len(expected) or {m["name"] for m in previous} != set(expected):
        raise ValueError("Saved fidelity grid/bases differ; choose a new directory")
    for meta in previous:
        experiment = expected[meta["name"]]
        if {
            k: v for k, v in meta.items() if k not in {"circuit_path", "dem_path"}
        } != experiment.metadata:
            raise ValueError("Saved experiment metadata changed")
        for key, content in (
            ("circuit_path", str(experiment.circuit)),
            ("dem_path", str(experiment.detector_error_model)),
        ):
            path = output / meta[key]
            if not path.resolve().is_relative_to(output.resolve()):
                raise ValueError("Artifact path is outside the output directory")
            if path.read_text().strip() != content.strip():
                raise ValueError("Saved circuit/DEM changed")
    saved_stats(output, experiments)


def runtime_identity() -> dict[str, Any]:
    from qldpc.decoders import dems, retrieval
    from qldpc.decoders import sinter as sinter_decoder

    return dict(
        versions={name: version(name) for name in ("stim", "sinter", "ldpc", "numpy")},
        decoder_source_hashes={
            m.__name__: digest(Path(str(m.__file__)).read_text())
            for m in (dems, retrieval, sinter_decoder)
        },
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_analysis(
    output: Path, experiments: list[BuiltExperiment], args: argparse.Namespace
) -> Path:
    stats = saved_stats(output, experiments)
    if len(stats) != len(experiments) or any(s.shots != args.max_shots for s in stats):
        raise ValueError("Analysis requires completed target counts for every condition")
    stage = output / f"analysis_{args.max_shots}"
    stage.mkdir(exist_ok=True)
    write_summary(stats, stage)
    label = (
        f"HGP simplex m={args.simplex_m} / unrotated surface d="
        f"{args.surface_distance or 2 ** (args.simplex_m - 1)}"
        if args.code_pair == "hgp-simplex-surface"
        else "Surface-3 / Steane"
    )
    subtitle = (
        f"Matching-basis; intrinsic hardware-only p={args.physical_error_rate:g}; "
        f"{args.preparation_rounds} ideal preparation rounds; {args.max_shots:,} shots/point.\n"
        "Fidelity-axis spacing is logarithmic in 1-F, with a separate near-ideal linear region."
    )
    write_fidelity_analysis(
        stage / "summary.csv", stage, title=label + " — Bell-fidelity sweep", subtitle=subtitle
    )
    records = [json.loads(line) for line in (output / "invocations.jsonl").read_text().splitlines()]
    batches = [
        json.loads(line) for line in (output / "batch_samples.jsonl").read_text().splitlines()
    ]
    timings = []
    for event in records:
        if event["status"] not in {"completed", "interrupted", "failed"}:
            continue
        current = [b for b in batches if b["invocation_id"] == event["invocation_id"]]
        if sum(b["shots"] for b in current) != event["new_shots"]:
            raise ValueError("Batch telemetry disagrees with checkpointed shots")
        for stat in stats:
            selected = [b for b in current if b["strong_id"] == stat.strong_id]
            shots = sum(b["shots"] for b in selected)
            if not shots:
                continue
            seconds = sum(b["worker_seconds"] for b in selected)
            timings.append(
                dict(
                    invocation_id=event["invocation_id"],
                    logical_basis=stat.json_metadata["logical_basis"],
                    bell_pair_fidelity=stat.json_metadata["bell_pair_fidelity"],
                    cumulative_shot_target=event["config"]["max_shots"],
                    num_workers=event["config"]["num_workers"],
                    new_shots=shots,
                    batch_worker_seconds=seconds,
                    worker_seconds_per_shot=seconds / shots,
                )
            )
    if timings:
        with (stage / "timing.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(timings[0]))
            writer.writeheader()
            writer.writerows(timings)
    completed = [e for e in records if e["status"] == "completed" and e["new_shots"] > 0]
    if completed:
        latest = completed[-1]
        wall = latest["sampling_decode_wall_seconds"]
        remaining = sum(max(0, 20000 - s.shots) for s in stats)
        write_json(
            stage / "timing_summary.json",
            dict(
                latest_invocation_id=latest["invocation_id"],
                num_workers=latest["config"]["num_workers"],
                latest_new_shots=latest["new_shots"],
                latest_shared_collection_wall_seconds=wall,
                remaining_shots_to_20000_per_condition=remaining,
                projected_extension_wall_seconds_at_latest_aggregate_throughput=remaining
                * wall
                / latest["new_shots"],
                caveat="All fidelities/bases shared this wall time. Projection includes startup/tail effects; not a guarantee.",
            ),
        )
    write_json(
        stage / "source_hashes.json",
        {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in (
                "manifest.json",
                "sinter_progress.csv",
                "run_config.json",
                "schedule_audit.json",
            )
        },
    )
    print(f"Plots and tables: {stage}", flush=True)
    return stage


def run(args: argparse.Namespace) -> dict[str, Any]:
    output: Path = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".collection.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another writer owns this sweep directory") from error
        started = time.perf_counter()
        if not (output / "manifest.json").exists() and any(
            p.name != ".collection.lock" for p in output.iterdir()
        ):
            raise ValueError("Nonempty output directory without a matching manifest")
        experiments, audit = build_sweep(args)
        check_output(output, experiments)
        identity = runtime_identity()
        if (output / "run_config.json").exists():
            prior = json.loads((output / "run_config.json").read_text())
            if prior["runtime_identity"] != identity:
                raise ValueError(
                    "Decoder/dependency identity changed; choose a new output directory"
                )
        if args.analyze_only:
            write_analysis(output, experiments, args)
            return {"status": "analyzed"}
        before_stats = saved_stats(output, experiments)
        if any(s.shots > args.max_shots for s in before_stats):
            raise ValueError("Requested cap is below existing counts; no samples may be discarded")
        write_artifacts(experiments, output)
        write_json(output / "schedule_audit.json", audit)
        config = {
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "runtime_identity": identity,
            "schedule_policy": PPM_BASIS_FIRST_VERSION,
            "noise_scope": "intrinsic",
            "adapter_noise_targets": "adapter_hardware",
            "detector_reduction": "inactive_rows",
            "dem_analysis": "undecomposed_exact",
            "include_idling_error": False,
            "compute_exact_distances": False,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        }
        write_json(output / "run_config.json", config)
        event: dict[str, Any] = dict(
            invocation_id=uuid.uuid4().hex,
            started_unix_time=time.time(),
            config=config,
            before=totals(before_stats),
            build_wall_seconds=time.perf_counter() - started,
        )

        def log(status: str) -> None:
            with (output / "invocations.jsonl").open("a") as stream:
                stream.write(json.dumps({**event, "status": status}, sort_keys=True) + "\n")

        log("built" if args.build_only else "sampling_started")
        if args.build_only:
            return {**event, "status": "built"}
        profile = CollectionProfile(output, event["invocation_id"])
        sample_started = time.perf_counter()
        status = "failed"
        try:
            profile.start()
            sample_experiments(
                experiments,
                output_dir=output,
                num_workers=args.num_workers,
                max_shots=args.max_shots,
                max_errors=args.max_shots,
                print_progress=args.verbose_progress,
                progress_callback=profile.progress,
                max_batch_size=args.max_batch_size,
                max_batch_seconds=args.max_batch_seconds,
            )
            status = "completed"
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        finally:
            after_stats = saved_stats(output, experiments)
            if after_stats:
                write_summary(after_stats, output)
            after = totals(after_stats)
            event.update(
                after=after,
                new_shots=after["shots"] - event["before"]["shots"],
                sampling_decode_wall_seconds=time.perf_counter() - sample_started,
                finished_unix_time=time.time(),
                profile=profile.finish(),
            )
            log(status)
        write_analysis(output, experiments, args)
        print(
            f"Finished {args.max_shots:,} shots for each of {len(experiments)} conditions; "
            f"{event['new_shots']:,} new shots in {event['sampling_decode_wall_seconds']:.1f}s.",
            flush=True,
        )
        return {**event, "status": status}


if __name__ == "__main__":
    run(parse_args())
