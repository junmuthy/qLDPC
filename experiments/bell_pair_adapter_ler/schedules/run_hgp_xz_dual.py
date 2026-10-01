"""Resumable ZZ-only exact-dual pilots; compare with read-only matching-basis data.

Defaults: m=4, surface d=8, four ideal preparation and eight adapter rounds,
256 then 1,000 cumulative ZZ shots. No fresh XX shots or automatic production run.
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
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterator

import matplotlib.pyplot as plt
import numpy as np
import sinter

from ..common.collection_profile import CollectionProfile
from ..common.paths import source_path
from ..common.run_experiment import (
    BuiltExperiment,
    _wilson_interval,
    sample_experiments,
    write_artifacts,
    write_summary,
)
from ..pilots.run_ppm_basis_first import totals
from .dual_timing import write_timing_comparison
from .hgp_xz_dual import (
    POLICY,
    DualBuild,
    build_dual_experiment,
    canonical,
    digest,
    validate_reference,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="Matching-basis manifest/circuit/DEM/checkpoint directory; read-only",
    )
    parser.add_argument("--simplex-m", type=int, default=4)
    parser.add_argument("--surface-distance", type=int)
    parser.add_argument("--left-logical-index", type=int, default=0)
    parser.add_argument("--right-logical-index", type=int, default=0)
    parser.add_argument("--preparation-rounds", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--physical-error-rate", type=float, default=0.001)
    parser.add_argument("--bell-pair-infidelity", type=float, default=0.0)
    parser.add_argument("--max-shots", type=int, default=1000, help="Cumulative ZZ shot cap")
    parser.add_argument(
        "--pilot-shots", type=int, default=256, help="First cumulative checkpoint stage"
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-batch-seconds", type=int, default=5)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--build-only", action="store_true")
    modes.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args(argv)
    if args.simplex_m < 2 or (args.surface_distance is not None and args.surface_distance < 2):
        parser.error("Simplex m and surface distance must be at least two")
    if (
        min(
            args.preparation_rounds,
            args.rounds,
            args.max_shots,
            args.pilot_shots,
            args.num_workers,
            args.max_batch_size,
            args.max_batch_seconds,
        )
        < 1
    ):
        parser.error("Rounds, shots, workers, and batch limits must be positive")
    if min(args.left_logical_index, args.right_logical_index) < 0:
        parser.error("Logical indices must be nonnegative")
    if not all(0 <= p <= 1 for p in (args.physical_error_rate, args.bell_pair_infidelity)):
        parser.error("Noise probabilities must be between zero and one")
    output, reference = args.output_dir.resolve(), args.reference_dir.resolve()
    if output.is_relative_to(reference) or reference.is_relative_to(output):
        parser.error("Output and reference directories must be separate, non-nested directories")
    return args


@contextmanager
def collection_lock(output: Path) -> Iterator[None]:
    """A persistent advisory lock file is harmless; OS releases the lock on exit."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".collection.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another writer owns this output directory") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def task_for(experiment: BuiltExperiment) -> sinter.Task:
    return sinter.Task(
        circuit=experiment.circuit,
        detector_error_model=experiment.detector_error_model,
        decoder="qldpc_bp_osd",
        json_metadata=experiment.metadata,
    )


def saved_stats(output: Path, experiment: BuiltExperiment) -> list[sinter.TaskStats]:
    path = output / "sinter_progress.csv"
    if not path.exists():
        return []
    stats = sinter.read_stats_from_csv_files(path)
    expected = task_for(experiment).strong_id()
    if any(
        s.strong_id != expected
        or s.decoder != "qldpc_bp_osd"
        or canonical(s.json_metadata) != experiment.metadata
        for s in stats
    ):
        raise ValueError("Checkpoint contains a different task; no foreign samples may be imported")
    return stats


def check_output(output: Path, experiment: BuiltExperiment) -> None:
    manifest = output / "manifest.json"
    if not manifest.exists():
        if any(p.name != ".collection.lock" for p in output.iterdir()):
            raise ValueError("Nonempty output directory has no matching manifest")
        return
    entries = json.loads(manifest.read_text())
    if (
        len(entries) != 1
        or canonical({k: v for k, v in entries[0].items() if k not in {"circuit_path", "dem_path"}})
        != experiment.metadata
    ):
        raise ValueError("Output contains a different experiment")
    for key, content in (
        ("circuit_path", str(experiment.circuit)),
        ("dem_path", str(experiment.detector_error_model)),
    ):
        if (output / entries[0][key]).read_text().strip() != content.strip():
            raise ValueError("Saved circuit/DEM changed; choose a new output directory")
    saved_stats(output, experiment)


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _events(path: Path) -> list[dict[str, Any]]:
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_reference_stats(
    reference: Path, built: DualBuild
) -> list[tuple[str, sinter.TaskStats, Path]]:
    entries = validate_reference(reference, built)
    reference_stats = sinter.read_stats_from_csv_files(reference / "sinter_progress.csv")
    series = []
    for basis in ("X", "Z"):
        entry = entries[basis]
        metadata = {k: v for k, v in entry.items() if k not in {"circuit_path", "dem_path"}}
        circuit = built.source if basis == "X" else built.matching_z
        expected = sinter.Task(
            circuit=circuit.circuit,
            detector_error_model=circuit.detector_error_model,
            decoder="qldpc_bp_osd",
            json_metadata=canonical(metadata),
        ).strong_id()
        stats = [s for s in reference_stats if s.strong_id == expected]
        if len(stats) != 1 or canonical(stats[0].json_metadata) != canonical(metadata):
            raise ValueError("Reference checkpoint identity does not match its validated artifact")
        series.append((f"Matching-basis {basis}{basis}\nexisting reference", stats[0], reference))
    return series


def compare(output: Path, reference: Path, built: DualBuild) -> list[dict[str, Any]]:
    series = load_reference_stats(reference, built)
    current = saved_stats(output, built.experiment)
    if len(current) != 1 or current[0].shots <= current[0].discards:
        raise ValueError("No accepted dual pilot samples yet")
    series.append(("Exact-dual ZZ\nnew independent samples", current[0], output))
    rows = []
    for label, stat, directory in series:
        accepted = stat.shots - stat.discards
        if not 0 <= stat.errors <= accepted or accepted < 1:
            raise ValueError("Invalid binomial counts")
        low, high = _wilson_interval(stat.errors, accepted)
        rows.append(
            dict(
                label=label.replace("\n", ": "),
                logical_basis=stat.json_metadata["logical_basis"],
                schedule_policy=stat.json_metadata["schedule_policy"],
                shots=stat.shots,
                errors=stat.errors,
                discards=stat.discards,
                logical_error_rate=stat.errors / accepted,
                ler_95ci_low=low,
                ler_95ci_high=high,
                worker_seconds=stat.seconds,
                worker_seconds_per_shot=stat.seconds / stat.shots,
                source_checkpoint=str(directory / "sinter_progress.csv"),
                strong_id=stat.strong_id,
            )
        )
    _csv(output / "comparison.csv", rows)
    fig, axis = plt.subplots(figsize=(8.4, 5.3))
    y, lows, highs = [
        np.array([r[k] for r in rows]) * 100
        for k in ("logical_error_rate", "ler_95ci_low", "ler_95ci_high")
    ]
    axis.errorbar(range(3), y, yerr=np.maximum([y - lows, highs - y], 0), fmt="o", capsize=5)
    axis.set_xticks(range(3), [label for label, _, _ in series])
    axis.set_ylabel("Adapter LER (%) — pointwise 95% Wilson intervals")
    axis.set_ylim(0, max(highs) * 1.4)
    axis.grid(axis="y", alpha=0.25)
    for i, row in enumerate(rows):
        axis.annotate(
            f"{row['errors']}/{row['shots']:,} failures",
            (i, highs[i]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )
    meta = built.experiment.metadata
    axis.set_title(
        f"{meta['left_code']} / {meta['right_code']}: hardware-only\n"
        f"{meta['preparation_rounds']} ideal preparation + {meta['surgery_rounds']} adapter rounds; "
        f"p={meta['physical_error_rate']:g}, Bell F={meta['bell_pair_fidelity']:g}"
    )
    fig.text(
        0.5,
        0.025,
        "Exact-dual ZZ and source XX have identical ordered detector-error models.\n"
        "Independent counts need not match. Broad pilot intervals do not establish equivalence.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    for extension in ("pdf", "png"):
        fig.savefig(output / f"hgp_xz_dual_comparison.{extension}", dpi=180)
    plt.close(fig)
    timings = []
    for event in _events(output / "invocations.jsonl"):
        if "after" not in event:
            continue
        new = event["new_shots"]
        wall = event["sampling_decode_wall_seconds"]
        profile = event.get("profile", {})
        timings.append(
            dict(
                invocation_id=event["invocation_id"],
                status=event["status"],
                num_workers=event["config"]["num_workers"],
                cumulative_shot_cap=event["shot_cap"],
                new_shots=new,
                collection_wall_seconds=wall,
                new_worker_seconds=event["new_worker_seconds"],
                worker_seconds_per_shot=event["new_worker_seconds"] / new if new else None,
                shots_per_wall_second=new / wall if wall else None,
                sampled_peak_sum_rss_bytes=profile.get("sampled_peak_sum_rss_bytes"),
            )
        )
    if timings:
        _csv(output / "stage_timings.csv", timings)
    write_timing_comparison(output, rows)
    return rows


def collect_stage(
    args: argparse.Namespace, built: DualBuild, cap: int, config: dict[str, Any]
) -> dict[str, Any]:
    output = args.output_dir
    before = totals(saved_stats(output, built.experiment))
    event: dict[str, Any] = dict(
        invocation_id=uuid.uuid4().hex,
        config=config,
        shot_cap=cap,
        before=before,
        started_unix_time=time.time(),
        status="sampling_started",
    )

    def log() -> None:
        with (output / "invocations.jsonl").open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")

    log()
    print(
        f"ZZ: resuming {before['shots']} → {cap} cumulative shots with {args.num_workers} workers",
        flush=True,
    )
    started = time.perf_counter()
    profile = None
    status = "failed"
    try:
        if args.profile:
            profile = CollectionProfile(output, event["invocation_id"])
            profile.start()
        sample_experiments(
            [built.experiment],
            output_dir=output,
            num_workers=args.num_workers,
            max_shots=cap,
            max_errors=cap,
            print_progress=False,
            progress_callback=profile.progress if profile else None,
            max_batch_size=args.max_batch_size,
            max_batch_seconds=args.max_batch_seconds,
        )
        if totals(saved_stats(output, built.experiment))["shots"] != cap:
            raise ValueError("Collection returned without reaching its cumulative shot cap")
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        wall = time.perf_counter() - started
        profile_summary = profile.finish() if profile else None
        stats = saved_stats(output, built.experiment)
        if stats:
            write_summary(stats, output)
        after = totals(stats)
        event.update(
            status=status,
            after=after,
            new_shots=after["shots"] - before["shots"],
            new_worker_seconds=after["worker_seconds"] - before["worker_seconds"],
            sampling_decode_wall_seconds=wall,
            finished_unix_time=time.time(),
        )
        if profile_summary is not None:
            event["profile"] = profile_summary
        log()
    print(
        f"Saved {after['shots']} ZZ shots, {after['errors']} failures; {event['new_shots']} new in {wall:.1f}s",
        flush=True,
    )
    return event


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    # Guard here too for callers that construct a Namespace without parse_args.
    output, reference = args.output_dir.resolve(), args.reference_dir.resolve()
    if output.is_relative_to(reference) or reference.is_relative_to(output):
        raise ValueError("Output and reference must be separate non-nested directories")
    with collection_lock(output):
        started = time.perf_counter()
        built = build_dual_experiment(
            simplex_m=args.simplex_m,
            surface_distance=args.surface_distance,
            left_logical_index=args.left_logical_index,
            right_logical_index=args.right_logical_index,
            preparation_rounds=args.preparation_rounds,
            surgery_rounds=args.rounds,
            physical_error_rate=args.physical_error_rate,
            bell_pair_infidelity=args.bell_pair_infidelity,
        )
        load_reference_stats(reference, built)
        check_output(output, built.experiment)
        if args.analyze_only:
            return compare(output, reference, built)
        write_artifacts([built.experiment], output)
        audit = dict(
            built.audit,
            source_matches_archived_matching_xx=True,
            matching_zz_matches_archive=True,
            reference_directory=str(reference),
            reference_manifest_sha256=digest((reference / "manifest.json").read_bytes()),
            reference_checkpoint_sha256=digest((reference / "sinter_progress.csv").read_bytes()),
        )
        _json(output / "schedule_audit.json", audit)
        config = {
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "schedule_policy": POLICY,
            "build_validation_wall_seconds": time.perf_counter() - started,
            "noise_scope": "intrinsic",
            "adapter_noise_targets": "adapter_hardware",
            "dem_analysis": "undecomposed_exact",
            "compute_exact_distances": False,
            "include_idling_error": False,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {k: version(k) for k in ("stim", "sinter", "ldpc", "numpy")},
            "threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
            "source_sha256": {
                name: digest(source_path(name).read_bytes())
                for name in (
                    "hgp_xz_dual.py",
                    "run_hgp_xz_dual.py",
                    "run_experiment.py",
                    "collection_profile.py",
                    "dual_timing.py",
                )
            },
        }
        _json(output / "run_config.json", config)
        print(
            f"Validated exact-dual ZZ: {built.experiment.circuit.num_detectors} detectors, "
            f"{built.experiment.detector_error_model.num_errors} DEM errors; ordered DEM equals archived XX",
            flush=True,
        )
        if args.build_only:
            return []
        for cap in sorted({min(args.pilot_shots, args.max_shots), args.max_shots}):
            if totals(saved_stats(output, built.experiment))["shots"] >= cap:
                continue
            collect_stage(args, built, cap, config)
            compare(output, reference, built)
        return compare(output, reference, built)


if __name__ == "__main__":
    run(parse_args())
