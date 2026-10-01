"""Checkpointed XX/eight-round sweep at two geometric-midpoint physical probabilities.

Four fresh tasks, one shared worker pool. The timing pilot is included in the
final cumulative cap. Frozen references are never imported as new-task samples.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import math
import os
from pathlib import Path
from typing import Any

import sinter

from . import run_d8_noise_extension as continuation
from . import run_d8_single_point_extension as single
from . import run_physical_noise_comparison as physical
from .run_d4_r8_control import write_comparison_rows
from .run_experiment import BuiltExperiment, write_artifacts
from .run_ppm_fidelity_sweep import (
    check_output,
    digest,
    runtime_identity,
    saved_stats,
    task_for,
    write_json,
)

CONFIGURATIONS = ((3, 4, 8), (4, 8, 8))
PROBABILITIES = (math.sqrt(1e-4 * 3e-4), math.sqrt(3e-4 * 1e-3))
DEFAULT_OUTPUT = physical.OUTPUTS / "ppm_basis_first_hgp_d4_d8_xx_surg8_log_midpoints_100000"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-shots", type=int, default=100000)
    parser.add_argument("--pilot-shots", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-batch-seconds", type=int, default=5)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args(argv)
    if (
        min(
            args.max_shots,
            args.pilot_shots,
            args.num_workers,
            args.max_batch_size,
            args.max_batch_seconds,
        )
        < 1
    ):
        parser.error("Shot, worker, and batch limits must be positive")
    if args.pilot_shots > args.max_shots:
        parser.error("Pilot shots must not exceed the cumulative target")
    args.physical_error_rates = list(PROBABILITIES)
    return args


def freeze_references() -> dict[str, Any]:
    """Validate the entire archive chain and snapshot the stopped partial run."""
    args = single.parse_args([])
    directory = single.DEFAULT_OUTPUT
    with (directory / ".collection.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        experiments, seeds, references, identity, _ = single.load_sources(args)
        current = saved_stats(directory, experiments)
        continuation.validate_counts(current, continuation.target_counts(seeds, 500000))
        # The extension may be deliberately stopped short of its original target.
        current = continuation.read_previous_extension(
            directory,
            experiments,
            seeds,
            identity,
            targets={s.strong_id: s.shots for s in current},
        )
        if current[0].shots < seeds[0].shots:
            raise ValueError("Partial reference has fewer samples than its imported seed")
        references = single.previous.replace_extended_rows(
            references,
            current,
            seeds,
            str(directory.resolve()),
            expected_conditions=1,
        )
        references = [r for r in references if r["surgery_rounds"] == 8]
        expected = {(m, p) for m in (3, 4) for p in (1e-4, 3e-4, 1e-3)}
        if (
            len(references) != 6
            or {(r["simplex_m"], r["physical_error_rate"]) for r in references} != expected
        ):
            raise ValueError("Expected six distinct eight-round reference points")
        return dict(
            captured_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            rows=references,
            partial_checkpoint_sha256=digest((directory / "sinter_progress.csv").read_text()),
            source_identity=identity,
            note="Frozen read-only reference counts; not imported into the new sampling checkpoint.",
        )


def build(args: argparse.Namespace) -> tuple[list[BuiltExperiment], list[dict[str, Any]]]:
    experiments, audits = physical.build_sweep(args, configurations=CONFIGURATIONS)
    baselines, _ = continuation.load_baselines(single.previous.BASELINE)
    original = {
        a["name"]: a for a in json.loads((single.previous.BASELINE / "preflight.json").read_text())
    }
    for experiment, audit in zip(experiments, audits):
        meta = experiment.metadata
        baseline = next(
            e
            for e in baselines
            if e.metadata["left_code"] == meta["left_code"]
            and e.metadata["surgery_rounds"] == 8
            and e.metadata["physical_error_rate"] == 1e-4
        )
        for key in meta.keys() | baseline.metadata.keys():
            if key not in {"name", "physical_error_rate"} and meta.get(
                key
            ) != baseline.metadata.get(key):
                raise ValueError(f"New physical-p condition changed {key}")
        previous = original[baseline.name]
        if audit["scaled_noise_sha256"] != physical.scaled_noise_signature(baseline.circuit, 1e-4):
            raise ValueError("New physical-p condition changed operations/noise locations")
        if audit["decoder_settings"] != previous["decoder_settings"]:
            raise ValueError("Production decoder settings changed")
        # Sinter hashes metadata in insertion order. Canonicalize before first use,
        # matching write_artifacts' JSON ordering and all archived production tasks.
        meta = json.loads(json.dumps(meta, sort_keys=True))
        experiment.metadata.clear()
        experiment.metadata.update(meta)
        audit.update(
            strong_id=task_for(experiment).strong_id(),
            matched_baseline_strong_id=task_for(baseline).strong_id(),
        )
    return experiments, audits


def with_zero_counts(output: Path, experiments: list[BuiltExperiment]) -> list[sinter.TaskStats]:
    current = {s.strong_id: s for s in saved_stats(output, experiments)}
    return [
        current.get(task_for(e).strong_id())
        or sinter.TaskStats(
            strong_id=task_for(e).strong_id(),
            decoder="qldpc_bp_osd",
            json_metadata=e.metadata,
        )
        for e in experiments
    ]


def analyze(
    args: argparse.Namespace,
    experiments: list[BuiltExperiment],
    references: dict[str, Any],
    target: int,
) -> Path:
    stats = saved_stats(args.output_dir, experiments)
    continuation.validate_counts(
        stats, {task_for(e).strong_id(): target for e in experiments}, complete=True
    )
    rows = references["rows"] + [physical.result_row(s, str(args.output_dir)) for s in stats]
    stage = args.output_dir / f"analysis_{target}"
    if stage.exists():
        # Keep earlier snapshots immutable, including a repeated analyze-only call.
        stage = args.output_dir / (
            f"analysis_{target}_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        )
    expected = {(m, 8, p) for m in (3, 4) for p in (1e-4, *PROBABILITIES, 3e-4, 1e-3)}
    kwargs = dict(
        only_rounds=8,
        expected_grid=expected,
        title="HGP simplex / unrotated surface — eight adapter rounds",
    )
    write_comparison_rows(stage, rows, error_bars="exact95", **kwargs)
    jackknife_available = all(
        r["errors"] > 1 and r["errors"] < r["shots"] - r["discards"] for r in rows
    )
    if jackknife_available:
        write_comparison_rows(stage, rows, error_bars="jackknife1sigma", **kwargs)
    remaining_worker_seconds = sum(
        max(0, args.max_shots - s.shots) * s.seconds / s.shots for s in stats
    )
    write_json(stage / "source_comparison_rows.json", rows)
    write_json(
        stage / "timing.json",
        dict(
            per_condition=[physical.result_row(s, "new_midpoint") for s in stats],
            num_workers=args.num_workers,
            remaining_worker_seconds_estimate=remaining_worker_seconds,
            ideal_remaining_wall_seconds_estimate=remaining_worker_seconds / args.num_workers,
            jackknife_plot_available=jackknife_available,
            caveat="Timing projection assumes representative shots and full worker utilization; startup, tails, and laptop sleep add time. Zero/one-failure points use the separate exact95 plot, not zero-width jackknife certainty.",
        ),
    )
    print(
        f"Estimated remaining time at {args.num_workers} workers: "
        f"{remaining_worker_seconds / args.num_workers / 3600:.2f} hours (ideal throughput)",
        flush=True,
    )
    return stage


def run(args: argparse.Namespace) -> None:
    output = args.output_dir = args.output_dir.resolve()
    source_args = single.parse_args([])
    sources = [
        single.DEFAULT_OUTPUT,
        *(r[2] for r in physical.REFERENCES),
        *(v for k, v in vars(source_args).items() if k.endswith("_dir") and k != "output_dir"),
    ]
    if any(
        output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output) for p in sources
    ):
        raise ValueError("Output must be separate from all source datasets")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (output / "manifest.json").exists() and any(
            p.name != ".collection.lock" for p in output.iterdir()
        ):
            raise ValueError("Nonempty output directory without a matching manifest")
        experiments, audits = build(args)
        check_output(output, experiments)
        reference_path = output / "reference_snapshot.json"
        references = (
            json.loads(reference_path.read_text())
            if reference_path.exists()
            else freeze_references()
        )
        identity = dict(
            runtime_identity=runtime_identity(),
            strong_ids=sorted(task_for(e).strong_id() for e in experiments),
            reference_sha256=digest(json.dumps(references, sort_keys=True)),
            targets_by_strong_id={task_for(e).strong_id(): args.max_shots for e in experiments},
            pilot_shots=args.pilot_shots,
        )
        identity_path = output / "identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError("Source/decoder/task/target identity changed")
        elif (output / "manifest.json").exists():
            raise ValueError("Existing run lacks identity; refusing to repair silently")
        if (
            not (output / "sinter_progress.csv").exists()
            and (output / "invocations.jsonl").exists()
        ):
            raise ValueError(
                "Sampling history exists but its checkpoint is missing; refusing to restart from zero"
            )
        before = with_zero_counts(output, experiments)
        continuation.validate_counts(before, identity["targets_by_strong_id"])
        if args.analyze_only:
            if not identity_path.exists():
                raise ValueError("Analyze-only requires an existing run")
            analyze(args, experiments, references, args.max_shots)
            return
        write_artifacts(experiments, output)
        write_json(reference_path, references)
        write_json(identity_path, identity)
        write_json(output / "preflight.json", audits)
        write_json(
            output / "run_config.json",
            {
                **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                **identity,
                "configurations": CONFIGURATIONS,
                "decoder_kwargs": {},
                "threads": {
                    k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
                },
            },
        )
        print(
            f"Exactly four fresh XX/eight-round tasks; {args.num_workers} workers TOTAL; "
            f"{args.max_shots:,} cumulative shots per task",
            flush=True,
        )
        if args.build_only:
            return
        for target in sorted({args.pilot_shots, args.max_shots}):
            current = with_zero_counts(output, experiments)
            if target == args.pilot_shots and all(s.shots >= target for s in current):
                if (
                    all(s.shots == target for s in current)
                    and not (output / f"analysis_{target}").exists()
                ):
                    analyze(args, experiments, references, target)
                continue
            targets = {s.strong_id: max(target, s.shots) for s in current}
            if any(s.shots < targets[s.strong_id] for s in current):
                continuation.collect(args, experiments, current, targets)
            if all(s.shots == target for s in saved_stats(output, experiments)):
                analyze(args, experiments, references, target)
        print("All four targets complete; no automatic further extension.", flush=True)


if __name__ == "__main__":
    run(parse_args())
