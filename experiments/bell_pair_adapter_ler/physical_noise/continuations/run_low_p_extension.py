"""Extend only the two eight-round XX p=1e-4 conditions from 100k to 200k shots.

Seed a NEW checkpoint once from validated original aggregates. Never modify the
source datasets, resample other conditions, or add the same seed twice on resume.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import sinter

from qldpc import decoders

from ...bell_fidelity.run_ppm_fidelity_sweep import (
    check_output,
    digest,
    runtime_identity,
    saved_stats,
    task_for,
    write_json,
)
from ...common.collection_profile import CollectionProfile
from ...common.run_experiment import (
    BuiltExperiment,
    sample_experiments,
    write_artifacts,
    write_summary,
)
from ...pilots.run_ppm_basis_first import totals
from ..run_d4_r8_control import (
    BASELINE,
    build_control,
    load_baselines,
    write_comparison_rows,
)
from ..run_d4_r8_control import (
    DEFAULT_OUTPUT as HIGH_P_CONTROL,
)
from ..run_physical_noise_comparison import OUTPUTS, REFERENCES, result_row

DEFAULT_OUTPUT = OUTPUTS / "ppm_basis_first_hgp_d4_d8_xx_p1e4_surg8_extension_200000"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE)
    parser.add_argument("--high-p-dir", type=Path, default=HIGH_P_CONTROL)
    parser.add_argument("--max-shots", type=int, default=200000, help="Cumulative shots/point")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-batch-seconds", type=int, default=5)
    fit_mode = parser.add_mutually_exclusive_group()
    fit_mode.add_argument(
        "--log-log-fits",
        action="store_true",
        help="Write separately named plots with jackknife-weighted log-log fits for each series",
    )
    fit_mode.add_argument(
        "--corrected-log-log-fits",
        action="store_true",
        help="Illustrative three-parameter A*p**w*(1+b*p) fits, with jackknife weights",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--build-only", action="store_true", help="Validate and seed without sampling"
    )
    mode.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args(argv)
    if min(args.max_shots, args.num_workers, args.max_batch_size, args.max_batch_seconds) < 1:
        parser.error("Shot, worker, and batch limits must be positive")
    return args


def select_experiments(experiments: list[BuiltExperiment]) -> list[BuiltExperiment]:
    selected = [
        e
        for e in experiments
        if e.metadata["surgery_rounds"] == 8 and e.metadata["physical_error_rate"] == 1e-4
    ]
    if len(selected) != 2 or {e.metadata["left_code"] for e in selected} != {
        "hgp_simplex_m3",
        "hgp_simplex_m4",
    }:
        raise ValueError("Expected exactly the two m=3/m=4 eight-round p=1e-4 conditions")
    expected = dict(
        logical_basis="X",
        preparation_rounds=4,
        surgery_rounds=8,
        physical_error_rate=1e-4,
        bell_pair_infidelity=0.0,
        noise_scope="intrinsic",
        adapter_noise_targets="adapter_hardware",
        detector_reduction="inactive_rows",
        include_idling_error=False,
        left_logical_index=0,
        right_logical_index=0,
        schedule_policy="ppm_basis_first_local_v1",
        decoder="qldpc_bp_osd",
        surgery_schedule=dict(
            local_order=["X", "Z"], bell_order=["X", "Z"], local_layers_sha256=None
        ),
    )
    for e in selected:
        for key, value in expected.items():
            if e.metadata.get(key) != value:
                raise ValueError(f"Selected source changed {key}")
        d = 4 if e.metadata["left_code"] == "hgp_simplex_m3" else 8
        if e.metadata["right_code"] != f"surface_unrotated_d{d}":
            raise ValueError("Selected source has the wrong surface block")
    return sorted(selected, key=lambda e: e.metadata["left_code"])


def seed_text(seeds: list[sinter.TaskStats]) -> str:
    return (
        sinter.CSV_HEADER
        + "\n"
        + "".join(s.to_csv_line() + "\n" for s in sorted(seeds, key=lambda s: s.strong_id))
    )


def load_sources(
    args: argparse.Namespace,
) -> tuple[
    list[BuiltExperiment],
    list[sinter.TaskStats],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    baselines, references = load_baselines(args.baseline_dir)
    selected = select_experiments(baselines)
    ids = {task_for(e).strong_id() for e in selected}
    seeds = [s for s in saved_stats(args.baseline_dir, baselines) if s.strong_id in ids]
    if len(seeds) != 2 or any(s.shots != 100000 for s in seeds):
        raise ValueError("Expected exactly 100,000 seed shots per selected point")
    original_audits = {
        a["name"]: a for a in json.loads((args.baseline_dir / "preflight.json").read_text())
    }
    audits = []
    for experiment in selected:
        original = original_audits[experiment.name]
        if digest(str(experiment.circuit)) != original["circuit_sha256"] or (
            digest(str(experiment.detector_error_model)) != original["dem_sha256"]
        ):
            raise ValueError("Saved source circuit/DEM changed since preflight")
        detectors, observables = (
            experiment.circuit.without_noise()
            .compile_detector_sampler(seed=731)
            .sample(shots=32, separate_observables=True)
        )
        if detectors.any() or observables.any() or experiment.circuit.num_observables != 1:
            raise ValueError("Noiseless XX validation failed")
        compiled = decoders.SinterDecoder().compile_decoder_for_dem(experiment.detector_error_model)
        settings = {key: getattr(compiled.decoder, key) for key in original["decoder_settings"]}
        if settings != original["decoder_settings"] or settings["osd_order"] != 0:
            raise ValueError("Production decoder settings changed")
        audits.append(
            dict(
                name=experiment.name,
                strong_id=task_for(experiment).strong_id(),
                decoder_settings=settings,
                noiseless_validation_shots=32,
                circuit_sha256=original["circuit_sha256"],
                dem_sha256=original["dem_sha256"],
            )
        )
        print(
            f"Validated archived XX {experiment.metadata['left_code']}, eight rounds, "
            f"p=0.0001; decoder {settings}",
            flush=True,
        )
    # The ninth comparison point is read-only and is never sent to the sampler.
    print("Checking the read-only p=0.001 reference below; it will NOT be sampled.", flush=True)
    control, _ = build_control(baselines)
    check_output(args.high_p_dir, [control])
    control_identity = json.loads((args.high_p_dir / "identity.json").read_text())
    if control_identity["runtime_identity"] != runtime_identity() or (
        control_identity["strong_id"] != task_for(control).strong_id()
    ):
        raise ValueError("High-p control decoder/task identity changed")
    control_stats = saved_stats(args.high_p_dir, [control])
    if len(control_stats) != 1 or control_stats[0].shots != 20000:
        raise ValueError("Expected the completed 20,000-shot high-p control")
    references.append(result_row(control_stats[0], str(args.high_p_dir.resolve())))
    identity = dict(
        runtime_identity=runtime_identity(),
        source_directories=[str(args.baseline_dir.resolve()), str(args.high_p_dir.resolve())],
        source_checkpoints_sha256={
            str(p.resolve()): digest(p.read_text())
            for p in (
                args.baseline_dir / "sinter_progress.csv",
                args.high_p_dir / "sinter_progress.csv",
            )
        },
        selected_strong_ids=sorted(ids),
        seed_csv_sha256=digest(seed_text(seeds)),
        reference_rows_sha256=digest(json.dumps(references, sort_keys=True)),
    )
    return selected, seeds, references, identity, audits


def ensure_seeded_checkpoint(
    output: Path,
    experiments: list[BuiltExperiment],
    seeds: list[sinter.TaskStats],
) -> list[sinter.TaskStats]:
    """Install an atomic seed once; a missing committed checkpoint is NOT reseeded."""
    path = output / "sinter_progress.csv"
    receipt_path = output / "seed_receipt.json"
    text = seed_text(seeds)
    receipt = dict(
        seed_csv_sha256=digest(text),
        seed_counts={s.strong_id: dict(shots=s.shots, errors=s.errors) for s in seeds},
    )
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise ValueError("Seed receipt changed")
    if not path.exists():
        if receipt_path.exists() or (output / "invocations.jsonl").exists():
            raise ValueError("Previously initialized checkpoint is missing; refusing to reseed")
        temporary = output / ".seed_checkpoint.tmp"
        with temporary.open("w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    # Collection only appends to the seeded prefix. This catches partial or foreign imports.
    with path.open() as stream:
        if stream.read(len(text)) != text:
            raise ValueError("Checkpoint seed prefix differs; refusing to overwrite data")
        if not receipt_path.exists() and stream.read(1):
            raise ValueError("Checkpoint has new samples but no seed receipt")
    if not receipt_path.exists():
        write_json(receipt_path, receipt)  # Recover a crash after seed rename, before receipt.
    stats = saved_stats(output, experiments)
    by_id = {s.strong_id: s for s in stats}
    if set(by_id) != {s.strong_id for s in seeds}:
        raise ValueError("Checkpoint is missing a selected condition")
    for seed in seeds:
        current = by_id[seed.strong_id]
        if current.shots < seed.shots or current.errors < seed.errors:
            raise ValueError("Checkpoint regressed below seeded counts")
    return stats


def replace_extended_rows(
    references: list[dict[str, Any]],
    stats: list[sinter.TaskStats],
    seeds: list[sinter.TaskStats],
    source: str,
    *,
    expected_conditions: int = 2,
) -> list[dict[str, Any]]:
    extended = {s.strong_id: s for s in stats}
    seed_by_id = {s.strong_id: s for s in seeds}
    if (
        expected_conditions < 1
        or set(extended) != set(seed_by_id)
        or len(extended) != expected_conditions
        or len(stats) != expected_conditions
        or len(seeds) != expected_conditions
    ):
        raise ValueError(
            f"Comparison needs exactly {expected_conditions} distinct extended conditions"
        )
    if not set(extended) <= {r["strong_id"] for r in references}:
        raise ValueError("An extended task is absent from the comparison")
    rows = []
    for reference in references:
        strong_id = reference["strong_id"]
        if strong_id not in extended:
            rows.append(dict(reference))
            continue
        stat, seed = extended[strong_id], seed_by_id[strong_id]
        row = result_row(stat, source)
        row.update(
            seed_shots=seed.shots,
            seed_errors=seed.errors,
            additional_shots=stat.shots - seed.shots,
            additional_errors=stat.errors - seed.errors,
            additional_worker_seconds=stat.seconds - seed.seconds,
        )
        rows.append(row)  # Replace the old estimate; do not add the seed a second time.
    return rows


def analyze(
    output: Path,
    experiments: list[BuiltExperiment],
    seeds: list[sinter.TaskStats],
    references: list[dict[str, Any]],
    target: int,
    *,
    log_log_fits: bool = False,
    corrected_log_log_fits: bool = False,
) -> None:
    stats = saved_stats(output, experiments)
    if len(stats) != 2 or any(s.shots != target for s in stats):
        raise ValueError("Analysis requires both completed target counts")
    rows = replace_extended_rows(references, stats, seeds, str(output.resolve()))
    stage = output / f"analysis_{target}"
    for convention in ("exact95", "jackknife1sigma"):
        write_comparison_rows(
            stage,
            rows,
            error_bars=convention,
            log_log_fits=log_log_fits,
            corrected_log_log_fits=corrected_log_log_fits,
        )
    events_path = output / "invocations.jsonl"
    events = (
        [json.loads(line) for line in events_path.read_text().splitlines()]
        if events_path.exists()
        else []
    )
    write_json(
        stage / "timing.json",
        dict(
            per_extended_condition=[r for r in rows if "additional_shots" in r],
            finished_invocations=[
                e for e in events if e["status"] in {"completed", "interrupted", "failed"}
            ],
            caveat="Additional worker seconds exclude the seed. Wall time includes pool startup/tails; hard-killed invocation wall time is unavailable.",
        ),
    )


def collect(
    args: argparse.Namespace,
    experiments: list[BuiltExperiment],
    before: list[sinter.TaskStats],
) -> None:
    output = args.output_dir
    event: dict[str, Any] = dict(
        invocation_id=uuid.uuid4().hex,
        target_per_condition=args.max_shots,
        num_workers=args.num_workers,
        before=totals(before),
        started_unix_time=time.time(),
    )
    profile = CollectionProfile(output, event["invocation_id"])
    counts = {s.strong_id: s.shots for s in before}
    labels = {task_for(e).strong_id(): e.metadata["left_code"] for e in experiments}
    last_print = 0.0

    def log(status: str) -> None:
        with (output / "invocations.jsonl").open("a") as stream:
            stream.write(json.dumps({**event, "status": status}, sort_keys=True) + "\n")

    def progress(update: sinter.Progress) -> None:
        nonlocal last_print
        profile.progress(update)
        for stat in update.new_stats:
            counts[stat.strong_id] += stat.shots
        if time.perf_counter() - last_print >= 30:
            print(
                "; ".join(f"{labels[key]}: {counts[key]:,}/{args.max_shots:,}" for key in labels),
                flush=True,
            )
            last_print = time.perf_counter()

    started, status = time.perf_counter(), "failed"
    log("sampling_started")
    try:
        profile.start()
        sample_experiments(
            experiments,
            output_dir=output,
            num_workers=args.num_workers,
            max_shots=args.max_shots,
            max_errors=args.max_shots,
            print_progress=False,
            progress_callback=progress,
            max_batch_size=args.max_batch_size,
            max_batch_seconds=args.max_batch_seconds,
        )
        stats = saved_stats(output, experiments)
        if len(stats) != 2 or any(s.shots != args.max_shots for s in stats):
            raise ValueError("Collection ended before both cumulative targets were reached")
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        stats = saved_stats(output, experiments)
        write_summary(stats, output)
        after = totals(stats)
        event.update(
            after=after,
            new_shots=after["shots"] - event["before"]["shots"],
            new_worker_seconds=after["worker_seconds"] - event["before"]["worker_seconds"],
            sampling_decode_wall_seconds=time.perf_counter() - started,
            finished_unix_time=time.time(),
            profile=profile.finish(),
        )
        log(status)
    print(
        f"Collected {event['new_shots']:,} additional shots in "
        f"{event['sampling_decode_wall_seconds']:.1f}s.",
        flush=True,
    )


def run(args: argparse.Namespace) -> None:
    args.output_dir = output = args.output_dir.resolve()
    for directory in [args.baseline_dir, args.high_p_dir, *(ref[2] for ref in REFERENCES)]:
        directory = directory.resolve()
        if output.is_relative_to(directory) or directory.is_relative_to(output):
            raise ValueError("Extension output must be separate from all source datasets")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".collection.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another writer owns the extension directory") from error
        if not (output / "manifest.json").exists() and any(
            p.name != ".collection.lock" for p in output.iterdir()
        ):
            raise ValueError("Nonempty output directory without a matching manifest")
        experiments, seeds, references, identity, audits = load_sources(args)
        check_output(output, experiments)
        identity_path = output / "identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError("Source/decoder/task identity changed")
        elif (output / "sinter_progress.csv").exists():
            raise ValueError("Existing checkpoint lacks extension provenance")
        if args.max_shots < max(s.shots for s in seeds):
            raise ValueError("Target is below the original seed counts")
        current = saved_stats(output, experiments)
        if any(s.shots > args.max_shots for s in current):
            raise ValueError("Target is below saved counts; refusing to discard data")
        if args.analyze_only:
            if not identity_path.exists() or not (output / "seed_receipt.json").exists():
                raise ValueError("Analyze-only requires a previously seeded checkpoint")
            ensure_seeded_checkpoint(output, experiments, seeds)
            analyze(
                output,
                experiments,
                seeds,
                references,
                args.max_shots,
                log_log_fits=args.log_log_fits,
                corrected_log_log_fits=args.corrected_log_log_fits,
            )
            return
        write_artifacts(experiments, output)
        write_json(identity_path, identity)
        before = ensure_seeded_checkpoint(output, experiments, seeds)
        write_json(output / "preflight.json", audits)
        write_json(output / "reference_rows.json", references)
        write_json(
            output / "run_config.json",
            {
                **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "runtime_identity": runtime_identity(),
                "selected_strong_ids": identity["selected_strong_ids"],
                "decoder_kwargs": {},
                "seed_shots_per_condition": {s.strong_id: s.shots for s in seeds},
                "threads": {
                    k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
                },
            },
        )
        print(
            f"Validated {len(experiments)} XX tasks, {totals(before)['shots']:,} retained shots; "
            f"target {args.max_shots:,} EACH, {args.num_workers} workers TOTAL.",
            flush=True,
        )
        if args.build_only:
            return
        if any(s.shots < args.max_shots for s in before):
            collect(args, experiments, before)
        analyze(
            output,
            experiments,
            seeds,
            references,
            args.max_shots,
            log_log_fits=args.log_log_fits,
            corrected_log_log_fits=args.corrected_log_log_fits,
        )
        print(
            "Both targets complete. No automatic extension beyond the requested total.", flush=True
        )


if __name__ == "__main__":
    run(parse_args())
