"""Add shots only to d=8/eight-round XX at p=1e-4 and 3e-4.

Seeds are fixed archived totals (300k and 100k), not the live continuation counts.
The default 200k additional shots therefore always means targets of 500k and 300k,
including on resume. Original datasets and the older continuation are read-only.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import sinter

from qldpc import decoders

from . import run_low_p_extension as previous
from .collection_profile import CollectionProfile
from .run_d4_r8_control import load_baselines, write_comparison_rows
from .run_experiment import BuiltExperiment, sample_experiments, write_artifacts, write_summary
from .run_physical_noise_comparison import OUTPUTS, REFERENCES, scaled_noise_signature
from .run_ppm_basis_first import totals
from .run_ppm_fidelity_sweep import (
    check_output,
    digest,
    runtime_identity,
    saved_stats,
    task_for,
    write_json,
)

DEFAULT_OUTPUT = OUTPUTS / "ppm_basis_first_hgp_d8_xx_p1e4_3e4_surg8_add200000"
SourceData = tuple[
    list[BuiltExperiment],
    list[sinter.TaskStats],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-dir", type=Path, default=previous.BASELINE)
    parser.add_argument("--high-p-dir", type=Path, default=previous.HIGH_P_CONTROL)
    parser.add_argument("--previous-extension-dir", type=Path, default=previous.DEFAULT_OUTPUT)
    parser.add_argument(
        "--additional-shots",
        type=int,
        default=200000,
        help="Additional to the fixed imported seeds, NOT additional on every restart",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-batch-seconds", type=int, default=5)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args(argv)
    if (
        min(args.additional_shots, args.num_workers, args.max_batch_size, args.max_batch_seconds)
        < 1
    ):
        parser.error("Shot, worker, and batch limits must be positive")
    return args


def select_experiments(experiments: list[BuiltExperiment]) -> list[BuiltExperiment]:
    selected = [
        e
        for e in experiments
        if (
            e.metadata["left_code"] == "hgp_simplex_m4"
            and e.metadata["surgery_rounds"] == 8
            and e.metadata["physical_error_rate"] in (1e-4, 3e-4)
        )
    ]
    if len(selected) != 2 or {e.metadata["physical_error_rate"] for e in selected} != {1e-4, 3e-4}:
        raise ValueError("Expected exactly the d=8/eight-round p=1e-4 and 3e-4 conditions")
    expected = dict(
        right_code="surface_unrotated_d8",
        logical_basis="X",
        preparation_rounds=4,
        bell_pair_infidelity=0.0,
        bell_pair_fidelity=1.0,
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
    return sorted(selected, key=lambda e: e.metadata["physical_error_rate"])


def read_previous_extension(
    directory: Path,
    experiments: list[BuiltExperiment],
    seeds: list[sinter.TaskStats],
    identity: dict[str, Any],
    *,
    targets: dict[str, int] | None = None,
) -> list[sinter.TaskStats]:
    """Validate the prior continuation without creating or repairing any source files."""
    if json.loads((directory / "identity.json").read_text()) != identity:
        raise ValueError("Previous extension source/decoder identity changed")
    check_output(directory, experiments)
    receipt = dict(
        seed_csv_sha256=digest(previous.seed_text(seeds)),
        seed_counts={s.strong_id: dict(shots=s.shots, errors=s.errors) for s in seeds},
    )
    if json.loads((directory / "seed_receipt.json").read_text()) != receipt:
        raise ValueError("Previous extension seed receipt changed")
    if not (directory / "sinter_progress.csv").read_text().startswith(previous.seed_text(seeds)):
        raise ValueError("Previous extension seed prefix changed")
    references = json.loads((directory / "reference_rows.json").read_text())
    if digest(json.dumps(references, sort_keys=True)) != identity["reference_rows_sha256"]:
        raise ValueError("Previous extension reference rows changed")
    stats = saved_stats(directory, experiments)
    if targets is not None:
        validate_counts(stats, targets, complete=True)
    elif len(stats) != 2 or any(s.shots != 300000 for s in stats):
        raise ValueError("Expected the completed 300,000-shot prior extension")
    return stats


def load_sources(
    args: argparse.Namespace,
) -> tuple[
    list[BuiltExperiment],
    list[sinter.TaskStats],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    # Reuse the original provenance checks, including the read-only high-p references.
    old_experiments, old_seeds, references, old_identity, _ = previous.load_sources(args)
    prior = read_previous_extension(
        args.previous_extension_dir, old_experiments, old_seeds, old_identity
    )
    references = previous.replace_extended_rows(
        references, prior, old_seeds, str(args.previous_extension_dir.resolve())
    )  # Retain the latest d=4 300k result as well as d=8 300k.
    baselines, _ = load_baselines(args.baseline_dir)
    selected = select_experiments(baselines)
    original = {
        a["name"]: a for a in json.loads((args.baseline_dir / "preflight.json").read_text())
    }
    baseline_stats = {s.strong_id: s for s in saved_stats(args.baseline_dir, baselines)}
    prior_stats = {s.strong_id: s for s in prior}
    audits, seeds, signatures = [], [], set()
    for e in selected:
        key = task_for(e).strong_id()
        p = e.metadata["physical_error_rate"]
        seed = prior_stats[key] if p == 1e-4 else baseline_stats[key]
        if seed.shots != (300000 if p == 1e-4 else 100000):
            raise ValueError("Unexpected seed count")
        seeds.append(seed)
        audit = original[e.name]
        if (
            digest(str(e.circuit)) != audit["circuit_sha256"]
            or digest(str(e.detector_error_model)) != audit["dem_sha256"]
        ):
            raise ValueError("Archived circuit/DEM differs from preflight")
        signatures.add(scaled_noise_signature(e.circuit, p))
        det, obs = (
            e.circuit.without_noise()
            .compile_detector_sampler(seed=731)
            .sample(shots=32, separate_observables=True)
        )
        if det.any() or obs.any() or e.circuit.num_observables != 1:
            raise ValueError("Noiseless XX validation failed")
        compiled = decoders.SinterDecoder().compile_decoder_for_dem(e.detector_error_model)
        settings = {k: getattr(compiled.decoder, k) for k in audit["decoder_settings"]}
        if settings != audit["decoder_settings"] or settings["osd_method"] != "OSD_0":
            raise ValueError("Production decoder settings changed")
        audits.append(
            {**audit, "strong_id": key, "decoder_settings": settings, "seed_shots": seed.shots}
        )
        print(
            f"Validated d=8 XX, eight rounds, p={p:g}: {seed.shots:,} seed shots, {seed.errors} failures",
            flush=True,
        )
    if len(signatures) != 1:
        raise ValueError("The two physical-p circuits changed operations or noise locations")
    files = [
        args.previous_extension_dir / name
        for name in (
            "sinter_progress.csv",
            "identity.json",
            "seed_receipt.json",
            "reference_rows.json",
        )
    ]
    identity = dict(
        runtime_identity=runtime_identity(),
        selected_strong_ids=sorted(s.strong_id for s in seeds),
        seed_csv_sha256=digest(previous.seed_text(seeds)),
        reference_rows_sha256=digest(json.dumps(references, sort_keys=True)),
        previous_extension_identity=old_identity,
        source_files_sha256={str(p.resolve()): digest(p.read_text()) for p in files},
    )
    return selected, seeds, references, identity, audits


def target_counts(seeds: list[sinter.TaskStats], additional: int) -> dict[str, int]:
    if additional < 1 or not seeds or len({s.strong_id for s in seeds}) != len(seeds):
        raise ValueError("Need distinct seeds and positive additional shots")
    return {s.strong_id: s.shots + additional for s in seeds}


def validate_counts(
    stats: list[sinter.TaskStats], targets: dict[str, int], *, complete: bool = False
) -> None:
    if {s.strong_id for s in stats} != set(targets) or len(stats) != len(targets):
        raise ValueError("Checkpoint does not contain exactly the selected tasks")
    if any(s.shots > targets[s.strong_id] for s in stats):
        raise ValueError("Target is below saved counts; refusing to discard data")
    if complete and any(s.shots != targets[s.strong_id] for s in stats):
        raise ValueError("All per-condition cumulative targets must be complete")


def collect(
    args: argparse.Namespace,
    experiments: list[BuiltExperiment],
    before: list[sinter.TaskStats],
    targets: dict[str, int],
) -> None:
    output = args.output_dir
    event: dict[str, Any] = dict(
        invocation_id=uuid.uuid4().hex,
        targets_by_strong_id=targets,
        num_workers=args.num_workers,
        before=totals(before),
        started_unix_time=time.time(),
    )
    profile = CollectionProfile(output, event["invocation_id"])
    counts = {s.strong_id: s.shots for s in before}
    labels = {
        task_for(e).strong_id(): (
            f"{e.metadata['right_code']} XX p={e.metadata['physical_error_rate']:g}"
        )
        for e in experiments
    }
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
                "; ".join(f"{labels[k]}: {counts[k]:,}/{targets[k]:,}" for k in labels), flush=True
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
            max_shots=max(targets.values()),
            max_shots_by_strong_id=targets,
            max_errors=max(targets.values()),
            print_progress=False,
            progress_callback=progress,
            max_batch_size=args.max_batch_size,
            max_batch_seconds=args.max_batch_seconds,
        )
        validate_counts(saved_stats(output, experiments), targets, complete=True)
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


def analyze(
    args: argparse.Namespace,
    experiments: list[BuiltExperiment],
    seeds: list[sinter.TaskStats],
    references: list[dict[str, Any]],
    targets: dict[str, int],
) -> None:
    stats = saved_stats(args.output_dir, experiments)
    validate_counts(stats, targets, complete=True)
    rows = previous.replace_extended_rows(
        references, stats, seeds, str(args.output_dir), expected_conditions=len(experiments)
    )
    stage = args.output_dir / f"analysis_add{args.additional_shots}"
    for error_bars in ("jackknife1sigma", "exact95"):
        write_comparison_rows(
            stage,
            rows,
            error_bars=error_bars,
            only_rounds=8,
            title="HGP simplex / unrotated surface — eight adapter rounds",
            show_fit_caption=False,
        )
    write_json(stage / "source_comparison_rows.json", rows)
    events_path = args.output_dir / "invocations.jsonl"
    events = (
        [json.loads(line) for line in events_path.read_text().splitlines()]
        if events_path.exists()
        else []
    )
    write_json(
        stage / "timing.json",
        dict(
            targets_by_strong_id=targets,
            extended_conditions=[r for r in rows if r["strong_id"] in targets],
            finished_invocations=[e for e in events if e["status"] != "sampling_started"],
            caveat="New worker seconds exclude imported data; wall time includes startup and tails.",
        ),
    )


def run(
    args: argparse.Namespace,
    *,
    source_loader: Callable[[argparse.Namespace], SourceData] | None = None,
    extra_source_directories: tuple[Path, ...] = (),
    analysis_writer: Callable[..., None] | None = None,
) -> None:
    args.output_dir = output = args.output_dir.resolve()
    for directory in [
        args.baseline_dir,
        args.high_p_dir,
        args.previous_extension_dir,
        *(r[2] for r in REFERENCES),
        *extra_source_directories,
    ]:
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
        experiments, seeds, references, identity, audits = (source_loader or load_sources)(args)
        check_output(output, experiments)
        identity_path = output / "identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError("Source/decoder/task identity changed")
        elif (output / "sinter_progress.csv").exists():
            raise ValueError("Existing checkpoint lacks extension provenance")
        targets = target_counts(seeds, args.additional_shots)
        current = saved_stats(output, experiments)
        if current:
            validate_counts(current, targets)
        if args.analyze_only:
            if not identity_path.exists() or not (output / "seed_receipt.json").exists():
                raise ValueError("Analyze-only requires a previously seeded checkpoint")
            previous.ensure_seeded_checkpoint(output, experiments, seeds)
            (analysis_writer or analyze)(args, experiments, seeds, references, targets)
            return
        write_artifacts(experiments, output)
        write_json(identity_path, identity)
        before = previous.ensure_seeded_checkpoint(output, experiments, seeds)
        validate_counts(before, targets)
        write_json(output / "preflight.json", audits)
        write_json(output / "reference_rows.json", references)
        write_json(
            output / "run_config.json",
            {
                **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "runtime_identity": runtime_identity(),
                "targets_by_strong_id": targets,
                "seed_shots_by_strong_id": {s.strong_id: s.shots for s in seeds},
                "decoder_kwargs": {},
                "threads": {
                    k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
                },
            },
        )
        for s in before:
            print(
                f"{s.json_metadata['right_code']} XX p={s.json_metadata['physical_error_rate']:g}: "
                f"{s.shots:,} saved / {targets[s.strong_id]:,} target",
                flush=True,
            )
        print(
            f"{args.num_workers} workers TOTAL; only {len(experiments)} selected XX task(s) will be sampled.",
            flush=True,
        )
        if args.build_only:
            return
        if any(s.shots < targets[s.strong_id] for s in before):
            collect(args, experiments, before, targets)
        (analysis_writer or analyze)(args, experiments, seeds, references, targets)
        print("All targets complete. No automatic further extension.", flush=True)


if __name__ == "__main__":
    run(parse_args())
