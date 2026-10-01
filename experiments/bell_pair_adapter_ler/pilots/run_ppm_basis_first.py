"""Hardware-only PPM-basis-first pilots for Surface/Steane and HGP-simplex/surface.

Use the generic build_experiment_for_codes API for other binary stabilizer CSS codes.
This runner never performs exact-distance searches or imports old shot checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import uuid
from importlib.metadata import version
from pathlib import Path
from typing import Any

import sinter

from qldpc import codes
from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli

from ..common.run_experiment import (
    PPM_BASIS_FIRST_VERSION,
    BuiltExperiment,
    _filter_stats_for_experiments,
    build_experiment_for_codes,
    sample_experiments,
    write_artifacts,
    write_summary,
)


def code_pair(
    preset: str, *, simplex_m: int = 3, surface_distance: int | None = None
) -> tuple[CSSCode, CSSCode, str, str]:
    """Construct a preset without computing distances for metadata."""
    if preset == "surface-steane":
        if simplex_m != 3 or surface_distance is not None:
            raise ValueError("simplex_m/surface_distance apply only to hgp-simplex-surface")
        return codes.SurfaceCode(3), codes.SteaneCode(), "rotated_surface_d3", "steane_d3"
    if preset != "hgp-simplex-surface":
        raise ValueError(f"Unknown code pair: {preset}")
    if simplex_m < 2:
        raise ValueError("simplex_m must be at least 2")
    distance = 2 ** (simplex_m - 1) if surface_distance is None else surface_distance
    if distance < 2:
        raise ValueError("surface_distance must be at least 2")
    return (
        codes.HGPCode(codes.SimplexCode(simplex_m)),
        codes.SurfaceCode(distance, rotated=False),
        f"hgp_simplex_m{simplex_m}",
        f"surface_unrotated_d{distance}",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New results or matching resume directory"
    )
    parser.add_argument(
        "--code-pair", choices=("surface-steane", "hgp-simplex-surface"), default="surface-steane"
    )
    parser.add_argument("--simplex-m", type=int, default=3)
    parser.add_argument(
        "--surface-distance", type=int, help="Unrotated surface distance; default 2**(m-1)"
    )
    parser.add_argument("--left-logical-index", type=int, default=0)
    parser.add_argument("--right-logical-index", type=int, default=0)
    parser.add_argument("--preparation-rounds", type=int, default=3)
    parser.add_argument("--rounds", type=int, nargs="+", default=[4])
    parser.add_argument("--bases", choices=("X", "Z"), nargs="+", default=["X", "Z"])
    parser.add_argument("--physical-error-rate", type=float, default=0.001)
    parser.add_argument("--bell-pair-infidelity", type=float, default=0.0)
    parser.add_argument(
        "--max-shots",
        type=int,
        default=1000,
        help="Cumulative shots per condition, including resumes",
    )
    parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Log batch timings and sampled process-tree RSS (requires psutil)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        help="Cap each sampling batch for pilot timing/checkpoint granularity",
    )
    parser.add_argument(
        "--max-batch-seconds",
        type=int,
        help="Target batch/flush interval; cannot interrupt a slow shot",
    )
    parser.add_argument(
        "--verbose-progress",
        action="store_true",
        help="Show Sinter's full metadata progress display",
    )
    args = parser.parse_args(argv)
    if args.max_batch_size is not None and args.max_batch_size < 1:
        parser.error("max-batch-size must be positive")
    if args.max_batch_seconds is not None and args.max_batch_seconds < 1:
        parser.error("max-batch-seconds must be positive")
    if min(args.max_shots, args.num_workers, args.preparation_rounds, *args.rounds) < 1:
        parser.error("Shots, workers, and round counts must be positive")
    if len(set(args.rounds)) != len(args.rounds) or len(set(args.bases)) != len(args.bases):
        parser.error("Round counts and bases must not contain duplicates")
    if min(args.left_logical_index, args.right_logical_index) < 0:
        parser.error("Logical indices must be nonnegative")
    for value in (args.physical_error_rate, args.bell_pair_infidelity):
        if not 0 <= value <= 1:
            parser.error("Error probabilities must be between zero and one")
    return args


def checkpoint_stats(output: Path, experiments: list[BuiltExperiment]) -> list[sinter.TaskStats]:
    path = output / "sinter_progress.csv"
    if not path.exists():
        return []
    return _filter_stats_for_experiments(sinter.read_stats_from_csv_files(path), experiments)


def totals(stats: list[sinter.TaskStats]) -> dict[str, float | int]:
    return {
        "shots": sum(s.shots for s in stats),
        "errors": sum(s.errors for s in stats),
        "worker_seconds": sum(s.seconds for s in stats),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output: Path = args.output_dir
    if output.exists() and any(output.iterdir()) and not (output / "manifest.json").exists():
        raise ValueError("Nonempty output directory without a manifest; choose a new directory")
    started = time.perf_counter()
    started_unix_time = time.time()
    left, right, left_label, right_label = code_pair(
        args.code_pair, simplex_m=args.simplex_m, surface_distance=args.surface_distance
    )
    code_build_seconds = time.perf_counter() - started
    experiments, builds = [], []
    for basis_name in args.bases:
        for rounds in args.rounds:
            build_started = time.perf_counter()
            experiment = build_experiment_for_codes(
                left_code=left,
                right_code=right,
                left_code_name=left_label,
                right_code_name=right_label,
                left_logical_index=args.left_logical_index,
                right_logical_index=args.right_logical_index,
                basis=Pauli[basis_name],
                preparation_rounds=args.preparation_rounds,
                surgery_rounds=rounds,
                physical_error_rate=args.physical_error_rate,
                bell_pair_infidelity=args.bell_pair_infidelity,
                noise_scope="intrinsic",
                include_idling_error=False,
                adapter_noise_targets="adapter_hardware",
                reduce_detectors=True,
                compute_exact_distances=False,
                schedule_policy="ppm_basis_first",
            )
            experiments.append(experiment)
            builds.append(
                {
                    "name": experiment.name,
                    "circuit_dem_build_wall_seconds": time.perf_counter() - build_started,
                }
            )
            meta = experiment.metadata
            print(
                f"{basis_name}{basis_name}, {rounds} rounds: local "
                f"{'→'.join(meta['surgery_schedule']['local_order'])}, Bell X→Z; "
                f"{meta['num_detectors']} detectors, {meta['num_dem_errors']} DEM errors; "
                f"build {builds[-1]['circuit_dem_build_wall_seconds']:.2f}s",
                flush=True,
            )
    # This rejects changes of codes, logical indices, schedule, noise, or rounds on resume.
    # Check canonical circuit/DEM content too: unchanged metadata alone is insufficient
    # if a dependency or code update alters fault propagation without changing counts.
    if (output / "manifest.json").exists():
        previous = {e["name"]: e for e in json.loads((output / "manifest.json").read_text())}
        for experiment in experiments:
            if experiment.name not in previous:
                continue  # write_artifacts rejects the differing metadata below.
            for key, content in (
                ("circuit_path", str(experiment.circuit)),
                ("dem_path", str(experiment.detector_error_model)),
            ):
                if (output / previous[experiment.name][key]).read_text().strip() != content.strip():
                    raise ValueError("Saved circuit/DEM changed; use a new output directory")
    write_artifacts(experiments, output)
    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "schedule_policy": PPM_BASIS_FIRST_VERSION,
        "noise_scope": "intrinsic",
        "adapter_noise_targets": "adapter_hardware",
        "detector_reduction": "inactive_rows",
        "dem_analysis": "undecomposed_exact",
        "include_idling_error": False,
        "compute_exact_distances": False,
        "versions": {k: version(k) for k in ("stim", "sinter", "ldpc", "numpy")},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "threads": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    before = totals(checkpoint_stats(output, experiments))
    invocation: dict[str, Any] = {
        "invocation_id": uuid.uuid4().hex,
        "started_unix_time": started_unix_time,
        "config": config,
        "code_construction_wall_seconds": code_build_seconds,
        "circuit_builds": builds,
        "before": before,
    }

    def log_event(event: dict[str, Any]) -> None:
        with (output / "invocations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")

    log_event({**invocation, "status": "built" if args.build_only else "sampling_started"})
    if args.build_only:
        return invocation
    sample_started = time.perf_counter()
    profile = None
    if args.profile:
        from ..common.collection_profile import CollectionProfile

        profile = CollectionProfile(output, invocation["invocation_id"])
        profile.start()
    status = "failed"
    try:
        sample_experiments(
            experiments,
            output_dir=output,
            num_workers=args.num_workers,
            max_shots=args.max_shots,
            max_errors=args.max_shots,  # Fixed shot budget, not an early error-count cap.
            print_progress=args.verbose_progress,
            progress_callback=profile.progress if profile else None,
            max_batch_size=args.max_batch_size,
            max_batch_seconds=args.max_batch_seconds,
        )
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        sample_wall = time.perf_counter() - sample_started
        # Recover flushed batches after Ctrl-C. An unflushed in-flight batch may be lost.
        saved = checkpoint_stats(output, experiments)
        if saved:
            write_summary(saved, output)
        after = totals(saved)
        invocation.update(
            status=status,
            after=after,
            new_shots=after["shots"] - before["shots"],
            new_worker_seconds=after["worker_seconds"] - before["worker_seconds"],
            sampling_decode_wall_seconds=sample_wall,
            total_invocation_wall_seconds=time.perf_counter() - started,
            finished_unix_time=time.time(),
        )
        if profile is not None:
            invocation["profile"] = profile.finish()
        log_event(invocation)
    print(
        f"Saved {after['shots']} cumulative shots; {invocation['new_shots']} new shots in "
        f"{sample_wall:.2f}s sampling/decoding wall time. Results: {output}",
        flush=True,
    )
    return invocation


if __name__ == "__main__":
    run(parse_args())
