"""Add 100k shots ONLY to d4/XX/eight rounds at sqrt(1e-4 * 3e-4).

Resume against a fixed 100k-shot seed, never the current accumulated count.
The four-condition source sweep and all existing plots remain read-only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
from pathlib import Path
from typing import Any

import sinter

from . import run_d8_noise_extension as continuation
from . import run_log_midpoint_sweep as midpoint
from .run_experiment import BuiltExperiment
from .run_ppm_fidelity_sweep import (
    check_output,
    digest,
    runtime_identity,
    saved_stats,
    task_for,
    write_json,
)

DEFAULT_OUTPUT = (
    midpoint.physical.OUTPUTS / "ppm_basis_first_hgp_d4_xx_surg8_low_midpoint_add100000"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-dir", type=Path, default=midpoint.DEFAULT_OUTPUT)
    parser.add_argument("--additional-shots", type=int, default=100000)
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
    # The common continuation runner also protects the older source directories.
    old = continuation.parse_args([])
    for name in ("baseline_dir", "high_p_dir", "previous_extension_dir"):
        setattr(args, name, getattr(old, name))
    return args


def select_experiment(experiments: list[BuiltExperiment]) -> BuiltExperiment:
    selected = [
        e
        for e in experiments
        if e.metadata["left_code"] == "hgp_simplex_m3"
        and e.metadata["physical_error_rate"] == midpoint.PROBABILITIES[0]
        and e.metadata["surgery_rounds"] == 8
    ]
    if len(selected) != 1:
        raise ValueError("Expected exactly one d4/lower-midpoint/eight-round experiment")
    expected = dict(
        right_code="surface_unrotated_d4",
        logical_basis="X",
        preparation_rounds=4,
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
    for key, value in expected.items():
        if selected[0].metadata.get(key) != value:
            raise ValueError(f"Selected experiment changed {key}")
    return selected[0]


def load_sources(args: argparse.Namespace) -> continuation.SourceData:
    source = args.source_dir.resolve()
    with (source / ".collection.lock").open("r") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        experiments, audits = midpoint.build(midpoint.parse_args([]))
        check_output(source, experiments)
        original = json.loads((source / "identity.json").read_text())
        snapshot = json.loads((source / "reference_snapshot.json").read_text())
        targets = {task_for(e).strong_id(): 100000 for e in experiments}
        if original != dict(
            runtime_identity=runtime_identity(),
            strong_ids=sorted(targets),
            reference_sha256=digest(json.dumps(snapshot, sort_keys=True)),
            targets_by_strong_id=targets,
            pilot_shots=1000,
        ):
            raise ValueError("Source sweep/decoder/reference identity changed")
        stats = saved_stats(source, experiments)
        continuation.validate_counts(stats, targets, complete=True)
        references = snapshot["rows"] + [
            midpoint.physical.result_row(s, str(source)) for s in stats
        ]
        saved_rows = json.loads(
            (source / "analysis_100000/source_comparison_rows.json").read_text()
        )
        if {r["strong_id"]: r for r in references} != {
            r["strong_id"]: r for r in saved_rows
        } or len(saved_rows) != 10:
            raise ValueError("Source comparison no longer agrees with the completed checkpoint")
        selected = select_experiment(experiments)
        key = task_for(selected).strong_id()
        seeds = [s for s in stats if s.strong_id == key]
        audit = next(a for a in audits if a["strong_id"] == key)
        audit.update(seed_shots=seeds[0].shots, seed_errors=seeds[0].errors)
        files = [
            source / name
            for name in (
                "identity.json",
                "sinter_progress.csv",
                "manifest.json",
                "preflight.json",
                "reference_snapshot.json",
                "analysis_100000/source_comparison_rows.json",
            )
        ]
        identity = dict(
            runtime_identity=runtime_identity(),
            selected_strong_ids=[key],
            seed_csv_sha256=digest(continuation.previous.seed_text(seeds)),
            reference_rows_sha256=digest(json.dumps(references, sort_keys=True)),
            source_files_sha256={str(p): digest(p.read_text()) for p in files},
        )
        print(
            f"Selected ONLY d4 XX, p={midpoint.PROBABILITIES[0]:.17g}, eight rounds: "
            f"{seeds[0].shots:,} seed shots / {seeds[0].errors} failures",
            flush=True,
        )
        return [selected], seeds, references, identity, [audit]


def analyze(
    args: argparse.Namespace,
    experiments: list[BuiltExperiment],
    seeds: list[sinter.TaskStats],
    references: list[dict[str, Any]],
    targets: dict[str, int],
) -> None:
    stats = saved_stats(args.output_dir, experiments)
    continuation.validate_counts(stats, targets, complete=True)
    rows = continuation.previous.replace_extended_rows(
        references, stats, seeds, str(args.output_dir), expected_conditions=1
    )
    stage = args.output_dir / f"analysis_add{args.additional_shots}"
    if stage.exists():
        stage = args.output_dir / (
            stage.name + "_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        )
    grid = {(m, 8, p) for m in (3, 4) for p in (1e-4, *midpoint.PROBABILITIES, 3e-4, 1e-3)}
    for error_bars in ("jackknife1sigma", "exact95"):
        midpoint.write_comparison_rows(
            stage,
            rows,
            error_bars=error_bars,
            only_rounds=8,
            expected_grid=grid,
            show_fit_caption=False,
            title="HGP simplex / unrotated surface — eight adapter rounds",
        )
    write_json(stage / "source_comparison_rows.json", rows)
    events = args.output_dir / "invocations.jsonl"
    write_json(
        stage / "timing.json",
        dict(
            targets_by_strong_id=targets,
            extended_conditions=[r for r in rows if r["strong_id"] in targets],
            invocations=[json.loads(line) for line in events.read_text().splitlines()]
            if events.exists()
            else [],
        ),
    )


def run(args: argparse.Namespace) -> None:
    continuation.run(
        args,
        source_loader=load_sources,
        extra_source_directories=(args.source_dir,),
        analysis_writer=analyze,
    )


if __name__ == "__main__":
    run(parse_args())
