"""Extend only d=8/eight-round XX at p=1e-4 from 500k to one million shots.

The additional-shot limit is relative to the fixed 500k imported checkpoint,
not the live count. Resume with the same command to reach the same target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import run_d8_noise_extension as continuation
from . import run_high_p_extension as high_p
from . import run_low_p_extension as previous
from .run_experiment import BuiltExperiment
from .run_physical_noise_comparison import OUTPUTS
from .run_ppm_fidelity_sweep import digest, runtime_identity, task_for

DEFAULT_OUTPUT = OUTPUTS / "ppm_basis_first_hgp_d8_xx_p1e4_surg8_add500000"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (
        ("output-dir", DEFAULT_OUTPUT),
        ("baseline-dir", previous.BASELINE),
        ("high-p-dir", previous.HIGH_P_CONTROL),
        ("previous-extension-dir", previous.DEFAULT_OUTPUT),
        ("comparison-dir", continuation.DEFAULT_OUTPUT),
        ("d8-source-dir", high_p.D8_ARCHIVE),
        ("high-extension-dir", high_p.DEFAULT_OUTPUT),
    ):
        parser.add_argument(f"--{name}", type=Path, default=default)
    parser.add_argument(
        "--additional-shots",
        type=int,
        default=500000,
        help="Additional to the fixed 500k seed, not each restart",
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


def select_experiment(baselines: list[BuiltExperiment]) -> BuiltExperiment:
    # Reuse the strict XX/protocol validation, then retain exactly one noise point.
    selected = [
        e
        for e in continuation.select_experiments(baselines)
        if e.metadata["physical_error_rate"] == 1e-4
    ]
    if len(selected) != 1:
        raise ValueError("Expected exactly one d=8/eight-round XX p=1e-4 task")
    return selected[0]


def load_sources(args: argparse.Namespace) -> continuation.SourceData:
    # Validate the complete chain once, without writing or sampling any source.
    high_experiments, high_seeds, references, high_identity, _ = high_p.load_sources(args)
    latest_high = continuation.read_previous_extension(
        args.high_extension_dir,
        high_experiments,
        high_seeds,
        high_identity,
        targets=continuation.target_counts(high_seeds, 20000),
    )
    references = previous.replace_extended_rows(
        references, latest_high, high_seeds, str(args.high_extension_dir.resolve())
    )
    baselines, _ = continuation.load_baselines(args.baseline_dir)
    selected = select_experiment(baselines)
    source_experiments = continuation.select_experiments(baselines)
    continuation.check_output(args.comparison_dir, source_experiments)
    stats = continuation.saved_stats(args.comparison_dir, source_experiments)
    key = task_for(selected).strong_id()
    seeds = [s for s in stats if s.strong_id == key]
    if len(seeds) != 1 or seeds[0].shots != 500000:
        raise ValueError("Expected the completed 500,000-shot d=8 p=1e-4 seed")
    reference = next(r for r in references if r["strong_id"] == key)
    if any(reference[k] != getattr(seeds[0], k) for k in ("shots", "errors", "discards")):
        raise ValueError("Selected seed differs from latest comparison")
    # The upstream preflight already verified this exact circuit/DEM/decoder.
    audit = next(
        a
        for a in json.loads((args.comparison_dir / "preflight.json").read_text())
        if a["strong_id"] == key
    )
    if audit["circuit_sha256"] != digest(str(selected.circuit)) or audit["dem_sha256"] != digest(
        str(selected.detector_error_model)
    ):
        raise ValueError("Selected archived preflight changed")
    audits = [{**audit, "seed_shots": seeds[0].shots, "seed_errors": seeds[0].errors}]
    files = [
        directory / filename
        for directory in (args.high_extension_dir, args.comparison_dir)
        for filename in (
            "identity.json",
            "seed_receipt.json",
            "reference_rows.json",
            "sinter_progress.csv",
            "preflight.json",
        )
    ]
    identity = dict(
        runtime_identity=runtime_identity(),
        selected_strong_ids=[key],
        seed_csv_sha256=digest(previous.seed_text(seeds)),
        reference_rows_sha256=digest(json.dumps(references, sort_keys=True)),
        latest_high_p_identity=high_identity,
        source_files_sha256={str(p.resolve()): digest(p.read_text()) for p in files},
    )
    print(
        f"Selected ONLY d=8 XX p=1e-4, eight rounds: {seeds[0].shots:,} seed shots / {seeds[0].errors} failures",
        flush=True,
    )
    return [selected], seeds, references, identity, audits


def run(args: argparse.Namespace) -> None:
    continuation.run(
        args,
        source_loader=load_sources,
        extra_source_directories=(args.comparison_dir, args.d8_source_dir, args.high_extension_dir),
    )


if __name__ == "__main__":
    run(parse_args())
