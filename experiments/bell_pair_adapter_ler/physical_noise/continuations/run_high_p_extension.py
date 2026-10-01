"""Add 20k shots each to d=4 and d=8 eight-round XX at physical p=0.001.

Resume targets are 40k and 30k, relative to fixed archived seeds of 20k and 10k.
The completed low-p extensions are read-only comparison data, never resampled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sinter
import stim

from qldpc import decoders

from ...bell_fidelity.run_ppm_fidelity_sweep import digest, runtime_identity, task_for
from ...common.run_experiment import BuiltExperiment
from ..run_physical_noise_comparison import OUTPUTS, scaled_noise_signature
from . import run_d8_noise_extension as continuation
from . import run_low_p_extension as previous

DEFAULT_OUTPUT = OUTPUTS / "ppm_basis_first_hgp_d4_d8_xx_p1e3_surg8_add20000"
D8_ARCHIVE = OUTPUTS / "ppm_basis_first_hgp_m4_d8_prep4_surg8_10000" / "samples"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-dir", type=Path, default=previous.BASELINE)
    parser.add_argument("--high-p-dir", type=Path, default=previous.HIGH_P_CONTROL)
    parser.add_argument("--previous-extension-dir", type=Path, default=previous.DEFAULT_OUTPUT)
    parser.add_argument("--comparison-dir", type=Path, default=continuation.DEFAULT_OUTPUT)
    parser.add_argument("--d8-source-dir", type=Path, default=D8_ARCHIVE)
    parser.add_argument(
        "--additional-shots",
        type=int,
        default=20000,
        help="Additional to fixed imported seeds, not each restart",
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


def load_archived_xx(
    directory: Path, simplex_m: int, expected_shots: int
) -> tuple[BuiltExperiment, sinter.TaskStats, str]:
    """Read only the matched XX task; an older archive can also contain ZZ."""
    current = runtime_identity()
    config = json.loads((directory / "run_config.json").read_text())
    if "runtime_identity" in config:
        if config["runtime_identity"] != current:
            raise ValueError("Archived decoder/dependency identity changed")
        provenance = "source hashes and versions verified"
    else:
        if config["versions"] != current["versions"] or config.get("decoder_kwargs", {}):
            raise ValueError("Legacy archive versions/default decoder differ")
        provenance = "versions/default decoder verified; archive has no decoder-source hashes"
    candidates = [
        m
        for m in json.loads((directory / "manifest.json").read_text())
        if m["logical_basis"] == "X"
        and m["surgery_rounds"] == 8
        and m["physical_error_rate"] == 0.001
        and m["bell_pair_infidelity"] == 0
    ]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one archived eight-round XX high-p point")
    archived = candidates[0]
    metadata = {k: v for k, v in archived.items() if k not in {"circuit_path", "dem_path"}}
    expected = dict(
        left_code=f"hgp_simplex_m{simplex_m}",
        right_code=f"surface_unrotated_d{2 ** (simplex_m - 1)}",
        logical_basis="X",
        surgery_rounds=8,
        preparation_rounds=4,
        physical_error_rate=0.001,
        bell_pair_fidelity=1.0,
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
        if metadata.get(key) != value:
            raise ValueError(f"Archived experiment differs in {key}")
    paths = [directory / archived[k] for k in ("circuit_path", "dem_path")]
    if any(not p.resolve().is_relative_to(directory.resolve()) for p in paths):
        raise ValueError("Archived artifact path escapes its directory")
    experiment = BuiltExperiment(
        archived["name"],
        stim.Circuit.from_file(paths[0]),
        stim.DetectorErrorModel.from_file(paths[1]),
        metadata,
    )
    key = task_for(experiment).strong_id()
    stats = [
        s
        for s in sinter.read_stats_from_csv_files(directory / "sinter_progress.csv")
        if s.strong_id == key
    ]
    if len(stats) != 1 or stats[0].json_metadata != metadata or stats[0].decoder != "qldpc_bp_osd":
        raise ValueError("Archived task identity changed")
    if stats[0].shots != expected_shots:
        raise ValueError("Unexpected high-p seed count")
    return experiment, stats[0], provenance


def validate_against_baseline(
    experiment: BuiltExperiment, baseline: BuiltExperiment, original: dict[str, Any]
) -> dict[str, Any]:
    for key in experiment.metadata.keys() | baseline.metadata.keys():
        if key not in {"name", "physical_error_rate"}:
            if experiment.metadata.get(key) != baseline.metadata.get(key):
                raise ValueError(f"High-p experiment changed {key}")
    signature = scaled_noise_signature(experiment.circuit, 0.001)
    if signature != scaled_noise_signature(
        baseline.circuit, baseline.metadata["physical_error_rate"]
    ):
        raise ValueError("High-p circuit changed operations or noise locations")
    det, obs = (
        experiment.circuit.without_noise()
        .compile_detector_sampler(seed=731)
        .sample(shots=32, separate_observables=True)
    )
    if det.any() or obs.any() or experiment.circuit.num_observables != 1:
        raise ValueError("Noiseless XX validation failed")
    compiled = decoders.SinterDecoder().compile_decoder_for_dem(experiment.detector_error_model)
    settings = {k: getattr(compiled.decoder, k) for k in original["decoder_settings"]}
    if settings != original["decoder_settings"] or settings["osd_method"] != "OSD_0":
        raise ValueError("High-p production decoder settings changed")
    return dict(
        name=experiment.name,
        strong_id=task_for(experiment).strong_id(),
        decoder_settings=settings,
        circuit_sha256=digest(str(experiment.circuit)),
        dem_sha256=digest(str(experiment.detector_error_model)),
        scaled_noise_sha256=signature,
        noiseless_validation_shots=32,
    )


def load_sources(args: argparse.Namespace) -> continuation.SourceData:
    low_experiments, low_seeds, references, low_identity, _ = continuation.load_sources(args)
    targets = continuation.target_counts(low_seeds, 200000)
    latest = continuation.read_previous_extension(
        args.comparison_dir, low_experiments, low_seeds, low_identity, targets=targets
    )
    references = previous.replace_extended_rows(
        references, latest, low_seeds, str(args.comparison_dir.resolve())
    )
    reference_by_id = {r["strong_id"]: r for r in references}
    baselines, _ = continuation.load_baselines(args.baseline_dir)
    original = {
        a["name"]: a for a in json.loads((args.baseline_dir / "preflight.json").read_text())
    }
    experiments, seeds, audits = [], [], []
    for m, directory, count in ((3, args.high_p_dir, 20000), (4, args.d8_source_dir, 10000)):
        experiment, seed, provenance = load_archived_xx(directory, m, count)
        reference = reference_by_id.get(seed.strong_id)
        if reference is None or any(
            reference[k] != getattr(seed, k) for k in ("shots", "errors", "discards")
        ):
            raise ValueError("High-p seed differs from the current comparison")
        baseline = next(
            e
            for e in baselines
            if e.metadata["left_code"] == f"hgp_simplex_m{m}"
            and e.metadata["surgery_rounds"] == 8
            and e.metadata["physical_error_rate"] == 1e-4
        )
        audit = validate_against_baseline(experiment, baseline, original[baseline.name])
        audit.update(
            seed_shots=seed.shots, seed_errors=seed.errors, seed_decoder_provenance=provenance
        )
        experiments.append(experiment)
        seeds.append(seed)
        audits.append(audit)
        print(
            f"Validated d={2 ** (m - 1)} XX p=0.001, eight rounds: {seed.shots:,} seed shots / {seed.errors} failures",
            flush=True,
        )
    files = [
        directory / filename
        for directory in (args.high_p_dir, args.d8_source_dir)
        for filename in ("sinter_progress.csv", "manifest.json", "run_config.json")
    ]
    files += [
        args.comparison_dir / filename
        for filename in (
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
        comparison_identity=low_identity,
        source_files_sha256={str(p.resolve()): digest(p.read_text()) for p in files},
        seed_decoder_provenance={a["strong_id"]: a["seed_decoder_provenance"] for a in audits},
    )
    return experiments, seeds, references, identity, audits


def run(args: argparse.Namespace) -> None:
    continuation.run(
        args,
        source_loader=load_sources,
        extra_source_directories=(args.comparison_dir, args.d8_source_dir),
    )


if __name__ == "__main__":
    run(parse_args())
