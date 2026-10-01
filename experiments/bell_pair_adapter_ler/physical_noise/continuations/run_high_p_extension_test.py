from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import sinter

from ...common.run_experiment import BuiltExperiment, write_artifacts
from ..run_physical_noise_comparison_test import FakeProfile
from . import run_high_p_extension as runner
from .run_low_p_extension_test import sources as original_sources  # noqa: F401


def args_for(output: Path, additional: int = 4, *extra: str) -> Any:
    return runner.parse_args(
        ["--output-dir", str(output), "--additional-shots", str(additional), *extra]
    )


@pytest.fixture
def sources(original_sources: Any) -> Any:  # noqa: F811
    all_experiments = original_sources[0]
    selected = [
        e
        for e in all_experiments
        if e.metadata["surgery_rounds"] == 8 and e.metadata["physical_error_rate"] == 0.001
    ]
    seeds = [
        sinter.TaskStats(
            strong_id=runner.task_for(e).strong_id(),
            decoder="qldpc_bp_osd",
            json_metadata=e.metadata,
            shots=20 if e.metadata["left_code"] == "hgp_simplex_m3" else 10,
            errors=2,
            seconds=1,
        )
        for e in selected
    ]
    references = original_sources[1][2]
    seed_by_id = {s.strong_id: s for s in seeds}
    references = [
        runner.previous.result_row(seed_by_id[r["strong_id"]], "archive")
        if r["strong_id"] in seed_by_id
        else dict(r)
        for r in references
    ]
    # Preserve the latest lower-p counts, not the old 100k comparison.
    for r in references:
        if r["surgery_rounds"] == 8 and r["physical_error_rate"] < 0.001:
            if r["nominal_distance"] == 8:
                r["shots"] = 500000 if r["physical_error_rate"] == 1e-4 else 300000
            elif r["physical_error_rate"] == 1e-4:
                r["shots"] = 300000
    identity = dict(
        runtime_identity=runner.runtime_identity(),
        selected_strong_ids=sorted(seed_by_id),
        seed_csv_sha256=runner.digest(runner.previous.seed_text(seeds)),
        reference_rows_sha256=runner.digest(json.dumps(references, sort_keys=True)),
    )
    return all_experiments, (selected, seeds, references, identity, [])


def test_defaults_and_isolation(tmp_path: Path) -> None:
    args = runner.parse_args([])
    assert args.additional_shots == 20000 and args.num_workers == 8
    assert args.max_batch_size == 32 and args.max_batch_seconds == 5
    for extra in (["--additional-shots", "0"], ["--bases", "Z"], ["--num-workers", "0"]):
        with pytest.raises(SystemExit):
            runner.parse_args(extra)
    for path in (args.comparison_dir, args.d8_source_dir, args.high_p_dir):
        with pytest.raises(ValueError, match="separate"):
            runner.run(args_for(path))


def test_archive_selects_only_xx_and_tracks_legacy_provenance(sources: Any, tmp_path: Path) -> None:
    experiment = sources[1][0][1]
    # Production builders canonicalize metadata before hashing. Sinter's hash
    # includes dictionary order, so do the same for this synthetic archive.
    experiment = BuiltExperiment(
        experiment.name,
        experiment.circuit,
        experiment.detector_error_model,
        json.loads(json.dumps(experiment.metadata, sort_keys=True)),
    )
    seed = sinter.TaskStats(
        strong_id=runner.task_for(experiment).strong_id(),
        decoder="qldpc_bp_osd",
        json_metadata=experiment.metadata,
        shots=10,
        errors=2,
        seconds=1,
    )
    zmeta = json.loads(
        json.dumps(
            {**experiment.metadata, "name": experiment.name + "_zz", "logical_basis": "Z"},
            sort_keys=True,
        )
    )
    zz = BuiltExperiment(zmeta["name"], experiment.circuit, experiment.detector_error_model, zmeta)
    write_artifacts([experiment, zz], tmp_path)
    runner.continuation.write_json(
        tmp_path / "run_config.json", dict(versions=runner.runtime_identity()["versions"])
    )
    zzstat = sinter.TaskStats(
        strong_id=runner.task_for(zz).strong_id(),
        decoder="qldpc_bp_osd",
        json_metadata=zmeta,
        shots=10000,
        errors=100,
    )
    (tmp_path / "sinter_progress.csv").write_text(
        sinter.CSV_HEADER + "\n" + seed.to_csv_line() + "\n" + zzstat.to_csv_line() + "\n"
    )
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    loaded, stat, provenance = runner.load_archived_xx(tmp_path, 4, 10)
    assert runner.task_for(loaded).strong_id() == seed.strong_id and stat == seed
    assert "no decoder-source hashes" in provenance
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="seed count"):
        runner.load_archived_xx(tmp_path, 4, 11)
    with pytest.raises(ValueError, match="left_code"):
        runner.load_archived_xx(tmp_path, 3, 10)
    runner.continuation.write_json(
        tmp_path / "run_config.json", dict(runtime_identity=runner.runtime_identity())
    )
    assert "source hashes" in runner.load_archived_xx(tmp_path, 4, 10)[2]
    runner.continuation.write_json(tmp_path / "run_config.json", dict(runtime_identity={}))
    with pytest.raises(ValueError, match="identity changed"):
        runner.load_archived_xx(tmp_path, 4, 10)


def test_circuit_noise_and_decoder_validation(sources: Any) -> None:
    for experiment in sources[1][0]:
        baseline = next(
            e
            for e in sources[0]
            if e.metadata["left_code"] == experiment.metadata["left_code"]
            and e.metadata["surgery_rounds"] == 8
            and e.metadata["physical_error_rate"] == 1e-4
        )
        decoder = (
            runner.decoders.SinterDecoder()
            .compile_decoder_for_dem(baseline.detector_error_model)
            .decoder
        )
        settings = {
            k: getattr(decoder, k)
            for k in ("osd_method", "osd_order", "bp_method", "schedule", "max_iter")
        }
        audit = runner.validate_against_baseline(
            experiment, baseline, dict(decoder_settings=settings)
        )
        assert audit["noiseless_validation_shots"] == 32
        assert audit["strong_id"] == runner.task_for(experiment).strong_id()
        with pytest.raises(ValueError, match="decoder settings"):
            runner.validate_against_baseline(
                experiment, baseline, dict(decoder_settings={**settings, "max_iter": 999})
            )
        with pytest.raises(ValueError, match="local.*|surgery_schedule"):
            changed = BuiltExperiment(
                experiment.name,
                experiment.circuit,
                experiment.detector_error_model,
                {**experiment.metadata, "surgery_schedule": {}},
            )
            runner.validate_against_baseline(changed, baseline, dict(decoder_settings=settings))


def test_different_targets_resume_and_latest_low_p_preserved(
    sources: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected, seeds, references, identity, _ = data = sources[1]
    monkeypatch.setattr(runner, "load_sources", lambda args: data)
    monkeypatch.setattr(runner.continuation, "CollectionProfile", FakeProfile)
    output = tmp_path / "extension"
    runner.run(args_for(output, 4, "--build-only"))
    prefix = (output / "sinter_progress.csv").read_bytes()
    stopped = False
    calls = []

    def collect(experiments: Any, **kwargs: Any) -> None:
        nonlocal stopped
        assert experiments == selected
        assert all(
            e.metadata["physical_error_rate"] == 0.001
            and e.metadata["surgery_rounds"] == 8
            and e.metadata["logical_basis"] == "X"
            for e in experiments
        )
        assert not kwargs.get("existing_data_filepaths")
        assert kwargs["num_workers"] == 8
        targets = kwargs["max_shots_by_strong_id"]
        calls.append(targets)
        current = {s.strong_id: s for s in runner.continuation.saved_stats(output, selected)}
        for e in selected:
            key = runner.task_for(e).strong_id()
            shots = targets[key] - current[key].shots
            if not shots:
                continue
            stat = sinter.TaskStats(
                strong_id=key,
                decoder="qldpc_bp_osd",
                json_metadata=e.metadata,
                shots=shots,
                errors=0,
                seconds=1,
            )
            with (output / "sinter_progress.csv").open("a") as stream:
                stream.write(stat.to_csv_line() + "\n")
            kwargs["progress_callback"](SimpleNamespace(new_stats=[stat]))
            if not stopped:
                stopped = True
                raise KeyboardInterrupt

    monkeypatch.setattr(runner.continuation, "sample_experiments", collect)
    with pytest.raises(KeyboardInterrupt):
        runner.run(args_for(output))
    assert sorted(s.shots for s in runner.continuation.saved_stats(output, selected)) == [10, 24]
    runner.run(args_for(output))
    runner.run(args_for(output))
    runner.run(args_for(output, 4, "--analyze-only"))
    assert len(calls) == 2 and sorted(calls[-1].values()) == [14, 24]
    rows = json.loads((output / "analysis_add4/source_comparison_rows.json").read_text())
    for ref, row in zip(references, rows):
        if ref["strong_id"] not in calls[0]:
            assert row == ref
        else:
            assert row["shots"] == calls[0][row["strong_id"]] and row["additional_shots"] == 4
    with (output / "analysis_add4/comparison_jackknife_1sigma_rounds8.csv").open() as stream:
        plotted = list(csv.DictReader(stream))
    assert len(plotted) == 6 and all(r["surgery_rounds"] == "8" for r in plotted)
    assert (output / "sinter_progress.csv").read_bytes().startswith(prefix)
    with pytest.raises(ValueError, match="below saved"):
        runner.run(args_for(output, 2))
    (output / "identity.json").write_text(json.dumps({**identity, "changed": True}))
    with pytest.raises(ValueError, match="identity changed"):
        runner.run(args_for(output))
