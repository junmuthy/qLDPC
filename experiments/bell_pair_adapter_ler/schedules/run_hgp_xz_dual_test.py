from __future__ import annotations

import dataclasses
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import sinter
import stim

from ..common.run_experiment import write_artifacts
from . import run_hgp_xz_dual as runner
from .hgp_xz_dual import DualBuild
from .hgp_xz_dual_test import REFERENCE


@pytest.fixture(scope="module")
def built(dual_m4: DualBuild) -> DualBuild:
    return dual_m4


@pytest.fixture(autouse=True)
def use_generated_reference(monkeypatch: pytest.MonkeyPatch, generated_m4_reference: Path) -> None:
    monkeypatch.setattr(sys.modules[__name__], "REFERENCE", generated_m4_reference)


def args_for(output: Path, *extra: str) -> Any:
    return runner.parse_args(
        [
            "--output-dir",
            str(output),
            "--reference-dir",
            str(REFERENCE),
            "--pilot-shots",
            "4",
            "--max-shots",
            "8",
            *extra,
        ]
    )


def append_batch(
    output: Path, built: DualBuild, shots: int, *, errors: int = 0
) -> sinter.TaskStats:
    stat = sinter.TaskStats(
        strong_id=runner.task_for(built.experiment).strong_id(),
        decoder="qldpc_bp_osd",
        json_metadata=built.experiment.metadata,
        shots=shots,
        errors=errors,
        seconds=shots / 100,
    )
    path = output / "sinter_progress.csv"
    exists = path.exists()
    with path.open("a") as stream:
        if not exists:
            stream.write(sinter.CSV_HEADER + "\n")
        stream.write(stat.to_csv_line() + "\n")
    return stat


def test_staged_collection_resume_profiles_and_plots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    built: DualBuild,
) -> None:
    monkeypatch.setattr(runner, "build_dual_experiment", lambda **kwargs: built)
    reference_before = {
        p.name: p.read_bytes()
        for p in (
            REFERENCE / "summary.csv",
            REFERENCE / "manifest.json",
            REFERENCE / "sinter_progress.csv",
        )
    }
    caps = []

    def collect(experiments: list[Any], **kwargs: Any) -> None:
        assert len(experiments) == 1 and experiments[0].metadata["logical_basis"] == "Z"
        assert kwargs["num_workers"] == 8
        assert kwargs["max_errors"] == kwargs["max_shots"]
        caps.append(kwargs["max_shots"])
        previous = runner.totals(runner.saved_stats(tmp_path, built.experiment))["shots"]
        delta = append_batch(tmp_path, built, int(kwargs["max_shots"] - previous))
        kwargs["progress_callback"](sinter.Progress(new_stats=(delta,), status_message="test"))

    monkeypatch.setattr(runner, "sample_experiments", collect)
    args = args_for(tmp_path)
    rows = runner.run(args)
    assert caps == [4, 8]
    assert [r["shots"] for r in rows] == [10000, 10000, 8]
    assert rows[-1]["ler_95ci_low"] == 0 and rows[-1]["ler_95ci_high"] > 0
    assert (tmp_path / "hgp_xz_dual_comparison.pdf").stat().st_size > 1000
    assert (tmp_path / "hgp_xz_dual_timing.pdf").stat().st_size > 1000
    events = runner._events(tmp_path / "invocations.jsonl")
    completed = [e for e in events if e["status"] == "completed"]
    assert [e["before"]["shots"] for e in completed] == [0, 4]
    assert [e["profile"]["reported_batch_shots"] for e in completed] == [4, 4]
    assert runner.totals(runner.saved_stats(tmp_path, built.experiment))["shots"] == 8
    runner.run(args)
    assert caps == [4, 8]  # completed resume never resamples
    args.max_shots = 10
    runner.run(args)
    assert caps == [4, 8, 10]
    args.analyze_only = True
    runner.run(args)
    assert caps == [4, 8, 10]
    for name, content in reference_before.items():
        assert (REFERENCE / name).read_bytes() == content


def test_interrupt_and_resume_retains_flushed_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    built: DualBuild,
) -> None:
    monkeypatch.setattr(runner, "build_dual_experiment", lambda **kwargs: built)

    def interrupt(*unused: Any, **kwargs: Any) -> None:
        append_batch(tmp_path, built, 2)
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "sample_experiments", interrupt)
    args = args_for(tmp_path, "--no-profile")
    with pytest.raises(KeyboardInterrupt):
        runner.run(args)
    event = runner._events(tmp_path / "invocations.jsonl")[-1]
    assert event["status"] == "interrupted" and event["after"]["shots"] == 2
    assert (tmp_path / "summary.csv").exists()

    def resume(*unused: Any, **kwargs: Any) -> None:
        previous = runner.totals(runner.saved_stats(tmp_path, built.experiment))["shots"]
        append_batch(tmp_path, built, int(kwargs["max_shots"] - previous))

    monkeypatch.setattr(runner, "sample_experiments", resume)
    runner.run(args)
    completed = [
        e for e in runner._events(tmp_path / "invocations.jsonl") if e["status"] == "completed"
    ]
    assert [e["new_shots"] for e in completed] == [2, 4]
    assert runner.totals(runner.saved_stats(tmp_path, built.experiment))["shots"] == 8


def test_artifact_round_trip_and_foreign_checkpoint_protection(
    tmp_path: Path, built: DualBuild
) -> None:
    write_artifacts([built.experiment], tmp_path)
    entry = json.loads((tmp_path / "manifest.json").read_text())[0]
    reloaded = sinter.Task(
        circuit=stim.Circuit.from_file(tmp_path / entry["circuit_path"]),
        detector_error_model=stim.DetectorErrorModel.from_file(tmp_path / entry["dem_path"]),
        decoder="qldpc_bp_osd",
        json_metadata={k: v for k, v in entry.items() if k not in {"circuit_path", "dem_path"}},
    )
    assert reloaded.strong_id() == built.audit["strong_id"]
    shutil.copy2(REFERENCE / "sinter_progress.csv", tmp_path / "sinter_progress.csv")
    with pytest.raises(ValueError, match="different task"):
        runner.check_output(tmp_path, built.experiment)


def test_output_guard_and_concurrent_writer(tmp_path: Path, built: DualBuild) -> None:
    with runner.collection_lock(tmp_path):
        with pytest.raises(ValueError, match="Another writer"):
            with runner.collection_lock(tmp_path):
                pytest.fail("Second writer acquired lock")
        runner.check_output(tmp_path, built.experiment)
    (tmp_path / "unrelated.txt").write_text("preserve")
    with pytest.raises(ValueError, match="Nonempty"):
        runner.check_output(tmp_path, built.experiment)
    output = tmp_path / "artifacts"
    write_artifacts([built.experiment], output)
    changed = dataclasses.replace(
        built.experiment, metadata=dict(built.experiment.metadata, surgery_rounds=7)
    )
    with pytest.raises(ValueError, match="different experiment"):
        runner.check_output(output, changed)
    entry = json.loads((output / "manifest.json").read_text())[0]
    (output / entry["dem_path"]).write_text("error(0.1) L0\n")
    with pytest.raises(ValueError, match="Saved circuit/DEM changed"):
        runner.check_output(output, built.experiment)


@pytest.mark.parametrize(
    "extra",
    [
        ["--rounds", "0"],
        ["--max-shots", "0"],
        ["--simplex-m", "1"],
        ["--num-workers", "0"],
        ["--left-logical-index", "-1"],
        ["--physical-error-rate", "nan"],
        ["--surface-distance", "1"],
        ["--build-only", "--analyze-only"],
    ],
)
def test_bad_cli(tmp_path: Path, extra: list[str]) -> None:
    with pytest.raises(SystemExit):
        args_for(tmp_path, *extra)


def test_defaults_and_reference_path_protection(tmp_path: Path) -> None:
    args = runner.parse_args(["--output-dir", str(tmp_path), "--reference-dir", str(REFERENCE)])
    assert (args.pilot_shots, args.max_shots, args.num_workers, args.rounds) == (256, 1000, 8, 8)
    for output in (REFERENCE, REFERENCE / "nested", REFERENCE.parent):
        with pytest.raises(SystemExit):
            runner.parse_args(["--output-dir", str(output), "--reference-dir", str(REFERENCE)])
