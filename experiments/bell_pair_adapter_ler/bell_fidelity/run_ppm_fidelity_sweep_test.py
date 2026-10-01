from __future__ import annotations

import csv
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import sinter
import stim

from qldpc.codes.common import CSSCode

from ..common.paths import OUTPUT_ROOT
from ..common.run_experiment import BuiltExperiment, write_artifacts
from . import run_ppm_fidelity_sweep as runner


def args_for(path: Path) -> Any:
    return runner.parse_args(
        [
            "--output-dir",
            str(path),
            "--bell-pair-fidelities",
            "1",
            "0.999",
            "0.97",
            "--max-shots",
            "3",
            "--num-workers",
            "1",
        ]
    )


@pytest.fixture(scope="module")
def built() -> tuple[list[BuiltExperiment], list[dict[str, Any]]]:
    return runner.build_sweep(args_for(Path("unused")))


def test_fidelity_grid_canonicalization() -> None:
    args = args_for(Path("unused"))
    assert args.bell_pair_infidelities == [0.0, 0.001, 0.03]
    default = runner.parse_args(["--output-dir", "unused"])
    assert len(default.bell_pair_infidelities) == 9
    assert default.max_shots == 1000 and default.num_workers == 8


@pytest.mark.parametrize(
    "extra",
    [
        ["--bell-pair-fidelities", "1", "nan"],
        ["--bell-pair-fidelities", "1", "1.1"],
        ["--bell-pair-fidelities", "1", "1"],
        ["--bell-pair-fidelities", "0.999", "0.99"],
        ["--bell-pair-fidelities", "1"],
        ["--rounds", "0"],
        ["--bases", "X", "X"],
        ["--num-workers", "0"],
    ],
)
def test_invalid_args(extra: list[str]) -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--output-dir", "unused", *extra])


def test_actual_hgp_grid_and_ideal_controls(
    built: tuple[list[BuiltExperiment], list[dict[str, Any]]],
) -> None:
    experiments, audit = built
    assert len(experiments) == len(audit) == 6
    for basis in "XZ":
        rows = [r for r in audit if r["logical_basis"] == basis]
        assert len({r["non_bell_operations_sha256"] for r in rows}) == 1
        assert [r["num_bell_source_channels"] for r in rows] == [0, 28, 28]
        assert all(r["bell_pairs_consumed"] == 28 for r in rows)


@pytest.mark.adapter_archive
def test_ideal_controls_match_archived_circuits(
    built: tuple[list[BuiltExperiment], list[dict[str, Any]]],
) -> None:
    experiments, _ = built
    for basis in "XZ":
        ideal = next(
            e
            for e in experiments
            if e.metadata["logical_basis"] == basis and e.metadata["bell_pair_fidelity"] == 1
        )
        reference = OUTPUT_ROOT / (
            "hgp_simplex3_surface_unrotated_d4_hardware_prep4_surg4_bell0_20k"
            if basis == "X"
            else "hgp_simplex3_surface_unrotated_d4_hardware_schedule_ablations_20k/Z/local_order"
        )
        meta = next(
            r
            for r in json.loads((reference / "manifest.json").read_text())
            if r["logical_basis"] == basis
        )
        assert str(ideal.circuit) == str(stim.Circuit.from_file(reference / meta["circuit_path"]))
        assert str(ideal.detector_error_model) == str(
            stim.DetectorErrorModel.from_file(reference / meta["dem_path"])
        )


def test_no_distance_search_and_surface_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*unused: Any, **kwargs: Any) -> None:
        raise AssertionError("Exact distance searches are forbidden")

    monkeypatch.setattr(CSSCode, "get_distance", fail)
    args = runner.parse_args(
        [
            "--output-dir",
            "unused",
            "--code-pair",
            "surface-steane",
            "--rounds",
            "1",
            "--bell-pair-fidelities",
            "1",
            "0.999",
        ]
    )
    experiments, audit = runner.build_sweep(args)
    assert len(experiments) == 4
    assert all(r["bell_pairs_consumed"] == 5 for r in audit)


def test_resume_artifact_and_task_guards(
    tmp_path: Path,
    built: tuple[list[BuiltExperiment], list[dict[str, Any]]],
) -> None:
    experiments, _ = built
    write_artifacts(experiments, tmp_path)
    runner.check_output(tmp_path, experiments)
    with pytest.raises(ValueError, match="grid/bases"):
        runner.check_output(tmp_path, experiments[:-1])
    path = tmp_path / "sinter_progress.csv"
    stat = sinter.TaskStats(
        strong_id="foreign", decoder="qldpc_bp_osd", json_metadata=experiments[0].metadata, shots=1
    )
    path.write_text(sinter.CSV_HEADER + "\n" + stat.to_csv_line() + "\n")
    with pytest.raises(ValueError, match="foreign task"):
        runner.saved_stats(tmp_path, experiments)
    path.unlink()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    circuit_path = tmp_path / manifest[0]["circuit_path"]
    circuit_path.write_text(circuit_path.read_text() + "X_ERROR(0.02) 0\n")
    with pytest.raises(ValueError, match="circuit/DEM changed"):
        runner.check_output(tmp_path, experiments)


def append_fake_samples(experiments: list[BuiltExperiment], **kwargs: Any) -> None:
    output = kwargs["output_dir"]
    previous = {s.strong_id: s.shots for s in runner.saved_stats(output, experiments)}
    path = output / "sinter_progress.csv"
    for experiment in experiments:
        identity = runner.task_for(experiment).strong_id()
        new_shots = kwargs["max_shots"] - previous.get(identity, 0)
        if not new_shots:
            continue
        stat = sinter.TaskStats(
            strong_id=identity,
            decoder="qldpc_bp_osd",
            json_metadata=experiment.metadata,
            shots=new_shots,
            errors=0,
            seconds=new_shots / 100,
        )
        exists = path.exists()
        with path.open("a") as stream:
            if not exists:
                stream.write(sinter.CSV_HEADER + "\n")
            stream.write(stat.to_csv_line() + "\n")
        kwargs["progress_callback"](SimpleNamespace(new_stats=[stat]))


def test_resumption_preserves_pilot_analysis_and_runtime_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    built: tuple[list[BuiltExperiment], list[dict[str, Any]]],
) -> None:
    monkeypatch.setattr(runner, "build_sweep", lambda args: built)
    monkeypatch.setattr(runner, "sample_experiments", append_fake_samples)
    args = args_for(tmp_path)
    first = runner.run(args)
    assert first["before"]["shots"] == 0 and first["new_shots"] == 18
    pilot_pdf = tmp_path / "analysis_3" / "adapter_ler_vs_bell_fidelity.pdf"
    digest = pilot_pdf.read_bytes()
    args.max_shots = 5
    second = runner.run(args)
    assert second["before"]["shots"] == 18 and second["new_shots"] == 12
    assert second["after"]["shots"] == 30
    assert pilot_pdf.read_bytes() == digest
    with (tmp_path / "analysis_5" / "summary.csv").open() as stream:
        assert {r["shots"] for r in csv.DictReader(stream)} == {"5"}
    args.max_shots = 3
    with pytest.raises(ValueError, match="below existing counts"):
        runner.run(args)
    args.max_shots = 5
    monkeypatch.setattr(runner, "runtime_identity", lambda: {"changed": True})
    with pytest.raises(ValueError, match="identity changed"):
        runner.run(args)


def test_interrupted_collection_keeps_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    built: tuple[list[BuiltExperiment], list[dict[str, Any]]],
) -> None:
    monkeypatch.setattr(runner, "build_sweep", lambda args: built)

    def interrupted(experiments: list[BuiltExperiment], **kwargs: Any) -> None:
        kwargs["max_shots"] = 1
        append_fake_samples(experiments, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "sample_experiments", interrupted)
    args = args_for(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        runner.run(args)
    records = [
        json.loads(line) for line in (tmp_path / "invocations.jsonl").read_text().splitlines()
    ]
    assert records[-1]["status"] == "interrupted" and records[-1]["after"]["shots"] == 6
    assert not (tmp_path / "analysis_3").exists()
    monkeypatch.setattr(runner, "sample_experiments", append_fake_samples)
    resumed = runner.run(args)
    assert resumed["before"]["shots"] == 6 and resumed["new_shots"] == 12


def test_writer_lock_and_nonempty_guard(tmp_path: Path) -> None:
    args = args_for(tmp_path)
    with (tmp_path / ".collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Another writer"):
            runner.run(args)
    (tmp_path / "unrelated.txt").write_text("Keep this file")
    with pytest.raises(ValueError, match="Nonempty"):
        runner.run(args)
    assert (tmp_path / "unrelated.txt").read_text() == "Keep this file"
