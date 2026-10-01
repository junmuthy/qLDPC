from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sinter
import stim

from qldpc import codes, decoders
from qldpc.circuits.surgery import JointPPMSchedule
from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli

from ..common import run_experiment as common
from ..common.paths import OUTPUT_ROOT
from . import run_ppm_basis_first as runner

OUTPUTS = OUTPUT_ROOT


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_generic_preset_matches_explicit_schedule_and_preserves_defaults(basis: Pauli) -> None:
    kwargs: dict[str, Any] = dict(
        left_code=codes.IcebergCode(4),
        right_code=codes.SteaneCode(),
        left_logical_index=1,
        basis=basis,
        preparation_rounds=2,
        surgery_rounds=2,
        adapter_noise_targets="adapter_hardware",
        reduce_detectors=True,
    )
    old = common.build_experiment_for_codes(**kwargs)
    explicit_old = common.build_experiment_for_codes(**kwargs, schedule_policy="legacy")
    new = common.build_experiment_for_codes(**kwargs, schedule_policy="ppm_basis_first")
    manual = common.build_experiment_for_codes(
        **kwargs, surgery_schedule=JointPPMSchedule.ppm_basis_first(basis)
    )
    assert old == explicit_old
    assert "schedule_policy" not in old.metadata
    assert new.circuit == manual.circuit
    assert new.detector_error_model == manual.detector_error_model
    assert new.name != old.name
    assert new.metadata["schedule_policy"] == common.PPM_BASIS_FIRST_VERSION
    assert new.metadata["dem_analysis"] == "undecomposed_exact"
    assert new.circuit.num_ticks == old.circuit.num_ticks
    if basis == Pauli.X:
        assert new.circuit == old.circuit
        assert new.detector_error_model == old.detector_error_model
    with pytest.raises(ValueError, match="not both"):
        common.build_experiment_for_codes(
            **kwargs, schedule_policy="ppm_basis_first", surgery_schedule=JointPPMSchedule()
        )
    with pytest.raises(ValueError, match="schedule_policy"):
        common.build_experiment_for_codes(**kwargs, schedule_policy="bad")  # type: ignore[arg-type]


@pytest.mark.parametrize("infidelity", [0.0, 0.001])
@pytest.mark.adapter_archive
def test_surface_preset_matches_all_existing_hardware_rounds(infidelity: float) -> None:
    directories = {
        "X": OUTPUTS
        / "adapter_hardware_prep3_round_sweep_1_5_comparison"
        / ("bell0" if infidelity == 0 else "bell0p001"),
        "Z": OUTPUTS / "surface_steane_hardware_local_order_prep3_rounds1_5_20k",
    }
    for experiment in common.build_experiment_matrix(
        rounds=(1, 2, 3, 4, 5),
        bell_pair_infidelity=infidelity,
        adapter_noise_targets="adapter_hardware",
        reduce_detectors=True,
        schedule_policy="ppm_basis_first",
    ):
        basis, rounds = experiment.metadata["logical_basis"], experiment.metadata["rounds"]
        directory = directories[basis]
        archived = next(
            r
            for r in json.loads((directory / "manifest.json").read_text())
            if r["logical_basis"] == basis
            and r["surgery_rounds"] == rounds
            and r["bell_pair_infidelity"] == infidelity
        )
        assert str(experiment.circuit) == str(
            stim.Circuit.from_file(directory / archived["circuit_path"])
        )
        assert str(experiment.detector_error_model) == str(
            stim.DetectorErrorModel.from_file(directory / archived["dem_path"])
        )


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
@pytest.mark.adapter_archive
def test_hgp_matches_archive_without_distance_search(
    basis: Pauli, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any, **kwargs: Any) -> int:
        raise AssertionError("Unexpected exact distance search")

    monkeypatch.setattr(CSSCode, "get_distance", fail)
    left, right, _, _ = runner.code_pair("hgp-simplex-surface")
    experiment = common.build_experiment_for_codes(
        left_code=left,
        right_code=right,
        basis=basis,
        preparation_rounds=4,
        surgery_rounds=4,
        adapter_noise_targets="adapter_hardware",
        reduce_detectors=True,
        schedule_policy="ppm_basis_first",
    )
    directory = OUTPUTS / (
        "hgp_simplex3_surface_unrotated_d4_hardware_prep4_surg4_bell0_20k"
        if basis == Pauli.X
        else "hgp_simplex3_surface_unrotated_d4_hardware_schedule_ablations_20k/Z/local_order"
    )
    archived = next(
        r
        for r in json.loads((directory / "manifest.json").read_text())
        if r["logical_basis"] == basis.name
    )
    assert str(experiment.circuit) == str(
        stim.Circuit.from_file(directory / archived["circuit_path"])
    )
    assert str(experiment.detector_error_model) == str(
        stim.DetectorErrorModel.from_file(directory / archived["dem_path"])
    )
    assert experiment.metadata["joint_code_distance_status"] == "not_computed"


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_ideal_and_noisy_decoding_smoke(basis: Pauli) -> None:
    for p in (0.0, 0.001):
        experiment = common.build_experiment(
            basis=basis,
            rounds=2,
            schedule_policy="ppm_basis_first",
            physical_error_rate=p,
            bell_pair_infidelity=p,
            adapter_noise_targets="adapter_hardware",
            reduce_detectors=True,
        )
        detectors, obs = experiment.circuit.compile_detector_sampler(seed=91).sample(
            32, separate_observables=True
        )
        predicted = (
            decoders.SinterDecoder()
            .compile_decoder_for_dem(experiment.detector_error_model)
            .decode_shots(detectors)
        )
        assert predicted.shape == obs.shape
        if p == 0:
            assert not np.any(detectors) and not np.any(obs) and not np.any(predicted)


def test_named_full_surgery_uses_saved_dem_and_keeps_legacy_auto(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    old = common.build_experiment(basis=Pauli.X, rounds=1)
    new = common.build_experiment(basis=Pauli.X, rounds=1, schedule_policy="ppm_basis_first")

    def collect(**kwargs: Any) -> list[sinter.TaskStats]:
        tasks = kwargs["tasks"]
        assert tasks[0].detector_error_model is None
        assert tasks[1].detector_error_model == new.detector_error_model
        return []

    monkeypatch.setattr(sinter, "collect", collect)
    common.sample_experiments(
        [old, new], output_dir=tmp_path, num_workers=1, max_shots=1, max_errors=1
    )
    common.write_artifacts([old], tmp_path)
    with pytest.raises(ValueError, match="different experiment"):
        common.write_artifacts([new], tmp_path)
    assert common.default_output_dir("full_surgery", False) != common.default_output_dir(
        "full_surgery", False, "ppm_basis_first"
    )


def test_summary_and_resume_timing_count_only_new_shots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = runner.parse_args(["--output-dir", str(tmp_path), "--rounds", "1", "--max-shots", "10"])
    saved: list[sinter.TaskStats] = []

    def checkpoint(*unused: Any) -> list[sinter.TaskStats]:
        return saved

    def collect(experiments: list[common.BuiltExperiment], **kwargs: Any) -> list[sinter.TaskStats]:
        assert kwargs["max_errors"] == kwargs["max_shots"]
        saved[:] = [
            sinter.TaskStats(
                strong_id=e.name,
                decoder="qldpc_bp_osd",
                json_metadata=e.metadata,
                shots=kwargs["max_shots"],
                errors=0,
                seconds=kwargs["max_shots"] / 100,
            )
            for e in experiments
        ]
        return saved

    monkeypatch.setattr(runner, "checkpoint_stats", checkpoint)
    monkeypatch.setattr(runner, "sample_experiments", collect)
    first = runner.run(args)
    assert first["new_shots"] == 20
    args.max_shots = 15
    second = runner.run(args)
    assert second["after"]["shots"] == 30 and second["new_shots"] == 10
    assert second["new_worker_seconds"] == pytest.approx(0.1)
    assert second["sampling_decode_wall_seconds"] <= second["total_invocation_wall_seconds"]
    with (tmp_path / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r["local_order"] for r in rows] == ["X,Z", "Z,X"]
    assert all(r["schedule_policy"] == common.PPM_BASIS_FIRST_VERSION for r in rows)
    assert len((tmp_path / "invocations.jsonl").read_text().splitlines()) == 4
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    saved_dem = tmp_path / manifest[0]["dem_path"]
    original = saved_dem.read_text()
    saved_dem.write_text("error(0.1) L0\n")
    with pytest.raises(ValueError, match="Saved circuit/DEM changed"):
        runner.run(args)
    saved_dem.write_text(original)
    args.rounds = [2]
    with pytest.raises(ValueError, match="different experiment"):
        runner.run(args)


def test_interruption_preserves_flushed_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = runner.parse_args(["--output-dir", str(tmp_path), "--rounds", "1"])
    saved: list[sinter.TaskStats] = []
    monkeypatch.setattr(runner, "checkpoint_stats", lambda *args: saved)

    def interrupted(experiments: list[common.BuiltExperiment], **kwargs: Any) -> None:
        saved.append(
            sinter.TaskStats(
                strong_id="test",
                decoder="qldpc_bp_osd",
                json_metadata=experiments[0].metadata,
                shots=5,
                errors=0,
                seconds=0.1,
            )
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "sample_experiments", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runner.run(args)
    last = json.loads((tmp_path / "invocations.jsonl").read_text().splitlines()[-1])
    assert last["status"] == "interrupted" and last["new_shots"] == 5
    with (tmp_path / "summary.csv").open() as stream:
        assert next(csv.DictReader(stream))["shots"] == "5"


@pytest.mark.parametrize(
    "extra", [["--max-shots", "0"], ["--rounds", "1", "1"], ["--bases", "X", "X"]]
)
def test_reject_bad_cli(extra: list[str], tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--output-dir", str(tmp_path), *extra])


def test_code_pair_validation() -> None:
    with pytest.raises(ValueError, match="only to"):
        runner.code_pair("surface-steane", surface_distance=4)
    with pytest.raises(ValueError, match="Unknown"):
        runner.code_pair("bad")
    with pytest.raises(ValueError, match="simplex_m"):
        runner.code_pair("hgp-simplex-surface", simplex_m=1)
    with pytest.raises(ValueError, match="surface_distance"):
        runner.code_pair("hgp-simplex-surface", surface_distance=1)
