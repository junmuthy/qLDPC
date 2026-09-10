from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sinter
import stim

from qldpc import decoders

from . import run_hgp_dual_schedule as runner
from .run_experiment import BuiltExperiment, write_artifacts


@pytest.fixture(scope="module")
def built() -> tuple[BuiltExperiment, dict[str, Any]]:
    return runner.build_dual_experiment()


def test_actual_hgp_duality_and_hardware_noise_window(
    built: tuple[BuiltExperiment, dict[str, Any]],
) -> None:
    experiment, audit = built
    assert experiment.metadata["logical_basis"] == "Z"
    assert experiment.metadata["schedule_policy"] == runner.SCHEDULE
    assert audit["source_matches_archived_hardware_xx"]
    assert audit["identical_ordered_dem_including_priors"]
    assert experiment.circuit.num_detectors == 340
    assert experiment.detector_error_model.num_errors == 2803
    assert audit["source_num_ticks"] == audit["dual_num_ticks"] == 332
    active = set(experiment.metadata["adapter_hardware_qubits"])
    measurements = 0
    crosses_boundary = False
    for op in experiment.circuit.flattened():
        qs = [t.value for t in op.targets_copy()]
        if op.name == "DEPOLARIZE2":
            assert measurements >= 4 * 122
            for a, b in zip(qs[::2], qs[1::2]):
                assert {a, b} & active
                crosses_boundary |= a < 123 or b < 123
        elif op.name in {"DEPOLARIZE1", "X_ERROR", "Y_ERROR", "Z_ERROR"}:
            assert measurements >= 4 * 122 and set(qs) <= active
        elif op.name in {"M", "MX"}:
            if op.gate_args_copy():
                assert measurements >= 4 * 122 and set(qs) <= active
            measurements += len(qs)
    assert crosses_boundary
    dets, obs = (
        experiment.circuit.without_noise()
        .compile_detector_sampler(seed=31)
        .sample(64, separate_observables=True)
    )
    assert not np.any(dets) and not np.any(obs)


def test_ordered_decoder_equivalence_on_same_syndromes(
    built: tuple[BuiltExperiment, dict[str, Any]],
) -> None:
    experiment, _ = built
    entry = next(
        e
        for e in json.loads((runner.REFERENCE / "manifest.json").read_text())
        if e["logical_basis"] == "X"
    )
    reference = stim.DetectorErrorModel.from_file(runner.REFERENCE / entry["dem_path"])
    dets, _, _ = reference.compile_sampler(seed=51).sample(16)
    a = decoders.SinterDecoder().compile_decoder_for_dem(reference)
    b = decoders.SinterDecoder().compile_decoder_for_dem(experiment.detector_error_model)
    assert np.array_equal(a.decode_shots(dets), b.decode_shots(dets))


def test_saved_artifacts_round_trip_task_identity_and_reject_other_runs(
    built: tuple[BuiltExperiment, dict[str, Any]],
    tmp_path: Path,
) -> None:
    experiment, audit = built
    write_artifacts([experiment], tmp_path)
    entry = json.loads((tmp_path / "manifest.json").read_text())[0]
    metadata = {k: v for k, v in entry.items() if k not in {"circuit_path", "dem_path"}}
    reloaded = sinter.Task(
        circuit=stim.Circuit.from_file(tmp_path / entry["circuit_path"]),
        detector_error_model=stim.DetectorErrorModel.from_file(tmp_path / entry["dem_path"]),
        decoder="qldpc_bp_osd",
        json_metadata=metadata,
    )
    assert reloaded.strong_id() == audit["strong_id"]
    previous = (tmp_path / "manifest.json").read_bytes()
    different = BuiltExperiment(
        experiment.name,
        experiment.circuit,
        experiment.detector_error_model,
        dict(experiment.metadata, schedule_policy="other"),
    )
    with pytest.raises(ValueError, match="different experiment"):
        write_artifacts([different], tmp_path)
    assert (tmp_path / "manifest.json").read_bytes() == previous


def test_comparison_with_zero_failures_preserves_reference(tmp_path: Path) -> None:
    before = (runner.REFERENCE / "summary.csv").read_bytes()
    with (runner.REFERENCE / "summary.csv").open() as source:
        rows = list(csv.DictReader(source))
    row = next(row for row in rows if row["logical_basis"] == "Z")
    row.update(name=row["name"] + "_" + runner.SCHEDULE, shots="1000", errors="0", seconds="10")
    with (tmp_path / "summary.csv").open("w", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    comparison = runner.write_comparison(tmp_path)
    assert len(comparison) == 3
    assert comparison[-1]["logical_error_rate"] == comparison[-1]["ler_95ci_low"] == 0
    assert comparison[-1]["ler_95ci_high"] > 0
    assert (tmp_path / "hgp_hardware_dual_schedule_comparison.pdf").stat().st_size > 1000
    assert (runner.REFERENCE / "summary.csv").read_bytes() == before
