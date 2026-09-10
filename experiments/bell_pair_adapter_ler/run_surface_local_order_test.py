from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sinter
import stim

from qldpc import decoders

from . import run_surface_local_order as runner
from .run_experiment import BuiltExperiment, write_artifacts, write_summary
from .run_hgp_schedule_ablations import _ideal_prefix


@pytest.fixture(scope="module")
def suite() -> tuple[list[BuiltExperiment], dict[str, Any]]:
    return runner.build_suite()


def test_ten_zz_variants_and_twenty_unchanged_references(
    suite: tuple[list[BuiltExperiment], dict[str, Any]],
) -> None:
    experiments, audit = suite
    assert len(experiments) == len(audit["experiments"]) == 10
    assert len(audit["references"]) == 20
    assert all(e["canonical_circuit_and_ordered_dem_match"] for e in audit["references"])
    assert {
        (
            e.metadata["logical_basis"],
            e.metadata["surgery_rounds"],
            e.metadata["bell_pair_infidelity"],
        )
        for e in experiments
    } == {("Z", r, p) for r in range(1, 6) for p in (0.0, 0.001)}
    for experiment, check in zip(experiments, audit["experiments"]):
        assert check["name"] == experiment.name
        assert check["local_prefix_exactly_z_then_x"] and check["local_family_layers_identical"]
        assert check["terminal_quantum_operations_identical"]
        assert check["legacy_total_ticks"] == check["variant_total_ticks"]
        assert _ideal_prefix(experiment.circuit).num_measurements == 3 * 14
        meta = experiment.metadata
        assert meta["joint_code_parameters"][0] == 23  # original ZZ geometry, not dual XX's 24
        assert meta["surgery_schedule"] == {
            "bell_order": ["X", "Z"],
            "local_order": ["Z", "X"],
            "local_layers_sha256": None,
        }
        active = set(meta["adapter_hardware_qubits"])
        bell_channels = 0
        for op in experiment.circuit.flattened():
            qs = [t.value for t in op.targets_copy()]
            if op.name == "DEPOLARIZE2":
                assert all({a, b} & active for a, b in zip(qs[::2], qs[1::2]))
            elif op.name in {"DEPOLARIZE1", "X_ERROR", "Y_ERROR", "Z_ERROR", "PAULI_CHANNEL_1"}:
                assert set(qs) <= active
                if op.name == "PAULI_CHANNEL_1":
                    bell_channels += 1
                    assert np.allclose(
                        op.gate_args_copy(), [(1 - meta["bell_pair_fidelity"]) / 3] * 3
                    )
            elif op.name in {"M", "MX"} and op.gate_args_copy():
                assert set(qs) <= active
        assert bell_channels == (
            meta["num_bell_pair_checks_per_round"] * meta["surgery_rounds"]
            if meta["bell_pair_infidelity"]
            else 0
        )
        decoder = decoders.SinterDecoder().compile_decoder_for_dem(experiment.detector_error_model)
        dets, obs = (
            experiment.circuit.without_noise()
            .compile_detector_sampler(seed=219)
            .sample(4, separate_observables=True)
        )
        assert np.array_equal(decoder.decode_shots(dets), obs)
    runner.verify_reference_hashes(audit)


def test_task_identity_roundtrip_and_reject_overwrite(
    suite: tuple[list[BuiltExperiment], dict[str, Any]],
    tmp_path: Path,
) -> None:
    experiments, audit = suite
    write_artifacts(experiments, tmp_path)
    before = (tmp_path / "manifest.json").read_bytes()
    for entry, check in zip(json.loads(before), audit["experiments"]):
        task = sinter.Task(
            circuit=stim.Circuit.from_file(tmp_path / entry["circuit_path"]),
            detector_error_model=stim.DetectorErrorModel.from_file(tmp_path / entry["dem_path"]),
            decoder="qldpc_bp_osd",
            json_metadata={k: v for k, v in entry.items() if k not in {"circuit_path", "dem_path"}},
        )
        assert task.strong_id() == check["strong_id"]
    with pytest.raises(ValueError, match="different experiment"):
        write_artifacts([dataclasses.replace(experiments[0], metadata={"other": True})], tmp_path)
    assert (tmp_path / "manifest.json").read_bytes() == before


@pytest.fixture
def completed(suite: tuple[list[BuiltExperiment], dict[str, Any]], tmp_path: Path) -> Path:
    experiments, audit = suite
    write_artifacts(experiments, tmp_path)
    (tmp_path / "schedule_audit.json").write_text(json.dumps(audit))
    write_summary(
        [
            sinter.TaskStats(
                strong_id=check["strong_id"],
                decoder="qldpc_bp_osd",
                json_metadata=e.metadata,
                shots=20000,
                errors=0,
                seconds=1,
            )
            for e, check in zip(experiments, audit["experiments"])
        ],
        tmp_path,
    )
    return tmp_path


def test_comparison_including_zero_failures_and_reused_xx(completed: Path) -> None:
    rows = runner.write_comparison(completed)
    assert len(rows) == 30
    assert len([r for r in rows if r["logical_basis"] == "X"]) == 10
    assert all(r["schedule_policy"] == "legacy" for r in rows if r["logical_basis"] == "X")
    for row in rows:
        if row["schedule_policy"] == runner.SCHEDULE:
            assert row["shots"] == 20000
            assert row["logical_error_rate"] == row["ler_95ci_low"] == 0
            assert row["ler_95ci_high"] > 0
    changes = runner._read_csv(completed / "differences.csv")
    assert len(changes) == 10
    assert all(float(r["fisher_p_holm_10"]) >= float(r["fisher_p"]) for r in changes)
    assert (completed / "adapter_ler_vs_rounds_local_order.pdf").stat().st_size > 1000


@pytest.mark.parametrize("defect", ["duplicate", "wrong_setting", "bad_counts"])
def test_reject_bad_summaries(completed: Path, defect: str) -> None:
    rows = runner._read_csv(completed / "summary.csv")
    if defect == "duplicate":
        rows[1] = dict(rows[0])
    elif defect == "wrong_setting":
        rows[0]["preparation_rounds"] = "4"
    else:
        rows[0]["errors"] = "20001"
    runner._write_csv(completed / "summary.csv", rows)
    with pytest.raises(ValueError):
        runner.comparison_rows(completed)


def test_reference_hash_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "REFERENCE", tmp_path)
    reference = tmp_path / "reference.csv"
    reference.write_text("original")
    audit = {"reference_files_sha256": {"reference.csv": runner._sha256(reference.read_bytes())}}
    runner.verify_reference_hashes(audit)
    reference.write_text("changed")
    with pytest.raises(ValueError, match="Reference archive changed"):
        runner.verify_reference_hashes(audit)
