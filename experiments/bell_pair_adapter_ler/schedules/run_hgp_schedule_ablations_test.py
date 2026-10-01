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

from ..common.run_experiment import BuiltExperiment, write_artifacts, write_summary
from . import run_hgp_schedule_ablations as runner


@pytest.fixture(scope="module")
def suite(generated_hgp_references: tuple[Path, Path]) -> Any:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, "REFERENCE", generated_hgp_references[0])
        patch.setattr(runner, "DUAL_REFERENCE", generated_hgp_references[1])
        yield runner.build_suite()


def test_six_actual_hgp_variants_and_phase_invariants(
    suite: tuple[list[BuiltExperiment], dict[str, Any]],
) -> None:
    experiments, audit = suite
    assert audit["baselines_match_archives_exactly"]
    assert len(experiments) == len(audit["experiments"]) == 6
    assert {(e.metadata["logical_basis"], e.metadata["schedule_policy"]) for e in experiments} == {
        (b, p) for b in ("X", "Z") for p in runner.POLICIES
    }
    for e, a in zip(experiments, audit["experiments"]):
        assert not a["ordered_dem_matches"]
        assert a["canonical_detector_and_observable_identities_identical"]
        assert a["physical_gate_and_noise_inventory_identical"]
        prefix = runner._ideal_prefix(e.circuit)
        assert prefix.num_measurements == 4 * 122
        active = set(e.metadata["adapter_hardware_qubits"])
        for op in e.circuit.flattened():
            qs = [t.value for t in op.targets_copy()]
            if op.name == "DEPOLARIZE2":
                assert all({a, b} & active for a, b in zip(qs[::2], qs[1::2]))
            elif op.name in {"DEPOLARIZE1", "X_ERROR", "Y_ERROR", "Z_ERROR"}:
                assert set(qs) <= active
            elif op.name in {"M", "MX"} and op.gate_args_copy():
                assert set(qs) <= active
        baseline_ticks = 332 if e.metadata["logical_basis"] == "X" else 328
        if e.metadata["schedule_policy"] != "local_layers":
            assert e.circuit.num_ticks == baseline_ticks
        else:
            assert e.circuit.num_ticks == 660 - baseline_ticks
        compiled = decoders.SinterDecoder().compile_decoder_for_dem(e.detector_error_model)
        dets, obs = (
            e.circuit.without_noise()
            .compile_detector_sampler(seed=173)
            .sample(4, separate_observables=True)
        )
        assert np.array_equal(compiled.decode_shots(dets), obs)


def test_roundtrip_strong_ids_and_archive_preservation(
    suite: tuple[list[BuiltExperiment], dict[str, Any]],
    tmp_path: Path,
) -> None:
    experiments, audit = suite
    before = (runner.REFERENCE / "summary.csv").read_bytes()
    write_artifacts(experiments, tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    for e, row in zip(audit["experiments"], manifest):
        task = sinter.Task(
            circuit=stim.Circuit.from_file(tmp_path / row["circuit_path"]),
            detector_error_model=stim.DetectorErrorModel.from_file(tmp_path / row["dem_path"]),
            decoder="qldpc_bp_osd",
            json_metadata={k: v for k, v in row.items() if k not in {"circuit_path", "dem_path"}},
        )
        assert task.strong_id() == e["strong_id"]
    with pytest.raises(ValueError, match="different experiment"):
        write_artifacts(
            [dataclasses.replace(experiments[0], metadata={"different": True})], tmp_path
        )
    assert (runner.REFERENCE / "summary.csv").read_bytes() == before


def test_comparison_zero_failures_and_multiple_testing(
    suite: tuple[list[BuiltExperiment], dict[str, Any]],
    tmp_path: Path,
) -> None:
    experiments, audit = suite
    write_artifacts(experiments, tmp_path)
    (tmp_path / "schedule_audit.json").write_text(json.dumps(audit))
    for e, a in zip(experiments, audit["experiments"]):
        output = tmp_path / e.metadata["logical_basis"] / e.metadata["schedule_policy"]
        write_artifacts([e], output)
        write_summary(
            [
                sinter.TaskStats(
                    strong_id=a["strong_id"],
                    decoder="qldpc_bp_osd",
                    json_metadata=e.metadata,
                    shots=20000,
                    errors=0,
                    seconds=100,
                )
            ],
            output,
        )
    rows = runner.write_comparison(tmp_path)
    assert len(rows) == 9
    for row in rows:
        if row["schedule_policy"] in runner.POLICIES:
            assert row["logical_error_rate"] == row["ler_95ci_low"] == 0
            assert row["ler_95ci_high"] > 0
    assert (tmp_path / "hgp_schedule_ablation_comparison.pdf").stat().st_size > 1000
    differences = runner._read_csv(tmp_path / "differences.csv")
    assert len(differences) == 6
    assert all(float(r["fisher_p_holm_6"]) >= float(r["fisher_p"]) for r in differences)


def test_holm_adjustment_and_semantic_record_permutation() -> None:
    assert runner.holm_adjust([0.03, 0.001, 0.02]) == [0.04, 0.003, 0.04]
    first = stim.Circuit("R 0 1\nM 0 1\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    second = stim.Circuit("R 0 1\nM 1 0\nDETECTOR rec[-1] rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-2]")
    assert runner._semantic_records(first) == runner._semantic_records(second)
    assert runner._circuit_inventory(first) == runner._circuit_inventory(second)
    second.append("X_ERROR", [0], 0.001)
    assert runner._circuit_inventory(first) != runner._circuit_inventory(second)
