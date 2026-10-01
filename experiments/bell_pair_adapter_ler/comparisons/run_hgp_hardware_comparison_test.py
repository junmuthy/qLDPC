from __future__ import annotations

import csv
from pathlib import Path

import pytest
import stim

from ..common.run_experiment import BuiltExperiment
from . import run_hgp_hardware_comparison as comparison


def test_build_settings_and_baseline_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    circuit = stim.Circuit("R 0\nM 0\nDETECTOR rec[-1]")
    calls = []

    def build(**kwargs: object) -> BuiltExperiment:
        calls.append(kwargs)
        return BuiltExperiment(
            "fixture", circuit, circuit.detector_error_model(), {"num_adapter_hardware_qubits": 1}
        )

    monkeypatch.setattr(comparison, "build_experiment_for_codes", build)
    monkeypatch.setattr(comparison.stim.Circuit, "from_file", lambda path: circuit)
    experiments, audit = comparison.build_hgp_experiments()
    assert len(experiments) == len(audit) == 2
    assert all(row["legacy_dem_matches_explicit_undecomposed_model"] for row in audit)
    assert all(call["compute_exact_distances"] is False for call in calls)
    assert all(call["preparation_rounds"] == call["surgery_rounds"] == 4 for call in calls)
    assert all(call["left_logical_index"] == call["right_logical_index"] == 0 for call in calls)
    assert all(call["physical_error_rate"] == 0.001 for call in calls)
    assert (
        calls[1]["adapter_noise_targets"] == calls[3]["adapter_noise_targets"] == "adapter_hardware"
    )
    assert calls[1]["reduce_detectors"] is calls[3]["reduce_detectors"] is True


@pytest.fixture
def results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    baseline, output = tmp_path / "baseline", tmp_path / "hardware"
    monkeypatch.setattr(comparison, "BASELINE", baseline)
    for policy, directory in (("full_surgery", baseline), ("adapter_hardware", output)):
        directory.mkdir()
        rows = [
            {
                "name": f"hgp_simplex3_98_18_4_surface_unrotated_d4_joint_{basis.lower()}",
                "logical_basis": basis,
                "noise_scope": "intrinsic",
                "decoder": "qldpc_bp_osd",
                "adapter_noise_targets": policy,
                "preparation_rounds": 4,
                "surgery_rounds": 4,
                "physical_error_rate": 0.001,
                "bell_pair_fidelity": 1.0,
                "shots": 10000,
                "discards": 0,
                "errors": 0 if policy == "adapter_hardware" else 30,
                "seconds": 5,
            }
            for basis in ("X", "Z")
        ]
        with (directory / "summary.csv").open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return output


def test_comparison_handles_zero_failures_and_preserves_archive(results: Path) -> None:
    previous = (comparison.BASELINE / "summary.csv").read_bytes()
    rows = comparison.write_comparison(results)
    assert len(rows) == 4
    hardware = [row for row in rows if row["policy"] == "adapter_hardware"]
    assert all(row["logical_error_rate"] == row["ler_95ci_low"] == 0 for row in hardware)
    assert all(row["ler_95ci_high"] > 0 for row in hardware)
    assert (results / "hgp_adapter_hardware_comparison.pdf").stat().st_size > 1000
    assert (comparison.BASELINE / "summary.csv").read_bytes() == previous


def test_comparison_rejects_unrelated_experiments(results: Path) -> None:
    path = results / "summary.csv"
    path.write_text(path.read_text().replace("intrinsic", "end-to-end"))
    with pytest.raises(ValueError, match="identity or noise policy"):
        comparison.write_comparison(results)
