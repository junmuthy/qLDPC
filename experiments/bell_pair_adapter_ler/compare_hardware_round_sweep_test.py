from __future__ import annotations

import json
from pathlib import Path

import pytest
import stim

from . import compare_hardware_round_sweep as comparison
from .run_experiment import BuiltExperiment


@pytest.fixture
def suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(comparison, "OUTPUTS", tmp_path / "old")
    output = tmp_path / "new"
    for infidelity, slug, baseline in comparison.SETTINGS:
        for hardware, directory in ((False, comparison.OUTPUTS / baseline), (True, output / slug)):
            directory.mkdir(parents=True)
            rows, manifest = [], []
            for basis in ("X", "Z"):
                for rounds in range(1, 6):
                    name = f"{basis}{rounds}"
                    rows.append(
                        {
                            "name": name,
                            "logical_basis": basis,
                            "surgery_rounds": rounds,
                            "preparation_rounds": 3,
                            "physical_error_rate": 0.001,
                            "bell_pair_infidelity": infidelity,
                            "noise_scope": "intrinsic",
                            "decoder": "qldpc_bp_osd",
                            "shots": 10000,
                            "errors": 0 if hardware else 10,
                            "discards": 0,
                            "seconds": 5,
                        }
                    )
                    metadata = {
                        "name": name,
                        "left_code": "rotated_surface_d3",
                        "right_code": "steane_d3",
                        "left_logical_index": 0,
                        "right_logical_index": 0,
                        "include_idling_error": False,
                        "num_detectors": 50,
                    }
                    if hardware:
                        metadata.update(
                            adapter_noise_targets="adapter_hardware",
                            detector_reduction="inactive_rows",
                        )
                    manifest.append(metadata)
            comparison.write_csv(directory / "summary.csv", rows)
            (directory / "manifest.json").write_text(json.dumps(manifest))
    directory = output / "full_surgery_exact"
    directory.mkdir()
    full_rows, full_manifest = [], []
    for _, slug, baseline in comparison.SETTINGS:
        baseline_dir = comparison.OUTPUTS / baseline
        for row in comparison.read_csv(baseline_dir / "summary.csv"):
            row["name"] = slug + row["name"]
            full_rows.append(row)
        for metadata in json.loads((baseline_dir / "manifest.json").read_text()):
            metadata.update(
                name=slug + metadata["name"],
                adapter_noise_targets="full_surgery",
                detector_reduction="none",
                dem_analysis="undecomposed_exact",
            )
            full_manifest.append(metadata)
    comparison.write_csv(directory / "summary.csv", full_rows)
    (directory / "manifest.json").write_text(json.dumps(full_manifest))
    return output


def test_matching_and_zero_error_plot(suite: Path) -> None:
    rows = comparison.comparison_rows(suite)
    assert len(rows) == 60
    assert rows[0]["logical_error_rate"] == 0.001
    assert rows[0]["worker_seconds_per_10000_shots"] == 5
    assert all(row["ler_95ci_low"] == 0 for row in rows if row["policy"] == "hardware_reduced")
    comparison.plot_ler(rows, suite)
    assert (suite / "adapter_ler_vs_rounds_comparison.pdf").stat().st_size > 1000


@pytest.mark.parametrize(
    "field,value",
    [
        ("shots", 500),
        ("preparation_rounds", 2),
        ("bell_pair_infidelity", 0.01),
        ("noise_scope", "end-to-end"),
    ],
)
def test_rejects_mismatched_or_unfinished_data(suite: Path, field: str, value: object) -> None:
    path = suite / "bell0" / "summary.csv"
    rows = comparison.read_csv(path)
    rows[0][field] = str(value)
    comparison.write_csv(path, rows)
    with pytest.raises(ValueError, match="Unfinished|Mismatched"):
        comparison.comparison_rows(suite)


def test_rejects_duplicate_points(suite: Path) -> None:
    path = suite / "bell0" / "summary.csv"
    rows = comparison.read_csv(path)
    comparison.write_csv(path, rows + rows[:1])
    with pytest.raises(ValueError, match="duplicate"):
        comparison.comparison_rows(suite)


def test_rejects_unmatched_dem_analysis(suite: Path) -> None:
    path = suite / "full_surgery_exact" / "manifest.json"
    rows = json.loads(path.read_text())
    rows[0].pop("dem_analysis")
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="Mismatched"):
        comparison.comparison_rows(suite)


def test_reference_run_passes_explicit_dem_without_changing_legacy_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = BuiltExperiment(
        "original", stim.Circuit(), stim.DetectorErrorModel(), {"name": "original"}
    )
    monkeypatch.setattr(comparison, "build_fidelity_sweep_matrix", lambda **kwargs: [experiment])
    captured = []

    def collect(experiments: list[BuiltExperiment], **kwargs: object) -> list:
        captured.extend(experiments)
        assert kwargs["max_shots"] == 10000
        assert kwargs["max_errors"] == 200
        return []

    monkeypatch.setattr(comparison, "sample_experiments", collect)
    comparison.sample_matched_baseline(tmp_path, num_workers=2)
    assert captured[0].metadata["adapter_noise_targets"] == "full_surgery"
    assert captured[0].metadata["dem_analysis"] == "undecomposed_exact"
    assert captured[0].name == "original_exact_dem"
    assert experiment.metadata == {"name": "original"}


def test_timing_summary_separates_policy_and_reduction_gain(tmp_path: Path) -> None:
    samples = []
    for infidelity, _, _ in comparison.SETTINGS:
        for basis in ("X", "Z"):
            for rounds in range(1, 6):
                for policy, duration in zip(comparison.POLICIES, (6, 3, 2)):
                    for repeat in range(3):
                        samples.append(
                            {
                                "logical_basis": basis,
                                "surgery_rounds": rounds,
                                "bell_pair_infidelity": infidelity,
                                "policy": policy,
                                "shots": 256,
                                "repeat": repeat,
                                "decode_ms_per_shot": duration,
                                "num_detectors": 50,
                                "num_decoder_error_variables": 500,
                                "paired_prediction_disagreements": 0,
                            }
                        )
    comparison.write_csv(tmp_path / "timing_samples.csv", samples)
    rows = comparison.summarize_timings(tmp_path)
    reduced = [r for r in rows if r["policy"] == "hardware_reduced"]
    assert len(reduced) == 20
    assert all(r["speedup_vs_full_surgery"] == 3 for r in reduced)
    assert all(r["reduction_only_speedup"] == 1.5 for r in reduced)
    assert (tmp_path / "adapter_decode_time_vs_rounds.pdf").stat().st_size > 1000
