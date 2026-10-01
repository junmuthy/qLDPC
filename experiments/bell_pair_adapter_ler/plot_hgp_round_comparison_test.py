import csv
import json
from pathlib import Path
from typing import Any

import pytest

from . import plot_hgp_round_comparison as comparison


def condition(directory: Path, basis: str, rounds: int) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    meta = dict(
        name=f"{basis}{rounds}",
        left_code="hgp_simplex_m4",
        right_code="surface_unrotated_d8",
        left_logical_index=0,
        right_logical_index=0,
        logical_basis=basis,
        preparation_rounds=4,
        surgery_rounds=rounds,
        rounds=rounds,
        physical_error_rate=0.001,
        bell_pair_infidelity=0.0,
        bell_pair_fidelity=1.0,
        noise_scope="intrinsic",
        adapter_noise_targets="adapter_hardware",
        include_idling_error=False,
        detector_reduction="inactive_rows",
        dem_analysis="undecomposed_exact",
        decoder="qldpc_bp_osd",
        schedule_policy="ppm_basis_first_local_v1",
        surgery_schedule=dict(
            local_order=["X", "Z"] if basis == "X" else ["Z", "X"],
            bell_order=["X", "Z"],
            local_layers_sha256=None,
        ),
    )
    (directory / "manifest.json").write_text(
        json.dumps([{**meta, "circuit_path": "unused.stim", "dem_path": "unused.dem"}])
    )
    # Initial pilot plus extension; the final rate must use both.
    comparison.write_csv(
        directory / "sinter_progress.csv",
        [
            dict(
                shots=n,
                errors=e,
                discards=0,
                strong_id="task",
                decoder="qldpc_bp_osd",
                json_metadata=json.dumps(meta),
            )
            for n, e in ((1000, 4), (9000, 36))
        ],
    )
    comparison.write_csv(
        directory / "summary.csv", [dict(name=meta["name"], shots=10000, errors=40, discards=0)]
    )
    return meta


def test_cumulative_counts_and_intervals(tmp_path: Path) -> None:
    condition(tmp_path, "X", 4)
    row = comparison.read_condition(tmp_path, "X", 4, 10000)
    assert row["shots"] == 10000 and row["errors"] == 40
    assert row["logical_error_rate"] == 0.004
    assert row["ler_95ci_low"] < 0.004 < row["ler_95ci_high"]
    with pytest.raises(ValueError, match="cumulative shots"):
        comparison.read_condition(tmp_path, "X", 4, 20000)
    comparison.write_csv(
        tmp_path / "summary.csv", [dict(name="X4", shots=1000, errors=4, discards=0)]
    )
    with pytest.raises(ValueError, match="summary and checkpoint"):
        comparison.read_condition(tmp_path, "X", 4, 10000)


@pytest.mark.parametrize(
    "key,value",
    [
        ("schedule_policy", "exact_xz_dual_from_matching_x_v1"),
        ("physical_error_rate", 0.002),
        ("preparation_rounds", 8),
    ],
)
def test_rejects_wrong_conditions(tmp_path: Path, key: str, value: object) -> None:
    meta = condition(tmp_path, "Z", 8)
    meta[key] = value
    with pytest.raises(ValueError, match=key):
        comparison.validate_metadata(meta, "Z", 8)


def test_ambiguous_checkpoint_identity(tmp_path: Path) -> None:
    condition(tmp_path, "X", 4)
    with (tmp_path / "sinter_progress.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    rows[1]["strong_id"] = "different task"
    comparison.write_csv(tmp_path / "sinter_progress.csv", rows)
    with pytest.raises(ValueError, match="ambiguous"):
        comparison.read_condition(tmp_path, "X", 4, 10000)


def test_completion_waits_for_final_event(tmp_path: Path) -> None:
    path = tmp_path / "invocations.jsonl"
    event = dict(status="sampling_started", config=dict(max_shots=10000, bases=["X", "Z"]))
    path.write_text(json.dumps(event) + "\n")
    assert not comparison.finished(tmp_path, 10000)
    complete = {**event, "status": "completed", "after": {"shots": 20000}}
    path.write_text(json.dumps(event) + "\n" + json.dumps(complete))
    assert not comparison.finished(tmp_path, 10000)  # Last write not flushed yet.
    path.write_text(json.dumps(complete) + "\n")
    assert comparison.finished(tmp_path, 10000)
    complete["status"] = "interrupted"
    path.write_text(json.dumps(complete) + "\n")
    with pytest.raises(RuntimeError, match="interrupted"):
        comparison.finished(tmp_path, 10000)


def test_plot_zero_failures(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = [
        dict(
            logical_basis=basis,
            surgery_rounds=rounds,
            shots=10000,
            errors=0,
            logical_error_rate=0.0,
            ler_95ci_low=0.0,
            ler_95ci_high=0.000384,
        )
        for basis in "XZ"
        for rounds in (4, 8)
    ]
    timings = [{**r, "worker_seconds_per_shot": r["surgery_rounds"] / 4} for r in rows]
    comparison.plot_results(tmp_path, rows, timings)
    for name in ("adapter_ler_4_vs_8_rounds", "simulation_timing_4_vs_8_rounds"):
        for suffix in ("pdf", "png"):
            assert (tmp_path / f"{name}.{suffix}").stat().st_size > 1000
