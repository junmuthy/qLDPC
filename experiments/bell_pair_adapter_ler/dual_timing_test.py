import json
from pathlib import Path

import pytest

from .dual_timing import timing_row


def test_timing_uses_requested_worker_stage_and_marks_shared_wall(tmp_path: Path) -> None:
    events = []
    batches = []
    for identity, workers, wall in (("old", 4, 100), ("new", 8, 50)):
        events.append(
            dict(
                invocation_id=identity,
                status="completed",
                config={"num_workers": workers},
                new_shots=200,
                sampling_decode_wall_seconds=wall,
                profile=dict(
                    reported_batch_shots=200,
                    resource_profile_error=None,
                    sampled_peak_sum_rss_bytes=123,
                ),
            )
        )
        for basis, seconds in (("X", 80), ("Z", 120)):
            batches.append(
                dict(
                    invocation_id=identity,
                    strong_id=basis,
                    logical_basis=basis,
                    shots=100,
                    worker_seconds=seconds,
                )
            )
    (tmp_path / "invocations.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    (tmp_path / "batch_samples.jsonl").write_text("\n".join(json.dumps(e) for e in batches))
    row = timing_row(tmp_path, basis="Z", strong_id="Z", workers=8, label="ZZ")
    assert row is not None
    assert row["timed_basis_shots"] == 100 and row["worker_seconds_per_shot"] == 1.2
    assert row["collection_wall_seconds"] == 50 and row["collection_total_new_shots"] == 200
    assert row["collection_shots_per_wall_second"] == 4
    assert row["collection_scope"].startswith("shared XX+ZZ")
    assert row["invocation_ids"] == "new"
    assert timing_row(tmp_path, basis="Z", strong_id="Z", workers=16, label="ZZ") is None
    events[1]["new_shots"] = 201
    (tmp_path / "invocations.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    with pytest.raises(ValueError, match="telemetry"):
        timing_row(tmp_path, basis="Z", strong_id="Z", workers=8, label="ZZ")
