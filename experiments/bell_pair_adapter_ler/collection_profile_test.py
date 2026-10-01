import json
from pathlib import Path

import sinter

from .collection_profile import CollectionProfile


def test_collection_profile_records_only_callback_batches(tmp_path: Path) -> None:
    profile = CollectionProfile(tmp_path, "test-invocation")
    profile.start()
    profile.progress(sinter.Progress(new_stats=(), status_message="starting"))
    for shots, errors in ((8, 0), (12, 1)):
        stats = sinter.TaskStats(
            strong_id="task",
            decoder="qldpc_bp_osd",
            json_metadata={"logical_basis": "X"},
            shots=shots,
            errors=errors,
            seconds=shots / 10,
        )
        profile.progress(sinter.Progress(new_stats=(stats,), status_message="running"))
    result = profile.finish()
    assert result["reported_batch_shots"] == 20
    assert result["reported_batches"] == 2
    assert result["reported_batch_worker_seconds"] == 2
    assert result["sampled_peak_sum_rss_bytes"] > 0
    assert result["resource_profile_error"] is None
    assert profile.thread is not None and not profile.thread.is_alive()
    rows = [
        json.loads(line) for line in (tmp_path / "batch_samples.jsonl").read_text().splitlines()
    ]
    assert [r["new_shots_this_invocation"] for r in rows] == [8, 20]
    assert all(r["invocation_id"] == "test-invocation" for r in rows)
    assert (tmp_path / "resource_samples.jsonl").exists()
