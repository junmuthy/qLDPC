from __future__ import annotations

import csv
from pathlib import Path

import pytest

from . import plot_hgp_failure_history as history


def test_history_keeps_original_batch_ranges_and_excludes_restart(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.csv"
    rows = [
        (2, 0, 0, history.ORIGINAL_ZZ),
        (1, 1, 0, "unrelated-task"),
        (22, 0, 0, history.RESTART_ZZ),
        (4, 1, 0, history.ORIGINAL_ZZ),
    ]
    with path.open("w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["shots", "errors", "discards", "strong_id"])
        writer.writerows(rows)
    batches, excluded = history.read_history(path)
    assert excluded == (22, 0)
    assert len(batches) == 2
    assert batches[-1] == {
        "source_csv_line": 5,
        "batch_index": 2,
        "recorded_shot_start": 3,
        "recorded_shot_end": 6,
        "batch_shots": 4,
        "batch_errors": 1,
        "cumulative_errors": 1,
    }


def test_archived_history_plot_preserves_checkpoint(tmp_path: Path) -> None:
    before = history.CHECKPOINT.read_bytes()
    history.plot_history(history.CHECKPOINT, tmp_path)
    assert history.CHECKPOINT.read_bytes() == before
    for extension in ("pdf", "png"):
        assert (tmp_path / f"zz_failure_history.{extension}").stat().st_size > 1000
    with (tmp_path / "zz_failure_batches.csv").open() as source:
        batches = list(csv.DictReader(source))
    assert len(batches) == 878
    assert int(batches[-1]["recorded_shot_end"]) == 10000
    assert sum(int(row["batch_errors"]) for row in batches) == 30
    assert sum(int(row["batch_shots"]) for row in batches) == 10000


def test_missing_task_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("shots,errors,discards,strong_id\n1,0,0,other\n")
    with pytest.raises(ValueError, match="absent"):
        history.read_history(path)
