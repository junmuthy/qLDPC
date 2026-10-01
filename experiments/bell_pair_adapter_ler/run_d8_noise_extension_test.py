from __future__ import annotations

import csv
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import sinter

from . import run_d8_noise_extension as runner
from . import run_experiment as common
from .run_low_p_extension_test import sources as original_sources  # noqa: F401
from .run_physical_noise_comparison_test import FakeProfile


def args_for(output: Path, additional: int = 4, *extra: str) -> Any:
    return runner.parse_args(
        [
            "--output-dir",
            str(output),
            "--additional-shots",
            str(additional),
            *extra,
        ]
    )


@pytest.fixture
def sources(original_sources: Any) -> Any:  # noqa: F811
    all_experiments = original_sources[0]
    selected = runner.select_experiments(all_experiments)
    seeds = [
        sinter.TaskStats(
            strong_id=runner.task_for(e).strong_id(),
            decoder="qldpc_bp_osd",
            json_metadata=e.metadata,
            shots=30 if e.metadata["physical_error_rate"] == 1e-4 else 10,
            errors=2,
            seconds=1,
        )
        for e in selected
    ]
    references = original_sources[1][2]
    by_id = {s.strong_id: s for s in seeds}
    references = [
        runner.previous.result_row(by_id[r["strong_id"]], "seed") if r["strong_id"] in by_id else r
        for r in references
    ]
    identity = dict(
        runtime_identity=runner.runtime_identity(),
        selected_strong_ids=sorted(by_id),
        seed_csv_sha256=runner.digest(runner.previous.seed_text(seeds)),
        reference_rows_sha256=runner.digest(json.dumps(references, sort_keys=True)),
    )
    return all_experiments, (selected, seeds, references, identity, [])


def test_selection_protocol_and_targets(sources: Any) -> None:
    args = runner.parse_args([])
    assert args.additional_shots == 200000 and args.num_workers == 8
    assert args.max_batch_size == 32 and args.max_batch_seconds == 5
    selected, seeds, _, _, _ = sources[1]
    assert len(selected) == 2
    assert {e.metadata["physical_error_rate"] for e in selected} == {1e-4, 3e-4}
    assert all(e.metadata["left_code"] == "hgp_simplex_m4" for e in selected)
    targets = runner.target_counts(seeds, 4)
    assert sorted(targets.values()) == [14, 34]
    selected[0].metadata["logical_basis"] = "Z"
    with pytest.raises(ValueError, match="logical_basis"):
        runner.select_experiments(sources[0])
    for extra in (["--additional-shots", "0"], ["--num-workers", "0"], ["--bases", "Z"]):
        with pytest.raises(SystemExit):
            runner.parse_args(extra)


def test_resume_different_targets_and_preserve_other_rows(
    sources: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected, seeds, references, identity, _ = data = sources[1]
    original = json.dumps(references, sort_keys=True)
    monkeypatch.setattr(runner, "load_sources", lambda args: data)
    monkeypatch.setattr(runner, "CollectionProfile", FakeProfile)
    output = tmp_path / "extension"
    runner.run(args_for(output, 4, "--build-only"))
    prefix = (output / "sinter_progress.csv").read_bytes()
    interrupted = False
    calls = []

    def collect(experiments: Any, **kwargs: Any) -> None:
        nonlocal interrupted
        assert experiments == selected
        assert not kwargs.get("existing_data_filepaths")
        targets = kwargs["max_shots_by_strong_id"]
        assert kwargs["max_shots"] == max(targets.values())
        assert kwargs["num_workers"] == 8
        calls.append(targets)
        old = {s.strong_id: s for s in runner.saved_stats(output, selected)}
        for e in experiments:
            key = runner.task_for(e).strong_id()
            remaining = targets[key] - old[key].shots
            if not remaining:
                continue
            stat = sinter.TaskStats(
                strong_id=key,
                decoder="qldpc_bp_osd",
                json_metadata=e.metadata,
                shots=remaining,
                errors=1,
                seconds=0.1,
            )
            with (output / "sinter_progress.csv").open("a") as stream:
                stream.write(stat.to_csv_line() + "\n")
            kwargs["progress_callback"](SimpleNamespace(new_stats=[stat]))
            if not interrupted:
                interrupted = True
                raise KeyboardInterrupt

    monkeypatch.setattr(runner, "sample_experiments", collect)
    with pytest.raises(KeyboardInterrupt):
        runner.run(args_for(output))
    assert sorted(s.shots for s in runner.saved_stats(output, selected)) == [10, 34]
    runner.run(args_for(output))
    assert sorted(s.shots for s in runner.saved_stats(output, selected)) == [14, 34]
    runner.run(args_for(output))
    runner.run(args_for(output, 4, "--analyze-only"))
    assert len(calls) == 2
    stage = output / "analysis_add4"
    png = stage / "adapter_ler_vs_physical_error_rate_jackknife_1sigma_rounds8.png"
    original_png = png.read_bytes()
    rows = json.loads((stage / "source_comparison_rows.json").read_text())
    for ref, row in zip(references, rows):
        if ref["strong_id"] not in calls[0]:
            assert row == ref
        else:
            assert row["shots"] == calls[0][row["strong_id"]]
            assert row["additional_shots"] == 4
    with (stage / "comparison_jackknife_1sigma_rounds8.csv").open() as stream:
        plotted = list(csv.DictReader(stream))
    assert len(plotted) == 6 and all(r["surgery_rounds"] == "8" for r in plotted)
    runner.run(args_for(output, 6))
    assert sorted(s.shots for s in runner.saved_stats(output, selected)) == [16, 36]
    assert png.read_bytes() == original_png
    assert (output / "sinter_progress.csv").read_bytes().startswith(prefix)
    assert json.dumps(references, sort_keys=True) == original
    with pytest.raises(ValueError, match="below saved"):
        runner.run(args_for(output, 4))
    (output / "identity.json").write_text(json.dumps({**identity, "changed": True}))
    with pytest.raises(ValueError, match="identity changed"):
        runner.run(args_for(output, 6))


def test_per_task_limits_keep_strong_ids(
    sources: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected, seeds, _, _, _ = sources[1]
    targets = runner.target_counts(seeds, 4)
    calls = []

    def fake_collect(**kwargs: Any) -> Any:
        calls.append(kwargs)
        assert kwargs["max_shots"] == 34
        assert {t.strong_id(): t.collection_options.max_shots for t in kwargs["tasks"]} == targets
        return []

    monkeypatch.setattr(sinter, "collect", fake_collect)
    kwargs = dict(output_dir=tmp_path, num_workers=8, max_shots=34, max_errors=34)
    common.sample_experiments(selected, **kwargs, max_shots_by_strong_id=targets)
    assert len(calls) == 1
    for limits in ({}, {"foreign": 1}, {k: 35 for k in targets}, {k: 0 for k in targets}):
        with pytest.raises(ValueError, match="shot limits"):
            common.sample_experiments(selected, **kwargs, max_shots_by_strong_id=limits)
    assert len(calls) == 1


def test_real_sinter_unequal_targets_resume(sources: Any, tmp_path: Path) -> None:
    selected, seeds, _, _, _ = sources[1]
    common.write_artifacts(selected, tmp_path)
    runner.previous.ensure_seeded_checkpoint(tmp_path, selected, seeds)
    for additional in (2, 2, 4):
        targets = runner.target_counts(seeds, additional)
        common.sample_experiments(
            selected,
            output_dir=tmp_path,
            num_workers=1,
            max_shots=max(targets.values()),
            max_errors=max(targets.values()),
            max_shots_by_strong_id=targets,
            max_batch_size=1,
            print_progress=False,
        )
        stats = runner.saved_stats(tmp_path, selected)
        assert {s.strong_id: s.shots for s in stats} == targets


def test_source_receipt_is_read_only(original_sources: Any, tmp_path: Path) -> None:  # noqa: F811
    experiments, seeds, refs, identity, _ = original_sources[1]
    common.write_artifacts(experiments, tmp_path)
    runner.write_json(tmp_path / "identity.json", identity)
    runner.write_json(tmp_path / "reference_rows.json", refs)
    runner.previous.ensure_seeded_checkpoint(tmp_path, experiments, seeds)
    with (tmp_path / "sinter_progress.csv").open("a") as stream:
        for s in seeds:
            stream.write(
                sinter.TaskStats(
                    strong_id=s.strong_id,
                    decoder=s.decoder,
                    json_metadata=s.json_metadata,
                    shots=300000 - s.shots,
                    errors=1,
                ).to_csv_line()
                + "\n"
            )
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    stats = runner.read_previous_extension(tmp_path, experiments, seeds, identity)
    assert [s.shots for s in stats] == [300000, 300000]
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    (tmp_path / "seed_receipt.json").unlink()
    with pytest.raises(FileNotFoundError):
        runner.read_previous_extension(tmp_path, experiments, seeds, identity)
    assert not (tmp_path / "seed_receipt.json").exists()


def test_locks_source_isolation_and_counts(sources: Any, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="separate"):
        runner.run(args_for(runner.previous.DEFAULT_OUTPUT))
    (tmp_path / "keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="Nonempty"):
        runner.run(args_for(tmp_path))
    with (tmp_path / ".collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Another writer"):
            runner.run(args_for(tmp_path))
    targets = runner.target_counts(sources[1][1], 4)
    with pytest.raises(ValueError, match="must be complete"):
        runner.validate_counts(sources[1][1], targets, complete=True)
    with pytest.raises(ValueError, match="exactly"):
        runner.validate_counts([], targets)
