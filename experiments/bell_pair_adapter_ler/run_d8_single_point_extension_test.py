from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import sinter

from . import run_d8_single_point_extension as runner
from .run_experiment import sample_experiments, write_artifacts
from .run_low_p_extension_test import sources as original_sources  # noqa: F401
from .run_physical_noise_comparison_test import FakeProfile


def args_for(path: Path, additional: int = 6, *extra: str) -> Any:
    return runner.parse_args(
        ["--output-dir", str(path), "--additional-shots", str(additional), *extra]
    )


@pytest.fixture
def sources(original_sources: Any) -> Any:  # noqa: F811
    all_experiments = original_sources[0]
    selected = runner.select_experiment(all_experiments)
    seed = sinter.TaskStats(
        strong_id=runner.task_for(selected).strong_id(),
        decoder="qldpc_bp_osd",
        json_metadata=selected.metadata,
        shots=30,
        errors=2,
        seconds=1,
    )
    rows = []
    for e in all_experiments:
        if e is selected:
            stat = seed
        else:
            meta = e.metadata
            # Include the latest high-p counts, so regression to stale plots is tested.
            high = meta["surgery_rounds"] == 8 and meta["physical_error_rate"] == 0.001
            shots = (40000 if meta["left_code"] == "hgp_simplex_m3" else 30000) if high else 100000
            stat = sinter.TaskStats(
                strong_id=runner.task_for(e).strong_id(),
                decoder="qldpc_bp_osd",
                json_metadata=meta,
                shots=shots,
                errors=4,
                seconds=1,
            )
        rows.append(runner.previous.result_row(stat, "latest_archive"))
    identity = dict(
        runtime_identity=runner.runtime_identity(),
        selected_strong_ids=[seed.strong_id],
        seed_csv_sha256=runner.digest(runner.previous.seed_text([seed])),
        reference_rows_sha256=runner.digest(json.dumps(rows, sort_keys=True)),
    )
    return all_experiments, ([selected], [seed], rows, identity, [])


def test_defaults_selection_and_cardinality(sources: Any) -> None:
    args = runner.parse_args([])
    assert args.additional_shots == 500000 and args.num_workers == 8
    assert args.max_batch_size == 32 and args.max_batch_seconds == 5
    selected, seeds, references, _, _ = sources[1]
    assert len(selected) == 1 and selected[0].metadata["physical_error_rate"] == 1e-4
    assert selected[0].metadata["left_code"] == "hgp_simplex_m4"
    assert list(runner.continuation.target_counts(seeds, 6).values()) == [36]
    for bad in ([], [seeds[0], seeds[0]]):
        with pytest.raises(ValueError, match="distinct seeds"):
            runner.continuation.target_counts(bad, 6)
    # Older two-point callers retain their strict default.
    with pytest.raises(ValueError, match="exactly 2"):
        runner.previous.replace_extended_rows(references, seeds, seeds, "new")
    rows = runner.previous.replace_extended_rows(
        references, seeds, seeds, "new", expected_conditions=1
    )
    assert len(rows) == 9
    selected[0].metadata["logical_basis"] = "Z"
    with pytest.raises(ValueError, match="logical_basis"):
        runner.select_experiment(sources[0])
    for extra in (["--additional-shots", "0"], ["--num-workers", "0"], ["--bases", "Z"]):
        with pytest.raises(SystemExit):
            runner.parse_args(extra)


def test_only_one_task_stop_resume_and_other_points_unchanged(
    sources: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected, seeds, references, identity, _ = data = sources[1]
    monkeypatch.setattr(runner, "load_sources", lambda args: data)
    monkeypatch.setattr(runner.continuation, "CollectionProfile", FakeProfile)
    output = tmp_path / "single"
    runner.run(args_for(output, 6, "--build-only"))
    prefix = (output / "sinter_progress.csv").read_bytes()
    calls = []
    stopped = False

    def collect(experiments: Any, **kwargs: Any) -> None:
        nonlocal stopped
        assert experiments == selected and len(experiments) == 1
        assert not kwargs.get("existing_data_filepaths")
        assert kwargs["num_workers"] == 8
        calls.append(kwargs["max_shots_by_strong_id"])
        key = seeds[0].strong_id
        current = runner.continuation.saved_stats(output, selected)[0]
        remaining = kwargs["max_shots_by_strong_id"][key] - current.shots
        count = 2 if not stopped else remaining
        stat = sinter.TaskStats(
            strong_id=key,
            decoder="qldpc_bp_osd",
            json_metadata=selected[0].metadata,
            shots=count,
            errors=0,
            seconds=0.1,
        )
        with (output / "sinter_progress.csv").open("a") as stream:
            stream.write(stat.to_csv_line() + "\n")
        kwargs["progress_callback"](SimpleNamespace(new_stats=[stat]))
        if not stopped:
            stopped = True
            raise KeyboardInterrupt

    monkeypatch.setattr(runner.continuation, "sample_experiments", collect)
    with pytest.raises(KeyboardInterrupt):
        runner.run(args_for(output))
    assert runner.continuation.saved_stats(output, selected)[0].shots == 32
    runner.run(args_for(output))
    runner.run(args_for(output))
    runner.run(args_for(output, 6, "--analyze-only"))
    assert len(calls) == 2 and list(calls[-1].values()) == [36]
    stage = output / "analysis_add6"
    plot = stage / "adapter_ler_vs_physical_error_rate_jackknife_1sigma_rounds8.png"
    original_plot = plot.read_bytes()
    rows = json.loads((stage / "source_comparison_rows.json").read_text())
    for ref, row in zip(references, rows):
        if ref["strong_id"] != seeds[0].strong_id:
            assert row == ref
        else:
            assert row["shots"] == 36 and row["additional_shots"] == 6
    runner.run(args_for(output, 10))
    assert runner.continuation.saved_stats(output, selected)[0].shots == 40
    assert plot.read_bytes() == original_plot
    assert (output / "sinter_progress.csv").read_bytes().startswith(prefix)
    with pytest.raises(ValueError, match="below saved"):
        runner.run(args_for(output, 6))
    (output / "identity.json").write_text(json.dumps({**identity, "changed": True}))
    with pytest.raises(ValueError, match="identity changed"):
        runner.run(args_for(output, 10))


def test_real_single_task_sinter_resume(sources: Any, tmp_path: Path) -> None:
    selected, seeds, _, _, _ = sources[1]
    write_artifacts(selected, tmp_path)
    runner.previous.ensure_seeded_checkpoint(tmp_path, selected, seeds)
    for added in (2, 2, 4):
        targets = runner.continuation.target_counts(seeds, added)
        sample_experiments(
            selected,
            output_dir=tmp_path,
            num_workers=2,
            max_shots=max(targets.values()),
            max_errors=max(targets.values()),
            max_shots_by_strong_id=targets,
            max_batch_size=1,
            print_progress=False,
        )
        result = runner.continuation.saved_stats(tmp_path, selected)
        assert len(result) == 1 and result[0].shots == 30 + added


def test_all_source_directories_protected() -> None:
    args = runner.parse_args([])
    for directory in (
        args.high_extension_dir,
        args.comparison_dir,
        args.d8_source_dir,
        args.previous_extension_dir,
    ):
        with pytest.raises(ValueError, match="separate"):
            runner.run(args_for(directory))
