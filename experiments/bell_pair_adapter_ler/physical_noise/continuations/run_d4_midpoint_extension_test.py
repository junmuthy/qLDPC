from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import sinter

from ...common.run_experiment import sample_experiments, write_artifacts
from ..run_log_midpoint_sweep_test import sources as midpoint_sources  # noqa: F401
from ..run_physical_noise_comparison_test import FakeProfile
from . import run_d4_midpoint_extension as runner
from .run_low_p_extension_test import sources as original_sources  # noqa: F401


@pytest.fixture
def data(midpoint_sources: Any) -> Any:  # noqa: F811
    experiments, snapshot = midpoint_sources
    selected = runner.select_experiment(experiments)
    stats = [
        sinter.TaskStats(
            strong_id=runner.task_for(e).strong_id(),
            decoder="qldpc_bp_osd",
            json_metadata=e.metadata,
            shots=100000,
            errors=5,
            seconds=7000,
        )
        for e in experiments
    ]
    seed = next(s for s in stats if s.strong_id == runner.task_for(selected).strong_id())
    rows = snapshot["rows"] + [runner.midpoint.physical.result_row(s, "source") for s in stats]
    identity = dict(
        runtime_identity=runner.runtime_identity(),
        reference_rows_sha256=runner.digest(json.dumps(rows, sort_keys=True)),
    )
    return [selected], [seed], rows, identity, []


def args_for(output: Path, *extra: str) -> Any:
    return runner.parse_args(["--output-dir", str(output), "--additional-shots", "4", *extra])


def test_defaults_and_selection(midpoint_sources: Any) -> None:  # noqa: F811
    args = runner.parse_args([])
    assert args.additional_shots == 100000 and args.num_workers == 8
    selected = runner.select_experiment(midpoint_sources[0])
    assert selected.metadata["left_code"] == "hgp_simplex_m3"
    assert selected.metadata["physical_error_rate"] == runner.midpoint.PROBABILITIES[0]
    assert selected.metadata["surgery_rounds"] == 8
    with pytest.raises(ValueError, match="exactly one"):
        runner.select_experiment([selected, selected])
    selected.metadata["logical_basis"] = "Z"
    with pytest.raises(ValueError, match="logical_basis"):
        runner.select_experiment([selected])


def test_interrupt_resume_and_ten_point_analysis(
    data: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiments, seeds, references, _, _ = data
    monkeypatch.setattr(runner, "load_sources", lambda args: data)
    monkeypatch.setattr(runner.continuation, "CollectionProfile", FakeProfile)
    output = tmp_path / "continuation"
    runner.run(args_for(output, "--build-only"))
    prefix = (output / "sinter_progress.csv").read_bytes()
    interrupted = False
    calls = []

    def sample(selected: Any, **kwargs: Any) -> None:
        nonlocal interrupted
        assert selected == experiments and len(selected) == 1
        assert kwargs["num_workers"] == 8 and not kwargs.get("existing_data_filepaths")
        calls.append(kwargs["max_shots_by_strong_id"])
        current = runner.saved_stats(output, selected)[0]
        count = 2 if not interrupted else kwargs["max_shots"] - current.shots
        stat = sinter.TaskStats(
            strong_id=seeds[0].strong_id,
            decoder="qldpc_bp_osd",
            json_metadata=selected[0].metadata,
            shots=count,
            seconds=0.1,
        )
        with (output / "sinter_progress.csv").open("a") as stream:
            stream.write(stat.to_csv_line() + "\n")
        kwargs["progress_callback"](SimpleNamespace(new_stats=[stat]))
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt

    monkeypatch.setattr(runner.continuation, "sample_experiments", sample)
    with pytest.raises(KeyboardInterrupt):
        runner.run(args_for(output))
    assert runner.saved_stats(output, experiments)[0].shots == 100002
    runner.run(args_for(output))
    assert runner.saved_stats(output, experiments)[0].shots == 100004
    assert (output / "sinter_progress.csv").read_bytes().startswith(prefix)
    runner.run(args_for(output))
    assert calls == [{seeds[0].strong_id: 100004}] * 2
    rows = json.loads((output / "analysis_add4/source_comparison_rows.json").read_text())
    assert len(rows) == 10
    for row in rows:
        if row["strong_id"] == seeds[0].strong_id:
            assert row["shots"] == 100004 and row["errors"] == 5
        else:
            assert row == next(r for r in references if r["strong_id"] == row["strong_id"])
    assert (
        output / "analysis_add4/adapter_ler_vs_physical_error_rate_jackknife_1sigma_rounds8.pdf"
    ).exists()


def test_real_single_task_seed_resume(data: Any, tmp_path: Path) -> None:
    experiments, seeds, _, _, _ = data
    write_artifacts(experiments, tmp_path)
    runner.continuation.previous.ensure_seeded_checkpoint(tmp_path, experiments, seeds)
    for target in (100002, 100003):
        sample_experiments(
            experiments,
            output_dir=tmp_path,
            num_workers=1,
            max_shots=target,
            max_errors=target,
            max_batch_size=1,
            max_batch_seconds=1,
            print_progress=False,
            max_shots_by_strong_id={seeds[0].strong_id: target},
        )
        stats = runner.saved_stats(tmp_path, experiments)
        assert len(stats) == 1 and stats[0].shots == target and stats[0].errors >= 5
