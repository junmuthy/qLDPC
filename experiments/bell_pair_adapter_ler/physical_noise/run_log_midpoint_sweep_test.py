from __future__ import annotations

import copy
import fcntl
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import sinter
import stim

from ..common.run_experiment import BuiltExperiment, sample_experiments
from . import run_log_midpoint_sweep as runner
from .continuations.run_low_p_extension_test import sources as original_sources  # noqa: F401
from .run_physical_noise_comparison_test import FakeProfile


def args_for(path: Path, *extra: str) -> Any:
    return runner.parse_args(
        [
            "--output-dir",
            str(path),
            "--pilot-shots",
            "2",
            "--max-shots",
            "6",
            *extra,
        ]
    )


@pytest.fixture
def sources(original_sources: Any) -> Any:  # noqa: F811
    old, (_, _, rows, _, _) = original_sources
    experiments = []
    for p in runner.PROBABILITIES:
        for m in (3, 4):
            source = next(
                e
                for e in old
                if e.metadata["left_code"] == f"hgp_simplex_m{m}"
                and e.metadata["surgery_rounds"] == 8
            )
            meta = copy.deepcopy(source.metadata)
            meta.update(name=f"m{m}_p{p}", physical_error_rate=p)
            meta = json.loads(json.dumps(meta, sort_keys=True))
            circuit = stim.Circuit(
                f"R 0\nX_ERROR({p}) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
            )
            experiments.append(
                BuiltExperiment(meta["name"], circuit, circuit.detector_error_model(), meta)
            )
    return experiments, {"rows": [r for r in rows if r["surgery_rounds"] == 8]}


def test_defaults_and_grid() -> None:
    args = runner.parse_args([])
    assert args.max_shots == 100000 and args.pilot_shots == 1000 and args.num_workers == 8
    assert runner.CONFIGURATIONS == ((3, 4, 8), (4, 8, 8))
    for p, low, high in zip(runner.PROBABILITIES, (1e-4, 3e-4), (3e-4, 1e-3)):
        assert math.log(p / low) == pytest.approx(math.log(high / p))
    for extra in (
        ["--max-shots", "0"],
        ["--num-workers", "0"],
        ["--max-shots", "10"],
        ["--bases", "Z"],
    ):
        with pytest.raises(SystemExit):
            runner.parse_args(extra)


def test_partial_resume_targets_and_frozen_references(
    sources: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiments, references = sources
    monkeypatch.setattr(runner, "build", lambda args: (experiments, []))
    monkeypatch.setattr(runner, "freeze_references", lambda: references)
    monkeypatch.setattr(runner.continuation, "CollectionProfile", FakeProfile)
    calls = []
    interrupt = True

    def sample(selected: list[BuiltExperiment], **kwargs: Any) -> None:
        nonlocal interrupt
        calls.append(kwargs)
        assert len(selected) == 4
        current = {s.strong_id: s for s in runner.with_zero_counts(kwargs["output_dir"], selected)}
        path = kwargs["output_dir"] / "sinter_progress.csv"
        with path.open("a") as stream:
            if path.stat().st_size == 0:
                stream.write(sinter.CSV_HEADER + "\n")
            for index, e in enumerate(selected):
                key = runner.task_for(e).strong_id()
                count = kwargs["max_shots_by_strong_id"][key] - current[key].shots
                if count:
                    stat = sinter.TaskStats(
                        strong_id=key,
                        decoder="qldpc_bp_osd",
                        json_metadata=e.metadata,
                        shots=count,
                        errors=0,
                        seconds=count * 0.1,
                    )
                    stream.write(stat.to_csv_line() + "\n")
                    stream.flush()
                    kwargs["progress_callback"](SimpleNamespace(new_stats=[stat]))
                if interrupt and index == 0:
                    interrupt = False
                    raise KeyboardInterrupt

    monkeypatch.setattr(runner.continuation, "sample_experiments", sample)
    output = tmp_path / "new"
    with pytest.raises(KeyboardInterrupt):
        runner.run(args_for(output))
    assert sum(s.shots for s in runner.saved_stats(output, experiments)) == 2
    frozen = (output / "reference_snapshot.json").read_bytes()
    monkeypatch.setattr(
        runner, "freeze_references", lambda: pytest.fail("Reference sources reloaded on resume")
    )
    runner.run(args_for(output))
    assert all(s.shots == 6 for s in runner.saved_stats(output, experiments))
    assert [c["max_shots"] for c in calls] == [2, 2, 6]
    assert all(c["num_workers"] == 8 and c["max_batch_size"] == 32 for c in calls)
    assert (output / "reference_snapshot.json").read_bytes() == frozen
    assert (
        len(json.loads((output / "analysis_6" / "source_comparison_rows.json").read_text())) == 10
    )
    assert not json.loads((output / "analysis_2" / "timing.json").read_text())[
        "jackknife_plot_available"
    ]
    before = (output / "sinter_progress.csv").read_bytes()
    runner.run(args_for(output))
    assert len(calls) == 3 and (output / "sinter_progress.csv").read_bytes() == before
    with pytest.raises(ValueError, match="identity changed"):
        runner.run(args_for(output, "--max-shots", "7"))
    identity = json.loads((output / "identity.json").read_text())
    identity["runtime_identity"]["versions"]["ldpc"] = "changed"
    (output / "identity.json").write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="identity changed"):
        runner.run(args_for(output))


def test_lock_and_output_protection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="separate"):
        runner.run(args_for(runner.single.DEFAULT_OUTPUT / "unsafe"))
    with (tmp_path / ".collection.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            runner.run(args_for(tmp_path))
    (tmp_path / "unknown.txt").write_text("user file")
    with pytest.raises(ValueError, match="Nonempty"):
        runner.run(args_for(tmp_path))


def test_new_grid_plot_and_missing_conditions(sources: Any, tmp_path: Path) -> None:
    experiments, references = sources
    rows = references["rows"] + [
        runner.physical.result_row(
            sinter.TaskStats(
                strong_id=runner.task_for(e).strong_id(),
                decoder="qldpc_bp_osd",
                json_metadata=e.metadata,
                shots=100000,
                errors=20,
            ),
            "new",
        )
        for e in experiments
    ]
    expected = {(m, 8, p) for m in (3, 4) for p in (1e-4, *runner.PROBABILITIES, 3e-4, 1e-3)}
    runner.write_comparison_rows(
        tmp_path, rows, expected_grid=expected, only_rounds=8, error_bars="jackknife1sigma"
    )
    assert (tmp_path / "adapter_ler_vs_physical_error_rate_jackknife_1sigma_rounds8.pdf").exists()
    with pytest.raises(ValueError, match="distinct"):
        runner.write_comparison_rows(tmp_path, rows[:-1], expected_grid=expected)


def test_real_collection_resumes_without_duplicates(sources: Any, tmp_path: Path) -> None:
    experiments, _ = sources
    for target in (2, 3):
        sample_experiments(
            experiments,
            output_dir=tmp_path,
            num_workers=1,
            max_shots=target,
            max_errors=target,
            print_progress=False,
            max_batch_size=1,
            max_batch_seconds=1,
            max_shots_by_strong_id={runner.task_for(e).strong_id(): target for e in experiments},
        )
        assert all(s.shots == target for s in runner.saved_stats(tmp_path, experiments))
    assert len(runner.saved_stats(tmp_path, experiments)) == 4
