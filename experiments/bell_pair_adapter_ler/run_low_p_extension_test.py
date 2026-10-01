from __future__ import annotations

import csv
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import sinter
import stim

from . import run_low_p_extension as runner
from .run_d4_r8_control import comparison_count_label
from .run_experiment import BuiltExperiment, write_artifacts
from .run_physical_noise_comparison_test import FakeProfile


def args_for(path: Path, target: int = 14, *extra: str) -> Any:
    return runner.parse_args(["--output-dir", str(path), "--max-shots", str(target), *extra])


@pytest.fixture
def sources() -> Any:
    experiments = []
    for m, distance, rounds in ((3, 4, 4), (3, 4, 8), (4, 8, 8)):
        for p in (1e-4, 3e-4, 0.001):
            name = f"m{m}_r{rounds}_p{p}"
            circuit = stim.Circuit(
                f"R 0\nX_ERROR({p}) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
            )
            meta = dict(
                name=name,
                left_code=f"hgp_simplex_m{m}",
                right_code=f"surface_unrotated_d{distance}",
                surgery_rounds=rounds,
                rounds=rounds,
                physical_error_rate=p,
                logical_basis="X",
                preparation_rounds=4,
                bell_pair_infidelity=0.0,
                bell_pair_fidelity=1.0,
                num_bell_pair_checks_per_round=1,
                noise_scope="intrinsic",
                adapter_noise_targets="adapter_hardware",
                detector_reduction="inactive_rows",
                include_idling_error=False,
                left_logical_index=0,
                right_logical_index=0,
                schedule_policy="ppm_basis_first_local_v1",
                decoder="qldpc_bp_osd",
                surgery_schedule=dict(
                    local_order=["X", "Z"], bell_order=["X", "Z"], local_layers_sha256=None
                ),
            )
            experiments.append(BuiltExperiment(name, circuit, circuit.detector_error_model(), meta))
    selected = runner.select_experiments(experiments)
    ids = {runner.task_for(e).strong_id() for e in selected}
    stats = [
        sinter.TaskStats(
            strong_id=runner.task_for(e).strong_id(),
            decoder="qldpc_bp_osd",
            json_metadata=e.metadata,
            shots=10 if runner.task_for(e).strong_id() in ids else 100000,
            errors=4,
            seconds=2,
        )
        for e in experiments
    ]
    seeds = [s for s in stats if s.strong_id in ids]
    references = [runner.result_row(s, "source_archive") for s in stats]
    identity = dict(
        runtime_identity=runner.runtime_identity(),
        selected_strong_ids=sorted(ids),
        seed_csv_sha256=runner.digest(runner.seed_text(seeds)),
        reference_rows_sha256=runner.digest(json.dumps(references, sort_keys=True)),
    )
    return experiments, (selected, seeds, references, identity, [])


def test_defaults_selection_and_protocol(sources: Any) -> None:
    args = runner.parse_args([])
    assert args.max_shots == 200000 and args.num_workers == 8
    assert args.max_batch_size == 32 and args.max_batch_seconds == 5
    assert args.log_log_fits is False
    assert runner.parse_args(["--analyze-only", "--log-log-fits"]).log_log_fits is True
    assert runner.parse_args(["--analyze-only", "--corrected-log-log-fits"]).corrected_log_log_fits
    with pytest.raises(SystemExit):
        runner.parse_args(["--log-log-fits", "--corrected-log-log-fits"])
    all_experiments, data = sources
    selected = runner.select_experiments(all_experiments)
    assert selected == data[0] and len(selected) == 2
    assert all(
        e.metadata["surgery_rounds"] == 8 and e.metadata["physical_error_rate"] == 1e-4
        for e in selected
    )
    selected[0].metadata["logical_basis"] = "Z"
    with pytest.raises(ValueError, match="logical_basis"):
        runner.select_experiments(all_experiments)
    for extra in (["--max-shots", "0"], ["--num-workers", "0"], ["--bases", "Z"]):
        with pytest.raises(SystemExit):
            runner.parse_args(extra)


def test_seed_is_idempotent_and_missing_checkpoint_is_not_reset(
    sources: Any, tmp_path: Path
) -> None:
    selected, seeds, _, _, _ = sources[1]
    write_artifacts(selected, tmp_path)
    runner.ensure_seeded_checkpoint(tmp_path, selected, seeds)
    checkpoint = tmp_path / "sinter_progress.csv"
    initial = checkpoint.read_bytes()
    runner.ensure_seeded_checkpoint(tmp_path, selected, seeds)
    assert checkpoint.read_bytes() == initial
    assert [s.shots for s in runner.saved_stats(tmp_path, selected)] == [10, 10]
    # Recover the narrow crash window between the atomic CSV rename and receipt.
    (tmp_path / "seed_receipt.json").unlink()
    runner.ensure_seeded_checkpoint(tmp_path, selected, seeds)
    assert checkpoint.read_bytes() == initial
    checkpoint.unlink()
    with pytest.raises(ValueError, match="refusing to reseed"):
        runner.ensure_seeded_checkpoint(tmp_path, selected, seeds)
    assert not checkpoint.exists()


def test_refuse_tampered_or_unreceipted_seed(sources: Any, tmp_path: Path) -> None:
    selected, seeds, _, _, _ = sources[1]
    text = runner.seed_text(seeds)
    checkpoint = tmp_path / "sinter_progress.csv"
    checkpoint.write_text(text + seeds[0].to_csv_line() + "\n")
    with pytest.raises(ValueError, match="no seed receipt"):
        runner.ensure_seeded_checkpoint(tmp_path, selected, seeds)
    checkpoint.write_text("tampered")
    with pytest.raises(ValueError, match="seed prefix"):
        runner.ensure_seeded_checkpoint(tmp_path, selected, seeds)
    assert checkpoint.read_text() == "tampered"


def test_interrupted_resume_and_extension(
    sources: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected, seeds, references, identity, _ = data = sources[1]
    source_before = json.dumps(references, sort_keys=True)
    monkeypatch.setattr(runner, "load_sources", lambda args: data)
    monkeypatch.setattr(runner, "CollectionProfile", FakeProfile)
    output = tmp_path / "extension"
    runner.run(args_for(output, 14, "--build-only"))
    prefix = (output / "sinter_progress.csv").read_bytes()
    calls = []
    interrupt = True

    def collect(experiments: list[BuiltExperiment], **kwargs: Any) -> None:
        nonlocal interrupt
        assert experiments == selected
        assert not kwargs.get("existing_data_filepaths")  # Seed already lives in the new CSV.
        calls.append(kwargs)
        old = {s.strong_id: s for s in runner.saved_stats(output, selected)}
        for e in experiments:
            strong_id = runner.task_for(e).strong_id()
            count = kwargs["max_shots"] - old[strong_id].shots
            if not count:
                continue
            new = sinter.TaskStats(
                strong_id=strong_id,
                decoder="qldpc_bp_osd",
                json_metadata=e.metadata,
                shots=count,
                errors=0,
                seconds=0.1,
            )
            with (output / "sinter_progress.csv").open("a") as stream:
                stream.write(new.to_csv_line() + "\n")
            kwargs["progress_callback"](SimpleNamespace(new_stats=[new]))
            if interrupt:
                interrupt = False
                raise KeyboardInterrupt

    monkeypatch.setattr(runner, "sample_experiments", collect)
    with pytest.raises(KeyboardInterrupt):
        runner.run(args_for(output))
    assert sorted(s.shots for s in runner.saved_stats(output, selected)) == [10, 14]
    runner.run(args_for(output))
    assert [s.shots for s in runner.saved_stats(output, selected)] == [14, 14]
    completed_plot = output / "analysis_14/adapter_ler_vs_physical_error_rate_jackknife_1sigma.png"
    first_plot = completed_plot.read_bytes()
    runner.run(args_for(output, 16))
    assert completed_plot.read_bytes() == first_plot
    assert [s.shots for s in runner.saved_stats(output, selected)] == [16, 16]
    assert (output / "sinter_progress.csv").read_bytes().startswith(prefix)
    runner.run(args_for(output, 16, "--analyze-only"))
    runner.run(args_for(output, 16))
    assert len(calls) == 3  # Stop/resume, then one extension; no sampling after reaching target.
    assert all(c["num_workers"] == 8 and c["max_batch_size"] == 32 for c in calls)
    assert all(c["max_errors"] == c["max_shots"] for c in calls)
    with (output / "analysis_16/comparison_jackknife_1sigma.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    extended = [r for r in rows if r["additional_shots"]]
    assert len(rows) == 9 and len(extended) == 2
    assert all(int(r["shots"]) == 16 and int(r["additional_shots"]) == 6 for r in extended)
    assert all(int(r["shots"]) == 100000 for r in rows if not r["additional_shots"])
    assert json.dumps(references, sort_keys=True) == source_before
    with pytest.raises(ValueError, match="below saved"):
        runner.run(args_for(output, 14))
    (output / "identity.json").write_text(json.dumps({**identity, "changed": True}))
    with pytest.raises(ValueError, match="identity changed"):
        runner.run(args_for(output, 16))


def test_mixed_count_labels_and_no_double_count(sources: Any) -> None:
    _, seeds, references, _, _ = sources[1]
    extended = [
        sinter.TaskStats(
            strong_id=s.strong_id,
            decoder=s.decoder,
            json_metadata=s.json_metadata,
            shots=200000,
            errors=8,
            seconds=10,
        )
        for s in seeds
    ]
    rows = runner.replace_extended_rows(references, extended, seeds, "extension")
    assert len(rows) == 9
    assert sorted(r["shots"] for r in rows) == [*([100000] * 7), 200000, 200000]
    caption = comparison_count_label(rows)
    assert "d4/4r: 100,000" in caption
    assert "d4/8r: 200,000" in caption and "d8/8r: 200,000" in caption
    assert "p=0.0003: 100,000 shots each" in caption


def test_refuse_foreign_directory_and_concurrent_writer(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="separate"):
        runner.run(args_for(runner.BASELINE))
    (tmp_path / "keep.txt").write_text("unchanged")
    with pytest.raises(ValueError, match="Nonempty"):
        runner.run(args_for(tmp_path))
    assert (tmp_path / "keep.txt").read_text() == "unchanged"
    with (tmp_path / ".collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Another writer"):
            runner.run(args_for(tmp_path))
