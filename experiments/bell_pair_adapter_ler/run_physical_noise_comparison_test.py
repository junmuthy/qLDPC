from __future__ import annotations

import fcntl
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import sinter
import stim

from qldpc import codes
from qldpc.codes.common import CSSCode

from . import run_physical_noise_comparison as runner
from .run_experiment import BuiltExperiment


def args_for(output: Path, shots: int = 2) -> Any:
    return runner.parse_args(
        [
            "--output-dir",
            str(output),
            "--max-shots",
            str(shots),
            "--without-archived-references",
        ]
    )


def test_defaults_and_invalid_args() -> None:
    args = runner.parse_args(["--output-dir", "unused"])
    assert args.physical_error_rates == [3e-4, 1e-4]
    assert args.num_workers == 4 and args.max_shots == 1000
    assert runner.CONFIGURATIONS == ((3, 4, 4), (3, 4, 8), (4, 8, 8))
    for extra in (
        ["--physical-error-rates", "nan"],
        ["--physical-error-rates", "0"],
        ["--physical-error-rates", "0.001", "0.001"],
        ["--num-workers", "0"],
        ["--max-shots", "0"],
        ["--bases", "Z"],
    ):
        with pytest.raises(SystemExit):
            runner.parse_args(["--output-dir", "unused", *extra])


def test_scaled_noise_signature_and_exact_intervals() -> None:
    def circuit(p: float, target: int = 0) -> stim.Circuit:
        return stim.Circuit(f"R 0 1\nX_ERROR({p}) {target}\nM({p}) 0 1")

    assert runner.scaled_noise_signature(circuit(1e-4), 1e-4) == (
        runner.scaled_noise_signature(circuit(3e-4), 3e-4)
    )
    assert runner.scaled_noise_signature(circuit(1e-4, 1), 1e-4) != (
        runner.scaled_noise_signature(circuit(1e-4), 1e-4)
    )
    with pytest.raises(ValueError, match="Bell-source"):
        runner.scaled_noise_signature(stim.Circuit("PAULI_CHANNEL_1(0.1,0,0) 0"), 0.1)
    assert runner.exact_interval(0, 10000) == pytest.approx((0, 0.0003688199146))
    assert runner.exact_interval(10000, 10000) == pytest.approx((0.9996311800854, 1))
    with pytest.raises(ValueError):
        runner.exact_interval(0, 0)


@pytest.fixture(scope="module")
def built() -> tuple[list[BuiltExperiment], list[dict[str, Any]]]:
    # Real builder/decoder on small blocks; the actual six larger circuits are
    # validated separately by the runner's --build-only preflight.
    with pytest.MonkeyPatch.context() as patch:

        def pair(preset: str, *, simplex_m: int) -> Any:
            return (
                codes.SurfaceCode(3),
                codes.SteaneCode(),
                f"hgp_simplex_m{simplex_m}",
                f"surface_unrotated_d{2 ** (simplex_m - 1)}",
            )

        def forbidden(*unused: Any, **kwargs: Any) -> None:
            raise AssertionError("An optional exact-distance search was attempted")

        patch.setattr(runner, "code_pair", pair)
        patch.setattr(CSSCode, "get_distance", forbidden)
        return runner.build_sweep(args_for(Path("unused")))


def test_build_matrix_and_noise_only_changes(built: Any) -> None:
    experiments, audits = built
    assert len(experiments) == len(audits) == 6
    assert len({runner.task_for(e).strong_id() for e in experiments}) == 6
    for e in experiments:
        meta = e.metadata
        assert meta["logical_basis"] == "X"
        assert meta["preparation_rounds"] == 4 and meta["bell_pair_fidelity"] == 1
        assert meta["adapter_noise_targets"] == "adapter_hardware"
        assert meta["noise_scope"] == "intrinsic"
        assert meta["surgery_schedule"]["local_order"] == ["X", "Z"]
    for m, _, rounds in runner.CONFIGURATIONS:
        selected = [a for a in audits if a["simplex_m"] == m and a["surgery_rounds"] == rounds]
        assert len({a["scaled_noise_sha256"] for a in selected}) == 1
        assert len({a["dem_sha256"] for a in selected}) == 2
        assert all(a["decoder_settings"]["osd_method"] == "OSD_0" for a in selected)


class FakeProfile:
    def __init__(self, *args: Any) -> None:
        pass

    def start(self) -> None:
        pass

    def progress(self, update: Any) -> None:
        pass

    def finish(self) -> dict[str, Any]:
        return {}


def test_checkpoint_resume_extension_and_analysis(
    built: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "sweep"
    monkeypatch.setattr(runner, "build_sweep", lambda args: built)
    monkeypatch.setattr(runner, "CollectionProfile", FakeProfile)
    calls = []
    interrupt = True

    def collect(experiments: list[BuiltExperiment], **kwargs: Any) -> None:
        nonlocal interrupt
        calls.append(kwargs)
        path = kwargs["output_dir"] / "sinter_progress.csv"
        previous = (
            {s.strong_id: s for s in sinter.read_stats_from_csv_files(path)}
            if path.exists()
            else {}
        )
        with path.open("a") as stream:
            if not previous:
                stream.write(sinter.CSV_HEADER + "\n")
            for e in experiments:
                strong_id = runner.task_for(e).strong_id()
                old = previous[strong_id].shots if strong_id in previous else 0
                count = kwargs["max_shots"] - old
                if not count:
                    continue
                stat = sinter.TaskStats(
                    strong_id=strong_id,
                    decoder="qldpc_bp_osd",
                    json_metadata=e.metadata,
                    shots=count,
                    errors=0,
                    seconds=count * 0.1,
                )
                stream.write(stat.to_csv_line() + "\n")
                kwargs["progress_callback"](SimpleNamespace(new_stats=[stat]))
        if interrupt:
            interrupt = False
            raise KeyboardInterrupt

    monkeypatch.setattr(runner, "sample_experiments", collect)
    with pytest.raises(KeyboardInterrupt):
        runner.run(args_for(output))
    assert sum(s.shots for s in runner.saved_stats(output, built[0])) == 6
    assert not (output / "analysis_2").exists()
    first = runner.run(args_for(output))
    assert first["new_shots"] == 6 and first["after"]["shots"] == 12
    pilot_plot = output / "analysis_2" / "adapter_ler_vs_physical_error_rate.png"
    assert pilot_plot.is_file()
    before_plot = pilot_plot.read_bytes()
    resumed = runner.run(args_for(output, shots=4))
    assert resumed["new_shots"] == 12 and resumed["after"]["shots"] == 24
    assert pilot_plot.read_bytes() == before_plot
    assert (output / "analysis_4" / "comparison.csv").is_file()
    assert all(c["num_workers"] == 4 and c["max_batch_size"] == 32 for c in calls)
    assert all(c["max_errors"] == c["max_shots"] for c in calls)
    with pytest.raises(ValueError, match="below saved counts"):
        runner.run(args_for(output, shots=2))
    config = json.loads((output / "run_config.json").read_text())
    config["runtime_identity"]["versions"]["ldpc"] = "changed"
    (output / "run_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="identity changed"):
        runner.run(args_for(output, shots=4))


def test_refuse_foreign_directory_and_concurrent_writer(tmp_path: Path) -> None:
    (tmp_path / "unrelated.txt").write_text("preserve me")
    with pytest.raises(ValueError, match="Nonempty"):
        runner.run(args_for(tmp_path))
    assert (tmp_path / "unrelated.txt").read_text() == "preserve me"
    with (tmp_path / ".collection.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="Another writer"):
            runner.run(args_for(tmp_path))
