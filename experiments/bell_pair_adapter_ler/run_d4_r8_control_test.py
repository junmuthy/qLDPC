from __future__ import annotations

import csv
import fcntl
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import sinter

from qldpc import codes
from qldpc.codes.common import CSSCode

from . import run_d4_r8_control as runner
from .run_experiment import BuiltExperiment, build_experiment_for_codes
from .run_physical_noise_comparison import exact_interval
from .run_physical_noise_comparison_test import FakeProfile


def args_for(output: Path, shots: int = 4) -> Any:
    return runner.parse_args(
        [
            "--output-dir",
            str(output),
            "--max-shots",
            str(shots),
            "--pilot-shots",
            "2",
        ]
    )


def test_defaults_and_invalid_args() -> None:
    args = runner.parse_args([])
    assert args.num_workers == 8 and args.max_shots == 20000 and args.pilot_shots == 1000
    assert args.max_batch_size == 32 and args.max_batch_seconds == 5
    assert args.error_bars == "exact95"
    for extra in (["--max-shots", "0"], ["--num-workers", "0"], ["--bases", "Z"]):
        with pytest.raises(SystemExit):
            runner.parse_args(extra)


@pytest.fixture(scope="module")
def built() -> tuple[BuiltExperiment, dict[str, Any]]:
    # Exercise the real circuit/DEM builder and production decoder on small blocks.
    # A separate --build-only preflight validates the actual HGP/surface archives.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            runner,
            "code_pair",
            lambda *a, **k: (
                codes.SurfaceCode(3),
                codes.SteaneCode(),
                "hgp_simplex_m3",
                "surface_unrotated_d4",
            ),
        )

        def forbidden(*a: Any, **k: Any) -> None:
            raise AssertionError("Optional exact-distance calculation attempted")

        patch.setattr(CSSCode, "get_distance", forbidden)
        captured = {}

        def capture(**kwargs: Any) -> BuiltExperiment:
            captured.update(kwargs)
            return build_experiment_for_codes(**kwargs)

        patch.setattr(runner, "build_experiment_for_codes", capture)
        with pytest.raises(ValueError, match="both saved"):
            runner.build_control([])
        baselines = [
            build_experiment_for_codes(**{**captured, "physical_error_rate": p})
            for p in (1e-4, 3e-4)
        ]
        experiment, audit = runner.build_control(baselines)
        assert audit["decoder_settings"]["osd_method"] == "OSD_0"
        assert len(audit["matched_baseline_strong_ids"]) == 2
        baselines[0].circuit.append("X_ERROR", [0], [1e-4])
        with pytest.raises(ValueError, match="noise locations"):
            runner.build_control(baselines)
        return experiment, audit


def reference_rows() -> list[dict[str, Any]]:
    rows = []
    for m, distance, rounds in runner.CONFIGURATIONS:
        for p in (1e-4, 3e-4, 1e-3):
            if (m, rounds, p) == (3, 8, 1e-3):
                continue
            shots = 100000 if p < 1e-3 else 20000 if m == 3 else 10000
            errors = 4 if p < 1e-3 else 31 if m == 3 else 21
            low, high = exact_interval(errors, shots)
            rows.append(
                dict(
                    simplex_m=m,
                    nominal_distance=distance,
                    surgery_rounds=rounds,
                    logical_basis="X",
                    physical_error_rate=p,
                    shots=shots,
                    errors=errors,
                    discards=0,
                    logical_error_rate=errors / shots,
                    ler_95ci_low=low,
                    ler_95ci_high=high,
                    source="old_read_only",
                )
            )
    return rows


def test_resume_plot_and_identity_guards(
    built: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment, _ = built
    output = tmp_path / "control"
    references = reference_rows()
    monkeypatch.setattr(runner, "load_baselines", lambda path: ([], references))
    monkeypatch.setattr(runner, "build_control", lambda baselines: built)
    monkeypatch.setattr(runner, "CollectionProfile", FakeProfile)
    calls = []
    interrupt = True

    def collect(experiments: list[BuiltExperiment], **kwargs: Any) -> None:
        nonlocal interrupt
        assert experiments == [experiment]
        calls.append(kwargs)
        path = kwargs["output_dir"] / "sinter_progress.csv"
        stats = runner.saved_stats(output, experiments)
        old = stats[0].shots if stats else 0
        count = kwargs["max_shots"] - old
        stat = sinter.TaskStats(
            strong_id=runner.task_for(experiment).strong_id(),
            decoder="qldpc_bp_osd",
            json_metadata=experiment.metadata,
            shots=count,
            errors=0,
            seconds=count * 0.1,
        )
        with path.open("a") as stream:
            if not stats:
                stream.write(sinter.CSV_HEADER + "\n")
            stream.write(stat.to_csv_line() + "\n")
        kwargs["progress_callback"](SimpleNamespace(new_stats=[stat]))
        if interrupt:
            interrupt = False
            raise KeyboardInterrupt

    monkeypatch.setattr(runner, "sample_experiments", collect)
    with pytest.raises(KeyboardInterrupt):
        runner.run(args_for(output))
    assert runner.saved_stats(output, [experiment])[0].shots == 2
    runner.run(args_for(output))
    assert [c["max_shots"] for c in calls] == [2, 4]
    pilot = output / "analysis_2" / "adapter_ler_vs_physical_error_rate.png"
    pilot_content = pilot.read_bytes()
    runner.run(args_for(output, 6))
    assert [c["max_shots"] for c in calls] == [2, 4, 6]
    assert pilot.read_bytes() == pilot_content
    assert all(c["num_workers"] == 8 and c["max_batch_size"] == 32 for c in calls)
    assert all(c["max_errors"] == c["max_shots"] for c in calls)
    with (output / "analysis_6" / "comparison.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 9
    assert sorted(int(r["shots"]) for r in rows) == [6, 10000, 20000, *([100000] * 6)]
    new = next(r for r in rows if r["source"] == "new_d4_r8_control")
    assert float(new["ler_95ci_high"]) == pytest.approx(exact_interval(0, 6)[1])
    with pytest.raises(ValueError, match="below saved"):
        runner.run(args_for(output, 2))
    identity_path = output / "identity.json"
    identity = json.loads(identity_path.read_text())
    identity["runtime_identity"]["versions"]["ldpc"] = "changed"
    identity_path.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="identity changed"):
        runner.run(args_for(output, 6))


def test_clean_labels_preserve_points_and_uncertainties(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from matplotlib.figure import Figure

    rows = reference_rows()
    control = next(r for r in rows if r["simplex_m"] == 3 and r["surgery_rounds"] == 8)
    rows.append(dict(control, physical_error_rate=1e-3))
    before = json.dumps(rows, sort_keys=True)
    runner.write_comparison_rows(
        tmp_path / "original", rows, error_bars="jackknife1sigma", only_rounds=8
    )
    original_savefig = Figure.savefig

    def savefig(figure: Any, *args: Any, **kwargs: Any) -> Any:
        axis = figure.axes[0]
        assert [t.get_text() for t in axis.get_legend().get_texts()] == ["d=4", "d=8"]
        assert not axis.texts  # No shot-count annotations remain.
        assert len(axis.containers) == 6  # All points and error bars retained.
        assert not figure.texts
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", savefig)
    runner.write_comparison_rows(
        tmp_path / "clean",
        rows,
        error_bars="jackknife1sigma",
        only_rounds=8,
        show_shot_counts=False,
        show_rounds_in_legend=False,
    )
    csv_name = "comparison_jackknife_1sigma_rounds8.csv"
    assert (tmp_path / "original" / csv_name).read_bytes() == (
        tmp_path / "clean" / csv_name
    ).read_bytes()
    assert json.dumps(rows, sort_keys=True) == before


def test_jackknife_matches_explicit_leave_one_out() -> None:
    for n in (2, 3, 7, 20):
        for errors in range(n + 1):
            outcomes = [1] * errors + [0] * (n - errors)
            estimates = [sum(outcomes[:i] + outcomes[i + 1 :]) / (n - 1) for i in range(n)]
            mean = sum(estimates) / n
            expected = math.sqrt(
                (n - 1) / n * sum((estimate - mean) ** 2 for estimate in estimates)
            )
            assert runner.jackknife_standard_error(errors, n) == pytest.approx(expected, abs=1e-15)
    assert runner.jackknife_standard_error(0, 100000) == 0
    assert runner.jackknife_standard_error(100000, 100000) == 0
    for errors, shots in ((0, 0), (0, 1), (-1, 10), (11, 10)):
        with pytest.raises(ValueError, match="Jackknife requires"):
            runner.jackknife_standard_error(errors, shots)


def test_log_log_fit_uses_jackknife_weights() -> None:
    import numpy as np

    rows = [
        dict(physical_error_rate=p, logical_error_rate=k / n, errors=k, shots=n, discards=0)
        for p, k, n in ((1e-4, 14, 200000), (3e-4, 22, 100000), (1e-3, 21, 20000))
    ]
    xs = np.log10([r["physical_error_rate"] for r in rows])
    ys = np.log10([r["logical_error_rate"] for r in rows])
    sigmas = np.array(
        [
            runner.jackknife_standard_error(r["errors"], r["shots"])
            / (r["logical_error_rate"] * math.log(10))
            for r in rows
        ]
    )
    # polyfit weights multiply residuals, so use 1/sigma, not 1/sigma**2.
    expected = np.polyfit(xs, ys, 1, w=1 / sigmas)
    fit = runner.log_log_fit(rows)
    assert fit["power_law_exponent"] == pytest.approx(expected[0])
    assert fit["log10_intercept"] == pytest.approx(expected[1])
    assert fit["prefactor"] == pytest.approx(10 ** expected[1])
    assert abs(fit["power_law_exponent"] - np.polyfit(xs, ys, 1)[0]) > 1e-3
    assert fit["num_points"] == 3
    # Discarded samples are not included in the jackknife denominator.
    discarded = [dict(r, shots=r["shots"] + 13, discards=13) for r in rows]
    assert runner.log_log_fit(discarded) == fit
    for rate in (0.0, -1.0, float("nan"), float("inf")):
        invalid = [dict(rows[0], logical_error_rate=rate), *rows[1:]]
        with pytest.raises(ValueError, match="finite, strictly positive"):
            runner.log_log_fit(invalid)
    with pytest.raises(ValueError, match="distinct"):
        runner.log_log_fit([rows[0], rows[0]])
    with pytest.raises(ValueError, match="jackknife uncertainty"):
        runner.log_log_fit([dict(rows[0], errors=200000, logical_error_rate=1), rows[1]])


@pytest.mark.parametrize("corrected", (False, True))
def test_fitted_plot_preserves_data_and_originals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corrected: bool,
) -> None:
    from matplotlib.axes import Axes

    rows = reference_rows()
    control = next(r for r in rows if r["simplex_m"] == 3 and r["surgery_rounds"] == 8)
    rows.append(dict(control, physical_error_rate=1e-3))
    before = json.dumps(rows, sort_keys=True)
    runner.write_comparison_rows(tmp_path, rows, error_bars="jackknife1sigma")
    originals = {p: p.read_bytes() for p in tmp_path.iterdir()}
    curves = []
    original_plot = Axes.plot

    def plot(axis: Any, *args: Any, **kwargs: Any) -> Any:
        if len(args) == 3 and args[2] == "--":
            curves.append((args[0], args[1], kwargs["color"]))
        return original_plot(axis, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", plot)
    runner.write_comparison_rows(
        tmp_path,
        rows,
        error_bars="jackknife1sigma",
        log_log_fits=not corrected,
        corrected_log_log_fits=corrected,
    )
    assert all(p.read_bytes() == content for p, content in originals.items())
    assert json.dumps(rows, sort_keys=True) == before
    assert len(curves) == 3
    suffix = "corrected_log_log_fits" if corrected else "log_log_fits"
    with (tmp_path / f"fit_parameters_jackknife_1sigma_{suffix}.csv").open() as stream:
        fits = list(csv.DictReader(stream))
    for index, (fit, (xs, ys, color)) in enumerate(zip(fits, curves)):
        assert color == f"C{index}"
        offset = (0.94, 1, 1.06)[index]
        assert [xs[0], xs[-1]] == pytest.approx([1e-4 * offset, 1e-3 * offset])
        assert len(xs) == (201 if corrected else 2)
        assert ys == pytest.approx(
            [
                float(fit["prefactor"])
                * (x / offset) ** float(fit["power_law_exponent"])
                * (1 + float(fit.get("correction_b", 0)) * x / offset)
                for x in xs
            ]
        )
        assert "weighted" in fit["method"] if corrected else "WLS" in fit["method"]
    assert (tmp_path / f"adapter_ler_vs_physical_error_rate_jackknife_1sigma_{suffix}.pdf").exists()


def test_eight_round_plot_only_has_two_weighted_fits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.figure import Figure

    rows = reference_rows()
    control = next(r for r in rows if r["simplex_m"] == 3 and r["surgery_rounds"] == 8)
    rows.append(dict(control, physical_error_rate=1e-3))
    before = json.dumps(rows, sort_keys=True)
    original_savefig = Figure.savefig

    def savefig(figure: Any, *args: Any, **kwargs: Any) -> Any:
        axis = figure.axes[0]
        assert not figure.texts
        assert len(axis.containers) == 6  # Six retained points with error bars.
        labels = [text.get_text() for text in axis.get_legend().get_texts()]
        assert len(labels) == 2 and all("8 adapter rounds" in label for label in labels)
        curves = [
            line for line in axis.lines if line.get_linestyle() == "--" and len(line.get_xdata())
        ]
        assert len(curves) == 2 and {line.get_color() for line in curves} == {"C1", "C2"}
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", savefig)
    runner.write_comparison_rows(
        tmp_path,
        rows,
        error_bars="jackknife1sigma",
        log_log_fits=True,
        only_rounds=8,
        show_fit_caption=False,
    )
    assert json.dumps(rows, sort_keys=True) == before
    with (tmp_path / "comparison_jackknife_1sigma_log_log_fits_rounds8.csv").open() as stream:
        plotted = list(csv.DictReader(stream))
    assert len(plotted) == 6 and all(int(row["surgery_rounds"]) == 8 for row in plotted)
    with (tmp_path / "fit_parameters_jackknife_1sigma_log_log_fits_rounds8.csv").open() as stream:
        fits = list(csv.DictReader(stream))
    assert len(fits) == 2
    for fit in fits:
        selected = [
            r for r in rows if r["simplex_m"] == int(fit["simplex_m"]) and r["surgery_rounds"] == 8
        ]
        expected = runner.log_log_fit(selected)
        assert float(fit["power_law_exponent"]) == pytest.approx(expected["power_law_exponent"])
    with pytest.raises(ValueError, match="round count is absent"):
        runner.write_comparison_rows(tmp_path, rows, only_rounds=16)


@pytest.mark.parametrize(
    "counts,shots,solutions,interpolates",
    [
        ([11, 16, 31], [100000, 100000, 20000], 1, False),
        ([14, 22, 21], [200000, 100000, 20000], 2, True),
        ([9, 12, 21], [200000, 100000, 10000], 1, False),
        ([100, 900, 10000], [100000000] * 3, 1, True),
        ([1000, 700, 50], [100000] * 3, 1, True),
    ],
)
def test_corrected_fit_branches_and_weighted_optimum(
    counts: list[int],
    shots: list[int],
    solutions: int,
    interpolates: bool,
) -> None:
    import numpy as np

    ps = np.array([1e-4, 3e-4, 1e-3])
    rows = [
        dict(
            physical_error_rate=p,
            logical_error_rate=k / n,
            errors=k,
            shots=n,
            discards=0,
        )
        for p, k, n in zip(ps, counts, shots)
    ]
    fit = runner.corrected_log_log_fit(rows)
    assert fit["number_of_finite_solutions"] == solutions
    assert fit["residual_degrees_of_freedom"] == 0
    assert (fit["weighted_residual_sum"] < 1e-10) == interpolates
    ys = np.log10(np.array(counts) / shots)
    sigmas = np.array(
        [
            runner.jackknife_standard_error(k, n) / (k / n * math.log(10))
            for k, n in zip(counts, shots)
        ]
    )
    for solution in [fit, *json.loads(fit["alternative_solutions_json"])]:
        b = solution["correction_b"]
        assert np.all(1 + b * np.geomspace(ps[0], ps[-1], 200) > 0)
        corr = np.log1p(b * ps) / math.log(10)
        expected_w, expected_intercept = np.polyfit(np.log10(ps), ys - corr, 1, w=1 / sigmas)
        assert solution["power_law_exponent"] == pytest.approx(expected_w)
        assert solution["log10_intercept"] == pytest.approx(expected_intercept)
        assert abs(b) >= abs(fit["correction_b"])
    # Check against a broad, independent profile scan, including negative b.
    for t in np.linspace(-12, 16, 300):
        b = np.expm1(t) / ps[-1]
        adjusted = ys - np.log1p(b * ps) / math.log(10)
        coeffs = np.polyfit(np.log10(ps), adjusted, 1, w=1 / sigmas)
        objective = np.sum(((np.polyval(coeffs, np.log10(ps)) - adjusted) / sigmas) ** 2)
        assert fit["weighted_residual_sum"] <= objective + 1e-8
    with pytest.raises(ValueError, match="exactly three"):
        runner.corrected_log_log_fit(rows[:2])


@pytest.mark.parametrize("error_bars", ("exact95", "jackknife1sigma"))
def test_plot_uses_consistent_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_bars: str,
) -> None:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    original_savefig = Figure.savefig

    def savefig(figure: Any, *args: Any, **kwargs: Any) -> Any:
        assert not figure.texts  # Data-only figures have no bottom caption.
        assert figure.axes[0].get_xlabel() == "Physical error probability p"
        assert figure.axes[0].get_legend() is not None
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", savefig)

    original = Axes.errorbar
    marker_styles = []
    plotted_errors = []

    def errorbar(axis: Any, *args: Any, **kwargs: Any) -> Any:
        marker_styles.append((kwargs["fmt"], kwargs["mfc"], kwargs["color"]))
        plotted_errors.append(kwargs["yerr"])
        return original(axis, *args, **kwargs)

    monkeypatch.setattr(Axes, "errorbar", errorbar)
    stat = sinter.TaskStats(
        strong_id="control",
        decoder="qldpc_bp_osd",
        json_metadata=dict(
            left_code="hgp_simplex_m3",
            right_code="surface_unrotated_d4",
            surgery_rounds=8,
            logical_basis="X",
            physical_error_rate=0.001,
        ),
        shots=20000,
        errors=21,
        seconds=1,
    )
    references = reference_rows()
    before = json.dumps(references, sort_keys=True)
    stage = tmp_path / "analysis_20000"
    if error_bars == "jackknife1sigma":
        runner.write_comparison(tmp_path, stat, references)
        preserved = {p: p.read_bytes() for p in stage.iterdir()}
        marker_styles.clear()
        plotted_errors.clear()
    runner.write_comparison(tmp_path, stat, references, error_bars=error_bars)
    assert len(marker_styles) == 9
    assert all(marker == "o" and fill == color for marker, fill, color in marker_styles)
    assert {color for _, _, color in marker_styles} == {"C0", "C1", "C2"}
    assert json.dumps(references, sort_keys=True) == before
    if error_bars == "jackknife1sigma":
        assert all(p.read_bytes() == content for p, content in preserved.items())
        with (stage / "comparison_jackknife_1sigma.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 9
        expected_sigmas = sorted(
            runner.jackknife_standard_error(
                int(row["errors"]), int(row["shots"]) - int(row["discards"])
            )
            for row in rows
        )
        assert sorted(plotted_errors) == pytest.approx(expected_sigmas)
        for row in rows:
            assert row["plotted_error_bars"] == "leave_one_shot_out_jackknife_1sigma"
            assert float(row["ler_jackknife_1sigma_high"]) - float(row["logical_error_rate"]) == (
                pytest.approx(float(row["ler_jackknife_se"]))
            )
        assert (stage / "adapter_ler_vs_physical_error_rate_jackknife_1sigma.pdf").exists()


def test_jackknife_refuses_degenerate_log_bars(tmp_path: Path) -> None:
    references = reference_rows()
    for errors in (0, 1, 20000):
        stat = sinter.TaskStats(
            strong_id="control",
            decoder="qldpc_bp_osd",
            json_metadata=dict(
                left_code="hgp_simplex_m3",
                right_code="surface_unrotated_d4",
                surgery_rounds=8,
                logical_basis="X",
                physical_error_rate=0.001,
            ),
            shots=20000,
            errors=errors,
            seconds=1,
        )
        with pytest.raises(ValueError, match="use exact95"):
            runner.write_comparison(tmp_path, stat, references, error_bars="jackknife1sigma")
    assert not (tmp_path / "analysis_20000").exists()


def test_protect_existing_data_and_concurrent_writer(tmp_path: Path) -> None:
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
