"""Complete the XX physical-noise comparison: m=3, d=4, eight rounds, p=0.001.

Only this missing point is sampled. Existing comparison data are read-only.
The default 1,000-shot timing pilot is retained in the 20,000-shot total.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any

import sinter
import stim

from qldpc import decoders
from qldpc.objects import Pauli

from ..bell_fidelity.run_ppm_fidelity_sweep import (
    check_output,
    digest,
    runtime_identity,
    saved_stats,
    task_for,
    write_json,
)
from ..common.collection_profile import CollectionProfile
from ..common.run_experiment import (
    BuiltExperiment,
    build_experiment_for_codes,
    sample_experiments,
    write_artifacts,
    write_summary,
)
from ..pilots.run_ppm_basis_first import code_pair, totals
from .run_physical_noise_comparison import (
    CONFIGURATIONS,
    OUTPUTS,
    REFERENCES,
    archived_references,
    result_row,
    scaled_noise_signature,
)

BASELINE = OUTPUTS / "ppm_basis_first_hgp_d4_d8_xx_physical_noise"
DEFAULT_OUTPUT = OUTPUTS / "ppm_basis_first_hgp_m3_d4_xx_prep4_surg8_p0p001"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE)
    parser.add_argument("--max-shots", type=int, default=20000, help="Cumulative, not additional")
    parser.add_argument("--pilot-shots", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--max-batch-seconds", type=int, default=5)
    parser.add_argument(
        "--error-bars",
        choices=("exact95", "jackknife1sigma"),
        default="exact95",
        help="Plot uncertainty only; jackknife mode writes separately named reports",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-only", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args(argv)
    if (
        min(
            args.max_shots,
            args.pilot_shots,
            args.num_workers,
            args.max_batch_size,
            args.max_batch_seconds,
        )
        < 1
    ):
        parser.error("Shot, worker, and batch limits must be positive")
    return args


def load_baselines(directory: Path) -> tuple[list[BuiltExperiment], list[dict[str, Any]]]:
    """Verify the completed six-condition sweep and the two high-p controls."""
    config = json.loads((directory / "run_config.json").read_text())
    if config["runtime_identity"] != runtime_identity():
        raise ValueError("Baseline decoder/dependency identity changed")
    experiments = []
    for meta in json.loads((directory / "manifest.json").read_text()):
        paths = [directory / meta[key] for key in ("circuit_path", "dem_path")]
        if any(not path.resolve().is_relative_to(directory.resolve()) for path in paths):
            raise ValueError("Baseline artifact path escapes its directory")
        metadata = {k: v for k, v in meta.items() if k not in {"circuit_path", "dem_path"}}
        experiments.append(
            BuiltExperiment(
                meta["name"],
                stim.Circuit.from_file(paths[0]),
                stim.DetectorErrorModel.from_file(paths[1]),
                metadata,
            )
        )
    stats = saved_stats(directory, experiments)
    expected = {(m, rounds, p) for m, _, rounds in CONFIGURATIONS for p in (1e-4, 3e-4)}
    rows = [result_row(stat, str(directory)) for stat in stats]
    keys = {(r["simplex_m"], r["surgery_rounds"], r["physical_error_rate"]) for r in rows}
    if len(experiments) != 6 or len(rows) != 6 or keys != expected:
        raise ValueError("Expected the completed six-condition low-p sweep")
    if any(r["shots"] != 100000 or r["logical_basis"] != "X" for r in rows):
        raise ValueError("Expected 100,000 XX shots per low-p condition")
    rows += archived_references(experiments)
    # Check counts against the previously presented plot, without rewriting it.
    with (directory / "analysis_100000" / "comparison.csv").open() as stream:
        snapshot = list(csv.DictReader(stream))
    observed = {(r["strong_id"], r["shots"], r["errors"], r["discards"]) for r in rows}
    expected_counts = {
        (r["strong_id"], int(r["shots"]), int(r["errors"]), int(r["discards"])) for r in snapshot
    }
    if len(snapshot) != 8 or observed != expected_counts:
        raise ValueError("Baseline checkpoint counts differ from the archived plot")
    return experiments, rows


def build_control(baselines: list[BuiltExperiment]) -> tuple[BuiltExperiment, dict[str, Any]]:
    started = time.perf_counter()
    left, right, left_name, right_name = code_pair("hgp-simplex-surface", simplex_m=3)
    experiment = build_experiment_for_codes(
        left_code=left,
        right_code=right,
        left_code_name=left_name,
        right_code_name=right_name,
        left_logical_index=0,
        right_logical_index=0,
        basis=Pauli.X,
        preparation_rounds=4,
        surgery_rounds=8,
        physical_error_rate=0.001,
        bell_pair_infidelity=0.0,
        noise_scope="intrinsic",
        include_idling_error=False,
        adapter_noise_targets="adapter_hardware",
        reduce_detectors=True,
        compute_exact_distances=False,
        schedule_policy="ppm_basis_first",
    )
    signature = scaled_noise_signature(experiment.circuit, 0.001)
    matched = []
    for baseline in baselines:
        meta = baseline.metadata
        if meta["left_code"] != "hgp_simplex_m3" or meta["surgery_rounds"] != 8:
            continue
        for key in meta.keys() | experiment.metadata.keys():
            if key not in {"name", "physical_error_rate"}:
                if meta.get(key) != experiment.metadata.get(key):
                    raise ValueError(f"Eight-round control changed {key}")
        if scaled_noise_signature(baseline.circuit, meta["physical_error_rate"]) != signature:
            raise ValueError("Eight-round control changed operations or noise locations")
        matched.append(task_for(baseline).strong_id())
    if len(matched) != 2:
        raise ValueError("Need both saved low-p eight-round controls")
    detectors, observables = (
        experiment.circuit.without_noise()
        .compile_detector_sampler(seed=731)
        .sample(shots=32, separate_observables=True)
    )
    if detectors.any() or observables.any() or experiment.circuit.num_observables != 1:
        raise ValueError("Noiseless detector/XX observable check failed")
    decoder = decoders.SinterDecoder().compile_decoder_for_dem(experiment.detector_error_model)
    settings = {
        key: getattr(decoder.decoder, key)
        for key in (
            "osd_method",
            "osd_order",
            "bp_method",
            "schedule",
            "max_iter",
        )
    }
    if settings["osd_method"] != "OSD_0" or settings["osd_order"] != 0:
        raise ValueError("Production decoder is no longer order zero")
    audit = dict(
        name=experiment.name,
        scaled_noise_sha256=signature,
        circuit_sha256=digest(str(experiment.circuit)),
        dem_sha256=digest(str(experiment.detector_error_model)),
        matched_baseline_strong_ids=matched,
        noiseless_validation_shots=32,
        decoder_settings=settings,
        build_wall_seconds=time.perf_counter() - started,
    )
    print(f"Validated one XX control: d=4, eight adapter rounds, p=0.001; {settings}", flush=True)
    return experiment, audit


def jackknife_standard_error(errors: int, shots: int) -> float:
    """Exact delete-one-shot jackknife SE of a Bernoulli mean, from counts.

    With x_i in {0, 1}, each leave-one-out estimate is (errors - x_i)/(N-1).
    (N-1)/N times their sum of squared deviations is p_hat*(1-p_hat)/(N-1).
    Each shot is the resampling unit, not a worker or an unequal checkpoint batch.
    The estimate is degenerate (zero) for all-identical outcomes, not proof that
    the true error probability is known exactly. N excludes discarded shots.
    """
    if shots < 2 or not 0 <= errors <= shots:
        raise ValueError("Jackknife requires at least two shots and valid failure counts")
    rate = errors / shots
    return math.sqrt(rate * (1 - rate) / (shots - 1))


def log_log_fit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Jackknife inverse-variance WLS in log10 coordinates, using actual p.

    Propagate SE with sigma_log10 = sigma_LER / (LER * ln(10)). This is a
    first-order approximation, not an exact binomial likelihood fit. Refuse
    zero-rate/zero-uncertainty points instead of inserting a pseudocount.
    With just three points, the exponent is descriptive, not an effective distance.
    """
    points = [(float(r["physical_error_rate"]), float(r["logical_error_rate"])) for r in rows]
    if len(points) < 2 or any(
        not math.isfinite(value) or value <= 0 for point in points for value in point
    ):
        raise ValueError("Log-log fits require at least two finite, strictly positive points")
    sigmas = [
        jackknife_standard_error(r["errors"], r["shots"] - r["discards"]) / (rate * math.log(10))
        for r, (_, rate) in zip(rows, points)
    ]
    if any(not math.isfinite(sigma) or sigma <= 0 for sigma in sigmas):
        raise ValueError("Log-log fits require strictly positive jackknife uncertainty")
    weights = [1 / sigma**2 for sigma in sigmas]
    total_weight = math.fsum(weights)
    xs, ys = zip(*[(math.log10(p), math.log10(rate)) for p, rate in points])
    mean_x = math.fsum(w * x for w, x in zip(weights, xs)) / total_weight
    mean_y = math.fsum(w * y for w, y in zip(weights, ys)) / total_weight
    variance = math.fsum(w * (x - mean_x) ** 2 for w, x in zip(weights, xs))
    if variance == 0:
        raise ValueError("Log-log fits require distinct physical error probabilities")
    slope = (
        math.fsum(w * (x - mean_x) * (y - mean_y) for w, x, y in zip(weights, xs, ys)) / variance
    )
    intercept = mean_y - slope * mean_x
    return dict(
        method="inverse_variance_WLS_log10_LER_vs_log10_p",
        uncertainty="delta_method_leave_one_shot_out_jackknife",
        num_points=len(points),
        power_law_exponent=slope,
        log10_intercept=intercept,
        prefactor=10**intercept,
        p_min=min(p for p, _ in points),
        p_max=max(p for p, _ in points),
    )


def corrected_log_log_fit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Profile the three-parameter model A*p**w*(1+b*p) on three points.

    For any fixed b, log10(A) and w are a weighted linear fit. The remaining
    residual is determined by one curvature contrast. The correction's contrast
    increases from -infinity to a positive maximum, then decreases toward zero.
    Thus positive curvature can have two exact solutions, or no exact solution.
    Find both branches, not just the local optimum nearest an arbitrary guess.
    Positivity is enforced over the measured interval; no extrapolation is made.
    """
    import numpy as np
    from scipy.optimize import brentq, minimize_scalar

    base = log_log_fit(rows)  # Validate positive rates and jackknife uncertainties.
    rows = sorted(rows, key=lambda row: row["physical_error_rate"])
    if len(rows) != 3 or len({r["physical_error_rate"] for r in rows}) != 3:
        raise ValueError("Corrected fits require exactly three distinct physical error rates")
    ps = np.array([r["physical_error_rate"] for r in rows])
    rates = np.array([r["logical_error_rate"] for r in rows])
    xs, ys = np.log10(ps), np.log10(rates)
    sigmas = np.array(
        [
            jackknife_standard_error(r["errors"], r["shots"] - r["discards"])
            / (rate * math.log(10))
            for r, rate in zip(rows, rates)
        ]
    )
    ratios = ps / ps[-1]
    log_complement = np.array([math.log1p(-r) if r < 1 else -math.inf for r in ratios])

    def correction(t: float) -> Any:
        # t = ln(1+b*p_max) parametrizes the entire positive-curve domain.
        # logaddexp is stable even when t is large or near the negative boundary.
        return np.logaddexp(log_complement, np.log(ratios) + t) / math.log(10)

    def contrast(values: Any) -> float:
        return float(np.diff(values)[1] / np.diff(xs)[1] - np.diff(values)[0] / np.diff(xs)[0])

    target = contrast(ys)
    peak = minimize_scalar(
        lambda t: -contrast(correction(t)),
        bounds=(0, math.log1p(100 * ps[-1] / ps[0])),
        method="bounded",
        options={"xatol": 1e-13},
    )
    if not peak.success:
        raise ValueError("Could not locate maximum curvature of the correction model")
    maximum = contrast(correction(peak.x))
    tolerance = 1e-12
    if target > maximum + tolerance:
        candidates = [peak.x]  # Best attainable curvature, not an exact interpolant.
        status = "curvature_limited_weighted_fit"
    elif abs(target - maximum) <= tolerance:
        candidates = [peak.x]
        status = "exact_at_maximum_curvature"
    elif abs(target) <= tolerance:
        candidates = [0.0]  # The other branch is only a b -> infinity limit.
        status = "exact_pure_power_law"
    else:

        def root(t: float) -> float:
            return contrast(correction(t)) - target

        if target > 0:
            high = max(2 * peak.x, 1.0)
            while root(high) > 0 and high < 64:
                high *= 2
            candidates = [brentq(root, 0, peak.x), brentq(root, peak.x, high)]
            status = "two_exact_solutions_smaller_abs_b_shown"
        else:
            low = -1.0
            while root(low) > 0 and low > -64:
                low *= 2
            candidates = [brentq(root, low, 0)]
            status = "one_exact_solution"

    solutions = []
    for t in candidates:
        corr = correction(t)
        exponent, intercept = np.polyfit(xs, ys - corr, 1, w=1 / sigmas)
        residuals = (intercept + exponent * xs + corr - ys) / sigmas
        b = math.expm1(t) / ps[-1]
        solutions.append(
            dict(
                power_law_exponent=float(exponent),
                log10_intercept=float(intercept),
                prefactor=float(10**intercept),
                correction_b=float(b),
                weighted_residual_sum=float(residuals @ residuals),
                max_abs_b_p=float(abs(b) * ps[-1]),
            )
        )
    solutions.sort(key=lambda fit: abs(fit["correction_b"]))
    return {
        **base,
        **solutions[0],
        "method": "jackknife_weighted_log10_profile_fit_A_p_w_1_plus_b_p",
        "fit_status": status,
        "num_parameters": 3,
        "residual_degrees_of_freedom": 0,
        "number_of_finite_solutions": len(solutions),
        "observed_slope_contrast": target,
        "maximum_model_slope_contrast": maximum,
        "alternative_solutions_json": json.dumps(solutions[1:], sort_keys=True),
        "caveat": "Illustration only; no parameter intervals or goodness-of-fit p-value; b*p need not be small",
    }


def write_comparison(
    output: Path,
    stat: sinter.TaskStats,
    references: list[dict[str, Any]],
    *,
    error_bars: str = "exact95",
) -> None:
    rows = [*references, result_row(stat, "new_d4_r8_control")]
    write_comparison_rows(output / f"analysis_{stat.shots}", rows, error_bars=error_bars)


def comparison_count_label(rows: list[dict[str, Any]]) -> str:
    """Describe actual counts, including selective extensions of only some points."""
    low_p = [row for row in rows if row["physical_error_rate"] < 0.001]
    counts = {row["shots"] for row in low_p}
    if len(counts) == 1:
        return f"Low-p points: {next(iter(counts)):,} shots each. High-p counts shown beside each point.\n"
    labels = []
    for p in sorted({row["physical_error_rate"] for row in low_p}):
        selected = sorted(
            [row for row in low_p if row["physical_error_rate"] == p],
            key=lambda row: (row["simplex_m"], row["surgery_rounds"]),
        )
        counts = {row["shots"] for row in selected}
        detail = (
            f"{next(iter(counts)):,} shots each"
            if len(counts) == 1
            else "; ".join(
                f"d{row['nominal_distance']}/{row['surgery_rounds']}r: {row['shots']:,}"
                for row in selected
            )
        )
        labels.append(f"p={p:g}: {detail}.")
    return "\n".join(labels) + " High-p counts shown beside each point.\n"


def write_comparison_rows(
    stage: Path,
    rows: list[dict[str, Any]],
    *,
    error_bars: str = "exact95",
    log_log_fits: bool = False,
    corrected_log_log_fits: bool = False,
    title: str = "HGP simplex / unrotated surface — completed noise grid",
    only_rounds: int | None = None,
    show_fit_caption: bool = True,
    expected_grid: set[tuple[int, int, float]] | None = None,
    show_shot_counts: bool = True,
    show_rounds_in_legend: bool = True,
) -> None:
    """Plot a validated grid (the archived nine-point grid by default)."""
    import matplotlib.pyplot as plt

    expected = expected_grid or {
        (m, rounds, p) for m, _, rounds in CONFIGURATIONS for p in (1e-4, 3e-4, 1e-3)
    }
    keys = {(r["simplex_m"], r["surgery_rounds"], r["physical_error_rate"]) for r in rows}
    if len(rows) != len(expected) or keys != expected:
        raise ValueError(
            "Comparison must contain exactly the expected distinct points (nine by default)"
        )
    if error_bars not in {"exact95", "jackknife1sigma"}:
        raise ValueError("Unknown error-bar convention")
    if log_log_fits and corrected_log_log_fits:
        raise ValueError("Choose either straight or corrected log-log fits, not both")
    configurations = [
        (index, m, distance, rounds)
        for index, (m, distance, rounds) in enumerate(CONFIGURATIONS)
        if only_rounds is None or rounds == only_rounds
    ]
    if not configurations:
        raise ValueError("Requested round count is absent from the comparison")
    if only_rounds is not None:
        rows = [r for r in rows if r["surgery_rounds"] == only_rounds]
    jackknife = error_bars == "jackknife1sigma"
    suffix = "_jackknife_1sigma" if jackknife else ""
    fits = []
    if log_log_fits or corrected_log_log_fits:
        suffix += "_corrected_log_log_fits" if corrected_log_log_fits else "_log_log_fits"
        fit_function = corrected_log_log_fit if corrected_log_log_fits else log_log_fit
        for _, m, distance, rounds in configurations:
            selected = [r for r in rows if r["simplex_m"] == m and r["surgery_rounds"] == rounds]
            fits.append(
                dict(
                    simplex_m=m,
                    nominal_distance=distance,
                    surgery_rounds=rounds,
                    **fit_function(selected),
                )
            )
    if only_rounds is not None:
        suffix += f"_rounds{only_rounds}"
    if jackknife:
        rows = [dict(row) for row in rows]  # Do not mutate the frozen reference metadata.
        for row in rows:
            n = row["shots"] - row["discards"]
            sigma = jackknife_standard_error(row["errors"], n)
            rate = row["logical_error_rate"]
            # A log plot cannot display a zero rate or a lower endpoint <= 0.
            # Refuse instead of fabricating a finite lower cap/upper bound.
            if not 0 < row["errors"] < n or rate - sigma <= 0:
                raise ValueError(
                    "Jackknife log plot needs nondegenerate, strictly positive error bars; "
                    "use exact95 for zero/one-failure or all-failure samples"
                )
            row.update(
                plotted_error_bars="leave_one_shot_out_jackknife_1sigma",
                ler_jackknife_se=sigma,
                ler_jackknife_1sigma_low=rate - sigma,
                ler_jackknife_1sigma_high=rate + sigma,
            )
    stage.mkdir(parents=True, exist_ok=True)
    with (stage / f"comparison{suffix}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    if fits:
        with (stage / f"fit_parameters{suffix}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fits[0]))
            writer.writeheader()
            writer.writerows(fits)
    fig, axis = plt.subplots(figsize=(8.5, 6.5 if corrected_log_log_fits else 6))
    for fit_index, (index, m, distance, rounds) in enumerate(configurations):
        color = f"C{index}"
        label = f"d={distance}"
        if show_rounds_in_legend:
            label += f", {rounds} adapter rounds"
        offset = (0.94, 1, 1.06)[index]
        if fits:
            fit = fits[fit_index]
            exponent = fit["power_law_exponent"]
            label += (
                rf"; $w={exponent:.2f}$"
                if corrected_log_log_fits
                else rf"; $\alpha={exponent:.2f}$"
            )
            # Two endpoints suffice for a pure power law; sample corrected curves.
            # Apply cosmetic x offsets only AFTER fitting actual p.
            ps = [fit["p_min"], fit["p_max"]]
            if corrected_log_log_fits:
                log_min, log_max = (math.log10(p) for p in ps)
                ps = [10 ** (log_min + i * (log_max - log_min) / 200) for i in range(201)]
            predictions = [10 ** (fit["log10_intercept"] + exponent * math.log10(p)) for p in ps]
            if corrected_log_log_fits:
                predictions = [
                    value * (1 + fit["correction_b"] * p) for p, value in zip(ps, predictions)
                ]
                if any(not math.isfinite(value) or value <= 0 for value in predictions):
                    raise ValueError("Corrected fit is not positive and finite over the data range")
            axis.plot(
                [p * offset for p in ps],
                predictions,
                "--",
                color=color,
                linewidth=1.4,
                zorder=1,
            )
        axis.plot([], [], "o--" if fits else "o", color=color, label=label)
        for row in (r for r in rows if r["simplex_m"] == m and r["surgery_rounds"] == rounds):
            x = row["physical_error_rate"] * offset
            y, low, high = (row[k] for k in ("logical_error_rate", "ler_95ci_low", "ler_95ci_high"))
            if jackknife:
                axis.errorbar(
                    x,
                    y,
                    yerr=row["ler_jackknife_se"],
                    fmt="o",
                    color=color,
                    mfc=color,
                    capsize=4,
                )
            elif row["errors"] == 0:
                axis.errorbar(x, high, yerr=high * 0.4, uplims=True, fmt="_", color=color)
            else:
                axis.errorbar(
                    x,
                    y,
                    yerr=[[y - low], [high - y]],
                    fmt="o",
                    color=color,
                    mfc=color,
                    capsize=4,
                )
            if show_shot_counts and row["physical_error_rate"] == 0.001:
                axis.annotate(
                    f"n={row['shots']:,}",
                    (x, y if y else high),
                    xytext=(7, 0),
                    textcoords="offset points",
                    color=color,
                    fontsize=8,
                )
    axis.set(
        xscale="log",
        yscale="log",
        xlabel="Physical error probability p",
        ylabel="XX adapter LER per complete PPM",
        title=title,
    )
    ticks = sorted({r["physical_error_rate"] for r in rows})
    axis.set_xticks(ticks, [f"{p:.3g}" for p in ticks])
    axis.set_xlim(7e-5, 1.8e-3)
    axis.grid(which="both", alpha=0.2)
    axis.legend()
    uncertainty_label = (
        "Production BP+OSD; +/-1 sigma leave-one-shot-out jackknife standard errors.\n"
        if jackknife
        else "Production BP+OSD; exact 95% intervals; arrows: zero-failure upper bounds.\n"
    )
    count_label = comparison_count_label(rows)
    if not show_shot_counts:
        count_label = count_label.replace(
            "High-p counts shown beside each point.", "High-p counts are recorded in the CSV."
        )
    fit_label = (
        "Dashed: jackknife-weighted log-log fits, LER = A p^alpha (3 points/color).\n"
        if fits
        else ""
    )
    if corrected_log_log_fits:
        fit_label = (
            "Dashed: weighted LER = A p^w (1+b p); 3 parameters / 3 points (illustration only).\n"
            "No extrapolation or parameter intervals; when solutions tie, smaller |b| is shown.\n"
            "At largest p, |b p| = "
            + ", ".join(f"{fit['max_abs_b_p']:.2f}" for fit in fits)
            + " ("
            + ", ".join(("blue", "orange", "green")[i] for i, _, _, _ in configurations)
            + "); these corrections are not small.\n"
        )
    if fits and show_fit_caption:
        fig.text(
            0.5,
            0.025,
            count_label
            + "Filled circles throughout; colors distinguish distance and adapter-round count.\n"
            "Four ideal preparation rounds; Bell-source F=1; hardware-only; matching-basis.\n"
            + uncertainty_label
            + fit_label
            + "Fit weights use propagated log-space SE; fits use actual p, before display offsets.",
            ha="center",
            fontsize=8,
        )
        fig.tight_layout(
            rect=(0, 0.18 + 0.025 * (count_label.count("\n") - 1 + fit_label.count("\n")), 1, 1)
        )
    else:
        # Omit the footer and reclaim its reserved whitespace when not requested.
        # Counts, uncertainty definitions, and provenance remain in the CSV/docs.
        fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(stage / f"adapter_ler_vs_physical_error_rate{suffix}.{extension}", dpi=180)
    plt.close(fig)
    print(f"Saved separate {len(rows)}-point comparison ({error_bars}): {stage}", flush=True)


def collect_stage(args: argparse.Namespace, experiment: BuiltExperiment, target: int) -> None:
    output = args.output_dir
    before = totals(saved_stats(output, [experiment]))
    event: dict[str, Any] = dict(
        invocation_id=uuid.uuid4().hex,
        target=target,
        num_workers=args.num_workers,
        before=before,
        started_unix_time=time.time(),
    )

    def log(status: str) -> None:
        with (output / "invocations.jsonl").open("a") as stream:
            stream.write(json.dumps({**event, "status": status}, sort_keys=True) + "\n")

    profile = CollectionProfile(output, event["invocation_id"])
    count, last_print = int(before["shots"]), 0.0

    def progress(update: sinter.Progress) -> None:
        nonlocal count, last_print
        profile.progress(update)
        count += sum(stat.shots for stat in update.new_stats)
        if time.perf_counter() - last_print >= 30:
            print(f"Checkpointed {count:,}/{target:,} XX shots", flush=True)
            last_print = time.perf_counter()

    status = "failed"
    started = time.perf_counter()
    log("sampling_started")
    try:
        profile.start()
        sample_experiments(
            [experiment],
            output_dir=output,
            num_workers=args.num_workers,
            max_shots=target,
            max_errors=target,
            print_progress=False,
            progress_callback=progress,
            max_batch_size=args.max_batch_size,
            max_batch_seconds=args.max_batch_seconds,
        )
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        stats = saved_stats(output, [experiment])
        after = totals(stats)
        if stats:
            write_summary(stats, output)
        event.update(
            after=after,
            new_shots=after["shots"] - before["shots"],
            sampling_decode_wall_seconds=time.perf_counter() - started,
            finished_unix_time=time.time(),
            profile=profile.finish(),
        )
        log(status)
    if after["shots"] != target:
        raise ValueError("Collection ended before reaching the cumulative target")
    remaining = args.max_shots - int(after["shots"])
    timing = {
        **event,
        "remaining_shots": remaining,
        "rough_remaining_wall_seconds": (
            remaining * event["sampling_decode_wall_seconds"] / event["new_shots"]
            if event["new_shots"]
            else None
        ),
        "caveat": "Elapsed sampling/decoding includes startup/tails; estimate is not a guarantee.",
    }
    write_json(output / f"timing_{target}.json", timing)
    print(
        f"Reached {target:,}: {after['errors']} errors; "
        f"{event['new_shots']:,} new shots in {event['sampling_decode_wall_seconds']:.1f}s. "
        f"Rough remaining seconds: {timing['rough_remaining_wall_seconds']}",
        flush=True,
    )


def run(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    args.output_dir = output
    # Never put new collection artifacts or reports inside an old dataset (or its parent).
    for directory in [args.baseline_dir, *(ref[2] for ref in REFERENCES)]:
        directory = directory.resolve()
        if output.is_relative_to(directory) or directory.is_relative_to(output):
            raise ValueError("New output must be separate from all existing datasets")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".collection.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another writer owns the control directory") from error
        if not (output / "manifest.json").exists() and any(
            p.name != ".collection.lock" for p in output.iterdir()
        ):
            raise ValueError("Nonempty output directory without a manifest")
        baselines, references = load_baselines(args.baseline_dir)
        experiment, audit = build_control(baselines)
        check_output(output, [experiment])
        frozen = dict(
            runtime_identity=runtime_identity(),
            baseline_dir=str(args.baseline_dir.resolve()),
            reference_rows_sha256=digest(json.dumps(references, sort_keys=True)),
            strong_id=task_for(experiment).strong_id(),
        )
        identity_path = output / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != frozen:
            raise ValueError("Decoder, reference data, or task identity changed")
        before = totals(saved_stats(output, [experiment]))
        if before["shots"] > args.max_shots:
            raise ValueError("Target is below saved shots; refusing to discard data")
        if args.analyze_only:
            stats = saved_stats(output, [experiment])
            if len(stats) != 1 or stats[0].shots != args.max_shots:
                raise ValueError("Analysis requires the completed target count")
            write_comparison(output, stats[0], references, error_bars=args.error_bars)
            return
        write_artifacts([experiment], output)
        write_json(identity_path, frozen)
        write_json(output / "preflight.json", audit)
        write_json(output / "archived_references.json", references)
        write_json(
            output / "run_config.json",
            {
                **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                **frozen,
                "decoder_kwargs": {},
                "threads": {
                    k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
                },
            },
        )
        if args.build_only:
            return
        for target in sorted({min(args.pilot_shots, args.max_shots), args.max_shots}):
            stats = saved_stats(output, [experiment])
            shots = stats[0].shots if stats else 0
            if shots > target:
                continue  # Resume past the pilot without reducing/resetting the checkpoint.
            if shots < target:
                print(f"Collecting to {target:,} with {args.num_workers} workers TOTAL", flush=True)
                collect_stage(args, experiment, target)
            write_comparison(
                output,
                saved_stats(output, [experiment])[0],
                references,
                error_bars=args.error_bars,
            )
        print(f"Finished at {args.max_shots:,} total XX shots; no automatic extension.", flush=True)


if __name__ == "__main__":
    run(parse_args())
