#!/usr/bin/env python3
"""Analyze and plot a Surface/Steane Bell-pair-fidelity sweep."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator, FuncFormatter

RESPONSE_FIELDS = (
    "name",
    "logical_basis",
    "preparation_rounds",
    "surgery_rounds",
    "physical_error_rate",
    "bell_pair_infidelity",
    "bell_pair_fidelity",
    "bell_pairs_consumed",
    "probability_any_bell_pair_error",
    "shots",
    "errors",
    "logical_error_rate",
    "ler_95ci_low",
    "ler_95ci_high",
    "ideal_bell_logical_error_rate",
    "excess_logical_error_rate",
    "excess_ler_95ci_low",
    "excess_ler_95ci_high",
    "relative_ler_overhead",
    "relative_ler_overhead_95ci_low",
    "relative_ler_overhead_95ci_high",
)

REQUIREMENT_FIELDS = (
    "logical_basis",
    "surgery_rounds",
    "allowable_relative_ler_overhead",
    "isotonic_ideal_bell_logical_error_rate",
    "max_tested_infidelity_meeting_point_estimate",
    "min_tested_fidelity_meeting_point_estimate",
    "next_tested_infidelity",
    "next_tested_fidelity",
    "point_estimate_status",
    "max_tested_infidelity_meeting_conservative_95ci",
    "min_tested_fidelity_meeting_conservative_95ci",
)


def read_summary(path: Path) -> list[dict[str, str]]:
    """Read a fidelity-sweep summary and verify the required columns."""
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    required = {
        "name",
        "logical_basis",
        "preparation_rounds",
        "surgery_rounds",
        "physical_error_rate",
        "bell_pair_infidelity",
        "bell_pair_fidelity",
        "num_bell_pair_checks_per_round",
        "shots",
        "errors",
        "logical_error_rate",
        "ler_95ci_low",
        "ler_95ci_high",
    }
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"summary is missing required columns: {sorted(missing)}")
    return rows


def build_response_rows(summary_rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    """Add ideal-pair comparisons and total Bell-pair exposure to summary rows."""
    ideal_rows: dict[tuple[str, int], dict[str, str]] = {}
    for row in summary_rows:
        key = (row["logical_basis"], int(row["surgery_rounds"]))
        if math.isclose(float(row["bell_pair_infidelity"]), 0.0, abs_tol=1e-15):
            if key in ideal_rows:
                raise ValueError(f"multiple ideal-Bell reference rows for {key}")
            ideal_rows[key] = row

    response_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        key = (row["logical_basis"], int(row["surgery_rounds"]))
        if key not in ideal_rows:
            raise ValueError(f"no ideal-Bell reference row for {key}")
        ideal = ideal_rows[key]
        infidelity = float(row["bell_pair_infidelity"])
        fidelity = float(row["bell_pair_fidelity"])
        rounds = int(row["surgery_rounds"])
        bell_pairs = rounds * int(row["num_bell_pair_checks_per_round"])
        ler = float(row["logical_error_rate"])
        low = float(row["ler_95ci_low"])
        high = float(row["ler_95ci_high"])
        ideal_ler = float(ideal["logical_error_rate"])
        ideal_low = float(ideal["ler_95ci_low"])
        ideal_high = float(ideal["ler_95ci_high"])

        if math.isclose(infidelity, 0.0, abs_tol=1e-15):
            excess = excess_low = excess_high = 0.0
            overhead = overhead_low = overhead_high = 0.0
        else:
            excess = ler - ideal_ler
            excess_low = low - ideal_high
            excess_high = high - ideal_low
            overhead = ler / ideal_ler - 1 if ideal_ler else math.nan
            overhead_low = low / ideal_high - 1 if ideal_high else math.nan
            overhead_high = high / ideal_low - 1 if ideal_low else math.inf

        response_rows.append(
            {
                "name": row["name"],
                "logical_basis": row["logical_basis"],
                "preparation_rounds": int(row["preparation_rounds"]),
                "surgery_rounds": rounds,
                "physical_error_rate": float(row["physical_error_rate"]),
                "bell_pair_infidelity": infidelity,
                "bell_pair_fidelity": fidelity,
                "bell_pairs_consumed": bell_pairs,
                "probability_any_bell_pair_error": 1 - fidelity**bell_pairs,
                "shots": int(row["shots"]),
                "errors": int(row["errors"]),
                "logical_error_rate": ler,
                "ler_95ci_low": low,
                "ler_95ci_high": high,
                "ideal_bell_logical_error_rate": ideal_ler,
                "excess_logical_error_rate": excess,
                "excess_ler_95ci_low": excess_low,
                "excess_ler_95ci_high": excess_high,
                "relative_ler_overhead": overhead,
                "relative_ler_overhead_95ci_low": overhead_low,
                "relative_ler_overhead_95ci_high": overhead_high,
            }
        )
    return sorted(
        response_rows,
        key=lambda row: (
            str(row["logical_basis"]),
            int(row["surgery_rounds"]),
            float(row["bell_pair_infidelity"]),
        ),
    )


def weighted_isotonic(values: Sequence[float], weights: Sequence[float]) -> list[float]:
    """Return a nondecreasing weighted least-squares fit using PAVA."""
    if len(values) != len(weights) or not values:
        raise ValueError("values and weights must have the same positive length")
    blocks: list[tuple[int, int, float, float]] = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        if weight <= 0:
            raise ValueError("weights must be positive")
        blocks.append((index, index + 1, value * weight, weight))
        while len(blocks) >= 2:
            previous = blocks[-2]
            current = blocks[-1]
            if previous[2] / previous[3] <= current[2] / current[3]:
                break
            blocks[-2:] = [
                (previous[0], current[1], previous[2] + current[2], previous[3] + current[3])
            ]

    fitted = [0.0] * len(values)
    for start, stop, weighted_sum, total_weight in blocks:
        fitted[start:stop] = [weighted_sum / total_weight] * (stop - start)
    return fitted


def build_requirement_rows(
    response_rows: Sequence[dict[str, Any]],
    relative_overheads: Sequence[float],
) -> list[dict[str, Any]]:
    """Bracket fidelity requirements using a monotone fit and conservative intervals."""
    grouped: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in response_rows:
        grouped[(str(row["logical_basis"]), int(row["surgery_rounds"]))].append(row)

    requirement_rows: list[dict[str, Any]] = []
    for (basis, rounds), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: float(row["bell_pair_infidelity"]))
        fitted = weighted_isotonic(
            [float(row["logical_error_rate"]) for row in ordered],
            [float(row["shots"]) for row in ordered],
        )
        ideal_fit = fitted[0]
        ideal_low = float(ordered[0]["ler_95ci_low"])
        for allowable_overhead in relative_overheads:
            if allowable_overhead < 0:
                raise ValueError("relative LER overheads must be nonnegative")
            target = ideal_fit * (1 + allowable_overhead)
            meeting_indices = [index for index, value in enumerate(fitted) if value <= target]
            last_index = max(meeting_indices)
            next_index = last_index + 1
            if next_index == len(ordered):
                next_infidelity: float | str = ""
                next_fidelity: float | str = ""
                status = "range_not_crossed"
            else:
                next_infidelity = float(ordered[next_index]["bell_pair_infidelity"])
                next_fidelity = float(ordered[next_index]["bell_pair_fidelity"])
                status = "bracketed_by_tested_points"

            conservative_index = 0
            for index, row in enumerate(ordered[1:], start=1):
                if float(row["ler_95ci_high"]) > (1 + allowable_overhead) * ideal_low:
                    break
                conservative_index = index

            requirement_rows.append(
                {
                    "logical_basis": basis,
                    "surgery_rounds": rounds,
                    "allowable_relative_ler_overhead": allowable_overhead,
                    "isotonic_ideal_bell_logical_error_rate": ideal_fit,
                    "max_tested_infidelity_meeting_point_estimate": float(
                        ordered[last_index]["bell_pair_infidelity"]
                    ),
                    "min_tested_fidelity_meeting_point_estimate": float(
                        ordered[last_index]["bell_pair_fidelity"]
                    ),
                    "next_tested_infidelity": next_infidelity,
                    "next_tested_fidelity": next_fidelity,
                    "point_estimate_status": status,
                    "max_tested_infidelity_meeting_conservative_95ci": float(
                        ordered[conservative_index]["bell_pair_infidelity"]
                    ),
                    "min_tested_fidelity_meeting_conservative_95ci": float(
                        ordered[conservative_index]["bell_pair_fidelity"]
                    ),
                }
            )
    return requirement_rows


def _write_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _group_rows(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["logical_basis"]), int(row["surgery_rounds"]))].append(row)
    return dict(grouped)


def _smallest_positive_infidelity(rows: Sequence[dict[str, Any]]) -> float:
    positive = [float(row["bell_pair_infidelity"]) for row in rows if row["bell_pair_infidelity"]]
    return min(positive)


def _format_fidelity_tick(infidelity: float, _position: float) -> str:
    return f"{1 - infidelity:.8f}".rstrip("0").rstrip(".")


def _label_fidelity_axis(axis: Axes, infidelities: Sequence[float]) -> None:
    """Show fidelity values while retaining logarithmic spacing in infidelity."""
    unique = sorted(set(infidelities))
    ticks = [
        infidelity
        for infidelity in unique
        if math.isclose(infidelity, 0.0, abs_tol=1e-15)
        or math.isclose(
            math.log10(infidelity),
            round(math.log10(infidelity)),
            abs_tol=1e-12,
        )
    ]
    if unique and unique[-1] not in ticks:
        ticks.append(unique[-1])
    axis.xaxis.set_major_locator(FixedLocator(ticks))
    axis.xaxis.set_major_formatter(FuncFormatter(_format_fidelity_tick))
    axis.invert_xaxis()
    axis.set_xlabel(r"Bell-pair fidelity $F_B$")


def plot_absolute_ler(rows: Sequence[dict[str, Any]], output_dir: Path) -> None:
    """Plot absolute adapter LER against Bell-pair fidelity with Wilson intervals."""
    grouped = _group_rows(rows)
    bases = sorted({basis for basis, _ in grouped})
    figure, axes = plt.subplots(1, len(bases), figsize=(6.1 * len(bases), 4.6), squeeze=False)
    linthresh = _smallest_positive_infidelity(rows) / 3
    colors = plt.get_cmap("viridis")
    all_rounds = sorted({rounds for _, rounds in grouped})

    for axis, basis in zip(axes[0], bases):
        for color_index, rounds in enumerate(all_rounds):
            group = grouped.get((basis, rounds))
            if not group:
                continue
            ordered = sorted(group, key=lambda row: float(row["bell_pair_infidelity"]))
            x = np.asarray([float(row["bell_pair_infidelity"]) for row in ordered])
            y = np.asarray([float(row["logical_error_rate"]) for row in ordered])
            low = np.asarray([float(row["ler_95ci_low"]) for row in ordered])
            high = np.asarray([float(row["ler_95ci_high"]) for row in ordered])
            color = colors(color_index / max(1, len(all_rounds) - 1))
            axis.errorbar(
                x,
                y,
                yerr=np.vstack((y - low, high - y)),
                marker="o",
                capsize=3,
                color=color,
                label=f"{rounds} surgery rounds",
            )
            ideal = ordered[0]
            axis.axhspan(
                float(ideal["ler_95ci_low"]),
                float(ideal["ler_95ci_high"]),
                color=color,
                alpha=0.09,
            )
        axis.set_xscale("symlog", linthresh=linthresh)
        _label_fidelity_axis(
            axis,
            [float(row["bell_pair_infidelity"]) for row in rows],
        )
        axis.set_yscale("log")
        axis.grid(which="both", alpha=0.25)
        axis.set_title(f"Logical {basis}{basis}")
        axis.legend()
    axes[0][0].set_ylabel("Intrinsic adapter logical error rate")
    figure.suptitle("Surface-3 / Steane adapter sensitivity to Bell-pair fidelity")
    figure.tight_layout()
    figure.savefig(output_dir / "adapter_ler_vs_bell_fidelity.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "adapter_ler_vs_bell_fidelity.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_excess_ler(rows: Sequence[dict[str, Any]], output_dir: Path) -> None:
    """Plot LER in excess of the ideal-Bell baseline with conservative intervals."""
    grouped = _group_rows(rows)
    bases = sorted({basis for basis, _ in grouped})
    figure, axes = plt.subplots(1, len(bases), figsize=(6.1 * len(bases), 4.6), squeeze=False)
    colors = plt.get_cmap("viridis")
    all_rounds = sorted({rounds for _, rounds in grouped})

    for axis, basis in zip(axes[0], bases):
        for color_index, rounds in enumerate(all_rounds):
            group = [
                row
                for row in grouped.get((basis, rounds), ())
                if float(row["bell_pair_infidelity"]) > 0
            ]
            if not group:
                continue
            ordered = sorted(group, key=lambda row: float(row["bell_pair_infidelity"]))
            x = np.asarray([float(row["bell_pair_infidelity"]) for row in ordered])
            y = np.asarray([float(row["excess_logical_error_rate"]) for row in ordered])
            low = np.asarray([float(row["excess_ler_95ci_low"]) for row in ordered])
            high = np.asarray([float(row["excess_ler_95ci_high"]) for row in ordered])
            color = colors(color_index / max(1, len(all_rounds) - 1))
            axis.errorbar(
                x,
                y,
                yerr=np.vstack((y - low, high - y)),
                marker="o",
                capsize=3,
                color=color,
                label=f"{rounds} surgery rounds",
            )
        axis.axhline(0, color="black", linestyle="--", linewidth=1)
        axis.set_xscale("log")
        _label_fidelity_axis(
            axis,
            [
                float(row["bell_pair_infidelity"])
                for row in rows
                if float(row["bell_pair_infidelity"]) > 0
            ],
        )
        axis.grid(which="both", alpha=0.25)
        axis.set_title(f"Logical {basis}{basis}")
        axis.legend()
    axes[0][0].set_ylabel(r"Excess LER $p_L(F_B)-p_L(1)$")
    figure.suptitle("Bell-induced excess adapter logical error rate")
    figure.tight_layout()
    figure.savefig(output_dir / "adapter_excess_ler_vs_bell_fidelity.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "adapter_excess_ler_vs_bell_fidelity.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_fidelity_analysis(
    summary_path: Path,
    output_dir: Path,
    *,
    relative_overheads: Sequence[float] = (0.1, 0.25),
) -> None:
    """Write response tables, engineering requirement brackets, and plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    response_rows = build_response_rows(read_summary(summary_path))
    requirement_rows = build_requirement_rows(response_rows, relative_overheads)
    _write_rows(output_dir / "fidelity_response.csv", RESPONSE_FIELDS, response_rows)
    _write_rows(output_dir / "fidelity_requirements.csv", REQUIREMENT_FIELDS, requirement_rows)
    plot_absolute_ler(response_rows, output_dir)
    plot_excess_ler(response_rows, output_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--relative-overheads", type=float, nargs="+", default=[0.1, 0.25])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir or args.summary.parent
    write_fidelity_analysis(
        args.summary,
        output_dir,
        relative_overheads=args.relative_overheads,
    )
    print(f"fidelity analysis written to {output_dir}")


if __name__ == "__main__":
    main()
