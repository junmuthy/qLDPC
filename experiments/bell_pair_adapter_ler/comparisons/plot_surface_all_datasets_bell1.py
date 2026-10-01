"""Replot all four Surface–Steane production round-sweep datasets at Bell F=1.

This reads saved counts only: no circuits are built and no shots are sampled.
Keep the automatic-DEM full-surgery archive, exact-DEM full-surgery reference,
original hardware schedule, and new hardware local-order schedule distinct.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.bell_pair_adapter_ler.common.run_experiment import _wilson_interval

from ..common.paths import OUTPUT_ROOT

OUTPUTS = OUTPUT_ROOT
ORIGINAL = OUTPUTS / "intrinsic_prep3_round_sweep_1_5"
HARDWARE = OUTPUTS / "adapter_hardware_prep3_round_sweep_1_5_comparison" / "bell0"
MATCHED = HARDWARE.parent / "full_surgery_exact"
LOCAL_ORDER = OUTPUTS / "surface_steane_hardware_local_order_prep3_rounds1_5_20k"
DESTINATION = OUTPUTS / "surface_steane_all_datasets_rounds1_5_bell1"
DATASETS = (
    ("original_full_surgery", ORIGINAL, "A: Original full surgery — automatic DEM", "X,Z"),
    ("matched_full_surgery", MATCHED, "B: Matched full surgery — undecomposed DEM", "X,Z"),
    ("original_hardware", HARDWARE, "C: Original hardware-only — local X→Z", "X,Z"),
    ("local_order_hardware", LOCAL_ORDER, "D: New hardware-only — local Z→X (ZZ only)", "Z,X"),
)


def source_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(OUTPUTS)): hashlib.sha256(path.read_bytes()).hexdigest()
        for _, directory, _, _ in DATASETS
        for path in (directory / "summary.csv", directory / "manifest.json")
    }


def load_comparison() -> list[dict[str, Any]]:
    """Validate source identities/settings, then compute rates from their actual counts."""
    result = []
    series = [
        (basis, policy, directory, label, local_order)
        for policy, directory, label, local_order in DATASETS
        for basis in (("Z",) if directory == LOCAL_ORDER else ("X", "Z"))
    ]
    for basis, policy, directory, label, local_order in series:
        manifest = {r["name"]: r for r in json.loads((directory / "manifest.json").read_text())}
        with (directory / "summary.csv").open(newline="") as stream:
            rows = [
                r
                for r in csv.DictReader(stream)
                if r["logical_basis"] == basis and float(r["bell_pair_infidelity"]) == 0
            ]
        if len(rows) != 5 or sorted(int(r["surgery_rounds"]) for r in rows) != list(range(1, 6)):
            raise ValueError(f"Expected exactly five F=1 round counts: {directory}, {basis}")
        full_surgery = policy in ("original_full_surgery", "matched_full_surgery")
        noise_targets = "full_surgery" if full_surgery else "adapter_hardware"
        reduction = "none" if full_surgery else "inactive_rows"
        for row in sorted(rows, key=lambda r: int(r["surgery_rounds"])):
            meta = manifest[row["name"]]
            for entry in (meta, row):
                if not (
                    entry["logical_basis"] == basis
                    and int(entry["surgery_rounds"]) == int(row["surgery_rounds"])
                    and int(entry["preparation_rounds"]) == 3
                    and float(entry["physical_error_rate"]) == 0.001
                    and float(entry["bell_pair_infidelity"]) == 0
                    and entry["noise_scope"] == "intrinsic"
                    and entry["decoder"] == "qldpc_bp_osd"
                    and entry.get("adapter_noise_targets", "full_surgery") == noise_targets
                    and entry.get("detector_reduction", "none") == reduction
                ):
                    raise ValueError(f"Unexpected source settings: {row['name']}")
            if not (
                meta["left_code"] == "rotated_surface_d3"
                and meta["right_code"] == "steane_d3"
                and meta["left_logical_index"] == meta["right_logical_index"] == 0
                and meta["include_idling_error"] is False
                and meta["bell_pair_fidelity"] == 1
            ):
                raise ValueError(f"Wrong code pair or noise settings: {row['name']}")
            if directory in (MATCHED, LOCAL_ORDER) and meta["dem_analysis"] != "undecomposed_exact":
                raise ValueError("Expected an explicitly undecomposed DEM")
            if directory == LOCAL_ORDER and meta["surgery_schedule"] != {
                "local_order": ["Z", "X"],
                "bell_order": ["X", "Z"],
                "local_layers_sha256": None,
            }:
                raise ValueError("ZZ must change only the local family order")
            shots, discards, errors = (int(row[k]) for k in ("shots", "discards", "errors"))
            if not (0 <= discards < shots and 0 <= errors <= shots - discards):
                raise ValueError(f"Invalid counts: {row['name']}")
            rate = errors / (shots - discards)
            low, high = _wilson_interval(errors, shots - discards)
            for key, value in zip(
                ("logical_error_rate", "ler_95ci_low", "ler_95ci_high"), (rate, low, high)
            ):
                if not math.isclose(float(row[key]), value, abs_tol=1e-14):
                    raise ValueError(f"Saved rate/interval disagrees with counts: {row['name']}")
            result.append(
                dict(
                    series=label,
                    logical_basis=basis,
                    policy=policy,
                    local_order=local_order,
                    bell_order="X,Z",
                    bell_pair_fidelity=1.0,
                    preparation_rounds=3,
                    surgery_rounds=int(row["surgery_rounds"]),
                    physical_error_rate=0.001,
                    noise_scope="intrinsic",
                    adapter_noise_targets=noise_targets,
                    dem_analysis="sinter_auto"
                    if policy == "original_full_surgery"
                    else "undecomposed_exact",
                    detector_reduction=reduction,
                    shots=shots,
                    discards=discards,
                    errors=errors,
                    logical_error_rate=rate,
                    ler_95ci_low=low,
                    ler_95ci_high=high,
                    source_name=row["name"],
                    source_summary=str((directory / "summary.csv").relative_to(OUTPUTS)),
                )
            )
    return result


def plot_comparison(rows: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8), sharex=True, sharey=True)
    for axis, basis in zip(axes, ("X", "Z")):
        for policy, color, marker, style in (
            ("original_full_surgery", "#555555", "o", "--"),
            ("matched_full_surgery", "#0072B2", "D", "-."),
            ("original_hardware", "#D55E00", "s", "-"),
            ("local_order_hardware", "#009E73", "^", "-"),
        ):
            selected = [r for r in rows if r["logical_basis"] == basis and r["policy"] == policy]
            if not selected:
                continue  # The new schedule reuses XX; do not duplicate those samples.
            x = np.array([r["surgery_rounds"] for r in selected])
            y, low, high = (
                np.array([r[key] for r in selected]) * 100
                for key in ("logical_error_rate", "ler_95ci_low", "ler_95ci_high")
            )
            nonzero = y > 0
            axis.errorbar(
                x[nonzero],
                y[nonzero],
                yerr=np.maximum([y[nonzero] - low[nonzero], high[nonzero] - y[nonzero]], 0),
                color=color,
                marker=marker,
                linestyle=style,
                capsize=4,
                label=selected[0]["series"],
            )
            if np.any(~nonzero):
                axis.errorbar(
                    x[~nonzero],
                    high[~nonzero],
                    yerr=0.2 * high[~nonzero],
                    uplims=True,
                    fmt=marker,
                    color=color,
                    capsize=4,
                )
        axis.set_title(f"{basis}{basis} logical parity measurement")
        axis.set_xlabel("Adapter syndrome rounds")
        axis.set_xticks(range(1, 6))
        axis.set_xlim(0.8, 5.2)
        axis.set_yscale("log")
        axis.grid(which="both", alpha=0.22)
    axes[0].set_ylabel("Adapter LER per PPM (%)")
    fig.suptitle(
        "Surface d=3 – Steane: all production round-sweep datasets\n"
        "Bell fidelity F=1; physical p=0.001; 3 ideal preparation rounds"
    )
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.07), ncol=2, fontsize=9)
    fig.text(
        0.5,
        0.025,
        "Pointwise 95% Wilson intervals. Both hardware datasets use undecomposed DEMs.\n"
        "D reuses C's XX samples (not plotted twice). All Bell extraction orders remain X→Z.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.195, 1, 0.91))
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"adapter_ler_all_datasets_bell1.{suffix}", dpi=180)
    plt.close(fig)


def main() -> None:
    hashes = source_hashes()
    rows = load_comparison()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with (DESTINATION / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_comparison(rows, DESTINATION)
    if source_hashes() != hashes:
        raise RuntimeError("Source data changed during plotting")
    (DESTINATION / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(f"Plotted {len(rows)} saved estimates (no new sampling): {DESTINATION}")


if __name__ == "__main__":
    main()
