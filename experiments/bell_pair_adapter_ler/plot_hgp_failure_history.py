"""Plot the archived full-surgery ZZ checkpoint history, without resampling."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, StrMethodFormatter

OUTPUTS = Path(__file__).resolve().parent / "outputs"
CHECKPOINT = (
    OUTPUTS
    / "hgp_simplex3_surface_unrotated_d4_intrinsic_prep4_surg4_bell0_10k"
    / "sinter_progress.csv"
)
DEFAULT_OUTPUT = OUTPUTS / "hgp_simplex3_surface_unrotated_d4_full_surgery_zz_failure_history"
ORIGINAL_ZZ = "c95edd6d15d1242ba996a4636541712add7f4ab78d4f50d7d76015b9e6d23ab6"
RESTART_ZZ = "26bde915a2ccfd33debf613c964bc32a1d6f24ac43526de6c502b81c9304971f"


def read_history(checkpoint: Path) -> tuple[list[dict[str, int]], tuple[int, int]]:
    """Preserve append order; a batch range is not an individual failed-shot index."""
    history = []
    total_shots = total_errors = excluded_shots = excluded_errors = 0
    with checkpoint.open(newline="", encoding="utf-8") as source:
        for line, row in enumerate(csv.DictReader(source, skipinitialspace=True), start=2):
            if row["strong_id"] == RESTART_ZZ:
                excluded_shots += int(row["shots"])
                excluded_errors += int(row["errors"])
            if row["strong_id"] != ORIGINAL_ZZ:
                continue
            shots, errors, discards = (int(row[key]) for key in ("shots", "errors", "discards"))
            if shots < 1 or not 0 <= errors <= shots or discards:
                raise ValueError("Expected positive, undiscarded batches with valid error counts")
            start = total_shots + 1
            total_shots += shots
            total_errors += errors
            history.append(
                {
                    "source_csv_line": line,
                    "batch_index": len(history) + 1,
                    "recorded_shot_start": start,
                    "recorded_shot_end": total_shots,
                    "batch_shots": shots,
                    "batch_errors": errors,
                    "cumulative_errors": total_errors,
                }
            )
    if not history:
        raise ValueError("Original ZZ task is absent from checkpoint")
    return history, (excluded_shots, excluded_errors)


def plot_history(checkpoint: Path, output_dir: Path) -> None:
    """Make a batch-resolution event trace and cumulative failures for the saved run."""
    history, excluded = read_history(checkpoint)
    observed = {row["recorded_shot_end"]: row["cumulative_errors"] for row in history}
    # Stop/resume boundaries recovered from the conversation; verify against the archive.
    expected = {22: 0, 1614: 4, 5789: 16, 10000: 30}
    if (
        any(observed.get(shots) != errors for shots, errors in expected.items())
        or history[-1]["recorded_shot_end"] != 10000
        or excluded != (22, 0)
    ):
        raise ValueError("Checkpoint no longer matches the audited 10,000-shot ZZ history")

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "zz_failure_batches.csv").open("w", newline="", encoding="utf-8") as dest:
        writer = csv.DictWriter(dest, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    figure, axes = plt.subplots(
        2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [1, 1.8]}
    )
    figure.suptitle("HGP simplex d=4 – surface d=4: full-surgery ZZ failure history", fontsize=15)
    figure.text(
        0.5,
        0.928,
        "4 ideal preparation + 4 noisy adapter rounds; p=0.001, Bell F=1 | "
        "30 failures / 10,000 shots = 0.300%",
        ha="center",
        fontsize=10,
    )
    segments = [
        (0, 1614, "#edf3f8", "4 failures"),
        (1614, 5789, "#fbf4e7", "12 additional failures"),
        (5789, 10000, "#eef6ed", "14 additional failures"),
    ]
    for axis in axes:
        for start, end, color, _ in segments:
            axis.axvspan(start, end, color=color, zorder=0)
        for boundary in (1614, 5789):
            axis.axvline(boundary, color="0.45", linestyle=":", linewidth=1)
        axis.grid(axis="y", color="0.8", linewidth=0.6, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    for start, end, _, label in segments:
        axes[0].text((start + end) / 2, 1.25, label, ha="center", fontsize=10)

    failures = [row for row in history if row["batch_errors"]]
    axes[0].bar(
        [row["recorded_shot_start"] - 0.5 for row in failures],
        [row["batch_errors"] for row in failures],
        width=[row["batch_shots"] for row in failures],
        align="edge",
        color="#a83c32",
        edgecolor="#a83c32",
        linewidth=0.6,
    )
    axes[0].set_ylim(0, 1.5)
    axes[0].set_yticks([0, 1])
    axes[0].set_ylabel("Failures in batch")
    axes[0].set_title("Each red bar spans a failure-containing batch", loc="left", fontsize=11)

    ends = [0] + [row["recorded_shot_end"] for row in history]
    counts = [0] + [row["cumulative_errors"] for row in history]
    axes[1].step(
        ends,
        counts,
        where="post",
        color="#245b85",
        linewidth=1.8,
        label="Observed (updated at each batch end)",
    )
    axes[1].plot(
        [0, 10000],
        [0, 30],
        "--",
        color="0.45",
        linewidth=1.2,
        label="Constant-rate reference: 0.300%",
    )
    for shots, errors, label in (
        (1614, 4, "Unexpected stop\n1,614 shots; 4 failures"),
        (5789, 16, "Overnight pause\n5,789 shots; 16 failures"),
    ):
        axes[1].plot(shots, errors, "o", color="#245b85", markersize=4)
        axes[1].annotate(
            label,
            (shots, errors),
            xytext=(10, 45 if shots == 1614 else -35),
            textcoords="offset points",
            fontsize=9,
            va="bottom" if shots == 1614 else "top",
        )
    axes[1].set_ylim(-3, 33)
    axes[1].set_yticks(range(0, 31, 5))
    axes[1].set_ylabel("Cumulative logical failures")
    axes[1].set_xlabel("Recorded ZZ shot count (completed-batch order, not wall-clock time)")
    axes[1].set_xlim(0, 10000)
    axes[1].xaxis.set_major_locator(MultipleLocator(1000))
    axes[1].xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes[1].legend(loc="upper left", framealpha=0.95, fontsize=9)
    figure.text(
        0.08,
        0.045,
        "Batch-level data only: the exact failed shot inside each red range is unknown. "
        "Parallel workers report batches in completion order.\n"
        "Excluded accidental fresh-task restart: 22 shots, 0 failures. "
        "Shading marks later stop/resume segments; the corrected restart was at 22 original shots.",
        fontsize=8.5,
        va="bottom",
    )
    figure.subplots_adjust(left=0.08, right=0.975, top=0.86, bottom=0.17, hspace=0.25)
    for extension in ("pdf", "png"):
        figure.savefig(output_dir / f"zz_failure_history.{extension}", dpi=180)
    plt.close(figure)
    print(f"Saved ZZ history to {output_dir}: 10,000 shots, 30 failures, {len(history)} batches")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot_history(CHECKPOINT, args.output_dir)


if __name__ == "__main__":
    main()
