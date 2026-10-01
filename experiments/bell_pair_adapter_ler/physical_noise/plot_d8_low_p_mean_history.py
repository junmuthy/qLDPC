"""Read-only cumulative LER history for the d=8, eight-round XX p=1e-4 point.

Stitch append-order checkpoints, excluding their verified imported seed rows.
The last source can be live: only its complete-line prefix is captured, without
locking or modifying the collector. These are batch endpoints, not per-shot data.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

from ..common.paths import OUTPUT_ROOT

OUTPUTS = OUTPUT_ROOT
STRONG_ID = "ebc359e13097938c90b7bbade7f1cfa3b692e292332cd0f9082c56456229f820"
SOURCE_NAMES = (
    "ppm_basis_first_hgp_d4_d8_xx_physical_noise",
    "ppm_basis_first_hgp_d4_d8_xx_p1e4_surg8_extension_200000",
    "ppm_basis_first_hgp_d8_xx_p1e4_3e4_surg8_add200000",
    "ppm_basis_first_hgp_d8_xx_p1e4_surg8_add500000",
)
DEFAULT_OUTPUT = OUTPUTS / "ppm_basis_first_hgp_d8_xx_p1e4_surg8_mean_history"


def stitch_history(
    directories: list[Path], strong_id: str = STRONG_ID
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    history, provenance = [], []
    shots = failures = 0
    metadata = None
    last_metadata_text = None
    for stage, directory in enumerate(directories):
        checkpoint = directory / "sinter_progress.csv"
        raw = checkpoint.read_bytes()
        end = raw.rfind(b"\n") + 1
        captured = raw[:end]
        if not captured:
            raise ValueError("Checkpoint has no complete lines")
        receipt_path = directory / "seed_receipt.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        if (stage == 0 and receipt is not None) or (stage > 0 and receipt is None):
            raise ValueError(
                "History must start with raw batches and have a receipt for each continuation"
            )
        if receipt is not None:
            prefix = b"".join(captured.splitlines(keepends=True)[: 1 + len(receipt["seed_counts"])])
            if (
                hashlib.sha256(prefix.decode().strip().encode()).hexdigest()
                != receipt["seed_csv_sha256"]
            ):
                raise ValueError("Imported seed prefix hash differs from its receipt")
            if receipt["seed_counts"].get(strong_id) != dict(shots=shots, errors=failures):
                raise ValueError("Imported seed does not match accumulated prior history")
        selected_rows = added_shots = added_failures = 0
        for line, row in enumerate(
            csv.DictReader(io.StringIO(captured.decode()), skipinitialspace=True), start=2
        ):
            if row["strong_id"] != strong_id:
                continue
            if row["decoder"] != "qldpc_bp_osd":
                raise ValueError("Decoder identity changed")
            if row["json_metadata"] != last_metadata_text:
                current = json.loads(row["json_metadata"])
                if metadata is None:
                    metadata = current
                    expected = dict(
                        left_code="hgp_simplex_m4",
                        right_code="surface_unrotated_d8",
                        physical_error_rate=1e-4,
                        surgery_rounds=8,
                        logical_basis="X",
                    )
                    if any(current.get(k) != v for k, v in expected.items()):
                        raise ValueError("Wrong experiment selected")
                elif current != metadata:
                    raise ValueError("Experiment metadata changed between batches")
                last_metadata_text = row["json_metadata"]
            n, k, discarded = (int(row[key]) for key in ("shots", "errors", "discards"))
            if n < 1 or not 0 <= k <= n or discarded:
                raise ValueError("Expected positive, undiscarded batches with valid failures")
            selected_rows += 1
            if receipt is not None and selected_rows == 1:
                if (n, k) != (shots, failures) or line > 1 + len(receipt["seed_counts"]):
                    raise ValueError("Expected the imported aggregate as the first selected row")
                continue  # This seed is already represented in the preceding stages.
            start = shots + 1
            shots += n
            failures += k
            added_shots += n
            added_failures += k
            rate = failures / shots
            se = math.sqrt(rate * (1 - rate) / (shots - 1)) if shots > 1 else None
            history.append(
                dict(
                    source=str(checkpoint.resolve()),
                    source_csv_line=line,
                    batch_index=len(history) + 1,
                    stage_index=stage,
                    batch_shot_start=start,
                    cumulative_shots=shots,
                    batch_shots=n,
                    batch_failures=k,
                    cumulative_failures=failures,
                    cumulative_ler=rate,
                    jackknife_1sigma=se,
                )
            )
        if not selected_rows:
            raise ValueError("Selected task is absent from checkpoint")
        provenance.append(
            dict(
                source=str(checkpoint.resolve()),
                selected_rows=selected_rows,
                imported_seed_shots=0
                if receipt is None
                else receipt["seed_counts"][strong_id]["shots"],
                added_shots=added_shots,
                added_failures=added_failures,
                cumulative_shots=shots,
                cumulative_failures=failures,
                captured_bytes=len(captured),
                ignored_partial_bytes=len(raw) - end,
                captured_prefix_sha256=hashlib.sha256(captured).hexdigest(),
            )
        )
    if not history:
        raise ValueError("No new batches found")
    return history, provenance


def write_plot(
    history: list[dict[str, Any]], provenance: list[dict[str, Any]], output: Path, captured_utc: str
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import StrMethodFormatter

    output.mkdir(parents=True, exist_ok=False)
    with (output / "cumulative_ler_history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    final = history[-1]
    milestones = [
        r for r in history if r["cumulative_shots"] in (100000, 200000, 300000, 500000, 1000000)
    ]
    if not milestones or milestones[-1] is not final:
        milestones.append(final)
    snapshot = dict(
        captured_utc=captured_utc,
        strong_id=STRONG_ID,
        sources=provenance,
        shots=final["cumulative_shots"],
        failures=final["cumulative_failures"],
        cumulative_ler=final["cumulative_ler"],
        milestones=milestones,
        interpretation="Cumulative means are correlated. Lines join saved-batch endpoints in checkpoint append order, not individual-shot or wall-clock order. Band is pointwise +/-1 jackknife SE, not a simultaneous confidence band. Zero-failure prefixes have no band, not zero uncertainty.",
    )
    (output / "snapshot.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    x = np.array([r["cumulative_shots"] for r in history])
    y = np.array([r["cumulative_ler"] for r in history])
    se = np.array(
        [
            r["jackknife_1sigma"]
            if 0 < r["cumulative_failures"] < r["cumulative_shots"]
            else np.nan
            for r in history
        ],
        dtype=float,
    )
    fig, axis = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    axis.fill_between(
        x, np.maximum(0, y - se), y + se, color="C2", alpha=0.16, label="Pointwise ±1 jackknife SE"
    )
    axis.plot(x, y, color="C2", linewidth=1.7, label="Cumulative mean (saved batches)")
    axis.axhline(
        final["cumulative_ler"],
        color="0.4",
        linestyle="--",
        linewidth=1,
        label=f"Latest: {final['cumulative_failures']:,}/{final['cumulative_shots']:,} = {final['cumulative_ler']:.3g}",
    )
    for r in milestones:
        axis.plot(r["cumulative_shots"], r["cumulative_ler"], "o", color="C2", markersize=4)
    for p in provenance[1:]:
        axis.axvline(p["imported_seed_shots"], color="0.65", linewidth=0.8, linestyle=":", zorder=0)
    axis.set(
        title="Cumulative XX adapter LER — d=8, eight rounds, p=10⁻⁴",
        xlabel="Accumulated shots (saved-batch order)",
        ylabel="Cumulative logical error rate",
        xlim=(0, float(x[-1]) * 1.025),
        ylim=(0, None),
    )
    axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right", fontsize=9)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"cumulative_adapter_ler.{suffix}", dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            dict(
                output=str(output.resolve()),
                shots=final["cumulative_shots"],
                failures=final["cumulative_failures"],
                ler=final["cumulative_ler"],
                milestones=[
                    {k: r[k] for k in ("cumulative_shots", "cumulative_failures", "cumulative_ler")}
                    for r in milestones
                ],
            ),
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    directories = [OUTPUTS / name for name in SOURCE_NAMES]
    destination = args.output_dir.resolve()
    for directory in directories:
        directory = directory.resolve()
        if destination.is_relative_to(directory) or directory.is_relative_to(destination):
            parser.error("History output must be separate from all source datasets")
    history, provenance = stitch_history(directories)
    now = datetime.datetime.now(datetime.timezone.utc)
    write_plot(
        history,
        provenance,
        destination / now.strftime("snapshot_%Y%m%dT%H%M%S_%fZ"),
        now.isoformat(),
    )


if __name__ == "__main__":
    main()
