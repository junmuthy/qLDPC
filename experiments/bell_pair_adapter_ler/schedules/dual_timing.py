"""Compare profiled work per shot without misattributing shared collection wall time."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _events(path: Path) -> list[dict[str, Any]]:
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )


def timing_row(
    directory: Path, *, basis: str, strong_id: str, workers: int, label: str
) -> dict[str, Any] | None:
    events = [
        e
        for e in _events(directory / "invocations.jsonl")
        if e["status"] == "completed"
        and e["config"]["num_workers"] == workers
        and e.get("profile", {}).get("reported_batch_shots", 0) > 0
    ]
    batches = _events(directory / "batch_samples.jsonl")
    selected_events, selected_batches = [], []
    for event in events:
        invocation_batches = [b for b in batches if b["invocation_id"] == event["invocation_id"]]
        if sum(b["shots"] for b in invocation_batches) != event["new_shots"]:
            raise ValueError("Timing telemetry disagrees with checkpointed invocation counts")
        chosen = [
            b
            for b in invocation_batches
            if b["strong_id"] == strong_id and b["logical_basis"] == basis
        ]
        if chosen:
            selected_events.append(event)
            selected_batches.extend(chosen)
    if not selected_batches:
        return None
    if any(e["profile"]["resource_profile_error"] is not None for e in selected_events):
        raise ValueError("Resource profiling was incomplete")
    shots = sum(b["shots"] for b in selected_batches)
    seconds = sum(b["worker_seconds"] for b in selected_batches)
    wall = sum(e["sampling_decode_wall_seconds"] for e in selected_events)
    all_shots = sum(e["new_shots"] for e in selected_events)
    shared = any(
        e["new_shots"]
        != sum(b["shots"] for b in selected_batches if b["invocation_id"] == e["invocation_id"])
        for e in selected_events
    )
    return dict(
        label=label,
        logical_basis=basis,
        num_workers=workers,
        timed_basis_shots=shots,
        batch_worker_seconds=seconds,
        worker_seconds_per_shot=seconds / shots,
        collection_wall_seconds=wall,
        collection_total_new_shots=all_shots,
        collection_shots_per_wall_second=all_shots / wall,
        collection_scope="shared XX+ZZ (not a ZZ-only wall time)"
        if shared
        else f"{basis}{basis} only",
        sampled_peak_sum_rss_bytes=max(
            e["profile"]["sampled_peak_sum_rss_bytes"] for e in selected_events
        ),
        invocation_ids=";".join(e["invocation_id"] for e in selected_events),
        source_directory=str(directory),
    )


def write_timing_comparison(output: Path, series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Use only the current pilot worker count; exclude historical lower-worker pilots."""
    completed = [e for e in _events(output / "invocations.jsonl") if e["status"] == "completed"]
    if not completed:
        return []
    workers = completed[-1]["config"]["num_workers"]
    rows = []
    for item in series:
        row = timing_row(
            Path(item["source_checkpoint"]).parent,
            basis=item["logical_basis"],
            strong_id=item["strong_id"],
            workers=workers,
            label=item["label"],
        )
        if row is None:
            (output / "timing_notes.json").write_text(
                json.dumps(
                    {
                        "status": "unavailable",
                        "reason": "All three conditions need completed batch profiles at the same worker count",
                        "num_workers": workers,
                    },
                    indent=2,
                )
                + "\n"
            )
            return []
        rows.append(row)
    with (output / "timing_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ratio = rows[2]["worker_seconds_per_shot"] / rows[1]["worker_seconds_per_shot"]
    (output / "timing_notes.json").write_text(
        json.dumps(
            dict(
                status="available",
                num_workers=workers,
                dual_to_matching_zz_worker_cost_ratio=ratio,
                interpretation="Descriptive sampled work ratio, not a controlled wall-time speedup or confidence interval",
                caveats=[
                    "Different shot counts and random faults; rare slow decodes affect pilot means",
                    "Archived workers shared XX and ZZ; dual pilot runs ZZ alone",
                    "Shared collection wall times must not be added across the two archived basis rows",
                    "Worker seconds include sampling/decoding, not pure decoder CPU time",
                    "Summed process RSS can double-count shared pages",
                ],
            ),
            indent=2,
        )
        + "\n"
    )
    fig, axis = plt.subplots(figsize=(8.5, 5.4))
    values = [r["worker_seconds_per_shot"] for r in rows]
    axis.bar(range(3), values, color=["C0", "C1", "C2"], width=0.55)
    axis.set_xticks(
        range(3),
        ["Matching-basis XX\narchived", "Matching-basis ZZ\narchived", "Exact-dual ZZ\nnew pilot"],
    )
    axis.set_ylabel("Sampling/decoding worker seconds per shot")
    axis.set_title(
        f"Hardware schedule timing — {workers} workers\nSame worker count; different sampled faults and collection workloads"
    )
    axis.set_ylim(0, max(values) * 1.35)
    for i, row in enumerate(rows):
        axis.annotate(
            f"{values[i]:.3f} s/shot\n{row['timed_basis_shots']:,} timed shots",
            (i, values[i]),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
        )
    axis.grid(axis="y", alpha=0.25)
    fig.text(
        0.5,
        0.025,
        f"Dual / native ZZ sampled worker cost: {ratio:.2f}×. Descriptive ratio, not a wall-time speedup.\n"
        "Archived collection shared XX+ZZ; new collection is ZZ only. Historical other-worker stages excluded.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    for extension in ("pdf", "png"):
        fig.savefig(output / f"hgp_xz_dual_timing.{extension}", dpi=180)
    plt.close(fig)
    return rows
