"""Compare completed four/eight-round matching-basis HGP runs, optionally waiting.

Read-only with respect to input runs. Never builds circuits or samples shots.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .dual_timing import timing_row
from .run_experiment import _wilson_interval

OUTPUTS = Path(__file__).resolve().parent / "outputs"
FOUR = OUTPUTS / "ppm_basis_first_hgp_m4_d8_prep4_surg4_10000"
EIGHT = OUTPUTS / "ppm_basis_first_hgp_m4_d8_prep4_surg8_10000" / "samples"
DESTINATION = OUTPUTS / "ppm_basis_first_hgp_m4_d8_prep4_rounds4_vs8_10000"
ROUND_DEPENDENT = {
    "name",
    "rounds",
    "surgery_rounds",
    "circuit_path",
    "dem_path",
    "num_detectors",
    "num_dem_errors",
    "num_full_detectors",
    "num_full_dem_errors",
    "retained_detector_ids",
}


def events(directory: Path) -> list[dict[str, Any]]:
    """Ignore an unterminated JSONL record while a collector appends its final event."""
    return [
        json.loads(line)
        for line in (directory / "invocations.jsonl").read_text().splitlines(keepends=True)
        if line.endswith("\n") and line.strip()
    ]


def finished(directory: Path, target: int) -> bool:
    records = events(directory)
    if not records:
        return False
    latest = records[-1]
    if latest["status"] in {"failed", "interrupted"}:
        raise RuntimeError(f"Collection {latest['status']}: {directory}; resume it first")
    return (
        latest["status"] == "completed"
        and latest["config"]["max_shots"] == target
        and latest["after"]["shots"] == target * len(latest["config"]["bases"])
    )


def validate_metadata(meta: dict[str, Any], basis: str, rounds: int) -> None:
    expected = dict(
        left_code="hgp_simplex_m4",
        right_code="surface_unrotated_d8",
        left_logical_index=0,
        right_logical_index=0,
        logical_basis=basis,
        preparation_rounds=4,
        surgery_rounds=rounds,
        rounds=rounds,
        physical_error_rate=0.001,
        bell_pair_infidelity=0.0,
        bell_pair_fidelity=1.0,
        noise_scope="intrinsic",
        adapter_noise_targets="adapter_hardware",
        include_idling_error=False,
        detector_reduction="inactive_rows",
        dem_analysis="undecomposed_exact",
        decoder="qldpc_bp_osd",
        schedule_policy="ppm_basis_first_local_v1",
        surgery_schedule=dict(
            local_order=["X", "Z"] if basis == "X" else ["Z", "X"],
            bell_order=["X", "Z"],
            local_layers_sha256=None,
        ),
    )
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"Unexpected {key} for {basis}{basis}, {rounds} rounds")


def read_condition(directory: Path, basis: str, rounds: int, target: int) -> dict[str, Any]:
    metas = [
        m
        for m in json.loads((directory / "manifest.json").read_text())
        if m["logical_basis"] == basis
    ]
    if len(metas) != 1:
        raise ValueError(f"Expected one {basis}{basis} condition in {directory}")
    meta = metas[0]
    validate_metadata(meta, basis, rounds)
    checkpoint_meta = {k: v for k, v in meta.items() if k not in {"circuit_path", "dem_path"}}
    with (directory / "sinter_progress.csv").open(newline="") as stream:
        samples = [
            r
            for r in csv.DictReader(stream, skipinitialspace=True)
            if json.loads(r["json_metadata"]) == checkpoint_meta
        ]
    ids = {r["strong_id"] for r in samples}
    if len(ids) != 1 or any(r["decoder"] != meta["decoder"] for r in samples):
        raise ValueError("Missing or ambiguous checkpoint task identity")
    shots, errors, discards = (
        sum(int(r[k]) for r in samples) for k in ("shots", "errors", "discards")
    )
    if shots != target or not (0 <= discards < shots and 0 <= errors <= shots - discards):
        raise ValueError(f"Expected {target} valid cumulative shots: {directory}, {basis}")
    with (directory / "summary.csv").open(newline="") as stream:
        summaries = [r for r in csv.DictReader(stream) if r["name"] == meta["name"]]
    if len(summaries) != 1 or tuple(
        int(summaries[0][k]) for k in ("shots", "errors", "discards")
    ) != (shots, errors, discards):
        raise ValueError("Completed summary and checkpoint counts disagree")
    low, high = _wilson_interval(errors, shots - discards)
    return dict(
        logical_basis=basis,
        surgery_rounds=rounds,
        shots=shots,
        errors=errors,
        discards=discards,
        logical_error_rate=errors / (shots - discards),
        ler_95ci_low=low,
        ler_95ci_high=high,
        strong_id=next(iter(ids)),
        source_directory=str(directory),
        metadata=meta,
    )


def load_comparison(four: Path, eight: Path, target: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for basis in ("X", "Z"):
        pair = [
            read_condition(four / basis, basis, 4, target),
            read_condition(eight, basis, 8, target),
        ]
        normalized = [
            {k: v for k, v in r["metadata"].items() if k not in ROUND_DEPENDENT} for r in pair
        ]
        if normalized[0] != normalized[1]:
            raise ValueError("Conditions differ in more than adapter round count")
        rows.extend({k: v for k, v in r.items() if k != "metadata"} for r in pair)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(output: Path, rows: list[dict[str, Any]], timings: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
    for axis, basis, color in zip(axes, ("X", "Z"), ("C0", "C1")):
        selected = [r for r in rows if r["logical_basis"] == basis]
        for row in selected:
            rate, low, high = [
                100 * row[k] for k in ("logical_error_rate", "ler_95ci_low", "ler_95ci_high")
            ]
            axis.errorbar(
                row["surgery_rounds"],
                rate,
                yerr=[[max(0, rate - low)], [max(0, high - rate)]],
                fmt="o",
                color=color,
                capsize=5,
                markersize=7,
            )
            axis.annotate(
                f"{row['errors']}/{row['shots']:,}",
                (row["surgery_rounds"], high),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )
        axis.set(
            title=f"{basis}{basis} logical parity",
            xlabel="Adapter syndrome rounds",
            xticks=[4, 8],
            xlim=(3, 9),
        )
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Adapter LER per PPM (%)")
    axes[0].set_ylim(0, max(r["ler_95ci_high"] for r in rows) * 100 * 1.25)
    fig.suptitle("HGP simplex m=4 / unrotated surface d=8 — matching-basis schedule")
    fig.text(
        0.5,
        0.02,
        "Four ideal preparation rounds; hardware-only p=0.001; Bell fidelity=1.\n"
        "Points: cumulative failure fractions. Error bars: 95% Wilson intervals.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"adapter_ler_4_vs_8_rounds.{suffix}", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5))
    for basis, color, marker in (("X", "C0", "o"), ("Z", "C1", "s")):
        selected = [r for r in timings if r["logical_basis"] == basis]
        axis.plot(
            [r["surgery_rounds"] for r in selected],
            [r["worker_seconds_per_shot"] for r in selected],
            marker=marker,
            color=color,
            label=f"{basis}{basis}",
        )
    axis.set(
        xlabel="Adapter syndrome rounds",
        xticks=[4, 8],
        xlim=(3, 9),
        ylabel="Sampling/decoding worker seconds per shot",
        title="Classical simulation work — eight-worker collection stages",
    )
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.text(
        0.5,
        0.025,
        "Earlier pilot stages excluded from timing, retained in LER counts.\n"
        "Four rounds: sequential XX, ZZ. Eight rounds: shared XX+ZZ pool.\n"
        "Descriptive work comparison; not a controlled speedup or physical circuit time.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"simulation_timing_4_vs_8_rounds.{suffix}", dpi=180)
    plt.close(fig)


def generate(four: Path, eight: Path, output: Path, target: int) -> None:
    rows = load_comparison(four, eight, target)
    timings = []
    for row in rows:
        timing = timing_row(
            Path(row["source_directory"]),
            basis=row["logical_basis"],
            strong_id=row["strong_id"],
            workers=8,
            label=f"{row['logical_basis'] * 2}, {row['surgery_rounds']} rounds",
        )
        if timing is None:
            raise ValueError("Missing completed eight-worker timing profile")
        timings.append({"surgery_rounds": row["surgery_rounds"], **timing})
    # Count shared eight-round invocations once, not once per basis.
    stages = []
    for rounds in (4, 8):
        selected = [r for r in timings if r["surgery_rounds"] == rounds]
        keys = {
            (r["source_directory"], identity)
            for r in selected
            for identity in r["invocation_ids"].split(";")
        }
        unique = [
            next(
                e
                for e in events(Path(d))
                if e["invocation_id"] == identity and e["status"] == "completed"
            )
            for d, identity in sorted(keys)
        ]
        stages.append(
            dict(
                surgery_rounds=rounds,
                num_workers=8,
                new_shots=sum(e["new_shots"] for e in unique),
                collection_wall_seconds=sum(e["sampling_decode_wall_seconds"] for e in unique),
                collection_scope="XX then ZZ" if rounds == 4 else "shared XX+ZZ",
            )
        )
    write_csv(output / "comparison.csv", rows)
    write_csv(output / "timing_comparison.csv", timings)
    write_csv(output / "collection_wall_times.csv", stages)
    hashes = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for d in (four / "X", four / "Z", eight)
        for p in (
            d / "manifest.json",
            d / "sinter_progress.csv",
            d / "summary.csv",
            d / "invocations.jsonl",
            d / "batch_samples.jsonl",
        )
    }
    (output / "source_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    plot_results(output, rows, timings)
    print(f"Completed comparison: {output / 'adapter_ler_4_vs_8_rounds.pdf'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--four-round-dir", type=Path, default=FOUR)
    parser.add_argument("--eight-round-dir", type=Path, default=EIGHT)
    parser.add_argument("--output-dir", type=Path, default=DESTINATION)
    parser.add_argument("--target-shots", type=int, default=10000)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--timeout-seconds", type=float, default=14400)
    args = parser.parse_args()
    if min(args.target_shots, args.poll_seconds, args.timeout_seconds) <= 0:
        parser.error("Shot target and wait intervals must be positive")
    sources = (args.four_round_dir / "X", args.four_round_dir / "Z", args.eight_round_dir)
    for source in (args.four_round_dir, args.eight_round_dir):
        if args.output_dir.resolve().is_relative_to(
            source.resolve()
        ) or source.resolve().is_relative_to(args.output_dir.resolve()):
            parser.error("Comparison output must be separate from the source runs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".comparison.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        while not all(finished(source, args.target_shots) for source in sources):
            if not args.wait:
                raise RuntimeError("Collections are not complete; use --wait")
            if time.monotonic() - started >= args.timeout_seconds:
                raise TimeoutError("Comparison wait expired; collectors are unaffected")
            print("Waiting for completed four/eight-round XX and ZZ collections...", flush=True)
            time.sleep(min(args.poll_seconds, 60))
        generate(args.four_round_dir, args.eight_round_dir, args.output_dir, args.target_shots)


if __name__ == "__main__":
    main()
