"""Validate completed larger-code pilot counts and summarize LER, workload, and timings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .run_experiment import _wilson_interval


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validation = {
        r["logical_basis"]: r for r in json.loads((output / "validation.json").read_text())
    }
    results, stages = [], []
    for basis in ("X", "Z"):
        directory = output / basis
        rows = read_csv(directory / "summary.csv")
        entries = json.loads((directory / "manifest.json").read_text())
        if len(rows) != 1 or len(entries) != 1:
            raise ValueError("Expected one condition per basis directory")
        row, meta = rows[0], entries[0]
        if not (
            row["name"] == meta["name"]
            and meta["logical_basis"] == basis
            and meta["left_code"] == "hgp_simplex_m4"
            and meta["right_code"] == "surface_unrotated_d8"
            and meta["left_logical_index"] == meta["right_logical_index"] == 0
            and meta["preparation_rounds"] == meta["surgery_rounds"] == 4
            and meta["physical_error_rate"] == 0.001
            and meta["bell_pair_fidelity"] == 1
            and meta["noise_scope"] == "intrinsic"
            and not meta["include_idling_error"]
            and meta["adapter_noise_targets"] == "adapter_hardware"
            and meta["detector_reduction"] == "inactive_rows"
            and meta["dem_analysis"] == "undecomposed_exact"
            and meta["schedule_policy"] == "ppm_basis_first_local_v1"
            and meta["surgery_schedule"]
            == {
                "local_order": ["X", "Z"] if basis == "X" else ["Z", "X"],
                "bell_order": ["X", "Z"],
                "local_layers_sha256": None,
            }
        ):
            raise ValueError("Pilot settings do not match the approved experiment")
        for path_key, hash_key in (
            ("circuit_path", "reduced_circuit_sha256"),
            ("dem_path", "reduced_dem_sha256"),
        ):
            digest = hashlib.sha256(
                (directory / meta[path_key]).read_text().strip().encode()
            ).hexdigest()
            if digest != validation[basis][hash_key]:
                raise ValueError(
                    "Sampled circuit/DEM does not match noiseless/structural validation"
                )
        invocations = [
            r for r in read_jsonl(directory / "invocations.jsonl") if r["status"] == "completed"
        ]
        if len(invocations) != 2 or [r["config"]["max_shots"] for r in invocations] != [256, 1000]:
            raise ValueError("Both the 256-shot stage and the extension to 1000 must be complete")
        batches = read_jsonl(directory / "batch_samples.jsonl")
        previous_shots = 0
        for invocation in invocations:
            if invocation["before"]["shots"] != previous_shots:
                raise ValueError("Resumed counts do not continue from the previous stage")
            selected = [r for r in batches if r["invocation_id"] == invocation["invocation_id"]]
            new_shots = sum(r["shots"] for r in selected)
            worker_seconds = sum(r["worker_seconds"] for r in selected)
            new_errors = sum(r["errors"] for r in selected)
            if not (
                new_shots
                == invocation["new_shots"]
                == invocation["profile"]["reported_batch_shots"]
                and new_errors == invocation["after"]["errors"] - invocation["before"]["errors"]
                and invocation["after"]["shots"] == invocation["config"]["max_shots"]
            ):
                raise ValueError("Batch telemetry and checkpoint totals disagree")
            if invocation["profile"]["resource_profile_error"] is not None:
                raise ValueError("Resource profiling was incomplete")
            wall = invocation["sampling_decode_wall_seconds"]
            stages.append(
                dict(
                    logical_basis=basis,
                    cumulative_shot_cap=invocation["config"]["max_shots"],
                    num_workers=invocation["config"]["num_workers"],
                    new_shots=new_shots,
                    new_errors=new_errors,
                    collection_wall_seconds=wall,
                    batch_worker_seconds=worker_seconds,
                    worker_seconds_per_shot=worker_seconds / new_shots,
                    new_shots_per_collection_wall_second=new_shots / wall,
                    sampled_peak_sum_rss_mib=invocation["profile"]["sampled_peak_sum_rss_bytes"]
                    / 2**20,
                    projected_20000_shot_wall_hours_at_this_stage_throughput=20000
                    * wall
                    / new_shots
                    / 3600,
                    invocation_id=invocation["invocation_id"],
                )
            )
            previous_shots = invocation["after"]["shots"]
        shots, errors, discards = (int(row[k]) for k in ("shots", "errors", "discards"))
        if (
            shots != 1000
            or shots != sum(r["shots"] for r in batches)
            or errors != sum(r["errors"] for r in batches)
        ):
            raise ValueError("Final summary double-counts or omits pilot shots")
        low, high = _wilson_interval(errors, shots - discards)
        results.append(
            dict(
                logical_basis=basis,
                shots=shots,
                errors=errors,
                discards=discards,
                logical_error_rate=errors / (shots - discards),
                ler_95ci_low=low,
                ler_95ci_high=high,
                total_collection_wall_seconds=sum(
                    r["sampling_decode_wall_seconds"] for r in invocations
                ),
                total_batch_worker_seconds=sum(r["worker_seconds"] for r in batches),
                num_detectors=meta["num_detectors"],
                num_dem_errors=meta["num_dem_errors"],
                source_summary=str((directory / "summary.csv").relative_to(output)),
            )
        )
    return results, stages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name, rows in zip(("pilot_summary.csv", "stage_timings.csv"), summarize(args.output_dir)):
        with (args.output_dir / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {name}: {len(rows)} rows", flush=True)
    source_names = (
        "run_experiment.py",
        "run_ppm_basis_first.py",
        "collection_profile.py",
        "validate_large_ppm_pilot.py",
        "summarize_large_ppm_pilot.py",
    )
    hashes = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in source_names
    }
    (args.output_dir / "runner_source_hashes.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
