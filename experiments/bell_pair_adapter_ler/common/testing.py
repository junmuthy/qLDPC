"""Synthetic checkpoint/reference fixtures, never copies of production results."""

import json
from pathlib import Path

import sinter

from ..bell_fidelity.run_ppm_fidelity_sweep import task_for
from .run_experiment import BuiltExperiment, write_artifacts, write_summary


def write_reference(experiments: list[BuiltExperiment], directory: Path) -> None:
    """Write generated circuits and synthetic counts/telemetry into a test directory."""
    write_artifacts(experiments, directory)
    stats = [
        sinter.TaskStats(
            strong_id=task_for(e).strong_id(),
            decoder="qldpc_bp_osd",
            json_metadata=e.metadata,
            shots=10000,
            errors=2,
            seconds=1,
        )
        for e in experiments
    ]
    write_summary(stats, directory)
    (directory / "sinter_progress.csv").write_text(
        sinter.CSV_HEADER + "\n" + "\n".join(s.to_csv_line() for s in stats) + "\n"
    )
    (directory / "run_config.json").write_text('{"synthetic_test_reference": true}\n')
    (directory / "invocations.jsonl").write_text(
        json.dumps(
            {
                "invocation_id": "synthetic",
                "status": "completed",
                "config": {"num_workers": 8},
                "new_shots": 10000 * len(stats),
                "sampling_decode_wall_seconds": 1,
                "profile": {
                    "reported_batch_shots": 10000 * len(stats),
                    "resource_profile_error": None,
                    "sampled_peak_sum_rss_bytes": 1024,
                },
            }
        )
        + "\n"
    )
    (directory / "batch_samples.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "invocation_id": "synthetic",
                    "strong_id": s.strong_id,
                    "logical_basis": s.json_metadata["logical_basis"],
                    "shots": s.shots,
                    "worker_seconds": s.seconds,
                }
            )
            + "\n"
            for s in stats
        )
    )
