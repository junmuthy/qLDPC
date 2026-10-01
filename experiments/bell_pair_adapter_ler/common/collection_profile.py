"""Optional collection telemetry; it never changes circuits, decoders, or shot identity."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import sinter


class CollectionProfile:
    """Log completed Sinter batches and sample parent/descendant RSS once per second.

    Requires psutil only when instantiated. Summed RSS can double-count shared
    pages and a sampled peak can miss brief spikes: it is not unique physical RAM.
    Worker seconds describe sampling/decoding batches, not pure decoder CPU time.
    """

    def __init__(self, directory: Path, invocation_id: str) -> None:
        import psutil

        self.psutil = psutil
        self.directory = directory
        self.invocation_id = invocation_id
        self.parent = psutil.Process(os.getpid())
        self.started = time.perf_counter()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.peak_rss = 0
        self.peak_processes = 0
        self.resource_error: str | None = None
        self.batch_shots = 0
        self.batch_seconds = 0.0
        self.batches = 0

    def _write(self, filename: str, row: dict[str, Any]) -> None:
        with (self.directory / filename).open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "invocation_id": self.invocation_id,
                        "unix_time": time.time(),
                        "elapsed_seconds": time.perf_counter() - self.started,
                        **row,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    def _resource_sample(self) -> None:
        processes = []
        for process in [self.parent, *self.parent.children(recursive=True)]:
            try:
                processes.append({"pid": process.pid, "rss_bytes": process.memory_info().rss})
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                continue
        rss = sum(p["rss_bytes"] for p in processes)
        self.peak_rss = max(self.peak_rss, rss)
        self.peak_processes = max(self.peak_processes, len(processes))
        self._write("resource_samples.jsonl", {"sum_rss_bytes": rss, "processes": processes})

    def start(self) -> None:
        self._resource_sample()

        def monitor() -> None:
            try:
                while not self.stop_event.wait(1):
                    self._resource_sample()
            except Exception as error:
                self.resource_error = repr(error)

        self.thread = threading.Thread(target=monitor, name="ppm-memory-profile", daemon=True)
        self.thread.start()

    def progress(self, progress: sinter.Progress) -> None:
        for stats in progress.new_stats:
            self.batches += 1
            self.batch_shots += stats.shots
            self.batch_seconds += stats.seconds
            self._write(
                "batch_samples.jsonl",
                {
                    "strong_id": stats.strong_id,
                    "logical_basis": stats.json_metadata["logical_basis"],
                    "shots": stats.shots,
                    "errors": stats.errors,
                    "discards": stats.discards,
                    "worker_seconds": stats.seconds,
                    "worker_seconds_per_shot": stats.seconds / stats.shots if stats.shots else None,
                    "new_shots_this_invocation": self.batch_shots,
                },
            )

    def finish(self) -> dict[str, Any]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
        return {
            "sampled_peak_sum_rss_bytes": self.peak_rss,
            "peak_process_count": self.peak_processes,
            "resource_profile_error": self.resource_error,
            "reported_batches": self.batches,
            "reported_batch_shots": self.batch_shots,
            "reported_batch_worker_seconds": self.batch_seconds,
            "resource_sample_interval_seconds": 1,
        }
