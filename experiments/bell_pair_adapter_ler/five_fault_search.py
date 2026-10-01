"""Bounded connected-triple/pair search for minimum five-fault PPM errors.

Requires a completed exclusion of weights one through four. A minimum error
must be connected in the single-fault detector-overlap graph: otherwise one
component is a smaller undetected logical error. Every connected five-vertex
graph contains a connected triple, which is enumerated as a two-edge wedge.
Thus completed coverage excludes weight five; partial coverage does not.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import effective_distance_audit as base


def neighbors(patterns: base.Patterns) -> list[list[int]]:
    incident: dict[int, set[int]] = collections.defaultdict(set)
    for i, value in enumerate(patterns.values):
        value >>= 1
        while value:
            bit = value & -value
            incident[bit.bit_length() - 1].add(i)
            value ^= bit
    adjacency: list[set[int]] = [set() for _ in patterns.values]
    for ids in incident.values():
        for i in ids:
            adjacency[i].update(ids)
    return [sorted(ids - {i}) for i, ids in enumerate(adjacency)]


def pivot_search(
    p: base.Patterns, pivot: int, adjacent: list[int], indexes: list[np.ndarray[Any, Any]]
) -> tuple[list[int] | None, int, int]:
    ia, ib = np.triu_indices(len(adjacent), 1)
    a, b = np.asarray(adjacent, dtype=np.uint32)[ia], np.asarray(adjacent, dtype=np.uint32)[ib]
    targets = p.keys[a] ^ p.keys[b] ^ p.keys[pivot] ^ np.uint64(1)
    buckets = ((targets >> 1) & (base.BUCKETS - 1)).astype(np.int64)
    order = np.lexsort((targets, buckets))
    offsets = np.concatenate(([0], np.cumsum(np.bincount(buckets, minlength=base.BUCKETS))))
    matches = examined = 0
    for bucket, index in enumerate(indexes):
        rows = order[offsets[bucket] : offsets[bucket + 1]]
        if not len(rows) or not len(index):
            continue
        keys = index["key"]
        low = np.searchsorted(keys, targets[rows], side="left")
        high = np.searchsorted(keys, targets[rows], side="right")
        examined += len(rows)
        for local in np.flatnonzero(low < high):
            row = rows[local]
            i, j = int(a[row]), int(b[row])
            value = p.values[pivot] ^ p.values[i] ^ p.values[j]
            for k in range(int(low[local]), int(high[local])):
                matches += 1
                x, y = base.unpack(int(index["pair"][k]))
                if value ^ p.values[x] ^ p.values[y] != 1:
                    continue
                ids = p.realization([pivot, i, j, x, y])
                if ids is not None:
                    return ids, examined, matches
    return None, len(targets), matches


def write_index(path: Path, records: np.ndarray[Any, Any]) -> str:
    records.sort(order="key")
    temporary = path.with_suffix(".part")
    with temporary.open("wb") as stream:
        np.save(stream, records, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return base.digest(path.read_bytes())


def condition(distance_audit: str, output: str, seconds: float) -> dict[str, Any]:
    old, dst = Path(distance_audit), Path(output)
    dst.mkdir(parents=True, exist_ok=True)
    with base.writer_lock(dst / ".writer.lock"):
        started = tick = time.perf_counter()
        old_identity = json.loads((old / "identity.json").read_text())
        progress = json.loads((old / "progress.json").read_text())
        if not progress["complete"] or progress["excluded_through_weight"] < 4:
            raise ValueError("Connected-triple proof requires complete exclusion through four")
        circuit, cat, frozen = base.load_source(Path(old_identity["source"]))
        if old_identity != frozen:
            raise ValueError("Source distance audit identity changed")
        identity = {
            "distance_audit": str(old.resolve()),
            "parent_identity": frozen,
            "parent_progress_sha256": base.digest((old / "progress.json").read_bytes()),
            "search_source_sha256": base.digest(Path(__file__).read_bytes()),
        }
        manifest = dst / "identity.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != identity:
                raise ValueError("Five-fault identity changed")
        else:
            if any(p.name != ".writer.lock" for p in dst.iterdir()):
                raise ValueError("Nonempty output without identity")
            base.atomic_json(manifest, identity)
        state_path = dst / "progress.json"
        state: dict[str, Any] = (
            json.loads(state_path.read_text())
            if state_path.exists()
            else {
                "index_hashes": [],
                "next_pivot": 0,
                "wedges_examined": 0,
                "fingerprint_matches": 0,
                "seconds": 0.0,
                "complete": False,
                "fault_distance_lower_bound": 5,
                "fault_distance_upper_bound": None,
            }
        )

        def checkpoint() -> None:
            nonlocal tick
            now = time.perf_counter()
            state["seconds"] += now - tick
            tick = now
            base.atomic_json(state_path, state)

        if not state["complete"]:
            # Check that the committed source chunks exhaust all unordered pairs.
            p = base.Patterns(cat)
            next_row = pairs = 0
            for chunk in progress["chunks"]:
                if chunk["start"] != next_row:
                    raise ValueError("Gap in source pair coverage")
                if base.digest((old / chunk["name"]).read_bytes()) != chunk["sha256"]:
                    raise ValueError("Source pair checkpoint changed")
                next_row = chunk["end"]
                pairs += chunk["pairs"]
            n = len(p.values)
            if next_row != n - 1 or pairs != n * (n - 1) // 2:
                raise ValueError("Source pairs are incomplete")
            for b, expected in enumerate(state["index_hashes"]):
                if base.digest((dst / f"index_{b:03d}.npy").read_bytes()) != expected:
                    raise ValueError("Sorted pair index changed")
            print(f"{old.name}: indexing/searching, {seconds:g}s budget", flush=True)
            while (
                len(state["index_hashes"]) < base.BUCKETS
                and time.perf_counter() - started < seconds
            ):
                b = len(state["index_hashes"])
                records = base.bucket_records(old, progress["chunks"], b)
                state["index_hashes"].append(write_index(dst / f"index_{b:03d}.npy", records))
                checkpoint()
            if len(state["index_hashes"]) == base.BUCKETS:
                graph = neighbors(p)
                order = sorted(range(n), key=lambda i: (not (p.values[i] & 1), len(graph[i]), i))
                state["pivots_total"] = n
                state["wedges_total"] = sum(len(ns) * (len(ns) - 1) // 2 for ns in graph)
                indexes = [
                    np.load(dst / f"index_{b:03d}.npy", mmap_mode="r", allow_pickle=False)
                    for b in range(base.BUCKETS)
                ]
                while state["next_pivot"] < n and time.perf_counter() - started < seconds:
                    pivot = order[state["next_pivot"]]
                    ids, count, matches = pivot_search(p, pivot, graph[pivot], indexes)
                    state["wedges_examined"] += count
                    state["fingerprint_matches"] += matches
                    if ids is not None:
                        witness = base.verify_witness(circuit, cat, ids)
                        if witness["weight"] != 5:
                            raise ValueError("Expected a five-fault witness")
                        base.atomic_json(dst / "witness.json", witness)
                        state.update(complete=True, fault_distance_upper_bound=5)
                        checkpoint()
                        break
                    state["next_pivot"] += 1
                    checkpoint()
                    if state["next_pivot"] % 100 == 0:
                        print(
                            f"{old.name}: {state['next_pivot']}/{n} pivots, "
                            f"{state['wedges_examined']:,} wedges",
                            flush=True,
                        )
                if state["next_pivot"] == n:
                    state.update(complete=True, fault_distance_lower_bound=6)
            checkpoint()
        summary = {k: v for k, v in state.items() if k != "index_hashes"}
        summary.update(
            condition=old.name,
            indexed_buckets=len(state["index_hashes"]),
            exact_distance=5 if state["fault_distance_upper_bound"] == 5 else None,
            caveat="Partial search does not raise the lower bound. Wedges may duplicate triangles. No decoder, noise, or circuit changes.",
        )
        base.atomic_json(dst / "summary.json", summary)
        print(json.dumps(summary), flush=True)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance-audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds-per-condition", type=float, default=600)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--conditions", nargs="+", default=["d8_r8_X", "d8_r8_Z", "d4_r8_X", "d4_r8_Z"]
    )
    args = parser.parse_args()
    if args.workers < 1 or args.seconds_per_condition <= 0:
        parser.error("Workers and time budget must be positive")
    valid = {f"d{d}_r8_{b}" for d in (4, 8) for b in ("X", "Z")}
    if len(args.conditions) != len(set(args.conditions)) or not set(args.conditions) <= valid:
        parser.error("Invalid or duplicate condition")
    if args.output_dir.resolve() == args.distance_audit_dir.resolve():
        parser.error("Use a separate output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with base.writer_lock(args.output_dir / ".writer.lock"):
        started = time.time()
        base.atomic_json(
            args.output_dir / f"invocation_{time.time_ns()}.json",
            {
                "started_unix": started,
                "workers": args.workers,
                "conditions": args.conditions,
                "seconds_per_condition": args.seconds_per_condition,
            },
        )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(
                    condition,
                    str(args.distance_audit_dir / n),
                    str(args.output_dir / n),
                    args.seconds_per_condition,
                )
                for n in args.conditions
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
                rows = [
                    json.loads(p.read_text())
                    for p in sorted(args.output_dir.glob("d*/summary.json"))
                ]
                base.atomic_json(args.output_dir / "summary.json", rows)
        print(f"Five-fault pass finished in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
