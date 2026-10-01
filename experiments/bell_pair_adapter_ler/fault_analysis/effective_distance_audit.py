"""Checkpointed, physical-location PPM fault-distance search through weight four.

The source single-fault audit is read-only. Full signatures validate all hash
matches: fingerprints only accelerate candidate lookup, never certify equality.
No noise sweep, circuit modification, or Monte Carlo LER collection is performed.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import contextlib
import csv
import fcntl
import gzip
import hashlib
import itertools
import json
import math
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import stim

RECORD = np.dtype([("key", "<u8"), ("pair", "<u8")])
BUCKETS = 128
ROWS_PER_CHUNK = 32
SCHEMA = 1


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextlib.contextmanager
def writer_lock(path: Path) -> Iterator[None]:
    with path.open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def signature_value(signature: dict[str, Any]) -> int:
    if any(o != 0 for o in signature["observables"]):
        raise ValueError("This audit targets exactly PPM observable zero")
    return sum(1 << (d + 1) for d in signature["detectors"]) | bool(signature["observables"])


class Patterns:
    """Distinct nonzero D/L signatures, retaining all compatible physical origins."""

    def __init__(self, catalogue: dict[str, Any]):
        self.catalogue = catalogue
        groups: dict[int, list[int]] = collections.defaultdict(list)
        for fid, fault in enumerate(catalogue["faults"]):
            value = signature_value(catalogue["signatures"][fault["signature"]])
            if value:
                groups[value].append(fid)
        self.values = sorted(groups)
        self.origins = [groups[value] for value in self.values]
        self.index = {value: i for i, value in enumerate(self.values)}
        self.sites = [
            (fault["instruction"], fault["target_start"]) for fault in catalogue["faults"]
        ]
        bits = max((v.bit_length() for v in self.values), default=1)
        # Linear fingerprint: D hashes into the upper 63 bits, L remains the low bit.
        basis = [1] + [
            (
                int.from_bytes(hashlib.sha256(f"ppm-distance:{i}".encode()).digest()[:8], "little")
                >> 1
            )
            << 1
            for i in range(1, bits)
        ]
        hashes = []
        for value in self.values:
            key = 0
            while value:
                bit = value & -value
                key ^= basis[bit.bit_length() - 1]
                value ^= bit
            hashes.append(key)
        self.keys = np.asarray(hashes, dtype=np.uint64)
        self.sorted_keys = np.unique(self.keys)

    def realization(self, ids: list[int]) -> list[int] | None:
        """Find a distinct-location realization; never silently pick just one origin."""
        if len(set(ids)) != len(ids):
            return None
        ordered = sorted(ids, key=lambda i: len(self.origins[i]))

        def visit(offset: int, used: set[tuple[int, int]], chosen: list[int]) -> list[int] | None:
            if offset == len(ordered):
                return chosen
            for fid in self.origins[ordered[offset]]:
                site = self.sites[fid]
                if site not in used:
                    result = visit(offset + 1, used | {site}, chosen + [fid])
                    if result is not None:
                        return result
            return None

        return visit(0, set(), [])


def low_weights(patterns: Patterns) -> list[int] | None:
    """Complete search of weights one and two."""
    if 1 in patterns.index:
        return patterns.realization([patterns.index[1]])
    for i, value in enumerate(patterns.values):
        j = patterns.index.get(value ^ 1)
        if j is not None and i < j:
            result = patterns.realization([i, j])
            if result is not None:
                return result
    return None


def make_pairs(patterns: Patterns, start: int, end: int) -> np.ndarray[Any, Any]:
    n = len(patterns.values)
    records = np.empty(sum(n - i - 1 for i in range(start, end)), dtype=RECORD)
    offset = 0
    for i in range(start, end):
        length = n - i - 1
        records["key"][offset : offset + length] = patterns.keys[i] ^ patterns.keys[i + 1 :]
        records["pair"][offset : offset + length] = np.uint64(i << 32) | np.arange(
            i + 1, n, dtype=np.uint64
        )
        offset += length
    return records


def unpack(pair: int) -> tuple[int, int]:
    return pair >> 32, pair & ((1 << 32) - 1)


def triples(patterns: Patterns, records: np.ndarray[Any, Any]) -> list[int] | None:
    """Every pair plus a single; fingerprint matches are checked with full D/L."""
    if not len(patterns.sorted_keys):
        return None
    targets = records["key"] ^ np.uint64(1)
    pos = np.searchsorted(patterns.sorted_keys, targets)
    pos = np.minimum(pos, len(patterns.sorted_keys) - 1)
    for row in np.flatnonzero(patterns.sorted_keys[pos] == targets):
        i, j = unpack(int(records["pair"][row]))
        k = patterns.index.get(patterns.values[i] ^ patterns.values[j] ^ 1)
        if k is not None:
            result = patterns.realization([i, j, k])
            if result is not None:
                return result
    return None


def quadruples(patterns: Patterns, records: np.ndarray[Any, Any]) -> list[int] | None:
    """Find opposite-L pairs with equal D; test every physical compatibility match."""
    if len(records) < 2:
        return None
    records.sort(order="key")
    keys = records["key"]
    transitions = (
        np.flatnonzero(
            ((keys[:-1] >> 1) == (keys[1:] >> 1)) & ((keys[:-1] & 1) == 0) & ((keys[1:] & 1) == 1)
        )
        + 1
    )
    for mid in transitions:
        first = int(np.searchsorted(keys, keys[mid] ^ np.uint64(1), side="left"))
        last = int(np.searchsorted(keys, keys[mid], side="right"))
        for a in range(first, int(mid)):
            i, j = unpack(int(records["pair"][a]))
            left = patterns.values[i] ^ patterns.values[j]
            for b in range(int(mid), last):
                k, ell = unpack(int(records["pair"][b]))
                if left ^ patterns.values[k] ^ patterns.values[ell] != 1:
                    continue
                result = patterns.realization([i, j, k, ell])
                if result is not None:
                    return result
    return None


def inject(circuit: stim.Circuit, faults: list[dict[str, Any]]) -> stim.Circuit:
    """Inject simultaneous Pauli and measurement faults at distinct primitive sites."""
    sites = [(f["instruction"], f["target_start"]) for f in faults]
    if len(sites) != len(set(sites)):
        raise ValueError("Mutually exclusive outcomes at one physical location")
    selected: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for fault in faults:
        selected[fault["instruction"]].append(fault)
    out = stim.Circuit()
    for index, op in enumerate(circuit.flattened()):
        chosen = selected.pop(index, [])
        if chosen and op.name in {"M", "MX", "MY"}:
            flips = {f["target_start"] for f in chosen}
            for start, target in enumerate(op.targets_copy()):
                out.append(op.name, [target], [float(start in flips)], tag=op.tag)
        else:
            piece = stim.Circuit()
            piece.append(op)
            out += piece.without_noise()
            for fault in chosen:
                if fault["gate"] != op.name or not fault["paulis"]:
                    raise ValueError("Fault does not match saved physical operation")
                targets = [
                    getattr(stim, f"target_{p[0].lower()}")(int(p[1:])) for p in fault["paulis"]
                ]
                out.append("CORRELATED_ERROR", targets, 1)
    if selected:
        raise ValueError("Fault instruction is outside the circuit")
    return out


def verify_history(circuit: stim.Circuit, cat: dict[str, Any], ids: list[int]) -> dict[str, Any]:
    faults = [cat["faults"][i] for i in ids]
    value = 0
    for f in faults:
        value ^= signature_value(cat["signatures"][f["signature"]])
    expected_d = np.asarray([(value >> (d + 1)) & 1 for d in range(circuit.num_detectors)])
    expected_o = np.asarray([value & 1])
    dets, obs = (
        inject(circuit, faults)
        .compile_detector_sampler(seed=719)
        .sample(16, separate_observables=True)
    )
    if not (np.all(dets == expected_d) and np.all(obs == expected_o)):
        raise ValueError("Multi-fault Stim replay disagrees with XOR prediction")
    sites = {(f["instruction"], f["target_start"]): f["site_probability"] for f in cat["faults"]}
    log_probability = sum(math.log1p(-p) for p in sites.values())
    log_probability += sum(math.log(f["probability"] / (1 - f["site_probability"])) for f in faults)
    return {
        "fault_ids": ids,
        "physical_faults": faults,
        "detectors": np.flatnonzero(expected_d).tolist(),
        "observable_flip": int(expected_o[0]),
        "verified_shots": 16,
        "exact_history_probability_no_other_faults": math.exp(log_probability),
    }


def verify_witness(circuit: stim.Circuit, cat: dict[str, Any], ids: list[int]) -> dict[str, Any]:
    combined = verify_history(circuit, cat, ids)
    if combined["detectors"] or combined["observable_flip"] != 1:
        raise ValueError("Candidate is not an undetected PPM flip")
    split = len(ids) // 2
    a = verify_history(circuit, cat, ids[:split])
    b = verify_history(circuit, cat, ids[split:])
    if a["detectors"] != b["detectors"] or a["observable_flip"] == b["observable_flip"]:
        raise ValueError("Candidate does not split into ambiguous histories")
    return {
        "weight": len(ids),
        "combined": combined,
        "ambiguous_histories": [a, b],
        "two_history_decoder_independent_lower_bound": min(
            a["exact_history_probability_no_other_faults"],
            b["exact_history_probability_no_other_faults"],
        ),
        "caveat": "Two exact mutually exclusive histories only, with all other sites fault-free. Not the total LER, not all realizations of this mechanism, and not a distance claim for other observables or noise locations.",
    }


def load_source(source: Path) -> tuple[stim.Circuit, dict[str, Any], dict[str, Any]]:
    identity = json.loads((source / "identity.json").read_text())
    circuit = stim.Circuit((source / "circuit.stim").read_text())
    dem = stim.DetectorErrorModel((source / "model.dem").read_text())
    raw = (source / "catalogue.json.gz").read_bytes()
    if digest(raw) != json.loads((source / "catalogue_hash.json").read_text())["sha256"]:
        raise ValueError("Source catalogue changed")
    if digest(str(circuit).encode()) != identity["circuit_sha256"]:
        raise ValueError("Source circuit changed")
    if digest(str(dem).encode()) != identity["dem_sha256"]:
        raise ValueError("Source DEM changed")
    if circuit.num_observables != 1 or circuit.detector_error_model() != dem:
        raise ValueError("Invalid source circuit/DEM")
    frozen = {
        "schema": SCHEMA,
        "source": str(source.resolve()),
        "source_identity": identity,
        "catalogue_sha256": digest(raw),
        "source_identity_sha256": digest((source / "identity.json").read_bytes()),
        "search_source_sha256": digest(Path(__file__).read_bytes()),
        "stim_version": stim.__version__,
        "numpy_version": np.__version__,
        "buckets": BUCKETS,
        "rows_per_chunk": ROWS_PER_CHUNK,
    }
    return circuit, json.loads(gzip.decompress(raw)), frozen


def save_chunk(
    directory: Path, records: np.ndarray[Any, Any], start: int, end: int
) -> dict[str, Any]:
    bucket = (records["key"] >> 1) & (BUCKETS - 1)
    order = np.argsort(bucket, kind="stable")
    counts = np.bincount(bucket.astype(np.int64), minlength=BUCKETS)
    offsets = np.concatenate(([0], np.cumsum(counts))).tolist()
    name = f"pairs_{start:06d}_{end:06d}.npy"
    temporary = directory / (name + ".part")
    with temporary.open("wb") as stream:
        np.save(stream, records[order], allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(directory / name)
    return {
        "name": name,
        "start": start,
        "end": end,
        "offsets": offsets,
        "sha256": digest((directory / name).read_bytes()),
        "pairs": len(records),
    }


def bucket_records(
    directory: Path, chunks: list[dict[str, Any]], bucket: int
) -> np.ndarray[Any, Any]:
    pieces = []
    for chunk in chunks:
        arr = np.load(directory / chunk["name"], mmap_mode="r", allow_pickle=False)
        begin, end = chunk["offsets"][bucket : bucket + 2]
        pieces.append(np.array(arr[begin:end]))
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=RECORD)


def search(
    directory: Path, circuit: stim.Circuit, cat: dict[str, Any], seconds: float
) -> dict[str, Any]:
    start_time = time.perf_counter()
    deadline = start_time + seconds
    path = directory / "progress.json"
    state: dict[str, Any] = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "excluded_through_weight": 0,
            "next_row": 0,
            "next_bucket": 0,
            "chunks": [],
            "search_seconds": 0.0,
            "complete": False,
        }
    )
    for chunk in state["chunks"]:
        if digest((directory / chunk["name"]).read_bytes()) != chunk["sha256"]:
            raise ValueError("Pair checkpoint changed")
    if state["complete"]:
        return state
    patterns = Patterns(cat)
    n = len(patterns.values)
    state["unique_nonzero_signatures"] = n
    state["total_signature_pairs"] = n * (n - 1) // 2

    def checkpoint() -> None:
        nonlocal start_time
        now = time.perf_counter()
        state["search_seconds"] += now - start_time
        start_time = now
        atomic_json(path, state)

    def found(ids: list[int]) -> dict[str, Any]:
        witness = verify_witness(circuit, cat, ids)
        atomic_json(directory / "witness.json", witness)
        state["upper_bound"] = witness["weight"]
        state["complete"] = True
        checkpoint()
        return state

    if state["excluded_through_weight"] == 0:
        ids = low_weights(patterns)
        if ids is not None:
            state["excluded_through_weight"] = len(ids) - 1
            return found(ids)
        state["excluded_through_weight"] = 2
        checkpoint()

    while state["next_row"] < max(0, n - 1):
        if time.perf_counter() >= deadline:
            checkpoint()
            return state
        start = state["next_row"]
        end = min(start + ROWS_PER_CHUNK, n - 1)
        records = make_pairs(patterns, start, end)
        ids = triples(patterns, records)
        if ids is not None:
            return found(ids)
        chunk = save_chunk(directory, records, start, end)
        state["chunks"].append(chunk)
        state["next_row"] = end
        checkpoint()
    state["excluded_through_weight"] = 3
    checkpoint()

    while state["next_bucket"] < BUCKETS:
        if time.perf_counter() >= deadline:
            checkpoint()
            return state
        records = bucket_records(directory, state["chunks"], state["next_bucket"])
        ids = quadruples(patterns, records)
        if ids is not None:
            return found(ids)
        state["next_bucket"] += 1
        checkpoint()
    state["excluded_through_weight"] = 4
    state["complete"] = True
    checkpoint()
    return state


def condition(source: str, output: str, seconds: float) -> dict[str, Any]:
    src, dst = Path(source), Path(output)
    dst.mkdir(parents=True, exist_ok=True)
    with writer_lock(dst / ".writer.lock"):
        circuit, cat, identity = load_source(src)
        identity_path = dst / "identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError("Audit identity changed; use a new output directory")
        else:
            if any(p.name != ".writer.lock" for p in dst.iterdir()):
                raise ValueError("Nonempty output without identity")
            atomic_json(identity_path, identity)
        print(f"{src.name}: starting/resuming, {seconds:g}s search budget", flush=True)
        state = search(dst, circuit, cat, seconds)
        summary = {
            "condition": src.name,
            "distance_label": identity["source_identity"]["distance_label"],
            "rounds": identity["source_identity"]["rounds"],
            "basis": identity["source_identity"]["basis"],
            "fault_distance_lower_bound": state["excluded_through_weight"] + 1,
            "fault_distance_upper_bound": state.get("upper_bound"),
            "exact_distance": state.get("upper_bound")
            if state.get("upper_bound") == state["excluded_through_weight"] + 1
            else None,
            "complete_through_four_or_minimum_found": state["complete"],
            "search_seconds": state["search_seconds"],
            "signature_pairs_generated": sum(c["pairs"] for c in state["chunks"]),
            "total_signature_pairs": state["total_signature_pairs"],
            "four_fault_buckets_excluded": state["next_bucket"],
            "checkpoint_bytes": sum((dst / c["name"]).stat().st_size for c in state["chunks"]),
        }
        if (dst / "witness.json").exists():
            witness = json.loads((dst / "witness.json").read_text())
            summary["verified_witness_weight"] = witness["weight"]
            summary["two_history_ler_lower_bound"] = witness[
                "two_history_decoder_independent_lower_bound"
            ]
        atomic_json(dst / "summary.json", summary)
        print(json.dumps(summary), flush=True)
        return summary


def summarize(root: Path) -> list[dict[str, Any]]:
    rows = [json.loads(p.read_text()) for p in sorted(root.glob("d*_r*_[XZ]/summary.json"))]
    atomic_json(root / "summary.json", rows)
    if rows:
        with (root / "summary.csv").open("w") as stream:
            writer = csv.DictWriter(stream, sorted(set().union(*(r.keys() for r in rows))))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seconds-per-condition", type=float, default=600)
    args = parser.parse_args()
    if args.workers < 1 or args.seconds_per_condition <= 0:
        parser.error("workers and seconds-per-condition must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_dir.resolve() == args.source_dir.resolve():
        parser.error("The source audit is read-only; choose another output directory")
    names = args.conditions or [
        f"d{d}_r{r}_{b}" for d, r, b in itertools.product((4, 8), (4, 8), ("X", "Z"))
    ]
    if len(names) != len(set(names)) or any(
        name not in {f"d{d}_r{r}_{b}" for d, r, b in itertools.product((4, 8), (4, 8), ("X", "Z"))}
        for name in names
    ):
        parser.error("Invalid or duplicate condition")
    with writer_lock(args.output_dir / ".writer.lock"):
        started = time.time()
        with (args.output_dir / "invocations.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "started_unix": started,
                        "conditions": names,
                        "workers": args.workers,
                        "seconds_per_condition": args.seconds_per_condition,
                    }
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(
                    condition,
                    str(args.source_dir / n),
                    str(args.output_dir / n),
                    args.seconds_per_condition,
                )
                for n in names
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
                summarize(args.output_dir)
        summarize(args.output_dir)
        print(f"Finished bounded audit in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
