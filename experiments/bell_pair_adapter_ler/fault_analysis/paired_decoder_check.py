"""Immutable Stim sample banks and checkpointed, paired BP+OSD timing checks.

Default scope is 100 shots per basis, not an automatic 1,000/10,000-shot run.
Truth is used by the scorer only, never by decoder configuration or prediction.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import stim
from scipy.stats import beta, binomtest

from qldpc.decoders import SinterDecoder

from .effective_distance_audit import atomic_json, digest, load_source, writer_lock
from .single_fault_audit import runtime_identity

CHUNK_SHOTS = 100
VARIANTS: dict[str, dict[str, Any]] = {
    "production": {},
    "osd_cs2": {"osd_method": "OSD_CS", "osd_order": 2},
}


def settings(decoder: Any) -> dict[str, Any]:
    result = {}
    for key in (
        "bp_method",
        "max_iter",
        "osd_method",
        "osd_order",
        "schedule",
        "ms_scaling_factor",
        "omp_thread_count",
        "random_serial_schedule",
    ):
        value = getattr(decoder, key, None)
        result[key] = value.item() if isinstance(value, np.generic) else value
    return result


def prepare(source: Path, directory: Path, shots: int, seed: int) -> None:
    circuit, _, frozen = load_source(source)
    if frozen["source_identity"]["runtime_identity"] != runtime_identity():
        raise ValueError("Archived decoder runtime changed")
    directory.mkdir(parents=True, exist_ok=True)
    identity = {
        "source": frozen,
        "master_seed": seed,
        "chunk_shots": CHUNK_SHOTS,
        "runner_sha256": digest(Path(__file__).read_bytes()),
        "variants": VARIANTS,
    }
    path = directory / "identity.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError("Sample-bank identity changed")
    else:
        if any(directory.iterdir()):
            raise ValueError("Nonempty sample directory without identity")
        atomic_json(path, identity)
    for chunk in range(math.ceil(shots / CHUNK_SHOTS)):
        manifest = directory / f"samples_{chunk:06d}.json"
        data_path = directory / f"samples_{chunk:06d}.npz"
        if manifest.exists():
            if digest(data_path.read_bytes()) != json.loads(manifest.read_text())["sha256"]:
                raise ValueError("Saved sample bank changed")
            continue
        chunk_seed = int(digest(f"{seed}:{chunk}".encode())[:16], 16)
        start = time.perf_counter()
        dets, truth = circuit.compile_detector_sampler(seed=chunk_seed).sample(
            CHUNK_SHOTS, separate_observables=True, bit_packed=True
        )
        temporary = data_path.with_suffix(".part")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, detectors=dets, truth=truth)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(data_path)
        atomic_json(
            manifest,
            {
                "chunk": chunk,
                "shot_start": chunk * CHUNK_SHOTS,
                "shots": CHUNK_SHOTS,
                "seed": chunk_seed,
                "sha256": digest(data_path.read_bytes()),
                "sampling_seconds": time.perf_counter() - start,
                "num_detectors": circuit.num_detectors,
                "num_observables": circuit.num_observables,
            },
        )


def bank(directory: Path, shots: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    detectors, truth = [], []
    for chunk in range(math.ceil(shots / CHUNK_SHOTS)):
        path = directory / f"samples_{chunk:06d}.npz"
        manifest = json.loads((directory / f"samples_{chunk:06d}.json").read_text())
        if digest(path.read_bytes()) != manifest["sha256"]:
            raise ValueError("Sample bank checksum mismatch")
        with np.load(path, allow_pickle=False) as data:
            d = np.unpackbits(
                data["detectors"], axis=1, bitorder="little", count=manifest["num_detectors"]
            )
            o = np.unpackbits(data["truth"], axis=1, bitorder="little", count=1)[:, 0]
        if len(d) != CHUNK_SHOTS or len(o) != CHUNK_SHOTS:
            raise ValueError("Invalid sample chunk shape")
        detectors.append(d)
        truth.append(o)
    return np.concatenate(detectors)[:shots], np.concatenate(truth)[:shots]


def decode(directory_text: str, variant: str, shots: int, seconds: float) -> dict[str, Any]:
    directory = Path(directory_text)
    output = directory / variant
    output.mkdir(exist_ok=True)
    with writer_lock(output / ".writer.lock"):
        identity = json.loads((directory / "identity.json").read_text())
        source = Path(identity["source"]["source"])
        _, _, frozen = load_source(source)
        if identity["source"] != frozen or identity["runner_sha256"] != digest(
            Path(__file__).read_bytes()
        ):
            raise ValueError("Decoder/source identity changed")
        if frozen["source_identity"]["runtime_identity"] != runtime_identity():
            raise ValueError("Decoder runtime changed")
        observations, _ = bank(directory, shots)
        start = time.perf_counter()
        model = stim.DetectorErrorModel((source / "model.dem").read_text())
        compiled = SinterDecoder(**VARIANTS[variant]).compile_decoder_for_dem(model)
        setup = time.perf_counter() - start
        resolved = settings(compiled.decoder)
        expected = json.loads((source / "decoder_settings.json").read_text())
        if variant == "osd_cs2":
            expected.update(osd_method="OSD_CS", osd_order=2)
        if any(resolved[k] != v for k, v in expected.items()):
            raise ValueError(f"Unexpected decoder settings: {resolved}, expected {expected}")
        config = {
            "variant": variant,
            "kwargs": VARIANTS[variant],
            "resolved": resolved,
            "sample_identity_sha256": digest((directory / "identity.json").read_bytes()),
        }
        config_path = output / "config.json"
        if config_path.exists():
            if json.loads(config_path.read_text()) != config:
                raise ValueError("Decoder configuration changed")
        else:
            atomic_json(config_path, config)
        completed = 0
        started = time.perf_counter()
        print(f"{directory.name}/{variant}: {shots} paired shots, setup {setup:.2f}s", flush=True)
        for shot_id, syndrome in enumerate(observations):
            path = output / f"shot_{shot_id:08d}.json"
            if path.exists():
                row = json.loads(path.read_text())
                if row["shot_id"] != shot_id or row["config_sha256"] != digest(
                    config_path.read_bytes()
                ):
                    raise ValueError("Foreign or corrupt prediction checkpoint")
                completed += 1
                continue
            if time.perf_counter() - started >= seconds:
                break
            atomic_json(
                output / "in_flight.json", {"shot_id": shot_id, "started_unix": time.time()}
            )
            wall, cpu = time.perf_counter(), time.process_time()
            prediction = int(compiled.decode(syndrome)[0])
            elapsed_cpu, elapsed_wall = time.process_time() - cpu, time.perf_counter() - wall
            atomic_json(
                path,
                {
                    "shot_id": shot_id,
                    "prediction": prediction,
                    "config_sha256": digest(config_path.read_bytes()),
                    "decode_wall_seconds": elapsed_wall,
                    "decode_cpu_seconds": elapsed_cpu,
                    "bp_converged": bool(getattr(compiled.decoder, "converge")),
                    "bp_iterations": int(getattr(compiled.decoder, "iter")),
                },
            )
            completed += 1
            atomic_json(output / "in_flight.json", {"shot_id": None, "completed": completed})
            if completed % 10 == 0:
                print(f"{directory.name}/{variant}: {completed}/{shots} checkpointed", flush=True)
        report = {
            "condition": directory.name,
            "variant": variant,
            "requested_shots": shots,
            "completed_shots": completed,
            "complete": completed == shots,
            "setup_seconds": setup,
            "invocation_wall_seconds": time.perf_counter() - started,
            "process_peak_rss_bytes_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
        }
        atomic_json(output / f"invocation_{time.time_ns()}.json", report)
        return report


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    return (
        float(beta.ppf(alpha / 2, k, n - k + 1)) if k else 0.0,
        float(beta.ppf(1 - alpha / 2, k + 1, n - k)) if k < n else 1.0,
    )


def paired_metrics(
    truth: np.ndarray[Any, Any], a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]
) -> dict[str, Any]:
    n = len(truth)
    if not n or len(a) != n or len(b) != n:
        raise ValueError("Paired comparison requires equal nonempty complete samples")
    fail_a, fail_b = a != truth, b != truth
    rescued = int(np.count_nonzero(fail_a & ~fail_b))
    introduced = int(np.count_nonzero(~fail_a & fail_b))
    lo_i, hi_i = clopper_pearson(introduced, n, alpha=0.025)
    lo_r, hi_r = clopper_pearson(rescued, n, alpha=0.025)
    return {
        "shots": n,
        "production_errors": int(sum(fail_a)),
        "alternative_errors": int(sum(fail_b)),
        "rescued": rescued,
        "introduced": introduced,
        "both_fail": int(sum(fail_a & fail_b)),
        "both_correct": int(sum(~fail_a & ~fail_b)),
        "delta_alternative_minus_production": (introduced - rescued) / n,
        "delta_95ci_conservative_paired": [lo_i - hi_r, hi_i - lo_r],
        "exact_mcnemar_pvalue": float(binomtest(introduced, introduced + rescued, 0.5).pvalue)
        if introduced + rescued
        else 1.0,
    }


def summarize(directory: Path, shots: int) -> dict[str, Any]:
    _, truth = bank(directory, shots)
    results, predictions, configs = {}, {}, {}
    for variant in VARIANTS:
        rows = [
            json.loads(path.read_text())
            for i in range(shots)
            if (path := directory / variant / f"shot_{i:08d}.json").exists()
        ]
        complete = len(rows) == shots
        wall = [r["decode_wall_seconds"] for r in rows]
        cpu = [r["decode_cpu_seconds"] for r in rows]
        errors = sum(r["prediction"] != truth[r["shot_id"]] for r in rows)
        results[variant] = {
            "requested_shots": shots,
            "completed_shots": len(rows),
            "complete": complete,
            "errors_on_completed_shots": int(errors),
            "ler": errors / shots if complete else None,
            "ler_95ci_exact": list(clopper_pearson(errors, shots)) if complete else None,
            "total_decode_cpu_seconds": sum(cpu),
            "total_decode_wall_seconds": sum(wall),
            "mean_decode_cpu_seconds": float(np.mean(cpu)) if cpu else None,
            "mean_decode_wall_seconds": float(np.mean(wall)) if wall else None,
            "median_decode_wall_seconds": float(np.median(wall)) if wall else None,
            "p95_decode_wall_seconds": float(np.quantile(wall, 0.95)) if wall else None,
            "max_decode_wall_seconds": max(wall) if wall else None,
            "bp_nonconverged_shots": sum(not r["bp_converged"] for r in rows),
        }
        if complete:
            predictions[variant] = np.asarray([r["prediction"] for r in rows])
        config_path = directory / variant / "config.json"
        if config_path.exists():
            configs[variant] = json.loads(config_path.read_text())["resolved"]
    if len(configs) == 2:
        unexpected = {
            k for k in configs["production"] if configs["production"][k] != configs["osd_cs2"][k]
        } - {"osd_method", "osd_order"}
        if unexpected:
            raise ValueError(f"Uncontrolled decoder settings: {unexpected}")
    report = {
        "condition": directory.name,
        "decoders": results,
        "paired": paired_metrics(truth, predictions["production"], predictions["osd_cs2"])
        if len(predictions) == 2
        else None,
        "caveat": "100-shot timing pilots are not precision LER estimates. Partial/time-censored samples do not receive LERs or paired significance claims. Paired interval uses conservative Bonferroni exact marginal bounds for rescued/introduced probabilities.",
    }
    atomic_json(directory / f"summary_{shots}.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seconds-per-decoder", type=float, default=600)
    parser.add_argument("--seed", type=int, default=9142026)
    args = parser.parse_args()
    if args.shots < 1 or args.workers < 1 or args.seconds_per_decoder <= 0:
        parser.error("Shot, worker, and time limits must be positive")
    if args.output_dir.resolve() == args.source_dir.resolve():
        parser.error("Choose a separate output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with writer_lock(args.output_dir / ".writer.lock"):
        started = time.time()
        for offset, basis in enumerate(("X", "Z")):
            name = f"d8_r8_{basis}"
            prepare(args.source_dir / name, args.output_dir / name, args.shots, args.seed + offset)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(
                    decode,
                    str(args.output_dir / f"d8_r8_{basis}"),
                    variant,
                    args.shots,
                    args.seconds_per_decoder,
                )
                for basis in ("X", "Z")
                for variant in VARIANTS
            ]
            for future in concurrent.futures.as_completed(futures):
                print(json.dumps(future.result()), flush=True)
                for basis in ("X", "Z"):
                    summarize(args.output_dir / f"d8_r8_{basis}", args.shots)
        rows = [summarize(args.output_dir / f"d8_r8_{b}", args.shots) for b in ("X", "Z")]
        atomic_json(args.output_dir / f"summary_{args.shots}.json", rows)
        atomic_json(
            args.output_dir / f"invocation_{time.time_ns()}.json",
            {
                "shots_per_basis": args.shots,
                "workers": args.workers,
                "wall_seconds": time.time() - started,
                "seconds_per_decoder": args.seconds_per_decoder,
            },
        )
        print(f"Paired check finished in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
