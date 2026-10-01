"""Probability accounting and unchanged-decoder checks for verified distance witnesses."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import stim

from qldpc.decoders import SinterDecoder

from .effective_distance_audit import atomic_json, digest, load_source, verify_history
from .single_fault_audit import runtime_identity


def realization_mass(cat: dict[str, Any], signature_ids: list[int], limit: int) -> dict[str, Any]:
    if len(signature_ids) != len(set(signature_ids)):
        raise ValueError("Expected distinct signatures in a minimum-weight witness")
    origins = [
        [i for i, f in enumerate(cat["faults"]) if f["signature"] == sid] for sid in signature_ids
    ]
    sites = {(f["instruction"], f["target_start"]): f["site_probability"] for f in cat["faults"]}
    p0 = math.exp(sum(math.log1p(-p) for p in sites.values()))
    candidates = math.prod(map(len, origins))
    checked = valid = 0
    mass = best = 0.0
    representative: list[int] | None = None
    for ids in itertools.islice(itertools.product(*origins), limit):
        checked += 1
        faults = [cat["faults"][i] for i in ids]
        if len({(f["instruction"], f["target_start"]) for f in faults}) != len(faults):
            continue
        valid += 1
        probability = p0 * math.prod(f["probability"] / (1 - f["site_probability"]) for f in faults)
        mass += probability
        if probability > best:
            best, representative = probability, list(ids)
    return {
        "signature_ids": signature_ids,
        "origins_per_signature": list(map(len, origins)),
        "candidate_realizations": candidates,
        "checked_realizations": checked,
        "valid_distinct_location_realizations": valid,
        "complete_within_this_signature_family": checked == candidates,
        "probability_mass_no_other_faults": mass,
        "most_probable_checked_realization": representative,
    }


def explain(directory: Path, limit: int) -> dict[str, Any]:
    output = directory / "witness_analysis.json"
    if output.exists():
        raise ValueError("Do not overwrite existing witness analysis")
    identity = json.loads((directory / "identity.json").read_text())
    source = Path(identity["source"])
    circuit, cat, frozen = load_source(source)
    if identity != frozen:
        raise ValueError("Search/source identity changed")
    if frozen["source_identity"]["runtime_identity"] != runtime_identity():
        raise ValueError("Production decoder runtime changed")
    witness = json.loads((directory / "witness.json").read_text())
    summaries = []
    for history in witness["ambiguous_histories"]:
        signatures = [cat["faults"][fid]["signature"] for fid in history["fault_ids"]]
        row = realization_mass(cat, signatures, limit)
        row["observable_flip"] = history["observable_flip"]
        if row["most_probable_checked_realization"] is not None:
            row["verified_representative"] = verify_history(
                circuit, cat, row["most_probable_checked_realization"]
            )
            if (
                row["verified_representative"]["detectors"] != history["detectors"]
                or row["verified_representative"]["observable_flip"] != history["observable_flip"]
            ):
                raise ValueError("Alternative physical origin changed the signature")
        summaries.append(row)
    dem = stim.DetectorErrorModel((source / "model.dem").read_text())
    start = time.perf_counter()
    decoder = SinterDecoder().compile_decoder_for_dem(dem)
    setup_seconds = time.perf_counter() - start
    syndrome = np.zeros(circuit.num_detectors, dtype=np.uint8)
    syndrome[witness["ambiguous_histories"][0]["detectors"]] = 1
    start = time.perf_counter()
    prediction = int(decoder.decode(syndrome)[0])
    report = {
        "condition": directory.name,
        "witness_sha256": digest((directory / "witness.json").read_bytes()),
        "analysis_source_sha256": digest(Path(__file__).read_bytes()),
        "histories": summaries,
        "decoder_prediction": prediction,
        "decoder_setup_seconds": setup_seconds,
        "decode_seconds": time.perf_counter() - start,
        "bp_converged": bool(getattr(decoder.decoder, "converge")),
        "bp_iterations": int(getattr(decoder.decoder, "iter")),
        "family_decoder_independent_ler_lower_bound": min(
            row["probability_mass_no_other_faults"] for row in summaries
        ),
        "family_production_decoder_ler_contribution": sum(
            row["probability_mass_no_other_faults"]
            for row in summaries
            if row["observable_flip"] != prediction
        ),
        "caveat": "Only these two disjoint signature families with all other physical sites fault-free; this is not a Monte Carlo estimate, not all failure mechanisms, and not a full-noise optimal-decoder calculation. If truncated, masses and contributions are partial lower bounds.",
    }
    atomic_json(output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "histories"}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--max-realizations", type=int, default=200000)
    args = parser.parse_args()
    if args.max_realizations < 1:
        parser.error("max-realizations must be positive")
    for directory in args.directories:
        explain(directory, args.max_realizations)


if __name__ == "__main__":
    main()
