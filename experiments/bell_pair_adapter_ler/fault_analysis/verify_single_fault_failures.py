"""Replay saved single-fault decoder failures without changing circuits or priors."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import stim

from qldpc.decoders import SinterDecoder

from .single_fault_audit import inject_one, runtime_identity, save_json


def verify(directory: Path) -> list[dict[str, Any]]:
    identity = json.loads((directory / "identity.json").read_text())
    if identity["runtime_identity"] != runtime_identity():
        raise ValueError("Decoder identity changed")
    with gzip.open(directory / "catalogue.json.gz", "rt") as stream:
        cat = json.load(stream)
    saved = [
        json.loads(line) for line in (directory / "decoder_progress.jsonl").read_text().splitlines()
    ]
    failures = [
        r
        for r in saved
        if r["predicted_observables"] != cat["signatures"][r["signature"]]["observables"]
    ]
    circuit = stim.Circuit((directory / "circuit.stim").read_text())
    dem = stim.DetectorErrorModel((directory / "model.dem").read_text())
    reports = []
    for row in failures:
        signature = cat["signatures"][row["signature"]]
        origins = [f for f in cat["faults"] if f["signature"] == row["signature"]]
        fault = origins[0]
        injected = inject_one(circuit, fault)
        dets, obs = injected.compile_detector_sampler(seed=42).sample(8, separate_observables=True)
        syndrome = np.zeros(circuit.num_detectors, dtype=int)
        syndrome[signature["detectors"]] = 1
        actual = np.zeros(circuit.num_observables, dtype=int)
        actual[signature["observables"]] = 1
        if not (np.all(dets == syndrome) and np.all(obs == actual)):
            raise ValueError("Physical injection did not reproduce the recorded signature")
        predictions, candidates = [], []
        for preceding_zero in (False, True):
            compiled = SinterDecoder().compile_decoder_for_dem(dem)
            if preceding_zero:
                compiled.decode(np.zeros(compiled.num_detectors, dtype=int))
            inferred = np.asarray(compiled.decoder.decode(syndrome), dtype=int)
            arrays = compiled.dem_arrays
            predicted = np.asarray(arrays.observable_flip_matrix @ inferred).ravel() % 2
            predictions.append(np.flatnonzero(predicted).tolist())
            columns = np.flatnonzero(inferred).tolist()
            costs = np.log((1 - arrays.error_probs) / arrays.error_probs)
            matching = []
            for j in range(arrays.num_errors):
                if np.array_equal(arrays.detector_flip_matrix[:, j].toarray().ravel(), syndrome):
                    matching.append(
                        {
                            "column": j,
                            "observables": np.flatnonzero(
                                arrays.observable_flip_matrix[:, j].toarray().ravel()
                            ).tolist(),
                            "log_odds_cost": float(costs[j]),
                        }
                    )
            candidates.append(
                {
                    "preceding_zero": preceding_zero,
                    "selected_dem_columns": columns,
                    "selected_log_odds_cost": float(costs[inferred.astype(bool)].sum()),
                    "syndrome_satisfied": bool(
                        np.array_equal(
                            np.asarray(arrays.detector_flip_matrix @ inferred).ravel() % 2, syndrome
                        )
                    ),
                    "matching_single_dem_columns": matching,
                }
            )
        report = {
            "signature_id": row["signature"],
            "signature": signature,
            "physical_origins": origins,
            "injected_shots": 8,
            "saved_prediction": row["predicted_observables"],
            "fresh_predictions": predictions,
            "reproduced": all(p == row["predicted_observables"] for p in predictions),
            "candidate_diagnostics": candidates,
            "caveat": "Costs compare individual DEM error patterns, not sums over logical classes. This is not an optimal logical-decoder calculation.",
        }
        reports.append(report)
        save_json(directory / "verified_decoder_failures.json", reports)
        print(
            directory.name,
            row["signature"],
            "reproduced",
            report["reproduced"],
            "candidates",
            candidates,
            flush=True,
        )
    return reports


def compare_eight_rounds(directory: Path) -> list[dict[str, Any]]:
    """Replay only the already-verified four-round physical faults in eight rounds."""
    target = directory.with_name(directory.name.replace("_r4_", "_r8_"))
    if target == directory:
        raise ValueError("Expected a four-round source directory")
    identity = json.loads((target / "identity.json").read_text())
    if identity["runtime_identity"] != runtime_identity():
        raise ValueError("Decoder identity changed")
    verified = json.loads((directory / "verified_decoder_failures.json").read_text())
    with gzip.open(target / "catalogue.json.gz", "rt") as stream:
        cat = json.load(stream)
    circuit = stim.Circuit((target / "circuit.stim").read_text())
    dem = stim.DetectorErrorModel((target / "model.dem").read_text())
    reports = []
    for entry in verified:
        origin = entry["physical_origins"][0]
        matches = [
            f
            for f in cat["faults"]
            if all(
                f[k] == origin[k] for k in ("gate", "qubits", "paulis", "tick", "phase", "round")
            )
        ]
        if len(matches) != 1:
            raise ValueError("Could not uniquely match physical fault to eight-round circuit")
        fault = matches[0]
        signature = cat["signatures"][fault["signature"]]
        dets, obs = (
            inject_one(circuit, fault)
            .compile_detector_sampler(seed=91)
            .sample(4, separate_observables=True)
        )
        syndrome = np.zeros(circuit.num_detectors, dtype=int)
        syndrome[signature["detectors"]] = 1
        actual = np.zeros(circuit.num_observables, dtype=int)
        actual[signature["observables"]] = 1
        if not (np.all(dets == syndrome) and np.all(obs == actual)):
            raise ValueError("Eight-round injection disagrees with catalogue")
        compiled = SinterDecoder().compile_decoder_for_dem(dem)
        prediction = np.flatnonzero(compiled.decode(syndrome)).tolist()
        row = {
            "source_signature": entry["signature_id"],
            "source_actual": entry["signature"],
            "source_prediction": entry["saved_prediction"],
            "target_signature": signature,
            "target_prediction": prediction,
            "target_correct": prediction == signature["observables"],
            "physical_fault": fault,
            "source_directory": str(directory),
            "target_directory": str(target),
        }
        reports.append(row)
        save_json(directory / "round_extension_replays.json", reports)
        print(
            directory.name,
            entry["signature_id"],
            "eight-round correct",
            row["target_correct"],
            flush=True,
        )
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--compare-eight-rounds", action="store_true")
    args = parser.parse_args()
    for directory in args.directories:
        if args.compare_eight_rounds:
            compare_eight_rounds(directory)
        else:
            verify(directory)
