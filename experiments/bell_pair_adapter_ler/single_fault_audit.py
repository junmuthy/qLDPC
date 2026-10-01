"""Physical single-fault propagation and bounded, resumable production-decoder audit.

No Monte Carlo shots, two-fault combinations, distance searches, or circuit changes.
The catalogue enumerates physical channel outcomes, including silent outcomes, rather
than treating merged detector-error-model terms as individual physical locations.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import dataclasses
import fcntl
import gzip
import hashlib
import itertools
import json
import math
import multiprocessing
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import stim

from qldpc.decoders import SinterDecoder
from qldpc.objects import Pauli

from .run_experiment import build_experiment_for_codes
from .run_ppm_basis_first import code_pair
from .run_ppm_fidelity_sweep import runtime_identity

ROOT = Path(__file__).resolve().parent / "outputs"
MEASUREMENTS = {"M", "MX", "MY"}
PAULI_ERRORS = {"X_ERROR": "X", "Y_ERROR": "Y", "Z_ERROR": "Z"}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


@dataclasses.dataclass
class Fault:
    instruction: int  # zero-based in the saved flattened circuit
    target_start: int  # zero-based within that instruction
    gate: str
    qubits: list[int]
    paulis: list[str]  # empty for a classical measurement-result flip
    probability: float  # physical, unconditional probability of this channel outcome
    site_probability: float  # total nonidentity probability of this primitive channel
    signature: int = 0
    tick: int = 0
    source_gate: str = ""
    source_instructions: list[int] = dataclasses.field(default_factory=list)
    phase: str = "unclassified"
    round: int | None = None

    @property
    def key(self) -> tuple[int, int, tuple[str, ...]]:
        return self.instruction, self.target_start, tuple(self.paulis)


def physical_faults(circuit: stim.Circuit) -> list[Fault]:
    """Enumerate mutually exclusive outcomes separately at every primitive channel."""
    faults = []
    tick = 0
    last_touch: dict[int, tuple[int, str]] = {}
    for index, op in enumerate(circuit):
        if isinstance(op, stim.CircuitRepeatBlock):
            raise ValueError("Catalogue requires a flattened circuit")
        if op.name == "TICK":
            tick += 1
        targets = op.targets_copy()
        args = op.gate_args_copy()
        outcomes: list[tuple[int, list[int], list[str], float, float]] = []
        if op.name in {"DEPOLARIZE1", "DEPOLARIZE2", "PAULI_CHANNEL_1", *PAULI_ERRORS}:
            width = 2 if op.name == "DEPOLARIZE2" else 1
            if len(targets) % width:
                raise ValueError("Malformed channel targets")
            site_p = sum(args) if op.name == "PAULI_CHANNEL_1" else args[0]
            for start in range(0, len(targets), width):
                qubits = [t.value for t in targets[start : start + width]]
                if op.name == "DEPOLARIZE2":
                    patterns = [(s, site_p / 15) for s in itertools.product("IXYZ", repeat=2)]
                elif op.name == "DEPOLARIZE1":
                    patterns = [((s,), site_p / 3) for s in "XYZ"]
                elif op.name == "PAULI_CHANNEL_1":
                    patterns = [((s,), p) for s, p in zip("XYZ", args)]
                else:
                    patterns = [((PAULI_ERRORS[op.name],), site_p)]
                for pattern, probability in patterns:
                    paulis = [f"{p}{q}" for p, q in zip(pattern, qubits) if p != "I"]
                    if probability and paulis:
                        outcomes.append((start, qubits, paulis, probability, site_p))
        elif op.name in MEASUREMENTS:
            p = args[0] if args else 0
            if p:
                outcomes.extend((j, [t.value], [], p, p) for j, t in enumerate(targets))
        elif stim.gate_data(op.name).is_noisy_gate:
            raise ValueError(f"Unsupported noise/readout channel: {op.name}")
        for start, qubits, paulis, probability, site_p in outcomes:
            if not 0 < site_p < 1:
                raise ValueError("Audit requires primitive noise probabilities strictly below one")
            sources = (
                [(index, op.name)]
                if op.name in MEASUREMENTS
                else sorted({last_touch.get(q, (-1, "unknown")) for q in qubits})
            )
            faults.append(
                Fault(
                    index,
                    start,
                    op.name,
                    qubits,
                    paulis,
                    probability,
                    site_p,
                    tick=tick,
                    source_gate="+".join(sorted({s[1] for s in sources})),
                    source_instructions=[s[0] for s in sources],
                )
            )
        if (
            op.name in MEASUREMENTS
            or stim.gate_data(op.name).is_unitary
            or op.name in {"R", "RX", "RY"}
        ):
            for target in targets:
                if target.is_qubit_target:
                    last_touch[target.value] = (index, op.name)
    return faults


def pauli_text(target: stim.GateTarget) -> str:
    axis = "X" if target.is_x_target else "Y" if target.is_y_target else "Z"
    return f"{axis}{target.value}"


def annotate_phases(circuit: stim.Circuit, faults: list[Fault], metadata: dict[str, Any]) -> None:
    """Use adapter-data reset/readout and absolute detector epochs, not proximity."""
    data = set(metadata["adapter_hardware_qubits"][: metadata["num_adapter_data_qubits"]])
    resets, readouts = [], []
    next_epoch: dict[int, int] = {}
    epoch = None
    for index in reversed(range(len(circuit))):
        op = circuit[index]
        if op.name == "DETECTOR":
            epoch = int(op.gate_args_copy()[2])
        if epoch is not None:
            next_epoch[index] = epoch
        targets = {t.value for t in op.targets_copy() if t.is_qubit_target}
        if targets == data:
            if op.name in {"R", "RX", "RY"}:
                resets.append(index)
            elif op.name in MEASUREMENTS:
                readouts.append(index)
    if len(resets) != 1 or len(readouts) != 1:
        raise ValueError("Could not uniquely locate adapter initialization and detachment")
    for fault in faults:
        if fault.source_instructions == resets:
            fault.phase = "adapter_initialization"
        elif fault.instruction == readouts[0]:
            fault.phase = "detachment"
        else:
            fault.phase = "surgery"
            fault.round = next_epoch[fault.instruction] - metadata["preparation_rounds"] + 1
            if not 1 <= fault.round <= metadata["surgery_rounds"]:
                raise ValueError("A noisy operation lies outside the expected adapter rounds")


def catalogue(circuit: stim.Circuit, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    circuit = circuit.flattened()
    # Explanation alone need not reject nondeterministic, noise-free detectors.
    # Require Stim's strict full model analysis before enumerating any faults.
    circuit.detector_error_model()
    faults = physical_faults(circuit)
    lookup = {f.key: f for f in faults}
    if len(lookup) != len(faults):
        raise ValueError("Duplicate physical fault identity")
    signatures: list[dict[str, list[int]]] = [{"detectors": [], "observables": []}]
    assigned = set()
    for explained in circuit.explain_detector_error_model_errors():
        detectors, observables = [], []
        for item in explained.dem_error_terms:
            target = item.dem_target
            if target.is_relative_detector_id():
                detectors.append(target.val)
            elif target.is_logical_observable_id():
                observables.append(target.val)
            else:
                raise ValueError("Unexpected decomposed error term")
        if not detectors and not observables:
            continue
        sid = len(signatures)
        signatures.append({"detectors": sorted(detectors), "observables": sorted(observables)})
        for location in explained.circuit_error_locations:
            if len(location.stack_frames) != 1:
                raise ValueError("Expected a single frame in the flattened circuit")
            key = (
                location.stack_frames[0].instruction_offset,
                location.instruction_targets.target_range_start,
                tuple(pauli_text(t.gate_target) for t in location.flipped_pauli_product),
            )
            if key not in lookup or key in assigned:
                raise ValueError(f"Unmatched or duplicate physical explanation: {key}")
            assigned.add(key)
            lookup[key].signature = sid
        if not explained.circuit_error_locations:
            raise ValueError("A single-fault signature has no physical origin")
    if metadata is not None:
        annotate_phases(circuit, faults, metadata)
    sites = {(f.instruction, f.target_start): f.site_probability for f in faults}
    no_fault_probability = math.exp(sum(math.log1p(-p) for p in sites.values()))
    by_syndrome: dict[tuple[int, ...], set[tuple[int, ...]]] = collections.defaultdict(set)
    for s in signatures:
        by_syndrome[tuple(s["detectors"])].add(tuple(s["observables"]))
    metrics = {
        "noise_locations": len(sites),
        "physical_fault_outcomes": len(faults),
        "silent_fault_outcomes": sum(f.signature == 0 for f in faults),
        "nontrivial_signatures": len(signatures) - 1,
        "distinct_syndromes_including_zero": len(by_syndrome),
        "logical_flip_outcomes": sum(bool(signatures[f.signature]["observables"]) for f in faults),
        "undetected_logical_outcomes": sum(
            bool(signatures[f.signature]["observables"])
            and not signatures[f.signature]["detectors"]
            for f in faults
        ),
        "ambiguous_single_fault_syndromes_including_zero": sum(
            len(v) > 1 for v in by_syndrome.values()
        ),
        "max_detectors_from_one_fault": max(len(s["detectors"]) for s in signatures),
        "expected_number_of_faulty_locations": sum(sites.values()),
        "no_fault_probability": no_fault_probability,
        "exactly_one_fault_probability": sum(
            f.probability * no_fault_probability / (1 - f.site_probability) for f in faults
        ),
    }
    return {
        "metrics": metrics,
        "signatures": signatures,
        "faults": [dataclasses.asdict(f) for f in faults],
    }


def inject_one(circuit: stim.Circuit, fault: dict[str, Any]) -> stim.Circuit:
    """Retain the ideal circuit and inject exactly the specified physical outcome."""
    injected = stim.Circuit()
    for index, op in enumerate(circuit.flattened()):
        if index != fault["instruction"]:
            piece = stim.Circuit()
            piece.append(op)
            injected += piece.without_noise()
        elif fault["gate"] in MEASUREMENTS:
            for start, target in enumerate(op.targets_copy()):
                injected.append(
                    op.name, [target], [float(start == fault["target_start"])], tag=op.tag
                )
        else:
            targets = [getattr(stim, f"target_{p[0].lower()}")(int(p[1:])) for p in fault["paulis"]]
            injected.append("CORRELATED_ERROR", targets, 1)
    return injected


def validate_injection(circuit: stim.Circuit, cat: dict[str, Any], count: int = 12) -> None:
    """Cross-check propagated signatures with deterministic-fault Stim sampling."""
    faults = cat["faults"]
    candidates = random.Random(4721).sample(range(len(faults)), min(count, len(faults)))
    for index in candidates:
        fault = faults[index]
        injected = inject_one(circuit, fault)
        dets, obs = injected.compile_detector_sampler(seed=17).sample(4, separate_observables=True)
        signature = cat["signatures"][fault["signature"]]
        expected_dets = np.zeros(circuit.num_detectors, dtype=bool)
        expected_obs = np.zeros(circuit.num_observables, dtype=bool)
        expected_dets[signature["detectors"]] = True
        expected_obs[signature["observables"]] = True
        if not (np.all(dets == expected_dets) and np.all(obs == expected_obs)):
            raise ValueError(f"Deterministic injected fault does not match catalogue: {fault}")


def trace_example(circuit: stim.Circuit, fault: dict[str, Any], block_size: int) -> dict[str, Any]:
    """A Pauli-frame representative, not a minimum-weight error modulo stabilizers.

    Stabilizer randomization is disabled only for tracing an already-validated
    physical fault; the catalogue's independent injection tests keep it enabled.
    """
    sim = stim.FlipSimulator(
        batch_size=1, num_qubits=circuit.num_qubits, disable_stabilizer_randomization=True, seed=1
    )
    peak: dict[str, Any] = {"block_paulis": [], "tick": 0}
    tick = 0
    for op in inject_one(circuit, fault):
        sim.do(op)
        if op.name == "TICK":
            tick += 1
            frame = sim.peek_pauli_flips(instance_index=0)
            support = [f"{'IXYZ'[frame[q]]}{q}" for q in range(block_size) if frame[q]]
            if len(support) > len(peak["block_paulis"]):
                peak = {"block_paulis": support, "tick": tick}
    return {
        "fault": fault,
        "peak_at_tick_boundaries": peak,
        "caveat": "Physical Pauli-frame representative; weight is not reduced modulo stabilizers and is not a circuit-distance bound.",
    }


def propagation_examples(
    circuit: stim.Circuit, cat: dict[str, Any], meta: dict[str, Any]
) -> list[dict[str, Any]]:
    """Trace six distinct illustrative signatures with large detector footprints."""
    chosen = []
    seen = set()
    for fault in sorted(
        cat["faults"], key=lambda f: -len(cat["signatures"][f["signature"]]["detectors"])
    ):
        signature = fault["signature"]
        if signature in seen:
            continue
        seen.add(signature)
        example = trace_example(
            circuit, fault, meta["left_code_parameters"][0] + meta["right_code_parameters"][0]
        )
        example["signature"] = cat["signatures"][signature]
        chosen.append(example)
        if len(chosen) == 6:
            break
    return chosen


def archive(
    distance: int, rounds: int, basis: str
) -> tuple[stim.Circuit, stim.DetectorErrorModel, dict[str, Any], str]:
    if distance == 4 and rounds == 8:
        left, right, left_name, right_name = code_pair(
            "hgp-simplex-surface", simplex_m=3, surface_distance=4
        )
        e = build_experiment_for_codes(
            left_code=left,
            right_code=right,
            left_code_name=left_name,
            right_code_name=right_name,
            basis=Pauli[basis],
            preparation_rounds=4,
            surgery_rounds=8,
            physical_error_rate=0.001,
            bell_pair_infidelity=0,
            include_idling_error=False,
            noise_scope="intrinsic",
            adapter_noise_targets="adapter_hardware",
            reduce_detectors=True,
            compute_exact_distances=False,
            schedule_policy="ppm_basis_first",
        )
        return e.circuit, e.detector_error_model, e.metadata, "new build; no Monte Carlo sampling"
    if distance == 4:
        directory = ROOT / "ppm_basis_first_hgp_m3_d4_prep4_surg4_bell_fidelity"
    elif rounds == 4:
        directory = ROOT / "ppm_basis_first_hgp_m4_d8_prep4_surg4_10000" / basis
    else:
        directory = ROOT / "ppm_basis_first_hgp_m4_d8_prep4_surg8_10000/samples"
    entries = json.loads((directory / "manifest.json").read_text())
    (entry,) = [
        e for e in entries if e["logical_basis"] == basis and e["bell_pair_infidelity"] == 0
    ]
    return (
        stim.Circuit((directory / entry["circuit_path"]).read_text()),
        stim.DetectorErrorModel((directory / entry["dem_path"]).read_text()),
        entry,
        str(directory),
    )


def verify_config(
    circuit: stim.Circuit,
    dem: stim.DetectorErrorModel,
    meta: dict[str, Any],
    rounds: int,
    basis: str,
) -> None:
    expected = dict(
        physical_error_rate=0.001,
        bell_pair_infidelity=0,
        preparation_rounds=4,
        surgery_rounds=rounds,
        logical_basis=basis,
        noise_scope="intrinsic",
        adapter_noise_targets="adapter_hardware",
        detector_reduction="inactive_rows",
        include_idling_error=False,
        left_logical_index=0,
        right_logical_index=0,
        schedule_policy="ppm_basis_first_local_v1",
    )
    for key, value in expected.items():
        if meta[key] != value:
            raise ValueError(f"Unexpected archived parameter {key}: {meta[key]}")
    if meta["surgery_schedule"] != dict(
        local_order=["X", "Z"] if basis == "X" else ["Z", "X"],
        bell_order=["X", "Z"],
        local_layers_sha256=None,
    ):
        raise ValueError("Unexpected archived schedule")
    if circuit.detector_error_model() != dem or circuit.num_observables != 1:
        raise ValueError("Circuit/DEM/observable mismatch")


def prepare(output: Path, distance: int, rounds: int, basis: str) -> None:
    start = time.perf_counter()
    directory = output / f"d{distance}_r{rounds}_{basis}"
    directory.mkdir(exist_ok=True)
    circuit, dem, meta, source = archive(distance, rounds, basis)
    verify_config(circuit, dem, meta, rounds, basis)
    circuit = circuit.flattened()
    identity = {
        "distance_label": distance,
        "rounds": rounds,
        "basis": basis,
        "source": source,
        "circuit_sha256": sha(str(circuit)),
        "dem_sha256": sha(str(dem)),
        "metadata": meta,
        "runtime_identity": runtime_identity(),
        "audit_source_sha256": sha(Path(__file__).read_text()),
    }
    manifest = directory / "identity.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError(f"Audit identity changed: {directory}")
        with gzip.open(directory / "catalogue.json.gz", "rt") as stream:
            cat = json.load(stream)
        if sha(str(circuit)) != sha((directory / "circuit.stim").read_text()):
            raise ValueError("Saved audit circuit changed")
        if sha(str(dem)) != sha((directory / "model.dem").read_text()):
            raise ValueError("Saved audit DEM changed")
        saved_hash = json.loads((directory / "catalogue_hash.json").read_text())["sha256"]
        if hashlib.sha256((directory / "catalogue.json.gz").read_bytes()).hexdigest() != saved_hash:
            raise ValueError("Saved physical catalogue changed")
        return
    if any(directory.iterdir()):
        raise ValueError(f"Nonempty condition directory without completed catalogue: {directory}")
    cat = catalogue(circuit, meta)
    validate_injection(circuit, cat)
    cat["metrics"]["propagation_and_validation_wall_seconds"] = time.perf_counter() - start
    (directory / "circuit.stim").write_text(str(circuit))
    (directory / "model.dem").write_text(str(dem))
    with gzip.open(directory / "catalogue.json.gz", "wt") as stream:
        json.dump(cat, stream)
    save_json(
        directory / "catalogue_hash.json",
        {"sha256": hashlib.sha256((directory / "catalogue.json.gz").read_bytes()).hexdigest()},
    )
    save_json(directory / "propagation_examples.json", propagation_examples(circuit, cat, meta))
    save_json(directory / "propagation_summary.json", cat["metrics"])
    save_json(manifest, identity)
    print(f"Catalogued {directory.name}: {cat['metrics']}", flush=True)


def decoder_order(cat: dict[str, Any]) -> list[int]:
    """Deterministic stratified diagnostic order, not an LER sampling distribution."""
    by_syndrome: dict[tuple[int, ...], int] = {}
    for sid, signature in enumerate(cat["signatures"]):
        by_syndrome.setdefault(tuple(signature["detectors"]), sid)
    groups: list[list[int]] = [[], []]
    for detectors, sid in by_syndrome.items():
        if detectors:
            groups[bool(cat["signatures"][sid]["observables"])].append(sid)
    rng = random.Random(92814)
    for group in groups:
        rng.shuffle(group)
    return [0] + [s for pair in itertools.zip_longest(*groups) for s in pair if s is not None]


def decode_condition(directory: str, limit: int, seconds: float) -> dict[str, Any]:
    path = Path(directory)
    with gzip.open(path / "catalogue.json.gz", "rt") as stream:
        cat = json.load(stream)
    signatures = cat["signatures"]
    progress = path / "decoder_progress.jsonl"
    prior = (
        [json.loads(line) for line in progress.read_text().splitlines()]
        if progress.exists()
        else []
    )
    done = {row["signature"] for row in prior}
    order = decoder_order(cat)[:limit]
    if len(done) != len(prior) or any(sid not in decoder_order(cat) for sid in done):
        raise ValueError("Duplicate or foreign decoder checkpoint entry")
    if json.loads((path / "identity.json").read_text())["runtime_identity"] != runtime_identity():
        raise ValueError("Decoder/runtime changed since catalogue creation")
    if done.issuperset(order):
        return {"condition": path.name, "status": "requested_limit_already_reached"}
    compiled = SinterDecoder().compile_decoder_for_dem(
        stim.DetectorErrorModel((path / "model.dem").read_text())
    )
    settings = {
        k: getattr(compiled.decoder, k, None)
        for k in ("max_iter", "bp_method", "osd_order", "osd_method")
    }
    save_json(path / "decoder_settings.json", settings)
    start = time.perf_counter()
    added = 0
    with progress.open("a", buffering=1) as stream:
        for sid in order:
            if sid in done:
                continue
            if time.perf_counter() - start >= seconds:
                break
            syndrome = np.zeros(compiled.num_detectors, dtype=int)
            syndrome[signatures[sid]["detectors"]] = 1
            tick = time.perf_counter()
            prediction = compiled.decode(syndrome)
            row = dict(
                signature=sid,
                predicted_observables=np.flatnonzero(prediction).tolist(),
                decode_seconds=time.perf_counter() - tick,
                bp_converged=bool(getattr(compiled.decoder, "converge", False)),
                bp_iterations=int(getattr(compiled.decoder, "iter", -1)),
            )
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            done.add(sid)
            added += 1
            if added % 16 == 0:
                print(
                    f"{path.name}: {len(done)} syndromes checkpointed in {time.perf_counter() - start:.1f}s",
                    flush=True,
                )
    result = dict(
        condition=path.name,
        added=added,
        saved=len(done),
        seconds=time.perf_counter() - start,
        status="requested_limit_reached" if done.issuperset(order) else "time_budget_reached",
    )
    print(result, flush=True)
    return result


def summarize(directory: Path) -> dict[str, Any]:
    with gzip.open(directory / "catalogue.json.gz", "rt") as stream:
        cat = json.load(stream)
    path = directory / "decoder_progress.jsonl"
    results = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    predictions = {
        tuple(cat["signatures"][r["signature"]]["detectors"]): r["predicted_observables"]
        for r in results
    }
    failures, covered = [], []
    for fault in cat["faults"]:
        signature = cat["signatures"][fault["signature"]]
        prediction = predictions.get(tuple(signature["detectors"]))
        if prediction is None:
            continue
        covered.append(fault)
        if prediction != signature["observables"]:
            failures.append(
                {**fault, "actual_signature": signature, "predicted_observables": prediction}
            )
    zero_failure = predictions.get(()) not in (None, [])
    no_fault_p = cat["metrics"]["no_fault_probability"]
    grouped = collections.Counter((f["phase"], f["round"], f["source_gate"]) for f in failures)
    report = {
        "condition": directory.name,
        **cat["metrics"],
        "decoded_syndromes": len(predictions),
        "covered_physical_outcomes": len(covered),
        "misdecoded_physical_outcomes": len(failures),
        "zero_syndrome_misdecoded": zero_failure,
        "misdecoded_distinct_syndromes": len(
            {tuple(f["actual_signature"]["detectors"]) for f in failures}
        ),
        "decode_seconds": sum(r["decode_seconds"] for r in results),
        "bp_nonconverged_syndromes": sum(not r["bp_converged"] for r in results),
        "complete_decoder_coverage": len(predictions)
        == cat["metrics"]["distinct_syndromes_including_zero"],
        "known_single_fault_sector_failure_probability": sum(
            f["probability"] * no_fault_p / (1 - f["site_probability"]) for f in failures
        ),
        "known_single_fault_failure_marginal_sum_not_ler": sum(f["probability"] for f in failures),
        "failure_groups": [
            {"phase": p, "round": r, "source_gate": g, "outcomes": n}
            for (p, r, g), n in grouped.items()
        ],
        "caveat": "Decoder coverage may be partial and selection is diagnostic, not statistical. Sector probability requires exactly one physical faulty location and no faults elsewhere; it is not the full LER. No two-fault or distance search was performed.",
    }
    save_json(directory / "audit_summary.json", report)
    save_json(directory / "decoder_failures.json", failures)
    return report


def write_reports(output: Path, directories: list[Path]) -> None:
    reports = [summarize(p) for p in directories if (p / "identity.json").exists()]
    save_json(output / "summary.json", reports)
    fields = [k for k in reports[0] if k not in {"failure_groups", "caveat"}] if reports else []
    with (output / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reports)


def run(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".audit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = output / "audit_config.json"
        config = {
            "p": 0.001,
            "bell_source_fidelity": 1,
            "preparation_rounds": 4,
            "distances": [4, 8],
            "rounds": [4, 8],
            "bases": ["X", "Z"],
            "runtime_identity": runtime_identity(),
            "audit_source_sha256": sha(Path(__file__).read_text()),
        }
        if config_path.exists():
            if json.loads(config_path.read_text()) != config:
                raise ValueError("Audit identity changed; choose a new directory")
        else:
            if any(p.name != ".audit.lock" for p in output.iterdir()):
                raise ValueError("Nonempty output directory without an audit configuration")
            save_json(config_path, config)
        directories = []
        for distance in (4, 8):
            for rounds in (4, 8):
                for basis in "XZ":
                    prepare(output, distance, rounds, basis)
                    directories.append(output / f"d{distance}_r{rounds}_{basis}")
        write_reports(output, directories)
        event = {
            "started_unix_time": time.time(),
            "workers": args.workers,
            "decode_limit": args.decode_limit,
            "seconds_per_condition": args.seconds_per_condition,
        }

        def log(status: str) -> None:
            with (output / "invocations.jsonl").open("a") as stream:
                stream.write(json.dumps({**event, "status": status}) + "\n")

        log("started")
        status = "failed"
        try:
            if args.decode_limit:
                context = multiprocessing.get_context("spawn")
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=args.workers, mp_context=context
                ) as pool:
                    futures = [
                        pool.submit(
                            decode_condition, str(p), args.decode_limit, args.seconds_per_condition
                        )
                        for p in directories
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        future.result()
            status = "completed_requested_stage"
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        finally:
            write_reports(output, directories)
            event["finished_unix_time"] = time.time()
            log(status)
        print(f"Audit saved to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--decode-limit",
        type=int,
        default=128,
        help="Cumulative distinct syndromes per condition; zero builds catalogues only",
    )
    parser.add_argument(
        "--seconds-per-condition",
        type=float,
        default=120,
        help="Wall-time budget per decoder invocation; one in-flight decode may exceed it",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.decode_limit < 0 or args.seconds_per_condition <= 0 or args.workers < 1:
        parser.error("Invalid decoder limit, time budget, or worker count")
    run(args)
