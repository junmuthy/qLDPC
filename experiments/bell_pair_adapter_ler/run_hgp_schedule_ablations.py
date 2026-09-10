"""One-factor hardware-only HGP/surface schedule comparisons against legacy XX/ZZ.

All variants start from their own legacy basis, not from a preceding variant.
Only merged extraction changes; ideal preparation and terminal boundaries stay
at their legacy settings. Existing results are read-only references.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import subprocess
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import sinter
import stim
from scipy.stats import fisher_exact

from qldpc import codes
from qldpc.circuits import QubitIDs
from qldpc.circuits.detector_reduction import remove_inactive_detectors
from qldpc.circuits.surgery import (
    CSSXZDuality,
    JointPPMSchedule,
    JointPPMSelectiveBellPairSyndrome,
    build_bridge,
    build_gadget,
)
from qldpc.circuits.surgery.circuit import _stitch_intercode
from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli

from .run_experiment import (
    BuiltExperiment,
    _wilson_interval,
    build_experiment_for_codes,
    sample_experiments,
    write_artifacts,
    write_summary,
)
from .run_hgp_dual_schedule import DEFAULT_OUTPUT as DUAL_REFERENCE
from .run_hgp_dual_schedule import REFERENCE, _sha256, _transpose_permutation

DEFAULT_OUTPUT = (
    REFERENCE.parent / "hgp_simplex3_surface_unrotated_d4_hardware_schedule_ablations_20k"
)
POLICIES = ("bell_order", "local_order", "local_layers")
LABELS = {
    "legacy": "Legacy",
    "bell_order": "A: Bell order",
    "local_order": "B: Local order",
    "local_layers": "C: Local layers",
    "exact_dual": "Exact dual\n(combined)",
}


@dataclasses.dataclass
class Layout:
    code: CSSCode
    ids: QubitIDs
    strategy: JointPPMSelectiveBellPairSyndrome

    @property
    def bell_checks(self) -> tuple[int, ...]:
        return tuple(self.ids.check[i] for i in self.strategy.selected_check_indices(self.code))


def _layout(left: CSSCode, right: CSSCode, basis: Pauli) -> Layout:
    assert basis in (Pauli.X, Pauli.Z)
    gl = build_gadget(left, left.get_logical_ops(basis)[0], basis=basis)
    gr = build_gadget(right, right.get_logical_ops(basis)[0], basis=basis)
    bridge = build_bridge(gl, gr)
    code = _stitch_intercode(gl, gr, bridge)
    strategy = JointPPMSelectiveBellPairSyndrome(gl, gr, bridge, independent_blocks=True)
    ids = QubitIDs.from_code(code, num_ancillas=len(strategy.selected_check_indices(code)))
    return Layout(code, ids, strategy)


def _circuit_inventory(circuit: stim.Circuit) -> Counter[tuple[Any, ...]]:
    """Per-operation physical gate/noise inventory, ignoring time and annotations."""
    result: Counter[tuple[Any, ...]] = Counter()
    for op in circuit.flattened():
        if op.name in {"TICK", "SHIFT_COORDS", "QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"}:
            continue
        targets = tuple(t.value for t in op.targets_copy())
        width = 2 if op.name in {"CX", "CZ", "DEPOLARIZE2"} else 1
        for index in range(0, len(targets), width):
            result[(op.name, tuple(op.gate_args_copy()), targets[index : index + width])] += 1
    return result


def _semantic_records(circuit: stim.Circuit) -> tuple[list[Any], list[Any]]:
    """Express detector/observable supports by physical qubit and measurement occurrence."""
    events = []
    occurrences: Counter[int] = Counter()
    detectors: list[Any] = []
    observables: list[Any] = []
    for op in circuit.flattened():
        if op.name in {"M", "MX"}:
            for target in op.targets_copy():
                occurrences[target.value] += 1
                events.append((target.value, occurrences[target.value], op.name))
        elif op.name in {"DETECTOR", "OBSERVABLE_INCLUDE"}:
            support: set[Any] = set()
            for target in op.targets_copy():
                support.symmetric_difference_update({events[len(events) + target.value]})
            row = (tuple(op.gate_args_copy()), tuple(sorted(support)))
            (detectors if op.name == "DETECTOR" else observables).append(row)
    return detectors, observables


def _ideal_prefix(circuit: stim.Circuit) -> stim.Circuit:
    prefix = stim.Circuit()
    for op in circuit.flattened():
        # Stim also classifies ideal M/MX as noisy-capable gates. Only stop at
        # an actual error channel or a measurement with nonzero error probability.
        if (stim.gate_data(op.name).is_noisy_gate and op.name not in {"M", "MX"}) or (
            op.name in {"M", "MX"} and any(op.gate_args_copy())
        ):
            break
        prefix.append(op)
    return prefix


def _bell_blocks(circuit: stim.Circuit) -> list[tuple[int, str]]:
    """One-round Bell blocks in extraction order, including reset and measurement."""
    blocks: list[tuple[int, str]] = []
    current: stim.Circuit | None = None
    owner = -1
    for op in circuit:
        if op.name == "R":
            if current is not None:
                blocks.append((owner, str(current)))
            current = stim.Circuit()
            owner = op.targets_copy()[0].value
        if current is not None:
            current.append(op)
    if current is not None:
        blocks.append((owner, str(current)))
    return blocks


def _local_prefix(circuit: stim.Circuit) -> stim.Circuit:
    result = stim.Circuit()
    for op in circuit:
        if op.name == "R":
            break
        result.append(op)
    return result


def audit_variant(
    legacy: BuiltExperiment,
    variant: BuiltExperiment,
    layout: Layout,
    schedule: JointPPMSchedule,
    policy: str,
) -> dict[str, Any]:
    """Fail closed if a purported one-factor experiment changes unrelated content."""
    if _circuit_inventory(legacy.circuit) != _circuit_inventory(variant.circuit):
        raise ValueError("Variant changed the physical gate/noise inventory")
    if _ideal_prefix(legacy.circuit) != _ideal_prefix(variant.circuit):
        raise ValueError("Variant changed ideal preparation or adapter initialization")
    if _semantic_records(legacy.circuit) != _semantic_records(variant.circuit):
        raise ValueError("Variant changed canonical detector identities or PPM observable")
    ideal = variant.circuit.without_noise()
    if ideal.detector_error_model().num_errors:
        raise ValueError("Variant has noisy terms with noise disabled")
    dets, obs = ideal.compile_detector_sampler(seed=310).sample(64, separate_observables=True)
    if np.any(dets) or np.any(obs):
        raise ValueError("Variant failed noiseless detector/observable sampling")
    old_round, _ = layout.strategy.get_circuit(layout.code, layout.ids)
    strategy = JointPPMSelectiveBellPairSyndrome(
        layout.strategy.g_l,
        layout.strategy.g_r,
        layout.strategy.bridge,
        independent_blocks=True,
        schedule=schedule,
    )
    new_round, _ = strategy.get_circuit(layout.code, layout.ids)
    old_bell, new_bell = _bell_blocks(old_round), _bell_blocks(new_round)
    if dict(old_bell) != dict(new_bell):
        raise ValueError("Variant changed within-check Bell gates or partner ownership")
    if policy == "bell_order":
        if _local_prefix(old_round) != _local_prefix(new_round):
            raise ValueError("Bell-order variant changed local extraction")
        expected_order = [q for q, _ in old_bell if q in layout.ids.checks_z] + [
            q for q, _ in old_bell if q in layout.ids.checks_x
        ]
        if [q for q, _ in new_bell] != expected_order:
            raise ValueError("Bell-order variant changed within-family check order")
    elif old_bell != new_bell:
        raise ValueError("Local variant changed Bell extraction order")
    if policy in {"bell_order", "local_order"} and old_round.num_ticks != new_round.num_ticks:
        raise ValueError("Order-only variant changed round depth")
    return {
        "policy": policy,
        "physical_gate_and_noise_inventory_identical": True,
        "ideal_preparation_and_initialization_identical": True,
        "canonical_detector_and_observable_identities_identical": True,
        "bell_partner_and_within_check_gates_identical": True,
        "noiseless_validation_passed": True,
        "legacy_ideal_extraction_ticks": old_round.num_ticks,
        "variant_ideal_extraction_ticks": new_round.num_ticks,
        "legacy_total_ticks": legacy.circuit.num_ticks,
        "variant_total_ticks": variant.circuit.num_ticks,
        "legacy_bell_order": [q for q, _ in old_bell],
        "variant_bell_order": [q for q, _ in new_bell],
        "ideal_prefix_sha256": _sha256(str(_ideal_prefix(variant.circuit))),
        "full_circuit_sha256": _sha256(str(variant.circuit)),
    }


def build_suite() -> tuple[list[BuiltExperiment], dict[str, Any]]:
    left, right = codes.HGPCode(codes.SimplexCode(3)), codes.SurfaceCode(4, rotated=False)
    layouts = {p: _layout(left, right, p) for p in (Pauli.X, Pauli.Z)}
    common: dict[str, Any] = dict(
        left_code=left,
        right_code=right,
        left_code_name="hgp_simplex3_98_18_4",
        right_code_name="surface_unrotated_d4",
        preparation_rounds=4,
        surgery_rounds=4,
        physical_error_rate=0.001,
        bell_pair_infidelity=0.0,
        include_idling_error=False,
        noise_scope="intrinsic",
        compute_exact_distances=False,
        adapter_noise_targets="adapter_hardware",
        reduce_detectors=False,
    )
    baselines = {p: build_experiment_for_codes(basis=p, **common) for p in layouts}
    references: list[tuple[str, stim.DetectorErrorModel]] = []
    for directory in (REFERENCE, DUAL_REFERENCE):
        for entry in json.loads((directory / "manifest.json").read_text()):
            model = stim.DetectorErrorModel.from_file(directory / entry["dem_path"])
            references.append((f"{directory.name}/{entry['name']}", model))
            if directory == REFERENCE:
                basis = Pauli[entry["logical_basis"]]
                reduced = remove_inactive_detectors(baselines[basis].circuit)
                old = stim.Circuit.from_file(directory / entry["circuit_path"])
                if reduced.circuit != old or reduced.detector_error_model != model:
                    raise ValueError(
                        "Current default no longer exactly reproduces archived baseline"
                    )
    experiments, audits = [], []
    for basis in (Pauli.Z, Pauli.X):
        target = layouts[basis]
        source = layouts[Pauli.X if basis == Pauli.Z else Pauli.Z]
        mapping = CSSXZDuality.from_codes(
            source.code,
            target.code,
            source.ids,
            target.ids,
            data_permutation=_transpose_permutation(len(source.code)),
            source_bell_checks=source.bell_checks,
            target_bell_checks=target.bell_checks,
        )
        source_layers = source.strategy.local_interaction_circuits(source.code, source.ids)
        # Dual source Z -> target X and source X -> target Z; keep TARGET X/Z family order.
        dual_layers = (mapping.transform(source_layers[1]), mapping.transform(source_layers[0]))
        schedules = (
            JointPPMSchedule(bell_order=(Pauli.Z, Pauli.X)),
            JointPPMSchedule(local_order=(Pauli.Z, Pauli.X)),
            JointPPMSchedule(local_layers=dual_layers),
        )
        for policy, schedule in zip(POLICIES, schedules):
            full = build_experiment_for_codes(basis=basis, surgery_schedule=schedule, **common)
            audit = audit_variant(baselines[basis], full, target, schedule, policy)
            reduction = remove_inactive_detectors(full.circuit)
            name = baselines[basis].name + f"_ablation_{policy}_v1"
            metadata = dict(full.metadata)
            metadata.update(
                name=name,
                schedule_policy=policy,
                schedule_version="single_factor_v1",
                detector_reduction="inactive_rows",
                detector_order="legacy_check_epoch",
                retained_detector_ids=list(reduction.retained_detector_ids),
                num_detectors=reduction.circuit.num_detectors,
                num_dem_errors=reduction.detector_error_model.num_errors,
                dem_analysis="undecomposed_exact",
            )
            # Match artifact order recursively: nested dict order enters Sinter identity too.
            metadata = json.loads(json.dumps(metadata, sort_keys=True))
            experiment = BuiltExperiment(
                name, reduction.circuit, reduction.detector_error_model, metadata
            )
            matches = [label for label, dem in references if dem == experiment.detector_error_model]
            audit.update(
                name=name,
                logical_basis=basis.name,
                num_detectors=experiment.circuit.num_detectors,
                num_dem_errors=experiment.detector_error_model.num_errors,
                ordered_dem_matches=matches,
                strong_id=sinter.Task(
                    circuit=experiment.circuit,
                    detector_error_model=experiment.detector_error_model,
                    json_metadata=metadata,
                    decoder="qldpc_bp_osd",
                ).strong_id(),
            )
            experiments.append(experiment)
            audits.append(audit)
            references.append((name, experiment.detector_error_model))
    return experiments, {
        "baselines_match_archives_exactly": True,
        "reference_summary_sha256": _sha256((REFERENCE / "summary.csv").read_bytes()),
        "dual_reference_summary_sha256": _sha256((DUAL_REFERENCE / "summary.csv").read_bytes()),
        "experiments": audits,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm familywise correction, valid with a shared baseline across comparisons."""
    result = [0.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(sorted(range(len(pvalues)), key=pvalues.__getitem__)):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        result[i] = running
    return result


def write_comparison(output: Path) -> list[dict[str, Any]]:
    manifest = {e["name"]: e for e in json.loads((output / "manifest.json").read_text())}
    audits = {
        e["name"]: e
        for e in json.loads((output / "schedule_audit.json").read_text())["experiments"]
    }
    rows = []
    directories = [(REFERENCE, "legacy"), (DUAL_REFERENCE, "exact_dual")]
    directories += [(output / b / p, p) for b in ("Z", "X") for p in POLICIES]
    for directory, policy in directories:
        if not (directory / "summary.csv").exists():
            continue
        entries = (
            manifest
            if policy in POLICIES
            else {e["name"]: e for e in json.loads((directory / "manifest.json").read_text())}
        )
        for item in _read_csv(directory / "summary.csv"):
            entry = entries[item["name"]]
            if not (
                entry["left_code"] == "hgp_simplex3_98_18_4"
                and entry["right_code"] == "surface_unrotated_d4"
                and entry["left_logical_index"] == entry["right_logical_index"] == 0
                and entry["preparation_rounds"] == entry["surgery_rounds"] == 4
                and entry["physical_error_rate"] == 0.001
                and entry["bell_pair_fidelity"] == 1
                and not entry["include_idling_error"]
                and entry["noise_scope"] == "intrinsic"
                and entry["adapter_noise_targets"] == "adapter_hardware"
                and entry["detector_reduction"] == "inactive_rows"
                and item["decoder"] == "qldpc_bp_osd"
            ):
                raise ValueError("Mismatched comparison settings")
            shots, errors, discards = (int(item[k]) for k in ("shots", "errors", "discards"))
            if shots <= discards:
                raise ValueError("Comparison requires accepted shots")
            low, high = _wilson_interval(errors, shots - discards)
            audit = audits.get(item["name"], {})
            if policy == "legacy":
                ticks = 332 if item["logical_basis"] == "X" else 328
            elif policy == "exact_dual":
                ticks = 332
            else:
                ticks = audit["variant_total_ticks"]
            rows.append(
                dict(
                    name=item["name"],
                    logical_basis=item["logical_basis"],
                    schedule_policy=policy,
                    shots=shots,
                    discards=discards,
                    errors=errors,
                    logical_error_rate=errors / (shots - discards),
                    ler_95ci_low=low,
                    ler_95ci_high=high,
                    worker_seconds=float(item["seconds"]),
                    worker_seconds_per_10000_shots=10000 * float(item["seconds"]) / shots,
                    total_ticks=ticks,
                    num_detectors=entry["num_detectors"],
                    num_dem_errors=entry["num_dem_errors"],
                    source_summary=str(directory / "summary.csv"),
                )
            )
    _write_csv(output / "comparison.csv", rows)
    comparisons = []
    for row in rows:
        if row["schedule_policy"] not in POLICIES:
            continue
        baseline = next(
            r
            for r in rows
            if r["schedule_policy"] == "legacy" and r["logical_basis"] == row["logical_basis"]
        )
        n, n0 = row["shots"] - row["discards"], baseline["shots"] - baseline["discards"]
        p, p0 = row["logical_error_rate"], baseline["logical_error_rate"]
        # Newcombe interval from Wilson limits for the difference of independent proportions.
        low = (
            p - p0 - np.sqrt((p - row["ler_95ci_low"]) ** 2 + (baseline["ler_95ci_high"] - p0) ** 2)
        )
        high = (
            p - p0 + np.sqrt((row["ler_95ci_high"] - p) ** 2 + (p0 - baseline["ler_95ci_low"]) ** 2)
        )
        comparisons.append(
            dict(
                logical_basis=row["logical_basis"],
                schedule_policy=row["schedule_policy"],
                ler_difference=p - p0,
                difference_95ci_low=float(low),
                difference_95ci_high=float(high),
                fisher_p=float(
                    fisher_exact(
                        [
                            [row["errors"], n - row["errors"]],
                            [baseline["errors"], n0 - baseline["errors"]],
                        ]
                    ).pvalue
                ),
            )
        )
    if comparisons:
        # Always reserve the six planned comparisons, including unfinished tasks.
        adjusted = holm_adjust(
            [r["fisher_p"] for r in comparisons] + [1.0] * (6 - len(comparisons))
        )
        for row, pvalue in zip(comparisons, adjusted):
            row["fisher_p_holm_6"] = pvalue
        _write_csv(output / "differences.csv", comparisons)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.7), sharey=True)
    for axis, basis in zip(axes, ("X", "Z")):
        policies = ["legacy", *POLICIES] + (["exact_dual"] if basis == "Z" else [])
        for index, policy in enumerate(policies):
            selected = [
                r for r in rows if r["logical_basis"] == basis and r["schedule_policy"] == policy
            ]
            if not selected:
                continue
            row = selected[0]
            y, low, high = [
                100 * row[k] for k in ("logical_error_rate", "ler_95ci_low", "ler_95ci_high")
            ]
            axis.errorbar(
                index,
                y,
                yerr=np.maximum([[y - low], [high - y]], 0),
                fmt="o",
                color="0.5" if policy == "exact_dual" else f"C{index}",
                capsize=5,
            )
            axis.annotate(
                f"{row['errors']}/{row['shots']:,}",
                (index, high),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        axis.set_xticks(range(len(policies)), [LABELS[p].replace(": ", ":\n") for p in policies])
        axis.set_title(f"{basis}{basis} PPM")
        axis.grid(axis="y", alpha=0.25)
        axis.set_xlim(-0.45, len(policies) - 0.55)
    axes[0].set_ylabel("Adapter LER (%) — 95% Wilson intervals")
    axes[0].set_ylim(0, max(r["ler_95ci_high"] for r in rows) * 135)
    figure.suptitle(
        "HGP simplex d=4 – surface d=4: one-factor hardware schedule tests\n"
        "4 ideal preparation + 4 adapter rounds; p=0.001, Bell F=1"
    )
    figure.text(
        0.5,
        0.02,
        "Each A/B/C variant changes only its named control from legacy. "
        "C replaces local coupling/layer assignments.\nExact dual is a combined-control reference, "
        "not a one-factor test. Error bars are pointwise; see differences.csv for Holm-adjusted tests.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.1, 1, 0.94))
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"hgp_schedule_ablation_comparison.{suffix}", dpi=180)
    plt.close(figure)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-shots", type=int, default=20000)
    parser.add_argument("--pilot-shots", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if min(args.max_shots, args.pilot_shots, args.num_workers) < 1:
        parser.error("Shot and worker counts must be positive")
    if args.analyze_only:
        write_comparison(args.output_dir)
        return
    started = time.perf_counter()
    experiments, audit = build_suite()
    write_artifacts(experiments, args.output_dir)
    config = dict(
        max_shots=args.max_shots,
        pilot_shots=args.pilot_shots,
        num_workers=args.num_workers,
        preparation_rounds=4,
        surgery_rounds=4,
        physical_error_rate=0.001,
        bell_pair_fidelity=1,
        include_idling_error=False,
        noise_scope="intrinsic",
        adapter_noise_targets="adapter_hardware",
        detector_reduction="inactive_rows",
        dem_analysis="undecomposed_exact",
        compute_exact_distances=False,
        policies=list(POLICIES),
        baseline_directory=str(REFERENCE),
        dual_reference_directory=str(DUAL_REFERENCE),
        versions={n: version(n) for n in ("stim", "sinter", "ldpc", "numpy", "scipy")},
        threads={n: os.environ.get(n) for n in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    for name, data in (("schedule_audit.json", audit), ("run_config.json", config)):
        (args.output_dir / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    for entry in audit["experiments"]:
        print(
            json.dumps(
                {
                    k: entry[k]
                    for k in (
                        "logical_basis",
                        "policy",
                        "variant_ideal_extraction_ticks",
                        "num_detectors",
                        "num_dem_errors",
                        "ordered_dem_matches",
                    )
                }
            ),
            flush=True,
        )
    if args.build_only:
        return
    # No task is silently resampled if exact reuse becomes possible: explicit accounting needed.
    if any(e["ordered_dem_matches"] for e in audit["experiments"]):
        raise ValueError("Exact ordered DEM reuse detected; resolve provenance before sampling")
    for experiment, entry in zip(experiments, audit["experiments"]):
        directory = (
            args.output_dir
            / experiment.metadata["logical_basis"]
            / experiment.metadata["schedule_policy"]
        )
        write_artifacts([experiment], directory)
        for cap in sorted({min(args.pilot_shots, args.max_shots), args.max_shots}):
            stage_started = time.perf_counter()
            print(
                f"Collect/resume {experiment.metadata['logical_basis']} {entry['policy']} to {cap:,} shots",
                flush=True,
            )
            stats = sample_experiments(
                [experiment],
                output_dir=directory,
                num_workers=args.num_workers,
                max_shots=cap,
                max_errors=cap,
                print_progress=False,
            )
            if len(stats) != 1 or stats[0].strong_id != entry["strong_id"]:
                raise ValueError("Sampling returned an unexpected task identity")
            write_summary(stats, directory)
            write_comparison(args.output_dir)
            record = dict(
                name=experiment.name,
                strong_id=stats[0].strong_id,
                stage_cap=cap,
                shots=stats[0].shots,
                errors=stats[0].errors,
                stage_wall_seconds=time.perf_counter() - stage_started,
                invocation_wall_seconds=time.perf_counter() - started,
            )
            with (args.output_dir / "invocations.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
