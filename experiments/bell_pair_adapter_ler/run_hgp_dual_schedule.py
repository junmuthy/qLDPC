"""Hardware-only ZZ using the exact X/Z dual of the archived XX schedule.

No full-surgery experiments or exact code-distance searches are constructed.
The default run checkpoints a 1,000-shot pilot, then resumes the same task to
20,000 shots. Archived XX/ZZ hardware estimates are references, not new shots.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import sinter
import stim

from qldpc import codes
from qldpc.circuits import QubitIDs
from qldpc.circuits.detector_reduction import remove_inactive_detectors
from qldpc.circuits.surgery import (
    AdapterHardware,
    CSSXZDuality,
    JointPPMSelectiveBellPairSyndrome,
    build_bridge,
    build_gadget,
)
from qldpc.circuits.surgery.circuit import _stitch_intercode
from qldpc.objects import Pauli

from .run_experiment import (
    BuiltExperiment,
    _wilson_interval,
    build_experiment_for_codes,
    sample_experiments,
    write_artifacts,
    write_summary,
)

OUTPUTS = Path(__file__).resolve().parent / "outputs"
REFERENCE = OUTPUTS / "hgp_simplex3_surface_unrotated_d4_hardware_prep4_surg4_bell0_20k"
DEFAULT_OUTPUT = (
    OUTPUTS / "hgp_simplex3_surface_unrotated_d4_hardware_xz_dual_prep4_surg4_bell0_20k"
)
SCHEDULE = "xz_dual_from_x_v1"


def _sha256(data: str | bytes) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def _transpose_permutation(num_data: int) -> tuple[int, ...]:
    """Explicit HGP/surface isomorphism, validated against every merged check below."""
    result = list(range(num_data))
    # Two 7x7 HGP sectors, then 4x4 vertex and 3x3 face surface-data sectors.
    # Gadget/bridge columns retain their order for these selected logicals.
    for offset, side in ((0, 7), (49, 7), (98, 4), (114, 3)):
        for index in range(side * side):
            result[offset + index] = offset + index % side * side + index // side
    return tuple(result)


def _validate_ppm_readout(
    circuit: stim.Circuit, check_ids: tuple[int, ...], partners: dict[int, int]
) -> None:
    """Ensure observable zero uses the final intended check/Bell measurements, not data readout."""
    events: list[int] = []
    observable_events: set[int] = set()
    for instruction in circuit.flattened():
        if instruction.name in {"M", "MX", "MR", "MRX"}:
            events.extend(t.value for t in instruction.targets_copy())
        elif instruction.name == "OBSERVABLE_INCLUDE":
            if instruction.gate_args_copy() != [0]:
                raise ValueError("Unexpected observable in dual PPM experiment")
            for target in instruction.targets_copy():
                if not target.is_measurement_record_target:
                    raise ValueError("Expected measurement-record observable")
                index = len(events) + target.value
                observable_events.symmetric_difference_update({index})
    last_event = {qubit: index for index, qubit in enumerate(events)}
    required = set(check_ids) | {partners[q] for q in check_ids if q in partners}
    if observable_events != {last_event[q] for q in required}:
        raise ValueError("Dual observable does not measure the intended final ZZ PPM checks")


def build_dual_experiment() -> tuple[BuiltExperiment, dict[str, Any]]:
    """Transport the complete noisy XX circuit, including ideal boundaries and records."""
    left = codes.HGPCode(codes.SimplexCode(3))
    right = codes.SurfaceCode(4, rotated=False)
    source = build_experiment_for_codes(
        left_code=left,
        right_code=right,
        left_code_name="hgp_simplex3_98_18_4",
        right_code_name="surface_unrotated_d4",
        left_logical_index=0,
        right_logical_index=0,
        basis=Pauli.X,
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
    layouts = []
    for basis in (Pauli.X, Pauli.Z):
        gl = build_gadget(left, left.get_logical_ops(basis)[0], basis=basis)
        gr = build_gadget(right, right.get_logical_ops(basis)[0], basis=basis)
        bridge = build_bridge(gl, gr)
        joint = _stitch_intercode(gl, gr, bridge)
        strategy = JointPPMSelectiveBellPairSyndrome(gl, gr, bridge, independent_blocks=True)
        selected = strategy.selected_check_indices(joint)
        ids = QubitIDs.from_code(joint, num_ancillas=len(selected))
        hardware = AdapterHardware.from_code(
            joint, ids, num_block_data_qubits=len(left) + len(right)
        )
        layouts.append((joint, ids, tuple(ids.check[i] for i in selected), hardware, gl, gr))
    sx, ix, bx, hx, glx, grx = layouts[0]
    sz, iz, bz, hz, glz, grz = layouts[1]
    permutation = _transpose_permutation(len(sx))
    source_support = set(glx.support) | {len(left) + i for i in grx.support}
    target_support = set(glz.support) | {len(left) + i for i in grz.support}
    if {permutation[i] for i in source_support} != target_support:
        raise ValueError("Data mapping does not carry the chosen XX logical support to ZZ")
    # Preserve the physical-block partition, not just the abstract stabilizer group.
    for first, last in (
        (0, len(left)),
        (len(left), len(left) + len(right)),
        (len(left) + len(right), len(sx)),
    ):
        if set(permutation[first:last]) != set(range(first, last)):
            raise ValueError("Duality crosses physical block/adapter boundaries")
    mapping = CSSXZDuality.from_codes(
        sx,
        sz,
        ix,
        iz,
        data_permutation=permutation,
        source_bell_checks=bx,
        target_bell_checks=bz,
    )
    for source_group, target_group in zip(
        (hx.data, hx.check, hx.ancilla), (hz.data, hz.check, hz.ancilla)
    ):
        if {mapping.qubit_map[q] for q in source_group} != set(target_group):
            raise ValueError("Duality changes the adapter-only noise selection")
    transformed = mapping.transform(source.circuit)
    ppm_start = left.num_checks_z + right.num_checks_z
    ppm_ids = iz.checks_z[ppm_start : ppm_start + len(glz.support) + len(grz.support)]
    _validate_ppm_readout(transformed, ppm_ids, dict(zip(bz, iz.ancilla)))
    # DEM construction rejects nondeterministic detectors/observables. Zero noise
    # must produce no stochastic detector/logical error terms.
    ideal_dem = transformed.without_noise().detector_error_model()
    if ideal_dem.num_errors:
        raise ValueError("Dual circuit has nontrivial errors without noise")
    original = remove_inactive_detectors(source.circuit)
    dual = remove_inactive_detectors(transformed)
    if original.retained_detector_ids != dual.retained_detector_ids:
        raise ValueError("Duality changed which detector events can fire")
    if original.detector_error_model != dual.detector_error_model:
        raise ValueError("Dual DEM is not exactly the reference XX DEM, including order and priors")

    entries = json.loads((REFERENCE / "manifest.json").read_text())
    reference_entry = next(e for e in entries if e["logical_basis"] == "X")
    reference_circuit = stim.Circuit.from_file(REFERENCE / reference_entry["circuit_path"])
    reference_dem = stim.DetectorErrorModel.from_file(REFERENCE / reference_entry["dem_path"])
    if (
        str(original.circuit) != str(reference_circuit)
        or original.detector_error_model != reference_dem
    ):
        raise ValueError(
            "Current reference XX construction no longer matches the archived hardware run"
        )
    name = reference_entry["name"].replace("_joint_x_", "_joint_z_") + f"_{SCHEDULE}"
    metadata = dict(source.metadata)
    metadata.update(
        name=name,
        logical_basis="Z",
        schedule_policy=SCHEDULE,
        schedule_reference_basis="X",
        schedule_reference_name=reference_entry["name"],
        schedule_reference_circuit_sha256=_sha256(str(reference_circuit)),
        schedule_mapping_sha256=_sha256(json.dumps(sorted(mapping.qubit_map.items()))),
        detector_reduction="inactive_rows",
        detector_order="transported_reference",
        adapter_hardware_qubits=list(hz.all_qubits),
        num_detectors=dual.circuit.num_detectors,
        num_dem_errors=dual.detector_error_model.num_errors,
        num_full_detectors=transformed.num_detectors,
        num_full_dem_errors=transformed.detector_error_model().num_errors,
        retained_detector_ids=list(dual.retained_detector_ids),
    )
    # Canonical order is part of new task identity; manifest reloads cannot cause
    # the insertion-order mismatch that affected an earlier legacy restart.
    metadata = dict(sorted(metadata.items()))
    experiment = BuiltExperiment(name, dual.circuit, dual.detector_error_model, metadata)
    task = sinter.Task(
        circuit=experiment.circuit,
        detector_error_model=experiment.detector_error_model,
        decoder="qldpc_bp_osd",
        json_metadata=metadata,
    )
    audit = {
        "schedule_policy": SCHEDULE,
        "strong_id": task.strong_id(),
        "source_matches_archived_hardware_xx": True,
        "logical_support_and_all_check_rows_mapped": True,
        "physical_partitions_and_hardware_mask_preserved": True,
        "target_zz_ppm_readout_validated": True,
        "identical_ordered_dem_including_priors": True,
        "identical_inactive_detector_selection": True,
        "source_data_columns_to_target": list(permutation),
        "source_qubits_to_target": {str(k): v for k, v in sorted(mapping.qubit_map.items())},
        "source_num_ticks": original.circuit.num_ticks,
        "dual_num_ticks": dual.circuit.num_ticks,
        "num_measurements": dual.circuit.num_measurements,
        "num_detectors": dual.circuit.num_detectors,
        "num_dem_errors": dual.detector_error_model.num_errors,
        "reference_summary_sha256": _sha256((REFERENCE / "summary.csv").read_bytes()),
    }
    return experiment, audit


def write_comparison(output_dir: Path) -> list[dict[str, Any]]:
    """Compare three estimates, with each archived reference counted only once."""
    rows = []
    for directory, policy in ((REFERENCE, "legacy"), (output_dir, SCHEDULE)):
        with (directory / "summary.csv").open(newline="") as source:
            for item in csv.DictReader(source):
                if not (
                    item["adapter_noise_targets"] == "adapter_hardware"
                    and item["noise_scope"] == "intrinsic"
                    and int(item["preparation_rounds"]) == int(item["surgery_rounds"]) == 4
                    and float(item["physical_error_rate"]) == 0.001
                    and float(item["bell_pair_fidelity"]) == 1
                    and item["decoder"] == "qldpc_bp_osd"
                    and item["name"].startswith("hgp_simplex3_98_18_4_surface_unrotated_d4_joint_")
                    and (policy == "legacy" or item["name"].endswith(SCHEDULE))
                ):
                    raise ValueError("Mismatched hardware schedule comparison settings")
                shots, discards, errors = (int(item[k]) for k in ("shots", "discards", "errors"))
                if shots <= discards:
                    raise ValueError("Comparison requires accepted shots")
                interval_low, interval_high = _wilson_interval(errors, shots - discards)
                rows.append(
                    dict(
                        logical_basis=item["logical_basis"],
                        schedule_policy=policy,
                        shots=shots,
                        errors=errors,
                        discards=discards,
                        logical_error_rate=errors / (shots - discards),
                        ler_95ci_low=interval_low,
                        ler_95ci_high=interval_high,
                        worker_seconds=float(item["seconds"]),
                        worker_seconds_per_10000_shots=10000 * float(item["seconds"]) / shots,
                        source_summary=str(directory / "summary.csv"),
                    )
                )
    if len(rows) != 3 or {(r["logical_basis"], r["schedule_policy"]) for r in rows} != {
        ("X", "legacy"),
        ("Z", "legacy"),
        ("Z", SCHEDULE),
    }:
        raise ValueError("Need archived XX/ZZ and one new dual-schedule ZZ estimate")
    rows.sort(key=lambda r: (r["schedule_policy"] != "legacy", r["logical_basis"]))
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as dest:
        writer = csv.DictWriter(dest, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axis = plt.subplots(figsize=(8.5, 5.4))
    y, low, high = [
        np.array([r[key] for r in rows]) * 100
        for key in ("logical_error_rate", "ler_95ci_low", "ler_95ci_high")
    ]
    axis.errorbar(range(3), y, yerr=np.maximum([y - low, high - y], 0), fmt="o", capsize=5)
    axis.set_xticks(
        range(3),
        [
            "XX: existing\nreference schedule",
            "ZZ: existing\nlegacy schedule",
            "ZZ: new\nexact dual of XX",
        ],
    )
    axis.set_xlim(-0.4, 2.4)
    axis.set_ylim(0, max(high) * 1.4)
    axis.set_ylabel("Adapter LER (%) — 95% Wilson intervals")
    axis.grid(axis="y", alpha=0.25)
    for i, row in enumerate(rows):
        axis.annotate(
            f"{row['errors']}/{row['shots']:,} failures",
            (i, high[i]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    axis.set_title(
        "HGP simplex d=4 – surface d=4: hardware-only schedules\n"
        "4 ideal preparation + 4 adapter rounds; p=0.001, Bell F=1"
    )
    fig.text(
        0.5,
        0.025,
        "Dual ZZ has the same ordered detector-error model as reference XX; "
        "samples are independent.\nThis comparison does not test block-only faults or full-surgery LER.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.095, 1, 1))
    for extension in ("pdf", "png"):
        fig.savefig(output_dir / f"hgp_hardware_dual_schedule_comparison.{extension}", dpi=180)
    plt.close(fig)
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
        parser.error("Shot counts and worker count must be positive")
    if args.analyze_only:
        write_comparison(args.output_dir)
        return
    started = time.perf_counter()
    experiment, audit = build_dual_experiment()
    write_artifacts([experiment], args.output_dir)
    for name, content in (
        ("schedule_audit.json", audit),
        (
            "run_config.json",
            {
                "schedule_policy": SCHEDULE,
                "max_shots": args.max_shots,
                "pilot_shots": args.pilot_shots,
                "num_workers": args.num_workers,
                "reference_directory": str(REFERENCE),
                "noise_scope": "intrinsic",
                "adapter_noise_targets": "adapter_hardware",
                "preparation_rounds": 4,
                "surgery_rounds": 4,
                "physical_error_rate": 0.001,
                "bell_pair_fidelity": 1.0,
                "include_idling_error": False,
                "compute_exact_distances": False,
                "dem_analysis": "undecomposed_exact",
                "versions": {name: version(name) for name in ("stim", "sinter", "ldpc", "numpy")},
                "threads": {
                    name: os.environ.get(name)
                    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
                },
            },
        ),
    ):
        (args.output_dir / name).write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
    print(
        f"Dual ZZ validated: {audit['num_detectors']} detectors, {audit['num_dem_errors']} DEM errors; "
        f"ordered XX DEM identical; task {audit['strong_id']}",
        flush=True,
    )
    if args.build_only:
        return
    for cap in sorted({min(args.pilot_shots, args.max_shots), args.max_shots}):
        stage_started = time.perf_counter()
        print(f"Collect/resume the same hardware-only ZZ task to {cap:,} shots", flush=True)
        stats = sample_experiments(
            [experiment],
            output_dir=args.output_dir,
            num_workers=args.num_workers,
            max_shots=cap,
            max_errors=cap,
            print_progress=False,
        )
        if len(stats) != 1 or stats[0].strong_id != audit["strong_id"]:
            raise ValueError("Sampling returned a different task identity")
        write_summary(stats, args.output_dir)
        write_comparison(args.output_dir)
        record = dict(
            stage_cap=cap,
            shots=stats[0].shots,
            errors=stats[0].errors,
            stage_wall_seconds=time.perf_counter() - stage_started,
            invocation_wall_seconds=time.perf_counter() - started,
            strong_id=stats[0].strong_id,
        )
        with (args.output_dir / "invocations.jsonl").open("a", encoding="utf-8") as dest:
            dest.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
