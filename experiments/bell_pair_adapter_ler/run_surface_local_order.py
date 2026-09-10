"""Surface–Steane hardware sweep: change only ZZ local-family order, reuse XX.

The ten new ZZ tasks use fixed shot caps, separate identities, and a shared
resumable Sinter checkpoint. Legacy XX/ZZ results are read-only references.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import sinter
import stim
from scipy.stats import fisher_exact

from qldpc import codes
from qldpc.circuits.detector_reduction import remove_inactive_detectors
from qldpc.circuits.surgery import JointPPMSchedule, JointPPMSelectiveBellPairSyndrome
from qldpc.objects import Pauli

from .run_experiment import (
    BuiltExperiment,
    _wilson_interval,
    build_experiment_for_codes,
    sample_experiments,
    write_artifacts,
    write_summary,
)
from .run_hgp_dual_schedule import _sha256
from .run_hgp_schedule_ablations import (
    _layout,
    _local_prefix,
    _read_csv,
    _write_csv,
    audit_variant,
    holm_adjust,
)

OUTPUTS = Path(__file__).resolve().parent / "outputs"
REFERENCE = OUTPUTS / "adapter_hardware_prep3_round_sweep_1_5_comparison"
DEFAULT_OUTPUT = OUTPUTS / "surface_steane_hardware_local_order_prep3_rounds1_5_20k"
SETTINGS = ((0.0, "bell0"), (0.001, "bell0p001"))
SCHEDULE = "ppm_basis_first_local_v1"


def _validate_entry(entry: dict[str, Any]) -> tuple[str, int, float]:
    basis, rounds, infidelity = (
        entry["logical_basis"],
        int(entry["surgery_rounds"]),
        float(entry["bell_pair_infidelity"]),
    )
    if not (
        basis in {"X", "Z"}
        and rounds in range(1, 6)
        and infidelity in {0.0, 0.001}
        and entry["left_code"] == "rotated_surface_d3"
        and entry["right_code"] == "steane_d3"
        and entry["left_logical_index"] == entry["right_logical_index"] == 0
        and entry["preparation_rounds"] == 3
        and entry["physical_error_rate"] == 0.001
        and entry["bell_pair_fidelity"] == 1 - infidelity
        and not entry["include_idling_error"]
        and entry["noise_scope"] == "intrinsic"
        and entry["adapter_noise_targets"] == "adapter_hardware"
        and entry["detector_reduction"] == "inactive_rows"
        and entry["decoder"] == "qldpc_bp_osd"
    ):
        raise ValueError("Mismatched Surface–Steane hardware settings")
    return basis, rounds, infidelity


def _terminal_quantum_suffix(circuit: stim.Circuit, num_data: int) -> stim.Circuit:
    """Quantum operations after the last check/auxiliary measurement, without record annotations."""
    suffix = stim.Circuit()
    for op in circuit.flattened():
        if op.name in {"M", "MX"} and any(t.value >= num_data for t in op.targets_copy()):
            suffix = stim.Circuit()
        elif op.name not in {"DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"}:
            suffix.append(op)
    return suffix


def verify_reference_hashes(audit: dict[str, Any]) -> None:
    for relative, expected in audit["reference_files_sha256"].items():
        if _sha256((REFERENCE / relative).read_bytes()) != expected:
            raise ValueError(f"Reference archive changed: {relative}")


def build_suite() -> tuple[list[BuiltExperiment], dict[str, Any]]:
    left, right = codes.SurfaceCode(3), codes.SteaneCode()
    # One-round audit templates are ideal (F=1); complete noisy inventories are
    # also checked separately for EACH actual Bell fidelity below.
    layout = _layout(left, right, Pauli.Z)
    schedule = JointPPMSchedule(local_order=(Pauli.Z, Pauli.X))
    strategy = JointPPMSelectiveBellPairSyndrome(
        layout.strategy.g_l,
        layout.strategy.g_r,
        layout.strategy.bridge,
        independent_blocks=True,
        schedule=schedule,
    )
    local_x, local_z = layout.strategy.local_interaction_circuits(layout.code, layout.ids)
    actual_layers = strategy.local_interaction_circuits(layout.code, layout.ids)
    if actual_layers != (local_x, local_z):
        raise ValueError("Order-only control changed the local family gate layers")
    ideal_round, _ = strategy.get_circuit(layout.code, layout.ids)
    old_round, _ = layout.strategy.get_circuit(layout.code, layout.ids)
    old_prefix = _local_prefix(old_round)
    expected_prefix = stim.Circuit()
    expected_prefix.append(old_prefix[0])  # unchanged local RX targets
    expected_prefix += local_z + local_x
    expected_prefix.append(old_prefix[-1])  # unchanged local MX targets
    if _local_prefix(ideal_round) != expected_prefix:
        raise ValueError("Local-order control changed more than the two family positions")
    common: dict[str, Any] = dict(
        left_code=left,
        right_code=right,
        left_code_name="rotated_surface_d3",
        right_code_name="steane_d3",
        preparation_rounds=3,
        physical_error_rate=0.001,
        include_idling_error=False,
        noise_scope="intrinsic",
        compute_exact_distances=False,
        adapter_noise_targets="adapter_hardware",
        reduce_detectors=False,
    )
    experiments: list[BuiltExperiment] = []
    checks: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for infidelity, slug in SETTINGS:
        directory = REFERENCE / slug
        for filename in ("manifest.json", "summary.csv", "run_config.json"):
            hashes[f"{slug}/{filename}"] = _sha256((directory / filename).read_bytes())
        entries = json.loads((directory / "manifest.json").read_text())
        expected = {(b, r, infidelity) for b in ("X", "Z") for r in range(1, 6)}
        keys = [_validate_entry(e) for e in entries]
        if len(entries) != 10 or set(keys) != expected:
            raise ValueError("Incomplete or duplicate legacy round sweep")
        for entry in entries:
            basis, rounds, _ = _validate_entry(entry)
            for key in ("circuit_path", "dem_path"):
                hashes[f"{slug}/{entry[key]}"] = _sha256((directory / entry[key]).read_bytes())
            legacy = build_experiment_for_codes(
                basis=Pauli[basis], surgery_rounds=rounds, bell_pair_infidelity=infidelity, **common
            )
            original = remove_inactive_detectors(legacy.circuit)
            archive_circuit = stim.Circuit.from_file(directory / entry["circuit_path"])
            archive_dem = stim.DetectorErrorModel.from_file(directory / entry["dem_path"])
            # Stim's text serialization rounds gate arguments. Compare canonical
            # serialized circuits, not the unrounded in-memory float bits, and
            # independently require EXACT ordered DEM equality (including priors).
            if (
                str(original.circuit) != str(archive_circuit)
                or original.detector_error_model != archive_dem
            ):
                raise ValueError(f"Default no longer matches archived circuit/DEM: {entry['name']}")
            references.append(
                dict(
                    name=entry["name"],
                    logical_basis=basis,
                    surgery_rounds=rounds,
                    bell_pair_infidelity=infidelity,
                    total_ticks=archive_circuit.num_ticks,
                    canonical_circuit_and_ordered_dem_match=True,
                    reused_without_new_shots=True,
                )
            )
            if basis == "X":
                continue
            full = build_experiment_for_codes(
                basis=Pauli.Z,
                surgery_rounds=rounds,
                bell_pair_infidelity=infidelity,
                surgery_schedule=schedule,
                **common,
            )
            check = audit_variant(legacy, full, layout, schedule, "local_order")
            if full.circuit.num_ticks != legacy.circuit.num_ticks:
                raise ValueError("Local-family swap changed complete circuit depth")
            if _terminal_quantum_suffix(full.circuit, len(layout.code)) != _terminal_quantum_suffix(
                legacy.circuit, len(layout.code)
            ):
                raise ValueError("Local-family swap changed detachment or terminal readout")
            reduction = remove_inactive_detectors(full.circuit)
            name = entry["name"] + "_local_zx_v1"
            metadata = dict(full.metadata)
            metadata.update(
                name=name,
                schedule_policy=SCHEDULE,
                detector_order="legacy_check_epoch",
                detector_reduction="inactive_rows",
                dem_analysis="undecomposed_exact",
                num_detectors=reduction.circuit.num_detectors,
                num_dem_errors=reduction.detector_error_model.num_errors,
                retained_detector_ids=list(reduction.retained_detector_ids),
                reference_name=entry["name"],
                reference_circuit_sha256=_sha256(str(archive_circuit)),
            )
            metadata = json.loads(json.dumps(metadata, sort_keys=True))
            experiment = BuiltExperiment(
                name, reduction.circuit, reduction.detector_error_model, metadata
            )
            check.update(
                name=name,
                logical_basis="Z",
                surgery_rounds=rounds,
                bell_pair_infidelity=infidelity,
                reference_name=entry["name"],
                local_family_layers_identical=True,
                local_prefix_exactly_z_then_x=True,
                terminal_quantum_operations_identical=True,
                num_detectors=experiment.circuit.num_detectors,
                num_dem_errors=experiment.detector_error_model.num_errors,
                ordered_dem_identical_to_legacy_zz=(experiment.detector_error_model == archive_dem),
                strong_id=sinter.Task(
                    circuit=experiment.circuit,
                    detector_error_model=experiment.detector_error_model,
                    decoder="qldpc_bp_osd",
                    json_metadata=metadata,
                ).strong_id(),
            )
            experiments.append(experiment)
            checks.append(check)
    experiments.sort(
        key=lambda e: (e.metadata["bell_pair_infidelity"], e.metadata["surgery_rounds"])
    )
    checks.sort(key=lambda e: (e["bell_pair_infidelity"], e["surgery_rounds"]))
    audit = dict(
        references=references,
        experiments=checks,
        reference_files_sha256=hashes,
        all_20_legacy_circuits_and_ordered_dems_match=True,
        xx_policy="unchanged_legacy_reused",
        zz_policy=SCHEDULE,
    )
    verify_reference_hashes(audit)
    return experiments, audit


def comparison_rows(output: Path) -> list[dict[str, Any]]:
    audit = json.loads((output / "schedule_audit.json").read_text())
    verify_reference_hashes(audit)
    info = {e["name"]: e for e in audit["references"] + audit["experiments"]}
    result = []
    seen: set[tuple[str, int, float, str]] = set()
    for directory, policy in [(REFERENCE / slug, "legacy") for _, slug in SETTINGS] + [
        (output, SCHEDULE)
    ]:
        manifest = {e["name"]: e for e in json.loads((directory / "manifest.json").read_text())}
        rows = _read_csv(directory / "summary.csv")
        if len(rows) != 10 or {r["name"] for r in rows} != set(manifest):
            raise ValueError("Incomplete, duplicate, or unexpected summary rows")
        for row in rows:
            entry = manifest[row["name"]]
            basis, rounds, infidelity = _validate_entry(entry)
            if not (
                row["logical_basis"] == basis
                and int(row["surgery_rounds"]) == rounds
                and int(row["preparation_rounds"]) == 3
                and row["noise_scope"] == "intrinsic"
                and float(row["physical_error_rate"]) == 0.001
                and float(row["bell_pair_infidelity"]) == infidelity
                and row["adapter_noise_targets"] == "adapter_hardware"
                and row["detector_reduction"] == "inactive_rows"
                and row["decoder"] == "qldpc_bp_osd"
            ):
                raise ValueError("Summary and manifest settings disagree")
            if policy == SCHEDULE and (basis != "Z" or entry.get("schedule_policy") != SCHEDULE):
                raise ValueError("Only the local-order ZZ variant belongs in this production run")
            key = (basis, rounds, infidelity, policy)
            if key in seen:
                raise ValueError("Duplicate comparison point")
            seen.add(key)
            n, k, discards = (int(row[key]) for key in ("shots", "errors", "discards"))
            if not (0 <= discards < n and 0 <= k <= n - discards):
                raise ValueError("Invalid shot/error counts")
            low, high = _wilson_interval(k, n - discards)
            result.append(
                dict(
                    name=row["name"],
                    logical_basis=basis,
                    surgery_rounds=rounds,
                    bell_pair_infidelity=infidelity,
                    bell_pair_fidelity=1 - infidelity,
                    schedule_policy=policy,
                    shots=n,
                    discards=discards,
                    errors=k,
                    logical_error_rate=k / (n - discards),
                    ler_95ci_low=low,
                    ler_95ci_high=high,
                    worker_seconds=float(row["seconds"]),
                    worker_seconds_per_10000_shots=10000 * float(row["seconds"]) / n,
                    total_ticks=info[row["name"]][
                        "total_ticks" if policy == "legacy" else "variant_total_ticks"
                    ],
                    num_detectors=entry["num_detectors"],
                    num_dem_errors=entry["num_dem_errors"],
                    early_error_cap_reached=(policy == "legacy" and n < 10000),
                    source_summary=str(directory / "summary.csv"),
                )
            )
    return result


def write_comparison(output: Path) -> list[dict[str, Any]]:
    rows = comparison_rows(output)
    _write_csv(output / "comparison.csv", rows)
    changes = []
    for row in rows:
        if row["schedule_policy"] != SCHEDULE:
            continue
        old = next(
            r
            for r in rows
            if r["schedule_policy"] == "legacy"
            and r["logical_basis"] == "Z"
            and r["surgery_rounds"] == row["surgery_rounds"]
            and r["bell_pair_infidelity"] == row["bell_pair_infidelity"]
        )
        n, n0 = row["shots"] - row["discards"], old["shots"] - old["discards"]
        p, p0 = row["logical_error_rate"], old["logical_error_rate"]
        changes.append(
            dict(
                surgery_rounds=row["surgery_rounds"],
                bell_pair_fidelity=row["bell_pair_fidelity"],
                legacy_shots=n0,
                legacy_errors=old["errors"],
                new_shots=n,
                new_errors=row["errors"],
                ler_difference=p - p0,
                difference_95ci_low=float(
                    p
                    - p0
                    - np.sqrt((p - row["ler_95ci_low"]) ** 2 + (old["ler_95ci_high"] - p0) ** 2)
                ),
                difference_95ci_high=float(
                    p
                    - p0
                    + np.sqrt((row["ler_95ci_high"] - p) ** 2 + (p0 - old["ler_95ci_low"]) ** 2)
                ),
                fisher_p=float(
                    fisher_exact(
                        [[row["errors"], n - row["errors"]], [old["errors"], n0 - old["errors"]]]
                    ).pvalue
                ),
                legacy_early_error_cap_reached=old["early_error_cap_reached"],
            )
        )
    for row, pvalue in zip(changes, holm_adjust([r["fisher_p"] for r in changes])):
        row["fisher_p_holm_10"] = pvalue
    _write_csv(output / "differences.csv", changes)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.7), sharex=True, sharey=True)
    for axis, (infidelity, _) in zip(axes, SETTINGS):
        for basis, policy, label, marker, color, style in (
            ("X", "legacy", "Legacy XX (unchanged, reused)", "^", "0.45", "--"),
            ("Z", "legacy", "Legacy ZZ: local X then Z", "o", "C1", "-"),
            ("Z", SCHEDULE, "New ZZ: local Z then X", "s", "C2", "-"),
        ):
            selected = sorted(
                (
                    r
                    for r in rows
                    if r["logical_basis"] == basis
                    and r["schedule_policy"] == policy
                    and r["bell_pair_infidelity"] == infidelity
                ),
                key=lambda r: r["surgery_rounds"],
            )
            x = np.array([r["surgery_rounds"] for r in selected])
            y, low, high = [
                np.array([r[key] for r in selected]) * 100
                for key in ("logical_error_rate", "ler_95ci_low", "ler_95ci_high")
            ]
            nonzero = y > 0
            axis.errorbar(
                x[nonzero],
                y[nonzero],
                yerr=np.maximum([y[nonzero] - low[nonzero], high[nonzero] - y[nonzero]], 0),
                marker=marker,
                color=color,
                linestyle=style,
                capsize=4,
                label=label,
            )
            if np.any(~nonzero):
                axis.errorbar(
                    x[~nonzero],
                    high[~nonzero],
                    yerr=high[~nonzero] * 0.2,
                    uplims=True,
                    fmt=marker,
                    color=color,
                    capsize=4,
                )
        axis.set_title(f"Bell fidelity {1 - infidelity:g}")
        axis.set_xlabel("Adapter syndrome rounds")
        axis.set_xticks(range(1, 6))
        axis.set_xlim(0.8, 5.2)
        axis.set_yscale("log")
        axis.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("Adapter LER (%) — 95% Wilson intervals")
    axes[0].legend(fontsize=9)
    fig.suptitle(
        "Surface d=3 – Steane: hardware-only local-order test\n"
        "3 ideal preparation rounds; physical p=0.001; unchanged adapter geometry and Bell order"
    )
    fig.text(
        0.5,
        0.025,
        "Only ZZ merged local-family order changes. XX data are reused, not resampled.\n"
        "Pointwise intervals; zero failures shown as upper limits. Legacy error-capped counts make nominal tests approximate.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.91))
    for extension in ("pdf", "png"):
        fig.savefig(output / f"adapter_ler_vs_rounds_local_order.{extension}", dpi=180)
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
        preparation_rounds=3,
        surgery_rounds=list(range(1, 6)),
        physical_error_rate=0.001,
        bell_pair_fidelities=[1.0, 0.999],
        include_idling_error=False,
        noise_scope="intrinsic",
        adapter_noise_targets="adapter_hardware",
        detector_reduction="inactive_rows",
        dem_analysis="undecomposed_exact",
        compute_exact_distances=False,
        schedule_policy=SCHEDULE,
        sampled_bases=["Z"],
        reused_bases=["X", "Z"],
        reference_directory=str(REFERENCE),
        versions={n: version(n) for n in ("stim", "sinter", "ldpc", "numpy", "scipy")},
        threads={n: os.environ.get(n) for n in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    for filename, data in (("schedule_audit.json", audit), ("run_config.json", config)):
        (args.output_dir / filename).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(
        "Validated all 20 archived XX/ZZ circuits and ten local-order-only ZZ variants.", flush=True
    )
    for e in audit["experiments"]:
        print(
            json.dumps(
                {
                    k: e[k]
                    for k in (
                        "surgery_rounds",
                        "bell_pair_infidelity",
                        "variant_total_ticks",
                        "num_detectors",
                        "ordered_dem_identical_to_legacy_zz",
                    )
                }
            ),
            flush=True,
        )
    if args.build_only:
        return
    expected_ids = {e["strong_id"] for e in audit["experiments"]}
    for cap in sorted({min(args.pilot_shots, args.max_shots), args.max_shots}):
        stage_started = time.perf_counter()
        print(f"Collect/resume all ten ZZ configurations to {cap:,} shots each", flush=True)
        stats = sample_experiments(
            experiments,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
            max_shots=cap,
            max_errors=cap,
            print_progress=False,
        )
        if len(stats) != 10 or {s.strong_id for s in stats} != expected_ids:
            raise ValueError("Sampling returned unexpected task identities")
        write_summary(stats, args.output_dir)
        write_comparison(args.output_dir)
        record = dict(
            stage_cap_per_task=cap,
            total_shots=sum(s.shots for s in stats),
            total_errors=sum(s.errors for s in stats),
            stage_wall_seconds=time.perf_counter() - stage_started,
            invocation_wall_seconds=time.perf_counter() - started,
        )
        with (args.output_dir / "invocations.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
