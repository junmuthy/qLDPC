"""Read-only circuit checks for the m=4/d=8 pilot; write a separate validation report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import stim

from qldpc.circuits.detector_reduction import remove_inactive_detectors
from qldpc.circuits.surgery import JointPPMSchedule
from qldpc.objects import Pauli

from .run_experiment import build_experiment_for_codes
from .run_hgp_schedule_ablations import _ideal_prefix, _layout, _local_prefix, audit_variant
from .run_ppm_basis_first import code_pair


def validate(*, surgery_rounds: int = 4) -> list[dict[str, Any]]:
    if surgery_rounds < 1:
        raise ValueError("surgery_rounds must be positive")
    left, right, left_label, right_label = code_pair(
        "hgp-simplex-surface", simplex_m=4, surface_distance=8
    )
    reports = []
    for basis in (Pauli.X, Pauli.Z):
        common: dict[str, Any] = dict(
            left_code=left,
            right_code=right,
            left_code_name=left_label,
            right_code_name=right_label,
            basis=basis,
            preparation_rounds=4,
            surgery_rounds=surgery_rounds,
            physical_error_rate=0.001,
            bell_pair_infidelity=0.0,
            adapter_noise_targets="adapter_hardware",
            reduce_detectors=False,
            noise_scope="intrinsic",
            include_idling_error=False,
            compute_exact_distances=False,
        )
        legacy = build_experiment_for_codes(**common)
        variant = build_experiment_for_codes(**common, schedule_policy="ppm_basis_first")
        layout = _layout(left, right, basis)
        schedule = JointPPMSchedule.ppm_basis_first(basis)
        report = audit_variant(legacy, variant, layout, schedule, "local_order")
        # Explicitly compare local families, not just their aggregate gate inventory.
        old_round, _ = layout.strategy.get_circuit(layout.code, layout.ids)
        old_prefix = _local_prefix(old_round)
        local_x, local_z = layout.strategy.local_interaction_circuits(layout.code, layout.ids)
        from qldpc.circuits.surgery import JointPPMSelectiveBellPairSyndrome

        strategy = JointPPMSelectiveBellPairSyndrome(
            layout.strategy.g_l,
            layout.strategy.g_r,
            layout.strategy.bridge,
            independent_blocks=True,
            schedule=schedule,
        )
        new_round, _ = strategy.get_circuit(layout.code, layout.ids)
        expected = stim.Circuit()
        expected.append(old_prefix[0])
        expected += local_x + local_z if basis == Pauli.X else local_z + local_x
        expected.append(old_prefix[-1])
        assert _local_prefix(new_round) == expected
        assert _ideal_prefix(variant.circuit).num_measurements == 4 * (
            left.num_checks + right.num_checks
        )
        block_ids = set(range(len(left) + len(right)))
        terminal = [
            op
            for op in variant.circuit.flattened()
            if op.name in {"M", "MX"} and any(t.value in block_ids for t in op.targets_copy())
        ]
        assert terminal and all(not any(op.gate_args_copy()) for op in terminal)
        assert {t.value for op in terminal for t in op.targets_copy()} == block_ids
        if basis == Pauli.X:
            assert variant.circuit == legacy.circuit
        reduced = remove_inactive_detectors(variant.circuit)
        dets, obs = (
            reduced.circuit.without_noise()
            .compile_detector_sampler(seed=815)
            .sample(128, separate_observables=True)
        )
        assert not dets.any() and not obs.any()
        report.update(
            logical_basis=basis.name,
            local_family_layers_unchanged=True,
            local_order=schedule.metadata()["local_order"],
            bell_order=schedule.metadata()["bell_order"],
            preparation_rounds=4,
            surgery_rounds=surgery_rounds,
            all_four_preparation_rounds_ideal=True,
            terminal_block_readout_ideal=True,
            noiseless_samples_full=64,
            noiseless_samples_reduced=128,
            reduced_detectors=reduced.circuit.num_detectors,
            reduced_dem_errors=reduced.detector_error_model.num_errors,
            reduced_circuit_sha256=hashlib.sha256(str(reduced.circuit).encode()).hexdigest(),
            reduced_dem_sha256=hashlib.sha256(
                str(reduced.detector_error_model).encode()
            ).hexdigest(),
        )
        reports.append(report)
        print(
            f"{basis.name}{basis.name}: validated ideal boundaries, local order, Bell checks, detector meanings, and noiseless outcomes",
            flush=True,
        )
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=4, help="Adapter extraction rounds")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    if args.report.exists():
        parser.error("Report already exists; choose a new path to preserve prior validation")
    reports = validate(surgery_rounds=args.rounds)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
