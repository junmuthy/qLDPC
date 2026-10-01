"""Validated exact X/Z-dual hardware circuits for simplex HGP / unrotated surface.

The dimension-derived permutation is a candidate, never an assumed symmetry:
logical supports, every check row, Bell ownership, and noise masks must match.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

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

from .run_experiment import BuiltExperiment, build_experiment_for_codes
from .run_hgp_dual_schedule import _validate_ppm_readout

POLICY = "exact_xz_dual_from_matching_x_v1"


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def canonical(value: Any) -> Any:
    """Recursively stabilize metadata ordering before Sinter computes task identity."""
    return json.loads(json.dumps(value, sort_keys=True))


def transpose_permutation(
    left: codes.HGPCode, surface_distance: int, num_data: int
) -> tuple[int, ...]:
    if not np.array_equal(left.code_a.matrix, left.code_b.matrix):
        raise ValueError("This preset requires identical HGP seeds")
    sides = (left.code_a.num_bits, left.code_a.num_checks, surface_distance, surface_distance - 1)
    if surface_distance < 2 or sum(side * side for side in sides) > num_data:
        raise ValueError("Invalid surface or merged-code dimensions")
    result = list(range(num_data))
    offset = 0
    for side in sides:
        for index in range(side * side):
            result[offset + index] = offset + index % side * side + index // side
        offset += side * side
    # Gadget and bridge columns keep their candidate order, subject to full validation.
    return tuple(result)


@dataclasses.dataclass
class DualBuild:
    experiment: BuiltExperiment
    source: BuiltExperiment
    matching_z: BuiltExperiment
    audit: dict[str, Any]


def _reduced(experiment: BuiltExperiment) -> BuiltExperiment:
    reduction = remove_inactive_detectors(experiment.circuit)
    meta = dict(experiment.metadata)
    meta.update(
        detector_reduction="inactive_rows",
        retained_detector_ids=list(reduction.retained_detector_ids),
        num_detectors=reduction.circuit.num_detectors,
        num_dem_errors=reduction.detector_error_model.num_errors,
    )
    return BuiltExperiment(experiment.name, reduction.circuit, reduction.detector_error_model, meta)


def build_dual_experiment(
    *,
    simplex_m: int = 4,
    surface_distance: int | None = None,
    left_logical_index: int = 0,
    right_logical_index: int = 0,
    preparation_rounds: int = 4,
    surgery_rounds: int = 8,
    physical_error_rate: float = 0.001,
    bell_pair_infidelity: float = 0.0,
) -> DualBuild:
    if simplex_m < 2 or min(preparation_rounds, surgery_rounds) < 1:
        raise ValueError("Invalid simplex dimension or round count")
    distance = 2 ** (simplex_m - 1) if surface_distance is None else surface_distance
    if distance < 2 or min(left_logical_index, right_logical_index) < 0:
        raise ValueError("Invalid surface distance or logical index")
    if not all(0 <= p <= 1 for p in (physical_error_rate, bell_pair_infidelity)):
        raise ValueError("Noise probabilities must be between zero and one")
    left = codes.HGPCode(codes.SimplexCode(simplex_m))
    right = codes.SurfaceCode(distance, rotated=False)
    if left_logical_index >= left.dimension or right_logical_index >= right.dimension:
        raise ValueError("Logical index is outside the selected block")
    full = {}
    layouts = []
    for basis in (Pauli.X, Pauli.Z):
        full[basis] = build_experiment_for_codes(
            left_code=left,
            right_code=right,
            left_code_name=f"hgp_simplex_m{simplex_m}",
            right_code_name=f"surface_unrotated_d{distance}",
            left_logical_index=left_logical_index,
            right_logical_index=right_logical_index,
            basis=basis,
            preparation_rounds=preparation_rounds,
            surgery_rounds=surgery_rounds,
            physical_error_rate=physical_error_rate,
            bell_pair_infidelity=bell_pair_infidelity,
            include_idling_error=False,
            noise_scope="intrinsic",
            compute_exact_distances=False,
            adapter_noise_targets="adapter_hardware",
            reduce_detectors=False,
            schedule_policy="ppm_basis_first",
        )
        gl = build_gadget(left, left.get_logical_ops(basis)[left_logical_index], basis=basis)
        gr = build_gadget(right, right.get_logical_ops(basis)[right_logical_index], basis=basis)
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
    permutation = transpose_permutation(left, distance, len(sx))
    source_support = set(glx.support) | {len(left) + i for i in grx.support}
    target_support = set(glz.support) | {len(left) + i for i in grz.support}
    if {permutation[i] for i in source_support} != target_support:
        raise ValueError("Selected logical supports are not counterparts under the preset duality")
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
    for a, b in zip((hx.data, hx.check, hx.ancilla), (hz.data, hz.check, hz.ancilla)):
        if {mapping.qubit_map[q] for q in a} != set(b):
            raise ValueError("Duality changes the hardware-only noise mask")
    transformed = mapping.transform(full[Pauli.X].circuit)
    ppm_start = left.num_checks_z + right.num_checks_z
    ppm_ids = iz.checks_z[ppm_start : ppm_start + len(glz.support) + len(grz.support)]
    _validate_ppm_readout(transformed, ppm_ids, dict(zip(bz, iz.ancilla)))
    source, matching_z = (_reduced(full[p]) for p in (Pauli.X, Pauli.Z))
    reduction = remove_inactive_detectors(transformed)
    if list(reduction.retained_detector_ids) != source.metadata["retained_detector_ids"]:
        raise ValueError("Duality changed inactive detector selection")
    if reduction.detector_error_model != source.detector_error_model:
        raise ValueError("Exact ordered DEM equality, including priors, failed")
    for circuit in (transformed, reduction.circuit):
        ideal = circuit.without_noise()
        if ideal.detector_error_model().num_errors:
            raise ValueError("Noiseless dual DEM has stochastic errors")
        dets, obs = ideal.compile_detector_sampler(seed=817).sample(64, separate_observables=True)
        if dets.any() or obs.any():
            raise ValueError("Noiseless dual sampling failed")
    if (source.circuit.num_ticks, source.circuit.num_measurements) != (
        transformed.num_ticks,
        transformed.num_measurements,
    ):
        raise ValueError("Duality changed circuit depth or measurement count")
    name = full[Pauli.Z].name.split("_schedule_")[0] + f"_{POLICY}"
    meta = dict(matching_z.metadata)
    meta.update(
        name=name,
        schedule_policy=POLICY,
        schedule_reference_basis="X",
        schedule_reference_circuit_sha256=digest(str(source.circuit)),
        schedule_reference_dem_sha256=digest(str(source.detector_error_model)),
        schedule_mapping_sha256=digest(json.dumps(sorted(mapping.qubit_map.items()))),
        surgery_schedule={
            "local_order": ["Z", "X"],
            "bell_order": ["Z", "X"],
            "local_layers_convention": "transported_reference_x",
        },
        boundary_convention="complete_reference_xz_transport",
        detector_order="transported_reference_x",
        dem_analysis="undecomposed_exact",
        adapter_hardware_qubits=list(hz.all_qubits),
        num_detectors=reduction.circuit.num_detectors,
        num_dem_errors=reduction.detector_error_model.num_errors,
        num_full_detectors=transformed.num_detectors,
        num_full_dem_errors=transformed.detector_error_model().num_errors,
        retained_detector_ids=list(reduction.retained_detector_ids),
    )
    experiment = BuiltExperiment(
        name, reduction.circuit, reduction.detector_error_model, canonical(meta)
    )
    audit = dict(
        schedule_policy=POLICY,
        logical_support_and_all_check_rows_mapped=True,
        physical_partitions_and_hardware_mask_preserved=True,
        target_zz_ppm_readout_validated=True,
        identical_ordered_dem_including_priors=True,
        identical_inactive_detector_selection=True,
        noiseless_full_and_reduced_samples_each=64,
        full_circuit_including_ideal_boundaries_transported=True,
        source_data_columns_to_target=list(permutation),
        source_qubits_to_target={str(k): v for k, v in sorted(mapping.qubit_map.items())},
        source_num_ticks=source.circuit.num_ticks,
        dual_num_ticks=transformed.num_ticks,
        num_measurements=transformed.num_measurements,
        source_circuit_sha256=digest(str(source.circuit)),
        dual_circuit_sha256=digest(str(experiment.circuit)),
        ordered_dem_sha256=digest(str(experiment.detector_error_model)),
        strong_id=sinter.Task(
            circuit=experiment.circuit,
            detector_error_model=experiment.detector_error_model,
            decoder="qldpc_bp_osd",
            json_metadata=experiment.metadata,
        ).strong_id(),
    )
    return DualBuild(experiment, source, matching_z, audit)


def validate_reference(directory: Path, built: DualBuild) -> dict[str, dict[str, Any]]:
    """Reference data are read-only, matched by settings AND canonical circuit/DEM."""
    entries = json.loads((directory / "manifest.json").read_text())
    selected = {}
    for basis, expected in (("X", built.source), ("Z", built.matching_z)):
        keys = (
            "logical_basis",
            "left_code",
            "right_code",
            "left_logical_index",
            "right_logical_index",
            "preparation_rounds",
            "surgery_rounds",
            "noise_scope",
            "adapter_noise_targets",
            "physical_error_rate",
            "bell_pair_infidelity",
            "include_idling_error",
            "decoder",
            "schedule_policy",
        )
        matches = [e for e in entries if all(e.get(k) == expected.metadata[k] for k in keys)]
        if len(matches) != 1:
            raise ValueError(
                f"Reference must contain one matching basis-first {basis}{basis} condition"
            )
        entry = matches[0]
        if (
            stim.Circuit.from_file(directory / entry["circuit_path"]) != expected.circuit
            or stim.DetectorErrorModel.from_file(directory / entry["dem_path"])
            != expected.detector_error_model
        ):
            raise ValueError("Reference circuit/DEM no longer matches the constructed baseline")
        selected[basis] = entry
    return selected
