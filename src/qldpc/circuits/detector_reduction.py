"""Remove detector rows that cannot respond to any circuit fault.

This reduction leaves all quantum operations and logical observables intact.
It removes only always-zero detector *events*, not individual stabilizer
measurements, whose ideal results may be random or entangled with other qubits.
"""

from __future__ import annotations

import dataclasses

import stim


@dataclasses.dataclass(frozen=True)
class DetectorReduction:
    circuit: stim.Circuit
    detector_error_model: stim.DetectorErrorModel
    num_original_detectors: int
    retained_detector_ids: tuple[int, ...]


def remove_inactive_detectors(circuit: stim.Circuit) -> DetectorReduction:
    """Keep every detector appearing in a nonzero-probability DEM error term.

    Stim's default analysis rejects nondeterministic detectors. The resulting
    fault support includes propagation through ideal operations and both time
    boundaries; no geometric proximity cutoff is applied. Logical-only error
    terms remain present. Errors are analyzed without graphlike decomposition
    or disjoint-error approximation. Unsupported circuits raise instead of
    silently changing the noise model.
    """
    dem = circuit.detector_error_model()
    active = {
        target.val
        for instruction in dem.flattened()
        if instruction.type == "error" and instruction.args_copy()[0] > 0
        for target in instruction.targets_copy()
        if target.is_relative_detector_id()
    }
    retained = tuple(sorted(active))
    reduced = stim.Circuit()
    detector_index = 0
    for instruction in circuit.flattened():
        if instruction.name == "DETECTOR":
            if detector_index in active:
                reduced.append(instruction)
            detector_index += 1
        else:
            reduced.append(instruction)
    reduced_dem = reduced.detector_error_model()
    return DetectorReduction(reduced, reduced_dem, circuit.num_detectors, retained)
