"""Transport a CSS extraction circuit through a validated X/Z duality.

This conjugates the *description* by Hadamards on data and relabels qubits;
it does not insert physical Hadamard gates. Measurement/observable/detector
ordering is retained, including ideal input/output boundaries. Unsupported
operations fail closed instead of silently receiving an incorrect transform.
"""

from __future__ import annotations

import collections
import dataclasses
import operator
from collections.abc import Sequence

import numpy as np
import stim

from qldpc.circuits.bookkeeping import QubitIDs
from qldpc.codes.common import CSSCode


@dataclasses.dataclass(frozen=True)
class CSSXZDuality:
    """Validated data/check/Bell-partner correspondence for an extraction circuit."""

    source_data: tuple[int, ...]
    qubit_map: dict[int, int]

    @classmethod
    def from_codes(
        cls,
        source: CSSCode,
        target: CSSCode,
        source_ids: QubitIDs,
        target_ids: QubitIDs,
        *,
        data_permutation: Sequence[int],
        source_bell_checks: Sequence[int] = (),
        target_bell_checks: Sequence[int] = (),
    ) -> CSSXZDuality:
        """Match opposite check rows and Bell partners under an explicit data map.

        ``data_permutation[i]`` is the target column for source column i.
        Bell-check sequences contain physical check IDs, one per auxiliary
        ancilla in allocation order. No code-family-specific IDs are assumed.
        """
        for code, ids in ((source, source_ids), (target, target_ids)):
            if code.field.order != 2 or code.is_subsystem_code:
                raise ValueError("Dual schedules require binary stabilizer CSS codes")
            QubitIDs.validated(ids, code)
            if len(set(ids.all_qubits)) != len(ids.all_qubits) or any(
                q < 0 for q in ids.all_qubits
            ):
                raise ValueError("Physical qubit IDs must be distinct nonnegative integers")
        try:
            permutation = tuple(operator.index(i) for i in data_permutation)
        except TypeError as ex:
            raise ValueError("data_permutation entries must be integers") from ex
        if len(source) != len(target) or sorted(permutation) != list(range(len(source))):
            raise ValueError("data_permutation must be a bijection between equal-size codes")
        mapping = {source_ids.data[i]: target_ids.data[j] for i, j in enumerate(permutation)}
        for source_matrix, source_checks, target_matrix, target_checks in (
            (source.matrix_x, source_ids.checks_x, target.matrix_z, target_ids.checks_z),
            (source.matrix_z, source_ids.checks_z, target.matrix_x, target_ids.checks_x),
        ):
            available: dict[tuple[int, ...], collections.deque[int]] = collections.defaultdict(
                collections.deque
            )
            for row, check in zip(target_matrix, target_checks):
                available[tuple(np.flatnonzero(np.asarray(row)))].append(check)
            for row, check in zip(source_matrix, source_checks):
                support = tuple(sorted(permutation[i] for i in np.flatnonzero(np.asarray(row))))
                if not available[support]:
                    raise ValueError("Check supports do not match under the requested X/Z duality")
                mapping[check] = available[support].popleft()
            if any(available.values()):
                raise ValueError("X/Z duality leaves unmatched target check rows")

        if (
            len(source_bell_checks) != len(source_ids.ancilla)
            or len(target_bell_checks) != len(target_ids.ancilla)
            or len(set(source_bell_checks)) != len(source_bell_checks)
            or len(set(target_bell_checks)) != len(target_bell_checks)
            or not set(source_bell_checks) <= set(source_ids.check)
            or not set(target_bell_checks) <= set(target_ids.check)
        ):
            raise ValueError("Bell-check owners must identify every auxiliary ancilla uniquely")
        target_partners = dict(zip(target_bell_checks, target_ids.ancilla))
        for check, partner in zip(source_bell_checks, source_ids.ancilla):
            if mapping[check] not in target_partners:
                raise ValueError("Bell-check selections are not X/Z counterparts")
            mapping[partner] = target_partners[mapping[check]]
        if set(mapping.values()) != set(target_ids.all_qubits) or len(mapping) != len(
            set(mapping.values())
        ):
            raise ValueError("The complete physical-qubit mapping is not bijective")
        return cls(source_ids.data, mapping)

    def transform(self, circuit: stim.Circuit) -> stim.Circuit:
        """Preserve all operation order, record references, tags, and noise probabilities.

        Coordinates follow the transported reference layout; detector IDs stay
        in reference order. The supported gate set is the one used by CSS
        controlled-Pauli/Bell extraction with independent Pauli noise.
        """
        data = set(self.source_data)
        one_qubit_duals = {
            "R": "RX",
            "RX": "R",
            "M": "MX",
            "MX": "M",
            "MR": "MRX",
            "MRX": "MR",
            "X": "Z",
            "Z": "X",
            "X_ERROR": "Z_ERROR",
            "Z_ERROR": "X_ERROR",
        }
        one_qubit_invariant = {"H", "I", "Y", "Y_ERROR", "DEPOLARIZE1"}
        result = stim.Circuit()
        for instruction in circuit:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                result.append(
                    stim.CircuitRepeatBlock(
                        instruction.repeat_count,
                        self.transform(instruction.body_copy()),
                        tag=instruction.tag,
                    )
                )
                continue
            name = instruction.name
            if name in {"TICK", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"}:
                if any(not t.is_measurement_record_target for t in instruction.targets_copy()):
                    raise ValueError(f"Unsupported annotation targets in {instruction}")
                result.append(instruction)
                continue
            targets = instruction.targets_copy()
            if any(not t.is_qubit_target or t.is_inverted_result_target for t in targets):
                raise ValueError(f"Unsupported dual-schedule targets in {instruction}")
            qubits = [t.value for t in targets]
            if not set(qubits) <= self.qubit_map.keys():
                raise ValueError(
                    f"Circuit uses qubits outside the validated duality: {instruction}"
                )
            args = instruction.gate_args_copy()
            if name == "QUBIT_COORDS":
                result.append(name, [self.qubit_map[q] for q in qubits], args, tag=instruction.tag)
            elif (
                name in one_qubit_duals or name in one_qubit_invariant or name == "PAULI_CHANNEL_1"
            ):
                for q in qubits:
                    dual_name = one_qubit_duals.get(name, name) if q in data else name
                    dual_args = (
                        [args[2], args[1], args[0]]
                        if name == "PAULI_CHANNEL_1" and q in data
                        else args
                    )
                    result.append(dual_name, [self.qubit_map[q]], dual_args, tag=instruction.tag)
            elif name in {"CX", "CZ", "DEPOLARIZE2"}:
                for a, b in zip(qubits[::2], qubits[1::2]):
                    if name != "DEPOLARIZE2" and a in data:
                        raise ValueError(
                            "Dual extraction requires check-controlled gates, not data controls"
                        )
                    dual_name = {"CX": "CZ", "CZ": "CX"}.get(name, name) if b in data else name
                    result.append(
                        dual_name, [self.qubit_map[a], self.qubit_map[b]], args, tag=instruction.tag
                    )
            else:
                raise ValueError(f"Unsupported gate in CSS dual schedule: {name}")
        return result
