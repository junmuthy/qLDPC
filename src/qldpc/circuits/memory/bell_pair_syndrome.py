from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
import stim

from qldpc import codes
from qldpc.objects import Pauli

from ..bookkeeping import ParityMeasurementRecord, QubitIDs
from ..common import restrict_to_qubits
from .syndrome_measurement import SyndromeMeasurementStrategy


PauliFactor = tuple[int, Pauli]
SupportSplit = Callable[
    [tuple[PauliFactor, ...], int],
    tuple[Sequence[PauliFactor], Sequence[PauliFactor]],
]


def balanced_split(
    factors: tuple[PauliFactor, ...],
    check_index: int,
) -> tuple[tuple[PauliFactor, ...], tuple[PauliFactor, ...]]:
    """Split a stabilizer support roughly in half."""
    midpoint = len(factors) // 2
    return factors[:midpoint], factors[midpoint:]


def _row_to_pauli_factors(
    row: npt.NDArray[np.int_],
    num_data_qubits: int,
) -> tuple[PauliFactor, ...]:
    """Convert one binary symplectic row [X | Z] into Pauli factors."""
    row = np.asarray(row, dtype=np.uint8) % 2
    expected_shape = (2 * num_data_qubits,)

    if row.shape != expected_shape:
        raise ValueError(f"Expected a stabilizer row of shape {expected_shape}, got {row.shape}.")

    factors: list[PauliFactor] = []
    for col, (x_value, z_value) in enumerate(
        zip(row[:num_data_qubits], row[num_data_qubits:])
    ):
        pauli = Pauli((int(x_value), int(z_value)))
        if pauli is not Pauli.I:
            factors.append((col, pauli))

    return tuple(factors)


def _validate_factor_split(
    factors: tuple[PauliFactor, ...],
    left_factors: Sequence[PauliFactor],
    right_factors: Sequence[PauliFactor],
) -> tuple[tuple[PauliFactor, ...], tuple[PauliFactor, ...]]:
    """Validate that left/right factors partition the stabilizer factors."""
    left = tuple(left_factors)
    right = tuple(right_factors)

    expected = set(factors)
    actual = set(left) | set(right)

    if actual != expected:
        raise ValueError(
            "split_support must cover exactly the stabilizer support. "
            f"Expected {_sorted_factors(expected)}, got {_sorted_factors(actual)}."
        )

    overlap = set(left) & set(right)
    if overlap:
        raise ValueError(
            "split_support must not put the same Pauli factor on both Bell halves. "
            f"Overlap: {_sorted_factors(overlap)}."
        )

    if len(left) + len(right) != len(factors):
        raise ValueError(
            "split_support must not duplicate Pauli factors. "
            f"left={left}, right={right}, factors={factors}."
        )

    return left, right


def _sorted_factors(factors: set[PauliFactor]) -> list[PauliFactor]:
    """Return Pauli factors in a stable order for validation messages."""
    return sorted(factors, key=lambda factor: (factor[0], factor[1].name))


def _append_bell_pair_check_measurement(
    circuit: stim.Circuit,
    qubit_ids: QubitIDs,
    *,
    check_id: int,
    bell_right: int,
    left_factors: Sequence[PauliFactor],
    right_factors: Sequence[PauliFactor],
    measurement_index: int,
) -> tuple[int, int]:
    """Append Bell-pair measurement of one stabilizer check and return raw event indices."""
    bell_left = check_id

    # Prepare |Phi+> = (|00> + |11>) / sqrt(2) on the two Bell halves.
    circuit.append("RZ", [bell_left, bell_right])
    circuit.append("H", [bell_left])
    circuit.append("CX", [bell_left, bell_right])
    circuit.append("TICK")

    # One interaction on each Bell half can be done in the same scheduled layer.
    for left_factor, right_factor in itertools.zip_longest(left_factors, right_factors):
        if left_factor is not None:
            data_col, pauli = left_factor
            circuit.append(f"C{pauli}", [bell_left, qubit_ids.data[data_col]])

        if right_factor is not None:
            data_col, pauli = right_factor
            circuit.append(f"C{pauli}", [bell_right, qubit_ids.data[data_col]])

        circuit.append("TICK")

    m_left = measurement_index
    m_right = measurement_index + 1

    circuit.append("MX", [bell_left, bell_right])
    circuit.append("TICK")

    return m_left, m_right


class BellPairParitySyndrome(SyndromeMeasurementStrategy):
    """Syndrome extraction where each check is an XOR of two Bell-half readouts.

    For each stabilizer generator S, represented as a general Pauli string:

    1. Prepare |Phi+> on two Bell halves a and b.
    2. Split the support of S into left and right pieces.
    3. For each left Pauli factor P_j, apply controlled-P_j from a to data qubit j.
    4. For each right Pauli factor P_j, apply controlled-P_j from b to data qubit j.
    5. Measure a and b in X.
    6. The effective syndrome bit is m_a XOR m_b.

    This supports arbitrary qubit stabilizer checks containing X, Y, and Z factors.
    """

    def __init__(
        self,
        *,
        split_support: SupportSplit = balanced_split,
        ancilla_offset: int | None = None,
    ) -> None:
        """Initialize the Bell-pair syndrome strategy.

        Args:
            split_support: Function that partitions a stabilizer's Pauli factors between the two
                Bell halves.
            ancilla_offset: Offset into qubit_ids.ancilla where the extra Bell halves begin.  If
                None, defaults to code.dimension so the first code.dimension ancillas remain
                available for basis=None memory experiments.  For fixed-basis CSS memory
                experiments, set ancilla_offset=0 to avoid reserving logical Bell-pair ancillas.
        """
        self.split_support = split_support
        self.ancilla_offset = ancilla_offset

    @restrict_to_qubits
    def get_circuit(
        self,
        code: codes.QuditCode,
        qubit_ids: QubitIDs | None = None,
    ) -> tuple[stim.Circuit, ParityMeasurementRecord]:
        """Construct one round of Bell-pair syndrome extraction."""
        if getattr(code.field, "order", None) != 2:
            raise ValueError("BellPairParitySyndrome only supports qubit codes over GF(2).")

        if code.is_subsystem_code:
            raise ValueError(
                "BellPairParitySyndrome expects a stabilizer code, not a subsystem code."
            )

        num_data_qubits = len(code)
        num_checks = code.num_checks
        ancilla_offset = code.dimension if self.ancilla_offset is None else self.ancilla_offset
        num_required_ancillas = ancilla_offset + num_checks

        if qubit_ids is None:
            qubit_ids = QubitIDs.from_code(code, num_ancillas=num_required_ancillas)
        else:
            qubit_ids = QubitIDs.validated(qubit_ids, code)
            qubit_ids.add_ancillas(num_required_ancillas - len(qubit_ids.ancilla))

        stabilizer_matrix = np.asarray(code.matrix, dtype=np.uint8) % 2
        expected_shape = (num_checks, 2 * num_data_qubits)

        if stabilizer_matrix.shape != expected_shape:
            raise ValueError(
                "Expected the stabilizer matrix to have shape "
                f"{expected_shape}, got {stabilizer_matrix.shape}."
            )

        circuit = stim.Circuit()
        record: dict[int, tuple[int, int]] = {}
        num_measurements = 0

        for check_index, row in enumerate(stabilizer_matrix):
            check_id = qubit_ids.check[check_index]
            bell_right = qubit_ids.ancilla[ancilla_offset + check_index]

            factors = _row_to_pauli_factors(row, num_data_qubits)
            left_factors, right_factors = _validate_factor_split(
                factors,
                *self.split_support(factors, check_index),
            )

            m_left, m_right = _append_bell_pair_check_measurement(
                circuit,
                qubit_ids,
                check_id=check_id,
                bell_right=bell_right,
                left_factors=left_factors,
                right_factors=right_factors,
                measurement_index=num_measurements,
            )
            num_measurements += 2

            record[check_id] = (m_left, m_right)

        return circuit, ParityMeasurementRecord(record, num_events=num_measurements)
