from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence

import numpy as np
import stim

from qldpc import codes, circuits
from qldpc.objects import Pauli

from ..bookkeeping import ParityMeasurementRecord, QubitIDs
from .syndrome_measurement import SyndromeMeasurementStrategy
# from __future__ import annotations

# from collections.abc import Callable, Sequence

# import numpy as np
# import stim

# from qldpc import circuits, codes
# from qldpc.objects import Pauli


SupportSplit = Callable[
    [list[int], int, Pauli],
    tuple[Sequence[int], Sequence[int]],
]


def balanced_split(
    support_cols: list[int],
    check_index: int,
    basis: Pauli,
) -> tuple[list[int], list[int]]:
    midpoint = len(support_cols) // 2
    return support_cols[:midpoint], support_cols[midpoint:]


class BellPairParitySyndrome(SyndromeMeasurementStrategy):
    """CSS syndrome extraction where each check syndrome is m_left XOR m_right."""

    def __init__(
        self,
        *,
        split_support: SupportSplit = balanced_split,
    ) -> None:
        self.split_support = split_support

    def get_circuit(
        self,
        code: codes.QuditCode,
        qubit_ids: circuits.QubitIDs | None = None,
    ) -> tuple[stim.Circuit, circuits.ParityMeasurementRecord]:
        if not isinstance(code, codes.CSSCode):
            raise ValueError("BellPairParitySyndrome currently supports CSS codes only.")

        qubit_ids = qubit_ids or circuits.QubitIDs.from_code(
            code,
            num_ancillas=code.num_checks,
        )

        if len(qubit_ids.ancilla) < code.num_checks:
            raise ValueError(
                f"Need at least {code.num_checks} ancilla qubits, one extra Bell half per check. "
                f"Got {len(qubit_ids.ancilla)}."
            )

        circuit = stim.Circuit()
        record: dict[int, tuple[int, int]] = {}
        num_measurements = 0

        check_specs: list[tuple[Pauli, np.ndarray, int]] = []

        # qLDPC orders CSS check qubits as X checks first, then Z checks.
        for local_row, check_id in enumerate(qubit_ids.checks_x):
            row = np.asarray(code.matrix_x[local_row], dtype=np.uint8) % 2
            check_specs.append((Pauli.X, row, check_id))

        for local_row, check_id in enumerate(qubit_ids.checks_z):
            row = np.asarray(code.matrix_z[local_row], dtype=np.uint8) % 2
            check_specs.append((Pauli.Z, row, check_id))

        for global_check_index, (basis, row, check_id) in enumerate(check_specs):
            bell_left = check_id
            bell_right = qubit_ids.ancilla[global_check_index]

            support_cols = [
                col
                for col, value in enumerate(row)
                if int(value) % 2
            ]

            left_cols, right_cols = self.split_support(
                support_cols,
                global_check_index,
                basis,
            )

            if set(left_cols) | set(right_cols) != set(support_cols):
                raise ValueError(
                    "split_support must cover the stabilizer support exactly."
                )

            if set(left_cols) & set(right_cols):
                raise ValueError(
                    "split_support must not put the same data qubit on both Bell halves."
                )

            # Prepare |Phi+> on the two Bell halves.
            circuit.append("RZ", [bell_left, bell_right])
            circuit.append("H", [bell_left])
            circuit.append("CX", [bell_left, bell_right])
            circuit.append("TICK")

            if basis is Pauli.Z:
                # Z-check:
                # CNOT data -> Bell half, then measure Bell halves in Z.
                targets: list[int] = []

                for col in left_cols:
                    targets += [qubit_ids.data[col], bell_left]

                for col in right_cols:
                    targets += [qubit_ids.data[col], bell_right]

                if targets:
                    circuit.append("CX", targets)

                circuit.append("TICK")

                m_left = num_measurements
                m_right = num_measurements + 1
                circuit.append("MZ", [bell_left, bell_right])
                num_measurements += 2

            elif basis is Pauli.X:
                # X-check:
                # CNOT Bell half -> data, then measure Bell halves in X.
                targets = []

                for col in left_cols:
                    targets += [bell_left, qubit_ids.data[col]]

                for col in right_cols:
                    targets += [bell_right, qubit_ids.data[col]]

                if targets:
                    circuit.append("CX", targets)

                circuit.append("TICK")

                m_left = num_measurements
                m_right = num_measurements + 1
                circuit.append("MX", [bell_left, bell_right])
                num_measurements += 2

            else:
                raise NotImplementedError(basis)

            # The effective syndrome bit is m_left XOR m_right.
            record[check_id] = (m_left, m_right)
            circuit.append("TICK")

        return circuit, circuits.ParityMeasurementRecord(
            record,
            num_events=num_measurements,
        )
# from __future__ import annotations

# import itertools
# from collections.abc import Callable, Sequence

# import numpy as np
# import stim

# from qldpc import codes
# from qldpc.objects import Pauli

# from ..bookkeeping import ParityMeasurementRecord, QubitIDs
# from .syndrome_measurement import SyndromeMeasurementStrategy


# PauliFactor = tuple[int, str]
# SupportSplit = Callable[
#     [tuple[PauliFactor, ...], int],
#     tuple[Sequence[PauliFactor], Sequence[PauliFactor]],
# ]


# def balanced_split(
#     factors: tuple[PauliFactor, ...],
#     check_index: int,
# ) -> tuple[tuple[PauliFactor, ...], tuple[PauliFactor, ...]]:
#     """Split a stabilizer support roughly in half."""
#     midpoint = len(factors) // 2
#     return factors[:midpoint], factors[midpoint:]


# def _row_to_pauli_factors(row: np.ndarray, num_data_qubits: int) -> tuple[PauliFactor, ...]:
#     """Convert one binary symplectic row [X | Z] into Pauli factors.

#     Returns:
#         A tuple of (data_column, pauli_char), where pauli_char is "X", "Y", or "Z".
#     """
#     row = np.asarray(row, dtype=np.uint8) % 2

#     x_support = row[:num_data_qubits]
#     z_support = row[num_data_qubits:]

#     factors: list[PauliFactor] = []

#     for col, (x_value, z_value) in enumerate(zip(x_support, z_support)):
#         x = int(x_value)
#         z = int(z_value)

#         if x == 0 and z == 0:
#             continue

#         if x == 1 and z == 0:
#             pauli = "X"
#         elif x == 0 and z == 1:
#             pauli = "Z"
#         elif x == 1 and z == 1:
#             pauli = "Y"
#         else:  # pragma: no cover; entries are binary after mod 2
#             raise ValueError(f"Invalid symplectic entry {(x, z)}")

#         factors.append((col, pauli))

#     return tuple(factors)


# def _validate_factor_split(
#     factors: tuple[PauliFactor, ...],
#     left_factors: Sequence[PauliFactor],
#     right_factors: Sequence[PauliFactor],
# ) -> tuple[tuple[PauliFactor, ...], tuple[PauliFactor, ...]]:
#     """Validate that left/right factors partition the stabilizer factors."""
#     left = tuple(left_factors)
#     right = tuple(right_factors)

#     expected = set(factors)
#     actual = set(left) | set(right)

#     if actual != expected:
#         raise ValueError(
#             "split_support must cover exactly the stabilizer support. "
#             f"Expected {sorted(expected)}, got {sorted(actual)}."
#         )

#     overlap = set(left) & set(right)
#     if overlap:
#         raise ValueError(
#             "split_support must not put the same Pauli factor on both Bell halves. "
#             f"Overlap: {sorted(overlap)}."
#         )

#     if len(left) + len(right) != len(factors):
#         raise ValueError(
#             "split_support must not duplicate Pauli factors. "
#             f"left={left}, right={right}, factors={factors}"
#         )

#     return left, right


# class BellPairParitySyndrome(SyndromeMeasurementStrategy):
#     """Syndrome extraction where each check syndrome is an XOR of two Bell-half readouts.

#     For each stabilizer generator S, represented as a general Pauli string:

#         1. Prepare |Phi+> on two Bell halves a,b.
#         2. Split the support of S into left and right pieces.
#         3. For each left Pauli factor P_j, apply controlled-P_j from a to data qubit j.
#         4. For each right Pauli factor P_j, apply controlled-P_j from b to data qubit j.
#         5. Measure a and b in X.
#         6. The effective syndrome is m_a XOR m_b.

#     This supports arbitrary qubit stabilizer checks containing X, Y, and Z factors.
#     """

#     def __init__(
#         self,
#         *,
#         split_support: SupportSplit = balanced_split,
#         ancilla_offset: int | None = None,
#     ) -> None:
#         """Initialize the Bell-pair syndrome strategy.

#         Args:
#             split_support:
#                 Function that partitions a stabilizer's Pauli factors between the two Bell halves.
#             ancilla_offset:
#                 Offset into qubit_ids.ancilla where the extra Bell halves begin.

#                 If None, defaults to code.dimension. This is the safest default for basis=None
#                 memory experiments, because qLDPC uses the first code.dimension ancillas for
#                 logical Bell-pair preparation in the combined-basis memory experiment.

#                 For fixed-basis CSS memory experiments, you may set ancilla_offset=0.
#         """
#         self.split_support = split_support
#         self.ancilla_offset = ancilla_offset

#     def get_circuit(
#         self,
#         code: codes.QuditCode,
#         qubit_ids: QubitIDs | None = None,
#     ) -> tuple[stim.Circuit, ParityMeasurementRecord]:
#         """Construct one round of Bell-pair syndrome extraction."""
#         if getattr(code.field, "order", None) != 2:
#             raise ValueError("BellPairParitySyndrome only supports qubit codes over GF(2).")

#         if code.is_subsystem_code:
#             raise ValueError("BellPairParitySyndrome expects a stabilizer code, not a subsystem code.")

#         qubit_ids = qubit_ids or QubitIDs.from_code(code)

#         num_data_qubits = len(code)
#         num_checks = code.num_checks

#         if len(qubit_ids.data) != num_data_qubits:
#             raise ValueError(
#                 f"Expected {num_data_qubits} data qubits, got {len(qubit_ids.data)}."
#             )

#         if len(qubit_ids.check) != num_checks:
#             raise ValueError(f"Expected {num_checks} check qubits, got {len(qubit_ids.check)}.")

#         ancilla_offset = code.dimension if self.ancilla_offset is None else self.ancilla_offset
#         num_required_ancillas = ancilla_offset + num_checks

#         if len(qubit_ids.ancilla) < num_required_ancillas:
#             raise ValueError(
#                 "Not enough ancillas for BellPairParitySyndrome. "
#                 f"Need at least {num_required_ancillas}: "
#                 f"{ancilla_offset} reserved ancillas plus {num_checks} Bell-pair ancillas. "
#                 f"Got {len(qubit_ids.ancilla)}."
#             )

#         stabilizer_matrix = np.asarray(code.matrix, dtype=np.uint8) % 2

#         if stabilizer_matrix.shape != (num_checks, 2 * num_data_qubits):
#             raise ValueError(
#                 "Expected the stabilizer matrix to have shape "
#                 f"{(num_checks, 2 * num_data_qubits)}, got {stabilizer_matrix.shape}."
#             )

#         circuit = stim.Circuit()
#         record: dict[int, tuple[int, int]] = {}
#         num_measurements = 0

#         for check_index, row in enumerate(stabilizer_matrix):
#             check_id = qubit_ids.check[check_index]

#             bell_left = check_id
#             bell_right = qubit_ids.ancilla[ancilla_offset + check_index]

#             factors = _row_to_pauli_factors(row, num_data_qubits)
#             left_factors, right_factors = _validate_factor_split(
#                 factors,
#                 *self.split_support(factors, check_index),
#             )

#             # Prepare |Phi+> = (|00> + |11>) / sqrt(2) on the two Bell halves.
#             circuit.append("R", [bell_left, bell_right])
#             circuit.append("H", [bell_left])
#             circuit.append("CX", [bell_left, bell_right])
#             circuit.append("TICK")

#             # Apply controlled-Pauli interactions. One interaction on each Bell half can be
#             # done in the same tick when both sides still have work to do.
#             for left_factor, right_factor in itertools.zip_longest(left_factors, right_factors):
#                 if left_factor is not None:
#                     data_col, pauli = left_factor
#                     circuit.append(f"C{pauli}", [bell_left, qubit_ids.data[data_col]])

#                 if right_factor is not None:
#                     data_col, pauli = right_factor
#                     circuit.append(f"C{pauli}", [bell_right, qubit_ids.data[data_col]])

#                 circuit.append("TICK")

#             # The check syndrome is the XOR/parity of these two X-basis measurements.
#             m_left = num_measurements
#             m_right = num_measurements + 1

#             circuit.append("MX", [bell_left, bell_right])
#             num_measurements += 2

#             record[check_id] = (m_left, m_right)
#             circuit.append("TICK")

#         return circuit, ParityMeasurementRecord(record, num_events=num_measurements)
