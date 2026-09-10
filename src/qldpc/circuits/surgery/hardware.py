"""Identify adapter hardware from a merged binary CSS code and its circuit IDs."""

from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np

from qldpc.circuits.bookkeeping import QubitIDs
from qldpc.codes.common import CSSCode

AdapterNoiseTargets = Literal["full_surgery", "adapter_hardware"]


@dataclasses.dataclass(frozen=True)
class AdapterHardware:
    """Added data, checks touching that data, and auxiliary surgery ancillas.

    Check ancillas are selected by their merged support, including original block
    checks extended onto gadget data. Auxiliary ancillas allocated by the surgery
    strategy (Bell partners for selective Bell extraction) are also included.
    """

    data: tuple[int, ...]
    check: tuple[int, ...]
    ancilla: tuple[int, ...]

    @property
    def all_qubits(self) -> tuple[int, ...]:
        return self.data + self.check + self.ancilla

    @classmethod
    def from_code(
        cls, code: CSSCode, qubit_ids: QubitIDs, *, num_block_data_qubits: int
    ) -> AdapterHardware:
        """Original data occupy the leading columns; added data occupy the rest."""
        if not 0 <= num_block_data_qubits <= len(code):
            raise ValueError("num_block_data_qubits is outside the joint-code data range")
        if code.field.order != 2 or code.is_subsystem_code:
            raise ValueError("Adapter hardware requires a binary stabilizer CSS code")
        QubitIDs.validated(qubit_ids, code)
        touches_adapter = np.concatenate(
            [
                np.any(np.asarray(matrix)[:, num_block_data_qubits:] != 0, axis=1)
                for matrix in (code.matrix_x, code.matrix_z)
            ]
        )
        return cls(
            data=qubit_ids.data[num_block_data_qubits:],
            check=tuple(q for q, selected in zip(qubit_ids.check, touches_adapter) if selected),
            ancilla=qubit_ids.ancilla,
        )
