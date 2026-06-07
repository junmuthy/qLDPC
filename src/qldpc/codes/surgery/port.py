"""Set-valued port function for multi-PPM and overlap-aware surgery.

A SetValuedPort maps each data qubit to the list of gadget (logical)
indices that include it in V_0. For disjoint supports, every list has
length 1. For overlap (Ide §VII C, Cain Processor with shared logicals),
shared qubits map to lists of length >= 2.

This module implements the set-valued port concept introduced in
Ide / Swaroop et al. arXiv:2410.03628 Appendix VIII (Theorem 11 / §VII C).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np


@dataclasses.dataclass(frozen=True)
class SetValuedPort:
    """Per-qubit list of gadget indices that include it in V_0.

    Attributes:
        qubit_to_gadgets: dict mapping data qubit index (int) -> sorted list
            of gadget indices that include it. Qubits not in any V_0 are
            omitted from the dict (queries return []).
    """

    qubit_to_gadgets: dict[int, list[int]]

    @classmethod
    def from_supports(cls, supports: Sequence[np.ndarray]) -> "SetValuedPort":
        """Build from a sequence of binary support vectors, one per gadget.

        Args:
            supports: t binary vectors of the same length n_data. supports[i][q]
                = 1 iff data qubit q is in V_0 of gadget i.

        Returns:
            SetValuedPort with qubit_to_gadgets populated only for qubits
            present in at least one support.
        """
        mapping: dict[int, list[int]] = {}
        for g_idx, supp in enumerate(supports):
            for q in np.flatnonzero(np.asarray(supp)).tolist():
                mapping.setdefault(int(q), []).append(g_idx)
        return cls(qubit_to_gadgets=mapping)

    def is_shared(self, qubit: int) -> bool:
        """True iff qubit is in V_0 of >= 2 gadgets."""
        return len(self.qubit_to_gadgets.get(qubit, [])) > 1

    def gadgets_for_qubit(self, qubit: int) -> list[int]:
        """Return the list of gadget indices that include qubit, or []."""
        return list(self.qubit_to_gadgets.get(qubit, []))

    def shared_qubits(self) -> list[int]:
        """Sorted list of qubits that appear in >= 2 gadget supports."""
        return sorted(q for q, gs in self.qubit_to_gadgets.items() if len(gs) > 1)
