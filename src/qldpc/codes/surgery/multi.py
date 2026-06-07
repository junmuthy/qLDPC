"""Multi-PPM (Pauli product measurement) surgery construction.

Measures t commuting Pauli logicals simultaneously via a single Webster
gadget on V_0 = union of supports. Each logical's chi-sum subset becomes
an HX-row-span stabilizer, consuming one logical DOF per measurement.

Overlapping supports are handled via SetValuedPort (Ide §VII C, Cain
Processor mode).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import galois
import numpy as np
import numpy.typing as npt

from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli

from .layered import SurgeryLayout, build_layered_surgery_code
from .port import SetValuedPort


@dataclasses.dataclass(frozen=True, eq=False)
class MultiSurgeryLayout:
    """Layout for a multi-PPM Webster gadget.

    Attributes:
        base_layout: SurgeryLayout from the Webster gadget on V_0_union.
        logical_ops: tuple of original binary support vectors, length t.
        set_valued_port: SetValuedPort mapping qubit -> list of logical indices.
        chi_group_per_logical: tuple of length t; chi_group_per_logical[i] is
            the tuple of chi row indices in merged.matrix_x whose sum modulo 2
            equals logical_ops[i].
    """

    base_layout: SurgeryLayout
    logical_ops: tuple[np.ndarray, ...]
    set_valued_port: SetValuedPort
    chi_group_per_logical: tuple[tuple[int, ...], ...]
