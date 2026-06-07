"""L=1 Webster gadget construction (see math.md §1, spec §2).

Three explicit named steps that map 1:1 to the paper:
    _step1_restriction  — math.md §1.1
    _step2_gauge_fix    — math.md §1.2
    _step3_assemble     — math.md §1.4
"""

from __future__ import annotations

import dataclasses

import galois
import numpy as np

from qldpc.codes.common import CSSCode

GF2 = galois.GF(2)


@dataclasses.dataclass(frozen=True, eq=False)
class GadgetLayout:
    code: CSSCode
    x: np.ndarray
    V0: tuple[int, ...]
    C0: tuple[int, ...]
    F: np.ndarray
    G: np.ndarray
    HX_merged: np.ndarray
    HZ_merged: np.ndarray
    kappa_qubits: tuple[int, ...]
