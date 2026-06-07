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


def _step1_restriction(
    code: CSSCode, x: np.ndarray
) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray]:
    """math.md §1.1 — V_0 = supp(x); C_0 = Z-checks touching V_0; F = H_Z[C_0, V_0]."""
    x = np.asarray(x).astype(np.uint8)
    if x.shape != (code.num_qudits,):
        raise ValueError(f"x has shape {x.shape}, expected ({code.num_qudits},)")
    V0 = tuple(int(i) for i in np.where(x)[0])
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    C0 = tuple(
        int(j) for j in range(HZ.shape[0]) if HZ[j, list(V0)].any()
    )
    F = HZ[np.ix_(C0, V0)] if C0 and V0 else np.zeros((len(C0), len(V0)), dtype=np.uint8)
    return V0, C0, F.astype(np.uint8)
