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


def _step2_gauge_fix(F: np.ndarray) -> np.ndarray:
    """math.md §1.2 — G whose rows form a canonical basis of ker(F.T) over GF(2).

    Uses galois ``left_null_space`` (row-reduced) so the basis is deterministic.
    """
    if F.size == 0:
        return np.zeros((0, F.shape[0]), dtype=np.uint8)
    G = GF2(F.astype(np.int_).tolist()).left_null_space()
    return np.asarray(G).astype(np.uint8)


def _step3_assemble(
    code: CSSCode,
    V0: tuple[int, ...],
    C0: tuple[int, ...],
    F: np.ndarray,
    G: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """math.md §1.4 — block assembly of HX_merged, HZ_merged."""
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    n = code.num_qudits
    mX, mZ = HX.shape[0], HZ.shape[0]
    nV, nC = len(V0), len(C0)
    r = G.shape[0]

    # E_{V0}^T : (nV × n), single 1 per row at position V0[i]
    E_V0_T = np.zeros((nV, n), dtype=np.uint8)
    for i, v in enumerate(V0):
        E_V0_T[i, v] = 1

    # F^T (nV × nC)
    F_T = F.T.astype(np.uint8)

    # \tilde F : (mZ × nC) selection matrix — F_tilde[j, k] = 1 iff j == C_0[k]
    # (math.md §1.4). Previous form F_tilde[j] = F[k] only worked when nV == nC.
    F_tilde = np.zeros((mZ, nC), dtype=np.uint8)
    for k, j in enumerate(C0):
        F_tilde[j, k] = 1

    HX_merged = np.block([
        [HX, np.zeros((mX, nC), dtype=np.uint8)],
        [E_V0_T, F_T],
    ]).astype(np.uint8)

    HZ_merged = np.block([
        [HZ, F_tilde],
        [np.zeros((r, n), dtype=np.uint8), G.astype(np.uint8)],
    ]).astype(np.uint8)

    return HX_merged, HZ_merged


def build_gadget(code: CSSCode, x: np.ndarray) -> GadgetLayout:
    """Webster L=1 gadget = steps 1+2+3 composed. Deterministic in (code, x)."""
    x = np.asarray(x).astype(np.uint8)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    if ((HZ @ x) % 2).any():
        raise ValueError("x is not a logical-X support (H_Z @ x != 0).")

    V0, C0, F = _step1_restriction(code, x)
    G = _step2_gauge_fix(F)
    HX_m, HZ_m = _step3_assemble(code, V0, C0, F, G)
    kappa_qubits = tuple(range(code.num_qudits, code.num_qudits + len(C0)))
    return GadgetLayout(
        code=code, x=x, V0=V0, C0=C0, F=F, G=G,
        HX_merged=HX_m, HZ_merged=HZ_m, kappa_qubits=kappa_qubits,
    )
