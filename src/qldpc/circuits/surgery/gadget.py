"""L=1 gadget construction (Webster, Smith, Cohen arXiv:2511.15989 §II.A).

Three explicit named steps that map 1:1 to the paper:
    _step1_restriction  — Webster §II.A step 1 (restriction)
    _step2_gauge_fix    — Webster §II.A step 2 (gauge fix)
    _step3_assemble     — Webster §II.A step 3 (block assembly)
"""

from __future__ import annotations

import dataclasses

import galois
import numpy as np

from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli, PauliXZ

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
    basis: PauliXZ


def _step1_restriction(
    code: CSSCode, x: np.ndarray, *, basis: PauliXZ = Pauli.X,
) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray]:
    """Webster §II.A step 1 — V_0 = supp(x); C_0 = checks touching V_0; F = H_complement[C_0, V_0].

    For basis=Pauli.X: F = H_Z[C_0, V_0] (the complementary basis to the measured logical).
    For basis=Pauli.Z: F = H_X[C_0, V_0].
    """
    x = np.asarray(x).astype(np.uint8)
    if x.shape != (code.num_qudits,):
        raise ValueError(f"x has shape {x.shape}, expected ({code.num_qudits},)")
    V0 = tuple(int(i) for i in np.where(x)[0])
    # Use the COMPLEMENTARY check matrix to the measured logical type
    H_complement = (
        np.asarray(code.matrix_z).astype(np.uint8)
        if basis is Pauli.X
        else np.asarray(code.matrix_x).astype(np.uint8)
    )
    C0 = tuple(
        int(j) for j in range(H_complement.shape[0]) if H_complement[j, list(V0)].any()
    )
    F = (
        H_complement[np.ix_(C0, V0)]
        if C0 and V0
        else np.zeros((len(C0), len(V0)), dtype=np.uint8)
    )
    return V0, C0, F.astype(np.uint8)


def _step2_gauge_fix(F: np.ndarray) -> np.ndarray:
    """Webster §II.A step 2 — G whose rows form a canonical basis of ker(F.T) over GF(2).

    Uses galois ``left_null_space`` (row-reduced) so the basis is deterministic.
    """
    if F.size == 0:
        return np.zeros((0, F.shape[0]), dtype=np.uint8)
    G = GF2(F.astype(np.int_).tolist()).left_null_space()
    return np.asarray(G).astype(np.uint8)


def _assemble_HX_L1(
    HX_data: np.ndarray,
    v0_indices: np.ndarray,
    F: np.ndarray,
) -> np.ndarray:
    """L=1 HX-side block assembly: [[HX_data, 0], [E_V0, F^T]] over GF(2).

    Used by _step3_assemble (initial gadget assembly) and
    build_gadget_augmented (post-boost rebuild). The Z-side
    assembly is NOT shared — the boost rebuild treats new κ' qubits as
    pure-gauge (no data-Z extension), unlike the initial assembly.

    Args:
        HX_data: original code's X-check matrix, shape (mX, n), uint8.
        v0_indices: indices of V_0 within the n data qubits, shape (|V_0|,).
        F: restriction matrix, shape (|C_0|, |V_0|), uint8.

    Returns:
        HX_merged: shape (mX + |V_0|, n + |C_0|), uint8.
    """
    mX, n = HX_data.shape
    n_v0, n_c0 = int(F.shape[1]), int(F.shape[0])
    n_merged = n + n_c0
    top = np.hstack([HX_data, np.zeros((mX, n_c0), dtype=np.uint8)]).astype(np.uint8)
    bot = np.zeros((n_v0, n_merged), dtype=np.uint8)
    bot[np.arange(n_v0), np.asarray(v0_indices)] = 1
    bot[:, n:] = F.T
    return np.vstack([top, bot]).astype(np.uint8)


def _step3_assemble(
    code: CSSCode,
    V0: tuple[int, ...],
    C0: tuple[int, ...],
    F: np.ndarray,
    G: np.ndarray,
    *,
    basis: PauliXZ = Pauli.X,
) -> tuple[np.ndarray, np.ndarray]:
    """Webster §II.A step 3 — block assembly of HX_merged, HZ_merged.

    basis=X (default): χ rows added to HX_merged, G to HZ_merged.
    basis=Z: χ rows added to HZ_merged, G to HX_merged (basis-symmetric dual).
    """
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    n = code.num_qudits
    mX, mZ = HX.shape[0], HZ.shape[0]
    nV, nC = len(V0), len(C0)
    r = G.shape[0]

    # F_tilde : (mZ_or_mX × nC) selection matrix — F_tilde[j, k] = 1 iff j == C_0[k]
    if basis is Pauli.X:
        F_tilde = np.zeros((mZ, nC), dtype=np.uint8)
    else:
        F_tilde = np.zeros((mX, nC), dtype=np.uint8)
    for k, j in enumerate(C0):
        if j < 0:
            continue        # sentinel for extra-κ rows from build_gadget_augmented
        F_tilde[j, k] = 1

    v0_arr = np.asarray(V0, dtype=np.int_)

    if basis is Pauli.X:
        # χ rows extend HX_merged; G rows extend HZ_merged
        HX_merged = _assemble_HX_L1(HX, v0_arr, F)
        HZ_merged = np.block([
            [HZ, F_tilde],
            [np.zeros((r, n), dtype=np.uint8), G.astype(np.uint8)],
        ]).astype(np.uint8)
    else:
        # basis=Z (symmetric dual): χ rows extend HZ_merged; G rows extend HX_merged
        HZ_merged = _assemble_HX_L1(HZ, v0_arr, F)
        HX_merged = np.block([
            [HX, F_tilde],
            [np.zeros((r, n), dtype=np.uint8), G.astype(np.uint8)],
        ]).astype(np.uint8)

    return HX_merged, HZ_merged


def build_gadget(
    code: CSSCode, x: np.ndarray, *, basis: PauliXZ,
) -> GadgetLayout:
    """Webster L=1 gadget = steps 1+2+3 composed. Deterministic in (code, x, basis).

    basis=Pauli.X: measures a logical X (PPM of X̄). Validates H_Z @ x == 0.
    basis=Pauli.Z: measures a logical Z (PPM of Z̄). Validates H_X @ x == 0.
    """
    x = np.asarray(x).astype(np.uint8)
    if basis is Pauli.X:
        H_check = np.asarray(code.matrix_z).astype(np.uint8)
        if ((H_check @ x) % 2).any():
            raise ValueError("x is not a logical-X support (H_Z @ x != 0).")
    elif basis is Pauli.Z:
        H_check = np.asarray(code.matrix_x).astype(np.uint8)
        if ((H_check @ x) % 2).any():
            raise ValueError("x is not a logical-Z support (H_X @ x != 0).")
    else:
        raise ValueError(f"basis must be Pauli.X or Pauli.Z, got {basis!r}")

    V0, C0, F = _step1_restriction(code, x, basis=basis)
    G = _step2_gauge_fix(F)
    HX_m, HZ_m = _step3_assemble(code, V0, C0, F, G, basis=basis)
    kappa_qubits = tuple(range(code.num_qudits, code.num_qudits + len(C0)))
    return GadgetLayout(
        code=code, x=x, V0=V0, C0=C0, F=F, G=G,
        HX_merged=HX_m, HZ_merged=HZ_m, kappa_qubits=kappa_qubits,
        basis=basis,
    )


def build_gadget_augmented(
    code: CSSCode,
    x: np.ndarray,
    F_extra: np.ndarray,
    *,
    basis: PauliXZ,
) -> GadgetLayout:
    """Rebuild a GadgetLayout with F augmented by extra weight-2 rows.

    Each row of ``F_extra`` has weight 2 and corresponds to a new κ qubit not
    backed by any original Z-check (basis=X) or X-check (basis=Z). The function:

    1. Stacks F_aug = [F; F_extra].
    2. Recomputes G_aug = ker(F_aug^T) via _step2_gauge_fix.
    3. Calls _step3_assemble with the original V_0 / C_0 plus the new κ rows.
       The extra columns of tilde_F are all zero (no original check sits on the
       new κ qubits).

    The returned ``GadgetLayout.C0`` and ``kappa_qubits`` are extended to cover
    the new κ qubits; the new κ indices come after the original ones.
    """
    x = np.asarray(x).astype(np.uint8)
    V0, C0, F = _step1_restriction(code, x, basis=basis)
    F_extra = np.asarray(F_extra).astype(np.uint8)
    if F_extra.shape[1] != len(V0):
        raise ValueError(
            f"F_extra has {F_extra.shape[1]} columns; expected {len(V0)} (= |V_0|)"
        )
    if F_extra.size and not np.all(F_extra.sum(axis=1) == 2):
        bad = np.flatnonzero(F_extra.sum(axis=1) != 2).tolist()
        raise ValueError(f"F_extra rows {bad} have weight != 2; required weight 2.")

    F_aug = np.vstack([F, F_extra]).astype(np.uint8)
    G_aug = _step2_gauge_fix(F_aug)

    # _step3_assemble computes tilde_F by indexing into C_0; we need an extended
    # C_0_aug that has the new rows as sentinels (their tilde_F columns must be 0).
    # Trick: pass C_0_aug = C_0 + (-1, -1, ...) sentinels which fall outside [0, mZ),
    # so the tilde_F loop sets nothing for those positions.
    n_extra = F_extra.shape[0]
    C0_aug = tuple(C0) + tuple([-1] * n_extra)
    HX_aug, HZ_aug = _step3_assemble(
        code, V0, C0_aug, F_aug, G_aug, basis=basis,
    )
    kappa_qubits_aug = tuple(range(code.num_qudits, code.num_qudits + len(C0_aug)))
    return GadgetLayout(
        code=code, x=x, V0=V0, C0=C0_aug, F=F_aug, G=G_aug,
        HX_merged=HX_aug, HZ_merged=HZ_aug, kappa_qubits=kappa_qubits_aug,
        basis=basis,
    )


