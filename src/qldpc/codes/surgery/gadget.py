"""L=1 Webster gadget construction (see math.md §1, spec §2).

Three explicit named steps that map 1:1 to the paper:
    _step1_restriction  — math.md §1.1
    _step2_gauge_fix    — math.md §1.2
    _step3_assemble     — math.md §1.4
"""

from __future__ import annotations

import dataclasses
import json as _json
import pathlib as _pathlib

import galois
import numpy as np

from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli, PauliXZ

GF2 = galois.GF(2)

_WEBSTER_APP_A_PATH = _pathlib.Path(__file__).resolve().parents[4] / "examples" / "webster_app_a.json"


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
    basis: PauliXZ = Pauli.X


def _step1_restriction(
    code: CSSCode, x: np.ndarray, *, basis: PauliXZ = Pauli.X,
) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray]:
    """math.md §1.1 — V_0 = supp(x); C_0 = checks touching V_0; F = H_complement[C_0, V_0].

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
    *,
    basis: PauliXZ = Pauli.X,
) -> tuple[np.ndarray, np.ndarray]:
    """math.md §1.4 — block assembly of HX_merged, HZ_merged.

    basis=X (default): χ rows added to HX_merged, G to HZ_merged.
    basis=Z: χ rows added to HZ_merged, G to HX_merged (basis-symmetric dual).
    """
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    n = code.num_qudits
    mX, mZ = HX.shape[0], HZ.shape[0]
    nV, nC = len(V0), len(C0)
    r = G.shape[0]

    E_V0_T = np.zeros((nV, n), dtype=np.uint8)
    for i, v in enumerate(V0):
        E_V0_T[i, v] = 1
    F_T = F.T.astype(np.uint8)

    # F_tilde : (mZ_or_mX × nC) selection matrix — F_tilde[j, k] = 1 iff j == C_0[k]
    if basis is Pauli.X:
        F_tilde = np.zeros((mZ, nC), dtype=np.uint8)
    else:
        F_tilde = np.zeros((mX, nC), dtype=np.uint8)
    for k, j in enumerate(C0):
        F_tilde[j, k] = 1

    if basis is Pauli.X:
        # χ rows extend HX_merged; G rows extend HZ_merged
        HX_merged = np.block([
            [HX, np.zeros((mX, nC), dtype=np.uint8)],
            [E_V0_T, F_T],
        ]).astype(np.uint8)
        HZ_merged = np.block([
            [HZ, F_tilde],
            [np.zeros((r, n), dtype=np.uint8), G.astype(np.uint8)],
        ]).astype(np.uint8)
    else:
        # basis=Z: χ rows extend HZ_merged; G rows extend HX_merged (symmetric)
        HZ_merged = np.block([
            [HZ, np.zeros((mZ, nC), dtype=np.uint8)],
            [E_V0_T, F_T],
        ]).astype(np.uint8)
        HX_merged = np.block([
            [HX, F_tilde],
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
        basis=Pauli.X,
    )


def load_webster_seed_set(code_index: int) -> dict:
    """Load Webster (arXiv:2511.15989) Appendix A data for code index 0..3.

    The 4 codes are generalised bicycle codes with l in {31, 63, 127, 255},
    each having 4 seed operators (X_bar_1, Z_bar_1, X_bar_{k/2+1}, Z_bar_{k/2+1}).
    The data is read from ``examples/webster_app_a.json``.

    Returns:
        A dict matching the JSON schema.

    Raises:
        IndexError: if code_index is not in 0..3.
        FileNotFoundError: if the JSON fixture is missing.
    """
    if not 0 <= code_index <= 3:
        raise IndexError(f"code_index must be in 0..3, got {code_index}")
    with _WEBSTER_APP_A_PATH.open() as fh:
        data = _json.load(fh)
    return data["codes"][code_index]


def _build_generalised_bicycle_code(l: int, A_set: list[int], B_set: list[int]) -> CSSCode:
    """Build a generalised bicycle code from cyclic exponent sets A, B.

    Per Kovalev-Pryadko (arXiv:1212.6703) and Swaroop's reference
    implementation (https://github.com/eswaroop/adapters-LDPC-surgery,
    ext/bivariate_bicyclic.py): given subsets A, B of Z_l, let A(x) =
    sum(x^a for a in A_set) and B(x) = sum(x^b for b in B_set) as cyclic
    matrices in F_2[Z_l]. Then H_X = [A | B] and H_Z = [B^T | A^T] define
    the bicycle code on 2l data qubits.

    Args:
        l: cyclic group order.
        A_set, B_set: subsets of {0, 1, ..., l-1}.

    Returns:
        CSSCode on 2l data qubits with check matrices [A | B] and
        [B^T | A^T] over GF(2).
    """
    I_l = np.eye(l, dtype=np.int_)
    # cyclic shift matrix S such that S^k is left-shift by k (zero-indexed)
    S = np.roll(I_l, shift=-1, axis=0)
    A = np.zeros((l, l), dtype=np.int_)
    for a in A_set:
        A = (A + np.linalg.matrix_power(S, a)) % 2
    B = np.zeros((l, l), dtype=np.int_)
    for b in B_set:
        B = (B + np.linalg.matrix_power(S, b)) % 2

    H_X = np.hstack([A, B])
    H_Z = np.hstack([B.T, A.T])

    return CSSCode(H_X, H_Z, is_subsystem_code=False)
