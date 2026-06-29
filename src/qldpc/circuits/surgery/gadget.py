"""L=1 gadget construction (Webster, Smith, Cohen arXiv:2511.15989 §II.A;
Cain et al. arXiv:2603.28627 §B.1).

Three explicit named steps that map 1:1 to the paper:
    _step1_restriction  — restriction: V₀=supp(x), C₀=checks touching V₀,
                          ∂_1 = π_{V₀} H_Z^T π_{C₀}^T (stored transposed as `incidence`)
    _step2_gauge_fix    — gauge fix: ∂_0 = ker(∂_1) over GF(2) (stored as `partial_0`)
    _step3_assemble     — block assembly of HX_merged, HZ_merged
"""

from __future__ import annotations

import dataclasses

import galois
import numpy as np

from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli, PauliXZ

GF2 = galois.GF(2)


def _projection(indices, N: int) -> np.ndarray:
    """π_S ∈ F₂^{|S|×N}: row i is the unit vector e_{indices[i]}.

    (π_S)_{i,j} = δ_{j, indices[i]}, so π_S M π_T^T = M[S, T] (numpy-style index).
    Entries outside [0, N) (e.g. the -1 sentinels build_gadget_augmented uses for
    boost-added Q' qubits with no backing check) give an all-zero row.
    """
    pi = np.zeros((len(indices), N), dtype=np.uint8)
    for i, s in enumerate(indices):
        if 0 <= s < N:
            pi[i, s] = 1
    return pi


def _restrict(
    H_complement: np.ndarray,
    x: np.ndarray,
) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray, np.ndarray]:
    """Unified single-gadget kernel — the useful quantities in one place
    (Webster, Smith, Cohen arXiv:2511.15989 §II.A; Cain et al. arXiv:2603.28627 §B.1).

    V₀ = support = supp(x); C₀ = data_checks = complementary-basis checks touching V₀;
    incidence = ∂_1^T = H_complement[C₀, V₀]  (|C₀|×|V₀|, edge×vertex);
    partial_0 = ∂_0 = ker(∂_1) = GF2(incidence).left_null_space() (row-reduced, deterministic).

    ``H_complement`` is the complementary check matrix to the measured logical type
    (H_Z when measuring X̄, H_X when measuring Z̄). This is the single primitive the
    joint-PPM and Y/mixed-PPM constructions also consume.
    """
    H_complement = np.asarray(H_complement).astype(np.uint8)
    x = np.asarray(x).astype(np.uint8)
    if x.shape != (H_complement.shape[1],):
        raise ValueError(f"x has shape {x.shape}, expected ({H_complement.shape[1]},)")
    support = tuple(int(i) for i in np.nonzero(x)[0])
    if support:
        V0 = np.array(support, dtype=np.int_)
        C0 = np.nonzero(H_complement[:, V0].any(axis=1))[0]
        incidence = H_complement[np.ix_(C0, V0)].astype(np.uint8)
    else:
        C0 = np.zeros(0, dtype=np.int_)
        incidence = np.zeros((0, 0), dtype=np.uint8)
    data_checks = tuple(int(j) for j in C0)
    if incidence.size:
        partial_0 = np.asarray(GF2(incidence).left_null_space()).astype(np.uint8)
    else:
        partial_0 = np.zeros((0, incidence.shape[0]), dtype=np.uint8)
    return support, data_checks, incidence, partial_0


def _x_merged(
    H_X: np.ndarray,
    H_Z: np.ndarray,
    x: np.ndarray,
    incidence_extra: np.ndarray | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form merged matrices for measuring X̄ (Webster, Smith, Cohen
    arXiv:2511.15989 §II.A; Cain et al. arXiv:2603.28627 §B.1; Ide, Gowda, Nadkarni,
    Dauphinais arXiv:2410.02753 Eq.(62)):

        H̃_X = [[H_X,   0 ],     H̃_Z = [[H_Z,   f_0 ],
               [f_1^T, ∂_1]]            [ 0,   ∂_0 ]]

    f_1^T = π_{V₀} (indicator on support data qubits); ∂_1 = incidence.T (vertex×edge,
    entered untransposed); f_0 = π_{C₀}^T (extends complementary checks onto Q'),
    with all-zero columns for boost-added κ that back no original check; ∂_0 = ker(∂_1).

    ``incidence_extra`` (weight-2 rows, |·|×|V₀|) stacks new κ onto ∂_1^T for the
    boost / joint path; ∂_0 is re-derived on the stacked incidence.
    """
    H_X = np.asarray(H_X).astype(np.uint8)
    H_Z = np.asarray(H_Z).astype(np.uint8)
    support, data_checks, incidence, partial_0 = _restrict(H_Z, x)
    n = H_X.shape[1]
    n_C0 = len(data_checks)
    if incidence_extra is not None:
        incidence_extra = np.asarray(incidence_extra).astype(np.uint8)
        n_extra = incidence_extra.shape[0]
        incidence = np.vstack([incidence, incidence_extra]).astype(np.uint8)
        if incidence.size:
            partial_0 = np.asarray(GF2(incidence).left_null_space()).astype(np.uint8)
        else:
            partial_0 = np.zeros((0, incidence.shape[0]), dtype=np.uint8)
        data_checks = tuple(data_checks) + tuple([-1] * n_extra)
    n_V0 = len(support)
    n_cols = incidence.shape[0]  # |C₀| + n_extra
    d1 = incidence.T.astype(np.uint8)  # ∂_1, |V₀|×n_cols
    f1T = np.zeros((n_V0, n), dtype=np.uint8)
    if n_V0:
        f1T[np.arange(n_V0), np.array(support, dtype=np.int_)] = 1
    f0 = np.zeros((H_Z.shape[0], n_cols), dtype=np.uint8)
    if n_C0:
        f0[np.array(data_checks[:n_C0], dtype=np.int_), np.arange(n_C0)] = 1
    HX = np.block(
        [[H_X, np.zeros((H_X.shape[0], n_cols), dtype=np.uint8)], [f1T, d1]]
    ).astype(np.uint8)
    HZ = np.block(
        [[H_Z, f0], [np.zeros((partial_0.shape[0], n), dtype=np.uint8), partial_0]]
    ).astype(np.uint8)
    return support, data_checks, incidence, partial_0, HX, HZ


@dataclasses.dataclass(frozen=True, eq=False)
class GadgetLayout:
    """Frozen result of a single L=1 gadget construction.

    Symbol mapping (Webster, Smith, Cohen arXiv:2511.15989 §II.A;
    Cain et al. arXiv:2603.28627 §B.1):
        support       = V₀  (qubit indices in supp(x))
        data_checks   = C₀  (complementary-basis check indices touching V₀;
                             -1 sentinels indicate boost-added Q' with no backing check)
        incidence     = ∂_1^T  (|C₀|×|V₀| edge×vertex matrix; ∂_1 = incidence.T)
        partial_0     = ∂_0    (basis of ker(∂_1) over GF(2))
        Q_prime       = Q' (κ qubits, indexed after the n data qubits)
    """

    code: CSSCode
    x: np.ndarray
    support: tuple[int, ...]
    data_checks: tuple[int, ...]
    incidence: np.ndarray
    partial_0: np.ndarray
    HX_merged: np.ndarray
    HZ_merged: np.ndarray
    Q_prime: tuple[int, ...]  # Q' ancilla qubit IDs
    basis: PauliXZ


def _step1_restriction(
    code: CSSCode,
    x: np.ndarray,
    *,
    basis: PauliXZ = Pauli.X,
) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray]:
    """Single-gadget restriction (Webster, Smith, Cohen arXiv:2511.15989 §II.A;
    Cain et al. arXiv:2603.28627 §B.1).

    V₀ = support = supp(x); C₀ = data_checks = complementary-basis checks touching V₀;
    ∂_1 = π_{V₀} H_Z^T π_{C₀}^T (stored transposed as `incidence`, shape |C₀|×|V₀|).

    basis=Pauli.X: complementary matrix is H_Z (measuring X̄, restricting via Z-checks).
    basis=Pauli.Z: complementary matrix is H_X (measuring Z̄, restricting via X-checks).
    """
    x = np.asarray(x).astype(np.uint8)
    if x.shape != (code.num_qudits,):
        raise ValueError(f"x has shape {x.shape}, expected ({code.num_qudits},)")
    support = tuple(int(i) for i in np.where(x)[0])
    # Use the COMPLEMENTARY check matrix to the measured logical type
    H_complement = (
        np.asarray(code.matrix_z).astype(np.uint8)
        if basis is Pauli.X
        else np.asarray(code.matrix_x).astype(np.uint8)
    )
    data_checks = tuple(
        int(j) for j in range(H_complement.shape[0]) if H_complement[j, list(support)].any()
    )
    # ∂_1 = π_{V₀} H_Z^T π_{C₀}^T  (vertex×edge incidence; Webster, Smith, Cohen
    # arXiv:2511.15989 §II.A; Cain et al. arXiv:2603.28627 §B.1; arXiv:2410.02753 Eq.(62)).
    # The stored `incidence` is its transpose ∂_1^T — the |C₀|×|V₀| edge×vertex form
    # (§4 y_gadget.py recovers ∂_1^x = incidence.T).
    n = code.num_qudits
    pi_V0 = _projection(support, n)                          # π_{V₀} = f_1^T ∈ F₂^{|V₀|×n}
    pi_C0 = _projection(data_checks, H_complement.shape[0])  # π_{C₀} ∈ F₂^{|C₀|×m_comp}
    partial_1 = (pi_V0 @ H_complement.T @ pi_C0.T) % 2       # ∂_1, |V₀|×|C₀|
    incidence = partial_1.T.astype(np.uint8)                # ∂_1^T, |C₀|×|V₀|
    return support, data_checks, incidence


def _step2_gauge_fix(incidence: np.ndarray) -> np.ndarray:
    """Gauge fix (Webster, Smith, Cohen arXiv:2511.15989 §II.A; Cain et al. arXiv:2603.28627 §B.1).

    ∂_0 = canonical row basis of ker(∂_1) = left_null_space(incidence) = ker(incidence^T) over GF(2).
    `incidence` = ∂_1^T (|C₀|×|V₀|); returns ∂_0 of shape (r, |C₀|) where r = |C₀| − rank(∂_1).

    Uses galois ``left_null_space`` (row-reduced) so the basis is deterministic.
    """
    if incidence.size == 0:
        return np.zeros((0, incidence.shape[0]), dtype=np.uint8)
    gauge = GF2(incidence.astype(np.int_).tolist()).left_null_space()
    return np.asarray(gauge).astype(np.uint8)


def _assemble_HX_L1(
    HX_data: np.ndarray,
    support_indices: np.ndarray,
    incidence: np.ndarray,
) -> np.ndarray:
    """L=1 X-side block assembly: [[HX_data, 0], [S_X']] over GF(2).

    S_X' = [f_1^T | ∂_1] where f_1^T = π_{V₀} (indicator on data qubits) and
    ∂_1 = incidence.T (Webster, Smith, Cohen arXiv:2511.15989 §II.A;
    Cain et al. arXiv:2603.28627 §B.1).

    Called by _step3_assemble (initial gadget assembly) and build_gadget_augmented
    (post-boost rebuild, with augmented incidence). The Z-side assembly is NOT shared —
    the boost rebuild treats new κ' qubits as pure-gauge (no data-Z extension).

    Args:
        HX_data: original code's X-check matrix, shape (mX, n), uint8.
        support_indices: indices of V₀ within the n data qubits, shape (|V₀|,).
        incidence: ∂_1^T, shape (|C₀|, |V₀|), uint8.

    Returns:
        HX_merged: shape (mX + |V₀|, n + |C₀|), uint8.
    """
    mX, n = HX_data.shape
    n_v0, n_c0 = int(incidence.shape[1]), int(incidence.shape[0])
    top = np.hstack([HX_data, np.zeros((mX, n_c0), dtype=np.uint8)]).astype(np.uint8)
    # S_X' rows = [f_1^T | ∂_1] : f_1^T = π_{V₀} on data, ∂_1 = incidence.T on Q'.
    f_1_T = np.zeros((n_v0, n), dtype=np.uint8)
    f_1_T[np.arange(n_v0), np.asarray(support_indices)] = 1
    partial_1 = incidence.T.astype(np.uint8)
    S_X_prime = np.hstack([f_1_T, partial_1]).astype(np.uint8)
    return np.vstack([top, S_X_prime]).astype(np.uint8)


def _step3_assemble(
    code: CSSCode,
    support: tuple[int, ...],
    data_checks: tuple[int, ...],
    incidence: np.ndarray,
    partial_0: np.ndarray,
    *,
    basis: PauliXZ = Pauli.X,
) -> tuple[np.ndarray, np.ndarray]:
    """Block assembly of HX_merged, HZ_merged (Webster, Smith, Cohen arXiv:2511.15989 §II.A;
    Cain et al. arXiv:2603.28627 §B.1).

    S'_meas rows go into the measurement-basis merged matrix; ∂_0 rows go into
    the complementary-basis merged matrix. f_0 = π_{C₀}^T extends original checks onto Q'.

    basis=X (default): S'_meas rows added to HX_merged; ∂_0 rows added to HZ_merged.
    basis=Z: S'_meas rows added to HZ_merged; ∂_0 rows added to HX_merged (basis-symmetric dual).
    """
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    n = code.num_qudits
    mX, mZ = HX.shape[0], HZ.shape[0]
    nC = len(data_checks)
    r = partial_0.shape[0]

    # f_0 = π_{C₀}^T : extends the original Z-checks (basis=X) onto the new Q' ancillas.
    # _projection's sentinel rule zeroes the columns of boost-added Q' (data_checks == -1).
    m_comp = mZ if basis is Pauli.X else mX
    f_0 = _projection(data_checks, m_comp).T.astype(np.uint8)   # (m_comp, |C₀|)

    support_arr = np.asarray(support, dtype=np.int_)

    if basis is Pauli.X:
        # S'_meas rows extend HX_merged; ∂_0 rows extend HZ_merged
        HX_merged = _assemble_HX_L1(HX, support_arr, incidence)
        HZ_merged = np.block(
            [
                [HZ, f_0],
                [np.zeros((r, n), dtype=np.uint8), partial_0.astype(np.uint8)],
            ]
        ).astype(np.uint8)
    else:
        # basis=Z (symmetric dual): S'_meas rows extend HZ_merged; ∂_0 rows extend HX_merged
        HZ_merged = _assemble_HX_L1(HZ, support_arr, incidence)
        HX_merged = np.block(
            [
                [HX, f_0],
                [np.zeros((r, n), dtype=np.uint8), partial_0.astype(np.uint8)],
            ]
        ).astype(np.uint8)

    return HX_merged, HZ_merged


def build_gadget(
    code: CSSCode,
    x: np.ndarray,
    *,
    basis: PauliXZ,
) -> GadgetLayout:
    """Full L=1 gadget: steps 1+2+3 composed (Webster, Smith, Cohen arXiv:2511.15989 §II.A;
    Cain et al. arXiv:2603.28627 §B.1). Deterministic in (code, x, basis).

    Q' ancilla qubits (κ) are indexed contiguously after the n data qubits.
    basis=Pauli.X: measures a logical X̄ (PPM of X̄). Validates H_Z @ x == 0 mod 2.
    basis=Pauli.Z: measures a logical Z̄ (PPM of Z̄). Validates H_X @ x == 0 mod 2.
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

    support, data_checks, incidence = _step1_restriction(code, x, basis=basis)
    partial_0 = _step2_gauge_fix(incidence)
    HX_m, HZ_m = _step3_assemble(code, support, data_checks, incidence, partial_0, basis=basis)
    Q_prime = tuple(range(code.num_qudits, code.num_qudits + len(data_checks)))  # Q' ancilla qubit IDs
    return GadgetLayout(
        code=code,
        x=x,
        support=support,
        data_checks=data_checks,
        incidence=incidence,
        partial_0=partial_0,
        HX_merged=HX_m,
        HZ_merged=HZ_m,
        Q_prime=Q_prime,
        basis=basis,
    )


def build_gadget_augmented(
    code: CSSCode,
    x: np.ndarray,
    incidence_extra: np.ndarray,
    *,
    basis: PauliXZ,
) -> GadgetLayout:
    """Rebuild a GadgetLayout with ∂_1^T augmented by extra weight-2 rows
    (Webster, Smith, Cohen arXiv:2511.15989 §II.A; Cain et al. arXiv:2603.28627 §B.1).

    Each row of ``incidence_extra`` (weight 2) corresponds to a new Q' (κ) qubit not
    backed by any original complementary-basis check. The function:

    1. Stacks incidence_aug = [∂_1^T; incidence_extra] (augmented |C₀|×|V₀| matrix).
    2. Recomputes ∂_0_aug = ker(incidence_aug^T) via _step2_gauge_fix.
    3. Calls _step3_assemble with -1 sentinels appended to C₀ for the new κ qubits
       (their f_0 columns are zero, as no original check maps onto them).

    The returned ``GadgetLayout.data_checks`` and ``Q_prime`` are extended to
    cover the new κ qubits; new κ indices come after the original ones.
    """
    x = np.asarray(x).astype(np.uint8)
    support, data_checks, incidence = _step1_restriction(code, x, basis=basis)
    incidence_extra = np.asarray(incidence_extra).astype(np.uint8)
    if incidence_extra.shape[1] != len(support):
        raise ValueError(
            f"incidence_extra has {incidence_extra.shape[1]} columns; expected {len(support)} (= |support|)"
        )
    if incidence_extra.size and not np.all(incidence_extra.sum(axis=1) == 2):
        bad = np.flatnonzero(incidence_extra.sum(axis=1) != 2).tolist()
        raise ValueError(f"incidence_extra rows {bad} have weight != 2; required weight 2.")

    incidence_aug = np.vstack([incidence, incidence_extra]).astype(np.uint8)
    partial_0_aug = _step2_gauge_fix(incidence_aug)

    # _step3_assemble builds f_0 = π_{C₀}^T; we need C₀_aug to include -1 sentinels
    # for the new κ qubits so their f_0 columns are all-zero (no original check maps
    # onto them). _projection's sentinel rule handles indices outside [0, m_comp).
    n_extra = incidence_extra.shape[0]
    data_checks_aug = tuple(data_checks) + tuple([-1] * n_extra)
    HX_aug, HZ_aug = _step3_assemble(
        code,
        support,
        data_checks_aug,
        incidence_aug,
        partial_0_aug,
        basis=basis,
    )
    Q_prime_aug = tuple(range(code.num_qudits, code.num_qudits + len(data_checks_aug)))  # Q' ancilla qubit IDs
    return GadgetLayout(
        code=code,
        x=x,
        support=support,
        data_checks=data_checks_aug,
        incidence=incidence_aug,
        partial_0=partial_0_aug,
        HX_merged=HX_aug,
        HZ_merged=HZ_aug,
        Q_prime=Q_prime_aug,
        basis=basis,
    )
