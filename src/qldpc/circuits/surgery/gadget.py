"""L=1 gadget construction (Webster, Smith, Cohen arXiv:2511.15989 §II.A;
Cain et al. arXiv:2603.28627 §B.1).

Three explicit named steps that map 1:1 to the paper:
    _step1_restriction  — restriction: V₀=supp(x), C₀=checks touching V₀,
                          H_X' = π_{V₀} H_Z^T π_{C₀}^T (stored transposed as `incidence`)
    _step2_gauge_fix    — gauge fix: G = ker((H_X')^T) over GF(2) (stored as `gauge`)
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


@dataclasses.dataclass(frozen=True, eq=False)
class GadgetLayout:
    """Frozen result of a single L=1 gadget construction.

    Symbol mapping (Webster, Smith, Cohen arXiv:2511.15989 §II.A;
    Cain et al. arXiv:2603.28627 §B.1):
        support       = V₀  (qubit indices in supp(x))
        data_checks   = C₀  (complementary-basis check indices touching V₀;
                             -1 sentinels indicate boost-added Q' with no backing check)
        incidence     = (H_X')^T  (|C₀|×|V₀| matrix; H_X' = π_{V₀}H_Z^Tπ_{C₀}^T)
        gauge         = G   (basis of ker((H_X')^T) over GF(2))
        Q_prime       = Q' (κ qubits, indexed after the n data qubits)
    """

    code: CSSCode
    x: np.ndarray
    support: tuple[int, ...]
    data_checks: tuple[int, ...]
    incidence: np.ndarray
    gauge: np.ndarray
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
    H_X' = π_{V₀} H_Z^T π_{C₀}^T (stored transposed as `incidence`, shape |C₀|×|V₀|).

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
    # H_X' = π_{V₀} H_Z^T π_{C₀}^T  (Webster, Smith, Cohen arXiv:2511.15989 §II.A;
    # Cain et al. arXiv:2603.28627 §B.1).  The stored `incidence` is its transpose —
    # the |C₀|×|V₀| vertex-edge incidence (§4 y_gadget.py calls this ∂₁ˣ).
    n = code.num_qudits
    pi_V0 = _projection(support, n)                  # π_{V₀} = f_X' ∈ F₂^{|V₀|×n}
    pi_C0 = _projection(data_checks, H_complement.shape[0])   # π_{C₀} ∈ F₂^{|C₀|×m_comp}
    H_X_prime = (pi_V0 @ H_complement.T @ pi_C0.T) % 2        # |V₀|×|C₀|
    incidence = H_X_prime.T.astype(np.uint8)         # |C₀|×|V₀|
    return support, data_checks, incidence


def _step2_gauge_fix(incidence: np.ndarray) -> np.ndarray:
    """Gauge fix (Webster, Smith, Cohen arXiv:2511.15989 §II.A; Cain et al. arXiv:2603.28627 §B.1).

    G = gauge = canonical row basis of ker((H_X')^T) = ker(incidence^T) over GF(2).
    `incidence` = (H_X')^T (|C₀|×|V₀|); returns G of shape (r, |C₀|) where r = |C₀| - rank(H_X').

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

    S_X' = [f_X' | H_X'] where f_X' = π_{V₀} (indicator on data qubits) and
    H_X' = incidence.T (Webster, Smith, Cohen arXiv:2511.15989 §II.A;
    Cain et al. arXiv:2603.28627 §B.1).

    Called by _step3_assemble (initial gadget assembly) and build_gadget_augmented
    (post-boost rebuild, with augmented incidence). The Z-side assembly is NOT shared —
    the boost rebuild treats new κ' qubits as pure-gauge (no data-Z extension).

    Args:
        HX_data: original code's X-check matrix, shape (mX, n), uint8.
        support_indices: indices of V₀ within the n data qubits, shape (|V₀|,).
        incidence: (H_X')^T, shape (|C₀|, |V₀|), uint8.

    Returns:
        HX_merged: shape (mX + |V₀|, n + |C₀|), uint8.
    """
    mX, n = HX_data.shape
    n_v0, n_c0 = int(incidence.shape[1]), int(incidence.shape[0])
    top = np.hstack([HX_data, np.zeros((mX, n_c0), dtype=np.uint8)]).astype(np.uint8)
    # S_X' rows = [f_X' | H_X'] : f_X' = π_{V₀} on data, H_X' = incidence.T on Q'.
    f_X_prime = np.zeros((n_v0, n), dtype=np.uint8)
    f_X_prime[np.arange(n_v0), np.asarray(support_indices)] = 1
    H_X_prime = incidence.T.astype(np.uint8)
    S_X_prime = np.hstack([f_X_prime, H_X_prime]).astype(np.uint8)
    return np.vstack([top, S_X_prime]).astype(np.uint8)


def _step3_assemble(
    code: CSSCode,
    support: tuple[int, ...],
    data_checks: tuple[int, ...],
    incidence: np.ndarray,
    gauge: np.ndarray,
    *,
    basis: PauliXZ = Pauli.X,
) -> tuple[np.ndarray, np.ndarray]:
    """Block assembly of HX_merged, HZ_merged (Webster, Smith, Cohen arXiv:2511.15989 §II.A;
    Cain et al. arXiv:2603.28627 §B.1).

    S'_meas rows go into the measurement-basis merged matrix; G=gauge rows go into
    the complementary-basis merged matrix. f_Z = π_{C₀}^T extends original checks onto Q'.

    basis=X (default): S'_meas rows added to HX_merged; G rows added to HZ_merged.
    basis=Z: S'_meas rows added to HZ_merged; G rows added to HX_merged (basis-symmetric dual).
    """
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    n = code.num_qudits
    mX, mZ = HX.shape[0], HZ.shape[0]
    nC = len(data_checks)
    r = gauge.shape[0]

    # f_Z = π_{C₀}^T : extends the original Z-checks (basis=X) onto the new Q' ancillas.
    # _projection's sentinel rule zeroes the columns of boost-added Q' (data_checks == -1).
    m_comp = mZ if basis is Pauli.X else mX
    f_Z = _projection(data_checks, m_comp).T.astype(np.uint8)   # (m_comp, |C₀|)

    support_arr = np.asarray(support, dtype=np.int_)

    if basis is Pauli.X:
        # S'_meas rows extend HX_merged; G rows extend HZ_merged
        HX_merged = _assemble_HX_L1(HX, support_arr, incidence)
        HZ_merged = np.block(
            [
                [HZ, f_Z],
                [np.zeros((r, n), dtype=np.uint8), gauge.astype(np.uint8)],
            ]
        ).astype(np.uint8)
    else:
        # basis=Z (symmetric dual): S'_meas rows extend HZ_merged; G rows extend HX_merged
        HZ_merged = _assemble_HX_L1(HZ, support_arr, incidence)
        HX_merged = np.block(
            [
                [HX, f_Z],
                [np.zeros((r, n), dtype=np.uint8), gauge.astype(np.uint8)],
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
    gauge = _step2_gauge_fix(incidence)
    HX_m, HZ_m = _step3_assemble(code, support, data_checks, incidence, gauge, basis=basis)
    Q_prime = tuple(range(code.num_qudits, code.num_qudits + len(data_checks)))  # Q' ancilla qubit IDs
    return GadgetLayout(
        code=code,
        x=x,
        support=support,
        data_checks=data_checks,
        incidence=incidence,
        gauge=gauge,
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
    """Rebuild a GadgetLayout with (H_X')^T augmented by extra weight-2 rows
    (Webster, Smith, Cohen arXiv:2511.15989 §II.A; Cain et al. arXiv:2603.28627 §B.1).

    Each row of ``incidence_extra`` (weight 2) corresponds to a new Q' (κ) qubit not
    backed by any original complementary-basis check. The function:

    1. Stacks incidence_aug = [(H_X')^T; incidence_extra] (augmented |C₀|×|V₀| matrix).
    2. Recomputes G_aug = ker(incidence_aug^T) via _step2_gauge_fix.
    3. Calls _step3_assemble with -1 sentinels appended to C₀ for the new κ qubits
       (their f_Z columns are zero, as no original check maps onto them).

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
    gauge_aug = _step2_gauge_fix(incidence_aug)

    # _step3_assemble builds f_Z = π_{C₀}^T; we need C₀_aug to include -1 sentinels
    # for the new κ qubits so their f_Z columns are all-zero (no original check maps
    # onto them). _projection's sentinel rule handles indices outside [0, m_comp).
    n_extra = incidence_extra.shape[0]
    data_checks_aug = tuple(data_checks) + tuple([-1] * n_extra)
    HX_aug, HZ_aug = _step3_assemble(
        code,
        support,
        data_checks_aug,
        incidence_aug,
        gauge_aug,
        basis=basis,
    )
    Q_prime_aug = tuple(range(code.num_qudits, code.num_qudits + len(data_checks_aug)))  # Q' ancilla qubit IDs
    return GadgetLayout(
        code=code,
        x=x,
        support=support,
        data_checks=data_checks_aug,
        incidence=incidence_aug,
        gauge=gauge_aug,
        HX_merged=HX_aug,
        HZ_merged=HZ_aug,
        Q_prime=Q_prime_aug,
        basis=basis,
    )
