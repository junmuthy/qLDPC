"""L=1 gadget construction (Webster, Smith, Cohen arXiv:2511.15989 §II.A;
Cain et al. arXiv:2603.28627 §B.1; Ide, Gowda, Nadkarni, Dauphinais
arXiv:2410.02753 Eq.(62)).

Closed-form structure (two primitives, basis dispatched via the X↔Z dual):
    _restrict   — restriction over GF(2): V₀=supp(x), C₀=complementary-basis
                  checks touching V₀, incidence=∂_1^T=H_complement[C₀,V₀],
                  partial_0=∂_0=ker(∂_1).
    _x_merged   — closed-form merged matrices H̃_X / H̃_Z for measuring X̄
                  (block assembly of f_1^T, ∂_1, f_0, ∂_0).
basis=Pauli.Z reuses the same primitives via the X↔Z dual (swap H_X/H_Z in,
swap the merged matrices out).
"""

from __future__ import annotations

import dataclasses

import galois
import numpy as np

from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli, PauliXZ

GF2 = galois.GF(2)


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
        # Only the first |C₀| columns map to a backing check; the boost-added κ
        # (data_checks[n_C0:] == -1 sentinels) keep all-zero f_0 columns.
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


def build_gadget(
    code: CSSCode,
    x: np.ndarray,
    *,
    basis: PauliXZ,
) -> GadgetLayout:
    """Full L=1 gadget: closed-form ``_x_merged`` with X↔Z dual dispatch
    (Webster, Smith, Cohen arXiv:2511.15989 §II.A; Cain et al. arXiv:2603.28627 §B.1;
    Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 Eq.(62)). Deterministic in
    (code, x, basis). basis=Pauli.X builds H̃_X/H̃_Z directly; basis=Pauli.Z feeds
    the swapped (H_Z, H_X) into ``_x_merged`` and swaps the merged matrices on the
    way out (the X↔Z dual of §II.A).

    Q' ancilla qubits (κ) are indexed contiguously after the n data qubits.
    basis=Pauli.X: measures a logical X̄ (PPM of X̄). Validates H_Z @ x == 0 mod 2.
    basis=Pauli.Z: measures a logical Z̄ (PPM of Z̄). Validates H_X @ x == 0 mod 2.
    """
    x = np.asarray(x).astype(np.uint8)
    if basis is Pauli.X:
        H_check = np.asarray(code.matrix_z).astype(np.uint8)
    elif basis is Pauli.Z:
        H_check = np.asarray(code.matrix_x).astype(np.uint8)
    else:
        raise ValueError(f"basis must be Pauli.X or Pauli.Z, got {basis!r}")
    if ((H_check @ x) % 2).any():
        which = "X" if basis is Pauli.X else "Z"
        comp = "H_Z" if basis is Pauli.X else "H_X"
        raise ValueError(f"x is not a logical-{which} support ({comp} @ x != 0).")

    # basis=X: X frame directly; basis=Z: X↔Z dual (swap H_X/H_Z in, swap merged out)
    # (Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III D — the dual of §2).
    if basis is Pauli.X:
        support, data_checks, incidence, partial_0, HX_m, HZ_m = _x_merged(
            code.matrix_x, code.matrix_z, x
        )
    else:
        support, data_checks, incidence, partial_0, HZ_m, HX_m = _x_merged(
            code.matrix_z, code.matrix_x, x
        )
    Q_prime = tuple(range(code.num_qudits, code.num_qudits + len(data_checks)))
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
    backed by any original complementary-basis check. The work is delegated to
    ``_x_merged(..., incidence_extra=...)``, which:

    1. Stacks incidence_aug = [∂_1^T; incidence_extra] (augmented |C₀|×|V₀| matrix).
    2. Re-derives ∂_0_aug = ker(∂_1_aug) on the stacked incidence over GF(2).
    3. Appends -1 sentinels to C₀ for the new κ qubits, so their f_0 columns are
       zero (no original check maps onto them) when H̃_X/H̃_Z are assembled.

    basis dispatch uses the X↔Z dual (swap H_X/H_Z in, swap merged out), matching
    ``build_gadget``. The returned ``GadgetLayout.data_checks`` and ``Q_prime`` are
    extended to cover the new κ qubits; new κ indices come after the original ones.
    """
    x = np.asarray(x).astype(np.uint8)
    incidence_extra = np.asarray(incidence_extra).astype(np.uint8)
    support_len = int(np.count_nonzero(x))
    if incidence_extra.shape[1] != support_len:
        raise ValueError(
            f"incidence_extra has {incidence_extra.shape[1]} columns; "
            f"expected {support_len} (= |support|)"
        )
    if incidence_extra.size and not np.all(incidence_extra.sum(axis=1) == 2):
        bad = np.flatnonzero(incidence_extra.sum(axis=1) != 2).tolist()
        raise ValueError(f"incidence_extra rows {bad} have weight != 2; required weight 2.")
    if basis is Pauli.X:
        support, data_checks, incidence, partial_0, HX_m, HZ_m = _x_merged(
            code.matrix_x, code.matrix_z, x, incidence_extra
        )
    elif basis is Pauli.Z:
        support, data_checks, incidence, partial_0, HZ_m, HX_m = _x_merged(
            code.matrix_z, code.matrix_x, x, incidence_extra
        )
    else:
        raise ValueError(f"basis must be Pauli.X or Pauli.Z, got {basis!r}")
    Q_prime = tuple(range(code.num_qudits, code.num_qudits + len(data_checks)))
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
