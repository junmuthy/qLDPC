"""L=1 gadget construction (Webster, Smith, Cohen arXiv:2511.15989 §II.A;
Cain et al. arXiv:2603.28627 §B.1; Ide, Gowda, Nadkarni, Dauphinais
arXiv:2410.02753 Eq.(62)).

``build_gadget`` (single-operator X̄/Z̄ measurement) assembles the merged
matrices from ``edge_expanded_maps`` — the full arXiv:2410.02753 Algorithm 3
(Cheeger ≥ 1 edge expansion + low-weight ∂_0) — per the mapping cone Eq.(12).

Closed-form primitives kept for the joint/Ȳ path (basis dispatched via the
X↔Z dual; consumed by ``PPM_joint.py`` and ``cheeger.py::boost_gadget``):
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

from .edge_expanded import edge_expanded_maps

GF2 = galois.GF(2)


def _gf2_rank(matrix: np.ndarray) -> int:
    """GF(2) row rank of a binary matrix (0 for an empty matrix)."""
    arr = np.asarray(matrix).astype(np.uint8) % 2
    return 0 if arr.shape[0] == 0 else int(GF2(arr).row_space().shape[0])


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
        f1 / f0       = the remaining two mapping-cone maps (Ide, Gowda, Nadkarni,
                        Dauphinais arXiv:2410.02753 Eq.(12): f_1 is the n×|V₀|
                        support indicator, f_0 the |checks|×|edges| check-to-edge
                        map). Optional: ``None`` on layouts produced by the
                        closed-form joint/boost path (``build_gadget_augmented``).
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
    f1: np.ndarray | None = None
    f0: np.ndarray | None = None


def build_gadget(
    code: CSSCode,
    x: np.ndarray,
    *,
    basis: PauliXZ,
    seed: int = 0,
    n_samples: int = 200,
    cellulate_to: int | str | None = "native",
) -> GadgetLayout:
    """Full L=1 gadget via the edge-expanded homological measurement (Ide, Gowda,
    Nadkarni, Dauphinais arXiv:2410.02753 Algorithm 3, assembled per Eq.(12)/(13)).

    ``edge_expanded_maps`` produces the four mapping-cone maps f_1, ∂_1, f_0, ∂_0
    with Cheeger(∂_1) ≥ 1 (fault distance preserved, arXiv:2410.02753 Thm 4) and a
    low-weight ∂_0 (Algorithm 2 random search; redundant cycles already removed).
    The merged matrices are the mapping-cone blocks

        H̃_X = [[H_X,   0 ],     H̃_Z = [[H_Z,  f_0],
               [f_1^T, ∂_1]]            [ 0,   ∂_0]]

    basis=Pauli.X: measures a logical X̄ (validates H_Z @ x == 0 mod 2);
    basis=Pauli.Z: measures a logical Z̄ via the X↔Z dual — same construction on
    the swapped (H_Z, H_X), merged matrices swapped on the way out
    (arXiv:2410.02753 §III D). Q' ancilla (edge) qubits are indexed contiguously
    after the n data qubits. Deterministic in (code, x, basis, seed).

    ``cellulate_to``: "desired" max ∂_0 row weight triggering Algorithm 3's
    hyperedge-expansion + cellulation branch. ``"native"`` (default) resolves to
    the max row weight of the complementary check matrix; an int overrides;
    ``None`` disables the branch (accept the main-path ∂_0 unconditionally).
    """
    x = np.asarray(x).astype(np.uint8)
    if basis is Pauli.X:
        H_meas = np.asarray(code.matrix_x).astype(np.uint8)
        H_comp = np.asarray(code.matrix_z).astype(np.uint8)
    elif basis is Pauli.Z:
        H_meas = np.asarray(code.matrix_z).astype(np.uint8)
        H_comp = np.asarray(code.matrix_x).astype(np.uint8)
    else:
        raise ValueError(f"basis must be Pauli.X or Pauli.Z, got {basis!r}")
    if ((H_comp @ x) % 2).any():
        which = "X" if basis is Pauli.X else "Z"
        comp = "H_Z" if basis is Pauli.X else "H_X"
        raise ValueError(f"x is not a logical-{which} support ({comp} @ x != 0).")

    ct = int(H_comp.sum(axis=1).max()) if cellulate_to == "native" else cellulate_to
    cm = edge_expanded_maps(H_comp, x, seed=seed, n_samples=n_samples, cellulate_to=ct)

    n = H_meas.shape[1]
    n_edges = cm.incidence.shape[0]
    d1 = cm.incidence.T.astype(np.uint8)  # ∂_1 (vertex×edge) for the H̃ block
    f1T = cm.f1.T.astype(np.uint8)  # f_1^T (|V₀|×n support indicator)
    # H̃_X = [[H_X, 0],[f1^T, ∂1]]; H̃_Z = [[H_Z, f0],[0, ∂0]] (arXiv:2410.02753 Eq 13)
    HX = np.block(
        [[H_meas, np.zeros((H_meas.shape[0], n_edges), np.uint8)], [f1T, d1]]
    ).astype(np.uint8)
    HZ = np.block(
        [[H_comp, cm.f0], [np.zeros((cm.partial_0.shape[0], n), np.uint8), cm.partial_0]]
    ).astype(np.uint8)
    if basis is Pauli.Z:  # X↔Z dual: swap the merged matrices on the way out
        HX, HZ = HZ, HX
    Q_prime = tuple(range(code.num_qudits, code.num_qudits + n_edges))
    return GadgetLayout(
        code=code,
        x=x,
        support=cm.support,
        data_checks=cm.data_checks,
        incidence=cm.incidence,
        partial_0=cm.partial_0,
        HX_merged=HX,
        HZ_merged=HZ,
        Q_prime=Q_prime,
        basis=basis,
        f1=cm.f1,
        f0=cm.f0,
    )


def minimize_z_checks(gadget: GadgetLayout) -> GadgetLayout:
    """Drop cycle checks that are redundant with the deformed original checks.

    ``partial_0`` (= ∂_0 = a full basis of the cycle space ker(∂_1)) generally
    contains rows that already lie in rowspan([H_complement | f_0]) — the cycle
    is a product of the deformed original checks, so it adds no new stabilizer
    (Cross, He, Rall, Yoder arXiv:2407.18393 Eq.(6), "redundant cycles"; the
    count removed is ``dim U``). This returns a new gadget keeping only the
    ``dim(ker ∂_1) − dim U`` independent cycle checks.

    The stabilizer group, code distance, ancilla qubits (``Q'``/edges), and the
    entire opposite-type check side are unchanged — only redundant cycle-check
    *generators* are removed. Consumed by the closed-form joint/Ȳ path
    (``cheeger.py::boost_gadget``); the edge-expanded ``build_gadget`` no longer
    needs it (Algorithm 2 already returns a non-redundant ∂_0, arXiv:2410.02753
    Alg 2 lines 1-6). For an X̄ gadget the cycle checks are Z-type (in
    ``HZ_merged``); for a Z̄ gadget they are X-type (in ``HX_merged``, the X↔Z
    dual).
    """
    if gadget.basis is Pauli.X:
        merged = np.asarray(gadget.HZ_merged).astype(np.uint8)
        n_comp = gadget.code.matrix_z.shape[0]
    else:
        merged = np.asarray(gadget.HX_merged).astype(np.uint8)
        n_comp = gadget.code.matrix_x.shape[0]

    top = merged[:n_comp]  # [H_complement | f_0] — deformed original checks
    cyc = merged[n_comp:]  # [0 | ∂_0] — candidate cycle checks (row-aligned with partial_0)

    # Greedily keep only cycle rows that increase the GF(2) rank, i.e. are not
    # already implied by the original checks + previously kept cycles.
    basis_arr = top.copy()
    rank = _gf2_rank(basis_arr)
    keep: list[int] = []
    for i in range(cyc.shape[0]):
        trial = np.vstack([basis_arr, cyc[i : i + 1]])
        trial_rank = _gf2_rank(trial)
        if trial_rank > rank:
            basis_arr, rank = trial, trial_rank
            keep.append(i)

    new_partial_0 = np.asarray(gadget.partial_0)[keep].astype(np.uint8)
    new_merged = np.vstack([top, cyc[keep]]).astype(np.uint8)
    if gadget.basis is Pauli.X:
        return dataclasses.replace(gadget, partial_0=new_partial_0, HZ_merged=new_merged)
    return dataclasses.replace(gadget, partial_0=new_partial_0, HX_merged=new_merged)


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
