"""Logical-Y measurement gadget primitives (Ȳ = iX̄Z̄).

Building blocks for measuring a logical Y operator via lattice surgery, following
the homological measurement construction of Ide, Gowda, Nadkarni, Dauphinais
arXiv:2410.02753 §III.C/§III.D. The
logical-Y support is the set ``W = supp(x) ∩ supp(z)`` of data qubits where the
logical-X string and the logical-Z string cross; ``|W|`` is odd (the strings
anticommute) and may be any size, so the merge handles general ``|W| ≥ 1``.

This module provides:
    _locate_overlaps — validate an (x, z) representative pair and return the
                       tuple ``W = supp(x) ∩ supp(z)`` of physical Y qubits.
    _steane_y_pair   — a concrete Steane-code fixture (code, x, z) whose
                       logical-X / logical-Z supports overlap on exactly one
                       qubit (the ``|W|=1`` case).
"""

from __future__ import annotations

import dataclasses
import itertools

import galois
import numpy as np

from qldpc import codes
from qldpc.codes.common import CSSCode, QuditCode
from qldpc.objects import Pauli

from .gadget import GadgetLayout, build_gadget
from .merge import apply_mixed_basis_merge

GF2 = galois.GF(2)


def _in_rowspace_gf2(M: np.ndarray, v: np.ndarray) -> bool:
    """Return True iff ``v`` (1D uint8) lies in the GF(2) row space of ``M``.

    Standard GF(2) membership test: ``v ∈ rowspace(M)`` exactly when appending
    ``v`` as a new row does not increase the rank, i.e. ``rank(M) == rank([M; v])``.
    Ranks are taken over GF(2) via ``galois``'s overload of
    ``numpy.linalg.matrix_rank`` on ``galois.GF(2)`` arrays.

    Used to certify that the symplectic vector ``[x | z]`` of Ȳ = iX̄Z̄ is a
    product of the merged-code stabilizers restricted to the original data
    qubits, the core correctness guarantee of the homological logical-Y merge of
    Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.C/§III.D.
    """
    M2 = GF2(np.asarray(M).astype(np.uint8))
    A = GF2(np.vstack([np.asarray(M).astype(np.uint8), np.asarray(v).astype(np.uint8)[None, :]]))
    return int(np.linalg.matrix_rank(M2)) == int(np.linalg.matrix_rank(A))


def _locate_overlap(code: CSSCode, x: np.ndarray, z: np.ndarray) -> int:
    """Return the single data qubit shared by logical-X support ``x`` and logical-Z support ``z``.

    Legacy ``|W|=1`` helper retained as a test fixture; the general-``|W|`` merge
    uses :func:`_locate_overlaps`. The single-overlap special case of the
    Ȳ-overlap ``W = supp(x) ∩ supp(z)`` of Ide, Gowda, Nadkarni, Dauphinais
    arXiv:2410.02753 §III.D: Ȳ = iX̄Z̄ is
    realised cleanly when the X̄ and Z̄ strings cross on exactly one data qubit.

    Validates, raising ``ValueError`` otherwise:
        * ``x`` is a logical-X representative: ``H_Z @ x == 0`` (mod 2),
        * ``z`` is a logical-Z representative: ``H_X @ z == 0`` (mod 2),
        * ``x`` and ``z`` anticommute: ``x · z`` has odd weight (mod 2),
        * their supports overlap on exactly one qubit: ``|supp(x) ∩ supp(z)| == 1``.

    Returns the index of that shared qubit.
    """
    x = np.asarray(x).astype(np.uint8)
    z = np.asarray(z).astype(np.uint8)
    n = code.num_qudits
    if x.shape != (n,):
        raise ValueError(f"x has shape {x.shape}, expected ({n},)")
    if z.shape != (n,):
        raise ValueError(f"z has shape {z.shape}, expected ({n},)")

    matrix_x = np.asarray(code.matrix_x).astype(np.uint8)  # H_X (X-type stabilizers)
    matrix_z = np.asarray(code.matrix_z).astype(np.uint8)  # H_Z (Z-type stabilizers)

    # x must commute with every Z-type stabilizer (i.e. be a logical-X representative).
    if ((matrix_z @ x) % 2 != 0).any():
        raise ValueError("x is not a logical-X representative: H_Z @ x != 0 (mod 2)")
    # z must commute with every X-type stabilizer (i.e. be a logical-Z representative).
    if ((matrix_x @ z) % 2 != 0).any():
        raise ValueError("z is not a logical-Z representative: H_X @ z != 0 (mod 2)")

    # x and z must anticommute: a logical-X and logical-Z of the SAME logical qubit
    # cross an odd number of times. A commuting pair (even overlap parity) cannot
    # form Ȳ = iX̄Z̄.
    if int(np.dot(x.astype(np.int64), z.astype(np.int64))) % 2 == 0:
        raise ValueError("x and z commute (x · z is even); they cannot form Ȳ = iX̄Z̄")

    overlap = np.where((x.astype(bool)) & (z.astype(bool)))[0]
    if overlap.size != 1:
        raise ValueError(
            f"x and z overlap on {overlap.size} qubits, expected exactly 1 "
            "(Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.D "
            "single-overlap case)"
        )
    return int(overlap[0])


def _locate_overlaps(code: CSSCode, x: np.ndarray, z: np.ndarray) -> tuple[int, ...]:
    """Return W = supp(x) ∩ supp(z), the physical Pauli-Y qubits of Ȳ = iX̄Z̄.

    Validates (Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.D):
      * x is a logical-X representative: H_Z @ x == 0 (mod 2),
      * z is a logical-Z representative: H_X @ z == 0 (mod 2),
      * x and z anticommute: x · z is odd (so |W| is odd, ≥ 1).
    """
    x = np.asarray(x).astype(np.uint8)
    z = np.asarray(z).astype(np.uint8)
    n = code.num_qudits
    if x.shape != (n,) or z.shape != (n,):
        raise ValueError(f"x/z must have shape ({n},); got {x.shape}, {z.shape}")
    if ((np.asarray(code.matrix_z).astype(np.uint8) @ x) % 2 != 0).any():
        raise ValueError("x is not a logical-X representative: H_Z @ x != 0 (mod 2)")
    if ((np.asarray(code.matrix_x).astype(np.uint8) @ z) % 2 != 0).any():
        raise ValueError("z is not a logical-Z representative: H_X @ z != 0 (mod 2)")
    if int(np.dot(x.astype(np.int64), z.astype(np.int64))) % 2 == 0:
        raise ValueError("x and z commute (x · z even); they cannot form Ȳ = iX̄Z̄")
    return tuple(int(i) for i in np.where(x.astype(bool) & z.astype(bool))[0])


def _steane_y_pair() -> tuple[CSSCode, np.ndarray, np.ndarray]:
    """Return a Steane code and a logical (x, z) pair overlapping on exactly one qubit.

    This is the clean single-overlap (``|W|=1``) fixture for the Ȳ-overlap
    ``W = supp(x) ∩ supp(z)`` of Ȳ = iX̄Z̄ (Ide, Gowda, Nadkarni, Dauphinais
    arXiv:2410.02753 §III.D). The canonical
    weight-3 logical-X and logical-Z
    representatives of the [[7,1,3]] Steane code already cross on a single data
    qubit; if a given representative pair does not, it is reduced over GF(2) by
    adding stabilizer rows (a logical-X support stays logical-X after XOR-ing any
    row of ``matrix_x``; a logical-Z support stays logical-Z after XOR-ing any row
    of ``matrix_z``) until the supports overlap on exactly one qubit.
    """
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)

    if _overlap_size(x, z) == 1:
        return code, x, z

    # Fallback: search small GF(2) combinations of stabilizer rows added to x and/or
    # z. Adding an X-stabilizer row to x leaves it a logical-X representative;
    # adding a Z-stabilizer row to z leaves it a logical-Z representative.
    rows_x = GF2(np.asarray(code.matrix_x).astype(np.uint8))
    rows_z = GF2(np.asarray(code.matrix_z).astype(np.uint8))
    x_gf = GF2(x)
    z_gf = GF2(z)
    n_rx = rows_x.shape[0]
    n_rz = rows_z.shape[0]

    for k_x in range(n_rx + 1):
        for cols_x in itertools.combinations(range(n_rx), k_x):
            x_cand = x_gf + (sum((rows_x[i] for i in cols_x), GF2.Zeros(x_gf.shape)))
            for k_z in range(n_rz + 1):
                for cols_z in itertools.combinations(range(n_rz), k_z):
                    z_cand = z_gf + (sum((rows_z[j] for j in cols_z), GF2.Zeros(z_gf.shape)))
                    x_arr = np.asarray(x_cand).astype(np.uint8)
                    z_arr = np.asarray(z_cand).astype(np.uint8)
                    if _overlap_size(x_arr, z_arr) == 1:
                        return code, x_arr, z_arr

    raise ValueError(
        "BLOCKED: no overlap-1 logical (x, z) pair reachable on the Steane code by "
        "stabilizer-row reduction; the multi-overlap (|W|≥2) merge would be required "
        "(Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.D)"
    )


def _overlap_size(x: np.ndarray, z: np.ndarray) -> int:
    """Number of data qubits in ``supp(x) ∩ supp(z)``."""
    return int(np.count_nonzero(x.astype(bool) & z.astype(bool)))


def _bb_y_pair(overlap: int = 1) -> tuple[CSSCode, np.ndarray, np.ndarray]:
    """BB [[36,8,4]] fixture for Ȳ on logical qubit 0 with chosen |W|.

    overlap=1: canonical reps (already single-overlap). overlap=3: add stabilizer
    rows (deterministic seed) until supp(x)∩supp(z) has size 3 — the |W|≥2
    crossing-cycle regime of arXiv:2410.02753 §III.D.
    """
    import sympy

    xs, ys = sympy.symbols("x y")
    code = codes.BBCode({xs: 3, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)
    n = code.num_qudits
    LX = np.asarray(code.get_logical_ops(Pauli.X)).astype(np.uint8)
    LZ = np.asarray(code.get_logical_ops(Pauli.Z)).astype(np.uint8)
    wide = LX.shape[1] == 2 * n
    x = (LX[0][:n] if wide else LX[0]).astype(np.uint8)
    z = (LZ[0][n:] if wide else LZ[0]).astype(np.uint8)
    if overlap == 1:
        return code, x, z
    if overlap == 3:
        HX = np.asarray(code.matrix_x).astype(np.uint8)
        HZ = np.asarray(code.matrix_z).astype(np.uint8)
        rng = np.random.default_rng(0)
        for _ in range(20000):
            ax = (
                rng.integers(0, 2, HX.shape[0])
                if rng.random() < 0.5
                else np.zeros(HX.shape[0], int)
            )
            az = rng.integers(0, 2, HZ.shape[0])
            xc = (x ^ (ax @ HX % 2)).astype(np.uint8)
            zc = (z ^ (az @ HZ % 2)).astype(np.uint8)
            if not xc.any() or not zc.any():
                continue
            if (
                int(np.count_nonzero(xc.astype(bool) & zc.astype(bool))) == 3
                and int(xc.sum()) <= 12
                and int(zc.sum()) <= 12
            ):
                return code, xc, zc
        raise ValueError("BLOCKED: no |W|=3 BB representative found in budget")
    raise ValueError(f"overlap must be 1 or 3, got {overlap}")


def _merged_incidence(
    g_x: GadgetLayout, g_z: GadgetLayout, x: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """Merged graph incidence ∂_1 (Ide, Gowda, Nadkarni, Dauphinais
    arXiv:2410.02753 Eq.(66)).

    Rows = vertices V_X ⊔ W ⊔ V_Z (data qubits of supp(x)/supp(z)); columns =
    edges (κ_X | κ_Z). ∂_1^x = g_x.incidence.T (rows=support, cols=κ_X); dual for
    ∂_1^z. The W rows stack the X- and Z-system incidences side by side, gluing
    the two graphs at the shared Y-qubits.
    """
    x = np.asarray(x).astype(np.uint8)
    z = np.asarray(z).astype(np.uint8)
    d1x = np.asarray(g_x.incidence).astype(np.uint8).T  # (|supp x|, k_x)
    d1z = np.asarray(g_z.incidence).astype(np.uint8).T  # (|supp z|, k_z)
    supx = list(g_x.support)
    supz = list(g_z.support)
    W = sorted(set(int(i) for i in np.where(x)[0]) & set(int(i) for i in np.where(z)[0]))
    VX = [v for v in supx if v not in W]
    VZ = [v for v in supz if v not in W]
    k_x = d1x.shape[1]
    k_z = d1z.shape[1]

    def rows_of(d: np.ndarray, sup: list[int], sel: list[int]) -> np.ndarray:
        idx = [sup.index(v) for v in sel]
        return d[idx] if idx else np.zeros((0, d.shape[1]), dtype=np.uint8)

    top = np.hstack([rows_of(d1x, supx, VX), np.zeros((len(VX), k_z), np.uint8)])
    mid = np.hstack([rows_of(d1x, supx, W), rows_of(d1z, supz, W)])
    bot = np.hstack([np.zeros((len(VZ), k_x), np.uint8), rows_of(d1z, supz, VZ)])
    return np.vstack([top, mid, bot]).astype(np.uint8), k_x, k_z


def _partial0_symplectic_rows(
    g_x: GadgetLayout,
    g_z: GadgetLayout,
    x: np.ndarray,
    z: np.ndarray,
    *,
    n: int,
    k_x: int,
    k_z: int,
) -> np.ndarray:
    """∂_0 = ker(merged ∂_1) as symplectic rows (Ide, Gowda, Nadkarni, Dauphinais
    arXiv:2410.02753 Eq.(67)/(68)).

    Column layout per half: [data (n) | κ_X | κ_Z]. A cycle c = (c_X | c_Z):
    its κ_X support enters as Z-part on κ_X (∂_0^(X)); its κ_Z support enters as
    X-part on κ_Z (∂_0^(Z)). A crossing cycle (|W|≥2) populates both → one
    non-CSS row.
    """
    D1, kx, kz = _merged_incidence(g_x, g_z, x, z)
    if (kx, kz) != (k_x, k_z):
        raise ValueError(f"incidence κ sizes {(kx, kz)} != ({k_x}, {k_z})")
    ker = np.asarray(GF2(D1.astype(int)).null_space()).astype(np.uint8)  # rows over (κ_X|κ_Z)
    nm = n + k_x + k_z
    out = np.zeros((ker.shape[0], 2 * nm), dtype=np.uint8)
    for i, c in enumerate(ker):
        c_x = c[:k_x]
        c_z = c[k_x:]
        out[i, n + k_x : nm] = c_z  # X-part on κ_Z  (∂_0^(Z))
        out[i, nm + n : nm + n + k_x] = c_x  # Z-part on κ_X  (∂_0^(X))
    return out


@dataclasses.dataclass(frozen=True)
class Obs0Row:
    """One merged-code check row contributing to the Ȳ readout product (obs0).

    Per Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.C: the obs0
    eigenvalue is the XOR of the in-circuit ancilla records of the merged-code
    rows whose product equals Ȳ on the original data columns. The picker
    :func:`_ybar_obs0_rows` solves over GF(2) for the rows whose product restricts
    to the literal Ȳ support ``[x | z]`` on data (X on V_X, Z on V_Z, Y on W) and
    is eigenbasis-compatible on the κ ancillas.

    Fields:
        sym_row — index of this row in ``YGadgetLayout.H_sym`` (equivalently in
                  ``merged_code.matrix``: ``build_y_gadget`` stacks them in the
                  same order, so the index is shared).
        family  — ``"X"`` (an X-type row, measured by an X-check ancilla),
                  ``"Z"`` (a Z-type row, measured by a Z-check ancilla),
                  or ``"Y"`` (the q1 mixed Y-stab row, measured by a y-ancilla MX).
        family_index — index of this row WITHIN its Pauli family, i.e. its
                  position among the merged code's pure-X rows (family ``"X"``),
                  pure-Z rows (``"Z"``), or mixed Y rows (``"Y"``). This is the
                  slot the circuit uses to look up the in-circuit ancilla record:
                  ``checks_x[family_index]`` / ``checks_z[family_index]`` /
                  ``y_ancilla_ids[family_index]``.
    """

    sym_row: int
    family: str
    family_index: int


@dataclasses.dataclass(frozen=True)
class Obs0ReadoutPlan:
    """How :func:`build_single_y_ppm_circuit` reads the DESTRUCTIVE Ȳ cross-check.

    NOTE on roles: the FAULT-TOLERANT physical readout (``obs0``) is the XOR of
    the selected rows' IN-CIRCUIT last-QEC-round ancilla outcomes (see
    :class:`Obs0Row` / :func:`_ybar_obs0_rows`). This plan describes the
    DESTRUCTIVE cross-check (``obs1``), the noiseless sibling of
    ``_surgery_observable``'s obs1: it reads the SAME §III.C product off the final
    destructive readouts. It is kept only as a cross-check (it collapses the
    data and is not a physical protocol on k>1 codes).

    The §III.C readout product (Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753
    §III.C) is a single merged-code stabilizer whose restriction to the original
    data columns equals the literal Ȳ support ``[x | z]``: Pauli X on
    ``V_X = supp(x) \\ W``, Pauli Z on ``V_Z = supp(z) \\ W``, and Pauli Y on
    ``W = supp(x) ∩ supp(z)``. Two facts shape how it is read:

      1. Sign / phase. ``Ȳ = iX̄Z̄`` carries an ``i``; the GF(2) row product
         only tracks Pauli SUPPORT, so the *signed* Pauli product of the chosen
         rows is ``-Y…`` (sign −1, computed with ``stim.PauliString``). XOR-ing
         the IN-CIRCUIT ancilla records of those rows is therefore deterministic
         but reads ``NOT(Ȳ eigenvalue)`` — a fixed global inversion. This is the
         honest, documented convention for ``obs0`` (a ``stim``
         ``OBSERVABLE_INCLUDE`` carries no constant offset to normalise it).
         Reading the SAME product off the destructive readouts (this plan)
         instead carries the physical Pauli-eigenvalue record signs, giving the
         un-inverted eigenvalue (Y+ → 0, Y− → 1) — the complement of ``obs0``.

      2. Eigenbasis. The product is deterministic only if every qubit's Pauli in
         it matches that qubit's prep/destructive-readout eigenbasis. With the
         ``[x | z]`` representative the data support is MIXED: V_X data → X (read
         ``MX``), V_Z data → Z (read ``M``), W data → Y (the Ȳ eigenstate, read
         ``MY``); κ_x → Z (``|0⟩``, read ``M``), κ_z → X (``|+⟩``, read ``MX``).
         :func:`_ybar_obs0_rows` solves over GF(2) for a row product meeting
         exactly this constraint.

    This plan stores, per data/ancilla qubit in the product's support, the Pauli
    to read there; the circuit maps each to its destructive record for ``obs1``.

    Fields:
        data_x   — original-data column indices read with ``MX`` (Pauli X, V_X).
        data_z   — original-data column indices read with ``M``  (Pauli Z, V_Z).
        data_y   — original-data column indices read with ``MY`` (Pauli Y, W).
        kx_z     — κ_x column indices (within ``[n0, n0+k_x)``) read with ``M`` (Z).
        kz_x     — κ_z column indices (within ``[n0+k_x, n_merged)``) read ``MX`` (X).
    """

    data_x: tuple[int, ...]
    data_z: tuple[int, ...]
    data_y: tuple[int, ...]
    kx_z: tuple[int, ...]
    kz_x: tuple[int, ...]


def _ybar_obs0_rows(
    H_sym: np.ndarray,
    code: CSSCode,
    x: np.ndarray,
    z: np.ndarray,
    *,
    n0: int,
    n_merged: int,
    k_x: int,
) -> tuple[tuple[Obs0Row, ...], Obs0ReadoutPlan]:
    """Pick the merged-code rows whose product is Ȳ = iX̄Z̄ (the obs0 readout).

    Solves over GF(2) for a row combination ``c`` such that the merged-code
    stabilizer ``S = cᵀ · H_sym``:

      * restricted to the original data columns equals the LITERAL Ȳ symplectic
        support ``[x | z]`` — Pauli X on ``V_X = supp(x) \\ W``, Pauli Z on
        ``V_Z = supp(z) \\ W``, Pauli Y on ``W = supp(x) ∩ supp(z)``. Because
        ``x`` is a logical-X representative and ``z`` a logical-Z representative,
        this product is automatically a logical-Y of the original qubit, so the
        old logical-Y commute/anticommute conditions are IMPLIED and dropped;
      * is measurable in the prepared/readout EIGENBASIS of every ANCILLA: κ_x
        carries Z only (``x_q == 0``, read ``|0⟩``), κ_z carries X only
        (``z_q == 0``, read ``|+⟩``), so the readout is DETERMINISTIC. The data
        eigenbasis is then mixed (X on V_X, Z on V_Z, Y on W) — the §III.C
        in-circuit readout, NOT an all-Y representative.

    This is the §III.C readout rule of Ide, Gowda, Nadkarni, Dauphinais
    arXiv:2410.02753 §III.C: the obs0 eigenvalue is the XOR of the in-circuit
    ancilla records of the merged-code rows whose product equals Ȳ on the data.
    The selected rows are exactly the new merge stabilizers (S_X' on V_X, S_Z' on
    V_Z, y_v on W), so obs0 is their BARE product. This literal ``[x | z]``
    support is feasible whenever Ȳ lies in the merged stabilizer center, and is
    deterministic in-circuit on a proper Ȳ-eigenstate codeword prep (the
    ``data_init="Y±"`` exact |Ȳ±⟩ = S̄|X̄±⟩ injection); an all-Y representative
    would only be deterministic on a physical ``∏_i|Y_i⟩`` prep and does not
    exist on a general code (e.g. BB ``[[36,8,4]]``), so it is not used.

    Returns
    -------
    rows
        The selected rows with per-row Pauli-family provenance
        (:class:`Obs0Row`).
    plan
        :class:`Obs0ReadoutPlan` listing, per support qubit of the product, the
        Pauli to read from its destructive record. The physical record signs
        give the correct Ȳ eigenvalue with no global offset.

    Raises ``ValueError`` if no such combination exists.
    """
    H = np.asarray(H_sym).astype(np.uint8)
    x = np.asarray(x).astype(np.uint8)
    z = np.asarray(z).astype(np.uint8)
    n_rows = H.shape[0]
    Hx = H[:, :n_merged]
    Hz = H[:, n_merged:]
    if n0 != code.num_qudits:
        raise ValueError(f"n0={n0} != code.num_qudits={code.num_qudits}")

    # obs0 = the BARE product of the new merge stabilizers (S_X' on V_X, S_Z' on
    # V_Z, y_v on W) = the literal [x | z] = iX̄Z̄ support. Solve A c = b for the
    # row-selection c such that the selected rows' product restricts to [x | z] on
    # data (X on V_X, Z on V_Z, Y on W) and is eigenbasis-compatible on the κ
    # ancillas (Z-only κ_x, X-only κ_z). Because x is a logical-X representative
    # and z a logical-Z representative, this product is automatically a logical-Y,
    # so the logical-Y commute/anticommute conditions are IMPLIED. On the exact
    # |Ȳ±⟩ codeword prep (|X̄+⟩ then transversal S) every code stabilizer is +1,
    # so this bare product agrees with Ȳ and its in-circuit XOR is DETERMINISTIC.
    # Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.C.
    a_rows: list[np.ndarray] = []
    b_vals: list[int] = []
    for q in range(n0):
        a_rows.append(Hx[:, q].copy())
        b_vals.append(int(x[q]))  # data X-part == x_q
        a_rows.append(Hz[:, q].copy())
        b_vals.append(int(z[q]))  # data Z-part == z_q
    # Ancilla eigenbasis: κ_x Z-only (X-part 0); κ_z X-only (Z-part 0).
    for q in range(n0, n0 + k_x):
        a_rows.append(Hx[:, q].copy())
        b_vals.append(0)
    for q in range(n0 + k_x, n_merged):
        a_rows.append(Hz[:, q].copy())
        b_vals.append(0)

    A = GF2(np.array(a_rows, dtype=np.uint8))
    b = GF2(np.array(b_vals, dtype=np.uint8).reshape(-1, 1))
    aug = GF2(np.hstack([np.asarray(A), np.asarray(b)]))
    rref = np.asarray(aug.row_reduce()).astype(np.uint8)
    c = np.zeros(n_rows, dtype=np.uint8)
    for r in range(rref.shape[0]):
        lead = np.flatnonzero(rref[r, :n_rows])
        if lead.size == 0:
            if rref[r, n_rows] == 1:
                raise ValueError(
                    "BLOCKED: no eigenbasis-compatible merged-code row product "
                    "equals Ȳ = iX̄Z̄ (Ide, Gowda, Nadkarni, Dauphinais "
                    "arXiv:2410.02753 §III.C)"
                )
            continue
        c[lead[0]] = rref[r, n_rows]

    S = (c @ H) % 2
    sx = S[:n_merged]
    sz = S[n_merged:]
    if not (np.array_equal(sx[:n0], x) and np.array_equal(sz[:n0], z)):
        raise ValueError("internal: obs0 product data part != Ȳ support [x|z]")
    if sx[n0 : n0 + k_x].any() or sz[n0 + k_x :].any():
        raise ValueError("internal: obs0 product not eigenbasis-compatible on κ ancillas")

    data_x = tuple(int(q) for q in range(n0) if sx[q] and not sz[q])
    data_z = tuple(int(q) for q in range(n0) if sz[q] and not sx[q])
    data_y = tuple(int(q) for q in range(n0) if sx[q] and sz[q])
    kx_z = tuple(int(q) for q in range(n0, n0 + k_x) if sz[q])
    kz_x = tuple(int(q) for q in range(n0 + k_x, n_merged) if sx[q])
    plan = Obs0ReadoutPlan(
        data_x=data_x, data_z=data_z, data_y=data_y, kx_z=kx_z, kz_x=kz_x
    )

    x_slot = z_slot = y_slot = 0
    rows: list[Obs0Row] = []
    for sym_row in range(n_rows):
        has_x = bool(Hx[sym_row].any())
        has_z = bool(Hz[sym_row].any())
        if has_x and has_z:
            family, family_index = "Y", y_slot
            y_slot += 1
        elif has_x:
            family, family_index = "X", x_slot
            x_slot += 1
        else:
            family, family_index = "Z", z_slot
            z_slot += 1
        if c[sym_row]:
            rows.append(Obs0Row(sym_row=sym_row, family=family, family_index=family_index))
    return tuple(rows), plan


@dataclasses.dataclass(frozen=True, eq=False)
class YGadgetLayout:
    """Merged code for a logical-Y measurement Ȳ = iX̄Z̄ (general ``|W|``).

    Output of :func:`build_y_gadget`, realising the homological logical-Y merge
    of Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.C/§III.D. The
    X-measurement system (the L=1 gadget
    of X̄) and the Z-measurement system (the L=1 gadget of Z̄) of ONE logical
    qubit are glued along the physical Y-support ``W = supp(x) ∩ supp(z)``: each
    ``v ∈ W`` carries a mixed-type check ``y_v`` obtained by the §III.D
    cross-merge of the S_X' row and the S_Z' row anchored at ``v``. The
    ``Y_stab`` rows are exactly these ``|W|`` symplectic Y rows.

    Per-system Cheeger boost (arXiv:2410.02753 §III.D). Each
    per-system gadget ``g_x`` / ``g_z`` is boosted (combinatorially) to boundary
    Cheeger constant ≥ 1 BEFORE merging, so the per-system distance argument
    holds. The boosted gadgets are stored, so ``k_x = len(g_x.Q_prime)``
    matches the merged-code width.

    ∂_0 block (no bridge). The merged check matrix appends ``∂_0 = ker(merged
    ∂_1)`` (arXiv:2410.02753 §III.D Eq.(66)/(67)), the cycle basis
    of the glued graph, in place of any per-system gauge blocks. At ``|W|=1`` the
    glued graph contributes no crossing cycle (0 ∂_0 rows); at ``|W|≥2`` it
    contributes ``|W|−1`` crossing rows. There is NO SkipTree adapter / bridge.

    Stabilizer code. The only pre-merge anticommutations — (S_X', S_Z') at each
    ``v ∈ W``, whose data supports cross — are consumed by the cross-merge into
    the ``y_v`` rows. The merged code is then a genuine STABILIZER code
    (``is_subsystem_code=False`` for Steane) encoding ``code.dimension − 1``
    logicals, with Ȳ = iX̄Z̄ in its stabilizer group.

    Fields:
        code        — the underlying CSS code (X̄ and Z̄ live on its qubits).
        x, z        — the logical-X / logical-Z supports (uint8); their overlap
                      ``supp(x) ∩ supp(z)`` is ``W``.
        W           — ``tuple[int, ...]`` of the data qubits in
                      ``supp(x) ∩ supp(z)`` (the physical Y qubits of Ȳ).
        g_x, g_z    — the per-system L=1 gadgets (Webster, Smith, Cohen
                      arXiv:2511.15989 §II.A) for X̄ (basis=X) and Z̄ (basis=Z),
                      each Cheeger-boosted to ≥ 1, built on the same ``code``
                      with disjoint κ-ancilla ranges.
        Y_stab      — ``(|W|, 2*n_merged)`` symplectic Y rows from the cross-
                      merge: the mixed checks ``y_v`` at each ``v ∈ W``.
        H_sym       — ``(rows, 2*n_merged)`` merged symplectic check matrix
                      = [HX_out | 0] ∪ [0 | HZ_out] ∪ Y_stab ∪ ∂_0. Column layout
                      in each half is ``[data (n) | κ_x | κ_z]``.
        merged_code — ``QuditCode(field(H_sym), is_subsystem_code=...)`` in
                      which Ȳ is in the measured stabilizer center; encodes one
                      fewer logical than ``code``.
        obs0_xor_map— the full Ȳ readout product (Ide, Gowda, Nadkarni,
                      Dauphinais arXiv:2410.02753 §III.D): the merged-code rows
                      whose eigenbasis-compatible GF(2) product equals the
                      symplectic Ȳ = [x | z] on the original data columns. Each
                      :class:`Obs0Row` records its ``H_sym`` row index plus
                      Pauli-family provenance (``"X"`` / ``"Z"`` / ``"Y"``, and
                      its index within that family). Solved over GF(2) by
                      :func:`_ybar_obs0_rows` — NOT hardcoded.
                      :func:`build_single_y_ppm_circuit` emits the FAULT-TOLERANT
                      ``obs0`` as the XOR of these rows' IN-CIRCUIT last-QEC-round
                      ancilla outcomes (family_index → ``checks_x`` / ``checks_z``
                      / ``y_ancilla_ids`` slot) — but ONLY when that XOR is
                      deterministic on the prepared state. The bare ``[x | z]``
                      product carries X on V_X / Z on V_Z data, so it is
                      deterministic exactly on a proper Ȳ-eigenstate codeword: the
                      ``data_init="Y±"`` prep injects the EXACT |Ȳ±⟩ = S̄|X̄±⟩
                      codeword, on which every code stabilizer is +1 and obs0 is
                      emitted; on a non-eigenstate prep (|0̄⟩/|+̄⟩) it is a genuine
                      50/50 and gated off (unless ``force_obs0``). The same bare
                      representative is the only feasible one on a general code (BB
                      [[36,8,4]], where an all-Y representative does not exist).
                      The bare product equals Ȳ = iX̄Z̄ EXACTLY (for Steane
                      [x | z] = X₂X₄Z₁Z₃Y₅ and iX̄Z̄ = +X₂X₄Z₁Z₃Y₅), so the raw
                      obs0 bit IS the Ȳ eigenvalue bit: Y+ → 0, Y− → 1.
        obs0_readout— :class:`Obs0ReadoutPlan`: the DESTRUCTIVE cross-check
                      (``obs1``, NOT the physical readout). Per support-qubit it
                      gives the Pauli to read off the final destructive readout:
                      V_X data → X, V_Z data → Z, W data → Y, κ_x → Z, κ_z → X.
                      This reads the SAME Ȳ = [x | z] product as ``obs0`` off the
                      destructive records, so it carries the same eigenvalue
                      (Y+ → 0, Y− → 1). :func:`build_single_y_ppm_circuit` uses it
                      in ``benchmark_y`` as ``obs0 ⊕ obs1`` (deterministic on ANY
                      input — both equal Ȳ); as a standalone ``obs1`` it is emitted
                      only when its destructive basis matches the readout
                      (gated by ``_observable_is_deterministic``).
    """

    code: CSSCode
    x: np.ndarray
    z: np.ndarray
    W: tuple[int, ...]
    g_x: GadgetLayout
    g_z: GadgetLayout
    Y_stab: np.ndarray
    H_sym: np.ndarray
    merged_code: QuditCode
    obs0_xor_map: tuple[Obs0Row, ...]
    obs0_readout: Obs0ReadoutPlan


def build_y_gadget(code: CSSCode, *, x: np.ndarray, z: np.ndarray) -> YGadgetLayout:
    """Assemble the homological logical-Y merged code for Ȳ = iX̄Z̄ (general ``|W|``).

    Implements the Eq.(68) merged code of Ide, Gowda, Nadkarni, Dauphinais
    arXiv:2410.02753 §III.C/§III.D:
    glue the X-measurement L=1 gadget of ``x`` and the Z-measurement L=1 gadget
    of ``z`` (both Webster, Smith, Cohen arXiv:2511.15989 §II.A) along the
    physical Y-support ``W = supp(x) ∩ supp(z)``. At each ``v ∈ W`` the S_X' row
    and S_Z' row anchored there are fused into one explicit mixed check ``y_v``
    via the §III.D cross-merge (NOT a single-qubit basis-change rotation — the
    non-CSS content lives in the ``y_v`` rows).

    Per-system Cheeger boost (arXiv:2410.02753 §III.D: per-system
    distance argument). Each gadget is boosted to boundary Cheeger constant ≥ 1
    BEFORE merging; the boosted gadgets are returned so the merged-code width
    ``n + k_x + k_z`` is consistent with ``len(g_x.Q_prime)``.

    The merged symplectic column space is ``[data (n) | κ_x | κ_z]``. The two
    systems share the data qubits, so the original code stabilizers appear ONCE:
    H_X is extended onto κ_z so it commutes with S_Z', and H_Z onto κ_x so it
    commutes with S_X' (the dual-extension of Webster §II.A step 3). NO per-system
    gauge rows are appended; instead ``∂_0 = ker(merged ∂_1)`` (the cycle basis
    of the glued graph, §III.D Eq.(66)/(67)) is appended. The only pairs that
    anticommute before the merge are (S_X', S_Z') at each ``v ∈ W``; fusing them
    removes that anticommutation, so the merged code is a genuine stabilizer code
    encoding ``code.dimension − 1`` logicals with Ȳ = iX̄Z̄ in its stabilizer
    group.

    The symplectic check matrix H̃ (Ide, Gowda, Nadkarni, Dauphinais
    arXiv:2410.02753 §III.D) is assembled in formula order with one named
    local variable per block.  Column layout per symplectic half:
    ``[ data (n) | κ_x (k_x) | κ_z (k_z) ]``::

        |   X-part (data | κ_x | κ_z)         |   Z-part (data | κ_x | κ_z)
        |--------------------------------------|------------------------------------
        |  H_X     0       π_{C₀^Z}^T         |   ·       ·        ·          block 1 (X-checks)
        |  π_{V_X} ∂₁ˣ|_{V_X}  0              |   ·       ·        ·          block 2 (S_X' on V_X)
        |  π_W     ∂₁ˣ|_W   0                 |  π_W      0       ∂₁ᶻ|_W      block 3 (Y on W, mixed)
        |  ·       ·        ·                 |  H_Z    π_{C₀^X}^T  0         block 4 (Z-checks)
        |  ·       ·        ·                 |  π_{V_Z}  0       ∂₁ᶻ|_{V_Z}  block 5 (S_Z' on V_Z)
        |  0       0        ∂₀^Z              |   0      ∂₀^X      0          block 6 (cycles ∂₀)

    Named-block mapping (each is a local variable in the assembly)::

        Xcheck_rows = HX_out[:m_x]   # block 1: H_X ext onto κ_z = π_{C₀^Z}^T
        SXprime_rows = HX_out[m_x:]  # block 2: π_{V_X} | ∂₁ˣ|_{V_X} | 0
        Ymix_rows   = Y_stab         # block 3: mixed S'@W rows (apply_mixed_basis_merge)
        Zcheck_rows = HZ_out[:m_z]   # block 4: H_Z ext onto κ_x = π_{C₀^X}^T
        SZprime_rows = HZ_out[m_z:]  # block 5: π_{V_Z} | 0 | ∂₁ᶻ|_{V_Z}
        cycle_rows  = partial0        # block 6: ker(merged ∂₁) = ∂₀^Z / ∂₀^X

    Args:
        code: a CSS code carrying the logical qubit to be Y-measured.
        x: logical-X support, ``H_Z @ x == 0`` (mod 2); shape ``(n,)``.
        z: logical-Z support, ``H_X @ z == 0`` (mod 2); shape ``(n,)``.

    Returns:
        A :class:`YGadgetLayout`. ``ValueError`` is raised (via
        :func:`_locate_overlaps`) if ``x``/``z`` are not valid anticommuting
        logicals (``H_Z @ x = 0``, ``H_X @ z = 0``, ``x · z`` odd).
    """
    x = np.asarray(x).astype(np.uint8)
    z = np.asarray(z).astype(np.uint8)
    W = _locate_overlaps(code, x, z)

    # Per-system L=1 gadgets, each Cheeger-boosted to ≥1
    # (arXiv:2410.02753 §III.D: per-system distance argument).
    from .cheeger import boost_gadget, cheeger_constant

    g_x = build_gadget(code, x, basis=Pauli.X)
    g_z = build_gadget(code, z, basis=Pauli.Z)
    if cheeger_constant(g_x) < 1.0:
        g_x = boost_gadget(g_x, method="combinatorial", target=1.0, seed=0)
    if cheeger_constant(g_z) < 1.0:
        g_z = boost_gadget(g_z, method="combinatorial", target=1.0, seed=0)

    field = code.field
    n = code.num_qudits
    m_x = int(np.asarray(code.matrix_x).shape[0])
    m_z = int(np.asarray(code.matrix_z).shape[0])
    k_x = len(g_x.Q_prime)
    k_z = len(g_z.Q_prime)
    n_merged = n + k_x + k_z

    # S_X' / S_Z' generator + extension blocks (Webster §II.A step 3 decomposition).
    SX_prime = np.asarray(g_x.HX_merged[m_x:]).astype(np.uint8)  # (|V0x|, n+k_x)
    hz_ext_kx = np.asarray(g_x.HZ_merged[:m_z]).astype(np.uint8)  # [H_Z | f_0^x]
    SZ_prime = np.asarray(g_z.HZ_merged[m_z:]).astype(np.uint8)  # (|V0z|, n+k_z)
    hx_ext_kz = np.asarray(g_z.HX_merged[:m_x]).astype(np.uint8)  # [H_X | f_0^z]

    def _embed(rows, *, data=None, kx=None, kz=None):
        out = np.zeros((rows, n_merged), dtype=np.uint8)
        if data is not None:
            out[:, :n] = data
        if kx is not None:
            out[:, n : n + k_x] = kx
        if kz is not None:
            out[:, n + k_x :] = kz
        return out

    # X-type rows: H_X extended onto κ_z; S_X' on [data | κ_x]. (NO per-system gauge.)
    HX_all = np.vstack(
        [
            _embed(m_x, data=hx_ext_kz[:, :n], kz=hx_ext_kz[:, n:]),
            _embed(SX_prime.shape[0], data=SX_prime[:, :n], kx=SX_prime[:, n:]),
        ]
    ).astype(np.uint8)
    # Z-type rows: H_Z extended onto κ_x; S_Z' on [data | κ_z].
    HZ_all = np.vstack(
        [
            _embed(m_z, data=hz_ext_kx[:, :n], kx=hz_ext_kx[:, n:]),
            _embed(SZ_prime.shape[0], data=SZ_prime[:, :n], kz=SZ_prime[:, n:]),
        ]
    ).astype(np.uint8)

    # Merge S_X'@v / S_Z'@v into one mixed y_v row for every v ∈ W (§III.D).
    HX_out, HZ_out, Y_stab, _obs0_y, _xl, _zl = apply_mixed_basis_merge(
        HX_all,
        HZ_all,
        merge_qubits=W,
        adapter_cols=tuple(range(n)),
    )
    HX_out = np.asarray(HX_out).astype(np.int_)
    HZ_out = np.asarray(HZ_out).astype(np.int_)
    if Y_stab is None or Y_stab.shape[0] < len(W):
        raise ValueError(
            f"BLOCKED: cross-merge produced {0 if Y_stab is None else Y_stab.shape[0]} "
            f"y_v rows, expected |W|={len(W)} (arXiv:2410.02753 §III.D)"
        )

    # ∂_0 = ker(merged ∂_1): cycle basis of the glued graph (§III.D Eq.66/67),
    # replacing the per-system gauge blocks. 0 rows at |W|=1, |W|−1 crossing rows
    # at |W|≥2.
    partial0 = _partial0_symplectic_rows(g_x, g_z, x, z, n=n, k_x=k_x, k_z=k_z)

    # --- Assemble H̃ block-by-block in formula order (Ide, Gowda, Nadkarni,
    # Dauphinais arXiv:2410.02753 §III.D; design spec §0). Column layout per
    # symplectic half: [ data (n) | κ_x (k_x) | κ_z (k_z) ].

    def _sym_x(rows: np.ndarray) -> np.ndarray:  # X-only rows → [X | 0]
        return np.hstack([rows, np.zeros_like(rows)]).astype(np.int_)

    def _sym_z(rows: np.ndarray) -> np.ndarray:  # Z-only rows → [0 | Z]
        return np.hstack([np.zeros_like(rows), rows]).astype(np.int_)

    # apply_mixed_basis_merge removed the S'@W rows into Y_stab, so:
    Xcheck_rows = HX_out[:m_x]   # block 1: [H_X | 0 | π_{C₀^Z}^T]
    SXprime_rows = HX_out[m_x:]  # block 2: [π_{V_X} | ∂₁ˣ|_{V_X} | 0]
    Ymix_rows   = Y_stab         # block 3: [π_W|∂₁ˣ|_W|0 ‖ π_W|0|∂₁ᶻ|_W]
    Zcheck_rows = HZ_out[:m_z]   # block 4: [H_Z | π_{C₀^X}^T | 0]
    SZprime_rows = HZ_out[m_z:]  # block 5: [π_{V_Z} | 0 | ∂₁ᶻ|_{V_Z}]
    cycle_rows  = partial0       # block 6: [0|0|∂₀^Z ‖ 0|∂₀^X|0]

    blocks = [
        _sym_x(Xcheck_rows),
        _sym_x(SXprime_rows),
        Ymix_rows.astype(np.int_),
        _sym_z(Zcheck_rows),
        _sym_z(SZprime_rows),
        cycle_rows.astype(np.int_),
    ]
    H_sym = (
        np.vstack([b for b in blocks if b.shape[0]])
        if any(b.shape[0] for b in blocks)
        else np.zeros((0, 2 * n_merged), dtype=np.int_)
    )

    Hx = H_sym[:, :n_merged]
    Hz = H_sym[:, n_merged:]
    comm = (Hx @ Hz.T + Hz @ Hx.T) % 2
    np.fill_diagonal(comm, 0)
    merged_code = QuditCode(field(H_sym), is_subsystem_code=bool(comm.any()))

    obs0_rows, obs0_readout = _ybar_obs0_rows(
        H_sym, code, x, z, n0=n, n_merged=n_merged, k_x=k_x
    )
    return YGadgetLayout(
        code=code,
        x=x,
        z=z,
        W=W,
        g_x=g_x,
        g_z=g_z,
        Y_stab=Y_stab,
        H_sym=H_sym,
        merged_code=merged_code,
        obs0_xor_map=obs0_rows,
        obs0_readout=obs0_readout,
    )
