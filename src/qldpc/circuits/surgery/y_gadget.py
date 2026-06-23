"""Logical-Y measurement gadget primitives (Ȳ = iX̄Z̄).

Building blocks for measuring a logical Y operator via lattice surgery, following
the homological measurement construction of Ide, Gowda, Nadkarni, Dauphinais
arXiv:2410.02753 §III.C/§III.D (see also docs/superpowers/docs/main.tex §4). The
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
    qubits, the core correctness guarantee of the single-overlap logical-Y
    merge of Cross, He, Rall, Yoder arXiv:2407.18393 §3.7.
    """
    M2 = GF2(np.asarray(M).astype(np.uint8))
    A = GF2(np.vstack([np.asarray(M).astype(np.uint8), np.asarray(v).astype(np.uint8)[None, :]]))
    return int(np.linalg.matrix_rank(M2)) == int(np.linalg.matrix_rank(A))


def _locate_overlap(code: CSSCode, x: np.ndarray, z: np.ndarray) -> int:
    """Return the single data qubit shared by logical-X support ``x`` and logical-Z support ``z``.

    Implements the single-overlap precondition of Cross, He, Rall, Yoder
    arXiv:2407.18393 §3.7: Ȳ = iX̄Z̄ is realised cleanly when the X̄ and Z̄
    strings cross on exactly one data qubit.

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
            "(Cross, He, Rall, Yoder arXiv:2407.18393 §3.7 single-overlap case)"
        )
    return int(overlap[0])


def _locate_overlaps(code: CSSCode, x: np.ndarray, z: np.ndarray) -> tuple[int, ...]:
    """Return W = supp(x) ∩ supp(z), the physical Pauli-Y qubits of Ȳ = iX̄Z̄.

    Validates (Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.D;
    docs/superpowers/docs/main.tex §4.1):
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

    This is the clean single-overlap fixture for Ȳ = iX̄Z̄ of Cross, He, Rall,
    Yoder arXiv:2407.18393 §3.7. The canonical weight-3 logical-X and logical-Z
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
        "stabilizer-row reduction; the out-of-scope multi-overlap §3.7 path would be "
        "required (Cross, He, Rall, Yoder arXiv:2407.18393 §3.7)"
    )


def _overlap_size(x: np.ndarray, z: np.ndarray) -> int:
    """Number of data qubits in ``supp(x) ∩ supp(z)``."""
    return int(np.count_nonzero(x.astype(bool) & z.astype(bool)))


def _bb_y_pair(overlap: int = 1) -> tuple[CSSCode, np.ndarray, np.ndarray]:
    """BB [[36,8,4]] fixture for Ȳ on logical qubit 0 with chosen |W|.

    overlap=1: canonical reps (already single-overlap). overlap=3: add stabilizer
    rows (deterministic seed) until supp(x)∩supp(z) has size 3 — the |W|≥2
    crossing-cycle regime of arXiv:2410.02753 §III.D / main.tex §4.
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
    """Merged graph incidence ∂_1 (arXiv:2410.02753 Eq.(66); main.tex §4.4).

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
    """∂_0 = ker(merged ∂_1) as symplectic rows (arXiv:2410.02753 Eq.(67)/(68)).

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

    Per Cross, He, Rall, Yoder arXiv:2407.18393 §3.2 (lines 562-563): the
    logical-measurement readout is the parity of ALL appended checks whose
    product is the logical operator. For Ȳ = iX̄Z̄ that product is the surviving
    χ_X rows ⊕ surviving χ_Z rows ⊕ the q1 Y-stab row (arXiv:2407.18393 line
    2433: ``X̄_M Z̄_M`` is the product of all interface + module checks), plus
    code stabilizers chosen so the product is measurable in the prepared/readout
    eigenbasis (see :func:`_ybar_obs0_rows`).

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
    ``_surgery_observable``'s obs1: it reads the SAME §3.2 product off the final
    destructive readouts. It is kept only as a cross-check (it collapses the
    data and is not a physical protocol on k>1 codes).

    The §3.2 readout product (Cross, He, Rall, Yoder arXiv:2407.18393 §3.2 lines
    562-563) is a single merged-code stabilizer equal to Ȳ on the logical qubit.
    Two facts (verified empirically on the Steane fixture) shape how it is read:

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
         it matches that qubit's prep/destructive-readout eigenbasis: data → Y
         (the Ȳ eigenstate, read ``MY``), κ_x → Z (``|0⟩``, read ``M``), κ_z → X
         (``|+⟩``, read ``MX``). :func:`_ybar_obs0_rows` solves over GF(2) for a
         row product meeting exactly this constraint (an "all-Y on data"
         representative).

    This plan stores, per data/ancilla qubit in the product's support, the Pauli
    to read there; the circuit maps each to its destructive record for ``obs1``.

    Fields:
        data_y   — original-data column indices read with ``MY`` (Pauli Y).
        kx_z     — κ_x column indices (within ``[n0, n0+k_x)``) read with ``M`` (Z).
        kz_x     — κ_z column indices (within ``[n0+k_x, n_merged)``) read ``MX`` (X).
    """

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

      * equals Ȳ = iX̄Z̄ on the original logical qubit — i.e. its restriction to
        the original data columns is a logical-Y representative (commutes with
        every code Z-stabilizer, anticommutes with the logical-Z support ``z``);
      * is measurable in the prepared/readout EIGENBASIS of every qubit, so the
        readout is DETERMINISTIC: data → Y (the Ȳ eigenstate), κ_x → Z (``|0⟩``),
        κ_z → X (``|+⟩``). Concretely ``S``'s symplectic vector obeys
        ``x_q == z_q`` on data (all-Y), ``x_q == 0`` on κ_x (Z-only),
        ``z_q == 0`` on κ_z (X-only).

    This is the §3.2 readout rule (Cross, He, Rall, Yoder arXiv:2407.18393 §3.2
    lines 562-563: output the parity of all appended checks whose product is the
    logical; arXiv:2407.18393 line 2433: ``X̄_M Z̄_M`` = product of interface +
    module checks), made concrete: the selected rows are the surviving χ_X / χ_Z
    rows, the q1 Y-stab row, AND whatever code stabilizers are needed to rotate
    the product into the eigenbasis. WITHOUT the eigenbasis constraint the bare
    χ_X ⊕ χ_Z ⊕ q1 product is non-deterministic (a Y representative that mixes
    X/Z paulis on Y-eigenbasis data qubits) and would suppress obs0.

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
    HZc = np.asarray(code.matrix_z).astype(np.uint8)  # code Z-stabilizers (m_z, n0)

    # Assemble the GF(2) constraint system  A c = b  on the row-selection c.
    a_rows: list[np.ndarray] = []
    b_vals: list[int] = []

    # Eigenbasis constraints on S = c·H:
    #   data q in [0, n0):       x_q == z_q   →  (Hx[:,q] ^ Hz[:,q]) · c = 0
    #   κ_x  q in [n0, n0+k_x):  x_q == 0     →  Hx[:,q] · c = 0
    #   κ_z  q in [n0+k_x, n):   z_q == 0     →  Hz[:,q] · c = 0
    for q in range(n0):
        a_rows.append(Hx[:, q] ^ Hz[:, q])
        b_vals.append(0)
    for q in range(n0, n0 + k_x):
        a_rows.append(Hx[:, q].copy())
        b_vals.append(0)
    for q in range(n0 + k_x, n_merged):
        a_rows.append(Hz[:, q].copy())
        b_vals.append(0)

    # Logical-Y constraints on the data X-part w_q = x_q of S (= z_q there):
    #   commute with every code Z-stabilizer:  HZc · w = 0
    #   anticommute with logical-Z support z:  z · w = 1   (so S is a logical-Y,
    #                                                        not a stabilizer/identity)
    for r in range(HZc.shape[0]):
        coef = np.zeros(n_rows, dtype=np.uint8)
        for q in range(n0):
            if HZc[r, q]:
                coef ^= Hx[:, q]
        a_rows.append(coef)
        b_vals.append(0)
    coef = np.zeros(n_rows, dtype=np.uint8)
    for q in range(n0):
        if z[q]:
            coef ^= Hx[:, q]
    a_rows.append(coef)
    b_vals.append(1)

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
                    "BLOCKED: no eigenbasis-compatible merged-code row product equals "
                    "Ȳ = iX̄Z̄; the §3.2 deterministic readout does not exist (Cross, "
                    "He, Rall, Yoder arXiv:2407.18393 §3.2)"
                )
            continue
        c[lead[0]] = rref[r, n_rows]

    # Resolve the product S = c·H and certify it (eigenbasis + Ȳ-on-data).
    S = (c @ H) % 2
    sx = S[:n_merged]
    sz = S[n_merged:]
    # data X-part must equal data Z-part (all-Y) and be a logical-Y representative.
    if not np.array_equal(sx[:n0], sz[:n0]):
        raise ValueError("internal: obs0 product is not all-Y on data")
    if ((HZc @ sx[:n0]) % 2 != 0).any() or int(np.dot(sx[:n0], z)) % 2 != 1:
        raise ValueError("internal: obs0 product data part is not a logical-Y representative")
    if sx[n0 : n0 + k_x].any() or sz[n0 + k_x :].any():
        raise ValueError("internal: obs0 product is not eigenbasis-compatible on κ ancillas")

    # Build the destructive-readout plan from the product's support.
    data_y = tuple(int(q) for q in range(n0) if sx[q])
    kx_z = tuple(int(q) for q in range(n0, n0 + k_x) if sz[q])
    kz_x = tuple(int(q) for q in range(n0 + k_x, n_merged) if sx[q])
    plan = Obs0ReadoutPlan(data_y=data_y, kx_z=kx_z, kz_x=kz_x)

    # Per-row provenance, classified the same way the circuit re-partitions the
    # merged-code matrix (``_split_quditcode_into_virtual_cssc``).
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
    of Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.C/§III.D (see also
    docs/superpowers/docs/main.tex §4). The X-measurement system (the L=1 gadget
    of X̄) and the Z-measurement system (the L=1 gadget of Z̄) of ONE logical
    qubit are glued along the physical Y-support ``W = supp(x) ∩ supp(z)``: each
    ``v ∈ W`` carries a mixed-type check ``y_v`` obtained by the §III.D
    cross-merge of the χ_X row and the χ_Z row anchored at ``v``. The
    ``Y_stab`` rows are exactly these ``|W|`` symplectic Y rows.

    Per-system Cheeger boost (main.tex §4.7 / arXiv:2410.02753 §III.D). Each
    per-system gadget ``g_x`` / ``g_z`` is boosted (combinatorially) to boundary
    Cheeger constant ≥ 1 BEFORE merging, so the per-system distance argument
    holds. The boosted gadgets are stored, so ``k_x = len(g_x.ancilla_qubits)``
    matches the merged-code width.

    ∂_0 block (no bridge). The merged check matrix appends ``∂_0 = ker(merged
    ∂_1)`` (arXiv:2410.02753 §III.D Eq.(66)/(67); main.tex §4.4), the cycle basis
    of the glued graph, in place of any per-system gauge blocks. At ``|W|=1`` the
    glued graph contributes no crossing cycle (0 ∂_0 rows); at ``|W|≥2`` it
    contributes ``|W|−1`` crossing rows. There is NO SkipTree adapter / bridge.

    Stabilizer code. The only pre-merge anticommutations — (χ_X, χ_Z) at each
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
                      / ``y_ancilla_ids`` slot). That readout is DETERMINISTIC but
                      measures −Ȳ (the GF(2) product drops the ``i`` of iX̄Z̄), so
                      the raw obs0 bit is NOT(Ȳ bit): Y+ → 1, Y− → 0 — handled by
                      a documented sign convention.
        obs0_readout— :class:`Obs0ReadoutPlan`: the DESTRUCTIVE cross-check
                      (``obs1``, NOT the physical readout). Per support-qubit it
                      gives the Pauli to read off the final destructive readout
                      (data → Y, κ_x → Z, κ_z → X). Those record signs carry the
                      physical Ȳ = iX̄Z̄ phase, so :func:`build_single_y_ppm_circuit`
                      emits it as ``obs1`` with the un-inverted eigenvalue
                      (Y+ → 0, Y− → 1) — the complement of the in-circuit ``obs0``
                      (whose signed product is −Ȳ; see ``obs0_xor_map``).
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
    arXiv:2410.02753 §III.C/§III.D (see also docs/superpowers/docs/main.tex §4):
    glue the X-measurement L=1 gadget of ``x`` and the Z-measurement L=1 gadget
    of ``z`` (both Webster, Smith, Cohen arXiv:2511.15989 §II.A) along the
    physical Y-support ``W = supp(x) ∩ supp(z)``. At each ``v ∈ W`` the χ_X row
    and χ_Z row anchored there are fused into one explicit mixed check ``y_v``
    via the §III.D cross-merge (NOT a single-qubit basis-change rotation — the
    non-CSS content lives in the ``y_v`` rows).

    Per-system Cheeger boost (main.tex §4.7 / arXiv:2410.02753 §III.D: per-system
    distance argument). Each gadget is boosted to boundary Cheeger constant ≥ 1
    BEFORE merging; the boosted gadgets are returned so the merged-code width
    ``n + k_x + k_z`` is consistent with ``len(g_x.ancilla_qubits)``.

    The merged symplectic column space is ``[data (n) | κ_x | κ_z]``. The two
    systems share the data qubits, so the original code stabilizers appear ONCE:
    H_X is extended onto κ_z so it commutes with χ_Z, and H_Z onto κ_x so it
    commutes with χ_X (the dual-extension of Webster §II.A step 3). NO per-system
    gauge rows are appended; instead ``∂_0 = ker(merged ∂_1)`` (the cycle basis
    of the glued graph, §III.D Eq.(66)/(67)) is appended. The only pairs that
    anticommute before the merge are (χ_X, χ_Z) at each ``v ∈ W``; fusing them
    removes that anticommutation, so the merged code is a genuine stabilizer code
    encoding ``code.dimension − 1`` logicals with Ȳ = iX̄Z̄ in its stabilizer
    group.

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

    # Per-system L=1 gadgets, each Cheeger-boosted to ≥1 (main.tex §4.7 /
    # arXiv:2410.02753 §III.D: per-system distance argument).
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
    k_x = len(g_x.ancilla_qubits)
    k_z = len(g_z.ancilla_qubits)
    n_merged = n + k_x + k_z

    # χ / extension blocks (Webster §II.A step 3 decomposition).
    chi_x = np.asarray(g_x.HX_merged[m_x:]).astype(np.uint8)  # (|V0x|, n+k_x)
    hz_ext_kx = np.asarray(g_x.HZ_merged[:m_z]).astype(np.uint8)  # [H_Z | F̃_x]
    chi_z = np.asarray(g_z.HZ_merged[m_z:]).astype(np.uint8)  # (|V0z|, n+k_z)
    hx_ext_kz = np.asarray(g_z.HX_merged[:m_x]).astype(np.uint8)  # [H_X | F̃_z]

    def _embed(rows, *, data=None, kx=None, kz=None):
        out = np.zeros((rows, n_merged), dtype=np.uint8)
        if data is not None:
            out[:, :n] = data
        if kx is not None:
            out[:, n : n + k_x] = kx
        if kz is not None:
            out[:, n + k_x :] = kz
        return out

    # X-type rows: H_X extended onto κ_z; χ_X on [data | κ_x]. (NO per-system gauge.)
    HX_all = np.vstack(
        [
            _embed(m_x, data=hx_ext_kz[:, :n], kz=hx_ext_kz[:, n:]),
            _embed(chi_x.shape[0], data=chi_x[:, :n], kx=chi_x[:, n:]),
        ]
    ).astype(np.uint8)
    # Z-type rows: H_Z extended onto κ_x; χ_Z on [data | κ_z].
    HZ_all = np.vstack(
        [
            _embed(m_z, data=hz_ext_kx[:, :n], kx=hz_ext_kx[:, n:]),
            _embed(chi_z.shape[0], data=chi_z[:, :n], kz=chi_z[:, n:]),
        ]
    ).astype(np.uint8)

    # Merge χ_X@v / χ_Z@v into one mixed y_v row for every v ∈ W (§III.D / §4.3).
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

    rows_sym: list[np.ndarray] = []
    for r in HX_out:
        rows_sym.append(np.concatenate([r, np.zeros(n_merged, dtype=np.int_)]))
    for r in HZ_out:
        rows_sym.append(np.concatenate([np.zeros(n_merged, dtype=np.int_), r]))
    for r in Y_stab:
        rows_sym.append(r.astype(np.int_))
    for r in partial0:
        rows_sym.append(r.astype(np.int_))
    H_sym = (
        np.array(rows_sym, dtype=np.int_)
        if rows_sym
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
