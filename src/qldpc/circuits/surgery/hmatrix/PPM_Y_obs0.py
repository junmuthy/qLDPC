"""Ȳ readout planner for the homological logical-Y merge (obs0 row picker).

The destructive/in-circuit Ȳ = iX̄Z̄ readout product following the homological
measurement construction of Ide, Gowda, Nadkarni, Dauphinais
arXiv:2410.02753 §III.D. The obs0 eigenvalue is the XOR of the in-circuit ancilla
records of the merged-code rows whose product restricts to the literal Ȳ support
``[x | z]`` on the original data columns. This module is self-contained: the
:func:`_ybar_obs0_rows` picker solves over GF(2) for that row combination and
calls none of the Y core helpers.

This module provides:
    Obs0Row          — one merged-code row contributing to the Ȳ readout product.
    Obs0ReadoutPlan  — how the destructive Ȳ cross-check (obs1) is read.
    _ybar_obs0_rows  — the GF(2) picker for the obs0 row product.
"""

from __future__ import annotations

import dataclasses

import galois
import numpy as np

from qldpc.codes.common import CSSCode

GF2 = galois.GF(2)


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
