"""Single-Ȳ surgery: shared context + state prep (split from the former y_circuit.py).

The foundation module of the Y trio: the cross-phase ``_YCtx`` dataclass, the
exact logical-Y eigenstate injection, the QUBIT_COORDS + state-prep phase, and
the virtual-CSS / center-mask helpers it relies on.

References:
    Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.C/§III.D
        — homological Ȳ = iX̄Z̄ measurement.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import stim

from qldpc.circuits.bookkeeping import QubitIDs
from qldpc.codes.common import CSSCode, QuditCode

from ..hmatrix.PPM_Y import YGadgetLayout


@dataclasses.dataclass
class _YCtx:
    """Cross-phase state for the Ȳ-PPM emitter.

    All fields are set by ``_y_state_prep`` and consumed by the subsequent
    phase functions.  The dataclass replaces the closure variables that
    previously lived inside ``build_single_y_ppm_circuit``.
    """

    # The merged subsystem code and its qubit layout.
    merged_code: QuditCode
    qubit_ids: QubitIDs
    # Virtual CSS sub-codes (pure-X / pure-Z subsets of merged_code.matrix).
    virtual_cssc_X: CSSCode
    virtual_cssc_Z: CSSCode
    # Merged-code column slices: data[0:n_code], κ_x [n_code:n_code+k_x], κ_z [n_code+k_x:].
    real_data_ids: tuple[int, ...]
    kx_ids: tuple[int, ...]
    kz_ids: tuple[int, ...]
    # Y-row ancilla IDs (one per mixed row in the merged stabiliser matrix).
    y_ancilla_ids: tuple[int, ...]
    # Dimension constants.
    n_code: int
    k_x: int
    k_z: int
    n_q: int
    # Stabiliser-center mask (True ↔ row commutes with every other row).
    center_mask: np.ndarray
    # Full symplectic stabiliser matrix (all rows, shape (n_rows, 2*n_q)).
    H_full: np.ndarray
    # Row-index partitions of H_full: pure-X, pure-Z, mixed-Y rows.
    x_row_idx: list[int]
    z_row_idx: list[int]
    mixed_row_idx: list[int]
    # Logical-representative column support (flat column indices into data qubits).
    x_cols: tuple[int, ...]
    z_cols: tuple[int, ...]


def _steane_logical_y_eigenstate_prep(
    yg: YGadgetLayout,
    real_data_ids: tuple[int, ...],
    *,
    data_init: str,
    ancilla_base: int,
) -> stim.Circuit:
    """Prepare the EXACT logical-Y eigenstate codeword |Ȳ±⟩ = S̄|X̄±⟩.

    The correct preparation of a logical-Y eigenstate is the codeword |X̄+⟩
    followed by the transversal phase gate S̄ — NOT the physical product
    ``∏_i |Y_i+⟩`` (which is a +1 eigenstate of every single-qubit Y_i but is
    NOT a codeword: the original code stabilizers are random on it). The
    distinction is decisive for the Ȳ readout: on the physical product the bare
    Ȳ representative ``[x | z]`` (= the product of the merged code's new
    S_X'·S_Z'·y_v rows) is non-deterministic, because ``[x | z]`` differs from an
    all-Y representative by code stabilizers that are random there. On the
    proper codeword every code stabilizer is +1, so all Ȳ representatives agree
    and the in-circuit ``[x | z]`` readout is DETERMINISTIC. The bare support is
    the LITERAL Ȳ = iX̄Z̄: for Steane ``[x | z] = X₂X₄Z₁Z₃Y₅``, and
    ``iX̄Z̄ = +X₂X₄Z₁Z₃Y₅`` (the ``i`` cancels the ``X₅Z₅ = −iY₅`` phase), so
    ⟨[x | z]⟩ = ⟨Ȳ⟩ EXACTLY — no sign convention.

    Construction (exact, noiseless codeword injection):
        RX^n                 → |+⟩^n  (X̄ = +1, X-syndrome 0, Z-syndrome random)
        measure H_Z, correct → |X̄+⟩  (feedback X^{R·s} cancels the random
                                       Z-syndrome s, where ``H_Z R = I``; the
                                       X-corrections commute with X̄ = +1, so the
                                       logical is untouched)
        S†^n (Y+) / S^n (Y-)  → |Ȳ±⟩  (transversal S† maps X̄ → +Ȳ, so
                                       S†|X̄+⟩ = |Ȳ+⟩; transversal S maps X̄ → −Ȳ,
                                       so S|X̄+⟩ = |Ȳ-⟩ — verified in stim)

    The feedback measurement+correction PROJECTS |+⟩^n onto the syndrome-0
    codeword |X̄+⟩ deterministically (the random Z-syndrome is cancelled by a
    Pauli correction, not merely tracked), so the state entering the merge is an
    exact Ȳ eigenstate and the merge's original-code H_X/H_Z rows are +1
    deterministic on it (verified). ``data_init="Y+"`` applies ``S†`` (→ |Ȳ+⟩,
    ⟨Ȳ⟩ = +1, bare obs0 = +1 → bit 0); ``"Y-"`` applies ``S`` (→ |Ȳ-⟩, ⟨Ȳ⟩ = −1,
    bare obs0 = −1 → bit 1).

    Transversal S is a logical operation only for a self-dual CSS code (its X-
    and Z-stabilizer row spaces coincide); we restrict to that case and raise
    otherwise. Homological Ȳ = iX̄Z̄ measurement of Ide, Gowda, Nadkarni,
    Dauphinais arXiv:2410.02753 §III.C/§III.D; the eigenstate prep is the standard
    transversal-Clifford state injection of self-dual CSS codes.

    Args:
        yg: the Y-gadget layout (provides the underlying self-dual CSS ``code``).
        real_data_ids: the merged-code data-qubit IDs (first ``code.num_qudits``
            columns) the codeword is prepared on.
        data_init: ``"Y+"`` (apply S) or ``"Y-"`` (apply S†).
        ancilla_base: first free qubit ID; the ``m_z`` Z-stabilizer measurement
            ancillas occupy ``ancilla_base .. ancilla_base + m_z - 1`` (disjoint
            from the merge qubits). These prep measurements are emitted before
            any merge measurement, so the merge's end-relative ``target_rec``
            offsets are unaffected.
    """
    import galois as _galois

    F2 = _galois.GF(2)
    code = yg.code
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    # Self-dual ⇔ X- and Z-stabilizer row spaces coincide (transversal H is
    # then a logical operation; transversal S is logical S for the Steane-like
    # triorthogonal/self-dual family).
    if not (HX.shape == HZ.shape and np.array_equal(np.sort(HX, axis=0), np.sort(HZ, axis=0))):
        raise ValueError(
            "data_init='Y+'/'Y-' transversal prep requires a self-dual CSS code "
            "(transversal H/S logical); got a non-self-dual code. Provide an "
            "explicit Ȳ-eigenstate prep for this code (Ide, Gowda, Nadkarni, "
            "Dauphinais arXiv:2410.02753 §III.C)."
        )
    ids = list(real_data_ids)
    n = len(ids)
    m_z = HZ.shape[0]

    # Right inverse R (n×m_z) of H_Z over GF(2): H_Z @ R = I. The Pauli X^{R·s}
    # then has Z-syndrome s, so applying it cancels a measured Z-syndrome s.
    R = np.zeros((n, m_z), dtype=np.uint8)
    eye = np.eye(m_z, dtype=np.uint8)
    for i in range(m_z):
        aug = F2(np.hstack([HZ, eye[i].reshape(-1, 1)]))
        rref = np.asarray(aug.row_reduce()).astype(np.uint8)
        col = np.zeros(n, dtype=np.uint8)
        for row in rref:
            lead = np.flatnonzero(row[:n])
            if lead.size:
                col[lead[0]] = row[n]
        R[:, i] = col
    if not np.array_equal((HZ @ R) % 2, eye):
        raise ValueError("internal: H_Z has no right inverse (code not full Z-rank)")

    circuit = stim.Circuit()
    circuit.append("RX", ids)  # |+⟩^n  (X̄ = +1, X-syndrome 0)
    # Measure each Z-stabilizer onto a fresh ancilla (random Z-syndrome on |+⟩^n).
    # Reset and measure ALL m_z ancillas in single ticks (one R layer, one M
    # layer) rather than a per-row R…CX…M staircase: the per-ancilla CX cascades
    # in between are independent (each writes a distinct ancilla target, sharing
    # only data controls, which commute), so batching is semantics-preserving and
    # keeps the timeline diagram a clean reset → extract → readout. The M order
    # (anc 0 … anc m_z-1) is unchanged, so the feedback target_rec(-(m_z-i))
    # offsets below still address stabilizer i.
    anc_ids = [ancilla_base + i for i in range(m_z)]
    circuit.append("R", anc_ids)
    for i in range(m_z):
        anc = anc_ids[i]
        for q in np.flatnonzero(HZ[i]):
            circuit.append("CX", [ids[int(q)], anc])  # Z-parity of data → ancilla
    circuit.append("M", anc_ids)
    # Feedback X^{R·s} to cancel the syndrome → exact |X̄+⟩. Stabilizer i is the
    # (m_z - i)-th most recent measurement record.
    for i in range(m_z):
        rec = stim.target_rec(-(m_z - i))
        for q in np.flatnonzero(R[:, i]):
            circuit.append("CX", [rec, ids[int(q)]])  # classically-controlled X_q
    # Transversal S† maps X̄ → +Ȳ (= iX̄Z̄), so S†|X̄+⟩ = |Ȳ+⟩ (the Ȳ = +1
    # eigenstate); transversal S maps X̄ → −Ȳ, so S|X̄+⟩ = |Ȳ-⟩. Verified in stim:
    # ⟨Ȳ⟩ = ⟨[x|z]⟩ = +1 on S†|X̄+⟩ and −1 on S|X̄+⟩.
    circuit.append("S_DAG" if data_init == "Y+" else "S", ids)  # → |Ȳ+⟩ / |Ȳ-⟩
    return circuit


def _y_state_prep(
    yg: YGadgetLayout,
    *,
    data_init: str | None,
) -> tuple[stim.Circuit, _YCtx]:
    """Emit QUBIT_COORDS + state-prep instructions; return the circuit and a
    cross-phase context object.

    Covers pipeline steps 1–2 of ``build_single_y_ppm_circuit``:
      1. QUBIT_COORDS for data + κ ancillas + Y-row ancillas.
      2. State prep: data via ``data_init``; κ_x ancillas |0⟩ (X-system), κ_z
         ancillas |+⟩ (Z-system).

    ``data_init`` must already be the normalised form (``"Z+"`` → ``None``,
    ``"X+"`` → ``"+"``); callers are responsible for that alias resolution.
    """
    merged_code = yg.merged_code
    field = merged_code.field
    n_q = merged_code.num_qudits
    n_code = yg.code.num_qudits
    k_x = len(yg.g_x.Q_prime)
    k_z = len(yg.g_z.Q_prime)
    assert n_q == n_code + k_x + k_z, (
        f"merged code width {n_q} != n_code {n_code} + k_x {k_x} + k_z {k_z}"
    )

    # Split the (possibly non-CSS) merged code into pure-X / pure-Z rows (driven
    # by EdgeColoring) and the mixed Y-type rows (driven by per-row CX/CY/CZ).
    virtual_cssc, HX, HZ, x_row_idx, z_row_idx, mixed_row_idx = (
        _split_quditcode_into_virtual_cssc(merged_code)
    )
    qubit_ids = QubitIDs.from_code(virtual_cssc)
    n_Y = len(mixed_row_idx)
    if n_Y:
        max_id = max(qubit_ids.all_qubits) if qubit_ids.all_qubits else -1
        y_ancilla_ids: tuple[int, ...] = tuple(range(max_id + 1, max_id + 1 + n_Y))
    else:
        y_ancilla_ids = ()

    # Merged-code column roles: [data (n_code) | κ_x (k_x) | κ_z (k_z)].
    data_cols = qubit_ids.data
    real_data_ids = data_cols[:n_code]
    kx_ids = data_cols[n_code : n_code + k_x]  # X-system ancillas → |0⟩, read Z
    kz_ids = data_cols[n_code + k_x :]  # Z-system ancillas → |+⟩, read X

    # First free qubit ID above all structural qubits — the base for the transient
    # Y-eigenstate state-injection ancillas (m_z Z-syndrome ancillas allocated in
    # ``_steane_logical_y_eigenstate_prep``). Reserve their IDs up front so they
    # get QUBIT_COORDS in the same layout pass; without this they were emitted as
    # coordinate-less wires (the R/M-only injection ancillas of the Y± preps).
    prep_base = (
        max([*qubit_ids.all_qubits, *y_ancilla_ids])
        if (qubit_ids.all_qubits or y_ancilla_ids)
        else -1
    ) + 1
    if data_init in ("Y+", "Y-"):
        n_prep = yg.code.matrix_z.shape[0]
        prep_ancilla_ids: tuple[int, ...] = tuple(range(prep_base, prep_base + n_prep))
    else:
        prep_ancilla_ids = ()

    # Original code check counts: the first m_x / m_z rows of checks_x / checks_z
    # are the original H_X / H_Z stabilizers; the rest are the new S_X' / S_Z'
    # merge rows (block order of build_y_gadget, preserved by the row split).
    m_x = yg.code.matrix_x.shape[0]
    m_z = yg.code.matrix_z.shape[0]
    circuit = _mixed_basis_qubit_coords(
        n_code, m_x, m_z, qubit_ids, y_ancilla_ids, prep_ancilla_ids
    )

    # --- State prep -----------------------------------------------------------
    # ``data_init`` is one of the six logical Pauli-basis eigenstates, named by
    # the operator they are a +/- eigenstate of:
    #   "Z+" (or None) → |0̄⟩ (Z̄ = +1)      "Z-" → |1̄⟩ (Z̄ = −1)
    #   "X+" (or "+")  → |+̄⟩ (X̄ = +1)      "X-" → |-̄⟩ (X̄ = −1)
    #   "Y+"           → |Ȳ+⟩ (Ȳ = +1)      "Y-" → |Ȳ-⟩ (Ȳ = −1)
    # The Ȳ measurement is deterministic ONLY on the Ȳ eigenstates (Y±); on the
    # Z̄/X̄ eigenstates Ȳ anticommutes with the prepared logical, so obs0 is a
    # genuine 50/50 (read via ``force_obs0``). None ("Z+") and "+" ("X+") are the
    # backward-compatible aliases for |0̄⟩ and |+̄⟩; internally |0̄⟩ is ``None``.
    x_cols = tuple(int(q) for q in np.flatnonzero(np.asarray(yg.x).astype(np.uint8)))
    z_cols = tuple(int(q) for q in np.flatnonzero(np.asarray(yg.z).astype(np.uint8)))
    if data_init is None:
        circuit.append("R", list(real_data_ids))  # |0⟩^n → logical |0̄…0̄⟩
    elif data_init == "Z-":
        circuit.append("R", list(real_data_ids))  # |0⟩^n
        circuit.append("X", [real_data_ids[q] for q in x_cols])  # X̄ on supp(x) → |1̄⟩
    elif data_init == "+":
        circuit.append("RX", list(real_data_ids))  # |+⟩^n → logical |+̄…+̄⟩
    elif data_init == "X-":
        circuit.append("RX", list(real_data_ids))  # |+⟩^n
        circuit.append("Z", [real_data_ids[q] for q in z_cols])  # Z̄ on supp(z) → |-̄⟩
    elif data_init in ("Y+", "Y-"):
        # ``prep_base`` (and the matching ``prep_ancilla_ids`` coords) were
        # reserved above so the injection ancillas land in the layout.
        circuit += _steane_logical_y_eigenstate_prep(
            yg, real_data_ids, data_init=data_init, ancilla_base=prep_base
        )
    else:
        raise ValueError(
            "data_init must be one of None/'Z+', 'Z-', '+'/'X+', 'X-', 'Y+', 'Y-' "
            f"for build_single_y_ppm_circuit; got {data_init!r}"
        )
    if kx_ids:
        circuit.append("R", list(kx_ids))  # X-system gadget ancilla |0⟩ (basis-complement)
    if kz_ids:
        circuit.append("RX", list(kz_ids))  # Z-system gadget ancilla |+⟩ (basis-complement)

    # Build the virtual CSS sub-codes (pure-X / pure-Z subsets) for EdgeColoring.
    HX_only = HX if HX.shape[0] else np.zeros((0, n_q), dtype=np.uint8)
    HZ_only = HZ if HZ.shape[0] else np.zeros((0, n_q), dtype=np.uint8)
    virtual_cssc_X = CSSCode(
        field(HX_only),
        field(np.zeros((0, n_q), dtype=np.uint8)),
        is_subsystem_code=False,
    )
    virtual_cssc_Z = CSSCode(
        field(np.zeros((0, n_q), dtype=np.uint8)),
        field(HZ_only),
        is_subsystem_code=False,
    )

    H_full = np.asarray(merged_code.matrix).astype(np.int_)
    center_mask = _compute_stabilizer_center_mask(H_full, n_q)

    ctx = _YCtx(
        merged_code=merged_code,
        qubit_ids=qubit_ids,
        virtual_cssc_X=virtual_cssc_X,
        virtual_cssc_Z=virtual_cssc_Z,
        real_data_ids=real_data_ids,
        kx_ids=kx_ids,
        kz_ids=kz_ids,
        y_ancilla_ids=y_ancilla_ids,
        n_code=n_code,
        k_x=k_x,
        k_z=k_z,
        n_q=n_q,
        center_mask=center_mask,
        H_full=H_full,
        x_row_idx=x_row_idx,
        z_row_idx=z_row_idx,
        mixed_row_idx=mixed_row_idx,
        x_cols=x_cols,
        z_cols=z_cols,
    )
    return circuit, ctx


def _split_quditcode_into_virtual_cssc(
    joint_code: QuditCode,
) -> tuple[CSSCode, np.ndarray, np.ndarray, list[int], list[int], list[int]]:
    """Partition ``joint_code.matrix`` rows by Pauli type into a virtual CSS subset.

    Mixed-basis subsystem-code helper (Cross, He, Rall, Yoder
    arXiv:2407.18393 Theorem 20 / Cowtan, He, Williamson, Yoder
    arXiv:2503.05003 §3.5 / Webster, Smith, Cohen arXiv:2511.15989 §II.B.2).
    The merged code is a SUBSYSTEM code: not all pairs of rows commute.

    The merged code (Webster cross-merge in ``y_gadget``) has three row classes:
      * pure-X rows (HX block): data X-stabilizers + comp-side rows.
      * pure-Z rows (HZ block): symmetric on Z side.
      * mixed Y rows (Y_stab block): one per merge qubit from the Webster
        cross-merge. Their X-part comes from an S_X' row, Z-part from an S_Z'
        row, with single-{q} adapter support on each side.

    This helper builds a "virtual" CSSCode from the pure-X / pure-Z subsets
    for reuse of the CSS syndrome-extraction pipeline (EdgeColoring). Y rows
    are reported separately via ``mixed_row_indices`` and are handled by
    dedicated CY/CZ-based extraction in the mixed-basis pipeline.

    Returns
    -------
    virtual_cssc
        ``CSSCode`` whose ``matrix_x`` is the pure-X subset, ``matrix_z`` is
        the pure-Z subset.
    HX
        Pure-X rows as an X-only matrix (shape (n_x, n_qudits)).
    HZ
        Pure-Z rows as a Z-only matrix (shape (n_z, n_qudits)).
    x_row_indices
        Indices into ``joint_code.matrix`` of the pure-X rows, in insertion
        order — used to locate S_X' ancilla IDs in the resulting qubit_ids.
    z_row_indices
        Indices of the pure-Z rows.
    mixed_row_indices
        Indices of any rows with both X and Z support (Y rows from the
        Webster cross-merge).
    """
    H = np.asarray(joint_code.matrix).astype(np.int_)
    n = joint_code.num_qudits
    Hx = H[:, :n]
    Hz = H[:, n:]
    x_mask = Hx.any(axis=1) & ~Hz.any(axis=1)
    z_mask = Hz.any(axis=1) & ~Hx.any(axis=1)
    mixed_mask = Hx.any(axis=1) & Hz.any(axis=1)

    x_idx = [int(i) for i in np.flatnonzero(x_mask)]
    z_idx = [int(i) for i in np.flatnonzero(z_mask)]
    mixed_idx = [int(i) for i in np.flatnonzero(mixed_mask)]

    HX = Hx[x_mask].astype(np.uint8)
    HZ = Hz[z_mask].astype(np.uint8)

    field = joint_code.field
    virtual_cssc = CSSCode(field(HX), field(HZ), is_subsystem_code=False)
    return virtual_cssc, HX, HZ, x_idx, z_idx, mixed_idx


def _mixed_basis_qubit_coords(
    n_code: int,
    m_x: int,
    m_z: int,
    qubit_ids: QubitIDs,
    y_ancilla_ids: tuple[int, ...] = (),
    prep_ancilla_ids: tuple[int, ...] = (),
) -> stim.Circuit:
    """Emit the per-role QUBIT_COORDS layout for the single-Y merged code.

    Follows the same lane convention as ``_surgery_qubit_coordinates`` (the CSS
    single/joint-PPM layout): data on y=0, gadget ancillas (Q') on y=1, and each
    check family SPLIT into the original code checks vs the new merge rows. The
    merged-code rows are assembled in the block order
    ``[H_X | S_X' ‖ y_v ‖ H_Z | S_Z' ‖ ∂_0]`` (Ide, Gowda, Nadkarni, Dauphinais
    arXiv:2410.02753 §III.D) and ``_split_quditcode_into_virtual_cssc`` preserves
    it, so the first ``m_x`` of ``checks_x`` are the original X-stabilizers (H_X)
    and the rest are the new S_X' merge rows — symmetrically for ``checks_z``.

    Lanes (x = index within the lane, restarting at 0 per lane):
      y=0  real data qubits      (first ``n_code`` columns of ``qubit_ids.data``)
      y=1  κ_x/κ_z gadget ancillas (Q'; remaining ``qubit_ids.data`` columns)
      y=2  original X-checks      (``checks_x[:m_x]`` = H_X)
      y=3  new X-checks           (``checks_x[m_x:]``, the X-type merge ancillas)
      y=4  original Z-checks      (``checks_z[:m_z]`` = H_Z)
      y=5  new Z-checks           (``checks_z[m_z:]``, the Z-type merge ancillas)
      y=6  mixed Y-rows: the ``y_v`` cross-merge ancillas + any ∂_0 cycle
           ancillas (Webster, Smith, Cohen arXiv:2511.15989 §II.B.2; |W|≥2 adds
           the cycle rows)
      y=7  transient Y-eigenstate state-injection ancillas (the m_z Z-syndrome
           measure ancillas of ``_steane_logical_y_eigenstate_prep``; present
           only for ``data_init`` in {"Y+", "Y-"})
    """
    circuit = stim.Circuit()
    for i, qid in enumerate(qubit_ids.data):
        lane, x = (0, i) if i < n_code else (1, i - n_code)
        circuit.append("QUBIT_COORDS", qid, (x, lane))
    # X-checks: original H_X on y=2, new X-checks on y=3.
    for i, qid in enumerate(qubit_ids.checks_x):
        lane, x = (2, i) if i < m_x else (3, i - m_x)
        circuit.append("QUBIT_COORDS", qid, (x, lane))
    # Z-checks: original H_Z on y=4, new Z-checks on y=5.
    for i, qid in enumerate(qubit_ids.checks_z):
        lane, x = (4, i) if i < m_z else (5, i - m_z)
        circuit.append("QUBIT_COORDS", qid, (x, lane))
    # Mixed Y-rows (y_v cross-merge + ∂_0 cycles) on y=6.
    for i, qid in enumerate(y_ancilla_ids):
        circuit.append("QUBIT_COORDS", qid, (i, 6))
    # Transient state-injection ancillas on y=7.
    for i, qid in enumerate(prep_ancilla_ids):
        circuit.append("QUBIT_COORDS", qid, (i, 7))
    return circuit


def _compute_stabilizer_center_mask(H_sym: np.ndarray, n: int) -> np.ndarray:
    """Mark each row of the symplectic matrix that commutes with ALL other rows.

    Returned mask is True for rows in the (algebraic) stabilizer center —
    measurements of these rows commute with every other gauge generator, so
    their syndrome-extraction outcomes are deterministic given any pure
    stabilizer state. Rows outside the center are gauge operators with
    random outcomes; we must NOT register detectors for them.
    """
    if H_sym.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    Hx = H_sym[:, :n].astype(np.int_)
    Hz = H_sym[:, n:].astype(np.int_)
    comm = (Hx @ Hz.T + Hz @ Hx.T) % 2
    # Zero out diagonal (a row trivially commutes with itself).
    np.fill_diagonal(comm, 0)
    return np.asarray(~comm.any(axis=1))
