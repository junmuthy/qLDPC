"""Single logical-Ȳ (Ȳ = iX̄Z̄) measurement circuit — non-CSS homological
surgery (Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.C/§III.D).
Emits the split X/Z/Y syndrome schedule
over the merged code H̃ (see y_gadget.build_y_gadget for the H̃ block layout)."""

from __future__ import annotations

import dataclasses

import numpy as np
import stim

from qldpc.circuits.bookkeeping import MeasurementRecord, QubitIDs
from qldpc.circuits.memory.syndrome_measurement import EdgeColoring
from qldpc.circuits.noise_model import NoiseModel
from qldpc.codes.common import CSSCode, QuditCode
from qldpc.objects import Pauli

from .y_gadget import YGadgetLayout


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


def _y_qec_cycle(
    ctx: _YCtx,
    yg: YGadgetLayout,
    *,
    data_init: str | None,
    rounds: int,
    batch_resets: bool = False,
) -> tuple[stim.Circuit, MeasurementRecord, dict[int, int], dict[int, Pauli]]:
    """Emit the split X/Z/Y multi-round QEC schedule and return the circuit +
    measurement record.

    Covers pipeline step 3 of ``build_single_y_ppm_circuit``:
      * X-phase → H̃ blocks 1,2; Z-phase → 4,5; Y-phase → block 3
      * Round-1 reliable detectors for center rows deterministic on the
        prepared state.
      * REPEAT block for rounds 2 … rounds (round-to-round difference
        detectors for all center rows).

    Returns a fresh ``stim.Circuit`` (the QEC rounds only, to be concatenated
    into the main circuit by the orchestrator), the corresponding
    ``MeasurementRecord`` covering all emitted measurements, the
    ``row_to_check`` map (merged-code row index → ancilla qubit ID), and the
    ``qubit_final_meas`` map (qubit ID → destructive readout ``Pauli``).
    """
    qubit_ids = ctx.qubit_ids
    virtual_cssc_X = ctx.virtual_cssc_X
    virtual_cssc_Z = ctx.virtual_cssc_Z
    real_data_ids = ctx.real_data_ids
    kx_ids = ctx.kx_ids
    kz_ids = ctx.kz_ids
    y_ancilla_ids = ctx.y_ancilla_ids
    n_q = ctx.n_q
    center_mask = ctx.center_mask
    H_full = ctx.H_full
    x_row_idx = ctx.x_row_idx
    z_row_idx = ctx.z_row_idx
    mixed_row_idx = ctx.mixed_row_idx
    x_cols = ctx.x_cols
    z_cols = ctx.z_cols

    # --- Build the split X / Z / Y per-round circuit --------------------------
    # X-phase → H̃ blocks 1,2; Z-phase → 4,5; Y-phase → block 3.
    # Determinism rationale: X-ancillas collapse before the Z-phase CZ gates
    # fire, so the data is in a definite X-stabilizer eigenstate when the
    # Z-phase starts.
    qubit_ids_x = QubitIDs(data=qubit_ids.data, check=qubit_ids.checks_x)
    qubit_ids_x.checks_x = qubit_ids.checks_x
    qubit_ids_z = QubitIDs(data=qubit_ids.data, check=qubit_ids.checks_z)
    qubit_ids_z.checks_z = qubit_ids.checks_z

    strategy = EdgeColoring()
    if virtual_cssc_X.matrix_x.shape[0]:
        x_phase_circuit, x_phase_record = strategy.get_circuit(virtual_cssc_X, qubit_ids_x)
    else:
        x_phase_circuit, x_phase_record = stim.Circuit(), MeasurementRecord()
    if virtual_cssc_Z.matrix_z.shape[0]:
        z_phase_circuit, z_phase_record = strategy.get_circuit(virtual_cssc_Z, qubit_ids_z)
    else:
        z_phase_circuit, z_phase_record = stim.Circuit(), MeasurementRecord()

    # Y-row extraction phase: one |+⟩ ancilla per Y_stab row; CX/CY/CZ entangle
    # it with the data per the Pauli at each column; MX records the eigenvalue.
    n_Y = len(mixed_row_idx)
    y_phase_circuit = stim.Circuit()
    y_phase_record = MeasurementRecord()
    if n_Y:
        y_phase_circuit.append("RX", list(y_ancilla_ids))
        for y_anc, orig_row_idx in zip(y_ancilla_ids, mixed_row_idx):
            row = H_full[orig_row_idx]
            x_part = row[:n_q]
            z_part = row[n_q:]
            cx_pairs: list[int] = []
            cy_pairs: list[int] = []
            cz_pairs: list[int] = []
            for q in range(n_q):
                xq, zq = int(x_part[q]), int(z_part[q])
                if xq == 1 and zq == 0:
                    cx_pairs.extend([y_anc, qubit_ids.data[q]])
                elif xq == 0 and zq == 1:
                    cz_pairs.extend([y_anc, qubit_ids.data[q]])
                elif xq == 1 and zq == 1:
                    cy_pairs.extend([y_anc, qubit_ids.data[q]])
            if cx_pairs:
                y_phase_circuit.append("CX", cx_pairs)
            if cy_pairs:
                y_phase_circuit.append("CY", cy_pairs)
            if cz_pairs:
                y_phase_circuit.append("CZ", cz_pairs)
        y_phase_circuit.append("MX", list(y_ancilla_ids))
        y_phase_record.append({q: i for i, q in enumerate(y_ancilla_ids)})

    # Assemble one round from the X / Z / Y phases. Two reset schedulings:
    #
    #   batch_resets=True (the noiseless circuit — the one that gets diagrammed):
    #     hoist EVERY phase's ancilla reset into a single RX layer at the round
    #     start, then emit the three measurement bodies (one TICK between each).
    #     The timeline then shows one reset column followed by the X/Z/Y
    #     measurement columns, with no reset interleaved into a measurement
    #     moment. Resets carry no ordering constraint (a check ancilla just idles
    #     in |+⟩ until its entangling gates), so this is semantics-free; the
    #     measurement order X→Z→Y is preserved, so the subsystem-gauge
    #     determinism is untouched.
    #
    #   batch_resets=False (under a noise model — the LER circuit): keep the
    #     just-in-time per-phase resets (each phase resets its ancillas right
    #     before using them), minimising ancilla idle time. Hoisting resets would
    #     lengthen ancilla idle windows and add idle-depolarisation locations, so
    #     we do NOT batch under noise.
    #
    # Either way the measurement record and the noiseless DEM are unchanged
    # (resets create no records; the MX order within/between phases is fixed).
    one_round = stim.Circuit()
    if batch_resets:
        # Diagram schedule: one reset layer, then a MERGED X+Z ancilla readout,
        # then the Y readout on its own. The pure-X and pure-Z merged-code rows
        # always commute (a pure-X row has zero Z-part, a pure-Z row zero X-part,
        # and their data supports meet in even overlap by CSS), so the two
        # |+>-ancilla MX readouts share one measurement moment with deterministic
        # outcomes; each X ancilla is touched only by its own CX gates, so
        # deferring the X readout past the Z-phase gates measures the same
        # stabiliser exactly. The mixed y_v row can anticommute with the gauge in
        # general, so the Y readout stays last and is never merged (Ide, Gowda,
        # Nadkarni, Dauphinais arXiv:2410.02753 §III.D). Used only for the
        # noiseless (diagrammed) circuit; the noisy LER path keeps the original
        # split schedule below.
        rx, x_body = _split_leading_reset(x_phase_circuit)
        rz, z_body = _split_leading_reset(z_phase_circuit)
        ry, y_body = _split_leading_reset(y_phase_circuit)
        x_gates, x_mx = _split_trailing_measure(x_body)
        z_gates, z_mx = _split_trailing_measure(z_body)
        y_gates, y_mx = _split_trailing_measure(y_body)

        def _tick() -> None:
            if len(one_round):
                one_round.append("TICK")

        all_resets = rx + rz + ry
        if all_resets:
            one_round.append("RX", all_resets)
        if len(x_gates):
            _tick()
            one_round += x_gates
        if len(z_gates):
            _tick()
            one_round += z_gates
        xz_mx = x_mx + z_mx
        if xz_mx:
            _tick()
            one_round.append("MX", xz_mx)  # merged X+Z ancilla readout
        if len(y_gates):
            _tick()
            one_round += y_gates
        if y_mx:
            _tick()
            one_round.append("MX", y_mx)  # mixed y_v readout, kept separate
    else:
        for ph in (x_phase_circuit, z_phase_circuit, y_phase_circuit):
            one_round += ph
    round_measurement_record = MeasurementRecord()
    round_measurement_record.append(x_phase_record)
    round_measurement_record.append(z_phase_record)
    round_measurement_record.append(y_phase_record)

    # --- Map joint rows → check ancilla IDs, classify the stabilizer center ---
    row_to_check: dict[int, int] = {}
    for slot, orig in enumerate(x_row_idx):
        row_to_check[orig] = qubit_ids.checks_x[slot]
    for slot, orig in enumerate(z_row_idx):
        row_to_check[orig] = qubit_ids.checks_z[slot]
    for slot, orig in enumerate(mixed_row_idx):
        row_to_check[orig] = y_ancilla_ids[slot]
    center_check_ids = tuple(
        row_to_check[orig] for orig in row_to_check if center_mask[orig]
    )

    # Per-qubit init Pauli + sign (used to find round-1 deterministic centers).
    #   real data: data_init → Pauli/sign; Y± is an Ȳ eigenstate, not a single-
    #     qubit product state, so we mark it None (no per-qubit single-stab is
    #     deterministic — only the Ȳ row product is, handled by obs0).
    #   κ_x: |0⟩ → +Z;  κ_z: |+⟩ → +X.
    qubit_init: dict[int, tuple[Pauli, int] | None] = {}
    if data_init in (None, "Z-"):
        flip = set(x_cols) if data_init == "Z-" else set()  # |1̄⟩ = X̄ on supp(x)
        for col, qid in enumerate(real_data_ids):
            qubit_init[qid] = (Pauli.Z, -1 if col in flip else +1)  # |0⟩ / |1⟩
    elif data_init in ("+", "X-"):
        flip = set(z_cols) if data_init == "X-" else set()  # |-̄⟩ = Z̄ on supp(z)
        for col, qid in enumerate(real_data_ids):
            qubit_init[qid] = (Pauli.X, -1 if col in flip else +1)  # |+⟩ / |-⟩
    else:  # "Y+"/"Y-": no single-qubit Pauli eigenstate per data qubit
        for qid in real_data_ids:
            qubit_init[qid] = None
    for qid in kx_ids:
        qubit_init[qid] = (Pauli.Z, +1)
    for qid in kz_ids:
        qubit_init[qid] = (Pauli.X, +1)

    # Per-qubit destructive readout basis (used by final detector emission):
    #   κ_x → Z (M), κ_z → X (MX), real data → basis matching data_init:
    #   Y± → Y;  X± ("+"/"X-") → X;  Z± (None/"1") → Z.
    data_final_pauli = (
        Pauli.Y
        if data_init in ("Y+", "Y-")
        else (Pauli.X if data_init in ("+", "X-") else Pauli.Z)
    )
    qubit_final_meas: dict[int, Pauli] = {}
    for qid in real_data_ids:
        qubit_final_meas[qid] = data_final_pauli
    for qid in kx_ids:
        qubit_final_meas[qid] = Pauli.Z
    for qid in kz_ids:
        qubit_final_meas[qid] = Pauli.X

    def _row_paulis(orig_row: int) -> dict[int, Pauli]:
        row = H_full[orig_row]
        out: dict[int, Pauli] = {}
        for q in range(n_q):
            xq, zq = int(row[q]), int(row[q + n_q])
            if xq == 0 and zq == 0:
                continue
            out[q] = Pauli.X if (xq, zq) == (1, 0) else (Pauli.Z if (xq, zq) == (0, 1) else Pauli.Y)
        return out

    # Round-1 reliable: center rows whose every non-I Pauli matches the init
    # eigenstate of that qubit, with net sign +1 (noiseless outcome 0).
    round1_reliable_check_ids: list[int] = []
    for orig in row_to_check:
        if not center_mask[orig]:
            continue
        cid = row_to_check[orig]
        sign = 1
        ok = True
        for q, pauli_q in _row_paulis(orig).items():
            init = qubit_init[qubit_ids.data[q]]
            if init is None or pauli_q is not init[0]:
                ok = False
                break
            sign *= init[1]
        if ok and sign == 1:
            round1_reliable_check_ids.append(cid)

    circuit = stim.Circuit()
    measurement_record = MeasurementRecord()

    # The merge (lattice surgery proper). On a Y±-eigenstate prep the data is the
    # exact |Ȳ±⟩ codeword (prepared by _steane_logical_y_eigenstate_prep above),
    # so the bare new-stabilizer product ∏(S_X'·S_Z'·y_v) = [x | z] first-measures
    # on a codeword and is the deterministic Ȳ readout (Ide, Gowda, Nadkarni,
    # Dauphinais arXiv:2410.02753 §III.C).
    circuit += one_round
    measurement_record.append(round_measurement_record)
    for cid in round1_reliable_check_ids:
        circuit.append(
            "DETECTOR", [measurement_record.get_target_rec(cid)], (cid, 0, 0)
        )

    if rounds > 1:
        repeat = one_round.copy()
        measurement_record.append(round_measurement_record)
        repeat.append("SHIFT_COORDS", [], (0, 0, 1))
        for cid in center_check_ids:
            repeat.append(
                "DETECTOR",
                [
                    measurement_record.get_target_rec(cid, -1),
                    measurement_record.get_target_rec(cid, -2),
                ],
                (cid, 0, 0),
            )
        circuit.append(stim.CircuitRepeatBlock(rounds - 1, repeat))
        measurement_record.append(round_measurement_record, repeat=rounds - 2)

    return circuit, measurement_record, row_to_check, qubit_final_meas


def _y_detach_and_readout(
    ctx: _YCtx,
    *,
    data_init: str | None,
    measurement_record: MeasurementRecord,
    destructive_measure_data: bool = True,
) -> stim.Circuit:
    """Detach the κ ancillas; optionally destructively read out the data.

    Covers pipeline step 4 of ``build_single_y_ppm_circuit``:
      * κ_x ancillas → M (read in Z).
      * κ_z ancillas → MX (read in X).
      * SHIFT_COORDS tick.
      * Data qubits → basis matching ``data_init`` (Y± → MY, X± → MX, Z± → M)
        — emitted only when ``destructive_measure_data`` is True.

    The κ detach is the split that returns the bare code; it always runs. When
    ``destructive_measure_data=False`` the real data qubits are left unmeasured
    (non-destructive / detach-only mode) — the Ȳ result is the in-circuit obs0.

    Mutates ``measurement_record`` in-place with the new measurement slots.
    Returns the new circuit fragment.
    """
    kx_ids = ctx.kx_ids
    kz_ids = ctx.kz_ids
    real_data_ids = ctx.real_data_ids

    circuit = stim.Circuit()

    # --- Detach (split): measure the κ gadget ancillas ------------------------
    if kx_ids:
        circuit.append("M", list(kx_ids))  # X-system ancilla read in Z
        measurement_record.append({q: i for i, q in enumerate(kx_ids)})
    if kz_ids:
        circuit.append("MX", list(kz_ids))  # Z-system ancilla read in X
        measurement_record.append({q: i for i, q in enumerate(kz_ids)})
    if not destructive_measure_data:
        return circuit  # detach-only: leave the data encoded

    # --- Destructive data readout ---------------------------------------------
    circuit.append("SHIFT_COORDS", [], (0, 0, 1))
    data_meas_op = (
        "MY"
        if data_init in ("Y+", "Y-")
        else ("MX" if data_init in ("+", "X-") else "M")
    )
    circuit.append(data_meas_op, list(real_data_ids))
    measurement_record.append({q: i for i, q in enumerate(real_data_ids)})

    return circuit


def _y_final_detectors(
    ctx: _YCtx,
    *,
    row_to_check: dict[int, int],
    qubit_final_meas: dict[int, Pauli],
    measurement_record: MeasurementRecord,
) -> stim.Circuit:
    """Emit final detectors for center rows reconstructable from destructive readouts.

    Covers pipeline step 5 of ``build_single_y_ppm_circuit``: same construction
    as the mixed-basis final detectors — a detector for each center row directly
    compatible with the destructive readout basis, and for readout-compatible
    null-space combinations of the remaining rows whose readout-incompatible
    parts cancel (Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.C/§III.D).

    Returns the new circuit fragment.  ``measurement_record`` is read but not
    mutated.
    """
    qubit_ids = ctx.qubit_ids
    n_q = ctx.n_q
    center_mask = ctx.center_mask
    H_full = ctx.H_full

    circuit = stim.Circuit()

    # --- Final detectors (center rows reconstructable from destructive readouts)
    center_idx = [orig for orig in row_to_check if center_mask[orig]]
    if center_idx:
        import galois as _galois

        F2 = _galois.GF(2)
        C = H_full[center_idx]  # (n_center, 2 n_q)

        def _row_destructive_compatible(combined_row: np.ndarray) -> bool:
            for q in range(n_q):
                xq, zq = int(combined_row[q]), int(combined_row[q + n_q])
                if xq == 0 and zq == 0:
                    continue
                P = qubit_final_meas[qubit_ids.data[q]]
                if P is Pauli.X and (xq, zq) != (1, 0):
                    return False
                if P is Pauli.Z and (xq, zq) != (0, 1):
                    return False
                if P is Pauli.Y and (xq, zq) != (1, 1):
                    return False
            return True

        def _emit_combo_detector(c_int: np.ndarray) -> None:
            combined = (c_int @ C) % 2
            targets: list[stim.GateTarget] = []
            for q in range(n_q):
                xq, zq = int(combined[q]), int(combined[q + n_q])
                if xq == 0 and zq == 0:
                    continue
                targets.append(measurement_record.get_target_rec(qubit_ids.data[q]))
            for slot, ci in enumerate(c_int):
                if ci:
                    cid_slot = row_to_check[center_idx[slot]]
                    targets.append(measurement_record.get_target_rec(cid_slot, -1))
            if targets:
                circuit.append("DETECTOR", targets, (0, 0, 0))

        F_rows: list[np.ndarray] = []
        for q in range(n_q):
            row_vec = np.zeros(2 * n_q, dtype=np.uint8)
            P = qubit_final_meas[qubit_ids.data[q]]
            if P is Pauli.X:
                row_vec[q + n_q] = 1
            elif P is Pauli.Z:
                row_vec[q] = 1
            else:  # Pauli.Y
                row_vec[q] = 1
                row_vec[q + n_q] = 1
            F_rows.append(row_vec)
        F_mat = np.stack(F_rows)
        A = F2((C @ F_mat.T) % 2)
        null_basis = np.asarray(A.T.null_space()).astype(np.int_)

        emitted_for: set[int] = set()
        for slot, orig in enumerate(center_idx):
            if orig in emitted_for:
                continue
            if _row_destructive_compatible(C[slot]):
                c = np.zeros(len(center_idx), dtype=np.int_)
                c[slot] = 1
                _emit_combo_detector(c)
                emitted_for.add(orig)
                continue
            cands = [(int(v.sum()), v) for v in null_basis if int(v[slot]) == 1]
            if not cands:
                continue
            cands.sort(key=lambda x: x[0])
            best_c = cands[0][1].astype(np.int_)
            _emit_combo_detector(best_c)
            for s2, val in enumerate(best_c):
                if val:
                    emitted_for.add(center_idx[s2])

    return circuit


def _y_emit_obs0(
    ctx: _YCtx,
    circuit: stim.Circuit,
    yg: YGadgetLayout,
    *,
    data_init: str | None,
    force_obs0: bool,
    measurement_record: MeasurementRecord,
) -> None:
    """Append OBSERVABLE_INCLUDE(0) for the Ȳ eigenvalue (§III.C readout product).

    Covers pipeline step 6a of ``build_single_y_ppm_circuit``:
    the obs0 eigenvalue is the XOR of the IN-CIRCUIT ancilla records of the
    merged-code rows whose product equals Ȳ on the original data columns (Ide,
    Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.C). Gated by
    ``_observable_is_deterministic``; emitted unconditionally when ``force_obs0``.
    """
    qubit_ids = ctx.qubit_ids
    y_ancilla_ids = ctx.y_ancilla_ids

    # --- obs0: the Ȳ eigenvalue (§III.C IN-CIRCUIT readout product) ------------
    # Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.C: the obs0
    # eigenvalue is the XOR of the IN-CIRCUIT ancilla records of the merged-code
    # rows whose product equals Ȳ on the original data columns. The picker
    # ``_ybar_obs0_rows`` solves this over GF(2): the selected rows' product is
    # the BARE new-stabilizer product ∏(S_X'·S_Z'·y_v), restricting to the literal
    # Ȳ support ``[x | z]`` on data (X on V_X, Z on V_Z, Y on W) and eigenbasis-
    # compatible on the κ ancillas (Z-only κ_x, X-only κ_z). ``yg.obs0_xor_map``
    # records, per selected row, its merged-code (``H_sym``) row index plus its
    # Pauli family (``"X"`` S_X' row, ``"Z"`` S_Z' row, or the ``"Y"`` y_v row) and
    # its index within that family. The family→ancilla map is the same one the
    # round circuit uses (``_split_quditcode_into_virtual_cssc`` partitions
    # ``merged_code.matrix`` in the SAME row order as ``H_sym``, so family_index
    # is the slot in ``checks_x`` / ``checks_z`` / ``y_ancilla_ids``):
    #   family "X" → ``qubit_ids.checks_x[family_index]``   (X-phase, M record)
    #   family "Z" → ``qubit_ids.checks_z[family_index]``   (Z-phase, M record)
    #   family "Y" → ``y_ancilla_ids[family_index]``        (Y-phase, MX record)
    #
    # DETERMINISM GATE. The bare ``[x | z]`` product carries Pauli-X on V_X and
    # Pauli-Z on V_Z data qubits. On a Y±-eigenstate prep the data is the EXACT
    # |Ȳ±⟩ codeword (state injection |X̄+⟩ then transversal S, see
    # ``_steane_logical_y_eigenstate_prep``), where every code stabilizer is +1,
    # so the bare product agrees with Ȳ and the in-circuit XOR is DETERMINISTIC
    # (Y+ → bit 1, Y- → bit 0; the GF(2) product drops the ``i`` of iX̄Z̄, so the
    # raw bit reads −Ȳ). On a non-eigenstate prep (|0̄⟩/|+̄⟩) the bare product is a
    # genuine 50/50, so ``_observable_is_deterministic`` gates obs0 OFF unless
    # ``force_obs0`` is set (the 50/50 cross-check). The previous ALL-Y-on-data
    # representative — a physical ∏_i|Y_i⟩ prep artifact, infeasible on general
    # codes such as BB [[36,8,4]] — is no longer targeted: the exact codeword
    # prep is what makes the bare product deterministic.
    obs0_recs: list[stim.GateTarget] = []
    for row in yg.obs0_xor_map:
        if row.family == "X":
            cid = qubit_ids.checks_x[row.family_index]
        elif row.family == "Z":
            cid = qubit_ids.checks_z[row.family_index]
        else:  # "Y" — the mixed y_v check, read MX in the Y-phase
            cid = y_ancilla_ids[row.family_index]
        obs0_recs.append(measurement_record.get_target_rec(cid))
    if force_obs0:
        # Emit obs0 even when it is NON-deterministic. On a non-Ȳ-eigenstate prep
        # (|0̄⟩ via data_init=None, or |+̄⟩ via data_init="+"), Ȳ anticommutes with
        # the prepared logical so the outcome is a genuine 50/50 — the DEM will NOT
        # compile, but raw sampling (``circuit.compile_sampler``) reads obs0
        # directly. Demonstrates that the merge measures Ȳ (Ide, Gowda, Nadkarni,
        # Dauphinais arXiv:2410.02753 §III.C).
        if obs0_recs:
            circuit.append("OBSERVABLE_INCLUDE", obs0_recs, 0)
    elif data_init in ("Y+", "Y-"):
        if obs0_recs and _observable_is_deterministic(circuit, obs0_recs):
            circuit.append("OBSERVABLE_INCLUDE", obs0_recs, 0)


def _y_emit_survivor_memory(
    ctx: _YCtx,
    circuit: stim.Circuit,
    *,
    memory_logical: int,
    data_init: str | None,
    measurement_record: MeasurementRecord,
) -> None:
    """Append OBSERVABLE_INCLUDE(0) for a surviving logical Z̄ (memory mode).

    Covers pipeline step 6c of ``build_single_y_ppm_circuit``:
    when ``memory_logical`` is given and ``data_init is None``, emits a single
    observable tracking the ``memory_logical``-th merged-code Z-logical as a
    logical-memory check (Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.C).
    """
    merged_code = ctx.merged_code
    real_data_ids = ctx.real_data_ids
    kx_ids = ctx.kx_ids
    n_code = ctx.n_code
    k_x = ctx.k_x
    n_q = ctx.n_q

    # --- survivor-memory observable --------------------------------------------
    # The Ȳ-on-q0 measurement preserves the other logicals; their Z̄ are
    # deterministic on the |0̄…0̄⟩ prep (data_init is None). Track one such
    # SURVIVING logical Z̄ off the final destructive readouts as a logical-memory
    # observable — the standard decodability check of the surgery (Ide, Gowda,
    # Nadkarni, Dauphinais arXiv:2410.02753 §III.C). This sidesteps the random Ȳ
    # outcome (obs0 stays gated off) by scoring a survivor instead. Gated on
    # ``data_init is None`` — obs0 is also gated off in this mode — so index 0 is
    # free; emitting there keeps the DEM at exactly one observable. Gated by
    # determinism, emitted before the noise block so the noise model wraps it.
    if memory_logical is not None and data_init is None:
        LZ = np.asarray(merged_code.get_logical_ops(Pauli.Z)).astype(np.uint8)
        row = LZ[memory_logical]  # (2*n_q,) symplectic [x | z]
        xpart, zpart = row[:n_q], row[n_q:]
        # readout-compatible survivor: pure-Z, no κ_z support (κ_z is read in X).
        # ``xpart.any()`` is a defensive guard: a ``get_logical_ops(Pauli.Z)`` row
        # is already pure-Z, so it never fires in practice, but it pins the
        # contract should a non-pure-Z representative ever be passed.
        if xpart.any() or zpart[n_code + k_x :].any():
            raise ValueError(
                f"merged Z-logical {memory_logical} is not Z-readout-compatible"
            )
        mem_recs: list[stim.GateTarget] = []
        for q in range(n_code):  # data Z -> M record
            if zpart[q]:
                mem_recs.append(measurement_record.get_target_rec(real_data_ids[q]))
        for q in range(k_x):  # κ_x Z -> M record (detach)
            if zpart[n_code + q]:
                mem_recs.append(measurement_record.get_target_rec(kx_ids[q]))
        if mem_recs and _observable_is_deterministic(circuit, mem_recs):
            circuit.append("OBSERVABLE_INCLUDE", mem_recs, 0)


def build_single_y_ppm_circuit(
    yg: YGadgetLayout,
    *,
    rounds: int,
    noise_model: NoiseModel | None = None,
    data_init: str | None = None,
    memory_logical: int | None = None,
    force_obs0: bool = False,
    destructive_measure_data: bool = True,
) -> stim.Circuit:
    """Single logical-Y PPM measurement circuit (Ȳ = iX̄Z̄) for ``yg``.

    Builds the syndrome-extraction circuit for the homological Y-gadget merged
    code of Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.C/§III.D.
    The merged code ``yg.merged_code`` is a ``QuditCode`` whose stabilizers are the
    original code's X-checks and Z-checks (dual-extended onto the κ_x / κ_z
    ancillas) plus the ``|W|`` mixed Y-type checks ``y_v`` (= the ``yg.Y_stab``
    rows), the Webster, Smith, Cohen arXiv:2511.15989 §II.B.2 cross-merge of the
    S_X' and S_Z' rows anchored at each crossing qubit ``v ∈ W``.

    The emission runs the split X / Z / Y syndrome schedule (the X-ancillas
    collapse before the Z-phase fires; the mixed Y-checks are measured last)
    directly on the single merged code ``yg.merged_code``. This is the sole
    non-CSS surgery emitter — ``build_joint_ppm_circuit`` handles only
    same-basis CSS joints and rejects mixed-basis inputs.

    Pipeline (mirrors ``build_single_ppm_circuit``):
      1. QUBIT_COORDS for data + κ ancillas + Y-row ancilla.
      2. State prep: data via ``data_init``; κ_x ancillas |0⟩ (X-system), κ_z
         ancillas |+⟩ (Z-system) — the basis-complement eigenstate of each
         per-system gadget.
      3. Multi-round QEC with a SPLIT X-phase (CX) / Z-phase (CZ) / Y-phase
         (per-``Y_stab``-row CX/CY/CZ → MX) schedule. Splitting the X- and
         Z-type extractions into non-overlapping phases keeps the individual
         gauge outcomes deterministic in the subsystem-code sense (Cohen, Kim,
         Bartlett, Brown arXiv:2110.10794 §II.B.2; Ide, Gowda, Nadkarni,
         Dauphinais arXiv:2410.02753 §III.C/§III.D). Round-1 detectors are emitted only for
         stabilizer-center rows that are deterministic on the prepared state;
         later rounds get round-to-round difference detectors for all center
         rows.
      4. Detach + destructive readout: κ_x in Z (M), κ_z in X (MX), data in the
         basis matching ``data_init`` (Y-eigenstate data → MY).
      5. Final detectors for center rows reconstructable from the destructive
         readouts (single-row or readout-compatible null-space combinations,
         same construction as the mixed-basis final detectors).
      6. obs0 — the Ȳ eigenvalue as the §III.C readout product (a merged-code
         stabilizer equal to Ȳ = [x|z] on data), read off the IN-CIRCUIT last-
         QEC-round ancilla outcomes of the picked rows (the fault-tolerant
         readout); emitted ONLY when that in-circuit XOR is deterministic on the
         prepared state (gated by ``_observable_is_deterministic``).

    ``data_init`` options — the six logical Pauli-basis eigenstates, named by the
    operator they are a ± eigenstate of:
      * ``"Z+"`` / ``None``: |0̄⟩ (Z̄ = +1, R).     ``"Z-"``: |1̄⟩ (Z̄ = −1, R + X̄).
      * ``"X+"`` / ``"+"``:  |+̄⟩ (X̄ = +1, RX).    ``"X-"``: |-̄⟩ (X̄ = −1, RX + Z̄).
      * ``"Y+"`` / ``"Y-"``: the EXACT logical-Y eigenstate codeword |Ȳ±⟩ = S̄|X̄±⟩
        (inject |X̄+⟩ then transversal S†/S; self-dual CSS code only, see
        ``_steane_logical_y_eigenstate_prep``).
    ``None`` and ``"+"`` are backward-compatible aliases for ``"Z+"`` and ``"X+"``.
    The Ȳ readout (obs0) is deterministic only on the Ȳ eigenstates (Y±); on the
    Z̄ (Z±) and X̄ (X±) eigenstates Ȳ anticommutes with the prepared logical, so
    obs0 is a genuine 50/50 (read via ``force_obs0``).

    ``destructive_measure_data`` (default True): when False, detach-only /
    non-destructive — the κ ancillas are measured (the split) but the data is
    left encoded as the post-measurement logical state. obs0 (the in-circuit Ȳ
    readout, fixed before detach) is still emitted; the destructive MY of the
    data and the destructive final detectors are skipped. This sidesteps the
    Y-specific X·Z=Y final-detector combination machinery entirely. Incompatible
    with ``memory_logical`` (which reads a survivor from the destructive readout).

    ``memory_logical`` (survivor-memory mode). The Ȳ-on-q0 measurement preserves
    the other logicals of the code; their Z̄ are deterministic on the |0̄…0̄⟩ prep
    (``data_init=None``). When ``memory_logical`` is an int and ``data_init is
    None``, this emits a single ``OBSERVABLE_INCLUDE`` (index 0 — obs0 is gated
    off in this mode, so the DEM carries exactly one observable) tracking the
    ``memory_logical``-th merged-code Z-logical as a surviving logical-memory
    observable read off the final destructive readouts.
    This is the standard logical-memory decodability check of the surgery (Ide,
    Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.C): it sidesteps the random
    Ȳ outcome (obs0 stays gated off — no Ȳ-eigenstate prep on |0̄…0̄⟩) by scoring
    a SURVIVING logical Z̄ instead. The chosen logical must be readout-compatible
    (pure-Z, no support on the κ_z ancillas, which are read in X); a non-compatible
    choice raises ``ValueError``. ``None`` (default) emits nothing, leaving the
    circuit byte-identical to the no-memory build.

    Bell/flag scope (Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.D).
    The brief's
    fault-tolerance refinement — splitting each ``y_v`` ancilla into a Bell/flag
    cell — is NOT built here. This function emits a straightforward Y-phase
    extraction of each ``y_v`` (one ancilla, RX → CX/CY/CZ → MX) that compiles to a
    DEM, which is the Task-4 acceptance. The Bell/flag cell is a distance
    refinement validated by the operational-distance task; it is added only if
    that task finds the operational distance has collapsed to 1.

    obs0 readout (§III.C product). Ide, Gowda, Nadkarni, Dauphinais
    arXiv:2410.02753 §III.C: the obs0 eigenvalue is the XOR of the IN-CIRCUIT
    ancilla records of the merged-code rows whose product equals Ȳ on the
    original data columns. The picker ``_ybar_obs0_rows`` solves over GF(2) for
    the product whose data restriction is the literal Ȳ support ``[x | z]`` (X on
    V_X, Z on V_Z, Y on W) and is eigenbasis-compatible on the κ ancillas (Z-only
    κ_x, X-only κ_z). obs0 is the FAULT-TOLERANT readout: the XOR of those rows'
    IN-CIRCUIT last-QEC-round ancilla outcomes (S_X' → ``checks_x`` M-record, S_Z' →
    ``checks_z`` M-record, q1 → ``y_ancilla`` MX-record), the same mechanism the
    X/Z-measurement sibling ``_surgery_observable`` uses on the final QEC round.

    Determinism gate. The bare ``[x | z]`` product carries Pauli-X on V_X /
    Pauli-Z on V_Z data qubits, so it is deterministic only on a PROPER Ȳ-
    eigenstate codeword — not on a non-eigenstate prep. With ``data_init`` in
    ``("Y+", "Y-")`` the data is the EXACT |Ȳ±⟩ codeword (state injection |X̄+⟩
    then transversal S), on which every code stabilizer is +1, so the bare
    product agrees with Ȳ and the in-circuit XOR is DETERMINISTIC; obs0 is
    emitted (DEM compiles). With the |0̄⟩/|+̄⟩ preps Ȳ anticommutes with the
    prepared logical, so the XOR is a genuine 50/50 and
    ``_observable_is_deterministic`` gates obs0 OFF unless ``force_obs0`` is set.
    The bare product equals Ȳ = iX̄Z̄ EXACTLY (for Steane ``[x | z] = X₂X₄Z₁Z₃Y₅``
    and ``iX̄Z̄ = +X₂X₄Z₁Z₃Y₅``), so the raw obs0 bit is the Ȳ eigenvalue bit:
    obs0 = 0 ↔ Ȳ = +1 (|Ȳ+⟩), obs0 = 1 ↔ Ȳ = −1 (|Ȳ-⟩). The per-system Cheeger
    boost
    (Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.D) is the
    FAULT-DISTANCE refinement, applied
    inside ``build_y_gadget``; it is not needed for this readout.
    """
    # Alias resolution (backward-compatible names; must happen before _y_state_prep).
    data_init = {"Z+": None, "X+": "+"}.get(data_init, data_init)

    if sum([bool(force_obs0), memory_logical is not None]) > 1:
        raise ValueError(
            "force_obs0 and memory_logical each use observable index 0; set at most one"
        )
    if memory_logical is not None and not destructive_measure_data:
        raise ValueError(
            "memory_logical reads a surviving logical from the destructive data "
            "readout; it is incompatible with destructive_measure_data=False"
        )

    # --- Phase 1: QUBIT_COORDS + state prep -----------------------------------
    circuit, ctx = _y_state_prep(yg, data_init=data_init)

    # --- Phase 2: split X/Z/Y QEC cycle (H̃ blocks 1,2 / 4,5 / 3) ------------
    # Batch every per-round ancilla reset into one front layer for the noiseless
    # circuit (clean timeline diagram); keep tight just-in-time resets under noise
    # (minimal ancilla idle for LER). Semantics-free either way (see _y_qec_cycle).
    qec_circuit, measurement_record, row_to_check, qubit_final_meas = _y_qec_cycle(
        ctx, yg, data_init=data_init, rounds=rounds, batch_resets=noise_model is None
    )
    circuit += qec_circuit

    # --- Phase 3: detach (+ destructive readout unless detach-only) -----------
    readout_circuit = _y_detach_and_readout(
        ctx, data_init=data_init,
        measurement_record=measurement_record,
        destructive_measure_data=destructive_measure_data,
    )
    circuit += readout_circuit

    # --- Phase 4: final detectors (need the destructive data readout) ---------
    if destructive_measure_data:
        circuit += _y_final_detectors(
            ctx,
            row_to_check=row_to_check,
            qubit_final_meas=qubit_final_meas,
            measurement_record=measurement_record,
        )

    # --- Phase 5a: obs0 (Ȳ eigenvalue, §III.C IN-CIRCUIT readout product) ----
    _y_emit_obs0(
        ctx, circuit, yg,
        data_init=data_init, force_obs0=force_obs0,
        measurement_record=measurement_record,
    )

    # --- Phase 5b: survivor-memory observable ---------------------------------
    if memory_logical is not None and data_init is None:
        _y_emit_survivor_memory(
            ctx, circuit,
            memory_logical=memory_logical, data_init=data_init,
            measurement_record=measurement_record,
        )

    if noise_model is not None:
        circuit = noise_model.noisy_circuit(circuit)

    return circuit


def _observable_is_deterministic(
    circuit: stim.Circuit, obs_targets: list[stim.GateTarget]
) -> bool:
    """Return True iff the XOR of ``obs_targets`` is deterministic in ``circuit``.

    Appends a probe OBSERVABLE_INCLUDE to a copy of ``circuit`` (which must be
    noiseless at this point) and asks stim whether it compiles: a
    non-deterministic observable makes ``detector_error_model()`` raise. We
    catch that to decide whether obs0 can be emitted. Used to gate obs0 in the
    regime where the Ȳ readout has an unpinned κ-gauge residual (Ide, Gowda,
    Nadkarni, Dauphinais arXiv:2410.02753 §III.C).
    """
    probe = circuit.copy()
    probe.append("OBSERVABLE_INCLUDE", obs_targets, 0)
    try:
        probe.detector_error_model()
        return True
    except ValueError:
        return False


def _split_leading_reset(phase: stim.Circuit) -> tuple[list[int], stim.Circuit]:
    """Split a phase circuit's leading ancilla reset off its body.

    Both the EdgeColoring CSS phases and the Y-phase emit their ancilla reset
    (``RX``) as the very first instruction (EdgeColoring then follows it with a
    ``TICK`` before the first gate layer). Returns ``(reset_targets, body)``:
    the qubit targets of that leading ``RX``, and the phase with the ``RX`` and
    any immediately-following ``TICK``\\ s removed (so re-emitting ``TICK`` +
    ``body`` yields exactly one moment barrier, no blank column). If the phase
    does not start with ``RX`` it is returned unchanged with no targets.

    Used by ``_y_qec_cycle`` under ``batch_resets`` to hoist all phase resets
    into a single reset layer for a clean timeline diagram.
    """
    insts = list(phase)
    if not insts or insts[0].name != "RX":
        return [], phase
    targets = [int(t.value) for t in insts[0].targets_copy()]
    i = 1
    while i < len(insts) and insts[i].name == "TICK":
        i += 1
    return targets, phase[i:]


def _split_trailing_measure(body: stim.Circuit) -> tuple[stim.Circuit, list[int]]:
    """Split a phase body's trailing ``MX`` off its gate layers.

    A phase body (the phase with its leading reset already removed by
    ``_split_leading_reset``) ends with the single ``MX`` that reads the phase's
    ancillas, optionally preceded by a ``TICK``. Returns ``(gates, mx_targets)``
    with that trailing ``MX`` and any ``TICK``\\ s right before it removed, so the
    caller can re-emit the gates and a *merged* measurement with deliberate
    moment barriers. If the body does not end with ``MX`` it is returned
    unchanged with no targets.

    Used by ``_y_qec_cycle`` under ``batch_resets`` to merge the X- and Z-phase
    ancilla readouts into one measurement moment.
    """
    insts = list(body)
    if not insts or insts[-1].name != "MX":
        return body, []
    targets = [int(t.value) for t in insts[-1].targets_copy()]
    g = len(insts) - 1
    while g > 0 and insts[g - 1].name == "TICK":
        g -= 1
    return body[:g], targets


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
