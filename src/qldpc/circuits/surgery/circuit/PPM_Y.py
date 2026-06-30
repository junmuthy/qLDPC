"""Single logical-Ȳ (Ȳ = iX̄Z̄) measurement circuit — non-CSS homological
surgery (split from the former y_circuit.py).

The orchestrator ``build_single_y_ppm_circuit`` plus the final-detector,
obs0, survivor-memory, and determinism-probe phases it drives. Emits the split
X/Z/Y syndrome schedule over the merged code H̃ (see hmatrix.PPM_Y.build_y_gadget
for the H̃ block layout).

References:
    Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.C/§III.D
        — homological Ȳ = iX̄Z̄ measurement.
"""

from __future__ import annotations

import numpy as np
import stim

from qldpc.circuits.bookkeeping import MeasurementRecord
from qldpc.circuits.noise_model import NoiseModel
from qldpc.objects import Pauli

from ..hmatrix.PPM_Y import YGadgetLayout
from .PPM_Y_prep import _y_state_prep, _YCtx
from .PPM_Y_qec import _y_detach_and_readout, _y_qec_cycle


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
