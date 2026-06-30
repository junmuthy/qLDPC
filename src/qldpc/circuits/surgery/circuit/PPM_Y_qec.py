"""Single-Ȳ surgery: split X/Z/Y QEC cycle + detach/readout (from the former y_circuit.py).

The multi-round split syndrome schedule (X-phase CX, Z-phase CZ, Y-phase
per-row CX/CY/CZ → MX) and the κ-detach + destructive data readout, plus the
two phase-splitter utilities used to batch the noiseless round's resets.

References:
    Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.C/§III.D
        — homological Ȳ = iX̄Z̄ measurement.
"""

from __future__ import annotations

import stim

from qldpc.circuits.bookkeeping import MeasurementRecord, QubitIDs
from qldpc.circuits.memory.syndrome_measurement import EdgeColoring
from qldpc.objects import Pauli

from ..hmatrix.PPM_Y import YGadgetLayout
from .PPM_Y_prep import _YCtx


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
