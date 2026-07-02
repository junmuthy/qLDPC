"""Surgery QEC engine (split from the former circuit.py).

The basis-agnostic phases shared by the single- and joint-PPM CSS builders:
reliable-check classification, state prep, the multi-round QEC cycle, the
observable set, the final detectors, and detach/readout.

References:
    Cain et al. arXiv:2603.28627 Appendix D  — single-PPM measurement protocol.
"""

from __future__ import annotations

import numpy as np
import stim

from qldpc.circuits.bookkeeping import DetectorRecord, MeasurementRecord, QubitIDs
from qldpc.circuits.memory.syndrome_measurement import EdgeColoring
from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli, PauliXZ

from ..hmatrix.PPM_joint import Bridge
from ..hmatrix.PPM_X_Z import GadgetLayout
from .support import (
    _block_observable_targets,
    _check_lane_index_map,
    _commuting_logical_basis,
)


def _reliable_checks(
    gadget: GadgetLayout,
    merged_code: CSSCode,
    qubit_ids: QubitIDs,
    *,
    experiment_basis: PauliXZ,
    n_data: int,
    joint: tuple[GadgetLayout, Bridge, bool] | None = None,
) -> tuple[int, ...]:
    """Merged checks deterministic at round 1 == reconstructable from final readout.

    A CSS check is deterministic iff every qubit in its support is initialized in
    the basis matching the check's Pauli type. Data qubits are in
    ``experiment_basis``; ``Q'``/bridge qubits are in complement(``gadget.basis``).
    This is the unifying rule of the design (Cain et al. arXiv:2603.28627
    Appendix D): the round-1-deterministic set and the final-reconstructable set
    coincide, computed directly from the merged check matrices + the per-qubit
    init basis (no index-slicing, so it is correct for single and joint alike).

    Emission order matches ``qubit_ids.check`` (all X-checks, then all Z-checks),
    so the round-1 and final detectors land in the same order as before.

    ``joint`` is accepted for signature parity with the QEC-cycle / final-detector
    call sites; the per-qubit basis vector already accounts for both gadgets'
    data columns (via ``n_data``) and the shared ``Q'``/bridge block, so the rule
    needs no extra branch for the joint case.
    """
    del joint  # rule is purely support + per-qubit init basis
    HX = np.asarray(merged_code.matrix_x).astype(np.uint8)
    HZ = np.asarray(merged_code.matrix_z).astype(np.uint8)
    n_merged = merged_code.num_qudits
    # per-qubit init basis: True = X-basis init, False = Z-basis init.
    # X-gadget -> Q'/bridge in Z (False); Z-gadget -> Q'/bridge in X (True).
    anc_is_x = gadget.basis is Pauli.Z
    data_is_x = experiment_basis is Pauli.X
    x_init = np.zeros(n_merged, dtype=bool)
    x_init[:n_data] = data_is_x
    x_init[n_data:] = anc_is_x

    reliable: list[int] = []
    # X-type checks: deterministic iff support is fully within X-init qubits.
    for r in range(HX.shape[0]):
        supp = np.nonzero(HX[r])[0]
        if supp.size and x_init[supp].all():
            reliable.append(qubit_ids.checks_x[r])
    # Z-type checks: deterministic iff support is fully within Z-init qubits.
    for r in range(HZ.shape[0]):
        supp = np.nonzero(HZ[r])[0]
        if supp.size and (~x_init[supp]).all():
            reliable.append(qubit_ids.checks_z[r])
    return tuple(reliable)


def _surgery_state_prep(
    gadget: GadgetLayout,
    data_ids: tuple[int, ...],
    ancilla_ids: tuple[int, ...],
    bridge_ids: tuple[int, ...] = (),
    *,
    experiment_basis: PauliXZ,
    data_init: str | None = None,
) -> stim.Circuit:
    """Init data/ancilla/bridge qubits at the start of a surgery PPM circuit.

    Data default init (``data_init=None``) follows ``experiment_basis``, which is
    decoupled from ``gadget.basis`` (Cain et al. arXiv:2603.28627 Appendix D):
      experiment_basis=X → data |+⟩ (RX)
      experiment_basis=Z → data |0⟩ (R)

    ancilla + bridge init follows the COMPLEMENT of ``gadget.basis`` (the merge
    mechanics, independent of ``experiment_basis``):
      basis=X → ancilla + bridge |0⟩ (R)
      basis=Z → ancilla + bridge |+⟩ (RX)

    Optional ``data_init`` overrides per-data-qubit initial state. Each
    character selects a state for the data qubit at the same position:

      "0" → |0⟩  (R)
      "1" → |1⟩  (R + post-init X)
      "+" → |+⟩  (RX)
      "-" → |-⟩  (RX + post-init Z)

    A length-1 string broadcasts to all data qubits; otherwise length must
    equal ``len(data_ids)``.  ancilla + bridge init is independent of ``data_init``
    and always follows the protocol default (basis-complement +1 eigenstate).
    """
    if data_init is None:
        default_char = "+" if experiment_basis is Pauli.X else "0"
        per_qubit = default_char * len(data_ids)
    else:
        if len(data_init) == 1:
            data_init = data_init * len(data_ids)
        if len(data_init) != len(data_ids):
            raise ValueError(
                f"data_init length {len(data_init)} does not match num data "
                f"qubits {len(data_ids)}; pass a length-1 string to broadcast"
            )
        invalid = sorted(set(data_init) - set("01+-"))
        if invalid:
            raise ValueError(
                f"data_init must contain only '0', '1', '+', '-'; got invalid chars {invalid}"
            )
        per_qubit = data_init

    r_data: list[int] = []
    rx_data: list[int] = []
    x_after: list[int] = []
    z_after: list[int] = []
    for q, c in zip(data_ids, per_qubit):
        if c == "0":
            r_data.append(q)
        elif c == "1":
            r_data.append(q)
            x_after.append(q)
        elif c == "+":
            rx_data.append(q)
        else:  # "-"
            rx_data.append(q)
            z_after.append(q)

    circuit = stim.Circuit()
    if r_data:
        circuit.append("R", r_data)
    if rx_data:
        circuit.append("RX", rx_data)
    if x_after:
        circuit.append("X", x_after)
    if z_after:
        circuit.append("Z", z_after)

    anc_ids = list(ancilla_ids) + (list(bridge_ids) if bridge_ids else [])
    if anc_ids:
        anc_init = "R" if gadget.basis is Pauli.X else "RX"
        circuit.append(anc_init, anc_ids)

    return circuit


def _surgery_qec_cycle(
    gadget: GadgetLayout,
    merged_code: CSSCode,
    num_rounds: int,
    qubit_ids: QubitIDs,
    *,
    experiment_basis: PauliXZ,
    n_data: int,
    joint: tuple[GadgetLayout, Bridge, bool] | None = None,
    single_sector: bool = False,
) -> tuple[stim.Circuit, MeasurementRecord, DetectorRecord]:
    """num_rounds of merged-code SE; round-1 detectors only for reliable checks.

    Single-PPM (``joint=None``) and joint-PPM (``joint=(g_r, bridge,
    intercode)``) share one round loop; only the reliable-check classifier and
    the check→lane map differ by whether the right gadget + bridge participate.

    ``single_sector`` (CSS-type PPM only): emit DETECTORs for the measured-basis
    checks alone (``checks_x`` for X̄, ``checks_z`` for Z̄), dropping the
    complementary sector. All checks are still *measured* (the merge needs them);
    only their detectors are skipped. Valid because the time-like L (X̄/Z̄) is flipped
    solely by the opposite single error type, which fires the measured-basis sector — so
    the complementary detectors carry no L fault distance, only correlated
    soft-info / off-basis error correction (arXiv:2410.02753 §3).
    """
    strategy = EdgeColoring()
    one_round, round_measurement_record = strategy.get_circuit(merged_code, qubit_ids)
    reliable = set(
        _reliable_checks(
            gadget,
            merged_code,
            qubit_ids,
            experiment_basis=experiment_basis,
            n_data=n_data,
            joint=joint,
        )
    )
    lane_idx = _check_lane_index_map(gadget, qubit_ids, joint=joint)
    all_check_ids = qubit_ids.check
    # Checks whose syndrome becomes a DETECTOR. single_sector keeps only the
    # measured-basis sector; all checks are still measured by ``one_round``.
    if single_sector:
        # Key off experiment_basis (the OBSERVABLE Pauli type), not gadget.basis:
        # the observables (block + time-like L) are experiment_basis-typed, so they
        # are flipped by the opposite single-qubit error type, caught by the
        # experiment_basis-stabilizer sector. (Match-basis: experiment_basis ==
        # gadget.basis, so this is unchanged; opposite-basis: keeps the correct
        # sector — gadget.basis would drop the detectors that catch the flips.)
        measured = set(qubit_ids.checks_x if experiment_basis is Pauli.X else qubit_ids.checks_z)
        detector_check_ids = tuple(cid for cid in all_check_ids if cid in measured)
    else:
        detector_check_ids = tuple(all_check_ids)

    circuit = stim.Circuit()
    measurement_record = MeasurementRecord()
    detector_record = DetectorRecord()

    # Round 1: emit DETECTORs only for reliable checks.
    circuit += one_round
    measurement_record.append(round_measurement_record)
    for check_id in detector_check_ids:
        if check_id in reliable:
            lane, idx = lane_idx[check_id]
            circuit.append(
                "DETECTOR", [measurement_record.get_target_rec(check_id)], (idx, lane, 0)
            )
    reliable_in_order = [cid for cid in detector_check_ids if cid in reliable]
    detector_record.append({cid: dd for dd, cid in enumerate(reliable_in_order)})

    if num_rounds > 1:
        repeat_circuit = one_round.copy()
        measurement_record.append(round_measurement_record)
        repeat_circuit.append("SHIFT_COORDS", [], (0, 0, 1))
        for check_id in detector_check_ids:
            lane, idx = lane_idx[check_id]
            repeat_circuit.append(
                "DETECTOR",
                [
                    measurement_record.get_target_rec(check_id, -1),
                    measurement_record.get_target_rec(check_id, -2),
                ],
                (idx, lane, 0),
            )
        circuit.append(stim.CircuitRepeatBlock(num_rounds - 1, repeat_circuit))
        measurement_record.append(round_measurement_record, repeat=num_rounds - 2)
        detector_record.append(
            {cid: dd for dd, cid in enumerate(detector_check_ids)},
            repeat=num_rounds - 1,
        )

    return circuit, measurement_record, detector_record


def _surgery_observable(
    gadget: GadgetLayout,
    *,
    experiment_basis: PauliXZ,
    merged_code: CSSCode,
    meas_check_ids: tuple[int, ...],
    logical_ops: np.ndarray,
    L_support: np.ndarray,
    n_data: int,
    data_ids: tuple[int, ...],
    qprime_ids: tuple[int, ...],
    bridge_ids: tuple[int, ...],
    measurement_record: MeasurementRecord,
) -> stim.Circuit:
    """Emit the Cain et al. arXiv:2603.28627 Appendix D surgery observable set.

    Block (space-like) observables: the commuting basis of ``experiment_basis``
    logicals (k if match-basis, k-1 if opposite — selected by
    ``_commuting_logical_basis`` against ``L_support``), each frame-corrected onto
    the Q'/bridge split records by ``_block_observable_targets``. They are emitted
    at indices ``0..m-1`` (m = number of basis rows).

    Time-like L observable (match-basis only, i.e. ``experiment_basis is
    gadget.basis``): the XOR of the FIRST-cycle S'_meas outcomes
    (``get_target_rec(cid, 0)`` over ``meas_check_ids``). It is appended at index
    ``m``, giving the k+1 layout. In the opposite-basis case L is not a readable
    eigenvalue (the data is initialized/read in the complementary basis), so only
    the k-1 block observables are emitted.

    ``logical_ops`` is the bare-code ``k×n_data`` support matrix of the
    ``experiment_basis`` logicals; ``L_support`` is the measured operator support
    (length n_data). For joint PPM these are the combined (block-diagonal /
    XOR'd) constructions assembled by the joint builder.
    """
    # column -> end measurement record (data final M; Q'/bridge split M)
    col_record: dict[int, stim.target_rec] = {}
    for col in range(n_data):
        col_record[col] = measurement_record.get_target_rec(data_ids[col])
    non_data_ids = tuple(qprime_ids) + tuple(bridge_ids)
    for j, qid in enumerate(non_data_ids):
        col_record[n_data + j] = measurement_record.get_target_rec(qid)

    circuit = stim.Circuit()
    # The measured operator L is ``gadget.basis``-type. A block logical of type
    # ``experiment_basis`` commutes with L automatically when match-basis (same
    # Pauli type always commute -> all k kept, no frame correction); only the
    # opposite-basis dot ``z_i · supp(L)`` is the genuine symplectic product, so we
    # zero the functional in the match-basis case (design §3.1) to keep all k.
    match_basis = experiment_basis is gadget.basis
    sympl_L = np.zeros(logical_ops.shape[1], dtype=np.uint8) if match_basis else L_support
    basis_ops = _commuting_logical_basis(logical_ops, sympl_L)
    idx = 0
    for w in basis_ops:
        targets = _block_observable_targets(merged_code, experiment_basis, w, n_data, col_record)
        circuit.append("OBSERVABLE_INCLUDE", targets, idx)
        idx += 1
    if match_basis:
        # Precondition: every meas-check ancilla was measured during the QEC cycle.
        for cid in meas_check_ids:
            assert measurement_record[cid], (
                f"meas-check {cid} has no measurement record; "
                f"_surgery_observable expects the QEC cycle to have run first."
            )
        L_targets = [measurement_record.get_target_rec(cid, 0) for cid in meas_check_ids]
        circuit.append("OBSERVABLE_INCLUDE", L_targets, idx)
    return circuit


def _surgery_final_detectors(
    gadget: GadgetLayout,
    merged_code: CSSCode,
    qubit_ids: QubitIDs,
    *,
    measurement_record: MeasurementRecord,
    experiment_basis: PauliXZ,
    n_data: int,
    joint: tuple[GadgetLayout, Bridge, bool] | None = None,
    single_sector: bool = False,
) -> stim.Circuit:
    """Emit DETECTORs for checks reconstructable from the final readouts.

    The reconstructable set is exactly ``_reliable_checks`` (same per-qubit basis
    rule used for round-1 detectors): a CSS check is final-reconstructable iff its
    support lies entirely within matching-basis qubits (data in
    ``experiment_basis``; ``Q'``/bridge in complement(``gadget.basis``)). For the
    match-basis paths this reproduces the prior "data H_X from M data + gauge G
    from M Q'" split exactly (Cain et al. arXiv:2603.28627 Appendix D).

    Each DETECTOR XORs ⊕(final M-record on the stab support) ⊕ last-round syndrome.
    Joint-PPM (``joint=(g_r, bridge, intercode)``) spans both gadgets' data rows
    automatically via ``n_data``.

    ``single_sector`` drops the complementary-basis gauge (G) detectors, matching
    the QEC-cycle filter — only the measured-basis data stabs are inferred.
    """
    HX = np.asarray(merged_code.matrix_x).astype(np.uint8)
    HZ = np.asarray(merged_code.matrix_z).astype(np.uint8)
    reliable = set(
        _reliable_checks(
            gadget,
            merged_code,
            qubit_ids,
            experiment_basis=experiment_basis,
            n_data=n_data,
            joint=joint,
        )
    )
    if single_sector:
        # Keep only measured-basis checks (drops the complementary gauge G). Key off
        # experiment_basis (the data readout / OBSERVABLE Pauli type), NOT gadget.basis,
        # to match the QEC-cycle round-detector filter and _reliable_checks: the final
        # detectors reconstruct the experiment_basis-sector data stabs from the M-record.
        # Opposite-basis (experiment_basis != gadget.basis) previously intersected the
        # reconstructable checks with the wrong sector -> ∅ final detectors -> the k−t
        # block observables were left undetectable (decoder blind, LER → raw flip rate).
        measured = set(
            qubit_ids.checks_x if experiment_basis is Pauli.X else qubit_ids.checks_z
        )
        reliable &= measured

    circuit = stim.Circuit()
    lane_idx = _check_lane_index_map(gadget, qubit_ids, joint=joint)

    def _emit_detector(stab_row: np.ndarray, check_id: int) -> None:
        supp = np.where(stab_row)[0]
        targets = [measurement_record.get_target_rec(qubit_ids.data[q]) for q in supp]
        targets.append(measurement_record.get_target_rec(check_id, -1))
        lane, idx = lane_idx[check_id]
        circuit.append("DETECTOR", targets, (idx, lane, 0))

    def _emit_sector(H: np.ndarray, check_ids: tuple[int, ...]) -> None:
        for kk in range(H.shape[0]):
            cid = check_ids[kk]
            if cid in reliable:
                _emit_detector(H[kk], cid)

    # Measured-basis sector first, then the complementary sector — matching the
    # prior emission order so existing match-basis circuits stay byte-identical.
    if gadget.basis is Pauli.X:
        _emit_sector(HX, qubit_ids.checks_x)
        _emit_sector(HZ, qubit_ids.checks_z)
    else:
        _emit_sector(HZ, qubit_ids.checks_z)
        _emit_sector(HX, qubit_ids.checks_x)

    return circuit


def _surgery_detach_and_readout(
    gadget: GadgetLayout,
    *,
    data_ids: tuple[int, ...],
    ancilla_ids: tuple[int, ...],
    bridge_ids: tuple[int, ...],
    measurement_record: MeasurementRecord,
    experiment_basis: PauliXZ,
    destructive_measure_data: bool = True,
) -> stim.Circuit:
    """Detach the κ/bridge ancillas; optionally destructively measure the data.

    Mκ (the split that returns the bare code) always runs. When
    ``destructive_measure_data`` is True the data is then measured destructively
    (SHIFT_COORDS then Mdata); when False the data is left encoded (detach-only).

    The ancilla/bridge detach op follows the COMPLEMENT of ``gadget.basis`` (the
    split): basis=X → M, basis=Z → MX. The data readout op follows
    ``experiment_basis`` (decoupled from gadget.basis, Cain et al.
    arXiv:2603.28627 Appendix D): experiment_basis=X → MX, experiment_basis=Z → M.
    """
    circuit = stim.Circuit()
    detach_qubits = list(ancilla_ids) + list(bridge_ids)
    ancilla_op = "M" if gadget.basis is Pauli.X else "MX"
    data_op = "MX" if experiment_basis is Pauli.X else "M"
    circuit.append(ancilla_op, detach_qubits)
    measurement_record.append({q: i for i, q in enumerate(detach_qubits)})
    if not destructive_measure_data:
        return circuit  # detach-only: leave the data encoded
    circuit.append("SHIFT_COORDS", [], (0, 0, 1))
    circuit.append(data_op, list(data_ids))
    measurement_record.append({q: i for i, q in enumerate(data_ids)})
    return circuit
