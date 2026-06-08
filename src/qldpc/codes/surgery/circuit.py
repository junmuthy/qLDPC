"""Stim surgery circuit construction (single-PPM and joint-PPM)."""

from __future__ import annotations

import numpy as np
import stim

from qldpc.circuits.bookkeeping import MeasurementRecord, DetectorRecord, QubitIDs
from qldpc.circuits.memory.memory import get_qubit_coordinates
from qldpc.circuits.memory.syndrome_measurement import EdgeColoring
from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli

from .bridge import Bridge
from .gadget import GadgetLayout


def _gadget_merged_csscode(g: GadgetLayout) -> CSSCode:
    return CSSCode(
        g.HX_merged.astype(np.int_),
        g.HZ_merged.astype(np.int_),
        is_subsystem_code=False,
    )


def build_single_ppm_circuit(
    gadget: GadgetLayout,
    *,
    rounds: int,
    noise_model=None,
) -> stim.Circuit:
    """Cain §III.A single-PPM measurement circuit for `gadget`."""
    merged_code = _gadget_merged_csscode(gadget)
    qubit_ids = QubitIDs.from_code(merged_code)
    n_data = gadget.code.num_qudits
    data_ids = qubit_ids.data[:n_data]
    kappa_ids = qubit_ids.data[n_data:]
    bridge_ids: tuple[int, ...] = ()

    circuit = get_qubit_coordinates(qubit_ids.data, qubit_ids.check)
    circuit += _surgery_state_prep(gadget, data_ids, kappa_ids, bridge_ids)
    qec_cycle, measurement_record, _ = _surgery_qec_cycle(
        gadget, merged_code, num_rounds=rounds, qubit_ids=qubit_ids,
    )
    circuit += qec_cycle
    circuit += _surgery_detach_and_readout(
        gadget, data_ids=data_ids, kappa_ids=kappa_ids, bridge_ids=bridge_ids,
        measurement_record=measurement_record,
    )

    m_X, m_Z, n_V = gadget.code.matrix_x.shape[0], gadget.code.matrix_z.shape[0], len(gadget.V0)
    if gadget.basis is Pauli.X:
        chi_check_ids = tuple(qubit_ids.checks_x[m_X : m_X + n_V])
    else:
        chi_check_ids = tuple(qubit_ids.checks_z[m_Z : m_Z + n_V])

    circuit += _surgery_observable(
        gadget,
        chi_check_ids=chi_check_ids,
        data_ids=data_ids,
        v0_indices=gadget.V0,
        num_rounds=rounds,
        measurement_record=measurement_record,
    )

    if noise_model is not None:
        circuit = noise_model.noisy_circuit(circuit)

    return circuit


def _stitch_to_joint_csscode(
    g1: GadgetLayout,
    g2: GadgetLayout,
    bridge: Bridge,
) -> CSSCode:
    """Assemble joint CSS code for two-PPM surgery (math.md §2.5–2.6)."""
    intercode = g1.code is not g2.code
    field = g1.code.field

    n_data_1 = g1.code.num_qubits
    n_data_2 = g2.code.num_qubits if intercode else 0
    n_anc_1, n_anc_2 = len(g1.C0), len(g2.C0)
    n_bridge = bridge.width
    n_merged = n_data_1 + n_data_2 + n_anc_1 + n_anc_2 + n_bridge
    mX1, mZ1 = int(g1.code.matrix_x.shape[0]), int(g1.code.matrix_z.shape[0])
    mX2, mZ2 = int(g2.code.matrix_x.shape[0]), int(g2.code.matrix_z.shape[0])
    HX1 = np.asarray(g1.HX_merged).astype(np.int_)
    HZ1 = np.asarray(g1.HZ_merged).astype(np.int_)
    HX2 = np.asarray(g2.HX_merged).astype(np.int_)
    HZ2 = np.asarray(g2.HZ_merged).astype(np.int_)

    if intercode:
        # Columns: [data_1 | data_2 | kappa_1 | kappa_2 | bridge]
        anc_off_1 = n_data_1 + n_data_2
        anc_off_2 = anc_off_1 + n_anc_1
        bridge_col_start = anc_off_2 + n_anc_2

        def _pad_g1(m: np.ndarray) -> np.ndarray:
            out = np.zeros((m.shape[0], n_merged), dtype=np.int_)
            out[:, :n_data_1] = m[:, :n_data_1]
            out[:, anc_off_1 : anc_off_1 + n_anc_1] = m[:, n_data_1:]
            return out
        def _pad_g2(m: np.ndarray) -> np.ndarray:
            out = np.zeros((m.shape[0], n_merged), dtype=np.int_)
            out[:, n_data_1 : n_data_1 + n_data_2] = m[:, :n_data_2]
            out[:, anc_off_2 : anc_off_2 + n_anc_2] = m[:, n_data_2:]
            return out
        HX1_pad, HZ1_pad = _pad_g1(HX1), _pad_g1(HZ1)
        HX2_pad, HZ2_pad = _pad_g2(HX2), _pad_g2(HZ2)
        # Chi row 0 of each gadget connects to its bridge endpoint.
        if g1.basis is Pauli.X:
            HX1_pad[mX1, bridge_col_start] = 1
            HX2_pad[mX2, bridge_col_start + n_bridge - 1] = 1
        else:
            HZ1_pad[mZ1, bridge_col_start] = 1
            HZ2_pad[mZ2, bridge_col_start + n_bridge - 1] = 1

        u_b = np.asarray(bridge.U_B).astype(np.int_)
        u_b_pad = np.zeros((u_b.shape[0], n_merged), dtype=np.int_)
        if u_b.shape[0] > 0:
            u_b_pad[:, bridge_col_start : bridge_col_start + n_bridge] = u_b

        if g1.basis is Pauli.X:
            HX_joint = field(np.vstack([HX1_pad, HX2_pad, u_b_pad]))
            HZ_joint = field(np.vstack([HZ1_pad, HZ2_pad]))
        else:
            HX_joint = field(np.vstack([HX1_pad, HX2_pad]))
            HZ_joint = field(np.vstack([HZ1_pad, HZ2_pad, u_b_pad]))

        return CSSCode(HX_joint, HZ_joint, is_subsystem_code=False)

    # Intra-code: shared data qubits.  Columns: [data | kappa_1 | kappa_2 | bridge]
    mX, mZ = mX1, mZ1

    def _pad(matrix: np.ndarray, *, anc_offset: int) -> np.ndarray:
        n_anc = matrix.shape[1] - n_data_1
        out = np.zeros((matrix.shape[0], n_merged), dtype=np.int_)
        out[:, :n_data_1] = matrix[:, :n_data_1]
        out[:, anc_offset : anc_offset + n_anc] = matrix[:, n_data_1:]
        return out

    anc_off_1 = n_data_1
    anc_off_2 = n_data_1 + n_anc_1
    bridge_col_start = n_data_1 + n_anc_1 + n_anc_2
    HX1_pad, HZ1_pad = _pad(HX1, anc_offset=anc_off_1), _pad(HZ1, anc_offset=anc_off_1)
    HX2_pad, HZ2_pad = _pad(HX2, anc_offset=anc_off_2), _pad(HZ2, anc_offset=anc_off_2)

    # Chi row 0 of each gadget connects to its bridge endpoint.
    if g1.basis is Pauli.X:
        HX1_pad[mX, bridge_col_start] = 1
        HX2_pad[mX, bridge_col_start + n_bridge - 1] = 1
    else:
        HZ1_pad[mZ, bridge_col_start] = 1
        HZ2_pad[mZ, bridge_col_start + n_bridge - 1] = 1
    for k, j in enumerate(g2.C0):
        HZ1_pad[int(j), anc_off_2 + k] = 1

    u_b = np.asarray(bridge.U_B).astype(np.int_)
    u_b_pad = np.zeros((u_b.shape[0], n_merged), dtype=np.int_)
    if u_b.shape[0] > 0:
        u_b_pad[:, bridge_col_start : bridge_col_start + n_bridge] = u_b

    if g1.basis is Pauli.X:
        HX_joint = field(np.vstack([HX1_pad, HX2_pad[mX:], u_b_pad]))
        HZ_joint = field(np.vstack([HZ1_pad, HZ2_pad[mZ:]]))
    else:
        HX_joint = field(np.vstack([HX1_pad, HX2_pad[mX:]]))
        HZ_joint = field(np.vstack([HZ1_pad, HZ2_pad[mZ:], u_b_pad]))

    return CSSCode(HX_joint, HZ_joint, is_subsystem_code=False)


def build_joint_ppm_circuit(
    g1: GadgetLayout,
    g2: GadgetLayout,
    bridge: Bridge,
    *,
    rounds: int,
    noise_model=None,
) -> tuple[stim.Circuit, CSSCode]:
    """Cain §III.A joint-PPM circuit; chi = χ^(1) ∪ χ^(2) ∪ U_B (math.md §2.7)."""
    joint_code = _stitch_to_joint_csscode(g1, g2, bridge)
    qubit_ids = QubitIDs.from_code(joint_code)
    intercode = g1.code is not g2.code
    n1 = g1.code.num_qudits
    n2 = g2.code.num_qudits if intercode else 0
    n_anc = len(g1.C0) + len(g2.C0)

    if intercode:
        data_ids = qubit_ids.data[: n1 + n2]
        v0_indices_combined = tuple(g1.V0) + tuple(n1 + i for i in g2.V0)
    else:
        data_ids = qubit_ids.data[:n1]
        v0_indices_combined = tuple(g1.V0) + tuple(g2.V0)

    kappa_ids = qubit_ids.data[n1 + n2 : n1 + n2 + n_anc]
    bridge_ids = qubit_ids.data[n1 + n2 + n_anc :]

    circuit = get_qubit_coordinates(qubit_ids.data, qubit_ids.check)
    circuit += _surgery_state_prep(g1, data_ids, kappa_ids, bridge_ids)
    qec_cycle, measurement_record, _ = _surgery_qec_cycle(
        g1, joint_code, num_rounds=rounds, qubit_ids=qubit_ids,
    )
    circuit += qec_cycle
    circuit += _surgery_detach_and_readout(
        g1, data_ids=data_ids, kappa_ids=kappa_ids, bridge_ids=bridge_ids,
        measurement_record=measurement_record,
    )

    # Chi check_ids: χ^(1) ∪ χ^(2) ∪ U_B (math.md §2.7). Row offsets mirror _stitch_to_joint_csscode.
    mX1, mZ1 = g1.code.matrix_x.shape[0], g1.code.matrix_z.shape[0]
    mX2 = g2.code.matrix_x.shape[0] if intercode else 0
    mZ2 = g2.code.matrix_z.shape[0] if intercode else 0
    n_V1, n_V2 = len(g1.V0), len(g2.V0)
    n_UB = bridge.U_B.shape[0]

    if g1.basis is Pauli.X:
        check_ids = qubit_ids.checks_x
        m1, m2 = mX1, mX2
    else:
        check_ids = qubit_ids.checks_z
        m1, m2 = mZ1, mZ2

    chi1_ids = tuple(check_ids[m1 : m1 + n_V1])
    off2 = m1 + n_V1 + m2   # offset to χ^(2) block (m2=0 for intracode)
    chi2_ids = tuple(check_ids[off2 : off2 + n_V2])
    ub_ids = tuple(check_ids[off2 + n_V2 : off2 + n_V2 + n_UB])
    chi_check_ids = chi1_ids + chi2_ids + ub_ids

    circuit += _surgery_observable(
        g1,
        chi_check_ids=chi_check_ids,
        data_ids=data_ids,
        v0_indices=v0_indices_combined,
        num_rounds=rounds,
        measurement_record=measurement_record,
    )

    if noise_model is not None:
        circuit = noise_model.noisy_circuit(circuit)

    return circuit, joint_code


def _classify_reliable_round1_checks(
    gadget: GadgetLayout,
    qubit_ids: QubitIDs,
) -> tuple[int, ...]:
    """Check ancillas with deterministic round-1 syndrome given surgery init state."""
    m_X, m_Z = gadget.code.matrix_x.shape[0], gadget.code.matrix_z.shape[0]
    if gadget.basis is Pauli.X:
        reliable_x = qubit_ids.checks_x[:m_X]   # data H_X rows (det. +1)
        reliable_z = qubit_ids.checks_z[m_Z:]    # gauge-fix G rows (det. +1)
    else:
        reliable_x = qubit_ids.checks_x[m_X:]   # gauge-fix G rows
        reliable_z = qubit_ids.checks_z[:m_Z]    # data H_Z rows

    return tuple(reliable_x) + tuple(reliable_z)


def _surgery_state_prep(
    gadget: GadgetLayout,
    data_ids: tuple[int, ...],
    kappa_ids: tuple[int, ...],
    bridge_ids: tuple[int, ...] = (),
) -> stim.Circuit:
    """Init data/κ/bridge: basis=X → data|+⟩, κ|0⟩; basis=Z → data|0⟩, κ|+⟩."""
    circuit = stim.Circuit()
    if gadget.basis is Pauli.X:
        circuit.append("RX", list(data_ids))
        circuit.append("R", list(kappa_ids) + (list(bridge_ids) if bridge_ids else []))
    else:
        circuit.append("R", list(data_ids))
        circuit.append("RX", list(kappa_ids) + (list(bridge_ids) if bridge_ids else []))
    return circuit


def _surgery_qec_cycle(
    gadget: GadgetLayout,
    merged_code: CSSCode,
    num_rounds: int,
    qubit_ids: QubitIDs,
) -> tuple[stim.Circuit, MeasurementRecord, DetectorRecord]:
    """num_rounds of merged-code SE; round-1 detectors only for reliable checks."""
    strategy = EdgeColoring()
    one_round, round_measurement_record = strategy.get_circuit(merged_code, qubit_ids)
    reliable = set(_classify_reliable_round1_checks(gadget, qubit_ids))
    all_check_ids = qubit_ids.check

    circuit = stim.Circuit()
    measurement_record = MeasurementRecord()
    detector_record = DetectorRecord()

    # Round 1: emit DETECTORs only for reliable checks.
    circuit += one_round
    measurement_record.append(round_measurement_record)
    for kk, check_id in enumerate(all_check_ids):
        if check_id in reliable:
            circuit.append("DETECTOR", [measurement_record.get_target_rec(check_id)], (0, 0, kk))
    reliable_in_order = [cid for cid in all_check_ids if cid in reliable]
    detector_record.append({cid: dd for dd, cid in enumerate(reliable_in_order)})

    if num_rounds > 1:
        repeat_circuit = one_round.copy()
        measurement_record.append(round_measurement_record)
        repeat_circuit.append("SHIFT_COORDS", [], (1, 0, 0))
        for kk, check_id in enumerate(all_check_ids):
            repeat_circuit.append("DETECTOR", [
                measurement_record.get_target_rec(check_id, -1),
                measurement_record.get_target_rec(check_id, -2),
            ], (0, 0, kk))
        circuit.append(stim.CircuitRepeatBlock(num_rounds - 1, repeat_circuit))
        measurement_record.append(round_measurement_record, repeat=num_rounds - 2)
        detector_record.append(
            {cid: dd for dd, cid in enumerate(all_check_ids)},
            repeat=num_rounds - 1,
        )

    return circuit, measurement_record, detector_record


def _surgery_observable(
    gadget: GadgetLayout,
    *,
    chi_check_ids: tuple[int, ...],
    data_ids: tuple[int, ...],
    v0_indices: tuple[int, ...],
    num_rounds: int,
    measurement_record: MeasurementRecord,
) -> stim.Circuit:
    """Obs 0 = ⊕ chi-XOR over rounds (Webster Eq. 1); Obs 1 = data on V_0."""
    circuit = stim.Circuit()
    chi_targets = [
        measurement_record.get_target_rec(cid, -1 - r)
        for r in range(num_rounds)
        for cid in chi_check_ids
    ]
    circuit.append("OBSERVABLE_INCLUDE", chi_targets, 0)
    data_targets = [
        measurement_record.get_target_rec(data_ids[i]) for i in v0_indices
    ]
    circuit.append("OBSERVABLE_INCLUDE", data_targets, 1)
    return circuit


def _surgery_detach_and_readout(
    gadget: GadgetLayout,
    *,
    data_ids: tuple[int, ...],
    kappa_ids: tuple[int, ...],
    bridge_ids: tuple[int, ...],
    measurement_record: MeasurementRecord,
) -> stim.Circuit:
    """Cain step 3 + final data measure. Mκ then SHIFT_COORDS then Mdata."""
    circuit = stim.Circuit()
    detach_qubits = list(kappa_ids) + list(bridge_ids)
    kappa_op = "M" if gadget.basis is Pauli.X else "MX"
    data_op = "MX" if gadget.basis is Pauli.X else "M"
    circuit.append(kappa_op, detach_qubits)
    measurement_record.append({q: i for i, q in enumerate(detach_qubits)})
    circuit.append("SHIFT_COORDS", [], (1, 0, 0))
    circuit.append(data_op, list(data_ids))
    measurement_record.append({q: i for i, q in enumerate(data_ids)})
    return circuit
