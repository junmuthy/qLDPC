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


def keep_only_observable(circuit: stim.Circuit, keep_idx: int) -> stim.Circuit:
    """Return a copy of ``circuit`` with all OBSERVABLE_INCLUDE entries dropped
    except the one whose first argument equals ``keep_idx``. Recurses into
    REPEAT blocks so observables inside loops are filtered the same way.

    Useful for sinter LER sweeps that compare one observable against a
    memory-experiment baseline — sinter expects exactly one observable per task.
    """
    out = stim.Circuit()
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            out.append(stim.CircuitRepeatBlock(
                op.repeat_count, keep_only_observable(op.body_copy(), keep_idx),
            ))
            continue
        if op.name == "OBSERVABLE_INCLUDE":
            if int(op.gate_args_copy()[0]) != keep_idx:
                continue
        out.append(op)
    return out


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
    circuit += _surgery_final_detectors(
        gadget, merged_code, qubit_ids,
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
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
) -> CSSCode:
    """Assemble merged CSSCode for two-PPM surgery (spec §3 block tables)."""
    intercode = g_l.code is not g_r.code
    if not intercode and bridge.basis is Pauli.X:
        return _stitch_intracode_joint_csscode(g_l, g_r, bridge)  # added in Task 9
    if not intercode and bridge.basis is Pauli.Z:
        return _stitch_intracode_joint_csscode_basis_z(g_l, g_r, bridge)  # Task 10
    if intercode and bridge.basis is Pauli.Z:
        return _stitch_intercode_joint_csscode_basis_z(g_l, g_r, bridge)  # Task 10

    field = g_l.code.field
    g_l_aug, g_r_aug = bridge.g_l_aug, bridge.g_r_aug
    n_l = g_l.code.num_qudits
    n_r = g_r.code.num_qudits
    k_l = g_l_aug.F.shape[0]      # |C_0^(l)| + e_l
    k_r = g_r_aug.F.shape[0]
    w = bridge.width
    n_merged = n_l + n_r + k_l + k_r + w
    mX_l = g_l.code.matrix_x.shape[0]
    mX_r = g_r.code.matrix_x.shape[0]
    mZ_l = g_l.code.matrix_z.shape[0]
    mZ_r = g_r.code.matrix_z.shape[0]
    r_l = g_l_aug.G.shape[0]      # r_l_aug
    r_r = g_r_aug.G.shape[0]

    # Column ranges
    cl_data = slice(0, n_l)
    cr_data = slice(n_l, n_l + n_r)
    cl_kappa = slice(n_l + n_r, n_l + n_r + k_l)
    cr_kappa = slice(n_l + n_r + k_l, n_l + n_r + k_l + k_r)
    c_adapter = slice(n_l + n_r + k_l + k_r, n_merged)

    HX = np.zeros((mX_l + mX_r + len(g_l.V0) + len(g_r.V0), n_merged), dtype=np.int_)
    HZ = np.zeros((mZ_l + mZ_r + r_l + r_r + (w - 1), n_merged), dtype=np.int_)

    HX_l = np.asarray(g_l_aug.HX_merged).astype(np.int_)
    HX_r = np.asarray(g_r_aug.HX_merged).astype(np.int_)
    HZ_l = np.asarray(g_l_aug.HZ_merged).astype(np.int_)
    HZ_r = np.asarray(g_r_aug.HZ_merged).astype(np.int_)

    # H_X rows: data H_X^(l), data H_X^(r), χ^(l), χ^(r)
    HX[: mX_l, cl_data] = HX_l[: mX_l, : n_l]
    HX[mX_l : mX_l + mX_r, cr_data] = HX_r[: mX_r, : n_r]
    # χ^(l) = HX_l[mX_l:, :]: cols [: n_l] -> cl_data, cols [n_l:] -> cl_kappa
    chi_l_rows = HX_l[mX_l :, :]
    HX[mX_l + mX_r : mX_l + mX_r + len(g_l.V0), cl_data] = chi_l_rows[:, : n_l]
    HX[mX_l + mX_r : mX_l + mX_r + len(g_l.V0), cl_kappa] = chi_l_rows[:, n_l :]
    chi_r_rows = HX_r[mX_r :, :]
    HX[mX_l + mX_r + len(g_l.V0) :, cr_data] = chi_r_rows[:, : n_r]
    HX[mX_l + mX_r + len(g_l.V0) :, cr_kappa] = chi_r_rows[:, n_r :]
    # Π_l, Π_r adapter extensions on χ rows
    for v_idx, lab in enumerate(bridge.label_l):
        if lab >= 0:
            HX[mX_l + mX_r + v_idx, c_adapter.start + lab] = 1
    for v_idx, lab in enumerate(bridge.label_r):
        if lab >= 0:
            HX[mX_l + mX_r + len(g_l.V0) + v_idx, c_adapter.start + lab] = 1

    # H_Z rows: data H_Z^(l) extended, data H_Z^(r) extended, G^(l)_aug, G^(r)_aug, new cycle-Z
    HZ[: mZ_l, cl_data] = HZ_l[: mZ_l, : n_l]
    HZ[: mZ_l, cl_kappa] = HZ_l[: mZ_l, n_l :]
    HZ[mZ_l : mZ_l + mZ_r, cr_data] = HZ_r[: mZ_r, : n_r]
    HZ[mZ_l : mZ_l + mZ_r, cr_kappa] = HZ_r[: mZ_r, n_r :]
    HZ[mZ_l + mZ_r : mZ_l + mZ_r + r_l, cl_kappa] = HZ_l[mZ_l :, n_l :]
    HZ[mZ_l + mZ_r + r_l : mZ_l + mZ_r + r_l + r_r, cr_kappa] = HZ_r[mZ_r :, n_r :]
    # New cycle Z-checks: [T_l | T_r | H_R]
    new_z_start = mZ_l + mZ_r + r_l + r_r
    HZ[new_z_start :, cl_kappa] = bridge.T_l
    HZ[new_z_start :, cr_kappa] = bridge.T_r
    HZ[new_z_start :, c_adapter] = bridge.H_R

    return CSSCode(field(HX), field(HZ), is_subsystem_code=False)


def _stitch_intracode_joint_csscode(g_l, g_r, bridge):
    """Spec §3 intra-code: shared data column block, χ^(l/r) extend onto same data cols."""
    assert g_l.code is g_r.code
    assert bridge.basis is Pauli.X  # basis=Z handled separately
    field = g_l.code.field
    g_l_aug, g_r_aug = bridge.g_l_aug, bridge.g_r_aug
    n = g_l.code.num_qudits
    k_l = g_l_aug.F.shape[0]
    k_r = g_r_aug.F.shape[0]
    w = bridge.width
    n_merged = n + k_l + k_r + w
    mX = g_l.code.matrix_x.shape[0]
    mZ = g_l.code.matrix_z.shape[0]
    r_l = g_l_aug.G.shape[0]
    r_r = g_r_aug.G.shape[0]

    c_data = slice(0, n)
    cl_kappa = slice(n, n + k_l)
    cr_kappa = slice(n + k_l, n + k_l + k_r)
    c_adapter = slice(n + k_l + k_r, n_merged)

    HX_l = np.asarray(g_l_aug.HX_merged).astype(np.int_)
    HX_r = np.asarray(g_r_aug.HX_merged).astype(np.int_)
    HZ_l = np.asarray(g_l_aug.HZ_merged).astype(np.int_)
    HZ_r = np.asarray(g_r_aug.HZ_merged).astype(np.int_)

    HX = np.zeros((mX + len(g_l.V0) + len(g_r.V0), n_merged), dtype=np.int_)
    HZ = np.zeros((mZ + r_l + r_r + (w - 1), n_merged), dtype=np.int_)

    # H_X: shared data H_X (from g_l_aug or g_r_aug, same)
    HX[: mX, c_data] = HX_l[: mX, : n]
    # χ^(l), χ^(r) onto shared data
    chi_l = HX_l[mX :, :]
    chi_r = HX_r[mX :, :]
    HX[mX : mX + len(g_l.V0), c_data] = chi_l[:, : n]
    HX[mX : mX + len(g_l.V0), cl_kappa] = chi_l[:, n :]
    HX[mX + len(g_l.V0) :, c_data] = chi_r[:, : n]
    HX[mX + len(g_l.V0) :, cr_kappa] = chi_r[:, n :]
    for v_idx, lab in enumerate(bridge.label_l):
        if lab >= 0:
            HX[mX + v_idx, c_adapter.start + lab] = 1
    for v_idx, lab in enumerate(bridge.label_r):
        if lab >= 0:
            HX[mX + len(g_l.V0) + v_idx, c_adapter.start + lab] = 1

    # H_Z: shared data H_Z with κ extension on BOTH sides
    HZ[: mZ, c_data] = HZ_l[: mZ, : n]
    HZ[: mZ, cl_kappa] = HZ_l[: mZ, n :]
    HZ[: mZ, cr_kappa] = HZ_r[: mZ, n :]
    HZ[mZ : mZ + r_l, cl_kappa] = HZ_l[mZ :, n :]
    HZ[mZ + r_l : mZ + r_l + r_r, cr_kappa] = HZ_r[mZ :, n :]
    new_z_start = mZ + r_l + r_r
    HZ[new_z_start :, cl_kappa] = bridge.T_l
    HZ[new_z_start :, cr_kappa] = bridge.T_r
    HZ[new_z_start :, c_adapter] = bridge.H_R

    return CSSCode(field(HX), field(HZ), is_subsystem_code=False)


def _stitch_intercode_joint_csscode_basis_z(g_l, g_r, bridge):
    """Symmetric dual of inter-code basis=X: χ rows live in HZ; new cycle in HX."""
    assert g_l.code is not g_r.code
    assert bridge.basis is Pauli.Z
    field = g_l.code.field
    g_l_aug, g_r_aug = bridge.g_l_aug, bridge.g_r_aug
    n_l = g_l.code.num_qudits
    n_r = g_r.code.num_qudits
    k_l = g_l_aug.F.shape[0]
    k_r = g_r_aug.F.shape[0]
    w = bridge.width
    n_merged = n_l + n_r + k_l + k_r + w
    mX_l = g_l.code.matrix_x.shape[0]
    mX_r = g_r.code.matrix_x.shape[0]
    mZ_l = g_l.code.matrix_z.shape[0]
    mZ_r = g_r.code.matrix_z.shape[0]
    r_l = g_l_aug.G.shape[0]
    r_r = g_r_aug.G.shape[0]

    cl_data = slice(0, n_l)
    cr_data = slice(n_l, n_l + n_r)
    cl_kappa = slice(n_l + n_r, n_l + n_r + k_l)
    cr_kappa = slice(n_l + n_r + k_l, n_l + n_r + k_l + k_r)
    c_adapter = slice(n_l + n_r + k_l + k_r, n_merged)

    HX_l = np.asarray(g_l_aug.HX_merged).astype(np.int_)
    HX_r = np.asarray(g_r_aug.HX_merged).astype(np.int_)
    HZ_l = np.asarray(g_l_aug.HZ_merged).astype(np.int_)
    HZ_r = np.asarray(g_r_aug.HZ_merged).astype(np.int_)

    # Roles swapped vs basis=X: χ rows live in H_Z; G rows + new cycle live in H_X.
    HZ = np.zeros((mZ_l + mZ_r + len(g_l.V0) + len(g_r.V0), n_merged), dtype=np.int_)
    HX = np.zeros((mX_l + mX_r + r_l + r_r + (w - 1), n_merged), dtype=np.int_)

    # H_Z rows: data H_Z^(l), data H_Z^(r), χ^(l), χ^(r)
    HZ[: mZ_l, cl_data] = HZ_l[: mZ_l, : n_l]
    HZ[mZ_l : mZ_l + mZ_r, cr_data] = HZ_r[: mZ_r, : n_r]
    chi_l = HZ_l[mZ_l :, :]
    chi_r = HZ_r[mZ_r :, :]
    HZ[mZ_l + mZ_r : mZ_l + mZ_r + len(g_l.V0), cl_data] = chi_l[:, : n_l]
    HZ[mZ_l + mZ_r : mZ_l + mZ_r + len(g_l.V0), cl_kappa] = chi_l[:, n_l :]
    HZ[mZ_l + mZ_r + len(g_l.V0) :, cr_data] = chi_r[:, : n_r]
    HZ[mZ_l + mZ_r + len(g_l.V0) :, cr_kappa] = chi_r[:, n_r :]
    # Π_l, Π_r adapter extensions on χ rows
    for v_idx, lab in enumerate(bridge.label_l):
        if lab >= 0:
            HZ[mZ_l + mZ_r + v_idx, c_adapter.start + lab] = 1
    for v_idx, lab in enumerate(bridge.label_r):
        if lab >= 0:
            HZ[mZ_l + mZ_r + len(g_l.V0) + v_idx, c_adapter.start + lab] = 1

    # H_X rows: data H_X^(l) extended, data H_X^(r) extended, G^(l)_aug, G^(r)_aug, new cycle
    HX[: mX_l, cl_data] = HX_l[: mX_l, : n_l]
    HX[: mX_l, cl_kappa] = HX_l[: mX_l, n_l :]
    HX[mX_l : mX_l + mX_r, cr_data] = HX_r[: mX_r, : n_r]
    HX[mX_l : mX_l + mX_r, cr_kappa] = HX_r[: mX_r, n_r :]
    HX[mX_l + mX_r : mX_l + mX_r + r_l, cl_kappa] = HX_l[mX_l :, n_l :]
    HX[mX_l + mX_r + r_l : mX_l + mX_r + r_l + r_r, cr_kappa] = HX_r[mX_r :, n_r :]
    # New cycle X-checks: [T_l | T_r | H_R]
    new_x_start = mX_l + mX_r + r_l + r_r
    HX[new_x_start :, cl_kappa] = bridge.T_l
    HX[new_x_start :, cr_kappa] = bridge.T_r
    HX[new_x_start :, c_adapter] = bridge.H_R

    return CSSCode(field(HX), field(HZ), is_subsystem_code=False)


def _stitch_intracode_joint_csscode_basis_z(g_l, g_r, bridge):
    """Symmetric dual of intra-code basis=X: χ rows in HZ; new cycle in HX."""
    assert g_l.code is g_r.code
    assert bridge.basis is Pauli.Z
    field = g_l.code.field
    g_l_aug, g_r_aug = bridge.g_l_aug, bridge.g_r_aug
    n = g_l.code.num_qudits
    k_l = g_l_aug.F.shape[0]
    k_r = g_r_aug.F.shape[0]
    w = bridge.width
    n_merged = n + k_l + k_r + w
    mX = g_l.code.matrix_x.shape[0]
    mZ = g_l.code.matrix_z.shape[0]
    r_l = g_l_aug.G.shape[0]
    r_r = g_r_aug.G.shape[0]

    c_data = slice(0, n)
    cl_kappa = slice(n, n + k_l)
    cr_kappa = slice(n + k_l, n + k_l + k_r)
    c_adapter = slice(n + k_l + k_r, n_merged)

    HX_l = np.asarray(g_l_aug.HX_merged).astype(np.int_)
    HX_r = np.asarray(g_r_aug.HX_merged).astype(np.int_)
    HZ_l = np.asarray(g_l_aug.HZ_merged).astype(np.int_)
    HZ_r = np.asarray(g_r_aug.HZ_merged).astype(np.int_)

    HZ = np.zeros((mZ + len(g_l.V0) + len(g_r.V0), n_merged), dtype=np.int_)
    HX = np.zeros((mX + r_l + r_r + (w - 1), n_merged), dtype=np.int_)

    # H_Z: shared data H_Z, then χ^(l), χ^(r) onto shared data
    HZ[: mZ, c_data] = HZ_l[: mZ, : n]
    chi_l = HZ_l[mZ :, :]
    chi_r = HZ_r[mZ :, :]
    HZ[mZ : mZ + len(g_l.V0), c_data] = chi_l[:, : n]
    HZ[mZ : mZ + len(g_l.V0), cl_kappa] = chi_l[:, n :]
    HZ[mZ + len(g_l.V0) :, c_data] = chi_r[:, : n]
    HZ[mZ + len(g_l.V0) :, cr_kappa] = chi_r[:, n :]
    for v_idx, lab in enumerate(bridge.label_l):
        if lab >= 0:
            HZ[mZ + v_idx, c_adapter.start + lab] = 1
    for v_idx, lab in enumerate(bridge.label_r):
        if lab >= 0:
            HZ[mZ + len(g_l.V0) + v_idx, c_adapter.start + lab] = 1

    # H_X: shared data H_X with κ extension on BOTH sides, G_aug rows, new cycle
    HX[: mX, c_data] = HX_l[: mX, : n]
    HX[: mX, cl_kappa] = HX_l[: mX, n :]
    HX[: mX, cr_kappa] = HX_r[: mX, n :]
    HX[mX : mX + r_l, cl_kappa] = HX_l[mX :, n :]
    HX[mX + r_l : mX + r_l + r_r, cr_kappa] = HX_r[mX :, n :]
    new_x_start = mX + r_l + r_r
    HX[new_x_start :, cl_kappa] = bridge.T_l
    HX[new_x_start :, cr_kappa] = bridge.T_r
    HX[new_x_start :, c_adapter] = bridge.H_R

    return CSSCode(field(HX), field(HZ), is_subsystem_code=False)


def build_joint_ppm_circuit(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
    *,
    rounds: int,
    noise_model=None,
) -> tuple[stim.Circuit, CSSCode]:
    """Joint-PPM circuit per spec §4 (universal adapter, no U_B in α*)."""
    joint_code = _stitch_to_joint_csscode(g_l, g_r, bridge)
    qubit_ids = QubitIDs.from_code(joint_code)
    intercode = g_l.code is not g_r.code

    g_l_aug, g_r_aug = bridge.g_l_aug, bridge.g_r_aug
    n_l = g_l.code.num_qudits
    n_r = g_r.code.num_qudits if intercode else 0
    k_l = g_l_aug.F.shape[0]
    k_r = g_r_aug.F.shape[0]
    w = bridge.width

    if intercode:
        data_ids = qubit_ids.data[: n_l + n_r]
        v0_combined = tuple(g_l.V0) + tuple(n_l + i for i in g_r.V0)
    else:
        data_ids = qubit_ids.data[: n_l]
        v0_combined = tuple(g_l.V0) + tuple(g_r.V0)
    kappa_ids = qubit_ids.data[n_l + n_r : n_l + n_r + k_l + k_r]
    bridge_ids = qubit_ids.data[n_l + n_r + k_l + k_r :]
    assert len(bridge_ids) == w

    circuit = get_qubit_coordinates(qubit_ids.data, qubit_ids.check)
    circuit += _surgery_state_prep(g_l, data_ids, kappa_ids, bridge_ids)
    qec_cycle, measurement_record, _ = _surgery_qec_cycle_joint(
        g_l, g_r, joint_code, num_rounds=rounds, qubit_ids=qubit_ids,
        intercode=intercode,
    )
    circuit += qec_cycle
    circuit += _surgery_detach_and_readout(
        g_l, data_ids=data_ids, kappa_ids=kappa_ids, bridge_ids=bridge_ids,
        measurement_record=measurement_record,
    )
    circuit += _surgery_final_detectors_joint(
        g_l, g_r, joint_code, qubit_ids,
        measurement_record=measurement_record,
        intercode=intercode,
    )

    # χ check IDs per spec §4: data H_X^(l) rows occupy first mX_l indices in
    # qubit_ids.checks_x, then m_X_r (inter-code), then χ^(l), then χ^(r).
    if bridge.basis is Pauli.X:
        check_ids = qubit_ids.checks_x
        m_l = g_l.code.matrix_x.shape[0]
        m_r = g_r.code.matrix_x.shape[0] if intercode else 0
    else:
        check_ids = qubit_ids.checks_z
        m_l = g_l.code.matrix_z.shape[0]
        m_r = g_r.code.matrix_z.shape[0] if intercode else 0
    n_V_l = len(g_l.V0)
    n_V_r = len(g_r.V0)
    chi_l_offset = m_l + m_r
    chi_r_offset = chi_l_offset + n_V_l
    chi_l_ids = tuple(check_ids[chi_l_offset : chi_l_offset + n_V_l])
    chi_r_ids = tuple(check_ids[chi_r_offset : chi_r_offset + n_V_r])
    chi_check_ids = chi_l_ids + chi_r_ids   # NO U_B / no adapter cycle-check ids

    circuit += _surgery_observable(
        g_l,
        chi_check_ids=chi_check_ids,
        data_ids=data_ids,
        v0_indices=v0_combined,
        num_rounds=rounds,
        measurement_record=measurement_record,
    )

    if noise_model is not None:
        circuit = noise_model.noisy_circuit(circuit)
    return circuit, joint_code


def _classify_reliable_round1_checks_joint(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    qubit_ids: QubitIDs,
    *,
    intercode: bool,
) -> tuple[int, ...]:
    """Joint-code variant: reliable checks across both gadgets + new cycle rows.

    basis=X (data |+⟩, κ + bridge |0⟩):
        H_X rows = [data H_X^(l), data H_X^(r), χ^(l), χ^(r)]
        H_Z rows = [data H_Z^(l) ext, data H_Z^(r) ext, G^(l)_aug, G^(r)_aug, new cycle-Z]

      Reliable X: data H_X rows of both gadgets.
      Reliable Z: G_aug rows + new cycle-Z rows (all act on κ ∪ bridge, all |0⟩).

    basis=Z is the X↔Z dual.
    """
    m_X_l = g_l.code.matrix_x.shape[0]
    m_X_r = g_r.code.matrix_x.shape[0] if intercode else 0
    m_Z_l = g_l.code.matrix_z.shape[0]
    m_Z_r = g_r.code.matrix_z.shape[0] if intercode else 0
    if g_l.basis is Pauli.X:
        reliable_x = qubit_ids.checks_x[: m_X_l + m_X_r]   # data H_X^(l/r)
        reliable_z = qubit_ids.checks_z[m_Z_l + m_Z_r :]   # G_aug + new cycle-Z
    else:
        reliable_x = qubit_ids.checks_x[m_X_l + m_X_r :]   # G_aug + new cycle-X
        reliable_z = qubit_ids.checks_z[: m_Z_l + m_Z_r]   # data H_Z^(l/r)
    return tuple(reliable_x) + tuple(reliable_z)


def _surgery_qec_cycle_joint(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    joint_code: CSSCode,
    num_rounds: int,
    qubit_ids: QubitIDs,
    *,
    intercode: bool,
) -> tuple[stim.Circuit, MeasurementRecord, DetectorRecord]:
    """Joint-code variant of _surgery_qec_cycle that classifies reliable checks
    across both gadgets + the bridge's new cycle-checks."""
    strategy = EdgeColoring()
    one_round, round_measurement_record = strategy.get_circuit(joint_code, qubit_ids)
    reliable = set(_classify_reliable_round1_checks_joint(
        g_l, g_r, qubit_ids, intercode=intercode,
    ))
    all_check_ids = qubit_ids.check

    circuit = stim.Circuit()
    measurement_record = MeasurementRecord()
    detector_record = DetectorRecord()

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


def _surgery_final_detectors_joint(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    joint_code: CSSCode,
    qubit_ids: QubitIDs,
    *,
    measurement_record: MeasurementRecord,
    intercode: bool,
) -> stim.Circuit:
    """Joint-code variant of _surgery_final_detectors.

    Emits detectors for the same reliable stabilizers as the round-1 classifier:
    basis=X: data H_X rows from both gadgets + G_aug + new cycle-Z rows.
    basis=Z: data H_Z rows from both gadgets + G_aug + new cycle-X rows.
    """
    m_X_l = g_l.code.matrix_x.shape[0]
    m_X_r = g_r.code.matrix_x.shape[0] if intercode else 0
    m_Z_l = g_l.code.matrix_z.shape[0]
    m_Z_r = g_r.code.matrix_z.shape[0] if intercode else 0
    HX = np.asarray(joint_code.matrix_x).astype(np.uint8)
    HZ = np.asarray(joint_code.matrix_z).astype(np.uint8)

    circuit = stim.Circuit()

    def _emit_detector(stab_row: np.ndarray, check_id: int, det_idx: int) -> None:
        supp = np.where(stab_row)[0]
        targets = [measurement_record.get_target_rec(qubit_ids.data[q]) for q in supp]
        targets.append(measurement_record.get_target_rec(check_id, -1))
        circuit.append("DETECTOR", targets, (0, 0, det_idx))

    if g_l.basis is Pauli.X:
        # data H_X rows from both gadgets
        for kk in range(m_X_l + m_X_r):
            _emit_detector(HX[kk], qubit_ids.checks_x[kk], kk)
        # G_aug rows + new cycle-Z rows: indices [m_Z_l + m_Z_r : HZ.shape[0])
        det_offset = m_X_l + m_X_r
        for offset, kk in enumerate(range(m_Z_l + m_Z_r, HZ.shape[0])):
            _emit_detector(HZ[kk], qubit_ids.checks_z[kk], det_offset + offset)
    else:
        # data H_Z rows from both gadgets
        for kk in range(m_Z_l + m_Z_r):
            _emit_detector(HZ[kk], qubit_ids.checks_z[kk], kk)
        det_offset = m_Z_l + m_Z_r
        for offset, kk in enumerate(range(m_X_l + m_X_r, HX.shape[0])):
            _emit_detector(HX[kk], qubit_ids.checks_x[kk], det_offset + offset)

    return circuit


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


def _surgery_final_detectors(
    gadget: GadgetLayout,
    merged_code: CSSCode,
    qubit_ids: QubitIDs,
    *,
    measurement_record: MeasurementRecord,
) -> stim.Circuit:
    """Emit DETECTORs for reliable stabs inferable from final readouts.

    For basis=X: data H_X (from Mx data) + G (from Mz κ).
    For basis=Z: data H_Z (from Mz data) + G (from Mx κ).
    Each DETECTOR XORs ⊕(final M-record on stab support) ⊕ last-round syndrome.
    """
    m_X = gadget.code.matrix_x.shape[0]
    m_Z = gadget.code.matrix_z.shape[0]
    HX = np.asarray(merged_code.matrix_x).astype(np.uint8)
    HZ = np.asarray(merged_code.matrix_z).astype(np.uint8)

    circuit = stim.Circuit()

    def _emit_detector(stab_row: np.ndarray, check_id: int, det_idx: int) -> None:
        supp = np.where(stab_row)[0]
        targets = [measurement_record.get_target_rec(qubit_ids.data[q]) for q in supp]
        targets.append(measurement_record.get_target_rec(check_id, -1))
        circuit.append("DETECTOR", targets, (0, 0, det_idx))

    if gadget.basis is Pauli.X:
        # data H_X rows (X-checks indices [:m_X])
        for kk in range(m_X):
            _emit_detector(HX[kk], qubit_ids.checks_x[kk], kk)
        # G rows (Z-checks indices [m_Z:])
        for offset, kk in enumerate(range(m_Z, HZ.shape[0])):
            _emit_detector(HZ[kk], qubit_ids.checks_z[kk], m_X + offset)
    else:  # Pauli.Z (symmetric: chi in HZ, G in HX)
        for kk in range(m_Z):
            _emit_detector(HZ[kk], qubit_ids.checks_z[kk], kk)
        for offset, kk in enumerate(range(m_X, HX.shape[0])):
            _emit_detector(HX[kk], qubit_ids.checks_x[kk], m_Z + offset)

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
