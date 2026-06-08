"""Stim surgery circuit construction.

build_single_ppm_circuit  — single-PPM measurement (gadget alone)
build_joint_ppm_circuit   — two-PPM joint measurement (gadget + gadget + bridge)
"""

from __future__ import annotations

import numpy as np
import stim

from qldpc.codes.common import CSSCode
from qldpc.circuits.bookkeeping import QubitIDs
from qldpc.circuits.memory.memory import get_memory_experiment
from qldpc.objects import Pauli, PauliXZ

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
    """Stim circuit for single-PPM measurement using `gadget`.

    Builds the merged CSS code (data + κ ancillas) and delegates to the
    existing memory experiment infrastructure.
    """
    merged = _gadget_merged_csscode(gadget)
    return get_memory_experiment(merged, basis=Pauli.X, num_rounds=rounds, noise_model=noise_model)


def _stitch_to_joint_csscode(
    g1: GadgetLayout,
    g2: GadgetLayout,
    bridge: Bridge,
) -> CSSCode:
    """Assemble the joint CSS code for two-PPM surgery.

    Intra-code (g1.code is g2.code) layout:
        [ data | g1-kappa | g2-kappa | bridge ]
        HX rows: g1 rows (all), g2 chi-rows (non-data), bridge U_B rows.
        HZ rows: g1 rows (all, with g2-kappa extensions spliced in),
                 g2 gauge-fix rows.

    Inter-code (g1.code is not g2.code) layout:
        [ data_1 | data_2 | g1-kappa | g2-kappa | bridge ]
        HX rows: g1 rows (all), g2 rows (all), bridge U_B rows.
        HZ rows: g1 rows (all), g2 rows (all).

    Per Cross §3.6 + math.md §2.5–2.6.
    """
    intercode = g1.code is not g2.code
    field = g1.code.field

    n_data_1 = g1.code.num_qubits
    n_data_2 = g2.code.num_qubits if intercode else 0
    n_anc_1 = len(g1.C0)       # kappa qubits for g1
    n_anc_2 = len(g2.C0)       # kappa qubits for g2
    n_bridge = bridge.width
    n_merged = n_data_1 + n_data_2 + n_anc_1 + n_anc_2 + n_bridge

    mX1 = int(g1.code.matrix_x.shape[0])
    mZ1 = int(g1.code.matrix_z.shape[0])
    mX2 = int(g2.code.matrix_x.shape[0])
    mZ2 = int(g2.code.matrix_z.shape[0])

    HX1 = np.asarray(g1.HX_merged).astype(np.int_)
    HZ1 = np.asarray(g1.HZ_merged).astype(np.int_)
    HX2 = np.asarray(g2.HX_merged).astype(np.int_)
    HZ2 = np.asarray(g2.HZ_merged).astype(np.int_)

    if intercode:
        # Inter-code: data_1 and data_2 are disjoint.
        # Column layout: [data_1 | data_2 | kappa_1 | kappa_2 | bridge]
        data_off_1 = 0
        data_off_2 = n_data_1
        anc_off_1 = n_data_1 + n_data_2
        anc_off_2 = anc_off_1 + n_anc_1
        bridge_col_start = anc_off_2 + n_anc_2

        def _pad_g1(matrix: np.ndarray) -> np.ndarray:
            out = np.zeros((matrix.shape[0], n_merged), dtype=np.int_)
            out[:, data_off_1 : data_off_1 + n_data_1] = matrix[:, :n_data_1]
            out[:, anc_off_1 : anc_off_1 + n_anc_1] = matrix[:, n_data_1:]
            return out

        def _pad_g2(matrix: np.ndarray) -> np.ndarray:
            out = np.zeros((matrix.shape[0], n_merged), dtype=np.int_)
            out[:, data_off_2 : data_off_2 + n_data_2] = matrix[:, :n_data_2]
            out[:, anc_off_2 : anc_off_2 + n_anc_2] = matrix[:, n_data_2:]
            return out

        HX1_pad = _pad_g1(HX1)
        HZ1_pad = _pad_g1(HZ1)
        HX2_pad = _pad_g2(HX2)
        HZ2_pad = _pad_g2(HZ2)

        # Bridge chi-endpoint extensions: chi row 0 of each gadget gets an
        # X on its bridge endpoint.
        HX1_pad[mX1 + 0, bridge_col_start + 0] = 1
        HX2_pad[mX2 + 0, bridge_col_start + n_bridge - 1] = 1

        # Inter-code: both gadgets contribute all rows (no duplicates).
        u_b = np.asarray(bridge.U_B).astype(np.int_)
        n_u_b = u_b.shape[0]
        u_b_pad = np.zeros((n_u_b, n_merged), dtype=np.int_)
        if n_u_b > 0:
            u_b_pad[:, bridge_col_start : bridge_col_start + n_bridge] = u_b

        HX_joint = field(np.vstack([HX1_pad, HX2_pad, u_b_pad]))
        HZ_joint = field(np.vstack([HZ1_pad, HZ2_pad]))

        return CSSCode(HX_joint, HZ_joint, is_subsystem_code=False)

    # Intra-code path (shared data qubits)
    n_data = n_data_1
    mX, mZ = mX1, mZ1

    def _pad(matrix: np.ndarray, *, anc_offset: int) -> np.ndarray:
        n_anc = matrix.shape[1] - n_data
        out = np.zeros((matrix.shape[0], n_merged), dtype=np.int_)
        out[:, :n_data] = matrix[:, :n_data]
        out[:, anc_offset : anc_offset + n_anc] = matrix[:, n_data:]
        return out

    anc_off_1 = n_data
    anc_off_2 = n_data + n_anc_1
    bridge_col_start = n_data + n_anc_1 + n_anc_2

    HX1_pad = _pad(HX1, anc_offset=anc_off_1)
    HX2_pad = _pad(HX2, anc_offset=anc_off_2)
    HZ1_pad = _pad(HZ1, anc_offset=anc_off_1)
    HZ2_pad = _pad(HZ2, anc_offset=anc_off_2)

    HX1_pad[mX + 0, bridge_col_start + 0] = 1
    HX2_pad[mX + 0, bridge_col_start + n_bridge - 1] = 1

    HX2_pad_nondata = HX2_pad[mX:]

    for k, j in enumerate(g2.C0):
        HZ1_pad[int(j), anc_off_2 + k] = 1

    HZ2_pad_gaugefix = HZ2_pad[mZ:]

    u_b = np.asarray(bridge.U_B).astype(np.int_)
    n_u_b = u_b.shape[0]
    u_b_pad = np.zeros((n_u_b, n_merged), dtype=np.int_)
    if n_u_b > 0:
        u_b_pad[:, bridge_col_start : bridge_col_start + n_bridge] = u_b

    HX_joint = field(np.vstack([HX1_pad, HX2_pad_nondata, u_b_pad]))
    HZ_joint = field(np.vstack([HZ1_pad, HZ2_pad_gaugefix]))

    return CSSCode(HX_joint, HZ_joint, is_subsystem_code=False)


def build_joint_ppm_circuit(
    g1: GadgetLayout,
    g2: GadgetLayout,
    bridge: Bridge,
    *,
    rounds: int,
    noise_model=None,
) -> tuple[stim.Circuit, CSSCode]:
    """Stim circuit + merged joint CSS code for two-PPM joint measurement.

    Intra-code path (g1.code is g2.code). Inter-code support is added in Task 16.
    """
    joint_code = _stitch_to_joint_csscode(g1, g2, bridge)
    circuit = get_memory_experiment(
        joint_code, basis=Pauli.X, num_rounds=rounds, noise_model=noise_model,
    )
    return circuit, joint_code


def _classify_reliable_round1_checks(
    gadget: GadgetLayout,
    merged_code: CSSCode,
    qubit_ids: QubitIDs,
) -> tuple[int, ...]:
    """Return the subset of merged-code check ancillas whose round-1 syndrome
    is reliable (= +1) given the surgery init state.

    For basis=Pauli.X (data in |+⟩, κ in |0⟩):
        reliable = data H_X rows (X-type, data |+⟩ → +1) +
                   gauge-fix G rows (Z-type, κ |0⟩ → +1)
        unreliable = χ rows (X on κ is random) + data H_Z rows (Z on data |+⟩ random)
    For basis=Pauli.Z (data in |0⟩, κ in |+⟩): swap X↔Z in the above.
    """
    m_X = gadget.code.matrix_x.shape[0]
    m_Z = gadget.code.matrix_z.shape[0]

    if gadget.basis is Pauli.X:
        # X-checks: first m_X are data H_X (reliable), next n_V are χ (unreliable)
        reliable_x = qubit_ids.checks_x[:m_X]
        # Z-checks: first m_Z are data H_Z (unreliable), last r are G (reliable)
        reliable_z = qubit_ids.checks_z[m_Z:]
    else:  # Pauli.Z (basis swap)
        # X-checks: first m_X are data H_X (unreliable), last r are G (reliable)
        reliable_x = qubit_ids.checks_x[m_X:]
        # Z-checks: first m_Z are data H_Z (reliable), next n_V are χ (unreliable)
        reliable_z = qubit_ids.checks_z[:m_Z]

    return tuple(reliable_x) + tuple(reliable_z)


def _surgery_state_prep(
    gadget: GadgetLayout,
    data_ids: tuple[int, ...],
    kappa_ids: tuple[int, ...],
    bridge_ids: tuple[int, ...] = (),
) -> stim.Circuit:
    """Cain step 1: init data in logical |+⟩ (basis=X) or |0⟩ (basis=Z),
    init κ ancillas in |0⟩ (basis=X) or |+⟩ (basis=Z). Bridge follows κ.
    """
    circuit = stim.Circuit()
    if gadget.basis is Pauli.X:
        circuit.append("RX", list(data_ids))
        circuit.append("R", list(kappa_ids))
        if bridge_ids:
            circuit.append("R", list(bridge_ids))
    else:  # Pauli.Z
        circuit.append("R", list(data_ids))
        circuit.append("RX", list(kappa_ids))
        if bridge_ids:
            circuit.append("RX", list(bridge_ids))
    return circuit
