"""Stim surgery circuit construction.

build_single_ppm_circuit  — single-PPM measurement (gadget alone)
build_joint_ppm_circuit   — two-PPM joint measurement (gadget + gadget + bridge)
"""

from __future__ import annotations

import numpy as np
import stim

from qldpc.codes.common import CSSCode
from qldpc.circuits.memory.memory import get_memory_experiment
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

    Qubit register: [ data | g1-kappa | g2-kappa | bridge ]
    HX rows: g1 rows (all), g2 chi-rows (non-data), bridge U_B rows.
    HZ rows: g1 rows (all, with g2-kappa extensions spliced in), g2 gauge-fix rows.

    Per Cross §3.6 + math.md §2.5–2.6.
    """
    import galois

    data_code = g1.code
    field = data_code.field

    n_data = data_code.num_qubits
    n_anc_1 = len(g1.C0)       # kappa qubits for g1
    n_anc_2 = len(g2.C0)       # kappa qubits for g2
    n_bridge = bridge.width
    n_merged = n_data + n_anc_1 + n_anc_2 + n_bridge

    mX = int(data_code.matrix_x.shape[0])   # data X-check rows
    mZ = int(data_code.matrix_z.shape[0])   # data Z-check rows

    HX1 = np.asarray(g1.HX_merged).astype(np.int_)  # shape (mX + nV1, n_data + n_anc_1)
    HZ1 = np.asarray(g1.HZ_merged).astype(np.int_)  # shape (mZ + r1, n_data + n_anc_1)
    HX2 = np.asarray(g2.HX_merged).astype(np.int_)  # shape (mX + nV2, n_data + n_anc_2)
    HZ2 = np.asarray(g2.HZ_merged).astype(np.int_)  # shape (mZ + r2, n_data + n_anc_2)

    def _pad(matrix: np.ndarray, *, anc_offset: int) -> np.ndarray:
        """Pad a (rows, n_data + n_anc) matrix into (rows, n_merged)."""
        n_anc = matrix.shape[1] - n_data
        out = np.zeros((matrix.shape[0], n_merged), dtype=np.int_)
        out[:, :n_data] = matrix[:, :n_data]
        out[:, anc_offset : anc_offset + n_anc] = matrix[:, n_data:]
        return out

    anc_off_1 = n_data
    anc_off_2 = n_data + n_anc_1

    HX1_pad = _pad(HX1, anc_offset=anc_off_1)
    HX2_pad = _pad(HX2, anc_offset=anc_off_2)
    HZ1_pad = _pad(HZ1, anc_offset=anc_off_1)
    HZ2_pad = _pad(HZ2, anc_offset=anc_off_2)

    # Extend chi row 0 of gadget1 with X on bridge qubit 0.
    # Chi rows in HX1 start at row mX; chi row 0 is row mX.
    bridge_col_start = n_data + n_anc_1 + n_anc_2
    HX1_pad[mX + 0, bridge_col_start + 0] = 1
    # Extend chi row 0 of gadget2 with X on bridge qubit w-1.
    HX2_pad[mX + 0, bridge_col_start + n_bridge - 1] = 1

    # Drop g2's data X-check rows (duplicates of g1's).
    HX2_pad_nondata = HX2_pad[mX:]   # keep only chi rows from g2

    # Splice g2-kappa extensions into g1's data Z rows.
    # For each k, g2.C0[k] is the data Z-check row j that gets a 1 on kappa2[k].
    # kappa2 columns start at anc_off_2 (they are the first n_anc_2 cols after g1's ancillas).
    for k, j in enumerate(g2.C0):
        HZ1_pad[int(j), anc_off_2 + k] = 1

    # Drop g2's data Z rows; keep only gauge-fix rows.
    HZ2_pad_gaugefix = HZ2_pad[mZ:]  # shape (r2, n_merged)

    # Bridge U_B X-stabilizers on bridge columns only.
    u_b = np.asarray(bridge.U_B).astype(np.int_)   # shape (w-1, w)
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

    Handles both intra-code (g1.code is g2.code) and inter-code paths via the
    `bridge.intercode` flag.
    """
    joint_code = _stitch_to_joint_csscode(g1, g2, bridge)
    circuit = get_memory_experiment(
        joint_code, basis=Pauli.X, num_rounds=rounds, noise_model=noise_model,
    )
    return circuit, joint_code
