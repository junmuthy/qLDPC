"""Tests for joint_layout.py — block-by-block joint PPM construction per main.tex §4."""

from __future__ import annotations

import numpy as np

from qldpc import codes
from qldpc.circuits.surgery.joint_layout import JointPPMLayout, column_slices_for_bridge
from qldpc.objects import Pauli


def test_layout_dataclass_construction() -> None:
    """JointPPMLayout holds three stabilizer matrices + provenance dicts."""
    H_X = np.zeros((2, 10), dtype=np.uint8)
    H_Z = np.zeros((3, 10), dtype=np.uint8)
    H_Y = np.zeros((1, 20), dtype=np.uint8)
    layout = JointPPMLayout(
        H_X=H_X,
        H_Z=H_Z,
        H_Y=H_Y,
        rows_data_x={"l": (0,), "r": (1,)},
        rows_data_z={"l": (0,), "r": (1,)},
        rows_chi={"l": (), "r": ()},
        rows_gauge={"l": (), "r": ()},
        rows_cycle={"l": (), "r": (2,)},
        rows_y=(0,),
        basis_l=Pauli.Z,
        basis_r=Pauli.X,
        column_slices={
            "Q_l": slice(0, 3),
            "Q_r": slice(3, 6),
            "k_l": slice(6, 7),
            "k_r": slice(7, 8),
            "A": slice(8, 10),
        },
    )
    assert layout.H_X.shape == (2, 10)
    assert layout.H_Z.shape == (3, 10)
    assert layout.H_Y.shape == (1, 20)
    assert layout.basis_l is Pauli.Z
    assert layout.column_slices["Q_l"] == slice(0, 3)


def test_column_slices_for_steane_pair() -> None:
    """Column slices partition (Q_l | Q_r | k_l | k_r | A) for an inter-code pair."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget

    code_l = codes.SteaneCode()
    code_r = codes.SteaneCode()
    x = np.asarray(code_l.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code_r.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code_l, x, basis=Pauli.X)
    g_r = build_gadget(code_r, z, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)

    slices = column_slices_for_bridge(g_l, g_r, bridge)
    n_l = g_l.code.num_qudits
    n_r = g_r.code.num_qudits
    k_l = bridge.g_l_aug.incidence.shape[0]
    k_r = bridge.g_r_aug.incidence.shape[0]
    w = bridge.width

    assert slices["Q_l"] == slice(0, n_l)
    assert slices["Q_r"] == slice(n_l, n_l + n_r)
    assert slices["k_l"] == slice(n_l + n_r, n_l + n_r + k_l)
    assert slices["k_r"] == slice(n_l + n_r + k_l, n_l + n_r + k_l + k_r)
    assert slices["A"] == slice(n_l + n_r + k_l + k_r, n_l + n_r + k_l + k_r + w)
