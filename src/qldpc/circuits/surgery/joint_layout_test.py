"""Tests for joint_layout.py — block-by-block joint PPM construction per main.tex §4."""

from __future__ import annotations

import numpy as np

from qldpc.circuits.surgery.joint_layout import JointPPMLayout
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
