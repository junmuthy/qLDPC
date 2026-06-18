"""Block-by-block joint PPM construction following docs/superpowers/docs/main.tex §3-§4.

Mixed-basis (e.g. Z̄_l ⊗ X̄_r) joint PPM produces a subsystem code with three
stabilizer matrices (H_X, H_Z, H_Y). Each row is built from a block in the
LaTeX matrices and tagged with its provenance so the downstream circuit
builder can emit obs0 = ⊕ m(χ_l) ⊕ ⊕ m(χ_r) ⊕ ⊕ m(y_q) per Lemma 2
(docs/superpowers/specs/2026-06-15-mixed-basis-joint-ppm-design.md §9).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from qldpc.objects import PauliXZ


@dataclasses.dataclass(frozen=True, eq=False)
class JointPPMLayout:
    """Joint PPM merged code with explicit row provenance.

    H_Y is empty (shape (0, 2N)) for same-basis joint PPM. The provenance
    dicts use side labels ``'l'`` / ``'r'`` and hold row indices into the
    matrix that contains those rows (e.g. rows_chi['l'] are indices into
    whichever of H_X or H_Z carries the left side's χ rows, determined by
    basis_l).
    """

    H_X: np.ndarray
    H_Z: np.ndarray
    H_Y: np.ndarray

    rows_data_x: dict[str, tuple[int, ...]]
    rows_data_z: dict[str, tuple[int, ...]]
    rows_chi: dict[str, tuple[int, ...]]
    rows_gauge: dict[str, tuple[int, ...]]
    rows_cycle: dict[str, tuple[int, ...]]
    rows_y: tuple[int, ...]

    basis_l: PauliXZ
    basis_r: PauliXZ

    column_slices: dict[str, slice]
