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

from qldpc.objects import Pauli, PauliXZ


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


def column_slices_for_bridge(g_l, g_r, bridge) -> dict[str, slice]:
    """Partition merged-code columns into Q_l | Q_r | k_l | k_r | A.

    Mirrors main.tex §4.2 qubit ordering. Inter-code: n_r = g_r.code.num_qudits;
    intra-code (g_l.code is g_r.code): n_r = 0 (shared data block) — caller is
    responsible for noting that Q_r aliases Q_l.
    """
    n_l = g_l.code.num_qudits
    n_r = g_r.code.num_qudits if g_l.code is not g_r.code else 0
    k_l = bridge.g_l_aug.incidence.shape[0]
    k_r = bridge.g_r_aug.incidence.shape[0]
    w = bridge.width
    return {
        "Q_l": slice(0, n_l),
        "Q_r": slice(n_l, n_l + n_r),
        "k_l": slice(n_l + n_r, n_l + n_r + k_l),
        "k_r": slice(n_l + n_r + k_l, n_l + n_r + k_l + k_r),
        "A": slice(n_l + n_r + k_l + k_r, n_l + n_r + k_l + k_r + w),
    }


def _block_data(g, *, basis_block: PauliXZ, side: str, slices: dict[str, slice], N: int) -> np.ndarray:
    """Build the data-stabilizer block (H_X or H_Z) for one side.

    Matches main.tex §4.2 row blocks 1 and 2 of $\\tilde H^{\\mathrm{joint,pre}}$:
    the side's original H_basis sits on Q_s; when basis_block is dual to the
    side's gadget basis, the rows whose index is in C_0^s extend into κ^s via
    f_basis^s = π_{C_0^s}^T (a single 1 per κ^s column at row C_0^s[k]).

    Args:
      g: GadgetLayout for the side.
      basis_block: Pauli.X to build a row of H_X, Pauli.Z to build a row of H_Z.
      side: 'l' or 'r'.
      slices: column partition from column_slices_for_bridge.
      N: total merged-code column count.

    Returns:
      Matrix of shape (m_basis, N). The data columns Q_s hold the original
      H_basis^s; the κ^s columns hold f_basis = π_{C_0^s}^T iff basis_block is
      the basis DUAL to g.basis (single-gadget extension). All other columns 0.
    """
    H = np.asarray(g.code.matrix_x if basis_block is Pauli.X else g.code.matrix_z).astype(np.uint8)
    m = H.shape[0]
    block = np.zeros((m, N), dtype=np.uint8)
    block[:, slices[f"Q_{side}"]] = H
    # f extension lives on κ^s iff this basis is dual to the side's gadget basis.
    if basis_block is not g.basis:
        kappa_slice = slices[f"k_{side}"]
        for k, j in enumerate(g.data_checks):
            if j < 0:
                continue  # sentinel for extra-κ rows from build_gadget_augmented
            block[j, kappa_slice.start + k] = 1
    return block
