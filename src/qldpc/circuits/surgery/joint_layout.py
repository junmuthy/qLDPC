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


def _block_chi(g_aug, *, side: str, slices: dict[str, slice], N: int,
               labels: tuple[int, ...]) -> np.ndarray:
    """Build the χ-row block — main.tex §4.2 row blocks 3 and 4.

    Each row i corresponds to a V_0^s vertex v_i. The row carries:
      * Q_s columns: π_{V_0^s} (single 1 at qubit v_i)
      * κ^s columns: H_{X/Z}'^{s,aug} (=g_aug.incidence^T) row i
      * Adapter columns: 1 at column labels[i] if labels[i] >= 0 (port row), else 0.

    The basis attribution (whether this block sits in H_X or H_Z) is the
    caller's responsibility — for basis_l=Z the left χ block belongs in H_Z,
    for basis_r=X the right χ block belongs in H_X.
    """
    n_V0 = len(g_aug.support)
    block = np.zeros((n_V0, N), dtype=np.uint8)
    # π_{V_0^s} on Q_s
    for i, v in enumerate(g_aug.support):
        block[i, slices[f"Q_{side}"].start + v] = 1
    # H'^{s,aug} = incidence^T on κ^s
    block[:, slices[f"k_{side}"]] = np.asarray(g_aug.incidence).astype(np.uint8).T
    # π_{P_s}^T P_{σ_s} on adapter via labels
    for i, lab in enumerate(labels):
        if lab >= 0:
            block[i, slices["A"].start + lab] = 1
    return block


def _block_gauge(g_aug, *, side: str, slices: dict[str, slice], N: int) -> np.ndarray:
    """Gauge block H_{X/Z}'^{s,aug} — main.tex §4.2 row block 3 (left) or
    analogous right row block. Supports only κ^s, zero elsewhere."""
    G = np.asarray(g_aug.gauge).astype(np.uint8)
    r = G.shape[0]
    block = np.zeros((r, N), dtype=np.uint8)
    block[:, slices[f"k_{side}"]] = G
    return block


def _block_cycle(T_s: np.ndarray, H_R: np.ndarray, *, side: str,
                 slices: dict[str, slice], N: int) -> np.ndarray:
    """Cycle row block T_s on κ^s + H_R on adapter — main.tex §4.2 last row block.

    Both T_s and H_R have w-1 rows; the returned block has the same row count.
    """
    n_rows = T_s.shape[0]
    block = np.zeros((n_rows, N), dtype=np.uint8)
    block[:, slices[f"k_{side}"]] = np.asarray(T_s).astype(np.uint8)
    block[:, slices["A"]] = np.asarray(H_R).astype(np.uint8)
    return block
