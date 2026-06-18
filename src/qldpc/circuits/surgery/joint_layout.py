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


def build_pre_merge_layout(g_l, g_r, bridge) -> JointPPMLayout:
    """Assemble the pre-merge (before cross-merge) joint check matrices per main.tex §4.2.

    Row order in H_X:
      1. data H_X^l (block 1)
      2. data H_X^r (block 2)
      3. gauge H_X'^{l,aug} if basis_l=Z (gauge sits in H_X when basis is dual)
      4. χ rows from the side whose basis is X (so χ rows live in H_X)
      5. cycle row from the side whose basis is Z (cycle lives in dual matrix)

    Mirror order in H_Z. The exact placement is basis-aware. ``rows_chi[side]``
    holds row indices into whichever of H_X/H_Z carries that side's χ rows.

    Note: this function assumes inter-code (g_l.code is not g_r.code). Intra-code
    is left for a separate task; the dispatcher in build_joint_layout will guard
    against it.
    """
    assert g_l.code is not g_r.code, "intra-code mixed-basis not yet implemented"
    slices = column_slices_for_bridge(g_l, g_r, bridge)
    N = slices["A"].stop

    # Per-side blocks
    data_x_l = _block_data(g_l, basis_block=Pauli.X, side="l", slices=slices, N=N)
    data_z_l = _block_data(g_l, basis_block=Pauli.Z, side="l", slices=slices, N=N)
    data_x_r = _block_data(g_r, basis_block=Pauli.X, side="r", slices=slices, N=N)
    data_z_r = _block_data(g_r, basis_block=Pauli.Z, side="r", slices=slices, N=N)
    chi_l = _block_chi(bridge.g_l_aug, side="l", slices=slices, N=N, labels=bridge.label_l)
    chi_r = _block_chi(bridge.g_r_aug, side="r", slices=slices, N=N, labels=bridge.label_r)
    gauge_l = _block_gauge(bridge.g_l_aug, side="l", slices=slices, N=N)
    gauge_r = _block_gauge(bridge.g_r_aug, side="r", slices=slices, N=N)
    cycle_l = _block_cycle(bridge.T_l, bridge.H_R, side="l", slices=slices, N=N)
    cycle_r = _block_cycle(bridge.T_r, bridge.H_R, side="r", slices=slices, N=N)

    # χ_s sits in H_basis_s; gauge_s and cycle_s sit in H_dual(basis_s).
    H_X_blocks: list[tuple[np.ndarray, str, str]] = [
        (data_x_l, "data_x", "l"),
        (data_x_r, "data_x", "r"),
    ]
    H_Z_blocks: list[tuple[np.ndarray, str, str]] = [
        (data_z_l, "data_z", "l"),
        (data_z_r, "data_z", "r"),
    ]
    for side_label, basis, chi_block, gauge_block, cycle_block in (
        ("l", bridge.basis_l, chi_l, gauge_l, cycle_l),
        ("r", bridge.basis_r, chi_r, gauge_r, cycle_r),
    ):
        if basis is Pauli.X:
            H_X_blocks.append((chi_block, "chi", side_label))
            H_Z_blocks.append((gauge_block, "gauge", side_label))
            H_Z_blocks.append((cycle_block, "cycle", side_label))
        else:
            H_Z_blocks.append((chi_block, "chi", side_label))
            H_X_blocks.append((gauge_block, "gauge", side_label))
            H_X_blocks.append((cycle_block, "cycle", side_label))

    def _stack_with_provenance(blocks):
        rows: list[np.ndarray] = []
        provenance: dict[tuple[str, str], list[int]] = {}
        for block, kind, side_label in blocks:
            start = len(rows)
            for r in np.asarray(block).astype(np.uint8):
                rows.append(r)
            end = len(rows)
            provenance.setdefault((kind, side_label), []).extend(range(start, end))
        if rows:
            mat = np.stack(rows).astype(np.uint8)
        else:
            mat = np.zeros((0, N), dtype=np.uint8)
        return mat, provenance

    H_X, prov_X = _stack_with_provenance(H_X_blocks)
    H_Z, prov_Z = _stack_with_provenance(H_Z_blocks)

    def _gather(prov, kind):
        out: dict[str, tuple[int, ...]] = {"l": (), "r": ()}
        for (k, side_label), idx in prov.items():
            if k == kind:
                out[side_label] = tuple(idx)
        return out

    H_Y = np.zeros((0, 2 * N), dtype=np.uint8)
    return JointPPMLayout(
        H_X=H_X,
        H_Z=H_Z,
        H_Y=H_Y,
        rows_data_x=_gather(prov_X, "data_x"),
        rows_data_z=_gather(prov_Z, "data_z"),
        rows_chi={
            "l": _gather(prov_X if bridge.basis_l is Pauli.X else prov_Z, "chi")["l"],
            "r": _gather(prov_X if bridge.basis_r is Pauli.X else prov_Z, "chi")["r"],
        },
        rows_gauge={
            "l": _gather(prov_Z if bridge.basis_l is Pauli.X else prov_X, "gauge")["l"],
            "r": _gather(prov_Z if bridge.basis_r is Pauli.X else prov_X, "gauge")["r"],
        },
        rows_cycle={
            "l": _gather(prov_Z if bridge.basis_l is Pauli.X else prov_X, "cycle")["l"],
            "r": _gather(prov_Z if bridge.basis_r is Pauli.X else prov_X, "cycle")["r"],
        },
        rows_y=(),
        basis_l=bridge.basis_l,
        basis_r=bridge.basis_r,
        column_slices=slices,
    )
