"""Webster–Smith–Cohen (arXiv:2511.15989 §II.B.2) mixed-basis merge.

Pure GF(2) row arithmetic on assembled (H_X, H_Z) matrices. Pair-merges
X-checks on each shared bridge qubit, pair-merges Z-checks, then
cross-merges the surviving X / Z leftover pair into a single Y-type
stabilizer row. The merged code is non-CSS but supports joint
Pauli-product measurement of operators of different Pauli type
(e.g. Z̄_l ⊗ X̄_r).
"""

from __future__ import annotations

import numpy as np


def _merge_at_qubit(
    H_X: np.ndarray,
    H_Z: np.ndarray,
    q: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int | None, int | None]:
    """In-place GF(2) row ops for one bridge qubit.

    Returns
    -------
    (H_X_out, H_Z_out, Y_row, leftover_x_idx, leftover_z_idx)
        H_X_out / H_Z_out: matrices after pair-merge (and row deletion if
            cross-merged).
        Y_row: symplectic row (length 2*n) if both X and Z conflicts present;
            else None.
        leftover_x_idx / leftover_z_idx: index of the leftover (pivot) row in
            the PRE-DELETION ``H_X`` / ``H_Z``. Always reported when a
            conflict was present at qubit ``q`` — even when ``Y_row`` is
            emitted and that row is then deleted. The caller needs the index
            in the pre-deletion frame to maintain its own bookkeeping of
            surviving leftover indices across successive ``np.delete`` calls.
    """
    n = H_X.shape[1]
    x_rows = np.flatnonzero(H_X[:, q])
    leftover_x: int | None = None
    if x_rows.size >= 1:
        pivot = int(x_rows[0])
        for r in x_rows[1:]:
            H_X[int(r)] = (H_X[int(r)] + H_X[pivot]) % 2
        leftover_x = pivot

    z_rows = np.flatnonzero(H_Z[:, q])
    leftover_z: int | None = None
    if z_rows.size >= 1:
        pivot = int(z_rows[0])
        for r in z_rows[1:]:
            H_Z[int(r)] = (H_Z[int(r)] + H_Z[pivot]) % 2
        leftover_z = pivot

    Y_row: np.ndarray | None = None
    if leftover_x is not None and leftover_z is not None:
        Y_row = np.zeros(2 * n, dtype=np.uint8)
        Y_row[:n] = H_X[leftover_x]
        Y_row[n:] = H_Z[leftover_z]
        # Remove both leftover rows; their product becomes the Y stabilizer.
        # NOTE: ``leftover_x`` / ``leftover_z`` are intentionally NOT nulled —
        # the caller must know which row indices were just deleted so it can
        # shift its leftover-row bookkeeping accordingly.
        H_X = np.delete(H_X, leftover_x, axis=0)
        H_Z = np.delete(H_Z, leftover_z, axis=0)

    return H_X, H_Z, Y_row, leftover_x, leftover_z


def apply_mixed_basis_merge(
    H_X: np.ndarray,
    H_Z: np.ndarray,
    merge_qubits: tuple[int, ...],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    list[int],
    list[int],
    list[int],
]:
    """Apply Webster–Smith–Cohen (arXiv:2511.15989 §II.B.2) merge.

    Parameters
    ----------
    H_X, H_Z
        Tentative X-stab / Z-stab matrices of the same-basis merged code,
        shape ``(N_X, n_merged)`` and ``(N_Z, n_merged)`` over GF(2).
    merge_qubits
        Bridge qubit column indices to merge at. Sorted ascending internally
        for determinism.

    Returns
    -------
    H_X_out, H_Z_out
        Modified stabilizer matrices (rows possibly removed where cross-merged).
    Y_stab
        ``(n_Y, 2*n_merged)`` symplectic Y-stab matrix, or ``None`` if no
        cross-merge occurred.
    obs0_y_indices
        Row indices into ``Y_stab`` whose outcomes XOR into obs0 (per Lemma 2
        of the design spec). All cross-merged qubits contribute; equals
        ``list(range(n_Y))``.
    x_leftover_indices
        Row indices into ``H_X_out`` of X-cycle pivots that survived pair-merge
        without a Z partner (i.e. their qubit had X conflict but no Z conflict).
        Contribute to obs0 per Lemma 2 §9.
    z_leftover_indices
        Symmetric on the Z side.

    Caller's ``H_X``, ``H_Z`` arrays are not mutated (internal copies are made).

    Notes
    -----
    Iteration order: ascending qubit index. Per Lemma 1 of the design spec,
    after Step A at qubit q no other X-row touches q, so subsequent merges
    at q' > q are independent of order on the X-side; symmetric for Z.
    """
    Hx = H_X.copy().astype(np.uint8)
    Hz = H_Z.copy().astype(np.uint8)
    Y_rows: list[np.ndarray] = []
    obs0_y: list[int] = []
    # Track leftover row indices in the *current* (post-deletion) matrices.
    # We refresh from scratch after each merge step.
    x_leftover_set: set[int] = set()
    z_leftover_set: set[int] = set()

    for q in sorted(int(q) for q in merge_qubits):
        Hx_before_rows = Hx.shape[0]
        Hz_before_rows = Hz.shape[0]
        Hx, Hz, Y_row, lx, lz = _merge_at_qubit(Hx, Hz, q)

        if Y_row is not None:
            obs0_y.append(len(Y_rows))
            Y_rows.append(Y_row)
            # Cross-merged rows are removed entirely. ``lx`` / ``lz`` are the
            # PRE-DELETION row indices that were just consumed. Update the
            # tracked leftover sets to the post-deletion frame:
            #   - drop any element equal to the deleted row
            #   - decrement any element strictly greater than the deleted row
            assert lx is not None and lz is not None
            assert Hx.shape[0] == Hx_before_rows - 1
            assert Hz.shape[0] == Hz_before_rows - 1
            new_x_leftover: set[int] = set()
            for old_idx in x_leftover_set:
                if old_idx == lx:
                    continue
                new_x_leftover.add(old_idx - 1 if old_idx > lx else old_idx)
            x_leftover_set = new_x_leftover
            new_z_leftover: set[int] = set()
            for old_idx in z_leftover_set:
                if old_idx == lz:
                    continue
                new_z_leftover.add(old_idx - 1 if old_idx > lz else old_idx)
            z_leftover_set = new_z_leftover
        else:
            if lx is not None:
                x_leftover_set.add(lx)
            if lz is not None:
                z_leftover_set.add(lz)

    Y_stab = np.array(Y_rows, dtype=np.uint8) if Y_rows else None
    return (
        Hx,
        Hz,
        Y_stab,
        obs0_y,
        sorted(x_leftover_set),
        sorted(z_leftover_set),
    )
