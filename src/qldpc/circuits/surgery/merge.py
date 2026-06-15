"""Webster–Smith–Cohen (arXiv:2511.15989 §II.B.2) mixed-basis merge.

Pure GF(2) operations on (H_X, H_Z). At each merge qubit q, identifies the
χ-style row whose support on the adapter columns is EXACTLY {q} as the
leftover X (and similarly Z). If both exist, cross-merges them into a single
Y-type symplectic row. Other rows — including SkipTree cycle rows with
weight-2 adapter support — are LEFT UNCHANGED.

The single-{q} adapter-col criterion is the structural signature of the
χ-rows from Webster's gadget construction (§II.A step 2): χ_v has adapter
support = {label(v)}. Cycle rows have weight-2 adapter support and would
break ⟨Y_q, cycle⟩ commutation if XOR'd into during pair-merge.
"""

from __future__ import annotations

import numpy as np


def _find_single_adapter_row(
    H: np.ndarray,
    q: int,
    adapter_set: frozenset[int],
    excluded: set[int],
) -> int | None:
    """Return the first row index r (not in ``excluded``) whose adapter support
    equals exactly ``{q}``, or ``None``."""
    for r in range(H.shape[0]):
        if r in excluded:
            continue
        # Adapter support of row r
        supp = {c for c in adapter_set if H[r, c] == 1}
        if supp == {q}:
            return r
    return None


def apply_mixed_basis_merge(
    H_X: np.ndarray,
    H_Z: np.ndarray,
    merge_qubits: tuple[int, ...],
    adapter_cols: tuple[int, ...] = (),
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    list[int],
    list[int],
    list[int],
]:
    """Webster–Smith–Cohen arXiv:2511.15989 §II.B.2 cross-merge.

    See module docstring for the algorithm. Caller's ``H_X`` / ``H_Z`` are not
    mutated.

    Parameters
    ----------
    H_X, H_Z
        ``(rows, n_merged)`` GF(2) stabilizer matrices.
    merge_qubits
        Adapter columns to attempt cross-merge at. Sorted ascending internally.
    adapter_cols
        Full set of adapter-column indices, used to compute the single-``{q}``
        adapter-col pivot criterion.

    Returns
    -------
    H_X_out, H_Z_out
        Modified matrices (rows removed where cross-merged).
    Y_stab
        ``(n_Y, 2*n_merged)`` symplectic Y rows, or ``None``.
    obs0_y_indices
        ``list(range(n_Y))`` — all Y rows contribute to obs0 per Lemma 2.
    x_leftover_indices, z_leftover_indices
        Always ``[]``. Returned for backward compatibility with the previous
        algorithm's interface; the new algorithm does not produce surviving
        leftovers (no pair-merge of multi-adapter rows is performed).
    """
    Hx = H_X.copy().astype(np.uint8)
    Hz = H_Z.copy().astype(np.uint8)
    adapter_set = frozenset(int(c) for c in adapter_cols)
    Y_rows: list[np.ndarray] = []
    obs0_y: list[int] = []
    rows_to_delete_x: set[int] = set()
    rows_to_delete_z: set[int] = set()

    n = Hx.shape[1]

    for q in sorted(int(q) for q in merge_qubits):
        x_pivot = _find_single_adapter_row(Hx, q, adapter_set, rows_to_delete_x)
        z_pivot = _find_single_adapter_row(Hz, q, adapter_set, rows_to_delete_z)
        if x_pivot is None or z_pivot is None:
            continue
        Y_row = np.zeros(2 * n, dtype=np.uint8)
        Y_row[:n] = Hx[x_pivot]
        Y_row[n:] = Hz[z_pivot]
        obs0_y.append(len(Y_rows))
        Y_rows.append(Y_row)
        rows_to_delete_x.add(x_pivot)
        rows_to_delete_z.add(z_pivot)

    Hx_out = np.delete(Hx, sorted(rows_to_delete_x), axis=0)
    Hz_out = np.delete(Hz, sorted(rows_to_delete_z), axis=0)

    Y_stab = np.array(Y_rows, dtype=np.uint8) if Y_rows else None
    return Hx_out, Hz_out, Y_stab, obs0_y, [], []
