"""Tests for the Webster–Smith–Cohen mixed-basis merge algorithm
(arXiv:2511.15989 §II.B.2).

The algorithm selects the cross-merge pivots by the structural criterion
"row whose support on the adapter columns equals exactly {q}". In the
seed-operator gadget this picks the χ row uniquely. Other rows (e.g.
SkipTree cycle rows with weight-2 adapter support) are LEFT UNCHANGED.
"""

from __future__ import annotations

import numpy as np

from qldpc.circuits.surgery.hmatrix.merge import apply_mixed_basis_merge


def _symplectic_inner(row_a: np.ndarray, row_b: np.ndarray, n: int) -> int:
    """⟨A,B⟩_s = A_x · B_z + A_z · B_x  (mod 2)."""
    ax, az = row_a[:n], row_a[n:]
    bx, bz = row_b[:n], row_b[n:]
    return int((ax @ bz + az @ bx) % 2)


def _row(x_supp, z_supp, n):
    out = np.zeros(2 * n, dtype=np.uint8)
    for q in x_supp:
        out[q] = 1
    for q in z_supp:
        out[n + q] = 1
    return out


def test_merge_empty_merge_qubits_is_identity() -> None:
    """If merge_qubits is empty, H_X and H_Z return unchanged and Y_stab is None."""
    H_X = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint8)
    H_Z = np.array([[0, 0, 1]], dtype=np.uint8)
    Hx2, Hz2, Y, obs0_y, x_l, z_l = apply_mixed_basis_merge(H_X, H_Z, (), adapter_cols=(0, 1, 2))
    assert np.array_equal(Hx2, H_X)
    assert np.array_equal(Hz2, H_Z)
    assert Y is None
    assert obs0_y == []
    assert x_l == [] and z_l == []


def test_merge_picks_chi_row_not_cycle_row() -> None:
    """When both a χ_l (single-adapter-col {q}) and a cycle (multi-adapter-col) row
    touch adapter[q], the χ_l row is picked as the X-pivot. The cycle row is NOT modified.

    Adapter cols are (3, 4, 5). The χ_l row has X support exactly on col 3 (single
    adapter col). The cycle row has X support on cols 3, 4 (multi adapter col).
    Suppose merge happens at q=3; there's no Z-pivot at q=3 so no Y row is formed,
    but verifying the H_X output: cycle row UNCHANGED, χ_l UNCHANGED.
    """
    # 6 cols total: cols 0–2 are data/κ, cols 3–5 are adapter.
    adapter = (3, 4, 5)
    cycle_row = _row([3, 4], [], 6)  # weight-2 adapter support
    chi_l_row = _row([0, 3], [], 6)  # single adapter col {3}
    H_X = np.vstack([cycle_row[:6][None, :], chi_l_row[:6][None, :]]).astype(np.uint8)
    H_Z = np.zeros((0, 6), dtype=np.uint8)
    Hx2, _, Y, _, _, _ = apply_mixed_basis_merge(H_X, H_Z, (3,), adapter_cols=adapter)
    # No Z partner → no Y row → no deletions.
    assert Y is None
    assert Hx2.shape == (2, 6)
    # H_X unchanged
    assert np.array_equal(Hx2, H_X)


def test_merge_chi_pair_cross_merges_into_y_row() -> None:
    """χ_l (X) and χ_r (Z) both with single-adapter-col {q} → cross-merge into Y_q."""
    adapter = (3, 4, 5)
    chi_l = _row([0, 3], [], 6)  # X on col 3
    chi_r = _row([], [1, 3], 6)  # Z on col 3
    H_X = chi_l[:6][None, :].astype(np.uint8)
    H_Z = chi_r[6:][None, :].astype(np.uint8)
    Hx2, Hz2, Y, obs0_y, x_l, z_l = apply_mixed_basis_merge(H_X, H_Z, (3,), adapter_cols=adapter)
    # Both rows consumed.
    assert Hx2.shape == (0, 6)
    assert Hz2.shape == (0, 6)
    assert Y is not None
    assert Y.shape == (1, 12)
    # Y row's X-part = chi_l's X-support; Z-part = chi_r's Z-support.
    assert np.array_equal(Y[0, :6], chi_l[:6])
    assert np.array_equal(Y[0, 6:], chi_r[6:])
    assert obs0_y == [0]
    assert x_l == [] and z_l == []


def test_merge_cycle_row_unchanged_when_chi_pair_present() -> None:
    """When both χ_l and χ_r are picked as pivots at q, the cycle row (multi-adapter)
    is NEVER modified — it stays in H_Z exactly as it was."""
    adapter = (3, 4, 5)
    chi_l = _row([0, 3], [], 6)  # X on adapter col 3
    chi_r = _row([], [1, 3], 6)  # Z on adapter col 3
    cycle = _row([], [3, 4], 6)  # Z, multi-adapter cols 3,4
    H_X = chi_l[:6][None, :].astype(np.uint8)
    # cycle goes first in H_Z, then chi_r
    H_Z = np.vstack([cycle[6:][None, :], chi_r[6:][None, :]]).astype(np.uint8)
    Hx2, Hz2, Y, _, _, _ = apply_mixed_basis_merge(H_X, H_Z, (3,), adapter_cols=adapter)
    # χ_l + χ_r consumed; cycle survives in H_Z exactly as input.
    assert Hx2.shape == (0, 6)
    assert Hz2.shape == (1, 6)
    assert Y is not None
    # cycle row unchanged
    assert np.array_equal(Hz2[0], cycle[6:])


def test_merge_two_independent_q_produces_two_y_rows() -> None:
    """Two cross-merges at q=3 and q=5 each produce one Y row."""
    adapter = (3, 4, 5)
    chi_l_3 = _row([0, 3], [], 6)
    chi_l_5 = _row([0, 5], [], 6)
    chi_r_3 = _row([], [1, 3], 6)
    chi_r_5 = _row([], [1, 5], 6)
    H_X = np.vstack([chi_l_3[:6][None, :], chi_l_5[:6][None, :]]).astype(np.uint8)
    H_Z = np.vstack([chi_r_3[6:][None, :], chi_r_5[6:][None, :]]).astype(np.uint8)
    Hx2, Hz2, Y, obs0_y, _, _ = apply_mixed_basis_merge(
        H_X, H_Z, (3, 5), adapter_cols=adapter
    )
    assert Hx2.shape == (0, 6)
    assert Hz2.shape == (0, 6)
    assert Y is not None and Y.shape == (2, 12)
    assert obs0_y == [0, 1]


def test_merge_no_chi_row_means_no_merge() -> None:
    """If no row has single-adapter-col support {q}, no merge happens at q."""
    adapter = (3, 4, 5)
    cycle = _row([], [3, 4], 6)  # multi-adapter, not a χ
    H_X = np.zeros((0, 6), dtype=np.uint8)
    H_Z = cycle[6:][None, :].astype(np.uint8)
    Hx2, Hz2, Y, obs0_y, _, _ = apply_mixed_basis_merge(
        H_X, H_Z, (3, 4), adapter_cols=adapter
    )
    assert Y is None
    assert np.array_equal(Hz2, H_Z)
    assert obs0_y == []


def test_merge_does_not_mutate_inputs() -> None:
    """Caller's H_X / H_Z arrays must be unchanged after apply_mixed_basis_merge."""
    H_X = np.array([[1, 0, 0, 1]], dtype=np.uint8)
    H_Z = np.array([[0, 1, 0, 1]], dtype=np.uint8)
    H_X_orig = H_X.copy()
    H_Z_orig = H_Z.copy()
    _ = apply_mixed_basis_merge(H_X, H_Z, (3,), adapter_cols=(3,))
    assert np.array_equal(H_X, H_X_orig)
    assert np.array_equal(H_Z, H_Z_orig)


def test_merge_iteration_order_deterministic() -> None:
    """merge_qubits=(3, 5) vs (5, 3) produce identical results (sorted internally)."""
    adapter = (3, 4, 5)
    chi_l_3 = _row([3], [], 6)
    chi_l_5 = _row([5], [], 6)
    chi_r_3 = _row([], [3], 6)
    chi_r_5 = _row([], [5], 6)
    H_X = np.vstack([chi_l_3[:6][None, :], chi_l_5[:6][None, :]]).astype(np.uint8)
    H_Z = np.vstack([chi_r_3[6:][None, :], chi_r_5[6:][None, :]]).astype(np.uint8)
    res_a = apply_mixed_basis_merge(H_X.copy(), H_Z.copy(), (3, 5), adapter_cols=adapter)
    res_b = apply_mixed_basis_merge(H_X.copy(), H_Z.copy(), (5, 3), adapter_cols=adapter)
    assert np.array_equal(res_a[0], res_b[0])
    assert np.array_equal(res_a[1], res_b[1])
    if res_a[2] is None:
        assert res_b[2] is None
    else:
        assert np.array_equal(res_a[2], res_b[2])
    assert res_a[3] == res_b[3]


def test_merge_y_rows_pairwise_commute_symplectically() -> None:
    """Lemma 1 part (a): ⟨Y_q, Y_q'⟩_s = 0 for all distinct q, q' merged.

    Pure unit test of ``apply_mixed_basis_merge``: builds χ_l / χ_r rows
    with single-adapter-col support {q} each, and verifies the resulting Y
    rows pairwise commute. The cycle-row commutation property requires the
    SkipTree identity (T_l · F_aug · P = H_R) from Swaroop arXiv:2407.18393
    §III and is exercised at the integration level by the joint-stitch
    commutation test in ``bridge_mixed_test.py``.

    Also verifies the cycle row is LEFT UNCHANGED by the algorithm (its row
    contents are identical to the input).
    """
    n = 8
    adapter = (5, 6, 7)
    # Construct several χ_l / χ_r rows with single-adapter-col support and
    # varied data + κ support, plus a cycle row.
    rows_X = [
        _row([0, 1, 5], [], n)[:n][None, :],  # χ_l at q=5
        _row([0, 2, 6], [], n)[:n][None, :],  # χ_l at q=6
        _row([1, 2, 7], [], n)[:n][None, :],  # χ_l at q=7
    ]
    cycle_z_input = _row([], [5, 6], n)[n:]  # cycle row (multi-adapter cols 5,6)
    rows_Z = [
        _row([], [3, 4, 5], n)[n:][None, :],  # χ_r at q=5
        _row([], [3, 4, 6], n)[n:][None, :],  # χ_r at q=6 (has support on κ_l col 3, κ_r col 4)
        _row([], [4, 7], n)[n:][None, :],     # χ_r at q=7
        cycle_z_input[None, :],
    ]
    H_X = np.vstack(rows_X).astype(np.uint8)
    H_Z = np.vstack(rows_Z).astype(np.uint8)
    Hx2, Hz2, Y, _, _, _ = apply_mixed_basis_merge(H_X, H_Z, (5, 6, 7), adapter_cols=adapter)
    assert Y is not None and Y.shape[0] == 3
    # Pairwise symplectic commutation among Y rows.
    for i in range(Y.shape[0]):
        for j in range(i + 1, Y.shape[0]):
            assert _symplectic_inner(Y[i], Y[j], n) == 0, (
                f"Y_{i} and Y_{j} anticommute"
            )
    # The cycle row survives in H_Z UNCHANGED (the new algorithm never modifies
    # multi-adapter-col rows during merge).
    assert Hz2.shape[0] == 1
    assert np.array_equal(Hz2[0], cycle_z_input)
