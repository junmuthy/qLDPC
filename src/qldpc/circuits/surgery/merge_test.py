"""Tests for the Webster–Smith–Cohen mixed-basis merge algorithm
(arXiv:2511.15989 §II.B.2)."""

from __future__ import annotations

import numpy as np

from qldpc.circuits.surgery.merge import apply_mixed_basis_merge


def _symplectic_inner(row_a: np.ndarray, row_b: np.ndarray, n: int) -> int:
    """⟨A,B⟩_s = A_x · B_z + A_z · B_x  (mod 2)."""
    ax, az = row_a[:n], row_a[n:]
    bx, bz = row_b[:n], row_b[n:]
    return int((ax @ bz + az @ bx) % 2)


def test_merge_with_no_conflict_qubits_is_identity() -> None:
    """If merge_qubits is empty, H_X and H_Z are returned unchanged with no Y rows."""
    H_X = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint8)
    H_Z = np.array([[0, 0, 1]], dtype=np.uint8)
    H_X2, H_Z2, Y, obs0_y, x_left, z_left = apply_mixed_basis_merge(H_X, H_Z, ())
    assert np.array_equal(H_X2, H_X)
    assert np.array_equal(H_Z2, H_Z)
    assert Y is None
    assert obs0_y == []
    assert x_left == []
    assert z_left == []


def test_merge_pair_only_x_creates_no_y_row() -> None:
    """When only X-checks touch the merge qubit (no Z-conflict), pair-merge X
    and record an X-leftover; no Y row generated."""
    # Two X-checks both touching qubit 0; no Z-check touches qubit 0
    H_X = np.array(
        [
            [1, 1, 0],
            [1, 0, 1],
        ],
        dtype=np.uint8,
    )
    H_Z = np.array([[0, 1, 1]], dtype=np.uint8)
    H_X2, H_Z2, Y, obs0_y, x_left, z_left = apply_mixed_basis_merge(H_X, H_Z, (0,))
    assert Y is None
    assert obs0_y == []
    # Pivot row 0 keeps qubit-0 entry; row 1 gets row0 XORed in → loses qubit 0
    assert H_X2.shape == (2, 3)
    assert H_X2[0, 0] == 1  # pivot retained
    assert H_X2[1, 0] == 0  # cancelled
    # H_Z unchanged (no Z-conflict at qubit 0)
    assert np.array_equal(H_Z2, H_Z)
    assert x_left == [0]  # row 0 is the leftover X
    assert z_left == []


def test_merge_cross_merge_produces_y_row() -> None:
    """When BOTH X-cycle and Z-cycle touch the merge qubit, leftover X + leftover Z
    are removed from H_X/H_Z and combined into a Y row in symplectic form."""
    # Single X-check at qubit 0; single Z-check at qubit 0
    H_X = np.array([[1, 1, 0]], dtype=np.uint8)
    H_Z = np.array([[1, 0, 1]], dtype=np.uint8)
    H_X2, H_Z2, Y, obs0_y, x_left, z_left = apply_mixed_basis_merge(H_X, H_Z, (0,))
    # The X-check and Z-check are removed; Y row carries both.
    assert H_X2.shape == (0, 3)
    assert H_Z2.shape == (0, 3)
    assert Y is not None
    assert Y.shape == (1, 6)  # (n_Y, 2*n_merged)
    # Y row's X-part = original H_X row; Z-part = original H_Z row
    assert np.array_equal(Y[0, :3], np.array([1, 1, 0], dtype=np.uint8))
    assert np.array_equal(Y[0, 3:], np.array([1, 0, 1], dtype=np.uint8))
    assert obs0_y == [0]


def test_merge_multiple_x_rows_paired_to_one_pivot() -> None:
    """Three X-checks at qubit 0 collapse to one pivot (rows 1, 2 get XORed with row 0)."""
    H_X = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
        ],
        dtype=np.uint8,
    )
    H_Z = np.zeros((0, 4), dtype=np.uint8)
    H_X2, _, Y, _, x_left, _ = apply_mixed_basis_merge(H_X, H_Z, (0,))
    assert Y is None
    assert H_X2.shape == (3, 4)
    # Row 0 pivot retains qubit 0
    assert H_X2[0, 0] == 1
    # Rows 1, 2 lose qubit 0 (XOR with pivot cancels)
    assert H_X2[1, 0] == 0
    assert H_X2[2, 0] == 0
    # Row 1 picks up row 0's other support: [1, 1, 1, 0] (was [1,0,1,0] + row0 [1,1,0,0])
    assert np.array_equal(H_X2[1], np.array([0, 1, 1, 0], dtype=np.uint8))
    assert np.array_equal(H_X2[2], np.array([0, 1, 0, 1], dtype=np.uint8))
    assert x_left == [0]


def test_merge_iteration_two_qubits_independent() -> None:
    """Two independent merge qubits; both produce Y rows when both have X+Z conflicts."""
    # 4 qubits. qubit 0 has X-cycle (row 0) and Z-cycle (row 0); qubit 2 has
    # X-cycle (row 1) and Z-cycle (row 1). No interaction.
    H_X = np.array(
        [
            [1, 1, 0, 0],  # X at q0
            [0, 0, 1, 1],  # X at q2
        ],
        dtype=np.uint8,
    )
    H_Z = np.array(
        [
            [1, 0, 0, 1],  # Z at q0
            [0, 1, 1, 0],  # Z at q2
        ],
        dtype=np.uint8,
    )
    H_X2, H_Z2, Y, obs0_y, x_left, z_left = apply_mixed_basis_merge(H_X, H_Z, (0, 2))
    # Both leftover X and Z get cross-merged → 2 Y rows; H_X / H_Z empty
    assert H_X2.shape == (0, 4)
    assert H_Z2.shape == (0, 4)
    assert Y is not None
    assert Y.shape == (2, 8)
    assert obs0_y == [0, 1]


def test_merge_post_merge_stabs_commute_symplectically() -> None:
    """Lemma 1: after merge, all surviving stabilizers commute pairwise
    under the symplectic inner product.

    Uses a valid CSS input (all H_X · H_Z^T = 0 mod 2) so Lemma 1's
    hypothesis is satisfied; the test then verifies the algorithm
    preserves commutation.
    """
    # Two X-checks (rows 0 and 2) and two Z-checks (rows 0 and 2) touch qubit 0;
    # row 1 in each is disjoint from q0. All rows commute pairwise X-vs-Z.
    H_X = np.array(
        [
            [1, 1, 0, 0],  # X at q0
            [0, 0, 1, 1],  # disjoint X (no q0)
            [1, 1, 1, 1],  # X at q0
        ],
        dtype=np.uint8,
    )
    H_Z = np.array(
        [
            [1, 1, 0, 0],  # Z at q0
            [0, 0, 1, 1],  # disjoint Z (no q0)
            [1, 1, 1, 1],  # Z at q0
        ],
        dtype=np.uint8,
    )
    n = 4
    H_X2, H_Z2, Y, _, _, _ = apply_mixed_basis_merge(H_X, H_Z, (0,))

    # Assemble all stabs as symplectic rows
    rows = []
    for r in H_X2:
        rows.append(np.concatenate([r, np.zeros(n, dtype=np.uint8)]))
    for r in H_Z2:
        rows.append(np.concatenate([np.zeros(n, dtype=np.uint8), r]))
    if Y is not None:
        for r in Y:
            rows.append(r.astype(np.uint8))

    for i, a in enumerate(rows):
        for j, b in enumerate(rows):
            if i >= j:
                continue
            assert _symplectic_inner(a, b, n) == 0, (
                f"rows {i} and {j} anticommute: {a} vs {b}"
            )


def test_merge_iteration_order_deterministic_ascending() -> None:
    """merge_qubits=(0, 2) vs (2, 0) — order should be ascending; results match (0, 2)."""
    H_X = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype=np.uint8,
    )
    H_Z = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype=np.uint8,
    )
    res_a = apply_mixed_basis_merge(H_X.copy(), H_Z.copy(), (0, 2))
    res_b = apply_mixed_basis_merge(H_X.copy(), H_Z.copy(), (2, 0))
    # Function should sort merge_qubits internally → same result either way
    assert np.array_equal(res_a[0], res_b[0])
    assert np.array_equal(res_a[1], res_b[1])
    if res_a[2] is None:
        assert res_b[2] is None
    else:
        assert np.array_equal(res_a[2], res_b[2])
    assert res_a[3] == res_b[3]


def test_merge_leftover_indices_account_for_row_deletion() -> None:
    """Regression: after a cross-merge deletes a row from H_X / H_Z, any
    previously-tracked leftover row index must be (a) dropped if it equals
    the deleted index, or (b) decremented if it sits above it.

    Scenario A — drop:
      q=0: X-only conflict at row 0 → x_leftover_set = {0}.
      q=1: X+Z cross-merge whose X-pivot is also row 0 → row 0 deleted.
      Expected: x_left == [] (the formerly-leftover row was consumed).

    Scenario B — decrement:
      q=0: X-only conflict at row 1 → x_leftover_set = {1}.
      q=1: X+Z cross-merge whose X-pivot is row 0 → row 0 deleted.
      Expected: x_left == [0], pointing to the surviving row that used to be
      at index 1 in the pre-delete H_X.
    """
    # Scenario A: leftover index equals deleted index.
    H_X_a = np.array(
        [
            [1, 1, 0],  # X at q0 and q1 (becomes pivot for both merges)
            [0, 0, 1],  # untouched by either merge
        ],
        dtype=np.uint8,
    )
    H_Z_a = np.array(
        [
            [0, 1, 1],  # Z at q1 only (no q0)
        ],
        dtype=np.uint8,
    )
    H_X2, H_Z2, Y, _obs0, x_left, z_left = apply_mixed_basis_merge(
        H_X_a, H_Z_a, (0, 1)
    )
    # Y row produced by the q=1 cross-merge.
    assert Y is not None and Y.shape == (1, 6)
    # Row 0 of H_X was consumed → only original row 1 survives at new index 0.
    assert H_X2.shape == (1, 3)
    assert np.array_equal(H_X2[0], np.array([0, 0, 1], dtype=np.uint8))
    # Row 0 of H_Z was consumed → empty.
    assert H_Z2.shape == (0, 3)
    # Previously-leftover row 0 is gone; nothing left.
    assert x_left == [], (
        f"Scenario A: expected x_left=[], got {x_left}; stale index points at "
        f"row content {H_X2[x_left[0]] if x_left else 'N/A'}"
    )
    assert z_left == []

    # Scenario B: leftover index above deleted index → must decrement.
    H_X_b = np.array(
        [
            [0, 1, 0],  # touches q1 only (becomes q=1 X-pivot, will be deleted)
            [1, 0, 1],  # touches q0 only (becomes q=0 leftover at row index 1)
        ],
        dtype=np.uint8,
    )
    H_Z_b = np.array(
        [
            [0, 1, 1],  # touches q1, used as q=1 Z-pivot
        ],
        dtype=np.uint8,
    )
    H_X2, H_Z2, Y, _obs0, x_left, z_left = apply_mixed_basis_merge(
        H_X_b, H_Z_b, (0, 1)
    )
    assert Y is not None and Y.shape == (1, 6)
    # H_X row 0 deleted by cross-merge → surviving row (was index 1) at index 0.
    assert H_X2.shape == (1, 3)
    assert np.array_equal(H_X2[0], np.array([1, 0, 1], dtype=np.uint8))
    # x_leftover should now point at new index 0 (the surviving X-leftover row).
    assert x_left == [0], (
        f"Scenario B: expected x_left=[0], got {x_left}; "
        f"surviving H_X rows = {H_X2.tolist()}"
    )
    # Verify the row at the reported leftover index is actually the row that
    # carried the q=0 X-leftover support — i.e. has support at q0.
    assert H_X2[x_left[0], 0] == 1, (
        "x_left index does not point at a row supporting q0 (the conflict qubit)"
    )
    assert z_left == []


def test_merge_does_not_mutate_inputs() -> None:
    """Caller's H_X / H_Z arrays must be unchanged after apply_mixed_basis_merge."""
    H_X = np.array([[1, 1, 0]], dtype=np.uint8)
    H_Z = np.array([[1, 0, 1]], dtype=np.uint8)
    H_X_orig = H_X.copy()
    H_Z_orig = H_Z.copy()
    _ = apply_mixed_basis_merge(H_X, H_Z, (0,))
    assert np.array_equal(H_X, H_X_orig)
    assert np.array_equal(H_Z, H_Z_orig)
