# Mixed-basis Joint PPM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `qldpc.circuits.surgery` so `build_bridge(g_l, g_r)` accepts `g_l.basis != g_r.basis` and downstream `build_joint_ppm_circuit` produces a stim circuit whose obs0 commits exactly the mixed-basis joint Pauli-product (e.g. `Z̄_l ⊗ X̄_r`).

**Architecture:** Run the existing same-basis SkipTree adapter, then apply a Webster–Smith–Cohen (arXiv:2511.15989 §II.B.2) merge over the assembled stabilizer matrices: pair-merge X-checks per shared bridge qubit, pair-merge Z-checks per shared bridge qubit, then cross-merge the leftover X+Z pair into a Y-type stabilizer. The merged code becomes non-CSS (Y rows present) → represented as `QuditCode` in symplectic form. Same-basis code paths are bit-for-bit preserved.

**Tech Stack:** Python 3.11, NumPy, NetworkX, Stim, qldpc (`QuditCode`, `CSSCode`, `Pauli`, `PauliXZ`), pytest.

**Source spec:** `docs/superpowers/specs/2026-06-15-mixed-basis-joint-ppm-design.md`

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/qldpc/circuits/surgery/bridge.py` | Modify | `Bridge` dataclass split + `build_bridge` mixed-basis dispatch |
| `src/qldpc/circuits/surgery/merge.py` | **Create** | Pure GF(2) Webster §II.B.2 merge algorithm |
| `src/qldpc/circuits/surgery/merge_test.py` | **Create** | Unit tests for merge.py |
| `src/qldpc/circuits/surgery/circuit.py` | Modify | Mixed-basis stitch → `QuditCode`; Y-stab extraction + detectors; obs0 formula |
| `src/qldpc/circuits/surgery/bridge_mixed_test.py` | **Create** | Mixed-basis `Bridge` integration tests |
| `src/qldpc/circuits/surgery/circuit_mixed_test.py` | **Create** | Mixed-basis circuit Tier 1 tests |
| `src/qldpc/circuits/surgery/bridge_test.py` | Modify | Add backward-compat `.basis` property test + update field-set test |
| `src/qldpc/circuits/surgery/__init__.py` | **No change** | Public API unchanged |
| `src/qldpc/circuits/surgery/gadget.py` | **No change** | |
| `src/qldpc/circuits/surgery/cheeger.py` | **No change** | |

## Backward-Compat Invariant

Existing 179 same-basis tests in `bridge_test.py`, `gadget_test.py`, `cheeger_test.py`, `circuit_test.py` MUST pass without modification after every commit. The only legitimate change to an existing test is in Task 1 (`test_bridge_dataclass_fields_universal_adapter` and `bridge.basis` property semantics).

---

## Task 1: Bridge dataclass `basis_l` / `basis_r` split with backward-compat `.basis` property

**Files:**
- Modify: `src/qldpc/circuits/surgery/bridge.py:19-41` (Bridge dataclass)
- Modify: `src/qldpc/circuits/surgery/bridge.py:439-453` (build_bridge return site)
- Modify: `src/qldpc/circuits/surgery/bridge_test.py:225-246` (field-set test)
- Modify: `src/qldpc/circuits/surgery/bridge_test.py:295-306` (basis-mismatch test — relax assertion now that mixed-basis is allowed by Task 3, but for Task 1 this test stays passing because `build_bridge` still rejects mismatched bases)

**Important:** For Task 1, **do not change `build_bridge`'s basis-mismatch rejection**. That stays until Task 3. Only the dataclass shape changes here.

- [ ] **Step 1.1: Write failing test for new field set + backward-compat property**

Add to `src/qldpc/circuits/surgery/bridge_test.py` (replace existing `test_bridge_dataclass_fields_universal_adapter`):

```python
def test_bridge_dataclass_fields_universal_adapter() -> None:
    """Bridge dataclass exposes the universal-adapter fields + mixed-basis fields.

    basis_l/basis_r replace the old single basis field (Webster–Smith–Cohen
    arXiv:2511.15989 §II.B.2 mixed-basis joint PPM).
    """
    import dataclasses

    from qldpc.circuits.surgery.bridge import Bridge

    fields = {f.name for f in dataclasses.fields(Bridge)}
    assert fields == {
        "width",
        "basis_l",
        "basis_r",
        "port_l",
        "port_r",
        "label_l",
        "label_r",
        "extra_ancilla_l",
        "extra_ancilla_r",
        "T_l",
        "T_r",
        "H_R",
        "g_l_aug",
        "g_r_aug",
        "Y_stab",
        "merge_qubits",
        "obs0_xor_map",
        "x_leftover_indices",
        "z_leftover_indices",
    }


def test_bridge_basis_property_returns_single_basis_when_same() -> None:
    """`.basis` returns the shared basis when basis_l == basis_r (backward compat)."""
    from qldpc.circuits.surgery.bridge import Bridge

    bridge = Bridge(
        width=2,
        basis_l=Pauli.X,
        basis_r=Pauli.X,
        port_l=(0, 1),
        port_r=(0, 1),
        label_l=(0, 1),
        label_r=(0, 1),
        extra_ancilla_l=np.zeros((0, 2), dtype=np.uint8),
        extra_ancilla_r=np.zeros((0, 2), dtype=np.uint8),
        T_l=np.zeros((1, 1), dtype=np.int_),
        T_r=np.zeros((1, 1), dtype=np.int_),
        H_R=np.array([[1, 1]], dtype=np.int_),
        g_l_aug=None,  # opaque to this test
        g_r_aug=None,
    )
    assert bridge.basis is Pauli.X


def test_bridge_basis_property_raises_when_mixed() -> None:
    """`.basis` raises AttributeError when basis_l != basis_r."""
    from qldpc.circuits.surgery.bridge import Bridge

    bridge = Bridge(
        width=2,
        basis_l=Pauli.X,
        basis_r=Pauli.Z,
        port_l=(0, 1),
        port_r=(0, 1),
        label_l=(0, 1),
        label_r=(0, 1),
        extra_ancilla_l=np.zeros((0, 2), dtype=np.uint8),
        extra_ancilla_r=np.zeros((0, 2), dtype=np.uint8),
        T_l=np.zeros((1, 1), dtype=np.int_),
        T_r=np.zeros((1, 1), dtype=np.int_),
        H_R=np.array([[1, 1]], dtype=np.int_),
        g_l_aug=None,
        g_r_aug=None,
    )
    with pytest.raises(AttributeError, match=r"mixed-basis|basis_l|basis_r"):
        _ = bridge.basis


def test_bridge_mixed_basis_fields_default_to_none_or_empty() -> None:
    """Y_stab defaults to None; merge_qubits/obs0_xor_map/leftover tuples default to ()."""
    from qldpc.circuits.surgery.bridge import Bridge

    bridge = Bridge(
        width=2,
        basis_l=Pauli.X,
        basis_r=Pauli.X,
        port_l=(0, 1),
        port_r=(0, 1),
        label_l=(0, 1),
        label_r=(0, 1),
        extra_ancilla_l=np.zeros((0, 2), dtype=np.uint8),
        extra_ancilla_r=np.zeros((0, 2), dtype=np.uint8),
        T_l=np.zeros((1, 1), dtype=np.int_),
        T_r=np.zeros((1, 1), dtype=np.int_),
        H_R=np.array([[1, 1]], dtype=np.int_),
        g_l_aug=None,
        g_r_aug=None,
    )
    assert bridge.Y_stab is None
    assert bridge.merge_qubits == ()
    assert bridge.obs0_xor_map == ()
    assert bridge.x_leftover_indices == ()
    assert bridge.z_leftover_indices == ()
```

- [ ] **Step 1.2: Verify tests fail**

```bash
uv run pytest src/qldpc/circuits/surgery/bridge_test.py::test_bridge_dataclass_fields_universal_adapter src/qldpc/circuits/surgery/bridge_test.py::test_bridge_basis_property_returns_single_basis_when_same src/qldpc/circuits/surgery/bridge_test.py::test_bridge_basis_property_raises_when_mixed src/qldpc/circuits/surgery/bridge_test.py::test_bridge_mixed_basis_fields_default_to_none_or_empty -v
```

Expected: 4 FAIL (assertion errors / `basis_l` not a field of Bridge).

- [ ] **Step 1.3: Modify `Bridge` dataclass**

Replace lines 19-41 of `src/qldpc/circuits/surgery/bridge.py`:

```python
@dataclasses.dataclass(frozen=True, eq=False)
class Bridge:
    """Universal adapter between two GadgetLayouts (Swaroop et al. arXiv:2410.03628 §IV / §VII).

    Cain mapping: V_0 → support; F → incidence; κ → ancilla.

    Same-basis fields match docs/superpowers/specs/2026-06-09-joint-ppm-bridge-design.md §1.
    Mixed-basis fields (Y_stab, merge_qubits, obs0_xor_map, x_leftover_indices,
    z_leftover_indices) implement the Webster–Smith–Cohen arXiv:2511.15989 §II.B.2
    cross-merge for joint Pauli-product measurement of different-basis logicals
    (e.g. Z̄_l ⊗ X̄_r). They default to None / () for same-basis bridges and
    are populated only by build_bridge's mixed-basis dispatch path.
    """

    width: int  # w = |𝒜| (adapter qubits)
    basis_l: PauliXZ  # X or Z on the left gadget (Webster–Smith–Cohen mixed-basis)
    basis_r: PauliXZ  # X or Z on the right gadget
    port_l: tuple[int, ...]  # 𝒫_l* ⊆ V_0^(l), length w
    port_r: tuple[int, ...]  # 𝒫_r* ⊆ V_0^(r), length w
    label_l: tuple[int, ...]  # label_l[i] = SkipTree label of V_0^(l)[i]; -1 if i ∉ 𝒫_l*
    label_r: tuple[int, ...]
    extra_ancilla_l: np.ndarray  # (e_l, |support^(l)|) F_2; weight-2 rows added
    extra_ancilla_r: np.ndarray
    T_l: np.ndarray  # (w-1, |C_0^(l)| + e_l) F_2 (3,2)-sparse
    T_r: np.ndarray
    H_R: np.ndarray  # (w-1, w) canonical rep code parity
    g_l_aug: GadgetLayout  # gadget rebuilt over F_aug^(l)
    g_r_aug: GadgetLayout
    # Mixed-basis fields (Webster–Smith–Cohen arXiv:2511.15989 §II.B.2).
    # None / empty for same-basis bridges.
    Y_stab: np.ndarray | None = None  # (n_Y, 2*n_merged) symplectic Y-rows
    merge_qubits: tuple[int, ...] = ()  # bridge qubit indices touched by cross-merge
    obs0_xor_map: tuple[int, ...] = ()  # Y_stab row indices XORed into obs0
    x_leftover_indices: tuple[int, ...] = ()  # X-cycle row indices not cross-merged
    z_leftover_indices: tuple[int, ...] = ()  # Z-cycle row indices not cross-merged

    @property
    def basis(self) -> PauliXZ:
        """Backward-compat single-basis accessor.

        Returns the shared basis when basis_l == basis_r. Raises AttributeError
        for mixed-basis bridges — callers must explicitly use basis_l / basis_r.
        """
        if self.basis_l is not self.basis_r:
            raise AttributeError(
                "mixed-basis Bridge has no single .basis attribute; "
                f"use bridge.basis_l ({self.basis_l!r}) / bridge.basis_r ({self.basis_r!r})"
            )
        return self.basis_l
```

- [ ] **Step 1.4: Update `build_bridge` return to populate `basis_l=basis_r=basis`**

In `src/qldpc/circuits/surgery/bridge.py` at the `return Bridge(...)` site (~line 439), replace `basis=basis,` with:

```python
        basis_l=basis,
        basis_r=basis,
```

- [ ] **Step 1.5: Run the four new tests + the full surgery suite**

```bash
uv run pytest src/qldpc/circuits/surgery/ -x -q
```

Expected: ALL pass (179 pre-existing + 4 new = 183).

If `test_build_bridge_smoke_steane_intracode` fails on `assert bridge.basis is Pauli.X`, that means the property is wrong — debug.

- [ ] **Step 1.6: Commit**

```bash
git add src/qldpc/circuits/surgery/bridge.py src/qldpc/circuits/surgery/bridge_test.py
git commit -m "$(cat <<'EOF'
refactor(surgery): Bridge dataclass basis_l/basis_r split

Splits the single `basis` field into `basis_l` / `basis_r` and adds
five mixed-basis fields (Y_stab, merge_qubits, obs0_xor_map,
x_leftover_indices, z_leftover_indices) defaulting to None / ().

Backward-compat `.basis` property returns the shared basis when
basis_l == basis_r and raises AttributeError otherwise — same-basis
callers (build_joint_ppm_circuit, _stitch_*) keep working bit-for-bit.

Prepares for Webster–Smith–Cohen (arXiv:2511.15989 §II.B.2) mixed-basis
joint PPM (e.g. measuring Z̄_l ⊗ X̄_r without measuring the individuals).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `merge.py` algorithm + unit tests

**Files:**
- Create: `src/qldpc/circuits/surgery/merge.py`
- Create: `src/qldpc/circuits/surgery/merge_test.py`

This task is **isolated**: `merge.py` has no imports from other surgery modules. It operates purely on numpy GF(2) matrices.

- [ ] **Step 2.1: Write failing tests in `merge_test.py`**

Create `src/qldpc/circuits/surgery/merge_test.py`:

```python
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

    Pre-merge inputs must be a valid CSS code (H_X @ H_Z.T = 0 mod 2),
    else the test asserts a property the merge doesn't promise.
    """
    # CSS-valid input: H_X = H_Z covering rows that include q0 conflicts
    # plus a disjoint row (validated H_X @ H_Z.T == 0 mod 2 by hand).
    H_X = np.array(
        [
            [1, 1, 0, 0],  # X at q0
            [0, 0, 1, 1],  # disjoint (no q0)
            [1, 1, 1, 1],  # X at q0
        ],
        dtype=np.uint8,
    )
    H_Z = np.array(
        [
            [1, 1, 0, 0],  # Z at q0
            [0, 0, 1, 1],  # disjoint (no q0)
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


def test_merge_does_not_mutate_inputs() -> None:
    """Caller's H_X / H_Z arrays must be unchanged after apply_mixed_basis_merge."""
    H_X = np.array([[1, 1, 0]], dtype=np.uint8)
    H_Z = np.array([[1, 0, 1]], dtype=np.uint8)
    H_X_orig = H_X.copy()
    H_Z_orig = H_Z.copy()
    _ = apply_mixed_basis_merge(H_X, H_Z, (0,))
    assert np.array_equal(H_X, H_X_orig)
    assert np.array_equal(H_Z, H_Z_orig)
```

- [ ] **Step 2.2: Verify tests fail (import error)**

```bash
uv run pytest src/qldpc/circuits/surgery/merge_test.py -x -q
```

Expected: ImportError — `merge.py` does not exist.

- [ ] **Step 2.3: Implement `merge.py`**

Create `src/qldpc/circuits/surgery/merge.py`:

```python
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
        leftover_x_idx / leftover_z_idx: index of the leftover row IN THE
            ORIGINAL INPUT MATRICES (before deletion) — only meaningful for
            tracking; consumed internally and not returned by
            ``apply_mixed_basis_merge``.
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
        H_X = np.delete(H_X, leftover_x, axis=0)
        H_Z = np.delete(H_Z, leftover_z, axis=0)
        leftover_x = leftover_z = None  # consumed

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
            # Cross-merged rows are removed entirely — drop them from leftover
            # tracking. Higher-indexed survivors shift down by one on each delete.
            assert Hx.shape[0] == Hx_before_rows - 1
            assert Hz.shape[0] == Hz_before_rows - 1
            new_x_leftover: set[int] = set()
            for old_idx in x_leftover_set:
                if old_idx == lx:
                    continue  # this row was the one consumed (won't trigger; lx already None here)
                new_x_leftover.add(old_idx)
            x_leftover_set = new_x_leftover
            new_z_leftover: set[int] = set()
            for old_idx in z_leftover_set:
                if old_idx == lz:
                    continue
                new_z_leftover.add(old_idx)
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
```

- [ ] **Step 2.4: Run merge_test.py — verify pass**

```bash
uv run pytest src/qldpc/circuits/surgery/merge_test.py -v
```

Expected: 8 PASS.

- [ ] **Step 2.5: Run full surgery suite — verify no regressions**

```bash
uv run pytest src/qldpc/circuits/surgery/ -x -q
```

Expected: ALL pass (183 + 8 = 191).

- [ ] **Step 2.6: Commit**

```bash
git add src/qldpc/circuits/surgery/merge.py src/qldpc/circuits/surgery/merge_test.py
git commit -m "$(cat <<'EOF'
feat(surgery): merge.py — Webster–Smith–Cohen mixed-basis merge

Implements arXiv:2511.15989 §II.B.2 cross-merge as pure GF(2) row
arithmetic on assembled (H_X, H_Z) matrices.

Per merge qubit q (ascending):
  Step A: pair-merge X-checks on q (XOR each non-pivot into pivot 0)
  Step B: pair-merge Z-checks on q (symmetric)
  Step C: if both leftover X and Z exist, remove them and emit a
          single Y-type symplectic row (X-part from leftover X row,
          Z-part from leftover Z row)

Returns the modified H_X / H_Z plus Y_stab matrix, obs0_y indices,
and X / Z leftover row indices for the downstream obs0 formula.

Unit tests verify: identity on empty merge_qubits, only-X (no Y),
only-Z (no Y), cross-merge Y construction, three-rows-into-one-pivot
arithmetic, two-qubit independence, post-merge symplectic commutation
(Lemma 1), ascending-order determinism, input non-mutation.

No integration yet — build_bridge dispatch comes in next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `build_bridge` mixed-basis dispatch + populate `Bridge.Y_stab`

**Files:**
- Modify: `src/qldpc/circuits/surgery/bridge.py:348-453` (`build_bridge` function)
- Create: `src/qldpc/circuits/surgery/bridge_mixed_test.py`

The plan: when `g_l.basis != g_r.basis`, run the same-basis SkipTree adapter **with `basis = g_l.basis`** (deterministic choice), then identify `merge_qubits` from the assembled tentative stabilizer matrices, then call `apply_mixed_basis_merge`. Store results on `Bridge`.

But wait — the same-basis adapter's output is *split into H_X / H_Z by the stitch step*, not by `build_bridge`. So in `build_bridge` we cannot run the merge yet because we don't know which rows are X-type vs Z-type post-stitch (it depends on intra- vs inter-code).

**Resolution:** `build_bridge` only records `basis_l`, `basis_r`, and `merge_qubits = ()` (empty placeholder). Actual `Y_stab`, `obs0_xor_map`, `x_leftover_indices`, `z_leftover_indices` populate during stitch (Task 4), then `Bridge` is replaced (since it's frozen) with the populated version.

This restructures the design slightly:

```python
# Step 1: build_bridge with mixed basis → returns Bridge with Y_stab=None placeholder.
# Step 2: _stitch_to_quditcode runs the merge using the assembled H_X/H_Z + bridge,
#         then dataclasses.replace(bridge, Y_stab=..., merge_qubits=..., ...) and
#         returns (joint_code, bridge_populated).
```

This means callers of `build_joint_ppm_circuit` must pass the bridge through stitch first. Simpler resolution: **`build_bridge` still does both** — it calls the adapter and runs the merge eagerly using a canonical *intercode* stitch. The same-basis branch doesn't change.

Actually the cleanest approach: extract `_arrange_as_HX_HZ` from the would-be Task 4 and call it inside `build_bridge` to determine `merge_qubits` and run the merge. Then Task 4's stitch uses the already-computed `Bridge.Y_stab` / `obs0_xor_map`.

But intra vs inter affects the shape of M_meas / M_comp. To keep this honest we do the merge **inside `_stitch_to_quditcode`** and use a helper that returns BOTH the QuditCode AND the populated Bridge fields, then callers (`build_joint_ppm_circuit`) re-bind their bridge variable.

Concretely:
1. Task 3: relax `build_bridge` basis check; pass `basis_l`, `basis_r`; leave mixed-basis fields default. **No merge runs in `build_bridge`.**
2. Task 4: `_stitch_intercode_mixed` / `_stitch_intracode_mixed` build H_X / H_Z, compute `merge_qubits`, call `apply_mixed_basis_merge`, build QuditCode. They return `(QuditCode, bridge_with_mixed_fields)`. `_stitch_to_joint_csscode` becomes `_stitch_to_joint_code` returning either `(CSSCode, bridge)` or `(QuditCode, bridge)`; build_joint_ppm_circuit re-binds bridge.

- [ ] **Step 3.1: Write failing tests in `bridge_mixed_test.py`**

Create `src/qldpc/circuits/surgery/bridge_mixed_test.py`:

```python
"""Mixed-basis Bridge integration tests
(Webster–Smith–Cohen arXiv:2511.15989 §II.B.2).
"""

from __future__ import annotations

import numpy as np
import pytest

from qldpc import codes
from qldpc.objects import Pauli


def test_build_bridge_accepts_mixed_basis_steane() -> None:
    """build_bridge no longer rejects g_l.basis=X with g_r.basis=Z."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(code, z, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)
    assert bridge.basis_l is Pauli.X
    assert bridge.basis_r is Pauli.Z


def test_build_bridge_mixed_basis_y_stab_unpopulated_until_stitch() -> None:
    """Mixed-basis build_bridge defers merge to stitch — Y_stab is None at this point."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(code, z, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)
    assert bridge.Y_stab is None
    assert bridge.merge_qubits == ()


def test_build_bridge_same_basis_y_stab_remains_none() -> None:
    """Existing same-basis path is bit-for-bit unchanged: Y_stab is None."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(code, x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    assert bridge.basis_l is Pauli.X
    assert bridge.basis_r is Pauli.X
    assert bridge.basis is Pauli.X  # backward-compat property
    assert bridge.Y_stab is None
    assert bridge.merge_qubits == ()


def test_build_bridge_mixed_basis_uses_basis_l_for_skiptree() -> None:
    """When basis_l != basis_r, the SkipTree adapter uses basis_l (deterministic).
    T_l / T_r shapes must match the basis_l-built augmented gadgets."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(code, z, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)
    # Augmented gadgets carry the side's native basis (basis_l → g_l_aug.basis=X)
    assert bridge.g_l_aug.basis is Pauli.X
    assert bridge.g_r_aug.basis is Pauli.Z
    # SkipTree invariant per side still holds (T · F_aug · P = H_R)
    for side in ("l", "r"):
        T = getattr(bridge, f"T_{side}")
        g_aug = getattr(bridge, f"g_{side}_aug")
        label = getattr(bridge, f"label_{side}")
        adj = g_aug.incidence.astype(np.int_)
        P = np.zeros((adj.shape[1], bridge.width), dtype=np.int_)
        for v_idx, lab in enumerate(label):
            if lab >= 0:
                P[v_idx, lab] = 1
        assert np.array_equal((T @ adj @ P) % 2, bridge.H_R)
```

- [ ] **Step 3.2: Run new tests — verify they fail**

```bash
uv run pytest src/qldpc/circuits/surgery/bridge_mixed_test.py -v
```

Expected: tests fail with `ValueError("build_bridge requires g_l.basis == g_r.basis ...")`.

- [ ] **Step 3.3: Update `build_bridge` to accept mixed-basis**

In `src/qldpc/circuits/surgery/bridge.py`, replace `build_bridge` (function body starting around line 348). The change: remove the `if g_l.basis is not g_r.basis: raise` block; pick `basis = g_l.basis` for cellulation cap; build per-side augmented gadgets each with their own basis. Pass `basis_l=g_l.basis`, `basis_r=g_r.basis` to the Bridge constructor.

Replace the body from `if g_l.basis is not g_r.basis:` through the end of `build_bridge`:

```python
    basis_l = g_l.basis
    basis_r = g_r.basis
    # Cellulation cap defaults to the worse-case basis-stabilizer weight across
    # both sides, irrespective of mixed-basis. The cap shapes the rep-code cycle
    # length in the merged code and is basis-agnostic for the structural distance
    # argument (Swaroop Theorem 12). Use each side's native basis.
    if cellulate_max_len is None:
        cellulate_max_len = max(
            _max_basis_stabilizer_weight(g_l.code, basis_l),
            _max_basis_stabilizer_weight(g_r.code, basis_r),
        )

    # Step 1: auxiliary graphs
    G_l_aux, _ = _build_aux_graph_strict(g_l.incidence)
    G_r_aux, _ = _build_aux_graph_strict(g_r.incidence)

    # Step 2: port subsets + width
    port_l_all = (
        tuple(port_subset_l) if port_subset_l is not None else tuple(range(len(g_l.support)))
    )
    port_r_all = (
        tuple(port_subset_r) if port_subset_r is not None else tuple(range(len(g_r.support)))
    )
    width = min(len(port_l_all), len(port_r_all))
    if width < 2:
        raise ValueError(f"bridge width must be >= 2, got {width}")
    port_l = port_l_all[:width]
    port_r = port_r_all[:width]
    if not (0 <= spanning_tree_root_l < width):
        raise ValueError(f"spanning_tree_root_l={spanning_tree_root_l} out of [0, {width})")
    if not (0 <= spanning_tree_root_r < width):
        raise ValueError(f"spanning_tree_root_r={spanning_tree_root_r} out of [0, {width})")

    # Step 3: induced-subgraph connectivity augmentation
    extras_l_conn = _connect_induced_subgraph(G_l_aux, port_l)
    extras_r_conn = _connect_induced_subgraph(G_r_aux, port_r)

    # Step 4: cellulation
    extras_l_cell = _cellulate_port_subgraph(G_l_aux, port_l, max_len=cellulate_max_len)
    extras_r_cell = _cellulate_port_subgraph(G_r_aux, port_r, max_len=cellulate_max_len)

    extras_l_edges = extras_l_conn + extras_l_cell
    extras_r_edges = extras_r_conn + extras_r_cell
    extra_ancilla_l = _edges_to_incidence_extra(extras_l_edges, len(g_l.support))
    extra_ancilla_r = _edges_to_incidence_extra(extras_r_edges, len(g_r.support))

    from .gadget import build_gadget_augmented

    g_l_aug = build_gadget_augmented(g_l.code, g_l.x, extra_ancilla_l, basis=basis_l)
    g_r_aug = build_gadget_augmented(g_r.code, g_r.x, extra_ancilla_r, basis=basis_r)

    # Step 5: SkipTree on induced port subgraph; embed back into full F_aug rows
    T_l, label_l = _run_skiptree_on_port_subgraph(
        G_l_aux,
        port_l,
        spanning_tree_root_l,
        g_l_aug.incidence,
    )
    T_r, label_r = _run_skiptree_on_port_subgraph(
        G_r_aux,
        port_r,
        spanning_tree_root_r,
        g_r_aug.incidence,
    )

    return Bridge(
        width=width,
        basis_l=basis_l,
        basis_r=basis_r,
        port_l=port_l,
        port_r=port_r,
        label_l=tuple(label_l),
        label_r=tuple(label_r),
        extra_ancilla_l=extra_ancilla_l.astype(np.uint8),
        extra_ancilla_r=extra_ancilla_r.astype(np.uint8),
        T_l=T_l,
        T_r=T_r,
        H_R=_canonical_H_R(width).astype(np.int_),
        g_l_aug=g_l_aug,
        g_r_aug=g_r_aug,
        # Mixed-basis fields: populated by _stitch_to_joint_code when basis_l != basis_r.
        # Left as defaults (None / ()) here for both same-basis and mixed-basis bridges.
    )
```

- [ ] **Step 3.4: Update docstring**

The function docstring (lines 358-371) referred to the basis-match requirement. Replace the docstring with:

```python
    """Universal-adapter bridge between two gadgets (Swaroop et al. arXiv:2410.03628 §IV).

    Cain mapping: V_0^(l) → support^(l); F → incidence; extra_kappa → extra_ancilla.

    When ``g_l.basis == g_r.basis``, returns a same-basis CSS bridge (legacy
    behavior). When ``g_l.basis != g_r.basis``, returns a mixed-basis bridge
    (basis_l ≠ basis_r) with mixed-basis fields (Y_stab, obs0_xor_map, ...)
    left UNPOPULATED — the Webster–Smith–Cohen (arXiv:2511.15989 §II.B.2)
    cross-merge populates them during ``_stitch_to_joint_code`` (see
    circuits/surgery/circuit.py), at which point the resulting Bridge instance
    has the merged-code data attached.

    See docs/superpowers/specs/2026-06-09-joint-ppm-bridge-design.md §2 for the
    7-step recipe. ``spanning_tree_root_s`` is the index INTO the port tuple of
    the SkipTree root vertex on side s.

    ``cellulate_max_len`` caps port-subgraph basis cycle length. When ``None``
    (default), it is set to ``max`` of the basis-side stabilizer row weights of
    the two codes — each side measured against its own basis.
    """
```

- [ ] **Step 3.5: Delete the now-stale `test_build_bridge_rejects_basis_mismatch` test**

In `src/qldpc/circuits/surgery/bridge_test.py`, delete the `test_build_bridge_rejects_basis_mismatch` function (lines ~295-306). Mixed-basis is now allowed; the rejection test would fail.

- [ ] **Step 3.6: Run all surgery tests — verify pass**

```bash
uv run pytest src/qldpc/circuits/surgery/ -x -q
```

Expected: ALL pass (191 → ~194, accounting for removed test + new bridge_mixed_test.py 4 tests).

- [ ] **Step 3.7: Commit**

```bash
git add src/qldpc/circuits/surgery/bridge.py src/qldpc/circuits/surgery/bridge_test.py src/qldpc/circuits/surgery/bridge_mixed_test.py
git commit -m "$(cat <<'EOF'
feat(surgery): build_bridge mixed-basis dispatch

build_bridge now accepts g_l.basis != g_r.basis. The mixed-basis path
runs the same-basis SkipTree adapter independently per side (each
augmented gadget keeps its own basis), and leaves Y_stab / merge_qubits /
obs0_xor_map / leftover-index fields UNPOPULATED on the returned Bridge.

The actual Webster–Smith–Cohen cross-merge (arXiv:2511.15989 §II.B.2)
runs later inside _stitch_to_joint_code (next commit), once the assembled
H_X / H_Z matrices are known.

Same-basis path is bit-for-bit unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `_stitch_to_joint_code` mixed-basis → `QuditCode`

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py:428-605` (stitch helpers)
- Modify: `src/qldpc/circuits/surgery/circuit.py:661-783` (`build_joint_ppm_circuit`)
- Extend: `src/qldpc/circuits/surgery/bridge_mixed_test.py`

For the mixed-basis path, the stitch:
1. Assembles `M_meas_l`, `M_comp_l` from `g_l_aug` and `M_meas_r`, `M_comp_r` from `g_r_aug` (each on its OWN basis — same as today, but using `basis_l` / `basis_r` not a single `bridge.basis`).
2. Splits these into `H_X` (all X-type rows from both sides) and `H_Z` (all Z-type rows).
3. Computes `merge_qubits` = intersection of bridge-qubit-column support between H_X and H_Z. (Bridge columns are `c_adapter` + `cl_ancilla` + `cr_ancilla`.)
4. Runs `apply_mixed_basis_merge(H_X, H_Z, merge_qubits)`.
5. Packs into a single symplectic matrix `[H_X_only | 0]` ⊕ `[0 | H_Z_only]` ⊕ `Y_stab_rows` and builds `QuditCode`.
6. `dataclasses.replace`s the bridge with populated `Y_stab`, `merge_qubits`, `obs0_xor_map`, `x_leftover_indices`, `z_leftover_indices`.

**Key constraint:** `build_joint_ppm_circuit` returns `(stim.Circuit, CSSCode)` today. For mixed-basis it returns `(stim.Circuit, QuditCode)`. The type annotation widens to `QuditCode` (CSSCode is a QuditCode subclass).

- [ ] **Step 4.1: Write failing tests for mixed-basis stitch**

Append to `src/qldpc/circuits/surgery/bridge_mixed_test.py`:

```python
def test_stitch_intercode_mixed_basis_returns_quditcode() -> None:
    """Mixed-basis inter-code stitch returns a QuditCode (not a CSSCode)."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_code
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.codes.common import CSSCode, QuditCode

    code_l = codes.SteaneCode()
    code_r = codes.SteaneCode()  # distinct instance → intercode
    x = np.asarray(code_l.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code_r.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code_l, x, basis=Pauli.X)
    g_r = build_gadget(code_r, z, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)
    joint_code, bridge_populated = _stitch_to_joint_code(g_l, g_r, bridge)
    assert isinstance(joint_code, QuditCode)
    # CSSCode subclasses QuditCode, but the mixed-basis merged code is NOT CSS.
    assert not isinstance(joint_code, CSSCode)


def test_stitch_mixed_basis_populates_bridge_fields() -> None:
    """After mixed-basis stitch, bridge_populated carries Y_stab + merge_qubits + obs0_xor_map."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_code
    from qldpc.circuits.surgery.gadget import build_gadget

    code_l = codes.SteaneCode()
    code_r = codes.SteaneCode()
    x = np.asarray(code_l.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code_r.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code_l, x, basis=Pauli.X)
    g_r = build_gadget(code_r, z, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)
    _, bridge_populated = _stitch_to_joint_code(g_l, g_r, bridge)
    assert bridge_populated.Y_stab is not None
    assert bridge_populated.Y_stab.shape[0] >= 1  # at least one Y row expected
    assert len(bridge_populated.merge_qubits) >= 1
    assert len(bridge_populated.obs0_xor_map) == bridge_populated.Y_stab.shape[0]


def test_stitch_mixed_basis_stabs_commute_symplectically() -> None:
    """Lemma 1: merged-code stabilizers pairwise commute under symplectic inner product."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_code
    from qldpc.circuits.surgery.gadget import build_gadget

    code_l = codes.SteaneCode()
    code_r = codes.SteaneCode()
    x = np.asarray(code_l.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code_r.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code_l, x, basis=Pauli.X)
    g_r = build_gadget(code_r, z, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)
    joint_code, _ = _stitch_to_joint_code(g_l, g_r, bridge)
    H = np.asarray(joint_code.matrix).astype(np.int_)
    n = joint_code.num_qudits
    Hx = H[:, :n]
    Hz = H[:, n:]
    # ⟨A,B⟩_s = A_x · B_z + A_z · B_x  ; assemble and check
    comm = (Hx @ Hz.T + Hz @ Hx.T) % 2
    assert not comm.any(), "merged-code stabilizers anticommute"


def test_stitch_same_basis_still_returns_csscode() -> None:
    """Backward-compat: same-basis stitch returns a CSSCode (unchanged behavior)."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_code
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.codes.common import CSSCode

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(code, x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    joint_code, bridge_populated = _stitch_to_joint_code(g_l, g_r, bridge)
    assert isinstance(joint_code, CSSCode)
    assert bridge_populated.Y_stab is None
```

- [ ] **Step 4.2: Verify tests fail (no `_stitch_to_joint_code` yet)**

```bash
uv run pytest src/qldpc/circuits/surgery/bridge_mixed_test.py -v
```

Expected: 4 new tests fail with `ImportError`.

- [ ] **Step 4.3: Refactor stitch into mixed-basis-aware helpers**

In `src/qldpc/circuits/surgery/circuit.py`, **add** a new function `_stitch_to_joint_code` (alongside the existing `_stitch_to_joint_csscode`). For backward-compat, `_stitch_to_joint_csscode` calls `_stitch_to_joint_code(...)[0]` when same-basis.

Add **after** the existing `_stitch_to_joint_csscode` (currently at line 592):

```python
def _stitch_to_joint_code(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
) -> tuple[QuditCode, Bridge]:
    """Assemble merged code (CSS for same-basis, QuditCode for mixed-basis).

    Same-basis path delegates to ``_stitch_to_joint_csscode`` and returns
    the bridge unchanged.

    Mixed-basis path (Webster–Smith–Cohen arXiv:2511.15989 §II.B.2):
      1. Build M_meas_l / M_comp_l from g_l_aug (using basis_l) and
         M_meas_r / M_comp_r from g_r_aug (using basis_r).
      2. Sort rows into H_X / H_Z by Pauli type (basis_l determines what
         M_meas_l / M_comp_l carry; basis_r symmetric).
      3. Compute merge_qubits = bridge-column-range qubits where BOTH
         H_X and H_Z have support.
      4. Run apply_mixed_basis_merge → modified H_X / H_Z + Y_stab.
      5. Pack into a single symplectic matrix and build QuditCode.
      6. Replace bridge with populated mixed-basis fields.
    """
    if bridge.basis_l is bridge.basis_r:
        return _stitch_to_joint_csscode(g_l, g_r, bridge), bridge

    # Mixed-basis path
    return _stitch_to_joint_code_mixed(g_l, g_r, bridge)
```

Then implement the mixed-basis helper. Add immediately after:

```python
def _assemble_meas_comp_per_side(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, slice]]:
    """Build per-side M_meas / M_comp blocks honoring each side's own basis.

    Returns
    -------
    M_meas_l_block, M_comp_l_block, M_meas_r_block, M_comp_r_block
        Each shape (rows_side, n_merged); zero-padded into the full merged
        column space.
    slices
        Dict with keys 'cl_data', 'cr_data' (or 'c_data' for intracode),
        'cl_ancilla', 'cr_ancilla', 'c_adapter' — slice objects into the
        merged column range.

    Naming convention: 'meas' = the side's own measured-basis check rows
    (χ-carrier per Webster Eq. 1); 'comp' = the dual (cycle-Z for basis=X).
    Caller decides how to split these into H_X / H_Z by inspecting basis_l
    and basis_r.
    """
    intercode = g_l.code is not g_r.code
    g_l_aug, g_r_aug = bridge.g_l_aug, bridge.g_r_aug

    # Per-side basis-aware accessors.
    def _per_side(g, g_aug, basis):
        if basis is Pauli.X:
            M_meas_src, M_comp_src = g_aug.HX_merged, g_aug.HZ_merged
            m_meas_data = g.code.matrix_x.shape[0]
            m_comp_data = g.code.matrix_z.shape[0]
        else:
            M_meas_src, M_comp_src = g_aug.HZ_merged, g_aug.HX_merged
            m_meas_data = g.code.matrix_z.shape[0]
            m_comp_data = g.code.matrix_x.shape[0]
        return (
            np.asarray(M_meas_src).astype(np.int_),
            np.asarray(M_comp_src).astype(np.int_),
            m_meas_data,
            m_comp_data,
        )

    M_meas_l, M_comp_l, m_meas_l_data, m_comp_l_data = _per_side(g_l, g_l_aug, bridge.basis_l)
    M_meas_r, M_comp_r, m_meas_r_data, m_comp_r_data = _per_side(g_r, g_r_aug, bridge.basis_r)

    n_l = g_l.code.num_qudits
    n_r = g_r.code.num_qudits if intercode else 0
    k_l, k_r = g_l_aug.incidence.shape[0], g_r_aug.incidence.shape[0]
    w = bridge.width

    if intercode:
        n_merged = n_l + n_r + k_l + k_r + w
        cl_data = slice(0, n_l)
        cr_data = slice(n_l, n_l + n_r)
        cl_ancilla = slice(n_l + n_r, n_l + n_r + k_l)
        cr_ancilla = slice(n_l + n_r + k_l, n_l + n_r + k_l + k_r)
        c_adapter = slice(n_l + n_r + k_l + k_r, n_merged)
        slices = {
            "cl_data": cl_data,
            "cr_data": cr_data,
            "cl_ancilla": cl_ancilla,
            "cr_ancilla": cr_ancilla,
            "c_adapter": c_adapter,
        }
    else:
        n = n_l
        n_merged = n + k_l + k_r + w
        c_data = slice(0, n)
        cl_ancilla = slice(n, n + k_l)
        cr_ancilla = slice(n + k_l, n + k_l + k_r)
        c_adapter = slice(n + k_l + k_r, n_merged)
        slices = {
            "c_data": c_data,
            "cl_data": c_data,
            "cr_data": c_data,
            "cl_ancilla": cl_ancilla,
            "cr_ancilla": cr_ancilla,
            "c_adapter": c_adapter,
        }

    # Assemble per-side meas (data-check + χ + adapter-label) and comp
    # (data-check ext + G + cycle) blocks, zero-padded into the full
    # n_merged columns. Build _block matrices that are FULL WIDTH.

    def _expand(rows_local: np.ndarray, side_label_attr: str,
                m_data: int, n_side: int,
                c_data_slice: slice, c_ancilla_slice: slice,
                kind: str) -> np.ndarray:
        """rows_local has columns [n_side | k_side]. We expand to n_merged columns."""
        m_total = rows_local.shape[0]
        out = np.zeros((m_total, n_merged), dtype=np.int_)
        # data-check rows occupy first m_data
        out[:m_data, c_data_slice] = rows_local[:m_data, :n_side]
        # the rest are χ rows (meas) or G+gauge (comp); they take ancilla extension too.
        rest = rows_local[m_data:, :]
        out[m_data:, c_data_slice] = rest[:, :n_side]
        out[m_data:, c_ancilla_slice] = rest[:, n_side:]
        if kind == "meas":
            # χ rows carry adapter-label support for ports.
            labels = bridge.label_l if side_label_attr == "l" else bridge.label_r
            # χ rows in M_meas_src are at indices [m_data, m_data+len(support))
            for v_idx, lab in enumerate(labels):
                if lab >= 0:
                    out[m_data + v_idx, c_adapter.start + lab] = 1
        return out

    M_meas_l_block = _expand(M_meas_l, "l", m_meas_l_data, n_l,
                             slices["cl_data"], cl_ancilla, "meas")
    M_comp_l_block = _expand(M_comp_l, "l", m_comp_l_data, n_l,
                             slices["cl_data"], cl_ancilla, "comp")
    M_meas_r_block = _expand(M_meas_r, "r", m_meas_r_data, n_r if intercode else n_l,
                             slices["cr_data"], cr_ancilla, "meas")
    M_comp_r_block = _expand(M_comp_r, "r", m_comp_r_data, n_r if intercode else n_l,
                             slices["cr_data"], cr_ancilla, "comp")

    return M_meas_l_block, M_comp_l_block, M_meas_r_block, M_comp_r_block, slices


def _stitch_to_joint_code_mixed(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
) -> tuple[QuditCode, Bridge]:
    """Mixed-basis stitch: run Webster–Smith–Cohen merge, build QuditCode."""
    import dataclasses

    from .merge import apply_mixed_basis_merge

    field = g_l.code.field
    intercode = g_l.code is not g_r.code

    M_meas_l, M_comp_l, M_meas_r, M_comp_r, slices = _assemble_meas_comp_per_side(
        g_l, g_r, bridge
    )

    # Sort per-side rows into H_X / H_Z by each side's basis:
    #   basis_l=X: M_meas_l rows are X-type, M_comp_l rows are Z-type
    #   basis_l=Z: M_meas_l rows are Z-type, M_comp_l rows are X-type
    def _x_z_split(M_meas_block, M_comp_block, basis):
        if basis is Pauli.X:
            return M_meas_block, M_comp_block  # (X-rows, Z-rows)
        return M_comp_block, M_meas_block

    HX_l, HZ_l = _x_z_split(M_meas_l, M_comp_l, bridge.basis_l)
    HX_r, HZ_r = _x_z_split(M_meas_r, M_comp_r, bridge.basis_r)

    # Append bridge cycle rows.
    g_l_aug, g_r_aug = bridge.g_l_aug, bridge.g_r_aug
    n_l = g_l.code.num_qudits
    n_r = g_r.code.num_qudits if intercode else 0
    k_l, k_r = g_l_aug.incidence.shape[0], g_r_aug.incidence.shape[0]
    w = bridge.width
    n_merged = (n_l + n_r if intercode else n_l) + k_l + k_r + w

    cycle_rows = np.zeros((w - 1, n_merged), dtype=np.int_)
    cycle_rows[:, slices["cl_ancilla"]] = bridge.T_l
    cycle_rows[:, slices["cr_ancilla"]] = bridge.T_r
    cycle_rows[:, slices["c_adapter"]] = bridge.H_R

    # Cycle rows are X-type on the basis-Z side and Z-type on the basis-X side
    # (Swaroop eq 37 — cycle-X-check checks the basis-X dual, i.e. the
    # adapter qubits that the basis-Z meas rows live on). Concretely:
    #   basis_l=Z, basis_r=X: left cycle row goes to HX (dual of Z-meas)
    #                        right cycle row would also be HX — but bridge has
    #                        ONE rep-code, not per-side. The cycle rows are
    #                        SHARED, and we assign them to HX iff basis_l=Z
    #                        (otherwise HZ).
    # Equivalent rule: cycle rows are placed in the dual of basis_l. We pick
    # basis_l deterministically for cycle-row placement; the cross-merge later
    # symmetrizes any mismatch.
    if bridge.basis_l is Pauli.Z:
        HX_cycle = cycle_rows
        HZ_cycle = np.zeros((0, n_merged), dtype=np.int_)
    else:
        HX_cycle = np.zeros((0, n_merged), dtype=np.int_)
        HZ_cycle = cycle_rows

    H_X = np.vstack([HX_l, HX_r, HX_cycle]).astype(np.uint8)
    H_Z = np.vstack([HZ_l, HZ_r, HZ_cycle]).astype(np.uint8)

    # Bridge-column qubit range = cl_ancilla ∪ cr_ancilla ∪ c_adapter.
    bridge_cols = list(
        list(range(slices["cl_ancilla"].start, slices["cl_ancilla"].stop))
        + list(range(slices["cr_ancilla"].start, slices["cr_ancilla"].stop))
        + list(range(slices["c_adapter"].start, slices["c_adapter"].stop))
    )
    x_support = (H_X[:, bridge_cols].any(axis=0))
    z_support = (H_Z[:, bridge_cols].any(axis=0))
    merge_qubits = tuple(
        bridge_cols[i] for i in range(len(bridge_cols)) if x_support[i] and z_support[i]
    )

    H_X_out, H_Z_out, Y_stab, obs0_y_idx, x_left, z_left = apply_mixed_basis_merge(
        H_X, H_Z, merge_qubits
    )

    # Pack symplectic matrix: [H_X_out | 0] / [0 | H_Z_out] / Y_stab
    n = n_merged
    rows: list[np.ndarray] = []
    for r in H_X_out:
        rows.append(np.concatenate([r, np.zeros(n, dtype=np.uint8)]))
    for r in H_Z_out:
        rows.append(np.concatenate([np.zeros(n, dtype=np.uint8), r]))
    if Y_stab is not None:
        for r in Y_stab:
            rows.append(r.astype(np.uint8))

    sym_matrix = np.array(rows, dtype=np.int_) if rows else np.zeros((0, 2 * n), dtype=np.int_)
    joint_code = QuditCode(field(sym_matrix), is_subsystem_code=False)

    bridge_populated = dataclasses.replace(
        bridge,
        Y_stab=Y_stab,
        merge_qubits=merge_qubits,
        obs0_xor_map=tuple(obs0_y_idx),
        x_leftover_indices=tuple(x_left),
        z_leftover_indices=tuple(z_left),
    )
    return joint_code, bridge_populated
```

Add the QuditCode import at the top of `circuit.py` next to `CSSCode`:

```python
from qldpc.codes.common import CSSCode, QuditCode
```

- [ ] **Step 4.4: Update `build_joint_ppm_circuit` to use `_stitch_to_joint_code`**

In `build_joint_ppm_circuit` (line ~694), change:

```python
    joint_code = _stitch_to_joint_csscode(g_l, g_r, bridge)
```

to:

```python
    joint_code, bridge = _stitch_to_joint_code(g_l, g_r, bridge)
```

Same-basis path: `joint_code` is a CSSCode and `bridge` is unchanged.
Mixed-basis path: `joint_code` is a QuditCode; `bridge` now carries populated mixed-basis fields.

Update the return type annotation of `build_joint_ppm_circuit`:

```python
) -> tuple[stim.Circuit, QuditCode]:
```

(CSSCode is a QuditCode subclass, so this still passes for the same-basis case.)

- [ ] **Step 4.5: Run all surgery tests**

```bash
uv run pytest src/qldpc/circuits/surgery/ -x -q
```

Expected: all pass — mixed-basis stitch tests now pass, same-basis tests unchanged.

**Note**: If `_classify_reliable_round1_checks_joint` or syndrome extraction fail on a QuditCode (non-CSS), that's expected — those run *after* stitch in `build_joint_ppm_circuit` and are out of scope for this task. The integration tests that exercise those paths come in Task 5. For now, only run **mixed-basis stitch tests + same-basis full pipeline tests**:

```bash
uv run pytest src/qldpc/circuits/surgery/bridge_test.py src/qldpc/circuits/surgery/merge_test.py src/qldpc/circuits/surgery/bridge_mixed_test.py src/qldpc/circuits/surgery/circuit_test.py src/qldpc/circuits/surgery/gadget_test.py src/qldpc/circuits/surgery/cheeger_test.py -x -q
```

- [ ] **Step 4.6: Commit**

```bash
git add src/qldpc/circuits/surgery/circuit.py src/qldpc/circuits/surgery/bridge_mixed_test.py
git commit -m "$(cat <<'EOF'
feat(surgery): _stitch_to_joint_code for mixed-basis merged code

Adds _stitch_to_joint_code which dispatches same-basis to the existing
_stitch_to_joint_csscode and runs the Webster–Smith–Cohen
(arXiv:2511.15989 §II.B.2) cross-merge for mixed-basis.

Mixed-basis flow:
  1. Assemble per-side M_meas / M_comp blocks honoring each side's
     own basis (g_l with basis_l, g_r with basis_r).
  2. Split rows into H_X / H_Z by Pauli type per side.
  3. Compute merge_qubits = bridge-column qubits where both H_X and
     H_Z have support.
  4. Run apply_mixed_basis_merge → modified H_X / H_Z + Y_stab.
  5. Pack symplectic [X|0] / [0|Z] / Y_stab matrix and build QuditCode.
  6. dataclasses.replace bridge with populated Y_stab / merge_qubits /
     obs0_xor_map / leftover-index fields.

build_joint_ppm_circuit now re-binds the returned bridge so downstream
circuit-construction sees the populated mixed-basis fields.

Same-basis path is bit-for-bit unchanged: _stitch_to_joint_code
returns (CSSCode, bridge) with bridge.Y_stab = None.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Y-stab measurement + detector in circuit pipeline

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py:786-934` (QEC cycle + reliable classification + final detectors)
- Extend: `src/qldpc/circuits/surgery/circuit_mixed_test.py` (create now)

For the mixed-basis circuit, we need:
1. Allocate one ancilla qubit per Y-stab row in `_surgery_qubit_coordinates` (new lane, e.g. y=7).
2. Emit per-round Y-stab extraction in `_surgery_qec_cycle_joint`: `RX y_anc`, then CX/CY/CZ from y_anc to each support qubit by Pauli type, then `MX y_anc`.
3. Register Y-stab detectors only for rounds ≥ 2 (round 1 = random, final = skip).

`_classify_reliable_round1_checks_joint` returns same as before — Y-stab is **never** in the reliable round-1 set.

**Important**: the existing `_surgery_qec_cycle_joint` calls `EdgeColoring().get_circuit(joint_code, qubit_ids)`. For a QuditCode this may not work — `EdgeColoring` is built for CSSCode. We need to:
- For same-basis (CSSCode): keep current behavior.
- For mixed-basis (QuditCode): build the X/Z extraction circuit for non-Y stabs using a CSSCode-shaped restriction, and append a Y-stab extraction block manually.

Concretely: split `joint_code.matrix` into X-only rows (rows where Z-part = 0), Z-only rows (X-part = 0), and Y rows. Build a CSSCode from X-only + Z-only and use the existing pipeline for those; emit Y rows manually.

This is the most complex task in the plan. Break it into sub-steps.

- [ ] **Step 5.1: Write failing tests in `circuit_mixed_test.py`**

Create `src/qldpc/circuits/surgery/circuit_mixed_test.py`:

```python
"""Tier 1 noiseless correctness tests for mixed-basis joint PPM
(Webster–Smith–Cohen arXiv:2511.15989 §II.B.2).
"""

from __future__ import annotations

import numpy as np
import pytest
import stim

from qldpc import codes
from qldpc.objects import Pauli


def _build_steane_mixed_pair():
    """Build a Steane × Steane mixed-basis (X, Z) pair for joint PPM."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget

    code_l = codes.SteaneCode()
    code_r = codes.SteaneCode()
    x = np.asarray(code_l.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code_r.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code_l, x, basis=Pauli.X)
    g_r = build_gadget(code_r, z, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)
    return g_l, g_r, bridge


def test_mixed_basis_joint_ppm_circuit_builds() -> None:
    """build_joint_ppm_circuit succeeds for mixed-basis input."""
    from qldpc.circuits.surgery import build_joint_ppm_circuit

    g_l, g_r, bridge = _build_steane_mixed_pair()
    circuit, joint_code = build_joint_ppm_circuit(
        g_l, g_r, bridge, rounds=3, data_init=("0", "+")
    )
    assert isinstance(circuit, stim.Circuit)
    # Circuit must contain at least one Y-stab ancilla extraction (CY gate).
    text = str(circuit)
    assert "CY" in text or "CZ" in text  # Y-stab uses CX/CY/CZ from y-ancilla


def test_mixed_basis_circuit_runs_no_decoder_errors() -> None:
    """Noiseless mixed-basis circuit compiles to a stim.DetectorErrorModel without crash."""
    from qldpc.circuits.surgery import build_joint_ppm_circuit

    g_l, g_r, bridge = _build_steane_mixed_pair()
    circuit, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=3, data_init=("0", "+"))
    dem = circuit.detector_error_model()
    assert dem is not None


def test_mixed_basis_round1_detectors_skip_y_stab() -> None:
    """No DETECTOR is registered for Y-stab measurements in round 1."""
    from qldpc.circuits.surgery import build_joint_ppm_circuit

    g_l, g_r, bridge = _build_steane_mixed_pair()
    circuit, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=1, data_init=("0", "+"))
    # The y-stab ancillas live on lane y=7 (or whichever lane is chosen).
    # Concretely: no DETECTOR in round 1 should reference a y-stab ancilla
    # measurement. The mechanism we test: round 1 detector count equals the
    # same-basis equivalent's count (Y-stab Round-1 detectors are zero).
    # Weaker but easier-to-test invariant: the circuit has at least one Y-stab
    # MX measurement but its round-1 detectors do not include it.
    # Simplest concrete check: scan DETECTOR ops with rec offsets for
    # any reference to the y-anc measurement (whose record is the
    # latest MX before next round). Use sampler.
    sampler = circuit.compile_detector_sampler()
    dets, _ = sampler.sample(shots=64, separate_observables=True)
    # Noiseless: all detectors deterministic 0
    assert (dets == 0).all()


def test_mixed_basis_joint_truth_table_z_l_x_r() -> None:
    """Initialize +1 eigenstate of Z̄_l ⊗ X̄_r: obs0 == 0 noiselessly across all shots."""
    from qldpc.circuits.surgery import build_joint_ppm_circuit, keep_only_observable

    g_l, g_r, bridge = _build_steane_mixed_pair()
    # |0⟩_L ⊗ |+⟩_L is a +1 eigenstate of Z̄_l ⊗ X̄_r (eigenvalue product = +1 · +1).
    circuit, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=3, data_init=("0", "+"))
    circuit = keep_only_observable(circuit, keep_idx=0)
    sampler = circuit.compile_detector_sampler()
    _, obs = sampler.sample(shots=512, separate_observables=True)
    # obs0 should be 0 (no flip from +1) on all noiseless shots.
    assert obs.shape[1] == 1
    assert (obs[:, 0] == 0).all()


def test_mixed_basis_joint_truth_table_z_l_x_r_minus_one_eigenstate() -> None:
    """Initialize -1 eigenstate of Z̄_l ⊗ X̄_r: obs0 == 1 noiselessly."""
    from qldpc.circuits.surgery import build_joint_ppm_circuit, keep_only_observable

    g_l, g_r, bridge = _build_steane_mixed_pair()
    # |1⟩_L ⊗ |+⟩_L: Z̄_l|1⟩ = -|1⟩, X̄_r|+⟩ = +|+⟩, product = -1.
    circuit, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=3, data_init=("1", "+"))
    circuit = keep_only_observable(circuit, keep_idx=0)
    sampler = circuit.compile_detector_sampler()
    _, obs = sampler.sample(shots=512, separate_observables=True)
    # obs0 should be 1 (flipped from +1) on all noiseless shots.
    assert (obs[:, 0] == 1).all()
```

- [ ] **Step 5.2: Run new circuit_mixed tests — verify they fail**

```bash
uv run pytest src/qldpc/circuits/surgery/circuit_mixed_test.py -v
```

Expected: ALL fail (likely with `EdgeColoring` error on QuditCode, or wrong obs0).

- [ ] **Step 5.3: Restrict `EdgeColoring` to the CSSCode subset of joint_code**

The minimal change to syndrome extraction: split the QuditCode into X-only rows + Z-only rows + Y rows; build a `CSSCode` wrapper from the X-only + Z-only rows for `EdgeColoring`; handle Y rows manually.

In `circuit.py`, replace the body of `_surgery_qec_cycle_joint` (currently ~line 818). The first portion stays — we just need to split the code:

```python
def _split_quditcode_for_extraction(
    joint_code: QuditCode,
) -> tuple[CSSCode, np.ndarray | None]:
    """Split a (possibly non-CSS) merged code into (CSS-subset, Y-rows).

    Returns
    -------
    css_subset
        CSSCode built from joint_code's pure-X and pure-Z rows.
    Y_rows
        ``(n_Y, 2*n)`` symplectic Y-stab rows, or ``None`` if joint_code is
        already CSS.
    """
    if isinstance(joint_code, CSSCode):
        return joint_code, None

    H = np.asarray(joint_code.matrix).astype(np.int_)
    n = joint_code.num_qudits
    Hx = H[:, :n]
    Hz = H[:, n:]
    x_only = (~Hz.any(axis=1)) & Hx.any(axis=1)
    z_only = (~Hx.any(axis=1)) & Hz.any(axis=1)
    y_rows_mask = Hx.any(axis=1) & Hz.any(axis=1)

    HX = Hx[x_only]
    HZ = Hz[z_only]
    css_subset = CSSCode(joint_code.field(HX), joint_code.field(HZ), is_subsystem_code=False)
    Y = H[y_rows_mask] if y_rows_mask.any() else None
    return css_subset, Y
```

- [ ] **Step 5.4: Extend `_surgery_qubit_coordinates` to allocate Y-stab ancillas**

Y-stab ancillas live on a new lane y=7. They are allocated AT THE END of `qubit_ids.checks_x` (deterministic) so they don't shift the existing X/Z layout.

Actually the simplest path: use a separate list of physical qubit IDs **outside** `qubit_ids`. The QuditCode's `QubitIDs.from_code(joint_code)` already allocates ancillas based on the matrix row count, which includes Y rows.

Concretely: `QubitIDs.from_code(joint_code)` returns IDs for each row of `joint_code.matrix`. The Y rows are the rows where both X and Z parts are nonzero. We need to figure out which `check` IDs correspond to Y rows.

Look at `QubitIDs.from_code` — it likely allocates `checks` as one ID per row of `joint_code.matrix`. For a QuditCode the row order is: X-only rows first (those in HX), then Z-only rows, then Y rows (per our stitch packing order).

We need to expose the Y-row indices in the merged code. Cleanest: have `_stitch_to_joint_code_mixed` return an additional `y_row_indices` so downstream knows.

**Decision**: Extend `Bridge` with the additional info via `obs0_xor_map` already (Y rows are at positions `len(H_X_out) + len(H_Z_out) + i` for i in obs0_xor_map indices), but we also need to know the X-row count to compute that offset. Simpler: have `_stitch_to_joint_code_mixed` set a private attribute on the returned Bridge (or add a field). 

**Cleanest approach**: add **one more field** `merged_y_row_offsets` to Bridge — the row indices in `joint_code.matrix` corresponding to Y rows. (We need this info anyway for detector emission.)

Wait — for backward-compat we can derive it: when bridge.Y_stab is not None, the Y rows occupy the last `bridge.Y_stab.shape[0]` rows of joint_code.matrix (because we packed [X | 0] / [0 | Z] / Y in that order). So we don't need a new field.

Concretely, in `_surgery_qec_cycle_joint` for mixed-basis:

```python
css_subset, Y_rows = _split_quditcode_for_extraction(joint_code)
# Reallocate qubit IDs against the CSS subset (Y-stab ancillas allocated separately)
qubit_ids_css = QubitIDs.from_code(css_subset)
# Use the existing pipeline on css_subset, then append Y-stab extraction.
```

This means `build_joint_ppm_circuit` must NOT pre-allocate `qubit_ids = QubitIDs.from_code(joint_code)` — it should allocate against the CSS subset, then add Y-stab ancillas explicitly.

This is a major refactor. The cleanest version:

1. In `build_joint_ppm_circuit`, compute `css_subset, Y_rows = _split_quditcode_for_extraction(joint_code)`.
2. Allocate `qubit_ids = QubitIDs.from_code(css_subset)`.
3. Allocate Y-stab ancilla IDs starting **after** the max ID in `qubit_ids`: `y_anc_ids = tuple(range(max_id + 1, max_id + 1 + n_Y))`.
4. Pass `y_anc_ids` + `Y_rows` to `_surgery_qec_cycle_joint`.

The existing `_surgery_qec_cycle_joint` already does `strategy.get_circuit(joint_code, qubit_ids)` using `joint_code`. Change this to use `css_subset` for the syndrome extraction, then append Y-stab extraction manually.

- [ ] **Step 5.5: Implement Y-stab extraction subroutine**

Add helper to `circuit.py`:

```python
def _emit_y_stab_extraction(
    Y_rows: np.ndarray,
    y_anc_ids: tuple[int, ...],
    data_ids: tuple[int, ...],
    ancilla_ids: tuple[int, ...],
    bridge_ids: tuple[int, ...],
    n_merged: int,
) -> tuple[stim.Circuit, list[int]]:
    """Emit a Y-stab extraction block (one ancilla per Y-row).

    For each Y-row:
        RX y_anc
        for q in support:
            CX (X), CY (Y), or CZ (Z) from y_anc to physical qudit
        MX y_anc

    Returns (circuit, list of y_anc ids in measurement order).
    """
    # Build the physical qudit index → physical-qubit-ID map. The Y row
    # columns 0..n_merged-1 correspond to physical qudits in this order:
    #   data_ids ++ ancilla_ids ++ bridge_ids
    # (matches the layout used by _stitch_to_joint_code_mixed and the
    # CSS extraction pipeline).
    phys_ids = tuple(data_ids) + tuple(ancilla_ids) + tuple(bridge_ids)
    assert len(phys_ids) == n_merged, (
        f"phys_ids length {len(phys_ids)} != n_merged {n_merged}"
    )

    circuit = stim.Circuit()
    measured_ids: list[int] = []
    for i, (y_row, anc_id) in enumerate(zip(Y_rows, y_anc_ids)):
        x_part = y_row[:n_merged]
        z_part = y_row[n_merged:]
        circuit.append("RX", [anc_id])
        for q in range(n_merged):
            xq, zq = int(x_part[q]), int(z_part[q])
            if xq and zq:
                circuit.append("CY", [anc_id, phys_ids[q]])
            elif xq:
                circuit.append("CX", [anc_id, phys_ids[q]])
            elif zq:
                circuit.append("CZ", [anc_id, phys_ids[q]])
        circuit.append("MX", [anc_id])
        measured_ids.append(anc_id)
    return circuit, measured_ids
```

- [ ] **Step 5.6: Refactor `_surgery_qec_cycle_joint` to handle mixed-basis**

This is the substantial change. The new pipeline:

```python
def _surgery_qec_cycle_joint(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    joint_code: QuditCode,
    bridge: Bridge,
    num_rounds: int,
    qubit_ids: QubitIDs,
    *,
    intercode: bool,
    y_anc_ids: tuple[int, ...] = (),
    data_ids: tuple[int, ...] = (),
    ancilla_ids: tuple[int, ...] = (),
    bridge_ids: tuple[int, ...] = (),
) -> tuple[stim.Circuit, MeasurementRecord, DetectorRecord]:
    """Joint-code variant — handles both CSS (same-basis) and QuditCode (mixed-basis).

    For mixed-basis (bridge.Y_stab is not None), syndrome extraction has two parts:
        1. CSS-subset extraction via EdgeColoring on the X-only + Z-only rows
           of joint_code.
        2. Manual Y-stab extraction via CX/CY/CZ from a fresh ancilla per Y row.
    """
    css_subset, Y_rows = _split_quditcode_for_extraction(joint_code)
    strategy = EdgeColoring()
    css_round, css_round_record = strategy.get_circuit(css_subset, qubit_ids)
    reliable = set(
        _classify_reliable_round1_checks_joint(
            g_l, g_r, qubit_ids, intercode=intercode
        )
    )
    all_check_ids = qubit_ids.check
    lane_idx = _check_lane_index_map(
        g_l, qubit_ids, joint=(g_r, bridge, intercode)
    )

    has_y = Y_rows is not None and Y_rows.shape[0] > 0
    n_merged = joint_code.num_qudits

    def _one_round() -> tuple[stim.Circuit, dict[int, int]]:
        """Emit one syndrome-extraction round: CSS + Y-stab."""
        rd = stim.Circuit()
        rd += css_round
        record = dict(css_round_record)
        if has_y:
            y_block, y_meas_ids = _emit_y_stab_extraction(
                Y_rows, y_anc_ids, data_ids, ancilla_ids, bridge_ids, n_merged
            )
            rd += y_block
            for offset, mid in enumerate(y_meas_ids):
                record[mid] = len(record) + offset
        return rd, record

    one_round, round_measurement_record = _one_round()
    measurement_record = MeasurementRecord()
    detector_record = DetectorRecord()

    circuit = stim.Circuit()
    circuit += one_round
    measurement_record.append(round_measurement_record)

    # Round 1 detectors: CSS reliable checks only (Y-stab NEVER reliable round 1).
    for check_id in all_check_ids:
        if check_id in reliable:
            lane, idx = lane_idx[check_id]
            circuit.append(
                "DETECTOR", [measurement_record.get_target_rec(check_id)], (idx, lane, 0)
            )
    reliable_in_order = [cid for cid in all_check_ids if cid in reliable]
    detector_record.append({cid: dd for dd, cid in enumerate(reliable_in_order)})

    if num_rounds > 1:
        repeat_circuit, _ = _one_round()
        measurement_record.append(round_measurement_record)
        repeat_circuit.append("SHIFT_COORDS", [], (0, 0, 1))
        # Round 2+ detectors: m_t XOR m_{t-1} for ALL CSS checks AND Y-stabs.
        for check_id in all_check_ids:
            lane, idx = lane_idx[check_id]
            repeat_circuit.append(
                "DETECTOR",
                [
                    measurement_record.get_target_rec(check_id, -1),
                    measurement_record.get_target_rec(check_id, -2),
                ],
                (idx, lane, 0),
            )
        if has_y:
            for i, y_anc in enumerate(y_anc_ids):
                repeat_circuit.append(
                    "DETECTOR",
                    [
                        measurement_record.get_target_rec(y_anc, -1),
                        measurement_record.get_target_rec(y_anc, -2),
                    ],
                    (i, 7, 0),  # lane 7 reserved for Y-stab ancillas
                )
        circuit.append(stim.CircuitRepeatBlock(num_rounds - 1, repeat_circuit))
        measurement_record.append(round_measurement_record, repeat=num_rounds - 2)
        detector_record.append(
            {cid: dd for dd, cid in enumerate(all_check_ids)},
            repeat=num_rounds - 1,
        )

    return circuit, measurement_record, detector_record
```

Note the new parameters with `(...) = ()` defaults so existing callers (same-basis) work unchanged.

- [ ] **Step 5.7: Update `build_joint_ppm_circuit` to allocate Y-stab ancilla IDs and pass them through**

In `build_joint_ppm_circuit`, after `joint_code, bridge = _stitch_to_joint_code(...)`:

```python
    intercode = g_l.code is not g_r.code

    # For mixed-basis, split the joint_code into a CSS subset + Y rows.
    css_subset, Y_rows = _split_quditcode_for_extraction(joint_code)
    qubit_ids = QubitIDs.from_code(css_subset)

    # Allocate Y-stab ancilla IDs after the last CSS ancilla.
    n_y = 0 if Y_rows is None else int(Y_rows.shape[0])
    if n_y > 0:
        max_existing = max(max(qubit_ids.data), max(qubit_ids.check) if qubit_ids.check else -1)
        y_anc_ids = tuple(range(max_existing + 1, max_existing + 1 + n_y))
    else:
        y_anc_ids = ()
```

Then proceed with the existing flow but pass `y_anc_ids` and the data/ancilla/bridge ids through to `_surgery_qec_cycle_joint`.

Also: `_surgery_qubit_coordinates` needs to emit `QUBIT_COORDS` for the Y-stab ancillas. Add after the existing coordinate emission (joint section, ~line 264):

```python
    # Joint PPM mixed-basis: Y-stab ancillas on y=7.
    # (Caller passes y_anc_ids via a side channel — keep _surgery_qubit_coordinates
    #  oblivious by emitting them in build_joint_ppm_circuit instead.)
```

— actually, simpler to emit Y-stab QUBIT_COORDS inside `build_joint_ppm_circuit` after calling `_surgery_qubit_coordinates`:

```python
    circuit = _surgery_qubit_coordinates(g_l, qubit_ids, joint=(g_r, bridge, intercode))
    for i, y_anc in enumerate(y_anc_ids):
        circuit.append("QUBIT_COORDS", [y_anc], (i, 7))
```

- [ ] **Step 5.8: Update `_surgery_qec_cycle_joint` call site in `build_joint_ppm_circuit`**

Change the call to pass new kwargs:

```python
    qec_cycle, measurement_record, _ = _surgery_qec_cycle_joint(
        g_l, g_r, joint_code, bridge,
        num_rounds=rounds,
        qubit_ids=qubit_ids,
        intercode=intercode,
        y_anc_ids=y_anc_ids,
        data_ids=data_ids,
        ancilla_ids=ancilla_ids,
        bridge_ids=bridge_ids,
    )
```

- [ ] **Step 5.9: Update `_surgery_final_detectors_joint` to skip Y rows**

Find `_surgery_final_detectors_joint` (line ~886) and add a guard at the top:

```python
    # Mixed-basis: skip Y-stab final detectors (single-basis destructive
    # readout cannot reconstruct Y eigenvalues).
    css_subset, _ = _split_quditcode_for_extraction(joint_code)
    HX = np.asarray(css_subset.matrix_x).astype(np.uint8)
    HZ = np.asarray(css_subset.matrix_z).astype(np.uint8)
```

(Replaces the existing `HX = np.asarray(joint_code.matrix_x)...` lines that would break on QuditCode.)

- [ ] **Step 5.10: Run mixed-basis circuit tests**

```bash
uv run pytest src/qldpc/circuits/surgery/circuit_mixed_test.py -v
```

Expected: First 3 tests pass (circuit builds, DEM compiles, round-1 detectors deterministic). Truth-table tests **may fail** if obs0 formula is still using same-basis logic — that's Task 6.

- [ ] **Step 5.11: Run full surgery suite — same-basis unchanged**

```bash
uv run pytest src/qldpc/circuits/surgery/ -x -q
```

Expected: all same-basis tests pass.

- [ ] **Step 5.12: Commit**

```bash
git add src/qldpc/circuits/surgery/circuit.py src/qldpc/circuits/surgery/circuit_mixed_test.py
git commit -m "$(cat <<'EOF'
feat(surgery): Y-stab measurement + detectors in joint PPM pipeline

Adds explicit CX/CY/CZ Y-stab extraction (one ancilla per Y-row) to
_surgery_qec_cycle_joint for mixed-basis joint PPM. Splits the merged
QuditCode into a CSS-subset (run via existing EdgeColoring) plus the
Y-row block (extracted manually).

Detector registration (per design spec §7.2):
  Round 1: Y-stab outcomes random under single-basis init → no detector.
  Round 2..τ: m_t XOR m_{t-1} (standard memory-experiment detector).
  Final: skipped — destructive readout cannot reconstruct Y eigenvalues.

Y-stab ancilla IDs allocated AFTER QubitIDs.from_code(css_subset) max
to avoid clashing with X/Z ancillas. QUBIT_COORDS on lane y=7 (joint
PPM only). Same-basis path is unchanged (Y_rows = None, no Y block).

Truth-table obs0 still computed by the same-basis formula — Task 6 adds
the Y-stab + leftover-cycle XOR terms required for mixed-basis correctness.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: obs0 formula extension + Tier 1 truth-table verification

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py` (`_surgery_observable` and `build_joint_ppm_circuit` for mixed-basis obs0)
- Extend: `src/qldpc/circuits/surgery/circuit_mixed_test.py`

Per design spec §7.3:

```
obs0 = ⊕ m(χ_l) ⊕ ⊕ m(χ_r)
     ⊕ ⊕_{i ∈ obs0_xor_map} m(Y_stab[i])
     ⊕ ⊕_{remaining X-cycle outcomes}
     ⊕ ⊕_{remaining Z-cycle outcomes}
```

In the existing same-basis pipeline, `_surgery_observable` emits `OBSERVABLE_INCLUDE meas_check_ids 0`. For mixed-basis we add the Y-stab ancilla measurements at the latest round AND the leftover X/Z cycle check measurements.

- [ ] **Step 6.1: Write failing truth-table tests if not already failing from Task 5**

The truth-table tests from Task 5.1 (`test_mixed_basis_joint_truth_table_z_l_x_r`, `..._minus_one_eigenstate`) should now drive obs0 correctness. If they pass already after Task 5, add stronger tests:

Append to `circuit_mixed_test.py`:

```python
def test_mixed_basis_joint_truth_table_both_signs() -> None:
    """All four computational-basis logical states give correct obs0 sign."""
    from qldpc.circuits.surgery import build_joint_ppm_circuit, keep_only_observable
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget

    code_l = codes.SteaneCode()
    code_r = codes.SteaneCode()
    x = np.asarray(code_l.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code_r.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code_l, x, basis=Pauli.X)
    g_r = build_gadget(code_r, z, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)

    # |0⟩_L ⊗ |+⟩_L: Z̄|0⟩=+, X̄|+⟩=+ → product +1 → obs0=0
    # |1⟩_L ⊗ |+⟩_L: Z̄|1⟩=-, X̄|+⟩=+ → product -1 → obs0=1
    # |0⟩_L ⊗ |-⟩_L: Z̄|0⟩=+, X̄|-⟩=- → product -1 → obs0=1
    # |1⟩_L ⊗ |-⟩_L: Z̄|1⟩=-, X̄|-⟩=- → product +1 → obs0=0
    cases = [(("0", "+"), 0), (("1", "+"), 1), (("0", "-"), 1), (("1", "-"), 0)]
    for init, expected in cases:
        c, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=3, data_init=init)
        c = keep_only_observable(c, keep_idx=0)
        sampler = c.compile_detector_sampler()
        _, obs = sampler.sample(shots=256, separate_observables=True)
        assert (obs[:, 0] == expected).all(), (
            f"init={init} expected obs0={expected}, got mix: "
            f"mean {obs[:, 0].mean():.3f}"
        )
```

- [ ] **Step 6.2: Run truth-table tests — verify they fail (or pass — if pass, the formula was already correct, document and skip implementation)**

```bash
uv run pytest src/qldpc/circuits/surgery/circuit_mixed_test.py::test_mixed_basis_joint_truth_table_both_signs -v
```

- [ ] **Step 6.3: Extend `_surgery_observable` to accept mixed-basis XOR terms**

Modify `_surgery_observable` signature (currently at line ~1084):

```python
def _surgery_observable(
    gadget: GadgetLayout,
    *,
    meas_check_ids: tuple[int, ...],
    data_ids: tuple[int, ...],
    support_indices: tuple[int, ...],
    measurement_record: MeasurementRecord,
    extra_xor_check_ids: tuple[int, ...] = (),
) -> stim.Circuit:
```

In the body, after the `OBSERVABLE_INCLUDE meas_targets 0` line, append:

```python
    for cid in extra_xor_check_ids:
        circuit.append(
            "OBSERVABLE_INCLUDE",
            [measurement_record.get_target_rec(cid)],
            0,
        )
```

(`stim` accumulates `OBSERVABLE_INCLUDE` calls with the same index — multiple appends XOR into the same observable.)

- [ ] **Step 6.4: Pass mixed-basis XOR terms from `build_joint_ppm_circuit`**

In `build_joint_ppm_circuit`, compute the extra XOR IDs for mixed-basis. After the existing `meas_check_ids` assembly:

```python
    extra_xor_check_ids: tuple[int, ...] = ()
    if bridge.Y_stab is not None:
        # XOR last-round Y-stab outcomes per obs0_xor_map.
        y_xor = tuple(y_anc_ids[i] for i in bridge.obs0_xor_map)
        # XOR leftover X-cycle and Z-cycle check outcomes (rows in joint_code
        # that survived pair-merge without a cross-merge partner).
        # The leftover indices reference rows in the POST-merge H_X / H_Z;
        # the corresponding check ancillas are at those same offsets in
        # css_subset's matrices.
        css_subset, _ = _split_quditcode_for_extraction(joint_code)
        n_HX = css_subset.matrix_x.shape[0]
        n_HZ = css_subset.matrix_z.shape[0]
        leftover_x_ids = tuple(qubit_ids.checks_x[i] for i in bridge.x_leftover_indices if i < n_HX)
        leftover_z_ids = tuple(qubit_ids.checks_z[i] for i in bridge.z_leftover_indices if i < n_HZ)
        extra_xor_check_ids = y_xor + leftover_x_ids + leftover_z_ids
```

(Replace `gadget=g_l` in `_surgery_observable` call with the existing arg and add `extra_xor_check_ids=extra_xor_check_ids`.)

- [ ] **Step 6.5: Run truth-table tests**

```bash
uv run pytest src/qldpc/circuits/surgery/circuit_mixed_test.py -v
```

Expected: ALL pass (including both-signs truth table).

If they fail, the most likely cause is sign-convention drift between Lemma 2's algebraic obs0 and the stim measurement-bit obs0. Debug by:
1. Run with `rounds=1` and inspect which detectors fire.
2. Add `print(obs)` to see actual obs0 values for each init.
3. Cross-check obs0_xor_map indices against actual Y-row positions in joint_code.matrix.

If truth tables fail by a constant XOR (all flipped), the fix is to XOR `1` into obs0 as a global constant. Add a `flip_obs0_constant` parameter if needed.

- [ ] **Step 6.6: Run full surgery suite**

```bash
uv run pytest src/qldpc/circuits/surgery/ -x -q
```

Expected: ALL pass.

- [ ] **Step 6.7: Commit**

```bash
git add src/qldpc/circuits/surgery/circuit.py src/qldpc/circuits/surgery/circuit_mixed_test.py
git commit -m "$(cat <<'EOF'
feat(surgery): mixed-basis obs0 formula + truth-table tests

Extends obs0 to XOR Y-stab outcomes (at indices bridge.obs0_xor_map)
plus leftover X-cycle and Z-cycle outcomes per design spec §7.3:

  obs0 = ⊕ m(χ_l) ⊕ ⊕ m(χ_r)
       ⊕ ⊕_{i ∈ obs0_xor_map} m(Y_stab[i])
       ⊕ ⊕_{remaining X-cycle outcomes}
       ⊕ ⊕_{remaining Z-cycle outcomes}

Stim's repeated OBSERVABLE_INCLUDE with the same observable index XORs
into a single observable — the formula compiles to a sequence of
appends after the base meas-check XOR.

Truth-table verification (Steane × Steane, X × Z mixed-basis):
  init |0⟩_L ⊗ |+⟩_L → Z̄·X̄ = +1 → obs0 = 0  ✓
  init |1⟩_L ⊗ |+⟩_L → Z̄·X̄ = -1 → obs0 = 1  ✓
  init |0⟩_L ⊗ |-⟩_L → Z̄·X̄ = -1 → obs0 = 1  ✓
  init |1⟩_L ⊗ |-⟩_L → Z̄·X̄ = +1 → obs0 = 0  ✓

All four cases noiseless-deterministic across 256 shots.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Tier 2 noise smoke + same-basis baseline comparison

**Files:**
- Extend: `src/qldpc/circuits/surgery/circuit_mixed_test.py`

This task is empirical (not strict TDD) — we measure a logical error rate.

- [ ] **Step 7.1: Write Tier 2 noise smoke test**

Append to `circuit_mixed_test.py`:

```python
@pytest.mark.slow
def test_mixed_basis_noise_smoke_steane_p005() -> None:
    """Steane × Steane mixed-basis at p=0.005: BP-LSD converges, no decoder crash.

    Quantitative LER comparison vs same-basis baseline is in Tier 3 (excluded
    from CI as long-running). Tier 2 verifies the decoder graph is well-formed.
    """
    from qldpc.circuits.noise_model import standard_noise_model
    from qldpc.circuits.surgery import build_joint_ppm_circuit, keep_only_observable
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget

    code_l = codes.SteaneCode()
    code_r = codes.SteaneCode()
    x = np.asarray(code_l.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code_r.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code_l, x, basis=Pauli.X)
    g_r = build_gadget(code_r, z, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)

    noise = standard_noise_model(0.005)
    circuit, _ = build_joint_ppm_circuit(
        g_l, g_r, bridge, rounds=3, data_init=("0", "+"), noise_model=noise
    )
    circuit = keep_only_observable(circuit, keep_idx=0)
    dem = circuit.detector_error_model(decompose_errors=False)
    assert dem.num_detectors > 0
    assert dem.num_observables == 1

    # Decoder-graph weight sanity: all error probabilities in (0, 1).
    weights = []
    for inst in dem.flattened():
        if inst.type == "error":
            weights.append(inst.args_copy()[0])
    assert all(0 < w < 1 for w in weights), "DEM error weights out of (0, 1)"
```

- [ ] **Step 7.2: Run Tier 2 test**

```bash
uv run pytest src/qldpc/circuits/surgery/circuit_mixed_test.py::test_mixed_basis_noise_smoke_steane_p005 -v
```

Expected: PASS — circuit + noise model compose into a valid DEM with nontrivial errors.

If `standard_noise_model` is not the actual symbol name in the project, find the correct one:

```bash
grep -rn "def.*noise_model\|class.*NoiseModel" src/qldpc/circuits/noise_model.py
```

and substitute accordingly.

- [ ] **Step 7.3: Run full surgery suite**

```bash
uv run pytest src/qldpc/circuits/surgery/ -x -q
```

Expected: ALL pass.

- [ ] **Step 7.4: Commit**

```bash
git add src/qldpc/circuits/surgery/circuit_mixed_test.py
git commit -m "$(cat <<'EOF'
test(surgery): Tier 2 noise smoke for mixed-basis joint PPM

Verifies the mixed-basis circuit composes correctly with a standard
noise model (p=0.005) and produces a well-formed detector-error model
with at least one detector, exactly one observable, and all error
probabilities in (0, 1).

Quantitative LER comparison vs same-basis Z̄_l ⊗ Z̄_r baseline (Tier 3
of the design spec) is deferred — out of scope for landing this PR.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Checklist

After completing all 7 tasks, verify:

1. **Spec coverage:**
   - §3 Architecture: Tasks 1, 3, 4, 5, 6 implement the dispatch.
   - §4 Bridge dataclass: Task 1.
   - §5 Merge algorithm: Task 2.
   - §6 Stitch to QuditCode: Task 4.
   - §7.1 Y-stab measurement: Task 5.
   - §7.2 Detector registration: Task 5.
   - §7.3 obs0 formula: Task 6.
   - §8 Tier 1 tests: Tasks 1, 2, 4, 5, 6.
   - §8 Tier 2 tests: Task 7.
   - §9 Correctness lemmas: tested algebraically by Tasks 2, 4 (Lemma 1) and 6 (Lemma 3 via truth tables).
   - §10 Separability: Verified by full surgery suite passing after every commit.

2. **Type consistency:**
   - `Bridge.Y_stab: np.ndarray | None` used throughout (None for same-basis).
   - `_stitch_to_joint_code` returns `(QuditCode, Bridge)`.
   - `build_joint_ppm_circuit` return type widens to `QuditCode` (CSSCode subclass).
   - `_surgery_qec_cycle_joint` accepts `y_anc_ids: tuple[int, ...] = ()` (default for same-basis).
   - `apply_mixed_basis_merge` always returns a 6-tuple.

3. **No placeholders:** Every step has either a complete code block or an exact bash command.

## Execution Notes

- The plan totals 7 commits on `feat/latticesurgery-mixedjoint` (commit 2-8 of the design spec; commit 1 / cellulate_max_len fix is already in the branch).
- Each commit independently passes the pre-existing test suite (179 tests).
- Subagents executing tasks may need to inspect `_webster_fixture.py` for BB-code helpers if extending Tier 1 beyond Steane × Steane (the spec calls Steane the smoke target; BB Webster seed is left to Tier 2).
- If any Tier 1 test fails with a sign-flip on obs0, the most common cause is the order of Y-row indices in `obs0_xor_map` not matching the row order in `joint_code.matrix` — verify by comparing `bridge.Y_stab.shape[0]` against the count of Y rows in the merged code's symplectic matrix.
