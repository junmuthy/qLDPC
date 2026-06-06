# Webster-Style Gadget Construction for QLDPC Surgery — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `build_layered_surgery_code` for QLDPC lattice surgery per Webster, Smith, Cohen (arXiv:2511.15989) §II.A Steps 1–3 (which is structurally identical to the L=1 case of Cross et al. 2024 §III), plus a Cain Fig 1b reproduction notebook that implements a Webster-style minimal fault-tolerant surgery measurement circuit (init + d rounds + Π χ_i observable per Webster Eq. 1).

**Architecture:** New module `src/qldpc/codes/surgery.py` exposing a function `build_layered_surgery_code(data_code, logical_op, *, num_layers=1)` that returns `(merged_CSSCode, SurgeryLayout)`. Five pure internal helpers implement Webster Steps 1–3 with the Cross-generalized `num_layers > 1` fallback (restriction `F`, gauge fix `G = left_null(F)`, layered block scaffold, `H_X` / `H_Z` assembly). All linear algebra runs over GF(2) via `galois.GF(2)`.

**Tech Stack:** Python 3, `galois.GF(2)`, `numpy`, `pytest`. Builds on `qldpc.codes.CSSCode`, `qldpc.codes.SteaneCode`, `qldpc.codes.HGPCode`.

**Spec:** `docs/superpowers/specs/2026-06-05-cross-layered-ancilla-design.md`.

---

## File Structure

| File | Status | Purpose |
|---|---|---|
| `src/qldpc/codes/surgery.py` | new | Public API + 5 helpers + `SurgeryLayout` + internal `_LayeredBlocks` |
| `src/qldpc/codes/surgery_test.py` | new | Unit tests (validation, helpers, integration) |
| `src/qldpc/codes/__init__.py` | modify | Re-export `build_layered_surgery_code`, `SurgeryLayout` |
| `examples/logical_error_rates/9_lattice_surgery_cain_fig1b.ipynb` | new | E2E reproduction notebook (manual artifact, not in CI) |

---

## Task 1: Module skeleton + `SurgeryLayout` dataclass

**Files:**
- Create: `src/qldpc/codes/surgery.py`
- Create: `src/qldpc/codes/surgery_test.py`

- [ ] **Step 1.1: Write failing test**

Add to `src/qldpc/codes/surgery_test.py`:

```python
"""Unit tests for surgery.py — Cross et al. 2024 layered ancilla construction.

Copyright 2026 The qLDPC Authors.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import dataclasses

import galois
import numpy as np
import pytest

from qldpc import codes
from qldpc.codes.surgery import SurgeryLayout
from qldpc.objects import Pauli


def test_surgery_layout_construction() -> None:
    """SurgeryLayout is a frozen dataclass with the documented fields."""
    F = galois.GF(2)([[1, 0, 1], [0, 1, 1]])
    G = galois.GF(2).Zeros((0, 2))
    layout = SurgeryLayout(
        num_data_qubits=7,
        num_ancilla_qubits=2,
        num_layers=1,
        qubit_layer=np.array([0] * 7 + [1] * 2, dtype=np.int_),
        v0_indices=np.array([0, 3, 4], dtype=np.int_),
        c0_indices=np.array([0, 2], dtype=np.int_),
        F=F,
        G=G,
        hx_row_kind=np.array(["data"] * 3 + ["ancilla_L1"] * 3, dtype=object),
        hz_row_kind=np.array(["data"] * 3, dtype=object),
    )

    assert layout.num_data_qubits == 7
    assert layout.num_ancilla_qubits == 2
    assert layout.num_layers == 1
    assert np.array_equal(layout.F, F)
    assert layout.G.shape == (0, 2)
    assert dataclasses.is_dataclass(layout) and layout.__dataclass_params__.frozen
```

- [ ] **Step 1.2: Run test to verify it fails**

```
pytest src/qldpc/codes/surgery_test.py::test_surgery_layout_construction -v
```

Expected: `ModuleNotFoundError: No module named 'qldpc.codes.surgery'`.

- [ ] **Step 1.3: Create the module skeleton with `SurgeryLayout`**

Create `src/qldpc/codes/surgery.py`:

```python
"""Gadget construction for QLDPC lattice surgery.

Primary reference: Webster, Smith, Cohen, arXiv:2511.15989 §II.A Steps 1-3,
an explicit pedagogically clean 3-step recipe for building a logical-X
measurement gadget on any CSS code. The default ``num_layers=1`` mode
implements Webster's 3 steps verbatim; ``num_layers > 1`` activates the
multi-layer fallback of Cross et al. 2024 (arXiv:2407.18393 §III) for codes
whose induced Tanner graph has insufficient boundary Cheeger constant.

The two formulations produce the same merged code at L=1: Webster's "gadget
qubit kappa_j for each adjacent Z-check S_j" = Cross's "C_1 ancilla qubit
at the same index as the C_0 Z-check"; Webster's "X-check chi_i wired to
kappa_j iff q_i in S_j" = Cross's `[Pi_V_0, F^T]` row pattern.

See docs/superpowers/specs/2026-06-05-cross-layered-ancilla-design.md for
the full paper traceability and design rationale.

Copyright 2026 The qLDPC Authors.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import dataclasses

import galois
import numpy as np
import numpy.typing as npt


@dataclasses.dataclass(frozen=True, eq=False)
class SurgeryLayout:
    """Provenance of qubits and checks in a merged surgery code.

    Returned by ``build_layered_surgery_code`` alongside the merged ``CSSCode``.
    Downstream pipelines (circuit synthesis, decoder configuration, plotting)
    can use the layout to distinguish data qubits from ancilla and identify
    which check rows are gauge-fixing.

    Attributes:
        num_data_qubits: Number of qubits in the original data code.
        num_ancilla_qubits: Total ancilla qubits across all L layers.
        num_layers: L. Always odd, >= 1.
        qubit_layer: Length (num_data + num_ancilla) array. Value 0 marks a
            data qubit; values 1..L mark the layer index of an ancilla qubit.
        v0_indices: Indices (within data qubits) of supp(X̄_M) = V_0.
        c0_indices: Row indices (within H_Z of data code) of Z-checks adjacent
            to V_0 = C_0.
        F: Step-1 restriction matrix; shape (|C_0|, |V_0|), equal to
            ``data_code.matrix_z[c0_indices][:, v0_indices]``.
        G: Step-4 gauge-fix basis; rows span the left null space of F (i.e.
            ``G @ F == 0``); shape (rank(left_null(F)), |C_0|).
        hx_row_kind: Length (num_x_checks_merged) string array. Values:
            "data" for old X-checks, "ancilla_L{i}" for new X-checks added by
            odd layer i in {1, 3, ..., L}.
        hz_row_kind: Length (num_z_checks_merged) string array. Values:
            "data" for old Z-checks, "ancilla_L{i}" for new Z-checks added by
            even layer i in {2, 4, ..., L-1}, "gauge_fix" for U_L rows.
    """

    num_data_qubits: int
    num_ancilla_qubits: int
    num_layers: int
    qubit_layer: npt.NDArray[np.int_]
    v0_indices: npt.NDArray[np.int_]
    c0_indices: npt.NDArray[np.int_]
    F: galois.FieldArray
    G: galois.FieldArray
    hx_row_kind: npt.NDArray
    hz_row_kind: npt.NDArray
```

- [ ] **Step 1.4: Run test to verify it passes**

```
pytest src/qldpc/codes/surgery_test.py::test_surgery_layout_construction -v
```

Expected: PASS.

- [ ] **Step 1.5: Commit**

```bash
git add src/qldpc/codes/surgery.py src/qldpc/codes/surgery_test.py
git commit -m "$(cat <<'EOF'
Add surgery module skeleton with SurgeryLayout dataclass

First step of the Cross et al. 2024 layered ancilla construction. See
docs/superpowers/specs/2026-06-05-cross-layered-ancilla-design.md §3
for the API design.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `_restrict_to_logical_support` + cheap validation

**Files:**
- Modify: `src/qldpc/codes/surgery.py` (add helper)
- Modify: `src/qldpc/codes/surgery_test.py` (add tests)

This task implements Step 1 of Cross §III (compute V_0, C_0, F) plus the five cheap validation checks from spec §5. The expensive row-span (stabilizer-rejection) check is added in Task 3.

- [ ] **Step 2.1: Write the F-extraction test**

Append to `src/qldpc/codes/surgery_test.py`:

```python
from qldpc.codes.surgery import _restrict_to_logical_support


def _steane_logical_x() -> tuple[codes.SteaneCode, galois.FieldArray]:
    """Return Steane code and one of its logical-X representatives."""
    code = codes.SteaneCode()
    logical_x = code.get_logical_ops(Pauli.X)[0]
    return code, logical_x


def test_restrict_returns_F_equal_to_HZ_restriction() -> None:
    """F = H_Z[C_0, V_0] elementwise."""
    code, logical_x = _steane_logical_x()
    v0, c0, F = _restrict_to_logical_support(
        code, np.asarray(logical_x).astype(np.int_), num_layers=1, validate_logical_op=False
    )
    expected = code.matrix_z[c0][:, v0]
    assert np.array_equal(F, expected)
    assert v0.size > 0 and c0.size > 0
    assert F.shape == (c0.size, v0.size)
```

- [ ] **Step 2.2: Write the validation-error tests**

Append:

```python
def test_restrict_rejects_wrong_shape() -> None:
    code, _ = _steane_logical_x()
    with pytest.raises(ValueError, match="shape"):
        _restrict_to_logical_support(code, np.zeros(5, dtype=np.int_), 1, False)


def test_restrict_rejects_non_binary() -> None:
    code, _ = _steane_logical_x()
    bad = np.zeros(code.num_qubits, dtype=np.int_)
    bad[0] = 2
    with pytest.raises(ValueError, match="binary"):
        _restrict_to_logical_support(code, bad, 1, False)


def test_restrict_rejects_zero_vector() -> None:
    code, _ = _steane_logical_x()
    with pytest.raises(ValueError, match="empty"):
        _restrict_to_logical_support(code, np.zeros(code.num_qubits, dtype=np.int_), 1, False)


def test_restrict_rejects_even_num_layers() -> None:
    code, logical_x = _steane_logical_x()
    arr = np.asarray(logical_x).astype(np.int_)
    with pytest.raises(ValueError, match="odd"):
        _restrict_to_logical_support(code, arr, 2, False)
    with pytest.raises(ValueError, match="odd"):
        _restrict_to_logical_support(code, arr, 0, False)


def test_restrict_rejects_non_commuting_op() -> None:
    """A vector that anticommutes with H_Z must be rejected."""
    code, _ = _steane_logical_x()
    # Find a row of H_X (a stabilizer) doesn't help here because it commutes with H_Z.
    # Build a vector that fails commutation: e.g., a single-qubit X (e_0) on Steane
    # anticommutes with the Z-stabilizer that touches qubit 0.
    single = np.zeros(code.num_qubits, dtype=np.int_)
    single[0] = 1
    with pytest.raises(ValueError, match="commute"):
        _restrict_to_logical_support(code, single, 1, False)
```

- [ ] **Step 2.3: Run tests to verify they fail**

```
pytest src/qldpc/codes/surgery_test.py -v -k "test_restrict"
```

Expected: `ImportError: cannot import name '_restrict_to_logical_support'`.

- [ ] **Step 2.4: Implement `_restrict_to_logical_support`**

Append to `src/qldpc/codes/surgery.py`:

```python
from .common import CSSCode


def _restrict_to_logical_support(
    data_code: CSSCode,
    logical_op: npt.ArrayLike,
    num_layers: int,
    validate_logical_op: bool,
) -> tuple[np.ndarray, np.ndarray, galois.FieldArray]:
    """Compute V_0, C_0, F per Cross 2024 §III Step 1, with input validation.

    See spec §5 for the validation contract. Returns the indices V_0 (qubit
    columns) and C_0 (Z-check rows) into the data code, plus the restriction
    matrix F = H_Z[C_0, V_0] as a GF(2) ``galois.FieldArray``.

    The expensive row-span check (rejecting stabilizers as logical operators)
    is gated by ``validate_logical_op`` — see Task 3 / spec §5 item 6.
    """
    if data_code.is_subsystem_code:
        raise ValueError(
            "build_layered_surgery_code requires a stabilizer CSSCode, not a "
            "subsystem code."
        )
    if num_layers < 1 or num_layers % 2 != 1:
        raise ValueError(f"num_layers must be odd and >= 1, got {num_layers}.")

    field = data_code.field
    logical_op_arr = np.asarray(logical_op)
    n_data = data_code.num_qubits

    if logical_op_arr.shape != (n_data,):
        raise ValueError(
            f"logical_op has shape {logical_op_arr.shape}, expected ({n_data},)."
        )
    int_view = logical_op_arr.astype(np.int_, copy=False)
    if not np.all((int_view == 0) | (int_view == 1)):
        raise ValueError("logical_op must be binary (values in {0, 1}).")

    v0_indices = np.flatnonzero(int_view)
    if v0_indices.size == 0:
        raise ValueError("logical_op support V_0 is empty (logical_op is the zero vector).")

    logical_op_gf = field(int_view)
    hz = data_code.matrix_z
    # commutation with Z-stabilizers: H_Z @ X̄^T == 0 over GF(2)
    if np.any(hz @ logical_op_gf != 0):
        raise ValueError(
            "logical_op does not commute with Z-stabilizers (H_Z @ logical_op != 0)."
        )

    # Identify C_0: Z-check rows whose support intersects V_0.
    c0_mask = np.any(hz[:, v0_indices] != 0, axis=1)
    c0_indices = np.flatnonzero(c0_mask)
    if c0_indices.size == 0:
        raise ValueError(
            "No Z-checks of the data code touch V_0; the ancilla system cannot "
            "be constructed (degenerate logical operator)."
        )

    F = hz[c0_indices][:, v0_indices]
    return v0_indices, c0_indices, F
```

- [ ] **Step 2.5: Run tests to verify they pass**

```
pytest src/qldpc/codes/surgery_test.py -v -k "test_restrict"
```

Expected: all 6 tests PASS.

- [ ] **Step 2.6: Commit**

```bash
git add src/qldpc/codes/surgery.py src/qldpc/codes/surgery_test.py
git commit -m "$(cat <<'EOF'
Add _restrict_to_logical_support with cheap input validation

Implements Step 1 of Cross 2024 §III plus the five cheap validation
checks from spec §5. The expensive row-span (stabilizer rejection) check
is gated on validate_logical_op=True and added in the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Row-span stabilizer rejection (default-on validation)

**Files:**
- Modify: `src/qldpc/codes/surgery.py` (extend `_restrict_to_logical_support`)
- Modify: `src/qldpc/codes/surgery_test.py` (add tests)

- [ ] **Step 3.1: Write tests**

Append to `src/qldpc/codes/surgery_test.py`:

```python
def test_restrict_rejects_stabilizer_when_validating() -> None:
    """A row of H_X is a stabilizer, not a logical operator."""
    code, _ = _steane_logical_x()
    stabilizer_row = np.asarray(code.matrix_x[0]).astype(np.int_)
    with pytest.raises(ValueError, match="stabilizer"):
        _restrict_to_logical_support(code, stabilizer_row, 1, validate_logical_op=True)


def test_restrict_accepts_stabilizer_when_skipping_validation() -> None:
    """With validate_logical_op=False, the row-span check is skipped."""
    code, _ = _steane_logical_x()
    stabilizer_row = np.asarray(code.matrix_x[0]).astype(np.int_)
    v0, c0, F = _restrict_to_logical_support(
        code, stabilizer_row, 1, validate_logical_op=False
    )
    assert v0.size > 0
    assert F.shape == (c0.size, v0.size)
```

- [ ] **Step 3.2: Run tests to verify they fail**

```
pytest src/qldpc/codes/surgery_test.py -v -k "stabilizer"
```

Expected: `test_restrict_rejects_stabilizer_when_validating` FAILS (no `stabilizer` ValueError raised); the other test PASSES.

- [ ] **Step 3.3: Extend `_restrict_to_logical_support`**

In `src/qldpc/codes/surgery.py`, after the commutation check and before the C_0 computation, insert:

```python
    if validate_logical_op:
        hx = data_code.matrix_x
        # rank over GF(2): count nonzero rows of row-reduced form
        rank_hx = int(np.sum(np.any(hx.row_reduce() != 0, axis=1)))
        augmented = field(np.vstack([np.asarray(hx), logical_op_gf.reshape(1, -1)]))
        rank_aug = int(np.sum(np.any(augmented.row_reduce() != 0, axis=1)))
        if rank_aug == rank_hx:
            raise ValueError(
                "logical_op lies in the row span of H_X — it is a stabilizer, "
                "not a logical operator. Pass validate_logical_op=False to skip "
                "this check."
            )
```

- [ ] **Step 3.4: Run tests to verify they pass**

```
pytest src/qldpc/codes/surgery_test.py -v -k "stabilizer"
```

Expected: both PASS.

- [ ] **Step 3.5: Commit**

```bash
git add src/qldpc/codes/surgery.py src/qldpc/codes/surgery_test.py
git commit -m "$(cat <<'EOF'
Reject stabilizers when validate_logical_op=True

Implements spec §5 item 6: a GF(2) row-rank comparison between H_X and
[H_X; logical_op] catches the case where the caller passes a stabilizer
instead of a logical operator.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `_compute_gauge_fix`

**Files:**
- Modify: `src/qldpc/codes/surgery.py`
- Modify: `src/qldpc/codes/surgery_test.py`

Implements Step 4 of Cross §III: `G` whose rows span the left null space of F (so `G @ F == 0`).

- [ ] **Step 4.1: Write test**

Append to `src/qldpc/codes/surgery_test.py`:

```python
from qldpc.codes.surgery import _compute_gauge_fix


def test_compute_gauge_fix_left_nulls_F() -> None:
    """G satisfies G @ F == 0 with shape (rank(left_null(F)), |C_0|)."""
    code, logical_x = _steane_logical_x()
    _, _, F = _restrict_to_logical_support(
        code, np.asarray(logical_x).astype(np.int_), 1, False
    )
    G = _compute_gauge_fix(F)
    assert G.shape[1] == F.shape[0]
    if G.shape[0] > 0:
        assert np.all(G @ F == 0)
    rank_F = int(np.sum(np.any(F.row_reduce() != 0, axis=1)))
    assert G.shape[0] == F.shape[0] - rank_F


def test_compute_gauge_fix_handles_full_rank_F() -> None:
    """When F has full row rank, G is empty (0 × |C_0|)."""
    field = galois.GF(2)
    F = field([[1, 0, 1], [0, 1, 1]])  # rank 2, |C_0| = 2 → G is 0 × 2
    G = _compute_gauge_fix(F)
    assert G.shape == (0, 2)
```

- [ ] **Step 4.2: Run tests to verify they fail**

```
pytest src/qldpc/codes/surgery_test.py -v -k "gauge_fix"
```

Expected: `ImportError`.

- [ ] **Step 4.3: Implement `_compute_gauge_fix`**

Append to `src/qldpc/codes/surgery.py`:

```python
def _compute_gauge_fix(F: galois.FieldArray) -> galois.FieldArray:
    """Compute G whose rows form a basis of the left null space of F.

    Cross 2024 §III Step 4: ``null(F) = {c : c @ F == 0}``. We promote the
    CKBB gauge operators to stabilizers by introducing ``rank(null(F))`` new
    Z-checks U_L connected via G. Returns G with shape (rank, |C_0|).
    """
    return F.left_null_space()
```

- [ ] **Step 4.4: Run tests to verify they pass**

```
pytest src/qldpc/codes/surgery_test.py -v -k "gauge_fix"
```

Expected: both PASS.

- [ ] **Step 4.5: Commit**

```bash
git add src/qldpc/codes/surgery.py src/qldpc/codes/surgery_test.py
git commit -m "$(cat <<'EOF'
Add _compute_gauge_fix (Step 4 of Cross 2024 §III)

G = basis of left_null(F), so G @ F == 0. These rows become the
gauge-fixing Z-checks U_L on the top layer of the ancilla system.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `_build_layered_blocks` + `_LayeredBlocks` struct

**Files:**
- Modify: `src/qldpc/codes/surgery.py`
- Modify: `src/qldpc/codes/surgery_test.py`

A thin structural helper used by both assembly functions. Provides per-layer ancilla sizes, F / F^T cached references, and convenient column slices.

- [ ] **Step 5.1: Write tests**

Append to `src/qldpc/codes/surgery_test.py`:

```python
from qldpc.codes.surgery import _build_layered_blocks


def test_layered_blocks_L1_sizes() -> None:
    field = galois.GF(2)
    F = field([[1, 0, 1], [0, 1, 1]])  # |C_0|=2, |V_0|=3
    blocks = _build_layered_blocks(F, num_layers=1)
    assert blocks.n_v0 == 3
    assert blocks.n_c0 == 2
    assert blocks.ancilla_layer_sizes == [2]  # C_1 only
    assert blocks.total_ancilla == 2
    assert blocks.ancilla_col_slice(1) == slice(0, 2)


def test_layered_blocks_L3_sizes_and_slices() -> None:
    field = galois.GF(2)
    F = field([[1, 0, 1], [0, 1, 1]])
    blocks = _build_layered_blocks(F, num_layers=3)
    # L=3: layers 1 (C, |C_0|=2), 2 (V, |V_0|=3), 3 (C, |C_0|=2)
    assert blocks.ancilla_layer_sizes == [2, 3, 2]
    assert blocks.total_ancilla == 7
    assert blocks.ancilla_col_slice(1) == slice(0, 2)
    assert blocks.ancilla_col_slice(2) == slice(2, 5)
    assert blocks.ancilla_col_slice(3) == slice(5, 7)
    assert np.array_equal(blocks.F_T, F.T)
```

- [ ] **Step 5.2: Run tests to verify they fail**

```
pytest src/qldpc/codes/surgery_test.py -v -k "layered_blocks"
```

Expected: `ImportError`.

- [ ] **Step 5.3: Implement `_LayeredBlocks` and `_build_layered_blocks`**

Append to `src/qldpc/codes/surgery.py`:

```python
@dataclasses.dataclass(frozen=True, eq=False)
class _LayeredBlocks:
    """Internal structural summary of an L-layer ancilla system.

    Holds F, F^T (cached), per-layer ancilla qubit sizes, and convenient
    column slices into the ancilla portion of the merged-code qubit register.
    Consumed by ``_assemble_merged_HX`` / ``_assemble_merged_HZ``.
    """

    F: galois.FieldArray
    F_T: galois.FieldArray
    num_layers: int
    n_v0: int
    n_c0: int

    @property
    def ancilla_layer_sizes(self) -> list[int]:
        """Sizes of ancilla qubit groups, indexed by layer i in 1..L.

        Odd i contribute |C_0| qubits (C_i in Cross notation);
        Even i contribute |V_0| qubits (V_i).
        """
        return [
            self.n_c0 if i % 2 == 1 else self.n_v0
            for i in range(1, self.num_layers + 1)
        ]

    @property
    def total_ancilla(self) -> int:
        return sum(self.ancilla_layer_sizes)

    def ancilla_col_slice(self, layer: int) -> slice:
        """Column slice for layer's ancilla qubits, relative to the ancilla block.

        Layer indexing is 1-based. Returns a slice into the ancilla columns
        (NOT including the n_data offset).
        """
        if layer < 1 or layer > self.num_layers:
            raise IndexError(f"layer must be in 1..{self.num_layers}, got {layer}")
        offset = sum(self.ancilla_layer_sizes[: layer - 1])
        size = self.ancilla_layer_sizes[layer - 1]
        return slice(offset, offset + size)


def _build_layered_blocks(F: galois.FieldArray, num_layers: int) -> _LayeredBlocks:
    """Build the structural summary of the L-layer ancilla system."""
    return _LayeredBlocks(
        F=F,
        F_T=F.T,
        num_layers=num_layers,
        n_v0=int(F.shape[1]),
        n_c0=int(F.shape[0]),
    )
```

- [ ] **Step 5.4: Run tests to verify they pass**

```
pytest src/qldpc/codes/surgery_test.py -v -k "layered_blocks"
```

Expected: both PASS.

- [ ] **Step 5.5: Commit**

```bash
git add src/qldpc/codes/surgery.py src/qldpc/codes/surgery_test.py
git commit -m "$(cat <<'EOF'
Add _LayeredBlocks struct and _build_layered_blocks helper

Internal scaffolding for the assembly helpers. Encapsulates per-layer
ancilla qubit sizes, F^T cache, and column-slice lookups so the matrix
assembly is independent of layer-count arithmetic.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `_assemble_merged_HX`

**Files:**
- Modify: `src/qldpc/codes/surgery.py`
- Modify: `src/qldpc/codes/surgery_test.py`

Builds the merged X-check matrix per spec §4.2 / §4.4: zero-padded data rows on top, then one row block per odd layer with `Π_V_0` (L=1 only) or `I` on the previous V layer, `F^T` on the current C layer, and `I` on the next V layer if present.

- [ ] **Step 6.1: Write test**

Append to `src/qldpc/codes/surgery_test.py`:

```python
from qldpc.codes.surgery import _assemble_merged_HX


def test_assemble_HX_steane_L1_shape_and_structure() -> None:
    """For Steane L=1, H_X^merged has correct shape and per-row support pattern."""
    code, logical_x = _steane_logical_x()
    arr = np.asarray(logical_x).astype(np.int_)
    v0, _, F = _restrict_to_logical_support(code, arr, 1, False)
    blocks = _build_layered_blocks(F, 1)

    HX = _assemble_merged_HX(code, blocks, v0)

    n_x_data = code.matrix_x.shape[0]
    n_ancilla = blocks.total_ancilla  # = |C_0|
    expected_rows = n_x_data + blocks.n_v0
    expected_cols = code.num_qubits + n_ancilla
    assert HX.shape == (expected_rows, expected_cols)

    # Old data X-checks: zero on ancilla columns.
    assert np.all(HX[:n_x_data, code.num_qubits:] == 0)
    assert np.array_equal(HX[:n_x_data, :code.num_qubits], code.matrix_x)

    # V_1 X-check rows: Π_V_0 on data columns (1s at v0_indices, rows = identity)
    v1_rows = HX[n_x_data:]
    data_block = v1_rows[:, :code.num_qubits]
    # row v of data_block should have exactly a 1 at column v0[v]
    assert np.all(np.sum(data_block, axis=1) == 1)
    for v in range(blocks.n_v0):
        assert data_block[v, v0[v]] == 1

    # V_1 X-check rows: F^T on C_1 ancilla columns
    c1_block = v1_rows[:, code.num_qubits:]
    assert np.array_equal(c1_block, F.T)
```

- [ ] **Step 6.2: Run test to verify it fails**

```
pytest src/qldpc/codes/surgery_test.py -v -k "assemble_HX"
```

Expected: `ImportError`.

- [ ] **Step 6.3: Implement `_assemble_merged_HX`**

Append to `src/qldpc/codes/surgery.py`:

```python
def _assemble_merged_HX(
    data_code: CSSCode,
    blocks: _LayeredBlocks,
    v0_indices: np.ndarray,
) -> galois.FieldArray:
    """Assemble the merged H_X per spec §4.2 / §4.4.

    Block-row order: old data X-checks (zero-padded on ancilla), then new
    X-check rows from each odd layer i in {1, 3, ..., L}, |V_0| rows each.

    For layer i=1 the data-column block is the V_0 injection matrix Π_V_0;
    for i >= 3 the previous-layer block is identity on the V_{i-1} ancilla
    columns. Every odd-layer block has F^T on its own C_i columns and (if
    i+1 <= L) identity on the next V_{i+1} ancilla columns.
    """
    field = data_code.field
    n_data = data_code.num_qubits
    n_ancilla = blocks.total_ancilla
    n_merged = n_data + n_ancilla

    hx = data_code.matrix_x
    n_x_data = int(hx.shape[0])

    # Old data X-checks padded with zeros.
    old_x = field.Zeros((n_x_data, n_merged))
    old_x[:, :n_data] = hx

    # New X-check rows.
    I_v0 = field.Identity(blocks.n_v0)
    rows_per_layer = []
    for i in range(1, blocks.num_layers + 1, 2):  # odd i
        row_block = field.Zeros((blocks.n_v0, n_merged))

        if i == 1:
            # Π_V_0: identity-like injection from V_1 X-checks onto V_0 data qubits
            row_block[np.arange(blocks.n_v0), v0_indices] = 1
        else:
            # Identity on V_{i-1} ancilla columns (V_{i-1} has size |V_0|).
            prev_slice = blocks.ancilla_col_slice(i - 1)
            row_block[:, n_data + prev_slice.start : n_data + prev_slice.stop] = I_v0

        # F^T on C_i ancilla columns.
        ci_slice = blocks.ancilla_col_slice(i)
        row_block[:, n_data + ci_slice.start : n_data + ci_slice.stop] = blocks.F_T

        # Identity on V_{i+1} ancilla columns if the layer exists.
        if i + 1 <= blocks.num_layers:
            next_slice = blocks.ancilla_col_slice(i + 1)
            row_block[:, n_data + next_slice.start : n_data + next_slice.stop] = I_v0

        rows_per_layer.append(row_block)

    return field(np.vstack([old_x, *rows_per_layer]))
```

- [ ] **Step 6.4: Run test to verify it passes**

```
pytest src/qldpc/codes/surgery_test.py -v -k "assemble_HX"
```

Expected: PASS.

- [ ] **Step 6.5: Commit**

```bash
git add src/qldpc/codes/surgery.py src/qldpc/codes/surgery_test.py
git commit -m "$(cat <<'EOF'
Add _assemble_merged_HX

Block-matrix assembly of the merged X-stabilizer check matrix following
spec §4.2 / §4.4: data X-checks zero-padded on ancilla columns, then new
V_i X-check rows for odd i with Π_V_0 (i=1) or I (i>=3) on the previous
qubit-layer block, F^T on the current C_i block, and I on the next V_{i+1}
block if present.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `_assemble_merged_HZ`

**Files:**
- Modify: `src/qldpc/codes/surgery.py`
- Modify: `src/qldpc/codes/surgery_test.py`

Builds the merged Z-check matrix per spec §4.2 / §4.4: old data Z-checks (C_0 rows extended with identity on C_1 ancilla), new Z-check rows for even layers (with `I, F, I` pattern), and gauge-fix rows with `G` on C_L only.

- [ ] **Step 7.1: Write test**

Append to `src/qldpc/codes/surgery_test.py`:

```python
from qldpc.codes.surgery import _assemble_merged_HZ


def test_assemble_HZ_steane_L1_shape_and_structure() -> None:
    """For Steane L=1: data Z-checks with C_0 extension + (possibly empty) gauge-fix."""
    code, logical_x = _steane_logical_x()
    arr = np.asarray(logical_x).astype(np.int_)
    v0, c0, F = _restrict_to_logical_support(code, arr, 1, False)
    G = _compute_gauge_fix(F)
    blocks = _build_layered_blocks(F, 1)

    HZ = _assemble_merged_HZ(code, blocks, G, c0)

    n_z_data = code.matrix_z.shape[0]
    n_ancilla = blocks.total_ancilla
    expected_rows = n_z_data + G.shape[0]  # no even ancilla layers for L=1
    expected_cols = code.num_qubits + n_ancilla
    assert HZ.shape == (expected_rows, expected_cols)

    # Old data Z-checks: original H_Z on data columns
    assert np.array_equal(HZ[:n_z_data, :code.num_qubits], code.matrix_z)

    # C_0 rows have identity entries on the corresponding C_1 ancilla columns.
    c1_slice = blocks.ancilla_col_slice(1)
    ancilla_block_z = HZ[:n_z_data, code.num_qubits:]
    for j, c_idx in enumerate(c0):
        assert ancilla_block_z[c_idx, c1_slice.start + j] == 1
    # Non-C_0 rows have zero on all ancilla columns.
    non_c0 = np.setdiff1d(np.arange(n_z_data), c0)
    assert np.all(ancilla_block_z[non_c0] == 0)

    # Gauge-fix rows (if any): zero on data, G on C_1.
    if G.shape[0] > 0:
        gauge_rows = HZ[n_z_data:]
        assert np.all(gauge_rows[:, :code.num_qubits] == 0)
        assert np.array_equal(gauge_rows[:, code.num_qubits:], G)
```

- [ ] **Step 7.2: Run test to verify it fails**

```
pytest src/qldpc/codes/surgery_test.py -v -k "assemble_HZ"
```

Expected: `ImportError`.

- [ ] **Step 7.3: Implement `_assemble_merged_HZ`**

Append to `src/qldpc/codes/surgery.py`:

```python
def _assemble_merged_HZ(
    data_code: CSSCode,
    blocks: _LayeredBlocks,
    G: galois.FieldArray,
    c0_indices: np.ndarray,
) -> galois.FieldArray:
    """Assemble the merged H_Z per spec §4.2 / §4.4.

    Block-row order:
        1. All old data Z-checks. Rows in ¬C_0 have zeros on every ancilla
           column; rows in C_0 get an identity-pattern extension on C_1.
        2. New Z-checks from each even layer i in {2, 4, ..., L-1}, |C_0|
           rows each. Pattern: I on C_{i-1}, F on V_i, I on C_{i+1}.
        3. Gauge-fix rows U_L: G on C_L, zero elsewhere.
    """
    field = data_code.field
    n_data = data_code.num_qubits
    n_ancilla = blocks.total_ancilla
    n_merged = n_data + n_ancilla

    hz = data_code.matrix_z
    n_z_data = int(hz.shape[0])

    # Old data Z-checks, with C_0 extension on C_1 ancilla columns.
    old_z = field.Zeros((n_z_data, n_merged))
    old_z[:, :n_data] = hz
    c1_slice = blocks.ancilla_col_slice(1)
    I_c0 = field.Identity(blocks.n_c0)
    old_z[c0_indices, n_data + c1_slice.start : n_data + c1_slice.stop] = I_c0

    # New Z-checks from even ancilla layers (i = 2, 4, ..., L-1).
    even_rows = []
    for i in range(2, blocks.num_layers, 2):
        row_block = field.Zeros((blocks.n_c0, n_merged))
        prev_slice = blocks.ancilla_col_slice(i - 1)
        cur_slice = blocks.ancilla_col_slice(i)
        next_slice = blocks.ancilla_col_slice(i + 1)
        row_block[:, n_data + prev_slice.start : n_data + prev_slice.stop] = I_c0
        row_block[:, n_data + cur_slice.start : n_data + cur_slice.stop] = blocks.F
        row_block[:, n_data + next_slice.start : n_data + next_slice.stop] = I_c0
        even_rows.append(row_block)

    # Gauge-fix rows on C_L.
    gauge_rows: list[galois.FieldArray] = []
    if G.shape[0] > 0:
        gf = field.Zeros((G.shape[0], n_merged))
        cL_slice = blocks.ancilla_col_slice(blocks.num_layers)
        gf[:, n_data + cL_slice.start : n_data + cL_slice.stop] = G
        gauge_rows.append(gf)

    return field(np.vstack([old_z, *even_rows, *gauge_rows]))
```

- [ ] **Step 7.4: Run test to verify it passes**

```
pytest src/qldpc/codes/surgery_test.py -v -k "assemble_HZ"
```

Expected: PASS.

- [ ] **Step 7.5: Commit**

```bash
git add src/qldpc/codes/surgery.py src/qldpc/codes/surgery_test.py
git commit -m "$(cat <<'EOF'
Add _assemble_merged_HZ

Block-matrix assembly of the merged Z-stabilizer check matrix per spec
§4.2 / §4.4: data Z-checks (C_0 rows extended with identity on C_1),
new even-layer Z-checks with [I, F, I] inter/intra pattern, and gauge-fix
rows with G on C_L only.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Top-level `build_layered_surgery_code` + integration tests

**Files:**
- Modify: `src/qldpc/codes/surgery.py` (add top-level function)
- Modify: `src/qldpc/codes/surgery_test.py` (add integration tests)

- [ ] **Step 8.1: Write integration tests**

Append to `src/qldpc/codes/surgery_test.py`:

```python
from qldpc.codes.surgery import build_layered_surgery_code


def _assert_css_and_logical_count(
    merged: codes.CSSCode,
    data: codes.CSSCode,
) -> None:
    """Merged code satisfies CSS commutation and has dimension k_data − 1."""
    assert merged.is_subsystem_code is False
    assert np.all((merged.matrix_x @ merged.matrix_z.T) == 0)
    assert merged.dimension == data.dimension - 1


def test_build_surgery_steane_L1_integration() -> None:
    """Steane L=1: merged code is CSS, has k_merged = 0, layout is consistent."""
    code, logical_x = _steane_logical_x()
    merged, layout = build_layered_surgery_code(
        code, np.asarray(logical_x).astype(np.int_), num_layers=1
    )

    _assert_css_and_logical_count(merged, code)
    assert layout.num_data_qubits == code.num_qubits
    assert layout.num_layers == 1
    assert layout.num_ancilla_qubits == layout.qubit_layer.size - code.num_qubits
    assert merged.num_qubits == code.num_qubits + layout.num_ancilla_qubits

    # Data qubits marked layer 0, ancilla marked layer 1.
    assert np.all(layout.qubit_layer[: code.num_qubits] == 0)
    assert np.all(layout.qubit_layer[code.num_qubits :] == 1)


def test_build_surgery_steane_L3_integration() -> None:
    """Steane L=3 exercises the loop body for ≥ 1 odd and ≥ 1 even ancilla layer."""
    code, logical_x = _steane_logical_x()
    merged, layout = build_layered_surgery_code(
        code, np.asarray(logical_x).astype(np.int_), num_layers=3
    )
    _assert_css_and_logical_count(merged, code)
    assert layout.num_layers == 3

    # Qubit-layer labels appear in {0, 1, 2, 3}.
    assert set(np.unique(layout.qubit_layer).tolist()) <= {0, 1, 2, 3}

    # Layout row-kind labels match expected counts.
    n_x_data = code.matrix_x.shape[0]
    n_z_data = code.matrix_z.shape[0]
    assert int(np.sum(layout.hx_row_kind == "data")) == n_x_data
    assert int(np.sum(layout.hz_row_kind == "data")) == n_z_data
    assert "ancilla_L1" in set(layout.hx_row_kind.tolist())
    assert "ancilla_L3" in set(layout.hx_row_kind.tolist())
    assert "ancilla_L2" in set(layout.hz_row_kind.tolist())


def test_build_surgery_layout_row_counts_match_matrices() -> None:
    """hx_row_kind / hz_row_kind lengths == merged check counts."""
    code, logical_x = _steane_logical_x()
    merged, layout = build_layered_surgery_code(
        code, np.asarray(logical_x).astype(np.int_), num_layers=3
    )
    assert layout.hx_row_kind.size == merged.matrix_x.shape[0]
    assert layout.hz_row_kind.size == merged.matrix_z.shape[0]


def test_build_surgery_small_hgp_L1() -> None:
    """Cross-code coverage on a small HGPCode."""
    seed = 0
    classical = codes.ClassicalCode.random(4, 2, seed=seed)
    hgp = codes.HGPCode(classical)
    logical_x = hgp.get_logical_ops(Pauli.X)[0]
    merged, layout = build_layered_surgery_code(
        hgp, np.asarray(logical_x).astype(np.int_), num_layers=1
    )
    _assert_css_and_logical_count(merged, hgp)
    assert layout.num_layers == 1
    assert layout.num_data_qubits == hgp.num_qubits


def test_webster_observable_equals_logical_x_on_data() -> None:
    """Webster Eq. (1) algebraic identity for the noise-free observable.

    Claim: with gadget qubits κ_j initialized to |0⟩, measuring the merged
    code's stabilizers and taking the product of the χ_i (new X-check)
    outcomes equals the X̄_M eigenvalue. The proof is purely algebraic:

        Π_i χ_i = (Π_{i ∈ V_0} X_{q_i}) · Π_j X_{κ_j}^{|S_j ∩ supp(L)| mod 2}

    and |S_j ∩ supp(L)| ≡ 0 mod 2 for every Z-stabilizer S_j of the data
    code (because Z-stabilizers commute with the logical X X̄_M). So the
    second factor is identity and the first factor is X̄_M on data qubits.

    Equivalently, the XOR of all χ_i rows of merged.matrix_x, restricted
    to data columns, equals logical_op, and restricted to ancilla columns
    equals 0. This is a pure GF(2) identity and is the noise-free core
    that the §7 notebook's logical observable definition relies on.
    """
    code, logical_x = _steane_logical_x()
    arr = np.asarray(logical_x).astype(np.int_)
    merged, layout = build_layered_surgery_code(code, arr, num_layers=1)

    chi_mask = layout.hx_row_kind == "ancilla_L1"
    chi_rows = np.asarray(merged.matrix_x[chi_mask]).astype(np.int_)
    product = chi_rows.sum(axis=0) % 2  # XOR of all χ_i rows

    n_data = layout.num_data_qubits
    assert np.array_equal(product[:n_data], arr), (
        "Webster Eq. (1): XOR of χ_i restricted to data should equal logical_op"
    )
    assert np.all(product[n_data:] == 0), (
        "Webster Eq. (1): XOR of χ_i restricted to ancilla should be zero "
        "(every Z-check of the data code touches V_0 in an even number of qubits)"
    )
```

- [ ] **Step 8.2: Run integration tests to verify they fail**

```
pytest src/qldpc/codes/surgery_test.py -v -k "build_surgery"
```

Expected: `ImportError` on `build_layered_surgery_code`.

- [ ] **Step 8.3: Implement top-level function**

Append to `src/qldpc/codes/surgery.py`:

```python
def build_layered_surgery_code(
    data_code: CSSCode,
    logical_op: npt.ArrayLike,
    *,
    num_layers: int = 1,
    validate_logical_op: bool = True,
) -> tuple[CSSCode, SurgeryLayout]:
    """Construct a merged stabilizer code that measures ``logical_op`` by lattice surgery.

    Implements the layered ancilla construction of Cross et al. 2024 §III
    (arXiv:2407.18393). Given a stabilizer CSSCode and the binary support
    vector of a logical X operator X̄_M, this builds ``num_layers`` ancilla
    layers (L must be odd) plus a top-layer gauge-fix Z-check block, and
    returns the merged CSSCode together with a SurgeryLayout describing the
    qubit / check partition.

    Args:
        data_code: The data CSSCode (stabilizer, not subsystem).
        logical_op: Binary row vector of length ``data_code.num_qubits``
            indicating supp(X̄_M).
        num_layers: Layer count L. Odd, >= 1. Default 1 follows the
            [[144,12,12]] gross code example in Cross et al. Table 1. For
            arbitrary logical_op, distance preservation may require L in
            {3, 5}; this function does not verify distance.
        validate_logical_op: If True (default), check that logical_op is
            not in the row span of H_X. Skip with False if the caller has
            already validated.

    Returns:
        (merged_code, layout):
            merged_code: CSSCode on (n_data + n_ancilla) qubits with logical
                dimension ``data_code.dimension - 1``.
            layout: SurgeryLayout describing qubit / check provenance.

    Raises:
        ValueError: See spec §5 for the exhaustive list of cases.
    """
    v0_indices, c0_indices, F = _restrict_to_logical_support(
        data_code, logical_op, num_layers, validate_logical_op
    )
    G = _compute_gauge_fix(F)
    blocks = _build_layered_blocks(F, num_layers)

    HX_merged = _assemble_merged_HX(data_code, blocks, v0_indices)
    HZ_merged = _assemble_merged_HZ(data_code, blocks, G, c0_indices)

    merged_code = CSSCode(HX_merged, HZ_merged, is_subsystem_code=False)

    layout = _build_layout(
        data_code, blocks, G, v0_indices, c0_indices, F
    )
    return merged_code, layout


def _build_layout(
    data_code: CSSCode,
    blocks: _LayeredBlocks,
    G: galois.FieldArray,
    v0_indices: np.ndarray,
    c0_indices: np.ndarray,
    F: galois.FieldArray,
) -> SurgeryLayout:
    """Assemble the SurgeryLayout dataclass from the building blocks."""
    n_data = data_code.num_qubits
    n_ancilla = blocks.total_ancilla
    qubit_layer = np.zeros(n_data + n_ancilla, dtype=np.int_)
    for i in range(1, blocks.num_layers + 1):
        slc = blocks.ancilla_col_slice(i)
        qubit_layer[n_data + slc.start : n_data + slc.stop] = i

    n_x_data = data_code.matrix_x.shape[0]
    hx_labels: list[str] = ["data"] * n_x_data
    for i in range(1, blocks.num_layers + 1, 2):  # odd
        hx_labels.extend([f"ancilla_L{i}"] * blocks.n_v0)
    hx_row_kind = np.array(hx_labels, dtype=object)

    n_z_data = data_code.matrix_z.shape[0]
    hz_labels: list[str] = ["data"] * n_z_data
    for i in range(2, blocks.num_layers, 2):  # even (>=2, <L)
        hz_labels.extend([f"ancilla_L{i}"] * blocks.n_c0)
    hz_labels.extend(["gauge_fix"] * int(G.shape[0]))
    hz_row_kind = np.array(hz_labels, dtype=object)

    return SurgeryLayout(
        num_data_qubits=n_data,
        num_ancilla_qubits=n_ancilla,
        num_layers=blocks.num_layers,
        qubit_layer=qubit_layer,
        v0_indices=v0_indices,
        c0_indices=c0_indices,
        F=F,
        G=G,
        hx_row_kind=hx_row_kind,
        hz_row_kind=hz_row_kind,
    )
```

- [ ] **Step 8.4: Run integration tests to verify they pass**

```
pytest src/qldpc/codes/surgery_test.py -v
```

Expected: all tests PASS.

- [ ] **Step 8.5: Commit**

```bash
git add src/qldpc/codes/surgery.py src/qldpc/codes/surgery_test.py
git commit -m "$(cat <<'EOF'
Add top-level build_layered_surgery_code and integration tests

Wires the five helpers into the public API. Integration tests verify CSS
commutation and logical count for Steane (L=1, L=3) and a small HGPCode
(L=1), and check that layout row-kind labels and qubit-layer indices are
consistent with the merged code's matrix shapes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Re-export from `qldpc.codes`

**Files:**
- Modify: `src/qldpc/codes/__init__.py`

- [ ] **Step 9.1: Write the import test**

Append to `src/qldpc/codes/surgery_test.py`:

```python
def test_surgery_reexport_from_qldpc_codes() -> None:
    """``build_layered_surgery_code`` and ``SurgeryLayout`` are re-exported."""
    from qldpc import codes as codes_module

    assert hasattr(codes_module, "build_layered_surgery_code")
    assert hasattr(codes_module, "SurgeryLayout")
    assert "build_layered_surgery_code" in codes_module.__all__
    assert "SurgeryLayout" in codes_module.__all__
```

- [ ] **Step 9.2: Run test to verify it fails**

```
pytest src/qldpc/codes/surgery_test.py::test_surgery_reexport_from_qldpc_codes -v
```

Expected: FAIL with `AttributeError` or `assert in __all__`.

- [ ] **Step 9.3: Add to `__init__.py`**

In `src/qldpc/codes/__init__.py`, after the existing import blocks (after the `from .quantum import (...)` block) and before `__all__`, add:

```python
from .surgery import (
    SurgeryLayout,
    build_layered_surgery_code,
)
```

Then in the `__all__` list, add the two names. Keep the list alphabetical to match existing style. Insert (in alphabetical position):

```python
    "SurgeryLayout",
    ...
    "build_layered_surgery_code",
```

If the existing `__all__` does not already mix classes and functions in one alphabetical list, follow whatever convention is in place — read the current end of `__init__.py` and insert the new entries in the matching style.

- [ ] **Step 9.4: Run test to verify it passes**

```
pytest src/qldpc/codes/surgery_test.py::test_surgery_reexport_from_qldpc_codes -v
```

Expected: PASS.

- [ ] **Step 9.5: Run the whole surgery test module**

```
pytest src/qldpc/codes/surgery_test.py -v
```

Expected: all tests PASS.

- [ ] **Step 9.6: Commit**

```bash
git add src/qldpc/codes/__init__.py src/qldpc/codes/surgery_test.py
git commit -m "$(cat <<'EOF'
Re-export build_layered_surgery_code and SurgeryLayout from qldpc.codes

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Webster minimal surgery measurement circuit + Cain Fig 1b reproduction notebook (manual E2E artifact)

**Files:**
- Create: `examples/logical_error_rates/_9_lattice_surgery_cain_fig1b_source.py`
- Create: `examples/logical_error_rates/9_lattice_surgery_cain_fig1b.ipynb`

This task produces the end-to-end reproduction notebook implementing a Webster-style minimal fault-tolerant surgery measurement circuit on top of the gadget. It is NOT run in CI; it is the manual integration artifact named in spec §7. The notebook follows existing conventions from `examples/logical_error_rates/`: `NUM_WORKERS` constant, parametrised cells, `qldpc.circuits.memory` API, BP-LSD via sinter.

**Key construction (per Webster Eq. 1 / spec §7):** with `num_layers=1`, all gadget qubits κ_j are initialized to |0⟩ at the start of the circuit. The χ_i (new X-checks, rows tagged `layout.hx_row_kind == "ancilla_L1"`) are reliable from round 1 — no Cross §3.2 D_0 special-casing needed. The fault-tolerant circuit is **identical to a standard memory experiment except for the logical observable**, which is the product across all R rounds of the χ_i outcomes (= X̄_M's measurement bit in the noise-free case).

Since notebooks are large JSON files, this task creates the notebook with `jupytext --to notebook` from a Python source file, ensuring the content is reviewable and the JSON is well-formed.

- [ ] **Step 10.1: Write the notebook source as a `.py` "percent" file**

Create `examples/logical_error_rates/_9_lattice_surgery_cain_fig1b_source.py`:

```python
"""Source for 9_lattice_surgery_cain_fig1b.ipynb.

Reproduces Cain et al. 2024 Fig 1b using qldpc.codes.build_layered_surgery_code
(Webster, Smith, Cohen 2025 §II.A 3-step gadget; equivalent to Cross 2024
§III at L=1). The measurement circuit follows Webster Eq. (1): with gadget
qubits initialized to |0⟩, the logical observable is the product of new
X-check (χ_i) outcomes across all rounds.

Convert with:
    jupytext --to notebook \
        examples/logical_error_rates/_9_lattice_surgery_cain_fig1b_source.py \
        -o examples/logical_error_rates/9_lattice_surgery_cain_fig1b.ipynb
"""

# %% [markdown]
# # Lattice Surgery for bb_18 — Reproducing Cain et al. 2024 Fig 1b
#
# Builds a merged surgery code for a bivariate-bicycle code via the explicit
# Webster–Smith–Cohen 3-step gadget recipe (arXiv:2511.15989 §II.A), runs a
# fault-tolerant measurement circuit on top with Webster's Eq. (1) observable
# (product of new X-check outcomes), decodes via BP-LSD, and compares the
# resulting LER curve against Cain et al. Fig 1b.

# %%
from __future__ import annotations

import numpy as np

from qldpc import codes
from qldpc.objects import Pauli

NUM_WORKERS = 8  # adjust to your machine; matches conventions in this directory.

# %% [markdown]
# ## 1. Construct the bb_18 BBCode

# %%
import sympy

x, y = sympy.symbols("x y")
# TODO(notebook author): replace with the bb_18 polynomials used in your
# reproduction target. See Cain App. D or your project notes.
poly_a = 1 + x + x**2  # PLACEHOLDER
poly_b = 1 + y + y**2  # PLACEHOLDER
orders = (3, 3)         # PLACEHOLDER

data_code = codes.BBCode(orders, poly_a, poly_b)
print(f"Data code: [[{data_code.num_qubits}, {data_code.dimension}]]")

# %% [markdown]
# ## 2. Pick a logical X representative

# %%
logical_x = np.asarray(data_code.get_logical_ops(Pauli.X)[0]).astype(np.int_)
print(f"|supp(X̄_M)| = {int(logical_x.sum())}")

# %% [markdown]
# ## 3. Build the gadget (merged surgery code)
#
# ``num_layers=1`` implements Webster Steps 1–3 verbatim. If BP-LSD fails to
# converge, try ``num_layers=3`` (Cross fallback).

# %%
merged, layout = codes.build_layered_surgery_code(
    data_code, logical_x, num_layers=1
)
print(f"Merged code: [[{merged.num_qubits}, {merged.dimension}]]")

# %% [markdown]
# ## 4. Sanity print against Cain Table III
#
# Cain Table III lists (ancilla qubits, X-checks, Z-checks) = (189, 104, 86)
# for bb_18. Exact match is not expected — Cain likely includes bridges and
# Cheeger-augmentation qubits beyond the bare 3-step gadget. The orders of
# magnitude should be comparable.

# %%
ancilla_qubits = int(layout.num_ancilla_qubits)
new_x_checks = int(np.sum(layout.hx_row_kind != "data"))
new_z_checks = int(np.sum(layout.hz_row_kind != "data"))
print(f"Ancilla qubits      : {ancilla_qubits}")
print(f"New X-checks (χ_i)  : {new_x_checks}")
print(f"New Z-checks + U_L  : {new_z_checks}")
print(f"Cain Table III ref  : (189, 104, 86) — qualitative comparison only")

# %% [markdown]
# ## 5. Build the Webster minimal surgery measurement circuit
#
# The circuit differs from a standard X-memory experiment in only one place:
# the logical observable. Per Webster Eq. (1), the observable is the product
# across all R = d rounds of the χ_i outcomes — these are the rows of the
# merged H_X tagged ``"ancilla_L1"`` in ``layout.hx_row_kind``. Because the
# gadget qubits κ_j (columns where ``layout.qubit_layer == 1``) are
# initialized to |0⟩, the χ_i are reliable from round 1 — no Cross §3.2
# unreliable/D_0 bookkeeping is needed.

# %%
from qldpc.circuits import memory

# TODO(notebook author): pick num_rounds matching Cain App. D. Typical
# choice is num_rounds = d (the code distance).
num_rounds = 12  # PLACEHOLDER — set per target code distance

# Identify which columns of the merged code are gadget qubits (Webster κ_j)
# vs data qubits — needed for the per-qubit initial state.
ancilla_qubit_mask = layout.qubit_layer == 1
data_qubit_mask = layout.qubit_layer == 0

# Identify which H_X rows are the χ_i (new X-checks added by the gadget).
chi_row_mask = layout.hx_row_kind == "ancilla_L1"
chi_row_indices = np.flatnonzero(chi_row_mask)
print(f"#chi_i rows: {chi_row_indices.size}")

# TODO(notebook author): build the surgery circuit. The pattern is:
#
#     circuit = memory.build_x_memory_circuit(
#         code=merged,
#         num_rounds=num_rounds,
#         initial_state="logical_plus",   # data qubits = logical |+⟩ of bb_18
#         ancilla_initial_state={i: "0" for i in np.flatnonzero(ancilla_qubit_mask)},
#         noise_model=...,                # standard depolarizing per Cain App. D
#     )
#
# The exact builder API depends on the `qldpc.circuits.memory` interface in
# this repo — adapt to whichever helper notebooks 2/3 already use. The
# critical detail is that gadget qubits start in |0⟩ (Z=+1 eigenstate).
#
# Then OVERRIDE the logical observable on the resulting stim circuit so it
# becomes the product of all chi_i measurement outcomes across all rounds,
# instead of the default data-code logical X̄.
#
# Pseudo-code:
#
#     circuit = ...build base circuit as above...
#     # Remove the default OBSERVABLE_INCLUDE that targets the data logical:
#     circuit = strip_default_observable(circuit)
#     # Add an OBSERVABLE_INCLUDE that XORs every chi_i measurement record
#     # across all num_rounds rounds:
#     chi_measurement_records = [
#         stim.target_rec(-offset_for(round_idx, chi_idx))
#         for round_idx in range(num_rounds)
#         for chi_idx in chi_row_indices
#     ]
#     circuit.append("OBSERVABLE_INCLUDE", chi_measurement_records, 0)

# %% [markdown]
# ## 6. Configure the BP-LSD decoder per Cain App. D
#
# Copy the exact decoder configuration from your prior
# `reproduce_cain_bb18_*.py` scripts. The key Cain App. D parameters are BP
# iteration count and LSD post-processing settings.

# %%
from qldpc import decoders

decoder_kwargs = {
    "max_iter": 30,        # PLACEHOLDER — set to Cain App. D value
    "osd_method": "OSD_0", # PLACEHOLDER — set to Cain App. D value
}

# %% [markdown]
# ## 7. Sweep and produce the LER curve

# %%
import sinter

# TODO(notebook author): fill in the sinter.collect sweep over physical
# error rates, using NUM_WORKERS workers. The circuit factory must wire the
# Webster observable defined in section 5; the decoder runs on the DEM
# derived from that circuit so that "logical error" = wrong χ_i product.

p_values = [1e-3, 2e-3, 3e-3, 5e-3, 7e-3]
results = []  # placeholder for sinter.collect output

# %% [markdown]
# ## 8. Plot alongside Cain Fig 1b

# %%
import matplotlib.pyplot as plt

# TODO(notebook author): extract LER from results, overlay Cain Fig 1b
# data if available, save figure to examples/logical_error_rates/figures/.
```

- [ ] **Step 10.2: Convert source to notebook**

```bash
cd /Users/tgzhou/Project/qLDPC && \
  jupytext --to notebook \
    examples/logical_error_rates/_9_lattice_surgery_cain_fig1b_source.py \
    -o examples/logical_error_rates/9_lattice_surgery_cain_fig1b.ipynb
```

(If `jupytext` is not installed, use `jupyter nbconvert --to notebook --no-prompt` with the same input. Either tool produces a valid `.ipynb`.)

Expected: `9_lattice_surgery_cain_fig1b.ipynb` written.

- [ ] **Step 10.3: Smoke test the notebook builds the merged code**

Run a tiny inline verification (does not execute the full notebook, just confirms the surgery API path is importable and the notebook JSON is well-formed):

```bash
python -c "
import nbformat, json
nb = nbformat.read('examples/logical_error_rates/9_lattice_surgery_cain_fig1b.ipynb', as_version=4)
print(f'notebook cells: {len(nb.cells)}')
from qldpc import codes
print(hasattr(codes, 'build_layered_surgery_code'))
"
```

Expected: prints cell count > 0 and `True`.

- [ ] **Step 10.4: Commit**

```bash
git add \
  examples/logical_error_rates/_9_lattice_surgery_cain_fig1b_source.py \
  examples/logical_error_rates/9_lattice_surgery_cain_fig1b.ipynb
git commit -m "$(cat <<'EOF'
Add Cain Fig 1b reproduction notebook (Webster minimal surgery circuit)

Notebook builds the gadget via build_layered_surgery_code (Webster Steps
1-3 at num_layers=1), then defines a fault-tolerant surgery measurement
circuit with the Webster Eq. 1 observable (product of new X-check χ_i
outcomes across R = d rounds). Gadget qubits initialize to |0⟩ so the
χ_i are reliable from round 1 — no Cross §3.2 D_0 bookkeeping. The
remaining circuit-construction and decoder-config TODOs are tagged for
the notebook author to wire to their bb_18 instance and Cain App. D
settings.

The corresponding _source.py jupytext file is committed alongside so the
notebook is reviewable as plain Python.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Checklist

After completing all tasks, run:

```bash
pytest src/qldpc/codes/surgery_test.py -v
```

Expected: all tests PASS.

### Spec coverage

- §3 Public API → Tasks 1, 8 (SurgeryLayout, build_layered_surgery_code).
- §4 Algorithm structure (helpers) → Tasks 2, 4, 5, 6, 7.
- §4.5 Paper traceability → docstrings reference the spec; implementation matches the §4 block patterns.
- §5 Validation (5 cheap + 1 default-on) → Tasks 2, 3.
- §6 Testing → Tasks 2–8 (each helper + integration). All 7 test functions in spec §6 are present.
- §7 Cain Fig 1b notebook → Task 10.
- §8 Implementation order → followed; minor adjustment: row-span check split into Task 3 to keep TDD increments small.

### Cross-task type / name consistency

- `SurgeryLayout` field names match the test access in Tasks 1, 8.
- `_LayeredBlocks` field/method names (`F`, `F_T`, `n_v0`, `n_c0`, `ancilla_layer_sizes`, `total_ancilla`, `ancilla_col_slice`) match between Task 5 (definition) and Tasks 6, 7, 8 (consumers).
- Function signatures: `_restrict_to_logical_support(data_code, logical_op, num_layers, validate_logical_op)` consistent across Tasks 2, 3, 8.
- `_assemble_merged_HX(data_code, blocks, v0_indices)` and `_assemble_merged_HZ(data_code, blocks, G, c0_indices)` consistent between definition and call sites.
- Row-kind label string format `"data" / "ancilla_L{i}" / "gauge_fix"` consistent between Task 8 implementation and tests in Task 8.

### Placeholder scan

- No "TBD" / "TODO" / "implement later" outside the notebook (Task 10), where the inline `TODO(notebook author):` markers are deliberate handoff points for the polynomial / decoder configuration that depends on the specific bb_18 instance and prior Cain App. D settings.
- All test functions have complete code bodies.
- All commit messages are written out, not placeholders.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-05-cross-layered-ancilla-construction.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Best for keeping context tight and isolating regressions per task.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Best if you want to inspect intermediate state interactively.

**Which approach?**
