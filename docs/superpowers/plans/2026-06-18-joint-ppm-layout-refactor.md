# Joint PPM Layout Refactor: Block-by-block Matrix Construction Following main.tex §4

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor mixed-basis joint PPM stitch + obs0 emission to construct
$\tilde H_X^{\mathrm{joint}}, \tilde H_Z^{\mathrm{joint}}, \tilde H_Y$ block-by-block per main.tex §4.2/§4.3, with explicit row provenance, so that obs0 = ⊕ m(χ_l) ⊕ ⊕ m(χ_r) ⊕ ⊕ m(y_q) directly follows Lemma 2 — unblocking the failing
`test_mixed_basis_circuit_compiles_to_dem` and the xfail truth-table test.

**Architecture:**
- New module `src/qldpc/circuits/surgery/joint_layout.py` with: (a) `JointPPMLayout` dataclass holding the three stabilizer matrices and per-row provenance dict; (b) block builders for each LaTeX block; (c) `build_joint_layout(g_l, g_r, bridge)` dispatcher that handles same-basis and mixed-basis under one interface.
- The current `_stitch_to_joint_code_mixed` is kept untouched as a fallback during transition; the new path replaces only the mixed-basis branch in `_build_joint_ppm_circuit_mixed_basis`.
- obs0 emission becomes provenance-driven: pick check IDs from rows tagged `chi_l`, `chi_r`, `y_q` in `JointPPMLayout`. Same-basis migration is explicitly OUT OF SCOPE for this plan.

**Tech Stack:** Python 3.12, NumPy, Stim, pytest, galois (existing surgery module dependencies).

---

## Background and Spec Trace

The mixed-basis joint PPM has been incrementally bolted on top of the same-basis machinery. The result: `_stitch_to_joint_code_mixed` (`circuit.py:763-940`) builds matrices by reusing `_assemble_meas_comp_per_side`, manually concatenating cycles, calling `apply_mixed_basis_merge`, then using structural heuristics (`_adapter_weight(row) >= 2`) to identify cycles vs χ. obs0 emission (`circuit.py:1596-1628`) only XORs `Y_stab[obs0_xor_map]` + empty leftover lists — missing the canonical $\prod \chi_l \cdot \prod \chi_r$ terms from Lemma 2 (design spec §9). Result: `test_mixed_basis_circuit_compiles_to_dem` fails with "non-deterministic observable L0".

The LaTeX in `docs/superpowers/docs/main.tex` §4 specifies the exact block structure for mixed-basis joint PPM. Implementing the stitch by reading that structure off the LaTeX gives explicit per-row provenance for free, eliminates the heuristics, and makes the obs0 formula direct.

Source-of-truth references:
- `docs/superpowers/docs/main.tex` §4.2 (pre-merge $\tilde H_X^{\mathrm{joint,pre}}, \tilde H_Z^{\mathrm{joint,pre}}$ block forms)
- `docs/superpowers/docs/main.tex` §4.3 (cross-merge: delete port rows, build $\tilde H_Y$)
- `docs/superpowers/docs/main.tex` §4.4-§4.5 (commutation + readout = $\prod \chi_l \cdot \prod \chi_r \cdot \prod y_q$)
- `docs/superpowers/specs/2026-06-15-mixed-basis-joint-ppm-design.md` §9 Lemma 2 (obs0 formula)

---

## File Structure

**Created files:**
- `src/qldpc/circuits/surgery/joint_layout.py` — `JointPPMLayout` dataclass + block builders + dispatcher (~280 lines)
- `src/qldpc/circuits/surgery/joint_layout_test.py` — unit tests for blocks, pre-merge, cross-merge, and end-to-end correctness (~250 lines)

**Modified files:**
- `src/qldpc/circuits/surgery/circuit.py` — `_build_joint_ppm_circuit_mixed_basis` switches to `build_joint_layout`; obs0 emission rewritten to use provenance. ~50 lines changed in two regions.
- `src/qldpc/circuits/surgery/circuit_mixed_test.py` — un-xfail `test_mixed_basis_joint_truth_table_x_l_z_r`, update docstrings reflecting Lemma 2 now closed.

**Out of scope (separate plan if desired):**
- Same-basis joint PPM migration to `joint_layout.py` (Phase 3 in earlier discussion)
- `Bridge` dataclass mixed-basis field cleanup (`Y_stab`, `obs0_xor_map`, `x_leftover_indices`, `z_leftover_indices`, `merge_qubits`)
- `_assemble_meas_comp_per_side` deprecation

---

## Task 1: `JointPPMLayout` dataclass skeleton + smoke test

**Files:**
- Create: `src/qldpc/circuits/surgery/joint_layout.py`
- Create: `src/qldpc/circuits/surgery/joint_layout_test.py`

- [ ] **Step 1: Write failing test for dataclass construction**

```python
# src/qldpc/circuits/surgery/joint_layout_test.py
"""Tests for joint_layout.py — block-by-block joint PPM construction per main.tex §4."""

from __future__ import annotations

import numpy as np

from qldpc.circuits.surgery.joint_layout import JointPPMLayout
from qldpc.objects import Pauli


def test_layout_dataclass_construction() -> None:
    """JointPPMLayout holds three stabilizer matrices + provenance dicts."""
    H_X = np.zeros((2, 10), dtype=np.uint8)
    H_Z = np.zeros((3, 10), dtype=np.uint8)
    H_Y = np.zeros((1, 20), dtype=np.uint8)
    layout = JointPPMLayout(
        H_X=H_X,
        H_Z=H_Z,
        H_Y=H_Y,
        rows_data_x={"l": (0,), "r": (1,)},
        rows_data_z={"l": (0,), "r": (1,)},
        rows_chi={"l": (), "r": ()},
        rows_gauge={"l": (), "r": ()},
        rows_cycle={"l": (), "r": (2,)},
        rows_y=(0,),
        basis_l=Pauli.Z,
        basis_r=Pauli.X,
        column_slices={
            "Q_l": slice(0, 3),
            "Q_r": slice(3, 6),
            "k_l": slice(6, 7),
            "k_r": slice(7, 8),
            "A": slice(8, 10),
        },
    )
    assert layout.H_X.shape == (2, 10)
    assert layout.H_Z.shape == (3, 10)
    assert layout.H_Y.shape == (1, 20)
    assert layout.basis_l is Pauli.Z
    assert layout.column_slices["Q_l"] == slice(0, 3)
```

- [ ] **Step 2: Run test to verify it fails (module does not exist yet)**

Run: `python -m pytest src/qldpc/circuits/surgery/joint_layout_test.py::test_layout_dataclass_construction -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qldpc.circuits.surgery.joint_layout'`.

- [ ] **Step 3: Write minimal joint_layout.py with dataclass**

```python
# src/qldpc/circuits/surgery/joint_layout.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest src/qldpc/circuits/surgery/joint_layout_test.py::test_layout_dataclass_construction -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/joint_layout.py src/qldpc/circuits/surgery/joint_layout_test.py
git commit -m "feat(surgery): JointPPMLayout dataclass skeleton

First task in the joint-PPM layout refactor (main.tex §4). The dataclass
holds the three stabilizer matrices (H_X, H_Z, H_Y) and per-row provenance
dicts so the downstream circuit builder can emit obs0 directly from row
indices instead of structural heuristics.
"
```

---

## Task 2: Column-slice helper

**Files:**
- Modify: `src/qldpc/circuits/surgery/joint_layout.py`
- Test: `src/qldpc/circuits/surgery/joint_layout_test.py`

- [ ] **Step 1: Write failing test for `column_slices_for_bridge`**

```python
# Append to src/qldpc/circuits/surgery/joint_layout_test.py

import pytest

from qldpc import codes
from qldpc.circuits.surgery.joint_layout import column_slices_for_bridge


def test_column_slices_for_steane_pair() -> None:
    """Column slices partition (Q_l | Q_r | k_l | k_r | A) for an inter-code pair."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(code, z, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)

    slices = column_slices_for_bridge(g_l, g_r, bridge)
    n_l = g_l.code.num_qudits
    n_r = g_r.code.num_qudits
    k_l = bridge.g_l_aug.incidence.shape[0]
    k_r = bridge.g_r_aug.incidence.shape[0]
    w = bridge.width

    assert slices["Q_l"] == slice(0, n_l)
    assert slices["Q_r"] == slice(n_l, n_l + n_r)
    assert slices["k_l"] == slice(n_l + n_r, n_l + n_r + k_l)
    assert slices["k_r"] == slice(n_l + n_r + k_l, n_l + n_r + k_l + k_r)
    assert slices["A"] == slice(n_l + n_r + k_l + k_r, n_l + n_r + k_l + k_r + w)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest src/qldpc/circuits/surgery/joint_layout_test.py::test_column_slices_for_steane_pair -v`
Expected: FAIL with `ImportError: cannot import name 'column_slices_for_bridge'`.

- [ ] **Step 3: Add `column_slices_for_bridge` to joint_layout.py**

```python
# Append to src/qldpc/circuits/surgery/joint_layout.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest src/qldpc/circuits/surgery/joint_layout_test.py::test_column_slices_for_steane_pair -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/joint_layout.py src/qldpc/circuits/surgery/joint_layout_test.py
git commit -m "feat(surgery): column_slices_for_bridge helper

Maps a (g_l, g_r, bridge) tuple to the (Q_l|Q_r|k_l|k_r|A) column slices
used throughout the joint_layout block builders. Intercode vs intracode
collapses Q_r to empty when g_l.code is g_r.code (shared data).
"
```

---

## Task 3: Data-row block builder

**Files:**
- Modify: `src/qldpc/circuits/surgery/joint_layout.py`
- Test: `src/qldpc/circuits/surgery/joint_layout_test.py`

- [ ] **Step 1: Write failing test for `_block_data`**

```python
# Append to src/qldpc/circuits/surgery/joint_layout_test.py

from qldpc.circuits.surgery.joint_layout import _block_data


def test_block_data_x_left_carries_f_X_on_kappa_l_for_basis_l_Z() -> None:
    """Left H_X^l data rows in mixed-basis Z⊗X must extend f_X^l = π_{C_0^l}^T into κ^l."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, z, basis=Pauli.Z)
    g_r = build_gadget(code, x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    slices = column_slices_for_bridge(g_l, g_r, bridge)
    N = slices["A"].stop

    block = _block_data(g_l, basis_block=Pauli.X, side="l", slices=slices, N=N)
    m_X_l = g_l.code.matrix_x.shape[0]
    assert block.shape == (m_X_l, N)
    # Data side equals original H_X^l on Q_l columns.
    assert (block[:, slices["Q_l"]] == np.asarray(g_l.code.matrix_x).astype(np.uint8)).all()
    # κ^l columns: f_X^l = π_{C_0^l}^T extends H_X^l rows whose index is in C_0^l.
    # C_0^l = X-checks touching V_0^l on the left's Z-gadget.
    c_0_l = list(g_l.data_checks)
    f_X_l_expected = np.zeros((m_X_l, bridge.g_l_aug.incidence.shape[0]), dtype=np.uint8)
    for k, j in enumerate(c_0_l):
        f_X_l_expected[j, k] = 1
    assert (block[:, slices["k_l"]] == f_X_l_expected).all()
    # Everything else zero.
    assert not block[:, slices["Q_r"]].any()
    assert not block[:, slices["k_r"]].any()
    assert not block[:, slices["A"]].any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest src/qldpc/circuits/surgery/joint_layout_test.py::test_block_data_x_left_carries_f_X_on_kappa_l_for_basis_l_Z -v`
Expected: FAIL with `ImportError: cannot import name '_block_data'`.

- [ ] **Step 3: Implement `_block_data`**

```python
# Append to src/qldpc/circuits/surgery/joint_layout.py
from qldpc.objects import Pauli


def _block_data(g, *, basis_block: PauliXZ, side: str, slices: dict[str, slice], N: int) -> np.ndarray:
    """Build the data-stabilizer block (H_X or H_Z) for one side.

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
        c_0 = list(g.data_checks)
        kappa_slice = slices[f"k_{side}"]
        for k, j in enumerate(c_0):
            block[j, kappa_slice.start + k] = 1
    return block
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest src/qldpc/circuits/surgery/joint_layout_test.py::test_block_data_x_left_carries_f_X_on_kappa_l_for_basis_l_Z -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/joint_layout.py src/qldpc/circuits/surgery/joint_layout_test.py
git commit -m "feat(surgery): _block_data builder

Data stabilizer rows extended with f_basis (= π_{C_0^s}^T) on the side's κ^s
columns iff the row's basis is dual to the side's gadget basis. Matches
main.tex §4.2 first two row blocks.
"
```

---

## Task 4: χ-row block builder

**Files:**
- Modify: `src/qldpc/circuits/surgery/joint_layout.py`
- Test: `src/qldpc/circuits/surgery/joint_layout_test.py`

- [ ] **Step 1: Write failing test for `_block_chi`**

```python
# Append to src/qldpc/circuits/surgery/joint_layout_test.py

from qldpc.circuits.surgery.joint_layout import _block_chi


def test_block_chi_left_z_basis_attaches_adapter_label() -> None:
    """χ_l rows (basis_l=Z) carry (π_{V_0^l} | H_Z'^{l,aug} | π_{P_l}^T P_{σ_l}) on (Q_l | k_l | A)."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, z, basis=Pauli.Z)
    g_r = build_gadget(code, x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    slices = column_slices_for_bridge(g_l, g_r, bridge)
    N = slices["A"].stop

    block = _block_chi(bridge.g_l_aug, side="l", slices=slices, N=N,
                       labels=bridge.label_l)
    n_V0_l = len(g_l.support)
    assert block.shape == (n_V0_l, N)
    # Q_l side: π_{V_0^l} — row i has 1 at column V_0^l[i].
    for i, v in enumerate(g_l.support):
        assert block[i, slices["Q_l"].start + v] == 1
    # k_l side: H_Z'^{l,aug} (incidence).
    assert (block[:, slices["k_l"]] == bridge.g_l_aug.incidence.T).all()
    # Adapter columns: row i has 1 at column A.start + label_l[i] when label_l[i] >= 0,
    # 0 otherwise.
    for i, lab in enumerate(bridge.label_l):
        if lab >= 0:
            assert block[i, slices["A"].start + lab] == 1
        # Non-port rows have 0 across all adapter columns.
        else:
            assert not block[i, slices["A"]].any()
    # No support on Q_r / k_r.
    assert not block[:, slices["Q_r"]].any()
    assert not block[:, slices["k_r"]].any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest src/qldpc/circuits/surgery/joint_layout_test.py::test_block_chi_left_z_basis_attaches_adapter_label -v`
Expected: FAIL with `ImportError: cannot import name '_block_chi'`.

- [ ] **Step 3: Implement `_block_chi`**

```python
# Append to src/qldpc/circuits/surgery/joint_layout.py


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest src/qldpc/circuits/surgery/joint_layout_test.py::test_block_chi_left_z_basis_attaches_adapter_label -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/joint_layout.py src/qldpc/circuits/surgery/joint_layout_test.py
git commit -m "feat(surgery): _block_chi builder

χ rows carry π_{V_0^s} on data, H_X/Z'^{s,aug} on κ^s, and π_{P_s}^T P_{σ_s}
on adapter (via the SkipTree label). Matches main.tex §4.2 row blocks 3-4.
"
```

---

## Task 5: Gauge and cycle block builders

**Files:**
- Modify: `src/qldpc/circuits/surgery/joint_layout.py`
- Test: `src/qldpc/circuits/surgery/joint_layout_test.py`

- [ ] **Step 1: Write failing tests for `_block_gauge` and `_block_cycle`**

```python
# Append to src/qldpc/circuits/surgery/joint_layout_test.py

from qldpc.circuits.surgery.joint_layout import _block_cycle, _block_gauge


def test_block_gauge_supports_only_kappa() -> None:
    """Gauge rows H_{X/Z}'^{s,aug} have support only on κ^s."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, z, basis=Pauli.Z)
    g_r = build_gadget(code, x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    slices = column_slices_for_bridge(g_l, g_r, bridge)
    N = slices["A"].stop

    block = _block_gauge(bridge.g_l_aug, side="l", slices=slices, N=N)
    r_l = bridge.g_l_aug.gauge.shape[0]
    assert block.shape == (r_l, N)
    assert (block[:, slices["k_l"]] == np.asarray(bridge.g_l_aug.gauge).astype(np.uint8)).all()
    assert not block[:, slices["Q_l"]].any()
    assert not block[:, slices["Q_r"]].any()
    assert not block[:, slices["k_r"]].any()
    assert not block[:, slices["A"]].any()


def test_block_cycle_carries_T_on_kappa_and_H_R_on_adapter() -> None:
    """Cycle row T_s on κ^s + H_R on adapter."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, z, basis=Pauli.Z)
    g_r = build_gadget(code, x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    slices = column_slices_for_bridge(g_l, g_r, bridge)
    N = slices["A"].stop

    block = _block_cycle(bridge.T_l, bridge.H_R, side="l", slices=slices, N=N)
    w_minus_1 = bridge.H_R.shape[0]
    assert block.shape == (w_minus_1, N)
    assert (block[:, slices["k_l"]] == bridge.T_l).all()
    assert (block[:, slices["A"]] == bridge.H_R).all()
    assert not block[:, slices["Q_l"]].any()
    assert not block[:, slices["Q_r"]].any()
    assert not block[:, slices["k_r"]].any()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest src/qldpc/circuits/surgery/joint_layout_test.py::test_block_gauge_supports_only_kappa src/qldpc/circuits/surgery/joint_layout_test.py::test_block_cycle_carries_T_on_kappa_and_H_R_on_adapter -v`
Expected: FAIL with `ImportError` for `_block_gauge` and `_block_cycle`.

- [ ] **Step 3: Implement both block builders**

```python
# Append to src/qldpc/circuits/surgery/joint_layout.py


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest src/qldpc/circuits/surgery/joint_layout_test.py::test_block_gauge_supports_only_kappa src/qldpc/circuits/surgery/joint_layout_test.py::test_block_cycle_carries_T_on_kappa_and_H_R_on_adapter -v`
Expected: PASS for both.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/joint_layout.py src/qldpc/circuits/surgery/joint_layout_test.py
git commit -m "feat(surgery): _block_gauge + _block_cycle builders

Gauge block carries the SkipTree gauge generators on κ^s only.
Cycle block carries T_s on κ^s and H_R on adapter, matching main.tex
§4.2's last row block (Swaroop Eq. 19 middle block).
"
```

---

## Task 6: `build_pre_merge` assembler with provenance

**Files:**
- Modify: `src/qldpc/circuits/surgery/joint_layout.py`
- Test: `src/qldpc/circuits/surgery/joint_layout_test.py`

- [ ] **Step 1: Write failing test for pre-merge layout structure**

```python
# Append to src/qldpc/circuits/surgery/joint_layout_test.py

from qldpc.circuits.surgery.joint_layout import build_pre_merge_layout


def test_pre_merge_layout_row_counts_mixed_basis_steane() -> None:
    """Mixed-basis (Z̄_l ⊗ X̄_r) pre-merge: row counts match §4.2 expectations."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, z, basis=Pauli.Z)
    g_r = build_gadget(code, x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)

    layout = build_pre_merge_layout(g_l, g_r, bridge)

    m_X_l = g_l.code.matrix_x.shape[0]
    m_X_r = g_r.code.matrix_x.shape[0]
    m_Z_l = g_l.code.matrix_z.shape[0]
    m_Z_r = g_r.code.matrix_z.shape[0]
    n_V0_l = len(g_l.support)
    n_V0_r = len(g_r.support)
    r_l_aug = bridge.g_l_aug.gauge.shape[0]
    r_r_aug = bridge.g_r_aug.gauge.shape[0]
    w_minus_1 = bridge.H_R.shape[0]

    # H_X rows: H_X^l + H_X^r + gauge_l + chi_r + cycle_l
    assert layout.H_X.shape[0] == m_X_l + m_X_r + r_l_aug + n_V0_r + w_minus_1
    # H_Z rows: H_Z^l + H_Z^r + chi_l + gauge_r + cycle_r
    assert layout.H_Z.shape[0] == m_Z_l + m_Z_r + n_V0_l + r_r_aug + w_minus_1
    # Pre-merge: H_Y empty.
    assert layout.H_Y.shape == (0, 2 * layout.column_slices["A"].stop)

    # Provenance: chi_l rows live in H_Z (basis_l=Z), chi_r in H_X (basis_r=X).
    assert len(layout.rows_chi["l"]) == n_V0_l
    assert len(layout.rows_chi["r"]) == n_V0_r
    assert layout.rows_y == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest src/qldpc/circuits/surgery/joint_layout_test.py::test_pre_merge_layout_row_counts_mixed_basis_steane -v`
Expected: FAIL with `ImportError: cannot import name 'build_pre_merge_layout'`.

- [ ] **Step 3: Implement `build_pre_merge_layout`**

```python
# Append to src/qldpc/circuits/surgery/joint_layout.py


def build_pre_merge_layout(g_l, g_r, bridge) -> JointPPMLayout:
    """Assemble the pre-merge (before cross-merge) joint check matrices per main.tex §4.2.

    Row order in H_X:
      1. data H_X^l (block 1)
      2. data H_X^r (block 2)
      3. gauge H_X'^{l,aug} if basis_l=Z, else gauge H_X'^{r,aug} on right side
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

    def _basis_to_block(g) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return (H_X rows, H_Z rows) for this side's blocks excluding cycle."""
        data_x_row = _block_data(g, basis_block=Pauli.X, side="l" if g is g_l else "r",
                                  slices=slices, N=N)
        data_z_row = _block_data(g, basis_block=Pauli.Z, side="l" if g is g_l else "r",
                                  slices=slices, N=N)
        side_label = "l" if g is g_l else "r"
        g_aug = bridge.g_l_aug if g is g_l else bridge.g_r_aug
        labels = bridge.label_l if g is g_l else bridge.label_r
        chi_block = _block_chi(g_aug, side=side_label, slices=slices, N=N, labels=labels)
        gauge_block = _block_gauge(g_aug, side=side_label, slices=slices, N=N)
        return data_x_row, data_z_row, chi_block, gauge_block

    data_x_l, data_z_l, chi_l, gauge_l = _basis_to_block(g_l)
    data_x_r, data_z_r, chi_r, gauge_r = _basis_to_block(g_r)
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
            rows.extend(np.asarray(block).astype(np.uint8))
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
        out = {"l": (), "r": ()}
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest src/qldpc/circuits/surgery/joint_layout_test.py::test_pre_merge_layout_row_counts_mixed_basis_steane -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/joint_layout.py src/qldpc/circuits/surgery/joint_layout_test.py
git commit -m "feat(surgery): build_pre_merge_layout assembler

Stacks block builders into H_X / H_Z with explicit per-row provenance.
χ rows go into the basis-native matrix; gauge and cycle into the dual.
The provenance dict is the central data structure for downstream obs0
emission per Lemma 2.
"
```

---

## Task 7: Cross-merge applying §4.3 (delete port rows + build $\tilde H_Y$)

**Files:**
- Modify: `src/qldpc/circuits/surgery/joint_layout.py`
- Test: `src/qldpc/circuits/surgery/joint_layout_test.py`

- [ ] **Step 1: Write failing test for cross-merge**

```python
# Append to src/qldpc/circuits/surgery/joint_layout_test.py

from qldpc.circuits.surgery.joint_layout import apply_cross_merge


def test_cross_merge_deletes_port_rows_and_builds_w_y_rows() -> None:
    """Cross-merge deletes one port row from H_X and H_Z each, builds w y_q rows."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, z, basis=Pauli.Z)
    g_r = build_gadget(code, x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    pre = build_pre_merge_layout(g_l, g_r, bridge)

    post = apply_cross_merge(pre, bridge)
    w = bridge.width

    # H_X loses w port-χ_r rows; H_Z loses w port-χ_l rows.
    assert post.H_X.shape[0] == pre.H_X.shape[0] - w
    assert post.H_Z.shape[0] == pre.H_Z.shape[0] - w
    # H_Y has w rows.
    assert post.H_Y.shape == (w, 2 * pre.column_slices["A"].stop)
    # rows_y == range(w).
    assert post.rows_y == tuple(range(w))
    # Surviving χ_l = non-port chi rows.
    surviving_chi_l = len(post.rows_chi["l"])
    surviving_chi_r = len(post.rows_chi["r"])
    assert surviving_chi_l == len(pre.rows_chi["l"]) - w
    assert surviving_chi_r == len(pre.rows_chi["r"]) - w
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest src/qldpc/circuits/surgery/joint_layout_test.py::test_cross_merge_deletes_port_rows_and_builds_w_y_rows -v`
Expected: FAIL with `ImportError: cannot import name 'apply_cross_merge'`.

- [ ] **Step 3: Implement `apply_cross_merge`**

```python
# Append to src/qldpc/circuits/surgery/joint_layout.py


def apply_cross_merge(pre: JointPPMLayout, bridge) -> JointPPMLayout:
    """Cross-merge per main.tex §4.3.

    For each adapter qubit q ∈ {0..w-1}:
      * Find the left port χ row at adapter column q (basis_l=Z → row in H_Z).
      * Find the right port χ row at adapter column q (basis_r=X → row in H_X).
      * Delete both rows; build y_q = (X-part-of-right-port | Z-part-of-left-port)
        as a symplectic row in H_Y.

    Port rows are identified by their adapter labels: for side s, the port row
    that contributes adapter X/Z at column q is the χ row whose label equals q.
    """
    w = bridge.width
    N = pre.column_slices["A"].stop

    chi_l_rows = list(pre.rows_chi["l"])  # indices into H_Z if basis_l=Z, into H_X if basis_l=X
    chi_r_rows = list(pre.rows_chi["r"])

    # Determine which matrix holds each side's χ block.
    def _matrix_for(basis):
        return "H_X" if basis is Pauli.X else "H_Z"

    chi_l_matrix = _matrix_for(pre.basis_l)
    chi_r_matrix = _matrix_for(pre.basis_r)
    assert chi_l_matrix != chi_r_matrix, "same-basis not supported here; use build_joint_layout for dispatch"

    # Map adapter label q → row index for each side.
    def _label_to_row_index(g_aug, labels, side_rows):
        """Returns dict {label_q: row_index_in_matrix_for_basis_s}.
        side_rows is the tuple of row indices reported in pre.rows_chi[side]
        — these enumerate V_0^s vertices in order, so label_q matches
        labels[chi_row_offset_within_block].
        """
        port_map: dict[int, int] = {}
        for offset, row_idx in enumerate(side_rows):
            lab = int(labels[offset])
            if lab >= 0:
                port_map[lab] = row_idx
        return port_map

    port_l_map = _label_to_row_index(bridge.g_l_aug, bridge.label_l, chi_l_rows)
    port_r_map = _label_to_row_index(bridge.g_r_aug, bridge.label_r, chi_r_rows)

    H_X_pre = pre.H_X.copy()
    H_Z_pre = pre.H_Z.copy()

    y_rows = np.zeros((w, 2 * N), dtype=np.uint8)
    port_l_delete = []
    port_r_delete = []
    for q in range(w):
        row_l = port_l_map[q]  # index into H_Z if basis_l=Z, H_X if basis_l=X
        row_r = port_r_map[q]  # opposite
        if pre.basis_l is Pauli.X:
            x_part = H_X_pre[row_l]
            z_part = H_Z_pre[row_r]
            port_l_delete.append(row_l)  # in H_X
            port_r_delete.append(row_r)  # in H_Z
        else:
            x_part = H_X_pre[row_r]
            z_part = H_Z_pre[row_l]
            port_l_delete.append(row_l)  # in H_Z
            port_r_delete.append(row_r)  # in H_X
        y_rows[q, :N] = x_part
        y_rows[q, N:] = z_part

    # Delete merged port rows from appropriate matrices.
    if pre.basis_l is Pauli.X:
        H_X_out = np.delete(H_X_pre, sorted(set(port_l_delete)), axis=0)
        H_Z_out = np.delete(H_Z_pre, sorted(set(port_r_delete)), axis=0)
    else:
        H_Z_out = np.delete(H_Z_pre, sorted(set(port_l_delete)), axis=0)
        H_X_out = np.delete(H_X_pre, sorted(set(port_r_delete)), axis=0)

    # Re-map row indices in provenance after deletion.
    def _remap(rows: tuple[int, ...], deleted: list[int]) -> tuple[int, ...]:
        deleted_sorted = sorted(deleted)
        new = []
        for r in rows:
            if r in deleted:
                continue
            shift = sum(1 for d in deleted_sorted if d < r)
            new.append(r - shift)
        return tuple(new)

    if pre.basis_l is Pauli.X:
        new_chi_l = _remap(pre.rows_chi["l"], port_l_delete)
        new_chi_r = _remap(pre.rows_chi["r"], port_r_delete)
        new_data_x_l = _remap(pre.rows_data_x["l"], port_l_delete)
        new_data_x_r = _remap(pre.rows_data_x["r"], port_l_delete)
        new_data_z_l = _remap(pre.rows_data_z["l"], port_r_delete)
        new_data_z_r = _remap(pre.rows_data_z["r"], port_r_delete)
        new_gauge_l = _remap(pre.rows_gauge["l"], port_r_delete)
        new_gauge_r = _remap(pre.rows_gauge["r"], port_r_delete)
        new_cycle_l = _remap(pre.rows_cycle["l"], port_r_delete)
        new_cycle_r = _remap(pre.rows_cycle["r"], port_r_delete)
    else:
        new_chi_l = _remap(pre.rows_chi["l"], port_l_delete)
        new_chi_r = _remap(pre.rows_chi["r"], port_r_delete)
        new_data_x_l = _remap(pre.rows_data_x["l"], port_r_delete)
        new_data_x_r = _remap(pre.rows_data_x["r"], port_r_delete)
        new_data_z_l = _remap(pre.rows_data_z["l"], port_l_delete)
        new_data_z_r = _remap(pre.rows_data_z["r"], port_l_delete)
        new_gauge_l = _remap(pre.rows_gauge["l"], port_l_delete)
        new_gauge_r = _remap(pre.rows_gauge["r"], port_l_delete)
        new_cycle_l = _remap(pre.rows_cycle["l"], port_l_delete)
        new_cycle_r = _remap(pre.rows_cycle["r"], port_l_delete)

    return JointPPMLayout(
        H_X=H_X_out,
        H_Z=H_Z_out,
        H_Y=y_rows,
        rows_data_x={"l": new_data_x_l, "r": new_data_x_r},
        rows_data_z={"l": new_data_z_l, "r": new_data_z_r},
        rows_chi={"l": new_chi_l, "r": new_chi_r},
        rows_gauge={"l": new_gauge_l, "r": new_gauge_r},
        rows_cycle={"l": new_cycle_l, "r": new_cycle_r},
        rows_y=tuple(range(w)),
        basis_l=pre.basis_l,
        basis_r=pre.basis_r,
        column_slices=pre.column_slices,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest src/qldpc/circuits/surgery/joint_layout_test.py::test_cross_merge_deletes_port_rows_and_builds_w_y_rows -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/joint_layout.py src/qldpc/circuits/surgery/joint_layout_test.py
git commit -m "feat(surgery): apply_cross_merge per main.tex §4.3

Port rows are identified by SkipTree label (label_s[i] == q), eliminating
the heuristic scan-for-single-{q} of the legacy merge.py. Y rows are built
in symplectic form (X-part-of-right | Z-part-of-left). Provenance is
re-mapped after deletion so downstream callers can XOR by row kind.
"
```

---

## Task 8: `build_joint_layout` dispatcher + symplectic commutation check

**Files:**
- Modify: `src/qldpc/circuits/surgery/joint_layout.py`
- Test: `src/qldpc/circuits/surgery/joint_layout_test.py`

- [ ] **Step 1: Write failing test for dispatcher + symplectic commutation**

```python
# Append to src/qldpc/circuits/surgery/joint_layout_test.py

from qldpc.circuits.surgery.joint_layout import build_joint_layout


def test_build_joint_layout_mixed_basis_satisfies_symplectic_commutation() -> None:
    """All pairwise rows in (H_X, H_Z, H_Y) commute symplectically over F_2."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, z, basis=Pauli.Z)
    g_r = build_gadget(code, x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)

    layout = build_joint_layout(g_l, g_r, bridge)
    N = layout.column_slices["A"].stop

    # Build symplectic representation: each row a vector in F_2^{2N}.
    def _sym(H, kind):
        if kind == "x":
            return np.hstack([H, np.zeros_like(H)])
        if kind == "z":
            return np.hstack([np.zeros_like(H), H])
        return H  # already symplectic

    full = np.vstack([
        _sym(layout.H_X, "x"),
        _sym(layout.H_Z, "z"),
        _sym(layout.H_Y, "y"),
    ])

    # Symplectic form J = [[0, I], [I, 0]] block (over F_2).
    a = full[:, :N]
    b = full[:, N:]
    # ⟨u, v⟩_J = u_X · v_Z + u_Z · v_X (mod 2).
    inner = (a @ b.T + b @ a.T) % 2
    assert (inner == 0).all(), f"symplectic commutation failed: nonzero inner products"


def test_build_joint_layout_raises_for_same_basis() -> None:
    """build_joint_layout only handles mixed-basis in this plan; same-basis raises."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(code, x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    with pytest.raises(NotImplementedError, match="same-basis"):
        build_joint_layout(g_l, g_r, bridge)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest src/qldpc/circuits/surgery/joint_layout_test.py::test_build_joint_layout_mixed_basis_satisfies_symplectic_commutation src/qldpc/circuits/surgery/joint_layout_test.py::test_build_joint_layout_raises_for_same_basis -v`
Expected: FAIL with `ImportError: cannot import name 'build_joint_layout'`.

- [ ] **Step 3: Implement dispatcher**

```python
# Append to src/qldpc/circuits/surgery/joint_layout.py


def build_joint_layout(g_l, g_r, bridge) -> JointPPMLayout:
    """Dispatcher: build pre-merge layout then apply cross-merge if mixed-basis.

    Same-basis joint PPM is out of scope for this iteration — it raises
    NotImplementedError. Callers (e.g. _build_joint_ppm_circuit_same_basis)
    continue to use the existing _stitch_intercode / _stitch_intracode path.
    """
    if bridge.basis_l is bridge.basis_r:
        raise NotImplementedError(
            "same-basis joint PPM remains on the legacy stitch in circuit.py; "
            "build_joint_layout handles only mixed-basis (basis_l != basis_r)"
        )
    pre = build_pre_merge_layout(g_l, g_r, bridge)
    return apply_cross_merge(pre, bridge)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest src/qldpc/circuits/surgery/joint_layout_test.py::test_build_joint_layout_mixed_basis_satisfies_symplectic_commutation src/qldpc/circuits/surgery/joint_layout_test.py::test_build_joint_layout_raises_for_same_basis -v`
Expected: PASS for both.

- [ ] **Step 5: Run full joint_layout_test.py to confirm regressions are clean**

Run: `python -m pytest src/qldpc/circuits/surgery/joint_layout_test.py -v`
Expected: ALL PASS.

- [ ] **Step 6: Commit**

```bash
git add src/qldpc/circuits/surgery/joint_layout.py src/qldpc/circuits/surgery/joint_layout_test.py
git commit -m "feat(surgery): build_joint_layout dispatcher + symplectic check

End-to-end JointPPMLayout for mixed-basis (e.g. Z̄_l ⊗ X̄_r). Asserts
pairwise symplectic commutation of all rows in H_X ∪ H_Z ∪ H_Y — Lemma 1
of the design spec.
"
```

---

## Task 9: `QuditCode` adapter so the circuit builder can consume `JointPPMLayout`

**Files:**
- Modify: `src/qldpc/circuits/surgery/joint_layout.py`
- Test: `src/qldpc/circuits/surgery/joint_layout_test.py`

- [ ] **Step 1: Write failing test for `to_quditcode`**

```python
# Append to src/qldpc/circuits/surgery/joint_layout_test.py

from qldpc.codes.quantum.qudit import QuditCode


def test_layout_to_quditcode_round_trips_rows() -> None:
    """JointPPMLayout.to_quditcode produces a QuditCode whose matrix concatenates
    [H_X | 0], [0 | H_Z], H_Y in that order."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, z, basis=Pauli.Z)
    g_r = build_gadget(code, x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    layout = build_joint_layout(g_l, g_r, bridge)

    qc = layout.to_quditcode()
    assert isinstance(qc, QuditCode)
    N = layout.column_slices["A"].stop
    expected = np.vstack([
        np.hstack([layout.H_X, np.zeros_like(layout.H_X)]),
        np.hstack([np.zeros_like(layout.H_Z), layout.H_Z]),
        layout.H_Y,
    ])
    assert (np.asarray(qc.matrix).astype(np.uint8) == expected).all()
    assert qc.num_qudits == N
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest src/qldpc/circuits/surgery/joint_layout_test.py::test_layout_to_quditcode_round_trips_rows -v`
Expected: FAIL with `AttributeError: 'JointPPMLayout' object has no attribute 'to_quditcode'`.

- [ ] **Step 3: Add `to_quditcode` method**

```python
# Modify the JointPPMLayout class in src/qldpc/circuits/surgery/joint_layout.py to add:

    def to_quditcode(self):
        """Bundle H_X, H_Z, H_Y into a single QuditCode matrix.

        Layout: [H_X | 0] rows first, then [0 | H_Z] rows, then H_Y rows
        (already symplectic). Useful for handing off to circuit builders
        that expect a unified stabilizer matrix.
        """
        from qldpc.codes.quantum.qudit import QuditCode

        N = self.column_slices["A"].stop
        zero = np.zeros_like(self.H_X)
        zero_z = np.zeros_like(self.H_Z)
        rows = np.vstack([
            np.hstack([self.H_X, zero]),
            np.hstack([zero_z, self.H_Z]),
            self.H_Y,
        ]).astype(np.uint8)
        from qldpc.codes.common import field_for_pauli
        # Pull the field directly from QuditCode constructor signature; mirror
        # _stitch_to_joint_code_mixed at circuit.py:912.
        field = field_for_pauli()  # placeholder; see implementation note below
        return QuditCode(field(rows), is_subsystem_code=True)
```

**Implementation note:** The actual field is obtained from `g_l.code.field`. Pass it through `to_quditcode(field)` rather than re-importing — keeps the layout decoupled from a specific code instance. Adjust the test to match.

- [ ] **Step 4: Adjust the method signature**

```python
# Replace to_quditcode with signature taking field explicitly:

    def to_quditcode(self, field):
        """Bundle H_X, H_Z, H_Y into a single QuditCode matrix.

        Args:
          field: GF(2) field implementation from the side codes (e.g. g_l.code.field).
        """
        from qldpc.codes.quantum.qudit import QuditCode

        N = self.column_slices["A"].stop
        zero = np.zeros_like(self.H_X)
        zero_z = np.zeros_like(self.H_Z)
        rows = np.vstack([
            np.hstack([self.H_X, zero]),
            np.hstack([zero_z, self.H_Z]),
            self.H_Y,
        ]).astype(np.uint8)
        return QuditCode(field(rows), is_subsystem_code=True)
```

- [ ] **Step 5: Adjust the test to pass field**

```python
# Update the test:
def test_layout_to_quditcode_round_trips_rows() -> None:
    ...
    qc = layout.to_quditcode(code.field)
    ...
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest src/qldpc/circuits/surgery/joint_layout_test.py::test_layout_to_quditcode_round_trips_rows -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/qldpc/circuits/surgery/joint_layout.py src/qldpc/circuits/surgery/joint_layout_test.py
git commit -m "feat(surgery): JointPPMLayout.to_quditcode(field) adapter

Bundles H_X, H_Z, H_Y into the unified QuditCode matrix expected by the
existing _build_joint_ppm_circuit_mixed_basis path. Subsystem-code flag is
True (the merged mixed-basis code is a subsystem code by construction).
"
```

---

## Task 10: Wire `joint_layout` into `_build_joint_ppm_circuit_mixed_basis` via new helper

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py:763` (insert new helper before existing function)
- Modify: `src/qldpc/circuits/surgery/circuit.py:608-630` (`_stitch_to_joint_code` dispatch — see below)

- [ ] **Step 1: Write integration test that exercises the dispatch**

```python
# Append to src/qldpc/circuits/surgery/joint_layout_test.py

def test_stitch_via_joint_layout_matches_legacy_row_counts_mixed_basis() -> None:
    """Layout-derived QuditCode and the legacy _stitch_to_joint_code_mixed
    produce stabilizer matrices with the same TOTAL row count (modulo provenance
    grouping). Verifies the new path is dimension-consistent with the existing
    implementation.
    """
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_code_mixed
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, z, basis=Pauli.Z)
    g_r = build_gadget(code, x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)

    legacy_code, _ = _stitch_to_joint_code_mixed(g_l, g_r, bridge)
    layout = build_joint_layout(g_l, g_r, bridge)
    new_code = layout.to_quditcode(code.field)
    assert np.asarray(new_code.matrix).shape[0] == np.asarray(legacy_code.matrix).shape[0]
    assert new_code.num_qudits == legacy_code.num_qudits
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `python -m pytest src/qldpc/circuits/surgery/joint_layout_test.py::test_stitch_via_joint_layout_matches_legacy_row_counts_mixed_basis -v`
Expected: PASS (if row counts match) or FAIL with an informative diff that drives debugging in Task 10 follow-up.

If the row counts differ, inspect by writing a temporary `print` of the provenance keys vs the legacy code's row breakdown (use Task 6 / Task 7 unit tests to localize). Adjust block ordering until counts match. Do NOT skip this gate — row count parity is the contract for downstream circuit consumption.

- [ ] **Step 3: Add `_build_mixed_basis_joint_code` to circuit.py**

```python
# Insert at src/qldpc/circuits/surgery/circuit.py, just above _stitch_to_joint_code_mixed (line 763):

def _build_mixed_basis_joint_code(
    g_l: GadgetLayout, g_r: GadgetLayout, bridge: Bridge
) -> tuple[QuditCode, Bridge, "JointPPMLayout"]:
    """New mixed-basis stitch via block-by-block layout (joint_layout module).

    Returns the QuditCode plus the JointPPMLayout itself; downstream callers
    (specifically the obs0 emission in _build_joint_ppm_circuit_mixed_basis)
    consume the layout's row provenance to construct obs0 = ⊕ m(χ_l) ⊕
    ⊕ m(χ_r) ⊕ ⊕ m(y_q) directly.
    """
    from qldpc.circuits.surgery.joint_layout import build_joint_layout

    layout = build_joint_layout(g_l, g_r, bridge)
    code = layout.to_quditcode(g_l.code.field)
    return code, bridge, layout
```

- [ ] **Step 4: Replace dispatch in `_stitch_to_joint_code`**

Find `_stitch_to_joint_code` (around `circuit.py:608`). Replace the body so that mixed-basis routes through the new function. Keep the same-basis path untouched:

```python
def _stitch_to_joint_code(
    g_l: GadgetLayout, g_r: GadgetLayout, bridge: Bridge
) -> tuple[QuditCode, Bridge]:
    """Joint stitch dispatcher.

    Same-basis (basis_l == basis_r): existing _stitch_intercode/_stitch_intracode
    (legacy CSS path).
    Mixed-basis: new _build_mixed_basis_joint_code via joint_layout.py.
    """
    if bridge.basis_l is bridge.basis_r:
        return _stitch_to_joint_csscode(g_l, g_r, bridge), bridge
    code, bridge_out, _layout = _build_mixed_basis_joint_code(g_l, g_r, bridge)
    return code, bridge_out
```

- [ ] **Step 5: Run circuit_mixed_test.py to verify no regression on existing tests that don't depend on layout**

Run: `python -m pytest src/qldpc/circuits/surgery/circuit_mixed_test.py::test_mixed_basis_joint_ppm_circuit_builds -v`
Expected: PASS (circuit still builds).

The existing `test_mixed_basis_circuit_compiles_to_dem` will still fail at this point because obs0 emission hasn't been updated yet. That's the next task.

- [ ] **Step 6: Commit**

```bash
git add src/qldpc/circuits/surgery/circuit.py src/qldpc/circuits/surgery/joint_layout_test.py
git commit -m "feat(surgery): route mixed-basis stitch through joint_layout

Mixed-basis joint PPM now builds its QuditCode via build_joint_layout(...)
instead of the legacy _stitch_to_joint_code_mixed. obs0 is still emitted
via the legacy path (next commit). The legacy function remains in the
module as a temporary fallback and reference.
"
```

---

## Task 11: Plumb `JointPPMLayout` through to obs0 emission

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py:996-1042` (`build_joint_ppm_circuit`) and `circuit.py:1313-1500` (`_build_joint_ppm_circuit_mixed_basis`)

- [ ] **Step 1: Pass layout into mixed-basis circuit builder**

Modify `build_joint_ppm_circuit` (around line 996) to call the new helper when basis_l != basis_r and pass the layout into the mixed-basis path:

```python
def build_joint_ppm_circuit(
    g_l, g_r, bridge,
    *,
    rounds: int,
    noise_model=None,
    data_init=None,
):
    """Dispatch joint PPM circuit. Same/mixed basis split."""
    if bridge.basis_l is bridge.basis_r:
        joint_code, bridge = _stitch_to_joint_code(g_l, g_r, bridge)
        return _build_joint_ppm_circuit_same_basis(
            g_l, g_r, bridge, joint_code,
            rounds=rounds, noise_model=noise_model, data_init=data_init,
        )
    joint_code, bridge, layout = _build_mixed_basis_joint_code(g_l, g_r, bridge)
    return _build_joint_ppm_circuit_mixed_basis(
        g_l, g_r, bridge, joint_code,
        layout=layout,
        rounds=rounds, noise_model=noise_model, data_init=data_init,
    )
```

- [ ] **Step 2: Update `_build_joint_ppm_circuit_mixed_basis` signature**

```python
def _build_joint_ppm_circuit_mixed_basis(
    g_l, g_r, bridge, joint_code,
    *,
    layout,  # NEW: JointPPMLayout
    rounds: int,
    noise_model,
    data_init,
) -> tuple[stim.Circuit, QuditCode]:
    """Mixed-basis joint PPM circuit (now consumes JointPPMLayout for obs0)."""
    # ...existing body unchanged through the obs0 emission section (line ~1596)...
```

- [ ] **Step 3: Run circuit build smoke test**

Run: `python -m pytest src/qldpc/circuits/surgery/circuit_mixed_test.py::test_mixed_basis_joint_ppm_circuit_builds -v`
Expected: PASS.

- [ ] **Step 4: Commit (no functional obs0 change yet, just plumbing)**

```bash
git add src/qldpc/circuits/surgery/circuit.py
git commit -m "refactor(surgery): plumb JointPPMLayout into mixed-basis circuit builder

build_joint_ppm_circuit now passes layout into the mixed-basis path, while
keeping same-basis on the legacy stitch. obs0 emission still uses the legacy
formula (next commit) — this commit is plumbing only.
"
```

---

## Task 12: Rewrite obs0 emission using `JointPPMLayout` row provenance

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py:1596-1628` (obs0 emission)

- [ ] **Step 1: Write failing test for `test_mixed_basis_circuit_compiles_to_dem` (already exists)**

This test was failing at the start of the plan. Re-run to confirm it still fails BEFORE the obs0 fix:

Run: `python -m pytest src/qldpc/circuits/surgery/circuit_mixed_test.py::test_mixed_basis_circuit_compiles_to_dem -v`
Expected: FAIL with "non-deterministic observables L0".

- [ ] **Step 2: Replace obs0 emission block with provenance-based version**

Find the existing obs0 block (around `circuit.py:1596-1628`). Replace with:

```python
    # obs0 per Lemma 2 (design spec §9 / main.tex §4.4):
    #   obs0 = ⊕_{i ∈ rows_chi['l']} m(check_for(i)) ⊕ ⊕_{i ∈ rows_chi['r']} m(check_for(i))
    #          ⊕ ⊕_{q ∈ rows_y} m(y_ancilla_ids[q])
    #
    # Surviving (non-port) χ rows and the cross-merged y_q rows together
    # multiply to Z̄_l ⊗ X̄_r as a product of merged-code stabilizers (Lemma 2).
    # In a noiseless run each m(·) is the corresponding stabilizer's
    # eigenvalue on the initial state, so the XOR is the joint logical bit.
    obs0_check_ids: list[int] = []
    # χ_l: row indices in layout.rows_chi['l'] index into H_X if basis_l=X,
    # H_Z if basis_l=Z. Map each row index to the check ancilla.
    def _row_to_check_id(row_idx: int, in_x_block: bool) -> int:
        # Mirror the lane assignment in _check_lane_index_map:
        #   χ rows for basis=X land on checks_x at offset m_X_total;
        #   χ rows for basis=Z land on checks_z at offset m_Z_total.
        # The provenance already gives the post-stitch row index in the
        # corresponding matrix; we just translate to a check ancilla ID.
        check_ids = qubit_ids.checks_x if in_x_block else qubit_ids.checks_z
        return check_ids[row_idx]

    for row_idx in layout.rows_chi["l"]:
        in_x = layout.basis_l is Pauli.X
        obs0_check_ids.append(_row_to_check_id(row_idx, in_x))
    for row_idx in layout.rows_chi["r"]:
        in_x = layout.basis_r is Pauli.X
        obs0_check_ids.append(_row_to_check_id(row_idx, in_x))
    for y_idx in layout.rows_y:
        obs0_check_ids.append(y_ancilla_ids[y_idx])

    if obs0_check_ids:
        obs0_targets = [
            measurement_record.get_target_rec(cid) for cid in obs0_check_ids
        ]
        circuit.append("OBSERVABLE_INCLUDE", obs0_targets, 0)
```

- [ ] **Step 3: Run the failing test**

Run: `python -m pytest src/qldpc/circuits/surgery/circuit_mixed_test.py::test_mixed_basis_circuit_compiles_to_dem -v`
Expected: PASS (obs0 is now deterministic).

If still failing, inspect the diagnostics:
1. Check `layout.rows_chi['l']` and `['r']` are nonempty.
2. Confirm `qubit_ids.checks_x[row_idx]` resolves to the correct ancilla per the
   `_check_lane_index_map` (lane 3 for χ).
3. The post-merge row index in `H_X` / `H_Z` may not equal the check-ancilla
   slot directly if `_split_quditcode_into_virtual_cssc` re-orders rows.
   If so, propagate the layout's row mapping through that split too — add a
   helper test before debugging blindly.

- [ ] **Step 4: Commit**

```bash
git add src/qldpc/circuits/surgery/circuit.py
git commit -m "fix(surgery): mixed-basis obs0 via row provenance — closes Lemma 2

obs0 = ⊕ m(χ_l) ⊕ ⊕ m(χ_r) ⊕ ⊕ m(y_q) — direct read of main.tex §4.4
Lemma 2. Driven by JointPPMLayout's per-row provenance instead of the
legacy Bridge fields (obs0_xor_map, x_leftover_indices, z_leftover_indices),
which carried only the y_q half of the formula.

Fixes test_mixed_basis_circuit_compiles_to_dem.
"
```

---

## Task 13: Un-xfail the truth-table test and verify noiseless correctness

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit_mixed_test.py:97-143`

- [ ] **Step 1: Remove the xfail marker on `test_mixed_basis_joint_truth_table_x_l_z_r`**

```python
# Replace the @pytest.mark.xfail(...) decorator with a plain definition:
def test_mixed_basis_joint_truth_table_x_l_z_r() -> None:
    """All four joint eigenstates of X̄_l ⊗ Z̄_r give correct obs0 sign noiselessly.

    With the JointPPMLayout-driven obs0 emission (main.tex §4.4 Lemma 2 closed),
    obs0 is deterministic in noiseless runs and equals the joint Pauli product.
    """
    from qldpc.circuits.surgery import build_joint_ppm_circuit, keep_only_observable

    g_l, g_r, bridge = _build_steane_mixed_pair()

    cases = [(("+", "0"), 0), (("-", "0"), 1), (("+", "1"), 1), (("-", "1"), 0)]
    for init, expected in cases:
        circuit, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=3, data_init=init)
        circuit = keep_only_observable(circuit, keep_idx=0)
        sampler = circuit.compile_detector_sampler()
        _, obs = sampler.sample(shots=256, separate_observables=True)
        assert obs.shape[1] == 1, f"expected 1 observable, got {obs.shape[1]}"
        assert (obs[:, 0] == expected).all(), (
            f"init={init} expected obs0={expected}, got mean {obs[:, 0].mean():.3f}"
        )
```

- [ ] **Step 2: Run the unxfailed test**

Run: `python -m pytest src/qldpc/circuits/surgery/circuit_mixed_test.py::test_mixed_basis_joint_truth_table_x_l_z_r -v`
Expected: PASS for all four init cases.

If any case fails, the most likely culprits are (in order of likelihood):
1. Wrong sign convention: check whether `m(·)` in Stim is `0` for `+1` eigenvalue or vice versa. Adjust the XOR with a constant if the convention is inverted.
2. Missing global-phase correction: per Lemma 2 the global phase `(-i)^w` cancels at the squared eigenvalue level — but if `w` is odd and some test cases have nontrivial initial phases, this can flip a bit. Confirm by sampling at `w=2` and `w=3` and seeing if exactly one parity flips.
3. Provenance row index mismatch (Task 12 step 3 diagnostic).

- [ ] **Step 3: Run the full mixed-basis test module**

Run: `python -m pytest src/qldpc/circuits/surgery/circuit_mixed_test.py -v`
Expected: All three tests PASS (no xfail left).

- [ ] **Step 4: Run all surgery tests as a regression gate**

Run: `python -m pytest src/qldpc/circuits/surgery/ -v --tb=short`
Expected: All same-basis tests still PASS (we did not touch the same-basis path). All mixed-basis tests PASS. If any same-basis test fails, investigate immediately — the dispatcher in `_stitch_to_joint_code` (Task 10) is the most likely regression site.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/circuit_mixed_test.py
git commit -m "test(surgery): un-xfail mixed-basis joint truth table — Lemma 2 closed

The truth-table test (X̄_l ⊗ Z̄_r noiseless correctness over all four joint
eigenstates) now passes with the JointPPMLayout-driven obs0 emission.
This closes the Tier 1 acceptance bar for the mixed-basis pipeline.
"
```

---

## Task 14: Update `_build_joint_ppm_circuit_mixed_basis` docstring + design spec status

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py:1313-1346` (function docstring)
- Modify: `docs/superpowers/specs/2026-06-15-mixed-basis-joint-ppm-design.md` (mark Lemma 2 closed)

- [ ] **Step 1: Update circuit.py docstring**

Find the existing docstring of `_build_joint_ppm_circuit_mixed_basis` (around line 1313) and replace the obs0 paragraph (currently around `6. obs0 per Lemma 2 (design spec §9): XOR of m(Y_stab[i])...`) with:

```python
    """Mixed-basis joint PPM circuit (subsystem merged code).

    [...existing intro about CKBB, CHRY, Cowtan et al...]

    Pipeline:
      1. _build_mixed_basis_joint_code (joint_layout.py) builds the merged
         QuditCode following docs/superpowers/docs/main.tex §4.2/§4.3
         block-by-block, returning a JointPPMLayout with per-row provenance.
      2. _split_quditcode_into_virtual_cssc partitions the joint-code matrix
         into pure-X / pure-Z rows (used by EdgeColoring) and Y-type rows
         (from the §4.3 cross-merge).
      3. Allocate ancillas: QubitIDs.from_code(virtual_cssc) for the CSS
         subset, then additional Y ancillas appended.
      4. Per-side state prep + detach (different bases for l / r).
      5. Per-round QEC: split X / Z / Y phases for determinism per
         Cohen-Kim-Bartlett-Brown arXiv:2110.10794 §II.B.2.
      6. obs0 = ⊕ m(χ_l) ⊕ ⊕ m(χ_r) ⊕ ⊕ m(y_q) per Lemma 2 of the design
         spec — implemented via JointPPMLayout row provenance.
    """
```

- [ ] **Step 2: Update design spec status**

Append to `docs/superpowers/specs/2026-06-15-mixed-basis-joint-ppm-design.md` (near the end) a status block:

```markdown
## Status (2026-06-18)

Tier 1 closed. obs0 deterministic for noiseless mixed-basis joint PPM via the
JointPPMLayout refactor (src/qldpc/circuits/surgery/joint_layout.py). The
legacy `_stitch_to_joint_code_mixed` and the `Bridge` fields `Y_stab`,
`obs0_xor_map`, `x_leftover_indices`, `z_leftover_indices`, `merge_qubits`
are now dead code paths and should be removed in a follow-up.
```

- [ ] **Step 3: Commit**

```bash
git add src/qldpc/circuits/surgery/circuit.py docs/superpowers/specs/2026-06-15-mixed-basis-joint-ppm-design.md
git commit -m "docs(surgery): document JointPPMLayout as the mixed-basis obs0 source

The circuit-builder docstring now references the new joint_layout module.
The design spec is updated to note Tier 1 is closed and the legacy stitch /
Bridge mixed-basis fields are slated for follow-up cleanup.
"
```

---

## Self-Review

**Spec coverage:**
- main.tex §4.2 pre-merge matrices → Tasks 3-6 build them block-by-block.
- main.tex §4.3 cross-merge → Task 7.
- main.tex §4.4 commutation → Task 8 (symplectic check).
- main.tex §4.5 deterministic readout (Lemma 2) → Tasks 12-13.
- Design spec §9 Lemma 1 (commutation) → Task 8.
- Design spec §9 Lemma 2 (obs0 formula) → Tasks 12-13.
- Design spec §9 Lemma 3 (noiseless correctness) → Task 13 truth-table test.
- Failing `test_mixed_basis_circuit_compiles_to_dem` → fixed in Task 12.
- xfail `test_mixed_basis_joint_truth_table_x_l_z_r` → un-xfailed in Task 13.

**Placeholder scan:**
- No "TBD", "add appropriate error handling", "similar to above" — all code shown.
- Task 9 Step 3 had a `field_for_pauli()` placeholder; corrected in Step 4-5 to take `field` as an argument explicitly.
- Task 10 Step 2 mentions "may fail" with diagnostic guidance — this is intentional, not a placeholder, because row-count parity is the contract that drives debugging.
- Task 12 Step 3 has a diagnostic list — same justification.

**Type consistency:**
- `JointPPMLayout` defined in Task 1, methods added in Task 9, consumed in Tasks 10-12. Names match.
- `column_slices_for_bridge` (Task 2) is called by `build_pre_merge_layout` (Task 6) and `apply_cross_merge` (Task 7).
- Block builders `_block_data`, `_block_chi`, `_block_gauge`, `_block_cycle` all called by `build_pre_merge_layout`. Signatures match.
- `apply_cross_merge(pre, bridge)` returns `JointPPMLayout`. Consumed by `build_joint_layout` (Task 8).
- `to_quditcode(field)` (Task 9) called by `_build_mixed_basis_joint_code` (Task 10) and `test_layout_to_quditcode_round_trips_rows` (Task 9 test).

All names consistent. No type drift.

---

## Out of Scope (Follow-up Plans)

After this plan lands and Tier 1 is green, the following cleanups remain:

1. **Same-basis migration to `joint_layout`** — port `_stitch_intercode` / `_stitch_intracode` to the same `build_joint_layout` pattern with `H_Y = ∅`. Risk: medium, since 30+ existing same-basis tests cover the legacy path.
2. **`Bridge` dataclass cleanup** — remove `Y_stab`, `obs0_xor_map`, `x_leftover_indices`, `z_leftover_indices`, `merge_qubits` once both paths flow through `JointPPMLayout`.
3. **Legacy `_stitch_to_joint_code_mixed` removal** — delete after a release cycle of dead-code monitoring.
4. **`merge.py` `apply_mixed_basis_merge` removal** — same as above.

Each warrants its own plan in `docs/superpowers/plans/`.
