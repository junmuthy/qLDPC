# Cheeger.py L=1 Collapse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the L≥1 "general layered" wrapper in `src/qldpc/codes/surgery/cheeger.py` to L=1 only (since Cain's protocol — the only one implemented end-to-end — uses L=1 + Cheeger boost), then dedupe the HX assembly between `gadget.py` and `cheeger.py`.

**Architecture:** Two semantic commits + one doc-cleanup commit. Commit 1 is purely mechanical (substitute `num_layers=1` into every loop, delete dead branches, slim `SurgeryLayout`); output matrices remain bit-identical. Commit 2 extracts the shared HX block-matrix assembly into one helper (`gadget.py:_assemble_HX_L1`) used by both `_step3_assemble` and `_reassemble_gadget_with_new_F`, and deletes two now-redundant functions in `cheeger.py`. Commit 3 deletes a stale doc comment.

**Tech Stack:** Python 3, numpy, galois (GF(2) linalg), pytest. Files: `src/qldpc/codes/surgery/cheeger.py` (953 LOC → ~795), `src/qldpc/codes/surgery/gadget.py` (+10 LOC), and one comment edit in `examples/logical_error_rates/_9_lattice_surgery_cain_fig1b_source.py`.

**Spec:** `docs/superpowers/specs/2026-06-09-cheeger-l1-collapse-design.md`

---

## File map

| File | Action | Why |
|---|---|---|
| `src/qldpc/codes/surgery/cheeger.py` | Modify (Task 1 + Task 2) | Main refactor target |
| `src/qldpc/codes/surgery/gadget.py` | Modify (Task 2) | Receives extracted `_assemble_HX_L1` helper |
| `src/qldpc/codes/surgery/_test.py` | Read-only (baseline regression check) | No edits — existing tests prove behavior preservation |
| `examples/logical_error_rates/_9_lattice_surgery_cain_fig1b_source.py` | Modify (Task 3) | Stale comment cleanup |
| `examples/scripts/cain_bb18_resource_exact_match.py` | Read-only (regression check) | Reproduces Cain Table III; must still print (39, 20, 20) |

This refactor preserves behavior, so the "test" is the existing test suite continuing to pass plus a one-off matrix-equality sanity check that is **not** committed. No new tests are added.

---

## Task 1: Mechanical collapse of cheeger.py (Step 1 in spec)

This task is large because the substeps are interdependent — deleting `_LayeredBlocks` requires fixing every caller, changing `SurgeryLayout` fields requires updating the two writer functions and one reader, etc. The plan minimises in-progress breakage by doing the changes in an order where the module compiles after each block of edits.

**Files:**
- Modify: `src/qldpc/codes/surgery/cheeger.py` (lines 19-197, 426-498, 830-887)

### Sub-step A: Baseline + sanity snapshot (not committed)

- [ ] **Step 1: Confirm baseline green**

Run: `pytest src/qldpc/codes/surgery/_test.py -x -q`
Expected: all tests pass (no failures, no errors).

- [ ] **Step 2: Create a one-off sanity snapshot script (DO NOT commit)**

Create a temporary file `/tmp/snapshot_HX_HZ.py`:

```python
"""One-off: snapshot HX_merged/HZ_merged for fixtures, used to verify Task 1 is mechanical."""
import pickle
import numpy as np
import sympy
from qldpc import codes
from qldpc.codes.surgery import build_gadget, boost_gadget

# Fixture 1: bb_18 (the Cain Table III example)
x, y = sympy.symbols("x y")
bb18 = codes.BBCode((31, 4), 1 + x**6 * y + x**27, y**2 + x**15 * y**3 + x**24)
target = codes.CSSCode(bb18.matrix_z, bb18.matrix_x, is_subsystem_code=False)

# Pick any single-weight x in ker(HZ) — use first logical X column
HZ = np.asarray(bb18.matrix_z).astype(np.uint8)
HX_data = np.asarray(bb18.matrix_x).astype(np.uint8)
# Use a deterministic logical X support: first nontrivial element of ker(HZ)
from qldpc.objects import Pauli
log_x_ops = bb18.get_logical_ops(Pauli.X)
vec = np.asarray(log_x_ops[0]).astype(np.uint8)
g = build_gadget(target, vec)

snapshot = {
    "HX_merged": g.HX_merged.copy(),
    "HZ_merged": g.HZ_merged.copy(),
    "F": g.F.copy(),
    "G": g.G.copy(),
    "V0": g.V0,
    "C0": g.C0,
    "kappa_qubits": g.kappa_qubits,
}
with open("/tmp/snapshot_baseline.pkl", "wb") as fh:
    pickle.dump(snapshot, fh)
print("Snapshot saved: shapes", g.HX_merged.shape, g.HZ_merged.shape)
```

Run: `python /tmp/snapshot_HX_HZ.py`
Expected: prints `Snapshot saved: shapes (...,...) (...,...)`.

(After Task 1 completes, re-run a comparison script — Step 14 below.)

### Sub-step B: Edit SurgeryLayout class (lines 19-51)

- [ ] **Step 3: Replace SurgeryLayout dataclass fields**

In `src/qldpc/codes/surgery/cheeger.py`, replace the body of `SurgeryLayout` (lines 19-51) with:

```python
@dataclasses.dataclass(frozen=True, eq=False)
class SurgeryLayout:
    """Provenance of qubits and checks in a merged L=1 surgery code.

    Internal type used by the boost dispatcher and the legacy-layout bridge
    (_gadget_to_legacy_layout / _legacy_to_gadget).

    Attributes:
        num_data_qubits: Number of qubits in the original data code.
        num_ancilla_qubits: Number of κ ancilla qubits (= |C_0| + boost extras).
        num_data_x_checks: Number of rows of HX_merged that are "data" rows
            (i.e. the original X-checks). Boost code uses this to slice off
            data rows from the merged check matrix when extracting the data
            block.
        num_data_z_checks: Number of rows of HZ_merged that are "data" rows.
        v0_indices: Indices (within data qubits) of supp(X̄_M) = V_0.
        c0_indices: Row indices (within H_Z of data code) of Z-checks adjacent
            to V_0 = C_0.
        F: Step-1 restriction matrix; shape (|C_0|, |V_0|).
        G: Step-2 gauge-fix basis; rows span the left null space of F^T.
    """

    num_data_qubits: int
    num_ancilla_qubits: int
    num_data_x_checks: int
    num_data_z_checks: int
    v0_indices: npt.NDArray[np.int_]
    c0_indices: npt.NDArray[np.int_]
    F: galois.FieldArray
    G: galois.FieldArray
```

(The `npt` import is already in the file. We drop `hx_row_kind`/`hz_row_kind` fields and the `import numpy.typing as npt` line stays in use via `v0_indices`/`c0_indices`.)

### Sub-step C: Inline `_LayeredBlocks` into all 4 callers (lines 93-197, 426-498)

`_LayeredBlocks` exists only to expose `n_v0`, `n_c0`, `F`, `F_T`, and a slice helper. At L=1 we can read these directly off `F` and use `slice(0, n_c0)` everywhere `ancilla_col_slice(1)` was used.

- [ ] **Step 4: Rewrite `_assemble_merged_HX` (lines 93-121)**

Replace the entire body of `_assemble_merged_HX` with:

```python
def _assemble_merged_HX(
    data_code: CSSCode,
    F: galois.FieldArray,
    v0_indices: np.ndarray,
) -> galois.FieldArray:
    """Assemble the merged H_X for L=1 surgery: [[HX_data, 0], [E_V0, F^T]]."""
    field = data_code.field
    n_data = data_code.num_qubits
    n_v0 = int(F.shape[1])
    n_c0 = int(F.shape[0])
    n_merged = n_data + n_c0

    hx = data_code.matrix_x
    n_x_data = int(hx.shape[0])
    top_block = field.Zeros((n_x_data, n_merged))
    top_block[:, :n_data] = hx

    chi_block = field.Zeros((n_v0, n_merged))
    chi_block[np.arange(n_v0), v0_indices] = 1
    chi_block[:, n_data:] = F.T

    return field(np.vstack([top_block, chi_block]))
```

(Signature change: `blocks: _LayeredBlocks` → `F: galois.FieldArray`. Drops the odd-layer loop, the `if i==1/else` branch, and the `if i+1 ≤ num_layers` branch — all evaluate to a single iteration with the first arm taken at L=1.)

- [ ] **Step 5: Rewrite `_assemble_merged_HZ` (lines 124-157) — actually, mark for deletion**

The whole function is dead (zero callers). Still rewrite first for safety, then delete in Step 7. Replace its body with a single raise, so any latent caller breaks loudly:

```python
def _assemble_merged_HZ(*args, **kwargs):
    raise NotImplementedError(
        "_assemble_merged_HZ has been removed during the L=1 collapse. "
        "Use the inline HZ-build inside _reassemble_gadget_with_new_F."
    )
```

(This temporary stub flushes out any caller we missed; we delete the function entirely after running the test suite.)

- [ ] **Step 6: Rewrite `_build_layout` (lines 160-197)**

Replace the entire body of `_build_layout` with:

```python
def _build_layout(
    data_code: CSSCode,
    F: galois.FieldArray,
    G: galois.FieldArray,
    v0_indices: np.ndarray,
    c0_indices: np.ndarray,
) -> SurgeryLayout:
    """Assemble SurgeryLayout from the building blocks (L=1)."""
    n_data = data_code.num_qubits
    n_ancilla = int(F.shape[0])
    num_data_x_checks = int(data_code.matrix_x.shape[0])
    num_data_z_checks = int(data_code.matrix_z.shape[0])
    return SurgeryLayout(
        num_data_qubits=n_data,
        num_ancilla_qubits=n_ancilla,
        num_data_x_checks=num_data_x_checks,
        num_data_z_checks=num_data_z_checks,
        v0_indices=v0_indices,
        c0_indices=c0_indices,
        F=F,
        G=G,
    )
```

(Signature change: drops `blocks: _LayeredBlocks`. Drops the `qubit_layer` array, the row_kind string-array construction, and the `for i in range(1, L+1, 2)` loop entirely.)

- [ ] **Step 7: Rewrite `_reassemble_gadget_with_new_F` (lines 426-498)**

Replace the entire body of `_reassemble_gadget_with_new_F` with:

```python
def _reassemble_gadget_with_new_F(
    merged: CSSCode,
    layout: SurgeryLayout,
    augmented_F: galois.FieldArray,
    n_extra: int,
) -> tuple[CSSCode, SurgeryLayout]:
    """Rebuild merged code + layout from an augmented restriction matrix (L=1).

    Boost-added κ' qubits (rows of F_aug beyond original C_0) are GAUGE qubits:
    they have no data-Z extension (no S_j in C_0 to extend). Their Z-stab
    contribution comes purely through the augmented gauge-fix matrix
    G_aug = basis of left null space of F_aug^T.
    """
    field = augmented_F.__class__
    G_aug = _compute_gauge_fix(augmented_F)
    n_data = layout.num_data_qubits

    data_x_arr = np.asarray(merged.matrix_x[:layout.num_data_x_checks]).astype(np.int_)
    data_z_arr = np.asarray(merged.matrix_z[:layout.num_data_z_checks]).astype(np.int_)
    data_x_gf = field(data_x_arr[:, :n_data])
    data_z_gf = field(data_z_arr[:, :n_data])

    data_code_proxy = CSSCode(data_x_gf, data_z_gf, is_subsystem_code=False)

    HX_new = _assemble_merged_HX(data_code_proxy, augmented_F, layout.v0_indices)

    # Manually build HZ_new: new κ' qubits get NO data-Z extension — only G_aug
    # rows mention them.
    n_c0 = int(augmented_F.shape[0])
    n_merged = n_data + n_c0
    n_kappa_orig = int(layout.F.shape[0])

    old_z = field.Zeros((data_z_gf.shape[0], n_merged))
    old_z[:, :n_data] = data_z_gf
    I_partial = field.Identity(n_kappa_orig)
    old_z[layout.c0_indices, n_data:n_data + n_kappa_orig] = I_partial

    gauge_rows: list[galois.FieldArray] = []
    if G_aug.shape[0] > 0:
        gf = field.Zeros((G_aug.shape[0], n_merged))
        gf[:, n_data:] = G_aug
        gauge_rows.append(gf)

    HZ_new = field(np.vstack([old_z, *gauge_rows]))

    boosted_merged = CSSCode(HX_new, HZ_new, is_subsystem_code=False)
    boosted_layout = _build_layout(
        data_code_proxy, augmented_F, G_aug, layout.v0_indices, layout.c0_indices,
    )
    return boosted_merged, boosted_layout
```

(Drops: `blocks` variable, `c1_slice` via `ancilla_col_slice(1)`, `even_rows` loop (empty at L=1), and the `hx_row_kind == "data"` boolean mask.)

- [ ] **Step 8: Rewrite `_gadget_to_legacy_layout` (lines 830-887)**

Replace the entire body of `_gadget_to_legacy_layout` with:

```python
def _gadget_to_legacy_layout(g):
    """Convert a GadgetLayout into the legacy (CSSCode, SurgeryLayout) pair
    consumed by boost_gadget_cheeger* / boost_gadget_distance.

    For basis=Pauli.Z, we SWAP HX/HZ so the legacy boost code (designed for
    X-basis chi rows in HX_merged) sees the chi rows where it expects them.
    The boost result is dual-swapped back in _legacy_to_gadget.
    """
    from qldpc.objects import Pauli
    F2 = galois.GF(2)
    n = g.code.num_qudits
    n_anc = len(g.C0)
    mX_data = int(g.code.matrix_x.shape[0])
    mZ_data = int(g.code.matrix_z.shape[0])

    if g.basis is Pauli.X:
        HX_for_legacy = g.HX_merged
        HZ_for_legacy = g.HZ_merged
        # After (no) swap: HX_for_legacy data rows = mX_data; HZ data rows = mZ_data.
        num_data_x_checks = mX_data
        num_data_z_checks = mZ_data
    else:  # Pauli.Z: swap so chi rows are in HX_for_legacy
        HX_for_legacy = g.HZ_merged
        HZ_for_legacy = g.HX_merged
        # After swap: HX_for_legacy data rows = mZ_data (the original HZ data part);
        #             HZ_for_legacy data rows = mX_data.
        num_data_x_checks = mZ_data
        num_data_z_checks = mX_data

    merged = CSSCode(
        F2(np.asarray(HX_for_legacy).astype(np.int_).tolist()),
        F2(np.asarray(HZ_for_legacy).astype(np.int_).tolist()),
        is_subsystem_code=False,
    )
    layout = SurgeryLayout(
        num_data_qubits=n,
        num_ancilla_qubits=n_anc,
        num_data_x_checks=num_data_x_checks,
        num_data_z_checks=num_data_z_checks,
        v0_indices=np.array(g.V0, dtype=np.int_),
        c0_indices=np.array(g.C0, dtype=np.int_),
        F=F2(np.asarray(g.F).astype(np.int_).tolist()),
        G=F2(np.asarray(g.G).astype(np.int_).tolist()),
    )
    return merged, layout
```

(Drops: `num_layers=1`, `qubit_layer`, `hx_row_kind`, `hz_row_kind` from the SurgeryLayout construction. Reads basis-aware data row counts directly off `g.code.matrix_x.shape[0]`.)

### Sub-step D: Delete the now-unused infrastructure

- [ ] **Step 9: Delete `_LayeredBlocks`, `_build_layered_blocks`, and `_assemble_merged_HZ`**

In `src/qldpc/codes/surgery/cheeger.py`:
- Delete the `@dataclasses.dataclass` decorator and `class _LayeredBlocks:` block (lines 59-85 in the original file).
- Delete the `def _build_layered_blocks(...)` function (lines 88-90 in the original file).
- Delete the `def _assemble_merged_HZ(*args, **kwargs):` stub created in Step 5.

Also delete the comment banner block at lines 14-17 and 200-202 (`# === SurgeryLayout and supporting helpers ===` / `# === End of inlined helpers ===`) — they no longer separate anything meaningful.

### Sub-step E: Verify and commit

- [ ] **Step 10: Re-run the full surgery test suite**

Run: `pytest src/qldpc/codes/surgery/_test.py -x -q`
Expected: all tests pass (same count as Step 1 baseline; no failures or errors).

If any test fails, do NOT proceed; the failure means the L=1 collapse introduced a behaviour change. Inspect the failure, fix in place, re-run.

- [ ] **Step 11: Re-run the matrix-equality sanity script**

Create temporary file `/tmp/compare_snapshot.py`:

```python
"""Verify Task 1 is purely mechanical: post-refactor matrices == baseline."""
import pickle
import numpy as np
import sympy
from qldpc import codes
from qldpc.codes.surgery import build_gadget
from qldpc.objects import Pauli

x, y = sympy.symbols("x y")
bb18 = codes.BBCode((31, 4), 1 + x**6 * y + x**27, y**2 + x**15 * y**3 + x**24)
target = codes.CSSCode(bb18.matrix_z, bb18.matrix_x, is_subsystem_code=False)
log_x_ops = bb18.get_logical_ops(Pauli.X)
vec = np.asarray(log_x_ops[0]).astype(np.uint8)
g = build_gadget(target, vec)

with open("/tmp/snapshot_baseline.pkl", "rb") as fh:
    baseline = pickle.load(fh)

for attr in ["HX_merged", "HZ_merged", "F", "G"]:
    new = np.asarray(getattr(g, attr))
    old = np.asarray(baseline[attr])
    assert new.shape == old.shape, f"{attr} shape mismatch: {new.shape} vs {old.shape}"
    assert np.array_equal(new, old), f"{attr} bit difference"
assert g.V0 == baseline["V0"], "V0 changed"
assert g.C0 == baseline["C0"], "C0 changed"
assert g.kappa_qubits == baseline["kappa_qubits"], "kappa_qubits changed"
print("OK — bit-identical to baseline.")
```

Run: `python /tmp/compare_snapshot.py`
Expected: `OK — bit-identical to baseline.`

- [ ] **Step 12: Run the Cain Table III regression**

Run: `python examples/scripts/cain_bb18_resource_exact_match.py`
Expected: terminates with `✓ EXACT MATCH with Cain Extended Data Table III` and `(Qubits=39, X-checks=20, Z-checks=20)`.

- [ ] **Step 13: Delete temporary files**

Run: `rm /tmp/snapshot_HX_HZ.py /tmp/snapshot_baseline.pkl /tmp/compare_snapshot.py`

- [ ] **Step 14: Commit**

Run: `git add src/qldpc/codes/surgery/cheeger.py`

Then:

```bash
git commit -m "$(cat <<'EOF'
refactor: collapse cheeger.py L≥1 wrapper to L=1 (mechanical)

- Drop SurgeryLayout.{num_layers, qubit_layer, hx_row_kind, hz_row_kind};
  add num_data_x_checks, num_data_z_checks (int).
- Delete _LayeredBlocks, _build_layered_blocks, _assemble_merged_HZ
  (already had zero callers).
- Inline odd-layer loops to single statements; delete empty even-layer
  loops in _assemble_merged_HX and _reassemble_gadget_with_new_F.
- _build_layout: drop qubit_layer + row_kind computation.
- _gadget_to_legacy_layout: read data-row counts directly off the
  input code's check matrices; basis=Z HX/HZ swap logic unchanged.

Purely mechanical: HX_merged/HZ_merged bit-identical to baseline on
the bb_18 fixture; full surgery test suite passes unchanged;
cain_bb18_resource_exact_match still prints (39, 20, 20).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Dedupe HX assembly between gadget.py and cheeger.py (Step 2 in spec)

The L=1 HX assembly `[[HX_data, 0], [E_V0, F^T]]` is built in two places: `gadget.py:_step3_assemble` and `cheeger.py:_assemble_merged_HX`. Extract the shared core into one helper in `gadget.py`, have both callers use it, then delete `cheeger.py:_assemble_merged_HX`.

**Files:**
- Modify: `src/qldpc/codes/surgery/gadget.py` (add helper, refactor `_step3_assemble`)
- Modify: `src/qldpc/codes/surgery/cheeger.py` (call helper from `_reassemble_gadget_with_new_F`, delete `_assemble_merged_HX`)

### Sub-step A: Add the shared helper to gadget.py

- [ ] **Step 1: Add `_assemble_HX_L1` to gadget.py**

Open `src/qldpc/codes/surgery/gadget.py`. Insert the following function **immediately before `_step3_assemble`** (i.e., before line 80 in the current file):

```python
def _assemble_HX_L1(
    HX_data: np.ndarray,
    v0_indices: np.ndarray,
    F: np.ndarray,
) -> np.ndarray:
    """L=1 HX-side block assembly: [[HX_data, 0], [E_V0, F^T]] over GF(2).

    Shared by _step3_assemble (initial gadget assembly) and
    cheeger._reassemble_gadget_with_new_F (post-boost rebuild). The Z-side
    assembly is NOT shared — the boost rebuild treats new κ' qubits as
    pure-gauge (no data-Z extension), unlike the initial assembly.

    Args:
        HX_data: original code's X-check matrix, shape (mX, n), uint8.
        v0_indices: indices of V_0 within the n data qubits, shape (|V_0|,).
        F: restriction matrix, shape (|C_0|, |V_0|), uint8.

    Returns:
        HX_merged: shape (mX + |V_0|, n + |C_0|), uint8.
    """
    mX, n = HX_data.shape
    n_v0, n_c0 = int(F.shape[1]), int(F.shape[0])
    n_merged = n + n_c0
    top = np.hstack([HX_data, np.zeros((mX, n_c0), dtype=np.uint8)]).astype(np.uint8)
    bot = np.zeros((n_v0, n_merged), dtype=np.uint8)
    bot[np.arange(n_v0), np.asarray(v0_indices)] = 1
    bot[:, n:] = F.T
    return np.vstack([top, bot]).astype(np.uint8)
```

### Sub-step B: Rewire `_step3_assemble` to call the helper

- [ ] **Step 2: Replace the basis-X / basis-Z block-matrix construction in `_step3_assemble`**

In `src/qldpc/codes/surgery/gadget.py`, find the block that begins:

```python
    if basis is Pauli.X:
        # χ rows extend HX_merged; G rows extend HZ_merged
        HX_merged = np.block([
            [HX, np.zeros((mX, nC), dtype=np.uint8)],
            [E_V0_T, F_T],
        ]).astype(np.uint8)
```

Replace the **entire `if basis is Pauli.X / else` block** (currently lines 114-133) with:

```python
    v0_arr = np.asarray(V0, dtype=np.int_)

    if basis is Pauli.X:
        # χ rows extend HX_merged; G rows extend HZ_merged
        HX_merged = _assemble_HX_L1(HX, v0_arr, F)
        HZ_merged = np.block([
            [HZ, F_tilde],
            [np.zeros((r, n), dtype=np.uint8), G.astype(np.uint8)],
        ]).astype(np.uint8)
    else:
        # basis=Z (symmetric dual): χ rows extend HZ_merged; G rows extend HX_merged
        HZ_merged = _assemble_HX_L1(HZ, v0_arr, F)
        HX_merged = np.block([
            [HX, F_tilde],
            [np.zeros((r, n), dtype=np.uint8), G.astype(np.uint8)],
        ]).astype(np.uint8)
```

(The local computations of `E_V0_T` and `F_T` at lines 101-104 become unused — also delete those four lines.)

- [ ] **Step 3: Run the surgery tests**

Run: `pytest src/qldpc/codes/surgery/_test.py -x -q`
Expected: all tests pass.

If any test fails (especially `test_basis_z_dual_equivalence` or anything matching `basis_z`), inspect the dual-symmetry handling in the `else` branch — the most likely source is dropping `HZ_data` where the original wrote `HX_data` or vice versa.

### Sub-step C: Switch `_reassemble_gadget_with_new_F` to the helper, delete `_assemble_merged_HX`

- [ ] **Step 4: Add an import for `_assemble_HX_L1` in cheeger.py**

At the top of `src/qldpc/codes/surgery/cheeger.py`, after the existing `from qldpc.codes.common import CSSCode` import, add:

```python
from .gadget import _assemble_HX_L1
```

- [ ] **Step 5: Rewrite `_reassemble_gadget_with_new_F` to use the helper**

In `src/qldpc/codes/surgery/cheeger.py`, find the line:

```python
    HX_new = _assemble_merged_HX(data_code_proxy, augmented_F, layout.v0_indices)
```

(This is the post-Task-1 form.) Replace it with:

```python
    HX_data_uint8 = np.asarray(data_x_gf).astype(np.uint8)
    F_uint8 = np.asarray(augmented_F).astype(np.uint8)
    HX_new_uint8 = _assemble_HX_L1(HX_data_uint8, layout.v0_indices, F_uint8)
    HX_new = field(HX_new_uint8.astype(np.int_).tolist())
```

(The helper returns numpy uint8; the rest of the function uses `galois.FieldArray`, so we round-trip via `field(...)`. Yes, this is slightly verbose at the boundary — acceptable cost for sharing the helper.)

- [ ] **Step 6: Delete `_assemble_merged_HX` from cheeger.py**

Delete the entire function `def _assemble_merged_HX(...) -> galois.FieldArray:` and its body (now the post-Task-1 ~12-line version). It has no callers left.

### Sub-step D: Verify and commit

- [ ] **Step 7: Run the surgery test suite**

Run: `pytest src/qldpc/codes/surgery/_test.py -x -q`
Expected: all tests pass.

- [ ] **Step 8: Run the Cain Table III regression**

Run: `python examples/scripts/cain_bb18_resource_exact_match.py`
Expected: `(Qubits=39, X-checks=20, Z-checks=20)`.

- [ ] **Step 9: Run a short Cain Fig 1b sanity check**

Run: `python examples/scripts/cain_fig1b_full_protocol.py --p-values 0.001 --shots 1000` if those flags are supported; otherwise just `python examples/scripts/cain_fig1b_full_protocol.py` and abort after the first noise point completes.

(If the script doesn't support short runs and a full run would take >10 minutes, skip this step — Task 7's full verification covers it.)

Expected: script runs without crashes; LER number printed at p=0.001 is within order-of-magnitude of pre-refactor (this is a smoke test, not a regression test).

- [ ] **Step 10: Commit**

Run: `git add src/qldpc/codes/surgery/cheeger.py src/qldpc/codes/surgery/gadget.py`

Then:

```bash
git commit -m "$(cat <<'EOF'
refactor: dedupe HX assembly between gadget.py and cheeger.py

- Extract _assemble_HX_L1 into gadget.py (~15 LOC pure function over
  numpy GF(2) arrays).
- _step3_assemble uses it for both basis=X and basis=Z (dual) paths.
- _reassemble_gadget_with_new_F uses it after boost augments F.
- Delete cheeger.py:_assemble_merged_HX (now redundant, single call
  site replaced).

cain_bb18_resource_exact_match still prints (39, 20, 20).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Doc cleanup (Step 3 in spec)

**Files:**
- Modify: `examples/logical_error_rates/_9_lattice_surgery_cain_fig1b_source.py:91-94`

- [ ] **Step 1: Read the comment context**

Run: `sed -n '85,100p' examples/logical_error_rates/_9_lattice_surgery_cain_fig1b_source.py`
Expected output: a comment block mentioning `layout.hx_row_kind` (line 92) and `layout.qubit_layer` (line 93).

- [ ] **Step 2: Delete the stale references**

The comment refers to `SurgeryLayout` attributes that no longer exist after Task 1. Use the Edit tool to replace the dead references. The original block reads roughly:

```python
# merged H_X tagged ``"ancilla_L1"`` in ``layout.hx_row_kind``. Because the
# gadget qubits κ_j (columns where ``layout.qubit_layer == 1``) are
```

Replace these two lines with a single concise comment that preserves the surrounding meaning. The simplest fix: rewrite to refer to GadgetLayout instead:

```python
# merged H_X chi rows (the last len(g.V0) rows) act on gadget qubits κ_j
# (columns at index ≥ g.code.num_qudits) and...
```

If the surrounding paragraph already adequately explains chi rows / κ qubits, the cleanest fix is to delete the two lines entirely. The implementer should read 5-10 lines of surrounding context to decide.

- [ ] **Step 3: Verify the file still parses (if it's executable Python)**

Run: `python -c "import ast; ast.parse(open('examples/logical_error_rates/_9_lattice_surgery_cain_fig1b_source.py').read())"`
Expected: no output (parse success).

- [ ] **Step 4: Commit**

Run: `git add examples/logical_error_rates/_9_lattice_surgery_cain_fig1b_source.py`

Then:

```bash
git commit -m "$(cat <<'EOF'
docs: drop stale qubit_layer / hx_row_kind comment in cain_fig1b source

These SurgeryLayout attributes were removed in the L=1 collapse refactor.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification

- [ ] **Step 1: Full surgery test suite, verbose**

Run: `pytest src/qldpc/codes/surgery/_test.py -v`
Expected: all tests pass.

- [ ] **Step 2: Cain Table III**

Run: `python examples/scripts/cain_bb18_resource_exact_match.py`
Expected: `✓ EXACT MATCH with Cain Extended Data Table III` + `(Qubits=39, X-checks=20, Z-checks=20)`.

- [ ] **Step 3: Confirm grep counts**

Run: `grep -c 'num_layers\|_LayeredBlocks\|hx_row_kind\|hz_row_kind\|qubit_layer\|_assemble_merged_HX\|_assemble_merged_HZ' src/qldpc/codes/surgery/cheeger.py`
Expected: `0`.

- [ ] **Step 4: Confirm LOC reduction**

Run: `wc -l src/qldpc/codes/surgery/cheeger.py src/qldpc/codes/surgery/gadget.py`
Expected: cheeger.py ≈ 795 (±20); gadget.py ≈ 232 (was 222, +10).

- [ ] **Step 5: Confirm git log shape**

Run: `git log --oneline -4`
Expected: three commits at the top — Task 3 (doc), Task 2 (dedupe), Task 1 (collapse), and the design-spec commit `aa293b5` below.

---

## What is NOT in this plan

- No new tests (existing tests cover the success criteria; refactor preserves behaviour).
- No `SurgeryLayout` deletion — the gadget↔legacy bridge still needs it.
- No basis=Z native handling in the boost — the HX/HZ swap hack in `_gadget_to_legacy_layout` stays.
- No `cheeger.py` line-budget enforcement — the prior LOC budget spec explicitly excluded `cheeger.py`.
