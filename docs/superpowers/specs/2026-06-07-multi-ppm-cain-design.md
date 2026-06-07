# Multi-PPM Surgery (Cain Processor) — Design Spec

**Status:** Draft 2026-06-07
**References:**
- Cain et al. Nature arXiv:2603.28627, Extended Data Table III
- Webster (Cross et al.) arXiv:2407.18393 §II.A 3-step layered ancilla
- Ide / Swaroop et al. arXiv:2410.03628 §VII C set-valued port function

---

## 1. Goal

Reach EXACT match with three remaining rows of Cain Extended Data Table III:

| Cain Table III Row | Target (κ, χ, G) | Status now |
|---|---|---|
| `bb_18` Processor \|P̄\|=9 | (189, 104, 86) | not matched (multi-target needed) |
| `lp_20^{3,5}` Processor \|P̄\|=69 | (813, 460, 357) | not matched (multi-target needed) |
| `lp_24^{3,7}` Memory \|P̄\|=1 | (364, 208, 157) | not matched (weight-skip on rep search) |

Success criterion: `(κ, χ, G)` exactly equal to Cain's tuple after running the
pipeline, **plus** CSS commutation holds and each measured logical is in
HX row span.

## 2. Architecture

New module layout under `src/qldpc/codes/surgery/`:

```
surgery/
  multi.py     # NEW: build_multi_target_surgery_code (Webster on V_0 union)
  port.py     # extend: SetValuedPort implementation (was a stub from Ide §VII C spec)
  layered.py  # unchanged (single-logical Webster reused)
  cellulation.py, cheeger.py, skiptree.py, joint.py: unchanged
```

Data flow per call:

```
data_code, [op_1, …, op_t]
      │
      ▼
Validate: all op_i commute pairwise; same Pauli type
      │
      ▼
SetValuedPort: data_qubit q → list of logical indices that include q
      │
      ▼
V_0_union = binary vector with q ∈ V_0 iff q in any supp(op_i)
      │
      ▼
build_layered_surgery_code(data_code, V_0_union)   # Webster single-target pipeline
      │
      ▼
For each i ∈ {0..t-1}:
  chi_group_i = list of chi row indices whose vertices are in supp(op_i)
  Verify Σ_{r ∈ chi_group_i} merged.HX[r] ≡ op_i (mod 2)
      │
      ▼
return (merged_code, MultiSurgeryLayout)
```

## 3. Core API

```python
@dataclasses.dataclass(frozen=True, eq=False)
class MultiSurgeryLayout:
    """Layout for multi-PPM Webster gadget.

    Attributes:
        base_layout: SurgeryLayout from build_layered_surgery_code on V_0_union.
        logical_ops: list of original binary support vectors, length t.
        set_valued_port: SetValuedPort instance mapping data qubit → list of
            logical indices that include it.
        chi_group_per_logical: list of length t. chi_group_per_logical[i] is
            the list of chi row indices in merged.matrix_x whose sum modulo 2
            equals logical_ops[i]. For disjoint supports, chi groups partition
            V_0; for overlap, they share rows.
    """
    base_layout: SurgeryLayout
    logical_ops: tuple[np.ndarray, ...]
    set_valued_port: SetValuedPort
    chi_group_per_logical: tuple[tuple[int, ...], ...]


def build_multi_target_surgery_code(
    data_code: CSSCode,
    logical_ops: Sequence[npt.ArrayLike],
    *, num_layers: int = 1, validate: bool = True,
) -> tuple[CSSCode, MultiSurgeryLayout]:
    """Webster gadget measuring t commuting Pauli logicals simultaneously.

    Each logical_ops[i] must commute pairwise with the others (X-type with
    X-type, or Z-type with Z-type). k_joint = k_data - t (each logical consumes
    one DOF). Supports may overlap; SetValuedPort handles shared qubits.

    Args:
        data_code: stabilizer CSSCode.
        logical_ops: t binary support vectors of length data_code.num_qubits.
        num_layers: Webster L (odd, ≥ 1).
        validate: input sanity checks.

    Returns:
        (merged_code, MultiSurgeryLayout).
    """
```

## 4. Set-valued port semantics

For each shared data qubit q with q ∈ supp(op_i) ∩ supp(op_j):
- The Webster gadget builds chi rows per V_0 vertex (chi rows are
  independent of which logical they "belong to").
- The chi row at vertex q is in `chi_group_per_logical[i]` AND
  `chi_group_per_logical[j]`.
- Sum over chi_group[i] = X̄_i ∈ HX row span (becomes a stab).
- Sum over chi_group[j] = X̄_j ∈ HX row span.
- Both work simultaneously because chi rows themselves don't track logical
  ownership — they're per-vertex, and the grouping is bookkeeping.

The construction in Ide §VII C is the inspiration. We support both disjoint
and overlapping PPM supports via SetValuedPort. This is pure single-block
multi-PPM (no graph-side adapter; not the inter-code joint scenario).

## 5. Cain match pipeline

For each Cain Table III row, a separate script under `examples/scripts/`:

```python
def cain_processor_match(data_code, target_ppms, target_shape, max_seeds=1000):
    """Run multi-target gadget + Cheeger boost, sweep seeds for exact match."""
    merged, layout = build_multi_target_surgery_code(data_code, target_ppms)
    for seed in range(max_seeds):
        boosted, bl = boost_gadget_cheeger_combinatorial(
            merged, layout.base_layout, target_h=1.0, seed=seed,
        )
        if (bl.num_ancilla_qubits, _chi_count(bl), _gauge_count(bl)) == target_shape:
            return seed, boosted, bl
    raise RuntimeError(f"No seed in 0..{max_seeds} produced shape {target_shape}.")
```

Logical-rep search for `bb_18` (9 of 10 Z-logicals):
- Enumerate combinations C(10, 9) = 10 sets.
- For each set, reduce by stabs to find low-weight reps.
- Combined V_0 union should have size 104.

Logical-rep search for `lp_20` (69 of 148 Z-logicals): much larger.
- Use BB-code-style algebraic structure if possible.
- Or BP+OSD-based greedy selection.

## 6. Tests

```python
# src/qldpc/codes/surgery_test.py — new tests

def test_multi_target_disjoint_synthetic():
    """Two disjoint Z-logicals on Steane; k_joint = -1 (Steane has k=1)."""
    # skip — Steane has only 1 logical
    pass

def test_multi_target_on_bb_disjoint_pair():
    """Two disjoint Z-logicals on BB_18; k_joint = 8."""
    bb = build_bb18()
    z_ops = bb.get_logical_ops(Pauli.Z)
    # Find two disjoint Z-logical reps (Pauli Z, supp disjoint)
    op_pair = find_disjoint_z_pair(bb)
    merged, layout = build_multi_target_surgery_code(bb_dual, op_pair)
    assert merged.dimension == bb.dimension - 2
    # Each X̄_i in row span
    for op in op_pair:
        op_pad = pad_to_merged(op, merged.num_qubits)
        assert is_in_row_span(op_pad, merged.matrix_x)

def test_multi_target_with_overlap_set_valued_port():
    """Two Z-logicals on BB with overlapping supports; verify SetValuedPort."""
    bb = build_bb18()
    z1 = z_logical_with_supp_1(bb)
    z3 = z_logical_with_supp_3_overlap(bb, z1)   # shares qubits with z1
    overlap = set(np.flatnonzero(z1)) & set(np.flatnonzero(z3))
    assert overlap, "test setup wrong: need non-empty overlap"
    merged, layout = build_multi_target_surgery_code(bb_dual, [z1, z3])
    for q in overlap:
        owners = layout.set_valued_port.gadgets_for_qubit(q)
        assert sorted(owners) == [0, 1]
    # Both logicals in row span
    for op in [z1, z3]:
        assert is_in_row_span(pad_to_merged(op, merged.num_qubits), merged.matrix_x)

# examples/scripts/cain_bb18_processor_exact_match.py — script

def main():
    bb18 = build_bb18()
    target_shape = (189, 104, 86)
    # Find 9 logical reps whose union covers ~104 qubits
    target_ppms = find_9_ppms_with_v0_size(bb18, target_v0_size=104, target_weight_avg=12)
    seed, boosted, bl = cain_processor_match(bb18, target_ppms, target_shape)
    print(f"Found at seed {seed}: ({bl.num_ancilla_qubits}, ...)")
    assert (bl.num_ancilla_qubits, _chi_count(bl), _gauge_count(bl)) == target_shape

# Similar scripts for lp_20 Processor, lp_24 Memory.
```

## 7. Risk register

| Risk | Mitigation |
|---|---|
| Cain's 9 Z-logical reps may not be disjoint or have specific structure | SetValuedPort handles overlap; rep search supports overlap-aware reduction |
| Cheeger boost seed search may not reach exact (κ, χ, G) for some rows | Try multiple boost variants; expand seed range; document if not reachable as Cain's specific algorithmic choice |
| BP+OSD rep search expensive on lp_20 / lp_24 (k > 100) | Use BB-code algebraic group action where applicable; for LP code, use greedy + stab reduction |
| Set-valued port spec says nothing about Cain-style construction | We are implementing inter-PPM semantics (chi sum grouping), not adapter port (Ide use). Keep `SetValuedPort` minimal: just qubit → owner list. |

## 8. Out of scope

- v3 SkipTree-based multi-PPM (preserving (3,2)-sparse adapter): YAGNI, Cain uses path-graph
- Inter-code multi-PPM (PPMs across different codes): out of scope; existing inter-code joint handles t=1 inter-code case
- Mixed Pauli-type PPMs (some X-type, some Z-type with non-disjoint supports): YAGNI

## 9. Open questions

None — all clarifying answered in brainstorming:
- Scope: all 3 remaining Cain rows
- Overlap: implement set-valued port (Case B) alongside disjoint (Case A) in one go
- Success bar: exact (κ, χ, G) match + CSS commute
