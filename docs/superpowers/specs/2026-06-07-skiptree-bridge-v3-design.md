# SkipTree-Based Joint Bridge (v3) — Design Spec

**Status:** Draft 2026-06-07 · supersedes v2 path-graph bridge
**References:**
- Ide / Swaroop et al. arXiv:2410.03628 (§III SkipTree, §IV Lemma 10, §VII B BB-LP, §VII C BB-BB, Appendix VIII Algorithm 2)
- Webster (Cross et al.) arXiv:2407.18393 §II.A 3-step layered ancilla
- Ide Zenodo supplementary data DOI:10.5281/zenodo.17527545 (`data_qLDPC_surgery.zip`)

---

## 1. Goal

Implement a joint-measurement code construction that reproduces the **stabilizer group** of Ide's two paper examples:

| Example | Source | Deformed parameters |
|---|---|---|
| BB-LP inter-code Z̄_1 Z̄_2 | §VII B / Table III | [[355, 25, 10]], max X-wt 8, Z-wt 7 |
| BB-BB intra-code Z̄_1 Z̄_3 (overlap on {17, 35}) | §VII C / Table IV | [[150, 5, 12]], max X-wt 8, Z-wt 6 |

"Stab-group equivalence" means: same H_X and H_Z **row spans** over the same column set. The literal row basis may differ from Ide's `.mtx` files (which depend on Ide's random spanning-tree seed).

### 1.1 Out of scope
- Cain Table III bb_18 Processor / lp_20 Processor — single-logical Cross §III layered ancilla, separate effort.
- Cellulation tuning, Cheeger boosting — already exist, used as-is.

---

## 2. Success criteria

A v3 implementation is correct iff all of the following hold:

| Bar | Verification |
|---|---|
| All 81 existing Webster single-logical tests pass | `pytest src/qldpc/codes/surgery_test.py` |
| BB-LP joint: n = 355, k_joint = 25 | rank(HX) + rank(HZ) + k = n |
| BB-LP joint: max stab weight ≤ 8 (X), ≤ 7 (Z) | row sums of joint H_X, H_Z |
| BB-LP joint: distance = 10 | BP+OSD or CPLEX minimum-weight logical search |
| BB-BB joint: n = 150, k_joint = 5, max-wt ≤ 8/6, distance = 12 | same |
| **Joint observable in row span**: X̄_1 X̄_2 ∈ span(HX_joint) | linear-algebra membership test over F_2 |
| **Singletons NOT in row span**: X̄_1 ∉ span(HX_joint), X̄_2 ∉ span(HX_joint) | same |
| CSS commutation: HX_joint · HZ_joint^T = 0 | matrix product mod 2 |

---

## 3. Architecture

### 3.1 Module split

Current `src/qldpc/codes/surgery.py` (1672 lines) splits into a sub-package:

```
src/qldpc/codes/surgery/
  __init__.py        # public API — re-exports for backwards compat
  layered.py         # Webster L-layer (build_layered_surgery_code) — unchanged logic
  skiptree.py        # _skip_tree (Alg 1) + _skip_tree_hr (Alg 2 flag-based)
  cellulation.py     # _cellulate_long_cycles
  cheeger.py         # boost_gadget_cheeger* + boost_gadget_distance
  joint.py           # NEW: SkipTree adapter (Lemma 10) — replaces v2 bridge
  port.py            # NEW: set-valued port for §VII C overlap
```

Re-exports preserve `from qldpc.codes.surgery import _skip_tree` (used by tests).

### 3.2 Data flow

```
                   ┌──────────────────────┐
data_code_1, op1 ─►│ Phase 1              │
data_code_2, op2 ─►│   build_layered_…    │─► gadget1 (HX1, HZ1, layout1)
                   │   per gadget         │─► gadget2 (HX2, HZ2, layout2)
                   └──────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ Phase 2              │
                   │   build aux graph G_s│
                   │   spanning tree      │─► T_1, P_1, H_R^{(w_1)}
                   │   SkipTree (Alg 1/2) │─► T_2, P_2, H_R^{(w_2)}
                   └──────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │ Phase 3              │
                   │   Lemma 10 stitch    │
                   │   extend χ, Z rows   │─► (HX_joint, HZ_joint)
                   │   adapter X-checks   │
                   └──────────────────────┘
```

---

## 4. Detailed construction

### 4.1 Auxiliary graph (Webster convention)

For each gadget s ∈ {1, 2}:
- `V_0_s = supp(op_s)` — set of data qubit indices.
- `C_0_s` = Z-checks of `data_code_s` touching V_0_s (with weight exactly 2 in V_0_s, for the simple-graph case; weight > 2 handled via cellulation as in existing code).
- `F_s` ∈ F_2^{|C_0_s| × |V_0_s|}: row c is the indicator of `supp(c) ∩ V_0_s`.
- `G_s` graph: vertices = V_0_s, edges = rows of F_s. Each edge corresponds to one κ_s ancilla qubit (Layer-1 C_1 in Webster).

### 4.2 SkipTree (Ide §III, Alg 1 + Appendix VIII Alg 2)

Apply `_skip_tree_hr(S_s)` where S_s = minimum spanning tree of G_s. Returns:
- `T_s` ∈ F_2^{(w_s − 1) × |E_{S_s}|}, (3, 2)-sparse (Theorem 7).
- `P_s` ∈ F_2^{w_s × w_s} permutation; `label_s[v] = l` iff P_s[v, l] = 1.
- Identity: `T_s · G_s · P_s = H_R(w_s)` where w_s = |V_0_s| and H_R is the canonical (w−1)×w parity check `[1,1,0…], [0,1,1,0…], …`.

Algorithm 2 differs from Algorithm 1 only by the `skip` flag toggling — needed because Alg 1 produces H_C (cyclic) but we want H_R (open path). On Hamilton-path G this yields the optimal (1,1)-sparse T.

### 4.3 Bridge

`w = min(w_1, w_2)`. Allocate `w` new data qubits `b_0, …, b_{w−1}` (the "adapter qubits A" of Lemma 10).

For each l ∈ {0, …, w − 1}:
- `v_l_1 = label_1^{−1}(l)` ∈ V_0_1
- `v_l_2 = label_2^{−1}(l)` ∈ V_0_2
- Bridge qubit `b_l` is conceptually an edge between v_l_1 and v_l_2 — physically a single new ancilla data qubit.

### 4.4 Adapter X-checks (Lemma 10, Eq 19 middle block)

`(w − 1)` new X-stabs. Row c, c ∈ {0, …, w−2}, has support:

| Column zone | Pattern |
|---|---|
| data (both codes) | 0 |
| κ_1 (Layer-1 of gadget 1) | T_1[c, :] (re-indexed by edge order of G_1) |
| κ_2 | T_2[c, :] |
| bridge b_0…b_{w−1} | H_R[c, :] = X on b_c, X on b_{c+1} |

### 4.5 χ row extension (HX)

Each Webster χ row at vertex v ∈ V_0_s carries `X_{data_v} + Σ_{e∋v in G_s} X_{κ_s[e]}`. We extend with **one** bridge X bit:

```
χ^(s)(v) → χ^(s)(v) + X_{b_{label_s[v]}}
```

This breaks `Σ_v χ^(s)(v) = X̄_s` (sum becomes `X̄_s + Σ_v X_{b_{label_s[v]}} = X̄_s + Σ_l X_{b_l}`), so X̄_s alone is **not** in HX row span.

Sum across both gadgets:

```
Σ_{s,v} χ^(s)(v) = X̄_1 + X̄_2 + 2 Σ_l X_{b_l} = X̄_1 + X̄_2 = X̄_1 X̄_2  (mod 2)
```

So **X̄_1 X̄_2 ∈ HX row span**. ✓

### 4.6 Webster Z-row extension (HZ)

Each Webster Z row at original Z-check c ∈ C_0_s carries `Z on supp(c) + Z_{κ_s[c]}`. To make the adapter X-check commute, extend with `Z` on bridge pattern `b_c` ∈ F_2^w that solves

```
H_R^T · b_c = T_s[:, c]
```

H_R has rank w − 1, so this system is consistent iff the column sum of T_s on c is even — true because T_s is a basis (row reduction of an incidence matrix). The solution is unique up to adding the all-ones vector; pick the unique solution with `b_c[0] = 0`:

```
b_c[l] = Σ_{j < l} T_s[j, c]   (running XOR)
```

For Webster Z-rows outside C_0_s (data Z-checks not touching V_0): unchanged.

### 4.7 χ–Z compatibility lemma (load-bearing)

For CSS commutation we need χ^(s)(v) extended and Z^(s)[c] extended to commute:

```
⟨χ^(s)(v), Z^(s)[c]⟩ = F[c, v] (data) + F[c, v] (κ_s) + α_v · b_c (bridge) = α_v · b_c (mod 2)
```

where α_v ∈ F_2^w is the single-bit indicator `e_{label_s[v]}`. So we need `α_v · b_c = b_c[label_s[v]] = 0` for all v ∈ V_0_s, c ∈ C_0_s.

**This is the key technical claim.** Concretely:

```
b_c[label_s[v]] = Σ_{j < label_s[v]} T_s[j, c]
```

The lemma asserts this sum is 0 for all (v, c). We prove this via:
- `T_s · G_s · P_s = H_R` (Theorem 7)
- so `T_s · G_s = H_R · P_s^T` and `(T_s · G_s)[c, v] = H_R[c, label_s[v]]`
- For c < label_s[v]: `H_R[c, label_s[v]] = δ_{c, label_s[v]−1}` — supports the running-XOR identity
- A direct cell-by-cell verification gives `b_c[label_s[v]] = (G_s · indicator(non-tree path))` reducing to 0 by spanning-tree property

**Verification path:** since the lemma is non-trivial, the implementation plan should include a numerical sanity check on BB-LP (compute α_v · b_c for all pairs, assert zero) BEFORE building the full joint matrix. If the lemma fails for some G (theoretically possible if SkipTree's path-vs-cycle structure misbehaves), the construction falls back to per-vertex bridge solving (extend chi χ^(s)(v) with a pattern α_v ∈ F_2^w that solves `H_R · α_v = e_{label_s[v]}`, plus per-row b_c).

### 4.8 Original code X and Z stabs

Both original codes' X-stabs and Z-stabs are kept **without bridge extension**. They commute with adapter trivially:
- Original X-stab S has zero overlap with χ extensions on bridge.
- Original Z-stab T has data support only (and possibly κ extension as in current Webster). Adapter has no data, so overlap on data is 0. κ-overlap follows existing Webster invariant.

### 4.9 Set-valued port (§VII C overlap, Theorem 11)

For intra-code joint with `supp(op_1) ∩ supp(op_2) ≠ ∅`:

```python
@dataclass(frozen=True)
class SetValuedPort:
    qubit_to_gadgets: dict[int, list[int]]  # data_qubit q → [gadget indices using q]
```

For each shared qubit q ∈ supp(op_1) ∩ supp(op_2):
- Gadget 1's V_0_1 contains q. Gadget 2's V_0_2 contains q.
- Gadget 1 creates κ_1 ancilla and chi/Z rows for q. Gadget 2 does the same independently.
- The data qubit q hosts **two** vertex Z-checks (one per gadget).
- Their product gives `Z_q² · (κ_1 ext) · (κ_2 ext) = (κ_1 ext) · (κ_2 ext)` — a pure ancilla stab.

Implementation effect: the joint HZ contains BOTH gadgets' vertex Z-stab rows even for shared q. The shared data column has degree 2 in HZ at the V_0 row positions. Webster Lemma 25 ensures this is consistent.

For Ide §VII C (Z̄_1 wt 14 ∩ Z̄_3 wt 12 = {17, 35}): qubits 17 and 35 appear in both V_0_1 and V_0_3. The joint code has w_1 = 14 labels for G_1 and w_3 = 12 labels for G_3. Bridge size `w = min(14, 12) = 12`.

---

## 5. Public API

```python
def build_joint_measurement_code(
    data_code_1: CSSCode,
    op1: ArrayLike,
    data_code_2: CSSCode,
    op2: ArrayLike,
    *,
    num_layers: int = 1,
    spanning_tree_seed: int = 0,
    validate: bool = True,
) -> tuple[CSSCode, JointSurgeryLayout]:
    """Construct merged code measuring op1 · op2 jointly.

    For inter-code joint (BB-LP example): pass distinct data_code_1, data_code_2.
    For intra-code joint (BB-BB example): pass data_code_1 == data_code_2.
    Overlap on data qubits handled via SetValuedPort (§4.9).
    """
```

`JointSurgeryLayout` retains v2's public attributes for API compatibility but `u_b_check_kind_mask` is reinterpreted as "rows that are adapter X-checks (not path stabs)". Old v2 path-bridge fields (`gadget1_bridge_qubit_idx` etc.) are dropped.

### 5.1 Breaking changes from v2

- `op1, op2` are now passed at separate positions with `data_code_1, data_code_2` between them (was: single `data_code, op1, op2`). Single-code callers: pass `data_code, op1, data_code, op2`.
- `_build_bridge_via_skiptree` and `_stitch_gadgets_with_bridge` are removed (internal).
- `_BridgeSpec` is removed.

---

## 6. Tests

### 6.1 New joint tests (`surgery_test.py`)

```python
def test_joint_BB_LP_inter_matches_ide_table_iii():
    bb = build_bb1()
    lp = build_lp2()
    z1_support = bb_z1_canonical_support()        # Ide's wt-14 representative
    z2_support = find_lp2_z2_equivalent_rep()     # search-based, deterministic seed
    merged, layout = build_joint_measurement_code(bb, z1_support, lp, z2_support)
    assert merged.num_qubits == 355
    assert merged.dimension == 25
    assert max_stab_weight(merged.matrix_x) <= 8
    assert max_stab_weight(merged.matrix_z) <= 7
    assert is_in_row_span(joint_observable(z1, z2, 98), merged.matrix_x)
    assert not is_in_row_span(z1_padded, merged.matrix_x)
    assert not is_in_row_span(z2_padded, merged.matrix_x)
    assert min_weight_logical_bp_osd(merged) == 10

def test_joint_BB_BB_intra_matches_ide_table_iv():
    bb = build_bb1()
    z1, z3 = bb_z1_canonical_support(), bb_z3_canonical_support()
    assert set(np.flatnonzero(z1)) & set(np.flatnonzero(z3)) == {17, 35}
    merged, layout = build_joint_measurement_code(bb, z1, bb, z3)
    assert merged.num_qubits == 150
    assert merged.dimension == 5
    assert max_stab_weight(merged.matrix_x) <= 8
    assert max_stab_weight(merged.matrix_z) <= 6
    assert is_in_row_span(joint_observable(z1, z3, 0), merged.matrix_x)
    assert min_weight_logical_bp_osd(merged) == 12

def test_chi_z_compatibility_lemma_on_bb_lp():
    """Verify §4.7 numerically: α_v · b_c = 0 for all (v, c) on BB-LP G_1, G_2."""
    # ... explicit computation, asserts identity
```

### 6.2 Preserved Webster tests

All 81 single-logical tests in `surgery_test.py` keep passing. Tests that referenced v2 `JointSurgeryLayout.bridge_qubit_slice` etc. (~5 tests) are dropped or rewritten against v3 layout.

---

## 7. Reference data integration (Ide Zenodo)

The Zenodo bundle `data_qLDPC_surgery.zip` contains:
- `BB_98_6_12/{original_codes,deformed_codes}/*.mtx` — original BB and Ide's deformed BB single-logical matrices
- `LP_200_20_10/{original_codes,deformed_codes}/*.mtx`
- `BB_98_LP_200_adapter/*.mtx` — Ide's joint BB-LP HX/HZ
- `BB_98_intracode_adapter/*.mtx` — Ide's joint BB-BB HX/HZ
- `*/skipTree_transformations/*.txt` — Ide's exact T, P, G matrices (Python array literals)

Approach: tests load Ide's HX/HZ as ground truth and assert **stab-group equality** (row span equality) against our generated joint code on the same qubit register.

`build_joint_measurement_code` accepts an optional `skiptree_override` parameter:

```python
@dataclass(frozen=True)
class SkipTreeOverride:
    G_s: tuple[np.ndarray, np.ndarray]   # incidence matrix per gadget
    T_s: tuple[np.ndarray, np.ndarray]
    P_s: tuple[np.ndarray, np.ndarray]
```

When provided, the function uses Ide's exact T, P, G matrices (loaded from `BB_98_6_12_Z_1_GTP.txt` etc.). When `None` (default), it generates its own via spanning-tree + SkipTree. Either path must produce a joint code whose row span equals Ide's; the override is for byte-exact reproducibility of Ide's specific basis when desired.

Storage: place the .mtx and .txt fixtures under `tests/fixtures/ide_zenodo/` (~35 KB total). Tests gate on file presence so non-fixture envs skip cleanly.

---

## 8. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| χ–Z compatibility lemma (§4.7) does not hold for some G | Medium | Numerical pre-check before stitching; fallback to per-vertex α_v bridge solving |
| Algorithm 2 port is subtle (skip flag toggling, root selection) | Medium | Cover with focused unit tests: path graph → (1,1)-sparse T, small cycle → match Alg 1 |
| Set-valued port (§4.9 for §VII C) needs per-qubit accounting | Medium | Implement disjoint case first; ship §VII B; tackle §VII C as a follow-up if needed |
| Module split breaks downstream imports | Low | `__init__.py` re-exports everything previously public |
| Webster single-logical tests fail due to import path changes | Low | Run full test suite after each module move; commit per file |

---

## 9. Open questions

None — all earlier clarifying questions resolved in the brainstorming session:
- Scope: §VII B + §VII C (both required)
- v2 fate: fully replace
- Test bar: Webster single-logical green + 2 new Ide joint tests
- Equality level: stab-group equality (not byte-identical mtx)
- Ide GTP files: optional `skiptree_override` parameter loads them for byte-exact reproduction; default generates our own T/P/G; tests assert stab-group equality either way
