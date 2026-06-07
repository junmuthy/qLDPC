# Reverse-Engineering Ide arXiv:2410.03628 §VII B Construction

**Date:** 2026-06-07
**Source:** Zenodo `data_qLDPC_surgery.zip` BB-LP joint matrices
**Status:** Partial decode — BB-side rule found, LP-side and adapter pending

This document records the structural rules of Ide's joint construction
that are **NOT in the published paper text**. The paper describes Lemma 10
(adapter cycle basis) and Eq 37 (adapter X-checks) but does not give
explicit algorithms for assembling the full joint HX, HZ. These rules are
recovered from the supplementary `.mtx` files.

## 1. Port function (decoded)

For BB_1 Z̄_1 (V_0_1 weight 14) ↔ LP_2 Z̄_2 inter-code joint, Ide's specific
random spanning-tree choice gives:

```python
IDE_BB_LP_PORT = {
    # BB qubit (V_0_1) → label = LP qubit index used for cross-block Vl
    6: 0,   8: 1,   32: 2,  33: 3,  93: 4,   35: 5,   36: 6,
    41: 7,  17: 8,  37: 9,  13: 10, 50: 11,  51: 12,  31: 13,
}
```

The 14 LP qubits {0, 1, …, 13} are **NOT V_0_2** of LP Z̄_2 — they don't
commute with LP's HX. They are arbitrary qubits chosen by Ide as port targets.

## 2. Vl rows (cross-block Z-checks)

The joint H_Z contains **14 cross-block rows** with structure:

```
Vl[v] = Z_{data_v} (BB)
      + Σ_{e ∋ v in G_1 (cellulated)} Z_{κ_1[e]}
      + Z_{LP_qubit_at_label_1[v]}
```

This formula is Ide's Fig 7 "Vl" rows materialized. The paper text never
gives this explicit formula.

## 3. BB-side X-stab deformation rule (decoded, verified 49/49)

For each original BB X-stab S, let U = supp(S) ∩ V_0_1.

- |U| = 0: deformed S' = S (unchanged on κ_1)
- |U| = 2 with {v_a, v_b} ⊆ V_0_1: deformed S' = S + X on κ_1[edge(v_a, v_b)]
  where edge(v_a, v_b) is the unique κ_1 ancilla qubit for the G_1 edge
  between v_a and v_b
- |U| = 4, 6, …: requires perfect matching of U via edges in G_1 (not observed
  in BB Z_1 example since X-stab weight 6 in BB rarely gives |U| ≥ 4)

This rule is what makes the **deformed S' commute with Vl rows directly**:

```
S' · Vl[v_a]:
  data: 1 (v_a ∈ supp(S))
  κ_1: 1 (κ_1[edge(v_a, v_b)] is incident to v_a)
  total: 0 ✓
```

In Webster's gadget, original X-stabs are unchanged and chi rows handle
commutation. Ide takes a different path: **deform original X-stabs to
absorb the chi-row role**. The 14 chi rows are not separately materialized.

## 4. BB κ_1 edge listing (cellulated G_1)

After cellulation +2 edges, G_1 has 23 edges with κ_1 mapping:

```
κ_1[0]: (6, 50)     κ_1[1]: (6, 51)     κ_1[2]: (13, 93)
κ_1[3]: (13, 31)    κ_1[4]: (8, 32)     κ_1[5]: (8, 33)
κ_1[6]: (17, 35)    κ_1[7]: (36, 50)    κ_1[8]: (37, 51)
κ_1[9]: (17, 41)    κ_1[10]: (31, 32)   κ_1[11]: (32, 33)
κ_1[12]: (33, 93)   κ_1[13]: (6, 31)    κ_1[14]: (8, 50)
κ_1[15]: (41, 51)   κ_1[16]: (35, 41)   κ_1[17]: (35, 36)
κ_1[18]: (36, 37)   κ_1[19]: (13, 37)   κ_1[20]: (17, 93)
κ_1[21]: (33, 37)   κ_1[22]: (8, 13)
```

## 5. Still UNDECODED

| Piece | Status | Notes |
|---|---|---|
| LP-side X-stab deformation | UNDECODED | LP single Z_2 rows are NOT in joint HX row span (0/103) — Ide uses a different LP-side construction |
| LP-side Vr rows | UNDECODED | No HZ rows have (single LP_d + LP_κ + bridge) pattern |
| 32 bridge-touching HX rows | UNDECODED | LP-side, signature lp_d + lp_κ + bridge; purpose unclear |
| 13 cross-block HX rows | UNDECODED | bb_κ + lp_d, no bridge; analog of Vl on X-side? |
| 14 bridge cols 341..354 | PARTIAL | Used by LP-side rows only; not directly V_0_2 |

## 6. Status of v3 implementation

- ✓ BB cellulated single gadget produces row-span-equivalent code to Ide's BB Z̄_1 single
- ✓ LP cellulated single gadget produces row-span-equivalent code to Ide's LP Z̄_2 single
- ✓ n = 355 matches via cellulation
- ✗ Joint code stab group does NOT match (177 X-stabs vs 166 Ide, 153 Z-stabs vs 164 Ide)
- ✗ Adding decoded Vl rows breaks CSS commute (134 anti-commute pairs) because
  our Webster chi rows are incompatible with Vl

## 7. Path forward for full match

To replicate Ide's joint stab group exactly:

1. **Replace BB Webster chi rows with Ide-style deformed X-stabs** using the
   rule in §3 above. This makes 49 deformed X-stabs + 10 cycle X-checks =
   same structure as Ide BB single Z_1.
2. **Decode LP-side construction**. Likely a similar "deform original X-stabs
   to absorb chi role" + asymmetric bridge extension. Estimated 2-3 hours.
3. **Decode adapter X-check structure** (the 13 BB_κ + LP_data rows). These
   may be the actual Lemma 10 adapter rows after row-reduction.
4. **Decode bridge cols 341..354 purpose**.

Total remaining: ~4-5 hours of focused reverse engineering. The decoded
pieces above are durable artifacts even if the full decode isn't completed.

## 8. Why LP-side is impossible to derive Webster-style (deeper investigation)

After further analysis, the LP-side gap is **architectural**, not just a row
basis difference:

### Min-weight Z̄_2 of joint LP-side is weight 1

Computing the minimum-weight Z-logical of joint H_X restricted to LP data
cols (mod LP_original H_Z) yields a **weight-1 representative** at LP qubit
**10** — which is one of the Vl port targets {0..13}.

This means Ide's joint code does NOT preserve LP single Z̄_2 as a subcode.
Instead:
- The joint code's "logical Z̄_2 class" is collapsed to single-qubit Z on
  one of the port LP qubits via the cross-block Vl stabs.
- LP single Z̄_2's specific weight-14 representative is irrelevant to the
  joint construction.

### What Ide actually does on LP-side (interpreted)

Ide's LP gadget is **gauge-fixed within the joint context**. Original LP
X-stabs are deformed onto bridge qubits 341..354 (gaining bridge support
to commute with the Vl rows). The 14 LP qubits {0..13} serve as the
"adapter port" through which BB V_0_1 vertices connect.

The construction does NOT build an independent LP single Z_2 gadget. The
LP-side structure is integral to the joint code; it can't be projected out
to recover LP single Z_2's deformed code.

### Implications for our v3

A Webster-style construction (build_layered_surgery_code per gadget +
combine via bridge) **cannot match Ide's joint stab group**. Ide uses a
different construction philosophy:

> Build the joint code in one pass, with the gadget structure emerging
> from the combined HX/HZ, rather than composing pre-built single-gadget
> components.

For project users:
- **General inter-code joint** (works for any two codes): use
  `build_joint_measurement_code_intercode(code1, op1, code2, op2,
  cellulate=True)`. Produces [[n, k, d]] valid joint with k = k_1 + k_2 - 1.
  Not stab-group-equivalent to Ide.
- **Ide's specific paper code**: use `build_joint_from_ide_fixture("BB_LP")`
  to load the Zenodo `.mtx` directly. The only practical path to
  byte-equivalence with the paper.

To algorithmically reproduce Ide's construction in general, one would need
to derive the "deform original X-stabs to absorb the Vl/Vr role" algorithm
from scratch, requiring careful analysis of Lemma 10's structure plus
unwritten implementation details.
