# SkipTree Bridge Refactor: Roadmap

User-provided quote from Swaroop et al. arXiv:2410.03628 §VII B clarifies the
target structure for SkipTree-based bridges:

```
HX|supp(E_1) = T_1
HX|supp(E_2) = T_2
HX|supp(E_A) = H_R(14)
```

Where E_1, E_2 are the edge sets of G_1, G_2 (= κ_1, κ_2 ancilla data
qubits), E_A is the bridge edge set, and H_R(14) is the canonical
parity-check of the length-14 repetition code.

The adapter X-checks are at most weight 8 (T_1, T_2 are (3,2)-sparse from
SkipTree, plus H_R weight 2 from bridge = max 5+2 = 7-8).

## Current v2 Bridge (working, 81/81 tests pass)

`_build_bridge_via_skiptree` + `_stitch_gadgets_with_bridge`:

- **Chi extensions**: ONE chi per gadget extended with X on bridge endpoint
  - χ_0^(1) ⊗ X_{b_0}
  - χ_0^(2) ⊗ X_{b_{w-1}}
- **Bridge X-stabs**: w-1 path stabs X_{b_i} X_{b_{i+1}}
- **Bridge Z-stabs**: none (relies on Cross §3.6 protocol α* identity)

This works (CSS commutes, joint operator in stab, k_joint = k_data - 1)
but stabilizer weights are NOT bounded by SkipTree's (3,2)-sparse guarantee.

## Target SkipTree Bridge (Swaroop §VII B, requires significant work)

To implement Swaroop's `HX|supp(E_s) = T_s` formula:

1. Apply SkipTree to spanning_tree(G_1), spanning_tree(G_2) → T_1, T_2, P_1, P_2.
2. **Replace** original chi rows of both gadgets with linear combinations
   defined by T_1, T_2 (row transformation of HX).
3. Each NEW joint stab acts on κ_1, κ_2, AND bridge with combined weight ≤ 8.
4. No separate "path stabs" — the H_R(w) pattern on bridge is part of the
   joint stab structure (not standalone X-stabs).

### Why this is non-trivial

- T_1 transforms 13 of 14 chi rows (V-1 dim). The "leftover" chi row needs
  special handling.
- The combined stab structure must preserve:
  - CSS commutation (against data Z-stabs, κ-extended Z-stabs, gauge-fix Z)
  - Joint observable membership (X̄_1 X̄_2 ∈ row span)
  - Singletons NOT in row span (X̄_1, X̄_2 individually NOT stabs)
  - k_joint = k_data - 1
- Chi rows from gadget 1 originally satisfy Σ χ^(1) = X̄_1 (κ-cancellation).
  After T_1 transformation, this identity is preserved IF T_1 is invertible
  on the chi row space. But the new generators have different STRUCTURE.

## Swaroop §VII B exact construction (from re-reading carefully)

For inter-code joint measurement BB_1 Z̄_1 ↔ LP_2 Z̄_2, Swaroop Table III lists:

  Auxiliary graph: G_1 (V_1, E_1) ∼_A G_2 (V_2, E_2)
  No. edges (addnl data qubits): 57 = 23 (BB_1) + 20 (LP_2) + 14 (adapter)
  No. vertices (Z-checks):       28 = 14 + 14
  No. cycles (X-checks):         17 (incl 2 new from cellulation)
                                + 13 (adapter)
  Adapter total: 14 data qubits + 13 X-checks = 27
  Deformed code: [[355, 25, 10]]
  Max stab weights (X, Z): 8, 7
  Adapter check weights observed: [4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 7, 8]

The 13 ADAPTER X-CHECKS (Eqn 37) have support:
  HX|supp(E_1) = T_1   (κ_1 part, (3,2)-sparse from SkipTree)
  HX|supp(E_2) = T_2   (κ_2 part, (3,2)-sparse)
  HX|supp(E_A) = H_R(14)   (bridge part, path-like)

The complete X-stabilizer set per Swaroop:
  1. Original code X-stabs (some deformed onto E_1, E_2)
  2. X cycle checks in BOTH G_1 and G_2 (gauge-fix per gadget)
  3. NEW adapter X-checks (Eqn 37)

## Why naive "extend all chi" fails

If we just extend ALL chi rows with bridge X (not just chi_0), the
Cross §3.6 α* identity breaks:

Extending ALL chi rows (not just chi_0) gives:
  Σ χ^(1) = X̄_1 + Σ_l X_{b_l} (full bridge X, not just X_{b_0})
  Σ χ^(2) = X̄_2 + Σ_l X_{b_l}
  Sum + path stabs ≠ X̄_1 X̄_2 (extra bridge support remains)

Cross §3.6 α* formula no longer gives joint op cleanly.

## Why Swaroop's adapter stabs don't immediately commute

Swaroop's adapter stab l has support T_1[l, :] on κ_1. For this to commute
with the chi rows (vertex Z-stabs in joint code), we need:

  T_1[l, :] · F_1[:, v] = 0   for all v ∈ V_0_1

But T_1 G_1 P_1 = H_R(14), and F_1 ≈ G_1 means T_1 F_1 has 1s at positions
(l, v) where label[v] ∈ {l, l+1}. So adapter stab l anti-commutes with
chi rows at vertices with labels l and l+1.

This contradiction suggests that in Swaroop's construction, EITHER:
(a) Chi rows are also row-transformed (e.g., T_1 acts on chi rows too),
(b) Some chi rows are dropped from the joint code,
(c) The adapter stabs include data support that I'm missing from Eqn 37, OR
(d) Theorem 7 / Algorithm 2 in Appendix VIII handles this via different math.

Without reading Swaroop Appendix VIII carefully, I cannot definitively answer
which. The current `_skip_tree` port (eswaroop, MIT) doesn't include the
Algorithm 2 flag-based variant either.

## Recommendation

Keep v2 path-graph bridge. The SkipTree's T_1 G_1 P_1 = H_R(14) is verified
on the AUXILIARY GRAPH (`swaroop_skiptree_verification.py`), confirming the
relabeling property. The full row-transformation refactor is a separate
v3 effort that should:

1. Design the FULL HX after transformation (including new joint stabs).
2. Verify all four invariants (CSS, joint op in stab, singletons not in
   stab, dimension count).
3. Demonstrate weight-bounded stabilizers (≤ 8 per Swaroop Theorem 7).

Estimated effort: 1-2 days. Defer until needed.
