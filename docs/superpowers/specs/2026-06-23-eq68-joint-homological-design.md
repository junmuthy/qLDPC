# Design: Logical-Y / mixed-Pauli measurement per Eq (68), general |W|, on a BB code

**Date:** 2026-06-23 (rev. 3, 2026-06-24)
**Status:** approved (design), pending spec review
**Scope:** end-to-end, BB `[[36,8,4]]` fixture, **general overlap `|W|` (computed,
not assumed)** — exercise both `|W|=1` and `|W|≥2`.

## Source of truth

1. Ide, Gowda, Nadkarni, Dauphinais, *Fault-tolerant logical measurements via
   homological measurement*, arXiv:2410.02753 — **§III.C** (FT protocol),
   **§III.D / `eq:joint_final` = "Eq (68)"** (run the homological measurement
   separately for `X̄`/`Z̄`, then merge anticommuting stabilizer pairs that
   overlap on a single qubit).
2. Repo `docs/superpowers/docs/main.tex` **§4 "Single logical mixed-Pauli PPM:
   system merge"** — the faithful rendering of Eq (68) this design implements,
   in the `π`/`∂_1`/`∂_0` convention, **general `|W|`**.

## Core principle: compute `W`, don't assume it

Given representatives `x` (`H_Z x = 0`) and `z` (`H_X z = 0`) for the same
logical qubit (`Ȳ = iX̄Z̄`), the support splits (main.tex §4.1):

```
V_X = supp(x)\supp(z)  (acts X)    V_Z = supp(z)\supp(x)  (acts Z)
W   = supp(x)∩supp(z)  (acts Y) = the physical Y-qubits, computed
```

`|W|` is odd for an anticommuting same-qubit pair. The construction handles **any
`|W|`**; the two regimes differ only in `∂_0 = ker(merged ∂_1)`:

- **`|W|=1`** (wedge of two graphs at one vertex): no crossing cycles, so
  `∂_0 = ker ∂_1^x ⊕ ker ∂_1^z = G_x ⊕ G_z` (two **pure-CSS** rows: Z-type on
  κ_X, X-type on κ_Z). Verified on Steane and on BB qubit 0.
- **`|W|≥2`**: `|W|−1` genuine **crossing cycles**, each a single **non-CSS**
  row carrying Z on κ_X and X on κ_Z (main.tex §4.4: "these two halves are the
  same stabilizer row, not two separate checks"). Verified on BB `[[36,8,4]]`: a
  `|W|=3` representative yields 1–2 crossing cycles.

So `∂_0` is computed as `ker(merged ∂_1)` in **all** cases — the `|W|=1`
pure-CSS split is just the degenerate output, not a special code path.

## Eq (68) transcribed (symplectic, X-block | Z-block)

Columns each side: `[ data (n) | κ_X | κ_Z ]` (κ_X = `X̄`-measurement ancillas,
κ_Z = `Z̄`-measurement ancillas).

```
             |  data         κ_X            κ_Z   |  data        κ_X         κ_Z
  H_X-checks |  H_X           ·          f_0^(Z)  |   ·           ·           ·
  V_X    χ   |  f_1^(X\Z)ᵀ   ∂_1^(X\Z)ᵀ     ·     |   ·           ·           ·
  W   Y-merge|  f_1^(Y|x)ᵀ   ∂_1^(Y|x)ᵀ     ·     |  f_1^(Y|z)ᵀ   ·       ∂_1^(Y|z)ᵀ
  H_Z-checks |   ·            ·             ·     |  H_Z        f_0^(X)       ·
  V_Z    χ   |   ·            ·             ·     |  f_1^(Z\X)ᵀ   ·       ∂_1^(Z\X)ᵀ
  ∂_0 cycles |   ·            ·          ∂_0^(Z)  |   ·         ∂_0^(X)       ·
```

Row-6 (main.tex line 903): `∂_0^(Z)` is X-type on κ_Z, `∂_0^(X)` is Z-type on
κ_X. A single `∂_0` row may populate both halves (crossing cycle ⇒ non-CSS).
The merged incidence is (main.tex §4.4, rows `V_X⊔W⊔V_Z`, cols `(κ_X | κ_Z)`):

```
∂_1 = [ ∂_1^x|_{V_X}   0          ]
      [ ∂_1^x|_W       ∂_1^z|_W    ]      ∂_0 = ker ∂_1 = (∂_0^(X) | ∂_0^(Z))
      [ 0              ∂_1^z|_{V_Z}]
```

## Distance mechanism (main.tex §4.7 / arXiv:2410.02753 §III.D)

Per-system: boost each of `∂_1^x`, `∂_1^z` to Cheeger `≥ 1`
(`cheeger.boost_gadget`); then any error `e = e_X + e_Z` has
`wt(e) ≥ min(wt e_X, wt e_Z)`, and the single-gadget distance bound applies to
each part independently — preserving `d` for **any `|W|`**. Crossing cycles
(`|W|≥2`) are needed not for this bound but to **remove the spurious low-weight
logicals the merge creates** (so `∂_0 = ker(merged ∂_1)` must include them).

**Why not Steane, why not `|W|=1`-only:** Steane's Ȳ-merge gives `k=0` (no
logical left, distance `nan`) — a degenerate witness; its old `xfail` "fault
distance = 1" is a `k=0`/`d=3` circuit-readout artifact, misattributed to a
missing §3.7 bridge. BB `[[36,8,4]]` keeps `k=7` at `d=4` and lets the boost do
real work. **No SkipTree / no §3.7 bridge** in this path (arXiv:2410.03628
removed) — a hard user constraint.

## Components

### Component 1 — `build_y_gadget`, general `|W|` (`y_gadget.py`)

- **Overlap, computed:** replace the `|W|=1`-only `_locate_overlap` with
  `_locate_overlaps(code, x, z) -> tuple[int,...]` returning the full
  `W = supp(x)∩supp(z)` (validates `x`/`z` are valid anticommuting logicals;
  no size restriction).
- **Per-system Cheeger boost (§4.7):** boost `g_x`, `g_z` to Cheeger `≥ 1`
  before merging.
- **Merge all of `W`:** `apply_mixed_basis_merge(..., merge_qubits=W)` fuses each
  χ_X@v / χ_Z@v into a `y_v` mixed row (it already accepts multiple qubits).
- **`∂_0 = ker(merged ∂_1)`, general:** build the merged incidence (above) from
  the **boosted** `∂_1^x`/`∂_1^z` and `W`; compute `∂_0 = ker(merged ∂_1)`
  (`galois` null space); embed each `∂_0` row as a symplectic row — κ_X support
  → Z-part on κ_X, κ_Z support → X-part on κ_Z (pure-CSS when the cycle lives in
  one system, non-CSS for a crossing cycle). **This replaces the current
  per-system `gauge_x_emb`/`gauge_z_emb` blocks**, which are correct only at
  `|W|=1`.
- **Drop SkipTree:** remove the `bridge` field, `build_bridge`/`Bridge` import,
  and §3.7 narrative.
- `obs0`: keep the `_ybar_obs0_rows` GF(2) solver (§III.C step 4: product of new
  stabilizers = `L`); re-cite to arXiv:2410.02753 §III.C, drop the §3.2 framing.

### Component 2 — BB fixture(s) (`y_gadget.py`)

- `_bb_y_pair(overlap=1)`: `BBCode({x:3,y:6}, x³+y+y², y³+x+x²)` (`[[36,8,4]]`).
  `overlap=1` → qubit-0 canonical reps. `overlap>=3` → a representative pair
  found by adding stabilizer rows so `|W|=3` with both gadget graphs cyclic
  (cache the concrete vector to keep the fixture deterministic; the search is in
  `examples/`-style helper, not run in tests).
- Keep `_steane_y_pair` only if still referenced elsewhere.

### Component 3 — circuit synthesis (`circuit.py`)

- `build_single_y_ppm_circuit` consumes `yg.merged_code` via
  `_split_quditcode_into_virtual_cssc`. Pure-CSS `∂_0` rows (`|W|=1`) flow
  through X/Z-phase scheduling unchanged. **Non-CSS crossing-cycle `∂_0` rows
  (`|W|≥2`) are mixed** and must be measured like `y_v` (mixed CX/CZ syndrome,
  MX y-ancilla). Verify the family-split routes them to the mixed-row path; if
  it only recognises the q0 `Y_stab` rows, extend it to treat **every** non-CSS
  row uniformly.
- Update docstrings/citations to arXiv:2410.02753 §III.C; remove Remark 23 /
  bridge narrative.

### Component 4 — tests (`y_gadget_test.py`, `circuit_single_y_test.py`)

- `y_gadget_test.py`: **parametrize over `|W| ∈ {1, 3}`** on the BB fixture.
  Assert: merged `k = 7`; distance not collapsed below `d_data = 4`; each system
  Cheeger `≥ 1` post-boost; `Ȳ` in the stabilizer center; rows commute; **no
  `bridge` field**; and for `|W|=3` that `∂_0` contains `|W|−1 = 2` **non-CSS**
  crossing-cycle rows (and `0` for `|W|=1`).
- `circuit_single_y_test.py`: BB fixture, both `|W|`; circuit compiles to a DEM;
  noiseless Ȳ readout deterministic; **measurement-fault-distance check** — no
  low-weight `error(p) L0` DEM term flips obs0 without firing a detector (the
  acceptance the Steane `k=0` fixture could not witness). Verified empirically;
  any failure is a circuit-readout bug fixed here, not a missing bridge.
- All citations unified to arXiv:2410.02753 + main.tex §4; remove
  arXiv:2407.18393 §3.7/§3.2 and arXiv:2410.03628 SkipTree from this path.

## Out of scope / unchanged (YAGNI)

- `gadget.py`, `cheeger.py`, `merge.py` implementations unchanged (used, not
  modified). `bridge.py` stays for other paths; this path stops importing it.
- The "matrices commute → merge any X-stab with any Z-stab" branch of §III.D
  (i.e. `X̄_i ⊗ Z̄_j` on **different** logical qubits, even `|W|`): deferred.
  This iteration covers `Ȳ = iX̄Z̄` on one qubit (odd `|W|`).

## Acceptance criteria

1. `build_y_gadget` computes `W` from `x`,`z` (any odd size) and returns a
   `YGadgetLayout` with **no `bridge` field**, `merged_code` `k = 7`, distance
   not collapsed below `d_data = 4` (confirm via `get_distance_exact`/tighter
   bound), `Ȳ = iX̄Z̄` in the stabilizer center, all rows commuting.
2. `∂_0 = ker(merged ∂_1)`: `0` crossing cycles at `|W|=1`, `|W|−1` non-CSS
   crossing cycles at `|W|≥2` (verified `|W|=3 → 2`); each system Cheeger `≥ 1`
   after boost.
3. `build_single_y_ppm_circuit` compiles to a DEM for both `|W|` regimes; the
   noiseless Ȳ truth table is deterministic.
4. Under a single-fault depolarizing model, **no** DEM term flips obs0 without a
   detector firing (measurement fault distance `> 1`), for both regimes; any
   shortfall documented as a circuit-readout issue with the offending chain.
5. No references to SkipTree / arXiv:2410.03628 §III, the §3.7 bridge, or
   arXiv:2407.18393 §3.2 remain in this path.
