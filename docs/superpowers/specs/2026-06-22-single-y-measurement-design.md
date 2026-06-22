# Single-Ȳ Measurement (§3.7 explicit mixed-check) — Design

- **Date:** 2026-06-22
- **Status:** Approved (design); pending spec review
- **Branch:** `feat/latticesurgery-mixedjoint`

## 1. Goal & scope

Implement a fault-tolerant logical **Y measurement** `Ȳ = iX̄Z̄` for a single logical
qubit of a CSS code, following the gauge-fixed Y ancilla system of Cross, He, Rall &
Yoder, "Improved QLDPC Surgery: Logical Measurements and Bridging Codes"
(arXiv:2407.18393), §3.7 — the construction that glues the X- and Z-measurement
systems of one logical qubit at their single overlap qubit `q₀` via a mixed-type
check `q₁`.

Deliverables:
- The merged stabilizer code in which `Ȳ` is the measured stabilizer.
- A stim syndrome-extraction circuit (split X/Z/Y schedule + Bell/flag cell for `q₁`).
- A **noiseless truth-table test** that the observable reads the `Ȳ` eigenvalue.

Validated on the **Steane [[7,1,3]]** code first (single logical qubit, `X̄`/`Z̄`
overlap = 1 qubit — the clean §3.7 case). The construction is general (any CSS code
with a single-overlap `X̄`/`Z̄` pair); gross-code tuning is out of scope this round.

**Hard constraint:** do NOT use the "apply a single-qubit Clifford that rotates the
local Pauli to Z" basis-change method (the existing `_rotate_x_side_to_z`
Hadamard-dual default). The non-CSS content lives explicitly in the mixed check `q₁`.

## 2. Background & references

- **Y construction:** Cross, He, Rall, Yoder, arXiv:2407.18393, §3.7 — mixed check
  `q₁` at the `X̄∩Z̄` overlap `q₀`; bridge qubits; Lemma 18 (gauge-check basis);
  Remark 19 (single-overlap assumption); §4.1 Bell/flag extraction of `q₁`.
- **Sparse bridge:** Swaroop, Jochym-O'Connor, Yoder, "Universal adapters between
  quantum LDPC codes" (arXiv:2410.03628), §III SkipTree (Theorem 7) + §II C
  cellulation — already implemented in `bridge.py`.
- **L=1 gadget + cross-merge:** Webster, Smith, Cohen (arXiv:2511.15989), §II.A
  (gadget) and §II.B.2 (mixed-basis cross-merge) — `gadget.py`, `merge.py`.

Why a mixed check is mandatory: `Ȳ` is mixed (XZ) on `q₀`, so the merged code cannot
be CSS (its X-part would have to be both an X-stabilizer and the logical `X̄`). The
non-CSS-ness is irreducible, lives in `q₁`, and is extracted with a Bell/flag gadget —
not basis-changed away.

## 3. Architecture / module layout

New isolated module `src/qldpc/circuits/surgery/y_gadget.py`:
- `YGadgetLayout` (frozen dataclass): `code`, `x`, `z`, `q0:int`, `g_x:GadgetLayout`,
  `g_z:GadgetLayout`, `bridge:Bridge`, `Y_stab:np.ndarray`,
  `H_sym:np.ndarray` (merged symplectic check matrix), `merged_code:QuditCode`,
  `ancilla_qubits`, `y_ancilla_qubits`, `obs0_xor_map:tuple[int,...]`.
- `build_y_gadget(code, *, x, z) -> YGadgetLayout`.

New circuit entry point in `circuit.py`:
- `build_single_y_ppm_circuit(yg, *, rounds, noise_model, data_init)` — sibling of
  `build_single_ppm_circuit`.

Public API: export `build_y_gadget`, `YGadgetLayout`, `build_single_y_ppm_circuit`
from `surgery/__init__.py`.

Reused unchanged: `build_gadget`, `build_gadget_augmented` (gadget.py);
`_build_aux_graph_strict`, `_cellulate_port_subgraph`,
`_run_skiptree_on_port_subgraph`, `Bridge` (bridge.py); `apply_mixed_basis_merge`
(merge.py); split X/Z/Y schedule + Y-row emission + `EdgeColoring` (circuit.py).

Factoring note: if the three bridge helpers are not cleanly importable, lift them to a
shared `surgery/_aux_graph.py` — a targeted, behavior-preserving refactor only if
needed.

## 4. Construction / data flow — `build_y_gadget`

1. **Validate & locate `q₀`.** `H_Z@x==0`, `H_X@z==0`, `x`/`z` anticommute (same
   logical qubit), and `|supp(x)∩supp(z)| == 1` → `q₀`. Overlap ≠ 1 → `ValueError`
   citing Remark 19 (out of scope).
2. **Build the two systems.** `g_x = build_gadget(code, x, basis=Pauli.X)`,
   `g_z = build_gadget(code, z, basis=Pauli.Z)`, same `code`, with disjoint κ ancilla
   ID ranges.
3. **Sparse bridge at `q₀`.** Build the auxiliary graph on each system's first-layer
   port, cellulate long cycles, run SkipTree to get the sparse adapter joining the X-
   and Z-system first layers (reuse the bridge helpers). Output: bridge qubits `B` +
   sparse gauge/cycle checks. ("Using cellulation and SkipTree.")
4. **Mixed check `q₁`.** `apply_mixed_basis_merge(HX, HZ, merge_qubits=(q₀,),
   adapter_cols=…)` fuses the χ_X-row and χ_Z-row anchored at `q₀` into one symplectic
   `Y_stab` row (`= q₁`) and removes those χ rows.
5. **Assemble merged code.** `QuditCode` from surviving X rows ⊕ Z rows ⊕ `Y_stab` ⊕
   bridge gauge checks. Assert: symplectic commutation `H_sym ⊙ H_symᵀ = 0`;
   `k_merged == k_code − 1`; `Ȳ` (product of appended χ_X·χ_Z) ∈ stabilizer.
6. **Observable.** `obs0 = ⊕ m(χ_x) ⊕ ⊕ m(χ_z) ⊕ ⊕ m(Y_stab)`; record
   `obs0_xor_map`. Completes `obs0` for the Y case (owned here, not via the two-block
   path).

## 5. Circuit — `build_single_y_ppm_circuit`

- Qubit coordinates + data state prep (reuse single-PPM helpers).
- Split-schedule QEC cycle: X-phase (CX), Z-phase (CZ), **Y-phase** (per `Y_stab` row
  decomposed into CX/CY/CZ → MX on the Y-ancilla).
- `q₁` extraction via the **Bell/flag cell**: split the `q₀`-side ancilla into 3
  qubits, Bell-init the outer two, Bell-measure at the end (Cross §4.1).
- `R≥d` rounds; per-round detectors; emit `obs0` from `obs0_xor_map`.

## 6. Error handling / preconditions

- Overlap ≠ 1 qubit → `ValueError` (Remark 19, out of scope).
- `x`/`z` not valid logicals, or not anticommuting → `ValueError`.
- Reducible `x`/`z`: document the irreducibility requirement; rely on `build_gadget`.
- Post-assembly asserts: commutation, `k−1` logicals, `Ȳ` ∈ stabilizer.

## 7. Testing (TDD, Steane first)

- `y_gadget_test.py`: build on Steane → valid stabilizer code, `k−1` logicals, `Ȳ` in
  stabilizer, all checks commute; overlap-≠1 raises.
- `circuit_single_y_test.py`: circuit compiles to a DEM (noiseless); **truth table** —
  prep `Ȳ=±1` eigenstates, confirm `obs0` reads the eigenvalue via the raw-sampler +
  manual XOR convention (`raw_observables`); DEM has no undetectable observable error.
- Optional: a small BB code (e.g. [[36,8]]) as a second case.

## 8. Out of scope (this round)

- Gross-code [[144,12,12]] instantiation/tuning (p,q,r,s operators, 11-bridge layout,
  automorphism `w`).
- Multi-qubit `X̄∩Z̄` overlap (Remark 19 extra gauge fixing).
- Circuit-level noise / fault-distance (LER, CPLEX/ILP).
- Changes to the two-block mixed bridge (`_rotate_x_side_to_z`, two-block `obs0`).

## 9. Open items to resolve in planning

- Exact column set for the `q₀` cross-merge in the intra-code case (§4 step 4).
- Whether the bridge helpers need lifting to a shared module (§3 factoring note).
- `Ȳ` eigenstate preparation on Steane for the truth table.
