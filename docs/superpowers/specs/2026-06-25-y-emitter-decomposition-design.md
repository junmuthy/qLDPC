# Ȳ emitter decomposition + H̃-faithful construction — design

**Date:** 2026-06-25
**Status:** approved (design)
**Scope:** `src/qldpc/circuits/surgery/` — the single logical-Ȳ measurement
construction (`y_gadget.py`) and emitter (`circuit.py` → new `y_circuit.py`).

## Motivation

After the Z̄⊗X̄ deletion and the `_surgery_*_joint` merge (done earlier on this
branch), the surgery module has three emitters:

| Emitter | Lines | Shape |
|---|---|---|
| `build_single_ppm_circuit` (X̄/Z̄) | ~85 | CSS, thin — calls `_surgery_*` helpers |
| `_build_joint_ppm_circuit_same_basis` (X̄X̄/Z̄Z̄) | ~98 | CSS, thin — same `_surgery_*` helpers |
| `build_single_y_ppm_circuit` (Ȳ) | ~635 | non-CSS, fully inline |

Two problems: (1) the 635-line Ȳ emitter is one opaque function in a 2003-line
`circuit.py`; (2) the merged-code construction in `y_gadget.py` uses names
(`hx_ext_kz`, `chi_x`, `partial0`) that **cannot be recognised against the
paper's check matrix** $\tilde H$, so the code is not verifiable against the
math.

This refactor (a) makes the construction directly correspond to $\tilde H$ —
formula-faithful names, explicit per-block stabilizer construction, rows
assembled in formula order; and (b) decomposes the emitter into named phases,
each annotated with the $\tilde H$ blocks it processes. It is
**behaviour-preserving at the DEM/observable level**, with two deliberate
exceptions (pruned obs1/benchmark_y; renumbered check IDs from formula-order
assembly).

## §0 — Reference: the merged-code check matrix H̃

The Ȳ merged code (Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.D;
main.tex §4) is the symplectic check matrix `[X-part | Z-part]`, each half with
column blocks `[ data (n) | κ_x (k_x) | κ_z (k_z) ]`:

```
        |   X-part (data | κ_x | κ_z)        |   Z-part (data | κ_x | κ_z)
H̃ =     |------------------------------------|------------------------------------
 row 1  |  H_X     0        π_{C₀^Z}^T        |   ·       ·        ·          X-checks
 row 2  |  π_{V_X} ∂₁ˣ|_{V_X}  0              |   ·       ·        ·          χ_X on V_X
 row 3  |  π_W     ∂₁ˣ|_W   0                 |  π_W      0       ∂₁ᶻ|_W       Y on W (mixed)
 row 4  |  ·       ·        ·                 |  H_Z    π_{C₀^X}^T  0          Z-checks
 row 5  |  ·       ·        ·                 |  π_{V_Z}  0       ∂₁ᶻ|_{V_Z}    χ_Z on V_Z
 row 6  |  0       0        ∂₀^Z              |   0      ∂₀^X      0           cycles ∂₀
```

### Block → name → construction

| Block | $\tilde H$ symbol | Python name | Constructed as |
|---|---|---|---|
| 1 | `H_X` | `H_X` | `code.matrix_x` |
| 1 | `π_{C₀^Z}^T` | `pi_C0z_T` | H_X's κ_z extension (`g_z` selection onto C₀^Z) |
| 2 | `π_{V_X}` | `pi_VX` | support-projection of χ_X (identity at V_X cols) |
| 2 | `∂₁ˣ\|_{V_X}` | `d1x_VX` | `g_x.incidence` restricted to V_X (κ_x cols of χ_X) |
| 3 | `π_W` | `pi_W` | support-projection at W cols (X- and Z-part) |
| 3 | `∂₁ˣ\|_W`, `∂₁ᶻ\|_W` | `d1x_W`, `d1z_W` | incidence restricted to W (via `apply_mixed_basis_merge`) |
| 4 | `H_Z` | `H_Z` | `code.matrix_z` |
| 4 | `π_{C₀^X}^T` | `pi_C0x_T` | H_Z's κ_x extension (`g_x` selection onto C₀^X) |
| 5 | `π_{V_Z}` | `pi_VZ` | support-projection of χ_Z |
| 5 | `∂₁ᶻ\|_{V_Z}` | `d1z_VZ` | `g_z.incidence` restricted to V_Z (κ_z cols of χ_Z) |
| 6 | `∂₀^Z`, `∂₀^X` | `d0_Z`, `d0_X` | cycle basis `ker(merged ∂₁)` (`_partial0_symplectic_rows`) |

`gadget.py`'s Cain-named internals (`incidence`, `incidence_tilde`, the χ/G
split) are **mapped** to these symbols at the `y_gadget.py` use sites
(`d1x_VX = g_x.incidence[...]  # ∂₁ˣ|_{V_X}`), **not renamed** — `gadget.py` is
shared with the CSS path and keeps its Cain/Webster conventions.

## §1 — H̃-faithful construction (`y_gadget.py`)

Rewrite the assembly inside `build_y_gadget` (currently
`y_gadget.py:676-738`) so each $\tilde H$ block is a **named local variable**
constructed explicitly, then stacked **in formula order**:

```python
# Each block named + built per §0; symplectic width 2*n_merged.
Xcheck_rows = _row([H_X,    zeros, pi_C0z_T], [])               # block 1
chiX_rows   = _row([pi_VX,  d1x_VX, zeros],   [])               # block 2
Ymix_rows   = Y_stab                                            # block 3 (mixed)
Zcheck_rows = _row([], [H_Z,   pi_C0x_T, zeros])               # block 4
chiZ_rows   = _row([], [pi_VZ, zeros,    d1z_VZ])              # block 5
cycle_rows  = _row([zeros, zeros, d0_Z], [zeros, d0_X, zeros]) # block 6
H_sym = np.vstack([Xcheck_rows, chiX_rows, Ymix_rows,
                   Zcheck_rows, chiZ_rows, cycle_rows])
```

(`_row(x_blocks, z_blocks)` is a small local that hstacks the `[data|κ_x|κ_z]`
column blocks for each half, padding empties with zeros — replacing the current
`_embed` helper.)

**Assembly order changes** from today's grouped `[HX_out | HZ_out | Y_stab |
∂₀]` to the formula's `1,2,3,4,5,6`. Consequences:

- `_ybar_obs0_rows`, `_split_quditcode_into_virtual_cssc`, and check-ID
  assignment all derive families/indices **from `H_sym`**, so they adapt to the
  new order automatically (family_index is the row's position within its Pauli
  family in `H_sym`).
- Check IDs and detector coordinates are **renumbered**. The DEM and observables
  are unchanged up to this relabelling, so logical tests pass; tests asserting
  exact check IDs / circuit text need updating (§4).

`build_y_gadget`'s docstring reproduces the §0 $\tilde H$ block and the
block→construction table.

## §2 — Module structure (`y_circuit.py`)

The entire Ȳ subsystem is already self-contained and **Ȳ-only** (all five
private helpers — `_steane_logical_y_eigenstate_prep`,
`_split_quditcode_into_virtual_cssc`, `_mixed_basis_qubit_coords`,
`_compute_stabilizer_center_mask`, `_observable_is_deterministic` — are called
only by the Ȳ emitter, which imports nothing from the CSS path).

Create `src/qldpc/circuits/surgery/y_circuit.py`; move
`build_single_y_ppm_circuit` + those five helpers + the nested `_row_paulis` +
the new phase helpers there. Update the one re-export in `surgery/__init__.py`.

Result: `circuit.py` 2003 → ~1100 lines (pure CSS); `y_circuit.py` ~640 lines.

## §3 — Phase decomposition (`build_single_y_ppm_circuit`)

The 635-line body becomes a ~40-line orchestrator over named phases, each
documented with the $\tilde H$ blocks it touches:

```
build_single_y_ppm_circuit(yg, *, rounds, noise_model, data_init,
                           memory_logical, force_obs0) -> stim.Circuit
  ├─ _y_state_prep            setup (split virtual CSSc, qubit-id arrays,
  │                           coords) + |Ȳ±⟩ injection + κ_x/κ_z init
  ├─ _y_qec_cycle             split X/Z/Y schedule + round-1 reliable detectors
  │                             X-phase → blocks 1,2 (+ ∂₀^Z of 6)
  │                             Z-phase → blocks 4,5 (+ ∂₀^X of 6)
  │                             Y-phase → block 3 (y_v)
  ├─ _y_detach_and_readout    mixed-basis MX/M/MY destructive readout
  ├─ _y_final_detectors       stabilizer-center rows from final readouts
  ├─ _y_emit_obs0             Ȳ eigenvalue product (force_obs0 / Y±-deterministic)
  └─ _y_emit_survivor_memory  survivor-Z̄ logical-memory observable
```

The orchestrator owns cross-phase state (merged code, `QubitIDs`, the id arrays,
`center_mask`, `MeasurementRecord`) and threads it explicitly — no closure/module
state. `_y_qec_cycle` is the explicit non-CSS counterpart to `_surgery_qec_cycle`;
the two stay separate (the chosen "two syndrome-schedule strategies" design).

## §4 — Prune (deliberate behaviour change)

Remove during extraction: the `benchmark_y` parameter + its block (obs0⊕obs1 LER
benchmark) and the `obs1` destructive-cross-check block — both untested, both
bring-up scaffolding. Keep `obs0` and the survivor-memory observable.
`_observable_is_deterministic` stays (used by both survivors). The CSS emitters'
own `obs1` (`_surgery_observable`, tested by
`circuit_test.py::test_build_joint_ppm_circuit_intercode_noiseless_observables_zero`)
is untouched.

## §5 — Success criteria & testing

- CSS circuits (X̄/Z̄/X̄X̄/Z̄Z̄) byte-identical — `circuit.py` CSS path untouched.
- Ȳ DEM + kept observables (obs0, survivor-memory) unchanged **as logical
  objects**; check IDs / detector coordinates renumber from formula-order
  assembly.
- Tests to update (enumerate in the plan): any asserting exact Ȳ check IDs /
  circuit text / detector coords; any importing the moved helpers (path
  `.circuit` → `.y_circuit`); none assert obs1/benchmark_y for Ȳ (verified).
  Logical/DEM/observable tests in `circuit_single_y_test.py` and
  `y_gadget_test.py` stay green unchanged.
- `ruff check` clean; no new public API; full surgery suite green.

## Out of scope

- The CSS emitters (single/joint) and `_surgery_*` helpers.
- Renaming `gadget.py`'s Cain/Webster internals (mapped, not renamed).
- The Ȳ algorithm / merge math itself (`apply_mixed_basis_merge`,
  `_partial0_symplectic_rows` logic) — only the assembly naming/order changes.
- `docs/superpowers/docs/main.tex` cleanup of deleted Z̄⊗X̄ sections.
