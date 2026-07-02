# Lean `lattice_surgery.ipynb` — Design

**Date:** 2026-06-25
**Status:** approved (brainstorm), pending spec review
**Scope:** Rewrite `examples/lattice_surgery.ipynb` into a lean demo; archive the current one.

## Goal

The current notebook (52 cells, ~850 KB) mixes an end-to-end surgery demo with a large
"vs published results" thread (Webster/Cain tables, Cross distance-preservation). Collapse it
to the essential narrative the user wants:

> pick a code, show single-qubit logical PPM (X̄/Ȳ/Z̄) runs correctly, show the joint Z̄⊗Z̄ runs
> correctly, then run LER for each.

Target: **3 sections, ~12–14 cells.** Prove correctness first (noiseless determinism + circuit
diagrams), then LER.

## Code choices

Correctness and LER want different codes, so the notebook uses one code *per purpose*:

| Purpose | Code | Why |
|---|---|---|
| Correctness (all cases) | Steane [[7,1,3]] | Small ⇒ exhaustive `data_init` truth tables are cheap and `timeslice-svg` diagrams stay legible. |
| LER X̄ single, Z̄⊗Z̄ joint | BB `(l=3, m=6)` = [[36,8]] | Real distance for a meaningful sweep; 36 qubits ⇒ sweeps are fast. |
| LER Ȳ single | Steane [[7,1,3]] | The \|Ȳ₊⟩ prep needs transversal S̄ ⇒ a doubly-even self-dual CSS code. BB does not qualify, so Ȳ LER cannot run on BB. |

## Notebook structure

### §0 Setup
Imports + shared helpers lifted from the current cells 3–4:
- `raw_observables(circuit, shots)` — truth-table helper using a raw `compile_sampler` + manual XOR
  (NOT `detector_sampler(separate_observables=True)`, which returns flips vs the noiseless baseline).
- noise model.
- `run_ler_sweep(...)` — thin wrapper over `sinter.collect` so each LER cell is ~3 lines, not a
  copy of the sweep boilerplate. Returns the surgery-vs-memory stats for plotting.

### §1 Correctness — Steane [[7,1,3]]
- **1.1 Single-qubit PPM.** For each of X̄ / Ȳ / Z̄:
  - render `circuit.diagram('timeslice-svg')` of the surgery gadget circuit (SVG saved in the .ipynb);
  - print the `data_init` truth table: scan the 6 single-qubit logical Pauli eigenstates
    (\|0̄⟩,\|1̄⟩,\|+̄⟩,\|-̄⟩,\|Ȳ₊⟩,\|Ȳ₋⟩); `obs0` is deterministic when the prepared logical commutes
    with the measured logical, and a genuine 50/50 when it anticommutes. Assert this.
- **1.2 Joint PPM.** Z̄⊗Z̄ on Steane×Steane (inter-code): same treatment — diagram + truth table.

**Diagram legibility:** the diagram cell may build a small-`rounds` circuit (e.g. `rounds=2`) just for
the picture, while the truth table runs the proper `rounds ≥ d` circuit. (Resolve in plan: same
`rounds` for both vs split.)

### §2 LER — surgery PPM vs memory baseline
Each cell = `run_ler_sweep(...)` + a plot of logical error rate vs physical error rate, surgery
overlaid on the memory baseline.
- **2.1 X̄ single** → BB [[36,8]].
- **2.2 Z̄⊗Z̄ joint** → BB [[36,8]].
- **2.3 Ȳ single** → Steane [[7,1,3]] (one-line note on why it stays on Steane).

> **LER-in-notebook is intentional and does not violate the "no LER in tests" rule.** That rule
> (`memory/feedback_no_ler_tests.md`) forbids `sinter.collect` sweeps in the pytest suite (slow,
> deemed meaningless there). A demo notebook is the correct place for LER curves.

## Dropped (vs current notebook)
- All of §3 — Webster Table I, Cain Table III `bb_18`, ×2 Cross distance-preservation (cells 18–28).
- §4.1 — Cain `bb_18` [[248,10]] LER.
- The joint-PPM "superposition variant" minimal example (truth table already proves joint PPM).
- The "how the cached (rep, seed) pair was found" reference cell.

## Archive + file handling
- The `.ipynb` is the source of truth (jupytext-flagged in metadata, but no paired `.py` on disk;
  cell 0 is only a module docstring).
- `git mv examples/lattice_surgery.ipynb examples/archive/lattice_surgery_full.ipynb` (new dir),
  preserving the full Cain/Webster/distance material.
- The new lean notebook takes the canonical name `examples/lattice_surgery.ipynb`.

## Open implementation details (resolve in the plan)
1. **Joint code for §2.2** — two logical qubits of a *single* [[36,8]] block (Z̄ᵢ⊗Z̄ⱼ, stays 36
   qubits) if the joint builder supports intra-code joints; otherwise two [[36,8]] blocks. Confirm
   against the joint-PPM builder API.
2. **Diagram `rounds`** — small-rounds-for-picture vs exact-circuit (above).
3. **Authoring mechanism** — hand-authored `.ipynb` JSON vs `py:percent` script + `jupytext --to
   notebook` + execute. Pick whichever produces a clean, top-to-bottom-runnable notebook.

## Success criteria
- New `examples/lattice_surgery.ipynb` runs top-to-bottom with no errors.
- §1 truth tables assert deterministic `obs0` on commuting preps and 50/50 on anticommuting preps,
  for X̄/Ȳ/Z̄ single and Z̄⊗Z̄ joint; each shows a `timeslice-svg` diagram.
- §2 produces three LER plots (X̄, Z̄⊗Z̄ on BB [[36,8]]; Ȳ on Steane) vs memory baselines.
- Current notebook preserved at `examples/archive/lattice_surgery_full.ipynb`.
- ~12–14 cells, 3 sections.
