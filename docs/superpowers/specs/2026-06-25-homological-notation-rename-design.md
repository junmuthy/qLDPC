# Surgery code → unified homological notation (f_1/∂_1/f_0/∂_0)

**Date:** 2026-06-25
**Status:** design (approved in brainstorm)
**Status:** implemented (2026-06-25)
**Scope:** `src/qldpc/circuits/surgery/` — `gadget.py`, `circuit.py`, `bridge.py`, `y_gadget.py`, `y_circuit.py` (+ their tests).
**Follows:** the just-completed `2026-06-25-single-joint-ppm-paper-notation` refactor. `main.tex` has since switched to ONE unified block-symbol set across §2/§3/§4 (arXiv:2410.02753 Eq.(62)/(68)); this pass re-aligns the code identifiers + docstrings to it.

## Goal

`main.tex` now writes the merged check matrices as
`H̃_X = (H_X 0 ; f_1^T ∂_1)`, `H̃_Z = (H_Z f_0 ; 0 ∂_0)` (Eq.(1)) with
`∂_1 = π_{V₀}H_Z^Tπ_{C₀}^T` (vertex×edge, untransposed in H̃_X), `f_1^T = π_{V₀}`,
`f_0 = π_{C₀}^T`, `∂_0 = ker ∂_1`. Apply the mapping `f_X'→f_1`, `H_X'→∂_1`, `f_Z→f_0`,
`H_Z'→∂_0` (§4 extension blocks `f_X/f_Z→f_0^z/f_0^x`) to the code so identifiers and
docstrings keep matching `main.tex`. (Ref: memory `feedback_surgery_naming_paper_symbols`.)

## Guarantee (safety net)

Pure rename + docstring edits — **zero behavioral change**. Every `HX_merged`/`HZ_merged`
(single, joint inter/intra) and every emitted circuit stays byte-identical. The existing
golden test `src/qldpc/circuits/surgery/refactor_snapshot_test.py` (12 sha256 hashes,
matrices unchanged) remains the authoritative gate — **its hashes must NOT change**.

## Notation note (orientation)

`∂_1` is vertex×edge (`|V₀|×|C₀|`) and enters `H̃_X` untransposed. The stored field
`incidence` is its transpose `∂_1^T` (edge×vertex, `|C₀|×|V₀|` = the paper's Eq.(62)
orientation); §4 already recovers `∂_1 = incidence.T`. So the field `incidence` is KEPT
(renaming it `partial_1` would lie about orientation); the local vertex×edge object is
`partial_1 = incidence.T`. `∂_0 = ker ∂_1 = left_null_space(incidence) = ker(incidence^T)`.

## Decisions (locked in brainstorm)

- **Rename field `gauge → partial_0`** (= ∂_0). Read at 6 sites in `circuit.py` + tests.
- **Keep field `incidence`** (= ∂_1^T), documented.
- **§4 (`y_gadget`/`y_circuit`): remove `χ` + align docstrings only.** Keep `d1x/d1z/D1/partial0`
  (already read as `∂_1^x/∂_1^z/∂_1/∂_0`).

## Identifier mapping

### `gadget.py`
| current | is | → |
|---|---|---|
| local `H_X_prime` | `∂_1` (vertex×edge) | **`partial_1`** |
| local `f_Z` | `f_0 = π_{C₀}^T` | **`f_0`** |
| local `f_X_prime` (`=π_{V₀}`) | `f_1^T` (data attachment) | **`f_1_T`** |
| field `gauge` | `∂_0 = ker ∂_1` | **`partial_0`** (field rename) |
| field `incidence` | `∂_1^T` | keep (doc `= ∂_1^T`) |
| `S_X_prime`, `pi_V0`, `pi_C0`, `_projection` | generators/projections | keep |

Also fix the doc-correctness slip in `_step2_gauge_fix`: it currently says
`ker((H_X')^T)`; correct text is `∂_0 = ker(∂_1) = left_null_space(incidence) = ker(incidence^T)`.

### `circuit.py`
- Docstrings/comments: `H_X'→∂_1`, `f_X'→f_1`, `f_Z→f_0`, `H_Z'→∂_0`.
- Field reads `.gauge → .partial_0` (lines ~170, 180, 186, 303, 312, 469). Keep the
  surrounding count-local NAMES unchanged (`n_gauge_*`, `r_l`, `r_r`); only the field
  access they read changes (`g.gauge.shape[0]` → `g.partial_0.shape[0]`).
- Keep `S_prime_l/r`, `Ql/Qr_prime`, `M_meas/M_comp` (generators/primes/matrix names).

### `bridge.py`
- Docstrings: `H_X'^{s,aug} → ∂_1^{s,aug}` (SkipTree identity, lines ~29, 31, 103, 356, 359, 452, 454); prose "gauge" → "∂_0". No code change.

### `y_gadget.py` / `y_circuit.py` (§4)
- Remove `χ`: identifier `chi_x/chi_z → SX_prime/SZ_prime`; prose `χ_X·χ_Z·y_v → S_X'·S_Z'·y_v`.
- Docstrings already use `∂_1/∂_0`; align any residual `H_X'`/`f_X`/`f_Z` to `∂_1`/`f_0^x`/`f_0^z`.
- **Keep** `d1x/d1z/D1/partial0` and `∂_0^{(X)}/∂_0^{(Z)}` (already homological).
- `.incidence` reads unchanged (field kept); y_gadget does NOT read `.gauge`, so the field
  rename does not touch it.

### tests
- Update `.gauge → .partial_0` reads and any renamed-local references in `gadget_test.py`,
  `circuit_test.py`, `cheeger_test.py`, `bridge_test.py`, `y_gadget_test.py`.

## Citations (also folded in — flag for review)

While editing §4 docstrings, **drop the `docs/superpowers/docs/main.tex §X` source
references** (14 in `y_gadget.py`, 7 in `y_circuit.py`) per the standing
`feedback_doc_citations` no-internal-doc rule — those docstrings already cite the primary
paper (Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753). Keep the arXiv citation; remove
only the `main.tex` tail. (This resolves the follow-up flagged after the prior refactor. If
you'd rather keep it a separate task, say so at spec review.)

## Test / verification plan

1. Golden gate: `refactor_snapshot_test.py` stays green, 12 hashes unchanged.
2. Full surgery suite passes unchanged: `python -m pytest src/qldpc/circuits/surgery/ -q`.
3. Jargon sweep — no `χ` anywhere in surgery source, no `H_X'`/`H_Z'`/`f_X'` leftover in the
   five touched files, no `docs/superpowers/docs/main.tex` in `y_gadget.py`/`y_circuit.py`:
   `grep -rn 'χ\|H_X.\|H_Z.\|main\.tex' src/qldpc/circuits/surgery/{gadget,circuit,bridge,y_gadget,y_circuit}.py`.
4. No LER / sinter tests.

## Execution

Subagent-driven; the golden test from the prior refactor is the regression gate throughout.
