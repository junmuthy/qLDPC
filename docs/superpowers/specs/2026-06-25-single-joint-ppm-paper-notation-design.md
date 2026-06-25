# Single & joint PPM: §2/§3 paper-notation refactor

**Date:** 2026-06-25
**Status:** design (approved in brainstorm)
**Status:** implemented (2026-06-25)
**Scope:** `src/qldpc/circuits/surgery/` — `gadget.py` (§2 single gadget), `circuit.py`
(§2 single + §3 joint stitch), `bridge.py` (§3 adapter), and mechanical touch-ups in
`y_gadget.py`/`y_circuit.py` (§4) + the surgery test suite.

## Goal

Make the §2 (single-PPM) and §3 (joint-PPM) surgery code **read against the
parity-check-matrix formulas** the way the §4 mixed-Pauli code (`y_gadget.py`) already
does. The reader should see the paper's symbols — `π_{V₀}`, `π_{C₀}`, `H_X'`, `f_X'`,
`f_Z`, `H_Z'`, `Q'`, `S_X'`, `T_l/T_r/H_R` — in the identifiers and block assembly,
instead of opaque names (`incidence_tilde`) or non-paper jargon (`χ`, `κ`,
`ancilla_qubits`).

The formulas being matched (from `docs/superpowers/docs/main.tex`, the project's
internal writeup — used here only to fix **notation**, never cited from source):
- §2 single: `H̃_X = [[H_X, 0],[f_X', H_X']]`, `H̃_Z = [[H_Z, f_Z],[0, H_Z']]` (Eq. (1), (7)).
- §3 joint: the 4-block-row `H̃_X^joint` and 5-block-row `H̃_Z^joint` (Eq. (18)–(19),
  main.tex lines 184–201).

## Non-goals / out of scope

- **No behavioral change.** No matrix reorder, no check-ID renumbering, no circuit
  change. (Both §2 and §3 are *already* in paper block-row order — verified below — so
  unlike the §4 Ȳ work, no reorder is needed.)
- **§4 `∂₁/∂₀` convention is left as-is.** `y_gadget.py`/`y_circuit.py` keep their own
  internal boundary-map names; only their *reads* of the renamed shared field are touched.
- No LER / `sinter` tests (per project rule). No Hadamard-dual / basis-rotation anything.

## Key finding that de-risks this: already in formula order

- `gadget._assemble_HX_L1` builds `[[H_X, 0],[π_{V₀}, H_X']]`; `_step3_assemble` builds
  `[[H_Z, f_Z],[0, gauge]]` — exactly Eq. (7).
- `circuit._stitch_intercode`/`_stitch_intracode` build `M_meas = [H_X^l; H_X^r; S_X'^l;
  S_X'^r]` and `M_comp = [H_Z^l; H_Z^r; H_Z'^l; H_Z'^r; cycle]` — exactly Eq. (18)–(19).

Therefore this is a **rename + re-express + document** refactor, not a restructuring.

## The guarantee (safety net)

Every merged `HX_merged`/`HZ_merged` (single, joint inter-code, joint intra-code) and
every emitted circuit stays **byte-identical** pre/post. Enforced by a regression test
that snapshots the matrices and circuit text on fixture codes and asserts array-equality,
including empty-support edge cases. The π-form construction (below) is identical over GF(2)
to the current fancy-indexing because `π_S M π_T^T = M[S,T]`.

## Notation decision (settled in brainstorm)

**Option A: each file uses its own main.tex section's symbols.** §2/§3 use
`H_X'/f/π/Q'`; §4 keeps `∂₁/∂₀`. A one-line doc note bridges them (`y_gadget`'s `∂₁ˣ`
≡ this `H_X'`). Chosen over forcing `∂₁` everywhere because it is most faithful to the
Eq. (7)/(18)–(19) formulas and keeps §4 churn out of scope.

## The `_projection` helper (paper's `π_S`)

Add to `gadget.py`:

```python
def _projection(indices, N):
    """π_S ∈ F₂^{|S|×N}: row i is the unit vector e_{indices[i]}.

    (π_S)_{i,j} = δ_{j, indices[i]}, so π_S M π_T^T = M[S, T] (numpy-style index).
    Negative index entries (sentinels for boost-added Q' with no backing check) give a
    zero row, matching build_gadget_augmented's tilde-F behavior.
    """
```

Used so the construction reads like the formula:

```python
# H_complement / m_comp are the complementary basis: (H_Z, m_Z) for basis=X,
# (H_X, m_X) for basis=Z. Do NOT hardcode m_Z.
pi_V0 = _projection(support, n)            # π_{V₀} ∈ F₂^{|V₀|×n}  (= f_X')
pi_C0 = _projection(data_checks, m_comp)   # π_{C₀} ∈ F₂^{|C₀|×m_comp}
H_X_prime = pi_V0 @ H_complement.T @ pi_C0.T   # = π_{V₀} H_Z^T π_{C₀}^T, shape |V₀|×|C₀|
# stored field `incidence` = H_X_prime.T  (the |C₀|×|V₀| vertex-edge incidence)
f_X_prime = pi_V0                          # data part of S_X'
f_Z       = pi_C0.T                        # original-check extension onto Q' (was incidence_tilde)
gauge     = left_null_space(incidence)     # H_Z' = ker(H_X'); unchanged computation
```

## Identifier mapping

### `gadget.py` — `GadgetLayout` fields + locals

| current | main.tex §2 symbol | action |
|---|---|---|
| `ancilla_qubits` (field) | `Q'` | **rename → `Q_prime`** |
| `incidence_tilde` (local) | `f_Z = π_{C₀}^T` | **rename → `f_Z`** (built as `pi_C0.T`) |
| "χ rows" / bottom `_assemble` block | `S_X' = [f_X' \| H_X']` | name **`S_X_prime`**, parts `f_X_prime` (`=pi_V0`) + `H_X_prime` (`=incidence.T`) |
| (new local) | `π_{V₀}`, `π_{C₀}` | introduce **`pi_V0`, `pi_C0`** via `_projection` (`pi_C0` width = complement check count: `m_Z` if basis=X, `m_X` if basis=Z) |
| `incidence` (field, `\|C₀\|×\|V₀\|`) | `(H_X')^T`, vertex-edge incidence | **keep**; doc `= (H_X')^T`; bridge note `≡ ∂₁ˣ` (§4) |
| `gauge` (field, `r×\|C₀\|`) | `H_Z' = ker(H_X')` | **keep**; doc `= H_Z'` (gauge-fixing rows, main.tex §2.1) |
| `support` (field) | `V₀ = supp(x)` | **keep**; doc `= V₀` |
| `data_checks` (field) | `C₀` | **keep**; doc `= C₀` |
| `HX_merged`/`HZ_merged` (fields) | `H̃_X`/`H̃_Z` | keep (already the formula's `H̃`) |

Keep `incidence`/`gauge`/`support`/`data_checks` as field names — all four are the exact
words main.tex §2.1–2.2 uses, and keeping them holds the field-rename blast radius to the
single `ancilla_qubits→Q_prime` change.

### `circuit.py` — joint (§3) + single locals

| current | main.tex §3 symbol | action |
|---|---|---|
| `meas_l_rows` / `meas_r_rows` ("χ rows") | `S_X'^l` / `S_X'^r` | **rename → `S_prime_l` / `S_prime_r`** |
| `cl_ancilla` / `cr_ancilla` (col slices) | `Q'_l` / `Q'_r` | **rename → `Ql_prime` / `Qr_prime`** |
| `M_meas` / `M_comp` | `H̃_X^joint` / `H̃_Z^joint` | keep; document each block-row against Eq. (18)–(19) |
| `c_adapter` | adapter `𝒜` | keep; doc |
| gauge-row / cycle-row blocks | `H_Z'^{l,r}` / `[T_l T_r H_R]` | comments only |

### `bridge.py`

Already paper-faithful (`width=w`, `port_l/r=𝒫_s`, `label=σ`, `T_l/T_r`, `H_R`,
`g_l_aug/g_r_aug`). **Docstring polish only:** document the port-label block
`π_{𝒫_s}^T P_{σ_s}` and the SkipTree identity; full citations (below).

### `y_gadget.py` / `y_circuit.py` (§4) — mechanical only

The shared field rename `ancilla_qubits → Q_prime` touches 6 read-sites here
(`len(g_x.Q_prime)` etc., incl. two docstrings). Internal `∂₁/∂₀` naming untouched.

### tests

Update renamed field/local references in `gadget_test.py`, `circuit_test.py`,
`bridge_test.py`, `y_gadget_test.py`. Add the byte-identical snapshot regression test.

## Docstring / citation strategy

Per `feedback_doc_citations` (project rule): **do not cite the internal `main.tex` from
source**; cite the primary papers fully (`Authors arXiv:ID §`). The code adopts main.tex's
*notation* only. Replace the old `"Cain mapping: V_0 → support"` bare-surname annotation
lines with full citations + plain symbol-mapping comments.

Verified citation strings (from `reference_surgery_citations`):
- §2 gadget — `Webster, Smith, Cohen arXiv:2511.15989 §II.A`; `Cain et al. arXiv:2603.28627 §B.1`.
- §3 joint — `Swaroop et al. (Swaroop, Jochym-O'Connor, Yoder) arXiv:2410.03628 §III` (SkipTree);
  `Cross et al. arXiv:2407.18393 Thm 6` (Cheeger distance preservation).
- boost / cellulation — `Williamson & Yoder arXiv:2410.02213`.

## Test / verification plan

1. **Byte-identical regression (new):** snapshot `HX_merged`/`HZ_merged` for single +
   joint inter-code + joint intra-code, and the emitted circuit text, on ≥2 fixture codes;
   assert array-equality and string-equality against pre-refactor snapshots. Cover the
   empty-support / boosted (sentinel) paths.
2. **Existing surgery suite passes unchanged** (`gadget_test`, `circuit_test`,
   `bridge_test`, `cheeger_test`, `merge_test`, `circuit_single_y_test`, `y_*_test`).
3. Deterministic structural checks only (matrix shapes, `num_observables`, DEM compile,
   truth tables) — **no LER / `sinter` sampling**.

## Execution shape

Subagent-driven per the implementation plan (writing-plans next), one file at a time,
each gated on the byte-identical snapshot + full suite staying green.
