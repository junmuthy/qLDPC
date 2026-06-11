# Cain-convention rename for surgery module

**Date:** 2026-06-11
**Goal:** Replace Webster's κ/χ/r/V_0/C_0/F/G vocabulary throughout `src/qldpc/circuits/surgery/` with Cain's `(Q, S_X, S_Z) / (Q', S'_X, S'_Z)`-style semantic names, so a reader who has not memorized Webster Table I can still understand `GadgetLayout` and the surgery pipeline.

## Motivation

The math in the surgery module is correct (validated by 147 pytest cases as of commit `a5638b8`). The remaining friction is *naming*: Webster's Greek letters κ (kappa), χ (chi), r and the index sets V_0, C_0 are concise and paper-faithful, but require the reader to hold an extra mental dictionary while reading the code. New code authors and reviewers consistently report that they can verify each formula but cannot intuit which symbol means what without flipping back to Webster §II.A.

Cain et al. use a more transparent vocabulary for the same construction. A data code is written as `Q(Q, S_X, S_Z)` (qubits / X-checks / Z-checks) and the ancilla system as `A(Q', S'_X, S'_Z)`. This separates the role (data vs ancilla) from the type (X-check vs Z-check) and avoids Greek-letter counters. We want the qLDPC surgery module to use this vocabulary while keeping all math identical.

## Scope

**In scope:**

* Rename the following attributes and local variables across `src/qldpc/circuits/surgery/`:

| Webster (current) | Cain (new) | Type | Meaning |
|---|---|---|---|
| `GadgetLayout.kappa_qubits` | `ancilla_qubits` | `tuple[int, ...]` | Cain Q' — new ancilla qubit IDs, one per data check in `data_checks` |
| `GadgetLayout.V0` | `support` | `tuple[int, ...]` | Indices in Q where the seed Pauli operator x has support |
| `GadgetLayout.C0` | `data_checks` | `tuple[int, ...]` | Indices of data checks (in S_X or S_Z, complementary to `basis`) that touch `support` |
| `GadgetLayout.F` | `incidence` | `np.ndarray (|data_checks|, |support|)` | Restriction of H_complement to (data_checks × support); incidence matrix between touched checks and support qubits |
| `GadgetLayout.G` | `gauge` | `np.ndarray (r, |data_checks|)` | Gauge-fix basis: rows are a canonical basis of `ker(incidence.T)` |
| `Bridge.extra_kappa_l` | `extra_ancilla_l` | `np.ndarray` | F-extra rows added during left-cellulation; each row = new ancilla qubit |
| `Bridge.extra_kappa_r` | `extra_ancilla_r` | `np.ndarray` | F-extra rows added during right-cellulation |
| local `kappa` (test) | `n_ancilla` | `int` | \|Q'\| |
| local `chi` (test) | `n_meas_checks` | `int` | Number of new measured-basis checks (= \|support\|) |
| local `r` (test) | `n_comp_checks` | `int` | Number of new complementary-basis (gauge-fix) checks |
| intermediate `F_aug` | `incidence_aug` | `np.ndarray` | Augmented incidence after boost/bridge |
| intermediate `F_extra` | `incidence_extra` | `np.ndarray` | Extra rows added during augmentation |

* Update all docstrings and code comments to use Cain vocabulary in prose. Webster paper citations stay (e.g. "Webster §II.A step 1") with a one-line Cain-translation when needed.
* Rename the affected test function:
  * `test_webster_table_i_kappa_chi_r_exact` → `test_webster_table_i_ancilla_meas_comp_exact`
  * `test_webster_table_i_z_basis_kappa_chi_r_exact` → `test_webster_table_i_z_basis_ancilla_meas_comp_exact`
  * Test docstrings continue to cite Webster Table I as the source of truth being reproduced.

**Out of scope:**

* Function names `_step1_restriction` and `_step2_gauge_fix` — these describe the step, not the returned matrix, and remain Webster-paper-aligned.
* CSS code matrix names `matrix_x` / `matrix_z` and merged matrices `HX_merged` / `HZ_merged` — these are CSS-code-level, not surgery-specific, and Cain uses S_X/S_Z anyway for the check *sets*.
* Numeric semantics (no algorithm changes, no test additions, no algebraic invariant changes).
* Public surface beyond `surgery/`: `qldpc/__init__.py` re-exports `GadgetLayout` but does not reference any renamed attribute, so it is unaffected.
* Notebook code (`docs/notebooks/*.ipynb`) is **not** touched by this rename — if any notebook references `g.V0` etc., we update it as a follow-up only if we find such references during implementation.

## Reasoning behind specific name choices

**`ancilla_qubits` (was `kappa_qubits`).** Directly mirrors Cain's Q'. Avoids the Greek letter that doesn't appear in the variable name (`kappa` is just a label, not a meaningful prefix).

**`support` (was `V0`).** The vector `x` is the seed Pauli operator; `support` is the standard term for `{i : x[i] = 1}`. Short and unambiguous in context.

**`data_checks` (was `C0`).** These are check indices into the *data* code's complementary check matrix (S_Z if basis=X, S_X if basis=Z). Distinct from the ancilla `S'_X`/`S'_Z`. Reader can disambiguate from `code.matrix_x.shape[0]` (total count of S_X) via context.

**`incidence` (was `F`).** F is a `(|C_0|, |V_0|)` 0/1 matrix where entry `(c, v) = 1` iff check `c` touches qubit `v` — definitionally an incidence matrix between two sets. Short enough not to bloat formulas like `incidence @ gauge.T`.

**`gauge` (was `G`).** Rows of G are gauge-fix stabilizers (in the complementary basis, acting on Q' only). "gauge" matches Webster §II.A step 2's own framing. Short.

**`n_meas_checks` / `n_comp_checks` (was χ, r).** Basis-symmetric semantic names. For basis=X, `n_meas_checks` counts the X-type new ancilla checks (which become `S'_X` rows); for basis=Z, they are Z-type (become `S'_Z` rows). The label refers to the role (measuring the seed observable vs gauge-fixing the complementary basis) not the literal basis, so it does not flip with `basis`.

**`extra_ancilla_l` / `extra_ancilla_r` (was `extra_kappa_l` / `extra_kappa_r`).** These are F-extra rows added during left/right cellulation, and each new row in F corresponds to a new ancilla qubit (since |Q'| = |C_0| and F has `|C_0|` rows). So "extra ancilla" is more direct than "extra kappa".

## Sites touched (approximate)

| File | Rename references |
|---|---|
| `src/qldpc/circuits/surgery/gadget.py` | 6 (dataclass field + `_step1_restriction` body + `build_gadget` + `build_gadget_augmented`) |
| `src/qldpc/circuits/surgery/bridge.py` | 15 (`Bridge` dataclass + `_edges_to_F_extra` + `build_bridge`) |
| `src/qldpc/circuits/surgery/cheeger.py` | 4 (`boost_gadget`, `boost_gadget_distance`, docstrings) |
| `src/qldpc/circuits/surgery/circuit.py` | 37 (heavy: `chi_l = len(g_l.V0)`, `g.kappa_qubits`, `M_chi[chi_start + len(g_l.V0):]`, etc.) |
| `src/qldpc/circuits/surgery/_test_gadget.py` | 23 |
| `src/qldpc/circuits/surgery/_test_circuit.py` | 6 |
| `src/qldpc/circuits/surgery/_test_bridge.py` | 7 |
| `src/qldpc/circuits/surgery/_test_cheeger.py` | 3 |
| `src/qldpc/circuits/surgery/__init__.py` | 0 (re-exports unchanged) |

**Total: ~101 references** across 8 files. Mostly mechanical attribute-access rename.

## Implementation strategy

Sequential, file-by-file. Within each file the rename is a multi-attribute regex on attribute access (`\.V0\b` → `.support`, `\.kappa_qubits\b` → `.ancilla_qubits`, `\.F\b` → `.incidence`, `\.G\b` → `.gauge`, `\.C0\b` → `.data_checks`, `extra_kappa_l` → `extra_ancilla_l`, `extra_kappa_r` → `extra_ancilla_r`). Each file is committed independently so any test regression localizes to one commit.

Local-variable renames in tests (`kappa` → `n_ancilla`, `chi` → `n_meas_checks`, `r` → `n_comp_checks`) are confined to test bodies and need careful scope: do NOT regex-rename `chi_l = len(g_l.V0)` style locals in `circuit.py`, since `chi_l` / `chi_r` / `chi_total` are merged-circuit-construction locals, not Webster Table I counters. Rename these to `n_meas_l`, `n_meas_r`, `n_meas_total` for consistency, but commit those separately for review clarity.

The rename order respects dataclass dependency:

1. `gadget.py` — rename `GadgetLayout` fields and `build_gadget` / `build_gadget_augmented` constructors.
2. `bridge.py` — rename `Bridge` fields and update `build_bridge` consumer of `g.V0`.
3. `cheeger.py` — update `boost_gadget` consumer of `g.kappa_qubits` / `g.F`.
4. `circuit.py` — update all consumers; rename `chi_l/chi_r/chi_total` locals to `n_meas_l/n_meas_r/n_meas_total`.
5. Each `_test_*.py` — update attribute references and rename test functions/local counters.

After each file commit, run `uv run pytest src/qldpc/circuits/surgery/ -q` and verify 147 passing.

## Architecture / file organization

No new files, no new modules. The rename is purely textual at attribute and local-variable level. Public re-exports in `__init__.py` (`GadgetLayout`, `build_gadget`, `Bridge`, `build_bridge`, `boost_gadget`, `cheeger_constant`) keep their names — the class identity is unchanged, only field names within it change.

## Testing

Same test suite, same expected counts.

* Before: 147 tests pass.
* After each commit: 147 tests pass.
* No new tests required (this is a rename, not a behavior change).
* The two renamed test functions (`test_webster_table_i_ancilla_meas_comp_exact` and its Z-basis variant) keep their `@pytest.mark.parametrize` cases and continue reproducing Webster Table I exactly.

## Risks and edge cases

* **`g.F` vs `g.G` regex collision.** `\.F\b` and `\.G\b` are single-letter attributes and could match in unrelated contexts (e.g. `H_F` in some other module). Mitigation: regex is `\.F\b` scoped to `surgery/` files only, and we do a `git grep` post-rename for stragglers. Verified that no test file uses `g.F` for any non-GadgetLayout object.
* **`chi_l / chi_r / chi_total` rename collision.** These are circuit-construction local variables, NOT references to `GadgetLayout.chi` (which never existed — `chi` was only a test-local counter equal to `len(g.V0)`). We rename them to `n_meas_l/n_meas_r/n_meas_total` for the same readability gain. Verified by searching that no production attribute is named `chi`.
* **Webster paper-citation comments** (e.g. "Webster §II.A step 1 — V_0 = supp(x); C_0 = checks touching V_0; F = H_complement[C_0, V_0].") should keep paper symbols when quoting the paper, but the same comment can also state the Cain equivalent on a second line for readability. Example:
  ```
  Webster §II.A step 1 — V_0 = supp(x); C_0 = checks touching V_0;
                          F = H_complement[C_0, V_0].
  Cain mapping: support = V_0; data_checks = C_0; incidence = F.
  ```
* **Notebook drift.** If `docs/notebooks/*.ipynb` references the old names, the notebook will break at runtime. We grep-check during implementation and update any hits as part of the same series of commits.

## Success criteria

* `uv run pytest src/qldpc/circuits/surgery/ -q` reports **147 passed** at the end (matches current count).
* `git grep -nE '\.(V0|C0|kappa_qubits|F|G)\b' src/qldpc/circuits/surgery/` returns **only** Webster-citation comments (no live attribute access).
* `git grep -n 'extra_kappa' src/qldpc/circuits/surgery/` returns nothing.
* `git grep -n 'chi_total\|chi_l\|chi_r' src/qldpc/circuits/surgery/` returns nothing.
* Webster paper-citation comments still cite Webster (no information loss).
* Each commit in the series leaves the test suite green (intermediate commits do not introduce a broken state).
