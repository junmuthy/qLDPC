# gadget.py closed-form flattening — design

**Date:** 2026-06-29
**Status:** approved (brainstorming), pending implementation plan
**Scope:** `src/qldpc/circuits/surgery/gadget.py` (+ a 1-line import change in `bridge.py`, + `gadget_test.py` test rewrites)

## 1. Motivation

`gadget.py` (309 lines) implements the single L=1 gadget — the merged check
matrices `H̃_X`, `H̃_Z` that turn a logical-`X̄` (or `Z̄`) measurement into extra
low-weight stabilizers. The construction is, mathematically, a closed form
written down directly in `docs/superpowers/docs/main.tex` §2.1, Eq. (merged):

```
H̃_X = [[H_X,   0  ],     H̃_Z = [[H_Z,   π_{C0}^T ],
       [π_{V0}, ∂_1 ]]            [ 0,   ker(∂_1) ]]
∂_1 = π_{V0} H_Z^T π_{C0}^T   (vertex-edge, |V0|×|C0|, enters H̃_X untransposed)
```

The current code expresses this through a three-step decomposition
(`_step1_restriction` / `_step2_gauge_fix` / `_step3_assemble`), a `_projection`
helper that builds explicit projection matrices, and an `_assemble_HX_L1` block
factored out for reuse by `build_gadget_augmented`. That indirection earns its
keep for paper-citation traceability and boost/joint reuse, but it makes the
closed form hard to read against the paper: a reviewer cannot see `H̃_X`/`H̃_Z`
in one place.

**Goal:** make `gadget.py` read 1:1 with main.tex §2.1 (closed form front and
centre) **and** flatten away the `_step*` / `_projection` indirection — while
keeping every downstream contract byte-identical.

## 2. Goals / non-goals

**Goals**
- `build_gadget` / `build_gadget_augmented` produce byte-identical
  `GadgetLayout` output (all 10 fields) on every code, for `basis ∈ {X, Z}`.
- The closed-form `H̃_X` / `H̃_Z` assembly is visible in a single core function
  that mirrors main.tex §2.1.
- Remove `_projection`, `_step1_restriction`, `_step2_gauge_fix`,
  `_step3_assemble`, `_assemble_HX_L1` as separate layers.
- `basis=Z` realised as the X↔Z dual of the X core (mirrors main.tex §4.2
  "apply the X↔Z dual of §2"), not a parallel branch.

### 2.1 `_restrict` as the shared kernel (forward context)

The joint-PPM (`circuit.py` `_stitch_*`) and Y/mixed-PPM (`y_gadget.py`)
constructions are slated for the same closed-form flattening in follow-on specs.
They are only loosely coupled to the single gadget: the **one** genuinely shared
piece is `_restrict` (the useful quantities V0 / C0 / incidence / gauge). Each
construction then assembles its own `H̃` directly from main.tex (§2.1 single,
§3 joint, §4 Y) rather than reusing another's merged matrices. This spec
therefore lands `_restrict` as a deliberately small, stable, unified primitive;
the joint/Y refactors consume it without depending on `_x_merged` or
`build_gadget_augmented`.

**Non-goals (out of scope)**
- No change to `GadgetLayout`'s fields, names, or types.
- No change to `circuit.py`, `cheeger.py`, `y_gadget.py`, `y_circuit.py`.
- No refactor of `bridge.py` beyond the single import line in §5.
- No change to the joint / boost / Y algorithms — only the single-gadget
  primitive they call.
- No LER / statistical tests (per project policy).

## 3. Consumer surface (constraints discovered)

`gadget.py` is a hub, not a leaf. The refactor must preserve:

| Symbol | Consumers (non-test) | Must preserve |
| --- | --- | --- |
| `GadgetLayout` (10 fields) | `circuit.py`, `bridge.py`, `y_gadget.py`, `y_circuit.py`, `cheeger.py` | fields + values |
| `build_gadget(code, x, *, basis)` | indirectly everywhere; tests | signature + output |
| `build_gadget_augmented(code, x, incidence_extra, *, basis)` | `bridge.py:532-533`, `cheeger.py:242/399/419` | signature + output |
| `_step1_restriction(code, x, *, basis)` | `bridge.py:525-526` | a restriction primitive must remain callable (renamed `_restrict`, see §5) |

Heaviest `GadgetLayout` field reads: `support`, `incidence`, `partial_0`,
`HX_merged`, `HZ_merged`, `Q_prime` (across `circuit.py` / `bridge.py` /
`y_gadget.py` / `cheeger.py`). All must stay byte-identical.

Both bases are load-bearing: `bridge.py` / `cheeger.py` call the augmented
builder with `basis=Z` (ZZ joints, dual boost); `gadget_test.py` tests
`basis=Z` directly. The X-only collapse is therefore a *core*, wrapped by a
basis dispatcher — not a replacement of the module.

## 4. Target structure of gadget.py

```python
GF2 = galois.GF(2)

@dataclass(frozen=True, eq=False)
class GadgetLayout:          # UNCHANGED — 10 fields
    code; x; support; data_checks; incidence; partial_0
    HX_merged; HZ_merged; Q_prime; basis

# ── unified shared kernel (the ONE reused primitive; bridge.py + the future
#    joint/Y refactors all consume this — see §2.1) ──
def _restrict(H_complement, x) -> (support, data_checks, incidence, partial_0):
    """The useful quantities of main.tex §2.1, in one place:
       V0 = supp(x); C0 = complement-basis checks touching V0;
       incidence = ∂_1^T = H_complement[C0, V0]  (|C0|×|V0|, edge-vertex);
       partial_0 = ker(∂_1) = GF2(incidence).left_null_space()."""

# ── core: main.tex §2.1 closed form, X̄ frame only ──
def _x_merged(H_X, H_Z, x, incidence_extra=None)
        -> (support, data_checks, incidence, partial_0, HX, HZ):
    support, C0, incidence, partial_0 = _restrict(H_Z, x)
    if incidence_extra is not None:                  # boost/joint κ rows
        incidence = vstack([incidence, incidence_extra])
        partial_0 = GF2(incidence).left_null_space()  # gauge re-derived on stacked ∂_1^T
    d1 = incidence.T                                  # ∂_1 (vertex-edge), |V0|×|C0_aug|
    f1T       = π_{V0}                                # |V0|×n indicator on support
    f0        = π_{C0}^T with zero columns for the extra κ   # replaces _projection sentinel
    HX = [[H_X, 0  ], [f1T, d1       ]]
    HZ = [[H_Z, f0 ], [0  , partial_0]]
    data_checks = tuple(C0) + (-1,) * n_extra        # sentinels preserved for augmented
    return support, data_checks, incidence, partial_0, HX, HZ

# ── public API: basis via X↔Z dual ──
def build_gadget(code, x, *, basis):
    validate H_check @ x == 0           # X→H_Z@x=0, Z→H_X@x=0
    if basis is Pauli.X:
        sup, dc, inc, p0, HX, HZ = _x_merged(code.matrix_x, code.matrix_z, x)
    else:                                # Z = X↔Z dual: swap inputs, swap merged outputs
        sup, dc, inc, p0, HZ, HX = _x_merged(code.matrix_z, code.matrix_x, x)
    Q_prime = tuple(range(n, n + len(dc)))
    return GadgetLayout(code, x, sup, dc, inc, p0, HX, HZ, Q_prime, basis)

def build_gadget_augmented(code, x, incidence_extra, *, basis):
    # same validation + dual dispatch, passing incidence_extra into _x_merged
    # (weight-2 row checks + width check retained from current implementation)
```

**Removed:** `_projection`, `_step1_restriction`, `_step2_gauge_fix`,
`_step3_assemble`, `_assemble_HX_L1`.

### 4.1 Key equivalences (verified before writing the spec)

- `incidence = H_complement[np.ix_(C0, V0)]` equals the current
  `_step1_restriction` output (`= π_{V0} H_c^T π_{C0}^T` transposed). The X core
  was checked byte-identical to `build_gadget` on Steane + the 4 Webster
  generalised-bicycle codes (`HX_merged`, `HZ_merged`, and `∏ S_X' = X̄`).
- `GF2(incidence).left_null_space()` (over `incidence = ∂_1^T`) equals
  `ker(∂_1)` = current `_step2_gauge_fix` output; galois returns the same RREF
  canonical basis, so `partial_0` is byte-identical.
- Dual frame: for `basis=Z` the dual call's internal `support / data_checks /
  incidence / partial_0` are computed from `complement = code.matrix_x`, exactly
  the values the current `basis=Z` path produces; only the merged-matrix
  assignment swaps (`S'` rows land in `HZ_merged`, `∂_0` in `HX_merged`).
- `_projection`'s only non-trivial behaviour (the `-1` sentinel → all-zero `f0`
  column for boost-added κ with no backing check) is reproduced by building `f0`
  with the original `C0` mapping and zero columns for the `n_extra` extras.

## 5. Cross-file change (the only one) ⚠️

`bridge.py:525` does `from .gadget import _step1_restriction` and calls it to
recover the un-augmented incidence. Resolution: keep `_restrict` as a real
primitive (it has a genuine second consumer — not redundant indirection) and
change the bridge import + call site from `_step1_restriction(code, x, basis=…)`
to `_restrict(H_complement, x)`, selecting `H_complement` by basis at the call
site (`code.matrix_z` for X, `code.matrix_x` for Z). bridge only needs the
`incidence` element of the tuple and ignores the unified `_restrict`'s extra
returns (`data_checks`, `partial_0`). This is a ~3-line change in `bridge.py`
(`_build_combined_extras` helper, lines ~517-526). No other file changes.

## 6. Test changes (`gadget_test.py`)

Per agreement, step-tests become equivalence/property tests on the public API:

- `test_step1_restriction_steane`, `test_step1_restriction_basis_z_uses_HX`,
  `test_step1_restriction_rejects_x_shape_mismatch` → retarget to `_restrict`
  (semantics preserved; shape validation moves into `build_gadget`/`_restrict`).
- `test_step2_gauge_fix_*` (4), `test_step3_assemble_*` (3), `test_projection_*`
  (4) → removed as helper-level tests; their assertions are re-expressed as
  **properties of `build_gadget` output**:
  - `H̃_X H̃_Z^T = 0` (CSS commutation) — was `test_step3_assemble_steane_css_commutes`.
  - `∂_1 ∂_0^T = 0` and `rank(∂_0) = |C0| − rank(∂_1)` — was `test_step2_gauge_fix_*`.
  - `∏ rows(S_X') = X̄` on data, identity on Q' — gadget logical-product invariant.
  - basis=Z places `S'` in `HZ_merged`, `∂_0` in `HX_merged` — was
    `test_step3_assemble_basis_z_*`.
- `build_gadget_augmented` tests and `GadgetLayout` frozen-dataclass test:
  unchanged (public contract).

## 7. Safety net — golden byte-identical regression

Land **before** the refactor, must stay green **after**:

- Basket: Steane + the 4 Webster generalised-bicycle codes, each for
  `basis ∈ {X, Z}`, plus ≥2 `build_gadget_augmented` cases (one X, one Z, with a
  representative weight-2 `incidence_extra`).
- For each: assert post-refactor `build_gadget(_augmented)` reproduces the
  pre-refactor `GadgetLayout` — all 10 fields, with `HX_merged`, `HZ_merged`,
  `incidence`, `partial_0`, `support`, `data_checks`, `Q_prime` compared exactly
  (`np.array_equal` / tuple equality).
- Baseline arrays captured from the current implementation (the X core was
  already shown byte-identical, so this regression is expected to pass 100%).

## 8. Acceptance criteria

1. Golden regression (§7) green.
2. Full `src/qldpc/circuits/surgery/` test suite green (no LER/sinter tests).
3. `gadget.py`: `_projection`, `_step1_restriction`, `_step2_gauge_fix`,
   `_step3_assemble`, `_assemble_HX_L1` gone; closed-form `H̃_X`/`H̃_Z` visible in
   one core function; `basis=Z` realised via the X↔Z dual.
4. Only `bridge.py` changed outside `gadget.py` (the §5 import/call site).
5. Docstrings cite papers fully (authors + arXiv:ID + §) per project policy —
   never `main.tex` or bare surnames.

## 9. Risks

- **Galois basis non-determinism** between `left_null_space(incidence)` and the
  old call: mitigated — both operate on the same matrix and galois returns RREF;
  verified byte-identical. The golden regression catches any regression anyway.
- **bridge.py basis selection** at the new `_restrict` call site: covered by the
  existing `bridge_test.py` regression (`test ...rebuilds g_l_aug via
  restriction`) plus the surgery suite.
