# Plan: SymCityCode with automatic check-orbit basis selection

Status: implemented on 2026-10-16 on branch `code/cyclic-physical-types`, based
on commit `2aa2e975`. The sections below preserve the construction contract.

## Delivery notes

Implemented `SymCityCode` with automatic check-orbit basis resolution, separate
check and logical solver budgets, and structured `CheckBasisResult` and
`CheckBasisError` outcomes. The solver applies to the general coefficient tensor
and finite abelian logical group: it tries parent orbits, alternative subsets,
and polynomial combinations, then independently validates any successful basis.
`SymCityCode` is the sole public family name. Explicit check and logical
overrides and the coefficient-list input remain available.

Renamed the private solver module to `src/qldpc/codes/_symcity.py` and the
notebook to `examples/symcity_codes.ipynb`. The notebook begins with a redundant
multi-family presentation over a noncyclic logical group and demonstrates
combination-only basis recovery and structured failures.

Validation in the existing Python 3.11 environment:

- All 78 tests in `quantum_test.py`, `_symcity_test.py`, and
  `transversal_test.py` pass. The repository's modular coverage check reports
  100% coverage for the three associated implementation modules.
- Repository formatting and linting pass for all changed Python and notebook files.
- All nine notebook code cells execute successfully, including assertions, with
  captured outputs. Execution ran in process because kernel sockets are restricted.
- Type checking reports only the existing unused `type: ignore` on the HGP
  `matrix_z` assignment, reproduced on the unchanged base checkout.
- Sphinx generates the API sources, but the full build stops at the existing
  `examples/basics.ipynb` because Pandoc is absent from this environment.

The binary-reduction logical-frame enhancement remains a separate, unimplemented
plan. No performance benchmark, distance certification, or noise simulation is
part of this change. Both general fallback solvers can have exponential cost;
budget exhaustion is reported as inconclusive, not as a proof of nonexistence.

## Objective

Make automatic check-basis resolution the default for the entire code family.
Use `SymCityCode` as the public family name. When no check
basis is supplied, determine an independent basis of complete logical-group
translation orbits from the supplied check polynomials, then run the existing
logical-frame construction and validation.

The algorithm must apply to every supported coefficient tensor, with arbitrary
numbers of check families and physical cycles, any supported physical-cycle
length, and any finite abelian logical group. It must not recognize particular
polynomials, code names, two-block layouts, or known row counts. Success depends
on the algebraic existence of a basis and the configured computational budget.
Preserve the coefficient-list API: this change does not add symbolic inputs.

This plan supersedes the original plan's default behavior of requiring the
entire parent check matrix to be independent. The
[original implementation plan](cyclic_physical_types_plan.md) remains a
historical record. The
[binary-reduction logical-frame plan](cyclic_physical_types_reduced_logicals_plan.md)
is a separate, compatible enhancement: its reduced-frame helper can run after
the automatic check basis has been found. Implementing that additional helper
is not a prerequisite for this general basis solver or part of this change.

## Construction contract

The input is still a `p x t x q` tensor of coefficients over the binary abelian
group algebra `R = F_2[G]`. The ring specifies `G`; no separate group argument
is necessary. Put `N = |G|`, `P = pq`, and `m = tq`.

Build the same polynomial parent `H_tilde` of shape `P x m`, and verify
`H_tilde H_tilde^dagger = 0` before selecting checks. Let

$$
M=\widetilde H_{\rm bin},\qquad
D=\operatorname{rank}_{\mathbb F_2}M.
$$

The automatic solver seeks a polynomial matrix `H` with `r` rows such that

$$
\operatorname{rowspan}_{\mathbb F_2}H_{\rm bin}
=\operatorname{rowspan}_{\mathbb F_2}M,
\qquad
\operatorname{rank}_{\mathbb F_2}H_{\rm bin}=rN=D.
$$

Its rows can be selected parent prototypes or polynomial combinations of
them. The latter is already supported by the explicit `check_basis` API.
Require `s = m - 2r >= 1`, and retain all existing requirements on `J`,
including its common X/Z pairing, augmentation, completeness, and regular
logical `G` action.

This change automates the existing orbit-basis contract. It does not silently
replace it with an arbitrary binary basis or return a code with an uncertified
logical action. A failed orbit-basis requirement does not mean the parent
fails to define an ordinary CSS code.

The broader proposal to construct directly from redundant binary checks and
certify a complete `J` without a free check-orbit basis remains a separate
extension. It is not needed for this automatic selection feature.

## Public API and naming

Use the following signature, retaining existing positional argument
behavior and explicit overrides:

```python
class SymCityCode(CSSCode):
    def __init__(
        self,
        block_coefficients: abstract.RingArray
        | Sequence[Sequence[Sequence[abstract.RingMember]]],
        logical_ops: abstract.RingArray | None = None,
        *,
        check_rows: Sequence[int] | None = None,
        check_basis: abstract.RingArray | None = None,
        check_search_limit: int | None = 10_000,
        logical_search_limit: int | None = 10_000,
    ) -> None:
        ...
```

The general default call becomes:

```python
from qldpc import codes

# block_coefficients contains the user's p x t x q group-ring coefficients.
code = codes.SymCityCode(block_coefficients)
assert code.check_basis_result.status == "found"
assert code.logical_frame_result.status == "found"
```

The user does not need to predict which rows will be independent. A successful
call certifies both the automatically resolved check basis and the compatible
logical frame. It is still possible for either solver to prove infeasibility
or exhaust its own budget; those outcomes must remain distinguishable.

`check_rows` and `check_basis` remain mutually exclusive. An explicit argument
bypasses automatic selection but still undergoes complete validation. Invalid
explicit inputs must raise an error rather than being replaced automatically.
An explicit `logical_ops` does not bypass check-basis validation or selection.

Name the actual class `SymCityCode`, including displayed names and error
messages, and export it through `qldpc.codes`. Use this name throughout the
library, tests, and examples. No compatibility alias is provided.

Rename the private helper module and its tests from `_cyclic.py` and
`_cyclic_test.py` to `_symcity.py` and `_symcity_test.py`, updating imports.
The existing public `LogicalFrameResult` and `LogicalFrameError` names remain.
The branch name need not change.

## Fast path: rank tests on complete orbit blocks

Use exact ranks of `galois.GF2` arrays throughout. Do not cast to ordinary
NumPy integer arrays and accidentally perform real-valued rank calculations.
Lift the parent once and reuse it: the binary block belonging to prototype
`i` is `M[i*N:(i+1)*N]`.

First handle the inexpensive decisive cases:

- If `D == 0`, use an empty polynomial basis of shape `(0, m)`.
- If `D % N != 0`, no complete independent orbit basis exists. Return a
  proved failure of this requirement.
- Otherwise set `r = D // N`. If `D == P*N`, use the entire parent directly.
- Check `s = m - 2r >= 1` before starting a potentially expensive fallback.

For the remaining cases, scan parent prototypes in their original order.
Let `B` contain the binary orbit blocks selected so far, and let `Q_i` be the
next complete orbit block. Compute

$$
\Delta_i=\operatorname{rank}_{\mathbb F_2}
\begin{pmatrix}B\\Q_i\end{pmatrix}
-\operatorname{rank}_{\mathbb F_2}B.
$$

- If `Delta_i == N`, append the entire orbit and record the prototype index.
- If `Delta_i == 0`, the orbit is redundant relative to `B`; skip it.
- If `0 < Delta_i < N`, the orbit is partially independent. Skip it for this
  greedy pass, but do not classify it as redundant or remove it from the
  fallback domain.

Stop successfully once the selected rank reaches `D`. At that point the
selected rows are independent and span the entire parent space. Selection
must be deterministic and preserve the order of retained parent prototypes.

Greedy failure is not a nonexistence proof. Do not freeze a failed greedy
prefix when beginning the fallback, or discard prototypes merely because
their rank increment was partial relative to that prefix.

## Bounded fallback solver

Prefer selected parent rows before constructing polynomial combinations.
This preserves the supplied checks' supports and generally keeps their
weights smaller. Both fallback stages use the same `check_search_limit`.

### 1. Alternative subsets of parent prototypes

Enumerate subsets of `r = D/N` parent prototypes deterministically. A subset
`S` succeeds precisely when the binary matrix formed from its complete orbit
blocks has rank `D`. Because it is a subset of the parent, equal rank also
certifies equal row space.

Use a simple bounded enumeration initially; reuse already evaluated subsets
where convenient. Do not assume a matroid exchange rule for orbit blocks.
If every subset fails, the proved conclusion is only that no basis can be
selected from the displayed prototypes. Polynomial combinations can still
produce a basis.

### 2. Polynomial combinations from the full check space

For a complete finite fallback domain, compute a binary row-space basis `U`
of `M`, with `D` rows. Every binary check vector is uniquely expressible as

$$
v=\lambda U,\qquad \lambda\in\mathbb F_2^D.
$$

Each such vector can be used as the identity-translate row of one polynomial
prototype. Its full `G` orbit lies in the parent check space, which is closed
under `G`. This gives a domain containing every possible polynomial basis
prototype in the check module, not just combinations with constant binary
coefficients on the original prototypes.

Account explicitly for the regular-representation convention when converting
between a binary vector and a polynomial row. If `v` is reshaped into `m`
blocks indexed by `g in G`, the coefficient of `g` in the corresponding
polynomial entry is the block entry at `g^{-1}`. Use `group.index`,
`group.identity`, and the actual enumeration from `group.generate()`;
do not assume identity is at index zero or use the group's possibly nonregular
default lift. Construct with `RingArray.from_field_array`, and verify that
the identity-index row of `regular_lift()` equals `v`.

Enumerate nonzero coefficient vectors lazily. Backtrack over collections of
`r` candidate prototypes, accepting an extension only when its complete orbit
increases the current rank by `N`. Use a fixed candidate ordering and increasing
candidate indices to avoid row-order permutations of the same trial basis.
Start with an empty selection rather than the failed greedy prefix.

This fallback is exhaustive when unbounded: if a free orbit basis exists,
its identity-translate vectors occur in the domain, and every prefix of that
basis passes the independence test. Exhausting all branches therefore proves
that no such basis exists. A resource limit does not establish that conclusion.

Generate candidates on demand, avoid materializing all `2**D` vectors or all
bases, and bound any caches. This is a correctness-oriented fallback for small
instances, with exponential worst-case cost; do not advertise it as an efficient
general module-basis algorithm. More specialized group-algebra methods can be
added later without changing the public result contract.

### Budget semantics

`check_search_limit` is a nonnegative integer or `None`; reject booleans and
other inappropriate types consistently with `logical_search_limit`.

The greedy scan, rank divisibility checks, and direct validations do not
consume this budget. In the fallback, each trial subset rank test and each
attempted candidate-orbit extension consumes one unit, including rejected
extensions. Maintain a shared counter across both fallback stages.

- `0` permits only the fast path, for any input on which that path succeeds.
- A finite value stops after at most that many fallback trials.
- `None` permits exhaustive fallback and can be expensive.

An exhausted subset stage may continue to combinations if budget remains.
If the budget ends before exhaustive combinations, report `inconclusive`,
even when all original-row subsets have already been ruled out. A success
on the final allowed trial must be returned as success. Keep the logical-frame
search budget separate.

## Results, errors, and final validation

Add frozen `CheckBasisResult` and `CheckBasisError(ValueError)` alongside the
existing logical-frame types. Export the public result and error types using
the repository's existing `quantum.py` and `codes/__init__.py` conventions.

The result should contain:

- `status`: `found`, `infeasible`, or `inconclusive`.
- `check_basis`: the validated polynomial basis on success, otherwise `None`.
- `selected_rows`: a tuple when the basis selects parent rows; `None` for
  combinations or a general explicitly supplied basis. Use `()` for zero rank.
- `parent_rank`: the exact binary rank `D`.
- `method`: `supplied`, `parent`, `greedy`, `subsets`, `combinations`, or `rank`.
- `num_candidates`: fallback trial count, excluding the fast path.
- `message`: a concise explanation of the result and, for failure, its scope.

Store successful metadata in `code.check_basis_result`; expose failure through
`CheckBasisError.result`. Preserve `code.check_basis`, `parent_matrix`,
`num_check_orbits`, `num_logical_orbits`, and `logical_frame_result`.

For explicit basis inputs, malformed data or failure of the requested basis
validation remains `TypeError` or `ValueError`. Automatic failure uses
`CheckBasisError`. A successfully resolved check basis followed by a failed
logical-frame construction still uses `LogicalFrameError`.

Before installing any automatically found basis, independently replay:

1. Ring and shape validation, and `rank(H_bin) == r*N == D`.
2. `rank(vstack([H_bin, M])) == D`, certifying the same check space.
3. Check commutation and `s >= 1`.
4. The existing complete logical-frame validation after deriving or accepting `J`.

A solver claiming success with an invalid basis is an internal implementation
error, not a reason to silently continue searching. Error messages must
distinguish a nonfree check module from an inconclusive computation, and must
not imply that the underlying commuting parent is an invalid CSS code.

## Constructor integration

Keep constructor orchestration in `quantum.py` and solver details in the
renamed `_symcity.py`, following the current separation of responsibilities.

1. Validate options, normalize/copy coefficient inputs, and build the full parent.
2. Validate full-parent commutation and compute its regular lift and binary rank.
3. Validate an explicit check selection/basis, or invoke automatic selection.
4. Record the independent polynomial basis, orbit counts, and basis result.
5. Accept and validate explicit `logical_ops`, or run the existing fiber
   shortcut and bounded logical-frame solver with their unchanged semantics.
6. Initialize `CSSCode` with the independent binary basis and install the
   validated common X/Z logicals as today.

Keep the full redundant parent available as `parent_matrix`; this plan does
not change `matrix_x`/`matrix_z` to store the full parent. It therefore preserves
the existing meaning of check counts and installed check matrices. Polynomial
combination bases can have larger check weights than the original parent;
report their method and preserve access to the original presentation. Do not
promise weight optimization, distance, or a decoder-performance improvement.

An independent basis need not itself have a cyclic row presentation. Its
equality of check space with the full parent preserves the physical-cycle
symmetry at the stabilizer-space level.

## Tests with mathematical fixtures

Add helper-level tests and constructor regressions that exercise different
mathematical outcomes, rather than only duplicating the solver implementation.

### General fast paths and supported domains

- A parent that is already independent, a zero parent, duplicate/zero
  prototypes, and trivial `G` (ordinary binary independence).
- Multiple check families and more than two physical cycles, with redundant
  prototypes in different positions. Compare automatic and explicit bases by
  their row spaces, rather than imposing a particular representative when
  alternatives are valid.
- Reorder parent families and apply invertible polynomial row combinations to
  generate presentations of the same check space. Verify correct rank and
  span certification even when the selected basis or solver stage changes.
- Retain even physical-cycle support and groups whose default matrix lift
  is not their regular representation. The new solver must not impose an
  odd-`q` or odd-`N` restriction.
- Exercise trivial, cyclic, and noncyclic groups, including both even and odd
  orders. Use the cyclic Steane presentation over `C_4` and `C_2 x C_2` as
  existing fixtures, alongside the general module fixtures below.

Retain STAR-56/66 only as existing integration regressions: construct with no
explicit basis, compare against their previous explicit constructions and
`BBCode`, and verify their parameters and logical action. They are not the
design target or a special branch of the algorithm.

### Greedy fails but another parent subset succeeds

Over `R = F_2[C_3]`, write

$$
e_0=1+x+x^2,\qquad e_1=x+x^2,
\qquad v_1=(1,1,0,0,0),\quad v_2=(0,0,1,1,0).
$$

Use physical cycle length `q = 1` and parent rows

$$
h_0=e_0v_1+e_1v_2,\qquad h_1=v_1,\qquad h_2=v_2.
$$

All rows commute. The orbit of `h_0` has rank 3. Adding `h_1` or `h_2`
raises the rank only to 5 or 4, respectively, so the first greedy pass stops
short of the full rank 6. The subset `(h_1,h_2)` has rank 6 and is a complete
orbit basis. Verify that the subset fallback replaces the greedy choice.
An explicit compatible frame for constructor isolation is
`J = (1,1,1,1,1)`.

### A basis requires polynomial combinations

Over the same ring, take `q = 1`, `v = (1,1,0)`, and parent rows
`e_0*v` and `e_1*v`. Their individual orbit ranks are 1 and 2, while the parent
rank is 3. No one-row subset is a free orbit basis. Their sum is `v`, whose
orbit has rank 3. Verify that exhausting parent subsets is not called global
infeasibility, and that the combinations fallback returns a valid basis.
Use `J = (1,1,1)` when isolating check-basis behavior from the logical solver.

### Genuine impossibility and incomplete computation

- Over `F_2[C_2]`, let `u = 1+x`. A parent with rows `(u,0,0,0)` and
  `(0,u,0,0)` has binary rank 2, divisible by `N = 2`, but every check is
  fixed by `x`. No candidate has a rank-2 orbit. Exhaustive fallback must
  report `infeasible`; a small interrupted budget must report `inconclusive`.
- A rank not divisible by `N` gives an immediate proved orbit-basis failure.
- Check exact budget boundaries, zero budget, unlimited search, invalid
  limits, and a success on the final allowed candidate.
- Verify the binary-vector/polynomial-prototype conversion with an order-3
  group, where inversion is nontrivial, and with a noncyclic group.
- An independent explicit basis spanning a different space must be rejected.
  Preserve explicit `check_rows` and `check_basis` errors and mutual exclusion.
- Force an invalid reported solver success to exercise final replay validation.
- Retain all existing logical-frame infeasible/inconclusive cases, so new
  check selection does not hide a logical-frame failure or change its status.

The two `C_3` fixtures and the conversion convention above are covered by
automated regression tests, including reordered group enumeration with the
identity away from index zero.

## Files, documentation, and validation

Update the class, imports, exports, and tests in:

- `src/qldpc/codes/quantum.py` and `quantum_test.py`.
- The renamed `src/qldpc/codes/_symcity.py` and `_symcity_test.py`.
- `src/qldpc/codes/__init__.py`, including sorted imports and `__all__`.
- `src/qldpc/circuits/transversal_test.py`, retaining the existing gate checks.

Rename `examples/cyclic_physical_types.ipynb` to `examples/symcity_codes.ipynb`.
Use `SymCityCode` throughout and regenerate stored outputs. Explain the general
default constructor, the coefficient tensor's shape, and the separate roles
of physical cycles and the logical group. Demonstrate redundant checks,
multiple check families, a noncyclic group, and inspecting both solver result
types; retain a short explicit-basis override example. Existing named-code
examples can remain as regressions and illustrations. Update the Sphinx examples
toctree in `docs/source/examples/index.rst` and add a concise family entry to
`README.md`.
Generated API documentation should follow the source exports.

Keep the older local plans as historical documents. Add a brief forward link
to this plan and use `SymCityCode` in their API examples and the pending
binary-reduction plan, with a basis resolved explicitly or automatically.
Do not rewrite unrelated experiments or historical outputs.

Run focused solver and constructor tests first, then affected construction,
solver, and transversal modules with the repository's modular coverage checks.
Follow `pyproject.toml` conventions: typed helpers, Google-style docstrings,
100-column formatting, and sorted public exports. Use the existing check scripts
for formatting, linting, typing, and coverage; investigate any new failures and
separate unchanged baseline issues from regressions.

Execute the updated notebook, including its assertions, and build the Sphinx
documentation in the configured development environment. Verify that objects
display the `SymCityCode` name and that library imports, tests, and examples
use that name consistently.

## Acceptance criteria

- Automatic basis resolution is the default for every supported input shape
  and group, with no special handling for a named code or polynomial template.
- Every successful result certifies complete independent `G` orbits spanning
  exactly the original check space; explicit overrides remain available.
- A greedy dead end can be recovered by another subset, and a basis that
  requires combinations can be recovered by the general fallback.
- Exhaustive failure and budget exhaustion have distinct, accurate results.
- Resolving the check basis preserves the existing logical-frame requirements
  and does not claim that a compatible `J` follows from check independence.
- `SymCityCode` is the sole exported and documented name for the family.

The implementation includes the rename, automatic basis
selection and fallback, updated examples and documentation, and regression tests
for the corresponding behavior changes.
