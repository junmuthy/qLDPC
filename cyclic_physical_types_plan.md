# Plan: polynomial CSS codes with cyclic physical types

Status: implemented on `code/cyclic-physical-types`. The sections below preserve
and explain the implementation contract; the delivery notes record validation.
The coefficient-list input is retained. No symbolic-input frontend is included.

The [SymCityCode follow-up](symcity_code_plan.md) is now implemented: the public
family is named `SymCityCode`, and check-orbit basis resolution is automatic
by default. The implementation and
notebook are now `_symcity.py` and `symcity_codes.ipynb`. The original contract
and delivery notes below are preserved as a historical record, with API names
updated to match the library.

## Delivery notes

Implemented `SymCityCode`, explicit or automatic polynomial logical frames,
independent check-orbit bases, and structured solver outcomes. Solver internals
live in `src/qldpc/codes/_symcity.py`; public classes are exported through
`qldpc.codes`. Added the executable notebook and existing-circuit integration tests.

Validation in the existing Python 3.11 environment:

- All 65 tests in `quantum_test.py`, `_symcity_test.py`, and `transversal_test.py`
  pass, with 100% modular coverage of the three associated implementation modules.
- Repository formatting passes; linting passes for every changed Python/notebook
  file. Whole-repository lint reports the existing `Pauli.I` E741 issue in
  `objects.py:38`, reproduced on the unchanged base checkout.
- Type checking the changed Python files reports only the existing unused
  `type: ignore` on the HGP `matrix_z` assignment, also reproduced on the base.
- All six notebook code cells were executed with captured outputs and assertions.
  They ran in process because this sandbox disallows Jupyter kernel sockets.
- Sphinx generated the API sources, but the full documentation build stopped at
  the existing `examples/basics.ipynb` because Pandoc is absent from this environment.
  A complete HTML build remains to be run in the documented development environment.

The overlapping-logical test below uses `C_4` and the correction
`(1+x+x**2+x**3)h_0`. The previously proposed `C_2` correction `(1+x)h_0`
changes representatives but leaves their translated supports disjoint.

The stronger reduction-at-`y=1` prescription is documented in the
[follow-up plan](cyclic_physical_types_reduced_logicals_plan.md), prepared
after implementation commit `2aa2e975`. It is not part of this implementation. Full-repository coverage, STAR distances, and noise simulations
were not run as part of this delivery.

Target branch: `code/cyclic-physical-types`, based on local `main` at
`ca2be1e72a47855b990d3d3245c360f7ec104d6d`.

## Objective and scope

Add a public `SymCityCode(CSSCode)` class for the family described in the
[cyclic physical-type construction note](../../megastaq/qec-code-search/docs/polynomial-css-cyclic-physical-types.md).
The class name is a proposal. Its contract should be settled before implementation.

The user supplies scalar block polynomials and, when needed, an independent
check basis. A common logical frame `J` is optional: when omitted, the class
attempts to determine it and reports whether it found a compatible frame, proved
that none exists, or stopped without a conclusion. A supplied frame is validated
and used as an override. A successful construction returns a normal qLDPC code
object with the stated dimension and logical translation action. Arbitrary
polynomials do not guarantee admissibility or distance.

Support arbitrary finite abelian logical groups, including even-order and
noncyclic groups; any positive number of equal-length physical cycles; multiple
check families; and any compatible positive logical orbit count. Include the
STAR cases and an `s=1` example. Do not impose the BB two-block layout, four-term
polynomials, or disjoint fiber logicals on the general class.

This addition includes a bounded logical-frame solver for fixed checks. Searching
over code parameters or generating admissible check polynomials and independent
check-orbit bases remains outside its scope. No new decoder or syndrome schedule
is needed. The solver must handle the general coefficient constraints, with
proved shortcuts for known cases; automatic mode must not be restricted to STAR
or fiber logicals.

## Existing implementation to follow

- [`GALACode`](src/qldpc/codes/quantum.py#L500) provides the closest input and
  presentation conventions: typed sequences of `RingMember`, defensive copies,
  block-circulant assembly, retained ring-valued parent matrices, descriptive
  validation errors, and a short executable construction in the class docstring.
  Its tests are colocated in [`quantum_test.py`](src/qldpc/codes/quantum_test.py#L126).
- [`GroupRing` and `RingArray`](src/qldpc/abstract/rings.py) already supply the
  coefficient algebra, adjoint, binary expansion, and coefficient access.
  `RingArray.T` is the conjugate transpose, including inversion of group elements.
  No additional polynomial-algebra dependency is needed.
- [`CSSCode`](src/qldpc/codes/common.py#L2177) supplies check matrices, dimensions,
  distance routines, Tanner graphs, and integration with the rest of the library.
  `set_logical_ops_xz` installs and validates a supplied pair of logical bases.
- `CSSCode.is_swel` and `set_swel_logical_ops` already address self-dual codes
  with equivalent logicals. An automatically selected binary SWEL basis does not
  establish the required arrangement into regular `G` orbits or the augmentation
  convention. Install the validated supplied or solver-generated `J` explicitly.
- [`QCCode` and `BBCode`](src/qldpc/codes/quantum.py#L651) supply polynomial-input
  examples and an independent comparison for STAR. Their lifting and coordinate
  conventions differ: `QCCode` transposes its lifted polynomial matrices.
- [`get_transversal_s`](src/qldpc/circuits/transversal.py#L36) already constructs a
  signed physical phase layer implementing logical S on every logical qubit of a
  SWEL code. Reuse it and test its action in the resolved frame.

Use direct inheritance from `CSSCode`. GALA permits latent parent rows that do
not define stabilizers and validates active orthogonality. In this family, the
full cyclic parent must commute, and omitted rows must be redundant. These are
different contracts even though the circulant indexing is the same.

Use a small private assembly helper in the new class initially. Calling GALA's
private helper would couple unrelated classes; extracting a shared helper is
reasonable only if it improves both implementations without changing GALA's API.

## Mathematical contract

Let `G` be finite abelian, `N=|G|`, and `R=F_2[G]`. Choose positive `p,t,q` and
scalar polynomials

$$
f_{ab}(\mathbf x,y)=\sum_{d=0}^{q-1} f_{ab,d}(\mathbf x)y^d,
\qquad f_{ab,d}\in R,
\qquad \mathcal H=(f_{ab})\in (R[y]/(y^q-1))^{p\times t}.
$$

Construct the parent over `R`, using zero-based indices,

$$
\widetilde H_{aq+u,bq+v}=f_{ab,(v-u)\bmod q}.
$$

It has `pq` polynomial check rows and `m=tq` physical types. Require an
`r x m` matrix `H` whose complete `G` translates are independent and span exactly
the parent check space. Set `s=m-2r>=1`. The supplied or automatically determined
`s x m` logical matrix `J` must satisfy

$$
\widetilde H\widetilde H^\dagger=0,\qquad
\operatorname{rank}_{\mathbb F_2}H_{\rm bin}=rN,\qquad
\operatorname{rowspan}H_{\rm bin}
 =\operatorname{rowspan}\widetilde H_{\rm bin},
$$

$$
HJ^\dagger=0,\qquad JJ^\dagger=I_s,\qquad
\mathbf1_sJ_{\rm aug}=\mathbf1_m.
$$

All binary matrices here use the **regular** representation of `G`.
The parent Gram condition is equivalent to `mathcal H mathcal H^dagger=0`.
Augmentation sums coefficients in `G` only; the physical cycle has already been
expanded into type indices. These conditions imply `HH^dagger=0`.

Initialize `H_X=H_Z=H_bin` and `L_X=L_Z=J_bin`, with the usual positive
generator convention. Then `n=tqN` and `k=sN`. The stored check matrices use the
independent `H`, matching the direct construction; keep the redundant parent as
metadata. It can support a later, explicitly designed option for redundant
syndrome measurements, whose decoder and circuit behavior can differ.

The physical `G x C_q` action preserves the code. Only `G` is prescribed to act
regularly on each of the `s` logical orbits. The additional cycle's logical action
must be computed; it is trivial on the displayed STAR and single-block frames.
For `s=1`, `tq` must be odd. Even `q` remains allowed for compatible `s`.

## Proposed public API

Place the class beside GALA/QC in `src/qldpc/codes/quantum.py` and export it through
`src/qldpc/codes/__init__.py`.

```python
class SymCityCode(CSSCode):
    def __init__(
        self,
        block_coefficients: Sequence[Sequence[Sequence[abstract.RingMember]]],
        logical_ops: abstract.RingArray | None = None,
        *,
        check_rows: Sequence[int] | None = None,
        check_basis: abstract.RingArray | None = None,
        logical_search_limit: int | None = 10_000,
    ) -> None:
        ...
```

`block_coefficients[a][b][d]` is the coefficient of `y**d` in `f_ab`.
Require a rectangular, nonempty `p x t x q` array. Every entry, including zero,
is an element of one binary commutative `GroupRing`. This is a coefficient
representation of scalar block polynomials, not a request to enter the expanded
check matrix. Coefficients can themselves be polynomial sums in any number of
logical-group generators. The API deliberately follows GALA's `RingMember`
convention; `GroupRing.eval` already accepts symbolic expressions for each
coefficient. A whole-expression SymPy convenience constructor can follow later.

When supplied, `logical_ops` is the polynomial `J`, not a binary matrix or full
symplectic matrix. Its entries belong to the same ring. Validate it directly;
do not replace an invalid explicit frame through an automatic retry. With
`logical_ops=None`, invoke the logical-frame solver after validating the checks.
Infer `G,p,t,q` from the block inputs, `r` from the chosen basis, and `s=tq-2r`.
Validate the supplied or generated number of logical rows rather than accepting
conflicting `r,s,k` arguments.

`logical_search_limit` bounds the number of candidates evaluated by the general
solver. The proposed initial default is 10,000; accept nonnegative integers or
`None` for explicitly unbounded enumeration. Zero skips enumeration but still
allows proved shortcuts and linear inconsistency checks. This is a candidate
limit, not a wall-clock guarantee. Validate the option even with an explicit
frame, but do not run the solver or consume its budget in that case.

There are three check-basis cases:

1. `check_rows` selects rows of the polynomial parent, each representing one
   complete `G` orbit. Validate integer indices, range, and uniqueness. Selected
   rows must span the entire parent and have binary rank `rN`.
2. `check_basis` supplies `H` directly over `R`, permitting polynomial row
   combinations when a row subset is insufficient. Validate independence and
   equality of the lifted row spaces. Never interpret it as a change of code.
3. With neither argument, use the entire parent and require it to be independent.
   A redundant parent raises an error explaining the two basis arguments.

Reject simultaneous `check_rows` and `check_basis`. Support `r=0` with an empty
selection or a correctly shaped empty `RingArray`, provided the parent is zero
and a supplied or generated `J` passes the conditions. Automatic selection of
the independent check-orbit basis remains deferred; automatic logical-frame
construction is part of this first API. Do not initially expose
`skip_validation`: the checks certify both supplied and generated data.

Retain copied `block_coefficients`, `ring`, `group`, `cycle_order`,
`num_type_cycles`, `num_check_families`, `num_check_orbits`, `num_logical_orbits`,
`parent_matrix`, `check_basis`, `logical_prototypes`, and a structured
`logical_frame_result`. Keep inherited
`matrix_x`, `matrix_z`, `dimension`, and `get_logical_ops` as the normal binary
interface. A concise `__str__` should report physical qubits, the logical group,
cycle length, and logical orbit count. Avoid GALA's name `num_active_rows`:
these selected rows are a basis of all checks.

An intended STAR-56 construction with automatic logicals is:

```python
from qldpc import abstract, codes

ring = abstract.GroupRing(abstract.CyclicGroup(4))
x = ring.generators[0]
zero, one = ring.zero, ring.one
q = 7

# a(x,y) = 1 + y**3 + x*y**2 + x*y**4
a = [one, zero, x, one, x, zero, zero]
a_dagger = [a[(-d) % q].T for d in range(q)]

code = codes.SymCityCode(
    block_coefficients=[[a, a_dagger]],
    check_rows=range(q - 1),
)
print(code.logical_frame_result.message)
assert code.logical_frame_result.status == "found"
assert code.num_qubits == 56
assert code.dimension == 8
assert code.get_weight() == 8
```

An explicit frame remains available when particular representatives are wanted:

```python
j = abstract.RingArray.build(
    [[1] * q + [0] * q, [0] * q + [1] * q], ring=ring
)
code = codes.SymCityCode(
    block_coefficients=[[a, a_dagger]],
    logical_ops=j,
    check_rows=range(q - 1),
)
```

These snippets illustrate the explicit-basis API. They make no distance claim
from the weight calculation.

## Automatic logical-frame construction and reporting

For fixed validated `H`, represent the unknown `J` by its `smN` binary
coefficients over `R`. Its expanded rows are the `G` translates of `s` prototypes;
the Gram condition then ensures their independence and complete regular orbits.
This imposes neither disjoint supports nor a fiber layout.

Use the following initial solver, with no new external solver dependency:

1. Try the proved fiber shortcut when `q` is odd, `s=t`, and every block
   polynomial obeys `f_ab(x,1)=0` in `R`. Form `J=I_t tensor 1_q` and pass it
   through the full validator. An inapplicable shortcut is not a failure of the
   general construction; continue to the coefficient solver.
2. Assemble the affine binary equations for `HJ^dagger=0` and
   `1_s J_aug=1_m`, and solve them by exact finite-field elimination. An
   inconsistent system proves that no compatible frame exists. Otherwise
   parameterize all solutions by a particular solution and a null-space basis.
3. Enumerate the free binary coefficients in a deterministic order, checking
   `JJ^dagger=I_s`. This provides a complete reference method if the affine
   solution space is exhausted. Enumerate lazily and stop at the candidate limit;
   the method is exponential and has no general efficiency guarantee. Avoid a
   hidden restriction to low-weight, constant, or fiber logicals.
4. Replay the full logical validator on any candidate returned as a solution,
   then install it using `set_logical_ops_xz`. A candidate that fails this replay
   is an implementation error, not a valid construction or a nonexistence result.

The quadratic constraints are now a software responsibility. Later SAT/SMT or
algebraic backends can replace enumeration without changing the input or outcome
contract. Existing binary SWEL routines may supply candidate frames, but those
candidates must also satisfy the polynomial orbit and augmentation requirements.

Define a small typed `LogicalFrameResult` carrying `status`, `logical_ops`
(`J` on success, otherwise `None`), `method`, `num_candidates`, and a readable
`message`. Expose it as `code.logical_frame_result` after success. Methods can
identify supplied data, the fiber shortcut, or coefficient solving. The outcomes
are:

- **`found`**: a complete frame passed every check. Example message:
  `Found a compatible J with 2 logical orbits (fiber shortcut).`
- **`infeasible`**: exact linear inconsistency, complete enumeration, or another
  proved obstruction establishes that no frame satisfies the stated conditions
  for these checks and group action. Example message:
  `No compatible J exists; all candidates in the affine solution space failed.`
- **`inconclusive`**: the candidate limit was reached while unexamined candidates
  remained. Example message:
  `No compatible J found within 10000 candidates; existence remains undecided.`
  State how to increase the limit or provide an explicit frame. Exhaustion on
  the final allowed candidate is `infeasible`, not `inconclusive`.

For `infeasible` or `inconclusive`, raise a public
`LogicalFrameError(ValueError)` with the result attached as `.result` and its
message as the exception text. This keeps normal constructor behavior while
making failure machine-readable. Do not return a partially initialized
`SymCityCode` or silently install a generic binary logical basis. Invalid
checks or an invalid supplied frame are input-validation errors, distinct from
solver outcomes. Internal exceptions are implementation failures and must not
be relabeled as mathematical infeasibility.

```python
try:
    code = codes.SymCityCode(
        block_coefficients=blocks,
        check_rows=selected_rows,
        logical_search_limit=10_000,
    )
except codes.LogicalFrameError as error:
    print(error.result.status, error.result.message)
else:
    print(code.logical_frame_result.status, code.logical_frame_result.message)
```

The library itself should not print; the result and exception provide the user
feedback. A failure concerns this family's compatible logical frame. The
validated checks can still define a valid ordinary CSS code. Report that
distinction without silently changing the requested family.

## Implementation sequence and validation

1. Normalize and copy inputs. Match GALA's typed methods, Google-style docstrings,
   and descriptive `ValueError`/`TypeError` messages. Check binary field, abelian
   group, rectangular coefficient data, compatible rings, and matrix shapes.
   Normalize entries to one chosen ring and group enumeration: equal group rings
   can carry different lift metadata. `RingArray.build(..., ring=ring)` copies
   entries; an object-array copy alone does not copy mutable `RingMember` objects.
2. Assemble `parent_matrix` with the formula above. Validate its Gram matrix using
   `RingArray.T`, not NumPy's ordinary transpose. Resolve `H` and check `s>=1`.
3. Expand using `RingArray.regular_lift()`. GALA's `.lift()` is intentionally more
   general and can use a nonregular representation. A representation of dimension
   `N` alone does not prove it is regular. Preserve `galois.GF2` array types for
   all ranks and row operations; casting to ordinary NumPy first changes the field.
4. Verify full check rank and equality of row spaces. For example, require both
   `rank(parent_bin)=rank(H_bin)=rN` and
   `rank(vstack([parent_bin,H_bin]))=rN`. Do not merely check that the binary rank
   is divisible by `N`, or reshape a binary row basis into polynomial rows.
5. Resolve the logical frame. Validate an explicit `J` or invoke the bounded
   solver above when it is omitted. In both paths use the same validator for
   logical commutation, orthonormality, and augmentation. Compute augmentation
   through `J.to_field_array().sum(axis=-1)` in the finite field. Preserve a valid
   explicit frame exactly; report it as `found` with method `supplied`. Validation
   failures should identify the violated condition and relevant shape or rank.
   Solver failure must carry its structured outcome and must not silently produce
   a generic CSS code with a different logical frame.
6. Initialize `CSSCode(H_bin, H_bin, is_subsystem_code=False)` and call
   `set_logical_ops_xz(J_bin, J_bin)` with validation enabled. The existing setter
   preserves the resolved row order. Do not call `set_swel_logical_ops` afterward.
   Compute weight and distance through inherited methods; do not cache an assumed
   distance. Existing methods already recognize equal X/Z checks and distances.
7. Document coordinate order and test symmetry explicitly. Use physical index
   `((b*q+v)*N+i)` and logical index `ell*N+i`, where `i` indexes
   `tuple(group.generate())`. Translation maps `(b,v,g)` to `(b,v,h*g)` and
   `(ell,g)` to `(ell,h*g)`; physical-type translation maps `v` to `v+1`.
   State whether any exposed permutation array maps old indices to new indices.
   Derive permutations from group multiplication, not assumptions about generator
   ordering. Check the cycle action on the stabilizer row space after basis
   selection; the chosen rows need not themselves be permuted as a list.

Keep permutation construction in examples/tests initially, using the documented
labels. Add a public helper only if a concrete consumer needs it. Logical-frame
guarantees refer to the installed frame; a later user-requested logical basis
change through inherited methods can change the displayed action.

The first implementation can use dense matrices, like GALA. Rank, row-space, and
logical affine-system elimination will limit its practical size; enumeration
can be substantially more expensive. Sparse/packed elimination, stronger logical
solvers, and component methods are later optimizations that must preserve the
same validation and outcome contract.

## Tests and acceptance fixtures

Add focused tests alongside GALA in `src/qldpc/codes/quantum_test.py`, with
independent small binary oracles rather than copying the implementation formula
as the only expected result.

- **Expansion and conventions.** Check a hand-written small circulant and its
  binary Kronecker expansion, coefficient sums and cancellations, inverse group
  elements, cyclic wraparound, and adjoints. Check a nontrivial even physical
  cycle as well as `q=1`. Verify two or more check families mutually commute when
  required; checking each block row against itself is insufficient.
- **STAR regressions.** For `(ell,q)=(4,7)` and `(3,11)`, use
  `a=1+y**3+x*y**2+x*y**4`, its adjoint, the first `q-1` parent rows, and two
  fiber logicals. Exercise both automatic and supplied-frame construction.
  Expect `(n,k,rank(H_X),weight)=(56,8,24,8)` and `(66,6,30,8)`.
  Compare both stabilizer row spaces against existing `BBCode` after an explicit
  coordinate permutation; compare the logical rows to the fiber oracle.
  These checks do not certify distances six and eight.
- **One block and one logical orbit.** Use `q=7`, `f=1+y+y**2+y**4`, the first
  three rows, and `J=(1,...,1)`. Test trivial `G`, `C_2 x C_2`, and `C_2 x C_3`.
  Cover omitted `J` as well as the explicit override.
  Expect `[[7N,N,3]]`; certify distance three on the seven-qubit fixture using
  all errors of weight at most two and the weight-three support `{0,1,5}`.
  Larger constant-coefficient instances are independent copies of this code.
- **General layout and frame.** Use three diagonal copies of the preceding
  block (`p=t=3`, `r=9`, `s=3`). Test an explicit check basis obtained through an
  invertible polynomial row operation. Also replace the one-block logical row
  by `J+(1+x+x**2+x**3)h_0` over `F_2[C_4]`: it preserves all conditions while introducing
  nonconstant logical entries and overlap between translated representatives.
  Preserve this explicit override rather than replacing it with the fiber frame.
  Do not enforce fiber support or disjointness in validation.
- **General automatic solving.** Use `q=1`, `p=1`, `t=3`, and parent
  `H=(1,1,0)` over `F_2[C_2 x C_2]`. Here `r=1`, `s=1`; the fiber shortcut is
  inapplicable because `s!=t`. The frame `J=(1,1,1)` proves feasibility, but do
  not supply it. Require the coefficient solver to return `found` and replay all
  identities, rank/completeness, and logical translations on its returned frame.
  Any valid frame is acceptable; do not require this particular representative.
- **Solver outcomes and limits.** For a proved negative case, use trivial `G`,
  `q=4`, `p=t=1`, `f=1+y+y**2+y**3`, and `check_rows=[0]`. Its independent
  check is `H=(1,1,1,1)`: every commuting support is even, so no orthonormal
  logical frame exists. Require `infeasible` and an explanatory result. Exercise
  consistent and inconsistent affine systems, complete candidate exhaustion,
  and success after rejecting candidates. On the preceding feasible nonfiber
  fixture, a zero candidate budget must yield `inconclusive`, not `infeasible`.
  Test success and exhaustion exactly at the limit, explicit unbounded mode on
  tiny fixtures, and invalid budgets. Assert that failures carry `.result`,
  success exposes `logical_frame_result`, and constructors do not print.
  Verify that explicit frames bypass solving and that solver failures never
  return a partially initialized family object. Use meaningful small constraints
  and deterministic limits rather than timing-sensitive tests.
- **Representation and symmetry.** Include `AbelianGroup(2,2,direct_sum=True)`:
  its available nonregular lift has dimension four, so a dimension-only guard
  would miss the mistake. Confirm use of the regular lift, the complete `G`
  logical permutation action, and invariance under the physical cycle. Assert
  that repeated access to inherited logical getters preserves the installed frame.
- **Independent validation failures.** Cover malformed shapes, empty coefficient
  axes, mixed rings, nonbinary and nonabelian inputs, conflicting basis arguments,
  invalid row indices, dependence, lost parent checks, added independent checks,
  noncommutation, incorrect logical counts, bad logical pairing, and failed
  augmentation. For the latter, `J+h_0` in the constant single-block example
  preserves commutation and norm but violates the stated augmentation convention.
  Cover `r=0` and defensive copying as well.
- **Nonfree check module.** Over `F_2[C_2]`, set `u=1+x` and take the `q=1`
  parent with rows `(u,0,0,0)` and `(0,u,0,0)`. It commutes and has binary rank
  `2=N`, but every vector is annihilated by nonzero `u`. No rank-one free
  `R`-basis exists. Test rejection of a proposed single-orbit basis despite the
  divisible binary rank. This is a requirement of the original direct family,
  not an obstruction introduced by physical-type cycles.
- **Existing circuit integration.** For a small fixture, use
  `circuits.get_logical_tableau` to check tensor Hadamard and the output of
  `circuits.get_transversal_s`, including logical signs. Put circuit-focused
  integration tests in `src/qldpc/circuits/transversal_test.py` if needed. Keep
  exact-distance searches and Monte Carlo noise studies out of constructor tests.

## Files, documentation, and checks

Implementation changes should initially touch:

- `src/qldpc/codes/quantum.py`: class, private normalization/assembly/validation
  and logical solver, `LogicalFrameResult`, `LogicalFrameError`, construction
  docstring, and readable representation. If the solver grows too large, move
  its internals to a private module with colocated tests.
- `src/qldpc/codes/__init__.py`: sorted public imports and `__all__` entries for
  `SymCityCode`, `LogicalFrameResult`, and `LogicalFrameError`.
- `src/qldpc/codes/quantum_test.py`: mathematical and API acceptance tests.
- `src/qldpc/circuits/transversal_test.py`: small integration checks if kept
  separate from code construction tests; no new phase-gate implementation needed.
- `examples/cyclic_physical_types.ipynb`: executable STAR and noncyclic `s=1`
  examples, automatic solving outside the fiber case, explicit overrides,
  coordinate conventions, overlapping logicals, and useful invalid inputs.
  Demonstrate success, proved nonexistence, and a budget-limited attempt.
  Explain what construction validates and what distance analysis adds.
- `docs/source/examples/cyclic_physical_types.ipynb`: relative symlink to that
  notebook, following the existing BB/transversal examples; add the notebook to
  `docs/source/examples/index.rst`. API pages are generated from class docstrings.

No initial change to `abstract`, `common.py`, dependencies, or existing GALA/BB
semantics should be necessary. Keep reference code and paper data out of runtime
dependencies. The sibling research note is design context, not a portable library
dependency. Do not edit existing untracked experiments or the local GALA notebook.

Follow the repository's 100-column Python/docstring limit, annotations, sorted
imports/exports, and colocated pytest conventions. Run the targeted construction
tests first, followed by GALA/BB regressions and any affected circuit tests. In the
development environment used by CI, the relevant commands are:

```bash
uv sync --extra dev
uv run python -m pytest src/qldpc/codes/quantum_test.py -k 'cyclic_type or gala or bivariate_bicycle'
uv run python -m pytest src/qldpc/circuits/transversal_test.py -k 'transversal_s or cyclic_type'
uv run python checks/format_.py --check
uv run python checks/lint_.py
uv run python checks/mypy_.py
uv run python checks/coverage_.py
uv run python checks/build_docs.py
```

The repository requires 100% modular coverage and forbids network access in tests.
Execute the example notebook before the documentation build: Sphinx has
`nbsphinx_execute="never"` and displays saved outputs. Use the existing environment
where appropriate; these commands describe implementation acceptance, not checks
already run for this planning-only change.

## Suggested delivery order and remaining mathematics

1. Implement the validated class with optional `logical_ops`, both check-basis
   inputs, the shared logical validator, structured results/errors, the fiber
   shortcut, and bounded general coefficient solving. Include STAR-56, the
   single-block noncyclic fixture, and the nonfiber automatic case. Automatic
   general solving and its three outcomes are part of this delivery, not deferred
   until after a mandatory-`J` version.
2. Complete solver and input failure cases, STAR-66 comparison, general-layout
   and symmetry tests, circuit integration, docstring, and notebook. Run repository
   checks.
3. Evaluate additional proved prescriptions through this API. Add named
   convenience factories only when they enforce all conditions for their stated
   parameter domains. A general SymPy block-polynomial frontend can be added
   independently, with explicit logical-variable and physical-cycle assignments.

For fixed valid checks, this plan specifies a complete finite logical-frame
procedure when allowed to exhaust the coefficient domain, and a bounded attempt
by default. It provides no guarantee of practical runtime or that a frame exists
for arbitrary checks. Generating commuting check polynomials, obtaining free
independent check-orbit bases, and proving distance bounds remain separate
problems. Efficient general logical-frame synthesis is also unfinished research;
bounded solver failure must remain inconclusive rather than excluding the code.

Existing semisimple Howell/component routines apply only in their stated domains;
the polynomial Howell routine is cyclic-only. Neither supplies a general
basis-and-logical-frame generator for even-order noncyclic groups. Binary SWEL
basis selection likewise does not establish the orbit contract. These limitations
do not require the caller to supply `J`: the coefficient solver covers the full
finite domain, subject to its explicit resource limit.

Arbitrary logical S/S-dagger direction patterns can be a later feature using the
documented finite-field phase equations. The existing all-S helper is sufficient
for the first integration tests. Specialized syndrome schedules and noise
experiments should follow only after the algebraic and logical-action tests pass.

## Evidence gathered while preparing this plan

Inspected the current GALA/QC/BB implementations and tests, group rings and their
regular lifts, CSS logical-basis methods, SWEL/transversal support, documentation
layout, and check/CI configuration at the base commit above.

A temporary probe using the sibling `multistaq/.venv` Python 3.11 environment and
this checkout's `src` verified, using existing classes only:

- STAR-56 and STAR-66 ranks, dimensions, check weights, logical identities,
  augmentation, and equality with both BB stabilizer row spaces after relabeling.
- The single-block construction for trivial `G`, `C_2 x C_2` with a nonregular
  default lift, and `C_2 x C_3`, with parameters `(7,1)`, `(28,4)`, and `(42,6)`.
- Installation and retrieval of the explicit logical frame, and the existing
  logical-S circuit on the seven-qubit fixture.
- The divisible-rank, nonfree-module counterexample described above.

For the STAR comparison, this probe maps new coordinate `(b,v,g)` to BB coordinate
`(b,-g mod ell,v)`. This accounts for the different order of tensor factors and
the logical-shift transpose convention. The future tests should state this map
explicitly rather than comparing dimensions alone.

The temporary probe is feasibility evidence, not an implementation of the proposed
class or a substitute for its tests. Its frames were explicitly constructed; it
does not validate the newly planned automatic solver. No full repository suite,
documentation build, STAR distance certification, or noise simulation was run for
this plan.
