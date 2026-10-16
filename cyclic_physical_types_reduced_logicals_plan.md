# Plan: construct cyclic-type logical frames through a binary reduction

Status: follow-up plan; not implemented. Prepared after implementation commit
`2aa2e9754cb45c6995b48701f1b9e67812448099` on `code/cyclic-physical-types`.

The [SymCityCode follow-up](symcity_code_plan.md) is now implemented. This plan
extends `SymCityCode`, with the check basis resolved automatically or supplied
explicitly. This binary-reduction
logical-frame enhancement remains a separate, unimplemented addition.

## Objective

Add a sufficient set of conditions on the cyclic check polynomials under which
the common logical frame `J` follows from binary linear algebra, without general
coefficient enumeration. Retain the coefficient-list API; do not add symbolic
inputs. Preserve the existing general family and explicit `logical_ops` override.

The initial implementation already validates arbitrary supported cyclic check
presentations, accepts a supplied frame, and otherwise tries disjoint fiber
logicals followed by bounded coefficient enumeration. This addition extends the
proved shortcut to nonzero binary reductions and overlapping logical supports.
It does not generate admissible check polynomials or independent orbit bases.

Relevant implementation:

- [`SymCityCode`](src/qldpc/codes/quantum.py): check validation and construction.
- [`_symcity.py`](src/qldpc/codes/_symcity.py): frame validation, fiber shortcut,
  affine coefficient equations, enumeration, and result types.
- [`CSSCode.get_swel_logical_ops`](src/qldpc/codes/common.py): binary common
  logical bases, using finite-field orthonormalization in
  [`math.py`](src/qldpc/math.py).

## Sufficient conditions on the supplied checks

Use the existing notation: `R=F_2[G]`, `N=|G|`, `p` check families, `t` physical
cycles of length `q`, and block polynomials

$$
\mathcal H(\mathbf x,y)=(f_{ab}),\qquad
f_{ab}=\sum_{d=0}^{q-1}f_{ab,d}(\mathbf x)y^d.
$$

Keep all existing checks: the full cyclic parent commutes; the supplied `r`
polynomial check rows are independent under regular lifting and span the full
parent; and `s=tq-2r>=1`.

The additional sufficient conditions are:

1. `q` is odd.
2. The reduction at `y=1` is a **constant binary matrix**:
   $$
   B=\mathcal H(\mathbf x,1)
    =\left(\sum_d f_{ab,d}(\mathbf x)\right)_{ab}
    \in\mathbb F_2^{p\times t}\subset R^{p\times t}.
   $$
   Each entry must be exactly `ring.zero` or `ring.one`. This is stronger than
   taking the augmentation in `G`; do not substitute `x_i=1` to force this test
   to pass. Individual `f_ab,d` may still contain nonconstant group elements.
3. With `rho=rank_F2(B)`, the already validated check rank obeys
   $$
   r=\frac{t(q-1)}2+\rho,
   \qquad\text{equivalently}\qquad s=t-2\rho\ge1.
   $$
4. The all-ones vector is not a binary check:
   $$
   \mathbf1_t\notin\operatorname{rowspan}_{\mathbb F_2}B.
   $$

The full commutation identity implies `B B^T=0` by evaluating at `y=1`.
Verify this identity when implementing the reduction as an internal consistency
check. Redundant rows of `B` are permitted.

These conditions work for any finite abelian `G`, including even-order and
noncyclic groups. They do not assume a semisimple logical group algebra.
They restrict this prescription to `s<=t` and to representatives fixed by the
physical cycle. They do not impose two blocks, `s=2`, four-term polynomials,
or disjoint logical supports. For `s=1`, take `t=2rho+1` when admissible checks
with the required rank have been supplied.

The check-basis validation remains essential. Divisibility of the full binary
rank by `N` does not prove that a free basis of complete `G` orbits exists.

## Construction of the reduced frame

The binary CSS code with common check matrix `B` has `s=t-2rho` logical qubits.
Its logical quotient is

$$
V=\ker B/\operatorname{rowspan}B.
$$

The induced symmetric pairing is nondegenerate. It is nonalternating exactly
when `1_t` is not in `rowspan B`: an odd vector exists in `ker B` precisely when
`1_t` is not orthogonal to all of `ker B`. In that case binary orthonormalization
gives representatives `L` satisfying

$$
BL^T=0,\qquad LL^T=I_s.
$$

Reuse `CSSCode(B,B).get_swel_logical_ops()` for this step. This is an ordinary
binary quotient computation, not a polynomial logical-frame search.

There is one further parity correction. Form

$$
\delta=\mathbf1_t+\mathbf1_sL.
$$

Then `delta` belongs to `rowspan B`. To see this, both `1_t` and `1_s L` pair
with any `v` in `ker B` as `v v^T`; their difference therefore lies in
`(ker B)^perp=rowspan B`. Add `delta` to the first logical row of `L`.
Because `delta` is a check, this preserves commutation and the identity Gram
matrix, and now

$$
\mathbf1_sL=\mathbf1_t.
$$

Use finite-field arrays throughout these operations, not ordinary integer sums.
Verify the corrected three binary identities before lifting. This correction
is required even if the existing SWEL helper usually returns convenient rows.

## Lift to the polynomial frame and prove completeness

Set

$$
J_{\lambda,bq+v}=L_{\lambda b}\,1_R
\quad(0\le v<q),
\qquad J=L\otimes\mathbf1_q.
$$

Every parent-row product with `J^dagger` is a row of `B L^T`, independent of
the physical-cycle position. Hence the full parent, and therefore its basis
`H`, commutes with `J`. Also,

$$
JJ^\dagger=qLL^T=I_s,\qquad
\mathbf1_sJ_{\rm aug}=\mathbf1_{tq}.
$$

The first equality uses odd `q`. The regular lift has `sN` independent logical
rows with identity pairing. Since `H_bin` has rank `rN` and
`tqN-2rN=sN`, these rows are a complete common logical frame.

Thus, after the stated checks on `H`, computing `J` requires binary linear
algebra and a Kronecker expansion. The full constructor still validates the
result with the existing polynomial validator before installing it.

The logical group `G` retains its prescribed regular action on each orbit.
This particular `J` is fixed by the additional physical cycle. Different rows
of `L` may overlap, so this does not restrict `J` to disjoint physical fibers.
It gives no distance guarantee.

## Integration and reporting

Keep the public constructor signature unchanged initially. The stronger
conditions describe a sufficient subclass, not new mandatory input restrictions.

1. Keep the explicit-frame path unchanged: validate and use `logical_ops`.
2. In automatic mode, retain the current zero-reduction fiber shortcut and its
   `method="fiber"` result. It is the special case `B=0`, `rho=0`, `L=I_t`.
3. Add a private reduction helper in `_symcity.py`. It receives the validated
   checks and coefficient tensor, tests the sufficient conditions, and returns
   either a validated frame or an explanation that this prescription does not
   apply. Identify the identity group element through the ring/group API when
   extracting constant coefficients; do not assume an arbitrary lift or ordering.
4. For a successful new shortcut, return `LogicalFrameResult` with
   `status="found"`, `method="cycle_reduction"`, and `num_candidates=0`.
   Extend the existing method type and documentation accordingly.
5. If the prescription does not apply, continue the existing general coefficient
   solver. Explain the reduction failure in its diagnostic message when useful.
   Failure of these sufficient conditions must not be labeled `infeasible`.
6. If the conditions hold but the derived frame fails validation, raise an
   internal-error exception. Do not disguise a broken implementation as a
   resource-limited attempt or silently fall back.

`logical_search_limit=0` will allow this new proved shortcut, just as it already
allows the fiber shortcut. A successful binary reduction consumes no enumeration
budget. The three result statuses and the meaning of `infeasible` remain intact.
Do not add a backend, symbolic input, CLI output, or new required dependency.

Import `CSSCode` directly from `.common` inside the solver module rather than
through the public package exports. Keep all existing general solver behavior
available when reduction is inapplicable.

## Test fixtures

Use `q=7`, `f=1+y+y^2+y^4`, and `e=1+y+...+y^6` for small exact examples.
These formulas describe coefficient lists; they do not require a symbolic API.

### Nonzero reduction and one logical orbit

For `t=3`, take the polynomial block matrix

$$
\mathcal H=\begin{pmatrix}
f&0&0\\0&f&0\\0&0&f\\e&e&0
\end{pmatrix}.
$$

The only nonzero row of `B` is `(1,1,0)`. Choose parent-row indices
`[0,1,2,7,8,9,14,15,16,21]`. Then `rho=1`, `r=10`, `s=1`, and
`(n,k)=(21N,N)`. The parity-corrected reduced frame is `L=(1,1,1)`.
Test trivial `G`, `C_4`, and `C_2 x C_2` with a nonregular default lift.
These constant-coefficient fixtures are independent copies over group sites.

### Overlapping logical representatives

For `t=5`, use `f I_5` and an additional block row `e(1,1,0,0,0)`.
Choose the first three rows of each diagonal block and the first row of the
additional family: `[7b+d for b in range(5) for d in range(3)] + [35]`.
Here `rho=1`, `r=16`, `s=3`, and `(n,k)=(35N,3N)`. One valid reduced frame is

$$
L=\begin{pmatrix}
1&1&1&0&0\\1&1&0&1&0\\1&1&0&0&1
\end{pmatrix}.
$$

The first two cycles are shared by all three rows. Validate the solver's output
identities without demanding this particular representative when alternatives
exist. Include a controlled reduced-frame test that needs a nonzero `delta`.

### Nonconstant checks with a nonzero binary reduction

Let `a=1+y^3+x y^2+x y^4` over `F_2[G]`, with `x` a group generator. Take

$$
\mathcal H=\begin{pmatrix}
a&a^\dagger&0\\0&0&f\\e&0&e
\end{pmatrix}.
$$

The adjoint inverts `y` and the logical group elements. The only nonzero row of
`B` is `(1,0,1)`. Select parent rows `[0,1,2,3,4,5,7,8,9,14]`.
For `G=C_4` and `G=C_2 x C_2`, this gives `r=10`, `s=1`, and `(n,k)=(84,4)`.
It exercises nonconstant coefficients while still reducing to binary `B`.

### Boundaries and regressions

- Require automatic success with a zero enumeration budget on the new fixtures;
  check `method`, common logicals, dimensions, augmentation, and group action.
- Preserve STAR-56/66 and existing fiber outcomes, including `method="fiber"`.
- Cover even `q`, a nonconstant reduction, incorrect reduced dimension, and
  `1_t` in `rowspan B`. Test helper rejection separately from constructor
  rejection or the general solver's eventual result.
- Exercise redundant binary check rows, empty binary check space, and a valid
  general check basis rather than only selected parent rows.
- Preserve supplied frames, including ones not fixed by the physical cycle.
- Verify fallback still finds nonfiber polynomial frames and distinguishes
  budget exhaustion from an exhaustive proof of infeasibility. Existing tests
  that intentionally exercise enumeration may need fixtures where the new
  sufficient conditions fail, or may test the coefficient helper directly.
- Replay full polynomial identities and both inherited logical getters. Check
  transversal Hadamard and signed logical S on a small new fixture.

## Files, validation, and delivery

Modify `_symcity.py` and its tests, the constructor/result docstrings in
`quantum.py`, relevant construction tests, and the existing example notebook.
Add the nonzero reduction example and explain when coefficient enumeration is
still needed. Keep the original plan as a record of the first implementation.

Run focused reduction tests, then all affected construction/solver/circuit
modules with the repository's 100% modular coverage check. Run formatting,
linting, and type checking on changed files; execute notebook cells and build
the documentation when Pandoc is available. Distinguish unchanged baseline
check failures from any regressions introduced by the new code.

Before writing this plan, eight concrete fixtures were checked with explicit
frames against the committed constructor: both diagonal families for three
groups, and the nonconstant family for two groups. Their rank, completeness,
pairing, augmentation, and dimensions passed. These probes establish useful
fixtures; the new automatic reduction helper and its regression tests remain
to be implemented.

A weaker condition allowing `B` to remain polynomial could reduce the logical
unknowns from `s*t*q*N` to `s*t*N`, but would still require solving a polynomial
orthonormal-frame problem. Defer that extension. Generating the full commuting
check space, proving that its orbit module is free, and certifying distance
remain separate tasks even with the binary reduction.
