"""Low-weight logical operators and canonical (symplectic) bases for CSS codes.

Low-weight logical operators are found by the random-window (information-set)
search of Leon [J. S. Leon, "A probabilistic algorithm for computing minimum
weights of large error-correcting codes", IEEE Trans. Inf. Theory 34(5), 1988] --
the same algorithm the GAP package QDistRnd [L. P. Pryadko, V. A. Shabashov,
V. K. Kozin, "QDistRnd: A GAP package for computing the distance of quantum
error-correcting codes", J. Open Source Softw. 7(71), 4120 (2022)] uses to bound
quantum code distances.  QDistRnd's ``DistRandCSS`` returns only the distance
bound, so we re-implement the sampler here to obtain the operators themselves.

``symplectic_logical_basis`` assembles the sampled operators into a canonical
logical basis {(X_i, Z_i)} with <X_i, Z_j> = delta_ij, keeping BOTH operator
types low-weight by *selecting* compatible low-weight operators from the pools
(rather than taking the forced dual of a fixed basis, which inflates the
weights).  It works for any CSS code: a constructive symplectic Gram-Schmidt in
the logical coordinate space falls back to the reference basis whenever the pools
run out of compatible low-weight operators, so it never dead-ends.  For the
bb18 = [[248, 10, 18]] bivariate-bicycle code this reproduces the basis of Cain
et al. [arXiv:2603.28627]: all ten X are weight-18 and Z is nine weight-18 plus
one weight-20 (2k-1 operators at the distance, one at distance+2).  Codes with no
such near-distance basis (e.g. the gross [[72, 12, 6]] code, whose canonical
logicals span weights 6..20) simply get their lowest-weight canonical basis.
"""
from __future__ import annotations

import numpy as np

from qldpc.objects import Pauli, PauliXZ


# --------------------------------------------------------------------------- #
# GF(2) linear algebra (numpy; the random-window hot loop wants tight XORs)
# --------------------------------------------------------------------------- #
def _gf2_nullspace(matrix: np.ndarray) -> np.ndarray:
    """Rows spanning {x : matrix @ x = 0} over GF(2)."""
    reduced = matrix.copy() % 2
    num_rows, num_cols = reduced.shape
    pivot_of_col: dict[int, int] = {}
    row = 0
    for col in range(num_cols):
        sel = next((r for r in range(row, num_rows) if reduced[r, col]), None)
        if sel is None:
            continue
        reduced[[row, sel]] = reduced[[sel, row]]
        for r in range(num_rows):
            if r != row and reduced[r, col]:
                reduced[r] ^= reduced[row]
        pivot_of_col[col] = row
        row += 1
    basis = []
    for free_col in (c for c in range(num_cols) if c not in pivot_of_col):
        vec = np.zeros(num_cols, dtype=np.uint8)
        vec[free_col] = 1
        for col, r in pivot_of_col.items():
            vec[col] = reduced[r, free_col]
        basis.append(vec)
    return np.array(basis, dtype=np.uint8).reshape(-1, num_cols)


def _gf2_rref(matrix: np.ndarray) -> np.ndarray:
    """Reduced row-echelon form over GF(2)."""
    reduced = matrix.copy() % 2
    num_rows, num_cols = reduced.shape
    row = 0
    for col in range(num_cols):
        pivot = next((i for i in range(row, num_rows) if reduced[i, col]), None)
        if pivot is None:
            continue
        reduced[[row, pivot]] = reduced[[pivot, row]]
        for i in range(num_rows):
            if i != row and reduced[i, col]:
                reduced[i] ^= reduced[row]
        row += 1
        if row == num_rows:
            break
    return reduced


def _gf2_rref_pivots(matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Reduced row-echelon rows (nonzero only) and their pivot columns over GF(2)."""
    reduced = matrix.copy() % 2
    num_rows, num_cols = reduced.shape
    pivots: list[int] = []
    row = 0
    for col in range(num_cols):
        pivot = next((i for i in range(row, num_rows) if reduced[i, col]), None)
        if pivot is None:
            continue
        reduced[[row, pivot]] = reduced[[pivot, row]]
        for i in range(num_rows):
            if i != row and reduced[i, col]:
                reduced[i] ^= reduced[row]
        pivots.append(col)
        row += 1
        if row == num_rows:
            break
    return reduced[:row], pivots


def _gf2_solve(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    """One solution ``x`` of ``matrix @ x == target`` (mod 2); assumes consistency."""
    num_cols = matrix.shape[1]
    aug = np.concatenate([matrix.copy() % 2, (target % 2).reshape(-1, 1)], axis=1)
    num_rows = aug.shape[0]
    pivot_cols: list[int] = []
    row = 0
    for col in range(num_cols):
        sel = next((r for r in range(row, num_rows) if aug[r, col]), None)
        if sel is None:
            continue
        aug[[row, sel]] = aug[[sel, row]]
        for r in range(num_rows):
            if r != row and aug[r, col]:
                aug[r] ^= aug[row]
        pivot_cols.append(col)
        row += 1
        if row == num_rows:
            break
    solution = np.zeros(num_cols, dtype=np.uint8)
    for i, col in enumerate(pivot_cols):
        solution[col] = aug[i, -1]
    return solution


def _css_matrices(code) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(HX, HZ, LX, LZ) as uint8 numpy arrays for a CSS ``code``."""
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    LX = np.asarray(code.get_logical_ops(Pauli.X)).astype(np.uint8)
    LZ = np.asarray(code.get_logical_ops(Pauli.Z)).astype(np.uint8)
    return HX, HZ, LX, LZ


# --------------------------------------------------------------------------- #
# Stage 1: random-window low-weight logical sampler (Leon / QDistRnd engine)
# --------------------------------------------------------------------------- #
def low_weight_logicals(
    code,
    basis: PauliXZ,
    *,
    weight_cap: int,
    trials: int,
    rng: np.random.Generator,
) -> dict[tuple[int, ...], np.ndarray]:
    """Sample distinct low-weight ``basis``-type logical operators of ``code``.

    Leon/QDistRnd random-window search: repeatedly take a random column
    permutation of a generator matrix of the operator space, row-reduce, and read
    off the low-weight rows.  A row is a *logical* (not a stabiliser) iff it has a
    non-trivial symplectic product with some opposite-type logical.

    Args:
        code: a CSS ``QuditCode``.
        basis: ``Pauli.X`` or ``Pauli.Z`` -- the operator type to sample.
        weight_cap: keep operators of weight <= this.
        trials: number of random windows.
        rng: numpy ``Generator``.

    Returns:
        ``{support-tuple: 0/1 vector}`` for every distinct qualifying logical.
    """
    HX, HZ, LX, LZ = _css_matrices(code)
    # basis-type operators live in the kernel of the opposite-type checks;
    # they are logical iff they pair non-trivially with the opposite-type logicals
    if basis is Pauli.X:
        generators, logical_check = _gf2_nullspace(HZ), LZ
    else:
        generators, logical_check = _gf2_nullspace(HX), LX
    n = generators.shape[1]
    pool: dict[tuple[int, ...], np.ndarray] = {}
    for _ in range(trials):
        perm = rng.permutation(n)
        reduced = _gf2_rref(generators[:, perm])
        unpermuted = np.zeros_like(reduced)
        unpermuted[:, perm] = reduced
        for row in unpermuted:
            weight = int(row.sum())
            if 0 < weight <= weight_cap and (logical_check @ row % 2).any():
                pool[tuple(np.nonzero(row)[0])] = row.copy()
    return pool


def logical_distance(code, *, trials: int = 3000, rng: np.random.Generator | None = None) -> int:
    """Random-window (QDistRnd-style) estimate of the code distance."""
    rng = np.random.default_rng() if rng is None else rng
    sampled = {
        **low_weight_logicals(code, Pauli.X, weight_cap=10 * 1024, trials=trials, rng=rng),
        **low_weight_logicals(code, Pauli.Z, weight_cap=10 * 1024, trials=trials, rng=rng),
    }
    if not sampled:
        raise RuntimeError("no logical operators sampled; raise trials")
    return min(int(v.sum()) for v in sampled.values())


# --------------------------------------------------------------------------- #
# Stage 2: symplectic Gram-Schmidt over low-weight pairs
# --------------------------------------------------------------------------- #
def symplectic_logical_basis(
    code,
    *,
    distance: int | None = None,
    weight_cap: int = 46,
    pool_size: int = 6000,
    trials: int = 3000,
    restarts: int = 40,
    pool_rounds: int = 3,
    optimize: PauliXZ = Pauli.X,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X_ops, Z_ops)``: a canonical basis with both types low-weight.

    Builds a canonical symplectic basis {(X_i, Z_i)} with ``X_i . Z_j = delta_ij``
    by a *constructive* symplectic Gram-Schmidt carried out in the k-dimensional
    logical coordinate space.  With reference bases satisfying ``LX . LZ^T = I``,
    every logical X operator ``p`` maps to the coordinate ``p . LZ^T`` and every
    logical Z operator ``s`` to ``s . LX^T``, so the symplectic product reduces to
    an ordinary GF(2) dot product of coordinates.  At step ``t`` the ``optimize``
    ("primary") type is drawn lowest-weight from a sampled pool -- restricted to
    operators that commute with the chosen secondaries and extend the primary span
    -- and its partner of the other ("secondary") type is drawn lowest-weight from
    its pool subject to ``coord . chosen_primaries == e_t`` (anticommute with the
    new primary, commute with the earlier ones).  Whenever a pool cannot supply a
    compatible operator the partner is *constructed* from the reference basis
    (``coord @ ref``; higher weight but always valid), so the routine completes for
    ANY CSS code instead of dead-ending when no near-distance basis exists.  The
    weight excess forced by the code's symplectic geometry therefore lands on the
    secondary type's later qubits; ``optimize`` selects which type stays small.
    Randomized restarts keep the lowest sorted-weight-profile basis found.

    For the bb18 = [[248, 10, 18]] bivariate-bicycle code this reproduces the
    near-distance basis of Cain et al. [arXiv:2603.28627] (primary all at the
    distance, one secondary at distance+2).  For codes admitting no such basis --
    e.g. the gross [[72, 12, 6]] code, whose canonical logicals unavoidably span
    weights 6..20 -- it returns the lowest-weight canonical basis it can build.

    Args:
        code: a CSS ``QuditCode``.
        distance: code distance; estimated by random-window sampling only when
            ``verbose`` and left ``None`` (reported, not used by the construction).
        weight_cap: max operator weight retained in either pool.
        pool_size: number of lowest-weight candidates kept per type.
        trials: random-window trials per operator type per pool round.
        restarts: randomized Gram-Schmidt restarts per pool round.
        pool_rounds: number of independent pool re-samples (best profile kept).
        optimize: which type is drawn weight-minimal (the "primary"); the other
            ("secondary") absorbs the symplectic weight excess. Default
            ``Pauli.X``; pass ``Pauli.Z`` to weight-minimize Z instead -- useful
            when the operator you will *measure* is Z, so its support stays small.
        rng: numpy ``Generator`` (fix for a reproducible basis).
        verbose: print pool sizes and the resulting weight profile per round.

    Returns:
        ``(X_ops, Z_ops)``: two ``(k, n)`` uint8 arrays, row ``i`` = ``(X_i, Z_i)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    HX, HZ, LX, LZ = _css_matrices(code)
    n, k = HX.shape[1], code.dimension
    if k == 0:
        return np.zeros((0, n), np.uint8), np.zeros((0, n), np.uint8)
    if distance is None and verbose:
        distance = logical_distance(code, trials=trials, rng=rng)

    # reference bases with LX . LZ^T = I: coord(prim p) = p . ref_sec^T,
    # coord(sec s) = s . ref_prim^T, and <p, s> = coord(p) . coord(s).
    prim_type = optimize
    sec_type = Pauli.Z if optimize is Pauli.X else Pauli.X
    ref_prim, ref_sec = (LX, LZ) if optimize is Pauli.X else (LZ, LX)
    unit = np.eye(k, dtype=np.uint8)

    def one_attempt(prim_ops, prim_co, sec_ops, sec_co):
        # arrays arrive weight-sorted, so the first admissible index is minimal-weight.
        chosen_p, chosen_s = [], []
        CP = np.zeros((0, k), np.uint8)  # primary coordinates chosen so far
        CS = np.zeros((0, k), np.uint8)  # secondary coordinates chosen so far
        for t in range(k):
            # primary: commute with chosen secondaries and extend the primary span
            commute = np.ones(len(prim_co), bool) if len(CS) == 0 else (prim_co @ CS.T % 2).sum(1) == 0
            if len(CP) == 0:
                indep = prim_co.any(1)
            else:
                rref, pivots = _gf2_rref_pivots(CP)
                residual = prim_co.copy()
                for prow, pcol in zip(rref, pivots):
                    residual[residual[:, pcol] == 1] ^= prow
                indep = residual.any(1)
            cand = np.nonzero(commute & indep)[0]
            if len(cand):
                cp, p_op = prim_co[cand[0]], prim_ops[cand[0]]
            else:  # construct: a coordinate in ker(CS) is disjoint from span(CP)
                nulls = _gf2_nullspace(CS) if len(CS) else unit
                cp = nulls[0]
                p_op = cp @ ref_prim % 2
            CP = np.vstack([CP, cp])
            # secondary: coord . CP == e_t (anticommute new primary, commute earlier)
            want = np.zeros(len(CP), np.uint8)
            want[-1] = 1
            hits = np.nonzero((sec_co @ CP.T % 2 == want).all(1))[0] if len(sec_co) else np.empty(0, int)
            if len(hits):
                cs, s_op = sec_co[hits[0]], sec_ops[hits[0]]
            else:  # construct: solve CP . cs == e_t (always consistent, CP full rank)
                cs = _gf2_solve(CP, want)
                s_op = cs @ ref_sec % 2
            CS = np.vstack([CS, cs])
            chosen_p.append(p_op)
            chosen_s.append(s_op)
        return np.array(chosen_p, np.uint8).reshape(k, n), np.array(chosen_s, np.uint8).reshape(k, n)

    best = None
    for pool_round in range(pool_rounds):
        pool_p = low_weight_logicals(code, prim_type, weight_cap=weight_cap, trials=trials, rng=rng)
        pool_s = low_weight_logicals(code, sec_type, weight_cap=weight_cap, trials=trials, rng=rng)
        prim_ops = np.array(sorted(pool_p.values(), key=lambda v: int(v.sum())), np.uint8).reshape(-1, n)[:pool_size]
        sec_ops = np.array(sorted(pool_s.values(), key=lambda v: int(v.sum())), np.uint8).reshape(-1, n)[:pool_size]
        prim_co = prim_ops @ ref_sec.T % 2
        sec_co = sec_ops @ ref_prim.T % 2
        prim_wt, sec_wt = prim_ops.sum(1), sec_ops.sum(1)
        if verbose:
            print(f"  round {pool_round + 1}: |{prim_type.name}<={weight_cap}|={len(prim_ops)}  |{sec_type.name}<={weight_cap}|={len(sec_ops)}")
        for _ in range(restarts):
            # weight order with a random tie-break so restarts explore distinct bases
            po = np.lexsort((rng.permutation(len(prim_ops)), prim_wt))
            so = np.lexsort((rng.permutation(len(sec_ops)), sec_wt))
            p_ops, s_ops = one_attempt(prim_ops[po], prim_co[po], sec_ops[so], sec_co[so])
            if not np.array_equal(p_ops @ s_ops.T % 2, unit):
                continue  # safety net; the construction guarantees the identity
            pw = sorted(int(v.sum()) for v in p_ops)
            sw = sorted(int(v.sum()) for v in s_ops)
            score = (max(pw), max(sw), sum(pw) + sum(sw))
            if best is None or score < best[0]:
                best = (score, p_ops, s_ops)
    assert best is not None  # reference fallback always completes a valid basis
    if verbose:
        pw = sorted(int(v.sum()) for v in best[1])
        sw = sorted(int(v.sum()) for v in best[2])
        print(f"  best: {prim_type.name} weights {pw}\n        {sec_type.name} weights {sw}")
    return (best[1], best[2]) if optimize is Pauli.X else (best[2], best[1])
