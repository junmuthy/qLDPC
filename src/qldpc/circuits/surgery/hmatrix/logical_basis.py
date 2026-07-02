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
types low-weight by *selecting* compatible low-weight anticommuting pairs (rather
than taking the forced dual of a fixed basis, which inflates the weights).  For
the bb18 = [[248, 10, 18]] bivariate-bicycle code this reproduces the basis of
Cain et al. [arXiv:2603.28627]: all ten X are weight-18 and Z is nine weight-18
plus one weight-20 -- the single weight excess is an irreducible symplectic
obstruction (2k-1 operators sit at the distance, exactly one at distance+2).
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
    x_weight_cap: int = 46,
    x_pool_size: int = 6000,
    trials: int = 3000,
    restarts: int = 250,
    pool_rounds: int = 4,
    optimize: PauliXZ = Pauli.X,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X_ops, Z_ops)``: a canonical basis with both types low-weight.

    Greedily builds a symplectic basis {(X_i, Z_i)} with ``X_i . Z_j = delta_ij``:
    for each qubit, pick the lowest-weight X and a Z (forced to the distance for
    the first k-1 qubits, allowed distance+2 for the last) that anticommute and
    commute with every pair chosen so far.  The commutation filters make the
    pairing matrix triangular with unit diagonal, i.e. exactly ``X . Z^T = I``.
    The ideal profile is 2k-1 operators at the distance and exactly one at
    distance+2 (the irreducible symplectic excess).  Each pool draw is retried up
    to ``restarts`` times; if none hits the ideal, the pools are re-sampled (up to
    ``pool_rounds`` times) since a single unlucky draw can admit no ideal basis.
    Falls back to the lowest max-weight basis seen if the ideal is never hit.

    Args:
        code: a CSS ``QuditCode``.
        distance: code distance; estimated by random-window sampling if ``None``.
        x_weight_cap: max weight retained in the X pool.
        x_pool_size: number of lowest-weight X candidates kept.
        trials: random-window trials per operator type per pool round.
        restarts: symplectic-greedy restarts per pool round.
        pool_rounds: max pool re-samples if the ideal profile is not hit.
        optimize: which operator type is drawn weight-minimal from the pool (the
            "primary"). The *other* type is forced to the distance and carries the
            lone ``distance+2`` excess. Default ``Pauli.X`` puts the excess on Z
            (so one Z sits at distance+2); pass ``Pauli.Z`` to weight-minimize Z
            instead (all Z at the distance, the excess falling on one X) — useful
            when the operator you will *measure* is Z, so its support stays small.
        rng: numpy ``Generator`` (fix for a reproducible basis).
        verbose: print pool sizes / restart on which the target was hit.

    Returns:
        ``(X_ops, Z_ops)``: two ``(k, n)`` uint8 arrays, row ``i`` = ``(X_i, Z_i)``.
    """
    rng = np.random.default_rng() if rng is None else rng
    HX, HZ, LX, LZ = _css_matrices(code)
    n, k = HX.shape[1], code.dimension
    if distance is None:
        distance = logical_distance(code, trials=trials, rng=rng)

    # ``prim`` is drawn weight-minimal from the pool; ``sec`` is forced to the
    # distance and carries the lone distance+2 excess. optimize selects which.
    prim_type, sec_type = (Pauli.X, Pauli.Z) if optimize is Pauli.X else (Pauli.Z, Pauli.X)

    def one_attempt(prim_all, sec_d, sec_d2):
        chosen_p = np.zeros((0, n), dtype=np.uint8)
        chosen_s = np.zeros((0, n), dtype=np.uint8)
        for t in range(k):
            s_candidates = sec_d if t < k - 1 else np.vstack([sec_d, sec_d2])
            s_candidates = s_candidates[rng.permutation(len(s_candidates))]
            cs = s_candidates if len(chosen_p) == 0 else s_candidates[(s_candidates @ chosen_p.T % 2).sum(1) == 0]
            cp = prim_all if len(chosen_s) == 0 else prim_all[(prim_all @ chosen_s.T % 2).sum(1) == 0]
            if len(cp) == 0 or len(cs) == 0:
                return None
            pair = None
            for p in cp:  # prim_all is weight-sorted -> first hit has minimal wt(p)
                anticommuting = np.nonzero(cs @ p % 2)[0]
                if len(anticommuting):
                    pair = (p, cs[anticommuting[0]])
                    break
            if pair is None:
                return None
            chosen_p = np.vstack([chosen_p, pair[0]])
            chosen_s = np.vstack([chosen_s, pair[1]])
        return chosen_p, chosen_s

    best = None
    for pool_round in range(pool_rounds):
        pool_p = low_weight_logicals(code, prim_type, weight_cap=x_weight_cap, trials=trials, rng=rng)
        pool_s = low_weight_logicals(code, sec_type, weight_cap=distance + 2, trials=trials, rng=rng)
        prim_all = np.array(sorted(pool_p.values(), key=lambda v: int(v.sum())), dtype=np.uint8).reshape(-1, n)
        prim_all = prim_all[:x_pool_size]
        sec_d = np.array([v for v in pool_s.values() if int(v.sum()) == distance], dtype=np.uint8).reshape(-1, n)
        sec_d2 = np.array([v for v in pool_s.values() if int(v.sum()) == distance + 2], dtype=np.uint8).reshape(-1, n)
        if verbose:
            print(
                f"  round {pool_round + 1}: |{prim_type.name}<={x_weight_cap}|={len(prim_all)}  "
                f"|{sec_type.name}_wt{distance}|={len(sec_d)}  |{sec_type.name}_wt{distance + 2}|={len(sec_d2)}"
            )
        if len(prim_all) == 0 or len(sec_d) < k - 1 or len(sec_d2) == 0:
            continue
        for restart in range(restarts):
            result = one_attempt(prim_all, sec_d, sec_d2)
            if result is None:
                continue
            p_ops, s_ops = result
            if not np.array_equal((p_ops @ s_ops.T) % 2, np.eye(k, dtype=int) % 2):
                continue
            pw = sorted(int(v.sum()) for v in p_ops)
            sw = sorted(int(v.sum()) for v in s_ops)
            # ideal: the primary (weight-optimized / measured) type sits entirely at
            # the distance, and the lone distance+2 excess is pushed onto the
            # secondary type. Score prioritizes a low primary max weight.
            target = max(pw) == distance and sw.count(distance) == k - 1 and sw.count(distance + 2) == 1
            score = (max(pw), max(sw), sum(pw) + sum(sw))
            if best is None or score < best[0]:
                best = (score, p_ops, s_ops)
            if target:
                if verbose:
                    print(f"  hit target profile (round {pool_round + 1}, restart {restart + 1})")
                return (p_ops, s_ops) if optimize is Pauli.X else (s_ops, p_ops)
    if best is None:
        raise RuntimeError("no valid canonical basis found; raise restarts / trials / pool_rounds")
    if verbose:
        print("  ideal profile not hit; returning lowest max-weight basis found")
    return (best[1], best[2]) if optimize is Pauli.X else (best[2], best[1])
