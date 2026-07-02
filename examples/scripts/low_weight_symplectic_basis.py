#!/usr/bin/env python
"""Find a low-weight canonical (symplectic) logical basis of a CSS code.

Two stages:

  1. QDistRnd-style random-window search (Leon's algorithm): sample low-weight
     logical operators of a given type by repeatedly taking a random column
     permutation of a generator matrix, row-reducing, and reading off the
     low-weight rows.  This is the engine inside GAP's QDistRnd `DistRandCSS`;
     re-implemented here so we get the actual operators (not just the distance).

  2. Symplectic Gram-Schmidt: assemble a canonical basis {(X_i, Z_i)} with
     <X_i, Z_j> = delta_ij by greedily picking low-weight anticommuting pairs
     that commute with every pair chosen so far.  The trick that keeps BOTH
     operator types short is to *select* already-compatible low-weight pairs
     rather than take the forced dual of a fixed basis (which blows the weights
     up).

For bb18 = [[248, 10, 18]] this reproduces the Cain et al. arXiv:2603.28627
basis: all ten X are weight-18 and Z is nine weight-18 + one weight-20 -- the
single weight-20 is the irreducible symplectic cost (you may place it on the X
or the Z side, but exactly one operator must exceed the distance).

Usage
-----
    python low_weight_symplectic_basis.py                 # bb18 (default)
    python low_weight_symplectic_basis.py --code gross    # [[144, 12, 12]]
    python low_weight_symplectic_basis.py --trials 4000 --restarts 120 --seed 7
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import sympy

from qldpc import codes
from qldpc.objects import Pauli


# --------------------------------------------------------------------------- #
# GF(2) linear algebra
# --------------------------------------------------------------------------- #
def gf2_nullspace(A: np.ndarray) -> np.ndarray:
    """Rows spanning {x : A x = 0} over GF(2)."""
    M = A.copy() % 2
    m, ncol = M.shape
    piv: dict[int, int] = {}
    row = 0
    for col in range(ncol):
        sel = next((r for r in range(row, m) if M[r, col]), None)
        if sel is None:
            continue
        M[[row, sel]] = M[[sel, row]]
        for r in range(m):
            if r != row and M[r, col]:
                M[r] ^= M[row]
        piv[col] = row
        row += 1
    basis = []
    for fc in [c for c in range(ncol) if c not in piv]:
        v = np.zeros(ncol, dtype=np.uint8)
        v[fc] = 1
        for col, r in piv.items():
            v[col] = M[r, fc]
        basis.append(v)
    return np.array(basis, dtype=np.uint8).reshape(-1, ncol)


def gf2_rref(M: np.ndarray) -> np.ndarray:
    """Reduced row-echelon form over GF(2)."""
    M = M.copy() % 2
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i, c]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
        if r == rows:
            break
    return M


# --------------------------------------------------------------------------- #
# Stage 1: QDistRnd random-window low-weight logical sampler
# --------------------------------------------------------------------------- #
def sample_low_weight_logicals(
    generators: np.ndarray,
    logical_check: np.ndarray,
    weight_cap: int,
    trials: int,
    rng: np.random.Generator,
) -> dict[tuple[int, ...], np.ndarray]:
    """QDistRnd/Leon random-window sampling of low-weight logical operators.

    ``generators`` spans the operator space (e.g. nullspace(H_X) for Z-type
    operators).  A row of the row-reduced random-column-permutation of that
    space is a codeword; it is a *logical* (not a stabiliser) iff it has a
    non-trivial symplectic product with some row of ``logical_check`` (the
    opposite-type logicals).  Returns {support-tuple: vector} for every distinct
    logical of weight <= ``weight_cap`` seen.
    """
    n = generators.shape[1]
    pool: dict[tuple[int, ...], np.ndarray] = {}
    for _ in range(trials):
        perm = rng.permutation(n)
        reduced = gf2_rref(generators[:, perm])
        unperm = np.zeros_like(reduced)
        unperm[:, perm] = reduced
        for row in unperm:
            w = int(row.sum())
            if 0 < w <= weight_cap and (logical_check @ row % 2).any():
                pool[tuple(np.nonzero(row)[0])] = row.copy()
    return pool


# --------------------------------------------------------------------------- #
# Stage 2: symplectic Gram-Schmidt over low-weight pairs
# --------------------------------------------------------------------------- #
def build_symplectic_basis(
    HX: np.ndarray,
    HZ: np.ndarray,
    LX: np.ndarray,
    LZ: np.ndarray,
    k: int,
    *,
    distance: int,
    x_weight_cap: int,
    x_pool_size: int,
    trials: int,
    restarts: int,
    rng: np.random.Generator,
    verbose: bool = True,
):
    """Return (Xs, Zs): a canonical basis with both operator types low-weight.

    Greedily, for each qubit, picks the lowest-weight X that commutes with the Z
    already chosen and the lowest-weight Z (forced to ``distance`` for the first
    k-1 qubits, allowed ``distance``+2 for the last) that anticommutes with it and
    commutes with the X already chosen.  The commutation filters make the pairing
    matrix triangular with unit diagonal, i.e. exactly ``X.Z^T = I``.  Restarts
    with reshuffled Z order until the target profile -- all X at ``distance`` and Z
    = (k-1) x distance + 1 x (distance+2) -- is reached; otherwise returns the
    lowest max-weight basis found.
    """
    n = HX.shape[1]
    t0 = time.time()
    poolX = sample_low_weight_logicals(gf2_nullspace(HZ), LZ, x_weight_cap, trials, rng)
    poolZ = sample_low_weight_logicals(gf2_nullspace(HX), LX, distance + 2, trials, rng)
    # full X pool, weight-sorted (prefers weight-``distance`` but keeps fallbacks)
    Xall = np.array(sorted(poolX.values(), key=lambda v: int(v.sum())), dtype=np.uint8).reshape(-1, n)
    Xall = Xall[:x_pool_size]
    Zd = np.array([v for v in poolZ.values() if int(v.sum()) == distance], dtype=np.uint8).reshape(-1, n)
    Zd2 = np.array([v for v in poolZ.values() if int(v.sum()) == distance + 2], dtype=np.uint8).reshape(-1, n)
    if verbose:
        print(
            f"  pools: |X<={x_weight_cap}|={len(Xall)} (min wt {int(Xall.sum(1).min()) if len(Xall) else '-'})  "
            f"|Z_wt{distance}|={len(Zd)}  |Z_wt{distance + 2}|={len(Zd2)}  ({time.time() - t0:.0f}s)"
        )
    if len(Xall) == 0 or len(Zd) < k - 1 or len(Zd2) == 0:
        raise RuntimeError("insufficient low-weight logicals sampled; raise --trials / --x-weight-cap")

    def one_attempt():
        Xs = np.zeros((0, n), dtype=np.uint8)
        Zs = np.zeros((0, n), dtype=np.uint8)
        for t in range(k):
            zc = Zd if t < k - 1 else np.vstack([Zd, Zd2])
            zc = zc[rng.permutation(len(zc))]
            # candidates that commute with everything chosen so far (vectorised)
            cz = zc if len(Xs) == 0 else zc[(zc @ Xs.T % 2).sum(1) == 0]
            cx = Xall if len(Zs) == 0 else Xall[(Xall @ Zs.T % 2).sum(1) == 0]
            if len(cx) == 0 or len(cz) == 0:
                return None
            pair = None
            for x in cx:  # Xall is weight-sorted -> first hit has minimal wt(x)
                hit = np.nonzero(cz @ x % 2)[0]  # z that anticommutes with x
                if len(hit):
                    pair = (x, cz[hit[0]])
                    break
            if pair is None:
                return None
            Xs = np.vstack([Xs, pair[0]])
            Zs = np.vstack([Zs, pair[1]])
        return Xs, Zs

    best = None
    for r in range(restarts):
        res = one_attempt()
        if res is None:
            continue
        Xs, Zs = res
        if not np.array_equal((Xs @ Zs.T) % 2, np.eye(k, dtype=int) % 2):
            continue
        allw = sorted(int(v.sum()) for v in np.vstack([Xs, Zs]))
        # ideal: all 2k operators at the distance except one at distance+2
        # (the single irreducible weight excess -- on the X or the Z side)
        target = allw.count(distance) == 2 * k - 1 and allw.count(distance + 2) == 1
        score = (max(allw), sum(allw))
        if best is None or score < best[0]:
            best = (score, Xs, Zs)
        if target:
            if verbose:
                print(f"  hit target profile on restart {r + 1}")
            return Xs, Zs
    if best is None:
        raise RuntimeError("no valid canonical basis found; raise --restarts / --trials")
    if verbose:
        print("  target profile not hit; returning lowest max-weight basis found")
    return best[1], best[2]


# --------------------------------------------------------------------------- #
# Verification + reporting
# --------------------------------------------------------------------------- #
def verify(Xs, Zs, HX, HZ, LX, LZ) -> None:
    k = len(Xs)
    assert np.array_equal((Xs @ Zs.T) % 2, np.eye(k, dtype=int) % 2), "X.Z^T != I"
    assert not (HZ @ Xs.T % 2).any(), "some X is not a valid X-operator (H_Z X != 0)"
    assert not (HX @ Zs.T % 2).any(), "some Z is not a valid Z-operator (H_X Z != 0)"
    assert all((LZ @ Xs[i] % 2).any() for i in range(k)), "some X is a stabiliser"
    assert all((LX @ Zs[i] % 2).any() for i in range(k)), "some Z is a stabiliser"


def report(Xs, Zs) -> None:
    print("VERIFIED: X.Z^T = I, valid operators, genuine logicals")
    print("wt(X):", [int(v.sum()) for v in Xs])
    print("wt(Z):", [int(v.sum()) for v in Zs])


# --------------------------------------------------------------------------- #
# Code registry + entry point
# --------------------------------------------------------------------------- #
def get_code(name: str):
    xs, ys = sympy.symbols("x y")
    if name == "bb18":  # Cain et al. arXiv:2603.28627 App. A Eq. A11, [[248,10,18]]
        return codes.BBCode((31, 4), 1 + xs**6 * ys + xs**27, ys**2 + xs**15 * ys**3 + xs**24)
    if name == "gross":  # [[144,12,12]]
        return codes.BBCode({xs: 12, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)
    raise ValueError(f"unknown code {name!r}; add it to get_code()")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--code", default="bb18", help="named code (bb18, gross)")
    ap.add_argument("--trials", type=int, default=3000, help="random-window trials per side")
    ap.add_argument("--restarts", type=int, default=100, help="symplectic-greedy restarts")
    ap.add_argument("--x-weight-cap", type=int, default=46, help="max weight kept in the X pool")
    ap.add_argument("--x-pool-size", type=int, default=6000, help="lowest-weight X candidates kept")
    ap.add_argument("--seed", type=int, default=1, help="rng seed (result is reproducible)")
    ap.add_argument("--out", default=None, help="optional JSON path for the basis supports")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    code = get_code(args.code)
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    LX = np.asarray(code.get_logical_ops(Pauli.X)).astype(np.uint8)
    LZ = np.asarray(code.get_logical_ops(Pauli.Z)).astype(np.uint8)
    k = code.dimension

    # distance = min weight over sampled logicals (QDistRnd estimate, both types)
    print(f"code {args.code}: [[{code.num_qudits}, {k}]]")
    probe = {
        **sample_low_weight_logicals(gf2_nullspace(HX), LX, 40, args.trials, rng),
        **sample_low_weight_logicals(gf2_nullspace(HZ), LZ, 40, args.trials, rng),
    }
    distance = min(int(v.sum()) for v in probe.values())
    print(f"min logical weight (distance estimate) = {distance}")

    Xs, Zs = build_symplectic_basis(
        HX, HZ, LX, LZ, k,
        distance=distance,
        x_weight_cap=args.x_weight_cap,
        x_pool_size=args.x_pool_size,
        trials=args.trials,
        restarts=args.restarts,
        rng=rng,
    )
    verify(Xs, Zs, HX, HZ, LX, LZ)
    report(Xs, Zs)

    if args.out:
        supports = {
            "X": [np.nonzero(v)[0].tolist() for v in Xs],
            "Z": [np.nonzero(v)[0].tolist() for v in Zs],
        }
        json.dump(supports, open(args.out, "w"))
        print(f"basis supports written to {args.out}")


if __name__ == "__main__":
    main()
