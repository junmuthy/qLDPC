"""Shared utilities for Cain Table III matching scripts.

- find_low_weight_z_rep: BP+OSD-style greedy reduction toward target weight
- combine_z_logicals: XOR multiple Z-logicals, with optional stab reduction
- enumerate_z_logical_subsets: yields (k choose t) combos of basis Z-logicals
- gadget_shape: returns (κ, χ, G) tuple for a (merged, SurgeryLayout) pair
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterable, Iterator

import galois
import numpy as np

GF2 = galois.GF(2)


def gadget_shape(layout) -> tuple[int, int, int]:
    """Return (κ, χ, G) where:
      κ = layout.num_ancilla_qubits  (total ancilla qubits)
      χ = number of chi rows = layout.v0_indices.size
      G = number of gauge-fix Z rows = (κ − rank(F))
    """
    F = np.asarray(layout.F).astype(int)
    rank_F = int(np.linalg.matrix_rank(GF2(F)))
    n_kappa = int(layout.num_ancilla_qubits)
    n_chi = int(layout.v0_indices.size)
    n_gauge = n_kappa - rank_F
    return (n_kappa, n_chi, n_gauge)


def combine_z_logicals(zls: np.ndarray, indices: Iterable[int]) -> np.ndarray:
    """XOR of zls[i] for i in indices."""
    out = np.zeros(zls.shape[1], dtype=int)
    for i in indices:
        out = (out + zls[i]) % 2
    return out


def stab_reduce(vec: np.ndarray, HZ: np.ndarray, *, max_steps: int = 50,
                seed: int = 0) -> np.ndarray:
    """Greedy stab reduction: XOR with HZ rows that strictly decrease weight."""
    cur = vec.copy()
    rng = random.Random(seed)
    for _ in range(max_steps):
        improved = False
        sample = rng.sample(range(HZ.shape[0]), min(30, HZ.shape[0]))
        for s_idx in sample:
            cand = (cur + HZ[s_idx]) % 2
            if int(cand.sum()) < int(cur.sum()):
                cur = cand
                improved = True
                break
        if not improved:
            break
    return cur


def find_low_weight_z_rep(
    code,
    *,
    target_weight: int,
    max_trials: int = 5000,
    max_indices: int = 8,
    seed: int = 0,
) -> np.ndarray | None:
    """Search for a Z-logical representative of given weight via XOR + reduce."""
    import qldpc.objects as _o
    HX = np.asarray(code.matrix_x).astype(int)
    HZ = np.asarray(code.matrix_z).astype(int)
    zls = np.asarray(code.get_logical_ops(_o.Pauli.Z)).astype(int)
    rng = random.Random(seed)
    for trial in range(max_trials):
        k = rng.randint(1, min(max_indices, code.dimension))
        idxs = rng.sample(range(code.dimension), k)
        cur = combine_z_logicals(zls, idxs)
        cur = stab_reduce(cur, HZ, seed=seed + trial)
        if int(cur.sum()) == target_weight and ((HX @ cur) % 2).sum() == 0:
            return cur
    return None


def enumerate_z_logical_subsets(
    n_logicals: int, t: int,
) -> Iterator[tuple[int, ...]]:
    """Iterate all C(n_logicals, t) combos of basis-logical indices."""
    yield from itertools.combinations(range(n_logicals), t)
