"""EXACT match for Cain Extended Data Table III lp_20^{3,5} Processor.

Target: (Qubits, X-checks, Z-checks) = (813, 460, 357).
Pipeline:
  1. Build lp_20^{3,5} [[1122, 148]] from Cain App. A Eq A3.
  2. Pick 69 of 148 Z-logical reps with |V_0_union| = 460.
  3. multi-target Webster gadget.
  4. Spectral Cheeger boost seed sweep until exact (kappa, chi, G).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import _cain_helpers as h

from qldpc import codes
from qldpc.abstract import CyclicGroup, GroupRing, RingArray
from qldpc.codes.common import CSSCode
from qldpc.codes.surgery import (
    boost_gadget_cheeger,
    build_multi_target_surgery_code,
)
from qldpc.objects import Pauli


TARGET = (813, 460, 357)
MAX_BOOST_SEEDS = 200


def build_lp20_3_5():
    """Cain App. A Eq A3: lp_20^{3,5}.

    l=33, with 3x5 RingArray over CyclicGroup(33). Reproduces [[1122, 148]].
    """
    l = 33
    group = CyclicGroup(l)
    xg = group.generators[0]
    ring = GroupRing(group)
    A = RingArray.build(
        [
            [1, 1, 1, 1, 1],
            [1, xg**14, xg**19, xg**11, xg**26],
            [1, xg**13, xg**2, xg**15, xg**21],
        ],
        ring,
    )
    return codes.LPCode(A)


def find_69_logical_reps(code, target_v0_size: int = 460,
                         max_attempts: int = 200) -> list[np.ndarray] | None:
    """Heuristic: pick 69 of 148 Z-logicals via random subset selection."""
    HX = np.asarray(code.matrix_x).astype(int)
    HZ = np.asarray(code.matrix_z).astype(int)
    zls = np.asarray(code.get_logical_ops(Pauli.Z)).astype(int)
    import random as _r

    best_dist: int | None = None
    best_ops: list[np.ndarray] = []
    best_size = 0

    for attempt in range(max_attempts):
        rng = _r.Random(attempt)
        chosen = rng.sample(range(code.dimension), 69)
        ops = []
        for i in chosen:
            v = h.stab_reduce(zls[i].copy(), HZ, max_steps=40,
                              seed=attempt * 1000 + i)
            assert ((HX @ v) % 2).sum() == 0
            ops.append(v)
        v0_union = np.zeros(code.num_qubits, dtype=int)
        for op in ops:
            v0_union = v0_union | op
        size = int(v0_union.sum())
        dist = abs(size - target_v0_size)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_ops = ops
            best_size = size
        if size == target_v0_size:
            return ops
        if attempt < 5 or attempt % 25 == 0:
            print(f"    attempt {attempt}: |V_0|={size} (best so far={best_size})")
    print(f"  best |V_0_union| over {max_attempts} attempts: {best_size}"
          f" (target {target_v0_size})")
    return None


def main() -> None:
    print("=" * 72)
    print("EXACT match for Cain Extended Data Table III lp_20^{3,5} Processor")
    print(f"Target (Qubits, X-checks, Z-checks): {TARGET}")
    print("=" * 72)

    lp = build_lp20_3_5()
    print(f"\nlp_20^{{3,5}}: [[{lp.num_qubits}, {lp.dimension}]]")
    print(f"  Expected [[1122, 148]] per Cain Eq A3")
    if (lp.num_qubits, lp.dimension) != (1122, 148):
        print(f"  params mismatch - polynomials may be wrong; aborting")
        return

    print("\nStep 1: find 69 Z-logical reps with |V_0_union| = 460")
    ops = find_69_logical_reps(lp, target_v0_size=460, max_attempts=200)
    if ops is None:
        print("  no 69-subset gives |V_0| = 460; need richer search")
        return
    print(f"  found {len(ops)} reps")

    print("\nStep 2: multi-target Webster gadget")
    lp_dual = CSSCode(lp.matrix_z, lp.matrix_x, is_subsystem_code=False)
    merged, layout = build_multi_target_surgery_code(lp_dual, ops, validate=False)
    base = layout.base_layout
    bare_shape = h.gadget_shape(base)
    print(f"  Bare gadget: (kappa, chi, G) = {bare_shape}")

    print(f"\nStep 3: Cheeger boost seed sweep (0..{MAX_BOOST_SEEDS - 1})")
    for seed in range(MAX_BOOST_SEEDS):
        boosted, b_layout, _r = boost_gadget_cheeger(
            merged, base, target_h=1.0, max_extra_qubits=500, seed=seed,
        )
        shape = h.gadget_shape(b_layout)
        if shape == TARGET:
            print(f"  EXACT MATCH at seed={seed}: {shape}")
            print("\n" + "=" * 72)
            print(f"EXACT MATCH: {shape} = Cain target {TARGET}")
            print("=" * 72)
            return
    print(f"  no seed in 0..{MAX_BOOST_SEEDS - 1} produced {TARGET};"
          f" expand search OR reps wrong")


if __name__ == "__main__":
    main()
