"""EXACT match for Cain Extended Data Table III lp_20^{3,5} Processor.

Target: (Qubits, X-checks, Z-checks) = (813, 460, 357) with |P̄|=69.

Cain interpretation (per Cain §"Concrete construction"):
  |P̄|=N denotes a SINGLE Pauli operator P̄ with logical weight N
  (acts non-trivially on N of the 148 logical qubits). Not "N PPMs in parallel".

  Pipeline (low-rate surgery on a high-weight P̄):
  1. Build lp_20^{3,5} [[1122, 148]] from Cain App. A Eq A3.
  2. Find a Pauli P̄ with logical weight 69 AND physical weight 460.
     Strategy: XOR random subsets of 69 basis Z-logicals + greedy stab reduction.
     (Cain samples 10^5 random multi-qubit X̄ operators per paper text.)
  3. build_layered_surgery_code(lp_dual, P̄)  — single PPM.
  4. Cheeger boost seed sweep until (κ, χ, G) = (813, 460, 357).
"""

from __future__ import annotations

import random as _random
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
    build_layered_surgery_code,
)
from qldpc.objects import Pauli


TARGET = (813, 460, 357)
TARGET_LOGICAL_WEIGHT = 69
TARGET_PHYSICAL_WEIGHT = 460
MAX_RANDOM_SAMPLES = 100_000
MAX_BOOST_SEEDS = 200


def build_lp20_3_5():
    """Cain App. A Eq A3: lp_20^{3,5} → [[1122, 148]]."""
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


def find_P_via_subspace_search(
    code, target_logical_wt: int, target_phys_wt: int,
    max_samples: int = MAX_RANDOM_SAMPLES, seed: int = 0,
) -> tuple[np.ndarray | None, np.ndarray, int]:
    """Search for a Pauli P̄ with logical weight target_logical_wt and
    physical weight target_phys_wt.

    Returns (matching_op_or_None, best_op_seen, best_phys_weight).

    Strategy:
      - Sample random subsets of basis Z-logicals of size target_logical_wt
      - XOR them together
      - Apply greedy stabilizer reduction to lower physical weight
      - Track candidate closest to target_phys_wt for diagnostics
    """
    HX = np.asarray(code.matrix_x).astype(int)
    HZ = np.asarray(code.matrix_z).astype(int)
    zls = np.asarray(code.get_logical_ops(Pauli.Z)).astype(int)
    k = code.dimension
    rng = _random.Random(seed)

    best_dist = None
    best_op = None
    best_phys = 0

    for trial in range(max_samples):
        # Vary subset size around target to enable some flexibility
        wt = target_logical_wt
        subset = rng.sample(range(k), wt)
        cur = np.zeros(code.num_qubits, dtype=int)
        for i in subset:
            cur = (cur + zls[i]) % 2
        cur = h.stab_reduce(cur, HZ, max_steps=40, seed=trial)
        if ((HX @ cur) % 2).sum() != 0:
            continue
        phys = int(cur.sum())
        dist = abs(phys - target_phys_wt)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_op = cur
            best_phys = phys
        if phys == target_phys_wt:
            return cur, best_op, best_phys
        # Progress logging at sparse intervals
        if trial < 5 or trial % 5000 == 0:
            print(f"    trial {trial}: phys_wt={phys} (best so far={best_phys})")

    return None, best_op, best_phys


def main() -> None:
    print("=" * 72)
    print("EXACT match for Cain Extended Data Table III lp_20^{3,5} Processor")
    print(f"Target (Qubits, X-checks, Z-checks): {TARGET}")
    print(f"Target operator: |P̄| (logical weight) = {TARGET_LOGICAL_WEIGHT},"
          f" physical weight = {TARGET_PHYSICAL_WEIGHT}")
    print("=" * 72)

    lp = build_lp20_3_5()
    print(f"\nlp_20^{{3,5}}: [[{lp.num_qubits}, {lp.dimension}]]")
    print(f"  Expected [[1122, 148]] per Cain Eq A3")
    if (lp.num_qubits, lp.dimension) != (1122, 148):
        print(f"  params mismatch - polynomials may be wrong; aborting")
        return

    print(f"\nStep 1: find P̄ with logical weight {TARGET_LOGICAL_WEIGHT}, "
          f"physical weight {TARGET_PHYSICAL_WEIGHT}")
    op, best_op, best_phys = find_P_via_subspace_search(
        lp,
        target_logical_wt=TARGET_LOGICAL_WEIGHT,
        target_phys_wt=TARGET_PHYSICAL_WEIGHT,
        max_samples=MAX_RANDOM_SAMPLES,
    )
    if op is None:
        print(f"  no exact-weight P̄ found in {MAX_RANDOM_SAMPLES} trials")
        print(f"  best sample: physical weight {best_phys} (target {TARGET_PHYSICAL_WEIGHT},"
              f" distance {abs(best_phys - TARGET_PHYSICAL_WEIGHT)})")
        print(f"  Cain uses Ref [113] algebraic construction for LP codes; we use random + reduce.")
        return
    else:
        print(f"  found P̄ with physical weight {int(op.sum())} = {TARGET_PHYSICAL_WEIGHT}")

    print("\nStep 2: build_layered_surgery_code(lp_dual, P̄)  [single-PPM Webster]")
    lp_dual = CSSCode(lp.matrix_z, lp.matrix_x, is_subsystem_code=False)
    merged, layout = build_layered_surgery_code(
        lp_dual, op, num_layers=1, validate_logical_op=False,
    )
    bare_shape = h.gadget_shape(layout)
    print(f"  Bare gadget: (kappa, chi, G) = {bare_shape}")

    print(f"\nStep 3: Cheeger boost seed sweep (0..{MAX_BOOST_SEEDS - 1})")
    for seed in range(MAX_BOOST_SEEDS):
        boosted, b_layout, _r = boost_gadget_cheeger(
            merged, layout, target_h=1.0, max_extra_qubits=500, seed=seed,
        )
        shape = h.gadget_shape(b_layout)
        if shape == TARGET:
            print(f"  EXACT MATCH at seed={seed}: {shape}")
            print("\n" + "=" * 72)
            print(f"✓ EXACT MATCH: {shape} = Cain target {TARGET}")
            print("=" * 72)
            return
    print(f"  no seed in 0..{MAX_BOOST_SEEDS - 1} produced {TARGET};"
          f" expand search OR reps wrong")


if __name__ == "__main__":
    main()
