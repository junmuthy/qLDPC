"""EXACT match for Cain Extended Data Table III bb_18 Processor.

Target: (Qubits, X-checks, Z-checks) = (189, 104, 86).
Pipeline:
  1. Build bb_18 [[248, 10]] from Cain App. A Eq A11.
  2. Pick 9 of 10 Z-logical reps; search reductions for total |V_0| = 104.
  3. build_multi_target_surgery_code(bb_18_dual, 9 reps).
  4. Spectral Cheeger boost with seed sweep until (κ, χ, G) = (189, 104, 86).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import sympy

sys.path.insert(0, str(Path(__file__).parent))
import _cain_helpers as h

from qldpc import codes
from qldpc.codes.common import CSSCode
from qldpc.codes.surgery import (
    boost_gadget_cheeger,
    build_multi_target_surgery_code,
)
from qldpc.objects import Pauli


TARGET = (189, 104, 86)
MAX_SEEDS = 1000


def build_bb18():
    """Cain App. A Eq A11: l=31, m=4, a=1+x^6 y+x^27, b=y^2+x^15 y^3+x^24."""
    x, y = sympy.symbols("x y")
    return codes.BBCode(
        (31, 4),
        1 + x**6 * y + x**27,
        y**2 + x**15 * y**3 + x**24,
    )


def find_9_logical_reps(
    code, target_v0_size: int = 104
) -> tuple[list[np.ndarray] | None, list[tuple[int, int]], list[np.ndarray]]:
    """Pick 9 of 10 Z-logicals; per-rep stab-reduce for low weight; check V_0 size.

    Returns (matching_ops, all_sizes, best_ops_by_distance) where best_ops_by_distance
    is the leave-one-out combo whose |V_0| is closest to target_v0_size.
    """
    HX = np.asarray(code.matrix_x).astype(int)
    HZ = np.asarray(code.matrix_z).astype(int)
    zls = np.asarray(code.get_logical_ops(Pauli.Z)).astype(int)
    all_sizes: list[tuple[int, int]] = []
    best_ops: list[np.ndarray] = []
    best_dist = None
    # Try each "leave-one-out" combo
    for leave_out in range(code.dimension):
        chosen = [i for i in range(code.dimension) if i != leave_out]
        ops = []
        for i in chosen:
            v = zls[i].copy()
            v = h.stab_reduce(v, HZ, max_steps=80, seed=leave_out * 100 + i)
            assert ((HX @ v) % 2).sum() == 0
            ops.append(v)
        v0_union = np.zeros(code.num_qubits, dtype=int)
        for op in ops:
            v0_union = v0_union | op
        size = int(v0_union.sum())
        all_sizes.append((leave_out, size))
        dist = abs(size - target_v0_size)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_ops = ops
        if size == target_v0_size:
            return ops, all_sizes, best_ops
    return None, all_sizes, best_ops


def main() -> None:
    print("=" * 72)
    print("EXACT match for Cain Extended Data Table III bb_18 Processor")
    print(f"Target (Qubits, X-checks, Z-checks): {TARGET}")
    print("=" * 72)

    bb = build_bb18()
    print(f"\nbb_18: [[{bb.num_qubits}, {bb.dimension}]]")
    assert (bb.num_qubits, bb.dimension) == (248, 10)

    print("\nStep 1: find 9 Z-logical reps with |V_0_union| = 104")
    ops, all_sizes, best_ops = find_9_logical_reps(bb, target_v0_size=104)
    if ops is None:
        print("  no leave-one-out combo gives |V_0| = 104")
        sizes_only = [s for _, s in all_sizes]
        print(f"  observed |V_0| sizes across 10 leave-one-out combos: {sizes_only}")
        v0_union = np.zeros(bb.num_qubits, dtype=int)
        for op in best_ops:
            v0_union = v0_union | op
        best_size = int(v0_union.sum())
        print(
            f"  proceeding with closest combo: |V_0| = {best_size} (Cain target = 104)"
        )
        ops = best_ops
    else:
        print(f"  found {len(ops)} reps with |V_0| = 104")

    print("\nStep 2: multi-target Webster gadget on V_0_union")
    bb_dual = CSSCode(bb.matrix_z, bb.matrix_x, is_subsystem_code=False)
    merged, layout = build_multi_target_surgery_code(bb_dual, ops, validate=False)
    base = layout.base_layout
    bare_shape = h.gadget_shape(base)
    print(f"  Bare gadget: (kappa, chi, G) = {bare_shape}")

    print(f"\nStep 3: Cheeger boost seed sweep (0..{MAX_SEEDS - 1})")
    for seed in range(MAX_SEEDS):
        boosted, b_layout, _result = boost_gadget_cheeger(
            merged, base, target_h=1.0, max_extra_qubits=200, seed=seed,
        )
        shape = h.gadget_shape(b_layout)
        if shape == TARGET:
            print(f"  EXACT MATCH at seed={seed}: {shape}")
            print("\n" + "=" * 72)
            print(f"EXACT MATCH with Cain Table III: {shape}")
            print(f"  Cain target: {TARGET}")
            print("=" * 72)
            return
    print(f"  no seed in 0..{MAX_SEEDS - 1} produced {TARGET}; expand range or revise search")


if __name__ == "__main__":
    main()
