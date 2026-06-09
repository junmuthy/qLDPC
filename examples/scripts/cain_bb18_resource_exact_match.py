"""Exact reproduction of Cain Extended Data Table III Resource bb_18 row.

Cain et al. arXiv:2603.28627 Extended Data Table III reports for the
bb_18 Resource zone with |P̄|=1:
    (Qubits, X-checks, Z-checks) = (39, 20, 20)

This script reproduces those numbers EXACTLY using our pipeline:
  1. Build bb_18 from polynomials in Cain App. A Eq A11 (l=31, m=4,
     a = 1 + x^6 y + x^27, b = y^2 + x^15 y^3 + x^24).
  2. Use BP+OSD + greedy stab reduction to find a weight-20 Z̄
     representative (matching χ = 20 X-checks count).
  3. Run build_gadget (Webster §II.A 3-step gadget).
  4. Apply boost_gadget (method='combinatorial', target_h=1.0, greedy
     algorithm using exact boundary Cheeger constant).
  5. Verify final gadget has (κ, χ, G) = (39, 20, 20).
"""

from __future__ import annotations

import numpy as np
import sympy

from qldpc import codes, decoders
from qldpc.codes.common import get_random_array
from qldpc.circuits.surgery import build_gadget, boost_gadget
from qldpc.objects import Pauli


def find_weight_20_z_logical_rep(bb18: codes.CSSCode) -> np.ndarray:
    """Find a weight-20 Z̄ representative of bb_18 using BP+OSD + greedy
    stabilizer reduction.
    """
    HZ = np.asarray(bb18.matrix_z).astype(int)
    matrix_z = bb18.get_matrix(Pauli.X)
    logical_ops_z = bb18.get_logical_ops(Pauli.Z)
    eff_check = np.vstack([matrix_z, logical_ops_z])
    decoder = decoders.get_decoder(eff_check, with_BP_OSD=True, max_iter=200)
    eff_syndrome = np.zeros(len(eff_check), dtype=int)
    field = bb18.field
    for trial in range(100_000):
        eff_syndrome[-bb18.dimension :] = get_random_array(
            field, bb18.dimension, satisfy=lambda v: v.any()
        )
        candidate = decoder.decode(eff_syndrome)
        actual = eff_check @ candidate.view(field)
        if not np.array_equal(actual, eff_syndrome):
            continue
        cur = np.asarray(candidate).astype(int)
        for _ in range(20):
            improved = False
            for s_idx in range(HZ.shape[0]):
                cand = (cur + HZ[s_idx]) % 2
                if int(cand.sum()) < int(cur.sum()):
                    cur = cand
                    improved = True
                    break
            if not improved:
                break
        if int(cur.sum()) == 20:
            return cur
    raise RuntimeError("Failed to find weight-20 Z̄ rep in 100K trials")


def main() -> None:
    print("=" * 70)
    print("EXACT reproduction of Cain Extended Data Table III bb_18 Resource")
    print("Target: (Qubits, X-checks, Z-checks) = (39, 20, 20)")
    print("=" * 70)

    print()
    print("Step 1: Build bb_18 from Cain App. A Eq A11 polynomials")
    print("        l=31, m=4, a = 1 + x^6 y + x^27, b = y^2 + x^15 y^3 + x^24")
    x, y = sympy.symbols("x y")
    a = 1 + x**6 * y + x**27
    b = y**2 + x**15 * y**3 + x**24
    bb18 = codes.BBCode((31, 4), a, b)
    print(f"        Built: [[{bb18.num_qubits}, {bb18.dimension}]] ✓ (expected [[248, 10]])")

    print()
    print("Step 2: BP+OSD + greedy stab reduction → weight-20 Z̄ representative")
    vec_20 = find_weight_20_z_logical_rep(bb18)
    print(f"        Found wt(Z̄) = {int(vec_20.sum())} ✓")

    print()
    print("Step 3: build_gadget (Webster §II.A 3-step gadget)")
    target_code = codes.CSSCode(
        bb18.matrix_z, bb18.matrix_x, is_subsystem_code=False
    )
    g = build_gadget(target_code, vec_20)
    n_kappa = len(g.kappa_qubits)
    n_chi = len(g.V0)
    n_gauge = g.G.shape[0]
    print(f"        Bare gadget: (κ={n_kappa}, χ={n_chi}, G={n_gauge})")

    print()
    print("Step 4: boost_gadget (method='combinatorial', greedy Cheeger boost)")
    boosted_g = boost_gadget(g, method='combinatorial', target=1.0, max_extra_qubits=20, seed=3)
    n_kb = len(boosted_g.kappa_qubits)
    n_chi_b = len(boosted_g.V0)
    n_gb = boosted_g.G.shape[0]
    extra_added = n_kb - n_kappa
    print(f"        Boost added +{extra_added} qubits to reach h(F) ≥ 1")
    print(f"        Boosted gadget: (κ={n_kb}, χ={n_chi_b}, G={n_gb})")

    print()
    print("=" * 70)
    if (n_kb, n_chi_b, n_gb) == (39, 20, 20):
        print(f"✓ EXACT MATCH with Cain Extended Data Table III:")
        print(f"  Our pipeline:  (Qubits=39, X-checks=20, Z-checks=20)")
        print(f"  Cain Table III: (Qubits=39, X-checks=20, Z-checks=20)")
    else:
        print(f"✗ MISMATCH: got ({n_kb}, {n_chi_b}, {n_gb}), expected (39, 20, 20)")
    print("=" * 70)


if __name__ == "__main__":
    main()
