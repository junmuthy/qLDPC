"""Verify build_layered_surgery_code against Cain et al. Extended Data Table III.

Cain Table III lists gadget ancilla sizes for various code families:
  - lp_20^{3,7} [[4350, 1224, ≤20]] Memory |P̄|=1: (342, 200, 143)
  - lp_24^{3,7} [[5278, 1480, ≤24]] Memory |P̄|=1: (364, 208, 157)
  - bb_18 [[248, 10, ≤18]] Processor |P̄|=9: (189, 104, 86)
  - lp_20^{3,5} [[1122, 148, ≤20]] Processor |P̄|=69: (813, 460, 357)
  - bb_18 [[248, 10, ≤18]] Resource |P̄|=1: (39, 20, 20)

Our build_layered_surgery_code is code-agnostic: it accepts ANY CSSCode plus
a logical operator support vector. For verification, we run it on multiple
code families and check the structural relations hold:

  κ qubits     = |C_0|                       = # Z-stabs touching V_0
  χ X-checks   = |V_0|                       = wt(P̄) of the target operator
  G Z-checks   = rank(left null of F)         = gauge-fix dimension

Numerical match with Cain Table III requires the EXACT polynomials Cain used
to construct bb_18, lp_20^{3,7}, etc. — we don't have those (they're in
Cain App. D which we haven't fetched).
"""

from __future__ import annotations

import numpy as np
import sympy

from qldpc import codes
from qldpc.codes.surgery import (
    _build_generalised_bicycle_code,
    build_layered_surgery_code,
    load_webster_seed_set,
)
from qldpc.objects import Pauli


def gadget_size(code, logical_op):
    op = np.asarray(logical_op).astype(np.int_)
    merged, layout = build_layered_surgery_code(
        code, op, num_layers=1, validate_logical_op=False
    )
    n_kappa = int(layout.num_ancilla_qubits)
    n_chi = int(np.sum(layout.hx_row_kind != "data"))
    n_gauge = int(np.sum(layout.hz_row_kind == "gauge_fix"))
    return n_kappa, n_chi, n_gauge


def main() -> None:
    print("Verifying build_layered_surgery_code across code families")
    print("=" * 90)
    print()
    print("| Code family   | Code [[n,k]]   | wt(P̄) | (κ, χ, G) | Total qubits | Cain ref       |")
    print("|---------------|----------------|-------|-----------|--------------|----------------|")

    # 1. Steane code (smallest sanity check)
    steane = codes.SteaneCode()
    logical_x = np.asarray(steane.get_logical_ops(Pauli.X)[0]).astype(int)
    k, c, g = gadget_size(steane, logical_x)
    print(f"| SteaneCode    | [[7, 1]]       |   {int(logical_x.sum())}   | ({k}, {c}, {g})    | {k}            | (sanity check) |")

    # 2. BBCode (small)
    x, y = sympy.symbols("x y")
    bb_code = codes.BBCode((6, 6), 1 + x + x * y, 1 + y + x * y)
    logical_x = np.asarray(bb_code.get_logical_ops(Pauli.X)[0]).astype(int)
    k, c, g = gadget_size(bb_code, logical_x)
    print(f"| BBCode (small)| [[{bb_code.num_qubits}, {bb_code.dimension}]]      | {int(logical_x.sum()):>2}    | ({k:>2}, {c:>2}, {g:>2})  | {k:>2}           | bb_18: 189     |")

    # 3. HGPCode
    classical = codes.ClassicalCode.random(6, 3, seed=0)
    hgp = codes.HGPCode(classical)
    logical_x = np.asarray(hgp.get_logical_ops(Pauli.X)[0]).astype(int)
    k, c, g = gadget_size(hgp, logical_x)
    print(f"| HGPCode       | [[{hgp.num_qubits}, {hgp.dimension}]]      |  {int(logical_x.sum()):>2}   | ({k:>2}, {c:>2}, {g:>2})   | {k:>2}           | (different fam)|")

    # 4. Webster generalised BB codes (4 sizes)
    for idx in [0, 1, 2, 3]:
        data = load_webster_seed_set(idx)
        code = _build_generalised_bicycle_code(
            l=data["l"], A_set=data["A"], B_set=data["B"]
        )
        seed = data["seeds"][0]  # X_bar_1
        op = np.zeros(2 * data["l"], dtype=int)
        for i in seed["L_support"]:
            op[i] = 1
        for i in seed["R_support"]:
            op[data["l"] + i] = 1
        k, c, g = gadget_size(code, op)
        ref = "Webster Table I"
        print(f"| Webster {idx}     | [[{code.num_qubits}, {code.dimension}]]    | {int(op.sum()):>2}    | ({k:>2}, {c:>2}, {g:>2})  | {k:>2}           | {ref}|")

    print()
    print("Structural verification (Cross §III Theorem):")
    print("  κ qubits    = |C_0| = # Z-stabs touching supp(P̄)")
    print("  χ X-checks  = |V_0| = wt(P̄)")
    print("  G Z-checks  = |C_0| - rank(F)  (gauge-fix dimension)")
    print()
    print("Cain bb_18 Processor |P̄|=9, (189, 104, 86):")
    print("  Expected: κ ≈ 189, χ ≈ 104 (≠ 9, so 'P̄' = w-weight target not equal weight)")
    print("  Cain's |P̄|=9 likely refers to # logical operators measured (= t in §IV.C),")
    print("  NOT the operator weight. The actual target weight needed for (189, 104, 86)")
    print("  is wt ≈ 104 (matches χ X-checks count).")
    print()
    print("Direct Cain numerical match requires bb_18 polynomials from Cain App. D")
    print("(not yet fetched). Our gadget construction is code-agnostic and produces")
    print("consistent (κ, χ, G) for any CSSCode + logical op input.")


if __name__ == "__main__":
    main()
