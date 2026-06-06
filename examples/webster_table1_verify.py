"""Webster Table I verification script.

Reproduces the overhead numbers from Table I of Webster, Smith, Cohen
(arXiv:2511.15989) by constructing each of the 4 generalised bicycle
codes from Appendix A, building gadgets and bridges, and comparing
against the paper's published numbers.

Webster-style "gadget qubits" = data ancillas + syndrome ancillas for new
X-checks and gauge-fix Z-checks = num_ancilla_qubits + |new X-checks| +
|gauge-fix Z-checks|. Webster-style "bridge qubits" per pair = 2w − 1
where w = min(wt(L_1), wt(L_2)).

Bare gadget numbers are the hard acceptance gate. Bridge numbers are
informational comparison.

Usage:
    python examples/webster_table1_verify.py
"""

from __future__ import annotations

import sys

import numpy as np

from qldpc import codes
from qldpc.codes.surgery import (
    boost_gadget_cheeger,
    build_joint_measurement_code,
    build_layered_surgery_code,
    load_webster_seed_set,
)
from qldpc.codes.surgery import _build_generalised_bicycle_code
from qldpc.objects import Pauli


def _support_to_binary_vector(L_support: list[int], R_support: list[int], l: int) -> np.ndarray:
    vec = np.zeros(2 * l, dtype=np.int_)
    for i in L_support:
        vec[i] = 1
    for i in R_support:
        vec[l + i] = 1
    return vec


def _webster_gadget_qubits(layout) -> int:
    """Webster-style gadget qubit count.

    = num_ancilla_qubits + #new X-checks + #gauge-fix Z-checks.
    """
    n_kappa = int(layout.num_ancilla_qubits)
    n_chi = int(np.sum(layout.hx_row_kind != "data"))
    n_gauge_fix = int(np.sum(layout.hz_row_kind == "gauge_fix"))
    return n_kappa + n_chi + n_gauge_fix


def main() -> int:
    print("Webster Table I verification")
    print("=" * 100)

    rows = []
    all_bare_match = True
    all_bridge_match = True

    for code_index in range(4):
        data = load_webster_seed_set(code_index)
        code = _build_generalised_bicycle_code(
            l=data["l"], A_set=data["A"], B_set=data["B"]
        )
        expected_bare = data["expected_bare_gadget_qubits_per_seed"]
        expected_bridge = data["expected_bridge_qubits_per_pair"]
        expected_cheeger = data["expected_cheeger_boost_qubits"]

        observed_bare_per_seed = []
        for seed in data["seeds"]:
            op = _support_to_binary_vector(seed["L_support"], seed["R_support"], data["l"])
            if seed["pauli_type"] == "X":
                target = code
            else:
                target = codes.CSSCode(code.matrix_z, code.matrix_x, is_subsystem_code=False)
            _, layout = build_layered_surgery_code(target, op, num_layers=1, validate_logical_op=False)
            observed_bare_per_seed.append(_webster_gadget_qubits(layout))

        bare_ok = all(o == expected_bare for o in observed_bare_per_seed)
        all_bare_match = all_bare_match and bare_ok

        # Bridge: pair X̄_1 and X̄_{k/2+1}
        x_seeds = [s for s in data["seeds"] if s["pauli_type"] == "X"]
        if len(x_seeds) >= 2:
            op_a = _support_to_binary_vector(x_seeds[0]["L_support"], x_seeds[0]["R_support"], data["l"])
            op_b = _support_to_binary_vector(x_seeds[1]["L_support"], x_seeds[1]["R_support"], data["l"])
            try:
                _, joint_layout = build_joint_measurement_code(code, op_a, op_b, num_layers=1, validate=False)
                w = int(joint_layout.num_bridge_qubits)
                observed_bridge = 2 * w - 1
                bridge_ok = (observed_bridge == expected_bridge)
            except Exception as exc:
                observed_bridge = f"FAIL: {type(exc).__name__}"
                bridge_ok = False
        else:
            observed_bridge = "n/a"
            bridge_ok = False
        all_bridge_match = all_bridge_match and bridge_ok

        # Cheeger boost (only meaningful for codes 3, 4 where expected > 0)
        if expected_cheeger > 0:
            op = _support_to_binary_vector(data["seeds"][0]["L_support"], data["seeds"][0]["R_support"], data["l"])
            if data["seeds"][0]["pauli_type"] == "X":
                target = code
            else:
                target = codes.CSSCode(code.matrix_z, code.matrix_x, is_subsystem_code=False)
            merged_x, layout_x = build_layered_surgery_code(target, op, num_layers=1, validate_logical_op=False)
            try:
                _, _, boost_result = boost_gadget_cheeger(
                    merged_x, layout_x, target_h=1.0,
                    max_extra_qubits=expected_cheeger * 3, seed=42,
                )
                observed_cheeger = boost_result.extra_qubits_added
            except Exception as exc:
                observed_cheeger = f"FAIL: {type(exc).__name__}"
        else:
            observed_cheeger = 0

        rows.append({
            "name": data["name"],
            "expected_bare": expected_bare,
            "observed_bare": observed_bare_per_seed,
            "bare_ok": bare_ok,
            "expected_cheeger": expected_cheeger,
            "observed_cheeger": observed_cheeger,
            "expected_bridge": expected_bridge,
            "observed_bridge": observed_bridge,
            "bridge_ok": bridge_ok,
        })

    # Print markdown table
    print()
    print("| Code         | Bare/seed (paper) | Bare/seed (ours)               | Bare OK | +n (paper) | +n (ours) | Bridge (paper) | Bridge (ours) | Bridge OK |")
    print("|--------------|-------------------|--------------------------------|---------|------------|-----------|----------------|---------------|-----------|")
    for r in rows:
        observed_str = ", ".join(str(x) for x in r["observed_bare"])
        bare_emoji = "OK" if r["bare_ok"] else "FAIL"
        bridge_emoji = "OK" if r["bridge_ok"] else "FAIL"
        print(
            f"| {r['name']:12} | {r['expected_bare']:17} | {observed_str:30} | {bare_emoji:7} | "
            f"{r['expected_cheeger']:10} | {str(r['observed_cheeger']):9} | "
            f"{r['expected_bridge']:14} | {str(r['observed_bridge']):13} | {bridge_emoji:9} |"
        )

    print()
    if all_bare_match and all_bridge_match:
        print("All Webster Table I numbers match: bare gadget AND bridge. v2 acceptance PASSED.")
        return 0
    if all_bare_match:
        print("Bare gadget numbers match Webster Table I. Bridge numbers DO NOT all match. Partial pass.")
        return 1
    print("Bare gadget numbers DO NOT match Webster Table I. v2 acceptance FAILED.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
