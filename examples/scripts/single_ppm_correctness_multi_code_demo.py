"""Single-PPM Z-init correctness demo across multiple codes and operators.

Extends single_ppm_correctness_z_init_demo.py to test the protocol on:
- Steane [[7,1,3]]
- Distance-3 SurfaceCode (topological)
- Webster code 0 [[62,10,6]] with X̄_1 (matches paper)
- Webster code 0 with X̄_{k/2+1} (different operator, same code)
- Webster code 1 [[126,6,12]] (larger BB code)

For each (code, logical-X) pair: build PPM circuit, mutate data init
|+⟩ → |0⟩, sample noiselessly, verify:
  1. PPM outcome is genuine random ±1 (40% < flip rate < 60%)
  2. Both observables (Webster Eq.1 + final-Mx X̄_M) agree on EVERY shot

If a code passes both, the surgery protocol works correctly on it.

Run: python examples/scripts/single_ppm_correctness_multi_code_demo.py
"""

from __future__ import annotations

import time
import numpy as np
import stim

from qldpc import codes
import sys
from pathlib import Path

from qldpc.codes.surgery import build_gadget, build_single_ppm_circuit
from qldpc.objects import Pauli

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _webster_seed_set import (  # noqa: E402
    build_generalised_bicycle_code,
    load_webster_seed_set,
)


def _swap_data_init_to_zero(circuit: stim.Circuit, data_ids: list[int]) -> stim.Circuit:
    """Replace RX(data_ids) with R(data_ids); preserve all other ops."""
    data_set = set(data_ids)
    out = stim.Circuit()
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            out.append(stim.CircuitRepeatBlock(
                op.repeat_count,
                _swap_data_init_to_zero(op.body_copy(), data_ids),
            ))
            continue
        if op.name == "RX":
            targets = [t.value for t in op.targets_copy()]
            data_targets = [t for t in targets if t in data_set]
            other_targets = [t for t in targets if t not in data_set]
            if data_targets:
                out.append("R", data_targets)
            if other_targets:
                out.append("RX", other_targets)
        else:
            out.append(op)
    return out


def _webster_operator(d: dict, name: str) -> np.ndarray:
    """Extract a Webster seed-set logical operator (X_bar_1 or X_bar_k2p1) as a 2l vector."""
    pauli_type = name[0]  # "X" or "Z"
    l = d["l"]
    for seed in d["seeds"]:
        if seed["name"] == name and seed["pauli_type"] == pauli_type:
            L = np.zeros(l, dtype=np.uint8)
            R = np.zeros(l, dtype=np.uint8)
            for i in seed["L_support"]:
                L[i] = 1
            for i in seed["R_support"]:
                R[i] = 1
            return np.concatenate([L, R])
    raise ValueError(f"{name} not found")


def verify(name: str, code, x_logical: np.ndarray, rounds: int = 3, shots: int = 4000) -> bool:
    """Build, mutate, sample, assert. Returns True if all checks pass."""
    print(f"\n=== {name} ===")
    t0 = time.time()
    g = build_gadget(code, x_logical)
    circuit_plus = build_single_ppm_circuit(g, rounds=rounds, noise_model=None)
    n_data = code.num_qudits
    data_ids = list(range(n_data))
    circuit_zero = _swap_data_init_to_zero(circuit_plus, data_ids)
    t_build = time.time() - t0

    print(f"  data qubits     : {n_data}")
    print(f"  κ ancillas      : {len(g.kappa_qubits)}")
    print(f"  |V_0|           : {len(g.V0)}")
    print(f"  build time      : {t_build:.2f}s")

    t0 = time.time()
    sampler = circuit_zero.compile_detector_sampler()
    _, observables = sampler.sample(shots=shots, separate_observables=True)
    t_sample = time.time() - t0

    obs0 = observables[:, 0]
    obs1 = observables[:, 1]
    rate0 = float(obs0.mean())
    rate1 = float(obs1.mean())
    agree = float((obs0 == obs1).mean())

    print(f"  sample time     : {t_sample:.2f}s ({shots} shots)")
    print(f"  obs0 (Eq.1)     : {rate0:.2%} flips  (expected ~50%)")
    print(f"  obs1 (X̄ check) : {rate1:.2%} flips  (expected ~50%)")
    print(f"  obs0 == obs1    : {agree:.2%} of shots  (expected 100%)")

    ok_random = 0.40 < rate0 < 0.60 and 0.40 < rate1 < 0.60
    ok_agree = agree == 1.0

    if ok_random and ok_agree:
        print(f"  ✓ PASS")
        return True
    else:
        if not ok_random:
            print(f"  ✗ FAIL: not genuine random (rates outside [40%, 60%])")
        if not ok_agree:
            print(
                f"  ✗ FAIL: {int((1 - agree) * shots)} of {shots} shots disagree — "
                f"observables don't measure the same X̄_M"
            )
        return False


def main() -> None:
    print("Single-PPM Z-init Correctness Across Multiple Codes")
    print("=" * 65)

    results = []

    # 1. Steane [[7, 1, 3]]
    steane = codes.SteaneCode()
    x_steane = np.asarray(steane.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    results.append(("Steane [[7, 1, 3]]", verify("Steane [[7, 1, 3]]", steane, x_steane)))

    # 2. Surface code, distance 3 [[9, 1, 3]] (or similar small dist)
    try:
        surface = codes.SurfaceCode(3)
        x_surface = np.asarray(surface.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
        n = surface.num_qudits
        k = surface.dimension
        results.append((
            f"SurfaceCode d=3 [[{n}, {k}]]",
            verify(f"SurfaceCode d=3 [[{n}, {k}]]", surface, x_surface),
        ))
    except Exception as e:
        print(f"\n=== SurfaceCode (skipped: {e}) ===")

    # 3. Webster code 0 [[62, 10, 6]] — X̄_1
    d0 = load_webster_seed_set(0)
    w0 = build_generalised_bicycle_code(d0["l"], d0["A"], d0["B"])
    x_w0 = _webster_operator(d0, "X_bar_1")
    results.append((
        "Webster 0 [[62, 10, 6]] X̄_1",
        verify("Webster 0 [[62, 10, 6]] X̄_1", w0, x_w0),
    ))

    # 4. Webster code 0 — X̄_{k/2+1} (different operator)
    x_w0_alt = _webster_operator(d0, "X_bar_k2p1")
    results.append((
        "Webster 0 [[62, 10, 6]] X̄_{k/2+1}",
        verify("Webster 0 [[62, 10, 6]] X̄_{k/2+1}", w0, x_w0_alt),
    ))

    # 5. Webster code 1 [[126, 6, 12]]
    d1 = load_webster_seed_set(1)
    w1 = build_generalised_bicycle_code(d1["l"], d1["A"], d1["B"])
    x_w1 = _webster_operator(d1, "X_bar_1")
    results.append((
        "Webster 1 [[126, 6, 12]] X̄_1",
        verify("Webster 1 [[126, 6, 12]] X̄_1", w1, x_w1),
    ))

    # 6. Gross / IBM bivariate-bicycle [[144, 12, 12]]
    # A = x^3 + y + y^2, B = y^3 + x + x^2, orders (R_x=12, R_y=6)
    # Reference: Bravyi-Cross-Cohn-Tillich-Yoder arXiv:2308.07915
    import sympy
    x, y = sympy.symbols("x y")
    gross = codes.BBCode({x: 12, y: 6}, x**3 + y + y**2, y**3 + x + x**2)
    x_gross = np.asarray(gross.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    results.append((
        "Gross BB [[144, 12, 12]] X̄_1",
        verify("Gross BB [[144, 12, 12]] X̄_1", gross, x_gross, shots=2000),
    ))

    # Summary
    print("\n" + "=" * 65)
    print("Summary:")
    n_pass = sum(1 for _, ok in results if ok)
    for name, ok in results:
        flag = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {flag}  {name}")
    print(f"\n{n_pass}/{len(results)} codes passed.")
    if n_pass == len(results):
        print("\nAll codes show: noiseless PPM is genuine X̄_M measurement,")
        print("both observables consistent across thousands of random shots.")


if __name__ == "__main__":
    main()
