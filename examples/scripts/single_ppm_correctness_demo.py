"""Single-PPM surgery correctness demo (public-API only).

Demonstrates that under noiseless conditions, build_single_ppm_circuit
deterministically returns the correct PPM result (X̄_M eigenvalue = +1)
across 1000 shots.

Logic:
- Initialize data qubits in logical |+⟩ (X-eigenstate). X̄_M |+⟩_logical = +1 |+⟩_logical.
- Run τ_s rounds of merged-code syndrome extraction.
- OBSERVABLE_INCLUDE(0) = ⊕ χ-row records across all rounds (Webster Eq. 1)
  = X̄_M eigenvalue when noiseless.
- OBSERVABLE_INCLUDE(1) = ⊕ final-data measurements on V_0 = X̄_M eigenvalue.

Under noiseless operation, both observables MUST be 0 (= +1 in stim convention)
in every shot. If even a single shot flips, the circuit is wrong.

Run with: python examples/scripts/single_ppm_correctness_demo.py
"""

from __future__ import annotations

import numpy as np

from qldpc import codes
import sys
from pathlib import Path

from qldpc.circuits.surgery import build_gadget, build_single_ppm_circuit
from qldpc.objects import Pauli

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _webster_seed_set import (  # noqa: E402
    build_generalised_bicycle_code,
    load_webster_seed_set,
)


def verify_ppm_correctness(name: str, code, x_logical: np.ndarray, rounds: int, shots: int) -> None:
    """Build single-PPM circuit, sample noiselessly, assert both observables = 0."""
    g = build_gadget(code, x_logical)
    circuit = build_single_ppm_circuit(g, rounds=rounds, noise_model=None)

    n_data = code.num_qudits
    print(f"\n=== {name} ===")
    print(f"  data qubits     : {n_data}")
    print(f"  κ ancillas      : {len(g.kappa_qubits)}")
    print(f"  |V_0|           : {len(g.V0)}  (number of χ rows = X̄_M support weight)")
    print(f"  merged code     : [[{n_data + len(g.kappa_qubits)}, ?]]")
    print(f"  circuit ops     : {len(circuit)} stim instructions")
    print(f"  num_detectors   : {circuit.num_detectors}")
    print(f"  num_observables : {circuit.num_observables}")

    sampler = circuit.compile_detector_sampler()
    detectors, observables = sampler.sample(shots=shots, separate_observables=True)

    # Both observables MUST be 0 in every shot if the protocol is correct.
    obs0_flips = int(observables[:, 0].sum())
    obs1_flips = int(observables[:, 1].sum())
    det_fires = int(detectors.sum())

    print(f"  shots           : {shots}")
    print(f"  detector fires  : {det_fires} (expected: 0)")
    print(f"  observable 0 flips (Webster Eq.1, PPM result): {obs0_flips} (expected: 0)")
    print(f"  observable 1 flips (X̄_M cross-check)        : {obs1_flips} (expected: 0)")

    assert det_fires == 0, f"{name}: detector fired noiselessly ({det_fires} times)"
    assert obs0_flips == 0, f"{name}: PPM observable flipped noiselessly ({obs0_flips} times)"
    assert obs1_flips == 0, f"{name}: data cross-check flipped noiselessly ({obs1_flips} times)"
    print(f"  ✓ noiseless PPM returns +1 in all {shots} shots")


def _x_bar_1_operator(d: dict) -> np.ndarray:
    """Extract X̄_1 from a Webster seed_set as a 2l binary vector."""
    l = d["l"]
    for seed in d["seeds"]:
        if seed["name"] == "X_bar_1" and seed["pauli_type"] == "X":
            L = np.zeros(l, dtype=np.uint8)
            R = np.zeros(l, dtype=np.uint8)
            for i in seed["L_support"]:
                L[i] = 1
            for i in seed["R_support"]:
                R[i] = 1
            return np.concatenate([L, R])
    raise ValueError("X_bar_1 not found")


def main() -> None:
    print("Single-PPM Surgery Correctness Demo")
    print("=" * 60)
    print("All assertions must pass for the circuit to be considered correct.")

    # Example 1: Steane code (smallest non-trivial CSS code)
    steane = codes.SteaneCode()
    x_steane = np.asarray(steane.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    verify_ppm_correctness("Steane [[7, 1, 3]]", steane, x_steane, rounds=3, shots=1000)

    # Example 2: Webster Appendix A code 0 — matches paper Table I
    data0 = load_webster_seed_set(0)
    webster0 = build_generalised_bicycle_code(data0["l"], data0["A"], data0["B"])
    x_webster0 = _x_bar_1_operator(data0)
    verify_ppm_correctness(
        "Webster code 0 [[62, 10, 6]]", webster0, x_webster0, rounds=3, shots=1000
    )

    print("\n" + "=" * 60)
    print("All demos passed. Single-PPM circuit is correct under noiseless conditions.")


if __name__ == "__main__":
    main()
