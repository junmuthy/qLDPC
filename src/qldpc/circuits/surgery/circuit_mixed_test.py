"""Tier 1 noiseless correctness tests for mixed-basis joint PPM.

Mixed-basis joint Pauli-product measurement of X̄_l ⊗ Z̄_r is built on top
of the Cross, He, Rall, Yoder (arXiv:2407.18393) Theorem 20 subsystem-code
construction, specialized to the lattice-surgery setting by Cowtan, He,
Williamson, Yoder (arXiv:2503.05003 §3.5).

This module verifies the *Tier 1* (noiseless circuit build + DEM compile)
acceptance bar for the mixed-basis pipeline:
  * ``build_joint_ppm_circuit`` produces a ``stim.Circuit`` without
    asserting CSSCode (it dispatches to the subsystem-code-aware path).
  * The resulting circuit compiles to a ``stim.DetectorErrorModel``
    successfully — the syndrome graph for the pure-X / pure-Z stabilizer
    block is well-formed.

obs0 is NOT emitted in Tier 1. As of the split-schedule fix
(``_build_joint_ppm_circuit_mixed_basis`` runs X-stab CX gates and X-MX
in a fully separate phase BEFORE the Z-stab CZ gates begin), individual
χ_l / bridge_gauge measurement outcomes are deterministic in the
subsystem-code sense. However, the canonical
``⊕ m(χ_l) ⊕ ⊕ m(χ_r) ⊕ ⊕ m(bridge_gauges)`` formula remains
non-deterministic shot-to-shot because the operator product
X̄_l ⊗ Z̄_r ⊗ Z^|B| anti-commutes with the χ_l and bridge_gauge
X-on-adapter rows themselves (Z-on-adapter from ∏ χ_r anti-commutes
with X-on-adapter from individual χ_l). Per the design spec §9 Lemma 2
the full obs0 must additionally XOR in (i) Y-stab rows from the
pair-merge step and (ii) leftover X-cycle / Z-cycle outcomes. Those
extra Bridge fields are not yet populated by
``_stitch_to_joint_code_mixed`` — Tier 2 work. See
``_build_joint_ppm_circuit_mixed_basis`` docstring for full details.

The full truth-table verification (obs0 sign for all four |ψ_l⟩ ⊗ |ψ_r⟩
joint eigenstates) is captured below as ``xfail`` — see
``test_mixed_basis_joint_truth_table_x_l_z_r`` for the design-spec
limitation.
"""

from __future__ import annotations

import numpy as np
import pytest
import stim

from qldpc import codes
from qldpc.objects import Pauli


def _build_steane_mixed_pair():
    """Build a Steane × Steane mixed-basis (X, Z) pair for joint PPM of X̄_l ⊗ Z̄_r."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget

    code_l = codes.SteaneCode()
    code_r = codes.SteaneCode()
    x = np.asarray(code_l.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code_r.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code_l, x, basis=Pauli.X)
    g_r = build_gadget(code_r, z, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)
    return g_l, g_r, bridge


def test_mixed_basis_joint_ppm_circuit_builds() -> None:
    """build_joint_ppm_circuit succeeds for mixed-basis input (no CSSCode assertion error).

    Uses ``data_init=("+", "0")`` which IS a true +1 eigenstate of X̄_l ⊗ Z̄_r
    (X̄_l|+⟩=+|+⟩, Z̄_r|0⟩=+|0⟩). The earlier ``("0", "+")`` choice was
    incorrect: |0⟩ is not an X̄_l eigenstate when basis_l=X measures X̄_l,
    so χ_l outcomes would not encode any well-defined X̄_l eigenvalue.

    Tier 1: obs0 is intentionally not emitted (see
    ``_build_joint_ppm_circuit_mixed_basis`` docstring) — the canonical
    Cross–He–Rall–Yoder formula
    ``⊕ m(χ_l) ⊕ ⊕ m(χ_r) ⊕ ⊕ m(bridge_gauges)`` is missing Y-stab and
    leftover X/Z cycle corrections (Lemma 2 of the design spec). The
    split X/Z syndrome schedule guarantees deterministic individual χ
    outcomes (a structural prerequisite for FT) but does NOT by itself
    close the operator-algebraic obs0 identity.
    """
    from qldpc.circuits.surgery import build_joint_ppm_circuit

    g_l, g_r, bridge = _build_steane_mixed_pair()
    circuit, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=3, data_init=("+", "0"))
    assert isinstance(circuit, stim.Circuit)


def test_mixed_basis_circuit_compiles_to_dem() -> None:
    """Noiseless mixed-basis circuit compiles to a stim.DetectorErrorModel without crash."""
    from qldpc.circuits.surgery import build_joint_ppm_circuit

    g_l, g_r, bridge = _build_steane_mixed_pair()
    circuit, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=3, data_init=("+", "0"))
    dem = circuit.detector_error_model()
    assert dem is not None


@pytest.mark.xfail(
    reason=(
        "obs0 = ⊕ m(χ_l) ⊕ ⊕ m(χ_r) ⊕ ⊕ m(bridge_gauges) is non-deterministic "
        "even with the split X-/Z-phase syndrome schedule (X-ancillas measured "
        "before Z-CZ gates fire). The operator product equals X̄_l ⊗ Z̄_r ⊗ "
        "Z^|B| as an OPERATOR, but this operator anti-commutes with the χ_l "
        "and bridge_gauge X-on-adapter gauge generators themselves (Z on "
        "adapter from ∏ χ_r ↔ X on adapter from individual χ_l rows). Hence "
        "the obs0 product is NOT in the algebraic stabilizer center of the "
        "gauge group, and Stim's detector_error_model correctly rejects it. "
        "Per Cohen-Kim-Bartlett-Brown arXiv:2110.10794 §II.B.2 and the design "
        "spec docs/superpowers/specs/2026-06-15-mixed-basis-joint-ppm-design"
        ".md §9 Lemma 2, the full obs0 formula additionally XORs in (a) Y-stab "
        "rows from the pair-merge step (obs0_xor_map) and (b) leftover "
        "X-cycle / Z-cycle rows (x_leftover_indices / z_leftover_indices). "
        "Tier 2 work: extend _stitch_to_joint_code_mixed to populate these "
        "Bridge fields via the merge.py pair-merge algorithm, then include "
        "them in obs0. The split syndrome schedule (DONE) is a structural "
        "prerequisite for any FT mixed-basis pipeline but does not by itself "
        "make obs0 deterministic — the missing piece is operator-algebraic "
        "(Y-stab / cycle corrections), not scheduling."
    ),
    strict=True,
)
def test_mixed_basis_joint_truth_table_x_l_z_r() -> None:
    """All four joint eigenstates of X̄_l ⊗ Z̄_r give correct obs0 sign noiselessly.

    For basis_l=X (measures X̄_l) and basis_r=Z (measures Z̄_r):
      ("+", "0"): X̄|+⟩=+, Z̄|0⟩=+, product=+1 → obs0=0
      ("−", "0"): X̄|−⟩=−, Z̄|0⟩=+, product=−1 → obs0=1
      ("+", "1"): X̄|+⟩=+, Z̄|1⟩=−, product=−1 → obs0=1
      ("−", "1"): X̄|−⟩=−, Z̄|1⟩=−, product=+1 → obs0=0
    """
    from qldpc.circuits.surgery import build_joint_ppm_circuit, keep_only_observable

    g_l, g_r, bridge = _build_steane_mixed_pair()

    cases = [(("+", "0"), 0), (("-", "0"), 1), (("+", "1"), 1), (("-", "1"), 0)]
    for init, expected in cases:
        circuit, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=3, data_init=init)
        circuit = keep_only_observable(circuit, keep_idx=0)
        sampler = circuit.compile_detector_sampler()
        _, obs = sampler.sample(shots=256, separate_observables=True)
        assert obs.shape[1] == 1, f"expected 1 observable, got {obs.shape[1]}"
        assert (obs[:, 0] == expected).all(), (
            f"init={init} expected obs0={expected}, got mean {obs[:, 0].mean():.3f}"
        )
