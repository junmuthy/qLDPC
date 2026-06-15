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

Full truth-table verification (obs0 sign for all four |ψ_l⟩ ⊗ |ψ_r⟩
computational eigenstates) is deferred to Task 6 (obs0 formula
extension), and is not exercised here.
"""

from __future__ import annotations

import numpy as np
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
    """build_joint_ppm_circuit succeeds for mixed-basis input (no CSSCode assertion error)."""
    from qldpc.circuits.surgery import build_joint_ppm_circuit

    g_l, g_r, bridge = _build_steane_mixed_pair()
    circuit, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=3, data_init=("0", "+"))
    assert isinstance(circuit, stim.Circuit)


def test_mixed_basis_circuit_compiles_to_dem() -> None:
    """Noiseless mixed-basis circuit compiles to a stim.DetectorErrorModel without crash."""
    from qldpc.circuits.surgery import build_joint_ppm_circuit

    g_l, g_r, bridge = _build_steane_mixed_pair()
    circuit, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=3, data_init=("0", "+"))
    dem = circuit.detector_error_model()
    assert dem is not None
