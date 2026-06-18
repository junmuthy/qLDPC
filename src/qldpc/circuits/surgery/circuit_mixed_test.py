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
        "obs0 emission per main.tex §4.5 Eq. eq:obs0-corrected is wired in "
        "_build_joint_ppm_circuit_mixed_basis with |Y_+⟩^⊗w bridge prep (RY), "
        "Y-basis bridge detach (MY), and ∏ Y_{a_q} XOR'd into obs0 (Cohen-"
        "Kim-Bartlett-Brown arXiv:2110.10794 §II.B.2 / Fig. 4 |Y⟩ ancilla "
        "protocol). Formula: obs0 = ⊕ m(χ_l surviving) ⊕ ⊕ m(χ_r surviving) "
        "⊕ ⊕ m(y_q) ⊕ ⊕ m_Y(a_q). With these fixes obs0 is now deterministic "
        "shot-to-shot (DEM compiles, obs0=0 for ALL data inits). BUT Steane × "
        "Steane is the DEGENERATE fixture where every V_0 vertex is a port "
        "(|V_0^l| = |V_0^r| = w = 3), so rows_chi['l'] = rows_chi['r'] = (). "
        "The remaining ⊕ m(y_q) ⊕ ⊕ m_Y(a_q) terms are both stabilizers of "
        "the joint-code initial state and yield +1 deterministically — they "
        "do NOT depend on the data prep. The data-dependent X̄_l ⊗ Z̄_r sign "
        "lives in the surviving χ rows (main.tex §4.5 Eq. eq:obs0-corrected "
        "requires non-empty surviving χ products on both sides). To un-xfail, "
        "replace the Steane fixture with a non-degenerate code where |V_0| > w "
        "(so surviving χ rows carry X̄_l, Z̄_r); see "
        "docs/superpowers/plans/2026-06-18-joint-ppm-layout-refactor.md "
        "§Task 13 follow-up."
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


def _raw_observables(circuit, shots):
    """Raw ±1 obs values via compile_sampler + manual XOR.

    Truth-table sign checks REQUIRE this API — detector_sampler.sample(
    separate_observables=True) returns flips relative to the noiseless
    reference (always 0 on noiseless run), hiding the sign. See
    examples/lattice_surgery.ipynb §0 ``raw_observables`` for the
    canonical helper this mirrors.
    """
    import numpy as np
    sampler = circuit.compile_sampler()
    raw = sampler.sample(shots=shots).astype(np.uint8)
    n_meas = raw.shape[1]
    obs_lines = [ln for ln in str(circuit).splitlines() if ln.startswith("OBSERVABLE_INCLUDE")]
    cols = []
    for line in obs_lines:
        offsets = [int(t.strip("rec[]")) for t in line.split() if t.startswith("rec[")]
        meas_idx = [n_meas + off for off in offsets]
        cols.append(np.bitwise_xor.reduce(raw[:, meas_idx], axis=1))
    return np.stack(cols, axis=1)


def test_mixed_basis_joint_truth_table_bb_xz() -> None:
    """Mixed-basis joint X̄_l ⊗ Z̄_r noiseless truth table on BB [[72,12]].

    This test uses a NON-DEGENERATE fixture (|V_0^l|=8 > w=4 < |V_0^r|=14)
    so the cross-merge leaves surviving χ rows on both sides. The Y-basis
    bridge protocol (main.tex §4.5 Eq. eq:obs0-corrected) + provenance-
    driven obs0 should then give the correct joint eigenvalue.

    Per the project convention this verifies via ``_raw_observables`` —
    NOT ``detector_sampler.sample(separate_observables=True)``, which
    returns flips relative to the noiseless reference (always 0 in a
    noiseless run, hiding the sign).
    """
    import sympy
    from qldpc.circuits.surgery import (
        build_bridge,
        build_gadget,
        build_joint_ppm_circuit,
        keep_only_observable,
        logical_state_init,
    )

    xs, ys = sympy.symbols("x y")
    c_l = codes.BBCode({xs: 6, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)
    c_r = codes.BBCode({xs: 6, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)
    x_l = np.asarray(c_l.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z_r = np.asarray(c_r.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(c_l, x_l, basis=Pauli.X)
    g_r = build_gadget(c_r, z_r, basis=Pauli.Z)
    bridge = build_bridge(
        g_l, g_r, port_subset_l=tuple(range(4)), port_subset_r=tuple(range(4)),
    )

    for s_l, s_r, expected in [
        ("+", "0", 0),
        ("-", "0", 1),
        ("+", "1", 1),
        ("-", "1", 0),
    ]:
        init_l = logical_state_init(c_l, s_l, log_idx=0)
        init_r = logical_state_init(c_r, s_r, log_idx=0)
        circuit, _ = build_joint_ppm_circuit(
            g_l, g_r, bridge, rounds=3, noise_model=None,
            data_init=(init_l, init_r),
        )
        circuit = keep_only_observable(circuit, keep_idx=0)
        obs = _raw_observables(circuit, shots=32)
        assert obs.shape[1] == 1, f"expected 1 observable, got {obs.shape[1]}"
        actual_bits = obs[:, 0]
        # In a noiseless run the eigenvalue is deterministic shot-to-shot.
        assert (actual_bits == expected).all(), (
            f"init=({s_l},{s_r}) expected obs0={expected}, got "
            f"bits {np.bincount(actual_bits, minlength=2).tolist()}"
        )
