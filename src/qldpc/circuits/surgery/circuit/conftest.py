"""Shared fixtures + helpers for the surgery circuit-layer tests.

pytest auto-discovers this conftest for every ``circuit/*_test.py`` sibling, so
the cross-file fixtures (``_steane_joint_fixture``) and plain helpers
(``_data_measured``, ``_bb_36_8_code``) live here instead of being duplicated.

The ``test_memory_experiment_*`` baseline (compares the surgery k+1/k-1 sets
against the public ``get_memory_experiment``) also lives here — it is a
cross-module baseline, not a unit test of any single circuit submodule.
"""

from __future__ import annotations

import numpy as np
import pytest
import stim

from qldpc import codes
from qldpc.objects import Pauli


def _data_measured(circuit: stim.Circuit, n_data: int) -> set[int]:
    """Real-data qubit IDs (< n_data) appearing under any measurement op."""
    return {
        t.qubit_value
        for inst in circuit.flattened()
        if inst.name in ("M", "MX", "MY", "MZ")
        for t in inst.targets_copy()
        if t.is_qubit_target and t.qubit_value < n_data
    }


def _bb_36_8_code() -> object:
    """In-repo BBCode [[36, 8]] (dimension 8) — the k>=2 fixture used elsewhere in
    this file (see test_build_single_ppm_circuit_block_observables_full_k_block)."""
    import sympy

    xs, ys = sympy.symbols("x y")
    return codes.BBCode({xs: 3, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)


@pytest.fixture
def _steane_joint_fixture():
    """Two [[7,1,3]] Steane patches joined by a bridge (basis=X). Returns
    (g_l, g_r, bridge) via the repo's real joint construction path."""
    from qldpc.circuits.surgery.hmatrix.PPM_joint import build_bridge
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget

    c1, c2 = codes.SteaneCode(), codes.SteaneCode()
    x1 = np.asarray(c1.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    x2 = np.asarray(c2.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(c1, x1, basis=Pauli.X)
    g_r = build_gadget(c2, x2, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    return g_l, g_r, bridge
