"""Tests for the simplified surgery package (see
docs/superpowers/specs/2026-06-07-surgery-simplification-design.md)."""

from __future__ import annotations

import dataclasses
import numpy as np
import pytest

from qldpc import codes
from qldpc.objects import Pauli


def test_gadget_layout_is_frozen_dataclass():
    from qldpc.codes.surgery.gadget import GadgetLayout
    assert dataclasses.is_dataclass(GadgetLayout)
    # frozen
    fields = {f.name for f in dataclasses.fields(GadgetLayout)}
    assert fields == {
        "code", "x", "V0", "C0", "F", "G",
        "HX_merged", "HZ_merged", "kappa_qubits",
    }
    # Verify actually frozen: mutation must raise
    inst = GadgetLayout(
        code=None, x=None, V0=(), C0=(),
        F=None, G=None, HX_merged=None, HZ_merged=None,
        kappa_qubits=(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        inst.code = object()


def test_step1_restriction_steane():
    from qldpc.codes.surgery.gadget import _step1_restriction
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    V0, C0, F = _step1_restriction(code, x)
    # V_0 = supp(x), sorted ascending
    assert V0 == tuple(int(i) for i in np.where(x)[0])
    assert list(V0) == sorted(V0)
    # C_0 = Z-checks touching V_0, sorted ascending
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    touched = sorted({j for j in range(HZ.shape[0])
                      for i in V0 if HZ[j, i] == 1})
    assert C0 == tuple(touched)
    assert list(C0) == sorted(C0)
    # F = H_Z[C_0, V_0]
    assert F.shape == (len(C0), len(V0))
    assert np.array_equal(F, HZ[np.ix_(C0, V0)])
    # F @ 1_{V0} == 0 (math.md §1.1 invariant)
    ones = np.ones(len(V0), dtype=np.uint8)
    assert np.array_equal((F @ ones) % 2, np.zeros(len(C0), dtype=np.uint8))


def test_step2_gauge_fix_basis_property():
    from qldpc.codes.surgery.gadget import _step1_restriction, _step2_gauge_fix
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    _, _, F = _step1_restriction(code, x)
    G = _step2_gauge_fix(F)
    # math.md §1.2: G F = 0 over GF(2)
    assert G.shape[1] == F.shape[0]
    GF = (G @ F) % 2
    assert np.array_equal(GF, np.zeros_like(GF))
    # rank(G) = |C_0| - rank(F)
    import galois
    r_expected = F.shape[0] - int(np.linalg.matrix_rank(galois.GF(2)(F.tolist())))
    assert G.shape[0] == r_expected


def test_step2_gauge_fix_deterministic():
    """Same F twice → byte-identical G."""
    from qldpc.codes.surgery.gadget import _step2_gauge_fix
    F = np.array([[1, 0, 1, 1], [0, 1, 1, 1]], dtype=np.uint8)
    G1 = _step2_gauge_fix(F)
    G2 = _step2_gauge_fix(F)
    assert np.array_equal(G1, G2)
