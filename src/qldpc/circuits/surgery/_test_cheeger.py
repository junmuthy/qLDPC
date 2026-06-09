"""Tests for src/qldpc/circuits/surgery/cheeger.py (cheeger_constant + boost_gadget)."""

from __future__ import annotations

import numpy as np
import pytest

from qldpc import codes
from qldpc.objects import Pauli

from ._test_helpers import (
    load_webster_seed_set,
    build_generalised_bicycle_code,
    _webster_x_bar_1_operator,
)


def test_cheeger_constant_matches_boost_target():
    """cheeger_constant(g) reports the Webster boundary Cheeger; boost raises it."""
    from qldpc.circuits.surgery import build_gadget, boost_gadget, cheeger_constant
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    h0 = cheeger_constant(g)
    assert h0 >= 0
    # Boosting to a higher target raises h(F) to (at least) that target.
    g_aug = boost_gadget(g, method="combinatorial", target=2.0, max_extra_qubits=30, seed=7)
    h1 = cheeger_constant(g_aug)
    assert h1 >= 2.0 - 1e-9, f"boost to 2.0 produced h={h1}"
    # No-op contract: if h0 already meets target, boost adds no rows.
    g_noop = boost_gadget(g, method="combinatorial", target=h0, max_extra_qubits=30, seed=7)
    assert g_noop.F.shape[0] == g.F.shape[0], "boost to current h should be a no-op"


def test_boost_gadget_dispatches_to_two_methods():
    from qldpc.circuits.surgery.gadget import (
        build_gadget, GadgetLayout,
    )
    from qldpc.circuits.surgery.cheeger import boost_gadget
    # Use Webster code 0 (l=31, k>=2): Steane gadget has dimension 0 (Steane
    # k=1 minus 1 gadget-consumed logical), which causes the BP+OSD decoder
    # used by boost_gadget_distance to hang searching for nonexistent logicals.
    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x = _webster_x_bar_1_operator(data)
    g = build_gadget(code, x)
    for method in ("combinatorial", "distance"):
        out = boost_gadget(g, method=method, target=1.0, seed=42)
        assert isinstance(out, GadgetLayout), f"method={method}"


def test_boost_gadget_seed_reproducible():
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.cheeger import boost_gadget
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    a = boost_gadget(g, method="combinatorial", target=1.0, seed=42)
    b = boost_gadget(g, method="combinatorial", target=1.0, seed=42)
    assert np.array_equal(a.F, b.F)
    assert np.array_equal(a.HX_merged, b.HX_merged)


@pytest.mark.parametrize("method", ["combinatorial", "distance"])
def test_boost_gadget_preserves_css_commutation(method):
    from qldpc.circuits.surgery.gadget import (
        build_gadget,
    )
    from qldpc.circuits.surgery.cheeger import boost_gadget
    # Webster code 0 — Steane causes distance-boost decoder to hang on k=0 merged.
    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x = _webster_x_bar_1_operator(data)
    g = build_gadget(code, x)
    boosted = boost_gadget(g, method=method, target=1.0, seed=0)
    product = (boosted.HX_merged @ boosted.HZ_merged.T) % 2
    assert np.array_equal(product, np.zeros_like(product))


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_boost_gadget_preserves_css_commutation_both_bases(basis):
    """boost_gadget on a basis=X or basis=Z gadget preserves CSS commutation."""
    from qldpc.circuits.surgery.gadget import (
        build_gadget,
    )
    from qldpc.circuits.surgery.cheeger import boost_gadget

    def operator(d, name):
        l = d["l"]
        for seed in d["seeds"]:
            if seed["name"] == name and seed["pauli_type"] == name[0]:
                L = np.zeros(l, dtype=np.uint8); R = np.zeros(l, dtype=np.uint8)
                for i in seed["L_support"]: L[i] = 1
                for i in seed["R_support"]: R[i] = 1
                return np.concatenate([L, R])
        raise ValueError(f"{name} not found")

    d = load_webster_seed_set(0)
    c = build_generalised_bicycle_code(d["l"], d["A"], d["B"])
    op_name = "X_bar_1" if basis is Pauli.X else "Z_bar_1"
    op = operator(d, op_name)
    g = build_gadget(c, op, basis=basis)
    boosted = boost_gadget(g, method="combinatorial", target=1.0, seed=0)
    product = (boosted.HX_merged @ boosted.HZ_merged.T) % 2
    assert np.array_equal(product, np.zeros_like(product))
    assert boosted.basis is basis  # boost preserves basis
