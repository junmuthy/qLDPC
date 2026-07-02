"""Tests for src/qldpc/circuits/surgery/gadget.py."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from qldpc import codes
from qldpc.circuits.surgery.conftest import (
    _webster_x_bar_operator,
    build_generalised_bicycle_code,
    load_webster_seed_set,
)
from qldpc.objects import Pauli

WEBSTER_TABLE_I_ANCILLA_MEAS_COMP = [(0, 19), (1, 31), (2, 49), (3, 79)]


def test_gadget_layout_is_frozen_dataclass() -> None:
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import GadgetLayout

    assert dataclasses.is_dataclass(GadgetLayout)
    # frozen
    fields = {f.name for f in dataclasses.fields(GadgetLayout)}
    assert fields == {
        "code",
        "x",
        "support",
        "data_checks",
        "incidence",
        "partial_0",
        "HX_merged",
        "HZ_merged",
        "Q_prime",
        "basis",
    }
    # Verify actually frozen: mutation must raise. None placeholders are fine here
    # — we only check FrozenInstanceError, never read the fields.
    inst = GadgetLayout(
        code=None,  # type: ignore[arg-type]
        x=None,  # type: ignore[arg-type]
        support=(),
        data_checks=(),
        incidence=None,  # type: ignore[arg-type]
        partial_0=None,  # type: ignore[arg-type]
        HX_merged=None,  # type: ignore[arg-type]
        HZ_merged=None,  # type: ignore[arg-type]
        Q_prime=(),
        basis=Pauli.X,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        inst.code = object()  # type: ignore[misc,assignment]


def test_restrict_steane_x_frame() -> None:
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import _restrict

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    support, data_checks, incidence, _ = _restrict(code.matrix_z, x)
    assert support == tuple(int(i) for i in np.where(x)[0])
    assert list(support) == sorted(support)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    touched = sorted({j for j in range(HZ.shape[0]) for i in support if HZ[j, i] == 1})
    assert data_checks == tuple(touched)
    assert incidence.shape == (len(data_checks), len(support))
    assert np.array_equal(incidence, HZ[np.ix_(data_checks, support)])
    ones = np.ones(len(support), dtype=np.uint8)
    assert np.array_equal((incidence @ ones) % 2, np.zeros(len(data_checks), dtype=np.uint8))


def test_restrict_basis_z_uses_HX() -> None:
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import _restrict

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    support, data_checks, incidence, _ = _restrict(code.matrix_x, z)
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    touched = sorted({j for j in range(HX.shape[0]) for i in support if HX[j, i] == 1})
    assert data_checks == tuple(touched)
    assert np.array_equal(incidence, HX[np.ix_(data_checks, support)])


def test_restrict_rejects_x_shape_mismatch() -> None:
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import _restrict

    code = codes.SteaneCode()
    bad_x = np.ones(code.num_qudits + 1, dtype=np.uint8)
    with pytest.raises(ValueError):
        _restrict(code.matrix_z, bad_x)


def test_build_gadget_css_commutes_and_gauge_kernel() -> None:
    """H̃_X H̃_Z^T = 0; ∂_1 ∂_0^T = 0; rank(∂_0) = |C0| - rank(∂_1)."""
    import galois

    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget

    GF = galois.GF(2)
    for basis in (Pauli.X, Pauli.Z):
        code = codes.SteaneCode()
        x = np.asarray(code.get_logical_ops(basis)[0]).astype(np.uint8)
        g = build_gadget(code, x, basis=basis)
        HX = np.asarray(g.HX_merged).astype(np.uint8)
        HZ = np.asarray(g.HZ_merged).astype(np.uint8)
        assert np.array_equal((HX @ HZ.T) % 2, np.zeros((HX.shape[0], HZ.shape[0]), np.uint8))
        inc = np.asarray(g.incidence).astype(np.uint8)  # ∂_1^T
        p0 = np.asarray(g.partial_0).astype(np.uint8)
        assert np.array_equal((p0 @ inc) % 2, np.zeros((p0.shape[0], inc.shape[1]), np.uint8))
        r_expected = inc.shape[0] - int(np.linalg.matrix_rank(GF(inc.tolist())))
        assert p0.shape[0] == r_expected


def test_build_gadget_basis_z_places_new_x_checks_in_HZ_merged() -> None:
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    m_Z = code.matrix_z.shape[0]
    n = code.num_qudits
    # the new S' rows live below H_Z in HZ_merged with f_1^T = π_{V0} on data
    new_rows = np.asarray(g.HZ_merged).astype(np.uint8)[m_Z:, :n]
    f1T = np.zeros((len(g.support), n), np.uint8)
    f1T[np.arange(len(g.support)), np.array(g.support, np.int_)] = 1
    assert np.array_equal(new_rows, f1T)


def test_build_gadget_product_of_new_x_checks_is_logical() -> None:
    """∏ rows(S_X') = X̄ on data, identity on Q' (Cain et al. arXiv:2603.28627 §B.1)."""
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    n = code.num_qudits
    m_X = code.matrix_x.shape[0]
    new_x_rows = np.asarray(g.HX_merged).astype(np.uint8)[m_X:]
    prod = new_x_rows.sum(axis=0) % 2
    assert np.array_equal(prod[:n], x)
    assert not prod[n:].any()


def test_build_gadget_steane_returns_valid_layout() -> None:
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import GadgetLayout, build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    assert isinstance(g, GadgetLayout)
    assert g.code is code
    assert np.array_equal(g.x, x)
    # κ qubits indexed contiguously after data qubits
    assert g.Q_prime == tuple(range(code.num_qudits, code.num_qudits + len(g.data_checks)))


def test_build_gadget_deterministic() -> None:
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g1 = build_gadget(code, x, basis=Pauli.X)
    g2 = build_gadget(code, x, basis=Pauli.X)
    assert g1.support == g2.support
    assert g1.data_checks == g2.data_checks
    assert np.array_equal(g1.incidence, g2.incidence)
    assert np.array_equal(g1.partial_0, g2.partial_0)
    assert np.array_equal(g1.HX_merged, g2.HX_merged)
    assert np.array_equal(g1.HZ_merged, g2.HZ_merged)
    assert g1.Q_prime == g2.Q_prime


def test_build_gadget_rejects_non_x_logical() -> None:
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget

    code = codes.SteaneCode()
    x = np.zeros(code.num_qudits, dtype=np.uint8)
    x[0] = 1  # not a logical X (HZ @ x ≠ 0 in general)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    if ((HZ @ x) % 2).any():
        with pytest.raises(ValueError, match="logical"):
            build_gadget(code, x, basis=Pauli.X)


@pytest.mark.parametrize("code_index,n_anc", WEBSTER_TABLE_I_ANCILLA_MEAS_COMP)
def test_webster_table_i_ancilla_meas_comp_exact(code_index: int, n_anc: int) -> None:
    """Webster Table I in Cain notation: |Q'| + |S'_meas| + |S'_comp| matches
    each of the 4 generalised-bicycle codes. Reproduces Webster Table I exactly."""
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import (
        build_gadget,
    )

    data = load_webster_seed_set(code_index)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x1 = _webster_x_bar_operator(data)
    g1 = build_gadget(code, x1, basis=Pauli.X, minimal_z_checks=False)
    n_ancilla = len(g1.Q_prime)
    n_meas_checks = int(g1.x.sum())  # |support|
    n_comp_checks = g1.partial_0.shape[0]
    assert n_ancilla + n_meas_checks + n_comp_checks == n_anc, (
        f"code {code_index}: |Q'|={n_ancilla}, |S'_meas|={n_meas_checks}, |S'_comp|={n_comp_checks}, "
        f"sum={n_ancilla + n_meas_checks + n_comp_checks}, expected {n_anc}"
    )


def test_build_gadget_basis_is_required() -> None:
    """basis has no default: a CSS code's X-logical and Z-logical can coincide
    (e.g. self-dual Steane), so the caller must declare intent explicitly.
    """
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    with pytest.raises(TypeError, match="basis"):
        build_gadget(code, x)  # type: ignore[call-arg]


def test_build_gadget_z_basis_css_commutation() -> None:
    """build_gadget(code, z_logical, basis=Pauli.Z) yields a CSS-commuting merged code."""
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    assert g.basis is Pauli.Z
    product = (g.HX_merged @ g.HZ_merged.T) % 2
    assert np.array_equal(product, np.zeros_like(product))


def test_build_gadget_z_basis_dual_matches_x_basis_on_dual_code() -> None:
    """basis-symmetric invariant: build_gadget(code, z, basis=Z) gives the same
    merged matrices as build_gadget(dual_code, z, basis=X), where dual_code has
    HX/HZ swapped. The swap labels swap too, so we compare HX_z vs HZ_dx_x and
    HZ_z vs HX_dx_x."""
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget
    from qldpc.codes.common import CSSCode

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_z = build_gadget(code, z, basis=Pauli.Z)
    # Dual code: swap matrix_x and matrix_z
    dual = CSSCode(
        np.asarray(code.matrix_z).astype(np.int_),
        np.asarray(code.matrix_x).astype(np.int_),
        is_subsystem_code=False,
    )
    g_dual = build_gadget(dual, z, basis=Pauli.X)
    # In the dual construction, the basis-X chi rows end up in dual.HX_merged
    # which corresponds to original.HZ_merged in the basis-Z construction.
    assert np.array_equal(g_z.HZ_merged, g_dual.HX_merged), (
        "basis-Z chi (in HZ_merged) should equal basis-X chi (in HX_merged) on dual"
    )
    assert np.array_equal(g_z.HX_merged, g_dual.HZ_merged), (
        "basis-Z gauge-fix (in HX_merged) should equal basis-X gauge-fix (in HZ_merged) on dual"
    )


def test_webster_table_i_z_basis_ancilla_meas_comp_exact() -> None:
    """Webster Z̄_1 seed in Cain notation: |Q'| + |S'_meas| + |S'_comp| matches
    (basis-symmetric dual; reproduces Webster Table I)."""
    from qldpc.circuits.surgery.conftest import _webster_z_bar_operator
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import (
        build_gadget,
    )

    for code_index, expected in [(0, 19), (1, 31), (2, 49), (3, 79)]:
        d = load_webster_seed_set(code_index)
        c = build_generalised_bicycle_code(d["l"], d["A"], d["B"])
        z = _webster_z_bar_operator(d)
        g = build_gadget(c, z, basis=Pauli.Z, minimal_z_checks=False)
        n_ancilla = len(g.Q_prime)
        n_meas_checks = len(g.support)
        n_comp_checks = g.partial_0.shape[0]
        assert n_ancilla + n_meas_checks + n_comp_checks == expected, (
            f"code {code_index}: Z-basis got |Q'|+|S'_meas|+|S'_comp|={n_ancilla + n_meas_checks + n_comp_checks}, expected {expected}"
        )


def test_build_gadget_augmented_extends_incidence_and_recomputes_gauge() -> None:
    """Augmenting with one weight-2 row adds a column to merged matrices and recomputes G."""
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget, build_gadget_augmented

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    # Pick two ports in V_0; create one extra weight-2 row connecting them
    support_a, support_b = g.support[0], g.support[1]
    extra_incidence = np.zeros((1, len(g.support)), dtype=np.uint8)
    idx_a = g.support.index(support_a)
    idx_b = g.support.index(support_b)
    extra_incidence[0, idx_a] = 1
    extra_incidence[0, idx_b] = 1
    g_aug = build_gadget_augmented(code, x, extra_incidence, basis=Pauli.X)

    # incidence_aug = [incidence | extra_incidence] vertically stacked
    assert g_aug.incidence.shape == (g.incidence.shape[0] + 1, g.incidence.shape[1])
    assert np.array_equal(g_aug.incidence[: g.incidence.shape[0]], g.incidence)
    assert np.array_equal(g_aug.incidence[g.incidence.shape[0] :], extra_incidence)
    # HX_merged has one extra column (one extra κ qubit); same number of rows
    assert g_aug.HX_merged.shape == (g.HX_merged.shape[0], g.HX_merged.shape[1] + 1)
    # CSS commutation
    product = (g_aug.HX_merged @ g_aug.HZ_merged.T) % 2
    assert np.array_equal(product, np.zeros_like(product))


def test_build_gadget_rejects_non_logical_input() -> None:
    """build_gadget rejects x that isn't a logical operator support.

    For basis=X: HZ @ x must be 0; for basis=Z: HX @ x must be 0. Single-qubit
    support [1,0,0,...] generally violates both (it's not in the codespace).
    """
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget

    code = codes.SteaneCode()
    bad = np.zeros(code.num_qudits, dtype=np.uint8)
    bad[0] = 1  # Single qubit support — not a logical operator on Steane.
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    assert ((HZ @ bad) % 2).any(), "fixture broken: single-qubit support should not be X-logical"
    assert ((HX @ bad) % 2).any(), "fixture broken: single-qubit support should not be Z-logical"
    with pytest.raises(ValueError, match="logical-X"):
        build_gadget(code, bad, basis=Pauli.X)
    with pytest.raises(ValueError, match="logical-Z"):
        build_gadget(code, bad, basis=Pauli.Z)


def test_build_gadget_rejects_invalid_basis() -> None:
    """build_gadget raises on basis that isn't Pauli.X or Pauli.Z."""
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    with pytest.raises(ValueError, match="basis must be"):
        build_gadget(code, x, basis=Pauli.Y)  # type: ignore[arg-type]


def test_build_gadget_augmented_rejects_wrong_width() -> None:
    """build_gadget_augmented rejects incidence_extra with wrong column count."""
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget_augmented

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    # support has 3 columns (Steane X-logical weight 3); pass 2-column incidence_extra.
    bad_extra = np.array([[1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="columns"):
        build_gadget_augmented(code, x, bad_extra, basis=Pauli.X)


def test_build_gadget_augmented_rejects_non_weight_2_rows() -> None:
    """build_gadget_augmented rejects incidence_extra rows with weight != 2."""
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget_augmented

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    # Width 3 (Steane X-logical), but a row with weight 1 (not 2)
    bad_extra = np.array([[1, 0, 0]], dtype=np.uint8)
    with pytest.raises(ValueError, match="weight"):
        build_gadget_augmented(code, x, bad_extra, basis=Pauli.X)


def test_x_merged_matches_legacy_build_gadget_x_frame() -> None:
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import _x_merged, build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    sup, dc, inc, p0, HX, HZ = _x_merged(code.matrix_x, code.matrix_z, x)
    g = build_gadget(code, x, basis=Pauli.X)
    assert sup == g.support and dc == g.data_checks
    assert np.array_equal(inc, g.incidence)
    assert np.array_equal(p0, g.partial_0)
    assert np.array_equal(HX, g.HX_merged)
    assert np.array_equal(HZ, g.HZ_merged)


# ---------------------------------------------------------------------------
# minimize_z_checks — opt-in removal of redundant cycle Z-checks
# (Cross, He, Rall, Yoder arXiv:2407.18393 Eq.(6): a cycle Z-check lying in
# rowspan([H_Z | f_0]) is already implied by the deformed original checks and
# can be dropped without changing the stabilizer group).
# ---------------------------------------------------------------------------


def _gf2_rank(matrix: np.ndarray) -> int:
    import galois

    arr = np.asarray(matrix).astype(np.uint8) % 2
    return 0 if arr.shape[0] == 0 else int(galois.GF(2)(arr).row_space().shape[0])


def _gross_x0_gadget():
    """Un-boosted gadget measuring X̄_0 of gross [[144,12,12]] (dim U = 6)."""
    import sympy

    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget

    xs, ys = sympy.symbols("x y")
    gross = codes.BBCode((12, 6), xs**3 + ys + ys**2, ys**3 + xs + xs**2)
    x = np.asarray(gross.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    # full basis (minimal_z_checks=False) so minimize_z_checks has redundancy to drop
    return build_gadget(gross, x, basis=Pauli.X, minimal_z_checks=False)


def test_minimize_z_checks_drops_redundant_rows() -> None:
    """Redundant cycle Z-checks are removed; every survivor stays independent."""
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import minimize_z_checks

    g = _gross_x0_gadget()
    g_min = minimize_z_checks(g)

    # This code has redundant cycles, so the count strictly drops.
    assert g_min.partial_0.shape[0] < g.partial_0.shape[0]

    # No redundant row remains: cycle rows are independent of each other AND of
    # the deformed original checks [H_Z | f_0] (the first n_Zorig rows).
    n_orig = g.code.matrix_z.shape[0]
    hz = np.asarray(g_min.HZ_merged)
    top, cyc = hz[:n_orig], hz[n_orig:]
    assert cyc.shape[0] == g_min.partial_0.shape[0]
    assert _gf2_rank(hz) == _gf2_rank(top) + cyc.shape[0]


def test_minimize_z_checks_preserves_stabilizer_group() -> None:
    """The merged Z-stabilizer group (rowspan of HZ_merged) is unchanged."""
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import minimize_z_checks

    g = _gross_x0_gadget()
    g_min = minimize_z_checks(g)

    full = np.asarray(g.HZ_merged)
    trimmed = np.asarray(g_min.HZ_merged)
    assert _gf2_rank(full) == _gf2_rank(trimmed) == _gf2_rank(np.vstack([full, trimmed]))


def test_minimize_z_checks_preserves_edges_and_x_side() -> None:
    """Only the Z-check block changes: qubits (edges) and X-checks are untouched."""
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import minimize_z_checks

    g = _gross_x0_gadget()
    g_min = minimize_z_checks(g)

    assert g_min.support == g.support
    assert g_min.Q_prime == g.Q_prime
    assert g_min.data_checks == g.data_checks
    assert np.array_equal(g_min.incidence, g.incidence)
    assert np.array_equal(g_min.HX_merged, g.HX_merged)
    assert np.array_equal(g_min.x, g.x)
    assert g_min.basis is g.basis
    assert g_min.code is g.code


def test_minimize_z_checks_is_noop_when_already_minimal() -> None:
    """Applying twice equals applying once (already-minimal gadget is a fixpoint)."""
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import minimize_z_checks

    g = _gross_x0_gadget()
    once = minimize_z_checks(g)
    twice = minimize_z_checks(once)
    assert twice.partial_0.shape[0] == once.partial_0.shape[0]
    assert np.array_equal(np.asarray(twice.HZ_merged), np.asarray(once.HZ_merged))


def test_minimize_z_checks_does_not_mutate_input() -> None:
    """The input gadget is left untouched (pure function)."""
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import minimize_z_checks

    g = _gross_x0_gadget()
    before = np.asarray(g.partial_0).copy()
    _ = minimize_z_checks(g)
    assert np.array_equal(np.asarray(g.partial_0), before)


def test_minimize_z_checks_handles_z_gadget_dual() -> None:
    """For a Z̄ gadget the cycle checks live in HX_merged; still trimmed correctly."""
    import sympy

    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget, minimize_z_checks

    xs, ys = sympy.symbols("x y")
    gross = codes.BBCode((12, 6), xs**3 + ys + ys**2, ys**3 + xs + xs**2)
    z = np.asarray(gross.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(gross, z, basis=Pauli.Z, minimal_z_checks=False)
    g_min = minimize_z_checks(g)

    # Cycle checks are X-type here → they live in HX_merged; HZ_merged is the
    # opposite (untouched) side.
    full = np.asarray(g.HX_merged)
    trimmed = np.asarray(g_min.HX_merged)
    assert _gf2_rank(full) == _gf2_rank(trimmed) == _gf2_rank(np.vstack([full, trimmed]))

    n_comp = g.code.matrix_x.shape[0]
    top, cyc = trimmed[:n_comp], trimmed[n_comp:]
    assert cyc.shape[0] == g_min.partial_0.shape[0] <= g.partial_0.shape[0]
    assert _gf2_rank(trimmed) == _gf2_rank(top) + cyc.shape[0]  # no redundant remains
    assert np.array_equal(g_min.HZ_merged, g.HZ_merged)  # opposite side untouched
    assert g_min.Q_prime == g.Q_prime


def test_build_gadget_minimizes_z_checks_by_default() -> None:
    """build_gadget default drops redundant cycle checks (covers the no-boost path)."""
    import sympy

    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget, minimize_z_checks

    xs, ys = sympy.symbols("x y")
    gross = codes.BBCode((12, 6), xs**3 + ys + ys**2, ys**3 + xs + xs**2)
    x = np.asarray(gross.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    default = build_gadget(gross, x, basis=Pauli.X)
    full = build_gadget(gross, x, basis=Pauli.X, minimal_z_checks=False)
    assert default.partial_0.shape[0] < full.partial_0.shape[0]
    # default is already minimal → applying minimize again is a no-op
    assert minimize_z_checks(default).partial_0.shape[0] == default.partial_0.shape[0]


def test_boost_gadget_minimizes_z_checks_by_default() -> None:
    """boost_gadget re-applies minimize after recomputing the full cycle basis."""
    import sympy

    from qldpc.circuits.surgery import boost_gadget, build_gadget

    xs, ys = sympy.symbols("x y")
    gross = codes.BBCode((12, 6), xs**3 + ys + ys**2, ys**3 + xs + xs**2)
    x = np.asarray(gross.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g0 = build_gadget(gross, x, basis=Pauli.X, minimal_z_checks=False)
    boost_kw = dict(method="combinatorial", target=1.0, max_extra_qubits=30, seed=3)
    default = boost_gadget(g0, **boost_kw)
    full = boost_gadget(g0, minimal_z_checks=False, **boost_kw)
    assert default.partial_0.shape[0] < full.partial_0.shape[0]
    # boost preserves the Cheeger boost (edges/incidence identical, only Z-checks trimmed)
    assert np.array_equal(default.incidence, full.incidence)
    assert default.Q_prime == full.Q_prime
