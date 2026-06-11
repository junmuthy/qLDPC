"""Tests for src/qldpc/circuits/surgery/gadget.py."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from qldpc import codes
from qldpc.objects import Pauli

from ._test_helpers import (
    load_webster_seed_set,
    build_generalised_bicycle_code,
    _webster_x_bar_1_operator,
    _webster_x_bar_operator,
)

WEBSTER_TABLE_I_KAPPA_CHI_R = [(0, 19), (1, 31), (2, 49), (3, 79)]


def test_gadget_layout_is_frozen_dataclass():
    from qldpc.circuits.surgery.gadget import GadgetLayout
    assert dataclasses.is_dataclass(GadgetLayout)
    # frozen
    fields = {f.name for f in dataclasses.fields(GadgetLayout)}
    assert fields == {
        "code", "x", "support", "C0", "F", "G",
        "HX_merged", "HZ_merged", "kappa_qubits", "basis",
    }
    # Verify actually frozen: mutation must raise
    inst = GadgetLayout(
        code=None, x=None, support=(), C0=(),
        F=None, G=None, HX_merged=None, HZ_merged=None,
        kappa_qubits=(), basis=Pauli.X,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        inst.code = object()


def test_step1_restriction_steane():
    from qldpc.circuits.surgery.gadget import _step1_restriction
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    support, C0, F = _step1_restriction(code, x)
    # V_0 = supp(x), sorted ascending
    assert support == tuple(int(i) for i in np.where(x)[0])
    assert list(support) == sorted(support)
    # C_0 = Z-checks touching V_0, sorted ascending
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    touched = sorted({j for j in range(HZ.shape[0])
                      for i in support if HZ[j, i] == 1})
    assert C0 == tuple(touched)
    assert list(C0) == sorted(C0)
    # F = H_Z[C_0, V_0]
    assert F.shape == (len(C0), len(support))
    assert np.array_equal(F, HZ[np.ix_(C0, support)])
    # F @ 1_{V0} == 0 (Webster §II.A step 1 invariant)
    ones = np.ones(len(support), dtype=np.uint8)
    assert np.array_equal((F @ ones) % 2, np.zeros(len(C0), dtype=np.uint8))


def test_step2_gauge_fix_basis_property():
    from qldpc.circuits.surgery.gadget import _step1_restriction, _step2_gauge_fix
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    _, _, F = _step1_restriction(code, x)
    G = _step2_gauge_fix(F)
    # Webster §II.A step 2: G F = 0 over GF(2)
    assert G.shape[1] == F.shape[0]
    GF = (G @ F) % 2
    assert np.array_equal(GF, np.zeros_like(GF))
    # rank(G) = |C_0| - rank(F)
    import galois
    r_expected = F.shape[0] - int(np.linalg.matrix_rank(galois.GF(2)(F.tolist())))
    assert G.shape[0] == r_expected


def test_step2_gauge_fix_deterministic():
    """Same F twice → byte-identical G (non-trivial: rank-deficient F → non-empty G)."""
    from qldpc.circuits.surgery.gadget import _step2_gauge_fix
    # 3x3 matrix with rank 2 (row 0 + row 1 = row 2 over GF(2)), so G has 1 row.
    F = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.uint8)
    G1 = _step2_gauge_fix(F)
    G2 = _step2_gauge_fix(F)
    assert G1.shape == (1, 3), f"expected G shape (1,3), got {G1.shape}"
    assert np.array_equal(G1, G2)
    # And sanity-check the basis property holds on this F too.
    assert np.array_equal((G1 @ F) % 2, np.zeros((1, F.shape[1]), dtype=np.uint8))


def test_step3_assemble_basis_z_places_chi_in_HZ_merged_and_G_in_HX_merged():
    """basis=Pauli.Z: χ rows added to HZ_merged (Z-type); G added to HX_merged (X-type)."""
    from qldpc.circuits.surgery.gadget import (
        _step1_restriction, _step2_gauge_fix, _step3_assemble,
    )
    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    support, C0, F = _step1_restriction(code, z, basis=Pauli.Z)
    G = _step2_gauge_fix(F)
    HX_m, HZ_m = _step3_assemble(code, support, C0, F, G, basis=Pauli.Z)

    n, mX, mZ = code.num_qudits, code.matrix_x.shape[0], code.matrix_z.shape[0]
    # For basis=Z: HX_merged grows by r rows (gauge-fix), HZ_merged by |V_0| rows (chi).
    assert HX_m.shape == (mX + G.shape[0], n + len(C0)), f"HX shape {HX_m.shape}"
    assert HZ_m.shape == (mZ + len(support), n + len(C0)), f"HZ shape {HZ_m.shape}"
    # CSS commutation
    product = (HX_m @ HZ_m.T) % 2
    assert np.array_equal(product, np.zeros_like(product))


def test_step3_assemble_steane_css_commutes():
    from qldpc.circuits.surgery.gadget import (
        _step1_restriction, _step2_gauge_fix, _step3_assemble,
    )
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    support, C0, F = _step1_restriction(code, x)
    G = _step2_gauge_fix(F)
    HX_m, HZ_m = _step3_assemble(code, support, C0, F, G)

    n, mX, mZ = code.num_qudits, code.matrix_x.shape[0], code.matrix_z.shape[0]
    assert HX_m.shape == (mX + len(support), n + len(C0))
    assert HZ_m.shape == (mZ + G.shape[0], n + len(C0))
    # Webster §II.A: H_X^merged @ H_Z^merged.T == 0 over GF(2) (CSS commutation)
    product = (HX_m @ HZ_m.T) % 2
    assert np.array_equal(product, np.zeros_like(product))


def test_step3_assemble_csscode_with_distinct_nV_nC():
    """Synthetic CSS code where nV != nC — catches F_tilde shape bug.

    Uses a 5-qubit CSS code with k=1, picking a logical-X representative
    whose support size (nV=4) differs from the number of Z-checks it
    touches (nC=2). With the buggy F_tilde[j] = F[k] form, numpy raises
    ValueError because F[k] has shape (nV=4,) but the row width is (nC=2).
    The fix (F_tilde[j, k] = 1) is the correct indicator/selection matrix.

    Verifies:
    1. CSS commutation: HX_merged @ HZ_merged.T == 0 over GF(2).
    2. Indicator form: each Z-check in C_0 attaches to EXACTLY ONE kappa
       ancilla (row-sum == 1 in the kappa block).
    """
    from qldpc.circuits.surgery.gadget import (
        _step1_restriction, _step2_gauge_fix, _step3_assemble,
    )

    # 5-qubit CSS code (k=1):
    #   HX = [[1,1,1,0,0],[0,0,0,1,1]]
    #   HZ = [[1,1,0,0,0],[1,0,1,0,0]]
    # Commutativity check (each pair of rows):
    #   row0(HX)·row0(HZ) = 1+1+0+0+0 = 0 mod 2 ✓
    #   row0(HX)·row1(HZ) = 1+0+1+0+0 = 0 mod 2 ✓
    #   row1(HX)·row0(HZ) = 0+0+0+0+0 = 0 mod 2 ✓
    #   row1(HX)·row1(HZ) = 0+0+0+0+0 = 0 mod 2 ✓
    HX_raw = np.array([[1, 1, 1, 0, 0],
                        [0, 0, 0, 1, 1]], dtype=np.uint8)
    HZ_raw = np.array([[1, 1, 0, 0, 0],
                        [1, 0, 1, 0, 0]], dtype=np.uint8)
    assert np.array_equal((HX_raw @ HZ_raw.T) % 2,
                           np.zeros((2, 2), dtype=np.uint8)), "CSS sanity failed"

    code = codes.CSSCode(HX_raw, HZ_raw)

    # Logical X rep: x = [1,1,1,1,0].
    #   HZ @ x = [1+1+0,1+0+1] = [0,0] mod 2  =>  x in ker(HZ) ✓
    #   row(HX) = span{[1,1,1,0,0],[0,0,0,1,1]}: cannot produce [1,1,1,1,0]
    #   because the last coord would require b=0 while 4th coord requires b=1 ✓ logical
    x_logical = np.array([1, 1, 1, 1, 0], dtype=np.uint8)
    assert np.array_equal((HZ_raw @ x_logical) % 2,
                           np.zeros(2, dtype=np.uint8)), "x_logical not in ker(HZ)"

    support, C0, F = _step1_restriction(code, x_logical)
    # V0 = {0,1,2,3} (nV=4); HZ row0 touches {0,1}, HZ row1 touches {0,2} -> C0=(0,1) (nC=2)
    assert len(support) != len(C0), (
        f"nV={len(support)} == nC={len(C0)}: this test requires nV != nC to catch the bug"
    )

    G = _step2_gauge_fix(F)
    HX_m, HZ_m = _step3_assemble(code, support, C0, F, G)

    # 1. CSS commutation
    product = (HX_m @ HZ_m.T) % 2
    assert np.array_equal(product, np.zeros_like(product)), (
        "CSS commutation failed: HX_merged @ HZ_merged.T != 0"
    )

    # 2. Indicator form: each Z-check j in C_0 should attach to exactly
    #    one kappa ancilla (column-slice after n data qubits in HZ_merged).
    n = code.num_qudits
    mZ = HZ_raw.shape[0]
    HZ_kappa_block = HZ_m[:mZ, n:]
    for k, j in enumerate(C0):
        row_sum = int(HZ_kappa_block[j].sum())
        assert row_sum == 1, (
            f"row j={j} of HZ kappa-block should have exactly 1 one (indicator form), "
            f"got {row_sum} — F_tilde indicator form violated"
        )


def test_build_gadget_steane_returns_valid_layout():
    from qldpc.circuits.surgery.gadget import build_gadget, GadgetLayout
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    assert isinstance(g, GadgetLayout)
    assert g.code is code
    assert np.array_equal(g.x, x)
    # κ qubits indexed contiguously after data qubits
    assert g.kappa_qubits == tuple(range(code.num_qudits, code.num_qudits + len(g.C0)))


def test_build_gadget_deterministic():
    from qldpc.circuits.surgery.gadget import build_gadget
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g1 = build_gadget(code, x, basis=Pauli.X)
    g2 = build_gadget(code, x, basis=Pauli.X)
    assert g1.support == g2.support
    assert g1.C0 == g2.C0
    assert np.array_equal(g1.F, g2.F)
    assert np.array_equal(g1.G, g2.G)
    assert np.array_equal(g1.HX_merged, g2.HX_merged)
    assert np.array_equal(g1.HZ_merged, g2.HZ_merged)
    assert g1.kappa_qubits == g2.kappa_qubits


def test_build_gadget_rejects_non_x_logical():
    from qldpc.circuits.surgery.gadget import build_gadget
    code = codes.SteaneCode()
    x = np.zeros(code.num_qudits, dtype=np.uint8)
    x[0] = 1  # not a logical X (HZ @ x ≠ 0 in general)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    if ((HZ @ x) % 2).any():
        with pytest.raises(ValueError, match="logical"):
            build_gadget(code, x, basis=Pauli.X)


def test_load_webster_seed_set_returns_known_shape():
    data = load_webster_seed_set(0)
    assert "l" in data and "A" in data and "B" in data
    assert "seeds" in data


def test_build_generalised_bicycle_code_constructs_css():
    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    assert code.num_qudits == 2 * data["l"]
    # CSS commutation
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    assert np.array_equal((HX @ HZ.T) % 2, np.zeros((HX.shape[0], HZ.shape[0]), dtype=np.uint8))


@pytest.mark.parametrize("code_index,n_anc", WEBSTER_TABLE_I_KAPPA_CHI_R)
def test_webster_table_i_kappa_chi_r_exact(code_index, n_anc):
    """Webster Table I: κ + χ + r matches for each of the 4 codes."""
    from qldpc.circuits.surgery.gadget import (
        build_gadget,
    )
    data = load_webster_seed_set(code_index)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x1 = _webster_x_bar_1_operator(data)
    g1 = build_gadget(code, x1, basis=Pauli.X)
    kappa = len(g1.kappa_qubits)
    chi = int(g1.x.sum())  # |V_0|
    r = g1.G.shape[0]
    assert kappa + chi + r == n_anc, (
        f"code {code_index}: κ={kappa}, χ={chi}, r={r}, "
        f"sum={kappa+chi+r}, expected {n_anc}"
    )


def test_build_gadget_basis_is_required():
    """basis has no default: a CSS code's X-logical and Z-logical can coincide
    (e.g. self-dual Steane), so the caller must declare intent explicitly.
    """
    from qldpc.circuits.surgery.gadget import build_gadget
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    with pytest.raises(TypeError, match="basis"):
        build_gadget(code, x)  # type: ignore[call-arg]


def test_step1_restriction_basis_z_uses_HX():
    """For basis=Pauli.Z, F = H_X[C_0, V_0] (not H_Z)."""
    from qldpc.circuits.surgery.gadget import _step1_restriction
    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    support, C0, F = _step1_restriction(code, z, basis=Pauli.Z)
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    # V_0 = supp(z)
    assert support == tuple(int(i) for i in np.where(z)[0])
    # C_0 = X-checks touching V_0
    touched = sorted({j for j in range(HX.shape[0]) for i in support if HX[j, i] == 1})
    assert C0 == tuple(touched)
    # F = H_X[C_0, V_0]
    assert np.array_equal(F, HX[np.ix_(C0, support)])
    # Webster §II.A step 1 invariant: F @ 1_{V0} = 0 (since H_X @ z = 0 for a logical Z)
    ones = np.ones(len(support), dtype=np.uint8)
    assert np.array_equal((F @ ones) % 2, np.zeros(len(C0), dtype=np.uint8))


def test_build_gadget_z_basis_css_commutation():
    """build_gadget(code, z_logical, basis=Pauli.Z) yields a CSS-commuting merged code."""
    from qldpc.circuits.surgery.gadget import build_gadget
    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    assert g.basis is Pauli.Z
    product = (g.HX_merged @ g.HZ_merged.T) % 2
    assert np.array_equal(product, np.zeros_like(product))


def test_build_gadget_z_basis_rejects_non_z_logical():
    """For basis=Pauli.Z, build_gadget checks HX @ x == 0 (z must be a Z-logical)."""
    from qldpc.circuits.surgery.gadget import build_gadget
    code = codes.SteaneCode()
    # An X-logical fails: HX @ x_logical_X is typically nonzero
    x_logical = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    if ((HX @ x_logical) % 2).any():
        with pytest.raises(ValueError, match="logical"):
            build_gadget(code, x_logical, basis=Pauli.Z)


def test_build_gadget_z_basis_dual_matches_x_basis_on_dual_code():
    """basis-symmetric invariant: build_gadget(code, z, basis=Z) gives the same
    merged matrices as build_gadget(dual_code, z, basis=X), where dual_code has
    HX/HZ swapped. The swap labels swap too, so we compare HX_z vs HZ_dx_x and
    HZ_z vs HX_dx_x."""
    from qldpc.circuits.surgery.gadget import build_gadget
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


def test_webster_table_i_z_basis_kappa_chi_r_exact():
    """Webster Z̄_1 seed produces the same κ+χ+r counts (basis-symmetric)."""
    from qldpc.circuits.surgery.gadget import (
        build_gadget,
    )

    def z_bar_1_operator(d: dict) -> np.ndarray:
        l = d["l"]
        for seed in d["seeds"]:
            if seed["name"] == "Z_bar_1" and seed["pauli_type"] == "Z":
                L = np.zeros(l, dtype=np.uint8); R = np.zeros(l, dtype=np.uint8)
                for i in seed["L_support"]:
                    L[i] = 1
                for i in seed["R_support"]:
                    R[i] = 1
                return np.concatenate([L, R])
        raise ValueError("Z_bar_1 not found")

    for code_index, expected in [(0, 19), (1, 31), (2, 49), (3, 79)]:
        d = load_webster_seed_set(code_index)
        c = build_generalised_bicycle_code(d["l"], d["A"], d["B"])
        z = z_bar_1_operator(d)
        g = build_gadget(c, z, basis=Pauli.Z)
        kappa = len(g.kappa_qubits)
        chi = len(g.support)
        r = g.G.shape[0]
        assert kappa + chi + r == expected, (
            f"code {code_index}: Z-basis got κ+χ+r={kappa+chi+r}, expected {expected}"
        )


def test_build_gadget_augmented_extends_F_and_recomputes_G():
    """Augmenting with one weight-2 row adds a column to merged matrices and recomputes G."""
    from qldpc.circuits.surgery.gadget import build_gadget, build_gadget_augmented
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    # Pick two ports in V_0; create one extra weight-2 row connecting them
    support_a, support_b = g.support[0], g.support[1]
    extra_F = np.zeros((1, len(g.support)), dtype=np.uint8)
    idx_a = g.support.index(support_a)
    idx_b = g.support.index(support_b)
    extra_F[0, idx_a] = 1
    extra_F[0, idx_b] = 1
    g_aug = build_gadget_augmented(code, x, extra_F, basis=Pauli.X)

    # F_aug = [F | extra_F] vertically stacked
    assert g_aug.F.shape == (g.F.shape[0] + 1, g.F.shape[1])
    assert np.array_equal(g_aug.F[: g.F.shape[0]], g.F)
    assert np.array_equal(g_aug.F[g.F.shape[0]:], extra_F)
    # HX_merged has one extra column (one extra κ qubit); same number of rows
    assert g_aug.HX_merged.shape == (g.HX_merged.shape[0], g.HX_merged.shape[1] + 1)
    # CSS commutation
    product = (g_aug.HX_merged @ g_aug.HZ_merged.T) % 2
    assert np.array_equal(product, np.zeros_like(product))


def test_step2_gauge_fix_rows_linearly_independent():
    """G rows from _step2_gauge_fix are linearly independent over GF(2).

    Webster §II.A step 3 requires |S_L| - wt(L) + 1 INDEPENDENT gauge
    constraints. The existing test verifies G @ F == 0 (i.e. G is in
    ker(F.T)) but not that G has full row rank.

    A degenerate F could let the gauge fix return redundant rows,
    inflating g.G.shape[0] without changing the actual gauge structure.
    The Cain Table III bb_18 G=20 reproduction would catch the final
    count but not the underlying rank degeneracy.
    """
    import sympy
    from qldpc.circuits.surgery.gadget import build_gadget
    import galois

    F2 = galois.GF(2)
    xs, ys = sympy.symbols("x y")

    cases: list[tuple[str, object, np.ndarray]] = []

    # Case 1: Steane
    steane = codes.SteaneCode()
    x_steane = np.asarray(steane.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    cases.append(("Steane", steane, x_steane))

    # Case 2: Webster GB code 0
    data = load_webster_seed_set(0)
    webster = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x_webster = _webster_x_bar_operator(data, "X_bar_1")
    cases.append(("Webster GB 0", webster, x_webster))

    # Case 3: Cain bb_18 (cached Z̄ support — same as notebook §3.2)
    bb18 = codes.BBCode(
        (31, 4),
        1 + xs**6 * ys + xs**27,
        ys**2 + xs**15 * ys**3 + xs**24,
    )
    # Use the same cached wt-20 Z̄ rep used by the notebook §3.2 cell to
    # exercise the largest realistic gauge-fix case (G=20 rows). Treat
    # via swap (matrix_z ↔ matrix_x) so vec_20 acts as the X̄ on
    # target_code (matches notebook usage).
    z_bar_support = [8, 9, 14, 18, 24, 34, 40, 56, 75, 76,
                     97, 111, 122, 171, 202, 208, 213, 218, 228, 238]
    from qldpc.codes.common import CSSCode
    vec_20 = np.zeros(bb18.num_qudits, dtype=np.uint8)
    vec_20[z_bar_support] = 1
    bb18_swapped = CSSCode(
        bb18.matrix_z, bb18.matrix_x, is_subsystem_code=False,
    )
    cases.append(("Cain bb_18 (swapped, wt-20)", bb18_swapped, vec_20))

    for label, code, seed_op in cases:
        g = build_gadget(code, seed_op, basis=Pauli.X)
        G = g.G
        if G.shape[0] == 0:
            # Steane has G empty; trivially row-rank == 0 == shape[0].
            assert G.shape[0] == 0
            continue
        rank = int(np.linalg.matrix_rank(F2(G.astype(np.uint8).tolist())))
        assert rank == G.shape[0], (
            f"{label}: gauge-fix G has {G.shape[0]} rows but rank only "
            f"{rank}. _step2_gauge_fix returned redundant rows on this F."
        )
        # Re-assert the existing G @ F == 0 invariant alongside.
        # (G is a basis of ker(F.T), i.e. G F = 0 over GF(2);
        # see gadget._step2_gauge_fix and existing test_step2_gauge_fix.)
        F_mat = g.F.astype(np.uint8)
        commute = (G.astype(np.uint8) @ F_mat) % 2
        assert not commute.any(), (
            f"{label}: G @ F != 0 (gauge-fix output failed commutation)."
        )
