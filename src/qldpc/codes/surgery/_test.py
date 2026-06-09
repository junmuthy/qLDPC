"""Tests for the simplified surgery package (see
docs/superpowers/specs/2026-06-07-surgery-simplification-design.md)."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

from qldpc import codes
from qldpc.objects import Pauli

# Webster seed-set helpers live under examples/ (the JSON fixture is there too).
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "examples"))
from _webster_seed_set import (  # noqa: E402
    load_webster_seed_set,
    build_generalised_bicycle_code,
)


def test_gadget_layout_is_frozen_dataclass():
    from qldpc.codes.surgery.gadget import GadgetLayout
    assert dataclasses.is_dataclass(GadgetLayout)
    # frozen
    fields = {f.name for f in dataclasses.fields(GadgetLayout)}
    assert fields == {
        "code", "x", "V0", "C0", "F", "G",
        "HX_merged", "HZ_merged", "kappa_qubits", "basis",
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
    """Same F twice → byte-identical G (non-trivial: rank-deficient F → non-empty G)."""
    from qldpc.codes.surgery.gadget import _step2_gauge_fix
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
    from qldpc.codes.surgery.gadget import (
        _step1_restriction, _step2_gauge_fix, _step3_assemble,
    )
    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    V0, C0, F = _step1_restriction(code, z, basis=Pauli.Z)
    G = _step2_gauge_fix(F)
    HX_m, HZ_m = _step3_assemble(code, V0, C0, F, G, basis=Pauli.Z)

    n, mX, mZ = code.num_qudits, code.matrix_x.shape[0], code.matrix_z.shape[0]
    # For basis=Z: HX_merged grows by r rows (gauge-fix), HZ_merged by |V_0| rows (chi).
    assert HX_m.shape == (mX + G.shape[0], n + len(C0)), f"HX shape {HX_m.shape}"
    assert HZ_m.shape == (mZ + len(V0), n + len(C0)), f"HZ shape {HZ_m.shape}"
    # CSS commutation
    product = (HX_m @ HZ_m.T) % 2
    assert np.array_equal(product, np.zeros_like(product))


def test_step3_assemble_steane_css_commutes():
    from qldpc.codes.surgery.gadget import (
        _step1_restriction, _step2_gauge_fix, _step3_assemble,
    )
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    V0, C0, F = _step1_restriction(code, x)
    G = _step2_gauge_fix(F)
    HX_m, HZ_m = _step3_assemble(code, V0, C0, F, G)

    n, mX, mZ = code.num_qudits, code.matrix_x.shape[0], code.matrix_z.shape[0]
    assert HX_m.shape == (mX + len(V0), n + len(C0))
    assert HZ_m.shape == (mZ + G.shape[0], n + len(C0))
    # math.md §1.5(a): H_X^merged @ H_Z^merged.T == 0 over GF(2)
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
    from qldpc.codes.surgery.gadget import (
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

    V0, C0, F = _step1_restriction(code, x_logical)
    # V0 = {0,1,2,3} (nV=4); HZ row0 touches {0,1}, HZ row1 touches {0,2} -> C0=(0,1) (nC=2)
    assert len(V0) != len(C0), (
        f"nV={len(V0)} == nC={len(C0)}: this test requires nV != nC to catch the bug"
    )

    G = _step2_gauge_fix(F)
    HX_m, HZ_m = _step3_assemble(code, V0, C0, F, G)

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
    from qldpc.codes.surgery.gadget import build_gadget, GadgetLayout
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    assert isinstance(g, GadgetLayout)
    assert g.code is code
    assert np.array_equal(g.x, x)
    # κ qubits indexed contiguously after data qubits
    assert g.kappa_qubits == tuple(range(code.num_qudits, code.num_qudits + len(g.C0)))


def test_build_gadget_deterministic():
    from qldpc.codes.surgery.gadget import build_gadget
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g1 = build_gadget(code, x)
    g2 = build_gadget(code, x)
    assert g1.V0 == g2.V0
    assert g1.C0 == g2.C0
    assert np.array_equal(g1.F, g2.F)
    assert np.array_equal(g1.G, g2.G)
    assert np.array_equal(g1.HX_merged, g2.HX_merged)
    assert np.array_equal(g1.HZ_merged, g2.HZ_merged)
    assert g1.kappa_qubits == g2.kappa_qubits


def test_build_gadget_rejects_non_x_logical():
    from qldpc.codes.surgery.gadget import build_gadget
    code = codes.SteaneCode()
    x = np.zeros(code.num_qudits, dtype=np.uint8)
    x[0] = 1  # not a logical X (HZ @ x ≠ 0 in general)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    if ((HZ @ x) % 2).any():
        with pytest.raises(ValueError, match="logical"):
            build_gadget(code, x)


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


WEBSTER_TABLE_I_KAPPA_CHI_R = [(0, 19), (1, 31), (2, 49), (3, 79)]


def _webster_x_bar_operator(data: dict, name: str = "X_bar_1", pauli_type: str = "X") -> np.ndarray:
    """Extract the named logical operator from a Webster seed_set dict.

    L_support and R_support are sparse index lists (positions within each l-half
    that are set to 1). Returns a dense binary vector of length 2l.

    Args:
        data: Webster seed set dict (from load_webster_seed_set).
        name: Seed name, e.g. "X_bar_1", "Z_bar_1".
        pauli_type: "X" or "Z"; filters seeds by pauli_type field.
    """
    l = data["l"]
    for seed in data["seeds"]:
        if seed["name"] == name and seed["pauli_type"] == pauli_type:
            v_L = np.zeros(l, dtype=np.uint8)
            v_L[seed["L_support"]] = 1
            v_R = np.zeros(l, dtype=np.uint8)
            v_R[seed["R_support"]] = 1
            return np.concatenate([v_L, v_R])
    raise ValueError(f"{name!r} (pauli_type={pauli_type!r}) seed not found")


def _webster_z_bar_operator(data: dict, name: str = "Z_bar_1") -> np.ndarray:
    """Extract the named Z-type logical operator from a Webster seed_set dict.

    Convenience wrapper around _webster_x_bar_operator with pauli_type="Z".
    """
    return _webster_x_bar_operator(data, name, pauli_type="Z")


def _webster_x_bar_1_operator(data: dict) -> np.ndarray:
    """Back-compat: returns X_bar_1; prefer _webster_x_bar_operator."""
    return _webster_x_bar_operator(data, "X_bar_1")


@pytest.mark.parametrize("code_index,n_anc", WEBSTER_TABLE_I_KAPPA_CHI_R)
def test_webster_table_i_kappa_chi_r_exact(code_index, n_anc):
    """Webster Table I: κ + χ + r matches for each of the 4 codes."""
    from qldpc.codes.surgery.gadget import (
        build_gadget,
    )
    data = load_webster_seed_set(code_index)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x1 = _webster_x_bar_1_operator(data)
    g1 = build_gadget(code, x1)
    kappa = len(g1.kappa_qubits)
    chi = int(g1.x.sum())  # |V_0|
    r = g1.G.shape[0]
    assert kappa + chi + r == n_anc, (
        f"code {code_index}: κ={kappa}, χ={chi}, r={r}, "
        f"sum={kappa+chi+r}, expected {n_anc}"
    )


def test_skip_tree_fullrank_on_K4_matches_H_R():
    """SkipTree full-rank: T_ind · G · P_ind = H_R for the complete graph K_4."""
    import networkx as nx
    from qldpc.codes.surgery.bridge import _skip_tree_fullrank, _canonical_H_R

    G_nx = nx.complete_graph(4)
    n = 4
    edges = sorted(tuple(sorted(e)) for e in G_nx.edges())
    edge_index_verts = {e: i for i, e in enumerate(edges)}
    G_mat = np.zeros((len(edges), n), dtype=np.int_)
    for (u, v), i in edge_index_verts.items():
        G_mat[i, u] = 1
        G_mat[i, v] = 1

    T_ind, P_ind = _skip_tree_fullrank(G_nx, root=0, edge_index_verts=edge_index_verts)
    H_R = _canonical_H_R(n)

    assert T_ind.shape == (n - 1, len(edges))
    assert P_ind.shape == (n, n)
    # SkipTree key identity: T_ind · G · P_ind == H_R over GF(2)
    product = (T_ind @ G_mat @ P_ind) % 2
    assert np.array_equal(product, H_R), f"got\n{product}\nwant\n{H_R}"
    # Paper Theorem 7: (3,2)-sparsity is a general invariant of SkipTree.
    assert T_ind.sum(axis=1).max() <= 3
    assert T_ind.sum(axis=0).max() <= 2


def test_build_single_ppm_circuit_noiseless_compiles():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import build_single_ppm_circuit
    import stim
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    circuit = build_single_ppm_circuit(g, rounds=2, noise_model=None)
    assert isinstance(circuit, stim.Circuit)
    assert len(circuit) > 0


def test_build_single_ppm_circuit_noiseless_no_detectors_fire():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import build_single_ppm_circuit
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    circuit = build_single_ppm_circuit(g, rounds=2, noise_model=None)
    sampler = circuit.compile_detector_sampler()
    samples = sampler.sample(shots=16)
    assert (samples == 0).all()


def test_build_single_ppm_circuit_with_noise_detectors_fire():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits.noise_model import DepolarizingNoiseModel
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    circuit = build_single_ppm_circuit(
        g, rounds=2, noise_model=DepolarizingNoiseModel(p=0.05),
    )
    samples = circuit.compile_detector_sampler().sample(shots=200)
    assert samples.any()  # at least one detector fires under noise


def test_boost_gadget_dispatches_to_three_methods():
    from qldpc.codes.surgery.gadget import (
        build_gadget, GadgetLayout,
    )
    from qldpc.codes.surgery.cheeger import boost_gadget
    # Use Webster code 0 (l=31, k>=2): Steane gadget has dimension 0 (Steane
    # k=1 minus 1 gadget-consumed logical), which causes the BP+OSD decoder
    # used by boost_gadget_distance to hang searching for nonexistent logicals.
    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x = _webster_x_bar_1_operator(data)
    g = build_gadget(code, x)
    for method in ("spectral", "combinatorial", "distance"):
        out = boost_gadget(g, method=method, target=1.0, seed=42)
        assert isinstance(out, GadgetLayout), f"method={method}"


def test_boost_gadget_seed_reproducible():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.cheeger import boost_gadget
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    a = boost_gadget(g, method="spectral", target=1.0, seed=42)
    b = boost_gadget(g, method="spectral", target=1.0, seed=42)
    assert np.array_equal(a.F, b.F)
    assert np.array_equal(a.HX_merged, b.HX_merged)


@pytest.mark.parametrize("method", ["spectral", "combinatorial", "distance"])
def test_boost_gadget_preserves_css_commutation(method):
    from qldpc.codes.surgery.gadget import (
        build_gadget,
    )
    from qldpc.codes.surgery.cheeger import boost_gadget
    # Webster code 0 — Steane causes distance-boost decoder to hang on k=0 merged.
    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x = _webster_x_bar_1_operator(data)
    g = build_gadget(code, x)
    boosted = boost_gadget(g, method=method, target=1.0, seed=0)
    product = (boosted.HX_merged @ boosted.HZ_merged.T) % 2
    assert np.array_equal(product, np.zeros_like(product))


def test_gadget_layout_has_basis_field():
    from qldpc.codes.surgery.gadget import GadgetLayout
    fields = {f.name for f in dataclasses.fields(GadgetLayout)}
    assert "basis" in fields, f"basis field missing; got {fields}"


def test_gadget_layout_basis_defaults_to_x_via_build_gadget():
    """Backward compatibility: build_gadget without explicit basis defaults to Pauli.X."""
    from qldpc.codes.surgery.gadget import build_gadget
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    assert g.basis is Pauli.X


def test_step1_restriction_basis_z_uses_HX():
    """For basis=Pauli.Z, F = H_X[C_0, V_0] (not H_Z)."""
    from qldpc.codes.surgery.gadget import _step1_restriction
    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    V0, C0, F = _step1_restriction(code, z, basis=Pauli.Z)
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    # V_0 = supp(z)
    assert V0 == tuple(int(i) for i in np.where(z)[0])
    # C_0 = X-checks touching V_0
    touched = sorted({j for j in range(HX.shape[0]) for i in V0 if HX[j, i] == 1})
    assert C0 == tuple(touched)
    # F = H_X[C_0, V_0]
    assert np.array_equal(F, HX[np.ix_(C0, V0)])
    # math.md §1.1 invariant: F @ 1_{V0} = 0 (since H_X @ z = 0 for a logical Z)
    ones = np.ones(len(V0), dtype=np.uint8)
    assert np.array_equal((F @ ones) % 2, np.zeros(len(C0), dtype=np.uint8))


def test_build_gadget_z_basis_css_commutation():
    """build_gadget(code, z_logical, basis=Pauli.Z) yields a CSS-commuting merged code."""
    from qldpc.codes.surgery.gadget import build_gadget
    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    assert g.basis is Pauli.Z
    product = (g.HX_merged @ g.HZ_merged.T) % 2
    assert np.array_equal(product, np.zeros_like(product))


def test_build_gadget_z_basis_rejects_non_z_logical():
    """For basis=Pauli.Z, build_gadget checks HX @ x == 0 (z must be a Z-logical)."""
    from qldpc.codes.surgery.gadget import build_gadget
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
    from qldpc.codes.surgery.gadget import build_gadget
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
    from qldpc.codes.surgery.gadget import (
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
        chi = len(g.V0)
        r = g.G.shape[0]
        assert kappa + chi + r == expected, (
            f"code {code_index}: Z-basis got κ+χ+r={kappa+chi+r}, expected {expected}"
        )


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_boost_gadget_preserves_css_commutation_both_bases(basis):
    """boost_gadget on a basis=X or basis=Z gadget preserves CSS commutation."""
    from qldpc.codes.surgery.gadget import (
        build_gadget,
    )
    from qldpc.codes.surgery.cheeger import boost_gadget

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


def test_classify_reliable_round1_checks_basis_x():
    """For basis=X: reliable round-1 checks are data H_X (first m_X X-checks)
    plus gauge-fix G (last r Z-checks)."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import _classify_reliable_round1_checks
    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.codes.common import CSSCode
    import galois
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    F2 = galois.GF(2)
    merged = CSSCode(
        F2(g.HX_merged.astype(np.int_).tolist()),
        F2(g.HZ_merged.astype(np.int_).tolist()),
        is_subsystem_code=False,
    )
    qubit_ids = QubitIDs.from_code(merged)
    reliable = _classify_reliable_round1_checks(g, qubit_ids)
    m_X = code.matrix_x.shape[0]
    m_Z = code.matrix_z.shape[0]
    # Reliable X-checks: first m_X of checks_x (the original data H_X rows)
    expected_x_reliable = set(qubit_ids.checks_x[:m_X])
    # Reliable Z-checks: last r of checks_z (the gauge-fix G rows)
    r = g.G.shape[0]
    expected_z_reliable = set(qubit_ids.checks_z[m_Z:])
    expected = expected_x_reliable | expected_z_reliable
    assert set(reliable) == expected, (
        f"reliable={set(reliable)}, expected={expected}"
    )


def test_classify_reliable_round1_checks_basis_z():
    """For basis=Z: reliable round-1 checks are data H_Z (first m_Z Z-checks)
    plus gauge-fix G (last r X-checks)."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import _classify_reliable_round1_checks
    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.codes.common import CSSCode
    import galois
    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    F2 = galois.GF(2)
    merged = CSSCode(
        F2(g.HX_merged.astype(np.int_).tolist()),
        F2(g.HZ_merged.astype(np.int_).tolist()),
        is_subsystem_code=False,
    )
    qubit_ids = QubitIDs.from_code(merged)
    reliable = _classify_reliable_round1_checks(g, qubit_ids)
    m_X = code.matrix_x.shape[0]
    m_Z = code.matrix_z.shape[0]
    r = g.G.shape[0]
    # basis=Z: data H_Z rows are first m_Z Z-checks; G rows are last r X-checks
    expected_z_reliable = set(qubit_ids.checks_z[:m_Z])
    expected_x_reliable = set(qubit_ids.checks_x[m_X:])
    expected = expected_z_reliable | expected_x_reliable
    assert set(reliable) == expected


def test_surgery_state_prep_basis_x_resets():
    """basis=X: data RX (→|+⟩), kappa R (→|0⟩)."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import _surgery_state_prep
    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.codes.common import CSSCode
    import galois
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    F2 = galois.GF(2)
    merged = CSSCode(
        F2(g.HX_merged.astype(np.int_).tolist()),
        F2(g.HZ_merged.astype(np.int_).tolist()),
        is_subsystem_code=False,
    )
    qubit_ids = QubitIDs.from_code(merged)
    n_data = code.num_qudits
    data_ids = qubit_ids.data[:n_data]
    kappa_ids = qubit_ids.data[n_data:]
    circuit = _surgery_state_prep(g, data_ids, kappa_ids, bridge_ids=())
    text = str(circuit)
    assert f"RX {' '.join(str(q) for q in data_ids)}" in text
    assert f"R {' '.join(str(q) for q in kappa_ids)}" in text


def test_surgery_state_prep_basis_z_resets():
    """basis=Z: data R (→|0⟩), kappa RX (→|+⟩)."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import _surgery_state_prep
    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.codes.common import CSSCode
    import galois
    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    F2 = galois.GF(2)
    merged = CSSCode(
        F2(g.HX_merged.astype(np.int_).tolist()),
        F2(g.HZ_merged.astype(np.int_).tolist()),
        is_subsystem_code=False,
    )
    qubit_ids = QubitIDs.from_code(merged)
    n_data = code.num_qudits
    data_ids = qubit_ids.data[:n_data]
    kappa_ids = qubit_ids.data[n_data:]
    circuit = _surgery_state_prep(g, data_ids, kappa_ids, bridge_ids=())
    text = str(circuit)
    assert f"R {' '.join(str(q) for q in data_ids)}" in text
    assert f"RX {' '.join(str(q) for q in kappa_ids)}" in text


def test_surgery_qec_cycle_round_1_detectors_classified():
    """Round-1 detectors are 1-arg only for RELIABLE checks; unreliable ones skipped."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import _surgery_qec_cycle, _classify_reliable_round1_checks
    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.codes.common import CSSCode
    import galois
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    F2 = galois.GF(2)
    merged = CSSCode(
        F2(g.HX_merged.astype(np.int_).tolist()),
        F2(g.HZ_merged.astype(np.int_).tolist()),
        is_subsystem_code=False,
    )
    qubit_ids = QubitIDs.from_code(merged)
    reliable = _classify_reliable_round1_checks(g, qubit_ids)

    circuit, meas_rec, det_rec = _surgery_qec_cycle(
        g, merged, num_rounds=2, qubit_ids=qubit_ids,
    )
    # Count round-1 1-arg DETECTORs (those appearing before any REPEAT_BLOCK).
    text = str(circuit)
    # Number of "DETECTOR" instructions in the first round (before the REPEAT block)
    # should equal len(reliable).
    first_round_str = text.split("REPEAT")[0]
    n_det = first_round_str.count("DETECTOR")
    assert n_det == len(reliable), (
        f"round-1 detectors={n_det}, expected len(reliable)={len(reliable)}"
    )


def test_surgery_detach_and_readout_basis_x_measures_kappa_then_data():
    """basis=X: detach with M (Z-basis) on κ, then MX on data."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import _surgery_detach_and_readout
    from qldpc.circuits.bookkeeping import MeasurementRecord
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    n_data = code.num_qudits
    data_ids = tuple(range(n_data))
    kappa_ids = tuple(range(n_data, n_data + len(g.kappa_qubits)))
    bridge_ids = ()
    meas_rec = MeasurementRecord()
    circuit = _surgery_detach_and_readout(
        g, data_ids=data_ids, kappa_ids=kappa_ids, bridge_ids=bridge_ids,
        measurement_record=meas_rec,
    )
    text = str(circuit)
    # κ measured first (in Z), then data (in X)
    m_kappa_idx = text.find(f"M {' '.join(str(q) for q in kappa_ids)}")
    m_data_idx = text.find(f"MX {' '.join(str(q) for q in data_ids)}")
    assert m_kappa_idx >= 0 and m_data_idx >= 0
    assert m_kappa_idx < m_data_idx


def test_surgery_detach_and_readout_basis_z_measures_kappa_in_x_then_data_in_z():
    """basis=Z: detach with MX on κ, then M on data."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import _surgery_detach_and_readout
    from qldpc.circuits.bookkeeping import MeasurementRecord
    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    n_data = code.num_qudits
    data_ids = tuple(range(n_data))
    kappa_ids = tuple(range(n_data, n_data + len(g.kappa_qubits)))
    meas_rec = MeasurementRecord()
    circuit = _surgery_detach_and_readout(
        g, data_ids=data_ids, kappa_ids=kappa_ids, bridge_ids=(),
        measurement_record=meas_rec,
    )
    text = str(circuit)
    m_kappa_idx = text.find(f"MX {' '.join(str(q) for q in kappa_ids)}")
    m_data_idx = text.find(f"M {' '.join(str(q) for q in data_ids)}")
    assert m_kappa_idx >= 0 and m_data_idx >= 0
    assert m_kappa_idx < m_data_idx


def test_surgery_observable_emits_two_observable_include():
    """Observable 0 = XOR of χ-row records across all rounds; Observable 1 = data measurement on V_0."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import _surgery_observable
    from qldpc.circuits.bookkeeping import MeasurementRecord
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    n_data = code.num_qudits
    chi_check_ids = tuple(range(100, 100 + len(g.V0)))   # placeholder ids
    data_ids = tuple(range(n_data))
    meas_rec = MeasurementRecord()
    # Simulate 2 rounds of chi-check measurements
    for _ in range(2):
        meas_rec.append({cid: i for i, cid in enumerate(chi_check_ids)})
    # Simulate final data measurement
    meas_rec.append({d: i for i, d in enumerate(data_ids)})

    circuit = _surgery_observable(
        g, chi_check_ids=chi_check_ids, data_ids=data_ids,
        v0_indices=g.V0, num_rounds=2, measurement_record=meas_rec,
    )
    text = str(circuit)
    assert text.count("OBSERVABLE_INCLUDE") == 2  # PPM + cross-check
    assert "(0)" in text and "(1)" in text  # two distinct observable indices


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_build_single_ppm_circuit_noiseless_observables_zero(basis):
    """Both OBSERVABLE_INCLUDEs evaluate to 0 (= +1) under no noise."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import build_single_ppm_circuit
    code = codes.SteaneCode()
    op = (code.get_logical_ops(Pauli.X)[0]
          if basis is Pauli.X
          else code.get_logical_ops(Pauli.Z)[0])
    op_arr = np.asarray(op).astype(np.uint8)
    g = build_gadget(code, op_arr, basis=basis)
    circuit = build_single_ppm_circuit(g, rounds=3, noise_model=None)
    # Sample observables; all should be 0.
    sampler = circuit.compile_detector_sampler()
    _, obs = sampler.sample(shots=16, separate_observables=True)
    assert (obs == 0).all(), (
        f"noiseless observables fired: {obs.sum()} flips across 16 shots"
    )


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_single_ppm_circuit_noise_flips_observable_at_high_p(basis):
    """At p=0.1, the PPM observable (observable 0) flips ≥ 5% of shots."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits.noise_model import DepolarizingNoiseModel
    code = codes.SteaneCode()
    op = (code.get_logical_ops(Pauli.X)[0]
          if basis is Pauli.X
          else code.get_logical_ops(Pauli.Z)[0])
    op_arr = np.asarray(op).astype(np.uint8)
    g = build_gadget(code, op_arr, basis=basis)
    circuit = build_single_ppm_circuit(
        g, rounds=3, noise_model=DepolarizingNoiseModel(p=0.1),
    )
    sampler = circuit.compile_detector_sampler()
    _, obs = sampler.sample(shots=400, separate_observables=True)
    # Observable 0 (PPM) flips a nontrivial fraction at p=0.1
    obs_0_flip_rate = float(obs[:, 0].mean())
    assert obs_0_flip_rate >= 0.05, (
        f"PPM observable flip rate {obs_0_flip_rate:.2%} too low at p=0.1"
    )


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_surgery_final_detectors_count_matches_reliable_round1(basis):
    """Number of final DETECTORs equals |reliable round-1 set|.

    Tests the helper in isolation: build a circuit through detach_and_readout,
    then call _surgery_final_detectors and count emitted DETECTOR instructions.
    """
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import (
        _surgery_state_prep, _surgery_qec_cycle, _surgery_detach_and_readout,
        _surgery_final_detectors, _classify_reliable_round1_checks,
        _gadget_merged_csscode,
    )
    from qldpc.circuits.bookkeeping import QubitIDs

    code = codes.SteaneCode()
    op = (code.get_logical_ops(Pauli.X)[0] if basis is Pauli.X
          else code.get_logical_ops(Pauli.Z)[0])
    op_arr = np.asarray(op).astype(np.uint8)
    g = build_gadget(code, op_arr, basis=basis)
    merged = _gadget_merged_csscode(g)
    qubit_ids = QubitIDs.from_code(merged)
    n_data = code.num_qudits
    data_ids = qubit_ids.data[:n_data]
    kappa_ids = qubit_ids.data[n_data:]

    # Simulate the pipeline through detach (we need measurement_record populated).
    _qec, mrec, _det = _surgery_qec_cycle(g, merged, num_rounds=2, qubit_ids=qubit_ids)
    _surgery_detach_and_readout(
        g, data_ids=data_ids, kappa_ids=kappa_ids, bridge_ids=(),
        measurement_record=mrec,
    )

    circuit = _surgery_final_detectors(g, merged, qubit_ids, measurement_record=mrec)
    n_final_det = str(circuit).count("DETECTOR")
    expected = len(_classify_reliable_round1_checks(g, qubit_ids))
    assert n_final_det == expected, (
        f"basis={basis}: emitted {n_final_det} DETECTORs, expected {expected}"
    )


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_build_single_ppm_circuit_noiseless_no_detector_fires(basis):
    """Noiseless: NO detector fires (including the new final detectors).

    The total detector count must equal: round-1 reliable + (rounds-1)*all_checks + final reliable.
    Under noiseless conditions all of them must remain silent.
    """
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import build_single_ppm_circuit
    code = codes.SteaneCode()
    op = (code.get_logical_ops(Pauli.X)[0] if basis is Pauli.X
          else code.get_logical_ops(Pauli.Z)[0])
    op_arr = np.asarray(op).astype(np.uint8)
    g = build_gadget(code, op_arr, basis=basis)
    circuit = build_single_ppm_circuit(g, rounds=3, noise_model=None)
    sampler = circuit.compile_detector_sampler()
    dets, _ = sampler.sample(shots=64, separate_observables=True)
    assert not dets.any(), (
        f"basis={basis}: {dets.sum()} detector fires noiselessly across {dets.shape[0]} shots"
    )


@pytest.mark.slow
def test_single_ppm_ler_monotone_in_p():
    """Tiny sinter sweep: PPM LER monotonically increasing in p.

    Catches gross protocol errors (wrong observable basis, sign flips, etc.).
    """
    import sinter
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits import DepolarizingNoiseModel
    from qldpc import decoders
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)

    error_rates = [0.001, 0.005, 0.02]
    tasks = []
    for p in error_rates:
        circuit = build_single_ppm_circuit(
            g, rounds=3, noise_model=DepolarizingNoiseModel(p),
        )
        tasks.append(sinter.Task(
            circuit=circuit,
            json_metadata={"p": float(p)},
        ))
    sinter_decoder = decoders.SinterDecoder()
    results = sinter.collect(
        tasks=tasks,
        decoders=["custom"],
        custom_decoders={"custom": sinter_decoder},
        num_workers=4,
        max_shots=2000,
        max_errors=30,
        print_progress=False,
    )
    by_p = {r.json_metadata["p"]: r.errors / max(r.shots, 1) for r in results}
    sorted_p = sorted(by_p.keys())
    ler_vals = [by_p[p] for p in sorted_p]
    print(f"LER values: {list(zip(sorted_p, ler_vals))}")
    # Monotonically non-decreasing (allow small statistical noise)
    for i in range(len(ler_vals) - 1):
        assert ler_vals[i] <= ler_vals[i + 1] * 1.5, (
            f"LER not monotonic: p={sorted_p[i]} → {ler_vals[i]}, "
            f"p={sorted_p[i+1]} → {ler_vals[i+1]}"
        )


@pytest.mark.slow
def test_single_ppm_ler_with_final_detectors_below_threshold():
    """With final detectors wired, LER at p=0.001 should be ≤ 0.01.

    Reference: before the final-detector wiring, LER at p=0.001 was ~0.024
    (from test_single_ppm_ler_monotone_in_p in the surgery-circuit-rewrite plan).
    Adding the inferred detectors should drop it significantly.
    """
    import sinter
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits import DepolarizingNoiseModel
    from qldpc import decoders

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)

    p = 0.001
    circuit = build_single_ppm_circuit(
        g, rounds=3, noise_model=DepolarizingNoiseModel(p),
    )
    sinter_decoder = decoders.SinterDecoder()
    results = sinter.collect(
        tasks=[sinter.Task(circuit=circuit, json_metadata={"p": float(p)})],
        decoders=["custom"],
        custom_decoders={"custom": sinter_decoder},
        num_workers=4,
        max_shots=5000,
        max_errors=50,
        print_progress=False,
    )
    assert len(results) == 1
    ler = results[0].errors / max(results[0].shots, 1)
    assert ler <= 0.01, (
        f"LER at p=0.001 = {ler:.4f} (errors={results[0].errors}/{results[0].shots} shots). "
        f"Expected ≤ 0.01 with final detectors wired. Was ~0.024 without them."
    )


def test_build_aux_graph_weight2_rows_become_edges():
    """F rows of weight 2 → graph edges; vertex set = {0, ..., |V_0|-1}."""
    from qldpc.codes.surgery.bridge import _build_aux_graph_strict
    F = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.uint8)
    G_nx, edge_idx = _build_aux_graph_strict(F)
    assert set(G_nx.nodes) == {0, 1, 2, 3}
    assert set(tuple(sorted(e)) for e in G_nx.edges) == {(0, 1), (1, 2), (2, 3)}
    assert edge_idx[(0, 1)] == 0
    assert edge_idx[(1, 2)] == 1
    assert edge_idx[(2, 3)] == 2


def test_build_aux_graph_filters_hyperedges():
    """F rows of weight >= 3 (hyperedges) are silently skipped; weight-2 rows survive."""
    from qldpc.codes.surgery.bridge import _build_aux_graph_strict
    F = np.array([
        [1, 1, 0, 0, 0],  # weight-2 → edge (0,1)
        [1, 1, 1, 1, 0],  # weight-4 hyperedge → skipped
        [0, 0, 1, 1, 0],  # weight-2 → edge (2,3)
        [0, 0, 0, 1, 1],  # weight-2 → edge (3,4)
    ], dtype=np.uint8)
    G_nx, edge_idx = _build_aux_graph_strict(F)
    assert set(G_nx.nodes) == {0, 1, 2, 3, 4}
    # Three weight-2 rows → three edges; hyperedge row contributes nothing
    assert G_nx.number_of_edges() == 3
    assert (0, 1) in edge_idx
    assert (2, 3) in edge_idx
    assert (3, 4) in edge_idx
    # Hyperedge would have produced edges (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    # but only edges from weight-2 rows are present
    assert (0, 2) not in edge_idx
    assert (0, 3) not in edge_idx
    assert (1, 3) not in edge_idx


def test_build_aux_graph_rejects_weight1_row():
    """F rows of weight 1 raise ValueError (dangling edge / no-op stabilizer)."""
    from qldpc.codes.surgery.bridge import _build_aux_graph_strict
    F = np.array([[1, 1, 0, 0], [0, 0, 1, 0]], dtype=np.uint8)
    with pytest.raises(ValueError, match=r"weight 1"):
        _build_aux_graph_strict(F)


def test_connect_induced_subgraph_no_op_when_connected():
    """If induced subgraph is already connected, no edges are added."""
    import networkx as nx
    from qldpc.codes.surgery.bridge import _connect_induced_subgraph
    G_aux = nx.path_graph(4)  # 0-1-2-3
    extra = _connect_induced_subgraph(G_aux, ports=(0, 1, 2, 3))
    assert extra == []
    assert set(tuple(sorted(e)) for e in G_aux.edges) == {(0, 1), (1, 2), (2, 3)}


def test_connect_induced_subgraph_adds_edges_to_disconnected_components():
    """Disconnected induced subgraph gets one bridging edge per missing connection."""
    import networkx as nx
    from qldpc.codes.surgery.bridge import _connect_induced_subgraph
    # G_aux: 0-1   2-3 (two separate components)
    G_aux = nx.Graph()
    G_aux.add_edges_from([(0, 1), (2, 3)])
    extra = _connect_induced_subgraph(G_aux, ports=(0, 1, 2, 3))
    assert len(extra) == 1  # exactly one bridge needed
    (u, v) = extra[0]
    # Endpoints must come from different original components
    assert {u, v} & {0, 1} and {u, v} & {2, 3}
    # G_aux mutated: induced subgraph now connected
    assert nx.is_connected(G_aux.subgraph((0, 1, 2, 3)))


def test_cellulate_caps_cycle_length():
    """After cellulation, every basis cycle has length <= cap."""
    import networkx as nx
    from qldpc.codes.surgery.bridge import _cellulate_port_subgraph
    # 10-cycle: 0-1-2-...-9-0 has one length-10 basis cycle
    G_aux = nx.cycle_graph(10)
    added = _cellulate_port_subgraph(G_aux, ports=tuple(range(10)), max_len=6)
    assert len(added) >= 1
    # All basis cycles now bounded
    sub = G_aux.subgraph(tuple(range(10)))
    assert max((len(c) for c in nx.cycle_basis(sub)), default=0) <= 6


def test_cellulate_no_op_when_already_short():
    """If all basis cycles are short, no edges are added."""
    import networkx as nx
    from qldpc.codes.surgery.bridge import _cellulate_port_subgraph
    G_aux = nx.cycle_graph(4)  # one 4-cycle
    added = _cellulate_port_subgraph(G_aux, ports=(0, 1, 2, 3), max_len=6)
    assert added == []


def test_cellulate_raises_when_port_cycle_has_no_available_chord():
    """RuntimeError when a port-subgraph cycle exists but every (i, j) pair
    is already an edge — i.e. the port subgraph is complete on those vertices."""
    import networkx as nx
    from qldpc.codes.surgery.bridge import _cellulate_port_subgraph
    # 7-cycle 0-1-2-3-4-5-6-0 plus ALL chords among {0..6} → complete graph K_7.
    # cycle_basis still surfaces cycles of length > max_len in K_7 (basis cycles
    # are length-3 triangles), so no long cycle exists in this case.
    # Instead: make a 7-cycle without any extra edges, then call with max_len=2.
    G = nx.cycle_graph(7)
    ports = tuple(range(7))
    # Already a complete graph K_7? No — cycle_graph(7) has only 7 edges.
    # Pre-saturate with all possible chords so no chord can be added:
    for i in range(7):
        for j in range(i + 2, 7):
            if not G.has_edge(i, j) and (i, j) != (0, 6):
                G.add_edge(i, j)
    # Now every (i, j) with j >= i+2 in the 7-cycle is already an edge.
    # A length-7 basis cycle no longer exists (it's broken into triangles),
    # so max_len=6 finds no long cycle and returns []. Use max_len=2 to force
    # the failure path:
    with pytest.raises(RuntimeError, match=r"No chord found"):
        _cellulate_port_subgraph(G, ports, max_len=2)


def test_cellulate_port_subgraph_breaks_long_port_cycle():
    """Ports are a strict subset of vertices, with a long cycle on the port
    subgraph. Cellulation breaks the port cycle without inspecting non-port
    edges elsewhere in G_aux."""
    import networkx as nx
    from qldpc.codes.surgery.bridge import _cellulate_port_subgraph
    G = nx.Graph()
    # 8-cycle on port vertices 0..7
    G.add_edges_from([(i, (i + 1) % 8) for i in range(8)])
    # Non-port "decoration": dangling vertex 100 attached to port 0
    G.add_edge(0, 100)
    ports = tuple(range(8))
    added = _cellulate_port_subgraph(G, ports, max_len=6)
    assert len(added) >= 1
    # All chord endpoints must be ports (cycle vertices are port vertices)
    for u, v in added:
        assert u in ports and v in ports
    # The non-port vertex 100 was not touched
    assert G.has_edge(0, 100)
    # All port-subgraph basis cycles now bounded
    sub = G.subgraph(ports)
    for c in nx.cycle_basis(sub):
        assert len(c) <= 6


def test_cellulate_port_subgraph_skips_non_port_cycle():
    """Long cycle entirely on non-port vertices is ignored; no edges added."""
    import networkx as nx
    from qldpc.codes.surgery.bridge import _cellulate_port_subgraph
    G = nx.Graph()
    # Long non-port cycle: 10-11-12-...-17-10 (length 8)
    G.add_edges_from([(10, 11), (11, 12), (12, 13), (13, 14),
                      (14, 15), (15, 16), (16, 17), (17, 10)])
    # Short port cycle: triangle on 0,1,2
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    ports = (0, 1, 2)
    n_edges_before = G.number_of_edges()
    added = _cellulate_port_subgraph(G, ports, max_len=6)
    assert added == []
    assert G.number_of_edges() == n_edges_before


def test_build_gadget_augmented_extends_F_and_recomputes_G():
    """Augmenting with one weight-2 row adds a column to merged matrices and recomputes G."""
    from qldpc.codes.surgery.gadget import build_gadget, build_gadget_augmented
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    # Pick two ports in V_0; create one extra weight-2 row connecting them
    v0_a, v0_b = g.V0[0], g.V0[1]
    extra_F = np.zeros((1, len(g.V0)), dtype=np.uint8)
    idx_a = g.V0.index(v0_a)
    idx_b = g.V0.index(v0_b)
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


def test_bridge_dataclass_fields_universal_adapter():
    """Bridge dataclass exposes the universal-adapter fields from spec §1."""
    import dataclasses
    from qldpc.codes.surgery.bridge import Bridge
    fields = {f.name for f in dataclasses.fields(Bridge)}
    assert fields == {
        "width", "basis",
        "port_l", "port_r",
        "label_l", "label_r",
        "extra_kappa_l", "extra_kappa_r",
        "T_l", "T_r", "H_R",
        "g_l_aug", "g_r_aug",
    }


def test_build_bridge_smoke_steane_intracode():
    """Steane × Steane intra-code joint X̄ X̄: build_bridge returns valid Bridge."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    code = codes.SteaneCode()
    x1 = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    x2 = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)  # same logical
    g_l = build_gadget(code, x1, basis=Pauli.X)
    g_r = build_gadget(code, x2, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    assert bridge.width == min(len(g_l.V0), len(g_r.V0))
    assert bridge.basis is Pauli.X
    assert len(bridge.port_l) == bridge.width
    assert len(bridge.port_r) == bridge.width
    assert bridge.T_l.shape == (bridge.width - 1, bridge.g_l_aug.F.shape[0])
    assert bridge.T_r.shape == (bridge.width - 1, bridge.g_r_aug.F.shape[0])
    assert bridge.H_R.shape == (bridge.width - 1, bridge.width)


def test_build_bridge_skiptree_invariant_holds():
    """T_s · G_s_aug · P_s = H_R for both sides on Steane × Steane."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(code, x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)

    for side in ("l", "r"):
        T = getattr(bridge, f"T_{side}")
        g_aug = getattr(bridge, f"g_{side}_aug")
        label = getattr(bridge, f"label_{side}")
        # G_aug = F_aug (incidence: rows = edges = κ qubits, cols = V_0 vertices)
        G_aug = g_aug.F.astype(np.int_)
        # P_s: |V_0^(s)| × w; P_s[v, k] = 1 iff v ∈ port AND label[v] == k
        P = np.zeros((G_aug.shape[1], bridge.width), dtype=np.int_)
        for v_idx, lab in enumerate(label):
            if lab >= 0:
                P[v_idx, lab] = 1
        lhs = (T @ G_aug @ P) % 2
        assert np.array_equal(lhs, bridge.H_R), f"side {side}:\n{lhs}\nvs\n{bridge.H_R}"


def test_build_bridge_rejects_basis_mismatch():
    """Bridge requires g_l.basis == g_r.basis."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(code, z, basis=Pauli.Z)
    with pytest.raises(ValueError, match=r"basis"):
        build_bridge(g_l, g_r)


def test_stitch_intercode_basis_x_css_commutation():
    """Inter-code Steane × Steane joint X̄X̄ merged code commutes."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import _stitch_to_joint_csscode
    code1 = codes.SteaneCode()
    code2 = codes.SteaneCode()
    x1 = np.asarray(code1.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    x2 = np.asarray(code2.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code1, x1, basis=Pauli.X)
    g_r = build_gadget(code2, x2, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    HZ = np.asarray(merged.matrix_z).astype(np.int_)
    product = (HX @ HZ.T) % 2
    assert np.array_equal(product, np.zeros_like(product))


def test_stitch_intercode_basis_x_k_reduces_by_one():
    """k_joint = k_l + k_r - 1 for inter-code Steane × Steane joint X̄X̄."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import _stitch_to_joint_csscode
    code1 = codes.SteaneCode()
    code2 = codes.SteaneCode()
    x1 = np.asarray(code1.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    x2 = np.asarray(code2.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code1, x1, basis=Pauli.X)
    g_r = build_gadget(code2, x2, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    assert merged.dimension == code1.dimension + code2.dimension - 1


def test_stitch_intercode_basis_x_joint_logical_in_stabilizer():
    """(x_1, x_2, 0, 0, 0) lies in rowspan(H_X^merged) — joint X̄_l X̄_r is a stabilizer."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import _stitch_to_joint_csscode
    code1 = codes.SteaneCode()
    code2 = codes.SteaneCode()
    x1 = np.asarray(code1.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    x2 = np.asarray(code2.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code1, x1, basis=Pauli.X)
    g_r = build_gadget(code2, x2, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    import galois
    GF2 = galois.GF(2)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    n_l = code1.num_qudits
    n_r = code2.num_qudits
    joint = np.zeros(HX.shape[1], dtype=np.int_)
    joint[:n_l] = x1
    joint[n_l : n_l + n_r] = x2
    augmented = np.vstack([HX, joint.reshape(1, -1)])
    assert np.linalg.matrix_rank(GF2(HX.tolist())) == np.linalg.matrix_rank(GF2(augmented.tolist()))


def test_stitch_intercode_basis_x_singletons_excluded():
    """(x_1, 0, ...) and (0, x_2, ...) alone are NOT in rowspan(H_X^merged)."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import _stitch_to_joint_csscode
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(codes.SteaneCode(), x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    import galois
    GF2 = galois.GF(2)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    n_l = code.num_qudits
    base = np.linalg.matrix_rank(GF2(HX.tolist()))
    for which in ("left", "right"):
        single = np.zeros(HX.shape[1], dtype=np.int_)
        if which == "left":
            single[:n_l] = x
        else:
            single[n_l : 2 * n_l] = x
        augmented = np.vstack([HX, single.reshape(1, -1)])
        assert np.linalg.matrix_rank(GF2(augmented.tolist())) == base + 1, which


def test_stitch_intracode_basis_x_css_commutation():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import _stitch_to_joint_csscode
    code = codes.SteaneCode()
    x1 = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    # Use Pauli.X logical 0 for both (same V_0); intra-code test
    g_l = build_gadget(code, x1, basis=Pauli.X)
    g_r = build_gadget(code, x1, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    HZ = np.asarray(merged.matrix_z).astype(np.int_)
    product = (HX @ HZ.T) % 2
    assert np.array_equal(product, np.zeros_like(product))


def test_stitch_intracode_basis_x_k_reduces_by_one():
    # Use Webster code 0 (k>=2) so the k_joint = k_data - 1 invariant is not
    # masked by the spurious bridge X-logical: Steane (k=1) with x_l = x_r is
    # the degenerate joint X̄ · X̄ = I case where the spurious bridge logical
    # leaves the dimension at k_data instead of k_data - 1.
    from qldpc.codes.surgery.gadget import (
        build_gadget,
    )
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import _stitch_to_joint_csscode

    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x1 = _webster_x_bar_operator(data, "X_bar_1")
    x2 = _webster_x_bar_operator(data, "X_bar_k2p1")
    g_l = build_gadget(code, x1, basis=Pauli.X)
    g_r = build_gadget(code, x2, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    assert merged.dimension == code.dimension - 1


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_stitch_intercode_both_bases_commute_and_singletons_excluded(basis):
    import galois
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import _stitch_to_joint_csscode
    GF2 = galois.GF(2)
    code = codes.SteaneCode()
    if basis is Pauli.X:
        x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    else:
        x = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=basis)
    g_r = build_gadget(codes.SteaneCode(), x, basis=basis)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    HZ = np.asarray(merged.matrix_z).astype(np.int_)
    product = (HX @ HZ.T) % 2
    assert np.array_equal(product, np.zeros_like(product))
    assert merged.dimension == 2 * code.dimension - 1
    # Singletons excluded: (x_l, 0, ...) and (0, x_r, ...) NOT in rowspan of the
    # check matrix that contains the joint stabilizer (HX for basis=X, HZ for Z).
    H_joint = HX if basis is Pauli.X else HZ
    n_l = code.num_qudits
    base_rank = np.linalg.matrix_rank(GF2(H_joint.tolist()))
    for which in ("left", "right"):
        single = np.zeros(H_joint.shape[1], dtype=np.int_)
        if which == "left":
            single[:n_l] = x
        else:
            single[n_l : 2 * n_l] = x
        augmented = np.vstack([H_joint, single.reshape(1, -1)])
        assert np.linalg.matrix_rank(GF2(augmented.tolist())) == base_rank + 1, which


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_stitch_intracode_both_bases_commute(basis):
    """Intra-code commutation for both bases. Use a Webster code with 2 distinct logicals.

    Steane intra-code (k=1) yields the degenerate joint X̄·X̄ = I case.
    """
    from qldpc.codes.surgery.gadget import (
        build_gadget,
    )
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import _stitch_to_joint_csscode
    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    if basis is Pauli.X:
        x1 = _webster_x_bar_operator(data, "X_bar_1")
        x2 = _webster_x_bar_operator(data, "X_bar_k2p1")
    else:
        # Z-type analogs from the Webster seed set.
        l = data["l"]
        def _z_op(name):
            for s in data["seeds"]:
                if s["name"] == name and s["pauli_type"] == "Z":
                    v = np.zeros(2 * l, dtype=np.uint8)
                    v[:l][s["L_support"]] = 1
                    v[l:][s["R_support"]] = 1
                    return v
            raise ValueError(f"{name} not found")
        x1 = _z_op("Z_bar_1")
        x2 = _z_op("Z_bar_k2p1")
    g_l = build_gadget(code, x1, basis=basis)
    g_r = build_gadget(code, x2, basis=basis)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    HZ = np.asarray(merged.matrix_z).astype(np.int_)
    product = (HX @ HZ.T) % 2
    assert np.array_equal(product, np.zeros_like(product))
    assert merged.dimension == code.dimension - 1


def test_build_bridge_bb18_hyperedge_and_long_cycle():
    """End-to-end: Cain bb_18 BBCode triggers both Bug 1 (hyperedge) and
    Bug 2 (long port-subgraph cycle). build_bridge must succeed and produce
    a merged code with k_merged = k_orig - 1 (intra-code joint Z̄_1 ⊗ Z̄_2).

    Two *distinct* Z-logicals are used so that the joint measurement reduces k
    by exactly 1.  Z-logical 0 has a weight-4 F row (triggers Bug 1); the pair
    together exercises the full _cellulate_port_subgraph path (Bug 2)."""
    import sympy
    from qldpc.codes.surgery import build_gadget, build_bridge
    from qldpc.codes.surgery.circuit import _stitch_to_joint_csscode

    x, y = sympy.symbols("x y")
    code = codes.BBCode(
        {x: 31, y: 4},
        1 + x**6 * y + x**27,
        y**2 + x**15 * y**3 + x**24,
    )
    z_ops = code.get_logical_ops(Pauli.Z)
    z0 = np.asarray(z_ops[0]).astype(np.uint8)  # hyperedge logical (Bug 1)
    z1 = np.asarray(z_ops[1]).astype(np.uint8)  # distinct second logical
    g_l = build_gadget(code, z0, basis=Pauli.Z)
    g_r = build_gadget(code, z1, basis=Pauli.Z)
    # Confirm we are actually exercising Bug 1 (hyperedge in left gadget):
    row_weights = np.asarray(g_l.F.sum(axis=1)).ravel().astype(int).tolist()
    assert max(row_weights) >= 4, "Test no longer triggers Bug 1 (no hyperedge)"
    # Build bridge (this used to raise NotImplementedError or RuntimeError)
    bridge = build_bridge(g_l, g_r)
    # Merged code construction must succeed
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    # Intra-code joint Z̄_1 ⊗ Z̄_2: k_merged == k_orig − 1
    assert merged.dimension == code.dimension - 1
    # CSS commutation on merged code
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    HZ = np.asarray(merged.matrix_z).astype(np.int_)
    assert not ((HX @ HZ.T) % 2).any(), "CSS commutation broken on merged code"


def test_build_joint_ppm_circuit_chi_check_ids_no_UB():
    """build_joint_ppm_circuit's noiseless first sample has zero detectors firing."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import build_joint_ppm_circuit
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(codes.SteaneCode(), x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    circuit, merged = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=2)
    # noiseless: all detectors must NOT fire on first sample
    sampler = circuit.compile_detector_sampler()
    dets, _ = sampler.sample(8, separate_observables=True)
    assert dets.sum() == 0


def test_build_joint_ppm_circuit_intercode_noiseless_observables_zero():
    """basis=X init: data in |+⟩, κ in |0⟩, adapter in |0⟩. Joint observables are deterministic."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import build_joint_ppm_circuit
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(codes.SteaneCode(), x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    circuit, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=2)
    sampler = circuit.compile_detector_sampler()
    _, obs = sampler.sample(8, separate_observables=True)
    # Joint observable 0 (chi XOR over rounds) deterministic = 0 in noiseless |+⟩^n init
    # Observable 1 (final M-X on V_0 ∪) deterministic = 0
    assert obs.sum() == 0


def test_adapter_cycle_check_weight_bounded():
    """Each new cycle-X row has weight <= 8 (SkipTree (3,2) + H_R weight 2). Basis=Z.

    For basis=Z, the new adapter cycle checks are placed in HX (the last w-1 rows).
    Each row has the form [T_l | H_R | T_r]:
      - T_l row: at most 3 entries on cl_kappa (SkipTree (3,2)-sparsity)
      - H_R row: exactly 2 entries on c_adapter (canonical rep code)
      - T_r row: at most 3 entries on cr_kappa (SkipTree (3,2)-sparsity)
    Total: weight <= 3 + 2 + 3 = 8.
    """
    from qldpc.codes.surgery.gadget import (
        build_gadget,
    )
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import _stitch_to_joint_csscode
    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    # Use Z̄_1 for both sides (intra-code, same logical) — bridge.width =
    # |V_0| = weight of Z̄_1, exercising the maximum-width cellulation path
    x = _webster_z_bar_operator(data, "Z_bar_1")
    g_l = build_gadget(code, x, basis=Pauli.Z)
    g_r = build_gadget(code, x, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    # basis=Z: new cycle-X-checks are the last (w-1) rows of HX
    new_x_rows = HX[-(bridge.width - 1):, :]
    max_w = int(new_x_rows.sum(axis=1).max())
    assert max_w <= 8, f"max new cycle-X weight {max_w} > 8"


def test_cellulation_caps_aug_aux_cycle_length_on_webster():
    """After cellulation, every basis cycle in the augmented aux graph has length <= 6."""
    import networkx as nx
    from qldpc.codes.surgery.gadget import (
        build_gadget,
    )
    from qldpc.codes.surgery.bridge import build_bridge, _build_aux_graph_strict
    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x = _webster_z_bar_operator(data, "Z_bar_1")
    g_l = build_gadget(code, x, basis=Pauli.Z)
    g_r = build_gadget(code, x, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r, cellulate_max_len=6)
    # Cellulation is now scoped to the port subgraph (where SkipTree runs).
    # Inspect cycles on the induced port subgraph, not the full graph.
    G_aux, _ = _build_aux_graph_strict(bridge.g_l_aug.F)
    sub = G_aux.subgraph(bridge.port_l)
    cycles = nx.cycle_basis(sub)
    if cycles:
        assert max(len(c) for c in cycles) <= 6, (
            f"max port-subgraph cycle length {max(len(c) for c in cycles)} > 6"
        )


@pytest.mark.slow
def test_joint_ppm_ler_monotone_steane_intercode():
    """LER non-increasing in p across {1e-4, 3e-4, 1e-3} for Steane × Steane."""
    from qldpc.circuits.noise_model import DepolarizingNoiseModel
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import build_joint_ppm_circuit
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(codes.SteaneCode(), x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    lers = []
    shots = 2000
    for p in (1e-3, 3e-4, 1e-4):
        nm = DepolarizingNoiseModel(p)
        circuit, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=3, noise_model=nm)
        sampler = circuit.compile_detector_sampler()
        _, obs = sampler.sample(shots, separate_observables=True)
        # logical error rate of OBS 0 (joint χ XOR)
        ler = (obs[:, 0] != 0).mean()
        lers.append(ler)
    # LER should be non-increasing as p decreases (tolerance 1.3× to absorb sampling noise)
    assert lers[0] >= lers[1] / 1.3, f"LER not monotone: {lers}"
    assert lers[1] >= lers[2] / 1.3, f"LER not monotone: {lers}"


@pytest.mark.parametrize("code_index", [0, 1, 2, 3])
def test_joint_xx_in_stabilizer_on_webster_intracode(code_index):
    """Webster BB codes 0..3 intra-code: (x_1, x_2 padded, 0...) is in rowspan(H_X^merged).

    Replaces deleted path-graph tests; pins the SkipTree adapter construction across
    the full Webster Table I code family rather than just code 0.
    """
    import galois
    from qldpc.codes.surgery.gadget import (
        build_gadget,
    )
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import _stitch_to_joint_csscode
    GF2 = galois.GF(2)
    data = load_webster_seed_set(code_index)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x1 = _webster_x_bar_operator(data, "X_bar_1")
    x2 = _webster_x_bar_operator(data, "X_bar_k2p1")
    g_l = build_gadget(code, x1, basis=Pauli.X)
    g_r = build_gadget(code, x2, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    joint = np.zeros(HX.shape[1], dtype=np.int_)
    n = code.num_qudits
    joint[:n] = (x1 + x2) % 2
    augmented = np.vstack([HX, joint.reshape(1, -1)])
    assert np.linalg.matrix_rank(GF2(HX.tolist())) == np.linalg.matrix_rank(
        GF2(augmented.tolist())
    )


def test_build_joint_ppm_circuit_intracode_noiseless_observables_zero():
    """Intra-code Webster joint X̄_1·X̄_{k/2+1}: noiseless detectors + observables = 0.

    Replaces deleted path-graph noiseless intracode tests.
    """
    from qldpc.codes.surgery.gadget import (
        build_gadget,
    )
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import build_joint_ppm_circuit
    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x1 = _webster_x_bar_operator(data, "X_bar_1")
    x2 = _webster_x_bar_operator(data, "X_bar_k2p1")
    g_l = build_gadget(code, x1, basis=Pauli.X)
    g_r = build_gadget(code, x2, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    circuit, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=2)
    sampler = circuit.compile_detector_sampler()
    dets, obs = sampler.sample(8, separate_observables=True)
    assert dets.sum() == 0, "noiseless intra-code: detectors should not fire"
    assert obs.sum() == 0, "noiseless intra-code: observables should be 0"
