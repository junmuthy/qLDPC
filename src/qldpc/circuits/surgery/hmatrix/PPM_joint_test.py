"""Tests for src/qldpc/circuits/surgery/hmatrix/PPM_joint.py."""

from __future__ import annotations

import numpy as np
import pytest

from qldpc import codes
from qldpc.circuits.surgery.conftest import (
    _webster_z_bar_operator,
    build_generalised_bicycle_code,
    load_webster_seed_set,
)
from qldpc.objects import Pauli


def test_bridge_dataclass_fields_universal_adapter() -> None:
    """Bridge dataclass exposes the universal-adapter fields + mixed-basis fields.

    basis_l/basis_r replace the old single basis field (Webster–Smith–Cohen
    arXiv:2511.15989 §II.B.2 mixed-basis joint PPM).
    """
    import dataclasses

    from qldpc.circuits.surgery.hmatrix.PPM_joint import Bridge

    fields = {f.name for f in dataclasses.fields(Bridge)}
    assert fields == {
        "width",
        "basis_l",
        "basis_r",
        "port_l",
        "port_r",
        "label_l",
        "label_r",
        "extra_ancilla_l",
        "extra_ancilla_r",
        "T_l",
        "T_r",
        "H_R",
        "g_l_aug",
        "g_r_aug",
        "Y_stab",
        "merge_qubits",
        "obs0_xor_map",
        "x_leftover_indices",
        "z_leftover_indices",
    }


def test_bridge_basis_property_returns_single_basis_when_same() -> None:
    """`.basis` returns the shared basis when basis_l == basis_r (backward compat)."""
    from qldpc.circuits.surgery.hmatrix.PPM_joint import Bridge

    bridge = Bridge(
        width=2,
        basis_l=Pauli.X,
        basis_r=Pauli.X,
        port_l=(0, 1),
        port_r=(0, 1),
        label_l=(0, 1),
        label_r=(0, 1),
        extra_ancilla_l=np.zeros((0, 2), dtype=np.uint8),
        extra_ancilla_r=np.zeros((0, 2), dtype=np.uint8),
        T_l=np.zeros((1, 1), dtype=np.int_),
        T_r=np.zeros((1, 1), dtype=np.int_),
        H_R=np.array([[1, 1]], dtype=np.int_),
        g_l_aug=None,  # opaque to this test
        g_r_aug=None,
    )
    assert bridge.basis is Pauli.X


def test_bridge_basis_property_raises_when_mixed() -> None:
    """`.basis` raises AttributeError when basis_l != basis_r."""
    from qldpc.circuits.surgery.hmatrix.PPM_joint import Bridge

    bridge = Bridge(
        width=2,
        basis_l=Pauli.X,
        basis_r=Pauli.Z,
        port_l=(0, 1),
        port_r=(0, 1),
        label_l=(0, 1),
        label_r=(0, 1),
        extra_ancilla_l=np.zeros((0, 2), dtype=np.uint8),
        extra_ancilla_r=np.zeros((0, 2), dtype=np.uint8),
        T_l=np.zeros((1, 1), dtype=np.int_),
        T_r=np.zeros((1, 1), dtype=np.int_),
        H_R=np.array([[1, 1]], dtype=np.int_),
        g_l_aug=None,
        g_r_aug=None,
    )
    with pytest.raises(AttributeError, match=r"mixed-basis|basis_l|basis_r"):
        _ = bridge.basis


def test_bridge_mixed_basis_fields_default_to_none_or_empty() -> None:
    """Y_stab defaults to None; merge_qubits/obs0_xor_map/leftover tuples default to ()."""
    from qldpc.circuits.surgery.hmatrix.PPM_joint import Bridge

    bridge = Bridge(
        width=2,
        basis_l=Pauli.X,
        basis_r=Pauli.X,
        port_l=(0, 1),
        port_r=(0, 1),
        label_l=(0, 1),
        label_r=(0, 1),
        extra_ancilla_l=np.zeros((0, 2), dtype=np.uint8),
        extra_ancilla_r=np.zeros((0, 2), dtype=np.uint8),
        T_l=np.zeros((1, 1), dtype=np.int_),
        T_r=np.zeros((1, 1), dtype=np.int_),
        H_R=np.array([[1, 1]], dtype=np.int_),
        g_l_aug=None,
        g_r_aug=None,
    )
    assert bridge.Y_stab is None
    assert bridge.merge_qubits == ()
    assert bridge.obs0_xor_map == ()
    assert bridge.x_leftover_indices == ()
    assert bridge.z_leftover_indices == ()


def test_build_bridge_smoke_steane_intracode() -> None:
    """Steane × Steane intra-code joint X̄ X̄: build_bridge returns valid Bridge."""
    from qldpc.circuits.surgery.hmatrix.PPM_joint import build_bridge
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    x1 = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    x2 = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)  # same logical
    g_l = build_gadget(code, x1, basis=Pauli.X)
    g_r = build_gadget(code, x2, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    assert bridge.width == min(len(g_l.support), len(g_r.support))
    assert bridge.basis is Pauli.X
    assert len(bridge.port_l) == bridge.width
    assert len(bridge.port_r) == bridge.width
    assert bridge.T_l.shape == (bridge.width - 1, bridge.g_l_aug.incidence.shape[0])
    assert bridge.T_r.shape == (bridge.width - 1, bridge.g_r_aug.incidence.shape[0])
    assert bridge.H_R.shape == (bridge.width - 1, bridge.width)


def _assert_skiptree_invariant(bridge, msg=None) -> None:
    """Shared SkipTree identity: T_s · (augmented incidence) · P_{σ_s} % 2 == H_R.

    Verifies the core SkipTree invariant for both sides s ∈ {l, r}.  Used by the
    plain, boosted, and duplicate-incidence-row regression tests — the only
    per-test variation is the *fixture* that produces ``bridge`` and the
    optional failure ``msg`` (a callable ``side -> str``).
    """
    for side in ("l", "r"):
        T = getattr(bridge, f"T_{side}")
        g_aug = getattr(bridge, f"g_{side}_aug")
        label = getattr(bridge, f"label_{side}")
        # adjacency = incidence_aug (rows = edges = ancilla qubits, cols = support vertices)
        adjacency = g_aug.incidence.astype(np.int_)
        # P_s: |V_0^(s)| × w; P_s[v, k] = 1 iff v ∈ port AND label[v] == k
        P = np.zeros((adjacency.shape[1], bridge.width), dtype=np.int_)
        for v_idx, lab in enumerate(label):
            if lab >= 0:
                P[v_idx, lab] = 1
        lhs = (T @ adjacency @ P) % 2
        detail = msg(side) if msg is not None else f"side {side}:\n{lhs}\nvs\n{bridge.H_R}"
        assert np.array_equal(lhs, bridge.H_R), detail


def test_build_bridge_skiptree_invariant_holds() -> None:
    """T_s · G_s_aug · P_s = H_R for both sides on Steane × Steane."""
    from qldpc.circuits.surgery.hmatrix.PPM_joint import build_bridge
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(code, x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)

    _assert_skiptree_invariant(bridge)


def test_build_bridge_bb18_hyperedge_and_long_cycle() -> None:
    """End-to-end: Cain bb_18 BBCode triggers both Bug 1 (hyperedge) and
    Bug 2 (long port-subgraph cycle). build_bridge must succeed and produce
    a merged code with k_merged = k_orig - 1 (intra-code joint Z̄_1 ⊗ Z̄_2).

    Two *distinct* Z-logicals are used so that the joint measurement reduces k
    by exactly 1.  Z-logical 0 has a weight-4 F row (triggers Bug 1); the pair
    together exercises the full _cellulate_port_subgraph path (Bug 2)."""
    import sympy

    from qldpc.circuits.surgery import build_bridge, build_gadget
    from qldpc.circuits.surgery.hmatrix.PPM_joint import _joint_merged_dispatch

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
    row_weights = np.asarray(g_l.incidence.sum(axis=1)).ravel().astype(int).tolist()
    assert max(row_weights) >= 4, "Test no longer triggers Bug 1 (no hyperedge)"
    # Build bridge (this used to raise NotImplementedError or RuntimeError)
    bridge = build_bridge(g_l, g_r)
    # Merged code construction must succeed
    merged = _joint_merged_dispatch(g_l, g_r, bridge)
    # Intra-code joint Z̄_1 ⊗ Z̄_2: k_merged == k_orig − 1
    assert merged.dimension == code.dimension - 1
    # CSS commutation on merged code
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    HZ = np.asarray(merged.matrix_z).astype(np.int_)
    assert not ((HX @ HZ.T) % 2).any(), "CSS commutation broken on merged code"


def test_adapter_cycle_check_weight_bounded() -> None:
    """Each new cycle-X row has weight <= 8 (SkipTree (3,2) + H_R weight 2). Basis=Z.

    For basis=Z, the new adapter cycle checks are placed in HX (the last w-1 rows).
    Each row has the form [T_l | H_R | T_r]:
      - T_l row: at most 3 entries on cl_ancilla (SkipTree (3,2)-sparsity)
      - H_R row: exactly 2 entries on c_adapter (canonical rep code)
      - T_r row: at most 3 entries on cr_ancilla (SkipTree (3,2)-sparsity)
    Total: weight <= 3 + 2 + 3 = 8.
    """
    from qldpc.circuits.surgery.hmatrix.PPM_joint import (
        _joint_merged_dispatch,
        build_bridge,
    )
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import (
        build_gadget,
    )

    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    # Use Z̄_1 for both sides (intra-code, same logical) — bridge.width =
    # |V_0| = weight of Z̄_1, exercising the maximum-width cellulation path
    x = _webster_z_bar_operator(data, "Z_bar_1")
    g_l = build_gadget(code, x, basis=Pauli.Z)
    g_r = build_gadget(code, x, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)
    merged = _joint_merged_dispatch(g_l, g_r, bridge)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    # basis=Z: new cycle-X-checks are the last (w-1) rows of HX
    new_x_rows = HX[-(bridge.width - 1) :, :]
    max_w = int(new_x_rows.sum(axis=1).max())
    assert max_w <= 8, f"max new cycle-X weight {max_w} > 8"


def test_cellulation_caps_aug_aux_cycle_length_on_webster() -> None:
    """After cellulation, every basis cycle in the augmented aux graph has length <= 6."""
    import networkx as nx

    from qldpc.circuits.surgery.hmatrix.PPM_joint import build_bridge
    from qldpc.circuits.surgery.hmatrix.PPM_joint_cellulation import _build_aux_graph_strict
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import (
        build_gadget,
    )

    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x = _webster_z_bar_operator(data, "Z_bar_1")
    g_l = build_gadget(code, x, basis=Pauli.Z)
    g_r = build_gadget(code, x, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r, cellulate_max_len=6)
    # Cellulation is now scoped to the port subgraph (where SkipTree runs).
    # Inspect cycles on the induced port subgraph, not the full graph.
    G_aux, _ = _build_aux_graph_strict(bridge.g_l_aug.incidence)
    sub = G_aux.subgraph(bridge.port_l)
    cycles = nx.cycle_basis(sub)
    if cycles:
        assert max(len(c) for c in cycles) <= 6, (
            f"max port-subgraph cycle length {max(len(c) for c in cycles)} > 6"
        )


def test_cellulate_max_len_defaults_to_max_basis_stabilizer_weight() -> None:
    """Default cellulate_max_len = max H_basis row weight (not a hardcoded 6).

    Surface code d=5 has weight-4 H_X and H_Z stabilizers, so the default cap
    must be 4. Passing the same code with an explicit cap=4 must reproduce the
    same port-subgraph cycle structure as the default.
    """
    import networkx as nx

    from qldpc.circuits.surgery.hmatrix.PPM_joint import build_bridge
    from qldpc.circuits.surgery.hmatrix.PPM_joint_cellulation import _build_aux_graph_strict
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    sc = codes.SurfaceCode(5)
    x = np.asarray(sc.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(sc, x, basis=Pauli.X)
    g_r = build_gadget(sc, x, basis=Pauli.X)

    HX = np.asarray(sc.matrix_x).astype(int)
    expected_cap = int(HX.sum(axis=1).max())

    bridge_default = build_bridge(g_l, g_r)
    bridge_explicit = build_bridge(g_l, g_r, cellulate_max_len=expected_cap)

    # Both runs produce the same extra ancilla shape (cellulation count matches).
    assert bridge_default.extra_ancilla_l.shape == bridge_explicit.extra_ancilla_l.shape
    assert bridge_default.extra_ancilla_r.shape == bridge_explicit.extra_ancilla_r.shape

    # Port-subgraph basis cycles all <= expected_cap.
    G_aux, _ = _build_aux_graph_strict(bridge_default.g_l_aug.incidence)
    sub = G_aux.subgraph(bridge_default.port_l)
    cycles = nx.cycle_basis(sub)
    if cycles:
        assert max(len(c) for c in cycles) <= expected_cap


def test_build_bridge_rejects_width_below_2() -> None:
    """build_bridge rejects port subsets that intersect to width < 2."""
    from qldpc.circuits.surgery.hmatrix.PPM_joint import build_bridge
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    with pytest.raises(ValueError, match="width must be >= 2"):
        build_bridge(g, g, port_subset_l=(0,), port_subset_r=(0,))


def test_build_bridge_rejects_spanning_tree_root_out_of_range_left() -> None:
    """build_bridge rejects spanning_tree_root_l outside [0, width)."""
    from qldpc.circuits.surgery.hmatrix.PPM_joint import build_bridge
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    with pytest.raises(ValueError, match="spanning_tree_root_l=99"):
        build_bridge(g, g, spanning_tree_root_l=99)


def test_build_bridge_rejects_spanning_tree_root_out_of_range_right() -> None:
    """build_bridge rejects spanning_tree_root_r outside [0, width)."""
    from qldpc.circuits.surgery.hmatrix.PPM_joint import build_bridge
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    with pytest.raises(ValueError, match="spanning_tree_root_r=99"):
        build_bridge(g, g, spanning_tree_root_r=99)


def _bb_72_12():
    """Cain et al. arXiv:2603.28627 Table I `[[72, 12]]` BB code (cheeger h<1)."""
    import sympy

    xs, ys = sympy.symbols("x y")
    return codes.BBCode({xs: 6, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)


def test_build_bridge_skiptree_invariant_holds_after_boost() -> None:
    """T_s · F_aug · P_s = H_R must hold even when g_l/g_r are boosted.

    Regression: build_bridge rebuilds g_l_aug via _restrict on the
    ORIGINAL (un-boosted) code+x+basis, dropping boost-added κ' rows from
    g_l.incidence. SkipTree T_l is computed against the boosted G_aux but
    embedded into unboosted g_l_aug.incidence → tree edges through boost-κ'
    are silently zeroed in T_full → invariant fails → joint_code cycle
    stabilizers are bogus → non-deterministic detector in joint PPM DEM.
    """
    from qldpc.circuits.surgery.hmatrix.cheeger import boost_gadget
    from qldpc.circuits.surgery.hmatrix.PPM_joint import build_bridge
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    z = np.asarray(_bb_72_12().get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l_raw = build_gadget(_bb_72_12(), z, basis=Pauli.Z)
    g_r_raw = build_gadget(_bb_72_12(), z, basis=Pauli.Z)
    g_l = boost_gadget(
        g_l_raw, method="combinatorial", target=1.0, max_extra_qubits=20, seed=3
    )
    g_r = boost_gadget(
        g_r_raw, method="combinatorial", target=1.0, max_extra_qubits=20, seed=3
    )
    assert g_l.incidence.shape[0] > g_l_raw.incidence.shape[0], "boost should add κ' rows"
    bridge = build_bridge(g_l, g_r)

    _assert_skiptree_invariant(
        bridge,
        msg=lambda side: (
            f"side {side}: T·F_aug·P ≠ H_R after boost — bridge dropped boost κ' rows"
        ),
    )


def _bb_36_8():
    """BBCode (l=3, m=6) [[36, 8]] — has *duplicate* weight-2 incidence rows
    when restricted to Z̄_0, exercising _run_skiptree_on_port_subgraph's
    duplicate-edge guard."""
    import sympy

    xs, ys = sympy.symbols("x y")
    return codes.BBCode({xs: 3, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)


def test_build_bridge_skiptree_invariant_holds_with_duplicate_incidence_rows() -> None:
    """T_s · F_aug · P_s = H_R must hold when F_aug has duplicate weight-2 rows.

    Regression: BBCode [[36, 8]] restricted to Z̄_0 has h(F)=1 (no boost
    needed) but the restricted incidence has two κ rows sharing the same
    (u, v) support — _build_aux_graph_strict dedups them to one G_aux edge.
    Pre-fix, _run_skiptree_on_port_subgraph assigned the *same* T_relab
    column to both duplicate κ rows, so their contributions to T · F_aug
    cancel mod 2 → invariant fails → joint_code cycle stabilizer non-trivially
    anti-commutes with the gauge → non-deterministic detector.
    """
    from qldpc.circuits.surgery.hmatrix.PPM_joint import build_bridge
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code_l = _bb_36_8()
    code_r = _bb_36_8()
    z = np.asarray(code_l.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code_l, z, basis=Pauli.Z)
    g_r = build_gadget(code_r, z, basis=Pauli.Z)
    inc = g_l.incidence.astype(np.int_)
    assert inc.shape[0] > np.unique(inc, axis=0).shape[0], (
        "test premise broken: BB [[36, 8]] restricted incidence should have duplicates"
    )
    bridge = build_bridge(g_l, g_r)

    _assert_skiptree_invariant(
        bridge,
        msg=lambda side: (
            f"side {side}: T·F_aug·P ≠ H_R with duplicate κ rows — bridge "
            f"duplicate-edge guard missing"
        ),
    )


def test_build_joint_ppm_circuit_dem_deterministic_bb_36_8() -> None:
    """Joint PPM DEM constructs without non-deterministic detectors on BB [[36, 8]].

    End-to-end regression for the duplicate-edge bug: BB [[36, 8]] Z̄⊗Z̄ joint
    PPM (h=1, no boost) previously crashed stim DEM with non-deterministic
    detectors because the SkipTree invariant failed on duplicate incidence
    rows.
    """
    from qldpc.circuits.noise_model import DepolarizingNoiseModel
    from qldpc.circuits.surgery.circuit.PPM_joint import build_joint_ppm_circuit
    from qldpc.circuits.surgery.circuit.support import keep_only_observable
    from qldpc.circuits.surgery.hmatrix.PPM_joint import build_bridge
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code_l, code_r = _bb_36_8(), _bb_36_8()
    z = np.asarray(code_l.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code_l, z, basis=Pauli.Z)
    g_r = build_gadget(code_r, z, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)

    noise = DepolarizingNoiseModel(1e-3, include_idling_error=False)
    circuit, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=3, noise_model=noise)
    stripped = keep_only_observable(circuit, keep_idx=0)
    dem = stripped.detector_error_model(approximate_disjoint_errors=True)
    assert dem.num_detectors > 0
