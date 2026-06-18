"""Tests for joint_layout.py — block-by-block joint PPM construction per main.tex §4."""

from __future__ import annotations

import numpy as np

from qldpc import codes
from qldpc.circuits.surgery.joint_layout import JointPPMLayout, column_slices_for_bridge
from qldpc.circuits.surgery.joint_layout import apply_cross_merge, build_pre_merge_layout
from qldpc.objects import Pauli


def test_layout_dataclass_construction() -> None:
    """JointPPMLayout holds three stabilizer matrices + provenance dicts."""
    H_X = np.zeros((2, 10), dtype=np.uint8)
    H_Z = np.zeros((3, 10), dtype=np.uint8)
    H_Y = np.zeros((1, 20), dtype=np.uint8)
    layout = JointPPMLayout(
        H_X=H_X,
        H_Z=H_Z,
        H_Y=H_Y,
        rows_data_x={"l": (0,), "r": (1,)},
        rows_data_z={"l": (0,), "r": (1,)},
        rows_chi={"l": (), "r": ()},
        rows_gauge={"l": (), "r": ()},
        rows_cycle={"l": (), "r": (2,)},
        rows_y=(0,),
        basis_l=Pauli.Z,
        basis_r=Pauli.X,
        column_slices={
            "Q_l": slice(0, 3),
            "Q_r": slice(3, 6),
            "k_l": slice(6, 7),
            "k_r": slice(7, 8),
            "A": slice(8, 10),
        },
    )
    assert layout.H_X.shape == (2, 10)
    assert layout.H_Z.shape == (3, 10)
    assert layout.H_Y.shape == (1, 20)
    assert layout.basis_l is Pauli.Z
    assert layout.column_slices["Q_l"] == slice(0, 3)


def test_column_slices_for_steane_pair() -> None:
    """Column slices partition (Q_l | Q_r | k_l | k_r | A) for an inter-code pair."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget

    code_l = codes.SteaneCode()
    code_r = codes.SteaneCode()
    x = np.asarray(code_l.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code_r.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code_l, x, basis=Pauli.X)
    g_r = build_gadget(code_r, z, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)

    slices = column_slices_for_bridge(g_l, g_r, bridge)
    n_l = g_l.code.num_qudits
    n_r = g_r.code.num_qudits
    k_l = bridge.g_l_aug.incidence.shape[0]
    k_r = bridge.g_r_aug.incidence.shape[0]
    w = bridge.width

    assert slices["Q_l"] == slice(0, n_l)
    assert slices["Q_r"] == slice(n_l, n_l + n_r)
    assert slices["k_l"] == slice(n_l + n_r, n_l + n_r + k_l)
    assert slices["k_r"] == slice(n_l + n_r + k_l, n_l + n_r + k_l + k_r)
    assert slices["A"] == slice(n_l + n_r + k_l + k_r, n_l + n_r + k_l + k_r + w)


def test_block_data_x_left_carries_f_X_on_kappa_l_for_basis_l_Z() -> None:
    """Left H_X^l data rows in mixed-basis Z⊗X must extend f_X^l = π_{C_0^l}^T into κ^l."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.joint_layout import _block_data

    code_l = codes.SteaneCode()
    code_r = codes.SteaneCode()
    z = np.asarray(code_l.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    x = np.asarray(code_r.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code_l, z, basis=Pauli.Z)
    g_r = build_gadget(code_r, x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    slices = column_slices_for_bridge(g_l, g_r, bridge)
    N = slices["A"].stop

    block = _block_data(g_l, basis_block=Pauli.X, side="l", slices=slices, N=N)
    m_X_l = g_l.code.matrix_x.shape[0]
    assert block.shape == (m_X_l, N)
    # Data side equals original H_X^l on Q_l columns.
    assert (block[:, slices["Q_l"]] == np.asarray(g_l.code.matrix_x).astype(np.uint8)).all()
    # κ^l columns: f_X^l = π_{C_0^l}^T extends H_X^l rows whose index is in C_0^l.
    # C_0^l = X-checks touching V_0^l on the left's Z-gadget.
    c_0_l = list(g_l.data_checks)
    f_X_l_expected = np.zeros((m_X_l, bridge.g_l_aug.incidence.shape[0]), dtype=np.uint8)
    for k, j in enumerate(c_0_l):
        f_X_l_expected[j, k] = 1
    assert (block[:, slices["k_l"]] == f_X_l_expected).all()
    # Everything else zero.
    assert not block[:, slices["Q_r"]].any()
    assert not block[:, slices["k_r"]].any()
    assert not block[:, slices["A"]].any()


def test_block_data_skips_sentinel_minus_one_indices_for_augmented_gadget() -> None:
    """If caller passes an augmented gadget (where data_checks contains -1 sentinels
    from build_gadget_augmented), _block_data must skip those rows rather than wrap
    around via NumPy negative indexing into the last row of the block.

    The Steane×Steane bridge happens not to require connectivity/cellulation
    augmentation, so bridge.g_l_aug.data_checks has no sentinels on that fixture.
    To make this regression deterministic we call build_gadget_augmented directly
    with a non-empty incidence_extra — that is exactly the code path the bug
    lives in (gadget.py:257 emits -1 sentinels).
    """
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget, build_gadget_augmented
    from qldpc.circuits.surgery.joint_layout import _block_data

    code_l = codes.SteaneCode()
    code_r = codes.SteaneCode()
    z = np.asarray(code_l.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    x = np.asarray(code_r.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code_l, z, basis=Pauli.Z)
    g_r = build_gadget(code_r, x, basis=Pauli.X)
    # Force sentinel-bearing augmentation by adding two weight-2 incidence_extra
    # rows directly (these would normally come from connectivity/cellulation
    # boosting). build_gadget_augmented appends (-1, -1) to data_checks.
    n_supp = len(g_l.support)
    incidence_extra = np.zeros((2, n_supp), dtype=np.uint8)
    incidence_extra[0, 0] = 1
    incidence_extra[0, 1] = 1
    incidence_extra[1, 1] = 1
    incidence_extra[1, 2] = 1
    g_l_aug = build_gadget_augmented(code_l, z, incidence_extra, basis=Pauli.Z)
    assert any(j < 0 for j in g_l_aug.data_checks), "fixture must contain -1 sentinels"

    # Build slices over g_l_aug's actual incidence row count (k_l grows by 2).
    bridge = build_bridge(g_l, g_r)
    n_l = g_l.code.num_qudits
    n_r = g_r.code.num_qudits
    k_l = g_l_aug.incidence.shape[0]
    k_r = bridge.g_r_aug.incidence.shape[0]
    w = bridge.width
    slices = {
        "Q_l": slice(0, n_l),
        "Q_r": slice(n_l, n_l + n_r),
        "k_l": slice(n_l + n_r, n_l + n_r + k_l),
        "k_r": slice(n_l + n_r + k_l, n_l + n_r + k_l + k_r),
        "A": slice(n_l + n_r + k_l + k_r, n_l + n_r + k_l + k_r + w),
    }
    N = slices["A"].stop

    block = _block_data(g_l_aug, basis_block=Pauli.X, side="l", slices=slices, N=N)
    m_X_l = g_l_aug.code.matrix_x.shape[0]
    # The LAST row of H_X^l must NOT have been wraparound-corrupted: every entry
    # in its κ^l block must come from a legitimate C_0 hit (i.e. real check
    # index == m_X_l - 1), never from a -1 sentinel masquerading as the last row.
    legit_kappa_positions = [
        k for k, j in enumerate(g_l_aug.data_checks)
        if 0 <= j and j == m_X_l - 1
    ]
    actual_last_row_kappa = np.flatnonzero(block[m_X_l - 1, slices["k_l"]])
    assert sorted(actual_last_row_kappa.tolist()) == sorted(legit_kappa_positions)


def test_block_chi_left_z_basis_attaches_adapter_label() -> None:
    """χ_l rows (basis_l=Z) carry (π_{V_0^l} | H_Z'^{l,aug} | π_{P_l}^T P_{σ_l}) on (Q_l | k_l | A)."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.joint_layout import _block_chi

    code_l = codes.SteaneCode()
    code_r = codes.SteaneCode()
    z = np.asarray(code_l.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    x = np.asarray(code_r.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code_l, z, basis=Pauli.Z)
    g_r = build_gadget(code_r, x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    slices = column_slices_for_bridge(g_l, g_r, bridge)
    N = slices["A"].stop

    block = _block_chi(bridge.g_l_aug, side="l", slices=slices, N=N,
                       labels=bridge.label_l)
    n_V0_l = len(g_l.support)
    assert block.shape == (n_V0_l, N)
    # Q_l side: π_{V_0^l} — row i has 1 at column V_0^l[i].
    for i, v in enumerate(g_l.support):
        assert block[i, slices["Q_l"].start + v] == 1
    # k_l side: H_Z'^{l,aug} (incidence^T).
    assert (block[:, slices["k_l"]] == bridge.g_l_aug.incidence.T).all()
    # Adapter columns: row i has 1 at column A.start + label_l[i] when label_l[i] >= 0,
    # 0 otherwise.
    for i, lab in enumerate(bridge.label_l):
        if lab >= 0:
            assert block[i, slices["A"].start + lab] == 1
        else:
            assert not block[i, slices["A"]].any()
    # No support on Q_r / k_r.
    assert not block[:, slices["Q_r"]].any()
    assert not block[:, slices["k_r"]].any()


def test_block_gauge_supports_only_kappa() -> None:
    """Gauge rows H_{X/Z}'^{s,aug} have support only on κ^s."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.joint_layout import _block_gauge

    code_l = codes.SteaneCode()
    code_r = codes.SteaneCode()
    z = np.asarray(code_l.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    x = np.asarray(code_r.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code_l, z, basis=Pauli.Z)
    g_r = build_gadget(code_r, x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    slices = column_slices_for_bridge(g_l, g_r, bridge)
    N = slices["A"].stop

    block = _block_gauge(bridge.g_l_aug, side="l", slices=slices, N=N)
    r_l = bridge.g_l_aug.gauge.shape[0]
    assert block.shape == (r_l, N)
    assert (block[:, slices["k_l"]] == np.asarray(bridge.g_l_aug.gauge).astype(np.uint8)).all()
    assert not block[:, slices["Q_l"]].any()
    assert not block[:, slices["Q_r"]].any()
    assert not block[:, slices["k_r"]].any()
    assert not block[:, slices["A"]].any()


def test_block_cycle_carries_T_on_kappa_and_H_R_on_adapter() -> None:
    """Cycle row T_s on κ^s + H_R on adapter."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.joint_layout import _block_cycle

    code_l = codes.SteaneCode()
    code_r = codes.SteaneCode()
    z = np.asarray(code_l.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    x = np.asarray(code_r.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code_l, z, basis=Pauli.Z)
    g_r = build_gadget(code_r, x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    slices = column_slices_for_bridge(g_l, g_r, bridge)
    N = slices["A"].stop

    block = _block_cycle(bridge.T_l, bridge.H_R, side="l", slices=slices, N=N)
    w_minus_1 = bridge.H_R.shape[0]
    assert block.shape == (w_minus_1, N)
    assert (block[:, slices["k_l"]] == bridge.T_l).all()
    assert (block[:, slices["A"]] == bridge.H_R).all()
    assert not block[:, slices["Q_l"]].any()
    assert not block[:, slices["Q_r"]].any()
    assert not block[:, slices["k_r"]].any()


def test_pre_merge_layout_row_counts_mixed_basis_steane() -> None:
    """Mixed-basis (Z̄_l ⊗ X̄_r) pre-merge: row counts match §4.2 expectations."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget

    code_l = codes.SteaneCode()
    code_r = codes.SteaneCode()
    z = np.asarray(code_l.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    x = np.asarray(code_r.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code_l, z, basis=Pauli.Z)
    g_r = build_gadget(code_r, x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)

    layout = build_pre_merge_layout(g_l, g_r, bridge)

    m_X_l = g_l.code.matrix_x.shape[0]
    m_X_r = g_r.code.matrix_x.shape[0]
    m_Z_l = g_l.code.matrix_z.shape[0]
    m_Z_r = g_r.code.matrix_z.shape[0]
    n_V0_l = len(g_l.support)
    n_V0_r = len(g_r.support)
    r_l_aug = bridge.g_l_aug.gauge.shape[0]
    r_r_aug = bridge.g_r_aug.gauge.shape[0]
    w_minus_1 = bridge.H_R.shape[0]

    # H_X rows: H_X^l + H_X^r + gauge_l + chi_r + cycle_l
    assert layout.H_X.shape[0] == m_X_l + m_X_r + r_l_aug + n_V0_r + w_minus_1
    # H_Z rows: H_Z^l + H_Z^r + chi_l + gauge_r + cycle_r
    assert layout.H_Z.shape[0] == m_Z_l + m_Z_r + n_V0_l + r_r_aug + w_minus_1
    # Pre-merge: H_Y empty.
    assert layout.H_Y.shape == (0, 2 * layout.column_slices["A"].stop)

    # Provenance: chi_l rows live in H_Z (basis_l=Z), chi_r in H_X (basis_r=X).
    assert len(layout.rows_chi["l"]) == n_V0_l
    assert len(layout.rows_chi["r"]) == n_V0_r
    assert layout.rows_y == ()


def test_cross_merge_deletes_port_rows_and_builds_w_y_rows() -> None:
    """Cross-merge deletes one port row from H_X and H_Z each, builds w y_q rows."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget

    code_l = codes.SteaneCode()
    code_r = codes.SteaneCode()
    z = np.asarray(code_l.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    x = np.asarray(code_r.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code_l, z, basis=Pauli.Z)
    g_r = build_gadget(code_r, x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    pre = build_pre_merge_layout(g_l, g_r, bridge)

    post = apply_cross_merge(pre, bridge)
    w = bridge.width

    # H_X loses w port-χ_r rows; H_Z loses w port-χ_l rows.
    assert post.H_X.shape[0] == pre.H_X.shape[0] - w
    assert post.H_Z.shape[0] == pre.H_Z.shape[0] - w
    # H_Y has w rows.
    assert post.H_Y.shape == (w, 2 * pre.column_slices["A"].stop)
    # rows_y == range(w).
    assert post.rows_y == tuple(range(w))
    # Surviving χ_l = non-port chi rows.
    surviving_chi_l = len(post.rows_chi["l"])
    surviving_chi_r = len(post.rows_chi["r"])
    assert surviving_chi_l == len(pre.rows_chi["l"]) - w
    assert surviving_chi_r == len(pre.rows_chi["r"]) - w
