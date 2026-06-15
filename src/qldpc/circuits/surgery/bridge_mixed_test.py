"""Mixed-basis Bridge integration tests
(Webster–Smith–Cohen arXiv:2511.15989 §II.B.2).
"""

from __future__ import annotations

import numpy as np

from qldpc import codes
from qldpc.objects import Pauli


def test_build_bridge_accepts_mixed_basis_steane() -> None:
    """build_bridge no longer rejects g_l.basis=X with g_r.basis=Z."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(code, z, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)
    assert bridge.basis_l is Pauli.X
    assert bridge.basis_r is Pauli.Z


def test_build_bridge_mixed_basis_y_stab_unpopulated_until_stitch() -> None:
    """Mixed-basis build_bridge defers merge to stitch — Y_stab is None at this point."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(code, z, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)
    assert bridge.Y_stab is None
    assert bridge.merge_qubits == ()


def test_build_bridge_same_basis_y_stab_remains_none() -> None:
    """Existing same-basis path is bit-for-bit unchanged: Y_stab is None."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(code, x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    assert bridge.basis_l is Pauli.X
    assert bridge.basis_r is Pauli.X
    assert bridge.basis is Pauli.X  # backward-compat property
    assert bridge.Y_stab is None
    assert bridge.merge_qubits == ()


def test_build_bridge_mixed_basis_uses_basis_l_for_skiptree() -> None:
    """When basis_l != basis_r, the SkipTree adapter uses basis_l (deterministic).
    T_l / T_r shapes must match the basis_l-built augmented gadgets."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(code, z, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)
    # Augmented gadgets carry the side's native basis (basis_l → g_l_aug.basis=X)
    assert bridge.g_l_aug.basis is Pauli.X
    assert bridge.g_r_aug.basis is Pauli.Z
    # SkipTree invariant per side still holds (T · F_aug · P = H_R)
    for side in ("l", "r"):
        T = getattr(bridge, f"T_{side}")
        g_aug = getattr(bridge, f"g_{side}_aug")
        label = getattr(bridge, f"label_{side}")
        adj = g_aug.incidence.astype(np.int_)
        P = np.zeros((adj.shape[1], bridge.width), dtype=np.int_)
        for v_idx, lab in enumerate(label):
            if lab >= 0:
                P[v_idx, lab] = 1
        assert np.array_equal((T @ adj @ P) % 2, bridge.H_R)


def test_stitch_intercode_mixed_basis_returns_quditcode() -> None:
    """Mixed-basis inter-code stitch returns a QuditCode (not a CSSCode)."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_code
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.codes.common import CSSCode, QuditCode

    code_l = codes.SteaneCode()
    code_r = codes.SteaneCode()  # distinct instance → intercode
    x = np.asarray(code_l.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code_r.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code_l, x, basis=Pauli.X)
    g_r = build_gadget(code_r, z, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)
    joint_code, bridge_populated = _stitch_to_joint_code(g_l, g_r, bridge)
    assert isinstance(joint_code, QuditCode)
    # CSSCode subclasses QuditCode, but the mixed-basis merged code is NOT CSS.
    assert not isinstance(joint_code, CSSCode)


def test_stitch_mixed_basis_populates_bridge_fields() -> None:
    """After mixed-basis stitch, bridge_populated records the gauge-extra block.

    NOTE: previously this asserted ``obs0_xor_map`` length matched
    ``Y_stab.shape[0]`` under the (incorrect) cross-merge approach. The
    correct construction per Cross-He-Rall-Yoder arXiv:2407.18393
    Appendix A.2 is a SUBSYSTEM code: there is no cross-merged Y stabilizer
    layer, and obs0_xor_map is unused. ``Y_stab`` is repurposed to record
    the |B| bridge-gauge + 2 SkipTree-cycle rows for documentation; the
    underlying joint code is a ``QuditCode(is_subsystem_code=True)``.
    """
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_code
    from qldpc.circuits.surgery.gadget import build_gadget

    code_l = codes.SteaneCode()
    code_r = codes.SteaneCode()
    x = np.asarray(code_l.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code_r.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code_l, x, basis=Pauli.X)
    g_r = build_gadget(code_r, z, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)
    _, bridge_populated = _stitch_to_joint_code(g_l, g_r, bridge)
    assert bridge_populated.Y_stab is not None
    assert bridge_populated.Y_stab.shape[0] >= 1  # gauge-extra rows recorded
    assert len(bridge_populated.merge_qubits) >= 1  # all w adapter columns
    # obs0_xor_map deliberately empty for the subsystem-code construction.
    assert bridge_populated.obs0_xor_map == ()


def test_stitch_mixed_basis_gauges_anticommute_on_adapter() -> None:
    """Subsystem-code property: χ_l (X-type) and χ_r (Z-type) anti-commute on
    shared adapter qubits, as required by Cross-He-Rall-Yoder arXiv:2407.18393
    Theorem 20.

    NOTE: this test REPLACES the previous ``test_stitch_mixed_basis_stabs_commute_symplectically``
    which asserted (incorrectly) that the merged-code generators all commute.
    The correct construction is a SUBSYSTEM code whose gauge generators may
    anti-commute; only the CENTER (the stabilizer group) commutes. qLDPC
    computes the center automatically via ``QuditCode(is_subsystem_code=True)``,
    so the merged code is well-defined despite anti-commuting generators.
    """
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_code
    from qldpc.circuits.surgery.gadget import build_gadget

    code_l = codes.SteaneCode()
    code_r = codes.SteaneCode()
    x = np.asarray(code_l.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code_r.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code_l, x, basis=Pauli.X)
    g_r = build_gadget(code_r, z, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)
    joint_code, _ = _stitch_to_joint_code(g_l, g_r, bridge)
    assert joint_code.is_subsystem_code  # gauge generators DO anti-commute
    H = np.asarray(joint_code.matrix).astype(np.int_)
    n = joint_code.num_qudits
    Hx = H[:, :n]
    Hz = H[:, n:]
    comm = (Hx @ Hz.T + Hz @ Hx.T) % 2
    # At least one pair anti-commutes — that's the structural signature of
    # a subsystem code. (For a stabilizer code we would assert ``not comm.any()``.)
    assert comm.any(), (
        "mixed-basis joint code should have anti-commuting gauge generators "
        "per Cross et al. arXiv:2407.18393 Theorem 20; got all-commuting "
        "matrix which suggests the construction degenerated to a stabilizer "
        "code."
    )


def test_stitch_mixed_basis_dimension_equals_k_l_plus_k_r_minus_one() -> None:
    """Mixed-basis joint code dim = k_l + k_r - 1 (one logical fixed by joint Z̄_l X̄_r)."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_code
    from qldpc.circuits.surgery.gadget import build_gadget

    code_l = codes.SteaneCode()
    code_r = codes.SteaneCode()
    x = np.asarray(code_l.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code_r.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code_l, x, basis=Pauli.X)
    g_r = build_gadget(code_r, z, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)
    joint_code, _ = _stitch_to_joint_code(g_l, g_r, bridge)
    expected_dim = code_l.dimension + code_r.dimension - 1
    assert joint_code.dimension == expected_dim, (
        f"expected dim {expected_dim}, got {joint_code.dimension}"
    )


def test_stitch_same_basis_still_returns_csscode() -> None:
    """Backward-compat: same-basis stitch returns a CSSCode (unchanged behavior)."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_code
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.codes.common import CSSCode

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(code, x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    joint_code, bridge_populated = _stitch_to_joint_code(g_l, g_r, bridge)
    assert isinstance(joint_code, CSSCode)
    assert bridge_populated.Y_stab is None
