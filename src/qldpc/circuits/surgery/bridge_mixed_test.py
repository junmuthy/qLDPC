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
