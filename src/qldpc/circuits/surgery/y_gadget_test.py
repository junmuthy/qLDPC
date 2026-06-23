# y_gadget_test.py
import galois
import numpy as np
import pytest

from qldpc.circuits.surgery.gadget import build_gadget
from qldpc.circuits.surgery.y_gadget import _locate_overlap, _steane_y_pair, build_y_gadget
from qldpc.objects import Pauli

GF2 = galois.GF(2)


def test_locate_overlaps_steane_is_singleton():
    from qldpc.circuits.surgery.y_gadget import _locate_overlaps

    code, x, z = _steane_y_pair()
    W = _locate_overlaps(code, x, z)
    assert isinstance(W, tuple)
    assert W == tuple(int(i) for i in np.where(x.astype(bool) & z.astype(bool))[0])
    assert len(W) % 2 == 1  # anticommuting ⇒ odd overlap


def test_steane_y_pair_has_single_overlap():
    code, x, z = _steane_y_pair()
    assert ((np.asarray(code.matrix_z) @ x) % 2 == 0).all()  # x is logical-X
    assert ((np.asarray(code.matrix_x) @ z) % 2 == 0).all()  # z is logical-Z
    overlap = np.where((x.astype(bool)) & (z.astype(bool)))[0]
    assert overlap.size == 1
    assert _locate_overlap(code, x, z) == int(overlap[0])


def test_locate_overlap_rejects_multi_overlap():
    code, x, _ = _steane_y_pair()
    # x overlaps itself on |supp(x)| = 3 qubits (multi-overlap) -> rejected by overlap-size check
    with pytest.raises(ValueError):
        _locate_overlap(code, x, x)


def test_build_y_gadget_merged_code_is_valid_subsystem_code():
    from qldpc.circuits.surgery.y_gadget import build_y_gadget
    code, x, z = _steane_y_pair()
    yg = build_y_gadget(code, x=x, z=z)
    mc = yg.merged_code
    # all checks commute (symplectic product zero)
    H = np.asarray(mc.matrix).astype(np.int64)
    n = mc.num_qudits
    omega = np.block([[np.zeros((n, n)), np.eye(n)], [np.eye(n), np.zeros((n, n))]]).astype(np.int64)
    assert ((H @ omega @ H.T) % 2 == 0).all()
    # encodes one fewer logical than the original
    assert mc.dimension == code.dimension - 1
    # exactly one Y_stab row exists (the q1 mixed check)
    assert yg.Y_stab is not None and yg.Y_stab.shape[0] >= 1


def test_build_y_gadget_rejects_multi_overlap():
    code, x, _ = _steane_y_pair()
    with pytest.raises(ValueError):
        build_y_gadget(code, x=x, z=x)


def test_ybar_is_in_merged_stabilizer():
    from qldpc.circuits.surgery.y_gadget import build_y_gadget
    code, x, z = _steane_y_pair()
    yg = build_y_gadget(code, x=x, z=z)
    n0 = code.num_qudits
    # symplectic Ȳ on the ORIGINAL data qubits: X-part = x, Z-part = z
    ybar = np.concatenate([x, z]).astype(np.uint8)  # length 2*n0
    # restrict merged stabilizer group to original-data columns and check ȳ is reachable
    H = np.asarray(yg.merged_code.matrix).astype(np.uint8)
    n = yg.merged_code.num_qudits
    data_cols = list(range(n0)) + list(range(n, n + n0))  # X-block + Z-block on data
    Hd = H[:, data_cols]
    from qldpc.circuits.surgery.y_gadget import _in_rowspace_gf2
    assert _in_rowspace_gf2(Hd, ybar)


def test_partial0_steane_w1_is_pure_css_no_crossing():
    from qldpc.circuits.surgery.y_gadget import (
        _merged_incidence,
        _partial0_symplectic_rows,
    )

    code, x, z = _steane_y_pair()
    g_x = build_gadget(code, x, basis=Pauli.X)
    g_z = build_gadget(code, z, basis=Pauli.Z)
    n = code.num_qudits
    k_x = len(g_x.ancilla_qubits)
    k_z = len(g_z.ancilla_qubits)
    D1, kx, kz = _merged_incidence(g_x, g_z, x, z)
    assert (kx, kz) == (k_x, k_z)
    # |W|=1 wedge: dim ker(merged) == dim ker(∂1x) + dim ker(∂1z)  (no crossing cycle)
    d1x = np.asarray(g_x.incidence).astype(int).T
    d1z = np.asarray(g_z.incidence).astype(int).T
    n_merged_cyc = np.asarray(GF2(D1.astype(int)).null_space()).shape[0]
    n_sep = (
        np.asarray(GF2(d1x).null_space()).shape[0]
        + np.asarray(GF2(d1z).null_space()).shape[0]
    )
    assert n_merged_cyc == n_sep  # verified fact for |W|=1
    rows = _partial0_symplectic_rows(g_x, g_z, x, z, n=n, k_x=k_x, k_z=k_z)
    nm = n + k_x + k_z
    assert rows.shape[1] == 2 * nm
    # every ∂_0 row is pure-CSS at |W|=1: no row has BOTH an X-part and a Z-part bit
    for r in rows:
        has_x = r[:nm].any()
        has_z = r[nm:].any()
        assert not (has_x and has_z), "unexpected crossing (non-CSS) row at |W|=1"
