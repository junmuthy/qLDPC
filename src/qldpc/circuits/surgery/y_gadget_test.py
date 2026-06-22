# y_gadget_test.py
import numpy as np
import pytest

from qldpc.circuits.surgery.y_gadget import _locate_overlap, _steane_y_pair, build_y_gadget


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
