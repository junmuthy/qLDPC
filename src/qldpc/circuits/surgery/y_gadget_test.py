# y_gadget_test.py
import numpy as np
import pytest

from qldpc.circuits.surgery.y_gadget import _locate_overlap, _steane_y_pair


def test_steane_y_pair_has_single_overlap():
    code, x, z = _steane_y_pair()
    assert ((np.asarray(code.matrix_z) @ x) % 2 == 0).all()  # x is logical-X
    assert ((np.asarray(code.matrix_x) @ z) % 2 == 0).all()  # z is logical-Z
    overlap = np.where((x.astype(bool)) & (z.astype(bool)))[0]
    assert overlap.size == 1
    assert _locate_overlap(code, x, z) == int(overlap[0])


def test_locate_overlap_rejects_multi_overlap():
    code, x, _ = _steane_y_pair()
    # x overlaps itself on |supp(x)| > 1 qubits and (x,x) commute -> rejected
    with pytest.raises(ValueError):
        _locate_overlap(code, x, x)
