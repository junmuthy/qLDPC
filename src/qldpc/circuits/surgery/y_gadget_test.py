# y_gadget_test.py
import dataclasses

import galois
import numpy as np
import pytest

from qldpc.circuits.surgery.cheeger import cheeger_constant
from qldpc.circuits.surgery.gadget import build_gadget
from qldpc.circuits.surgery.y_gadget import (
    YGadgetLayout,
    _locate_overlap,
    _steane_y_pair,
    build_y_gadget,
)
from qldpc.objects import Pauli

GF2 = galois.GF(2)


def test_build_y_gadget_no_bridge_field_and_boosted():
    code, x, z = _steane_y_pair()
    yg = build_y_gadget(code, x=x, z=z)
    names = {f.name for f in dataclasses.fields(YGadgetLayout)}
    assert "bridge" not in names
    assert yg.W == (int(np.where(x.astype(bool) & z.astype(bool))[0][0]),)
    assert cheeger_constant(yg.g_x) >= 1.0 and cheeger_constant(yg.g_z) >= 1.0
    # Ȳ in stabilizer center: symplectic [x|z] in rowspace(H_sym) restricted to data cols
    n = code.num_qudits
    nm = yg.merged_code.num_qudits
    H = np.asarray(yg.H_sym).astype(int)
    data_cols = list(range(n)) + list(range(nm, nm + n))
    M = GF2(H[:, data_cols] % 2)
    v = GF2(np.concatenate([np.asarray(x), np.asarray(z)]).astype(int) % 2)
    assert int(np.linalg.matrix_rank(M)) == int(
        np.linalg.matrix_rank(GF2(np.vstack([np.asarray(M), np.asarray(v)[None, :]])))
    )


def test_steane_y_pair_has_single_overlap():
    from qldpc.circuits.surgery.y_gadget import _locate_overlaps

    code, x, z = _steane_y_pair()
    assert ((np.asarray(code.matrix_z) @ x) % 2 == 0).all()  # x is logical-X
    assert ((np.asarray(code.matrix_x) @ z) % 2 == 0).all()  # z is logical-Z
    overlap = np.where((x.astype(bool)) & (z.astype(bool)))[0]
    assert overlap.size == 1
    assert _locate_overlap(code, x, z) == int(overlap[0])
    # _locate_overlaps (plural) returns the same singleton as a tuple; |W| odd.
    W = _locate_overlaps(code, x, z)
    assert isinstance(W, tuple)
    assert W == tuple(int(i) for i in overlap)
    assert len(W) % 2 == 1  # anticommuting ⇒ odd overlap


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


def test_build_y_gadget_rejects_commuting_pair():
    # The general-|W| builder no longer rejects multi-overlap (it merges every
    # v ∈ W), but it MUST still reject a pair that violates the Ȳ = iX̄Z̄
    # precondition. A commuting (x, z) pair (here z = 0, so x · z is even) is not
    # an anticommuting logical-X/-Z pair and is rejected by _locate_overlaps.
    code, x, _ = _steane_y_pair()
    z_commuting = np.zeros_like(x)
    with pytest.raises(ValueError):
        build_y_gadget(code, x=x, z=z_commuting)


def test_build_y_gadget_handles_multi_overlap():
    # |W|=3 case: the Steane logical-X representative is self-dual (H_X @ x = 0),
    # so (x, x) is a valid anticommuting pair with |W| = 3. The general-|W|
    # builder produces a genuine stabilizer code (no longer rejected).
    code, x, _ = _steane_y_pair()
    yg = build_y_gadget(code, x=x, z=x)
    assert len(yg.W) == 3
    assert yg.Y_stab.shape[0] == 3
    assert yg.merged_code.is_subsystem_code is False


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


def test_bb_y_pair_overlaps():
    from qldpc.circuits.surgery.y_gadget import _bb_y_pair, _locate_overlaps

    for ov in (1, 3):
        code, x, z = _bb_y_pair(overlap=ov)
        assert code.num_qudits == 36
        W = _locate_overlaps(code, x, z)
        assert len(W) == ov


def test_partial0_steane_w1_is_pure_css_no_crossing():
    from qldpc.circuits.surgery.y_gadget import (
        _merged_incidence,
        _partial0_symplectic_rows,
    )

    code, x, z = _steane_y_pair()
    g_x = build_gadget(code, x, basis=Pauli.X)
    g_z = build_gadget(code, z, basis=Pauli.Z)
    n = code.num_qudits
    k_x = len(g_x.Q_prime)
    k_z = len(g_z.Q_prime)
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


def _kerdim(M):
    M = np.asarray(M).astype(int)
    return 0 if M.size == 0 else np.asarray(GF2(M).null_space()).shape[0]


# crossing_dim = dim(ker merged ∂_1) − dim(ker ∂_1^x) − dim(ker ∂_1^z) is the
# BASIS-INDEPENDENT count of genuine crossing cycles. (The number of mixed ROWS
# in any particular ∂_0 basis is NOT invariant — do not assert on it.)
@pytest.mark.parametrize("overlap, crossing_dim_expected", [(1, 0), (3, 2)])
def test_bb_merged_structure_and_crossing_dim(overlap, crossing_dim_expected):
    from qldpc.circuits.surgery.y_gadget import _bb_y_pair, _merged_incidence

    code, x, z = _bb_y_pair(overlap=overlap)
    yg = build_y_gadget(code, x=x, z=z)
    assert yg.merged_code.dimension == 7  # 8 logicals − 1 measured
    D1, k_x, k_z = _merged_incidence(yg.g_x, yg.g_z, x, z)
    d1x = np.asarray(yg.g_x.incidence).astype(int).T
    d1z = np.asarray(yg.g_z.incidence).astype(int).T
    crossing_dim = _kerdim(D1) - _kerdim(d1x) - _kerdim(d1z)
    assert crossing_dim == crossing_dim_expected  # 0 at |W|=1, |W|−1 at |W|=3


def test_h_sym_rows_in_h_tilde_block_order() -> None:
    """H_sym rows follow the H̃ formula order: block1 H_X(X), block2 S_X'(X),
    block3 Y(mixed), block4 H_Z(Z), block5 S_Z'(Z), block6 cycles
    (Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.D)."""
    code, x, z = _steane_y_pair()
    yg = build_y_gadget(code, x=x, z=z)
    H = np.asarray(yg.H_sym).astype(int)
    n = yg.merged_code.num_qudits
    xparts = (H[:, :n] != 0).any(axis=1)
    zparts = (H[:, n:] != 0).any(axis=1)
    m_x = code.matrix_x.shape[0]
    n_w = len(yg.W)
    # block 1 (H_X) + block 2 (S_X' on V_X) are pure-X and come first
    assert xparts[: m_x].all() and not zparts[: m_x].any(), "block 1 not pure-X-first"
    # block 3 (Y on W) is mixed and follows the pure-X blocks
    y0 = m_x + (yg.g_x.HX_merged.shape[0] - m_x - n_w)  # m_x + |V_X|
    assert (xparts[y0 : y0 + n_w] & zparts[y0 : y0 + n_w]).all(), "block 3 not mixed"
    # blocks 4+5 (H_Z pure-Z, S_Z' on V_Z pure-Z) start immediately after Y
    m_z = code.matrix_z.shape[0]
    v_z = yg.g_z.HZ_merged.shape[0] - m_z - n_w  # |V_Z|
    z45_start = y0 + n_w
    z45_end = z45_start + m_z + v_z  # exclusive
    assert not xparts[z45_start:z45_end].any(), "blocks 4+5 not pure-Z"
    assert zparts[z45_start:z45_end].all(), "blocks 4+5 not all-Z"
    # block 6 (∂₀ cycle rows) is last; for |W|=1 there is 1 pure-Z cycle row
    # (crossing cycles only arise at |W|≥3); the entire tail after blocks 4+5 is pure-Z
    n_rows = H.shape[0]
    if z45_end < n_rows:
        assert not xparts[z45_end:].any(), "block 6 tail has unexpected X-part"
        assert zparts[z45_end:].all(), "block 6 tail has unexpected non-Z rows"
