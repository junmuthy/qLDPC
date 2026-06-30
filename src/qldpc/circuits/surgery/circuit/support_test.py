"""Tests for qldpc.circuits.surgery.circuit.support (GF(2) algebra, observable
targets, logical_state_init, keep_only_observable)."""

from __future__ import annotations

import numpy as np
import pytest
import stim

from qldpc import codes
from qldpc.objects import Pauli


def test_logical_state_init_zero_and_plus_broadcast() -> None:
    """'0' and '+' return length-n broadcast strings — trivial CSS prep."""
    from qldpc.circuits.surgery.circuit.support import logical_state_init

    code = codes.SteaneCode()
    n = code.num_qudits
    assert logical_state_init(code, "0", log_idx=0) == "0" * n
    assert logical_state_init(code, "+", log_idx=0) == "+" * n


def test_logical_state_init_one_flips_x_bar_support() -> None:
    """'1' = X̄_0 |0⟩_L: '1' on supp(X̄_0), '0' elsewhere."""
    from qldpc.circuits.surgery.circuit.support import logical_state_init

    code = codes.SteaneCode()
    n = code.num_qudits
    x_bar = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    s = logical_state_init(code, "1", log_idx=0)
    assert len(s) == n
    expected_ones = set(int(i) for i in np.where(x_bar)[0])
    actual_ones = {i for i, c in enumerate(s) if c == "1"}
    actual_zeros = {i for i, c in enumerate(s) if c == "0"}
    assert actual_ones == expected_ones
    assert actual_zeros == set(range(n)) - expected_ones


def test_logical_state_init_minus_flips_z_bar_support() -> None:
    """'-' = Z̄_0 |+⟩_L: '-' on supp(Z̄_0), '+' elsewhere."""
    from qldpc.circuits.surgery.circuit.support import logical_state_init

    code = codes.SteaneCode()
    n = code.num_qudits
    z_bar = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    s = logical_state_init(code, "-", log_idx=0)
    assert len(s) == n
    expected_minus = set(int(i) for i in np.where(z_bar)[0])
    actual_minus = {i for i, c in enumerate(s) if c == "-"}
    actual_plus = {i for i, c in enumerate(s) if c == "+"}
    assert actual_minus == expected_minus
    assert actual_plus == set(range(n)) - expected_minus


@pytest.mark.parametrize("bad", ["2", "x", "", "01", "0 ", " 0"])
def test_logical_state_init_invalid_state_raises(bad: str) -> None:
    """Anything outside {'0', '1', '+', '-'} raises ValueError."""
    from qldpc.circuits.surgery.circuit.support import logical_state_init

    code = codes.SteaneCode()
    with pytest.raises(ValueError, match="state"):
        logical_state_init(code, bad, log_idx=0)


def test_logical_state_init_missing_log_idx_raises() -> None:
    """log_idx is keyword-only with no default — omitting it raises TypeError."""
    from qldpc.circuits.surgery.circuit.support import logical_state_init

    code = codes.SteaneCode()
    with pytest.raises(TypeError, match="log_idx"):
        logical_state_init(code, "0")  # type: ignore[call-arg]


def test_logical_state_init_log_idx_selects_different_logical_qubit() -> None:
    """log_idx=i flips supp(X̄_i) — distinct from X̄_0 on k>1 codes."""
    import sympy

    from qldpc.circuits.surgery.circuit.support import logical_state_init

    xs, ys = sympy.symbols("x y")
    code = codes.BBCode({xs: 3, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)
    # k = 8 logical qubits — pick two distinct indices.
    s0 = logical_state_init(code, "1", log_idx=0)
    s3 = logical_state_init(code, "1", log_idx=3)
    x0 = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    x3 = np.asarray(code.get_logical_ops(Pauli.X)[3]).astype(np.uint8)
    n = code.num_qudits
    expected_s0 = "".join("1" if x0[i] else "0" for i in range(n))
    expected_s3 = "".join("1" if x3[i] else "0" for i in range(n))
    assert s0 == expected_s0
    assert s3 == expected_s3
    assert s0 != s3, "different log_idx must give different prep strings"


@pytest.mark.parametrize("log_idx", [-1, 1, 7, 100])
def test_logical_state_init_log_idx_out_of_range_raises(log_idx: int) -> None:
    """log_idx outside [0, code.dimension) raises IndexError."""
    from qldpc.circuits.surgery.circuit.support import logical_state_init

    code = codes.SteaneCode()  # k = 1; only log_idx=0 is valid
    with pytest.raises(IndexError, match="log_idx"):
        logical_state_init(code, "1", log_idx=log_idx)


def test_keep_only_observable_drops_others_and_recurses_into_repeat() -> None:
    """keep_only_observable retains the matching OBSERVABLE_INCLUDE and recurses
    into REPEAT blocks, dropping all other observable IDs."""
    from qldpc.circuits.surgery.circuit.support import keep_only_observable

    inner = stim.Circuit("""
        TICK
        OBSERVABLE_INCLUDE(0) rec[-1]
        OBSERVABLE_INCLUDE(1) rec[-2]
    """)
    outer = stim.Circuit()
    outer.append("M", [0, 1])
    outer.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 1)
    outer.append(stim.CircuitRepeatBlock(2, inner))
    outer.append("OBSERVABLE_INCLUDE", [stim.target_rec(-2)], 0)

    kept = keep_only_observable(outer, keep_idx=0)
    text = str(kept)
    # obs(0) outside REPEAT preserved
    assert "OBSERVABLE_INCLUDE(0)" in text
    # obs(1) outside REPEAT removed
    assert text.count("OBSERVABLE_INCLUDE(1)") == 0
    # REPEAT block still present and filtered (only obs(0) inside)
    assert "REPEAT 2" in text
    repeat_body_lines = [ln.strip() for ln in text.splitlines() if "OBSERVABLE_INCLUDE" in ln]
    assert all("OBSERVABLE_INCLUDE(0)" in ln for ln in repeat_body_lines)


def test_gf2_solve_consistent_returns_particular_solution():
    from qldpc.circuits.surgery.circuit.support import _gf2_solve

    A = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    b = np.array([1, 0], dtype=np.uint8)
    x = _gf2_solve(A, b)
    assert x is not None
    assert np.array_equal((A @ x) % 2, b)


def test_gf2_solve_inconsistent_returns_none():
    from qldpc.circuits.surgery.circuit.support import _gf2_solve

    A = np.array([[1, 0], [1, 0], [0, 0]], dtype=np.uint8)
    b = np.array([1, 0, 0], dtype=np.uint8)  # rows 0,1 demand x0=1 and x0=0
    assert _gf2_solve(A, b) is None


def test_gf2_solve_zero_rhs_returns_zero_vector():
    from qldpc.circuits.surgery.circuit.support import _gf2_solve

    A = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    b = np.array([0, 0], dtype=np.uint8)
    x = _gf2_solve(A, b)
    assert x is not None
    assert np.array_equal(x, np.zeros(2, dtype=np.uint8))


def test_commuting_basis_all_commute_returns_all():
    from qldpc.circuits.surgery.circuit.support import _commuting_logical_basis

    logical_ops = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint8)
    L = np.array([0, 0, 0], dtype=np.uint8)  # symplectic product 0 with everything
    basis = _commuting_logical_basis(logical_ops, L)
    assert basis.shape == (2, 3)
    assert np.array_equal(basis, logical_ops)


def test_commuting_basis_drops_one_when_one_anticommutes():
    from qldpc.circuits.surgery.circuit.support import _commuting_logical_basis

    logical_ops = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint8)
    L = np.array([1, 0, 0], dtype=np.uint8)  # anticommutes only with row 0
    basis = _commuting_logical_basis(logical_ops, L)
    assert basis.shape == (1, 3)
    assert ((basis @ L) % 2 == 0).all()  # all commute with L
    assert np.array_equal(basis[0], np.array([0, 1, 0], dtype=np.uint8))


def test_commuting_basis_general_L_combines_multiple_anticommuters():
    from qldpc.circuits.surgery.circuit.support import _commuting_logical_basis

    # L overlaps rows 0 AND 1 (both anticommute); result must be k-1 = 1, commuting.
    logical_ops = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.uint8)
    L = np.array([1, 0, 0], dtype=np.uint8)  # dot row0=1, row1=1 -> both anticommute
    basis = _commuting_logical_basis(logical_ops, L)
    assert basis.shape == (1, 3)
    assert ((basis @ L) % 2 == 0).all()


def test_block_observable_targets_no_deformation_when_data_only_valid():
    from qldpc.circuits.surgery.circuit.support import _block_observable_targets
    from qldpc.codes.common import CSSCode

    # Merged code = a code where a data-only Z logical already commutes with all X.
    # Use a 2-qubit code with HX empty, HZ empty (1 logical), Q' = none.
    merged = CSSCode(
        np.zeros((0, 1), dtype=int),
        np.zeros((0, 1), dtype=int),
        is_subsystem_code=False,
    )
    col_record = {0: stim.target_rec(-1)}
    w = np.array([1], dtype=np.uint8)  # Z on the single data qubit
    targets = _block_observable_targets(merged, Pauli.Z, w, n_data=1, col_record=col_record)
    assert targets == [stim.target_rec(-1)]


def test_block_observable_targets_adds_qprime_records_for_deformation():
    from qldpc.circuits.surgery.circuit.support import _block_observable_targets
    from qldpc.codes.common import CSSCode

    # merged X-check forces a Z logical to deform onto the Q' column.
    # cols: 0 = data, 1 = Q'.  HX_merged = [[1,1]] (one X-check on data0 & Q').
    merged = CSSCode(
        np.array([[1, 1]], dtype=int),
        np.zeros((0, 2), dtype=int),
        is_subsystem_code=False,
    )
    col_record = {0: stim.target_rec(-2), 1: stim.target_rec(-1)}
    w = np.array([1, 0], dtype=np.uint8)  # data-only Z on col 0 anticommutes with the X-check
    targets = _block_observable_targets(merged, Pauli.Z, w, n_data=1, col_record=col_record)
    # deformed rep must add the Q' column (col 1) so it commutes with the X-check
    assert set(targets) == {stim.target_rec(-2), stim.target_rec(-1)}
