"""data_init truth-table / tuple / validation tests for
qldpc.circuits.surgery.circuit.PPM_joint (build_joint_ppm_circuit,
_expand_joint_data_init).

The merge/dimension/coords/observable tests live in PPM_joint_test.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from qldpc import codes
from qldpc.circuits.surgery.conftest import (
    _webster_x_bar_operator,
    build_generalised_bicycle_code,
    load_webster_seed_set,
)
from qldpc.objects import Pauli


def test_joint_ppm_data_init_truth_table() -> None:
    """Joint Z̄⊗Z̄ on two Steane copies: the time-like L (index k=k_l+k_r) encodes
    the joint parity across the 4 |a⟩|b⟩ inits.

    Match-basis joint emits k_l+k_r+1 = 3 observables: block Z̄_l (index 0),
    block Z̄_r (index 1), and the time-like joint L = Z̄_l⊗Z̄_r (index 2). The
    joint parity truth table now lives on the time-like L, not the old obs0
    (which is now a single block logical).
    """
    from qldpc.circuits.surgery.circuit.PPM_joint import build_joint_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_joint import build_bridge
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget

    c1, c2 = codes.SteaneCode(), codes.SteaneCode()
    z1 = np.asarray(c1.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    z2 = np.asarray(c2.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g1 = build_gadget(c1, z1, basis=Pauli.Z)
    g2 = build_gadget(c2, z2, basis=Pauli.Z)
    bridge = build_bridge(g1, g2)
    n1 = c1.num_qudits
    k = c1.dimension + c2.dimension  # time-like L lives at index k
    cases = [
        ("0" * n1 + "0" * n1, 0),
        ("0" * n1 + "1" * n1, 1),
        ("1" * n1 + "0" * n1, 1),
        ("1" * n1 + "1" * n1, 0),
    ]
    for data_init, expected in cases:
        circuit, _ = build_joint_ppm_circuit(
            g1, g2, bridge, rounds=3, noise_model=None, data_init=data_init
        )
        assert circuit.num_observables == k + 1
        raw = circuit.compile_sampler().sample(shots=16).astype(np.uint8)
        n_meas = raw.shape[1]
        obs_lines = [ln for ln in str(circuit).splitlines() if ln.startswith("OBSERVABLE_INCLUDE")]
        offsets = [int(t.strip("rec[]")) for t in obs_lines[k].split() if t.startswith("rec[")]
        time_L = np.bitwise_xor.reduce(raw[:, [n_meas + off for off in offsets]], axis=1)
        rate = float(time_L.mean())
        assert rate == float(expected), (
            f"data_init={data_init!r} gave time-like L rate {rate:.3f}, expected {expected}"
        )


def test_joint_ppm_data_init_superposition() -> None:
    """c1 |0⟩ × c2 |+⟩: block Z̄_r is random (c2 in a Z-superposition), yet the
    time-like L still equals block Z̄_l ⊕ block Z̄_r every shot (§3.4).

    Match-basis joint emits 3 observables: block Z̄_l (index 0, deterministic
    here), block Z̄_r (index 1, random), time-like L = Z̄_l⊗Z̄_r (index 2). The
    folded cross-check L == Z̄_l ⊕ Z̄_r is load-bearing even when a block logical
    is itself random — replacing the old obs0==obs1 cross-check.
    """
    from qldpc.circuits.surgery.circuit.PPM_joint import build_joint_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_joint import build_bridge
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget

    c1, c2 = codes.SteaneCode(), codes.SteaneCode()
    z1 = np.asarray(c1.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    z2 = np.asarray(c2.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g1 = build_gadget(c1, z1, basis=Pauli.Z)
    g2 = build_gadget(c2, z2, basis=Pauli.Z)
    bridge = build_bridge(g1, g2)
    n = c1.num_qudits
    k = c1.dimension + c2.dimension  # = 2; time-like L at index k
    circuit, _ = build_joint_ppm_circuit(
        g1, g2, bridge, rounds=3, noise_model=None, data_init="0" * n + "+" * n
    )
    assert circuit.num_observables == k + 1
    raw = circuit.compile_sampler().sample(shots=64).astype(np.uint8)
    n_meas = raw.shape[1]
    obs_lines = [ln for ln in str(circuit).splitlines() if ln.startswith("OBSERVABLE_INCLUDE")]
    cols = []
    for line in obs_lines:
        offsets = [int(t.strip("rec[]")) for t in line.split() if t.startswith("rec[")]
        cols.append(np.bitwise_xor.reduce(raw[:, [n_meas + off for off in offsets]], axis=1))
    block_l, block_r, time_L = cols[0], cols[1], cols[k]
    assert block_r.min() != block_r.max(), "premise: c2 |+⟩ should make block Z̄_r random"
    assert (time_L == (block_l ^ block_r)).all(), (
        f"time-like L != block_l XOR block_r on {(time_L != (block_l ^ block_r)).sum()}/64 shots"
    )


def test_joint_ppm_data_init_tuple_matches_per_qubit_string() -> None:
    """data_init=("0", "+") produces the same circuit as "0"*n + "+"*n."""
    from qldpc.circuits.surgery.circuit.PPM_joint import build_joint_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_joint import build_bridge
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget

    c1, c2 = codes.SteaneCode(), codes.SteaneCode()
    z1 = np.asarray(c1.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    z2 = np.asarray(c2.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g1 = build_gadget(c1, z1, basis=Pauli.Z)
    g2 = build_gadget(c2, z2, basis=Pauli.Z)
    bridge = build_bridge(g1, g2)
    n = c1.num_qudits
    c_tuple, _ = build_joint_ppm_circuit(
        g1,
        g2,
        bridge,
        rounds=3,
        noise_model=None,
        data_init=("0", "+"),
    )
    c_string, _ = build_joint_ppm_circuit(
        g1,
        g2,
        bridge,
        rounds=3,
        noise_model=None,
        data_init="0" * n + "+" * n,
    )
    assert str(c_tuple) == str(c_string)


def test_joint_ppm_data_init_tuple_per_qubit_entry() -> None:
    """Each tuple entry may be per-qubit (length n_code), not only len-1 broadcast."""
    from qldpc.circuits.surgery.circuit.PPM_joint import build_joint_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_joint import build_bridge
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget

    c1, c2 = codes.SteaneCode(), codes.SteaneCode()
    z1 = np.asarray(c1.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    z2 = np.asarray(c2.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g1 = build_gadget(c1, z1, basis=Pauli.Z)
    g2 = build_gadget(c2, z2, basis=Pauli.Z)
    bridge = build_bridge(g1, g2)
    n = c1.num_qudits
    spec_l = "0011010"
    spec_r = "+"
    c_tuple, _ = build_joint_ppm_circuit(
        g1,
        g2,
        bridge,
        rounds=3,
        noise_model=None,
        data_init=(spec_l, spec_r),
    )
    c_string, _ = build_joint_ppm_circuit(
        g1,
        g2,
        bridge,
        rounds=3,
        noise_model=None,
        data_init=spec_l + "+" * n,
    )
    assert str(c_tuple) == str(c_string)


@pytest.mark.parametrize(
    "bad_init,error_substr",
    [
        (("0",), "must have 2 entries"),
        (("0", "+", "-"), "must have 2 entries"),
        (("00", "+"), "data_init\\[0\\] length 2 does not match c_l data count 7"),
        (("0", "++"), "data_init\\[1\\] length 2 does not match c_r data count 7"),
        ((0, "+"), "must be str"),
    ],
)
def test_joint_ppm_data_init_tuple_validation(bad_init: object, error_substr: str) -> None:
    from qldpc.circuits.surgery.circuit.PPM_joint import build_joint_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_joint import build_bridge
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget

    c1, c2 = codes.SteaneCode(), codes.SteaneCode()
    z1 = np.asarray(c1.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    z2 = np.asarray(c2.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g1 = build_gadget(c1, z1, basis=Pauli.Z)
    g2 = build_gadget(c2, z2, basis=Pauli.Z)
    bridge = build_bridge(g1, g2)
    expected = TypeError if "must be str" in error_substr else ValueError
    with pytest.raises(expected, match=error_substr):
        build_joint_ppm_circuit(
            g1,
            g2,
            bridge,
            rounds=3,
            noise_model=None,
            data_init=bad_init,  # type: ignore[arg-type]
        )


def test_joint_ppm_data_init_tuple_rejects_intracode() -> None:
    """Tuple form is invalid for intracode joint PPM (single data set)."""
    from qldpc.circuits.surgery.circuit.PPM_joint import build_joint_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_joint import build_bridge
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget

    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x1 = _webster_x_bar_operator(data, "X_bar_1")
    x2 = _webster_x_bar_operator(data, "X_bar_k2p1")
    g_l = build_gadget(code, x1, basis=Pauli.X)
    g_r = build_gadget(code, x2, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    assert g_l.code is g_r.code, "intracode setup precondition"
    with pytest.raises(ValueError, match="intracode joint has a single data set"):
        build_joint_ppm_circuit(
            g_l,
            g_r,
            bridge,
            rounds=3,
            noise_model=None,
            data_init=("0", "0"),
        )


def test_joint_ppm_even_rounds_truth_table() -> None:
    """The time-like L must encode logical X̄_l X̄_r parity correctly at EVEN rounds.

    Regression test for the bug where _surgery_observable XOR'd meas-check
    syndromes across all rounds (R · m_v ≡ 0 mod 2 for even R) instead of using a
    single round's product (Webster, Smith, Cohen arXiv:2511.15989 §II.A: Z̄ = ∏_v
    A_v). The fix reads the FIRST-cycle merge checks; the time-like L lives at
    index k=k_l+k_r. Uses ``compile_sampler`` + manual XOR to read the raw
    observable bit. Also checks §3.4: L == block X̄_l ⊕ block X̄_r every shot.
    """
    from qldpc.circuits.surgery.circuit.PPM_joint import build_joint_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_joint import build_bridge
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(codes.SteaneCode(), x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    k = g_l.code.dimension + g_r.code.dimension  # time-like L at index k
    # basis=X, so we sweep ("+", "+"), ("-", "+"), ("+", "-"), ("-", "-").
    # "-" on data flips X̄ to -1; X̄_l X̄_r = product → parity bit.
    cases = [
        (("+", "+"), 0),
        (("-", "+"), 1),
        (("+", "-"), 1),
        (("-", "-"), 0),
    ]
    for data_init, expected in cases:
        circuit, _ = build_joint_ppm_circuit(
            g_l,
            g_r,
            bridge,
            rounds=2,
            noise_model=None,
            data_init=data_init,
        )
        raw = circuit.compile_sampler().sample(shots=16).astype(np.uint8)
        n_meas = raw.shape[1]
        obs_lines = [ln for ln in str(circuit).splitlines() if ln.startswith("OBSERVABLE_INCLUDE")]
        vals = []
        for ln in obs_lines:
            offs = [int(t.strip("rec[]")) for t in ln.split() if t.startswith("rec[")]
            vals.append(np.bitwise_xor.reduce(raw[:, [n_meas + o for o in offs]], axis=1))
        block_l, block_r, time_L = vals[0], vals[1], vals[k]
        assert (time_L == expected).all(), (
            f"data_init={data_init!r}: time-like L has {(time_L != expected).sum()}/"
            f"16 shots disagreeing with expected parity bit {expected}"
        )
        # §3.4 folded cross-check: time-like L == block X̄_l ⊕ block X̄_r.
        assert (time_L == (block_l ^ block_r)).all(), (
            f"data_init={data_init!r}: time-like L != block_l XOR block_r"
        )


def test_expand_joint_data_init_rejects_non_str_non_seq_type() -> None:
    """_expand_joint_data_init raises TypeError on data_init that isn't str/tuple/list/None."""
    from qldpc.circuits.surgery.circuit.PPM_joint import _expand_joint_data_init

    with pytest.raises(TypeError, match="data_init must be"):
        _expand_joint_data_init({"bad": "input"}, n_l=4, n_r=4, intercode=True)  # type: ignore[arg-type]
