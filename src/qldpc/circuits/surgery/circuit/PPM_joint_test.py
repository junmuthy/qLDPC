"""Merge/dimension/coords/observable tests for
qldpc.circuits.surgery.circuit.PPM_joint (build_joint_ppm_circuit).

Includes the joint-merged-code structural tests (which exercise
``hmatrix.PPM_joint._joint_merged_dispatch``), the joint code dimension formula,
the intercode QUBIT_COORDS layout, non-destructive detach, the folded
cross-check, and the experiment_basis observable counts. The data_init
truth-table/tuple tests live in PPM_joint_data_init_test.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from qldpc import codes
from qldpc.circuits.surgery.circuit.conftest import _data_measured
from qldpc.circuits.surgery.conftest import (
    _webster_x_bar_operator,
    build_generalised_bicycle_code,
    load_webster_seed_set,
)
from qldpc.objects import Pauli, PauliXZ


def test_joint_merged_intercode_basis_x_joint_logical_in_stabilizer() -> None:
    """(x_1, x_2, 0, 0, 0) lies in rowspan(H_X^merged) — joint X̄_l X̄_r is a stabilizer."""
    from qldpc.circuits.surgery.hmatrix.PPM_joint import (
        _joint_merged_dispatch,
        build_bridge,
    )
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget

    code1 = codes.SteaneCode()
    code2 = codes.SteaneCode()
    x1 = np.asarray(code1.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    x2 = np.asarray(code2.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code1, x1, basis=Pauli.X)
    g_r = build_gadget(code2, x2, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    merged = _joint_merged_dispatch(g_l, g_r, bridge)
    import galois

    GF2 = galois.GF(2)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    n_l = code1.num_qudits
    n_r = code2.num_qudits
    joint = np.zeros(HX.shape[1], dtype=np.int_)
    joint[:n_l] = x1
    joint[n_l : n_l + n_r] = x2
    augmented = np.vstack([HX, joint.reshape(1, -1)])
    assert np.linalg.matrix_rank(GF2(HX.tolist())) == np.linalg.matrix_rank(GF2(augmented.tolist()))


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_joint_merged_intercode_both_bases_commute_and_singletons_excluded(basis: PauliXZ) -> None:
    import galois

    from qldpc.circuits.surgery.hmatrix.PPM_joint import (
        _joint_merged_dispatch,
        build_bridge,
    )
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget

    GF2 = galois.GF(2)
    code = codes.SteaneCode()
    if basis is Pauli.X:
        x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    else:
        x = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=basis)
    g_r = build_gadget(codes.SteaneCode(), x, basis=basis)
    bridge = build_bridge(g_l, g_r)
    merged = _joint_merged_dispatch(g_l, g_r, bridge)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    HZ = np.asarray(merged.matrix_z).astype(np.int_)
    product = (HX @ HZ.T) % 2
    assert np.array_equal(product, np.zeros_like(product))
    assert merged.dimension == 2 * code.dimension - 1
    # Singletons excluded: (x_l, 0, ...) and (0, x_r, ...) NOT in rowspan of the
    # check matrix that contains the joint stabilizer (HX for basis=X, HZ for Z).
    H_joint = HX if basis is Pauli.X else HZ
    n_l = code.num_qudits
    base_rank = np.linalg.matrix_rank(GF2(H_joint.tolist()))
    for which in ("left", "right"):
        single = np.zeros(H_joint.shape[1], dtype=np.int_)
        if which == "left":
            single[:n_l] = x
        else:
            single[n_l : 2 * n_l] = x
        augmented = np.vstack([H_joint, single.reshape(1, -1)])
        assert np.linalg.matrix_rank(GF2(augmented.tolist())) == base_rank + 1, which


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_joint_merged_intracode_both_bases_commute(basis: PauliXZ) -> None:
    """Intra-code commutation for both bases. Use a Webster code with 2 distinct logicals.

    Steane intra-code (k=1) yields the degenerate joint X̄·X̄ = I case.
    """
    from qldpc.circuits.surgery.hmatrix.PPM_joint import (
        _joint_merged_dispatch,
        build_bridge,
    )
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import (
        build_gadget,
    )

    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    if basis is Pauli.X:
        x1 = _webster_x_bar_operator(data, "X_bar_1")
        x2 = _webster_x_bar_operator(data, "X_bar_k2p1")
    else:
        from qldpc.circuits.surgery.conftest import _webster_z_bar_operator

        x1 = _webster_z_bar_operator(data, "Z_bar_1")
        x2 = _webster_z_bar_operator(data, "Z_bar_k2p1")
    g_l = build_gadget(code, x1, basis=basis)
    g_r = build_gadget(code, x2, basis=basis)
    bridge = build_bridge(g_l, g_r)
    merged = _joint_merged_dispatch(g_l, g_r, bridge)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    HZ = np.asarray(merged.matrix_z).astype(np.int_)
    product = (HX @ HZ.T) % 2
    assert np.array_equal(product, np.zeros_like(product))
    assert merged.dimension == code.dimension - 1


def test_build_joint_ppm_circuit_meas_check_ids_no_UB() -> None:
    """build_joint_ppm_circuit's noiseless first sample has zero detectors firing."""
    from qldpc.circuits.surgery.circuit.PPM_joint import build_joint_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_joint import build_bridge
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(codes.SteaneCode(), x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    circuit, merged = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=2)
    # noiseless: all detectors must NOT fire on first sample
    sampler = circuit.compile_detector_sampler()
    dets, _ = sampler.sample(8, separate_observables=True)
    assert dets.sum() == 0


def test_build_joint_ppm_circuit_intercode_folded_cross_check() -> None:
    """Folded cross-check (design §3.4): in a noiseless match-basis joint run the
    time-like L (index k=k_l+k_r) equals the GF(2) sum of the block observables
    (here X̄_l ⊕ X̄_r), across all 4 parity inits.

    Replaces the removed obs0==obs1 mechanism. The match-basis joint emits
    k_l+k_r+1 = 3 observables: block X̄_l (index 0), block X̄_r (index 1), and the
    time-like joint L = X̄_l⊗X̄_r (index 2). Sweeping non-trivial parity inits
    catches a regression in either the block readout or the time-like L.
    """
    from qldpc.circuits.surgery.circuit.PPM_joint import build_joint_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_joint import build_bridge
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(codes.SteaneCode(), x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    k = g_l.code.dimension + g_r.code.dimension  # = 2; L lives at index k
    # ("-" flips X̄ to -1.) Expected: block_l = a, block_r = b, L = a XOR b.
    cases = [(("+", "+"), 0, 0), (("-", "+"), 1, 0), (("+", "-"), 0, 1), (("-", "-"), 1, 1)]
    for data_init, exp_l, exp_r in cases:
        circuit, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=2, data_init=data_init)
        assert circuit.num_observables == k + 1
        raw = circuit.compile_sampler().sample(shots=8).astype(np.uint8)
        n_meas = raw.shape[1]
        obs_lines = [ln for ln in str(circuit).splitlines() if ln.startswith("OBSERVABLE_INCLUDE")]
        vals = []
        for ln in obs_lines:
            offs = [int(t.strip("rec[]")) for t in ln.split() if t.startswith("rec[")]
            vals.append(np.bitwise_xor.reduce(raw[:, [n_meas + o for o in offs]], axis=1))
        block_l, block_r, time_L = vals[0], vals[1], vals[k]
        assert (block_l == exp_l).all(), f"{data_init!r}: block X̄_l != {exp_l}"
        assert (block_r == exp_r).all(), f"{data_init!r}: block X̄_r != {exp_r}"
        # §3.4 folded cross-check: time-like L == X̄_l ⊕ X̄_r every shot.
        assert (time_L == (block_l ^ block_r)).all(), (
            f"{data_init!r}: time-like L != block_l XOR block_r on "
            f"{(time_L != (block_l ^ block_r)).sum()}/8 noiseless shots"
        )


@pytest.mark.parametrize("code_index", [0, 1, 2, 3])
def test_joint_xx_in_stabilizer_on_webster_intracode(code_index: int) -> None:
    """Webster BB codes 0..3 intra-code: (x_1, x_2 padded, 0...) is in rowspan(H_X^merged).

    Replaces deleted path-graph tests; pins the SkipTree adapter construction across
    the full Webster Table I code family. Code 3 ([[510,16]], |support|=26) is cheap
    again now that build_gadget uses the Fiedler path above |V|=20 (~0.07 s, was
    ~200 s under the old exact O(2^26) Cheeger sweep).
    """
    import galois

    from qldpc.circuits.surgery.hmatrix.PPM_joint import (
        _joint_merged_dispatch,
        build_bridge,
    )
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import (
        build_gadget,
    )

    GF2 = galois.GF(2)
    data = load_webster_seed_set(code_index)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x1 = _webster_x_bar_operator(data, "X_bar_1")
    x2 = _webster_x_bar_operator(data, "X_bar_k2p1")
    g_l = build_gadget(code, x1, basis=Pauli.X)
    g_r = build_gadget(code, x2, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    merged = _joint_merged_dispatch(g_l, g_r, bridge)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    joint = np.zeros(HX.shape[1], dtype=np.int_)
    n = code.num_qudits
    joint[:n] = (x1 + x2) % 2
    augmented = np.vstack([HX, joint.reshape(1, -1)])
    assert np.linalg.matrix_rank(GF2(HX.tolist())) == np.linalg.matrix_rank(GF2(augmented.tolist()))


def test_build_joint_ppm_circuit_intracode_noiseless_observables_zero() -> None:
    """Intra-code Webster joint X̄_1·X̄_{k/2+1}: noiseless detectors + observables = 0.

    Replaces deleted path-graph noiseless intracode tests.
    """
    from qldpc.circuits.surgery.circuit.PPM_joint import build_joint_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_joint import build_bridge
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import (
        build_gadget,
    )

    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x1 = _webster_x_bar_operator(data, "X_bar_1")
    x2 = _webster_x_bar_operator(data, "X_bar_k2p1")
    g_l = build_gadget(code, x1, basis=Pauli.X)
    g_r = build_gadget(code, x2, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    circuit, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=2)
    sampler = circuit.compile_detector_sampler()
    dets, obs = sampler.sample(8, separate_observables=True)
    assert dets.sum() == 0, "noiseless intra-code: detectors should not fire"
    assert obs.sum() == 0, "noiseless intra-code: observables should be 0"


def test_joint_ppm_non_destructive_detach_only() -> None:
    """ZZ joint: ``destructive_measure_data=False`` detaches but keeps data encoded."""
    from qldpc.circuits.surgery.circuit.PPM_joint import build_joint_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_joint import build_bridge
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget

    c1, c2 = codes.SteaneCode(), codes.SteaneCode()
    z1 = np.asarray(c1.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    z2 = np.asarray(c2.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g1 = build_gadget(c1, z1, basis=Pauli.Z)
    g2 = build_gadget(c2, z2, basis=Pauli.Z)
    bridge = build_bridge(g1, g2)

    full, _ = build_joint_ppm_circuit(g1, g2, bridge, rounds=2, noise_model=None)
    lean, _ = build_joint_ppm_circuit(
        g1, g2, bridge, rounds=2, noise_model=None, destructive_measure_data=False
    )
    n_data = c1.num_qudits + c2.num_qudits  # left+right data qubits
    # destructive joint emits the match-basis (k_l+k_r)+1 set; non-destructive has
    # no final data readout to build it from, so 0 observables.
    assert full.num_observables == c1.dimension + c2.dimension + 1
    assert _data_measured(lean, n_data) == set()  # data left encoded
    assert lean.num_observables == 0  # no destructive readout => no observable set
    assert lean.num_detectors < full.num_detectors
    lean.detector_error_model()  # still compiles


def test_joint_ppm_qubit_coords_intercode_layout() -> None:
    """Intercode joint Z̄⊗Z̄ on two Steane copies: QUBIT_COORDS lanes correct.

    n_l = n_r = 7; left data on y=0 at x=0..6; right data on y=0 at x=7..13.
    κ ancillas on y=1. Bridge data + cycle ancillas on y=6.
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
    circuit, _ = build_joint_ppm_circuit(
        g1,
        g2,
        bridge,
        rounds=1,
        noise_model=None,
    )

    # Parse QUBIT_COORDS and group qubit ids by y.
    by_y: dict[int, list[tuple[int, int]]] = {}
    for line in str(circuit).splitlines():
        line = line.strip()
        if not line.startswith("QUBIT_COORDS"):
            continue
        head, qid_str = line.rsplit(" ", 1)
        tup = head[len("QUBIT_COORDS(") : -1]
        x_str, y_str = [t.strip() for t in tup.split(",")]
        x, y = int(x_str), int(y_str)
        qid = int(qid_str)
        by_y.setdefault(y, []).append((x, qid))

    # y=0 must have n_l + n_r = 14 qubits at x=0..13.
    y0 = sorted(by_y.get(0, []))
    assert len(y0) == 14, f"y=0 expected 14 data qubits, got {len(y0)}"
    assert [x for x, _ in y0] == list(range(14)), (
        f"y=0 x positions: expected 0..13, got {[x for x, _ in y0]}"
    )

    # y=1 (was y=3) must have κ_l + κ_r qubits (depends on bridge augmentation).
    y1 = sorted(by_y.get(1, []))
    assert len(y1) >= 2, f"y=1 expected at least 2 κ qubits, got {len(y1)}"

    # y=6 holds the w bridge-data (adapter) qubits at x=0..w-1; y=7 holds the
    # w-1 bridge cycle ancillas at x=0..w-2 on their own lane (regression: both
    # used to share y=6, with cycle check i colliding with bridge-data qubit i).
    w = bridge.width
    y6 = sorted(by_y.get(6, []))
    assert [x for x, _ in y6] == list(range(w)), (
        f"y=6 expected bridge data at x=0..{w - 1}, got {[x for x, _ in y6]}"
    )
    y7 = sorted(by_y.get(7, []))
    assert [x for x, _ in y7] == list(range(w - 1)), (
        f"y=7 expected {w - 1} cycle ancillas at x=0..{w - 2}, got {[x for x, _ in y7]}"
    )


def test_joint_code_dimension_steane_x_steane_equals_one() -> None:
    """Intercode Steane × Steane joint PPM gives joint_code.dimension == 1.

    Formula: k_l + k_r − 1 because Z̄_l ⊗ Z̄_r becomes a stabilizer of
    the joint code after surgery. For k_l = k_r = 1, that's 1.

    Catches a stitching bug in _joint_merged_intercode that drops or
    duplicates a stabilizer row — CSS commutation would still hold
    but the joint code's logical dimension would shift.
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
    _, joint_code = build_joint_ppm_circuit(
        g1,
        g2,
        bridge,
        rounds=3,
        noise_model=None,
    )
    expected = c1.dimension + c2.dimension - 1  # 1 + 1 - 1 = 1
    assert joint_code.dimension == expected, (
        f"Steane × Steane intercode joint_code.dimension = "
        f"{joint_code.dimension}, expected {expected}"
    )


def test_joint_code_dimension_webster_x_steane_equals_ten() -> None:
    """Intercode Webster GB code 0 × Steane joint PPM gives dim == k_l + k_r − 1 = 10.

    Webster GB code 0 is [[62, 10, _]]; k_l = 10. Steane is k_r = 1.
    Expected: 10 + 1 − 1 = 10.

    The k_l > 1 case exposes the −1 reduction in the formula. A
    stitching bug that fails to add the Z̄_l ⊗ Z̄_r constraint would
    surface as dim = 11.
    """
    from qldpc.circuits.surgery.circuit.PPM_joint import build_joint_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_joint import build_bridge
    from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget

    data = load_webster_seed_set(0)
    webster = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    z_webster = _webster_x_bar_operator(data, "Z_bar_1", pauli_type="Z")
    steane = codes.SteaneCode()
    z_steane = np.asarray(steane.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(webster, z_webster, basis=Pauli.Z)
    g_r = build_gadget(steane, z_steane, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)
    _, joint_code = build_joint_ppm_circuit(
        g_l,
        g_r,
        bridge,
        rounds=3,
        noise_model=None,
    )
    expected = webster.dimension + steane.dimension - 1  # 10 + 1 - 1 = 10
    assert joint_code.dimension == expected, (
        f"Webster × Steane intercode joint_code.dimension = "
        f"{joint_code.dimension}, expected {expected}"
    )


def test_joint_ppm_match_basis_emits_kl_plus_kr_plus_1(_steane_joint_fixture) -> None:
    """Match-basis joint PPM: (k_l + k_r) block + 1 time-like L."""
    from qldpc.circuits.surgery.circuit.PPM_joint import build_joint_ppm_circuit

    g_l, g_r, bridge = _steane_joint_fixture
    circ, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=3, experiment_basis=Pauli.X)
    assert circ.num_observables == g_l.code.dimension + g_r.code.dimension + 1


def test_joint_ppm_opposite_basis_emits_kl_plus_kr_minus_1(_steane_joint_fixture) -> None:
    """Opposite-basis joint PPM: (k_l + k_r - 1) block logicals commuting with L."""
    from qldpc.circuits.surgery.circuit.PPM_joint import build_joint_ppm_circuit

    g_l, g_r, bridge = _steane_joint_fixture
    circ, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=3, experiment_basis=Pauli.Z)
    assert circ.num_observables == g_l.code.dimension + g_r.code.dimension - 1


def test_joint_ppm_observables_deterministic_noiseless(_steane_joint_fixture) -> None:
    """Every joint observable is deterministic with no noise."""
    from qldpc.circuits.surgery.circuit.PPM_joint import build_joint_ppm_circuit

    g_l, g_r, bridge = _steane_joint_fixture
    circ, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=3, experiment_basis=Pauli.X)
    _, obs = circ.compile_detector_sampler().sample(shots=64, separate_observables=True)
    assert not obs.any()


def test_joint_ppm_single_sector_preserves_observables_shrinks_dem(_steane_joint_fixture) -> None:
    """single_sector keeps the full (k_l + k_r + 1) match-basis observable set
    (every observable is the measured Pauli type, decodable from the measured-basis
    sector alone) while dropping the complementary-sector detectors."""
    from qldpc.circuits.surgery.circuit.PPM_joint import build_joint_ppm_circuit

    g_l, g_r, bridge = _steane_joint_fixture
    full, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=3)
    ss, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=3, single_sector=True)
    n_obs = g_l.code.dimension + g_r.code.dimension + 1
    assert ss.num_observables == full.num_observables == n_obs
    assert ss.num_detectors < full.num_detectors  # complementary sector dropped
    _, obs = ss.compile_detector_sampler().sample(shots=64, separate_observables=True)
    assert not obs.any()  # all observables still deterministic from the kept sector


def test_joint_ppm_single_sector_opposite_basis_observable_detectability(
    _steane_joint_fixture,
) -> None:
    """Regression: single_sector on the OPPOSITE-basis joint experiment
    (experiment_basis != bridge.basis) keeps its (k_l + k_r) − 1 block observables
    detectable.

    The shared ``_surgery_final_detectors`` single_sector filter must key off
    ``experiment_basis`` (the data-readout Pauli type), NOT the gadget/bridge basis
    — otherwise it intersects the reconstructable checks with the wrong sector,
    emits ZERO final detectors, and the block observables become undetectable
    (decoder blind, LER → raw flip rate). Cain et al. arXiv:2603.28627 Appendix D
    averages the X- and Z-basis experiments on one gadget/bridge, so both must
    decode. Mirrors the single-PPM ``test_single_sector_opposite_basis_*``.
    """
    from qldpc.circuits import DepolarizingNoiseModel
    from qldpc.circuits.surgery.circuit.PPM_joint import build_joint_ppm_circuit

    g_l, g_r, bridge = _steane_joint_fixture
    opposite = Pauli.Z if bridge.basis is Pauli.X else Pauli.X
    noise = DepolarizingNoiseModel(0.01)
    ss, _ = build_joint_ppm_circuit(
        g_l, g_r, bridge, rounds=3, noise_model=noise,
        single_sector=True, experiment_basis=opposite,
    )
    assert ss.num_observables == g_l.code.dimension + g_r.code.dimension - 1
    dem = ss.detector_error_model()
    undetectable = sum(
        1
        for e in dem.flattened()
        if e.type == "error"
        and any(t.is_logical_observable_id() for t in e.targets_copy())
        and not any(t.is_relative_detector_id() for t in e.targets_copy())
    )
    assert undetectable == 0, (
        f"opposite-basis joint (exp={opposite.name}): {undetectable} observable-flipping "
        f"errors became undetectable (single_sector dropped final detectors)"
    )
