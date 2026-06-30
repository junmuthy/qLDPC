"""Unit/structural tests for qldpc.circuits.surgery.circuit.PPM_XZ
(build_single_ppm_circuit): observable counts, single_sector DEM shrink,
data_init validation, qubit/detector coordinate layout, non-destructive detach.

The heavy end-to-end / truth-table / x-error-locality tests live in
PPM_XZ_e2e_test.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from qldpc import codes
from qldpc.circuits.surgery.circuit.conftest import _bb_36_8_code, _data_measured
from qldpc.objects import Pauli, PauliXZ


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_build_single_ppm_circuit_noiseless_observables_zero(basis: PauliXZ) -> None:
    """Both OBSERVABLE_INCLUDEs evaluate to 0 (= +1) under no noise."""
    from qldpc.circuits.surgery.circuit.PPM_XZ import build_single_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    op = code.get_logical_ops(Pauli.X)[0] if basis is Pauli.X else code.get_logical_ops(Pauli.Z)[0]
    op_arr = np.asarray(op).astype(np.uint8)
    g = build_gadget(code, op_arr, basis=basis)
    circuit = build_single_ppm_circuit(g, rounds=3, noise_model=None)
    # Sample observables; all should be 0.
    sampler = circuit.compile_detector_sampler()
    _, obs = sampler.sample(shots=16, separate_observables=True)
    assert (obs == 0).all(), f"noiseless observables fired: {obs.sum()} flips across 16 shots"


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_build_single_ppm_circuit_noiseless_no_detector_fires(basis: PauliXZ) -> None:
    """Noiseless: NO detector fires (including the new final detectors).

    The total detector count must equal: round-1 reliable + (rounds-1)*all_checks + final reliable.
    Under noiseless conditions all of them must remain silent.
    """
    from qldpc.circuits.surgery.circuit.PPM_XZ import build_single_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    op = code.get_logical_ops(Pauli.X)[0] if basis is Pauli.X else code.get_logical_ops(Pauli.Z)[0]
    op_arr = np.asarray(op).astype(np.uint8)
    g = build_gadget(code, op_arr, basis=basis)
    circuit = build_single_ppm_circuit(g, rounds=3, noise_model=None)
    sampler = circuit.compile_detector_sampler()
    dets, _ = sampler.sample(shots=64, separate_observables=True)
    assert not dets.any(), (
        f"basis={basis}: {dets.sum()} detector fires noiselessly across {dets.shape[0]} shots"
    )


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
@pytest.mark.parametrize("destructive", [False, True])
def test_single_sector_keeps_only_measured_basis(basis: PauliXZ, destructive: bool) -> None:
    """single_sector=True keeps only the measured-basis detectors (lanes 2,3 for X̄;
    4,5 for Z̄), preserving the observable while shrinking the DEM. The complementary
    sector is still physically measured (the merge needs it) — only its DETECTORs drop.

    Valid only for CSS-type PPM; obs0 = X̄/Z̄ is flipped solely by the opposite single
    error type, which fires the measured-basis sector (Bombin/Cohen homological
    measurement, arXiv:2410.02753 §3).
    """
    from qldpc.circuits import DepolarizingNoiseModel
    from qldpc.circuits.surgery.circuit.PPM_XZ import build_single_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    op = code.get_logical_ops(Pauli.X)[0] if basis is Pauli.X else code.get_logical_ops(Pauli.Z)[0]
    op_arr = np.asarray(op).astype(np.uint8)
    g = build_gadget(code, op_arr, basis=basis)
    noise = DepolarizingNoiseModel(0.01)
    kw = dict(rounds=3, noise_model=noise, destructive_measure_data=destructive)
    full = build_single_ppm_circuit(g, **kw)
    single = build_single_ppm_circuit(g, single_sector=True, **kw)

    assert single.num_observables == full.num_observables  # observable preserved
    full_dem, single_dem = full.detector_error_model(), single.detector_error_model()
    assert single_dem.num_detectors < full_dem.num_detectors
    assert single_dem.num_errors < full_dem.num_errors
    kept = {2, 3} if basis is Pauli.X else {4, 5}  # measured-basis lanes only
    lanes = {int(c[1]) for c in single.get_detector_coordinates().values()}
    assert lanes <= kept, f"single_sector kept complementary-basis lanes {lanes - kept}"


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_single_sector_preserves_observable_detectability(basis: PauliXZ) -> None:
    """Dropping the complementary sector leaves every observable-flipping error
    detectable: no error flips an observable with an empty detector set (fault
    distance intact).

    Re-anchored to the DESTRUCTIVE single-sector circuit. The prior
    ``destructive_measure_data=False`` build now emits 0 observables (the Cain et
    al. Appendix D experiment set is read from the final destructive data
    measurement), which made the assertion vacuous. The destructive single-sector
    build emits the k+1 match-basis set and is the meaningful fault-distance case.
    """
    from qldpc.circuits import DepolarizingNoiseModel
    from qldpc.circuits.surgery.circuit.PPM_XZ import build_single_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    op = code.get_logical_ops(Pauli.X)[0] if basis is Pauli.X else code.get_logical_ops(Pauli.Z)[0]
    op_arr = np.asarray(op).astype(np.uint8)
    g = build_gadget(code, op_arr, basis=basis)
    noise = DepolarizingNoiseModel(0.01)
    single = build_single_ppm_circuit(
        g, rounds=3, noise_model=noise, destructive_measure_data=True, single_sector=True
    )
    assert single.num_observables == code.dimension + 1  # real observables to protect
    dem = single.detector_error_model()
    undetectable = sum(
        1
        for e in dem.flattened()
        if e.type == "error"
        and any(t.is_logical_observable_id() for t in e.targets_copy())
        and not any(t.is_relative_detector_id() for t in e.targets_copy())
    )
    assert undetectable == 0, f"{undetectable} observable-flipping errors became undetectable"


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_build_single_ppm_circuit_block_observables_full_k_block(basis: PauliXZ) -> None:
    """Match-basis single PPM emits the full k-logical block + time-like L = k+1
    observables on the k=8 BBCode [[36, 8]], all deterministic noiselessly.

    The k block logicals (indices 0..k-1) are the Cain et al. arXiv:2603.28627
    Appendix D 'block error' set (failure = ANY logical Pauli error), the same
    metric get_memory_experiment uses for the idling baseline; the time-like L at
    index k catches the merge's time-like errors. Replaces the removed
    block_observables=True flag (the experiment set is now always emitted).
    """
    import sympy

    from qldpc.circuits.surgery.circuit.PPM_XZ import build_single_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    xs, ys = sympy.symbols("x y")
    code = codes.BBCode({xs: 3, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)  # [[36, 8]]
    op = code.get_logical_ops(Pauli.X)[0] if basis is Pauli.X else code.get_logical_ops(Pauli.Z)[0]
    g = build_gadget(code, np.asarray(op).astype(np.uint8), basis=basis)
    circuit = build_single_ppm_circuit(
        g,
        rounds=3,
        noise_model=None,
        destructive_measure_data=True,
        single_sector=True,
        experiment_basis=basis,  # match-basis -> k+1
    )
    # match-basis: k block logicals (0..k-1) + time-like L (index k) = k+1
    assert circuit.num_observables == code.dimension + 1
    # noiseless: every emitted observable (block + L) is deterministic (= 0)
    _, obs = circuit.compile_detector_sampler().sample(shots=32, separate_observables=True)
    assert not obs.any(), f"basis={basis}: observable fired noiselessly ({obs.sum()})"


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_build_single_ppm_circuit_opposite_basis_k_minus_1(basis: PauliXZ) -> None:
    """Opposite-basis single PPM emits the k-1 block logicals commuting with L (no
    time-like L) on the k=8 BBCode [[36, 8]], all deterministic noiselessly.

    Replaces the removed block_observables flag: the opposite-basis experiment is
    the k-t = k-1 set of Cain et al. arXiv:2603.28627 Appendix D.
    """
    import sympy

    from qldpc.circuits.surgery.circuit.PPM_XZ import build_single_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    xs, ys = sympy.symbols("x y")
    code = codes.BBCode({xs: 3, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)  # [[36, 8]]
    op = code.get_logical_ops(Pauli.X)[0] if basis is Pauli.X else code.get_logical_ops(Pauli.Z)[0]
    g = build_gadget(code, np.asarray(op).astype(np.uint8), basis=basis)
    opp = Pauli.Z if basis is Pauli.X else Pauli.X
    circuit = build_single_ppm_circuit(
        g, rounds=3, noise_model=None, destructive_measure_data=True, experiment_basis=opp
    )
    assert circuit.num_observables == code.dimension - 1
    _, obs = circuit.compile_detector_sampler().sample(shots=32, separate_observables=True)
    assert not obs.any(), f"basis={basis}: opposite-basis observable fired noiselessly ({obs.sum()})"


def test_single_ppm_data_init_default_matches_pre_kwarg() -> None:
    """build_single_ppm_circuit(g, rounds=3) ≡ data_init=None ≡ data_init='+' for basis=X."""
    from qldpc.circuits.surgery.circuit.PPM_XZ import build_single_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    c_no_kwarg = build_single_ppm_circuit(g, rounds=3, noise_model=None)
    c_none = build_single_ppm_circuit(g, rounds=3, noise_model=None, data_init=None)
    c_plus = build_single_ppm_circuit(g, rounds=3, noise_model=None, data_init="+")
    assert str(c_no_kwarg) == str(c_none), "data_init=None must match no-kwarg call"
    assert str(c_no_kwarg) == str(c_plus), "data_init='+' broadcast must match default for basis=X"


@pytest.mark.parametrize(
    "bad_init,error_substr",
    [
        ("00", "does not match num data qubits"),  # wrong length: too short
        ("0" * 8, "does not match num data qubits"),  # wrong length: too long (Steane n=7)
        ("@" * 7, "invalid chars"),  # invalid character
        ("0123456", "invalid chars"),  # mixed valid + invalid
    ],
)
def test_data_init_validation(bad_init: object, error_substr: str) -> None:
    """Bad data_init raises ValueError with informative message."""
    from qldpc.circuits.surgery.circuit.PPM_XZ import build_single_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    with pytest.raises(ValueError, match=error_substr):
        build_single_ppm_circuit(g, rounds=3, noise_model=None, data_init=bad_init)  # type: ignore[arg-type]


def test_qubit_coords_layout_steane() -> None:
    """Steane single-PPM circuit emits QUBIT_COORDS in 6 semantic lanes.

    y=0 data (Steane ids 0..6), y=1 κ ancillas (3), y=2 data H_X ancillas
    (3), y=3 χ ancillas (3), y=4 data H_Z ancillas (3), y=5 G ancilla (1).
    Ordering chosen so y is monotonic in qubit ID for basis=X.
    """
    from qldpc.circuits.surgery.circuit.PPM_XZ import build_single_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    circuit = build_single_ppm_circuit(g, rounds=1, noise_model=None)

    # Parse QUBIT_COORDS lines: each line is "QUBIT_COORDS(x, y) qubit_id"
    coord_map: dict[int, tuple[int, int]] = {}
    for line in str(circuit).splitlines():
        line = line.strip()
        if not line.startswith("QUBIT_COORDS"):
            continue
        # "QUBIT_COORDS(x, y) qid" — parse "(x, y)" and qid
        head, qid_str = line.rsplit(" ", 1)
        tup = head[len("QUBIT_COORDS(") : -1]
        x_str, y_str = [t.strip() for t in tup.split(",")]
        coord_map[int(qid_str)] = (int(x_str), int(y_str))

    expected = {
        # data qubits on y=0 (unchanged)
        0: (0, 0),
        1: (1, 0),
        2: (2, 0),
        3: (3, 0),
        4: (4, 0),
        5: (5, 0),
        6: (6, 0),
        # κ ancillas on y=1 (was y=3)
        7: (0, 1),
        8: (1, 1),
        9: (2, 1),
        # data H_X ancillas on y=2 (was y=1)
        10: (0, 2),
        11: (1, 2),
        12: (2, 2),
        # χ ancillas on y=3 (was y=4)
        13: (0, 3),
        14: (1, 3),
        15: (2, 3),
        # data H_Z ancillas on y=4 (was y=2)
        16: (0, 4),
        17: (1, 4),
        18: (2, 4),
        # G ancilla on y=5 (unchanged)
        19: (0, 5),
    }
    assert coord_map == expected, f"\nexpected: {expected}\ngot:      {coord_map}"


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_single_ppm_non_destructive_detach_only(basis: PauliXZ) -> None:
    """``destructive_measure_data=False`` detaches κ but leaves the data encoded.

    Non-destructive (detach-only) mode: the κ ancillas are still measured (the
    split that restores the bare code), but the real data qubits are not. The
    Cain et al. arXiv:2603.28627 Appendix D experiment set is built from the
    FINAL destructive data readout, so with the data left encoded there is no
    end-of-circuit data measurement to form any block / time-like observable —
    the whole observable set and the destructive final detectors are dropped, so
    the detector count drops too. The full (destructive) build emits the k+1
    match-basis set, whereas the lean build emits 0.
    """
    from qldpc.circuits.surgery.circuit.PPM_XZ import build_single_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    log = np.asarray(code.get_logical_ops(basis)[0]).astype(np.uint8)
    g = build_gadget(code, log, basis=basis)

    full = build_single_ppm_circuit(g, rounds=2, noise_model=None)
    lean = build_single_ppm_circuit(
        g, rounds=2, noise_model=None, destructive_measure_data=False
    )

    assert full.num_observables == code.dimension + 1  # destructive: k+1 match-basis set
    assert _data_measured(lean, code.num_qudits) == set()  # data left encoded
    assert lean.num_observables == 0  # no destructive readout => no observable set
    assert lean.num_detectors < full.num_detectors  # destructive detectors gone
    lean.detector_error_model()  # still compiles


def test_single_ppm_destructive_default_unchanged() -> None:
    """``destructive_measure_data=True`` (default) is byte-identical to omitting it."""
    from qldpc.circuits.surgery.circuit.PPM_XZ import build_single_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    a = build_single_ppm_circuit(g, rounds=2, noise_model=None)
    b = build_single_ppm_circuit(
        g, rounds=2, noise_model=None, destructive_measure_data=True
    )
    assert str(a) == str(b)


def test_detector_coords_steane_round_1_reliable() -> None:
    """Steane single-PPM round-1 reliable detectors have lane ∈ {2, 5}.

    Round-1 reliable for basis=X gadget: 3 data H_X checks (lane=2) + 1 G
    check (lane=5). No χ or data H_Z because those aren't deterministic
    on the protocol-default |+⟩ init.

    DETECTOR coord order is ``(idx, lane, t)`` per stim convention
    (time last). The first two components ``(idx, lane)`` exactly match
    the QUBIT_COORDS ``(x, y)`` of the ancilla being measured.
    """
    from qldpc.circuits.surgery.circuit.PPM_XZ import build_single_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    circuit = build_single_ppm_circuit(g, rounds=1, noise_model=None)

    detector_coords: set[tuple[int, int, int]] = set()
    for line in str(circuit).splitlines():
        line = line.strip()
        if not line.startswith("DETECTOR"):
            continue
        # "DETECTOR(idx, lane, t) rec[-N] ..." — extract the tuple
        head = line.split(")")[0]
        tup = head[len("DETECTOR(") :]
        parts = [int(p.strip()) for p in tup.split(",")]
        assert len(parts) == 3
        detector_coords.add((parts[0], parts[1], parts[2]))

    expected = {(0, 2, 0), (1, 2, 0), (2, 2, 0), (0, 5, 0)}
    assert detector_coords == expected, f"\nexpected: {expected}\ngot:      {detector_coords}"


def test_detector_coords_basis_z_pauli_type_keyed_lanes() -> None:
    """basis=Z gadget: lanes are Pauli-type-keyed, NOT measured-vs-gauge keyed.

    Under the unified convention (matching basis=X, the ZZ joint, and the Y
    layout), each check family splits by Pauli type into original vs new rows:
    lane 2 = H_X, lane 3 = X-type extras (the X-gauge G for basis=Z), lane 4 =
    H_Z, lane 5 = Z-type extras (the measured S_Z'). So S_Z' lands on lane 5 —
    NOT lane 3 — even though it is the measured surgery stabilizer.

    For Steane logical-Z (3-qubit support, 3 X/Z-checks): G = H_X[C_0, V_0] is
    full-rank ⇒ the X-gauge is empty (lane 3 absent), and the 3 S_Z' rows sit on
    lane 5. Round-1 reliable detectors are the data H_Z rows (lane 4); S_Z' and
    the empty G are not deterministic on the |0⟩^n protocol-default init.

    DETECTOR coord order is ``(idx, lane, t)`` per stim convention; the QUBIT_COORDS
    coord is ``(idx, lane)`` — both carry the lane at the same slot.
    """
    from qldpc.circuits.surgery.circuit.PPM_XZ import build_single_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    circuit = build_single_ppm_circuit(g, rounds=1, noise_model=None)

    # The measured S_Z' rows (checks_z past the m_Z original H_Z rows) are
    # Z-type, so they MUST be on lane 5 (Pauli-type-keyed), not lane 3.
    m_Z = code.matrix_z.shape[0]
    n_meas = len(g.support)
    from qldpc.circuits.surgery.circuit.support import QubitIDs, _gadget_merged_csscode

    qubit_ids = QubitIDs.from_code(_gadget_merged_csscode(g))
    coords = circuit.get_final_qubit_coordinates()
    sz_prime_ids = qubit_ids.checks_z[m_Z : m_Z + n_meas]
    assert n_meas == 3, f"Steane Z̄ has 3 S_Z' rows; got {n_meas}"
    sz_lanes = {int(coords[cid][1]) for cid in sz_prime_ids}
    assert sz_lanes == {5}, (
        f"S_Z' (measured, Z-type) must be lane 5 under Pauli-type keying; got {sz_lanes}"
    )
    # X-gauge G is empty for this fixture ⇒ lane 3 carries no qubit.
    lane3_ids = [q for q, (_x, y) in coords.items() if int(y) == 3]
    assert lane3_ids == [], f"basis=Z Steane has empty X-gauge ⇒ lane 3 empty; got {lane3_ids}"

    detector_lanes: set[int] = set()
    for line in str(circuit).splitlines():
        line = line.strip()
        if not line.startswith("DETECTOR"):
            continue
        head = line.split(")")[0]
        tup = head[len("DETECTOR(") :]
        parts = [int(p.strip()) for p in tup.split(",")]
        detector_lanes.add(parts[1])

    # Round-1 reliable detectors are the data H_Z rows (lane 4) only.
    assert detector_lanes == {4}, (
        f"basis=Z round-1 reliable detectors should be data H_Z (lane 4) only; "
        f"got {detector_lanes}"
    )


def test_single_ppm_match_basis_emits_k_plus_1_observables() -> None:
    """Match-basis single PPM: k block observables + 1 time-like L = k+1."""
    from qldpc.circuits.surgery.circuit.PPM_XZ import build_single_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()  # k=1
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    circ = build_single_ppm_circuit(g, rounds=3, experiment_basis=Pauli.X)
    assert circ.num_observables == code.dimension + 1  # k + t, t=1


def test_single_ppm_opposite_basis_emits_k_minus_1_observables() -> None:
    """Opposite-basis single PPM: only the (k-1) block logicals commuting with L."""
    from qldpc.circuits.surgery.circuit.PPM_XZ import build_single_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    bb = _bb_36_8_code()  # dimension 8 -> k-1 = 7
    xop = np.asarray(bb.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    gbb = build_gadget(bb, xop, basis=Pauli.X)
    circ = build_single_ppm_circuit(gbb, rounds=3, experiment_basis=Pauli.Z)
    assert circ.num_observables == bb.dimension - 1


def test_single_ppm_observables_deterministic_noiseless() -> None:
    """Every observable (block + time-like L) is deterministic with no noise."""
    from qldpc.circuits.surgery.circuit.PPM_XZ import build_single_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    xop = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, xop, basis=Pauli.X)
    circ = build_single_ppm_circuit(g, rounds=3, experiment_basis=Pauli.X)
    sampler = circ.compile_detector_sampler()
    _, obs = sampler.sample(shots=64, separate_observables=True)
    assert not obs.any()  # every observable deterministic (=0) with no noise


# --- Task 8: memory-experiment observable count (baseline) ---


@pytest.mark.parametrize("make_code", [lambda: codes.SteaneCode(), _bb_36_8_code])
def test_memory_experiment_emits_k_logical_x_observables(make_code) -> None:
    """The memory experiment emits exactly k logical-X block observables.

    Reuses the existing public ``get_memory_experiment`` (NO new production code,
    NO ``keep_only_observable``). Per the design §3.3 the memory experiment is the
    apples-to-apples baseline for the surgery k+1/k-1 sets: it initializes |+>^k,
    runs QEC and a transversal X readout, emitting the k logical-X̄ block
    observables (Cain et al. arXiv:2603.28627 Appendix D, Memory experiment).
    Covers k=1 (Steane [[7,1,3]]) and k=8 (BBCode [[36,8]]).
    """
    from qldpc.circuits import get_memory_experiment

    code = make_code()
    circ = get_memory_experiment(code, basis=Pauli.X, num_rounds=2)
    assert circ.num_observables == code.dimension
