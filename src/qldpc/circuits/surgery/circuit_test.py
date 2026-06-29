"""Tests for src/qldpc/circuits/surgery/circuit.py (single + joint PPM)."""

from __future__ import annotations

import numpy as np
import pytest
import stim

from qldpc import codes
from qldpc.objects import Pauli, PauliXZ

from ._webster_fixture import (
    _webster_x_bar_operator,
    build_generalised_bicycle_code,
    load_webster_seed_set,
)


def test_classify_reliable_round1_checks_basis_x() -> None:
    """For basis=X: reliable round-1 checks are data H_X (first m_X X-checks)
    plus gauge-fix G (last n_comp_checks Z-checks)."""
    import galois

    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.circuits.surgery.circuit import _classify_reliable_round1_checks
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.codes.common import CSSCode

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    F2 = galois.GF(2)
    merged = CSSCode(
        F2(g.HX_merged.astype(np.int_).tolist()),
        F2(g.HZ_merged.astype(np.int_).tolist()),
        is_subsystem_code=False,
    )
    qubit_ids = QubitIDs.from_code(merged)
    reliable = _classify_reliable_round1_checks(g, qubit_ids)
    m_X = code.matrix_x.shape[0]
    m_Z = code.matrix_z.shape[0]
    # Reliable X-checks: first m_X of checks_x (the original data H_X rows)
    expected_x_reliable = set(qubit_ids.checks_x[:m_X])
    # Reliable Z-checks: last g.partial_0.shape[0] of checks_z (the gauge-fix G rows)
    expected_z_reliable = set(qubit_ids.checks_z[m_Z:])
    expected = expected_x_reliable | expected_z_reliable
    assert set(reliable) == expected, f"reliable={set(reliable)}, expected={expected}"


def test_classify_reliable_round1_checks_basis_z() -> None:
    """For basis=Z: reliable round-1 checks are data H_Z (first m_Z Z-checks)
    plus gauge-fix G (last n_comp_checks X-checks)."""
    import galois

    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.circuits.surgery.circuit import _classify_reliable_round1_checks
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.codes.common import CSSCode

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    F2 = galois.GF(2)
    merged = CSSCode(
        F2(g.HX_merged.astype(np.int_).tolist()),
        F2(g.HZ_merged.astype(np.int_).tolist()),
        is_subsystem_code=False,
    )
    qubit_ids = QubitIDs.from_code(merged)
    reliable = _classify_reliable_round1_checks(g, qubit_ids)
    m_X = code.matrix_x.shape[0]
    m_Z = code.matrix_z.shape[0]
    # basis=Z: data H_Z rows are first m_Z Z-checks; G rows are last g.partial_0.shape[0] X-checks
    expected_z_reliable = set(qubit_ids.checks_z[:m_Z])
    expected_x_reliable = set(qubit_ids.checks_x[m_X:])
    expected = expected_z_reliable | expected_x_reliable
    assert set(reliable) == expected


def test_surgery_state_prep_basis_x_resets() -> None:
    """basis=X: data RX (→|+⟩), kappa R (→|0⟩)."""
    import galois

    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.circuits.surgery.circuit import _surgery_state_prep
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.codes.common import CSSCode

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    F2 = galois.GF(2)
    merged = CSSCode(
        F2(g.HX_merged.astype(np.int_).tolist()),
        F2(g.HZ_merged.astype(np.int_).tolist()),
        is_subsystem_code=False,
    )
    qubit_ids = QubitIDs.from_code(merged)
    n_data = code.num_qudits
    data_ids = qubit_ids.data[:n_data]
    ancilla_ids = qubit_ids.data[n_data:]
    circuit = _surgery_state_prep(g, data_ids, ancilla_ids, bridge_ids=())
    text = str(circuit)
    assert f"RX {' '.join(str(q) for q in data_ids)}" in text
    assert f"R {' '.join(str(q) for q in ancilla_ids)}" in text


def test_surgery_state_prep_basis_z_resets() -> None:
    """basis=Z: data R (→|0⟩), kappa RX (→|+⟩)."""
    import galois

    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.circuits.surgery.circuit import _surgery_state_prep
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.codes.common import CSSCode

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    F2 = galois.GF(2)
    merged = CSSCode(
        F2(g.HX_merged.astype(np.int_).tolist()),
        F2(g.HZ_merged.astype(np.int_).tolist()),
        is_subsystem_code=False,
    )
    qubit_ids = QubitIDs.from_code(merged)
    n_data = code.num_qudits
    data_ids = qubit_ids.data[:n_data]
    ancilla_ids = qubit_ids.data[n_data:]
    circuit = _surgery_state_prep(g, data_ids, ancilla_ids, bridge_ids=())
    text = str(circuit)
    assert f"R {' '.join(str(q) for q in data_ids)}" in text
    assert f"RX {' '.join(str(q) for q in ancilla_ids)}" in text


def test_surgery_qec_cycle_round_1_detectors_classified() -> None:
    """Round-1 detectors are 1-arg only for RELIABLE checks; unreliable ones skipped."""
    import galois

    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.circuits.surgery.circuit import _classify_reliable_round1_checks, _surgery_qec_cycle
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.codes.common import CSSCode

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    F2 = galois.GF(2)
    merged = CSSCode(
        F2(g.HX_merged.astype(np.int_).tolist()),
        F2(g.HZ_merged.astype(np.int_).tolist()),
        is_subsystem_code=False,
    )
    qubit_ids = QubitIDs.from_code(merged)
    reliable = _classify_reliable_round1_checks(g, qubit_ids)

    circuit, meas_rec, det_rec = _surgery_qec_cycle(
        g,
        merged,
        num_rounds=2,
        qubit_ids=qubit_ids,
    )
    # Count round-1 1-arg DETECTORs (those appearing before any REPEAT_BLOCK).
    text = str(circuit)
    # Number of "DETECTOR" instructions in the first round (before the REPEAT block)
    # should equal len(reliable).
    first_round_str = text.split("REPEAT")[0]
    n_det = first_round_str.count("DETECTOR")
    assert n_det == len(reliable), (
        f"round-1 detectors={n_det}, expected len(reliable)={len(reliable)}"
    )


def test_surgery_detach_and_readout_basis_x_measures_ancilla_then_data() -> None:
    """basis=X: detach with M (Z-basis) on ancilla, then MX on data."""
    from qldpc.circuits.bookkeeping import MeasurementRecord
    from qldpc.circuits.surgery.circuit import _surgery_detach_and_readout
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    n_data = code.num_qudits
    data_ids = tuple(range(n_data))
    ancilla_ids = tuple(range(n_data, n_data + len(g.Q_prime)))
    bridge_ids = ()
    meas_rec = MeasurementRecord()
    circuit = _surgery_detach_and_readout(
        g,
        data_ids=data_ids,
        ancilla_ids=ancilla_ids,
        bridge_ids=bridge_ids,
        measurement_record=meas_rec,
    )
    text = str(circuit)
    # ancilla measured first (in Z), then data (in X)
    m_ancilla_idx = text.find(f"M {' '.join(str(q) for q in ancilla_ids)}")
    m_data_idx = text.find(f"MX {' '.join(str(q) for q in data_ids)}")
    assert m_ancilla_idx >= 0 and m_data_idx >= 0
    assert m_ancilla_idx < m_data_idx


def test_surgery_detach_and_readout_basis_z_measures_ancilla_in_x_then_data_in_z() -> None:
    """basis=Z: detach with MX on ancilla, then M on data."""
    from qldpc.circuits.bookkeeping import MeasurementRecord
    from qldpc.circuits.surgery.circuit import _surgery_detach_and_readout
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    n_data = code.num_qudits
    data_ids = tuple(range(n_data))
    ancilla_ids = tuple(range(n_data, n_data + len(g.Q_prime)))
    meas_rec = MeasurementRecord()
    circuit = _surgery_detach_and_readout(
        g,
        data_ids=data_ids,
        ancilla_ids=ancilla_ids,
        bridge_ids=(),
        measurement_record=meas_rec,
    )
    text = str(circuit)
    m_ancilla_idx = text.find(f"MX {' '.join(str(q) for q in ancilla_ids)}")
    m_data_idx = text.find(f"M {' '.join(str(q) for q in data_ids)}")
    assert m_ancilla_idx >= 0 and m_data_idx >= 0
    assert m_ancilla_idx < m_data_idx


def test_surgery_observable_emits_two_observable_include() -> None:
    """Direct unit test on _surgery_observable: emits two OBSERVABLE_INCLUDE entries.

    Observable 0 = XOR of the last QEC round's meas-check records (Webster,
    Smith, Cohen single-round identity Z̄ = ∏_v A_v, arXiv:2511.15989 §II.A).
    Observable 1 = XOR of data records on support (destructive cross-check).
    Asserts exactly two OBSERVABLE_INCLUDE lines are emitted with distinct
    observable indices."""
    from qldpc.circuits.bookkeeping import MeasurementRecord
    from qldpc.circuits.surgery.circuit import _surgery_observable
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    n_data = code.num_qudits
    meas_check_ids = tuple(range(100, 100 + len(g.support)))  # placeholder ids
    data_ids = tuple(range(n_data))
    meas_rec = MeasurementRecord()
    # Simulate 2 rounds of meas-check measurements
    for _ in range(2):
        meas_rec.append({cid: i for i, cid in enumerate(meas_check_ids)})
    # Simulate final data measurement
    meas_rec.append({d: i for i, d in enumerate(data_ids)})

    circuit = _surgery_observable(
        g,
        meas_check_ids=meas_check_ids,
        data_ids=data_ids,
        support_indices=g.support,
        measurement_record=meas_rec,
    )
    text = str(circuit)
    assert text.count("OBSERVABLE_INCLUDE") == 2  # PPM + cross-check
    assert "(0)" in text and "(1)" in text  # two distinct observable indices


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_build_single_ppm_circuit_noiseless_observables_zero(basis: PauliXZ) -> None:
    """Both OBSERVABLE_INCLUDEs evaluate to 0 (= +1) under no noise."""
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

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
def test_surgery_final_detectors_count_matches_reliable_round1(basis: PauliXZ) -> None:
    """Number of final DETECTORs equals |reliable round-1 set|.

    Tests the helper in isolation: build a circuit through detach_and_readout,
    then call _surgery_final_detectors and count emitted DETECTOR instructions.
    """
    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.circuits.surgery.circuit import (
        _classify_reliable_round1_checks,
        _gadget_merged_csscode,
        _surgery_detach_and_readout,
        _surgery_final_detectors,
        _surgery_qec_cycle,
    )
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    op = code.get_logical_ops(Pauli.X)[0] if basis is Pauli.X else code.get_logical_ops(Pauli.Z)[0]
    op_arr = np.asarray(op).astype(np.uint8)
    g = build_gadget(code, op_arr, basis=basis)
    merged = _gadget_merged_csscode(g)
    qubit_ids = QubitIDs.from_code(merged)
    n_data = code.num_qudits
    data_ids = qubit_ids.data[:n_data]
    ancilla_ids = qubit_ids.data[n_data:]

    # Simulate the pipeline through detach (we need measurement_record populated).
    _qec, mrec, _det = _surgery_qec_cycle(g, merged, num_rounds=2, qubit_ids=qubit_ids)
    _surgery_detach_and_readout(
        g,
        data_ids=data_ids,
        ancilla_ids=ancilla_ids,
        bridge_ids=(),
        measurement_record=mrec,
    )

    circuit = _surgery_final_detectors(g, merged, qubit_ids, measurement_record=mrec)
    n_final_det = str(circuit).count("DETECTOR")
    expected = len(_classify_reliable_round1_checks(g, qubit_ids))
    assert n_final_det == expected, (
        f"basis={basis}: emitted {n_final_det} DETECTORs, expected {expected}"
    )


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_build_single_ppm_circuit_noiseless_no_detector_fires(basis: PauliXZ) -> None:
    """Noiseless: NO detector fires (including the new final detectors).

    The total detector count must equal: round-1 reliable + (rounds-1)*all_checks + final reliable.
    Under noiseless conditions all of them must remain silent.
    """
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

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
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

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
    """Dropping the complementary sector leaves every obs0-flipping error detectable:
    no error flips the observable with an empty detector set (fault distance intact)."""
    from qldpc.circuits import DepolarizingNoiseModel
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    op = code.get_logical_ops(Pauli.X)[0] if basis is Pauli.X else code.get_logical_ops(Pauli.Z)[0]
    op_arr = np.asarray(op).astype(np.uint8)
    g = build_gadget(code, op_arr, basis=basis)
    noise = DepolarizingNoiseModel(0.01)
    single = build_single_ppm_circuit(
        g, rounds=3, noise_model=noise, destructive_measure_data=False, single_sector=True
    )
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
def test_build_single_ppm_circuit_block_observables(basis: PauliXZ) -> None:
    """block_observables=True emits the full logical block — one observable per logical
    operator (k = code.dimension), read from the destructive data measurement — instead
    of just the measured operator obs0. This is the Cain et al. arXiv:2603.28627
    Ext. Data Fig. 1 'block error' convention (failure = ANY logical Pauli error), the
    same metric get_memory_experiment uses for the idling baseline.
    """
    import sympy

    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

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
        block_observables=True,
    )
    # full block: one observable per logical operator, not obs0/obs1
    assert circuit.num_observables == code.dimension
    # noiseless: every block observable is deterministic (= 0)
    _, obs = circuit.compile_detector_sampler().sample(shots=32, separate_observables=True)
    assert not obs.any(), f"basis={basis}: block observable fired noiselessly ({obs.sum()})"


def test_block_observables_requires_destructive_readout() -> None:
    """block_observables needs the destructive data measurement to infer all logicals."""
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    op = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, op, basis=Pauli.X)
    with pytest.raises(ValueError, match="block_observables"):
        build_single_ppm_circuit(g, rounds=2, destructive_measure_data=False, block_observables=True)


def test_stitch_intercode_basis_x_joint_logical_in_stabilizer() -> None:
    """(x_1, x_2, 0, 0, 0) lies in rowspan(H_X^merged) — joint X̄_l X̄_r is a stabilizer."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_csscode
    from qldpc.circuits.surgery.gadget import build_gadget

    code1 = codes.SteaneCode()
    code2 = codes.SteaneCode()
    x1 = np.asarray(code1.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    x2 = np.asarray(code2.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code1, x1, basis=Pauli.X)
    g_r = build_gadget(code2, x2, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
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
def test_stitch_intercode_both_bases_commute_and_singletons_excluded(basis: PauliXZ) -> None:
    import galois

    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_csscode
    from qldpc.circuits.surgery.gadget import build_gadget

    GF2 = galois.GF(2)
    code = codes.SteaneCode()
    if basis is Pauli.X:
        x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    else:
        x = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=basis)
    g_r = build_gadget(codes.SteaneCode(), x, basis=basis)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
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
def test_stitch_intracode_both_bases_commute(basis: PauliXZ) -> None:
    """Intra-code commutation for both bases. Use a Webster code with 2 distinct logicals.

    Steane intra-code (k=1) yields the degenerate joint X̄·X̄ = I case.
    """
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_csscode
    from qldpc.circuits.surgery.gadget import (
        build_gadget,
    )

    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    if basis is Pauli.X:
        x1 = _webster_x_bar_operator(data, "X_bar_1")
        x2 = _webster_x_bar_operator(data, "X_bar_k2p1")
    else:
        from ._webster_fixture import _webster_z_bar_operator

        x1 = _webster_z_bar_operator(data, "Z_bar_1")
        x2 = _webster_z_bar_operator(data, "Z_bar_k2p1")
    g_l = build_gadget(code, x1, basis=basis)
    g_r = build_gadget(code, x2, basis=basis)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    HZ = np.asarray(merged.matrix_z).astype(np.int_)
    product = (HX @ HZ.T) % 2
    assert np.array_equal(product, np.zeros_like(product))
    assert merged.dimension == code.dimension - 1


def test_build_joint_ppm_circuit_meas_check_ids_no_UB() -> None:
    """build_joint_ppm_circuit's noiseless first sample has zero detectors firing."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

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


def test_build_joint_ppm_circuit_intercode_noiseless_observables_zero() -> None:
    """Cross-check obs0 == obs1 per shot across all 4 parity inits.

    Previously asserted only ``obs.sum() == 0`` (via compile_detector_sampler)
    for a single |+⟩^n init, which was vacuous: noiseless flips are 0
    regardless of obs0's correctness, and parity=+1 trivially gave the
    expected 0. Now uses compile_sampler + raw XOR so noiseless obs0 and
    obs1 are the actual eigenvalue bits, and sweeps non-trivial parity inits
    so a regression in obs0 is caught.
    """
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(codes.SteaneCode(), x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    for data_init in [("+", "+"), ("-", "+"), ("+", "-"), ("-", "-")]:
        circuit, _ = build_joint_ppm_circuit(
            g_l,
            g_r,
            bridge,
            rounds=2,
            data_init=data_init,
        )
        raw = circuit.compile_sampler().sample(shots=8).astype(np.uint8)
        n_meas = raw.shape[1]
        obs_lines = [ln for ln in str(circuit).splitlines() if ln.startswith("OBSERVABLE_INCLUDE")]
        offs0 = [int(t.strip("rec[]")) for t in obs_lines[0].split() if t.startswith("rec[")]
        offs1 = [int(t.strip("rec[]")) for t in obs_lines[1].split() if t.startswith("rec[")]
        obs0 = np.bitwise_xor.reduce(raw[:, [n_meas + o for o in offs0]], axis=1)
        obs1 = np.bitwise_xor.reduce(raw[:, [n_meas + o for o in offs1]], axis=1)
        assert (obs0 == obs1).all(), (
            f"data_init={data_init!r}: obs0 disagrees with obs1 on "
            f"{(obs0 != obs1).sum()}/8 noiseless shots"
        )


@pytest.mark.parametrize("code_index", [0, 1, 2, 3])
def test_joint_xx_in_stabilizer_on_webster_intracode(code_index: int) -> None:
    """Webster BB codes 0..3 intra-code: (x_1, x_2 padded, 0...) is in rowspan(H_X^merged).

    Replaces deleted path-graph tests; pins the SkipTree adapter construction across
    the full Webster Table I code family rather than just code 0.
    """
    import galois

    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_csscode
    from qldpc.circuits.surgery.gadget import (
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
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
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
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit
    from qldpc.circuits.surgery.gadget import (
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


def test_single_ppm_data_init_default_matches_pre_kwarg() -> None:
    """build_single_ppm_circuit(g, rounds=3) ≡ data_init=None ≡ data_init='+' for basis=X."""
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    c_no_kwarg = build_single_ppm_circuit(g, rounds=3, noise_model=None)
    c_none = build_single_ppm_circuit(g, rounds=3, noise_model=None, data_init=None)
    c_plus = build_single_ppm_circuit(g, rounds=3, noise_model=None, data_init="+")
    assert str(c_no_kwarg) == str(c_none), "data_init=None must match no-kwarg call"
    assert str(c_no_kwarg) == str(c_plus), "data_init='+' broadcast must match default for basis=X"


def test_single_ppm_data_init_zero_obs0_obs1_agree() -> None:
    """data_init='0' on basis=X gadget → logical |0⟩: obs0 ≡ obs1 every shot."""
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    circuit = build_single_ppm_circuit(g, rounds=3, noise_model=None, data_init="0")
    sampler = circuit.compile_detector_sampler()
    _, observables = sampler.sample(shots=8, separate_observables=True)
    obs0, obs1 = observables[:, 0], observables[:, 1]
    agree = float((obs0 == obs1).mean())
    assert agree == 1.0, f"obs0 vs obs1 disagree on {int((1 - agree) * 8)} of 8 shots"


def test_joint_ppm_data_init_truth_table() -> None:
    """Joint Z̄⊗Z̄ on two Steane copies: 4 |a⟩|b⟩ inits give expected parity."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

    c1, c2 = codes.SteaneCode(), codes.SteaneCode()
    z1 = np.asarray(c1.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    z2 = np.asarray(c2.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g1 = build_gadget(c1, z1, basis=Pauli.Z)
    g2 = build_gadget(c2, z2, basis=Pauli.Z)
    bridge = build_bridge(g1, g2)
    n1 = c1.num_qudits
    cases = [
        ("0" * n1 + "0" * n1, 0),
        ("0" * n1 + "1" * n1, 1),
        ("1" * n1 + "0" * n1, 1),
        ("1" * n1 + "1" * n1, 0),
    ]
    for data_init, expected in cases:
        circuit, _ = build_joint_ppm_circuit(
            g1,
            g2,
            bridge,
            rounds=3,
            noise_model=None,
            data_init=data_init,
        )
        sampler = circuit.compile_sampler()
        raw = sampler.sample(shots=16).astype(np.uint8)
        n_meas = raw.shape[1]
        obs_lines = [ln for ln in str(circuit).splitlines() if ln.startswith("OBSERVABLE_INCLUDE")]
        offsets = [int(t.strip("rec[]")) for t in obs_lines[0].split() if t.startswith("rec[")]
        meas_idx = [n_meas + off for off in offsets]
        obs0 = np.bitwise_xor.reduce(raw[:, meas_idx], axis=1)
        rate = float(obs0.mean())
        assert rate == float(expected), (
            f"data_init={data_init!r} gave obs0 rate {rate:.3f}, expected {expected}"
        )


def test_joint_ppm_data_init_superposition() -> None:
    """c1 |0⟩ × c2 |+⟩: Z̄_2 random → obs0 ≡ obs1 every shot."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

    c1, c2 = codes.SteaneCode(), codes.SteaneCode()
    z1 = np.asarray(c1.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    z2 = np.asarray(c2.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g1 = build_gadget(c1, z1, basis=Pauli.Z)
    g2 = build_gadget(c2, z2, basis=Pauli.Z)
    bridge = build_bridge(g1, g2)
    n = c1.num_qudits
    circuit, _ = build_joint_ppm_circuit(
        g1,
        g2,
        bridge,
        rounds=3,
        noise_model=None,
        data_init="0" * n + "+" * n,
    )
    sampler = circuit.compile_sampler()
    raw = sampler.sample(shots=8).astype(np.uint8)
    n_meas = raw.shape[1]
    obs_lines = [ln for ln in str(circuit).splitlines() if ln.startswith("OBSERVABLE_INCLUDE")]
    cols = []
    for line in obs_lines:
        offsets = [int(t.strip("rec[]")) for t in line.split() if t.startswith("rec[")]
        meas_idx = [n_meas + off for off in offsets]
        cols.append(np.bitwise_xor.reduce(raw[:, meas_idx], axis=1))
    obs = np.stack(cols, axis=1)
    agree = float((obs[:, 0] == obs[:, 1]).mean())
    assert agree == 1.0, f"obs0 vs obs1 disagree on {int((1 - agree) * 8)} of 8 shots"


def test_joint_ppm_data_init_tuple_matches_per_qubit_string() -> None:
    """data_init=("0", "+") produces the same circuit as "0"*n + "+"*n."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

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
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

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
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

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
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

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
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

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
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

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


def _data_measured(circuit: stim.Circuit, n_data: int) -> set[int]:
    """Real-data qubit IDs (< n_data) appearing under any measurement op."""
    return {
        t.qubit_value
        for inst in circuit.flattened()
        if inst.name in ("M", "MX", "MY", "MZ")
        for t in inst.targets_copy()
        if t.is_qubit_target and t.qubit_value < n_data
    }


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_single_ppm_non_destructive_detach_only(basis: PauliXZ) -> None:
    """``destructive_measure_data=False`` detaches κ but leaves the data encoded.

    Non-destructive (detach-only) mode: the κ ancillas are still measured (the
    split), but the real data qubits are not. obs0 (the logical readout) is still
    emitted from the in-circuit meas-checks; obs1 (the destructive cross-check)
    and the destructive final detectors are dropped, so the detector count drops.
    """
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    log = np.asarray(code.get_logical_ops(basis)[0]).astype(np.uint8)
    g = build_gadget(code, log, basis=basis)

    full = build_single_ppm_circuit(g, rounds=2, noise_model=None)
    lean = build_single_ppm_circuit(
        g, rounds=2, noise_model=None, destructive_measure_data=False
    )

    assert _data_measured(lean, code.num_qudits) == set()  # data left encoded
    assert lean.num_observables == 1  # obs0 only (obs1 dropped)
    assert lean.num_detectors < full.num_detectors  # destructive detectors gone
    lean.detector_error_model()  # still compiles


def test_single_ppm_destructive_default_unchanged() -> None:
    """``destructive_measure_data=True`` (default) is byte-identical to omitting it."""
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    a = build_single_ppm_circuit(g, rounds=2, noise_model=None)
    b = build_single_ppm_circuit(
        g, rounds=2, noise_model=None, destructive_measure_data=True
    )
    assert str(a) == str(b)


def test_joint_ppm_non_destructive_detach_only() -> None:
    """ZZ joint: ``destructive_measure_data=False`` detaches but keeps data encoded."""
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

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
    assert _data_measured(lean, n_data) == set()  # data left encoded
    assert lean.num_observables == 1  # obs0 only (obs1 dropped)
    assert lean.num_detectors < full.num_detectors
    lean.detector_error_model()  # still compiles


def test_detector_coords_steane_round_1_reliable() -> None:
    """Steane single-PPM round-1 reliable detectors have lane ∈ {2, 5}.

    Round-1 reliable for basis=X gadget: 3 data H_X checks (lane=2) + 1 G
    check (lane=5). No χ or data H_Z because those aren't deterministic
    on the protocol-default |+⟩ init.

    DETECTOR coord order is ``(idx, lane, t)`` per stim convention
    (time last). The first two components ``(idx, lane)`` exactly match
    the QUBIT_COORDS ``(x, y)`` of the ancilla being measured.
    """
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

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
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    circuit = build_single_ppm_circuit(g, rounds=1, noise_model=None)

    # The measured S_Z' rows (checks_z past the m_Z original H_Z rows) are
    # Z-type, so they MUST be on lane 5 (Pauli-type-keyed), not lane 3.
    m_Z = code.matrix_z.shape[0]
    n_meas = len(g.support)
    from qldpc.circuits.surgery.circuit import QubitIDs, _gadget_merged_csscode

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


def test_joint_ppm_qubit_coords_intercode_layout() -> None:
    """Intercode joint Z̄⊗Z̄ on two Steane copies: QUBIT_COORDS lanes correct.

    n_l = n_r = 7; left data on y=0 at x=0..6; right data on y=0 at x=7..13.
    κ ancillas on y=1. Bridge data + cycle ancillas on y=6.
    """
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

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


def test_logical_state_init_zero_and_plus_broadcast() -> None:
    """'0' and '+' return length-n broadcast strings — trivial CSS prep."""
    from qldpc.circuits.surgery.circuit import logical_state_init

    code = codes.SteaneCode()
    n = code.num_qudits
    assert logical_state_init(code, "0", log_idx=0) == "0" * n
    assert logical_state_init(code, "+", log_idx=0) == "+" * n


def test_logical_state_init_one_flips_x_bar_support() -> None:
    """'1' = X̄_0 |0⟩_L: '1' on supp(X̄_0), '0' elsewhere."""
    from qldpc.circuits.surgery.circuit import logical_state_init

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
    from qldpc.circuits.surgery.circuit import logical_state_init

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
    from qldpc.circuits.surgery.circuit import logical_state_init

    code = codes.SteaneCode()
    with pytest.raises(ValueError, match="state"):
        logical_state_init(code, bad, log_idx=0)


def test_logical_state_init_missing_log_idx_raises() -> None:
    """log_idx is keyword-only with no default — omitting it raises TypeError."""
    from qldpc.circuits.surgery.circuit import logical_state_init

    code = codes.SteaneCode()
    with pytest.raises(TypeError, match="log_idx"):
        logical_state_init(code, "0")  # type: ignore[call-arg]


def test_logical_state_init_log_idx_selects_different_logical_qubit() -> None:
    """log_idx=i flips supp(X̄_i) — distinct from X̄_0 on k>1 codes."""
    import sympy

    from qldpc.circuits.surgery.circuit import logical_state_init

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
    from qldpc.circuits.surgery.circuit import logical_state_init

    code = codes.SteaneCode()  # k = 1; only log_idx=0 is valid
    with pytest.raises(IndexError, match="log_idx"):
        logical_state_init(code, "1", log_idx=log_idx)


@pytest.mark.parametrize("state,expected_obs0", [("0", 0), ("1", 1)])
def test_logical_state_init_end_to_end_steane_basis_z(state: str, expected_obs0: int) -> None:
    """Steane single-PPM (basis=Z) reads obs0 = int(state) deterministically.

    Steane has wt(Z̄_0) = 3 (odd), so naive broadcast `"1" * n` ALSO works
    — this test pins the helper to the textbook expectation on the
    historically-working code, catching any regression where the helper
    accidentally diverges from naive on this code.
    """
    from qldpc.circuits.surgery.circuit import (
        build_single_ppm_circuit,
        logical_state_init,
    )
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    z_bar = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z_bar, basis=Pauli.Z)
    circuit = build_single_ppm_circuit(
        g,
        rounds=3,
        noise_model=None,
        data_init=logical_state_init(code, state, log_idx=0),
    )
    # Raw measurement records — see lattice_surgery.ipynb §0 raw_observables.
    raw = circuit.compile_sampler().sample(shots=16).astype(np.uint8)
    n_meas = raw.shape[1]
    obs0_recs = []
    for ln in str(circuit).splitlines():
        if ln.startswith("OBSERVABLE_INCLUDE(0)"):
            obs0_recs = [int(t.strip("rec[]")) for t in ln.split() if t.startswith("rec[")]
            break
    obs0 = np.bitwise_xor.reduce(raw[:, [n_meas + off for off in obs0_recs]], axis=1)
    rate = float(obs0.mean())
    assert rate == float(expected_obs0), (
        f"state={state!r}: obs0 rate {rate:.3f} != expected {expected_obs0}"
    )


@pytest.mark.parametrize("state,expected_obs0", [("0", 0), ("1", 1)])
def test_logical_state_init_end_to_end_bbcode_basis_z(state: str, expected_obs0: int) -> None:
    """BBCode [[36, 8]] single-PPM (basis=Z): regression for even-weight Z̄.

    For BBCode (l=3, m=6) the chosen Z̄_0 has weight 8 (even), so naive
    broadcast `"1"*36` produces logical |0⟩_L (NOT |1⟩_L) and obs0=0,
    silently failing any truth table that hardcodes expected=1 for "1".

    The helper uses X̄_0 to flip the correct support, so obs0 tracks the
    textbook expectation. If this test ever returns obs0=0 for state="1",
    the helper has regressed to naive broadcast.
    """
    import sympy

    from qldpc.circuits.surgery.circuit import (
        build_single_ppm_circuit,
        logical_state_init,
    )
    from qldpc.circuits.surgery.gadget import build_gadget

    xs, ys = sympy.symbols("x y")
    code = codes.BBCode({xs: 3, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)
    z_bar = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    assert int(z_bar.sum()) % 2 == 0, "test premise broken: this BBCode should have even-wt Z̄_0"
    g = build_gadget(code, z_bar, basis=Pauli.Z)
    circuit = build_single_ppm_circuit(
        g,
        rounds=3,
        noise_model=None,
        data_init=logical_state_init(code, state, log_idx=0),
    )
    raw = circuit.compile_sampler().sample(shots=200).astype(np.uint8)
    n_meas = raw.shape[1]
    obs0_recs = []
    for ln in str(circuit).splitlines():
        if ln.startswith("OBSERVABLE_INCLUDE(0)"):
            obs0_recs = [int(t.strip("rec[]")) for t in ln.split() if t.startswith("rec[")]
            break
    obs0 = np.bitwise_xor.reduce(raw[:, [n_meas + off for off in obs0_recs]], axis=1)
    rate = float(obs0.mean())
    assert rate == float(expected_obs0), (
        f"state={state!r}: obs0 rate {rate:.3f} != expected {expected_obs0}. "
        f"This is the BBCode even-wt regression test — failure here means "
        f"logical_state_init is no better than naive '{state}' * n broadcast."
    )


@pytest.mark.parametrize("rounds", [1, 2, 3, 5, 10])
@pytest.mark.parametrize("state", ["0", "1"])
def test_multi_round_invariance_steane_basis_z(rounds: int, state: str) -> None:
    """obs0 reads the merged Z̄ eigenvalue independently of R.

    Webster, Smith, Cohen arXiv:2511.15989 §II.A gives the single-round
    identity Z̄ = ∏_{v ∈ support} A_v on the merged stabilizer group: the XOR
    of one round's meas-check outcomes equals the eigenvalue bit of Z̄. Cain
    et al. arXiv:2603.28627 §B.1 selects the final QEC round as the readout
    point; detectors carry the FT load round-to-round.

    Therefore obs0 = int(state) for every R ≥ 1:
      * state="0" (|0⟩^n → Z̄=+1): obs0 = 0
      * state="1" (|1⟩^n → Z̄=−1, wt(Z̄_Steane)=3 odd): obs0 = 1

    This R-invariance is exactly what the single-round identity guarantees;
    any round-index drift in _surgery_qec_cycle, _surgery_observable, or
    MeasurementRecord.get_target_rec would break it for some R. The previous
    XOR-across-R-rounds formula collapsed to R·m_v mod 2, which was silently
    0 for every even R — the bug this test now guards against.
    """
    from qldpc.circuits.surgery.circuit import (
        build_single_ppm_circuit,
        logical_state_init,
    )
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    z_bar = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z_bar, basis=Pauli.Z)
    circuit = build_single_ppm_circuit(
        g,
        rounds=rounds,
        noise_model=None,
        data_init=logical_state_init(code, state, log_idx=0),
    )
    raw = circuit.compile_sampler().sample(shots=200).astype(np.uint8)
    n_meas = raw.shape[1]
    obs0_recs = []
    for ln in str(circuit).splitlines():
        if ln.startswith("OBSERVABLE_INCLUDE(0)"):
            obs0_recs = [int(t.strip("rec[]")) for t in ln.split() if t.startswith("rec[")]
            break
    obs0 = np.bitwise_xor.reduce(raw[:, [n_meas + off for off in obs0_recs]], axis=1)
    rate = float(obs0.mean())
    # Webster single-round identity: obs0 = last-round XOR of meas-checks
    # = eigenvalue bit of Z̄ on the merged group, independent of R.
    expected_obs0 = int(state)
    assert rate == float(expected_obs0), (
        f"rounds={rounds}, state={state!r}: obs0 rate {rate:.3f} != "
        f"expected {expected_obs0} (Webster Z̄=∏A_v should hold for any R)"
    )


@pytest.mark.parametrize("error_qubit", list(range(7)))
def test_single_qubit_x_error_triggers_only_neighboring_z_checks_steane(
    error_qubit: int,
) -> None:
    """Inject X_ERROR(1.0) on data qubit ``error_qubit`` between state
    prep and the first QEC round of the Steane basis=Z PPM. Assert
    exactly the round-1 Z-stab detectors whose support contains
    ``error_qubit`` fire (by row index, not just count).

    Why X_ERROR (not data_init):
    * Stim's detector sampler reports ``actual XOR tableau-predicted``.
      A state-prep-only change is already known to the tableau, so
      detectors stay 0 (no deviation from prediction).
    * X_ERROR(1.0) is a noise channel — the tableau prediction is
      computed without noise, so applying X always deviates the
      measured Z-stab parities from the prediction, firing the
      affected detectors.

    Why this catches stim wiring bugs:
    * Round-1 reliable Z-checks compare measured syndrome to +1.
    * An X error on data qubit i flips the parity of every Z-stab whose
      support contains i — exactly those detectors must fire, no others.
    * CX target/control swap, wrong measurement basis, or EdgeColoring
      delaying a check to a later round all break this exact-match
      pattern loudly.
    * The assertion checks the FIRED SET against the expected set of
      Z-stab row indices (not just the count) — a bug that swaps rows
      while preserving cardinality is caught.
    """
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    z_bar = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z_bar, basis=Pauli.Z)
    clean_circuit = build_single_ppm_circuit(
        g,
        rounds=1,
        noise_model=None,
        data_init="0" * 7,
    )

    # Splice X_ERROR(1.0) at the boundary between state prep and QEC.
    # _surgery_state_prep emits only R, RX, X, Z instructions (closed
    # set) before the QEC cycle begins. Scan for the LAST such op and
    # insert immediately after — this is robust to future QEC ops
    # (MPP, XCX, etc.) that an open-set heuristic would misclassify.
    lines = str(clean_circuit).splitlines()
    prep_ops = ("R", "RX", "X", "Z")
    last_prep_idx = -1
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            continue  # pragma: no cover  -- stim's str() never emits blank lines today
        op = s.split()[0].split("(")[0]
        if op in prep_ops:
            last_prep_idx = i
    assert last_prep_idx >= 0, "could not locate any prep op (R/RX/X/Z) in Steane PPM circuit"
    injected_lines = (
        lines[: last_prep_idx + 1] + [f"X_ERROR(1.0) {error_qubit}"] + lines[last_prep_idx + 1 :]
    )
    injected_circuit = stim.Circuit("\n".join(injected_lines))

    sampler = injected_circuit.compile_detector_sampler()
    detection_events, _ = sampler.sample(
        shots=1,
        separate_observables=True,
    )
    events = detection_events[0]

    # Identify ROUND-1 reliable Z-side detectors via the clean reference:
    # deterministic-0 detectors emitted in the round-1 slab (time-coord
    # 0, before SHIFT_COORDS). Steane basis=Z rounds=1 emits 6 such
    # detectors total — 3 reliable round-1 Z-checks (time=0) and 3
    # final-readout cross-checks (time=1, after SHIFT_COORDS). We want
    # only the round-1 set: those are the ones flipped by X errors
    # injected before the first CZ extraction (the post-SHIFT detectors
    # check (round-1 syndrome) XOR (data-derived syndrome), which is
    # invariant under prep-time X errors and therefore stays at 0).
    #
    # The round-1 reliable detectors are emitted in data-H_Z row order
    # (set by _classify_reliable_round1_checks iterating
    # qubit_ids.checks_z[:m_Z]), so deterministic_zero_round1[j]
    # corresponds to H_Z row j.
    clean_sampler = clean_circuit.compile_detector_sampler()
    clean_events, _ = clean_sampler.sample(
        shots=256,
        separate_observables=True,
    )
    all_det_zero = np.where(clean_events.sum(axis=0) == 0)[0]
    det_coords = clean_circuit.get_detector_coordinates()
    deterministic_zero = np.array(
        [d for d in all_det_zero if det_coords[d][2] == 0.0],
        dtype=int,
    )

    HZ = np.asarray(code.matrix_z).astype(int)
    n_reliable_z = HZ.shape[0]  # 3 for Steane
    assert len(deterministic_zero) == n_reliable_z, (
        f"expected exactly {n_reliable_z} round-1 deterministic-zero "
        f"detectors on clean Steane basis=Z PPM (rounds=1), got "
        f"{len(deterministic_zero)} — reliable-check emission order may "
        f"have changed"
    )

    # Steane Z-stabs touching error_qubit (row indices)
    z_stabs_touching = set(int(j) for j in np.where(HZ[:, error_qubit] == 1)[0])
    # Map each round-1 deterministic-zero detector position (sorted by
    # emission order) to its corresponding Z-stab row index. The fired
    # set is the set of row indices whose detector fired.
    fired_z_stab_rows = {j for j in range(len(deterministic_zero)) if events[deterministic_zero[j]]}
    assert fired_z_stab_rows == z_stabs_touching, (
        f"X_ERROR on qubit {error_qubit}: expected Z-stab rows "
        f"{sorted(z_stabs_touching)} to fire, got "
        f"{sorted(fired_z_stab_rows)}. This is the syndrome-extraction "
        f"wiring regression: CX swap, wrong measurement basis, "
        f"EdgeColoring schedule bug, or a stabilizer row that was "
        f"reordered/replaced. The set comparison catches bugs that "
        f"swap detector contents while preserving cardinality."
    )


def test_joint_code_dimension_steane_x_steane_equals_one() -> None:
    """Intercode Steane × Steane joint PPM gives joint_code.dimension == 1.

    Formula: k_l + k_r − 1 because Z̄_l ⊗ Z̄_r becomes a stabilizer of
    the joint code after surgery. For k_l = k_r = 1, that's 1.

    Catches a stitching bug in _stitch_intercode that drops or
    duplicates a stabilizer row — CSS commutation would still hold
    but the joint code's logical dimension would shift.
    """
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

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
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

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


def test_joint_ppm_even_rounds_truth_table() -> None:
    """obs0 must encode logical X̄_l X̄_r parity correctly at EVEN rounds.

    Regression test for the bug where _surgery_observable XOR'd meas-check
    syndromes across all rounds (R · m_v ≡ 0 mod 2 for even R) instead of
    using a single round's product (Webster, Smith, Cohen arXiv:2511.15989
    §II.A: Z̄ = ∏_v A_v). Uses
    ``compile_sampler`` + manual XOR so we read the raw observable bit,
    not stim's noiseless-flip from its (possibly wrong) prediction.
    """
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(codes.SteaneCode(), x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
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
        offs0 = [int(t.strip("rec[]")) for t in obs_lines[0].split() if t.startswith("rec[")]
        offs1 = [int(t.strip("rec[]")) for t in obs_lines[1].split() if t.startswith("rec[")]
        obs0 = np.bitwise_xor.reduce(raw[:, [n_meas + o for o in offs0]], axis=1)
        obs1 = np.bitwise_xor.reduce(raw[:, [n_meas + o for o in offs1]], axis=1)
        assert (obs0 == expected).all(), (
            f"data_init={data_init!r}: obs0 has {(obs0 != expected).sum()}/"
            f"16 shots disagreeing with expected parity bit {expected}"
        )
        assert (obs0 == obs1).all(), f"data_init={data_init!r}: obs0 != obs1 in noiseless run"


def test_single_ppm_even_rounds_truth_table() -> None:
    """obs0 must encode single-patch X̄ (or Z̄) parity at EVEN rounds.

    Same regression as test_joint_ppm_even_rounds_truth_table but for the
    single-patch PPM construction. Sweeps "+" and "-" data inits in basis=X
    and "0", "1" in basis=Z to expose the cumulative-XOR bug at even rounds.
    Uses compile_sampler + manual XOR for the same reason as Task 1.
    """
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit, logical_state_init
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    basis_cases: list[tuple[PauliXZ, list[tuple[str, int]]]] = [
        (Pauli.X, [("+", 0), ("-", 1)]),
        (Pauli.Z, [("0", 0), ("1", 1)]),
    ]
    for basis, cases in basis_cases:
        op = (
            code.get_logical_ops(Pauli.X)[0]
            if basis is Pauli.X
            else code.get_logical_ops(Pauli.Z)[0]
        )
        op_arr = np.asarray(op).astype(np.uint8)
        g = build_gadget(code, op_arr, basis=basis)
        for state, expected in cases:
            data_init = logical_state_init(code, state=state, log_idx=0)
            circuit = build_single_ppm_circuit(
                g,
                rounds=2,
                noise_model=None,
                data_init=data_init,
            )
            raw = circuit.compile_sampler().sample(shots=16).astype(np.uint8)
            n_meas = raw.shape[1]
            obs_lines = [
                ln for ln in str(circuit).splitlines() if ln.startswith("OBSERVABLE_INCLUDE")
            ]
            offs0 = [int(t.strip("rec[]")) for t in obs_lines[0].split() if t.startswith("rec[")]
            offs1 = [int(t.strip("rec[]")) for t in obs_lines[1].split() if t.startswith("rec[")]
            obs0 = np.bitwise_xor.reduce(raw[:, [n_meas + o for o in offs0]], axis=1)
            obs1 = np.bitwise_xor.reduce(raw[:, [n_meas + o for o in offs1]], axis=1)
            assert (obs0 == expected).all(), (
                f"basis={basis!r} state={state!r}: obs0 has "
                f"{(obs0 != expected).sum()}/16 shots disagreeing with "
                f"expected parity bit {expected}"
            )
            assert (obs0 == obs1).all(), (
                f"basis={basis!r} state={state!r}: obs0 != obs1 in noiseless run"
            )


def test_keep_only_observable_drops_others_and_recurses_into_repeat() -> None:
    """keep_only_observable retains the matching OBSERVABLE_INCLUDE and recurses
    into REPEAT blocks, dropping all other observable IDs."""
    from qldpc.circuits.surgery.circuit import keep_only_observable

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


def test_expand_joint_data_init_rejects_non_str_non_seq_type() -> None:
    """_expand_joint_data_init raises TypeError on data_init that isn't str/tuple/list/None."""
    from qldpc.circuits.surgery.circuit import _expand_joint_data_init

    with pytest.raises(TypeError, match="data_init must be"):
        _expand_joint_data_init({"bad": "input"}, n_l=4, n_r=4, intercode=True)  # type: ignore[arg-type]


def test_single_ppm_dem_ok_bb_36_8_with_boost() -> None:
    """Single-PPM DEM constructs cleanly on BB [[36, 8]] with boost.

    Contract test: single-PPM does NOT call build_bridge / SkipTree, so the
    joint-PPM boost-drop and duplicate-edge bugs (fixed in bridge.py) cannot
    affect it. This regression locks that property in — both BB [[36, 8]]
    (duplicate weight-2 incidence rows on Z̄_0) AND boost (Cheeger h<1)
    simultaneously, the double-boundary case for the bridge bugs. If a future
    refactor accidentally routes single-PPM through bridge code, this test
    will catch it via stim's non-deterministic-detector rejection.
    """
    import sympy

    from qldpc.circuits.noise_model import DepolarizingNoiseModel
    from qldpc.circuits.surgery.cheeger import boost_gadget, cheeger_constant
    from qldpc.circuits.surgery.circuit import (
        build_single_ppm_circuit,
        keep_only_observable,
    )
    from qldpc.circuits.surgery.gadget import build_gadget

    xs, ys = sympy.symbols("x y")
    code = codes.BBCode({xs: 3, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    # Premise: restricted incidence has duplicate weight-2 rows.
    assert g.incidence.shape[0] > np.unique(g.incidence, axis=0).shape[0], (
        "test premise broken: BB [[36, 8]] Z̄_0 restriction should have duplicate κ rows"
    )
    if cheeger_constant(g) < 1.0:
        g = boost_gadget(g, method="combinatorial", target=1.0, max_extra_qubits=20, seed=3)

    noise = DepolarizingNoiseModel(1e-3, include_idling_error=False)
    circuit = build_single_ppm_circuit(g, rounds=3, noise_model=noise)
    stripped = keep_only_observable(circuit, keep_idx=0)
    dem = stripped.detector_error_model(approximate_disjoint_errors=True)
    assert dem.num_detectors > 0


def test_gf2_solve_consistent_returns_particular_solution():
    from qldpc.circuits.surgery.circuit import _gf2_solve

    A = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    b = np.array([1, 0], dtype=np.uint8)
    x = _gf2_solve(A, b)
    assert x is not None
    assert np.array_equal((A @ x) % 2, b)


def test_gf2_solve_inconsistent_returns_none():
    from qldpc.circuits.surgery.circuit import _gf2_solve

    A = np.array([[1, 0], [1, 0], [0, 0]], dtype=np.uint8)
    b = np.array([1, 0, 0], dtype=np.uint8)  # rows 0,1 demand x0=1 and x0=0
    assert _gf2_solve(A, b) is None


def test_gf2_solve_zero_rhs_returns_zero_vector():
    from qldpc.circuits.surgery.circuit import _gf2_solve

    A = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    b = np.array([0, 0], dtype=np.uint8)
    x = _gf2_solve(A, b)
    assert x is not None
    assert np.array_equal(x, np.zeros(2, dtype=np.uint8))
