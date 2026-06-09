"""Tests for src/qldpc/circuits/surgery/circuit.py (single + joint PPM)."""

from __future__ import annotations

import numpy as np
import pytest
import stim

from qldpc import codes
from qldpc.objects import Pauli

from ._test_helpers import (
    load_webster_seed_set,
    build_generalised_bicycle_code,
    _webster_x_bar_operator,
)


def test_build_single_ppm_circuit_noiseless_compiles():
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    import stim
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    circuit = build_single_ppm_circuit(g, rounds=2, noise_model=None)
    assert isinstance(circuit, stim.Circuit)
    assert len(circuit) > 0


def test_build_single_ppm_circuit_noiseless_no_detectors_fire():
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    circuit = build_single_ppm_circuit(g, rounds=2, noise_model=None)
    sampler = circuit.compile_detector_sampler()
    samples = sampler.sample(shots=16)
    assert (samples == 0).all()


def test_build_single_ppm_circuit_with_noise_detectors_fire():
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits.noise_model import DepolarizingNoiseModel
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    circuit = build_single_ppm_circuit(
        g, rounds=2, noise_model=DepolarizingNoiseModel(p=0.05),
    )
    samples = circuit.compile_detector_sampler().sample(shots=200)
    assert samples.any()  # at least one detector fires under noise


def test_classify_reliable_round1_checks_basis_x():
    """For basis=X: reliable round-1 checks are data H_X (first m_X X-checks)
    plus gauge-fix G (last r Z-checks)."""
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.circuit import _classify_reliable_round1_checks
    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.codes.common import CSSCode
    import galois
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
    # Reliable Z-checks: last r of checks_z (the gauge-fix G rows)
    r = g.G.shape[0]
    expected_z_reliable = set(qubit_ids.checks_z[m_Z:])
    expected = expected_x_reliable | expected_z_reliable
    assert set(reliable) == expected, (
        f"reliable={set(reliable)}, expected={expected}"
    )


def test_classify_reliable_round1_checks_basis_z():
    """For basis=Z: reliable round-1 checks are data H_Z (first m_Z Z-checks)
    plus gauge-fix G (last r X-checks)."""
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.circuit import _classify_reliable_round1_checks
    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.codes.common import CSSCode
    import galois
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
    r = g.G.shape[0]
    # basis=Z: data H_Z rows are first m_Z Z-checks; G rows are last r X-checks
    expected_z_reliable = set(qubit_ids.checks_z[:m_Z])
    expected_x_reliable = set(qubit_ids.checks_x[m_X:])
    expected = expected_z_reliable | expected_x_reliable
    assert set(reliable) == expected


def test_surgery_state_prep_basis_x_resets():
    """basis=X: data RX (→|+⟩), kappa R (→|0⟩)."""
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.circuit import _surgery_state_prep
    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.codes.common import CSSCode
    import galois
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
    kappa_ids = qubit_ids.data[n_data:]
    circuit = _surgery_state_prep(g, data_ids, kappa_ids, bridge_ids=())
    text = str(circuit)
    assert f"RX {' '.join(str(q) for q in data_ids)}" in text
    assert f"R {' '.join(str(q) for q in kappa_ids)}" in text


def test_surgery_state_prep_basis_z_resets():
    """basis=Z: data R (→|0⟩), kappa RX (→|+⟩)."""
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.circuit import _surgery_state_prep
    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.codes.common import CSSCode
    import galois
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
    kappa_ids = qubit_ids.data[n_data:]
    circuit = _surgery_state_prep(g, data_ids, kappa_ids, bridge_ids=())
    text = str(circuit)
    assert f"R {' '.join(str(q) for q in data_ids)}" in text
    assert f"RX {' '.join(str(q) for q in kappa_ids)}" in text


def test_surgery_qec_cycle_round_1_detectors_classified():
    """Round-1 detectors are 1-arg only for RELIABLE checks; unreliable ones skipped."""
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.circuit import _surgery_qec_cycle, _classify_reliable_round1_checks
    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.codes.common import CSSCode
    import galois
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
        g, merged, num_rounds=2, qubit_ids=qubit_ids,
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


def test_surgery_detach_and_readout_basis_x_measures_kappa_then_data():
    """basis=X: detach with M (Z-basis) on κ, then MX on data."""
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.circuit import _surgery_detach_and_readout
    from qldpc.circuits.bookkeeping import MeasurementRecord
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    n_data = code.num_qudits
    data_ids = tuple(range(n_data))
    kappa_ids = tuple(range(n_data, n_data + len(g.kappa_qubits)))
    bridge_ids = ()
    meas_rec = MeasurementRecord()
    circuit = _surgery_detach_and_readout(
        g, data_ids=data_ids, kappa_ids=kappa_ids, bridge_ids=bridge_ids,
        measurement_record=meas_rec,
    )
    text = str(circuit)
    # κ measured first (in Z), then data (in X)
    m_kappa_idx = text.find(f"M {' '.join(str(q) for q in kappa_ids)}")
    m_data_idx = text.find(f"MX {' '.join(str(q) for q in data_ids)}")
    assert m_kappa_idx >= 0 and m_data_idx >= 0
    assert m_kappa_idx < m_data_idx


def test_surgery_detach_and_readout_basis_z_measures_kappa_in_x_then_data_in_z():
    """basis=Z: detach with MX on κ, then M on data."""
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.circuit import _surgery_detach_and_readout
    from qldpc.circuits.bookkeeping import MeasurementRecord
    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    n_data = code.num_qudits
    data_ids = tuple(range(n_data))
    kappa_ids = tuple(range(n_data, n_data + len(g.kappa_qubits)))
    meas_rec = MeasurementRecord()
    circuit = _surgery_detach_and_readout(
        g, data_ids=data_ids, kappa_ids=kappa_ids, bridge_ids=(),
        measurement_record=meas_rec,
    )
    text = str(circuit)
    m_kappa_idx = text.find(f"MX {' '.join(str(q) for q in kappa_ids)}")
    m_data_idx = text.find(f"M {' '.join(str(q) for q in data_ids)}")
    assert m_kappa_idx >= 0 and m_data_idx >= 0
    assert m_kappa_idx < m_data_idx


def test_surgery_observable_emits_two_observable_include():
    """Observable 0 = XOR of χ-row records across all rounds; Observable 1 = data measurement on V_0."""
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.circuit import _surgery_observable
    from qldpc.circuits.bookkeeping import MeasurementRecord
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    n_data = code.num_qudits
    chi_check_ids = tuple(range(100, 100 + len(g.V0)))   # placeholder ids
    data_ids = tuple(range(n_data))
    meas_rec = MeasurementRecord()
    # Simulate 2 rounds of chi-check measurements
    for _ in range(2):
        meas_rec.append({cid: i for i, cid in enumerate(chi_check_ids)})
    # Simulate final data measurement
    meas_rec.append({d: i for i, d in enumerate(data_ids)})

    circuit = _surgery_observable(
        g, chi_check_ids=chi_check_ids, data_ids=data_ids,
        v0_indices=g.V0, num_rounds=2, measurement_record=meas_rec,
    )
    text = str(circuit)
    assert text.count("OBSERVABLE_INCLUDE") == 2  # PPM + cross-check
    assert "(0)" in text and "(1)" in text  # two distinct observable indices


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_build_single_ppm_circuit_noiseless_observables_zero(basis):
    """Both OBSERVABLE_INCLUDEs evaluate to 0 (= +1) under no noise."""
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    code = codes.SteaneCode()
    op = (code.get_logical_ops(Pauli.X)[0]
          if basis is Pauli.X
          else code.get_logical_ops(Pauli.Z)[0])
    op_arr = np.asarray(op).astype(np.uint8)
    g = build_gadget(code, op_arr, basis=basis)
    circuit = build_single_ppm_circuit(g, rounds=3, noise_model=None)
    # Sample observables; all should be 0.
    sampler = circuit.compile_detector_sampler()
    _, obs = sampler.sample(shots=16, separate_observables=True)
    assert (obs == 0).all(), (
        f"noiseless observables fired: {obs.sum()} flips across 16 shots"
    )


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_single_ppm_circuit_noise_flips_observable_at_high_p(basis):
    """At p=0.1, the PPM observable (observable 0) flips ≥ 5% of shots."""
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits.noise_model import DepolarizingNoiseModel
    code = codes.SteaneCode()
    op = (code.get_logical_ops(Pauli.X)[0]
          if basis is Pauli.X
          else code.get_logical_ops(Pauli.Z)[0])
    op_arr = np.asarray(op).astype(np.uint8)
    g = build_gadget(code, op_arr, basis=basis)
    circuit = build_single_ppm_circuit(
        g, rounds=3, noise_model=DepolarizingNoiseModel(p=0.1),
    )
    sampler = circuit.compile_detector_sampler()
    _, obs = sampler.sample(shots=400, separate_observables=True)
    # Observable 0 (PPM) flips a nontrivial fraction at p=0.1
    obs_0_flip_rate = float(obs[:, 0].mean())
    assert obs_0_flip_rate >= 0.05, (
        f"PPM observable flip rate {obs_0_flip_rate:.2%} too low at p=0.1"
    )


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_surgery_final_detectors_count_matches_reliable_round1(basis):
    """Number of final DETECTORs equals |reliable round-1 set|.

    Tests the helper in isolation: build a circuit through detach_and_readout,
    then call _surgery_final_detectors and count emitted DETECTOR instructions.
    """
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.circuit import (
        _surgery_state_prep, _surgery_qec_cycle, _surgery_detach_and_readout,
        _surgery_final_detectors, _classify_reliable_round1_checks,
        _gadget_merged_csscode,
    )
    from qldpc.circuits.bookkeeping import QubitIDs

    code = codes.SteaneCode()
    op = (code.get_logical_ops(Pauli.X)[0] if basis is Pauli.X
          else code.get_logical_ops(Pauli.Z)[0])
    op_arr = np.asarray(op).astype(np.uint8)
    g = build_gadget(code, op_arr, basis=basis)
    merged = _gadget_merged_csscode(g)
    qubit_ids = QubitIDs.from_code(merged)
    n_data = code.num_qudits
    data_ids = qubit_ids.data[:n_data]
    kappa_ids = qubit_ids.data[n_data:]

    # Simulate the pipeline through detach (we need measurement_record populated).
    _qec, mrec, _det = _surgery_qec_cycle(g, merged, num_rounds=2, qubit_ids=qubit_ids)
    _surgery_detach_and_readout(
        g, data_ids=data_ids, kappa_ids=kappa_ids, bridge_ids=(),
        measurement_record=mrec,
    )

    circuit = _surgery_final_detectors(g, merged, qubit_ids, measurement_record=mrec)
    n_final_det = str(circuit).count("DETECTOR")
    expected = len(_classify_reliable_round1_checks(g, qubit_ids))
    assert n_final_det == expected, (
        f"basis={basis}: emitted {n_final_det} DETECTORs, expected {expected}"
    )


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_build_single_ppm_circuit_noiseless_no_detector_fires(basis):
    """Noiseless: NO detector fires (including the new final detectors).

    The total detector count must equal: round-1 reliable + (rounds-1)*all_checks + final reliable.
    Under noiseless conditions all of them must remain silent.
    """
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    code = codes.SteaneCode()
    op = (code.get_logical_ops(Pauli.X)[0] if basis is Pauli.X
          else code.get_logical_ops(Pauli.Z)[0])
    op_arr = np.asarray(op).astype(np.uint8)
    g = build_gadget(code, op_arr, basis=basis)
    circuit = build_single_ppm_circuit(g, rounds=3, noise_model=None)
    sampler = circuit.compile_detector_sampler()
    dets, _ = sampler.sample(shots=64, separate_observables=True)
    assert not dets.any(), (
        f"basis={basis}: {dets.sum()} detector fires noiselessly across {dets.shape[0]} shots"
    )


@pytest.mark.slow
def test_single_ppm_ler_monotone_in_p():
    """Tiny sinter sweep: PPM LER monotonically increasing in p.

    Catches gross protocol errors (wrong observable basis, sign flips, etc.).
    """
    import sinter
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits import DepolarizingNoiseModel
    from qldpc import decoders
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)

    error_rates = [0.001, 0.005, 0.02]
    tasks = []
    for p in error_rates:
        circuit = build_single_ppm_circuit(
            g, rounds=3, noise_model=DepolarizingNoiseModel(p),
        )
        tasks.append(sinter.Task(
            circuit=circuit,
            json_metadata={"p": float(p)},
        ))
    sinter_decoder = decoders.SinterDecoder()
    results = sinter.collect(
        tasks=tasks,
        decoders=["custom"],
        custom_decoders={"custom": sinter_decoder},
        num_workers=4,
        max_shots=2000,
        max_errors=30,
        print_progress=False,
    )
    by_p = {r.json_metadata["p"]: r.errors / max(r.shots, 1) for r in results}
    sorted_p = sorted(by_p.keys())
    ler_vals = [by_p[p] for p in sorted_p]
    print(f"LER values: {list(zip(sorted_p, ler_vals))}")
    # Monotonically non-decreasing (allow small statistical noise)
    for i in range(len(ler_vals) - 1):
        assert ler_vals[i] <= ler_vals[i + 1] * 1.5, (
            f"LER not monotonic: p={sorted_p[i]} → {ler_vals[i]}, "
            f"p={sorted_p[i+1]} → {ler_vals[i+1]}"
        )


@pytest.mark.slow
def test_single_ppm_ler_with_final_detectors_below_threshold():
    """With final detectors wired, LER at p=0.001 should be ≤ 0.01.

    Reference: before the final-detector wiring, LER at p=0.001 was ~0.024
    (from test_single_ppm_ler_monotone_in_p in the surgery-circuit-rewrite plan).
    Adding the inferred detectors should drop it significantly.
    """
    import sinter
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits import DepolarizingNoiseModel
    from qldpc import decoders

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)

    p = 0.001
    circuit = build_single_ppm_circuit(
        g, rounds=3, noise_model=DepolarizingNoiseModel(p),
    )
    sinter_decoder = decoders.SinterDecoder()
    results = sinter.collect(
        tasks=[sinter.Task(circuit=circuit, json_metadata={"p": float(p)})],
        decoders=["custom"],
        custom_decoders={"custom": sinter_decoder},
        num_workers=4,
        max_shots=5000,
        max_errors=50,
        print_progress=False,
    )
    assert len(results) == 1
    ler = results[0].errors / max(results[0].shots, 1)
    assert ler <= 0.01, (
        f"LER at p=0.001 = {ler:.4f} (errors={results[0].errors}/{results[0].shots} shots). "
        f"Expected ≤ 0.01 with final detectors wired. Was ~0.024 without them."
    )


def test_stitch_intercode_basis_x_css_commutation():
    """Inter-code Steane × Steane joint X̄X̄ merged code commutes."""
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_csscode
    code1 = codes.SteaneCode()
    code2 = codes.SteaneCode()
    x1 = np.asarray(code1.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    x2 = np.asarray(code2.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code1, x1, basis=Pauli.X)
    g_r = build_gadget(code2, x2, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    HZ = np.asarray(merged.matrix_z).astype(np.int_)
    product = (HX @ HZ.T) % 2
    assert np.array_equal(product, np.zeros_like(product))


def test_stitch_intercode_basis_x_k_reduces_by_one():
    """k_joint = k_l + k_r - 1 for inter-code Steane × Steane joint X̄X̄."""
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_csscode
    code1 = codes.SteaneCode()
    code2 = codes.SteaneCode()
    x1 = np.asarray(code1.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    x2 = np.asarray(code2.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code1, x1, basis=Pauli.X)
    g_r = build_gadget(code2, x2, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    assert merged.dimension == code1.dimension + code2.dimension - 1


def test_stitch_intercode_basis_x_joint_logical_in_stabilizer():
    """(x_1, x_2, 0, 0, 0) lies in rowspan(H_X^merged) — joint X̄_l X̄_r is a stabilizer."""
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_csscode
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


def test_stitch_intercode_basis_x_singletons_excluded():
    """(x_1, 0, ...) and (0, x_2, ...) alone are NOT in rowspan(H_X^merged)."""
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_csscode
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(codes.SteaneCode(), x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    import galois
    GF2 = galois.GF(2)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    n_l = code.num_qudits
    base = np.linalg.matrix_rank(GF2(HX.tolist()))
    for which in ("left", "right"):
        single = np.zeros(HX.shape[1], dtype=np.int_)
        if which == "left":
            single[:n_l] = x
        else:
            single[n_l : 2 * n_l] = x
        augmented = np.vstack([HX, single.reshape(1, -1)])
        assert np.linalg.matrix_rank(GF2(augmented.tolist())) == base + 1, which


def test_stitch_intracode_basis_x_css_commutation():
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_csscode
    code = codes.SteaneCode()
    x1 = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    # Use Pauli.X logical 0 for both (same V_0); intra-code test
    g_l = build_gadget(code, x1, basis=Pauli.X)
    g_r = build_gadget(code, x1, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    HZ = np.asarray(merged.matrix_z).astype(np.int_)
    product = (HX @ HZ.T) % 2
    assert np.array_equal(product, np.zeros_like(product))


def test_stitch_intracode_basis_x_k_reduces_by_one():
    # Use Webster code 0 (k>=2) so the k_joint = k_data - 1 invariant is not
    # masked by the spurious bridge X-logical: Steane (k=1) with x_l = x_r is
    # the degenerate joint X̄ · X̄ = I case where the spurious bridge logical
    # leaves the dimension at k_data instead of k_data - 1.
    from qldpc.circuits.surgery.gadget import (
        build_gadget,
    )
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_csscode

    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x1 = _webster_x_bar_operator(data, "X_bar_1")
    x2 = _webster_x_bar_operator(data, "X_bar_k2p1")
    g_l = build_gadget(code, x1, basis=Pauli.X)
    g_r = build_gadget(code, x2, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    assert merged.dimension == code.dimension - 1


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_stitch_intercode_both_bases_commute_and_singletons_excluded(basis):
    import galois
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_csscode
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
def test_stitch_intracode_both_bases_commute(basis):
    """Intra-code commutation for both bases. Use a Webster code with 2 distinct logicals.

    Steane intra-code (k=1) yields the degenerate joint X̄·X̄ = I case.
    """
    from qldpc.circuits.surgery.gadget import (
        build_gadget,
    )
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_csscode
    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    if basis is Pauli.X:
        x1 = _webster_x_bar_operator(data, "X_bar_1")
        x2 = _webster_x_bar_operator(data, "X_bar_k2p1")
    else:
        # Z-type analogs from the Webster seed set.
        l = data["l"]
        def _z_op(name):
            for s in data["seeds"]:
                if s["name"] == name and s["pauli_type"] == "Z":
                    v = np.zeros(2 * l, dtype=np.uint8)
                    v[:l][s["L_support"]] = 1
                    v[l:][s["R_support"]] = 1
                    return v
            raise ValueError(f"{name} not found")
        x1 = _z_op("Z_bar_1")
        x2 = _z_op("Z_bar_k2p1")
    g_l = build_gadget(code, x1, basis=basis)
    g_r = build_gadget(code, x2, basis=basis)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    HZ = np.asarray(merged.matrix_z).astype(np.int_)
    product = (HX @ HZ.T) % 2
    assert np.array_equal(product, np.zeros_like(product))
    assert merged.dimension == code.dimension - 1


def test_build_joint_ppm_circuit_chi_check_ids_no_UB():
    """build_joint_ppm_circuit's noiseless first sample has zero detectors firing."""
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit
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


def test_build_joint_ppm_circuit_intercode_noiseless_observables_zero():
    """basis=X init: data in |+⟩, κ in |0⟩, adapter in |0⟩. Joint observables are deterministic."""
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(codes.SteaneCode(), x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    circuit, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=2)
    sampler = circuit.compile_detector_sampler()
    _, obs = sampler.sample(8, separate_observables=True)
    # Joint observable 0 (chi XOR over rounds) deterministic = 0 in noiseless |+⟩^n init
    # Observable 1 (final M-X on V_0 ∪) deterministic = 0
    assert obs.sum() == 0


@pytest.mark.slow
def test_joint_ppm_ler_monotone_steane_intercode():
    """LER non-increasing in p across {1e-4, 3e-4, 1e-3} for Steane × Steane."""
    from qldpc.circuits.noise_model import DepolarizingNoiseModel
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(codes.SteaneCode(), x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    lers = []
    shots = 2000
    for p in (1e-3, 3e-4, 1e-4):
        nm = DepolarizingNoiseModel(p)
        circuit, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=3, noise_model=nm)
        sampler = circuit.compile_detector_sampler()
        _, obs = sampler.sample(shots, separate_observables=True)
        # logical error rate of OBS 0 (joint χ XOR)
        ler = (obs[:, 0] != 0).mean()
        lers.append(ler)
    # LER should be non-increasing as p decreases (tolerance 1.3× to absorb sampling noise)
    assert lers[0] >= lers[1] / 1.3, f"LER not monotone: {lers}"
    assert lers[1] >= lers[2] / 1.3, f"LER not monotone: {lers}"


@pytest.mark.parametrize("code_index", [0, 1, 2, 3])
def test_joint_xx_in_stabilizer_on_webster_intracode(code_index):
    """Webster BB codes 0..3 intra-code: (x_1, x_2 padded, 0...) is in rowspan(H_X^merged).

    Replaces deleted path-graph tests; pins the SkipTree adapter construction across
    the full Webster Table I code family rather than just code 0.
    """
    import galois
    from qldpc.circuits.surgery.gadget import (
        build_gadget,
    )
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_csscode
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
    assert np.linalg.matrix_rank(GF2(HX.tolist())) == np.linalg.matrix_rank(
        GF2(augmented.tolist())
    )


def test_build_joint_ppm_circuit_intracode_noiseless_observables_zero():
    """Intra-code Webster joint X̄_1·X̄_{k/2+1}: noiseless detectors + observables = 0.

    Replaces deleted path-graph noiseless intracode tests.
    """
    from qldpc.circuits.surgery.gadget import (
        build_gadget,
    )
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit
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
