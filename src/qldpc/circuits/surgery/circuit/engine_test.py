"""Tests for qldpc.circuits.surgery.circuit.engine (reliable-check classification,
state prep, QEC cycle, observable emission, final detectors, detach + readout)."""

from __future__ import annotations

import numpy as np
import pytest

from qldpc import codes
from qldpc.objects import Pauli, PauliXZ


def test_reliable_checks_match_basis_x_reproduces_hx_and_gauge_slice() -> None:
    """Match-basis (experiment_basis=X on X-gadget): _reliable_checks reproduces
    the prior index-slicing — data H_X (first m_X X-checks) plus gauge-fix G (last
    n_comp_checks Z-checks)."""
    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.circuits.surgery.circuit.engine import _reliable_checks
    from qldpc.circuits.surgery.circuit.support import _gadget_merged_csscode
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    merged = _gadget_merged_csscode(g)
    qubit_ids = QubitIDs.from_code(merged)
    n_data = code.num_qudits
    reliable = _reliable_checks(
        g, merged, qubit_ids, experiment_basis=Pauli.X, n_data=n_data
    )
    m_X = code.matrix_x.shape[0]
    m_Z = code.matrix_z.shape[0]
    # Reliable X-checks: first m_X of checks_x (the original data H_X rows)
    expected_x_reliable = set(qubit_ids.checks_x[:m_X])
    # Reliable Z-checks: last g.partial_0.shape[0] of checks_z (the gauge-fix G rows)
    expected_z_reliable = set(qubit_ids.checks_z[m_Z:])
    expected = expected_x_reliable | expected_z_reliable
    assert set(reliable) == expected, f"reliable={set(reliable)}, expected={expected}"


def test_reliable_checks_match_basis_z_reproduces_hz_and_gauge_slice() -> None:
    """Match-basis (experiment_basis=Z on Z-gadget): _reliable_checks reproduces
    the prior index-slicing — data H_Z (first m_Z Z-checks) plus gauge-fix G (last
    n_comp_checks X-checks)."""
    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.circuits.surgery.circuit.engine import _reliable_checks
    from qldpc.circuits.surgery.circuit.support import _gadget_merged_csscode
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    merged = _gadget_merged_csscode(g)
    qubit_ids = QubitIDs.from_code(merged)
    n_data = code.num_qudits
    reliable = _reliable_checks(
        g, merged, qubit_ids, experiment_basis=Pauli.Z, n_data=n_data
    )
    m_X = code.matrix_x.shape[0]
    m_Z = code.matrix_z.shape[0]
    # basis=Z: data H_Z rows are first m_Z Z-checks; G rows are last g.partial_0.shape[0] X-checks
    expected_z_reliable = set(qubit_ids.checks_z[:m_Z])
    expected_x_reliable = set(qubit_ids.checks_x[m_X:])
    expected = expected_z_reliable | expected_x_reliable
    assert set(reliable) == expected


def test_reliable_checks_match_basis_x_gadget_reproduces_hx_and_gauge() -> None:
    """Match-basis (experiment_basis=X on X-gadget): the per-qubit basis rule
    marks the original H_X data rows reliable and the original H_Z data rows
    (which touch X-init data) unreliable."""
    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.circuits.surgery.circuit.engine import _reliable_checks
    from qldpc.circuits.surgery.circuit.support import _gadget_merged_csscode
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()  # [[7,1,3]] Steane; repo fixture for the brief's _x_gadget
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    merged = _gadget_merged_csscode(g)
    qids = QubitIDs.from_code(merged)
    n_data = code.num_qudits
    rel = set(_reliable_checks(g, merged, qids, experiment_basis=Pauli.X, n_data=n_data))
    m_X = code.matrix_x.shape[0]
    m_Z = code.matrix_z.shape[0]
    # original H_X rows are reliable; original H_Z rows (touch X-init data) are NOT
    assert set(qids.checks_x[:m_X]) <= rel
    assert not (set(qids.checks_z[:m_Z]) & rel)


def test_reliable_checks_opposite_basis_z_experiment_makes_all_z_checks_reliable() -> None:
    """Opposite-basis (experiment_basis=Z on X-gadget): data |0> + Q' |0> means
    every Z-type merged check is reliable and no X-type check is."""
    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.circuits.surgery.circuit.engine import _reliable_checks
    from qldpc.circuits.surgery.circuit.support import _gadget_merged_csscode
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    merged = _gadget_merged_csscode(g)
    qids = QubitIDs.from_code(merged)
    n_data = code.num_qudits
    rel = set(_reliable_checks(g, merged, qids, experiment_basis=Pauli.Z, n_data=n_data))
    # data |0> + Q' |0>  => every Z-type merged check is reliable; no X-type check is
    assert set(qids.checks_z) <= rel
    assert not (set(qids.checks_x) & rel)


def test_surgery_state_prep_basis_x_resets() -> None:
    """basis=X: data RX (→|+⟩), kappa R (→|0⟩)."""
    import galois

    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.circuits.surgery.circuit.engine import _surgery_state_prep
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget
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
    circuit = _surgery_state_prep(
        g, data_ids, ancilla_ids, bridge_ids=(), experiment_basis=g.basis
    )
    text = str(circuit)
    assert f"RX {' '.join(str(q) for q in data_ids)}" in text
    assert f"R {' '.join(str(q) for q in ancilla_ids)}" in text


def test_surgery_state_prep_basis_z_resets() -> None:
    """basis=Z: data R (→|0⟩), kappa RX (→|+⟩)."""
    import galois

    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.circuits.surgery.circuit.engine import _surgery_state_prep
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget
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
    circuit = _surgery_state_prep(
        g, data_ids, ancilla_ids, bridge_ids=(), experiment_basis=g.basis
    )
    text = str(circuit)
    assert f"R {' '.join(str(q) for q in data_ids)}" in text
    assert f"RX {' '.join(str(q) for q in ancilla_ids)}" in text


def test_surgery_qec_cycle_round_1_detectors_classified() -> None:
    """Round-1 detectors are 1-arg only for RELIABLE checks; unreliable ones skipped."""
    import galois

    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.circuits.surgery.circuit.engine import _reliable_checks, _surgery_qec_cycle
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget
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
    reliable = _reliable_checks(
        g, merged, qubit_ids, experiment_basis=g.basis, n_data=n_data
    )

    circuit, meas_rec, det_rec = _surgery_qec_cycle(
        g,
        merged,
        num_rounds=2,
        qubit_ids=qubit_ids,
        experiment_basis=g.basis,
        n_data=n_data,
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
    from qldpc.circuits.surgery.circuit.engine import _surgery_detach_and_readout
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

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
        experiment_basis=g.basis,
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
    from qldpc.circuits.surgery.circuit.engine import _surgery_detach_and_readout
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

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
        experiment_basis=g.basis,
    )
    text = str(circuit)
    m_ancilla_idx = text.find(f"MX {' '.join(str(q) for q in ancilla_ids)}")
    m_data_idx = text.find(f"M {' '.join(str(q) for q in data_ids)}")
    assert m_ancilla_idx >= 0 and m_data_idx >= 0
    assert m_ancilla_idx < m_data_idx


def test_state_prep_z_experiment_on_x_gadget_inits_data_in_z() -> None:
    """experiment_basis=Z on an X-gadget: data init in Z (R), no RX on data.

    The data init/readout basis is decoupled from gadget.basis (Cain et al.
    arXiv:2603.28627 Appendix D). For an X-gadget the ancilla (basis-X
    complement) inits with R, so an experiment_basis=Z data init (also R) means
    the circuit contains no RX at all.
    """
    from qldpc.circuits.surgery.circuit.engine import _surgery_state_prep
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()  # [[7,1,3]] Steane; repo fixture for the brief's _x_gadget
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    data_ids = tuple(range(g.code.num_qudits))
    anc_ids = tuple(range(g.code.num_qudits, g.code.num_qudits + len(g.Q_prime)))
    circ = _surgery_state_prep(g, data_ids, anc_ids, experiment_basis=Pauli.Z)
    text = str(circ)
    # data in Z -> R on data; ancilla (X-gadget complement = Z) -> R on ancilla; no RX
    assert "RX" not in text
    assert "R " in text or text.strip().startswith("R")


def test_detach_readout_z_experiment_on_x_gadget_measures_data_in_z() -> None:
    """experiment_basis=Z on an X-gadget: data readout in Z (M), no MX.

    Data readout basis is decoupled from gadget.basis (Cain et al.
    arXiv:2603.28627 Appendix D): experiment_basis=Z -> M on data. The X-gadget
    ancilla detach op is also M, so the circuit contains no MX.
    """
    from qldpc.circuits.bookkeeping import MeasurementRecord
    from qldpc.circuits.surgery.circuit.engine import _surgery_detach_and_readout
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    data_ids = tuple(range(g.code.num_qudits))
    anc_ids = tuple(range(g.code.num_qudits, g.code.num_qudits + len(g.Q_prime)))
    rec = MeasurementRecord()
    circ = _surgery_detach_and_readout(
        g,
        data_ids=data_ids,
        ancilla_ids=anc_ids,
        bridge_ids=(),
        measurement_record=rec,
        experiment_basis=Pauli.Z,
    )
    text = str(circ)
    assert "MX" not in text  # data measured with M (Z), ancilla measured with M (Z)
    assert "M " in text or "\nM" in text


def test_surgery_observable_match_basis_emits_k_block_plus_time_like_L() -> None:
    """Direct unit test on _surgery_observable (new k+1 match-basis layout).

    Match-basis (experiment_basis == gadget.basis) emits the k block logicals of
    experiment_basis at indices 0..k-1 (frame-corrected from the final data
    readout) plus the time-like L (XOR of the FIRST-cycle S'_meas outcomes) at
    index k — total k+1, per Cain et al. arXiv:2603.28627 Appendix D. For Steane
    (k=1) that is index 0 = block X̄, index 1 = time-like L.
    """
    from qldpc.circuits.bookkeeping import MeasurementRecord, QubitIDs
    from qldpc.circuits.surgery.circuit.engine import _surgery_observable
    from qldpc.circuits.surgery.circuit.support import _gadget_merged_csscode
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    merged = _gadget_merged_csscode(g)
    qids = QubitIDs.from_code(merged)
    n_data = code.num_qudits
    data_ids = qids.data[:n_data]
    qprime_ids = qids.data[n_data:]
    m_X = code.matrix_x.shape[0]
    n_V = len(g.support)
    meas_check_ids = tuple(qids.checks_x[m_X : m_X + n_V])

    meas_rec = MeasurementRecord()
    # 2 QEC rounds measure the meas-checks; then the detach measures Q' and data.
    for _ in range(2):
        meas_rec.append(dict.fromkeys(meas_check_ids, 0))
    meas_rec.append(dict.fromkeys(qprime_ids, 0))
    meas_rec.append(dict.fromkeys(data_ids, 0))

    logical_ops = np.asarray(code.get_logical_ops(Pauli.X)).astype(np.uint8)
    circuit = _surgery_observable(
        g,
        experiment_basis=Pauli.X,
        merged_code=merged,
        meas_check_ids=meas_check_ids,
        logical_ops=logical_ops,
        L_support=np.asarray(g.x).astype(np.uint8),
        n_data=n_data,
        data_ids=data_ids,
        qprime_ids=qprime_ids,
        bridge_ids=(),
        measurement_record=meas_rec,
    )
    text = str(circuit)
    # k=1 block logical at index 0 + time-like L at index 1 = k+1 = 2 entries.
    assert text.count("OBSERVABLE_INCLUDE") == code.dimension + 1
    assert "(0)" in text and "(1)" in text  # block X̄ + time-like L


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_surgery_final_detectors_count_matches_reliable_round1(basis: PauliXZ) -> None:
    """Number of final DETECTORs equals |reliable round-1 set|.

    Tests the helper in isolation: build a circuit through detach_and_readout,
    then call _surgery_final_detectors and count emitted DETECTOR instructions.
    """
    from qldpc.circuits.bookkeeping import QubitIDs
    from qldpc.circuits.surgery.circuit.engine import (
        _reliable_checks,
        _surgery_detach_and_readout,
        _surgery_final_detectors,
        _surgery_qec_cycle,
    )
    from qldpc.circuits.surgery.circuit.support import _gadget_merged_csscode
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

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
    _qec, mrec, _det = _surgery_qec_cycle(
        g, merged, num_rounds=2, qubit_ids=qubit_ids, experiment_basis=g.basis, n_data=n_data
    )
    _surgery_detach_and_readout(
        g,
        data_ids=data_ids,
        ancilla_ids=ancilla_ids,
        bridge_ids=(),
        measurement_record=mrec,
        experiment_basis=g.basis,
    )

    circuit = _surgery_final_detectors(
        g, merged, qubit_ids, measurement_record=mrec, experiment_basis=g.basis, n_data=n_data
    )
    n_final_det = str(circuit).count("DETECTOR")
    expected = len(
        _reliable_checks(g, merged, qubit_ids, experiment_basis=g.basis, n_data=n_data)
    )
    assert n_final_det == expected, (
        f"basis={basis}: emitted {n_final_det} DETECTORs, expected {expected}"
    )
