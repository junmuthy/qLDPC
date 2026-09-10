"""Physical error propagation and logical information survive detector pruning."""

import numpy as np
import pytest
import stim

from qldpc.circuits.detector_reduction import remove_inactive_detectors


def test_reduction_keeps_propagated_errors_logical_only_errors_and_record_offsets() -> None:
    circuit = stim.Circuit("""
        R 0 1 2 3
        X_ERROR(0.1) 0
        CX 0 1
        CX 1 2
        M 0 1 2 3
        DETECTOR(0, 0) rec[-1]
        DETECTOR(2, 0) rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-3]
        REPEAT 2 {
            R 3
            M 3
            DETECTOR rec[-1]
        }
        R 3
        X_ERROR(0.2) 3
        M 3
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    result = remove_inactive_detectors(circuit)
    assert result.num_original_detectors == 4
    assert result.retained_detector_ids == (1,)
    assert result.detector_error_model == stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.2) L0
        detector(2, 0) D0
    """)
    # Use the same measurement records in both converters (not independent RNG streams).
    records = circuit.compile_sampler(seed=5).sample(1000)
    dets, obs = circuit.compile_m2d_converter().convert(
        measurements=records, separate_observables=True
    )
    reduced_dets, reduced_obs = result.circuit.compile_m2d_converter().convert(
        measurements=records, separate_observables=True
    )
    assert np.array_equal(dets[:, result.retained_detector_ids], reduced_dets)
    assert np.array_equal(obs, reduced_obs)
    assert not np.any(dets[:, [0, 2, 3]])


def test_reduction_rejects_gauge_detectors_and_preserves_noiseless_observable() -> None:
    with pytest.raises(ValueError, match="non-deterministic detectors"):
        remove_inactive_detectors(stim.Circuit("RX 0\nM 0\nDETECTOR rec[-1]"))
    result = remove_inactive_detectors(
        stim.Circuit("R 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")
    )
    assert result.circuit.num_detectors == 0
    assert result.circuit.num_observables == 1
    assert result.circuit.num_measurements == 1
