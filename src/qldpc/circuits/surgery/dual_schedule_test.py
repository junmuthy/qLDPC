"""Exact gate/noise transport, check validation, and record-order preservation."""

import numpy as np
import pytest
import stim

from qldpc import codes
from qldpc.circuits import QubitIDs
from qldpc.circuits.surgery import CSSXZDuality


def test_generic_css_mapping_and_bell_owners() -> None:
    # Unequal X/Z check counts: no Steane/surface/HGP or self-dual-code assumption.
    source = codes.CSSCode([[1, 1, 0]], [[1, 1, 0], [0, 0, 1]])
    target = codes.CSSCode(source.matrix_z, source.matrix_x)
    a = QubitIDs([8, 2, 9], [11, 15, 14], [20])
    b = QubitIDs([3, 4, 1], [10, 12, 13], [21])
    mapping = CSSXZDuality.from_codes(
        source,
        target,
        a,
        b,
        data_permutation=[0, 1, 2],
        source_bell_checks=[11],
        target_bell_checks=[13],
    )
    assert mapping.qubit_map == {8: 3, 2: 4, 9: 1, 11: 13, 15: 10, 14: 12, 20: 21}
    with pytest.raises(ValueError, match="Bell-check selections"):
        CSSXZDuality.from_codes(
            source,
            target,
            a,
            b,
            data_permutation=[0, 1, 2],
            source_bell_checks=[11],
            target_bell_checks=[10],
        )
    with pytest.raises(ValueError, match="owners"):
        CSSXZDuality.from_codes(source, target, a, b, data_permutation=[0, 1, 2])
    with pytest.raises(ValueError, match="bijection"):
        CSSXZDuality.from_codes(source, target, a, b, data_permutation=[0, 0, 2])
    with pytest.raises(ValueError, match="integers"):
        CSSXZDuality.from_codes(source, target, a, b, data_permutation=[0.0, 1, 2])
    with pytest.raises(ValueError, match="Check supports"):
        CSSXZDuality.from_codes(source, target, a, b, data_permutation=[2, 1, 0])


def test_transform_mixed_targets_noise_tags_and_repeats() -> None:
    mapping = CSSXZDuality((0,), {0: 4, 1: 3, 2: 2})
    original = stim.Circuit("""
        QUBIT_COORDS(0, 1) 0
        RX[prepare] 0 1
        X_ERROR(0.1) 0 1
        PAULI_CHANNEL_1(0.01, 0.02, 0.03) 0 1
        CX 1 0 1 2
        DEPOLARIZE2(0.04) 1 0 1 2
        REPEAT 2 {
            CZ 1 0
            TICK
            MRX(0.01) 0 1
            DETECTOR rec[-1] rec[-2]
        }
        OBSERVABLE_INCLUDE(0) rec[-2]
        SHIFT_COORDS(0, 0, 1)
    """)
    transformed = mapping.transform(original)
    expected = stim.Circuit("""
        QUBIT_COORDS(0, 1) 4
        R[prepare] 4
        RX[prepare] 3
        Z_ERROR(0.1) 4
        X_ERROR(0.1) 3
        PAULI_CHANNEL_1(0.03, 0.02, 0.01) 4
        PAULI_CHANNEL_1(0.01, 0.02, 0.03) 3
        CZ 3 4
        CX 3 2
        DEPOLARIZE2(0.04) 3 4 3 2
        REPEAT 2 {
            CX 3 4
            TICK
            MR(0.01) 4
            MRX(0.01) 3
            DETECTOR rec[-1] rec[-2]
        }
        OBSERVABLE_INCLUDE(0) rec[-2]
        SHIFT_COORDS(0, 0, 1)
    """)
    assert transformed == expected
    inverse = CSSXZDuality((4,), {v: k for k, v in mapping.qubit_map.items()})
    assert inverse.transform(transformed) == original
    assert transformed.num_ticks == original.num_ticks
    assert transformed.num_measurements == original.num_measurements


def test_noisy_quantum_channel_and_ideal_boundary_equivalence() -> None:
    reference = stim.Circuit("""
        RX 0 1
        CX 1 0
        DEPOLARIZE2(0.01) 1 0
        PAULI_CHANNEL_1(0.01, 0.02, 0.03) 0
        MX(0.02) 1
        MX 0
        DETECTOR rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    dual = CSSXZDuality((0,), {0: 2, 1: 3}).transform(reference)
    assert dual.detector_error_model() == reference.detector_error_model()
    assert dual.without_noise().detector_error_model().num_errors == 0
    dets, obs = (
        dual.without_noise().compile_detector_sampler(seed=1).sample(32, separate_observables=True)
    )
    assert not np.any(dets) and not np.any(obs)


@pytest.mark.parametrize("circuit", ["S 0", "CY 1 0", "CX 0 1", "M !0", "CX rec[-1] 0", "H 5"])
def test_unsupported_transform_fails_closed(circuit: str) -> None:
    with pytest.raises(ValueError):
        CSSXZDuality((0,), {0: 2, 1: 3}).transform(stim.Circuit(circuit))
