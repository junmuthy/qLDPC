from __future__ import annotations

from typing import Any

import pytest

from qldpc import codes
from qldpc.objects import Pauli

from .run_experiment import (
    BuiltExperiment,
    build_experiment,
    build_experiment_for_codes,
    build_experiment_matrix,
)


@pytest.mark.parametrize(
    ("basis", "rounds", "expected_params", "expected_qubits"),
    [
        (Pauli.X, 3, (24, 1, 3), 52),
        (Pauli.Z, 3, (23, 1, 3), 50),
    ],
)
def test_build_experiment(
    basis: Pauli,
    rounds: int,
    expected_params: tuple[int, int, int],
    expected_qubits: int,
) -> None:
    experiment = build_experiment(basis=basis, rounds=rounds)

    assert experiment.metadata["joint_code_parameters"] == list(expected_params)
    assert experiment.metadata["num_qubits"] == expected_qubits
    assert experiment.metadata["num_observables"] == 1
    assert experiment.metadata["num_bell_pair_checks_per_round"] == 5
    assert experiment.metadata["preparation_rounds"] == 3
    assert experiment.metadata["surgery_rounds"] == rounds
    assert experiment.metadata["noise_scope"] == "intrinsic"
    assert experiment.detector_error_model.num_detectors == experiment.circuit.num_detectors
    assert experiment.detector_error_model.num_errors > 0
    assert "PAULI_CHANNEL_1" not in str(experiment.circuit)


def test_requested_matrix_contains_both_bases_and_round_counts() -> None:
    experiments = build_experiment_matrix()

    actual = {
        (experiment.metadata["logical_basis"], experiment.metadata["rounds"])
        for experiment in experiments
    }
    assert actual == {("X", 3), ("X", 5), ("Z", 3), ("Z", 5)}


@pytest.mark.parametrize(
    ("keyword", "value"),
    [("physical_error_rate", -0.1), ("bell_pair_infidelity", 1.1)],
)
def test_build_experiment_rejects_invalid_probabilities(keyword: str, value: float) -> None:
    kwargs: dict[str, Any] = {keyword: value}
    with pytest.raises(ValueError, match=keyword):
        build_experiment(basis=Pauli.X, rounds=3, **kwargs)


def test_build_experiment_for_multi_logical_css_code() -> None:
    """The reusable experiment accepts non-preset codes and logical indices."""
    left_code = codes.IcebergCode(4)
    right_code = codes.SteaneCode()
    experiment = build_experiment_for_codes(
        left_code=left_code,
        right_code=right_code,
        left_code_name="iceberg_4",
        right_code_name="steane",
        left_logical_index=1,
        basis=Pauli.X,
        preparation_rounds=2,
        surgery_rounds=2,
    )

    assert experiment.metadata["left_code"] == "iceberg_4"
    assert experiment.metadata["left_logical_index"] == 1
    assert experiment.metadata["left_code_parameters"] == list(left_code.get_code_params())
    assert experiment.metadata["num_observables"] == 1
    assert experiment.detector_error_model.num_errors > 0


def test_build_experiment_for_same_css_object_uses_independent_blocks() -> None:
    code = codes.SteaneCode()
    experiment = build_experiment_for_codes(
        left_code=code,
        right_code=code,
        basis=Pauli.Z,
        preparation_rounds=2,
        surgery_rounds=2,
    )

    assert experiment.metadata["joint_code_parameters"][0] >= 2 * code.num_qudits


def test_build_experiment_rejects_bad_scope_and_logical_index() -> None:
    with pytest.raises(ValueError, match="noise_scope"):
        build_experiment(
            basis=Pauli.X,
            rounds=2,
            noise_scope="unknown",  # type: ignore[arg-type]
        )
    with pytest.raises(IndexError, match="logical index"):
        build_experiment_for_codes(
            left_code=codes.SteaneCode(),
            right_code=codes.SteaneCode(),
            basis=Pauli.X,
            preparation_rounds=2,
            surgery_rounds=2,
            left_logical_index=1,
        )


def test_noise_scopes_control_preparation_and_closure() -> None:
    intrinsic = build_experiment(
        basis=Pauli.X,
        rounds=2,
        preparation_rounds=2,
        noise_scope="intrinsic",
    )
    operational = build_experiment(
        basis=Pauli.X,
        rounds=2,
        preparation_rounds=2,
        noise_scope="operational",
    )
    end_to_end = build_experiment(
        basis=Pauli.X,
        rounds=2,
        preparation_rounds=2,
        noise_scope="end-to-end",
    )

    noise_names = {
        "DEPOLARIZE1",
        "DEPOLARIZE2",
        "X_ERROR",
        "Y_ERROR",
        "Z_ERROR",
        "PAULI_CHANNEL_1",
        "PAULI_CHANNEL_2",
    }

    def measurements_before_first_noise(experiment: BuiltExperiment) -> int:
        measurements = 0
        for instruction in experiment.circuit.flattened():
            if instruction.name in noise_names:
                return measurements
            measurements += instruction.num_measurements
        raise AssertionError("experiment contains no physical noise instruction")

    def final_measurement_args(experiment: BuiltExperiment) -> list[float]:
        circuit = experiment.circuit
        return next(
            instruction.gate_args_copy()
            for instruction in reversed(list(circuit.flattened()))
            if instruction.name in {"M", "MX"}
        )

    expected_preparation_measurements = 2 * (
        codes.SurfaceCode(3).num_checks + codes.SteaneCode().num_checks
    )
    assert measurements_before_first_noise(intrinsic) == expected_preparation_measurements
    assert measurements_before_first_noise(operational) == expected_preparation_measurements
    assert measurements_before_first_noise(end_to_end) == 0
    assert final_measurement_args(intrinsic) == []
    assert final_measurement_args(operational) == [0.001]
    assert final_measurement_args(end_to_end) == [0.001]
