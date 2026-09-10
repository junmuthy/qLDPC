from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sinter

from qldpc import codes, decoders
from qldpc.circuits.detector_reduction import remove_inactive_detectors
from qldpc.codes.common import CSSCode
from qldpc.decoders.dems import DetectorErrorModelArrays
from qldpc.objects import Pauli

from .run_experiment import (
    BuiltExperiment,
    _filter_stats_for_experiments,
    _wilson_interval,
    build_experiment,
    build_experiment_for_codes,
    build_experiment_matrix,
    build_fidelity_sweep_matrix,
    write_artifacts,
)


def test_wilson_zero_and_all_error_endpoints_are_exact() -> None:
    assert _wilson_interval(0, 1000)[0] == 0
    assert _wilson_interval(1000, 1000)[1] == 1


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_hardware_reduction_preserves_error_distribution_and_paired_predictions(
    basis: Pauli,
) -> None:
    full = build_experiment(
        basis=basis,
        rounds=4,
        adapter_noise_targets="adapter_hardware",
        bell_pair_infidelity=0.001,
    )
    reduced = remove_inactive_detectors(full.circuit)
    assert 0 < reduced.circuit.num_detectors < full.circuit.num_detectors
    a = DetectorErrorModelArrays(full.detector_error_model)
    b = DetectorErrorModelArrays(reduced.detector_error_model)
    projected = DetectorErrorModelArrays.from_arrays(
        a.detector_flip_matrix[list(reduced.retained_detector_ids)],
        a.observable_flip_matrix,
        a.error_probs,
    )
    expected = {targets: prob for prob, targets in a.get_circuit_errors(projected.to_dem())}
    actual = {targets: prob for prob, targets in b.get_circuit_errors(b.to_dem())}
    assert actual == pytest.approx(expected, abs=1e-14)
    records = full.circuit.compile_sampler(seed=32).sample(128)
    detectors, obs = full.circuit.compile_m2d_converter().convert(
        measurements=records, separate_observables=True
    )
    small_detectors, small_obs = reduced.circuit.compile_m2d_converter().convert(
        measurements=records, separate_observables=True
    )
    assert np.array_equal(small_detectors, detectors[:, reduced.retained_detector_ids])
    assert np.array_equal(obs, small_obs)
    removed = sorted(set(range(full.circuit.num_detectors)) - set(reduced.retained_detector_ids))
    assert not np.any(detectors[:, removed])
    d1 = decoders.SinterDecoder().compile_decoder_for_dem(full.detector_error_model)
    d2 = decoders.SinterDecoder().compile_decoder_for_dem(reduced.detector_error_model)
    assert np.array_equal(d1.decode_shots(detectors), d2.decode_shots(small_detectors))


def test_variants_have_distinct_artifacts_and_keep_default_identity(tmp_path: Path) -> None:
    full = build_experiment(basis=Pauli.X, rounds=1)
    explicit = build_experiment(basis=Pauli.X, rounds=1, adapter_noise_targets="full_surgery")
    hardware = build_experiment(basis=Pauli.X, rounds=1, adapter_noise_targets="adapter_hardware")
    reduced = build_experiment(
        basis=Pauli.X, rounds=1, adapter_noise_targets="adapter_hardware", reduce_detectors=True
    )
    assert full.circuit == explicit.circuit
    assert list(full.metadata.items()) == list(explicit.metadata.items())
    assert "adapter_noise_targets" not in full.metadata  # Legacy Sinter identity stays unchanged.
    assert len({full.name, hardware.name, reduced.name}) == 3
    assert reduced.metadata["num_full_detectors"] == full.circuit.num_detectors
    assert len(reduced.metadata["retained_detector_ids"]) == reduced.circuit.num_detectors
    write_artifacts([full], tmp_path)
    previous = (tmp_path / "manifest.json").read_bytes()
    write_artifacts([explicit], tmp_path)
    with pytest.raises(ValueError, match="different experiment"):
        write_artifacts([hardware], tmp_path)
    assert (tmp_path / "manifest.json").read_bytes() == previous
    with pytest.raises(ValueError, match="adapter_noise_targets"):
        build_experiment(basis=Pauli.X, rounds=1, adapter_noise_targets="typo")  # type: ignore[arg-type]


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_hgp_hardware_mode_builds_without_distance_search(basis: Pauli) -> None:
    result = build_experiment_for_codes(
        left_code=codes.HGPCode(codes.SimplexCode(3)),
        right_code=codes.SurfaceCode(4, rotated=False),
        basis=basis,
        preparation_rounds=1,
        surgery_rounds=1,
        adapter_noise_targets="adapter_hardware",
        reduce_detectors=True,
    )
    assert result.metadata["left_code_distance_status"] == "not_computed"
    assert result.metadata["num_detectors"] < result.metadata["num_full_detectors"]


def test_noise_target_selection_is_independent_of_phase_scope() -> None:
    experiment = build_experiment(
        basis=Pauli.X,
        rounds=1,
        noise_scope="end-to-end",
        adapter_noise_targets="adapter_hardware",
    )
    assert "X_ERROR" in str(experiment.circuit) or "Z_ERROR" in str(experiment.circuit)
    terminal = next(
        op for op in reversed(list(experiment.circuit.flattened())) if op.name in {"M", "MX"}
    )
    assert terminal.gate_args_copy() == [0.001]


@pytest.mark.parametrize(
    ("basis", "rounds", "expected_params", "expected_qubits"),
    [
        (Pauli.X, 3, (24, 1, None), 52),
        (Pauli.Z, 3, (23, 1, None), 50),
    ],
)
def test_build_experiment(
    basis: Pauli,
    rounds: int,
    expected_params: tuple[int, int, int | None],
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
    assert experiment.metadata["left_code_distance_status"] == "known"
    assert experiment.metadata["right_code_distance_status"] == "known"
    assert experiment.metadata["joint_code_distance_status"] == "not_computed"
    assert experiment.detector_error_model.num_detectors == experiment.circuit.num_detectors
    assert experiment.detector_error_model.num_errors > 0
    assert "PAULI_CHANNEL_1" not in str(experiment.circuit)


def test_exact_distance_metadata_is_explicitly_opt_in() -> None:
    experiment = build_experiment(
        basis=Pauli.X,
        rounds=2,
        compute_exact_distances=True,
    )

    assert experiment.metadata["joint_code_parameters"] == [24, 1, 3]
    assert experiment.metadata["joint_code_distance_status"] == "exact"


def test_default_metadata_does_not_search_for_distance(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(*args: Any, **kwargs: Any) -> int:
        raise AssertionError("exact distance search should be opt-in")

    monkeypatch.setattr(CSSCode, "get_distance", fail_if_called)
    experiment = build_experiment(basis=Pauli.X, rounds=2)

    assert experiment.metadata["joint_code_parameters"] == [24, 1, None]
    assert experiment.metadata["joint_code_distance_status"] == "not_computed"


def test_requested_matrix_contains_both_bases_and_round_counts() -> None:
    experiments = build_experiment_matrix()

    actual = {
        (experiment.metadata["logical_basis"], experiment.metadata["rounds"])
        for experiment in experiments
    }
    assert actual == {("X", 3), ("X", 5), ("Z", 3), ("Z", 5)}


def test_fidelity_sweep_matrix_contains_every_requested_combination() -> None:
    experiments = build_fidelity_sweep_matrix(
        bell_pair_infidelities=(0.0, 0.001),
        bases=(Pauli.X,),
        rounds=(3, 4),
    )

    actual = {
        (
            experiment.metadata["logical_basis"],
            experiment.metadata["rounds"],
            experiment.metadata["bell_pair_infidelity"],
        )
        for experiment in experiments
    }
    assert actual == {
        ("X", 3, 0.0),
        ("X", 3, 0.001),
        ("X", 4, 0.0),
        ("X", 4, 0.001),
    }
    assert len({experiment.name for experiment in experiments}) == len(experiments)


@pytest.mark.parametrize("infidelities", [(), (0.001, 0.001)])
def test_fidelity_sweep_matrix_rejects_empty_or_duplicate_grid(
    infidelities: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="bell_pair_infidelities"):
        build_fidelity_sweep_matrix(bell_pair_infidelities=infidelities)


def test_auxiliary_sinter_data_are_filtered_by_exact_metadata() -> None:
    experiment = build_experiment(basis=Pauli.X, rounds=2)
    legacy_metadata = dict(experiment.metadata)
    legacy_metadata.pop("joint_code_distance_status")
    requested = sinter.TaskStats(
        strong_id="requested",
        decoder="qldpc_bp_osd",
        json_metadata=experiment.metadata,
        shots=100,
    )
    legacy = sinter.TaskStats(
        strong_id="legacy",
        decoder="qldpc_bp_osd",
        json_metadata=legacy_metadata,
        shots=100,
    )

    assert _filter_stats_for_experiments([requested, legacy], [experiment]) == [requested]


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
    assert experiment.metadata["left_code_parameters"] == [
        len(left_code),
        left_code.dimension,
        left_code.get_distance_if_known(),
    ]
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


def test_large_hgp_adapter_does_not_compute_unknown_distances() -> None:
    hgp_code = codes.HGPCode(codes.SimplexCode(3))
    surface_code = codes.SurfaceCode(4, rotated=False)
    experiment = build_experiment_for_codes(
        left_code=hgp_code,
        right_code=surface_code,
        basis=Pauli.X,
        preparation_rounds=1,
        surgery_rounds=1,
    )

    assert experiment.metadata["left_code_parameters"] == [98, 18, None]
    assert experiment.metadata["left_code_distance_status"] == "not_computed"
    assert experiment.metadata["right_code_parameters"] == [25, 1, 4]
    assert experiment.metadata["right_code_distance_status"] == "known"
    assert experiment.metadata["joint_code_parameters"] == [136, 18, None]
    assert experiment.metadata["joint_code_distance_status"] == "not_computed"


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
