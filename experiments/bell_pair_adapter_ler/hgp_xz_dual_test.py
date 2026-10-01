from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import stim

from qldpc import codes, decoders
from qldpc.codes.common import CSSCode

from . import hgp_xz_dual as dual
from .run_hgp_schedule_ablations import _ideal_prefix

OUTPUTS = Path(__file__).with_name("outputs")
REFERENCE = OUTPUTS / "ppm_basis_first_hgp_m4_d8_prep4_surg8_10000/samples"


@pytest.fixture(scope="module")
def large() -> dual.DualBuild:
    return dual.build_dual_experiment()


def test_m3_reproduces_archived_source_and_dual_without_distance_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> int:
        raise AssertionError("Exact distance search is forbidden")

    monkeypatch.setattr(CSSCode, "get_distance", fail)
    built = dual.build_dual_experiment(simplex_m=3, surface_distance=4, surgery_rounds=4)
    for label, experiment, basis in (
        ("hgp_simplex3_surface_unrotated_d4_hardware_prep4_surg4_bell0_20k", built.source, "X"),
        (
            "hgp_simplex3_surface_unrotated_d4_hardware_xz_dual_prep4_surg4_bell0_20k",
            built.experiment,
            "Z",
        ),
    ):
        directory = OUTPUTS / label
        entry = next(
            e
            for e in json.loads((directory / "manifest.json").read_text())
            if e["logical_basis"] == basis
        )
        assert experiment.circuit == stim.Circuit.from_file(directory / entry["circuit_path"])
        assert experiment.detector_error_model == stim.DetectorErrorModel.from_file(
            directory / entry["dem_path"]
        )


def test_m4_reference_mapping_noise_window_and_exact_dem(large: dual.DualBuild) -> None:
    entries = dual.validate_reference(REFERENCE, large)
    assert set(entries) == {"X", "Z"}
    assert large.experiment.detector_error_model == large.source.detector_error_model
    assert large.experiment.circuit.num_detectors == 1426
    assert large.experiment.detector_error_model.num_errors == 12884
    assert large.matching_z.detector_error_model != large.source.detector_error_model
    meta = large.experiment.metadata
    assert meta["surgery_schedule"]["local_order"] == ["Z", "X"]
    assert meta["surgery_schedule"]["bell_order"] == ["Z", "X"]
    assert meta["boundary_convention"] == "complete_reference_xz_transport"
    mapping = large.audit["source_data_columns_to_target"]
    assert mapping[1] == 15 and mapping[226] == 240
    assert mapping[451] == 458 and mapping[515] == 521
    assert mapping[563:] == list(range(563, 590))
    left, right = codes.HGPCode(codes.SimplexCode(4)), codes.SurfaceCode(8, rotated=False)
    assert _ideal_prefix(large.experiment.circuit).num_measurements == 4 * (
        left.num_checks + right.num_checks
    )
    active = set(meta["adapter_hardware_qubits"])
    terminal = set()
    for op in large.experiment.circuit.flattened():
        qs = [t.value for t in op.targets_copy()]
        if op.name == "DEPOLARIZE2":
            assert all({a, b} & active for a, b in zip(qs[::2], qs[1::2]))
        elif op.name in {"DEPOLARIZE1", "X_ERROR", "Y_ERROR", "Z_ERROR", "PAULI_CHANNEL_1"}:
            assert set(qs) <= active
        elif op.name in {"M", "MX"}:
            if any(op.gate_args_copy()):
                assert set(qs) <= active
            if set(qs) & set(range(563)):
                assert not any(op.gate_args_copy())
                terminal.update(qs)
    assert terminal == set(range(563))
    dets, obs = (
        large.experiment.circuit.without_noise()
        .compile_detector_sampler(seed=12)
        .sample(32, separate_observables=True)
    )
    assert not dets.any() and not obs.any()


def test_same_ordered_decoder_predictions(large: dual.DualBuild) -> None:
    dets, _, _ = large.source.detector_error_model.compile_sampler(seed=817).sample(4)
    source = decoders.SinterDecoder().compile_decoder_for_dem(large.source.detector_error_model)
    target = decoders.SinterDecoder().compile_decoder_for_dem(large.experiment.detector_error_model)
    assert np.array_equal(source.decode_shots(dets), target.decode_shots(dets))


def test_nonzero_noise_parameters_still_dual() -> None:
    built = dual.build_dual_experiment(
        simplex_m=2,
        surface_distance=2,
        preparation_rounds=2,
        surgery_rounds=2,
        physical_error_rate=0.002,
        bell_pair_infidelity=0.003,
    )
    assert built.experiment.detector_error_model == built.source.detector_error_model
    assert built.experiment.metadata["physical_error_rate"] == 0.002
    assert built.experiment.metadata["bell_pair_infidelity"] == 0.003


def test_reject_invalid_logical_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dual, "transpose_permutation", lambda left, distance, n: tuple(range(n)))
    with pytest.raises(ValueError, match="logical supports|Check supports"):
        dual.build_dual_experiment(simplex_m=3, surface_distance=4, surgery_rounds=1)


def test_reject_reference_mismatch(large: dual.DualBuild) -> None:
    bad = dataclasses.replace(large.source, metadata=dict(large.source.metadata, surgery_rounds=7))
    with pytest.raises(ValueError, match="matching basis-first"):
        dual.validate_reference(REFERENCE, dataclasses.replace(large, source=bad))
    bad = dataclasses.replace(large.source, circuit=large.source.circuit + stim.Circuit("TICK"))
    with pytest.raises(ValueError, match="circuit/DEM"):
        dual.validate_reference(REFERENCE, dataclasses.replace(large, source=bad))


@pytest.mark.parametrize(
    "options",
    [
        dict(simplex_m=1),
        dict(surface_distance=1),
        dict(surgery_rounds=0),
        dict(left_logical_index=-1),
        dict(physical_error_rate=float("nan")),
        dict(right_logical_index=1),
    ],
)
def test_bad_parameters(options: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        dual.build_dual_experiment(**options)  # type: ignore[arg-type]
