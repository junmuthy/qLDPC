"""Build fresh references for schedule tests instead of reading local production data."""

from pathlib import Path

import pytest

from qldpc import codes
from qldpc.objects import Pauli

from ..common.run_experiment import build_experiment, build_experiment_for_codes
from ..common.testing import write_reference
from . import hgp_xz_dual, run_hgp_dual_schedule


@pytest.fixture(scope="module")
def generated_hgp_references(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    reference = tmp_path_factory.mktemp("hgp-reference")
    left, right = codes.HGPCode(codes.SimplexCode(3)), codes.SurfaceCode(4, rotated=False)
    experiments = [
        build_experiment_for_codes(
            left_code=left,
            right_code=right,
            left_code_name="hgp_simplex3_98_18_4",
            right_code_name="surface_unrotated_d4",
            basis=basis,
            preparation_rounds=4,
            surgery_rounds=4,
            adapter_noise_targets="adapter_hardware",
            reduce_detectors=True,
        )
        for basis in (Pauli.X, Pauli.Z)
    ]
    write_reference(experiments, reference)
    dual_reference = tmp_path_factory.mktemp("hgp-dual-reference")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(run_hgp_dual_schedule, "REFERENCE", reference)
        dual, _ = run_hgp_dual_schedule.build_dual_experiment()
    write_reference([dual], dual_reference)
    return reference, dual_reference


@pytest.fixture(scope="module")
def generated_surface_reference(tmp_path_factory: pytest.TempPathFactory) -> Path:
    reference = tmp_path_factory.mktemp("surface-reference")
    for p, slug in ((0, "bell0"), (0.001, "bell0p001")):
        experiments = [
            build_experiment(
                basis=basis,
                rounds=rounds,
                bell_pair_infidelity=p,
                adapter_noise_targets="adapter_hardware",
                reduce_detectors=True,
            )
            for basis in (Pauli.X, Pauli.Z)
            for rounds in range(1, 6)
        ]
        write_reference(experiments, reference / slug)
    return reference


@pytest.fixture(scope="session")
def dual_m4() -> hgp_xz_dual.DualBuild:
    return hgp_xz_dual.build_dual_experiment()


@pytest.fixture(scope="session")
def generated_m4_reference(
    tmp_path_factory: pytest.TempPathFactory,
    dual_m4: hgp_xz_dual.DualBuild,
) -> Path:
    reference = tmp_path_factory.mktemp("m4-reference")
    write_reference([dual_m4.source, dual_m4.matching_z], reference)
    return reference
