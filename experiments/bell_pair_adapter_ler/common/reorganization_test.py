"""Guard circuit identity, default paths, and compatibility across directory moves."""

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from qldpc.objects import Pauli

from ..bell_fidelity.run_ppm_fidelity_sweep import task_for
from ..pilots.run_ppm_basis_first import code_pair
from .checkpoint_compatibility import matches_relocated_audit
from .paths import OUTPUT_ROOT, PACKAGE_ROOT, SOURCE_LOCATIONS, source_path
from .run_experiment import build_experiment_for_codes, default_output_dir

BASELINES = json.loads(Path(__file__).with_name("reorganization_baselines.json").read_text())


@pytest.mark.parametrize("row", BASELINES["conditions"])
def test_relocation_preserves_circuit_dem_and_task_id(row: dict[str, object]) -> None:
    left, right, ln, rn = code_pair(str(row["preset"]), simplex_m=int(row["m"]))
    experiment = build_experiment_for_codes(
        left_code=left,
        right_code=right,
        left_code_name=ln,
        right_code_name=rn,
        basis=Pauli[str(row["basis"])],
        preparation_rounds=4,
        surgery_rounds=int(row["rounds"]),
        adapter_noise_targets="adapter_hardware",
        reduce_detectors=True,
        schedule_policy="ppm_basis_first",
    )
    assert hashlib.sha256(str(experiment.circuit).encode()).hexdigest() == row["circuit"]
    assert hashlib.sha256(str(experiment.detector_error_model).encode()).hexdigest() == row["dem"]
    assert task_for(experiment).strong_id() == row["strong_id"]


def test_stable_paths_and_imports() -> None:
    assert OUTPUT_ROOT == PACKAGE_ROOT / "outputs"
    assert default_output_dir("full_surgery", False) == OUTPUT_ROOT
    for name, relative in SOURCE_LOCATIONS.items():
        module = importlib.import_module(
            "experiments.bell_pair_adapter_ler." + relative[:-3].replace("/", ".")
        )
        assert source_path(name) == Path(module.__file__).resolve()
        assert source_path(name).is_file()
    with pytest.raises(KeyError):
        source_path("../unknown.py")


@pytest.mark.parametrize(
    "name",
    [
        "run_experiment",
        "run_ppm_basis_first",
        "run_ppm_fidelity_sweep",
        "run_fidelity_sweep",
        "run_physical_noise_comparison",
    ],
)
def test_legacy_imports_alias_same_module(name: str) -> None:
    old = importlib.import_module("experiments.bell_pair_adapter_ler." + name)
    path = SOURCE_LOCATIONS[name + ".py"]
    new = importlib.import_module(
        "experiments.bell_pair_adapter_ler." + path[:-3].replace("/", ".")
    )
    assert old is new


def test_source_hash_compatibility_is_narrow_and_preserves_identity(tmp_path: Path) -> None:
    mapping = json.loads(Path(__file__).with_name("source_relocations.json").read_text())[
        "single_fault_audit.py"
    ]
    current = hashlib.sha256(source_path("single_fault_audit.py").read_bytes()).hexdigest()
    assert current == mapping["new_sha256"]
    saved = {"audit_source_sha256": mapping["old_sha256"], "circuit_sha256": "unchanged"}
    expected = dict(saved, audit_source_sha256=current)
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps(saved))
    before = identity.read_bytes()
    assert matches_relocated_audit(saved, expected, tmp_path)
    assert matches_relocated_audit(saved, expected, tmp_path)
    assert identity.read_bytes() == before
    assert (tmp_path / "source_relocation.json").is_file()
    assert not matches_relocated_audit(saved, dict(expected, circuit_sha256="changed"), tmp_path)
    assert not matches_relocated_audit(
        saved, dict(expected, audit_source_sha256="unknown"), tmp_path
    )
    assert not matches_relocated_audit(
        dict(saved, audit_source_sha256="unknown"), expected, tmp_path
    )
    (tmp_path / "source_relocation.json").write_text("{}")
    with pytest.raises(ValueError, match="receipt changed"):
        matches_relocated_audit(saved, expected, tmp_path)
