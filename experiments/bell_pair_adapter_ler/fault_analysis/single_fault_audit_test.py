from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import stim

from ..common.testing import write_reference
from . import single_fault_audit as audit


def example(reverse: bool = False) -> stim.Circuit:
    targets = "1 0" if reverse else "0 1"
    return stim.Circuit(
        f"R 0 1\nDEPOLARIZE2(0.001) {targets}\nM(0.001) 0 1\n"
        "DETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]"
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_all_physical_outcomes_and_injection(reverse: bool) -> None:
    circuit = example(reverse)
    cat = audit.catalogue(circuit)
    m = cat["metrics"]
    assert m["noise_locations"] == 3
    assert m["physical_fault_outcomes"] == 17
    assert m["silent_fault_outcomes"] == 3
    assert m["logical_flip_outcomes"] == 9
    assert m["undetected_logical_outcomes"] == 5
    assert m["ambiguous_single_fault_syndromes_including_zero"] == 2
    assert m["no_fault_probability"] == pytest.approx(0.999**3)
    assert m["exactly_one_fault_probability"] == pytest.approx(3 * 0.001 * 0.999**2)
    audit.validate_injection(circuit, cat, count=17)


def test_channels_and_silent_measurement_fault() -> None:
    circuit = stim.Circuit(
        "R 0 1\nDEPOLARIZE1(0.03) 0\nPAULI_CHANNEL_1(0.01,0.02,0.03) 1\n"
        "X_ERROR(0.001) 0\nY_ERROR(0.002) 1\nZ_ERROR(0.003) 0\n"
        "M(0.004) 0 1\nOBSERVABLE_INCLUDE(0) rec[-1]"
    )
    cat = audit.catalogue(circuit)
    assert cat["metrics"]["physical_fault_outcomes"] == 11
    assert any(f["gate"] == "M" and f["signature"] == 0 for f in cat["faults"])
    audit.validate_injection(circuit, cat, 11)


def test_unsupported_noise_is_not_silently_ignored() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        audit.catalogue(stim.Circuit("R 0\nE(0.001) X0\nM 0\nDETECTOR rec[-1]"))


def test_nondeterministic_detectors_rejected() -> None:
    with pytest.raises(ValueError, match="non-deterministic"):
        audit.catalogue(stim.Circuit("R 0\nMX 0\nDETECTOR rec[-1]"))


def test_trace_hook_frame_is_not_a_logical_failure() -> None:
    circuit = stim.Circuit(
        "R 0 1 2\nH 2\nX_ERROR(0.001) 2\nCX 2 0 2 1\nTICK\nMX 2\nM 0 1\nDETECTOR rec[-1] rec[-2]"
    )
    cat = audit.catalogue(circuit)
    assert cat["faults"][0]["signature"] == 0
    trace = audit.trace_example(circuit, cat["faults"][0], 2)
    assert trace["peak_at_tick_boundaries"]["block_paulis"] == ["X0", "X1"]


def test_generated_m3_reference_phase_annotations_and_injection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left, right, ln, rn = audit.code_pair("hgp-simplex-surface")
    experiment = audit.build_experiment_for_codes(
        left_code=left,
        right_code=right,
        left_code_name=ln,
        right_code_name=rn,
        basis=audit.Pauli.X,
        preparation_rounds=4,
        surgery_rounds=4,
        adapter_noise_targets="adapter_hardware",
        reduce_detectors=True,
        schedule_policy="ppm_basis_first",
    )
    write_reference([experiment], tmp_path / "ppm_basis_first_hgp_m3_d4_prep4_surg4_bell_fidelity")
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    circuit, dem, meta, _ = audit.archive(4, 4, "X")
    audit.verify_config(circuit, dem, meta, 4, "X")
    cat = audit.catalogue(circuit, meta)
    assert cat["metrics"]["physical_fault_outcomes"] == 7430
    assert cat["metrics"]["undetected_logical_outcomes"] == 0
    assert cat["metrics"]["ambiguous_single_fault_syndromes_including_zero"] == 0
    assert {f["round"] for f in cat["faults"] if f["phase"] == "surgery"} == {1, 2, 3, 4}
    assert sum(f["phase"] == "adapter_initialization" for f in cat["faults"]) == 13
    assert sum(f["phase"] == "detachment" for f in cat["faults"]) == 13
    audit.validate_injection(circuit, cat)


def make_saved_example(path: Path) -> dict[str, Any]:
    circuit = example()
    cat = audit.catalogue(circuit)
    with gzip.open(path / "catalogue.json.gz", "wt") as stream:
        json.dump(cat, stream)
    (path / "model.dem").write_text(str(circuit.detector_error_model()))
    audit.save_json(path / "identity.json", {"runtime_identity": audit.runtime_identity()})
    return cat


def test_resumable_decoder_and_ambiguous_syndrome_accounting(tmp_path: Path) -> None:
    cat = make_saved_example(tmp_path)
    audit.decode_condition(str(tmp_path), 1, 5)
    first = (tmp_path / "decoder_progress.jsonl").read_text()
    audit.decode_condition(str(tmp_path), 2, 5)
    progress = (tmp_path / "decoder_progress.jsonl").read_text()
    assert progress.startswith(first)
    assert len(progress.splitlines()) == 2
    audit.decode_condition(str(tmp_path), 2, 5)
    assert (tmp_path / "decoder_progress.jsonl").read_text() == progress
    report = audit.summarize(tmp_path)
    assert report["complete_decoder_coverage"]
    assert report["covered_physical_outcomes"] == 17
    assert report["misdecoded_physical_outcomes"] >= 5
    failures = json.loads((tmp_path / "decoder_failures.json").read_text())
    expected = sum(f["probability"] * 0.999**2 for f in failures)
    assert report["known_single_fault_sector_failure_probability"] == pytest.approx(expected)
    assert len(audit.decoder_order(cat)) == 2


def test_decoder_budget_and_resume_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_saved_example(tmp_path)
    with pytest.raises(ValueError, match="Decoder/runtime changed"):
        monkeypatch.setattr(audit, "runtime_identity", lambda: {})
        audit.decode_condition(str(tmp_path), 2, 5)


def test_single_fault_injection_preserves_observables_and_measurements() -> None:
    circuit = example()
    cat = audit.catalogue(circuit)
    for f in cat["faults"]:
        injected = audit.inject_one(circuit, f)
        assert injected.num_measurements == circuit.num_measurements
        assert injected.num_detectors == circuit.num_detectors
        assert injected.num_observables == circuit.num_observables
        assert np.isclose(sum(g["probability"] for g in cat["faults"]), 0.003)
