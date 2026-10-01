from __future__ import annotations

import gzip
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import stim

from . import effective_distance_audit as audit
from .single_fault_audit import catalogue


def repetition(distance: int) -> stim.Circuit:
    circuit = stim.Circuit()
    circuit.append("R", range(distance))
    circuit.append("X_ERROR", range(distance), 0.002)
    circuit.append("M", range(distance))
    for i in range(distance - 1):
        circuit.append(
            "DETECTOR", [stim.target_rec(i - distance), stim.target_rec(i + 1 - distance)]
        )
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-distance)], 0)
    return circuit


@pytest.mark.parametrize("distance", [1, 2, 3, 4, 5])
def test_exact_bounds_and_verified_witnesses(tmp_path: Path, distance: int) -> None:
    circuit = repetition(distance)
    cat = catalogue(circuit)
    result = audit.search(tmp_path, circuit, cat, 30)
    assert result["complete"]
    assert result["excluded_through_weight"] == min(distance - 1, 4)
    if distance <= 4:
        assert result["upper_bound"] == distance
        witness = json.loads((tmp_path / "witness.json").read_text())
        assert witness["combined"]["detectors"] == []
        assert witness["combined"]["observable_flip"] == 1
        a, b = witness["ambiguous_histories"]
        assert a["detectors"] == b["detectors"]
        assert a["observable_flip"] != b["observable_flip"]
        if distance == 4:
            assert witness["two_history_decoder_independent_lower_bound"] == pytest.approx(
                0.002**2 * 0.998**2
            )
    else:
        assert "upper_bound" not in result
        assert result["next_bucket"] == audit.BUCKETS
    first = (tmp_path / "progress.json").read_bytes()
    audit.search(tmp_path, circuit, cat, 30)
    assert (tmp_path / "progress.json").read_bytes() == first


def fake_catalogue(values: list[int], sites: list[int]) -> dict[str, Any]:
    return {
        "signatures": [
            {
                "detectors": [d for d in range(v.bit_length()) if (v >> (d + 1)) & 1],
                "observables": [0] if v & 1 else [],
            }
            for v in values
        ],
        "faults": [
            {"signature": i, "instruction": site, "target_start": 0} for i, site in enumerate(sites)
        ],
    }


def brute(cat: dict[str, Any]) -> int | None:
    for weight in range(1, 5):
        for ids in itertools.combinations(range(len(cat["faults"])), weight):
            faults = [cat["faults"][i] for i in ids]
            if len({(f["instruction"], f["target_start"]) for f in faults}) != weight:
                continue
            value = 0
            for f in faults:
                value ^= audit.signature_value(cat["signatures"][f["signature"]])
            if value == 1:
                return weight
    return None


@pytest.mark.parametrize("colliding_hashes", [False, True])
def test_against_exhaustive_physical_search(colliding_hashes: bool) -> None:
    rng = np.random.default_rng(817)
    for _ in range(80):
        values = rng.integers(0, 64, size=9).tolist()
        sites = rng.integers(0, 7, size=9).tolist()
        cat = fake_catalogue(values, sites)
        p = audit.Patterns(cat)
        if colliding_hashes:
            p.keys = np.asarray([v & 1 for v in p.values], dtype=np.uint64)
            p.sorted_keys = np.unique(p.keys)
        pairs = audit.make_pairs(p, 0, max(0, len(p.values) - 1))
        found = audit.low_weights(p) or audit.triples(p, pairs) or audit.quadruples(p, pairs)
        assert (len(found) if found else None) == brute(cat)


def test_distinct_location_matching_retains_alternative_origins() -> None:
    p = audit.Patterns(fake_catalogue([2, 3, 2], [0, 0, 1]))
    found = audit.low_weights(p)
    assert found is not None
    assert {p.sites[i] for i in found} == {(0, 0), (1, 0)}
    p = audit.Patterns(fake_catalogue([2, 3], [0, 0]))
    assert audit.low_weights(p) is None


def test_multi_fault_injection_noise_and_measurement_same_instruction() -> None:
    circuit = stim.Circuit(
        "R 0 1 2 3\nDEPOLARIZE2(0.001) 0 1 2 3\nM(0.001) 0 1 2 3\n"
        "DETECTOR rec[-1]\nDETECTOR rec[-2]\nOBSERVABLE_INCLUDE(0) rec[-4]"
    )
    cat = catalogue(circuit)
    ids = [
        next(
            i
            for i, f in enumerate(cat["faults"])
            if f["gate"] == gate
            and f["target_start"] == target
            and (not paulis or f["paulis"] == paulis)
        )
        for gate, target, paulis in [
            ("DEPOLARIZE2", 0, ["X0"]),
            ("DEPOLARIZE2", 2, ["X2"]),
            ("M", 0, []),
            ("M", 2, []),
        ]
    ]
    result = audit.verify_history(circuit, cat, ids)
    assert result["detectors"] == []
    assert result["observable_flip"] == 0
    with pytest.raises(ValueError, match="Mutually exclusive"):
        audit.inject(circuit, [cat["faults"][ids[0]]] * 2)


def test_budget_resume_and_pair_checkpoint_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    circuit = repetition(70)
    cat = catalogue(circuit)
    with monkeypatch.context() as patch:
        clock = itertools.count()
        patch.setattr(audit.time, "perf_counter", lambda: float(next(clock)))
        result = audit.search(tmp_path, circuit, cat, 3)
    assert not result["complete"]
    assert result["excluded_through_weight"] == 2
    assert result["next_row"] == 32
    chunk = tmp_path / result["chunks"][0]["name"]
    saved = chunk.read_bytes()
    chunk.write_bytes(saved[:-1])
    with pytest.raises(ValueError, match="checkpoint changed"):
        audit.search(tmp_path, circuit, cat, 30)
    chunk.write_bytes(saved)
    complete = audit.search(tmp_path, circuit, cat, 30)
    assert complete["complete"]
    assert complete["excluded_through_weight"] == 4


def save_source(path: Path) -> None:
    path.mkdir()
    circuit = repetition(4)
    (path / "circuit.stim").write_text(str(circuit))
    (path / "model.dem").write_text(str(circuit.detector_error_model()))
    with gzip.open(path / "catalogue.json.gz", "wt") as stream:
        json.dump(catalogue(circuit), stream)
    audit.atomic_json(
        path / "catalogue_hash.json",
        {"sha256": audit.digest((path / "catalogue.json.gz").read_bytes())},
    )
    audit.atomic_json(
        path / "identity.json",
        {
            "circuit_sha256": audit.digest(str(circuit).encode()),
            "dem_sha256": audit.digest(str(circuit.detector_error_model()).encode()),
            "distance_label": 4,
            "rounds": 4,
            "basis": "X",
        },
    )


def test_frozen_source_and_readonly_resume(tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    save_source(source)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    result = audit.condition(str(source), str(output), 30)
    assert result["exact_distance"] == 4
    audit.condition(str(source), str(output), 30)
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    (source / "circuit.stim").write_text("R 0")
    with pytest.raises(ValueError, match="circuit changed"):
        audit.condition(str(source), str(output), 30)


def test_single_fault_decoder_error_is_not_structural_distance(tmp_path: Path) -> None:
    # A deliberately bad decoder could flip the answer on every single error.
    # This independent audit must still return the repetition circuit's distance.
    circuit = repetition(4)
    assert audit.search(tmp_path, circuit, catalogue(circuit), 30)["upper_bound"] == 4
