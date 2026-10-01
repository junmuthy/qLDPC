from __future__ import annotations

import gzip
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from . import effective_distance_audit as base
from . import five_fault_search as search
from .effective_distance_audit_test import fake_catalogue, repetition
from .single_fault_audit import catalogue


def find_five(cat: dict[str, Any], collide: bool = False) -> list[int] | None:
    p = base.Patterns(cat)
    if collide:
        p.keys = np.asarray([v & 1 for v in p.values], dtype=np.uint64)
    pairs = base.make_pairs(p, 0, max(0, len(p.values) - 1))
    indexes = []
    for bucket in range(base.BUCKETS):
        rows = pairs[((pairs["key"] >> 1) & (base.BUCKETS - 1)) == bucket].copy()
        rows.sort(order="key")
        indexes.append(rows)
    for pivot, ns in enumerate(search.neighbors(p)):
        result, _, _ = search.pivot_search(p, pivot, ns, indexes)
        if result is not None:
            return result
    return None


@pytest.mark.parametrize("distance", [5, 6, 8])
@pytest.mark.parametrize("collide", [False, True])
def test_known_distance_and_hash_collisions(distance: int, collide: bool) -> None:
    circuit = repetition(distance)
    cat = catalogue(circuit)
    found = find_five(cat, collide)
    if distance == 5:
        assert found is not None
        assert base.verify_witness(circuit, cat, found)["weight"] == 5
    else:
        assert found is None


def test_connected_search_matches_exhaustive_five_faults() -> None:
    rng = np.random.default_rng(715)
    checked = 0
    for _ in range(200):
        values = rng.integers(2, 1024, size=10).tolist()
        sites = rng.integers(0, 10, size=10).tolist()
        cat = fake_catalogue(values, sites)
        minimum = None
        for w in range(1, 6):
            for ids in itertools.combinations(range(10), w):
                if len({sites[i] for i in ids}) != w:
                    continue
                v = 0
                for i in ids:
                    v ^= values[i]
                if v == 1:
                    minimum = w
                    break
            if minimum is not None:
                break
        if minimum is not None and minimum < 5:
            continue
        checked += 1
        found = find_five(cat)
        assert (len(found) if found is not None else None) == minimum
    assert checked > 100


def test_sorted_index_roundtrip(tmp_path: Path) -> None:
    p = base.Patterns(catalogue(repetition(5)))
    pairs = base.make_pairs(p, 0, len(p.values) - 1)
    path = tmp_path / "index.npy"
    digest = search.write_index(path, pairs)
    assert digest == base.digest(path.read_bytes())
    saved = np.load(path)
    assert np.all(saved["key"][1:] >= saved["key"][:-1])


def test_budget_then_resume_finds_verified_minimum(tmp_path: Path) -> None:
    source, previous, output = tmp_path / "source", tmp_path / "four", tmp_path / "five"
    source.mkdir()
    circuit = repetition(5)
    dem = circuit.detector_error_model()
    (source / "circuit.stim").write_text(str(circuit))
    (source / "model.dem").write_text(str(dem))
    with gzip.open(source / "catalogue.json.gz", "wt") as stream:
        json.dump(catalogue(circuit), stream)
    base.atomic_json(
        source / "catalogue_hash.json",
        {"sha256": base.digest((source / "catalogue.json.gz").read_bytes())},
    )
    base.atomic_json(
        source / "identity.json",
        {
            "circuit_sha256": base.digest(str(circuit).encode()),
            "dem_sha256": base.digest(str(dem).encode()),
            "distance_label": 5,
            "rounds": 8,
            "basis": "X",
        },
    )
    base.condition(str(source), str(previous), 30)
    before = {p: p.read_bytes() for p in previous.iterdir()}
    partial = search.condition(str(previous), str(output), 0)
    assert not partial["complete"]
    assert partial["fault_distance_lower_bound"] == 5
    complete = search.condition(str(previous), str(output), 30)
    assert complete["exact_distance"] == 5
    assert complete["complete"]
    witness = json.loads((output / "witness.json").read_text())
    assert len(witness["ambiguous_histories"][0]["physical_faults"]) == 2
    assert len(witness["ambiguous_histories"][1]["physical_faults"]) == 3
    saved = (output / "progress.json").read_bytes()
    search.condition(str(previous), str(output), 30)
    assert saved == (output / "progress.json").read_bytes()
    assert before == {p: p.read_bytes() for p in before}
