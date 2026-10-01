from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import stim

from qldpc.decoders import SinterDecoder

from . import paired_decoder_check as paired
from .effective_distance_audit import atomic_json, digest
from .single_fault_audit import catalogue, runtime_identity


def source_fixture(path: Path) -> None:
    path.mkdir()
    circuit = stim.Circuit("R 0 1 2 3 4 5 6\nX_ERROR(0.03) 0 1 2 3 4 5 6\nM 0 1 2 3 4 5 6")
    for bit in range(3):
        circuit.append(
            "DETECTOR", [stim.target_rec(i - 7) for i in range(7) if (i + 1) & (1 << bit)]
        )
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-7)], 0)
    dem = circuit.detector_error_model()
    (path / "circuit.stim").write_text(str(circuit))
    (path / "model.dem").write_text(str(dem))
    with gzip.open(path / "catalogue.json.gz", "wt") as stream:
        json.dump(catalogue(circuit), stream)
    atomic_json(
        path / "catalogue_hash.json", {"sha256": digest((path / "catalogue.json.gz").read_bytes())}
    )
    atomic_json(
        path / "identity.json",
        {
            "circuit_sha256": digest(str(circuit).encode()),
            "dem_sha256": digest(str(dem).encode()),
            "runtime_identity": runtime_identity(),
        },
    )
    decoder = SinterDecoder().compile_decoder_for_dem(dem)
    atomic_json(
        path / "decoder_settings.json",
        {
            k: paired.settings(decoder.decoder)[k]
            for k in ("max_iter", "bp_method", "osd_method", "osd_order")
        },
    )


def test_immutable_banks_and_prefix_extension(tmp_path: Path) -> None:
    src, dst = tmp_path / "source", tmp_path / "samples"
    source_fixture(src)
    old = {p: p.read_bytes() for p in src.iterdir()}
    paired.prepare(src, dst, 100, 97)
    first = {p: p.read_bytes() for p in dst.iterdir()}
    d1, o1 = paired.bank(dst, 100)
    paired.prepare(src, dst, 200, 97)
    assert first == {p: p.read_bytes() for p in first}
    d2, o2 = paired.bank(dst, 200)
    assert np.array_equal(d1, d2[:100]) and np.array_equal(o1, o2[:100])
    assert old == {p: p.read_bytes() for p in old}
    with pytest.raises(ValueError, match="identity changed"):
        paired.prepare(src, dst, 200, 98)
    target = dst / "samples_000000.npz"
    target.write_bytes(target.read_bytes()[:-1])
    with pytest.raises(ValueError, match="checksum"):
        paired.bank(dst, 100)


def test_decoder_resume_and_completed_pairing(tmp_path: Path) -> None:
    src, dst = tmp_path / "source", tmp_path / "samples"
    source_fixture(src)
    paired.prepare(src, dst, 8, 891)
    for variant in paired.VARIANTS:
        result = paired.decode(str(dst), variant, 8, 60)
        assert result["complete"]
        before = {p: p.read_bytes() for p in (dst / variant).glob("shot_*.json")}
        paired.decode(str(dst), variant, 8, 60)
        assert before == {p: p.read_bytes() for p in before}
    summary = paired.summarize(dst, 8)
    assert summary["paired"]["shots"] == 8
    assert all(v["complete"] for v in summary["decoders"].values())


def test_budget_never_silently_discards_unfinished_shots(tmp_path: Path) -> None:
    src, dst = tmp_path / "source", tmp_path / "samples"
    source_fixture(src)
    paired.prepare(src, dst, 8, 312)
    result = paired.decode(str(dst), "production", 8, 0)
    assert not result["complete"]
    summary = paired.summarize(dst, 8)
    assert summary["paired"] is None
    assert summary["decoders"]["production"]["ler"] is None


def test_rescued_and_introduced_failures_are_both_counted() -> None:
    truth = np.zeros(100, dtype=np.uint8)
    a, b = truth.copy(), truth.copy()
    a[:4] = 1
    b[3:5] = 1
    row = paired.paired_metrics(truth, a, b)
    assert row["rescued"] == 3
    assert row["introduced"] == 1
    assert row["both_fail"] == 1
    assert row["delta_alternative_minus_production"] == -0.02


def test_zero_observed_failures_does_not_prove_equal_performance() -> None:
    zeros = np.zeros(100, dtype=np.uint8)
    row = paired.paired_metrics(zeros, zeros, zeros)
    lower, upper = row["delta_95ci_conservative_paired"]
    assert lower < 0 < upper
    assert paired.clopper_pearson(0, 100)[1] > 0.03
