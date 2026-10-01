from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from . import plot_d8_low_p_mean_history as plot


def write_checkpoint(
    directory: Path,
    counts: list[tuple[int, int]],
    seed: tuple[int, int] | None = None,
    partial: bytes = b"",
) -> None:
    directory.mkdir()
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["shots", "errors", "discards", "strong_id", "decoder", "json_metadata"])
    metadata = json.dumps(
        dict(
            left_code="hgp_simplex_m4",
            right_code="surface_unrotated_d8",
            physical_error_rate=1e-4,
            surgery_rounds=8,
            logical_basis="X",
        )
    )
    if seed is not None:
        writer.writerow([*seed, 0, plot.STRONG_ID, "qldpc_bp_osd", metadata])
        receipt = dict(
            seed_counts={plot.STRONG_ID: dict(shots=seed[0], errors=seed[1])},
            seed_csv_sha256=hashlib.sha256(stream.getvalue().strip().encode()).hexdigest(),
        )
        (directory / "seed_receipt.json").write_text(json.dumps(receipt))
    writer.writerow([1, 1, 0, "unrelated", "decoder", "{}"])
    for n, k in counts:
        writer.writerow([n, k, 0, plot.STRONG_ID, "qldpc_bp_osd", metadata])
    (directory / "sinter_progress.csv").write_bytes(stream.getvalue().encode() + partial)


def test_seed_seams_no_double_count_and_partial_tail(tmp_path: Path) -> None:
    a, b, c = [tmp_path / name for name in ("a", "b", "c")]
    write_checkpoint(a, [(10, 0), (10, 1)])
    write_checkpoint(b, [(30, 1)], seed=(20, 1))
    write_checkpoint(c, [(50, 1)], seed=(50, 2), partial=b"12,0,0,")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    rows, provenance = plot.stitch_history([a, b, c])
    assert [r["cumulative_shots"] for r in rows] == [10, 20, 50, 100]
    assert [r["cumulative_failures"] for r in rows] == [0, 1, 2, 3]
    assert sum(r["batch_shots"] for r in rows) == 100
    assert rows[-1]["cumulative_ler"] == 0.03
    assert rows[-1]["jackknife_1sigma"] == pytest.approx((0.03 * 0.97 / 99) ** 0.5)
    assert provenance[-1]["ignored_partial_bytes"] == 7
    assert [p["added_shots"] for p in provenance] == [20, 30, 50]
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_bad_seeds_and_batches_refused(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    write_checkpoint(a, [(10, 1)])
    write_checkpoint(b, [(10, 0)], seed=(10, 2))
    with pytest.raises(ValueError, match="match accumulated"):
        plot.stitch_history([a, b])
    receipt = json.loads((b / "seed_receipt.json").read_text())
    receipt["seed_csv_sha256"] = "bad"
    (b / "seed_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="prefix hash"):
        plot.stitch_history([a, b])
    (b / "seed_receipt.json").unlink()
    with pytest.raises(ValueError, match="receipt"):
        plot.stitch_history([a, b])
    c = tmp_path / "c"
    write_checkpoint(c, [(10, 11)])
    with pytest.raises(ValueError, match="valid failures"):
        plot.stitch_history([c])


def test_plot_outputs_no_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_checkpoint(source, [(100, 0), (100, 1), (100, 0)])
    rows, provenance = plot.stitch_history([source])
    output = tmp_path / "report"
    plot.write_plot(rows, provenance, output, "snapshot time")
    assert (output / "cumulative_adapter_ler.pdf").stat().st_size > 1000
    assert (output / "cumulative_adapter_ler.png").stat().st_size > 1000
    snapshot = json.loads((output / "snapshot.json").read_text())
    assert snapshot["shots"] == 300 and snapshot["failures"] == 1
    with pytest.raises(FileExistsError):
        plot.write_plot(rows, provenance, output, "later")
