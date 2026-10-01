from pathlib import Path

import pytest

from . import plot_surface_all_datasets_bell1 as plotter


@pytest.mark.adapter_archive
def test_seven_distinct_series_at_fidelity_one() -> None:
    hashes = plotter.source_hashes()
    rows = plotter.load_comparison()
    expected = {
        (basis, policy, rounds)
        for policy, _, _, _ in plotter.DATASETS
        for basis in (("Z",) if policy == "local_order_hardware" else ("X", "Z"))
        for rounds in range(1, 6)
    }
    assert len(rows) == len(expected) == 35
    assert {(r["logical_basis"], r["policy"], r["surgery_rounds"]) for r in rows} == expected
    assert len({(r["source_summary"], r["source_name"]) for r in rows}) == 35
    assert all(r["bell_pair_fidelity"] == 1 for r in rows)
    assert len(hashes) == 8
    assert plotter.source_hashes() == hashes


@pytest.mark.adapter_archive
def test_original_and_matched_full_surgery_are_not_conflated() -> None:
    rows = plotter.load_comparison()
    x3 = {r["policy"]: r for r in rows if r["logical_basis"] == "X" and r["surgery_rounds"] == 3}
    assert x3["original_full_surgery"]["errors"] == 42
    assert x3["matched_full_surgery"]["errors"] == 132
    assert x3["original_hardware"]["errors"] == 79
    assert x3["original_full_surgery"]["dem_analysis"] == "sinter_auto"
    assert x3["matched_full_surgery"]["dem_analysis"] == "undecomposed_exact"
    new = [r for r in rows if r["policy"] == "local_order_hardware"]
    assert all(r["logical_basis"] == "Z" and r["shots"] == 20000 for r in new)
    assert all(r["local_order"] == "Z,X" and r["bell_order"] == "X,Z" for r in new)


def test_plot_including_zero_failure_limit(tmp_path: Path) -> None:
    rows = [
        dict(
            logical_basis=basis,
            policy=policy,
            series=label,
            surgery_rounds=rounds,
            logical_error_rate=0.01,
            ler_95ci_low=0.005,
            ler_95ci_high=0.02,
        )
        for policy, _, label, _ in plotter.DATASETS
        for basis in (("Z",) if policy == "local_order_hardware" else ("X", "Z"))
        for rounds in range(1, 6)
    ]
    rows[-1].update(logical_error_rate=0, ler_95ci_low=0)
    plotter.plot_comparison(rows, tmp_path)
    for suffix in ("pdf", "png"):
        assert (tmp_path / f"adapter_ler_all_datasets_bell1.{suffix}").stat().st_size > 1000
