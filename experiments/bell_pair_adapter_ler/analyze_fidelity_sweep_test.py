from __future__ import annotations

import math

import pytest

from .analyze_fidelity_sweep import (
    build_requirement_rows,
    build_response_rows,
    weighted_isotonic,
)


def _summary_row(
    *,
    infidelity: float,
    ler: float,
    low: float,
    high: float,
    shots: int = 10_000,
) -> dict[str, str]:
    return {
        "name": f"example_bell_{infidelity}",
        "logical_basis": "X",
        "preparation_rounds": "3",
        "surgery_rounds": "4",
        "physical_error_rate": "0.001",
        "bell_pair_infidelity": str(infidelity),
        "bell_pair_fidelity": str(1 - infidelity),
        "num_bell_pair_checks_per_round": "5",
        "shots": str(shots),
        "errors": str(round(shots * ler)),
        "logical_error_rate": str(ler),
        "ler_95ci_low": str(low),
        "ler_95ci_high": str(high),
    }


def test_build_response_rows_compares_with_ideal_bell_reference() -> None:
    rows = build_response_rows(
        [
            _summary_row(infidelity=0.0, ler=0.01, low=0.008, high=0.012),
            _summary_row(infidelity=0.001, ler=0.015, low=0.012, high=0.018),
        ]
    )

    assert rows[0]["excess_logical_error_rate"] == 0
    assert rows[1]["bell_pairs_consumed"] == 20
    assert rows[1]["probability_any_bell_pair_error"] == pytest.approx(1 - 0.999**20)
    assert rows[1]["excess_logical_error_rate"] == pytest.approx(0.005)
    assert rows[1]["excess_ler_95ci_low"] == pytest.approx(0)
    assert rows[1]["excess_ler_95ci_high"] == pytest.approx(0.01)
    assert rows[1]["relative_ler_overhead"] == pytest.approx(0.5)


def test_build_response_rows_requires_ideal_reference() -> None:
    with pytest.raises(ValueError, match="no ideal-Bell reference"):
        build_response_rows([_summary_row(infidelity=0.001, ler=0.015, low=0.012, high=0.018)])


def test_response_rejects_mixing_hardware_and_full_surgery_noise() -> None:
    full = _summary_row(infidelity=0, ler=0.01, low=0.008, high=0.012)
    hardware = dict(full, adapter_noise_targets="adapter_hardware")
    with pytest.raises(ValueError, match="each noise policy"):
        build_response_rows([full, hardware])


def test_weighted_isotonic_pools_decreasing_neighbors() -> None:
    assert weighted_isotonic([0.02, 0.01, 0.03], [1, 1, 1]) == pytest.approx([0.015, 0.015, 0.03])


def test_requirement_rows_bracket_point_estimate_and_report_ci_limit() -> None:
    response = build_response_rows(
        [
            _summary_row(infidelity=0.0, ler=0.01, low=0.008, high=0.012),
            _summary_row(infidelity=0.001, ler=0.009, low=0.007, high=0.011),
            _summary_row(infidelity=0.01, ler=0.02, low=0.016, high=0.025),
        ]
    )

    requirement = build_requirement_rows(response, [0.25])[0]

    assert requirement["isotonic_ideal_bell_logical_error_rate"] == pytest.approx(0.0095)
    assert requirement["max_tested_infidelity_meeting_point_estimate"] == 0.001
    assert requirement["min_tested_fidelity_meeting_point_estimate"] == 0.999
    assert requirement["next_tested_infidelity"] == 0.01
    assert requirement["point_estimate_status"] == "bracketed_by_tested_points"
    assert requirement["max_tested_infidelity_meeting_conservative_95ci"] == 0
    assert math.isclose(requirement["min_tested_fidelity_meeting_conservative_95ci"], 1)
