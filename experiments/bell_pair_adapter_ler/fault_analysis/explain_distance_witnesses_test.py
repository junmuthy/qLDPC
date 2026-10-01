from __future__ import annotations

import pytest

from .effective_distance_audit_test import repetition
from .explain_distance_witnesses import realization_mass
from .single_fault_audit import catalogue


def test_probability_mass_is_exact_sector_not_marginal_probability() -> None:
    cat = catalogue(repetition(4))
    signatures = [f["signature"] for f in cat["faults"][:2]]
    result = realization_mass(cat, signatures, 100)
    assert result["probability_mass_no_other_faults"] == pytest.approx(0.002**2 * 0.998**2)
    assert result["complete_within_this_signature_family"]
    assert result["valid_distinct_location_realizations"] == 1


def test_duplicate_signatures_and_truncation_are_not_miscounted() -> None:
    cat = catalogue(repetition(4))
    sid = cat["faults"][0]["signature"]
    with pytest.raises(ValueError, match="distinct signatures"):
        realization_mass(cat, [sid, sid], 100)
    result = realization_mass(cat, [sid], 0)
    assert result["probability_mass_no_other_faults"] == 0
    assert not result["complete_within_this_signature_family"]


def test_multiple_origins_cannot_double_count_one_location() -> None:
    cat = catalogue(repetition(4))
    a, b = cat["faults"][:2]
    # Give signature b a second origin at a's site, which must be excluded.
    duplicate = dict(b, instruction=a["instruction"], target_start=a["target_start"])
    cat["faults"].append(duplicate)
    result = realization_mass(cat, [a["signature"], b["signature"]], 100)
    assert result["candidate_realizations"] == 2
    assert result["valid_distinct_location_realizations"] == 1
    assert result["probability_mass_no_other_faults"] == pytest.approx(0.002**2 * 0.998**2)
