"""Shared fixtures + helpers for the split surgery test files."""

from __future__ import annotations

import numpy as np

from ._test_webster_fixture import (
    load_webster_seed_set,
    build_generalised_bicycle_code,
)


def _webster_x_bar_operator(
    data: dict, name: str = "X_bar_1", pauli_type: str = "X",
) -> np.ndarray:
    """Extract the named logical operator from a Webster seed_set dict.

    L_support and R_support are sparse index lists (positions within each l-half
    that are set to 1). Returns a dense binary vector of length 2l.

    Args:
        data: Webster seed set dict (from load_webster_seed_set).
        name: Seed name, e.g. "X_bar_1", "Z_bar_1".
        pauli_type: "X" or "Z"; filters seeds by pauli_type field.
    """
    l = data["l"]
    for seed in data["seeds"]:
        if seed["name"] == name and seed["pauli_type"] == pauli_type:
            v_L = np.zeros(l, dtype=np.uint8)
            v_L[seed["L_support"]] = 1
            v_R = np.zeros(l, dtype=np.uint8)
            v_R[seed["R_support"]] = 1
            return np.concatenate([v_L, v_R])
    raise ValueError(f"{name!r} (pauli_type={pauli_type!r}) seed not found")


def _webster_z_bar_operator(data: dict, name: str = "Z_bar_1") -> np.ndarray:
    """Extract the named Z-type logical operator from a Webster seed_set dict.

    Convenience wrapper around _webster_x_bar_operator with pauli_type="Z".
    """
    return _webster_x_bar_operator(data, name, pauli_type="Z")


def _webster_x_bar_1_operator(data: dict) -> np.ndarray:
    """Back-compat: returns X_bar_1; prefer _webster_x_bar_operator."""
    return _webster_x_bar_operator(data, "X_bar_1")
