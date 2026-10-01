from __future__ import annotations

import itertools

import numpy as np
import pytest

from .adapter_structure_audit import connectivity


def test_boundary_expansion_keeps_parallel_edges_and_hyperedges() -> None:
    rng = np.random.default_rng(182)
    for n in range(2, 9):
        f = rng.integers(0, 2, (11, n), dtype=np.uint8)
        f[1] = f[0]
        expected = min(
            np.count_nonzero(f[:, subset].sum(axis=1) % 2) / len(subset)
            for size in range(1, n // 2 + 1)
            for subset in itertools.combinations(range(n), size)
        )
        result = connectivity(f)
        assert result["boundary_cheeger_exact"] == pytest.approx(expected)
        assert result["boundary_size"] == len(result["boundary_rows"])


def test_path_and_disconnected_graphs() -> None:
    f = np.zeros((7, 8), dtype=np.uint8)
    for i in range(7):
        f[i, i : i + 2] = 1
    assert connectivity(f)["boundary_cheeger_exact"] == 0.25
    assert connectivity(f)["expansion_condition_met"] is False
    f[3] = 0
    assert connectivity(f)["boundary_cheeger_exact"] == 0
    assert len(connectivity(f)["connected_components"]) == 2


def test_large_graph_is_not_silently_certified() -> None:
    result = connectivity(np.zeros((2, 21), dtype=np.uint8))
    assert result["boundary_cheeger_exact"] is None
    assert result["expansion_condition_met"] is None
