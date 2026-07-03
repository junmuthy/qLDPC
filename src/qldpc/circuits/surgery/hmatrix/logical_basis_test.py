"""Tests for canonical symplectic logical-basis construction (general CSS codes).

Small codes only — the gross [[72,12,6]] basis costs ~12 s per build and must not
run in pytest. [[8,2]] toric / [[7,1,3]] Steane keep every test well under a second.
"""
from __future__ import annotations

import numpy as np
import pytest

from qldpc import codes
from qldpc.objects import Pauli
from qldpc.circuits.surgery.hmatrix.logical_basis import symplectic_logical_basis


def _assert_canonical(code, Lx: np.ndarray, Lz: np.ndarray) -> None:
    """Lx, Lz form a valid canonical basis: X.Z^T = I, and each row is a logical."""
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    k, n = code.dimension, len(code)
    assert Lx.shape == (k, n)
    assert Lz.shape == (k, n)
    # canonical (symplectic) pairing X_i . Z_j = delta_ij
    assert np.array_equal(Lx @ Lz.T % 2, np.eye(k, dtype=np.uint8))
    # X-type operators commute with every Z-check; Z-type with every X-check
    assert not (HZ @ Lx.T % 2).any()
    assert not (HX @ Lz.T % 2).any()
    # every row is a *nontrivial* logical (nonzero); the pairing already forces independence
    assert (Lx.sum(1) > 0).all()
    assert (Lz.sum(1) > 0).all()


@pytest.mark.parametrize("optimize", [Pauli.X, Pauli.Z])
def test_completes_for_both_optimize(optimize):
    """A canonical symplectic basis is built and valid for both optimize types."""
    code = codes.ToricCode(2)                     # [[8, 2]]
    Lx, Lz = symplectic_logical_basis(code, rng=np.random.default_rng(2), optimize=optimize)
    _assert_canonical(code, Lx, Lz)


def test_optimize_bounds_weight_of_chosen_type():
    """optimize=Z weight-minimizes Z, so max Z weight <= max X weight."""
    code = codes.ToricCode(2)
    Lx_z, Lz_z = symplectic_logical_basis(code, rng=np.random.default_rng(5), optimize=Pauli.Z)
    assert max(int(r.sum()) for r in Lz_z) <= max(int(r.sum()) for r in Lx_z)


def test_generalizes_to_other_css_code():
    """A structurally different (non-BB) code also yields a canonical basis."""
    code = codes.SteaneCode()                     # [[7, 1, 3]]
    Lx, Lz = symplectic_logical_basis(code, rng=np.random.default_rng(0))
    _assert_canonical(code, Lx, Lz)
