"""Tests for the affine logical-frame solver and its independent validator.

Copyright 2026 The qLDPC Authors and Infleqtion Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import itertools

import galois
import numpy as np
import pytest

from qldpc import abstract

from . import _cyclic


@pytest.mark.parametrize(
    "matrix,rhs",
    [
        ([[1, 1, 0], [0, 1, 1]], [1, 0]),
        ([[1, 0], [0, 1]], [1, 1]),
        ([[0, 0]], [0]),
        ([[1, 1], [1, 1]], [0, 1]),
    ],
)
def test_affine_solutions(matrix: list[list[int]], rhs: list[int]) -> None:
    """The returned affine space equals the full brute-force solution set."""
    aa, bb = galois.GF2(matrix), galois.GF2(rhs)
    expected = {
        vv
        for vv in itertools.product((0, 1), repeat=aa.shape[1])
        if np.array_equal(aa @ galois.GF2(vv), bb)
    }
    solution = _cyclic.affine_solutions(aa, bb)
    if solution is None:
        assert not expected
    else:
        particular, kernel = solution
        actual = {
            tuple(map(int, particular + galois.GF2(bits) @ kernel))
            for bits in itertools.product((0, 1), repeat=len(kernel))
        }
        assert actual == expected


def test_copy_and_validate() -> None:
    """Copied frames keep their ring and must satisfy every independent logical condition."""
    ring = abstract.GroupRing(abstract.TrivialGroup())
    hh = abstract.RingArray.build([[1, 1, 0]], ring)
    jj = abstract.RingArray.build([[1, 1, 1]], ring)
    copied = _cyclic.copy_matrix(jj, ring, "frame")
    assert np.array_equal(copied, jj)
    assert copied[0, 0] is not jj[0, 0]
    _cyclic.validate_logicals(hh, copied)
    with pytest.raises(TypeError, match="RingArray"):
        _cyclic.copy_matrix([[1]], ring, "frame")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="two-dimensional"):
        _cyclic.copy_matrix(jj[0], ring, "frame")
    with pytest.raises(ValueError, match="same group ring"):
        _cyclic.copy_matrix(jj, abstract.GroupRing(abstract.CyclicGroup(2)), "frame")
    for vals, message in (
        ([[1]], "shape"),
        ([[1, 0, 0]], "commutation"),
        ([[0, 0, 0]], "orthonormal"),
        ([[0, 0, 1]], "augmentation"),
    ):
        with pytest.raises(ValueError, match=message):
            _cyclic.validate_logicals(hh, abstract.RingArray.build(vals, ring))


def test_find_logicals() -> None:
    """Accept full frames and distinguish linear, nonlinear, and budget failures."""
    ring = abstract.GroupRing(abstract.TrivialGroup())
    hh = abstract.RingArray.build([[1, 1, 0, 0, 0]], ring)
    coefficients = abstract.RingArray(hh.reshape(1, 5, 1))
    found = _cyclic.find_logicals(hh, coefficients, None)
    assert found.status == "found"
    assert found.logical_ops is not None
    assert found.num_candidates > 1
    _cyclic.validate_logicals(hh, found.logical_ops)
    limited = _cyclic.find_logicals(hh, coefficients, found.num_candidates - 1)
    assert limited.status == "inconclusive"
    error = _cyclic.LogicalFrameError(limited)
    assert error.result is limited
    assert str(error) == limited.message

    hh = abstract.RingArray.build([[1, 1, 1, 1]], ring)
    impossible = _cyclic.find_logicals(hh, abstract.RingArray(hh.reshape(1, 4, 1)), None)
    assert impossible.status == "infeasible"
    assert impossible.num_candidates == 8
    # The affine helper also handles inconsistent systems independently of check validation.
    hh = abstract.RingArray.build([[1, 0, 0]], ring)
    inconsistent = _cyclic.find_logicals(hh, abstract.RingArray(hh.reshape(1, 3, 1)), 0)
    assert inconsistent.status == "infeasible"
    assert inconsistent.num_candidates == 0
    assert "inconsistent" in inconsistent.message

    hh = abstract.RingArray.build(
        [[1, 1, 1, 0, 1, 0, 0], [0, 1, 1, 1, 0, 1, 0], [0, 0, 1, 1, 1, 0, 1]], ring
    )
    fiber = _cyclic.find_logicals(hh, abstract.RingArray(hh[:1].reshape(1, 1, 7)), 0)
    assert fiber.status == "found"
    assert fiber.method == "fiber"
