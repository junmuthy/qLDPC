"""Tests for independent check-orbit bases and compatible logical frames.

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

from . import _symcity


def resolve(parent: abstract.RingArray, limit: int | None = None) -> _symcity.CheckBasisResult:
    """Resolve a basis and independently check every successful result."""
    binary = parent.regular_lift()
    rank = int(np.linalg.matrix_rank(binary))
    result = _symcity.find_check_basis(parent, binary, rank, limit)
    if result.status == "found":
        assert result.check_basis is not None
        actual = _symcity.validate_check_basis(result.check_basis, binary, rank)
        assert np.array_equal(actual.row_space(), binary.row_space())
    return result


def test_search_limits() -> None:
    """Normalize integer budgets and reject ambiguous or negative limits."""
    assert _symcity.validate_search_limit(None, "limit") is None
    assert _symcity.validate_search_limit(0, "limit") == 0
    assert _symcity.validate_search_limit(np.int64(4), "limit") == 4  # type: ignore[arg-type]
    for value in (True, 1.5):
        with pytest.raises(TypeError, match="limit must be a nonnegative integer"):
            _symcity.validate_search_limit(value, "limit")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="limit must be nonnegative"):
        _symcity.validate_search_limit(-1, "limit")


@pytest.mark.parametrize(
    "group",
    [
        abstract.TrivialGroup(),
        abstract.CyclicGroup(3),
        abstract.AbelianGroup(2, 2, direct_sum=True),
    ],
)
def test_check_fast_paths(group: abstract.Group) -> None:
    """Rank selection handles zero/duplicate checks, different groups, and equivalent generators."""
    ring = abstract.GroupRing(group)
    basis = abstract.RingArray.build([[1, 1, 0, 0, 0], [0, 0, 1, 1, 0]], ring)
    assert resolve(basis, 0).method == "parent"
    zero = abstract.RingArray.build(np.zeros((3, 5), dtype=int), ring)
    result = resolve(zero, 0)
    assert result.selected_rows == ()
    assert result.parent_rank == 0
    assert result.check_basis is not None and result.check_basis.shape == (0, 5)
    redundant = abstract.RingArray.build(np.vstack([zero[:1], basis[:1], basis]), ring)
    result = resolve(redundant, 0)
    assert result.method == "greedy"
    assert result.selected_rows == (1, 3)
    # An invertible elementary polynomial row operation and row reordering preserve the space.
    transformed = abstract.RingArray.build(basis, ring)
    transformed[0] = transformed[0] + (ring.one + ring.generators[0]) * transformed[1]
    transformed = abstract.RingArray.build(np.vstack([transformed[1:], transformed]), ring)
    actual = resolve(transformed, 0).check_basis
    assert actual is not None
    assert np.array_equal(actual.regular_lift().row_space(), basis.regular_lift().row_space())


def test_check_subsets() -> None:
    """An independently chosen first orbit can prevent greedy completion from parent rows."""
    ring = abstract.GroupRing(abstract.CyclicGroup(3))
    xx = ring.generators[0]
    e0, e1 = ring.one + xx + xx**2, xx + xx**2
    v1 = abstract.RingArray.build([[1, 1, 0, 0, 0]], ring)
    v2 = abstract.RingArray.build([[0, 0, 1, 1, 0]], ring)
    parent = abstract.RingArray.build(np.vstack([e0 * v1 + e1 * v2, v1, v2]), ring)
    assert resolve(parent, 0).status == "inconclusive"
    result = resolve(parent)
    assert (result.method, result.selected_rows, result.num_candidates) == ("subsets", (1, 2), 3)
    assert resolve(parent, 3).status == "found"
    limited = resolve(parent, 2)
    assert (limited.status, limited.num_candidates) == ("inconclusive", 2)
    error = _symcity.CheckBasisError(limited)
    assert error.result is limited and str(error) == limited.message


@pytest.mark.parametrize("num_orbits", [1, 2])
def test_check_combinations(num_orbits: int) -> None:
    """A free module can be presented by prototypes none of which has a full-rank orbit."""
    ring = abstract.GroupRing(abstract.CyclicGroup(3))
    xx = ring.generators[0]
    e0, e1 = ring.one + xx + xx**2, xx + xx**2
    basis = abstract.RingArray.build(
        np.hstack(
            [np.kron(np.eye(num_orbits, dtype=int), [[1, 1]]), np.zeros((num_orbits, 1), dtype=int)]
        ),
        ring,
    )
    parent = abstract.RingArray.build(np.vstack([e0 * basis, e1 * basis]), ring)
    result = resolve(parent)
    assert result.method == "combinations"
    assert result.selected_rows is None
    assert result.check_basis is not None
    assert np.array_equal(
        result.check_basis.regular_lift().row_space(), basis.regular_lift().row_space()
    )
    assert resolve(parent, result.num_candidates).status == "found"
    assert resolve(parent, result.num_candidates - 1).status == "inconclusive"


@pytest.mark.parametrize("free_summand", [False, True])
def test_check_nonfree(free_summand: bool) -> None:
    """Exhaustive module failure differs from subset failure or an interrupted traversal."""
    ring = abstract.GroupRing(abstract.CyclicGroup(2))
    uu = ring.one + ring.generators[0]
    parent = abstract.RingArray.build([[0, 0, uu, 0, 0], [0, 0, 0, uu, 0]], ring)
    if free_summand:
        parent = abstract.RingArray.build(np.vstack([[[1, 1, 0, 0, 0]], parent]), ring)
    result = resolve(parent)
    assert (result.status, result.method) == ("infeasible", "combinations")
    assert result.parent_rank == (4 if free_summand else 2)
    assert resolve(parent, result.num_candidates).status == "infeasible"
    limited = resolve(parent, result.num_candidates - 1)
    assert (limited.status, limited.method) == ("inconclusive", "combinations")
    assert limited.num_candidates == result.num_candidates - 1
    # An odd binary rank cannot be a sum of complete independent C_2 orbits.
    result = resolve(abstract.RingArray(parent[-1:]), 0)
    assert (result.status, result.method, result.num_candidates) == ("infeasible", "rank", 0)


def test_prototype_from_vector() -> None:
    """Inversion and identity location respect the actual group enumeration."""
    for group in (abstract.CyclicGroup(3), abstract.AbelianGroup(2, 3, direct_sum=True)):
        members = list(group.generate())
        reordered = members[1:] + members[:1]
        reordered_group = abstract.Group(*group.generators, generate_func=lambda: iter(reordered))
        ring = abstract.GroupRing(reordered_group)
        vector = ring.field((np.arange(3 * reordered_group.order) % 3 == 0).astype(int))
        prototype = _symcity.prototype_from_vector(vector, ring)
        lifted = prototype.regular_lift()
        identity_index = reordered_group.index(reordered_group.identity)
        assert np.array_equal(lifted[identity_index], vector)
        assert np.array_equal(
            _symcity.prototype_from_vector(lifted[identity_index], ring), prototype
        )


def test_check_validator() -> None:
    """Replay validation independently checks shape, span, rank, commutation, and dimensions."""
    ring = abstract.GroupRing(abstract.TrivialGroup())
    good = abstract.RingArray.build([[1, 1, 0]], ring)
    binary = good.regular_lift()
    for bad, message in (
        ([[1, 1]], "tq columns"),
        ([[0, 0, 0]], "binary rank"),
        ([[1, 0, 1]], "span exactly"),
    ):
        with pytest.raises(ValueError, match=message):
            _symcity.validate_check_basis(abstract.RingArray.build(bad, ring), binary, 1)
    noncommuting = abstract.RingArray.build([[1, 0, 0]], ring)
    with pytest.raises(ValueError, match="does not commute"):
        _symcity.validate_check_basis(noncommuting, noncommuting.regular_lift(), 1)
    full = abstract.RingArray.build([[1, 1]], ring)
    with pytest.raises(ValueError, match="s = tq - 2r"):
        _symcity.validate_check_basis(full, full.regular_lift(), 1)


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
    solution = _symcity.affine_solutions(aa, bb)
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
    copied = _symcity.copy_matrix(jj, ring, "frame")
    assert np.array_equal(copied, jj)
    assert copied[0, 0] is not jj[0, 0]
    _symcity.validate_logicals(hh, copied)
    with pytest.raises(TypeError, match="RingArray"):
        _symcity.copy_matrix([[1]], ring, "frame")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="two-dimensional"):
        _symcity.copy_matrix(jj[0], ring, "frame")
    with pytest.raises(ValueError, match="same group ring"):
        _symcity.copy_matrix(jj, abstract.GroupRing(abstract.CyclicGroup(2)), "frame")
    for vals, message in (
        ([[1]], "shape"),
        ([[1, 0, 0]], "commutation"),
        ([[0, 0, 0]], "orthonormal"),
        ([[0, 0, 1]], "augmentation"),
    ):
        with pytest.raises(ValueError, match=message):
            _symcity.validate_logicals(hh, abstract.RingArray.build(vals, ring))


def test_find_logicals() -> None:
    """Accept full frames and distinguish linear, nonlinear, and budget failures."""
    ring = abstract.GroupRing(abstract.TrivialGroup())
    hh = abstract.RingArray.build([[1, 1, 0, 0, 0]], ring)
    coefficients = abstract.RingArray(hh.reshape(1, 5, 1))
    found = _symcity.find_logicals(hh, coefficients, None)
    assert found.status == "found"
    assert found.logical_ops is not None
    assert found.num_candidates > 1
    _symcity.validate_logicals(hh, found.logical_ops)
    limited = _symcity.find_logicals(hh, coefficients, found.num_candidates - 1)
    assert limited.status == "inconclusive"
    error = _symcity.LogicalFrameError(limited)
    assert error.result is limited
    assert str(error) == limited.message

    hh = abstract.RingArray.build([[1, 1, 1, 1]], ring)
    impossible = _symcity.find_logicals(hh, abstract.RingArray(hh.reshape(1, 4, 1)), None)
    assert impossible.status == "infeasible"
    assert impossible.num_candidates == 8
    # The affine helper also handles inconsistent systems independently of check validation.
    hh = abstract.RingArray.build([[1, 0, 0]], ring)
    inconsistent = _symcity.find_logicals(hh, abstract.RingArray(hh.reshape(1, 3, 1)), 0)
    assert inconsistent.status == "infeasible"
    assert inconsistent.num_candidates == 0
    assert "inconsistent" in inconsistent.message

    hh = abstract.RingArray.build(
        [[1, 1, 1, 0, 1, 0, 0], [0, 1, 1, 1, 0, 1, 0], [0, 0, 1, 1, 1, 0, 1]], ring
    )
    fiber = _symcity.find_logicals(hh, abstract.RingArray(hh[:1].reshape(1, 1, 7)), 0)
    assert fiber.status == "found"
    assert fiber.method == "fiber"
