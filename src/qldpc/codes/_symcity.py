"""Check-orbit and logical-frame validation and bounded solving for SymCity codes.

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
from dataclasses import dataclass
from typing import Literal

import galois
import numpy as np

from qldpc import abstract


@dataclass(frozen=True)
class CheckBasisResult:
    """Outcome of validating or seeking a complete independent check-orbit basis.

    Attributes:
        status: "found", "infeasible" (proved impossible), or "inconclusive" (budget exhausted).
        check_basis: A polynomial basis on success, otherwise None; replay validation before use.
        selected_rows: Parent-row indices for a selected basis, otherwise None.
        parent_rank: Binary rank of the full parent check matrix.
        method: "supplied", "parent", "greedy", "subsets", "combinations", or "rank".
        num_candidates: Fallback rank trials, excluding fast paths and final validation.
        message: Human-readable explanation of the outcome.
    """

    status: Literal["found", "infeasible", "inconclusive"]
    check_basis: abstract.RingArray | None
    selected_rows: tuple[int, ...] | None
    parent_rank: int
    method: Literal["supplied", "parent", "greedy", "subsets", "combinations", "rank"]
    num_candidates: int
    message: str


class CheckBasisError(ValueError):
    """Check-orbit basis resolution failed; inspect ``result`` for its scope and status."""

    result: CheckBasisResult

    def __init__(self, result: CheckBasisResult) -> None:
        self.result = result
        super().__init__(result.message)


def validate_search_limit(value: int | None, name: str) -> int | None:
    """Normalize a nonnegative candidate budget without accepting booleans."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a nonnegative integer or None")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return int(value)


def validate_check_basis(
    basis: abstract.RingArray, parent_binary: galois.FieldArray, parent_rank: int
) -> galois.FieldArray:
    """Validate independence, span, commutation, and logical count; return the binary basis."""
    num_checks, num_types = basis.shape
    if num_types * basis.group.order != parent_binary.shape[1]:
        raise ValueError("check_basis must have one column per physical type (tq columns)")
    matrix = basis.regular_lift()
    rank = np.linalg.matrix_rank(matrix)
    expected_rank = num_checks * basis.group.order
    if rank != expected_rank:
        raise ValueError(
            f"Check basis has binary rank {rank}, expected rN = {expected_rank}; "
            "provide independent complete G orbits using check_rows or check_basis"
        )
    if rank != parent_rank or np.linalg.matrix_rank(np.vstack((matrix, parent_binary))) != rank:
        raise ValueError("The check basis must span exactly the full parent check space")
    if np.any(basis @ basis.T):
        raise ValueError("The check basis does not commute")
    if num_types - 2 * num_checks < 1:
        raise ValueError("The check basis must leave s = tq - 2r >= 1 logical orbits")
    return matrix


def prototype_from_vector(
    vector: galois.FieldArray, ring: abstract.GroupRing
) -> abstract.RingArray:
    """Recover a polynomial prototype from the identity row of its left-regular lift."""
    inverses = [ring.group.index(~gg) for gg in ring.group.generate()]
    coefficients = vector.reshape(1, -1, ring.group.order)[:, :, inverses]
    return abstract.RingArray.from_field_array(coefficients, ring)


def find_check_basis(
    parent: abstract.RingArray,
    parent_binary: galois.FieldArray,
    parent_rank: int,
    max_candidates: int | None,
) -> CheckBasisResult:
    """Select independent G orbits, then try subsets and the full check-module domain.

    The caller supplies a commuting parent and its exact binary lift and rank. Fast paths
    consume no candidates. Each fallback subset or orbit-extension rank test consumes one.
    The combination stage explores all nonzero binary check vectors and independent orbit
    collections; only exhaustion of that full domain proves a nonfree check module. No
    special polynomial form or group-order parity is assumed. Worst-case cost is exponential.
    """
    order = parent.group.order
    empty = parent_binary[:0]
    count = 0

    def found_rows(
        rows: tuple[int, ...], method: Literal["parent", "greedy", "subsets"]
    ) -> CheckBasisResult:
        return CheckBasisResult(
            "found",
            abstract.RingArray.build(parent[list(rows)], parent.ring),
            rows,
            parent_rank,
            method,
            count,
            f"Found {len(rows)} independent check orbits ({method}).",
        )

    def limited(method: Literal["subsets", "combinations"]) -> CheckBasisResult:
        return CheckBasisResult(
            "inconclusive",
            None,
            None,
            parent_rank,
            method,
            count,
            f"No complete independent check-orbit basis found within {count} fallback trials; "
            "existence remains undecided. Increase check_search_limit or supply check_basis.",
        )

    if not parent_rank:
        return found_rows((), "parent")
    if parent_rank % order:
        return CheckBasisResult(
            "infeasible",
            None,
            None,
            parent_rank,
            "rank",
            0,
            f"No complete independent G-orbit basis exists: binary rank {parent_rank} is not "
            f"divisible by |G| = {order}. This does not invalidate the parent CSS checks.",
        )
    num_orbits = parent_rank // order
    if num_orbits == len(parent):
        return found_rows(tuple(range(len(parent))), "parent")

    selected: list[int] = []
    matrix = empty
    for index in range(len(parent)):
        candidate: galois.FieldArray = np.vstack(
            (matrix, parent_binary[index * order : (index + 1) * order])
        ).view(parent.field)
        if np.linalg.matrix_rank(candidate) == (len(selected) + 1) * order:
            selected.append(index)
            matrix = candidate
            if len(selected) == num_orbits:
                return found_rows(tuple(selected), "greedy")

    for rows in itertools.combinations(range(len(parent)), num_orbits):
        if max_candidates is not None and count >= max_candidates:
            return limited("subsets")
        indices = (np.array(rows)[:, None] * order + np.arange(order)).ravel()
        count += 1
        if np.linalg.matrix_rank(parent_binary[indices]) == parent_rank:
            return found_rows(rows, "subsets")

    # Each nonzero vector in the binary check space is the identity translate of a
    # polynomial prototype. Its whole orbit stays in that G-invariant check space.
    # The explicit DFS stack avoids Python's recursion limit and exponential caches.
    row_space = parent_binary.row_space()
    total = 1 << parent_rank
    stack: list[tuple[int, galois.FieldArray]] = [(1, empty)]
    prototypes: list[abstract.RingArray] = []
    while stack:
        index, matrix = stack[-1]
        if index == total:
            stack.pop()
            if prototypes:
                prototypes.pop()
            continue
        if max_candidates is not None and count >= max_candidates:
            return limited("combinations")
        stack[-1] = (index + 1, matrix)
        vector = row_space[[ii for ii in range(parent_rank) if index >> ii & 1]].sum(axis=0)
        prototype = prototype_from_vector(vector, parent.ring)
        candidate = np.vstack((matrix, prototype.regular_lift())).view(parent.field)
        count += 1
        if np.linalg.matrix_rank(candidate) != (len(prototypes) + 1) * order:
            continue
        prototypes.append(prototype)
        if len(prototypes) == num_orbits:
            basis = abstract.RingArray.build(np.vstack(prototypes), parent.ring)
            return CheckBasisResult(
                "found",
                basis,
                None,
                parent_rank,
                "combinations",
                count,
                f"Found {num_orbits} independent check orbits using polynomial combinations.",
            )
        stack.append((index + 1, candidate))

    return CheckBasisResult(
        "infeasible",
        None,
        None,
        parent_rank,
        "combinations",
        count,
        "No complete independent G-orbit basis exists: exhausted the full check-module domain. "
        "The check module is not free; this does not invalidate the parent CSS checks.",
    )


@dataclass(frozen=True)
class LogicalFrameResult:
    """Outcome of validating or seeking a common polynomial logical frame.

    Attributes:
        status: "found", "infeasible" (proved impossible), or "inconclusive" (budget exhausted).
        logical_ops: A validated polynomial frame on success, otherwise None.
        method: "supplied", "fiber", or "coefficients".
        num_candidates: Number of candidates tested by coefficient enumeration.
        message: Human-readable explanation of the outcome.
    """

    status: Literal["found", "infeasible", "inconclusive"]
    logical_ops: abstract.RingArray | None
    method: Literal["supplied", "fiber", "coefficients"]
    num_candidates: int
    message: str


class LogicalFrameError(ValueError):
    """A logical-frame solver did not find a frame; inspect ``result`` for its outcome."""

    result: LogicalFrameResult

    def __init__(self, result: LogicalFrameResult) -> None:
        self.result = result
        super().__init__(result.message)


def copy_matrix(
    matrix: abstract.RingArray, ring: abstract.GroupRing, name: str
) -> abstract.RingArray:
    """Validate a matrix's ring and make an entrywise copy in a common enumeration."""
    if not isinstance(matrix, abstract.RingArray):
        raise TypeError(f"{name} must be a RingArray")
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if matrix.ring != ring:
        raise ValueError(f"{name} must use the same group ring as the check polynomials")
    return abstract.RingArray.build(matrix, ring=ring)


def validate_logicals(checks: abstract.RingArray, logicals: abstract.RingArray) -> None:
    """Check the complete common logical frame, including augmentation parity."""
    num_checks, num_types = checks.shape
    num_logicals = num_types - 2 * num_checks
    if logicals.shape != (num_logicals, num_types):
        raise ValueError(
            f"logical_ops must have shape {(num_logicals, num_types)}, got {logicals.shape}"
        )
    if np.any(checks @ logicals.T):
        raise ValueError("logical_ops violate check/logical commutation (H J^dagger != 0)")
    identity = abstract.RingArray.build(np.eye(num_logicals, dtype=int), ring=checks.ring)
    if not np.array_equal(logicals @ logicals.T, identity):
        raise ValueError("logical_ops must be orthonormal (J J^dagger = I)")
    if not np.all(logicals.to_field_array().sum(axis=(0, 2)) == 1):
        raise ValueError("logical_ops violate augmentation parity (1_s J_aug != 1_m)")


def affine_solutions(
    matrix: galois.FieldArray, rhs: galois.FieldArray
) -> tuple[galois.FieldArray, galois.FieldArray] | None:
    """Return a particular solution and a kernel basis, or prove linear inconsistency."""
    field = type(matrix)
    num_variables = matrix.shape[1]
    reduced = np.column_stack((matrix, rhs)).view(field).row_reduce()
    nonzero = np.any(reduced[:, :num_variables], axis=1)
    if np.any(reduced[~nonzero, -1]):
        return None
    reduced = reduced[nonzero]
    pivots = np.array([np.flatnonzero(row[:-1])[0] for row in reduced], dtype=int)
    free = np.setdiff1d(np.arange(num_variables), pivots)
    particular = field.Zeros(num_variables)
    particular[pivots] = reduced[:, -1]
    kernel = field.Zeros((len(free), num_variables))
    kernel[:, free] = field.Identity(len(free))
    kernel[:, pivots] = -reduced[:, free].T
    return particular, kernel


def find_logicals(
    checks: abstract.RingArray,
    block_coefficients: abstract.RingArray,
    max_candidates: int | None,
) -> LogicalFrameResult:
    """Try the fiber frame, then enumerate the full affine coefficient domain up to a limit.

    The linear system enforces check/logical commutation and augmentation. The remaining Gram
    identity is checked on each candidate. Exhausting the affine space proves infeasibility;
    stopping early leaves existence undecided. No support or weight restriction is imposed.
    """
    ring = checks.ring
    field = ring.field
    group = ring.group
    num_checks, num_types = checks.shape
    num_logicals = num_types - 2 * num_checks
    _, num_cycles, cycle_order = block_coefficients.shape
    if (
        cycle_order % 2
        and num_logicals == num_cycles
        and not np.any(block_coefficients.sum(axis=2))
    ):
        logicals = abstract.RingArray.build(
            np.kron(np.eye(num_cycles, dtype=int), np.ones((1, cycle_order), dtype=int)), ring
        )
        validate_logicals(checks, logicals)
        return LogicalFrameResult(
            "found",
            logicals,
            "fiber",
            0,
            f"Found a compatible J with {num_logicals} logical orbits (fiber shortcut).",
        )

    # Variables are J[logical, physical_type, group_coefficient]. Taking the adjoint in
    # H J^dagger reverses each group coefficient before applying the regular lift of H.
    inverses = np.array([group.index(~gg) for gg in group.generate()])
    inverse_columns = (np.arange(num_types)[:, None] * group.order + inverses).ravel()
    check_equations = checks.regular_lift()[:, inverse_columns]
    matrix = np.kron(np.eye(num_logicals, dtype=int), check_equations).view(field)
    augmentation = np.tile(
        np.kron(np.eye(num_types, dtype=int), np.ones((1, group.order), dtype=int)),
        (1, num_logicals),
    ).view(field)
    rhs = field.Zeros(matrix.shape[0] + num_types)
    rhs[-num_types:] = 1
    solutions = affine_solutions(np.vstack((matrix, augmentation)).view(field), rhs)
    if solutions is None:
        return LogicalFrameResult(
            "infeasible",
            None,
            "coefficients",
            0,
            "No compatible J exists; check commutation and augmentation are inconsistent.",
        )

    vector, kernel = solutions
    total = 1 << len(kernel)
    count = total if max_candidates is None else min(total, max_candidates)
    identity = abstract.RingArray.build(np.eye(num_logicals, dtype=int), ring)
    for index in range(count):
        # Gray order visits every coefficient assignment, changing just one kernel vector.
        if index:
            vector += kernel[(index & -index).bit_length() - 1]
        logicals = abstract.RingArray.from_field_array(
            vector.reshape(num_logicals, num_types, group.order), ring
        )
        if np.array_equal(logicals @ logicals.T, identity):
            validate_logicals(checks, logicals)
            return LogicalFrameResult(
                "found",
                logicals,
                "coefficients",
                index + 1,
                f"Found a compatible J after testing {index + 1} coefficient candidates.",
            )
    if count == total:
        return LogicalFrameResult(
            "infeasible",
            None,
            "coefficients",
            count,
            "No compatible J exists; all candidates in the affine solution space failed.",
        )
    return LogicalFrameResult(
        "inconclusive",
        None,
        "coefficients",
        count,
        f"No compatible J found within {count} candidates; existence remains undecided. "
        "Increase logical_search_limit or supply logical_ops explicitly.",
    )
