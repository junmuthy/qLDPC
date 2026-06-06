"""Gadget construction for QLDPC lattice surgery.

Primary reference: Webster, Smith, Cohen, arXiv:2511.15989 §II.A Steps 1-3,
an explicit pedagogically clean 3-step recipe for building a logical-X
measurement gadget on any CSS code. The default ``num_layers=1`` mode
implements Webster's 3 steps verbatim; ``num_layers > 1`` activates the
multi-layer fallback of Cross et al. 2024 (arXiv:2407.18393 §III) for codes
whose induced Tanner graph has insufficient boundary Cheeger constant.

The two formulations produce the same merged code at L=1: Webster's "gadget
qubit kappa_j for each adjacent Z-check S_j" = Cross's "C_1 ancilla qubit
at the same index as the C_0 Z-check"; Webster's "X-check chi_i wired to
kappa_j iff q_i in S_j" = Cross's `[Pi_V_0, F^T]` row pattern.

See docs/superpowers/specs/2026-06-05-cross-layered-ancilla-design.md for
the full paper traceability and design rationale.

Copyright 2026 The qLDPC Authors.
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import dataclasses

import galois
import networkx as nx
import numpy as np
import numpy.typing as npt

from qldpc.objects import Pauli

from .common import CSSCode


@dataclasses.dataclass(frozen=True, eq=False)
class SurgeryLayout:
    """Provenance of qubits and checks in a merged surgery code.

    Returned by ``build_layered_surgery_code`` alongside the merged ``CSSCode``.
    Downstream pipelines (circuit synthesis, decoder configuration, plotting)
    can use the layout to distinguish data qubits from ancilla and identify
    which check rows are gauge-fixing.

    Attributes:
        num_data_qubits: Number of qubits in the original data code.
        num_ancilla_qubits: Total ancilla qubits across all L layers.
        num_layers: L. Always odd, >= 1.
        qubit_layer: Length (num_data + num_ancilla) array. Value 0 marks a
            data qubit; values 1..L mark the layer index of an ancilla qubit.
        v0_indices: Indices (within data qubits) of supp(X̄_M) = V_0.
        c0_indices: Row indices (within H_Z of data code) of Z-checks adjacent
            to V_0 = C_0.
        F: Step-1 restriction matrix; shape (|C_0|, |V_0|), equal to
            ``data_code.matrix_z[c0_indices][:, v0_indices]``.
        G: Step-4 gauge-fix basis; rows span the left null space of F (i.e.
            ``G @ F == 0``); shape (rank(left_null(F)), |C_0|).
        hx_row_kind: Length (num_x_checks_merged) string array. Values:
            "data" for old X-checks, "ancilla_L{i}" for new X-checks added by
            odd layer i in {1, 3, ..., L}.
        hz_row_kind: Length (num_z_checks_merged) string array. Values:
            "data" for old Z-checks, "ancilla_L{i}" for new Z-checks added by
            even layer i in {2, 4, ..., L-1}, "gauge_fix" for U_L rows.
    """

    num_data_qubits: int
    num_ancilla_qubits: int
    num_layers: int
    qubit_layer: npt.NDArray[np.int_]
    v0_indices: npt.NDArray[np.int_]
    c0_indices: npt.NDArray[np.int_]
    F: galois.FieldArray
    G: galois.FieldArray
    hx_row_kind: npt.NDArray
    hz_row_kind: npt.NDArray


def _restrict_to_logical_support(
    data_code: CSSCode,
    logical_op: npt.ArrayLike,
    num_layers: int,
    validate_logical_op: bool,
) -> tuple[np.ndarray, np.ndarray, galois.FieldArray]:
    """Compute V_0, C_0, F per Cross 2024 §III Step 1, with input validation.

    See spec §5 for the validation contract. Returns the indices V_0 (qubit
    columns) and C_0 (Z-check rows) into the data code, plus the restriction
    matrix F = H_Z[C_0, V_0] as a GF(2) ``galois.FieldArray``.

    The expensive row-span check (rejecting stabilizers as logical operators)
    is gated by ``validate_logical_op`` — see Task 3 / spec §5 item 6.
    """
    if data_code.is_subsystem_code:
        raise ValueError(
            "build_layered_surgery_code requires a stabilizer CSSCode, not a "
            "subsystem code."
        )
    if num_layers < 1 or num_layers % 2 != 1:
        raise ValueError(f"num_layers must be odd and >= 1, got {num_layers}.")

    field = data_code.field
    logical_op_arr = np.asarray(logical_op)
    n_data = data_code.num_qubits

    if logical_op_arr.shape != (n_data,):
        raise ValueError(
            f"logical_op has shape {logical_op_arr.shape}, expected ({n_data},)."
        )
    int_view = logical_op_arr.astype(np.int_, copy=False)
    if not np.all((int_view == 0) | (int_view == 1)):
        raise ValueError("logical_op must be binary (values in {0, 1}).")

    v0_indices = np.flatnonzero(int_view)
    if v0_indices.size == 0:
        raise ValueError("logical_op support V_0 is empty (logical_op is the zero vector).")

    logical_op_gf = field(int_view)
    hz = data_code.matrix_z
    # commutation with Z-stabilizers: H_Z @ X̄^T == 0 over GF(2)
    if np.any(hz @ logical_op_gf != 0):
        raise ValueError(
            "logical_op does not commute with Z-stabilizers (H_Z @ logical_op != 0)."
        )

    if validate_logical_op:
        hx = data_code.matrix_x
        # rank over GF(2): count nonzero rows of row-reduced form
        rank_hx = int(np.sum(np.any(hx.row_reduce() != 0, axis=1)))
        augmented = field(np.vstack([np.asarray(hx), logical_op_gf.reshape(1, -1)]))
        rank_aug = int(np.sum(np.any(augmented.row_reduce() != 0, axis=1)))
        if rank_aug == rank_hx:
            raise ValueError(
                "logical_op lies in the row span of H_X — it is a stabilizer, "
                "not a logical operator. Pass validate_logical_op=False to skip "
                "this check."
            )

    # Identify C_0: Z-check rows whose support intersects V_0.
    c0_mask = np.any(hz[:, v0_indices] != 0, axis=1)
    c0_indices = np.flatnonzero(c0_mask)
    if c0_indices.size == 0:
        raise ValueError(
            "No Z-checks of the data code touch V_0; the ancilla system cannot "
            "be constructed (degenerate logical operator)."
        )

    F = hz[c0_indices][:, v0_indices]
    return v0_indices, c0_indices, F


def _compute_gauge_fix(F: galois.FieldArray) -> galois.FieldArray:
    """Compute G whose rows form a basis of the left null space of F.

    Cross 2024 §III Step 4: ``null(F) = {c : c @ F == 0}``. We promote the
    CKBB gauge operators to stabilizers by introducing ``rank(null(F))`` new
    Z-checks U_L connected via G. Returns G with shape (rank, |C_0|).
    """
    return F.left_null_space()


@dataclasses.dataclass(frozen=True, eq=False)
class _LayeredBlocks:
    """Internal structural summary of an L-layer ancilla system.

    Holds F, F^T (cached), per-layer ancilla qubit sizes, and convenient
    column slices into the ancilla portion of the merged-code qubit register.
    Consumed by ``_assemble_merged_HX`` / ``_assemble_merged_HZ``.
    """

    F: galois.FieldArray
    F_T: galois.FieldArray
    num_layers: int
    n_v0: int
    n_c0: int

    @property
    def ancilla_layer_sizes(self) -> list[int]:
        """Sizes of ancilla qubit groups, indexed by layer i in 1..L.

        Odd i contribute |C_0| qubits (C_i in Cross notation);
        Even i contribute |V_0| qubits (V_i).
        """
        return [
            self.n_c0 if i % 2 == 1 else self.n_v0
            for i in range(1, self.num_layers + 1)
        ]

    @property
    def total_ancilla(self) -> int:
        return sum(self.ancilla_layer_sizes)

    def ancilla_col_slice(self, layer: int) -> slice:
        """Column slice for layer's ancilla qubits, relative to the ancilla block.

        Layer indexing is 1-based. Returns a slice into the ancilla columns
        (NOT including the n_data offset).
        """
        if layer < 1 or layer > self.num_layers:
            raise IndexError(f"layer must be in 1..{self.num_layers}, got {layer}")
        offset = sum(self.ancilla_layer_sizes[: layer - 1])
        size = self.ancilla_layer_sizes[layer - 1]
        return slice(offset, offset + size)


def _build_layered_blocks(F: galois.FieldArray, num_layers: int) -> _LayeredBlocks:
    """Build the structural summary of the L-layer ancilla system."""
    return _LayeredBlocks(
        F=F,
        F_T=F.T,
        num_layers=num_layers,
        n_v0=int(F.shape[1]),
        n_c0=int(F.shape[0]),
    )


def _assemble_merged_HX(
    data_code: CSSCode,
    blocks: _LayeredBlocks,
    v0_indices: np.ndarray,
) -> galois.FieldArray:
    """Assemble the merged H_X per spec §4.2 / §4.4.

    Block-row order: old data X-checks (zero-padded on ancilla), then new
    X-check rows from each odd layer i in {1, 3, ..., L}, |V_0| rows each.

    For layer i=1 the data-column block is the V_0 injection matrix Π_V_0;
    for i >= 3 the previous-layer block is identity on the V_{i-1} ancilla
    columns. Every odd-layer block has F^T on its own C_i columns and (if
    i+1 <= L) identity on the next V_{i+1} ancilla columns.
    """
    field = data_code.field
    n_data = data_code.num_qubits
    n_ancilla = blocks.total_ancilla
    n_merged = n_data + n_ancilla

    hx = data_code.matrix_x
    n_x_data = int(hx.shape[0])

    # Old data X-checks padded with zeros.
    old_x = field.Zeros((n_x_data, n_merged))
    old_x[:, :n_data] = hx

    # New X-check rows.
    I_v0 = field.Identity(blocks.n_v0)
    rows_per_layer = []
    for i in range(1, blocks.num_layers + 1, 2):  # odd i
        row_block = field.Zeros((blocks.n_v0, n_merged))

        if i == 1:
            # Π_V_0: identity-like injection from V_1 X-checks onto V_0 data qubits
            row_block[np.arange(blocks.n_v0), v0_indices] = 1
        else:
            # Identity on V_{i-1} ancilla columns (V_{i-1} has size |V_0|).
            prev_slice = blocks.ancilla_col_slice(i - 1)
            row_block[:, n_data + prev_slice.start : n_data + prev_slice.stop] = I_v0

        # F^T on C_i ancilla columns.
        ci_slice = blocks.ancilla_col_slice(i)
        row_block[:, n_data + ci_slice.start : n_data + ci_slice.stop] = blocks.F_T

        # Identity on V_{i+1} ancilla columns if the layer exists.
        if i + 1 <= blocks.num_layers:
            next_slice = blocks.ancilla_col_slice(i + 1)
            row_block[:, n_data + next_slice.start : n_data + next_slice.stop] = I_v0

        rows_per_layer.append(row_block)

    return field(np.vstack([old_x, *rows_per_layer]))


def _assemble_merged_HZ(
    data_code: CSSCode,
    blocks: _LayeredBlocks,
    G: galois.FieldArray,
    c0_indices: np.ndarray,
) -> galois.FieldArray:
    """Assemble the merged H_Z per spec §4.2 / §4.4.

    Block-row order:
        1. All old data Z-checks. Rows in ¬C_0 have zeros on every ancilla
           column; rows in C_0 get an identity-pattern extension on C_1.
        2. New Z-checks from each even layer i in {2, 4, ..., L-1}, |C_0|
           rows each. Pattern: I on C_{i-1}, F on V_i, I on C_{i+1}.
        3. Gauge-fix rows U_L: G on C_L, zero elsewhere.
    """
    field = data_code.field
    n_data = data_code.num_qubits
    n_ancilla = blocks.total_ancilla
    n_merged = n_data + n_ancilla

    hz = data_code.matrix_z
    n_z_data = int(hz.shape[0])

    # Old data Z-checks, with C_0 extension on C_1 ancilla columns.
    old_z = field.Zeros((n_z_data, n_merged))
    old_z[:, :n_data] = hz
    c1_slice = blocks.ancilla_col_slice(1)
    I_c0 = field.Identity(blocks.n_c0)
    old_z[c0_indices, n_data + c1_slice.start : n_data + c1_slice.stop] = I_c0

    # New Z-checks from even ancilla layers (i = 2, 4, ..., L-1).
    even_rows = []
    for i in range(2, blocks.num_layers, 2):
        row_block = field.Zeros((blocks.n_c0, n_merged))
        prev_slice = blocks.ancilla_col_slice(i - 1)
        cur_slice = blocks.ancilla_col_slice(i)
        next_slice = blocks.ancilla_col_slice(i + 1)
        row_block[:, n_data + prev_slice.start : n_data + prev_slice.stop] = I_c0
        row_block[:, n_data + cur_slice.start : n_data + cur_slice.stop] = blocks.F
        row_block[:, n_data + next_slice.start : n_data + next_slice.stop] = I_c0
        even_rows.append(row_block)

    # Gauge-fix rows on C_L.
    gauge_rows: list[galois.FieldArray] = []
    if G.shape[0] > 0:
        gf = field.Zeros((G.shape[0], n_merged))
        cL_slice = blocks.ancilla_col_slice(blocks.num_layers)
        gf[:, n_data + cL_slice.start : n_data + cL_slice.stop] = G
        gauge_rows.append(gf)

    return field(np.vstack([old_z, *even_rows, *gauge_rows]))


def build_layered_surgery_code(
    data_code: CSSCode,
    logical_op: npt.ArrayLike,
    *,
    num_layers: int = 1,
    validate_logical_op: bool = True,
) -> tuple[CSSCode, SurgeryLayout]:
    """Construct a merged stabilizer code that measures ``logical_op`` by lattice surgery.

    Implements the layered ancilla construction of Cross et al. 2024 §III
    (arXiv:2407.18393). Given a stabilizer CSSCode and the binary support
    vector of a logical X operator X̄_M, this builds ``num_layers`` ancilla
    layers (L must be odd) plus a top-layer gauge-fix Z-check block, and
    returns the merged CSSCode together with a SurgeryLayout describing the
    qubit / check partition.

    Args:
        data_code: The data CSSCode (stabilizer, not subsystem).
        logical_op: Binary row vector of length ``data_code.num_qubits``
            indicating supp(X̄_M).
        num_layers: Layer count L. Odd, >= 1. Default 1 follows the
            [[144,12,12]] gross code example in Cross et al. Table 1. For
            arbitrary logical_op, distance preservation may require L in
            {3, 5}; this function does not verify distance.
        validate_logical_op: If True (default), check that logical_op is
            not in the row span of H_X. Skip with False if the caller has
            already validated.

    Returns:
        (merged_code, layout):
            merged_code: CSSCode on (n_data + n_ancilla) qubits with logical
                dimension ``data_code.dimension - 1``.
            layout: SurgeryLayout describing qubit / check provenance.

    Raises:
        ValueError: See spec §5 for the exhaustive list of cases.
    """
    v0_indices, c0_indices, F = _restrict_to_logical_support(
        data_code, logical_op, num_layers, validate_logical_op
    )
    G = _compute_gauge_fix(F)
    blocks = _build_layered_blocks(F, num_layers)

    HX_merged = _assemble_merged_HX(data_code, blocks, v0_indices)
    HZ_merged = _assemble_merged_HZ(data_code, blocks, G, c0_indices)

    merged_code = CSSCode(HX_merged, HZ_merged, is_subsystem_code=False)

    layout = _build_layout(
        data_code, blocks, G, v0_indices, c0_indices, F
    )
    return merged_code, layout


def _build_layout(
    data_code: CSSCode,
    blocks: _LayeredBlocks,
    G: galois.FieldArray,
    v0_indices: np.ndarray,
    c0_indices: np.ndarray,
    F: galois.FieldArray,
) -> SurgeryLayout:
    """Assemble the SurgeryLayout dataclass from the building blocks."""
    n_data = data_code.num_qubits
    n_ancilla = blocks.total_ancilla
    qubit_layer = np.zeros(n_data + n_ancilla, dtype=np.int_)
    for i in range(1, blocks.num_layers + 1):
        slc = blocks.ancilla_col_slice(i)
        qubit_layer[n_data + slc.start : n_data + slc.stop] = i

    n_x_data = data_code.matrix_x.shape[0]
    hx_labels: list[str] = ["data"] * n_x_data
    for i in range(1, blocks.num_layers + 1, 2):  # odd
        hx_labels.extend([f"ancilla_L{i}"] * blocks.n_v0)
    hx_row_kind = np.array(hx_labels, dtype=object)

    n_z_data = data_code.matrix_z.shape[0]
    hz_labels: list[str] = ["data"] * n_z_data
    for i in range(2, blocks.num_layers, 2):  # even (>=2, <L)
        hz_labels.extend([f"ancilla_L{i}"] * blocks.n_c0)
    hz_labels.extend(["gauge_fix"] * int(G.shape[0]))
    hz_row_kind = np.array(hz_labels, dtype=object)

    return SurgeryLayout(
        num_data_qubits=n_data,
        num_ancilla_qubits=n_ancilla,
        num_layers=blocks.num_layers,
        qubit_layer=qubit_layer,
        v0_indices=v0_indices,
        c0_indices=c0_indices,
        F=F,
        G=G,
        hx_row_kind=hx_row_kind,
        hz_row_kind=hz_row_kind,
    )


import json as _json
import pathlib as _pathlib


_WEBSTER_APP_A_PATH = _pathlib.Path(__file__).resolve().parents[3] / "examples" / "webster_app_a.json"


def load_webster_seed_set(code_index: int) -> dict:
    """Load Webster (arXiv:2511.15989) Appendix A data for code index 0..3.

    The 4 codes are generalised bicycle codes with l in {31, 63, 127, 255},
    each having 4 seed operators (X_bar_1, Z_bar_1, X_bar_{k/2+1}, Z_bar_{k/2+1}).
    The data is read from ``examples/webster_app_a.json``.

    Returns:
        A dict matching the JSON schema.

    Raises:
        IndexError: if code_index is not in 0..3.
        FileNotFoundError: if the JSON fixture is missing.
    """
    if not 0 <= code_index <= 3:
        raise IndexError(f"code_index must be in 0..3, got {code_index}")
    with _WEBSTER_APP_A_PATH.open() as fh:
        data = _json.load(fh)
    return data["codes"][code_index]


def _build_generalised_bicycle_code(l: int, A_set: list[int], B_set: list[int]) -> CSSCode:
    """Build a generalised bicycle code from cyclic exponent sets A, B.

    Per Kovalev-Pryadko (arXiv:1212.6703) and Swaroop's reference
    implementation (https://github.com/eswaroop/adapters-LDPC-surgery,
    ext/bivariate_bicyclic.py): given subsets A, B of Z_l, let A(x) =
    sum(x^a for a in A_set) and B(x) = sum(x^b for b in B_set) as cyclic
    matrices in F_2[Z_l]. Then H_X = [A | B] and H_Z = [B^T | A^T] define
    the bicycle code on 2l data qubits.

    Args:
        l: cyclic group order.
        A_set, B_set: subsets of {0, 1, ..., l-1}.

    Returns:
        CSSCode on 2l data qubits with check matrices [A | B] and
        [B^T | A^T] over GF(2).
    """
    I_l = np.eye(l, dtype=np.int_)
    # cyclic shift matrix S such that S^k is left-shift by k (zero-indexed)
    S = np.roll(I_l, shift=-1, axis=0)
    A = np.zeros((l, l), dtype=np.int_)
    for a in A_set:
        A = (A + np.linalg.matrix_power(S, a)) % 2
    B = np.zeros((l, l), dtype=np.int_)
    for b in B_set:
        B = (B + np.linalg.matrix_power(S, b)) % 2

    H_X = np.hstack([A, B])
    H_Z = np.hstack([B.T, A.T])

    return CSSCode(H_X, H_Z, is_subsystem_code=False)


def _skip_tree(
    S: nx.Graph,
    root: int = 0,
    edge_index_verts: dict[tuple[int, int], int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """SkipTree basis transformation (Swaroop et al. arXiv:2410.03628 §III).

    Direct port of skipTree() in https://github.com/eswaroop/adapters-LDPC-surgery
    (MIT, 2025) skip_tree_algorithm.py with attribution. The qldpc project
    is Apache 2.0; MIT and Apache 2.0 are compatible for redistribution.

    Args:
        S: connected simple graph.
        root: vertex to start the labelling at.
        edge_index_verts: optional override mapping each edge ``tuple(sorted)``
            to a column index in T. If None, columns are indexed by
            ``S.edges()`` order.

    Returns:
        T: shape (n-1, |E|) edge-incidence matrix. T[l, e] = 1 iff edge e
            lies on the shortest path from vertex labeled l to vertex
            labeled (l+1) mod n.
        P: shape (n, n) permutation matrix. P[v, l] = 1 iff vertex v has
            label l.
    """
    n = S.number_of_nodes()
    index = 0
    label = [0] * n
    visited: set[int] = set()

    def label_first(v: int, skip: bool) -> None:
        nonlocal index
        visited.add(v)
        label[index] = v
        index = index + 1

        children = [nbr for nbr in S.neighbors(v) if nbr not in visited]
        for child_idx, child in enumerate(children):
            last_in_gen = child_idx == len(children) - 1
            if last_in_gen and not skip:
                label_first(child, skip=False)
            else:
                label_last(child)

    def label_last(v: int) -> None:
        nonlocal index
        visited.add(v)
        for child in S.neighbors(v):
            if child not in visited:
                label_first(child, skip=True)
        label[index] = v
        index = index + 1

    label_first(root, skip=False)

    P = np.zeros((n, n), dtype=np.int_)
    for l_idx, v in enumerate(label):
        P[v, l_idx] = 1

    if not edge_index_verts:
        edge_index_verts = {tuple(sorted(e)): i for i, e in enumerate(S.edges())}

    T = np.zeros((n - 1, len(edge_index_verts)), dtype=np.int_)

    for l_idx in range(n - 1):
        path = nx.shortest_path(S, source=label[l_idx], target=label[(l_idx + 1) % n])
        for u, v in zip(path[:-1], path[1:]):
            e = tuple(sorted((u, v)))
            T[l_idx, edge_index_verts[e]] = 1
    return T, P


def _cellulate_long_cycles(
    G: nx.Graph,
    edge_qubit_to_vertices: dict[int, tuple[int, int]],
    vert_to_edge: dict[tuple[int, int], int],
    G_mat: np.ndarray,
    max_len: int = 6,
) -> tuple[list[tuple[int, int]], dict[int, tuple[int, int]], dict[tuple[int, int], int], np.ndarray]:
    """Cellulation: break cycles longer than max_len by adding chord edges.

    Direct port of cellulate_long_cycles() in
    https://github.com/eswaroop/adapters-LDPC-surgery cellulation.py
    (MIT, 2025). Implements Lemma 14 of Swaroop et al. arXiv:2410.03628.

    For each cycle of length > max_len in nx.cycle_basis(G), adds a chord
    edge between vertex 0 and vertex n//2 of the cycle, then recomputes
    the cycle basis. Mutates G, edge_qubit_to_vertices, vert_to_edge, and
    G_mat in place.

    Args:
        G: graph to mutate.
        edge_qubit_to_vertices: dict mapping edge-qubit index -> vertex pair.
        vert_to_edge: inverse mapping.
        G_mat: edge-vertex incidence matrix (shape: |E| x |V|).
        max_len: maximum allowed cycle length. Default 6.

    Returns:
        (new_edges_added, edge_qubit_to_vertices, vert_to_edge, G_mat).
    """
    new_edges = []
    next_edge_index = (max(edge_qubit_to_vertices.keys()) + 1) if edge_qubit_to_vertices else 0

    while True:
        cycles = nx.cycle_basis(G)
        long_cycles = [c for c in cycles if len(c) > max_len]
        if not long_cycles:
            break
        cycle = long_cycles[0]
        n = len(cycle)
        u = cycle[0]
        v = cycle[(n // 2) % n]
        u, v = sorted((u, v))

        if not G.has_edge(u, v):
            G.add_edge(u, v)
            new_edges.append((u, v))
            edge_qubit_to_vertices[next_edge_index] = (u, v)
            vert_to_edge[(u, v)] = next_edge_index
            n_vertices = G_mat.shape[1]
            new_row = np.zeros((1, n_vertices), dtype=np.int_)
            new_row[0, u] = 1
            new_row[0, v] = 1
            G_mat = np.vstack([G_mat, new_row])
            next_edge_index += 1

    return new_edges, edge_qubit_to_vertices, vert_to_edge, G_mat


@dataclasses.dataclass(frozen=True, eq=False)
class BoostResult:
    """Statistics about a Cheeger boost run."""

    extra_qubits_added: int
    final_h_lower_bound: float
    iterations: int
    terminated_by: str  # "target_reached" | "max_qubits_exhausted" | "no_progress"


def _spectral_cheeger_lower_bound(F: galois.FieldArray) -> float:
    """Spectral lower bound on the boundary Cheeger constant of F.

    Returns ``lambda_2(F_float @ F_float.T) / 2.0``, where F_float =
    F.astype(np.float64). This is a tractable lower bound based on the
    discrete Cheeger inequality and is what boost_gadget_cheeger uses to
    decide when to stop adding augmentation qubits.

    Args:
        F: GF(2) restriction matrix of shape (|C_0|, |V_0|).

    Returns:
        Non-negative float lower bound on h(F).
    """
    F_float = np.asarray(F).astype(np.float64)
    if F_float.shape[0] < 2:
        return 0.0
    M = F_float @ F_float.T
    eigenvalues = np.linalg.eigvalsh(M)
    lambda_2 = float(eigenvalues[1])
    return max(0.0, lambda_2 / 2.0)


def boost_gadget_cheeger(
    merged: CSSCode,
    layout: SurgeryLayout,
    *,
    target_h: float = 1.0,
    max_extra_qubits: int | None = None,
    seed: int | None = None,
) -> tuple[CSSCode, SurgeryLayout, BoostResult]:
    """Heuristic Cheeger augmentation by random degree-2 edge addition.

    Implements Webster (arXiv:2511.15989) §II.A end's "+n" trick:
    iteratively add new κ' ancilla qubits to the gadget, each connecting
    a random pair of X-checks (χ_i, χ_j) not already directly connected
    via another κ, until the spectral lower bound on the boundary Cheeger
    constant of F reaches target_h.

    Args:
        merged: merged CSSCode returned by build_layered_surgery_code.
        layout: the associated SurgeryLayout (used to read F).
        target_h: target Cheeger lower bound. Default 1.0.
        max_extra_qubits: cap on additions. None = unbounded.
        seed: RNG seed for reproducibility.

    Returns:
        (boosted_merged, boosted_layout, result).

    Raises:
        ValueError: target_h <= 0, max_extra_qubits < 0, or F too small.
    """
    if target_h <= 0:
        raise ValueError(f"target_h must be positive, got {target_h}.")
    if max_extra_qubits is not None and max_extra_qubits < 0:
        raise ValueError(f"max_extra_qubits must be >= 0, got {max_extra_qubits}.")
    if layout.F.shape[1] < 2:
        raise ValueError(
            f"F has {layout.F.shape[1]} columns; need >= 2 X-checks to add an edge."
        )

    rng = np.random.default_rng(seed)
    field = layout.F.__class__
    F = np.asarray(layout.F).astype(np.int_).copy()
    n_X = F.shape[1]

    def _existing_pairs(F_arr: np.ndarray) -> set[tuple[int, int]]:
        pairs: set[tuple[int, int]] = set()
        for row in F_arr:
            ones = np.flatnonzero(row)
            for i in range(len(ones)):
                for j in range(i + 1, len(ones)):
                    pairs.add((int(ones[i]), int(ones[j])))
        return pairs

    extra = 0
    iterations = 0
    terminated_by = "no_progress"
    h_lb = _spectral_cheeger_lower_bound(field(F))
    max_iter_inner = 10 * n_X * n_X

    while True:
        iterations += 1
        h_lb = _spectral_cheeger_lower_bound(field(F))
        if h_lb >= target_h:
            terminated_by = "target_reached"
            break
        if max_extra_qubits is not None and extra >= max_extra_qubits:
            terminated_by = "max_qubits_exhausted"
            break
        if iterations > max_iter_inner:
            terminated_by = "no_progress"
            break

        pairs = _existing_pairs(F)
        candidate = None
        for _attempt in range(n_X * 2):
            i, j = sorted(int(x) for x in rng.choice(n_X, 2, replace=False))
            if (i, j) not in pairs:
                candidate = (i, j)
                break
        if candidate is None:
            terminated_by = "no_progress"
            break

        new_row = np.zeros(n_X, dtype=np.int_)
        new_row[candidate[0]] = 1
        new_row[candidate[1]] = 1
        F = np.vstack([F, new_row])
        extra += 1

    augmented_F = field(F)
    G = _compute_gauge_fix(augmented_F)
    blocks = _build_layered_blocks(augmented_F, layout.num_layers)
    n_data = layout.num_data_qubits

    data_x = np.asarray(merged.matrix_x[layout.hx_row_kind == "data"]).astype(np.int_)
    data_z = np.asarray(merged.matrix_z[layout.hz_row_kind == "data"]).astype(np.int_)
    data_x = field(data_x[:, :n_data])
    data_z = field(data_z[:, :n_data])
    data_code_proxy = CSSCode(data_x, data_z, is_subsystem_code=False)

    HX_new = _assemble_merged_HX(data_code_proxy, blocks, layout.v0_indices)
    HZ_new = _assemble_merged_HZ(data_code_proxy, blocks, G, layout.c0_indices)

    boosted_merged = CSSCode(HX_new, HZ_new, is_subsystem_code=False)
    boosted_layout = _build_layout(
        data_code_proxy, blocks, G, layout.v0_indices, layout.c0_indices, augmented_F
    )
    return boosted_merged, boosted_layout, BoostResult(
        extra_qubits_added=extra,
        final_h_lower_bound=float(h_lb),
        iterations=iterations,
        terminated_by=terminated_by,
    )


@dataclasses.dataclass(frozen=True, eq=False)
class JointSurgeryLayout:
    """Provenance of qubits and checks in a merged joint-measurement code.

    Returned by ``build_joint_measurement_code`` alongside the merged
    CSSCode. Captures the two individual gadget layouts plus bridge
    metadata.

    Attributes:
        gadget_layouts: Pair of SurgeryLayout instances, one per logical op.
        pauli_type: Pauli.X for X̄_1 X̄_2; Pauli.Z for Z̄_1 Z̄_2.
        num_data_qubits: Number of qubits in the original data code.
        num_ancilla_qubits: gadget1.num_ancilla + gadget2.num_ancilla.
        num_bridge_qubits: Bridge qubits introduced by SkipTree.
        bridge_qubit_slice: Column slice for bridge qubits within the
            merged qubit register (after data + both gadget ancillas).
        u_b_check_kind_mask: Boolean mask over merged H_Z rows marking the
            U_B bridge stabilizer rows.
    """

    gadget_layouts: tuple[SurgeryLayout, SurgeryLayout]
    pauli_type: Pauli
    num_data_qubits: int
    num_ancilla_qubits: int
    num_bridge_qubits: int
    bridge_qubit_slice: slice
    u_b_check_kind_mask: npt.NDArray[np.bool_]


def _validate_joint_logical_ops(
    data_code: CSSCode,
    op1: np.ndarray,
    op2: np.ndarray,
) -> Pauli:
    """Validate joint-measurement inputs and return the detected Pauli type.

    Detects whether (op1, op2) are both logical-X (commute with H_Z) or
    both logical-Z (commute with H_X), and rejects mixed types.

    Raises:
        ValueError: data_code.dimension < 2, mixed Pauli types, or either
            op fails the v1 single-operator validation contract.
    """
    if data_code.dimension < 2:
        raise ValueError(
            f"joint measurement requires at least 2 logical qubits, got "
            f"data_code.dimension={data_code.dimension}."
        )

    field = data_code.field

    def _is_x_type(op: np.ndarray) -> bool:
        gf_op = field(op)
        return bool(np.all((data_code.matrix_z @ gf_op) == 0))

    def _is_z_type(op: np.ndarray) -> bool:
        gf_op = field(op)
        return bool(np.all((data_code.matrix_x @ gf_op) == 0))

    op1_x = _is_x_type(op1)
    op2_x = _is_x_type(op2)
    op1_z = _is_z_type(op1)
    op2_z = _is_z_type(op2)

    if op1_x and op2_x and not (op1_z and op2_z):
        return Pauli.X
    if op1_z and op2_z and not (op1_x and op2_x):
        return Pauli.Z
    if op1_x and op2_z and not op2_x:
        raise ValueError("op1 and op2 must be the same Pauli type (op1 is X, op2 is Z).")
    if op1_z and op2_x and not op1_x:
        raise ValueError("op1 and op2 must be the same Pauli type (op1 is Z, op2 is X).")
    raise ValueError(
        "Could not detect a consistent Pauli type for op1 and op2; check that "
        "each is a valid logical operator of data_code."
    )


@dataclasses.dataclass(frozen=True, eq=False)
class _BridgeSpec:
    """Internal output of _build_bridge_via_skiptree.

    Attributes:
        num_bridge_qubits: number of bridge qubits introduced.
        u_b_rows: shape (n_u_b, |C_0_1| + |C_0_2| + num_bridge_qubits)
            GF(2) matrix. Each row is one U_B Z-stabilizer over
            (κ_j_1, κ_j_2, bridge) qubits.
        interface_vertex_to_qubit: dict mapping interface-graph vertex
            index → (block, kappa_index) where block ∈ {0, 1}.
    """

    num_bridge_qubits: int
    u_b_rows: galois.FieldArray
    interface_vertex_to_qubit: dict[int, tuple[int, int]]


def _build_bridge_via_skiptree(
    layout1: SurgeryLayout,
    layout2: SurgeryLayout,
) -> _BridgeSpec:
    """Construct the same-block joint-measurement bridge.

    Algorithm:
    1. Interface graph S = (V, E):
       V = κ_j_1 vertices (one per layout1.c0_indices) ∪ κ_j_2 vertices.
       E = pairs (κ_j_1, κ_k_2) where data Z-checks indexed j and k share
           at least one qubit in supp(op1) ∩ supp(op2).
    2. Add chord edges via _cellulate_long_cycles(S, max_len=6).
    3. Run _skip_tree(S, root=0) → T (n-1, |E|) and P (n, n).
    4. Bridge qubits = n - 1.
       For row r of T, U_B stabilizer acts on:
         interface vertices that appear an ODD number of times in T[r]
         (i.e., XOR of endpoints of all edges in T[r])
         plus the bridge qubit r.

    Raises:
        ValueError: if the interface graph has no edges or is disconnected.
    """
    n_kappa_1 = layout1.F.shape[0]
    n_kappa_2 = layout2.F.shape[0]
    field = layout1.F.__class__

    S = nx.Graph()
    S.add_nodes_from(range(n_kappa_1 + n_kappa_2))
    interface_vertex_to_qubit: dict[int, tuple[int, int]] = {}
    for j_idx in range(n_kappa_1):
        interface_vertex_to_qubit[j_idx] = (0, j_idx)
    for k_idx in range(n_kappa_2):
        interface_vertex_to_qubit[n_kappa_1 + k_idx] = (1, k_idx)

    F1 = np.asarray(layout1.F).astype(np.int_)
    F2 = np.asarray(layout2.F).astype(np.int_)
    v0_1 = layout1.v0_indices
    v0_2 = layout2.v0_indices
    common_qubits = np.intersect1d(v0_1, v0_2)

    edge_qubit_to_vertices: dict[int, tuple[int, int]] = {}
    vert_to_edge: dict[tuple[int, int], int] = {}
    next_edge_index = 0
    for q in common_qubits:
        col1 = int(np.where(v0_1 == q)[0][0])
        col2 = int(np.where(v0_2 == q)[0][0])
        j_indices = np.flatnonzero(F1[:, col1])
        k_indices = np.flatnonzero(F2[:, col2])
        for j in j_indices:
            for k in k_indices:
                u, v = sorted((int(j), int(n_kappa_1 + k)))
                if (u, v) not in vert_to_edge:
                    S.add_edge(u, v)
                    edge_qubit_to_vertices[next_edge_index] = (u, v)
                    vert_to_edge[(u, v)] = next_edge_index
                    next_edge_index += 1

    if next_edge_index == 0:
        raise ValueError(
            "interface graph has no edges; gadgets' V_0 supports do not overlap "
            "via any data Z-check, so no same-block bridge is possible."
        )
    if not nx.is_connected(S.subgraph([v for v in S.nodes if S.degree(v) > 0])):
        raise ValueError(
            "interface graph between the two gadgets is disconnected; "
            "same-block joint-measurement bridge requires a connected interface."
        )

    G_mat = np.zeros((next_edge_index, n_kappa_1 + n_kappa_2), dtype=np.int_)
    for ei, (u, v) in edge_qubit_to_vertices.items():
        G_mat[ei, u] = 1
        G_mat[ei, v] = 1
    _, edge_qubit_to_vertices, vert_to_edge, G_mat = _cellulate_long_cycles(
        S, edge_qubit_to_vertices, vert_to_edge, G_mat, max_len=6
    )

    # Restrict S to vertices with degree > 0 for SkipTree (the algorithm needs a connected component)
    connected_vertices = sorted([v for v in S.nodes if S.degree(v) > 0])
    if len(connected_vertices) < 2:
        raise ValueError("interface graph has fewer than 2 connected vertices.")
    # _skip_tree assumes vertices are labeled 0..n-1; relabel and translate back at the end.
    relabel_map = {v: i for i, v in enumerate(connected_vertices)}
    inv_relabel = {i: v for v, i in relabel_map.items()}
    S_sub = nx.relabel_nodes(S.subgraph(connected_vertices).copy(), relabel_map)
    root = relabel_map[connected_vertices[0]]

    T, P = _skip_tree(S_sub, root=root)
    num_bridge_qubits = T.shape[0]
    n_interface = n_kappa_1 + n_kappa_2
    u_b_rows_arr = np.zeros((num_bridge_qubits, n_interface + num_bridge_qubits), dtype=np.int_)
    # Build an edge_to_index map for the subgraph, then translate vertices back to the original S labels.
    sub_edge_index_verts = {tuple(sorted(e)): i for i, e in enumerate(S_sub.edges())}
    inv_sub_edges = {i: tuple(sorted(e)) for e, i in sub_edge_index_verts.items()}

    for r in range(num_bridge_qubits):
        # Edges on this row's shortest path in S_sub
        for e_idx in np.flatnonzero(T[r]):
            u_sub, v_sub = inv_sub_edges[e_idx]
            u = inv_relabel[u_sub]
            v = inv_relabel[v_sub]
            u_b_rows_arr[r, u] = (u_b_rows_arr[r, u] + 1) % 2
            u_b_rows_arr[r, v] = (u_b_rows_arr[r, v] + 1) % 2
        # bridge qubit r
        u_b_rows_arr[r, n_interface + r] = 1

    return _BridgeSpec(
        num_bridge_qubits=num_bridge_qubits,
        u_b_rows=field(u_b_rows_arr),
        interface_vertex_to_qubit=interface_vertex_to_qubit,
    )


def _stitch_gadgets_with_bridge(
    data_code: CSSCode,
    merged1: CSSCode,
    layout1: SurgeryLayout,
    merged2: CSSCode,
    layout2: SurgeryLayout,
    bridge: _BridgeSpec,
    *,
    pauli_type: Pauli,
) -> tuple[CSSCode, JointSurgeryLayout]:
    """Combine two single-operator merged codes with bridge into a joint CSSCode.

    Qubit register:
        [ data qubits | layout1.ancilla | layout2.ancilla | bridge ]
        n_data         n_anc_1            n_anc_2           n_bridge

    Construction (X-type joint, ``pauli_type=Pauli.X``):
        - H_X rows: HX1 padded to the joint register, plus HX2's non-"data"
          rows padded (the "data" X-checks of merged2 duplicate those of
          merged1 on data columns and are zero on every ancilla, so they
          are removed).
        - H_Z rows: HZ1 padded + HZ2's non-"data" rows padded + U_B rows.
          HZ2's "data" rows are subsumed by HZ1's "data" rows, but for each
          j in ``layout2.c0_indices`` we splice in the gadget-2 ancilla
          connection (identity on the κ_j_2 column) onto the corresponding
          HZ1 row so the gadget-2 c0 connection isn't lost.
        - Each U_B[r] row carries Z support on the joint register equal to
          the SUM of the gadget data Z-checks at the path endpoints (i.e.
          for endpoint vertex j in block 1: full ``HZ1`` row at C0-row
          ``c0_1_to_row[j]``; for endpoint vertex k in block 2: the gadget-2
          ancilla connection ``Z_{κ_k_2}`` plus the data Z-check S_{c0_2[k]}).
          The "Z on bridge qubit r" lives on the bridge column. Because this
          U_B is itself a SUM of existing data Z-stabilizers (which already
          commute with both gadgets' X-stabilizers) plus a Z on a fresh
          bridge qubit, CSS commutation is guaranteed.

    Returns the joint merged CSSCode and the JointSurgeryLayout.
    """
    field = data_code.field
    n_data = data_code.num_qubits
    n_anc_1 = layout1.num_ancilla_qubits
    n_anc_2 = layout2.num_ancilla_qubits
    n_bridge = bridge.num_bridge_qubits
    n_merged = n_data + n_anc_1 + n_anc_2 + n_bridge

    HX1 = np.asarray(merged1.matrix_x).astype(np.int_)
    HX2 = np.asarray(merged2.matrix_x).astype(np.int_)
    HZ1 = np.asarray(merged1.matrix_z).astype(np.int_)
    HZ2 = np.asarray(merged2.matrix_z).astype(np.int_)

    def _pad_row(matrix: np.ndarray, *, ancilla_block: int) -> np.ndarray:
        """Embed a row of merged_i into the joint register column layout."""
        out = np.zeros((matrix.shape[0], n_merged), dtype=np.int_)
        out[:, :n_data] = matrix[:, :n_data]
        if ancilla_block == 0:
            out[:, n_data : n_data + n_anc_1] = matrix[:, n_data:]
        else:
            out[:, n_data + n_anc_1 : n_data + n_anc_1 + n_anc_2] = matrix[:, n_data:]
        return out

    HX1_padded = _pad_row(HX1, ancilla_block=0)
    HX2_padded = _pad_row(HX2, ancilla_block=1)
    HZ1_padded = _pad_row(HZ1, ancilla_block=0)
    HZ2_padded = _pad_row(HZ2, ancilla_block=1)

    # De-duplicate gadget2's "data" X-checks (they replicate gadget1's after
    # padding — same on data columns, zero on every ancilla and bridge column).
    is_data_hx_2 = layout2.hx_row_kind == "data"
    HX2_padded = HX2_padded[~is_data_hx_2]

    # Splice the gadget-2 κ_k_2 connection onto HZ1's c0_2 rows, then drop
    # HZ2's "data" rows (now subsumed). For each j ∈ layout2.c0_indices, the
    # k-th row of layout2.F maps to data Z-check row j of the data code; the
    # j-th data row of HZ1_padded becomes (S_j on data) + Z_κ_j_1 (if j ∈
    # c0_1, from merged1) + Z_κ_j_2 (newly added from merged2 connection).
    n_z_data = int(data_code.matrix_z.shape[0])
    is_data_hz_1 = layout1.hz_row_kind == "data"
    data_hz1_row_indices = np.flatnonzero(is_data_hz_1)
    # Map: data Z-check index j  ->  row index in HZ1_padded
    # The "data" rows of HZ1 are the first n_z_data rows (see _assemble_merged_HZ).
    assert data_hz1_row_indices.size == n_z_data
    blocks2 = _build_layered_blocks(layout2.F, layout2.num_layers)
    c1_slice_2 = blocks2.ancilla_col_slice(1)
    # offset of κ_k_2 within joint register
    kappa2_col_start = n_data + n_anc_1 + c1_slice_2.start
    for k, j in enumerate(layout2.c0_indices):
        row_in_hz1 = int(data_hz1_row_indices[j])
        HZ1_padded[row_in_hz1, kappa2_col_start + k] = 1

    is_data_hz_2 = layout2.hz_row_kind == "data"
    HZ2_padded = HZ2_padded[~is_data_hz_2]

    # U_B rows: bridge.u_b_rows has columns [n_kappa_1 | n_kappa_2 | n_bridge].
    # Map each interface-vertex column to the FULL data Z-check of the matching
    # gadget, so each U_B row is (sum of existing data Z-stabilizers of the two
    # gadgets at the path endpoints) + Z on bridge qubit r. This is the only
    # construction that preserves CSS commutation with chi_v X-checks of both
    # gadgets (since each summand individually commutes with H_X).
    n_k1 = layout1.F.shape[0]
    n_k2 = layout2.F.shape[0]
    u_b_arr = np.asarray(bridge.u_b_rows).astype(np.int_)
    if u_b_arr.shape[1] != n_k1 + n_k2 + n_bridge:
        raise ValueError(
            f"bridge.u_b_rows width {u_b_arr.shape[1]} != n_kappa_1 + n_kappa_2 "
            f"+ n_bridge ({n_k1} + {n_k2} + {n_bridge}); cannot stitch."
        )

    blocks1 = _build_layered_blocks(layout1.F, layout1.num_layers)
    c1_slice_1 = blocks1.ancilla_col_slice(1)
    kappa1_col_start = n_data + c1_slice_1.start

    u_b_padded = np.zeros((u_b_arr.shape[0], n_merged), dtype=np.int_)
    for r in range(u_b_arr.shape[0]):
        # Each block-1 endpoint κ_j_1 contributes spliced HZ1's data row at
        # c0_1[j]: this row carries S_{c0_1[j]} on data, Z_κ_j_1 on gadget-1
        # ancilla, and (if c0_1[j] ∈ c0_2) Z_κ_{k}_2 on gadget-2 ancilla. It
        # already commutes with chi rows of BOTH gadgets (see the splice
        # analysis above), so any sum of such rows + Z on a fresh bridge qubit
        # also commutes.
        for j_in_k1 in range(n_k1):
            if u_b_arr[r, j_in_k1] == 1:
                row_in_hz1 = int(data_hz1_row_indices[layout1.c0_indices[j_in_k1]])
                u_b_padded[r] = (u_b_padded[r] + HZ1_padded[row_in_hz1]) % 2
        # Each block-2 endpoint κ_k_2 ALSO contributes spliced HZ1's data row
        # at c0_2[k] — by the splice, this row carries Z_κ_k_2 on gadget-2
        # ancilla (and S_{c0_2[k]} on data, and Z_κ_j_1 if c0_2[k] ∈ c0_1).
        # Using the spliced row guarantees the cross-gadget commutation.
        for k_in_k2 in range(n_k2):
            if u_b_arr[r, n_k1 + k_in_k2] == 1:
                row_in_hz1 = int(data_hz1_row_indices[layout2.c0_indices[k_in_k2]])
                u_b_padded[r] = (u_b_padded[r] + HZ1_padded[row_in_hz1]) % 2
        # Bridge qubit r.
        for r2 in range(n_bridge):
            if u_b_arr[r, n_k1 + n_k2 + r2] == 1:
                u_b_padded[r, n_data + n_anc_1 + n_anc_2 + r2] ^= 1

    HX_joint = field(np.vstack([HX1_padded, HX2_padded]))
    HZ_joint = field(np.vstack([HZ1_padded, HZ2_padded, u_b_padded]))

    joint_merged = CSSCode(HX_joint, HZ_joint, is_subsystem_code=False)

    bridge_slice = slice(n_data + n_anc_1 + n_anc_2, n_merged)
    u_b_check_kind_mask = np.zeros(HZ_joint.shape[0], dtype=bool)
    if u_b_arr.shape[0] > 0:
        u_b_check_kind_mask[-u_b_arr.shape[0]:] = True

    joint_layout = JointSurgeryLayout(
        gadget_layouts=(layout1, layout2),
        pauli_type=pauli_type,
        num_data_qubits=n_data,
        num_ancilla_qubits=n_anc_1 + n_anc_2,
        num_bridge_qubits=n_bridge,
        bridge_qubit_slice=bridge_slice,
        u_b_check_kind_mask=u_b_check_kind_mask,
    )
    return joint_merged, joint_layout
