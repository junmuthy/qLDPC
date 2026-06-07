"""Webster (Cross et al. arXiv:2407.18393) L-layer surgery construction.

build_layered_surgery_code is the single-logical workhorse — preserved
unchanged from the v1/v2 codebase.
"""

from __future__ import annotations

import dataclasses

import galois
import numpy as np
import numpy.typing as npt

from qldpc.codes.common import CSSCode


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
