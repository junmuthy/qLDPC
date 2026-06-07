"""Joint-measurement construction (v2 path-graph bridge).

This v2 implementation is preserved here as a holding place. It will be
replaced by the SkipTree-based v3 (per Ide arXiv:2410.03628 §VII B + §VII C)
in subsequent tasks (8–13) of the v3 plan.
"""

from __future__ import annotations

import dataclasses

import galois
import networkx as nx
import numpy as np
import numpy.typing as npt

from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli

from .layered import (
    SurgeryLayout,
    _build_layered_blocks,
    build_layered_surgery_code,
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
        u_b_check_kind_mask: Boolean mask over merged H_X rows (since the
            bridge stabilizers are X-type in the joint-X measurement case)
            marking the U_B bridge stabilizer rows.
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
    """Internal bridge specification for joint-measurement code construction.

    Per Cross §3.6 + Webster (arXiv:2511.15989) paragraph on bridges. The
    bridge is a path graph of w data qubits with w-1 X-type stabilizers,
    where w = min(wt(L_1), wt(L_2)) = min(|V_0_1|, |V_0_2|). One χ from
    each gadget is extended with X on a bridge endpoint:
    - χ_0^(1) gets X on bridge qubit 0
    - χ_0^(2) gets X on bridge qubit w-1

    Attributes:
        num_bridge_qubits: w.
        u_b_x_rows: shape (w-1, w) GF(2) matrix. Row i has 1s at columns
            i and i+1, encoding the X_{b_i} X_{b_{i+1}} path stabilizer.
        gadget1_extended_chi_row_in_v1: index of the χ in gadget1's V_1
            block (0-indexed) that gets extended with X on bridge qubit 0.
            For this simple design, always 0.
        gadget2_extended_chi_row_in_v1: index for gadget2. Always 0.
        gadget1_bridge_qubit_idx: which bridge qubit attaches to gadget 1.
            For this design, always 0.
        gadget2_bridge_qubit_idx: which bridge qubit attaches to gadget 2.
            For this design, always w-1.
    """
    num_bridge_qubits: int
    u_b_x_rows: galois.FieldArray
    gadget1_extended_chi_row_in_v1: int
    gadget2_extended_chi_row_in_v1: int
    gadget1_bridge_qubit_idx: int
    gadget2_bridge_qubit_idx: int


def _build_bridge_via_skiptree(
    layout1: SurgeryLayout,
    layout2: SurgeryLayout,
) -> _BridgeSpec:
    """Bridge construction for joint X̄_1 X̄_2 measurement.

    Per Cross §3.6 (arXiv:2407.18393) + Webster (arXiv:2511.15989)
    paragraph on bridges with disjoint logical support:

    - w = min(wt(L_1), wt(L_2)) bridge data qubits b_0..b_{w-1}
    - w-1 X-type stabilizers: X_{b_i} X_{b_{i+1}} for i in 0..w-2 (path)
    - Endpoint attachment: χ_0^(1) ⊗ X_{b_0}, χ_0^(2) ⊗ X_{b_{w-1}}

    SkipTree on a path graph is the identity (no optimization needed for
    the simple case). The function name retains "via_skiptree" for API
    consistency; future generalizations to non-path interface graphs
    would use the SkipTree primitive.

    Raises:
        ValueError: if w = 0 (one of the logical operators has empty
            support — caller should have rejected this earlier).
    """
    w = min(int(layout1.v0_indices.size), int(layout2.v0_indices.size))
    if w == 0:
        raise ValueError(
            "min(wt(L_1), wt(L_2)) is 0; cannot build a bridge between "
            "logical operators with empty support."
        )
    field = layout1.F.__class__

    # Path-graph X-stabilizers on bridge qubits only.
    u_b_x_rows = np.zeros((max(w - 1, 0), w), dtype=np.int_)
    for i in range(w - 1):
        u_b_x_rows[i, i] = 1
        u_b_x_rows[i, i + 1] = 1

    return _BridgeSpec(
        num_bridge_qubits=w,
        u_b_x_rows=field(u_b_x_rows),
        gadget1_extended_chi_row_in_v1=0,
        gadget2_extended_chi_row_in_v1=0,
        gadget1_bridge_qubit_idx=0,
        gadget2_bridge_qubit_idx=w - 1,
    )


def _solve_gf2_system(A: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """Solve A x = b over GF(2). Returns x or None if infeasible.

    A has shape (m, n), b has shape (m,), x has shape (n,).
    """
    GF2 = galois.GF(2)
    m, n = A.shape
    aug = GF2(np.hstack([A.astype(np.int_), b.reshape(-1, 1).astype(np.int_)]))
    rref = np.asarray(aug.row_reduce())
    x = np.zeros(n, dtype=np.int_)
    for r in range(m):
        nz = np.flatnonzero(rref[r, :n])
        if nz.size == 0:
            if rref[r, n] == 1:
                return None
            continue
        x[int(nz[0])] = int(rref[r, n])
    return x


def _find_bridge_z_stab_data_logical(
    data_code: CSSCode,
    op1: np.ndarray,
    op2: np.ndarray,
) -> np.ndarray | None:
    """Find a Z-Pauli y_d of data code that anti-commutes with both op1 and op2.

    Such y_d corresponds to (Z̄_1 + Z̄_2)-class representative. Used to build
    the bridge Z-stab that constrains the bridge gauge X-DOFs.

    Returns y_d as binary vector of length n_data, or None if not found.
    """
    for zl in data_code.get_logical_ops(Pauli.Z):
        zl_arr = np.asarray(zl).astype(np.int_)
        if (zl_arr @ op1) % 2 == 1 and (zl_arr @ op2) % 2 == 1:
            return zl_arr
    return None


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
    """Combine two single-operator merged codes with a path-graph bridge.

    Per Cross §3.6 + Webster paragraph: the bridge introduces w data
    qubits with w-1 X-type stabilizers. One χ in each gadget is extended
    with X on the corresponding bridge endpoint.

    Qubit register layout:
        [ data | layout1.ancilla | layout2.ancilla | bridge ]
        n_data  n_anc_1            n_anc_2           n_bridge

    H_X rows (in order):
        - data X-checks (from data_code, identical between the two gadgets)
        - layout1 V_1 X-check rows (with χ_0^(1) extended by X_{b_0})
        - layout2 V_1 X-check rows (with χ_0^(2) extended by X_{b_{w-1}})
        - bridge U_B X-stab path rows (w-1 rows)

    H_Z rows:
        - layout1's data + gadget Z rows (already correct: data Z extensions
          onto layout1 κ_j; gauge-fix rows on layout1's C_L)
        - layout2's gadget-side Z rows ONLY (data extension to layout2 κ_j;
          gauge-fix on layout2's C_L); the non-C_0_2 data Z rows are
          duplicates already in layout1's output and are dropped.
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

    # Extend ONE χ row in each gadget's HX with X on the bridge endpoint.
    # χ rows in HX1 are at indices where layout1.hx_row_kind == "ancilla_L1".
    n_x_data = int(np.sum(layout1.hx_row_kind == "data"))
    # χ_0^(1) is the first χ row after the data X-check rows.
    chi_row_1 = n_x_data + bridge.gadget1_extended_chi_row_in_v1
    chi_row_2 = int(np.sum(layout2.hx_row_kind == "data")) + bridge.gadget2_extended_chi_row_in_v1
    bridge_col_start = n_data + n_anc_1 + n_anc_2

    HX1_padded[chi_row_1, bridge_col_start + bridge.gadget1_bridge_qubit_idx] = 1
    HX2_padded[chi_row_2, bridge_col_start + bridge.gadget2_bridge_qubit_idx] = 1

    # De-duplicate gadget2's data X-checks (= gadget1's).
    is_data_hx_2 = layout2.hx_row_kind == "data"
    HX2_padded_nondata = HX2_padded[~is_data_hx_2]

    # Compose gadget1's data Z rows with gadget2's κ_2 extensions: for each
    # j ∈ layout2.c0_indices, splice the κ_2 column onto HZ1's data row j so
    # that the merged Z-stabilizer carries BOTH κ_1 (if j ∈ c0_1) and κ_2 (if
    # j ∈ c0_2) extensions. This guarantees chi rows from BOTH gadgets commute
    # with every Z-stabilizer (overlap on data is cancelled by overlap on the
    # respective κ block).
    is_data_hz_1 = layout1.hz_row_kind == "data"
    data_hz1_row_indices = np.flatnonzero(is_data_hz_1)
    n_z_data = int(data_code.matrix_z.shape[0])
    assert data_hz1_row_indices.size == n_z_data
    blocks2 = _build_layered_blocks(layout2.F, layout2.num_layers)
    c1_slice_2 = blocks2.ancilla_col_slice(1)
    kappa2_col_start = n_data + n_anc_1 + c1_slice_2.start
    for k, j in enumerate(layout2.c0_indices):
        row_in_hz1 = int(data_hz1_row_indices[j])
        HZ1_padded[row_in_hz1, kappa2_col_start + k] = 1

    # Drop ALL of HZ2's data rows: the data Z stabilizers are now fully
    # captured by HZ1_padded (with κ_1 + κ_2 extensions spliced in above).
    is_data_hz_2 = layout2.hz_row_kind == "data"
    HZ2_padded_filtered = HZ2_padded[~is_data_hz_2]

    # Bridge U_B X-type stabilizers, padded onto joint register (zero on data,
    # zero on both gadget ancillas, the X-stab pattern on bridge cols).
    n_u_b = bridge.u_b_x_rows.shape[0]
    u_b_x_arr = np.asarray(bridge.u_b_x_rows).astype(np.int_)
    u_b_padded = np.zeros((n_u_b, n_merged), dtype=np.int_)
    if n_u_b > 0:
        u_b_padded[:, bridge_col_start : bridge_col_start + n_bridge] = u_b_x_arr

    # IMPORTANT NOTE on bridge X-logical structure:
    #
    # Each bridge qubit X_{b_q} appears as a weight-1 representative of the
    # X̄_1 ≡ X̄_2 equivalence class (linked via Σ_i χ_i^(s) = X̄_s on data +
    # e_{b_endpoint} on bridge, then chained through path stabilizers). This
    # is a FEATURE, not a bug: the weight-1 bridge representative is exactly
    # the low-weight logical observable that the surgery protocol measures to
    # read out the joint X̄_1 X̄_2 eigenvalue.
    #
    # For MEMORY experiments on the joint code (artificial; joint code is
    # transient in real surgery use), the weight-1 representative gives
    # d_X(joint) = 1. This is NOT meaningful for joint surgery protocol
    # fault tolerance, which involves repeated measurements + readout, not
    # state storage. The joint code's actual role is as a transient code
    # during the merge phase; readout via bridge X is by design.
    #
    # Adding a Z-stab to eliminate the bridge representatives would also
    # consume the X̄_1 ≡ X̄_2 class as a logical (k_joint = k_data - 2 instead
    # of k_data - 1), violating Cross §3.6's "one logical DOF consumed".
    # See test_joint_code_has_spurious_bridge_weight1_xlogicals_v2_limitation
    # for diagnostic.

    # Per Cross §3.6: a joint X̄_1 X̄_2 measurement consumes ONE logical degree of
    # freedom (the product's eigenvalue). k_joint = k_data - 1.
    HX_joint = field(np.vstack([HX1_padded, HX2_padded_nondata, u_b_padded]))
    HZ_joint = field(np.vstack([HZ1_padded, HZ2_padded_filtered]))

    joint_merged = CSSCode(HX_joint, HZ_joint, is_subsystem_code=False)

    bridge_slice = slice(bridge_col_start, n_merged)
    # u_b_check_kind_mask: True on the bridge X-stab rows in HX_joint
    # (the bridge stabilizers are X-type in the joint-X measurement case).
    u_b_check_kind_mask = np.zeros(HX_joint.shape[0], dtype=bool)
    if n_u_b > 0:
        # The bridge rows are the last n_u_b rows (path stabs only).
        u_b_check_kind_mask[-n_u_b:] = True

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


def build_joint_measurement_code(
    data_code: CSSCode,
    op1: npt.ArrayLike,
    op2: npt.ArrayLike,
    *,
    num_layers: int = 1,
    validate: bool = True,
) -> tuple[CSSCode, JointSurgeryLayout]:
    """Construct a merged stabilizer code measuring op1 · op2 by lattice surgery.

    Implements the same-block joint X̄X̄' (or Z̄Z̄') measurement: builds two
    single-operator gadgets via build_layered_surgery_code, connects them
    with a Cross §3.6-style bridge (path graph of w = min(wt(L_1), wt(L_2))
    data qubits + w-1 X-type path stabilizers + χ endpoint extension), and
    stitches the result into a joint CSSCode of dimension k_data - 1
    (per Cross §3.6: a joint X̄_1 X̄_2 measurement consumes one logical
    degree of freedom).

    Args:
        data_code: stabilizer CSSCode with dimension >= 2.
        op1, op2: same-Pauli-type logical operator support vectors,
            length data_code.num_qubits each.
        num_layers: layer count for each component gadget.
        validate: if True, run all validation checks.

    Returns:
        (merged_code, joint_layout).

    Raises:
        ValueError: per spec §5 v2 validation rules.
    """
    op1_arr = np.asarray(op1).astype(np.int_)
    op2_arr = np.asarray(op2).astype(np.int_)

    if validate:
        pauli_type = _validate_joint_logical_ops(data_code, op1_arr, op2_arr)
    else:
        pauli_type = Pauli.X

    if pauli_type == Pauli.X:
        target_code = data_code
    else:
        # For Z-type joint, work on the ZX-dual.
        target_code = CSSCode(
            data_code.matrix_z, data_code.matrix_x, is_subsystem_code=False
        )

    merged1, layout1 = build_layered_surgery_code(
        target_code, op1_arr, num_layers=num_layers, validate_logical_op=validate
    )
    merged2, layout2 = build_layered_surgery_code(
        target_code, op2_arr, num_layers=num_layers, validate_logical_op=validate
    )

    bridge = _build_bridge_via_skiptree(layout1, layout2)
    joint_merged, joint_layout = _stitch_gadgets_with_bridge(
        target_code, merged1, layout1, merged2, layout2, bridge, pauli_type=pauli_type,
    )
    return joint_merged, joint_layout


# ---------------------------------------------------------------------------
# v3 SkipTree bridge construction helpers (Tasks 8+ of v3 plan).
# These are stateless helpers used by the v3 build_joint_measurement_code
# implementation introduced in Task 13.
# ---------------------------------------------------------------------------


def _build_auxiliary_graph(
    F: np.ndarray | galois.FieldArray,
) -> tuple[nx.Graph, dict[int, tuple[int, int]]]:
    """Build aux graph G_s from Webster F matrix.

    Vertices = V_0_s (columns of F).
    Edges = rows of F with weight exactly 2 (one per kappa_s ancilla qubit).
    Returns G and a dict mapping kappa_s qubit index -> sorted (u, v) vertex pair.
    """
    F_arr = np.asarray(F).astype(int)
    n_V = F_arr.shape[1]
    G = nx.Graph()
    G.add_nodes_from(range(n_V))
    edge_qubit_to_vertices: dict[int, tuple[int, int]] = {}
    for i, row in enumerate(F_arr):
        eps = sorted(np.flatnonzero(row).tolist())
        if len(eps) == 2:
            u, v = eps[0], eps[1]
            edge_qubit_to_vertices[i] = (u, v)
            if not G.has_edge(u, v):
                G.add_edge(u, v)
    return G, edge_qubit_to_vertices


def _label_inverse(P: np.ndarray) -> list[int]:
    """Return list ``inv[l] = vertex v`` such that P[v, l] = 1.

    P is a permutation matrix with exactly one 1 per row and per column.
    """
    n = P.shape[0]
    inv = [-1] * n
    for v in range(n):
        for l in range(n):
            if P[v, l] == 1:
                inv[l] = v
                break
    return inv


def canonical_HR(w: int) -> np.ndarray:
    """Canonical (w-1) x w parity-check matrix of the length-w repetition code.

    Row l: 1 at columns l and l+1, 0 elsewhere.
    """
    H = np.zeros((w - 1, w), dtype=np.int_)
    for l in range(w - 1):
        H[l, l] = 1
        H[l, l + 1] = 1
    return H


def _running_xor_b_c(T_col: np.ndarray) -> np.ndarray:
    """Compute b in F_2^w from T_col in F_2^{w-1} via running XOR.

    Solves H_R @ b = T_col with the canonical choice b[0] = 0, where
    H_R = canonical_HR(w) is the (w-1) x w open-path repetition parity
    check. Row l of H_R yields b[l] + b[l+1] = T_col[l]; the running
    XOR b[l] = b[l-1] XOR T_col[l-1] (with b[0] = 0) solves this.
    """
    w_minus_1 = T_col.shape[0]
    w = w_minus_1 + 1
    b = np.zeros(w, dtype=np.int_)
    for l in range(1, w):
        b[l] = (b[l - 1] + int(T_col[l - 1])) % 2
    return b
