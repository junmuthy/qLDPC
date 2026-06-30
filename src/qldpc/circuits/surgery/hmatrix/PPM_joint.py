"""Standalone bridge adapter for two-PPM joint surgery
(Swaroop et al. arXiv:2410.03628 §IV / §VII).

Handles both intra-code (g1.code is g2.code) and inter-code joints.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli, PauliXZ

from .PPM_joint_cellulation import (
    _build_aux_graph_strict,
    _canonical_H_R,
    _cellulate_port_subgraph,
    _connect_induced_subgraph,
    _edges_to_incidence_extra,
    _run_skiptree_on_port_subgraph,
)
from .PPM_XZ import GadgetLayout


@dataclasses.dataclass(frozen=True, eq=False)
class Bridge:
    """Universal adapter between two GadgetLayouts.

    Implements the universal adapter construction of Swaroop et al.
    (Swaroop, Jochym-O'Connor, Yoder) arXiv:2410.03628 §III.

    For each side s ∈ {l, r}, the SkipTree (Swaroop et al. arXiv:2410.03628 §III)
    produces a matrix T_s and a permutation P_{σ_s} acting on the port 𝒫_s.
    The port-label block π_{𝒫_s}^T P_{σ_s} selects and reorders the port columns
    of the augmented incidence matrix ∂_1^{s,aug}, satisfying the SkipTree identity:

        T_s (∂_1^{s,aug})^T π_{𝒫_s}^T = H_R P_{σ_s}^T

    where H_R is the canonical repetition-code parity-check matrix stored in
    ``Bridge.H_R``.  This identity guarantees CSS commutation between the adapter
    ancillas and the merged-code ∂_0 generators.

    Cheeger-distance preservation (same-basis) is guaranteed by
    Cross et al. arXiv:2407.18393 Thm 6.  Port-subgraph cellulation (to cap
    basis cycle length) follows Williamson & Yoder arXiv:2410.02213.

    Same-basis fields (``width``, ``port_l/r``, ``label_l/r``, ``T_l/r``,
    ``H_R``, ``g_l/r_aug``) are populated for both same-basis and mixed-basis
    bridges.  Mixed-basis fields (``Y_stab``, ``merge_qubits``, ``obs0_xor_map``,
    ``x_leftover_indices``, ``z_leftover_indices``) implement the
    Webster–Smith–Cohen arXiv:2511.15989 §II.B.2 cross-merge for joint
    Pauli-product measurement of different-basis logicals (e.g. Z̄_l ⊗ X̄_r).
    They default to None / () for same-basis bridges and are populated only by
    the mixed-basis dispatch path in ``_stitch_to_joint_code``.
    """

    width: int  # w = |𝒜| (adapter qubits)
    basis_l: PauliXZ  # X or Z on the left gadget (Webster–Smith–Cohen mixed-basis)
    basis_r: PauliXZ  # X or Z on the right gadget
    port_l: tuple[int, ...]  # 𝒫_l* ⊆ V_0^(l), length w
    port_r: tuple[int, ...]  # 𝒫_r* ⊆ V_0^(r), length w
    label_l: tuple[int, ...]  # label_l[i] = SkipTree label of V_0^(l)[i]; -1 if i ∉ 𝒫_l*
    label_r: tuple[int, ...]
    extra_ancilla_l: np.ndarray  # (e_l, |support^(l)|) F_2; weight-2 rows added
    extra_ancilla_r: np.ndarray
    T_l: np.ndarray  # (w-1, |C_0^(l)| + e_l) F_2 (3,2)-sparse
    T_r: np.ndarray
    H_R: np.ndarray  # (w-1, w) canonical rep code parity
    g_l_aug: GadgetLayout  # gadget rebuilt over F_aug^(l)
    g_r_aug: GadgetLayout
    # Mixed-basis fields (Webster–Smith–Cohen arXiv:2511.15989 §II.B.2).
    # None / empty for same-basis bridges.
    Y_stab: np.ndarray | None = None  # (n_Y, 2*n_merged) symplectic Y-rows
    merge_qubits: tuple[int, ...] = ()  # bridge qubit indices touched by cross-merge
    obs0_xor_map: tuple[int, ...] = ()  # Y_stab row indices XORed into obs0
    x_leftover_indices: tuple[int, ...] = ()  # X-cycle row indices not cross-merged
    z_leftover_indices: tuple[int, ...] = ()  # Z-cycle row indices not cross-merged

    @property
    def basis(self) -> PauliXZ:
        """Backward-compat single-basis accessor.

        Returns the shared basis when basis_l == basis_r. Raises AttributeError
        for mixed-basis bridges — callers must explicitly use basis_l / basis_r.
        """
        if self.basis_l is not self.basis_r:
            raise AttributeError(
                "mixed-basis Bridge has no single .basis attribute; "
                f"use bridge.basis_l ({self.basis_l!r}) / bridge.basis_r ({self.basis_r!r})"
            )
        return self.basis_l


def _max_basis_stabilizer_weight(code, basis: PauliXZ) -> int:
    """Max row weight of the underlying code's stabilizers on the gadget's basis side."""
    from qldpc.objects import Pauli

    H = code.matrix_z if basis is Pauli.Z else code.matrix_x
    return int(np.asarray(H).astype(int).sum(axis=1).max())


def build_bridge(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    *,
    port_subset_l: tuple[int, ...] | None = None,
    port_subset_r: tuple[int, ...] | None = None,
    spanning_tree_root_l: int = 0,
    spanning_tree_root_r: int = 0,
    cellulate_max_len: int | None = None,
) -> Bridge:
    """Build a universal-adapter bridge between two gadgets.

    Implements the construction of Swaroop et al. (Swaroop, Jochym-O'Connor,
    Yoder) arXiv:2410.03628 §III.  For each side s ∈ {l, r}:

      1. Build the auxiliary graph G_aux^(s) from weight-2 rows of F^(s)
         (the gadget incidence matrix, notation from arXiv:2410.03628 §III).
      2. Augment G_aux^(s) so that the induced subgraph on port 𝒫_s is
         connected (connectivity edges → extra_ancilla_s).
      3. Cellulate the port subgraph to cap basis cycle length, following
         Williamson & Yoder arXiv:2410.02213.
         Cheeger-distance preservation is guaranteed by Cross et al.
         arXiv:2407.18393 Thm 6.
      4. Run SkipTree (Swaroop et al. arXiv:2410.03628 §III) on the spanning
         tree of the port subgraph, yielding T_s and permutation P_{σ_s}.
         The port-label block π_{𝒫_s}^T P_{σ_s} selects and reorders port
         columns of ∂_1^{s,aug}, satisfying the SkipTree identity:

             T_s (∂_1^{s,aug})^T π_{𝒫_s}^T = H_R P_{σ_s}^T

      5. Rebuild augmented gadgets g_l_aug / g_r_aug from the combined extras.

    ``spanning_tree_root_s`` is the index INTO the port tuple of the SkipTree
    root vertex on side s.

    When ``g_l.basis == g_r.basis``, returns a same-basis CSS bridge.
    When ``g_l.basis != g_r.basis``, returns a mixed-basis bridge (basis_l ≠
    basis_r) with mixed-basis fields (Y_stab, obs0_xor_map, ...) left
    UNPOPULATED — the Webster–Smith–Cohen (arXiv:2511.15989 §II.B.2)
    cross-merge populates them during ``_stitch_to_joint_code``.

    ``cellulate_max_len`` caps port-subgraph basis cycle length.  When ``None``
    (default), it is set to the maximum basis-side stabilizer row weight across
    both sides.
    """
    basis_l = g_l.basis
    basis_r = g_r.basis
    # Cellulation cap defaults to the worse-case basis-stabilizer weight across
    # both sides, irrespective of mixed-basis. The cap shapes the rep-code cycle
    # length in the merged code and is basis-agnostic for the structural distance
    # argument (Swaroop Theorem 12). Use each side's native basis.
    if cellulate_max_len is None:
        cellulate_max_len = max(
            _max_basis_stabilizer_weight(g_l.code, basis_l),
            _max_basis_stabilizer_weight(g_r.code, basis_r),
        )

    # Step 1: auxiliary graphs
    G_l_aux, _ = _build_aux_graph_strict(g_l.incidence)
    G_r_aux, _ = _build_aux_graph_strict(g_r.incidence)

    # Step 2: port subsets + width
    port_l_all = (
        tuple(port_subset_l) if port_subset_l is not None else tuple(range(len(g_l.support)))
    )
    port_r_all = (
        tuple(port_subset_r) if port_subset_r is not None else tuple(range(len(g_r.support)))
    )
    width = min(len(port_l_all), len(port_r_all))
    if width < 2:
        raise ValueError(f"bridge width must be >= 2, got {width}")
    port_l = port_l_all[:width]
    port_r = port_r_all[:width]
    if not (0 <= spanning_tree_root_l < width):
        raise ValueError(f"spanning_tree_root_l={spanning_tree_root_l} out of [0, {width})")
    if not (0 <= spanning_tree_root_r < width):
        raise ValueError(f"spanning_tree_root_r={spanning_tree_root_r} out of [0, {width})")

    # Step 3: induced-subgraph connectivity augmentation
    extras_l_conn = _connect_induced_subgraph(G_l_aux, port_l)
    extras_r_conn = _connect_induced_subgraph(G_r_aux, port_r)

    # Step 4: cellulation
    extras_l_cell = _cellulate_port_subgraph(G_l_aux, port_l, max_len=cellulate_max_len)
    extras_r_cell = _cellulate_port_subgraph(G_r_aux, port_r, max_len=cellulate_max_len)

    extras_l_edges = extras_l_conn + extras_l_cell
    extras_r_edges = extras_r_conn + extras_r_cell
    extra_ancilla_l = _edges_to_incidence_extra(extras_l_edges, len(g_l.support))
    extra_ancilla_r = _edges_to_incidence_extra(extras_r_edges, len(g_r.support))

    from qldpc.objects import Pauli

    from .PPM_XZ import _restrict, build_gadget_augmented

    # boost_gadget appends weight-2 κ' rows to g_l.incidence beyond the original
    # _restrict incidence (∂_1^T). These rows must be preserved when assembling
    # g_l_aug — SkipTree runs against G_aux (built from boosted g_l.incidence)
    # but T_full is embedded into g_l_aug.incidence; dropping boost rows leaves
    # tree edges through boost-κ' silently zeroed and breaks the invariant
    # T_s · F_aug · P_s = H_R.
    _Hc_l = g_l.code.matrix_z if basis_l is Pauli.X else g_l.code.matrix_x
    _Hc_r = g_r.code.matrix_z if basis_r is Pauli.X else g_r.code.matrix_x
    _orig_inc_l = _restrict(_Hc_l, g_l.x)[2]
    _orig_inc_r = _restrict(_Hc_r, g_r.x)[2]
    boost_extras_l = g_l.incidence[_orig_inc_l.shape[0] :].astype(np.uint8)
    boost_extras_r = g_r.incidence[_orig_inc_r.shape[0] :].astype(np.uint8)
    combined_extras_l = np.vstack([boost_extras_l, extra_ancilla_l.astype(np.uint8)])
    combined_extras_r = np.vstack([boost_extras_r, extra_ancilla_r.astype(np.uint8)])

    g_l_aug = build_gadget_augmented(g_l.code, g_l.x, combined_extras_l, basis=basis_l)
    g_r_aug = build_gadget_augmented(g_r.code, g_r.x, combined_extras_r, basis=basis_r)

    # Step 5: SkipTree on induced port subgraph; embed back into full F_aug rows
    T_l, label_l = _run_skiptree_on_port_subgraph(
        G_l_aux,
        port_l,
        spanning_tree_root_l,
        g_l_aug.incidence,
    )
    T_r, label_r = _run_skiptree_on_port_subgraph(
        G_r_aux,
        port_r,
        spanning_tree_root_r,
        g_r_aug.incidence,
    )

    return Bridge(
        width=width,
        basis_l=basis_l,
        basis_r=basis_r,
        port_l=port_l,
        port_r=port_r,
        label_l=tuple(label_l),
        label_r=tuple(label_r),
        extra_ancilla_l=extra_ancilla_l.astype(np.uint8),
        extra_ancilla_r=extra_ancilla_r.astype(np.uint8),
        T_l=T_l,
        T_r=T_r,
        H_R=_canonical_H_R(width).astype(np.int_),
        g_l_aug=g_l_aug,
        g_r_aug=g_r_aug,
        # Mixed-basis fields: populated by _stitch_to_joint_code when basis_l != basis_r.
        # Left as defaults (None / ()) here for both same-basis and mixed-basis bridges.
    )


def _select_meas_comp(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int, int]:
    """Basis-dispatched (M_meas, M_comp) sources + data-row counts.

    M_meas holds the measured-basis merged check rows, M_comp the complementary;
    for basis=X these are (HX_merged, HZ_merged), swapped for basis=Z. Mirrors the
    abstraction of the prior ``_stitch_*`` helpers (Swaroop et al. arXiv:2410.03628 §III).
    """
    g_l_aug, g_r_aug = bridge.g_l_aug, bridge.g_r_aug
    if bridge.basis is Pauli.X:
        M_meas_l_src, M_comp_l_src = g_l_aug.HX_merged, g_l_aug.HZ_merged
        M_meas_r_src, M_comp_r_src = g_r_aug.HX_merged, g_r_aug.HZ_merged
        m_meas_l_data = g_l.code.matrix_x.shape[0]
        m_meas_r_data = g_r.code.matrix_x.shape[0]
        m_comp_l_data = g_l.code.matrix_z.shape[0]
        m_comp_r_data = g_r.code.matrix_z.shape[0]
    else:
        M_meas_l_src, M_comp_l_src = g_l_aug.HZ_merged, g_l_aug.HX_merged
        M_meas_r_src, M_comp_r_src = g_r_aug.HZ_merged, g_r_aug.HX_merged
        m_meas_l_data = g_l.code.matrix_z.shape[0]
        m_meas_r_data = g_r.code.matrix_z.shape[0]
        m_comp_l_data = g_l.code.matrix_x.shape[0]
        m_comp_r_data = g_r.code.matrix_x.shape[0]
    return (
        np.asarray(M_meas_l_src).astype(np.int_),
        np.asarray(M_meas_r_src).astype(np.int_),
        np.asarray(M_comp_l_src).astype(np.int_),
        np.asarray(M_comp_r_src).astype(np.int_),
        m_meas_l_data,
        m_meas_r_data,
        m_comp_l_data,
        m_comp_r_data,
    )


def _port_label_block(label: tuple[int, ...], width: int) -> np.ndarray:
    """Π_s = π_{𝒫_s}^T P_{σ_s} ∈ F_2^{|label|×w}: row i has a 1 at column
    ``label[i]`` when ``label[i] >= 0`` (port vertex), else all-zero."""
    blk = np.zeros((len(label), width), dtype=np.int_)
    for v_idx, lab in enumerate(label):
        if lab >= 0:
            blk[v_idx, lab] = 1
    return blk


def _joint_merged_intercode(g_l: GadgetLayout, g_r: GadgetLayout, bridge: Bridge) -> CSSCode:
    """Inter-code joint merge (g_l.code is not g_r.code), closed form of
    Swaroop et al. arXiv:2410.03628 §III Eq. (H̃_X/H̃_Z^joint). Columns
    (Q_l | Q_r | Q'_l | Q'_r | 𝒜) with separate data blocks Q_l, Q_r."""
    assert g_l.code is not g_r.code
    field = g_l.code.field
    (M_meas_l, M_meas_r, M_comp_l, M_comp_r,
     m_meas_l, m_meas_r, m_comp_l, m_comp_r) = _select_meas_comp(g_l, g_r, bridge)

    n_l, n_r = g_l.code.num_qudits, g_r.code.num_qudits
    k_l = bridge.g_l_aug.incidence.shape[0]
    k_r = bridge.g_r_aug.incidence.shape[0]
    w = bridge.width

    HX_l = M_meas_l[:m_meas_l, :n_l]
    HX_r = M_meas_r[:m_meas_r, :n_r]
    f1T_l = M_meas_l[m_meas_l:, :n_l]
    d1_l = M_meas_l[m_meas_l:, n_l:]
    f1T_r = M_meas_r[m_meas_r:, :n_r]
    d1_r = M_meas_r[m_meas_r:, n_r:]

    HZ_l = M_comp_l[:m_comp_l, :n_l]
    f0_l = M_comp_l[:m_comp_l, n_l:]
    d0_l = M_comp_l[m_comp_l:, n_l:]
    HZ_r = M_comp_r[:m_comp_r, :n_r]
    f0_r = M_comp_r[:m_comp_r, n_r:]
    d0_r = M_comp_r[m_comp_r:, n_r:]

    Pi_l = _port_label_block(bridge.label_l, w)
    Pi_r = _port_label_block(bridge.label_r, w)
    T_l = np.asarray(bridge.T_l).astype(np.int_)
    T_r = np.asarray(bridge.T_r).astype(np.int_)
    H_R = np.asarray(bridge.H_R).astype(np.int_)
    sup_l, sup_r = len(bridge.label_l), len(bridge.label_r)
    r_l, r_r = d0_l.shape[0], d0_r.shape[0]

    def Z(rows: int, cols: int) -> np.ndarray:
        return np.zeros((rows, cols), dtype=np.int_)

    M_meas = np.block([
        [HX_l,         Z(m_meas_l, n_r), Z(m_meas_l, k_l), Z(m_meas_l, k_r), Z(m_meas_l, w)],
        [Z(m_meas_r, n_l), HX_r,         Z(m_meas_r, k_l), Z(m_meas_r, k_r), Z(m_meas_r, w)],
        [f1T_l,        Z(sup_l, n_r),    d1_l,             Z(sup_l, k_r),    Pi_l],
        [Z(sup_r, n_l), f1T_r,           Z(sup_r, k_l),    d1_r,             Pi_r],
    ]).astype(np.int_)

    M_comp = np.block([
        [HZ_l,         Z(m_comp_l, n_r), f0_l,             Z(m_comp_l, k_r), Z(m_comp_l, w)],
        [Z(m_comp_r, n_l), HZ_r,         Z(m_comp_r, k_l), f0_r,             Z(m_comp_r, w)],
        [Z(r_l, n_l),  Z(r_l, n_r),      d0_l,             Z(r_l, k_r),      Z(r_l, w)],
        [Z(r_r, n_l),  Z(r_r, n_r),      Z(r_r, k_l),      d0_r,             Z(r_r, w)],
        [Z(w - 1, n_l), Z(w - 1, n_r),   T_l,              T_r,              H_R],
    ]).astype(np.int_)

    if bridge.basis is Pauli.X:
        return CSSCode(field(M_meas), field(M_comp), is_subsystem_code=False)
    return CSSCode(field(M_comp), field(M_meas), is_subsystem_code=False)


def _joint_merged_intracode(g_l: GadgetLayout, g_r: GadgetLayout, bridge: Bridge) -> CSSCode:
    """Intra-code joint merge (g_l.code is g_r.code), closed form of Swaroop et al.
    arXiv:2410.03628 §III. Columns (Q | Q'_l | Q'_r | 𝒜): the two sides SHARE the
    single data column block Q and the single H_X/H_Z data-row block (written once)."""
    assert g_l.code is g_r.code
    field = g_l.code.field
    (M_meas_l, M_meas_r, M_comp_l, M_comp_r,
     m_meas_data, _mr, m_comp_data, _cr) = _select_meas_comp(g_l, g_r, bridge)

    n = g_l.code.num_qudits
    k_l = bridge.g_l_aug.incidence.shape[0]
    k_r = bridge.g_r_aug.incidence.shape[0]
    w = bridge.width

    HX = M_meas_l[:m_meas_data, :n]
    f1T_l = M_meas_l[m_meas_data:, :n]
    d1_l = M_meas_l[m_meas_data:, n:]
    f1T_r = M_meas_r[m_meas_data:, :n]
    d1_r = M_meas_r[m_meas_data:, n:]

    HZ = M_comp_l[:m_comp_data, :n]
    f0_l = M_comp_l[:m_comp_data, n:]
    f0_r = M_comp_r[:m_comp_data, n:]
    d0_l = M_comp_l[m_comp_data:, n:]
    d0_r = M_comp_r[m_comp_data:, n:]

    Pi_l = _port_label_block(bridge.label_l, w)
    Pi_r = _port_label_block(bridge.label_r, w)
    T_l = np.asarray(bridge.T_l).astype(np.int_)
    T_r = np.asarray(bridge.T_r).astype(np.int_)
    H_R = np.asarray(bridge.H_R).astype(np.int_)
    sup_l, sup_r = len(bridge.label_l), len(bridge.label_r)
    r_l, r_r = d0_l.shape[0], d0_r.shape[0]

    def Z(rows: int, cols: int) -> np.ndarray:
        return np.zeros((rows, cols), dtype=np.int_)

    M_meas = np.block([
        [HX,           Z(m_meas_data, k_l), Z(m_meas_data, k_r), Z(m_meas_data, w)],
        [f1T_l,        d1_l,                Z(sup_l, k_r),       Pi_l],
        [f1T_r,        Z(sup_r, k_l),       d1_r,                Pi_r],
    ]).astype(np.int_)

    M_comp = np.block([
        [HZ,           f0_l,                f0_r,                Z(m_comp_data, w)],
        [Z(r_l, n),    d0_l,                Z(r_l, k_r),         Z(r_l, w)],
        [Z(r_r, n),    Z(r_r, k_l),         d0_r,                Z(r_r, w)],
        [Z(w - 1, n),  T_l,                 T_r,                 H_R],
    ]).astype(np.int_)

    if bridge.basis is Pauli.X:
        return CSSCode(field(M_meas), field(M_comp), is_subsystem_code=False)
    return CSSCode(field(M_comp), field(M_meas), is_subsystem_code=False)


def _joint_merged_dispatch(g_l: GadgetLayout, g_r: GadgetLayout, bridge: Bridge) -> CSSCode:
    """Assemble the merged joint CSSCode; intra (shared data) vs inter dispatch."""
    if g_l.code is g_r.code:
        return _joint_merged_intracode(g_l, g_r, bridge)
    return _joint_merged_intercode(g_l, g_r, bridge)
