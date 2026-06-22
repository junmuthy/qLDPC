"""Logical-Y measurement gadget primitives (Ȳ = iX̄Z̄).

Building blocks for measuring a logical Y operator via lattice surgery in the
clean single-overlap regime of Cross, He, Rall, Yoder arXiv:2407.18393 §3.7,
where the logical-X support and the logical-Z support of the measured qubit
intersect on exactly one data qubit. That single shared qubit is where the
X̄ and Z̄ strings cross to realise Ȳ = iX̄Z̄.

This module provides:
    _locate_overlap  — validate an (x, z) representative pair and return the
                       index of their single shared (crossing) data qubit.
    _steane_y_pair   — a concrete Steane-code fixture (code, x, z) whose
                       logical-X / logical-Z supports overlap on exactly one
                       qubit, i.e. the §3.7 single-overlap case.
"""

from __future__ import annotations

import dataclasses
import itertools

import galois
import numpy as np

from qldpc import codes
from qldpc.codes.common import CSSCode, QuditCode
from qldpc.objects import Pauli

from .bridge import Bridge, build_bridge
from .gadget import GadgetLayout, build_gadget
from .merge import apply_mixed_basis_merge

GF2 = galois.GF(2)


def _locate_overlap(code: CSSCode, x: np.ndarray, z: np.ndarray) -> int:
    """Return the single data qubit shared by logical-X support ``x`` and logical-Z support ``z``.

    Implements the single-overlap precondition of Cross, He, Rall, Yoder
    arXiv:2407.18393 §3.7: Ȳ = iX̄Z̄ is realised cleanly when the X̄ and Z̄
    strings cross on exactly one data qubit.

    Validates, raising ``ValueError`` otherwise:
        * ``x`` is a logical-X representative: ``H_Z @ x == 0`` (mod 2),
        * ``z`` is a logical-Z representative: ``H_X @ z == 0`` (mod 2),
        * ``x`` and ``z`` anticommute: ``x · z`` has odd weight (mod 2),
        * their supports overlap on exactly one qubit: ``|supp(x) ∩ supp(z)| == 1``.

    Returns the index of that shared qubit.
    """
    x = np.asarray(x).astype(np.uint8)
    z = np.asarray(z).astype(np.uint8)
    n = code.num_qudits
    if x.shape != (n,):
        raise ValueError(f"x has shape {x.shape}, expected ({n},)")
    if z.shape != (n,):
        raise ValueError(f"z has shape {z.shape}, expected ({n},)")

    matrix_x = np.asarray(code.matrix_x).astype(np.uint8)  # H_X (X-type stabilizers)
    matrix_z = np.asarray(code.matrix_z).astype(np.uint8)  # H_Z (Z-type stabilizers)

    # x must commute with every Z-type stabilizer (i.e. be a logical-X representative).
    if ((matrix_z @ x) % 2 != 0).any():
        raise ValueError("x is not a logical-X representative: H_Z @ x != 0 (mod 2)")
    # z must commute with every X-type stabilizer (i.e. be a logical-Z representative).
    if ((matrix_x @ z) % 2 != 0).any():
        raise ValueError("z is not a logical-Z representative: H_X @ z != 0 (mod 2)")

    # x and z must anticommute: a logical-X and logical-Z of the SAME logical qubit
    # cross an odd number of times. A commuting pair (even overlap parity) cannot
    # form Ȳ = iX̄Z̄.
    if int(np.dot(x.astype(np.int64), z.astype(np.int64))) % 2 == 0:
        raise ValueError("x and z commute (x · z is even); they cannot form Ȳ = iX̄Z̄")

    overlap = np.where((x.astype(bool)) & (z.astype(bool)))[0]
    if overlap.size != 1:
        raise ValueError(
            f"x and z overlap on {overlap.size} qubits, expected exactly 1 "
            "(Cross, He, Rall, Yoder arXiv:2407.18393 §3.7 single-overlap case)"
        )
    return int(overlap[0])


def _steane_y_pair() -> tuple[CSSCode, np.ndarray, np.ndarray]:
    """Return a Steane code and a logical (x, z) pair overlapping on exactly one qubit.

    This is the clean single-overlap fixture for Ȳ = iX̄Z̄ of Cross, He, Rall,
    Yoder arXiv:2407.18393 §3.7. The canonical weight-3 logical-X and logical-Z
    representatives of the [[7,1,3]] Steane code already cross on a single data
    qubit; if a given representative pair does not, it is reduced over GF(2) by
    adding stabilizer rows (a logical-X support stays logical-X after XOR-ing any
    row of ``matrix_x``; a logical-Z support stays logical-Z after XOR-ing any row
    of ``matrix_z``) until the supports overlap on exactly one qubit.
    """
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)

    if _overlap_size(x, z) == 1:
        return code, x, z

    # Fallback: search small GF(2) combinations of stabilizer rows added to x and/or
    # z. Adding an X-stabilizer row to x leaves it a logical-X representative;
    # adding a Z-stabilizer row to z leaves it a logical-Z representative.
    rows_x = GF2(np.asarray(code.matrix_x).astype(np.uint8))
    rows_z = GF2(np.asarray(code.matrix_z).astype(np.uint8))
    x_gf = GF2(x)
    z_gf = GF2(z)
    n_rx = rows_x.shape[0]
    n_rz = rows_z.shape[0]

    for k_x in range(n_rx + 1):
        for cols_x in itertools.combinations(range(n_rx), k_x):
            x_cand = x_gf + (sum((rows_x[i] for i in cols_x), GF2.Zeros(x_gf.shape)))
            for k_z in range(n_rz + 1):
                for cols_z in itertools.combinations(range(n_rz), k_z):
                    z_cand = z_gf + (sum((rows_z[j] for j in cols_z), GF2.Zeros(z_gf.shape)))
                    x_arr = np.asarray(x_cand).astype(np.uint8)
                    z_arr = np.asarray(z_cand).astype(np.uint8)
                    if _overlap_size(x_arr, z_arr) == 1:
                        return code, x_arr, z_arr

    raise ValueError(
        "BLOCKED: no overlap-1 logical (x, z) pair reachable on the Steane code by "
        "stabilizer-row reduction; the out-of-scope multi-overlap §3.7 path would be "
        "required (Cross, He, Rall, Yoder arXiv:2407.18393 §3.7)"
    )


def _overlap_size(x: np.ndarray, z: np.ndarray) -> int:
    """Number of data qubits in ``supp(x) ∩ supp(z)``."""
    return int(np.count_nonzero(x.astype(bool) & z.astype(bool)))


@dataclasses.dataclass(frozen=True, eq=False)
class YGadgetLayout:
    """Merged code for a single logical-Y measurement Ȳ = iX̄Z̄.

    Output of :func:`build_y_gadget`, realising the gauge-fixed Y ancilla
    system of Cross, He, Rall, Yoder arXiv:2407.18393 §3.7 in the clean
    single-overlap regime (Remark 19): the X-measurement system (the L=1
    gadget of X̄) and the Z-measurement system (the L=1 gadget of Z̄) of ONE
    logical qubit are glued at their single shared data qubit ``q0`` via an
    explicit mixed-type check ``q1``. ``q1`` is the symplectic Y row
    ``Y_stab`` obtained by the Webster, Smith, Cohen arXiv:2511.15989 §II.B.2
    cross-merge of the χ_X row and χ_Z row anchored at ``q0``.

    Fields:
        code        — the underlying CSS code (X̄ and Z̄ live on its qubits).
        x, z        — the logical-X / logical-Z supports (uint8), overlapping
                      on exactly the one data qubit ``q0``.
        q0          — index of that single shared (crossing) data qubit.
        g_x, g_z    — the per-system L=1 gadgets (Webster, Smith, Cohen
                      arXiv:2511.15989 §II.A) for X̄ (basis=X) and Z̄ (basis=Z),
                      built on the same ``code`` with disjoint κ-ancilla ranges.
        bridge      — the sparse SkipTree+cellulation adapter joining the two
                      systems (Swaroop, Jochym-O'Connor, Yoder arXiv:2410.03628
                      §III / §II C). Stored for downstream circuit synthesis;
                      the single-overlap merged code itself needs only ``q1``.
        Y_stab      — ``(n_Y, 2*n_merged)`` symplectic Y rows from the cross-
                      merge (n_Y == 1 here: the single mixed check ``q1``).
        H_sym       — ``(rows, 2*n_merged)`` merged symplectic check matrix
                      = [HX_out | 0] ∪ [0 | HZ_out] ∪ Y_stab.
        merged_code — ``QuditCode(field(H_sym))`` in which Ȳ is the measured
                      stabilizer; encodes one fewer logical than ``code``.
        obs0_xor_map— Y_stab row indices XORed into obs0 (per Webster, Smith,
                      Cohen arXiv:2511.15989 Lemma 2): the Ȳ eigenvalue rides
                      on these rows because Steane is degenerate and no surviving
                      χ rows carry it.

    Column layout of the merged symplectic space (both X and Z halves):
    ``[data (n) | κ_x | κ_z]``. The X̄ system contributes its X-checks and χ_X
    rows on ``[data | κ_x]``; the Z̄ system its Z-checks and χ_Z rows on
    ``[data | κ_z]``. The single overlap means only ``q1`` is non-CSS, so no
    bridge-adapter columns are required (Remark 19 single-overlap case).
    """

    code: CSSCode
    x: np.ndarray
    z: np.ndarray
    q0: int
    g_x: GadgetLayout
    g_z: GadgetLayout
    bridge: Bridge
    Y_stab: np.ndarray
    H_sym: np.ndarray
    merged_code: QuditCode
    obs0_xor_map: tuple[int, ...]


def build_y_gadget(code: CSSCode, *, x: np.ndarray, z: np.ndarray) -> YGadgetLayout:
    """Assemble the §3.7 logical-Y merged code for Ȳ = iX̄Z̄ on one logical qubit.

    Implements the single-overlap (Remark 19) construction of Cross, He, Rall,
    Yoder arXiv:2407.18393 §3.7: glue the X-measurement L=1 gadget of ``x`` and
    the Z-measurement L=1 gadget of ``z`` (both Webster, Smith, Cohen
    arXiv:2511.15989 §II.A) at their single shared data qubit ``q0``, fusing the
    χ_X and χ_Z rows anchored there into one explicit mixed check ``q1`` via the
    Webster, Smith, Cohen arXiv:2511.15989 §II.B.2 cross-merge (NOT a single-
    qubit basis-change rotation — the non-CSS content lives in ``q1``).

    The merged symplectic column space is ``[data (n) | κ_x | κ_z]``. The two
    systems share the data qubits, so the original code stabilizers appear
    ONCE: H_X is extended onto κ_z so it commutes with χ_Z, and H_Z onto κ_x so
    it commutes with χ_X (the dual-extension of Webster §II.A step 3). The only
    pair that anticommutes before the merge is (χ_X, χ_Z) at ``q0`` (their data
    supports cross exactly once); fusing them removes that anticommutation, so
    the merged code is a genuine stabilizer code encoding ``code.dimension − 1``
    logicals with Ȳ = iX̄Z̄ in its stabilizer group.

    Args:
        code: a CSS code carrying the logical qubit to be Y-measured.
        x: logical-X support, ``H_Z @ x == 0`` (mod 2); shape ``(n,)``.
        z: logical-Z support, ``H_X @ z == 0`` (mod 2); shape ``(n,)``.

    Returns:
        A :class:`YGadgetLayout`. ``ValueError`` is raised (via
        :func:`_locate_overlap`) if ``x``/``z`` are not valid anticommuting
        logicals overlapping on exactly one qubit (multi-overlap is the
        out-of-scope Remark 19 extension).
    """
    x = np.asarray(x).astype(np.uint8)
    z = np.asarray(z).astype(np.uint8)
    # Validate the single-overlap precondition and locate the crossing qubit q0.
    q0 = _locate_overlap(code, x, z)

    # Per-system L=1 gadgets on the SAME code (disjoint κ-ancilla index ranges).
    g_x = build_gadget(code, x, basis=Pauli.X)
    g_z = build_gadget(code, z, basis=Pauli.Z)

    # Sparse adapter joining the two systems (Swaroop, Jochym-O'Connor, Yoder
    # arXiv:2410.03628 §III SkipTree + §II C cellulation). Stored on the layout
    # for downstream circuit synthesis; the single-overlap merged code needs
    # only the q1 mixed check, so the bridge columns do not enter H_sym.
    bridge = build_bridge(g_x, g_z)

    field = code.field
    n = code.num_qudits
    m_x = int(np.asarray(code.matrix_x).shape[0])
    m_z = int(np.asarray(code.matrix_z).shape[0])
    k_x = len(g_x.ancilla_qubits)
    k_z = len(g_z.ancilla_qubits)
    n_merged = n + k_x + k_z

    # --- Decompose each gadget into stabilizer / χ / gauge blocks --------------
    # X̄ system (Webster §II.A step 3, basis=X):
    #   HX_merged = [ [H_X | 0]        (X-checks, κ_x part zero)
    #                 [E_x | F_x^T] ]  (χ_X rows: one data vertex + κ_x columns)
    #   HZ_merged = [ [H_Z | F̃_x]      (Z-checks dual-extended onto κ_x)
    #                 [0   | G_x] ]     (gauge rows on κ_x)
    chi_x = np.asarray(g_x.HX_merged[m_x:]).astype(np.uint8)  # (|V0x|, n+k_x)
    hz_ext_kx = np.asarray(g_x.HZ_merged[:m_z]).astype(np.uint8)  # [H_Z | F̃_x]
    gauge_x = np.asarray(g_x.HZ_merged[m_z:, n:]).astype(np.uint8)  # (r_x, k_x)

    # Z̄ system (basis=Z, symmetric dual):
    #   HZ_merged = [ [H_Z | 0] ; [E_z | F_z^T] ]  (Z-checks; χ_Z rows on κ_z)
    #   HX_merged = [ [H_X | F̃_z] ; [0 | G_z] ]   (X-checks dual-extended; gauge on κ_z)
    chi_z = np.asarray(g_z.HZ_merged[m_z:]).astype(np.uint8)  # (|V0z|, n+k_z)
    hx_ext_kz = np.asarray(g_z.HX_merged[:m_x]).astype(np.uint8)  # [H_X | F̃_z]
    gauge_z = np.asarray(g_z.HX_merged[m_x:, n:]).astype(np.uint8)  # (r_z, k_z)

    def _embed(
        data_kx_block: np.ndarray | None = None,
        *,
        kx_cols: np.ndarray | None = None,
        kz_cols: np.ndarray | None = None,
        data_cols: np.ndarray | None = None,
        rows: int = 0,
    ) -> np.ndarray:
        """Place a block into the merged ``[data | κ_x | κ_z]`` column space."""
        out = np.zeros((rows, n_merged), dtype=np.uint8)
        if data_cols is not None:
            out[:, :n] = data_cols
        if kx_cols is not None:
            out[:, n : n + k_x] = kx_cols
        if kz_cols is not None:
            out[:, n + k_x : n + k_x + k_z] = kz_cols
        return out

    # --- X-type rows of the merged code ---------------------------------------
    # H_X extended onto κ_z (commutes with χ_Z); χ_X on [data | κ_x]; gauge G_z on κ_z.
    HX_orig = _embed(data_cols=hx_ext_kz[:, :n], kz_cols=hx_ext_kz[:, n:], rows=m_x)
    chi_x_emb = _embed(data_cols=chi_x[:, :n], kx_cols=chi_x[:, n:], rows=chi_x.shape[0])
    gauge_z_emb = _embed(kz_cols=gauge_z, rows=gauge_z.shape[0])
    HX_all = np.vstack([HX_orig, chi_x_emb, gauge_z_emb]).astype(np.uint8)

    # --- Z-type rows of the merged code ---------------------------------------
    # H_Z extended onto κ_x (commutes with χ_X); χ_Z on [data | κ_z]; gauge G_x on κ_x.
    HZ_orig = _embed(data_cols=hz_ext_kx[:, :n], kx_cols=hz_ext_kx[:, n:], rows=m_z)
    chi_z_emb = _embed(data_cols=chi_z[:, :n], kz_cols=chi_z[:, n:], rows=chi_z.shape[0])
    gauge_x_emb = _embed(kx_cols=gauge_x, rows=gauge_x.shape[0])
    HZ_all = np.vstack([HZ_orig, chi_z_emb, gauge_x_emb]).astype(np.uint8)

    # --- Cross-merge the χ_X / χ_Z rows anchored at q0 into q1 -----------------
    # adapter_cols = the DATA columns: a χ row has exactly ONE data qubit in its
    # support (its vertex), so the single-{q0} criterion selects precisely χ_X
    # and χ_Z at q0 (original stabilizers have ≥2 data qubits). The fused row IS
    # the mixed check q1 (Webster, Smith, Cohen arXiv:2511.15989 §II.B.2).
    HX_out, HZ_out, Y_stab, obs0_y, _x_left, _z_left = apply_mixed_basis_merge(
        HX_all,
        HZ_all,
        merge_qubits=(q0,),
        adapter_cols=tuple(range(n)),
    )
    HX_out = np.asarray(HX_out).astype(np.int_)
    HZ_out = np.asarray(HZ_out).astype(np.int_)
    if Y_stab is None or Y_stab.shape[0] < 1:
        raise ValueError(
            "BLOCKED: cross-merge produced no Y_stab row at q0 — the χ_X/χ_Z "
            f"pair anchored at q0={q0} was not found (Cross, He, Rall, Yoder "
            "arXiv:2407.18393 §3.7 mixed check q1 must exist)"
        )

    # --- Pack the merged symplectic matrix [HX_out|0] ∪ [0|HZ_out] ∪ Y_stab ----
    rows_sym: list[np.ndarray] = []
    for r in HX_out:
        rows_sym.append(np.concatenate([r, np.zeros(n_merged, dtype=np.int_)]))
    for r in HZ_out:
        rows_sym.append(np.concatenate([np.zeros(n_merged, dtype=np.int_), r]))
    for r in Y_stab:
        rows_sym.append(r.astype(np.int_))
    H_sym = (
        np.array(rows_sym, dtype=np.int_)
        if rows_sym
        else np.zeros((0, 2 * n_merged), dtype=np.int_)
    )

    # Genuine stabilizer code in the single-overlap case: the only pre-merge
    # anticommutation (χ_X, χ_Z) at q0 is consumed by q1. Detect any residual
    # gauge anticommutation defensively (would arise only for multi-overlap).
    Hx = H_sym[:, :n_merged]
    Hz = H_sym[:, n_merged:]
    comm = (Hx @ Hz.T + Hz @ Hx.T) % 2
    np.fill_diagonal(comm, 0)
    is_subsystem = bool(comm.any())

    merged_code = QuditCode(field(H_sym), is_subsystem_code=is_subsystem)

    return YGadgetLayout(
        code=code,
        x=x,
        z=z,
        q0=q0,
        g_x=g_x,
        g_z=g_z,
        bridge=bridge,
        Y_stab=Y_stab,
        H_sym=H_sym,
        merged_code=merged_code,
        obs0_xor_map=tuple(obs0_y),
    )
