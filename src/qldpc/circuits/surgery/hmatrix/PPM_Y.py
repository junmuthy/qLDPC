"""Logical-Y measurement gadget primitives (Ȳ = iX̄Z̄).

Building blocks for measuring a logical Y operator via lattice surgery, following
the homological measurement construction of Ide, Gowda, Nadkarni, Dauphinais
arXiv:2410.02753 §III.C/§III.D. The
logical-Y support is the set ``W = supp(x) ∩ supp(z)`` of data qubits where the
logical-X string and the logical-Z string cross; ``|W|`` is odd (the strings
anticommute) and may be any size, so the merge handles general ``|W| ≥ 1``.

This module provides:
    _locate_overlaps — validate an (x, z) representative pair and return the
                       tuple ``W = supp(x) ∩ supp(z)`` of physical Y qubits.
"""

from __future__ import annotations

import dataclasses

import galois
import numpy as np

from qldpc.codes.common import CSSCode, QuditCode
from qldpc.objects import Pauli

from .merge import apply_mixed_basis_merge
from .PPM_XZ import GadgetLayout, build_gadget
from .PPM_Y_obs0 import Obs0ReadoutPlan, Obs0Row, _ybar_obs0_rows

GF2 = galois.GF(2)


def _in_rowspace_gf2(M: np.ndarray, v: np.ndarray) -> bool:
    """Return True iff ``v`` (1D uint8) lies in the GF(2) row space of ``M``.

    Standard GF(2) membership test: ``v ∈ rowspace(M)`` exactly when appending
    ``v`` as a new row does not increase the rank, i.e. ``rank(M) == rank([M; v])``.
    Ranks are taken over GF(2) via ``galois``'s overload of
    ``numpy.linalg.matrix_rank`` on ``galois.GF(2)`` arrays.

    Used to certify that the symplectic vector ``[x | z]`` of Ȳ = iX̄Z̄ is a
    product of the merged-code stabilizers restricted to the original data
    qubits, the core correctness guarantee of the homological logical-Y merge of
    Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.C/§III.D.
    """
    M2 = GF2(np.asarray(M).astype(np.uint8))
    A = GF2(np.vstack([np.asarray(M).astype(np.uint8), np.asarray(v).astype(np.uint8)[None, :]]))
    return int(np.linalg.matrix_rank(M2)) == int(np.linalg.matrix_rank(A))


def _locate_overlap(code: CSSCode, x: np.ndarray, z: np.ndarray) -> int:
    """Return the single data qubit shared by logical-X support ``x`` and logical-Z support ``z``.

    Legacy ``|W|=1`` helper retained as a test fixture; the general-``|W|`` merge
    uses :func:`_locate_overlaps`. The single-overlap special case of the
    Ȳ-overlap ``W = supp(x) ∩ supp(z)`` of Ide, Gowda, Nadkarni, Dauphinais
    arXiv:2410.02753 §III.D: Ȳ = iX̄Z̄ is
    realised cleanly when the X̄ and Z̄ strings cross on exactly one data qubit.

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
            "(Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.D "
            "single-overlap case)"
        )
    return int(overlap[0])


def _locate_overlaps(code: CSSCode, x: np.ndarray, z: np.ndarray) -> tuple[int, ...]:
    """Return W = supp(x) ∩ supp(z), the physical Pauli-Y qubits of Ȳ = iX̄Z̄.

    Validates (Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.D):
      * x is a logical-X representative: H_Z @ x == 0 (mod 2),
      * z is a logical-Z representative: H_X @ z == 0 (mod 2),
      * x and z anticommute: x · z is odd (so |W| is odd, ≥ 1).
    """
    x = np.asarray(x).astype(np.uint8)
    z = np.asarray(z).astype(np.uint8)
    n = code.num_qudits
    if x.shape != (n,) or z.shape != (n,):
        raise ValueError(f"x/z must have shape ({n},); got {x.shape}, {z.shape}")
    if ((np.asarray(code.matrix_z).astype(np.uint8) @ x) % 2 != 0).any():
        raise ValueError("x is not a logical-X representative: H_Z @ x != 0 (mod 2)")
    if ((np.asarray(code.matrix_x).astype(np.uint8) @ z) % 2 != 0).any():
        raise ValueError("z is not a logical-Z representative: H_X @ z != 0 (mod 2)")
    if int(np.dot(x.astype(np.int64), z.astype(np.int64))) % 2 == 0:
        raise ValueError("x and z commute (x · z even); they cannot form Ȳ = iX̄Z̄")
    return tuple(int(i) for i in np.where(x.astype(bool) & z.astype(bool))[0])


def _overlap_size(x: np.ndarray, z: np.ndarray) -> int:
    """Number of data qubits in ``supp(x) ∩ supp(z)``."""
    return int(np.count_nonzero(x.astype(bool) & z.astype(bool)))


def _merged_incidence(
    g_x: GadgetLayout, g_z: GadgetLayout, x: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """Merged graph incidence ∂_1 (Ide, Gowda, Nadkarni, Dauphinais
    arXiv:2410.02753 Eq.(66)).

    Rows = vertices V_X ⊔ W ⊔ V_Z (data qubits of supp(x)/supp(z)); columns =
    edges (κ_X | κ_Z). ∂_1^x = g_x.incidence.T (rows=support, cols=κ_X); dual for
    ∂_1^z. The W rows stack the X- and Z-system incidences side by side, gluing
    the two graphs at the shared Y-qubits.
    """
    x = np.asarray(x).astype(np.uint8)
    z = np.asarray(z).astype(np.uint8)
    d1x = np.asarray(g_x.incidence).astype(np.uint8).T  # (|supp x|, k_x)
    d1z = np.asarray(g_z.incidence).astype(np.uint8).T  # (|supp z|, k_z)
    supx = list(g_x.support)
    supz = list(g_z.support)
    W = sorted(set(int(i) for i in np.where(x)[0]) & set(int(i) for i in np.where(z)[0]))
    VX = [v for v in supx if v not in W]
    VZ = [v for v in supz if v not in W]
    k_x = d1x.shape[1]
    k_z = d1z.shape[1]

    def rows_of(d: np.ndarray, sup: list[int], sel: list[int]) -> np.ndarray:
        idx = [sup.index(v) for v in sel]
        return d[idx] if idx else np.zeros((0, d.shape[1]), dtype=np.uint8)

    top = np.hstack([rows_of(d1x, supx, VX), np.zeros((len(VX), k_z), np.uint8)])
    mid = np.hstack([rows_of(d1x, supx, W), rows_of(d1z, supz, W)])
    bot = np.hstack([np.zeros((len(VZ), k_x), np.uint8), rows_of(d1z, supz, VZ)])
    return np.vstack([top, mid, bot]).astype(np.uint8), k_x, k_z


def _partial0_symplectic_rows(
    g_x: GadgetLayout,
    g_z: GadgetLayout,
    x: np.ndarray,
    z: np.ndarray,
    *,
    n: int,
    k_x: int,
    k_z: int,
) -> np.ndarray:
    """∂_0 = ker(merged ∂_1) as symplectic rows (Ide, Gowda, Nadkarni, Dauphinais
    arXiv:2410.02753 Eq.(67)/(68)).

    Column layout per half: [data (n) | κ_X | κ_Z]. A cycle c = (c_X | c_Z):
    its κ_X support enters as Z-part on κ_X (∂_0^(X)); its κ_Z support enters as
    X-part on κ_Z (∂_0^(Z)). A crossing cycle (|W|≥2) populates both → one
    non-CSS row.
    """
    D1, kx, kz = _merged_incidence(g_x, g_z, x, z)
    if (kx, kz) != (k_x, k_z):
        raise ValueError(f"incidence κ sizes {(kx, kz)} != ({k_x}, {k_z})")
    ker = np.asarray(GF2(D1.astype(int)).null_space()).astype(np.uint8)  # rows over (κ_X|κ_Z)
    nm = n + k_x + k_z
    out = np.zeros((ker.shape[0], 2 * nm), dtype=np.uint8)
    for i, c in enumerate(ker):
        c_x = c[:k_x]
        c_z = c[k_x:]
        out[i, n + k_x : nm] = c_z  # X-part on κ_Z  (∂_0^(Z))
        out[i, nm + n : nm + n + k_x] = c_x  # Z-part on κ_X  (∂_0^(X))
    return out


@dataclasses.dataclass(frozen=True, eq=False)
class YGadgetLayout:
    """Merged code for a logical-Y measurement Ȳ = iX̄Z̄ (general ``|W|``).

    Output of :func:`build_y_gadget`, realising the homological logical-Y merge
    of Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.C/§III.D. The
    X-measurement system (the L=1 gadget
    of X̄) and the Z-measurement system (the L=1 gadget of Z̄) of ONE logical
    qubit are glued along the physical Y-support ``W = supp(x) ∩ supp(z)``: each
    ``v ∈ W`` carries a mixed-type check ``y_v`` obtained by the §III.D
    cross-merge of the S_X' row and the S_Z' row anchored at ``v``. The
    ``Y_stab`` rows are exactly these ``|W|`` symplectic Y rows.

    Per-system Cheeger boost (arXiv:2410.02753 §III.D). Each
    per-system gadget ``g_x`` / ``g_z`` is boosted (combinatorially) to boundary
    Cheeger constant ≥ 1 BEFORE merging, so the per-system distance argument
    holds. The boosted gadgets are stored, so ``k_x = len(g_x.Q_prime)``
    matches the merged-code width.

    ∂_0 block (no bridge). The merged check matrix appends ``∂_0 = ker(merged
    ∂_1)`` (arXiv:2410.02753 §III.D Eq.(66)/(67)), the cycle basis
    of the glued graph, in place of any per-system gauge blocks. At ``|W|=1`` the
    glued graph contributes no crossing cycle (0 ∂_0 rows); at ``|W|≥2`` it
    contributes ``|W|−1`` crossing rows. There is NO SkipTree adapter / bridge.

    Stabilizer code. The only pre-merge anticommutations — (S_X', S_Z') at each
    ``v ∈ W``, whose data supports cross — are consumed by the cross-merge into
    the ``y_v`` rows. The merged code is then a genuine STABILIZER code
    (``is_subsystem_code=False`` for Steane) encoding ``code.dimension − 1``
    logicals, with Ȳ = iX̄Z̄ in its stabilizer group.

    Fields:
        code        — the underlying CSS code (X̄ and Z̄ live on its qubits).
        x, z        — the logical-X / logical-Z supports (uint8); their overlap
                      ``supp(x) ∩ supp(z)`` is ``W``.
        W           — ``tuple[int, ...]`` of the data qubits in
                      ``supp(x) ∩ supp(z)`` (the physical Y qubits of Ȳ).
        g_x, g_z    — the per-system L=1 gadgets (Webster, Smith, Cohen
                      arXiv:2511.15989 §II.A) for X̄ (basis=X) and Z̄ (basis=Z),
                      each Cheeger-boosted to ≥ 1, built on the same ``code``
                      with disjoint κ-ancilla ranges.
        Y_stab      — ``(|W|, 2*n_merged)`` symplectic Y rows from the cross-
                      merge: the mixed checks ``y_v`` at each ``v ∈ W``.
        H_sym       — ``(rows, 2*n_merged)`` merged symplectic check matrix
                      = [HX_out | 0] ∪ [0 | HZ_out] ∪ Y_stab ∪ ∂_0. Column layout
                      in each half is ``[data (n) | κ_x | κ_z]``.
        merged_code — ``QuditCode(field(H_sym), is_subsystem_code=...)`` in
                      which Ȳ is in the measured stabilizer center; encodes one
                      fewer logical than ``code``.
        obs0_xor_map— the full Ȳ readout product (Ide, Gowda, Nadkarni,
                      Dauphinais arXiv:2410.02753 §III.D): the merged-code rows
                      whose eigenbasis-compatible GF(2) product equals the
                      symplectic Ȳ = [x | z] on the original data columns. Each
                      :class:`Obs0Row` records its ``H_sym`` row index plus
                      Pauli-family provenance (``"X"`` / ``"Z"`` / ``"Y"``, and
                      its index within that family). Solved over GF(2) by
                      :func:`_ybar_obs0_rows` — NOT hardcoded.
                      :func:`build_single_y_ppm_circuit` emits the FAULT-TOLERANT
                      ``obs0`` as the XOR of these rows' IN-CIRCUIT last-QEC-round
                      ancilla outcomes (family_index → ``checks_x`` / ``checks_z``
                      / ``y_ancilla_ids`` slot) — but ONLY when that XOR is
                      deterministic on the prepared state. The bare ``[x | z]``
                      product carries X on V_X / Z on V_Z data, so it is
                      deterministic exactly on a proper Ȳ-eigenstate codeword: the
                      ``data_init="Y±"`` prep injects the EXACT |Ȳ±⟩ = S̄|X̄±⟩
                      codeword, on which every code stabilizer is +1 and obs0 is
                      emitted; on a non-eigenstate prep (|0̄⟩/|+̄⟩) it is a genuine
                      50/50 and gated off (unless ``force_obs0``). The same bare
                      representative is the only feasible one on a general code (BB
                      [[36,8,4]], where an all-Y representative does not exist).
                      The bare product equals Ȳ = iX̄Z̄ EXACTLY (for Steane
                      [x | z] = X₂X₄Z₁Z₃Y₅ and iX̄Z̄ = +X₂X₄Z₁Z₃Y₅), so the raw
                      obs0 bit IS the Ȳ eigenvalue bit: Y+ → 0, Y− → 1.
        obs0_readout— :class:`Obs0ReadoutPlan`: the DESTRUCTIVE cross-check
                      (``obs1``, NOT the physical readout). Per support-qubit it
                      gives the Pauli to read off the final destructive readout:
                      V_X data → X, V_Z data → Z, W data → Y, κ_x → Z, κ_z → X.
                      This reads the SAME Ȳ = [x | z] product as ``obs0`` off the
                      destructive records, so it carries the same eigenvalue
                      (Y+ → 0, Y− → 1). :func:`build_single_y_ppm_circuit` uses it
                      in ``benchmark_y`` as ``obs0 ⊕ obs1`` (deterministic on ANY
                      input — both equal Ȳ); as a standalone ``obs1`` it is emitted
                      only when its destructive basis matches the readout
                      (gated by ``_observable_is_deterministic``).
    """

    code: CSSCode
    x: np.ndarray
    z: np.ndarray
    W: tuple[int, ...]
    g_x: GadgetLayout
    g_z: GadgetLayout
    Y_stab: np.ndarray
    H_sym: np.ndarray
    merged_code: QuditCode
    obs0_xor_map: tuple[Obs0Row, ...]
    obs0_readout: Obs0ReadoutPlan


def build_y_gadget(code: CSSCode, *, x: np.ndarray, z: np.ndarray) -> YGadgetLayout:
    """Assemble the homological logical-Y merged code for Ȳ = iX̄Z̄ (general ``|W|``).

    Implements the Eq.(68) merged code of Ide, Gowda, Nadkarni, Dauphinais
    arXiv:2410.02753 §III.C/§III.D:
    glue the X-measurement L=1 gadget of ``x`` and the Z-measurement L=1 gadget
    of ``z`` (both Webster, Smith, Cohen arXiv:2511.15989 §II.A) along the
    physical Y-support ``W = supp(x) ∩ supp(z)``. At each ``v ∈ W`` the S_X' row
    and S_Z' row anchored there are fused into one explicit mixed check ``y_v``
    via the §III.D cross-merge (NOT a single-qubit basis-change rotation — the
    non-CSS content lives in the ``y_v`` rows).

    Per-system Cheeger boost (arXiv:2410.02753 §III.D: per-system
    distance argument). Each gadget is boosted to boundary Cheeger constant ≥ 1
    BEFORE merging; the boosted gadgets are returned so the merged-code width
    ``n + k_x + k_z`` is consistent with ``len(g_x.Q_prime)``.

    The merged symplectic column space is ``[data (n) | κ_x | κ_z]``. The two
    systems share the data qubits, so the original code stabilizers appear ONCE:
    H_X is extended onto κ_z so it commutes with S_Z', and H_Z onto κ_x so it
    commutes with S_X' (the dual-extension of Webster §II.A step 3). NO per-system
    gauge rows are appended; instead ``∂_0 = ker(merged ∂_1)`` (the cycle basis
    of the glued graph, §III.D Eq.(66)/(67)) is appended. The only pairs that
    anticommute before the merge are (S_X', S_Z') at each ``v ∈ W``; fusing them
    removes that anticommutation, so the merged code is a genuine stabilizer code
    encoding ``code.dimension − 1`` logicals with Ȳ = iX̄Z̄ in its stabilizer
    group.

    The symplectic check matrix H̃ (Ide, Gowda, Nadkarni, Dauphinais
    arXiv:2410.02753 §III.D) is assembled in formula order with one named
    local variable per block.  Column layout per symplectic half:
    ``[ data (n) | κ_x (k_x) | κ_z (k_z) ]``::

        |   X-part (data | κ_x | κ_z)         |   Z-part (data | κ_x | κ_z)
        |--------------------------------------|------------------------------------
        |  H_X     0       π_{C₀^Z}^T         |   ·       ·        ·          block 1 (X-checks)
        |  π_{V_X} ∂₁ˣ|_{V_X}  0              |   ·       ·        ·          block 2 (S_X' on V_X)
        |  π_W     ∂₁ˣ|_W   0                 |  π_W      0       ∂₁ᶻ|_W      block 3 (Y on W, mixed)
        |  ·       ·        ·                 |  H_Z    π_{C₀^X}^T  0         block 4 (Z-checks)
        |  ·       ·        ·                 |  π_{V_Z}  0       ∂₁ᶻ|_{V_Z}  block 5 (S_Z' on V_Z)
        |  0       0        ∂₀^Z              |   0      ∂₀^X      0          block 6 (cycles ∂₀)

    Named-block mapping (each is a local variable in the assembly)::

        Xcheck_rows = HX_out[:m_x]   # block 1: H_X ext onto κ_z = π_{C₀^Z}^T
        SXprime_rows = HX_out[m_x:]  # block 2: π_{V_X} | ∂₁ˣ|_{V_X} | 0
        Ymix_rows   = Y_stab         # block 3: mixed S'@W rows (apply_mixed_basis_merge)
        Zcheck_rows = HZ_out[:m_z]   # block 4: H_Z ext onto κ_x = π_{C₀^X}^T
        SZprime_rows = HZ_out[m_z:]  # block 5: π_{V_Z} | 0 | ∂₁ᶻ|_{V_Z}
        cycle_rows  = partial0        # block 6: ker(merged ∂₁) = ∂₀^Z / ∂₀^X

    Args:
        code: a CSS code carrying the logical qubit to be Y-measured.
        x: logical-X support, ``H_Z @ x == 0`` (mod 2); shape ``(n,)``.
        z: logical-Z support, ``H_X @ z == 0`` (mod 2); shape ``(n,)``.

    Returns:
        A :class:`YGadgetLayout`. ``ValueError`` is raised (via
        :func:`_locate_overlaps`) if ``x``/``z`` are not valid anticommuting
        logicals (``H_Z @ x = 0``, ``H_X @ z = 0``, ``x · z`` odd).
    """
    x = np.asarray(x).astype(np.uint8)
    z = np.asarray(z).astype(np.uint8)
    W = _locate_overlaps(code, x, z)

    # Per-system L=1 gadgets, each Cheeger-boosted to ≥1
    # (arXiv:2410.02753 §III.D: per-system distance argument).
    from .cheeger import boost_gadget, cheeger_constant

    g_x = build_gadget(code, x, basis=Pauli.X)
    g_z = build_gadget(code, z, basis=Pauli.Z)
    if cheeger_constant(g_x) < 1.0:
        g_x = boost_gadget(g_x, method="combinatorial", target=1.0, seed=0)
    if cheeger_constant(g_z) < 1.0:
        g_z = boost_gadget(g_z, method="combinatorial", target=1.0, seed=0)

    field = code.field
    n = code.num_qudits
    m_x = int(np.asarray(code.matrix_x).shape[0])
    m_z = int(np.asarray(code.matrix_z).shape[0])
    k_x = len(g_x.Q_prime)
    k_z = len(g_z.Q_prime)
    n_merged = n + k_x + k_z

    # S_X' / S_Z' generator + extension blocks (Webster §II.A step 3 decomposition).
    SX_prime = np.asarray(g_x.HX_merged[m_x:]).astype(np.uint8)  # (|V0x|, n+k_x)
    hz_ext_kx = np.asarray(g_x.HZ_merged[:m_z]).astype(np.uint8)  # [H_Z | f_0^x]
    SZ_prime = np.asarray(g_z.HZ_merged[m_z:]).astype(np.uint8)  # (|V0z|, n+k_z)
    hx_ext_kz = np.asarray(g_z.HX_merged[:m_x]).astype(np.uint8)  # [H_X | f_0^z]

    def _embed(rows, *, data=None, kx=None, kz=None):
        out = np.zeros((rows, n_merged), dtype=np.uint8)
        if data is not None:
            out[:, :n] = data
        if kx is not None:
            out[:, n : n + k_x] = kx
        if kz is not None:
            out[:, n + k_x :] = kz
        return out

    # X-type rows: H_X extended onto κ_z; S_X' on [data | κ_x]. (NO per-system gauge.)
    HX_all = np.vstack(
        [
            _embed(m_x, data=hx_ext_kz[:, :n], kz=hx_ext_kz[:, n:]),
            _embed(SX_prime.shape[0], data=SX_prime[:, :n], kx=SX_prime[:, n:]),
        ]
    ).astype(np.uint8)
    # Z-type rows: H_Z extended onto κ_x; S_Z' on [data | κ_z].
    HZ_all = np.vstack(
        [
            _embed(m_z, data=hz_ext_kx[:, :n], kx=hz_ext_kx[:, n:]),
            _embed(SZ_prime.shape[0], data=SZ_prime[:, :n], kz=SZ_prime[:, n:]),
        ]
    ).astype(np.uint8)

    # Merge S_X'@v / S_Z'@v into one mixed y_v row for every v ∈ W (§III.D).
    HX_out, HZ_out, Y_stab, _obs0_y, _xl, _zl = apply_mixed_basis_merge(
        HX_all,
        HZ_all,
        merge_qubits=W,
        adapter_cols=tuple(range(n)),
    )
    HX_out = np.asarray(HX_out).astype(np.int_)
    HZ_out = np.asarray(HZ_out).astype(np.int_)
    if Y_stab is None or Y_stab.shape[0] < len(W):
        raise ValueError(
            f"BLOCKED: cross-merge produced {0 if Y_stab is None else Y_stab.shape[0]} "
            f"y_v rows, expected |W|={len(W)} (arXiv:2410.02753 §III.D)"
        )

    # ∂_0 = ker(merged ∂_1): cycle basis of the glued graph (§III.D Eq.66/67),
    # replacing the per-system gauge blocks. 0 rows at |W|=1, |W|−1 crossing rows
    # at |W|≥2.
    partial0 = _partial0_symplectic_rows(g_x, g_z, x, z, n=n, k_x=k_x, k_z=k_z)

    # --- Assemble H̃ block-by-block in formula order (Ide, Gowda, Nadkarni,
    # Dauphinais arXiv:2410.02753 §III.D; design spec §0). Column layout per
    # symplectic half: [ data (n) | κ_x (k_x) | κ_z (k_z) ].

    def _sym_x(rows: np.ndarray) -> np.ndarray:  # X-only rows → [X | 0]
        return np.hstack([rows, np.zeros_like(rows)]).astype(np.int_)

    def _sym_z(rows: np.ndarray) -> np.ndarray:  # Z-only rows → [0 | Z]
        return np.hstack([np.zeros_like(rows), rows]).astype(np.int_)

    # apply_mixed_basis_merge removed the S'@W rows into Y_stab, so:
    Xcheck_rows = HX_out[:m_x]   # block 1: [H_X | 0 | π_{C₀^Z}^T]
    SXprime_rows = HX_out[m_x:]  # block 2: [π_{V_X} | ∂₁ˣ|_{V_X} | 0]
    Ymix_rows   = Y_stab         # block 3: [π_W|∂₁ˣ|_W|0 ‖ π_W|0|∂₁ᶻ|_W]
    Zcheck_rows = HZ_out[:m_z]   # block 4: [H_Z | π_{C₀^X}^T | 0]
    SZprime_rows = HZ_out[m_z:]  # block 5: [π_{V_Z} | 0 | ∂₁ᶻ|_{V_Z}]
    cycle_rows  = partial0       # block 6: [0|0|∂₀^Z ‖ 0|∂₀^X|0]

    blocks = [
        _sym_x(Xcheck_rows),
        _sym_x(SXprime_rows),
        Ymix_rows.astype(np.int_),
        _sym_z(Zcheck_rows),
        _sym_z(SZprime_rows),
        cycle_rows.astype(np.int_),
    ]
    H_sym = (
        np.vstack([b for b in blocks if b.shape[0]])
        if any(b.shape[0] for b in blocks)
        else np.zeros((0, 2 * n_merged), dtype=np.int_)
    )

    Hx = H_sym[:, :n_merged]
    Hz = H_sym[:, n_merged:]
    comm = (Hx @ Hz.T + Hz @ Hx.T) % 2
    np.fill_diagonal(comm, 0)
    merged_code = QuditCode(field(H_sym), is_subsystem_code=bool(comm.any()))

    obs0_rows, obs0_readout = _ybar_obs0_rows(
        H_sym, code, x, z, n0=n, n_merged=n_merged, k_x=k_x
    )
    return YGadgetLayout(
        code=code,
        x=x,
        z=z,
        W=W,
        g_x=g_x,
        g_z=g_z,
        Y_stab=Y_stab,
        H_sym=H_sym,
        merged_code=merged_code,
        obs0_xor_map=obs0_rows,
        obs0_readout=obs0_readout,
    )
