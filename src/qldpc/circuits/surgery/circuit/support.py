"""Surgery circuit support utilities (split from the former circuit.py).

GF(2) algebra, logical-basis selection, block-observable targets, merged-code
construction, observable filtering, logical state-init strings, and the
QUBIT_COORDS / check-lane layout helpers shared by the single- and joint-PPM
builders.

References:
    Cain et al. arXiv:2603.28627 §B.1  — single-PPM measurement protocol.
    Webster, Smith, Cohen arXiv:2511.15989  — gadget Eq. 1 observable.
"""

from __future__ import annotations

import galois
import numpy as np
import stim

from qldpc.circuits.bookkeeping import QubitIDs
from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli

from ..hmatrix.PPM_joint import Bridge
from ..hmatrix.PPM_X_Z import GadgetLayout

GF2 = galois.GF(2)


def _gf2_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """Particular solution to ``A x = b`` over GF(2), or None if inconsistent.

    Free variables are set to 0. Uses row reduction of the augmented matrix
    [A | b] via galois; a pivot-free row with nonzero RHS means inconsistency.
    """
    A = np.asarray(A).astype(np.int_) % 2
    b = np.asarray(b).astype(np.int_).reshape(-1) % 2
    m, n = A.shape
    if m == 0:
        return np.zeros(n, dtype=np.uint8)
    aug = GF2(np.hstack([A, b.reshape(-1, 1)]))
    rref = np.asarray(aug.row_reduce()).astype(np.int_)
    x = np.zeros(n, dtype=np.uint8)
    for row in rref:
        nz = np.nonzero(row[:n])[0]
        if nz.size == 0:
            if row[n]:
                return None  # 0 == 1 : inconsistent
            continue
        x[nz[0]] = row[n]
    return x


def _commuting_logical_basis(logical_ops: np.ndarray, L_support: np.ndarray) -> np.ndarray:
    """Basis of the bare-code logicals (rows of ``logical_ops``) commuting with L.

    The symplectic functional a_i = (L_support . logical_ops[i]) mod 2 is a linear
    functional on the k-dim logical space; its kernel (dim k or k-1) is the
    commuting subspace. When some a_i == 1, pick a pivot p with a_p == 1 and return
    {ops[i] : a_i == 0} ∪ {ops[i] ⊕ ops[p] : a_i == 1, i != p}. When all a_i == 0
    (same Pauli type / match-basis) return all k rows unchanged.

    Construction mirrors the gauge-fix logic of Webster, Smith, Cohen
    arXiv:2511.15989 §II.A used by build_gadget; here it selects the k-t readout
    observables of Cain et al. arXiv:2603.28627 Appendix D (t=1).
    """
    logical_ops = np.asarray(logical_ops).astype(np.uint8)
    L_support = np.asarray(L_support).astype(np.uint8).reshape(-1)
    a = (logical_ops @ L_support) % 2
    ones = np.nonzero(a)[0]
    if ones.size == 0:
        return logical_ops.copy()
    p = int(ones[0])
    rows = [logical_ops[i] for i in range(logical_ops.shape[0]) if a[i] == 0]
    rows += [(logical_ops[i] ^ logical_ops[p]) for i in ones[1:]]
    return np.array(rows, dtype=np.uint8).reshape(-1, logical_ops.shape[1])


def _block_observable_targets(
    merged_code: CSSCode,
    experiment_basis: Pauli,
    w: np.ndarray,
    n_data: int,
    col_record: dict[int, stim.target_rec],
) -> list[stim.target_rec]:
    """End-of-circuit targets for the frame-corrected merged-code rep of logical ``w``.

    ``w`` is the bare-code logical support (length n_data, ``experiment_basis`` type).
    The deformed representative v = w ⊕ c commutes with every opposite-type merged
    check, where c lives on the non-data columns (Q' + bridge) and is read from the
    split. Pauli-frame correction folded into the observable (no physical gates),
    per Cain et al. arXiv:2603.28627 Appendix D.
    """
    # opposite-type checks: an experiment_basis logical must commute with them.
    M_opp = (
        np.asarray(merged_code.matrix_z).astype(np.uint8)
        if experiment_basis is Pauli.X
        else np.asarray(merged_code.matrix_x).astype(np.uint8)
    )
    n_merged = merged_code.num_qudits
    corr_cols = np.arange(n_data, n_merged)  # Q' + bridge
    # ``w`` is the bare-code logical support on the data columns (length n_data);
    # accept a longer (full merged-width) array by reading only its data prefix.
    w = np.asarray(w).astype(np.uint8).reshape(-1)[:n_data]
    if M_opp.shape[0] == 0:
        c = np.zeros(corr_cols.size, dtype=np.uint8)
    else:
        syndrome = (M_opp[:, :n_data] @ w) % 2
        c = _gf2_solve(M_opp[:, corr_cols], syndrome)
        assert c is not None, "logical w does not commute with L (unsolvable deformation)"
    support_cols = list(np.nonzero(w)[0]) + [int(corr_cols[j]) for j in np.nonzero(c)[0]]
    return [col_record[col] for col in support_cols]


def _gadget_merged_csscode(g: GadgetLayout) -> CSSCode:
    return CSSCode(
        g.HX_merged.astype(np.int_),
        g.HZ_merged.astype(np.int_),
        is_subsystem_code=False,
    )


def keep_only_observable(circuit: stim.Circuit, keep_idx: int) -> stim.Circuit:
    """Return a copy of ``circuit`` with all OBSERVABLE_INCLUDE entries dropped
    except the one whose first argument equals ``keep_idx``. Recurses into
    REPEAT blocks so observables inside loops are filtered the same way.

    The CSS PPM builders (``build_single_ppm_circuit`` /
    ``build_joint_ppm_circuit``) no longer emit the old obs0/obs1 pair; their
    observable layout is now the Cain et al. arXiv:2603.28627 Appendix D
    experiment set — ``k + 1`` entries in match-basis (``k`` frame-corrected
    block logicals at indices ``0..k-1`` plus the time-like ``L`` at index ``k``)
    or ``k - 1`` in opposite-basis (block logicals commuting with ``L``, no
    time-like ``L``). See ``_surgery_observable`` for the exact layout.

    This helper survives because the **Ȳ / mixed surgery path** (``y_circuit.py``)
    still emits a single forced Ȳ readout observable for which sinter LER sweeps
    want exactly one observable per task. For a CSS PPM circuit, ``keep_idx=0``
    retains block logical 0 of ``experiment_basis``; choose the index that
    matches the baseline you are comparing against (sinter expects exactly one
    observable per task).
    """
    out = stim.Circuit()
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            out.append(
                stim.CircuitRepeatBlock(
                    op.repeat_count,
                    keep_only_observable(op.body_copy(), keep_idx),
                )
            )
            continue
        if op.name == "OBSERVABLE_INCLUDE":
            if int(op.gate_args_copy()[0]) != keep_idx:
                continue
        out.append(op)
    return out


def logical_state_init(code: CSSCode, state: str, *, log_idx: int) -> str:
    """Per-qubit ``data_init`` string preparing a Pauli logical state on
    logical qubit ``log_idx`` of a CSS code.

    ``state`` ∈ {"0", "1", "+", "-"}:
      * "0" → ``"0" * n``  — |0⟩^n projects to |0⟩_L^{⊗k} for any CSS code
      * "1" → "1" on supp(X̄_{log_idx}), "0" elsewhere — flips logical qubit
        ``log_idx`` from |0⟩_L to |1⟩_L; other logical qubits stay at |0⟩_L
      * "+" → ``"+" * n``  — |+⟩^n projects to |+⟩_L^{⊗k} for any CSS code
      * "-" → "-" on supp(Z̄_{log_idx}), "+" elsewhere — flips logical qubit
        ``log_idx`` from |+⟩_L to |-⟩_L; other logical qubits stay at |+⟩_L

    X̄_{log_idx} and Z̄_{log_idx} are taken from
    ``code.get_logical_ops(Pauli.X)[log_idx]`` and ``[Pauli.Z][log_idx]``;
    qldpc guarantees they form an anti-commuting symplectic pair on that
    logical qubit, so the prep is correct for ANY CSS code regardless of
    the parity of wt(X̄) / wt(Z̄). Naive broadcast ``data_init = "1" * n``
    is correct only when those weights are odd, and silently produces the
    wrong logical state on codes where they are even (e.g. BBCode [[36, 8]]
    with wt(Z̄_0) = 8).

    ``log_idx`` is REQUIRED (keyword-only, no default) — there is no
    universally "right" logical qubit choice on a k>1 code, so the
    caller must declare intent explicitly. Even for state="0" / "+"
    (which physically broadcast and don't depend on log_idx), supplying
    log_idx makes the targeted logical qubit unambiguous in the call
    site. To get a meaningful PPM truth-table check, ``log_idx`` MUST
    match the logical qubit chosen for the gadget's measured Z̄ (or X̄)
    — i.e. the gadget's seed operator should be
    ``code.get_logical_ops(Pauli.Z)[log_idx]`` (or ``[Pauli.X]`` for
    basis=X). The helper does NOT verify this; if indices disagree the
    prep targets a logical qubit that the gadget doesn't measure, and
    the L / block readout is silently random.

    The returned string has length ``code.num_qudits``. Plug it straight
    into ``build_single_ppm_circuit(..., data_init=...)`` or wrap with a
    tuple for ``build_joint_ppm_circuit(..., data_init=(s_l, s_r))``.

    Raises
    ------
    ValueError
        If ``state`` is not one of "0", "1", "+", "-".
    IndexError
        If ``log_idx`` is out of range for ``code.dimension``.
    """
    if state not in ("0", "1", "+", "-"):
        raise ValueError(f"state must be one of '0', '1', '+', '-'; got {state!r}")
    if not 0 <= log_idx < code.dimension:
        raise IndexError(f"log_idx={log_idx} out of range for code with dimension={code.dimension}")
    n = code.num_qudits
    if state in ("0", "+"):
        return state * n
    if state == "1":
        flip = np.asarray(code.get_logical_ops(Pauli.X)[log_idx]).astype(np.uint8)
        flip_char, base_char = "1", "0"
    else:  # state == "-"
        flip = np.asarray(code.get_logical_ops(Pauli.Z)[log_idx]).astype(np.uint8)
        flip_char, base_char = "-", "+"
    return "".join(flip_char if flip[i] else base_char for i in range(n))


def _surgery_qubit_coordinates(
    gadget: GadgetLayout,
    qubit_ids: QubitIDs,
    *,
    joint: tuple[GadgetLayout, Bridge, bool] | None = None,
) -> stim.Circuit:
    """Emit QUBIT_COORDS in surgery's per-role semantic lane layout.

    Symbols: V₀ → support; ancillas Q' → Q_prime. Each check family splits into
    the original code checks vs the new merge checks added by the gadget (we do
    NOT distinguish measured S' rows from gauge ∂_0 — both are "new checks").

    Lanes are Pauli-type-keyed and basis-independent (the SAME rule the Y mixed
    layout ``_mixed_basis_qubit_coords`` uses)::

      y=0  data qubits         (originally data + κ + bridge in qubit_ids.data
                                slot; we split them across y=0/1/6 here).
      y=1  ancilla qubits (Q')
      y=2  original X-checks    (H_X = checks_x[:m_X])
      y=3  new X-checks         (checks_x[m_X:], any X-type merge ancillas)
      y=4  original Z-checks    (H_Z = checks_z[:m_Z])
      y=5  new Z-checks         (checks_z[m_Z:], any Z-type merge ancillas)
      y=6  bridge data (adapter qubits; joint PPM only)
      y=7  bridge cycle ancillas (joint PPM only)

    The lane is fixed by Pauli TYPE only. This keeps X, Z, ZZ-joint, and Y
    measurements on one convention — lanes 2/3 are always the X-check family
    (original/new), lanes 4/5 the Z-check family.

    `joint=None` → single PPM. Otherwise pass (g_r, bridge, intercode).
    """
    circuit = stim.Circuit()

    if joint is None:
        g_l = gadget
        g_r = None
        bridge = None
        intercode = False
    else:
        g_l = gadget
        g_r, bridge, intercode = joint

    # Sizes for left side (always present).
    n_l = g_l.code.num_qudits
    m_X_l = g_l.code.matrix_x.shape[0]
    m_Z_l = g_l.code.matrix_z.shape[0]
    n_meas_l = len(g_l.support)
    n_gauge_l = g_l.partial_0.shape[0]
    k_l = len(g_l.Q_prime)

    # Sizes for right side (joint+intercode only — intracode shares data).
    if joint is not None and intercode:
        assert g_r is not None
        n_r = g_r.code.num_qudits
        m_X_r = g_r.code.matrix_x.shape[0]
        m_Z_r = g_r.code.matrix_z.shape[0]
        n_meas_r = len(g_r.support)
        n_gauge_r = g_r.partial_0.shape[0]
    elif joint is not None:  # intracode: data shared, ancillas separate per gadget
        assert g_r is not None
        n_r = 0
        m_X_r = m_Z_r = 0  # data checks not duplicated for intracode
        n_meas_r = len(g_r.support)
        n_gauge_r = g_r.partial_0.shape[0]
    else:
        n_r = 0
        m_X_r = m_Z_r = 0
        n_meas_r = n_gauge_r = k_r = 0

    # For joint PPM, the in-circuit κ count is the augmented value (bridge may
    # have added κ' ancillas during cellulation); use the bridge's augmented
    # gadgets as the source of truth.
    if joint is not None:
        assert bridge is not None
        k_l = bridge.g_l_aug.incidence.shape[0]
        k_r = bridge.g_r_aug.incidence.shape[0]

    n_data_total = n_l + n_r
    w = bridge.width if joint is not None and bridge is not None else 0

    # y=0 data
    for i in range(n_data_total):
        circuit.append("QUBIT_COORDS", qubit_ids.data[i], (i, 0))

    # y=1 κ
    for i in range(k_l + k_r):
        circuit.append("QUBIT_COORDS", qubit_ids.data[n_data_total + i], (i, 1))

    # y=6 bridge data (joint PPM only)
    for i in range(w):
        circuit.append(
            "QUBIT_COORDS",
            qubit_ids.data[n_data_total + k_l + k_r + i],
            (i, 6),
        )

    # Check ancillas are Pauli-type-keyed (same rule for basis=X, basis=Z, and
    # the Y mixed layout): each check family splits into the original code checks
    # and the new merge checks. Original H_X on y=2 and all new X-type checks on
    # y=3; original H_Z on y=4 and all new Z-type checks on y=5. The new checks
    # are just the merge ancillas past the originals (the S'/∂_0 measured-vs-gauge
    # split is irrelevant to the lane — only Pauli type matters); cycle ancillas,
    # if any, are peeled off to y=7 below.
    is_basis_x = g_l.basis is Pauli.X
    m_X_total = m_X_l + m_X_r
    m_Z_total = m_Z_l + m_Z_r
    n_meas_total = n_meas_l + n_meas_r
    n_gauge_total = n_gauge_l + n_gauge_r
    # New (non-cycle) checks past the originals in each array.
    n_new_x = n_meas_total if is_basis_x else n_gauge_total
    n_new_z = n_gauge_total if is_basis_x else n_meas_total

    for i in range(m_X_total):
        circuit.append("QUBIT_COORDS", qubit_ids.checks_x[i], (i, 2))
    for i in range(n_new_x):
        circuit.append("QUBIT_COORDS", qubit_ids.checks_x[m_X_total + i], (i, 3))

    for i in range(m_Z_total):
        circuit.append("QUBIT_COORDS", qubit_ids.checks_z[i], (i, 4))
    for i in range(n_new_z):
        circuit.append("QUBIT_COORDS", qubit_ids.checks_z[m_Z_total + i], (i, 5))

    # Joint PPM: bridge cycle ancillas get their own lane y=7 (a check role,
    # kept off the bridge-data row y=6). Using x=i makes the qubit coord equal
    # the detector coord from _check_lane_index_map (lane 7), preserving the
    # detector-coord == qubit-coord invariant (see test_detector_coords_*).
    # Before this, cycle check i shared (i, 6) with bridge-data qubit i and
    # overlapped on the diagram (e.g. q19/q29 on a w=3 Steane joint).
    if joint is not None and w > 1:
        # The new cycle checks live at the end of checks_x (basis=Z) or
        # checks_z (basis=X). They're (w - 1) of them.
        if is_basis_x:
            cycle_check_ids = qubit_ids.checks_z[m_Z_total + n_gauge_total :]
        else:
            cycle_check_ids = qubit_ids.checks_x[m_X_total + n_gauge_total :]
        for i, cid in enumerate(cycle_check_ids):
            circuit.append("QUBIT_COORDS", cid, (i, 7))

    return circuit


def _check_lane_index_map(
    gadget: GadgetLayout,
    qubit_ids: QubitIDs,
    *,
    joint: tuple[GadgetLayout, Bridge, bool] | None = None,
) -> dict[int, tuple[int, int]]:
    """Build a {check_id: (lane, idx)} map matching the QUBIT_COORDS layout.

    Pauli-type-keyed, basis-independent (mirrors ``_surgery_qubit_coordinates``);
    each check family splits into original code checks vs new merge checks
    (measured-vs-gauge is not distinguished — both are new checks):
      lane=2: original X-checks H_X (checks_x[:m_X_total])
      lane=3: new X-checks (checks_x[m_X:], any X-type merge ancillas)
      lane=4: original Z-checks H_Z (checks_z[:m_Z_total])
      lane=5: new Z-checks (checks_z[m_Z:], any Z-type merge ancillas)
      lane=7: bridge cycle check ancillas (joint PPM only; lane=6 = bridge data).
    """
    is_basis_x = gadget.basis is Pauli.X

    if joint is None:
        m_X_total = gadget.code.matrix_x.shape[0]
        m_Z_total = gadget.code.matrix_z.shape[0]
        n_meas_total = len(gadget.support)
        n_gauge_total = gadget.partial_0.shape[0]
    else:
        g_r, bridge, intercode = joint
        m_X_total = gadget.code.matrix_x.shape[0]
        m_Z_total = gadget.code.matrix_z.shape[0]
        if intercode:
            m_X_total += g_r.code.matrix_x.shape[0]
            m_Z_total += g_r.code.matrix_z.shape[0]
        n_meas_total = len(gadget.support) + len(g_r.support)
        n_gauge_total = gadget.partial_0.shape[0] + g_r.partial_0.shape[0]

    result: dict[int, tuple[int, int]] = {}

    # data H_X on lane=2
    for i in range(m_X_total):
        result[qubit_ids.checks_x[i]] = (2, i)
    # data H_Z on lane=4
    for i in range(m_Z_total):
        result[qubit_ids.checks_z[i]] = (4, i)

    # Pauli-type-keyed: all new X-checks (checks_x past H_X) on lane 3, all new
    # Z-checks (checks_z past H_Z) on lane 5 — same as the QUBIT_COORDS layout.
    # n_new_x/n_new_z exclude the cycle rows, routed to lane 7 below.
    n_new_x = n_meas_total if is_basis_x else n_gauge_total
    n_new_z = n_gauge_total if is_basis_x else n_meas_total
    for i in range(n_new_x):
        result[qubit_ids.checks_x[m_X_total + i]] = (3, i)
    for i in range(n_new_z):
        result[qubit_ids.checks_z[m_Z_total + i]] = (5, i)

    # Joint PPM bridge cycle ancillas on lane=7 (lane=6 holds the bridge data).
    if joint is not None:
        if is_basis_x:
            cycle_ids = qubit_ids.checks_z[m_Z_total + n_gauge_total :]
        else:
            cycle_ids = qubit_ids.checks_x[m_X_total + n_gauge_total :]
        for i, cid in enumerate(cycle_ids):
            result[cid] = (7, i)

    return result
