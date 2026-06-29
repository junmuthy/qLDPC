"""Stim surgery circuit construction (single-PPM and joint-PPM).

References:
    Cain et al. arXiv:2603.28627 §B.1  — single-PPM measurement protocol.
    Webster, Smith, Cohen arXiv:2511.15989  — gadget Eq. 1 observable.
"""

from __future__ import annotations

import galois
import numpy as np
import stim

from qldpc.circuits.bookkeeping import DetectorRecord, MeasurementRecord, QubitIDs
from qldpc.circuits.memory.syndrome_measurement import EdgeColoring
from qldpc.circuits.noise_model import NoiseModel
from qldpc.codes.common import CSSCode, QuditCode
from qldpc.objects import Pauli

from .bridge import Bridge
from .gadget import GadgetLayout

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

    For surgery PPM circuits, pass ``keep_idx=0`` to retain only obs0
    (Webster Eq. 1, the physical syndrome-based readout). obs1 is an
    implementation cross-check that directly measures the data on V_0 and
    is NOT part of any physical protocol — keeping it for an LER run would
    sample the wrong distribution.

    Useful for sinter LER sweeps that compare one observable against a
    memory-experiment baseline — sinter expects exactly one observable per task.
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
    the obs0 outcome is silently random.

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


def build_single_ppm_circuit(
    gadget: GadgetLayout,
    *,
    rounds: int,
    noise_model: NoiseModel | None = None,
    data_init: str | None = None,
    destructive_measure_data: bool = True,
    single_sector: bool = False,
    block_observables: bool = False,
) -> stim.Circuit:
    """Cain §III.A single-PPM measurement circuit for `gadget`.

    Emits two OBSERVABLE_INCLUDE entries (see ``_surgery_observable`` for
    full semantics):

      * obs0 — Single-round Z̄ = ∏_{v ∈ support} A_v readout (Webster, Smith,
        Cohen arXiv:2511.15989 §II.A, gadget Eq. 1). XOR of the **last** QEC
        round's meas-check outcomes. The repeated rounds give FT distance
        via the detector layer; following Cain et al. arXiv:2603.28627 §B.1
        we read the logical eigenvalue from the final round.
      * obs1 — Direct destructive M on ``support`` qubits; noiseless
        cross-check, not a physical protocol.

    For LER / noisy runs, use ``keep_only_observable(circuit, keep_idx=0)``.

    ``data_init`` (optional): per-data-qubit init override; see
    ``_surgery_state_prep`` for the character-to-state mapping.

    ``destructive_measure_data`` (default True): when True the data is read out
    destructively at the end (emitting obs1 + the destructive final detectors).
    When False, the circuit is **detach-only / non-destructive** — the κ ancillas
    are still measured (the split that restores the bare code) but the data qubits
    are left encoded. The logical result is still obs0 (the in-circuit last-round
    meas-check product, fixed before detach), so obs1 and the destructive final
    detectors are dropped and the detector count drops accordingly.

    ``single_sector`` (default False): emit DETECTORs for the measured-basis sector
    only (X-checks for X̄, Z-checks for Z̄), dropping the complementary sector. obs0
    = X̄/Z̄ is flipped solely by the opposite single error type, which fires the
    measured-basis sector, so this preserves obs0's fault distance exactly while
    shrinking the DEM ~8× (the complementary detectors carried only correlated
    soft-info + off-basis correction). It does NOT correct the complementary error
    type on the data, so it is valid for an isolated obs0 readout/LER but not when
    the merged register must be handed back intact. CSS-type PPM only (X̄ or Z̄);
    inapplicable to Ȳ / mixed joints, which need both sectors.

    ``block_observables`` (default False): emit one OBSERVABLE_INCLUDE per logical
    operator of ``gadget.code`` (the full logical block, read from the destructive
    data measurement), instead of the single measured-operator obs0/obs1. This is
    the block-error convention used for the idling baseline (get_memory_experiment)
    and by Cain et al. arXiv:2603.28627 Ext. Data Fig. 1 ("failure = any logical
    Pauli error"), so a surgery-vs-idling block-error-per-cycle comparison is
    apples-to-apples. Requires ``destructive_measure_data=True``.
    """
    if block_observables and not destructive_measure_data:
        raise ValueError(
            "block_observables=True requires destructive_measure_data=True "
            "(the data readout is needed to infer all logical operators)"
        )
    merged_code = _gadget_merged_csscode(gadget)
    qubit_ids = QubitIDs.from_code(merged_code)
    n_data = gadget.code.num_qudits
    data_ids = qubit_ids.data[:n_data]
    Q_prime_ids = qubit_ids.data[n_data:]  # Q' ancilla qubit IDs
    bridge_ids: tuple[int, ...] = ()

    circuit = _surgery_qubit_coordinates(gadget, qubit_ids)
    circuit += _surgery_state_prep(
        gadget,
        data_ids,
        Q_prime_ids,
        bridge_ids,
        data_init=data_init,
    )
    qec_cycle, measurement_record, _ = _surgery_qec_cycle(
        gadget,
        merged_code,
        num_rounds=rounds,
        qubit_ids=qubit_ids,
        single_sector=single_sector,
    )
    circuit += qec_cycle
    circuit += _surgery_detach_and_readout(
        gadget,
        data_ids=data_ids,
        ancilla_ids=Q_prime_ids,
        bridge_ids=bridge_ids,
        measurement_record=measurement_record,
        destructive_measure_data=destructive_measure_data,
    )
    if destructive_measure_data:
        circuit += _surgery_final_detectors(
            gadget,
            merged_code,
            qubit_ids,
            measurement_record=measurement_record,
            single_sector=single_sector,
        )

    m_X, m_Z, n_V = (
        gadget.code.matrix_x.shape[0],
        gadget.code.matrix_z.shape[0],
        len(gadget.support),
    )
    if gadget.basis is Pauli.X:
        meas_check_ids = tuple(qubit_ids.checks_x[m_X : m_X + n_V])
    else:
        meas_check_ids = tuple(qubit_ids.checks_z[m_Z : m_Z + n_V])

    circuit += _surgery_observable(
        gadget,
        meas_check_ids=meas_check_ids,
        data_ids=data_ids,
        support_indices=gadget.support,
        measurement_record=measurement_record,
        destructive_measure_data=destructive_measure_data,
        block_observables=block_observables,
    )

    if noise_model is not None:
        circuit = noise_model.noisy_circuit(circuit)

    return circuit

def _stitch_intercode(g_l: GadgetLayout, g_r: GadgetLayout, bridge: Bridge) -> CSSCode:
    """Inter-code joint stitch (g_l.code is not g_r.code). Handles both bases.

    Builds M_meas (= H̃_X^joint when basis=X), the measured merged check matrix,
    and M_comp (= H̃_Z^joint when basis=X), the complementary merged check matrix,
    following Swaroop et al. (Swaroop, Jochym-O'Connor, Yoder) arXiv:2410.03628 §III.
    """
    assert g_l.code is not g_r.code
    field = g_l.code.field
    g_l_aug, g_r_aug = bridge.g_l_aug, bridge.g_r_aug

    # measured-basis abstraction: M_meas holds the new meas-basis check rows;
    # M_comp holds the dual cycle/comp-basis rows.
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

    M_meas_l = np.asarray(M_meas_l_src).astype(np.int_)
    M_meas_r = np.asarray(M_meas_r_src).astype(np.int_)
    M_comp_l = np.asarray(M_comp_l_src).astype(np.int_)
    M_comp_r = np.asarray(M_comp_r_src).astype(np.int_)

    n_l, n_r = g_l.code.num_qudits, g_r.code.num_qudits
    k_l, k_r = g_l_aug.incidence.shape[0], g_r_aug.incidence.shape[0]
    w = bridge.width
    n_merged = n_l + n_r + k_l + k_r + w
    r_l, r_r = g_l_aug.partial_0.shape[0], g_r_aug.partial_0.shape[0]

    cl_data = slice(0, n_l)
    cr_data = slice(n_l, n_l + n_r)
    Ql_prime = slice(n_l + n_r, n_l + n_r + k_l)
    Qr_prime = slice(n_l + n_r + k_l, n_l + n_r + k_l + k_r)
    c_adapter = slice(n_l + n_r + k_l + k_r, n_merged)

    # H̃_X^joint block structure (Swaroop et al. arXiv:2410.03628 §III):
    #   row-block 0: H_X^(l) data check rows       (cols: Q_l)
    #   row-block 1: H_X^(r) data check rows       (cols: Q_r)
    #   row-block 2: S_X'^l rows = [f_1^{l,T} | ∂_1^l | Π_l labels]  (cols: Q_l, Q'_l, 𝒜)
    #   row-block 3: S_X'^r rows = [f_1^{r,T} | ∂_1^r | Π_r labels]  (cols: Q_r, Q'_r, 𝒜)
    M_meas = np.zeros(
        (m_meas_l_data + m_meas_r_data + len(g_l.support) + len(g_r.support), n_merged),
        dtype=np.int_,
    )
    M_meas[:m_meas_l_data, cl_data] = M_meas_l[:m_meas_l_data, :n_l]
    M_meas[m_meas_l_data : m_meas_l_data + m_meas_r_data, cr_data] = M_meas_r[:m_meas_r_data, :n_r]
    S_prime_l = M_meas_l[m_meas_l_data:, :]  # S_X'^l rows = [f_1^{l,T} | ∂_1^l]
    S_prime_r = M_meas_r[m_meas_r_data:, :]  # S_X'^r rows = [f_1^{r,T} | ∂_1^r]
    meas_start = m_meas_l_data + m_meas_r_data
    M_meas[meas_start : meas_start + len(g_l.support), cl_data] = S_prime_l[:, :n_l]
    M_meas[meas_start : meas_start + len(g_l.support), Ql_prime] = S_prime_l[:, n_l:]
    M_meas[meas_start + len(g_l.support) :, cr_data] = S_prime_r[:, :n_r]
    M_meas[meas_start + len(g_l.support) :, Qr_prime] = S_prime_r[:, n_r:]
    for v_idx, lab in enumerate(bridge.label_l):
        if lab >= 0:
            M_meas[meas_start + v_idx, c_adapter.start + lab] = 1
    for v_idx, lab in enumerate(bridge.label_r):
        if lab >= 0:
            M_meas[meas_start + len(g_l.support) + v_idx, c_adapter.start + lab] = 1

    # H̃_Z^joint block structure (Swaroop et al. arXiv:2410.03628 §III):
    #   row-block 0: H_Z^(l) data check rows + f_0^(l) ext. onto Q'_l  (cols: Q_l, Q'_l)
    #   row-block 1: H_Z^(r) data check rows + f_0^(r) ext. onto Q'_r  (cols: Q_r, Q'_r)
    #   row-block 2: G_l gauge rows                                      (cols: Q'_l)
    #   row-block 3: G_r gauge rows                                      (cols: Q'_r)
    #   row-block 4: bridge cycle rows [T_l | T_r | H_R]                (cols: Q'_l, Q'_r, 𝒜)
    M_comp = np.zeros(
        (m_comp_l_data + m_comp_r_data + r_l + r_r + (w - 1), n_merged),
        dtype=np.int_,
    )
    M_comp[:m_comp_l_data, cl_data] = M_comp_l[:m_comp_l_data, :n_l]
    M_comp[:m_comp_l_data, Ql_prime] = M_comp_l[:m_comp_l_data, n_l:]
    M_comp[m_comp_l_data : m_comp_l_data + m_comp_r_data, cr_data] = M_comp_r[:m_comp_r_data, :n_r]
    M_comp[m_comp_l_data : m_comp_l_data + m_comp_r_data, Qr_prime] = M_comp_r[
        :m_comp_r_data, n_r:
    ]
    g_start = m_comp_l_data + m_comp_r_data
    M_comp[g_start : g_start + r_l, Ql_prime] = M_comp_l[m_comp_l_data:, n_l:]
    M_comp[g_start + r_l : g_start + r_l + r_r, Qr_prime] = M_comp_r[m_comp_r_data:, n_r:]
    cyc_start = g_start + r_l + r_r
    M_comp[cyc_start:, Ql_prime] = bridge.T_l
    M_comp[cyc_start:, Qr_prime] = bridge.T_r
    M_comp[cyc_start:, c_adapter] = bridge.H_R

    if bridge.basis is Pauli.X:
        return CSSCode(field(M_meas), field(M_comp), is_subsystem_code=False)
    return CSSCode(field(M_comp), field(M_meas), is_subsystem_code=False)


def _stitch_intracode(g_l: GadgetLayout, g_r: GadgetLayout, bridge: Bridge) -> CSSCode:
    """Intra-code joint stitch (g_l.code is g_r.code). Handles both bases.

    Builds M_meas (= H̃_X^joint when basis=X), the measured merged check matrix,
    and M_comp (= H̃_Z^joint when basis=X), the complementary merged check matrix,
    following Swaroop et al. (Swaroop, Jochym-O'Connor, Yoder) arXiv:2410.03628 §III.

    Differences from _stitch_intercode:
      - Shared data check rows (count = m_meas/comp_data once, not l+r).
      - Shared data column block (n columns, not n_l + n_r).
      - S_X'^s rows from both sides write into the SAME data-column slice.
    """
    assert g_l.code is g_r.code
    field = g_l.code.field
    g_l_aug, g_r_aug = bridge.g_l_aug, bridge.g_r_aug

    if bridge.basis is Pauli.X:
        M_meas_l_src, M_comp_l_src = g_l_aug.HX_merged, g_l_aug.HZ_merged
        M_meas_r_src, M_comp_r_src = g_r_aug.HX_merged, g_r_aug.HZ_merged
        m_meas_data = g_l.code.matrix_x.shape[0]
        m_comp_data = g_l.code.matrix_z.shape[0]
    else:
        M_meas_l_src, M_comp_l_src = g_l_aug.HZ_merged, g_l_aug.HX_merged
        M_meas_r_src, M_comp_r_src = g_r_aug.HZ_merged, g_r_aug.HX_merged
        m_meas_data = g_l.code.matrix_z.shape[0]
        m_comp_data = g_l.code.matrix_x.shape[0]

    M_meas_l = np.asarray(M_meas_l_src).astype(np.int_)
    M_meas_r = np.asarray(M_meas_r_src).astype(np.int_)
    M_comp_l = np.asarray(M_comp_l_src).astype(np.int_)
    M_comp_r = np.asarray(M_comp_r_src).astype(np.int_)

    n = g_l.code.num_qudits
    k_l, k_r = g_l_aug.incidence.shape[0], g_r_aug.incidence.shape[0]
    w = bridge.width
    n_merged = n + k_l + k_r + w
    r_l, r_r = g_l_aug.partial_0.shape[0], g_r_aug.partial_0.shape[0]

    c_data = slice(0, n)
    Ql_prime = slice(n, n + k_l)
    Qr_prime = slice(n + k_l, n + k_l + k_r)
    c_adapter = slice(n + k_l + k_r, n_merged)

    # H̃_X^joint block structure (Swaroop et al. arXiv:2410.03628 §III):
    #   row-block 0: H_X data check rows (shared)                     (cols: Q)
    #   row-block 1: S_X'^l rows = [f_1^{l,T} | ∂_1^l | Π_l labels]   (cols: Q, Q'_l, 𝒜)
    #   row-block 2: S_X'^r rows = [f_1^{r,T} | ∂_1^r | Π_r labels]   (cols: Q, Q'_r, 𝒜)
    M_meas = np.zeros(
        (m_meas_data + len(g_l.support) + len(g_r.support), n_merged),
        dtype=np.int_,
    )
    M_meas[:m_meas_data, c_data] = M_meas_l[:m_meas_data, :n]  # shared
    S_prime_l = M_meas_l[m_meas_data:, :]  # S_X'^l rows = [f_1^{l,T} | ∂_1^l]
    S_prime_r = M_meas_r[m_meas_data:, :]  # S_X'^r rows = [f_1^{r,T} | ∂_1^r]
    M_meas[m_meas_data : m_meas_data + len(g_l.support), c_data] = S_prime_l[:, :n]
    M_meas[m_meas_data : m_meas_data + len(g_l.support), Ql_prime] = S_prime_l[:, n:]
    M_meas[m_meas_data + len(g_l.support) :, c_data] = S_prime_r[:, :n]
    M_meas[m_meas_data + len(g_l.support) :, Qr_prime] = S_prime_r[:, n:]
    for v_idx, lab in enumerate(bridge.label_l):
        if lab >= 0:
            M_meas[m_meas_data + v_idx, c_adapter.start + lab] = 1
    for v_idx, lab in enumerate(bridge.label_r):
        if lab >= 0:
            M_meas[m_meas_data + len(g_l.support) + v_idx, c_adapter.start + lab] = 1

    # H̃_Z^joint block structure (Swaroop et al. arXiv:2410.03628 §III):
    #   row-block 0: H_Z data check rows + f_0^(l) onto Q'_l + f_0^(r) onto Q'_r
    #                                                         (cols: Q, Q'_l, Q'_r)
    #   row-block 1: G_l gauge rows                           (cols: Q'_l)
    #   row-block 2: G_r gauge rows                           (cols: Q'_r)
    #   row-block 3: bridge cycle rows [T_l | T_r | H_R]     (cols: Q'_l, Q'_r, 𝒜)
    M_comp = np.zeros(
        (m_comp_data + r_l + r_r + (w - 1), n_merged),
        dtype=np.int_,
    )
    M_comp[:m_comp_data, c_data] = M_comp_l[:m_comp_data, :n]
    M_comp[:m_comp_data, Ql_prime] = M_comp_l[:m_comp_data, n:]
    M_comp[:m_comp_data, Qr_prime] = M_comp_r[:m_comp_data, n:]
    M_comp[m_comp_data : m_comp_data + r_l, Ql_prime] = M_comp_l[m_comp_data:, n:]
    M_comp[m_comp_data + r_l : m_comp_data + r_l + r_r, Qr_prime] = M_comp_r[m_comp_data:, n:]
    cyc_start = m_comp_data + r_l + r_r
    M_comp[cyc_start:, Ql_prime] = bridge.T_l
    M_comp[cyc_start:, Qr_prime] = bridge.T_r
    M_comp[cyc_start:, c_adapter] = bridge.H_R

    if bridge.basis is Pauli.X:
        return CSSCode(field(M_meas), field(M_comp), is_subsystem_code=False)
    return CSSCode(field(M_comp), field(M_meas), is_subsystem_code=False)


def _stitch_to_joint_csscode(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
) -> CSSCode:
    """Assemble merged CSSCode for two-PPM surgery.

    Dispatches on the structural axis (g_l.code is g_r.code → intra-code
    shares data; otherwise inter-code).  Each branch handles both
    bridge.basis values internally via the M_meas/M_comp abstraction.
    """
    if g_l.code is g_r.code:
        return _stitch_intracode(g_l, g_r, bridge)
    return _stitch_intercode(g_l, g_r, bridge)


def _stitch_to_joint_code(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
) -> tuple[QuditCode, Bridge]:
    """Assemble merged CSSCode for same-basis two-PPM surgery.

    Delegates to ``_stitch_to_joint_csscode`` and returns the bridge
    unchanged. Mixed-basis joints are rejected upstream in
    ``build_joint_ppm_circuit`` (no valid CSS merged code exists).
    """
    return _stitch_to_joint_csscode(g_l, g_r, bridge), bridge


def _expand_joint_data_init(
    data_init: str | tuple[str, ...] | list[str] | None,
    n_l: int,
    n_r: int,
    intercode: bool,
) -> str | None:
    """Normalize ``data_init`` to a per-physical-qubit string.

    Two accepted shapes:

      * ``str`` (or ``None``) — passed through verbatim to ``_surgery_state_prep``
        (length-1 broadcasts to all data qubits; length n_l + n_r is per-qubit).

      * ``tuple[str, str]`` (or list) — per-code logical-init spec. Each entry
        is a string that is itself per-code broadcast (length 1) or per-qubit
        (length n_code). Tuple form is only valid for intercode joint PPM
        (intracode has a single data set; use a plain string instead).
        Example: ``("0", "+")`` initializes c_l data to |0⟩^{n_l} and c_r data
        to |+⟩^{n_r} — which, after the first round of merged-code SE projects
        into the codespace, equals logical |0⟩_L ⊗ |+⟩_L for any CSS code.
    """
    if data_init is None or isinstance(data_init, str):
        return data_init
    if not isinstance(data_init, (tuple, list)):
        raise TypeError(
            f"data_init must be str, tuple, list, or None; got {type(data_init).__name__}"
        )
    if not intercode:
        raise ValueError(
            "tuple/list data_init only valid for intercode joint PPM; "
            "intracode joint has a single data set, pass a plain string instead"
        )
    if len(data_init) != 2:
        raise ValueError(
            f"data_init tuple must have 2 entries (one per code), got {len(data_init)}"
        )
    spec_l, spec_r = data_init
    if not isinstance(spec_l, str) or not isinstance(spec_r, str):
        raise TypeError(
            f"data_init tuple entries must be str, got "
            f"({type(spec_l).__name__}, {type(spec_r).__name__})"
        )
    if len(spec_l) == 1:
        spec_l = spec_l * n_l
    if len(spec_r) == 1:
        spec_r = spec_r * n_r
    if len(spec_l) != n_l:
        raise ValueError(f"data_init[0] length {len(spec_l)} does not match c_l data count {n_l}")
    if len(spec_r) != n_r:
        raise ValueError(f"data_init[1] length {len(spec_r)} does not match c_r data count {n_r}")
    return spec_l + spec_r


_H_DATA_INIT = {"+": "0", "-": "1", "0": "+", "1": "-"}


def build_joint_ppm_circuit(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
    *,
    rounds: int,
    noise_model: NoiseModel | None = None,
    data_init: str | tuple[str, ...] | list[str] | None = None,
    destructive_measure_data: bool = True,
) -> tuple[stim.Circuit, QuditCode]:
    """Joint-PPM circuit for same-basis logical measurement (X̄_l⊗X̄_r / Z̄_l⊗Z̄_r).

    Emits two OBSERVABLE_INCLUDE entries (see ``_surgery_observable`` for
    full semantics):

      * obs0 — Single-round joint readout via Webster's identity
        ∏_{v ∈ support_l ∪ support_r} A_v = X̄_l ⊗ X̄_r (or Z̄_l ⊗ Z̄_r for
        basis=Z). See Webster, Smith, Cohen arXiv:2511.15989 §II.A. XOR of
        the **last** QEC round's meas-check outcomes on both patches.
        Detectors carry the FT load; following Cain et al.
        arXiv:2603.28627 §B.1 the final round is the readout point.
      * obs1 — Direct destructive M on ``support_l ∪ support_r``; noiseless
        cross-check, not a physical protocol.

    For LER / noisy runs, use ``keep_only_observable(circuit, keep_idx=0)``.

    ``data_init`` (optional): override the per-code data init.

      * ``str`` — per-physical-qubit (or len-1 broadcast). For intercode,
        positions [0:n_l) are left, [n_l:n_l+n_r) are right; for intracode,
        length is n_l. See ``_surgery_state_prep`` for the char-to-state mapping.
      * ``tuple[str, str]`` (intercode only) — per-code logical-init spec.
        ``data_init=("0", "+")`` → c_l in |0⟩_L, c_r in |+⟩_L.

    Mixed-basis joints (Z̄_l ⊗ X̄_r) are not supported: the Z-check and
    X-check anticommute on the shared bridge qubit, so no valid CSS merged
    code exists — joint PPMs are same-type only (Cross, He, Rall, Yoder
    arXiv:2407.18393). Use single-qubit Ȳ surgery
    (``build_single_y_ppm_circuit``) for mixed / non-CSS logical measurements.

    ``destructive_measure_data`` (default True): when False, detach-only /
    non-destructive — the κ + bridge ancillas are measured (the split) but the
    data is left encoded. obs0 (the in-circuit Z̄⊗Z̄ / X̄⊗X̄ readout) is still
    emitted; obs1 and the destructive final detectors are dropped.
    """
    if bridge.basis_l is not bridge.basis_r:
        raise NotImplementedError(
            "Mixed-basis joint PPM (e.g. Z̄ ⊗ X̄) is not supported: the Z- and "
            "X-checks anticommute on the bridge qubit, so no CSS merged code "
            "exists (Cross, He, Rall, Yoder arXiv:2407.18393, joint PPMs are "
            "same-type only). Use build_single_y_ppm_circuit for mixed / "
            "non-CSS logical measurements."
        )
    joint_code, bridge = _stitch_to_joint_code(g_l, g_r, bridge)
    return _build_joint_ppm_circuit_same_basis(
        g_l, g_r, bridge, joint_code,
        rounds=rounds, noise_model=noise_model, data_init=data_init,
        destructive_measure_data=destructive_measure_data,
    )


def _build_joint_ppm_circuit_same_basis(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
    joint_code: CSSCode,
    *,
    rounds: int,
    noise_model: NoiseModel | None,
    data_init: str | tuple[str, ...] | list[str] | None,
    destructive_measure_data: bool = True,
) -> tuple[stim.Circuit, QuditCode]:
    """Original same-basis joint PPM pipeline (CSS merged code)."""
    qubit_ids = QubitIDs.from_code(joint_code)
    intercode = g_l.code is not g_r.code

    g_l_aug, g_r_aug = bridge.g_l_aug, bridge.g_r_aug
    n_l = g_l.code.num_qudits
    n_r = g_r.code.num_qudits if intercode else 0
    k_l = g_l_aug.incidence.shape[0]
    k_r = g_r_aug.incidence.shape[0]
    w = bridge.width

    if intercode:
        data_ids = qubit_ids.data[: n_l + n_r]
        support_combined = tuple(g_l.support) + tuple(n_l + i for i in g_r.support)
    else:
        data_ids = qubit_ids.data[:n_l]
        support_combined = tuple(g_l.support) + tuple(g_r.support)
    Q_prime_ids = qubit_ids.data[n_l + n_r : n_l + n_r + k_l + k_r]  # Q' ancilla qubit IDs
    bridge_ids = qubit_ids.data[n_l + n_r + k_l + k_r :]
    assert len(bridge_ids) == w

    circuit = _surgery_qubit_coordinates(
        g_l,
        qubit_ids,
        joint=(g_r, bridge, intercode),
    )
    expanded_data_init = _expand_joint_data_init(data_init, n_l, n_r, intercode)
    circuit += _surgery_state_prep(
        g_l,
        data_ids,
        Q_prime_ids,
        bridge_ids,
        data_init=expanded_data_init,
    )
    qec_cycle, measurement_record, _ = _surgery_qec_cycle(
        g_l,
        joint_code,
        num_rounds=rounds,
        qubit_ids=qubit_ids,
        joint=(g_r, bridge, intercode),
    )
    circuit += qec_cycle
    circuit += _surgery_detach_and_readout(
        g_l,
        data_ids=data_ids,
        ancilla_ids=Q_prime_ids,
        bridge_ids=bridge_ids,
        measurement_record=measurement_record,
        destructive_measure_data=destructive_measure_data,
    )
    if destructive_measure_data:
        circuit += _surgery_final_detectors(
            g_l,
            joint_code,
            qubit_ids,
            measurement_record=measurement_record,
            joint=(g_r, bridge, intercode),
        )

    # S_X'^s check IDs: data H_X^(l) rows occupy first mX_l indices in
    # qubit_ids.checks_x, then m_X_r (inter-code), then S_X'^l, then S_X'^r.
    if bridge.basis is Pauli.X:
        check_ids = qubit_ids.checks_x
        m_l = g_l.code.matrix_x.shape[0]
        m_r = g_r.code.matrix_x.shape[0] if intercode else 0
    else:
        check_ids = qubit_ids.checks_z
        m_l = g_l.code.matrix_z.shape[0]
        m_r = g_r.code.matrix_z.shape[0] if intercode else 0
    n_V_l = len(g_l.support)
    n_V_r = len(g_r.support)
    meas_l_offset = m_l + m_r
    meas_r_offset = meas_l_offset + n_V_l
    meas_l_ids = tuple(check_ids[meas_l_offset : meas_l_offset + n_V_l])
    meas_r_ids = tuple(check_ids[meas_r_offset : meas_r_offset + n_V_r])
    meas_check_ids = meas_l_ids + meas_r_ids  # NO U_B / no adapter cycle-check ids

    circuit += _surgery_observable(
        g_l,
        meas_check_ids=meas_check_ids,
        data_ids=data_ids,
        support_indices=support_combined,
        measurement_record=measurement_record,
        destructive_measure_data=destructive_measure_data,
    )

    if noise_model is not None:
        circuit = noise_model.noisy_circuit(circuit)
    return circuit, joint_code


def _classify_reliable_round1_checks(
    gadget: GadgetLayout,
    qubit_ids: QubitIDs,
    *,
    g_r: GadgetLayout | None = None,
    intercode: bool = False,
) -> tuple[int, ...]:
    """Check ancillas with deterministic round-1 syndrome given surgery init state.

    Single-gadget (``g_r=None``): reliable = data-basis rows + gauge (S'_comp)
    rows. Joint-code (``g_r`` set): the same split spanning both gadgets — for
    inter-code the right gadget's data-check rows extend the data block
    (offsets m_X_l+m_X_r etc.); the bridge's new cycle rows live in the gauge
    block. Intra-code joints share data, so ``intercode=False`` (m_*_r = 0).
    """
    m_X_l = gadget.code.matrix_x.shape[0]
    m_Z_l = gadget.code.matrix_z.shape[0]
    m_X_r = g_r.code.matrix_x.shape[0] if (g_r is not None and intercode) else 0
    m_Z_r = g_r.code.matrix_z.shape[0] if (g_r is not None and intercode) else 0
    if gadget.basis is Pauli.X:
        reliable_x = qubit_ids.checks_x[: m_X_l + m_X_r]  # data S_X rows (det. +1)
        reliable_z = qubit_ids.checks_z[m_Z_l + m_Z_r :]  # gauge (S'_comp) + cycle
    else:
        reliable_x = qubit_ids.checks_x[m_X_l + m_X_r :]  # gauge (S'_comp) + cycle
        reliable_z = qubit_ids.checks_z[: m_Z_l + m_Z_r]  # data S_Z rows
    return tuple(reliable_x) + tuple(reliable_z)


def _surgery_state_prep(
    gadget: GadgetLayout,
    data_ids: tuple[int, ...],
    ancilla_ids: tuple[int, ...],
    bridge_ids: tuple[int, ...] = (),
    *,
    data_init: str | None = None,
) -> stim.Circuit:
    """Init data/ancilla/bridge qubits at the start of a surgery PPM circuit.

    Default (``data_init=None``):
      basis=X → data |+⟩ (RX), ancilla + bridge |0⟩ (R)
      basis=Z → data |0⟩ (R),  ancilla + bridge |+⟩ (RX)

    Optional ``data_init`` overrides per-data-qubit initial state. Each
    character selects a state for the data qubit at the same position:

      "0" → |0⟩  (R)
      "1" → |1⟩  (R + post-init X)
      "+" → |+⟩  (RX)
      "-" → |-⟩  (RX + post-init Z)

    A length-1 string broadcasts to all data qubits; otherwise length must
    equal ``len(data_ids)``.  ancilla + bridge init is independent of ``data_init``
    and always follows the protocol default (basis-complement +1 eigenstate).
    """
    if data_init is None:
        default_char = "+" if gadget.basis is Pauli.X else "0"
        per_qubit = default_char * len(data_ids)
    else:
        if len(data_init) == 1:
            data_init = data_init * len(data_ids)
        if len(data_init) != len(data_ids):
            raise ValueError(
                f"data_init length {len(data_init)} does not match num data "
                f"qubits {len(data_ids)}; pass a length-1 string to broadcast"
            )
        invalid = sorted(set(data_init) - set("01+-"))
        if invalid:
            raise ValueError(
                f"data_init must contain only '0', '1', '+', '-'; got invalid chars {invalid}"
            )
        per_qubit = data_init

    r_data: list[int] = []
    rx_data: list[int] = []
    x_after: list[int] = []
    z_after: list[int] = []
    for q, c in zip(data_ids, per_qubit):
        if c == "0":
            r_data.append(q)
        elif c == "1":
            r_data.append(q)
            x_after.append(q)
        elif c == "+":
            rx_data.append(q)
        else:  # "-"
            rx_data.append(q)
            z_after.append(q)

    circuit = stim.Circuit()
    if r_data:
        circuit.append("R", r_data)
    if rx_data:
        circuit.append("RX", rx_data)
    if x_after:
        circuit.append("X", x_after)
    if z_after:
        circuit.append("Z", z_after)

    anc_ids = list(ancilla_ids) + (list(bridge_ids) if bridge_ids else [])
    if anc_ids:
        anc_init = "R" if gadget.basis is Pauli.X else "RX"
        circuit.append(anc_init, anc_ids)

    return circuit


def _surgery_qec_cycle(
    gadget: GadgetLayout,
    merged_code: CSSCode,
    num_rounds: int,
    qubit_ids: QubitIDs,
    *,
    joint: tuple[GadgetLayout, Bridge, bool] | None = None,
    single_sector: bool = False,
) -> tuple[stim.Circuit, MeasurementRecord, DetectorRecord]:
    """num_rounds of merged-code SE; round-1 detectors only for reliable checks.

    Single-PPM (``joint=None``) and joint-PPM (``joint=(g_r, bridge,
    intercode)``) share one round loop; only the reliable-check classifier and
    the check→lane map differ by whether the right gadget + bridge participate.

    ``single_sector`` (CSS-type PPM only): emit DETECTORs for the measured-basis
    checks alone (``checks_x`` for X̄, ``checks_z`` for Z̄), dropping the
    complementary sector. All checks are still *measured* (the merge needs them);
    only their detectors are skipped. Valid because obs0 = X̄/Z̄ is flipped solely
    by the opposite single error type, which fires the measured-basis sector — so
    the complementary detectors carry no obs0 fault distance, only correlated
    soft-info / off-basis error correction (arXiv:2410.02753 §3).
    """
    strategy = EdgeColoring()
    one_round, round_measurement_record = strategy.get_circuit(merged_code, qubit_ids)
    if joint is None:
        reliable = set(_classify_reliable_round1_checks(gadget, qubit_ids))
        lane_idx = _check_lane_index_map(gadget, qubit_ids)
    else:
        g_r, _bridge, intercode = joint
        reliable = set(
            _classify_reliable_round1_checks(
                gadget, qubit_ids, g_r=g_r, intercode=intercode
            )
        )
        lane_idx = _check_lane_index_map(gadget, qubit_ids, joint=joint)
    all_check_ids = qubit_ids.check
    # Checks whose syndrome becomes a DETECTOR. single_sector keeps only the
    # measured-basis sector; all checks are still measured by ``one_round``.
    if single_sector:
        measured = set(qubit_ids.checks_x if gadget.basis is Pauli.X else qubit_ids.checks_z)
        detector_check_ids = tuple(cid for cid in all_check_ids if cid in measured)
    else:
        detector_check_ids = tuple(all_check_ids)

    circuit = stim.Circuit()
    measurement_record = MeasurementRecord()
    detector_record = DetectorRecord()

    # Round 1: emit DETECTORs only for reliable checks.
    circuit += one_round
    measurement_record.append(round_measurement_record)
    for check_id in detector_check_ids:
        if check_id in reliable:
            lane, idx = lane_idx[check_id]
            circuit.append(
                "DETECTOR", [measurement_record.get_target_rec(check_id)], (idx, lane, 0)
            )
    reliable_in_order = [cid for cid in detector_check_ids if cid in reliable]
    detector_record.append({cid: dd for dd, cid in enumerate(reliable_in_order)})

    if num_rounds > 1:
        repeat_circuit = one_round.copy()
        measurement_record.append(round_measurement_record)
        repeat_circuit.append("SHIFT_COORDS", [], (0, 0, 1))
        for check_id in detector_check_ids:
            lane, idx = lane_idx[check_id]
            repeat_circuit.append(
                "DETECTOR",
                [
                    measurement_record.get_target_rec(check_id, -1),
                    measurement_record.get_target_rec(check_id, -2),
                ],
                (idx, lane, 0),
            )
        circuit.append(stim.CircuitRepeatBlock(num_rounds - 1, repeat_circuit))
        measurement_record.append(round_measurement_record, repeat=num_rounds - 2)
        detector_record.append(
            {cid: dd for dd, cid in enumerate(detector_check_ids)},
            repeat=num_rounds - 1,
        )

    return circuit, measurement_record, detector_record


def _surgery_observable(
    gadget: GadgetLayout,
    *,
    meas_check_ids: tuple[int, ...],
    data_ids: tuple[int, ...],
    support_indices: tuple[int, ...],
    measurement_record: MeasurementRecord,
    destructive_measure_data: bool = True,
    block_observables: bool = False,
) -> stim.Circuit:
    """Emit OBSERVABLE_INCLUDE entries for the surgery PPM.

    ``block_observables=True`` overrides obs0/obs1 and instead emits one observable
    per logical operator of ``gadget.code`` (the full logical block), read from the
    destructive data measurement via ``get_observables`` — the block-error
    convention (Cain et al. arXiv:2603.28627 Ext. Data Fig. 1). Otherwise emits:

    obs0 — physical readout of the logical Pauli. The merged stabilizer group
        satisfies the single-round identity Z̄ = ∏_{v ∈ support} A_v (Webster,
        Smith, Cohen arXiv:2511.15989 §II.A, gadget Eq. 1). We point
        ``OBSERVABLE_INCLUDE`` at the **last** QEC round's meas-check (S'_meas)
        outcomes — their XOR is the eigenvalue bit of Z̄ (or X̄ for basis=X).
        Detectors carry the FT load via round-to-round consistency; following
        Cain et al. arXiv:2603.28627 §B.1 the final round is the natural
        readout point.

    obs1 — Direct stim measurement of the data qubits on ``support``. NOT a
        physical protocol — destructively projects the data — but a useful
        noiseless cross-check: in any noiseless shot ``obs0 == obs1``.

    For LER sweeps and any noisy run, keep ONLY obs0 via
    ``keep_only_observable(circuit, keep_idx=0)``.
    """
    if block_observables:
        # Full logical block from the destructive data readout: one observable per
        # logical operator of the bare code, ordered data qubit 0..n-1 (data_ids[i]
        # is code qubit i). Same machinery as get_memory_experiment's block readout.
        from qldpc.circuits.memory.memory import get_observables

        data_targets = [measurement_record.get_target_rec(d) for d in data_ids]
        return get_observables(
            gadget.code,
            data_qubits=list(range(len(data_ids))),
            basis=gadget.basis,
            on_measurements=data_targets,
        )
    # Precondition: every meas-check ancilla must have been measured during
    # the QEC cycle. Detach/readout only touches data + κ + bridge, never the
    # meas-check ancillas, so ``get_target_rec(cid)`` (default -1) is
    # guaranteed to resolve to the last QEC round. Fail loudly if a future
    # refactor breaks this.
    for cid in meas_check_ids:
        assert measurement_record[cid], (
            f"meas-check {cid} has no measurement record; "
            f"_surgery_observable expects the QEC cycle to have run first."
        )
    circuit = stim.Circuit()
    meas_targets = [measurement_record.get_target_rec(cid) for cid in meas_check_ids]
    circuit.append("OBSERVABLE_INCLUDE", meas_targets, 0)
    if not destructive_measure_data:
        return circuit  # detach-only: no destructive data to cross-check (obs1)
    data_targets = [measurement_record.get_target_rec(data_ids[i]) for i in support_indices]
    circuit.append("OBSERVABLE_INCLUDE", data_targets, 1)
    return circuit


def _surgery_final_detectors(
    gadget: GadgetLayout,
    merged_code: CSSCode,
    qubit_ids: QubitIDs,
    *,
    measurement_record: MeasurementRecord,
    joint: tuple[GadgetLayout, Bridge, bool] | None = None,
    single_sector: bool = False,
) -> stim.Circuit:
    """Emit DETECTORs for reliable stabs inferable from final readouts.

    For basis=X: data H_X (from Mx data) + G (from Mz κ).
    For basis=Z: data H_Z (from Mz data) + G (from Mx κ).
    Each DETECTOR XORs ⊕(final M-record on stab support) ⊕ last-round syndrome.
    Joint-PPM (``joint=(g_r, bridge, intercode)``) spans both gadgets' data rows.

    ``single_sector`` drops the complementary-basis gauge (G) detectors, matching
    the QEC-cycle filter — only the measured-basis data stabs are inferred.
    """
    m_X = gadget.code.matrix_x.shape[0]
    m_Z = gadget.code.matrix_z.shape[0]
    if joint is not None:
        g_r, _bridge, intercode = joint
        if intercode:
            m_X += g_r.code.matrix_x.shape[0]
            m_Z += g_r.code.matrix_z.shape[0]
    HX = np.asarray(merged_code.matrix_x).astype(np.uint8)
    HZ = np.asarray(merged_code.matrix_z).astype(np.uint8)

    circuit = stim.Circuit()
    lane_idx = _check_lane_index_map(gadget, qubit_ids, joint=joint)

    def _emit_detector(stab_row: np.ndarray, check_id: int) -> None:
        supp = np.where(stab_row)[0]
        targets = [measurement_record.get_target_rec(qubit_ids.data[q]) for q in supp]
        targets.append(measurement_record.get_target_rec(check_id, -1))
        lane, idx = lane_idx[check_id]
        circuit.append("DETECTOR", targets, (idx, lane, 0))

    if gadget.basis is Pauli.X:
        for kk in range(m_X):
            _emit_detector(HX[kk], qubit_ids.checks_x[kk])
        if not single_sector:  # complementary-basis gauge G (Mz κ)
            for kk in range(m_Z, HZ.shape[0]):
                _emit_detector(HZ[kk], qubit_ids.checks_z[kk])
    else:  # Pauli.Z (symmetric: S_X' in HZ, G in HX)
        for kk in range(m_Z):
            _emit_detector(HZ[kk], qubit_ids.checks_z[kk])
        if not single_sector:  # complementary-basis gauge G (Mx κ)
            for kk in range(m_X, HX.shape[0]):
                _emit_detector(HX[kk], qubit_ids.checks_x[kk])

    return circuit


def _surgery_detach_and_readout(
    gadget: GadgetLayout,
    *,
    data_ids: tuple[int, ...],
    ancilla_ids: tuple[int, ...],
    bridge_ids: tuple[int, ...],
    measurement_record: MeasurementRecord,
    destructive_measure_data: bool = True,
) -> stim.Circuit:
    """Detach the κ/bridge ancillas; optionally destructively measure the data.

    Mκ (the split that returns the bare code) always runs. When
    ``destructive_measure_data`` is True the data is then measured destructively
    (SHIFT_COORDS then Mdata); when False the data is left encoded (detach-only).
    """
    circuit = stim.Circuit()
    detach_qubits = list(ancilla_ids) + list(bridge_ids)
    ancilla_op = "M" if gadget.basis is Pauli.X else "MX"
    data_op = "MX" if gadget.basis is Pauli.X else "M"
    circuit.append(ancilla_op, detach_qubits)
    measurement_record.append({q: i for i, q in enumerate(detach_qubits)})
    if not destructive_measure_data:
        return circuit  # detach-only: leave the data encoded
    circuit.append("SHIFT_COORDS", [], (0, 0, 1))
    circuit.append(data_op, list(data_ids))
    measurement_record.append({q: i for i, q in enumerate(data_ids)})
    return circuit
