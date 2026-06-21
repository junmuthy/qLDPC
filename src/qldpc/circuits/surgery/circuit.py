"""Stim surgery circuit construction (single-PPM and joint-PPM).

References:
    Cain et al. arXiv:2603.28627 §B.1  — single-PPM measurement protocol.
    Webster, Smith, Cohen arXiv:2511.15989  — gadget Eq. 1 observable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import stim

from qldpc.circuits.bookkeeping import DetectorRecord, MeasurementRecord, QubitIDs
from qldpc.circuits.memory.syndrome_measurement import EdgeColoring
from qldpc.circuits.noise_model import NoiseModel
from qldpc.codes.common import CSSCode, QuditCode
from qldpc.objects import Pauli

from .bridge import Bridge
from .gadget import GadgetLayout

if TYPE_CHECKING:
    from qldpc.circuits.surgery.joint_layout import JointPPMLayout


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

    Cain mapping: V_0 → support; κ ancillas → ancilla qubits (Q');
    χ ancillas → S'_meas ancillas (= χ rows); G ancillas → S'_comp ancillas (= G rows).

    Lanes:
      y=0  data qubits         (originally data + κ + bridge in qubit_ids.data
                                slot; we split them across y=0/1/6 here).
      y=1  ancilla qubits (Q')
      y=2  data H_X ancillas   (checks_x[:m_X])
      y=3  S'_meas ancillas (= χ rows)
                               (basis=X: checks_x[m_X:]; basis=Z: checks_z[m_Z:])
      y=4  data H_Z ancillas   (checks_z[:m_Z])
      y=5  S'_comp ancillas (= G rows)
                               (basis=X: checks_z[m_Z:]; basis=Z: checks_x[m_X:])
      y=6  bridge data (adapter qubits; joint PPM only)
      y=7  bridge cycle ancillas (joint PPM only)

    For basis=X, y is monotonic in qubit ID order (ids 0..6→y=0, 7..9→y=1,
    10..12→y=2, 13..15→y=3, 16..18→y=4, 19→y=5), so QUBIT_COORDS lines in
    the stringified circuit dump appear in increasing y order. basis=Z
    breaks monotonicity because χ and G swap matrix slots, but the lane
    numbers remain stable: S'_meas always y=3, S'_comp always y=5.

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
    n_gauge_l = g_l.gauge.shape[0]
    k_l = len(g_l.ancilla_qubits)

    # Sizes for right side (joint+intercode only — intracode shares data).
    if joint is not None and intercode:
        assert g_r is not None
        n_r = g_r.code.num_qudits
        m_X_r = g_r.code.matrix_x.shape[0]
        m_Z_r = g_r.code.matrix_z.shape[0]
        n_meas_r = len(g_r.support)
        n_gauge_r = g_r.gauge.shape[0]
    elif joint is not None:  # intracode: data shared, ancillas separate per gadget
        assert g_r is not None
        n_r = 0
        m_X_r = m_Z_r = 0  # data checks not duplicated for intracode
        n_meas_r = len(g_r.support)
        n_gauge_r = g_r.gauge.shape[0]
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

    # X-check ancillas: data H_X on y=2, then either χ on y=3 (basis=X) or G on y=5 (basis=Z).
    is_basis_x = g_l.basis is Pauli.X
    m_X_total = m_X_l + m_X_r
    n_meas_total = n_meas_l + n_meas_r
    n_gauge_total = n_gauge_l + n_gauge_r

    for i in range(m_X_total):
        circuit.append("QUBIT_COORDS", qubit_ids.checks_x[i], (i, 2))
    if is_basis_x:
        # χ rows on y=3 (within checks_x)
        for i in range(n_meas_total):
            circuit.append(
                "QUBIT_COORDS",
                qubit_ids.checks_x[m_X_total + i],
                (i, 3),
            )
    else:
        # G rows on y=5 (within checks_x for basis=Z)
        for i in range(n_gauge_total):
            circuit.append(
                "QUBIT_COORDS",
                qubit_ids.checks_x[m_X_total + i],
                (i, 5),
            )

    # Z-check ancillas: data H_Z on y=4, then either G on y=5 (basis=X) or χ on y=3 (basis=Z).
    m_Z_total = m_Z_l + m_Z_r
    for i in range(m_Z_total):
        circuit.append("QUBIT_COORDS", qubit_ids.checks_z[i], (i, 4))
    if is_basis_x:
        for i in range(n_gauge_total):
            circuit.append(
                "QUBIT_COORDS",
                qubit_ids.checks_z[m_Z_total + i],
                (i, 5),
            )
    else:
        for i in range(n_meas_total):
            circuit.append(
                "QUBIT_COORDS",
                qubit_ids.checks_z[m_Z_total + i],
                (i, 3),
            )

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

    Lanes for checks (idx is x position within lane):
      lane=2: data H_X check ancillas (checks_x[:m_X_total])
      lane=3: χ check ancillas (basis=X: checks_x[m_X:]; basis=Z: checks_z[m_Z:])
      lane=4: data H_Z check ancillas (checks_z[:m_Z_total])
      lane=5: G check ancillas (basis=X: checks_z[m_Z:]; basis=Z: checks_x[m_X:])
      lane=7: bridge cycle check ancillas (joint PPM only; lane=6 = bridge data).
    """
    is_basis_x = gadget.basis is Pauli.X

    if joint is None:
        m_X_total = gadget.code.matrix_x.shape[0]
        m_Z_total = gadget.code.matrix_z.shape[0]
        n_meas_total = len(gadget.support)
        n_gauge_total = gadget.gauge.shape[0]
    else:
        g_r, bridge, intercode = joint
        m_X_total = gadget.code.matrix_x.shape[0]
        m_Z_total = gadget.code.matrix_z.shape[0]
        if intercode:
            m_X_total += g_r.code.matrix_x.shape[0]
            m_Z_total += g_r.code.matrix_z.shape[0]
        n_meas_total = len(gadget.support) + len(g_r.support)
        n_gauge_total = gadget.gauge.shape[0] + g_r.gauge.shape[0]

    result: dict[int, tuple[int, int]] = {}

    # data H_X on lane=2
    for i in range(m_X_total):
        result[qubit_ids.checks_x[i]] = (2, i)
    # data H_Z on lane=4
    for i in range(m_Z_total):
        result[qubit_ids.checks_z[i]] = (4, i)

    if is_basis_x:
        # χ on lane=3 in checks_x[m_X:]; G on lane=5 in checks_z[m_Z:]
        for i in range(n_meas_total):
            result[qubit_ids.checks_x[m_X_total + i]] = (3, i)
        for i in range(n_gauge_total):
            result[qubit_ids.checks_z[m_Z_total + i]] = (5, i)
    else:
        # G on lane=5 in checks_x[m_X:]; χ on lane=3 in checks_z[m_Z:]
        for i in range(n_gauge_total):
            result[qubit_ids.checks_x[m_X_total + i]] = (5, i)
        for i in range(n_meas_total):
            result[qubit_ids.checks_z[m_Z_total + i]] = (3, i)

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
    """
    merged_code = _gadget_merged_csscode(gadget)
    qubit_ids = QubitIDs.from_code(merged_code)
    n_data = gadget.code.num_qudits
    data_ids = qubit_ids.data[:n_data]
    ancilla_ids = qubit_ids.data[n_data:]
    bridge_ids: tuple[int, ...] = ()

    circuit = _surgery_qubit_coordinates(gadget, qubit_ids)
    circuit += _surgery_state_prep(
        gadget,
        data_ids,
        ancilla_ids,
        bridge_ids,
        data_init=data_init,
    )
    qec_cycle, measurement_record, _ = _surgery_qec_cycle(
        gadget,
        merged_code,
        num_rounds=rounds,
        qubit_ids=qubit_ids,
    )
    circuit += qec_cycle
    circuit += _surgery_detach_and_readout(
        gadget,
        data_ids=data_ids,
        ancilla_ids=ancilla_ids,
        bridge_ids=bridge_ids,
        measurement_record=measurement_record,
    )
    circuit += _surgery_final_detectors(
        gadget,
        merged_code,
        qubit_ids,
        measurement_record=measurement_record,
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
    )

    if noise_model is not None:
        circuit = noise_model.noisy_circuit(circuit)

    return circuit


def _stitch_intercode(g_l: GadgetLayout, g_r: GadgetLayout, bridge: Bridge) -> CSSCode:
    """Inter-code joint stitch (g_l.code is not g_r.code). Handles both bases."""
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
    r_l, r_r = g_l_aug.gauge.shape[0], g_r_aug.gauge.shape[0]

    cl_data = slice(0, n_l)
    cr_data = slice(n_l, n_l + n_r)
    cl_ancilla = slice(n_l + n_r, n_l + n_r + k_l)
    cr_ancilla = slice(n_l + n_r + k_l, n_l + n_r + k_l + k_r)
    c_adapter = slice(n_l + n_r + k_l + k_r, n_merged)

    # Build M_meas: data χ-carrier rows (left & right) + χ rows + adapter Π labels.
    M_meas = np.zeros(
        (m_meas_l_data + m_meas_r_data + len(g_l.support) + len(g_r.support), n_merged),
        dtype=np.int_,
    )
    M_meas[:m_meas_l_data, cl_data] = M_meas_l[:m_meas_l_data, :n_l]
    M_meas[m_meas_l_data : m_meas_l_data + m_meas_r_data, cr_data] = M_meas_r[:m_meas_r_data, :n_r]
    meas_l_rows = M_meas_l[m_meas_l_data:, :]
    meas_r_rows = M_meas_r[m_meas_r_data:, :]
    meas_start = m_meas_l_data + m_meas_r_data
    M_meas[meas_start : meas_start + len(g_l.support), cl_data] = meas_l_rows[:, :n_l]
    M_meas[meas_start : meas_start + len(g_l.support), cl_ancilla] = meas_l_rows[:, n_l:]
    M_meas[meas_start + len(g_l.support) :, cr_data] = meas_r_rows[:, :n_r]
    M_meas[meas_start + len(g_l.support) :, cr_ancilla] = meas_r_rows[:, n_r:]
    for v_idx, lab in enumerate(bridge.label_l):
        if lab >= 0:
            M_meas[meas_start + v_idx, c_adapter.start + lab] = 1
    for v_idx, lab in enumerate(bridge.label_r):
        if lab >= 0:
            M_meas[meas_start + len(g_l.support) + v_idx, c_adapter.start + lab] = 1

    # Build M_comp: co-carrier data rows (with κ extension) + G_aug + new cycle.
    M_comp = np.zeros(
        (m_comp_l_data + m_comp_r_data + r_l + r_r + (w - 1), n_merged),
        dtype=np.int_,
    )
    M_comp[:m_comp_l_data, cl_data] = M_comp_l[:m_comp_l_data, :n_l]
    M_comp[:m_comp_l_data, cl_ancilla] = M_comp_l[:m_comp_l_data, n_l:]
    M_comp[m_comp_l_data : m_comp_l_data + m_comp_r_data, cr_data] = M_comp_r[:m_comp_r_data, :n_r]
    M_comp[m_comp_l_data : m_comp_l_data + m_comp_r_data, cr_ancilla] = M_comp_r[
        :m_comp_r_data, n_r:
    ]
    g_start = m_comp_l_data + m_comp_r_data
    M_comp[g_start : g_start + r_l, cl_ancilla] = M_comp_l[m_comp_l_data:, n_l:]
    M_comp[g_start + r_l : g_start + r_l + r_r, cr_ancilla] = M_comp_r[m_comp_r_data:, n_r:]
    cyc_start = g_start + r_l + r_r
    M_comp[cyc_start:, cl_ancilla] = bridge.T_l
    M_comp[cyc_start:, cr_ancilla] = bridge.T_r
    M_comp[cyc_start:, c_adapter] = bridge.H_R

    if bridge.basis is Pauli.X:
        return CSSCode(field(M_meas), field(M_comp), is_subsystem_code=False)
    return CSSCode(field(M_comp), field(M_meas), is_subsystem_code=False)


def _stitch_intracode(g_l: GadgetLayout, g_r: GadgetLayout, bridge: Bridge) -> CSSCode:
    """Intra-code joint stitch (g_l.code is g_r.code). Handles both bases.

    Differences from _stitch_intercode:
      - Shared data check rows (count = m_meas/comp_data once, not l+r).
      - Shared data column block (n columns, not n_l + n_r).
      - χ rows from both sides write into the SAME data-column slice.
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
    r_l, r_r = g_l_aug.gauge.shape[0], g_r_aug.gauge.shape[0]

    c_data = slice(0, n)
    cl_ancilla = slice(n, n + k_l)
    cr_ancilla = slice(n + k_l, n + k_l + k_r)
    c_adapter = slice(n + k_l + k_r, n_merged)

    # Build M_meas: shared data check rows + χ rows (both sides into shared data).
    M_meas = np.zeros(
        (m_meas_data + len(g_l.support) + len(g_r.support), n_merged),
        dtype=np.int_,
    )
    M_meas[:m_meas_data, c_data] = M_meas_l[:m_meas_data, :n]  # shared
    meas_l_rows = M_meas_l[m_meas_data:, :]
    meas_r_rows = M_meas_r[m_meas_data:, :]
    M_meas[m_meas_data : m_meas_data + len(g_l.support), c_data] = meas_l_rows[:, :n]
    M_meas[m_meas_data : m_meas_data + len(g_l.support), cl_ancilla] = meas_l_rows[:, n:]
    M_meas[m_meas_data + len(g_l.support) :, c_data] = meas_r_rows[:, :n]
    M_meas[m_meas_data + len(g_l.support) :, cr_ancilla] = meas_r_rows[:, n:]
    for v_idx, lab in enumerate(bridge.label_l):
        if lab >= 0:
            M_meas[m_meas_data + v_idx, c_adapter.start + lab] = 1
    for v_idx, lab in enumerate(bridge.label_r):
        if lab >= 0:
            M_meas[m_meas_data + len(g_l.support) + v_idx, c_adapter.start + lab] = 1

    # Build M_comp: shared data co-carrier rows with κ extension on BOTH sides,
    # then G_l, G_r, then new cycle.
    M_comp = np.zeros(
        (m_comp_data + r_l + r_r + (w - 1), n_merged),
        dtype=np.int_,
    )
    M_comp[:m_comp_data, c_data] = M_comp_l[:m_comp_data, :n]
    M_comp[:m_comp_data, cl_ancilla] = M_comp_l[:m_comp_data, n:]
    M_comp[:m_comp_data, cr_ancilla] = M_comp_r[:m_comp_data, n:]
    M_comp[m_comp_data : m_comp_data + r_l, cl_ancilla] = M_comp_l[m_comp_data:, n:]
    M_comp[m_comp_data + r_l : m_comp_data + r_l + r_r, cr_ancilla] = M_comp_r[m_comp_data:, n:]
    cyc_start = m_comp_data + r_l + r_r
    M_comp[cyc_start:, cl_ancilla] = bridge.T_l
    M_comp[cyc_start:, cr_ancilla] = bridge.T_r
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
    bridge.basis values internally via the χ-carrier abstraction.
    """
    if g_l.code is g_r.code:
        return _stitch_intracode(g_l, g_r, bridge)
    return _stitch_intercode(g_l, g_r, bridge)


def _stitch_to_joint_code(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
) -> tuple[QuditCode, Bridge]:
    """Assemble merged code (CSSCode for same-basis, QuditCode subsystem for mixed-basis).

    Same-basis path delegates to ``_stitch_to_joint_csscode`` and returns
    the bridge unchanged.

    Mixed-basis path (Cross, He, Rall, Yoder arXiv:2407.18393 Appendix A.2
    Theorem 20 proof; Cowtan, He, Williamson, Yoder arXiv:2503.05003 §3.5)
    delegates to ``_build_mixed_basis_joint_code`` which assembles a SUBSYSTEM
    code via the block-by-block ``joint_layout`` module: the per-side χ_l
    (basis_l-type) and χ_r (basis_r-type) seed-operator rows remain as
    separate gauge generators whose pairwise anti-commutation on shared
    adapter qubits is structural and expected. The stabilizer center is
    computed automatically by ``QuditCode(is_subsystem_code=True)``.
    """
    if bridge.basis_l is bridge.basis_r:
        return _stitch_to_joint_csscode(g_l, g_r, bridge), bridge

    code, bridge_out, _layout = _build_mixed_basis_joint_code(g_l, g_r, bridge)
    return code, bridge_out


def _assemble_meas_comp_per_side(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, slice]]:
    """Build per-side M_meas / M_comp blocks honoring each side's own basis.

    Each block is expanded to the full n_merged column width and zero-padded
    outside its native data + ancilla columns.

    Returns
    -------
    M_meas_l_block, M_comp_l_block, M_meas_r_block, M_comp_r_block
        Each shape (rows_side, n_merged); zero-padded into the full merged
        column space.
    slices
        Dict with keys 'cl_data', 'cr_data' (or 'c_data' for intracode),
        'cl_ancilla', 'cr_ancilla', 'c_adapter' — slice objects into the
        merged column range.

    Naming convention: 'meas' = the side's own measured-basis check rows
    (χ-carrier per Webster Eq. 1); 'comp' = the dual (cycle-Z for basis=X).
    Caller decides how to split these into H_X / H_Z by inspecting basis_l
    and basis_r.
    """
    intercode = g_l.code is not g_r.code
    g_l_aug, g_r_aug = bridge.g_l_aug, bridge.g_r_aug

    def _per_side(
        g: GadgetLayout, g_aug: GadgetLayout, basis: Pauli
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        if basis is Pauli.X:
            M_meas_src, M_comp_src = g_aug.HX_merged, g_aug.HZ_merged
            m_meas_data = g.code.matrix_x.shape[0]
            m_comp_data = g.code.matrix_z.shape[0]
        else:
            M_meas_src, M_comp_src = g_aug.HZ_merged, g_aug.HX_merged
            m_meas_data = g.code.matrix_z.shape[0]
            m_comp_data = g.code.matrix_x.shape[0]
        return (
            np.asarray(M_meas_src).astype(np.int_),
            np.asarray(M_comp_src).astype(np.int_),
            m_meas_data,
            m_comp_data,
        )

    M_meas_l, M_comp_l, m_meas_l_data, m_comp_l_data = _per_side(g_l, g_l_aug, bridge.basis_l)
    M_meas_r, M_comp_r, m_meas_r_data, m_comp_r_data = _per_side(g_r, g_r_aug, bridge.basis_r)

    n_l = g_l.code.num_qudits
    n_r = g_r.code.num_qudits if intercode else 0
    k_l, k_r = g_l_aug.incidence.shape[0], g_r_aug.incidence.shape[0]
    w = bridge.width

    if intercode:
        n_merged = n_l + n_r + k_l + k_r + w
        cl_data = slice(0, n_l)
        cr_data = slice(n_l, n_l + n_r)
        cl_ancilla = slice(n_l + n_r, n_l + n_r + k_l)
        cr_ancilla = slice(n_l + n_r + k_l, n_l + n_r + k_l + k_r)
        c_adapter = slice(n_l + n_r + k_l + k_r, n_merged)
        slices = {
            "cl_data": cl_data,
            "cr_data": cr_data,
            "cl_ancilla": cl_ancilla,
            "cr_ancilla": cr_ancilla,
            "c_adapter": c_adapter,
        }
    else:
        n = n_l
        n_merged = n + k_l + k_r + w
        c_data = slice(0, n)
        cl_ancilla = slice(n, n + k_l)
        cr_ancilla = slice(n + k_l, n + k_l + k_r)
        c_adapter = slice(n + k_l + k_r, n_merged)
        slices = {
            "c_data": c_data,
            "cl_data": c_data,
            "cr_data": c_data,
            "cl_ancilla": cl_ancilla,
            "cr_ancilla": cr_ancilla,
            "c_adapter": c_adapter,
        }

    def _expand(
        rows_local: np.ndarray,
        side_label_attr: str,
        m_data: int,
        n_side: int,
        c_data_slice: slice,
        c_ancilla_slice: slice,
        kind: str,
    ) -> np.ndarray:
        # Data rows: meas-side data checks live on data cols only (no κ extension).
        # Comp-side data checks get extended to commute with χ rows acting X on κ,
        # so they carry both data and κ-ancilla support — same convention as
        # the same-basis _stitch_intercode / _stitch_intracode helpers.
        m_total = rows_local.shape[0]
        out = np.zeros((m_total, n_merged), dtype=np.int_)
        out[:m_data, c_data_slice] = rows_local[:m_data, :n_side]
        if kind == "comp":
            out[:m_data, c_ancilla_slice] = rows_local[:m_data, n_side:]
        rest = rows_local[m_data:, :]
        out[m_data:, c_data_slice] = rest[:, :n_side]
        out[m_data:, c_ancilla_slice] = rest[:, n_side:]
        if kind == "meas":
            labels = bridge.label_l if side_label_attr == "l" else bridge.label_r
            for v_idx, lab in enumerate(labels):
                if lab >= 0:
                    out[m_data + v_idx, c_adapter.start + lab] = 1
        return out

    n_side_l = n_l
    n_side_r = n_r if intercode else n_l

    M_meas_l_block = _expand(
        M_meas_l, "l", m_meas_l_data, n_side_l, slices["cl_data"], cl_ancilla, "meas"
    )
    M_comp_l_block = _expand(
        M_comp_l, "l", m_comp_l_data, n_side_l, slices["cl_data"], cl_ancilla, "comp"
    )
    M_meas_r_block = _expand(
        M_meas_r, "r", m_meas_r_data, n_side_r, slices["cr_data"], cr_ancilla, "meas"
    )
    M_comp_r_block = _expand(
        M_comp_r, "r", m_comp_r_data, n_side_r, slices["cr_data"], cr_ancilla, "comp"
    )

    return M_meas_l_block, M_comp_l_block, M_meas_r_block, M_comp_r_block, slices


def _build_mixed_basis_joint_code(
    g_l: GadgetLayout, g_r: GadgetLayout, bridge: Bridge
) -> tuple[QuditCode, Bridge, "JointPPMLayout"]:
    """New mixed-basis stitch via block-by-block layout (joint_layout module).

    Returns the QuditCode plus the JointPPMLayout itself; downstream callers
    (specifically the obs0 emission in _build_joint_ppm_circuit_mixed_basis)
    consume the layout's row provenance to construct obs0 = ⊕ m(χ_l) ⊕
    ⊕ m(χ_r) ⊕ ⊕ m(y_q) directly.
    """
    from qldpc.circuits.surgery.joint_layout import JointPPMLayout, build_joint_layout

    layout = build_joint_layout(g_l, g_r, bridge)
    code = layout.to_quditcode(g_l.code.field)
    return code, bridge, layout


def _stitch_to_joint_code_mixed(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
) -> tuple[QuditCode, Bridge]:
    """Mixed-basis stitch: assemble code via Webster-Smith-Cohen cross-merge.

    The merged code for an inter-code mixed-basis joint Pauli measurement
    (e.g. X̄_l ⊗ Z̄_r) is built per:

      * Webster, Smith, Cohen arXiv:2511.15989 §II.B.2 — the cross-merge
        recipe that combines a χ_l (X-type) and χ_r (Z-type) pair on each
        shared bridge qubit into a single Y-type stabilizer row.
      * Cross, He, Rall, Yoder arXiv:2407.18393 Appendix A.2 (Theorem 20)
        — the underlying subsystem-code dimension analysis.
      * Cowtan, He, Williamson, Yoder arXiv:2503.05003 §3.5 — lattice-
        surgery specialization.

    Construction (inter-code Steane × Steane, k_l = k_r = 1, w = 3):

      1. Per-side blocks via ``_assemble_meas_comp_per_side`` (X/Z split
         by each side's own basis).
      2. Cycle rows on BOTH duals: cycle_l of comp_l-type with support
         T_l on cl_ancilla + H_R on adapter; cycle_r symmetric. These
         anti-commute pairwise on the adapter via H_R · H_R^T — structural,
         expected for the subsystem code.
      3. Cross-merge step (``apply_mixed_basis_merge`` from merge.py): for
         each adapter qubit q where both an X- and Z-row have single-{q}
         adapter support, fuse them into a Y-stab. The χ_l (X-type on
         data + adapter) and χ_r (Z-type) rows are the canonical pivots.
         Multi-adapter-col rows (cycle rows) are left unchanged.
      4. Final symplectic matrix = [HX_out | 0] ∪ [0 | HZ_out] ∪ Y_stab.

    Returns a ``QuditCode`` — tried as stabilizer code first, falls back
    to subsystem if any gauge generator pair anti-commutes (cycle_l /
    cycle_r anti-commutation is the normal failure mode).

    Bridge fields populated:
      * ``Y_stab``: the cross-merge Y rows (shape ``(n_Y, 2*n_merged)``)
      * ``obs0_xor_map``: list of Y_stab row indices XORed into obs0
        (per Lemma 2). All Y rows contribute.
      * ``x_leftover_indices`` / ``z_leftover_indices``: indices (within
        the X-row / Z-row blocks of the final code) of cycle rows that
        contribute to obs0 to cancel residual X^A / Z^A on the adapter.
      * ``merge_qubits``: adapter columns processed by the cross-merge.
    """
    import dataclasses

    field = g_l.code.field
    intercode = g_l.code is not g_r.code

    M_meas_l, M_comp_l, M_meas_r, M_comp_r, slices = _assemble_meas_comp_per_side(
        g_l, g_r, bridge
    )

    def _x_z_split(
        M_meas_block: np.ndarray, M_comp_block: np.ndarray, basis: Pauli
    ) -> tuple[np.ndarray, np.ndarray]:
        if basis is Pauli.X:
            return M_meas_block, M_comp_block
        return M_comp_block, M_meas_block

    HX_l, HZ_l = _x_z_split(M_meas_l, M_comp_l, bridge.basis_l)
    HX_r, HZ_r = _x_z_split(M_meas_r, M_comp_r, bridge.basis_r)

    g_l_aug, g_r_aug = bridge.g_l_aug, bridge.g_r_aug
    n_l = g_l.code.num_qudits
    n_r = g_r.code.num_qudits if intercode else 0
    k_l, k_r = g_l_aug.incidence.shape[0], g_r_aug.incidence.shape[0]
    w = bridge.width
    n_merged = (n_l + n_r if intercode else n_l) + k_l + k_r + w

    cl_ancilla = slices["cl_ancilla"]
    cr_ancilla = slices["cr_ancilla"]
    c_adapter = slices["c_adapter"]

    # Step 2: cycle rows on both duals.
    cycle_l = np.zeros((bridge.T_l.shape[0], n_merged), dtype=np.int_)
    cycle_l[:, cl_ancilla] = bridge.T_l
    cycle_l[:, c_adapter] = bridge.H_R

    cycle_r = np.zeros((bridge.T_r.shape[0], n_merged), dtype=np.int_)
    cycle_r[:, cr_ancilla] = bridge.T_r
    cycle_r[:, c_adapter] = bridge.H_R

    # Build pre-merge HX_all / HZ_all.
    # cycle_l is in the basis dual to basis_l; cycle_r in the basis dual to basis_r.
    HX_pre_rows: list[np.ndarray] = list(HX_l.astype(np.int_)) + list(HX_r.astype(np.int_))
    HZ_pre_rows: list[np.ndarray] = list(HZ_l.astype(np.int_)) + list(HZ_r.astype(np.int_))
    if bridge.basis_l is Pauli.X:
        HZ_pre_rows.extend(cycle_l.astype(np.int_))
    else:
        HX_pre_rows.extend(cycle_l.astype(np.int_))
    if bridge.basis_r is Pauli.X:
        HZ_pre_rows.extend(cycle_r.astype(np.int_))
    else:
        HX_pre_rows.extend(cycle_r.astype(np.int_))

    HX_all = (
        np.array(HX_pre_rows, dtype=np.int_)
        if HX_pre_rows
        else np.zeros((0, n_merged), dtype=np.int_)
    )
    HZ_all = (
        np.array(HZ_pre_rows, dtype=np.int_)
        if HZ_pre_rows
        else np.zeros((0, n_merged), dtype=np.int_)
    )

    # Step 3: cross-merge. merge_qubits = adapter columns with both X and Z support.
    from .merge import apply_mixed_basis_merge

    adapter_cols = tuple(range(c_adapter.start, c_adapter.stop))
    merge_qubits = tuple(
        q for q in adapter_cols if HX_all[:, q].any() and HZ_all[:, q].any()
    )
    HX_out, HZ_out, Y_stab, obs0_y, x_left, z_left = apply_mixed_basis_merge(
        HX_all.astype(np.uint8),
        HZ_all.astype(np.uint8),
        merge_qubits,
        adapter_cols=adapter_cols,
    )
    HX_out = np.asarray(HX_out).astype(np.int_)
    HZ_out = np.asarray(HZ_out).astype(np.int_)

    # Identify the leftover cycle row indices (within HX_out / HZ_out) for
    # obs0 cancellation per Lemma 2. Cycle rows have weight ≥ 2 support on
    # the adapter, so we identify them structurally. For our construction,
    # cycle_r (X-type) lives in HX_out (when basis_r=Z), and cycle_l (Z-type)
    # lives in HZ_out (when basis_l=X). The merge algorithm never deletes
    # multi-adapter rows so they survive in order at the END of HX_out/HZ_out.
    def _adapter_weight(row: np.ndarray) -> int:
        return int(row[c_adapter.start : c_adapter.stop].sum())

    x_cycle_indices_in_out = tuple(
        i for i in range(HX_out.shape[0]) if _adapter_weight(HX_out[i]) >= 2
    )
    z_cycle_indices_in_out = tuple(
        i for i in range(HZ_out.shape[0]) if _adapter_weight(HZ_out[i]) >= 2
    )

    # Step 4: pack symplectic matrix.
    rows_sym: list[np.ndarray] = []
    for r in HX_out:
        rows_sym.append(np.concatenate([r, np.zeros(n_merged, dtype=np.int_)]))
    for r in HZ_out:
        rows_sym.append(np.concatenate([np.zeros(n_merged, dtype=np.int_), r]))
    if Y_stab is not None:
        for r in Y_stab:
            rows_sym.append(r.astype(np.int_))

    sym_matrix = (
        np.array(rows_sym, dtype=np.int_)
        if rows_sym
        else np.zeros((0, 2 * n_merged), dtype=np.int_)
    )

    # Try stabilizer code first; fall back to subsystem if cycle anti-commutation
    # makes the matrix non-CSS-commuting.
    is_subsystem = False
    if rows_sym:
        Hx = sym_matrix[:, :n_merged].astype(np.int_)
        Hz = sym_matrix[:, n_merged:].astype(np.int_)
        comm = (Hx @ Hz.T + Hz @ Hx.T) % 2
        np.fill_diagonal(comm, 0)
        is_subsystem = bool(comm.any())

    joint_code = QuditCode(field(sym_matrix), is_subsystem_code=is_subsystem)

    bridge_populated = dataclasses.replace(
        bridge,
        Y_stab=Y_stab if Y_stab is not None else None,
        merge_qubits=merge_qubits,
        obs0_xor_map=tuple(obs0_y),
        x_leftover_indices=x_cycle_indices_in_out,
        z_leftover_indices=z_cycle_indices_in_out,
    )
    return joint_code, bridge_populated


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


def _dual_csscode(code: CSSCode) -> CSSCode:
    """Transversal-Hadamard dual of a CSS code (matrix_x ↔ matrix_z).

    Applying a physical Hadamard to every data qubit of a CSS code maps it to
    its dual: H_X ↔ H_Z, X̄ ↔ Z̄ (SJOY24 / Swaroop et al. arXiv:2410.03628 §II,
    "appropriate local basis for each qubit"). The dual is itself CSS because
    H_X H_Z^T = 0 is symmetric. This is the local/transversal-H frame change
    used to turn an X-side gadget into a Z-type one so the existing same-basis
    bridge path applies verbatim.
    """
    return CSSCode(
        np.asarray(code.matrix_z).astype(np.int_),
        np.asarray(code.matrix_x).astype(np.int_),
        is_subsystem_code=False,
    )


def _rotate_x_side_to_z(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
) -> tuple[GadgetLayout, GadgetLayout, Bridge, bool]:
    """Rotate the X-basis side of a mixed-basis joint to Z-type via a dual frame.

    Mixed joint (one side X, one side Z, e.g. X̄_l ⊗ Z̄_r) is turned into an
    ordinary same-basis Z⊗Z joint by replacing the X-side code with its
    transversal-Hadamard dual (``_dual_csscode``). Because the gadget's V_0,
    incidence, data_checks and gauge depend only on the *complementary* check
    matrix of the measured basis — which is invariant under the X↔Z swap of a
    dual code — the rebuilt Z-gadget on the dual code has bit-for-bit identical
    support / incidence / ports, so SkipTree sees exactly the same Z-type port
    graph (hard constraint 1). The same-basis bridge is then rebuilt over the
    rotated gadgets, preserving the caller's port subsets.

    SJOY24 / Swaroop et al. arXiv:2410.03628 §II: the universal adapter is only
    ever defined for a Z-type operator, and arbitrary Paulis are reduced to it
    by a local basis change. Here the X-side becomes Z-type, the right side is
    already Z, so both are uniform and the CSS same-basis path (already
    |Y_+⟩-free) applies. Cross, He, Rall, Yoder arXiv:2407.18393 §3.7 gauging
    measurement is realized without the |Y_+⟩ adapter or null-space synthesis.

    Returns
    -------
    g_l_rot, g_r_rot
        The rotated gadgets (both Z-basis). The side that was already Z is
        returned unchanged; the X-side is the dual-code Z-gadget.
    bridge_rot
        A same-basis (Z) bridge over the rotated gadgets.
    left_is_x
        True iff the LEFT side carried the X basis (so its data_init must be
        H-transformed). Exactly one side is X for a mixed joint.
    """
    from .bridge import build_bridge
    from .gadget import build_gadget

    left_is_x = bridge.basis_l is Pauli.X
    intracode = g_l.code is g_r.code

    def _rotate(g: GadgetLayout) -> GadgetLayout:
        dual = _dual_csscode(g.code)
        # supp(x) is a logical-X of the original code → a logical-Z of the dual
        # (same bit support), so build a Z-basis gadget on the dual with seed x.
        return build_gadget(dual, np.asarray(g.x).astype(np.uint8), basis=Pauli.Z)

    if intracode:
        # Rotating one side to its dual makes the two sides use different code
        # objects, so a single-patch (intra-code) mixed joint cannot be routed
        # through the same-basis path as an intra-code stitch. This is a genuine
        # mixed single-qubit Ȳ-style overlap (CHRY §3.7 q_0/q_1 machinery) and
        # is out of scope for the disjoint-code dual-frame reduction.
        raise NotImplementedError(
            "intra-code mixed-basis joint via Hadamard-dual frame is not "
            "supported (the dual rotation makes the two sides use different "
            "codes); use distinct code instances for the two logicals"
        )

    if left_is_x:
        g_l_rot = _rotate(g_l)
        g_r_rot = g_r
    else:
        g_l_rot = g_l
        g_r_rot = _rotate(g_r)

    bridge_rot = build_bridge(
        g_l_rot,
        g_r_rot,
        port_subset_l=tuple(bridge.port_l),
        port_subset_r=tuple(bridge.port_r),
    )
    return g_l_rot, g_r_rot, bridge_rot, left_is_x


def _h_transform_left_data_init(
    data_init: str | tuple[str, ...] | list[str] | None,
    n_l: int,
    n_r: int,
    *,
    left_is_x: bool,
) -> tuple[str, ...] | None:
    """Normalize ``data_init`` to a (spec_l, spec_r) pair, H-transforming the
    X-side spec into the dual frame (+↔0, -↔1).

    The rotated joint runs a Z⊗Z surgery on the dual frame, so a prepared
    X̄-eigenstate on the original X-side must be re-expressed as the
    corresponding Z̄-eigenstate of the dual code: H|+⟩=|0⟩, H|-⟩=|1⟩. Tracking
    the transversal H purely in the init string (and the dual code's
    stabilizers) is physically identical to inserting a depth-1 H layer, with
    no Y states anywhere. The non-X side passes through unchanged.

    Returns ``None`` (use builder defaults) iff ``data_init`` is None.
    """
    if data_init is None:
        return None
    if isinstance(data_init, str):
        if len(data_init) == 1:
            spec_l, spec_r = data_init * n_l, data_init * n_r
        else:
            spec_l, spec_r = data_init[:n_l], data_init[n_l:]
    elif isinstance(data_init, (tuple, list)):
        if len(data_init) != 2:
            raise ValueError(
                f"data_init tuple must have 2 entries (one per code), got {len(data_init)}"
            )
        sl, sr = data_init
        spec_l = sl * n_l if len(sl) == 1 else sl
        spec_r = sr * n_r if len(sr) == 1 else sr
    else:
        raise TypeError(
            f"data_init must be str, tuple, list, or None; got {type(data_init).__name__}"
        )
    if len(spec_l) != n_l:
        raise ValueError(f"data_init left length {len(spec_l)} != n_l {n_l}")
    if len(spec_r) != n_r:
        raise ValueError(f"data_init right length {len(spec_r)} != n_r {n_r}")
    if left_is_x:
        spec_l = "".join(_H_DATA_INIT[c] for c in spec_l)
    else:
        spec_r = "".join(_H_DATA_INIT[c] for c in spec_r)
    return (spec_l, spec_r)


def build_joint_ppm_circuit(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
    *,
    rounds: int,
    noise_model: NoiseModel | None = None,
    data_init: str | tuple[str, ...] | list[str] | None = None,
    mixed_strategy: str = "hadamard_dual",
) -> tuple[stim.Circuit, QuditCode]:
    """Joint-PPM circuit (universal adapter; no U_B in α*).

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

    ``mixed_strategy`` (mixed-basis joints only): selects how X̄ ⊗ Z̄-style
    mixed measurements are realized.

      * ``"hadamard_dual"`` (default) — rotate the X-side to a Z-type dual code
        via a transversal/local Hadamard frame (SJOY24 / Swaroop et al.
        arXiv:2410.03628 §II), then run the EXISTING CSS same-basis bridge path
        (Cross, He, Rall, Yoder arXiv:2407.18393 §3.7 gauging measurement).
        Fully CSS: all ancillas in |0⟩/|+⟩ measured in Z/X, no |Y_+⟩, no
        null-space final-detector synthesis. obs0 = ∏ m_v over in-circuit
        vertex checks.
      * ``"cohen"`` — legacy Cohen–Kim–Bartlett–Brown arXiv:2110.10794 |Y_+⟩
        cross-merge path. Retained for comparison during bring-up; uses RY/MY
        and a null-space combination-detector block. Will be removed.
    """
    if bridge.basis_l is bridge.basis_r:
        joint_code, bridge = _stitch_to_joint_code(g_l, g_r, bridge)
        return _build_joint_ppm_circuit_same_basis(
            g_l, g_r, bridge, joint_code,
            rounds=rounds, noise_model=noise_model, data_init=data_init,
        )

    if mixed_strategy == "hadamard_dual":
        g_l_rot, g_r_rot, bridge_rot, left_is_x = _rotate_x_side_to_z(g_l, g_r, bridge)
        n_l = g_l_rot.code.num_qudits
        n_r = g_r_rot.code.num_qudits
        rot_data_init = _h_transform_left_data_init(
            data_init, n_l, n_r, left_is_x=left_is_x
        )
        joint_code, bridge_rot = _stitch_to_joint_code(g_l_rot, g_r_rot, bridge_rot)
        return _build_joint_ppm_circuit_same_basis(
            g_l_rot, g_r_rot, bridge_rot, joint_code,
            rounds=rounds, noise_model=noise_model, data_init=rot_data_init,
        )
    if mixed_strategy != "cohen":
        raise ValueError(
            f"mixed_strategy must be 'hadamard_dual' or 'cohen', got {mixed_strategy!r}"
        )

    # Legacy Cohen |Y_+⟩ path (kept reachable during bring-up per design constraint).
    joint_code, bridge, layout = _build_mixed_basis_joint_code(g_l, g_r, bridge)
    return _build_joint_ppm_circuit_mixed_basis(
        g_l, g_r, bridge, joint_code,
        layout=layout,
        rounds=rounds, noise_model=noise_model, data_init=data_init,
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
    ancilla_ids = qubit_ids.data[n_l + n_r : n_l + n_r + k_l + k_r]
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
        ancilla_ids,
        bridge_ids,
        data_init=expanded_data_init,
    )
    qec_cycle, measurement_record, _ = _surgery_qec_cycle_joint(
        g_l,
        g_r,
        joint_code,
        bridge,
        num_rounds=rounds,
        qubit_ids=qubit_ids,
        intercode=intercode,
    )
    circuit += qec_cycle
    circuit += _surgery_detach_and_readout(
        g_l,
        data_ids=data_ids,
        ancilla_ids=ancilla_ids,
        bridge_ids=bridge_ids,
        measurement_record=measurement_record,
    )
    circuit += _surgery_final_detectors_joint(
        g_l,
        g_r,
        joint_code,
        bridge,
        qubit_ids,
        measurement_record=measurement_record,
        intercode=intercode,
    )

    # χ check IDs: data H_X^(l) rows occupy first mX_l indices in
    # qubit_ids.checks_x, then m_X_r (inter-code), then χ^(l), then χ^(r).
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
    )

    if noise_model is not None:
        circuit = noise_model.noisy_circuit(circuit)
    return circuit, joint_code


def _split_quditcode_into_virtual_cssc(
    joint_code: QuditCode,
) -> tuple[CSSCode, np.ndarray, np.ndarray, list[int], list[int], list[int]]:
    """Partition ``joint_code.matrix`` rows by Pauli type into a virtual CSS subset.

    Mixed-basis subsystem-code helper (Cross, He, Rall, Yoder
    arXiv:2407.18393 Theorem 20 / Cowtan, He, Williamson, Yoder
    arXiv:2503.05003 §3.5 / Webster, Smith, Cohen arXiv:2511.15989 §II.B.2).
    The merged code is a SUBSYSTEM code: not all pairs of rows commute.

    The stitch ``_stitch_to_joint_code_mixed`` produces three row classes:
      * pure-X rows (HX block): data X-stabilizers + comp_r-side rows +
        cycle_r if basis_r=Z.
      * pure-Z rows (HZ block): symmetric on Z side.
      * mixed Y rows (Y_stab block): one per merge qubit from the Webster
        cross-merge. Their X-part comes from a χ_l row, Z-part from a χ_r
        row, with single-{q} adapter support on each side.

    This helper builds a "virtual" CSSCode from the pure-X / pure-Z subsets
    for reuse of the CSS syndrome-extraction pipeline (EdgeColoring). Y rows
    are reported separately via ``mixed_row_indices`` and are handled by
    dedicated CY/CZ-based extraction in the mixed-basis pipeline.

    Returns
    -------
    virtual_cssc
        ``CSSCode`` whose ``matrix_x`` is the pure-X subset, ``matrix_z`` is
        the pure-Z subset.
    HX
        Pure-X rows as an X-only matrix (shape (n_x, n_qudits)).
    HZ
        Pure-Z rows as a Z-only matrix (shape (n_z, n_qudits)).
    x_row_indices
        Indices into ``joint_code.matrix`` of the pure-X rows, in insertion
        order — used to locate χ_l ancilla IDs in the resulting qubit_ids.
    z_row_indices
        Indices of the pure-Z rows.
    mixed_row_indices
        Indices of any rows with both X and Z support (Y rows from the
        Webster cross-merge).
    """
    H = np.asarray(joint_code.matrix).astype(np.int_)
    n = joint_code.num_qudits
    Hx = H[:, :n]
    Hz = H[:, n:]
    x_mask = Hx.any(axis=1) & ~Hz.any(axis=1)
    z_mask = Hz.any(axis=1) & ~Hx.any(axis=1)
    mixed_mask = Hx.any(axis=1) & Hz.any(axis=1)

    x_idx = [int(i) for i in np.flatnonzero(x_mask)]
    z_idx = [int(i) for i in np.flatnonzero(z_mask)]
    mixed_idx = [int(i) for i in np.flatnonzero(mixed_mask)]

    HX = Hx[x_mask].astype(np.uint8)
    HZ = Hz[z_mask].astype(np.uint8)

    field = joint_code.field
    virtual_cssc = CSSCode(field(HX), field(HZ), is_subsystem_code=False)
    return virtual_cssc, HX, HZ, x_idx, z_idx, mixed_idx


def _mixed_basis_qubit_coords(
    n_data: int,
    qubit_ids: QubitIDs,
    y_ancilla_ids: tuple[int, ...] = (),
) -> stim.Circuit:
    """Emit a simple sequential QUBIT_COORDS layout for mixed-basis joint PPM.

    Lanes: y=0 data, y=6 ancilla+bridge, y=2 X-checks, y=4 Z-checks,
    y=3 Y-stab ancillas (cross-merge per Webster, Smith, Cohen
    arXiv:2511.15989 §II.B.2).
    """
    circuit = stim.Circuit()
    for i, qid in enumerate(qubit_ids.data):
        lane = 0 if i < n_data else 6
        circuit.append("QUBIT_COORDS", qid, (i, lane))
    for i, qid in enumerate(qubit_ids.checks_x):
        circuit.append("QUBIT_COORDS", qid, (i, 2))
    for i, qid in enumerate(qubit_ids.checks_z):
        circuit.append("QUBIT_COORDS", qid, (i, 4))
    for i, qid in enumerate(y_ancilla_ids):
        circuit.append("QUBIT_COORDS", qid, (i, 3))
    return circuit


def _mixed_basis_state_prep(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
    data_l_ids: tuple[int, ...],
    data_r_ids: tuple[int, ...],
    ancilla_l_ids: tuple[int, ...],
    ancilla_r_ids: tuple[int, ...],
    bridge_ids: tuple[int, ...],
    *,
    data_init_l: str,
    data_init_r: str,
) -> stim.Circuit:
    """Init data + ancillas + bridge for the mixed-basis joint PPM.

    Per-side rules (mirroring ``_surgery_state_prep``):
      * data_s: per-qubit init from data_init_s string ('0','1','+','-').
      * ancilla_s: basis_s-complement +1 eigenstate
        (basis_s = X → |0⟩ via R; basis_s = Z → |+⟩ via RX).
      * bridge (adapter) qubits: initialized in basis_l-complement
        deterministically. The cross-merge / bridge-gauge structure
        makes this choice arbitrary up to gauge — basis_l is selected
        to match the placement convention in
        ``_stitch_to_joint_code_mixed`` (cycle_l in the basis_l-dual).
    """
    circuit = stim.Circuit()

    def _emit_data(ids: tuple[int, ...], spec: str) -> None:
        r_ids, rx_ids, x_after, z_after = [], [], [], []
        for q, c in zip(ids, spec):
            if c == "0":
                r_ids.append(q)
            elif c == "1":
                r_ids.append(q)
                x_after.append(q)
            elif c == "+":
                rx_ids.append(q)
            elif c == "-":
                rx_ids.append(q)
                z_after.append(q)
            else:
                raise ValueError(f"invalid data_init char {c!r}; must be one of 0/1/+/-")
        if r_ids:
            circuit.append("R", r_ids)
        if rx_ids:
            circuit.append("RX", rx_ids)
        if x_after:
            circuit.append("X", x_after)
        if z_after:
            circuit.append("Z", z_after)

    _emit_data(data_l_ids, data_init_l)
    _emit_data(data_r_ids, data_init_r)

    l_complement = "R" if bridge.basis_l is Pauli.X else "RX"
    r_complement = "R" if bridge.basis_r is Pauli.X else "RX"
    if ancilla_l_ids:
        circuit.append(l_complement, list(ancilla_l_ids))
    if ancilla_r_ids:
        circuit.append(r_complement, list(ancilla_r_ids))
    # Bridge prep:
    #   * Same-basis (basis_l is basis_r): basis_l-complement init matches the
    #     adapter stabilizer choice in ``_stitch_to_joint_code_mixed``.
    #   * Mixed-basis (basis_l ≠ basis_r): |Y_+⟩^⊗w so that ∏ Y_{a_q} is a
    #     stabilizer with eigenvalue +1, absorbing the adapter Y residual per
    #     main.tex §4.5 Eq. eq:obs0-corrected (Cohen-Kim-Bartlett-Brown
    #     arXiv:2110.10794 §II.B.2 / Fig. 4 |Y⟩ ancilla protocol).
    if bridge_ids:
        if bridge.basis_l is bridge.basis_r:
            circuit.append(l_complement, list(bridge_ids))
        else:
            circuit.append("RY", list(bridge_ids))
    return circuit


def _compute_stabilizer_center_mask(H_sym: np.ndarray, n: int) -> np.ndarray:
    """Mark each row of the symplectic matrix that commutes with ALL other rows.

    Returned mask is True for rows in the (algebraic) stabilizer center —
    measurements of these rows commute with every other gauge generator, so
    their syndrome-extraction outcomes are deterministic given any pure
    stabilizer state. Rows outside the center are gauge operators with
    random outcomes; we must NOT register detectors for them.
    """
    if H_sym.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    Hx = H_sym[:, :n].astype(np.int_)
    Hz = H_sym[:, n:].astype(np.int_)
    comm = (Hx @ Hz.T + Hz @ Hx.T) % 2
    # Zero out diagonal (a row trivially commutes with itself).
    np.fill_diagonal(comm, 0)
    return np.asarray(~comm.any(axis=1))


def _build_joint_ppm_circuit_mixed_basis(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
    joint_code: QuditCode,
    *,
    layout: JointPPMLayout,
    rounds: int,
    noise_model: NoiseModel | None,
    data_init: str | tuple[str, ...] | list[str] | None,
) -> tuple[stim.Circuit, QuditCode]:
    """Mixed-basis joint PPM circuit (subsystem merged code).

    Cross, He, Rall, Yoder (arXiv:2407.18393) Theorem 20 / Cowtan, He,
    Williamson, Yoder (arXiv:2503.05003 §3.5) / Webster, Smith, Cohen
    (arXiv:2511.15989 §II.B.2) construction. The merged code is a
    subsystem code with anti-commuting gauge generators (the SkipTree
    cycle rows on opposite Pauli types overlap on the adapter via
    H_R · H_R^T).

    Pipeline:
      1. _build_mixed_basis_joint_code (joint_layout.py) builds the merged
         QuditCode following docs/superpowers/docs/main.tex §4.2/§4.3
         block-by-block, returning a JointPPMLayout with per-row provenance.
      2. _split_quditcode_into_virtual_cssc partitions the joint-code matrix
         into pure-X / pure-Z rows (used by EdgeColoring) and Y-type rows
         (from the §4.3 cross-merge).
      3. Allocate ancillas: QubitIDs.from_code(virtual_cssc) for the CSS
         subset, then additional Y ancillas appended.
      4. Per-side state prep + detach (different bases for l / r).
      5. Per-round QEC: split X / Z / Y phases for determinism per
         Cohen-Kim-Bartlett-Brown arXiv:2110.10794 §II.B.2.
      6. obs0 = ⊕ m(χ_l) ⊕ ⊕ m(χ_r) ⊕ ⊕ m(y_q) per Lemma 2 of the design
         spec — implemented via JointPPMLayout row provenance. For the
         degenerate fixture where every V_0 vertex is a port (e.g.
         Steane × Steane), surviving χ rows are empty and obs0 reduces to
         ⊕ m(y_q) alone, which has a residual ∏ Y on the adapter; in that
         regime obs0 emission is suppressed (test_mixed_basis_circuit_
         compiles_to_dem passes vacuously, truth-table test stays xfail).
    """
    intercode = g_l.code is not g_r.code
    g_l_aug, g_r_aug = bridge.g_l_aug, bridge.g_r_aug
    n_l = g_l.code.num_qudits
    n_r = g_r.code.num_qudits if intercode else 0
    k_l = g_l_aug.incidence.shape[0]
    k_r = g_r_aug.incidence.shape[0]
    w = bridge.width

    virtual_cssc, _HX, _HZ, x_row_idx, z_row_idx, mixed_row_idx = (
        _split_quditcode_into_virtual_cssc(joint_code)
    )

    qubit_ids = QubitIDs.from_code(virtual_cssc)
    # Allocate Y-row ancillas appended after the virtual_cssc's check ids.
    n_Y = len(mixed_row_idx)
    if n_Y:
        max_id = max(qubit_ids.all_qubits) if qubit_ids.all_qubits else -1
        y_ancilla_ids: tuple[int, ...] = tuple(range(max_id + 1, max_id + 1 + n_Y))
    else:
        y_ancilla_ids = ()
    n_data_total = n_l + n_r if intercode else n_l

    if intercode:
        data_l_ids = qubit_ids.data[:n_l]
        data_r_ids = qubit_ids.data[n_l : n_l + n_r]
    else:
        data_l_ids = qubit_ids.data[:n_l]
        data_r_ids = data_l_ids  # shared
    ancilla_l_ids = qubit_ids.data[n_data_total : n_data_total + k_l]
    ancilla_r_ids = qubit_ids.data[n_data_total + k_l : n_data_total + k_l + k_r]
    bridge_ids = qubit_ids.data[n_data_total + k_l + k_r :]
    assert len(bridge_ids) == w

    # Normalize data_init to per-side strings.
    if data_init is None:
        spec_l = ("+" if bridge.basis_l is Pauli.X else "0") * n_l
        spec_r = ("+" if bridge.basis_r is Pauli.X else "0") * n_r if intercode else spec_l
    elif isinstance(data_init, str):
        if not intercode:
            spec_l = data_init * n_l if len(data_init) == 1 else data_init
            spec_r = spec_l
        else:
            if len(data_init) == 1:
                spec_l = data_init * n_l
                spec_r = data_init * n_r
            else:
                spec_l = data_init[:n_l]
                spec_r = data_init[n_l:]
    elif isinstance(data_init, (tuple, list)):
        if not intercode:
            raise ValueError(
                "tuple/list data_init only valid for intercode joint PPM"
            )
        if len(data_init) != 2:
            raise ValueError(
                f"data_init tuple must have 2 entries, got {len(data_init)}"
            )
        sl, sr = data_init
        spec_l = sl * n_l if len(sl) == 1 else sl
        spec_r = sr * n_r if len(sr) == 1 else sr
    else:
        raise TypeError(f"data_init has unsupported type {type(data_init).__name__}")
    if len(spec_l) != n_l:
        raise ValueError(f"data_init left length {len(spec_l)} != n_l {n_l}")
    if intercode and len(spec_r) != n_r:
        raise ValueError(f"data_init right length {len(spec_r)} != n_r {n_r}")

    circuit = _mixed_basis_qubit_coords(n_data_total, qubit_ids, y_ancilla_ids)
    circuit += _mixed_basis_state_prep(
        g_l, g_r, bridge,
        data_l_ids=data_l_ids,
        data_r_ids=data_r_ids if intercode else (),
        ancilla_l_ids=ancilla_l_ids,
        ancilla_r_ids=ancilla_r_ids,
        bridge_ids=bridge_ids,
        data_init_l=spec_l,
        data_init_r=spec_r,
    )

    # Multi-round QEC via *split* EdgeColoring on the virtual CSS subset.
    #
    # IMPORTANT (mixed-basis determinism). A naive call to
    # ``EdgeColoring().get_circuit(virtual_cssc, qubit_ids)`` initializes
    # ALL ancillas in |+⟩ at the start of the round, fires the X-stab CX
    # subgraph, then the Z-stab CZ subgraph, and finally measures ALL
    # ancillas in X-basis. Because all X-ancillas remain in superposition
    # while the Z-stab CZ gates fire, the X-CX gates entangle the data
    # in a way that can rotate the adapter qubit's Z eigenvalue before
    # the Z-stab ancillas record it. The result is that χ_r outcomes
    # become non-deterministic shot-to-shot whenever χ_l shares an
    # adapter qubit with χ_r — the canonical failure mode noted in the
    # Tier 1 design docstring.
    #
    # Per Cohen, Kim, Bartlett, Brown arXiv:2110.10794 §II.B.2 (mixed-
    # basis joint PPM) and Cross, He, Rall, Yoder arXiv:2407.18393
    # Theorem 20 (subsystem-code construction), the X-type and Z-type
    # gauge measurements must be scheduled in SEPARATE non-overlapping
    # circuit phases. We split the per-round circuit into:
    #   • X-phase: RX(X-ancillas) → CX gates → MX(X-ancillas)
    #   • Z-phase: RX(Z-ancillas) → CZ gates → MX(Z-ancillas)
    # X-ancillas collapse before Z-CZ gates fire, so the data is in a
    # definite X-stabilizer eigenstate when the Z-phase starts. With
    # this schedule the individual χ_l (X-type) and χ_r (Z-type) gauge
    # measurements become deterministic in the subsystem-code sense.
    qubit_ids_x = QubitIDs(data=qubit_ids.data, check=qubit_ids.checks_x)
    qubit_ids_x.checks_x = qubit_ids.checks_x
    qubit_ids_z = QubitIDs(data=qubit_ids.data, check=qubit_ids.checks_z)
    qubit_ids_z.checks_z = qubit_ids.checks_z

    field = joint_code.field
    n_qudits = joint_code.num_qudits
    HX_only = _HX if _HX.shape[0] else np.zeros((0, n_qudits), dtype=np.uint8)
    HZ_only = _HZ if _HZ.shape[0] else np.zeros((0, n_qudits), dtype=np.uint8)
    virtual_cssc_X = CSSCode(
        field(HX_only),
        field(np.zeros((0, n_qudits), dtype=np.uint8)),
        is_subsystem_code=False,
    )
    virtual_cssc_Z = CSSCode(
        field(np.zeros((0, n_qudits), dtype=np.uint8)),
        field(HZ_only),
        is_subsystem_code=False,
    )

    strategy = EdgeColoring()
    if HX_only.shape[0]:
        x_phase_circuit, x_phase_record = strategy.get_circuit(virtual_cssc_X, qubit_ids_x)
    else:
        x_phase_circuit, x_phase_record = stim.Circuit(), MeasurementRecord()
    if HZ_only.shape[0]:
        z_phase_circuit, z_phase_record = strategy.get_circuit(virtual_cssc_Z, qubit_ids_z)
    else:
        z_phase_circuit, z_phase_record = stim.Circuit(), MeasurementRecord()

    # Y-row syndrome extraction phase. Each Y-stab row gets a dedicated
    # ancilla initialized in |+⟩ (RX); CX/CY/CZ gates entangle the ancilla
    # with the data depending on the Pauli at each qubit; finally MX
    # collapses the ancilla to record the eigenvalue.
    y_phase_circuit = stim.Circuit()
    y_phase_record = MeasurementRecord()
    if n_Y:
        H_full = np.asarray(joint_code.matrix).astype(np.int_)
        n_q = joint_code.num_qudits
        if y_ancilla_ids:
            y_phase_circuit.append("RX", list(y_ancilla_ids))
        # Emit per-Pauli gate lists (collected then appended in canonical order).
        for y_anc, orig_row_idx in zip(y_ancilla_ids, mixed_row_idx):
            row = H_full[orig_row_idx]
            x_part = row[:n_q]
            z_part = row[n_q:]
            cx_pairs: list[int] = []
            cy_pairs: list[int] = []
            cz_pairs: list[int] = []
            for q in range(n_q):
                xq, zq = int(x_part[q]), int(z_part[q])
                if xq == 1 and zq == 0:
                    cx_pairs.extend([y_anc, qubit_ids.data[q]])
                elif xq == 0 and zq == 1:
                    cz_pairs.extend([y_anc, qubit_ids.data[q]])
                elif xq == 1 and zq == 1:
                    cy_pairs.extend([y_anc, qubit_ids.data[q]])
            if cx_pairs:
                y_phase_circuit.append("CX", cx_pairs)
            if cy_pairs:
                y_phase_circuit.append("CY", cy_pairs)
            if cz_pairs:
                y_phase_circuit.append("CZ", cz_pairs)
        if y_ancilla_ids:
            y_phase_circuit.append("MX", list(y_ancilla_ids))
            y_phase_record.append({q: i for i, q in enumerate(y_ancilla_ids)})

    one_round = stim.Circuit()
    one_round += x_phase_circuit
    one_round += z_phase_circuit
    one_round += y_phase_circuit
    round_measurement_record = MeasurementRecord()
    round_measurement_record.append(x_phase_record)
    round_measurement_record.append(z_phase_record)
    round_measurement_record.append(y_phase_record)

    measurement_record = MeasurementRecord()

    # Determine which check IDs belong to rows in the algebraic stabilizer
    # center — i.e. rows that commute with every other gauge generator.
    # Only these are safe to register as detectors (deterministic outcomes).
    H_sym = np.asarray(joint_code.matrix).astype(np.int_)
    center_mask = _compute_stabilizer_center_mask(H_sym, joint_code.num_qudits)
    # Map original row index in joint_code.matrix → measurement-record ancilla ID.
    row_to_check: dict[int, int] = {}
    for slot, orig in enumerate(x_row_idx):
        row_to_check[orig] = qubit_ids.checks_x[slot]
    for slot, orig in enumerate(z_row_idx):
        row_to_check[orig] = qubit_ids.checks_z[slot]
    for slot, orig in enumerate(mixed_row_idx):
        row_to_check[orig] = y_ancilla_ids[slot]
    center_check_ids = tuple(
        row_to_check[orig] for orig in row_to_check if center_mask[orig]
    )

    # Per-qubit init Pauli + sign (one stabilizer per product-state qubit).
    # Used by round-1 + final detector emission below to identify center rows
    # whose joint eigenvalue is deterministic on the prepared state.
    #   data_l[i]/data_r[i]: spec char → +X (|+⟩), -X (|-⟩), +Z (|0⟩), -Z (|1⟩)
    #   ancilla_{l,r}: basis-complement init → +Z (|0⟩) or +X (|+⟩)
    #   bridge: same-basis → basis_l-complement; mixed-basis → +Y (|Y_+⟩, RY)
    _data_init_to_pauli: dict[str, tuple[Pauli, int]] = {
        "+": (Pauli.X, +1), "-": (Pauli.X, -1),
        "0": (Pauli.Z, +1), "1": (Pauli.Z, -1),
    }
    qubit_init: dict[int, tuple[Pauli, int]] = {}
    for i, qid in enumerate(data_l_ids):
        qubit_init[qid] = _data_init_to_pauli[spec_l[i]]
    if intercode:
        for i, qid in enumerate(data_r_ids):
            qubit_init[qid] = _data_init_to_pauli[spec_r[i]]
    _anc_l_pauli = Pauli.Z if bridge.basis_l is Pauli.X else Pauli.X
    _anc_r_pauli = Pauli.Z if bridge.basis_r is Pauli.X else Pauli.X
    for qid in ancilla_l_ids:
        qubit_init[qid] = (_anc_l_pauli, +1)
    for qid in ancilla_r_ids:
        qubit_init[qid] = (_anc_r_pauli, +1)
    if bridge.basis_l is bridge.basis_r:
        _bridge_init_pauli = _anc_l_pauli
    else:
        _bridge_init_pauli = Pauli.Y
    for qid in bridge_ids:
        qubit_init[qid] = (_bridge_init_pauli, +1)

    # Per-qubit destructive readout basis (used by final detector emission).
    qubit_final_meas: dict[int, Pauli] = {}
    for qid in data_l_ids:
        qubit_final_meas[qid] = bridge.basis_l  # MX → X, M → Z
    if intercode:
        for qid in data_r_ids:
            qubit_final_meas[qid] = bridge.basis_r
    for qid in ancilla_l_ids:
        qubit_final_meas[qid] = _anc_l_pauli
    for qid in ancilla_r_ids:
        qubit_final_meas[qid] = _anc_r_pauli
    for qid in bridge_ids:
        qubit_final_meas[qid] = _bridge_init_pauli

    def _row_pauli_per_qubit(orig_row: int) -> dict[int, Pauli]:
        """Return {data-col → Pauli} for non-I support of joint matrix row."""
        H_full_ = np.asarray(joint_code.matrix).astype(np.int_)
        row = H_full_[orig_row]
        out: dict[int, Pauli] = {}
        for q in range(joint_code.num_qudits):
            xq, zq = int(row[q]), int(row[q + joint_code.num_qudits])
            if xq == 0 and zq == 0:
                continue
            if xq == 1 and zq == 0:
                out[q] = Pauli.X
            elif xq == 0 and zq == 1:
                out[q] = Pauli.Z
            else:
                out[q] = Pauli.Y
        return out

    # Center rows mapped to (check_id, per-qubit Pauli support).
    center_rows: list[tuple[int, int, dict[int, Pauli]]] = [
        (orig_row, row_to_check[orig_row], _row_pauli_per_qubit(orig_row))
        for orig_row in row_to_check
        if center_mask[orig_row]
    ]

    # Classify center rows that are deterministic on the prepared state.
    # Round-1 single-target detector fires iff sign = +1 (so noiseless
    # measurement outcome is 0). Sign = -1 rows are still caught by the
    # inter-round diff detectors, so skipping them here is safe.
    round1_reliable_check_ids: list[int] = []
    for _, cid, row_paulis in center_rows:
        sign = 1
        ok = True
        for q, pauli_q in row_paulis.items():
            qid = qubit_ids.data[q]
            init_pauli, init_sign = qubit_init[qid]
            if pauli_q is not init_pauli:
                ok = False
                break
            sign *= init_sign
        if ok and sign == 1:
            round1_reliable_check_ids.append(cid)

    circuit += one_round
    measurement_record.append(round_measurement_record)
    for cid in round1_reliable_check_ids:
        circuit.append(
            "DETECTOR",
            [measurement_record.get_target_rec(cid)],
            (cid, 0, 0),
        )

    if rounds > 1:
        repeat = one_round.copy()
        measurement_record.append(round_measurement_record)
        repeat.append("SHIFT_COORDS", [], (0, 0, 1))
        for cid in center_check_ids:
            repeat.append(
                "DETECTOR",
                [
                    measurement_record.get_target_rec(cid, -1),
                    measurement_record.get_target_rec(cid, -2),
                ],
                (cid, 0, 0),
            )
        circuit.append(stim.CircuitRepeatBlock(rounds - 1, repeat))
        measurement_record.append(round_measurement_record, repeat=rounds - 2)

    # Detach + destructive readout per side. Mixed-basis: each side uses its
    # own basis-complement ancilla measurement and its own basis-aligned data
    # measurement.
    l_anc_op = "M" if bridge.basis_l is Pauli.X else "MX"
    r_anc_op = "M" if bridge.basis_r is Pauli.X else "MX"
    l_data_op = "MX" if bridge.basis_l is Pauli.X else "M"
    r_data_op = "MX" if bridge.basis_r is Pauli.X else "M"

    # Measure ancillas + bridge.
    detach_l = list(ancilla_l_ids)
    detach_r = list(ancilla_r_ids)
    if detach_l:
        circuit.append(l_anc_op, detach_l)
        measurement_record.append({q: i for i, q in enumerate(detach_l)})
    if detach_r:
        circuit.append(r_anc_op, detach_r)
        measurement_record.append({q: i for i, q in enumerate(detach_r)})
    if bridge_ids:
        # Mixed-basis: bridge measured in Y basis (matches |Y_+⟩ prep) so that
        # ∏ m(a_q) gives the ∏ Y_{a_q} eigenvalue per main.tex §4.5
        # Eq. eq:obs0-corrected. Same-basis: matches basis_l-complement init.
        if bridge.basis_l is bridge.basis_r:
            bridge_meas_op = l_anc_op
        else:
            bridge_meas_op = "MY"
        circuit.append(bridge_meas_op, list(bridge_ids))
        measurement_record.append({q: i for i, q in enumerate(bridge_ids)})
    circuit.append("SHIFT_COORDS", [], (0, 0, 1))

    # Measure data — left in basis_l, right in basis_r.
    circuit.append(l_data_op, list(data_l_ids))
    measurement_record.append({q: i for i, q in enumerate(data_l_ids)})
    if intercode:
        circuit.append(r_data_op, list(data_r_ids))
        measurement_record.append({q: i for i, q in enumerate(data_r_ids)})

    # Final detectors. A center row's eigenvalue can be reconstructed from the
    # destructive readout only when every non-I Pauli of the row matches that
    # qubit's fixed destructive measurement basis (data_l→basis_l, data_r→
    # basis_r, κ_s→gadget basis, adapter→Y). For those directly compatible
    # rows we emit the row-by-row detector
    #   DETECTOR( XOR per-qubit destructive readouts on supp(r)
    #             ⊕ last-round in-circuit ancilla measurement of r ).
    #
    # In the mixed-basis (non-CSS) merge this is NOT enough: the cycle rows
    # (H_R blocks of BOTH H̃_X^joint and H̃_Z^joint) deposit X_{a_q} / Z_{a_q}
    # on the adapter, which §4.5 forces to be measured in Y, so they are not
    # individually reconstructable; yet their last-round outcomes (and the
    # adapter Y readouts) enter obs0 and must be pinned. We therefore also emit
    # detectors for *linear combinations* of center rows whose incompatible
    # parts cancel into a readout-compatible support — exactly the null space of
    # the per-qubit incompatibility constraint F. Without these combination
    # detectors a single fault on the Y-measured adapter flips obs0 with no
    # detector firing (operational distance collapses to 1). See main.tex §4.6.
    H_full = np.asarray(joint_code.matrix).astype(np.int_)
    center_idx = [orig for orig in row_to_check if center_mask[orig]]
    if center_idx:
        import galois as _galois
        F2 = _galois.GF(2)
        C = H_full[center_idx]  # (n_center, 2n)
        n_q = joint_code.num_qudits

        def _row_destructive_compatible(combined_row: np.ndarray) -> bool:
            for q in range(n_q):
                xq, zq = int(combined_row[q]), int(combined_row[q + n_q])
                if xq == 0 and zq == 0:
                    continue
                P = qubit_final_meas[qubit_ids.data[q]]
                if P is Pauli.X and (xq, zq) != (1, 0):
                    return False
                if P is Pauli.Z and (xq, zq) != (0, 1):
                    return False
                if P is Pauli.Y and (xq, zq) != (1, 1):
                    return False
            return True

        def _emit_combo_detector(c_int: np.ndarray) -> None:
            combined = (c_int @ C) % 2
            targets: list[stim.GateTarget] = []
            for q in range(n_q):
                xq, zq = int(combined[q]), int(combined[q + n_q])
                if xq == 0 and zq == 0:
                    continue
                targets.append(measurement_record.get_target_rec(qubit_ids.data[q]))
            for slot, ci in enumerate(c_int):
                if ci:
                    cid_slot = row_to_check[center_idx[slot]]
                    targets.append(measurement_record.get_target_rec(cid_slot, -1))
            if targets:
                circuit.append("DETECTOR", targets, (0, 0, 0))

        # Build constraint matrix F such that F @ row_sym = 0 iff row is
        # destructive-compatible (one constraint per qubit).
        F_rows: list[np.ndarray] = []
        for q in range(n_q):
            row_vec = np.zeros(2 * n_q, dtype=np.uint8)
            P = qubit_final_meas[qubit_ids.data[q]]
            if P is Pauli.X:
                row_vec[q + n_q] = 1
            elif P is Pauli.Z:
                row_vec[q] = 1
            else:  # Pauli.Y
                row_vec[q] = 1
                row_vec[q + n_q] = 1
            F_rows.append(row_vec)
        F_mat = np.stack(F_rows)
        A = F2((C @ F_mat.T) % 2)
        null_basis = np.asarray(A.T.null_space()).astype(np.int_)
        # Rows of null_basis span all destructive-compatible combos.

        emitted_for: set[int] = set()
        for slot, orig in enumerate(center_idx):
            if orig in emitted_for:
                continue
            single_row = C[slot]
            if _row_destructive_compatible(single_row):
                c = np.zeros(len(center_idx), dtype=np.int_)
                c[slot] = 1
                _emit_combo_detector(c)
                emitted_for.add(orig)
                continue
            # Row's Pauli is incompatible with the destructive readout
            # basis somewhere (e.g. Y_stab row with X-on-left-ancilla).
            # Pick a min-weight null-space vector that includes this row;
            # the combination cancels the incompatible parts. One detector
            # per Y_stab row keeps the Tanner graph bounded.
            cands = [(int(v.sum()), v) for v in null_basis if int(v[slot]) == 1]
            if not cands:
                continue  # row genuinely unreachable via destructive readouts
            cands.sort(key=lambda x: x[0])
            best_c = cands[0][1].astype(np.int_)
            _emit_combo_detector(best_c)
            for s2, val in enumerate(best_c):
                if val:
                    emitted_for.add(center_idx[s2])

    # obs0 per main.tex §4.5 Eq. eq:obs0-corrected:
    #   Z̄_l ⊗ X̄_r = ∏ χ_l surviving · ∏ χ_r surviving · ∏ y_q · ∏ Y_{a_q}
    # The last product (∏ Y_{a_q}) over bridge destructive Y-basis readouts
    # closes the adapter Y residual that arises in the mixed-basis case
    # (Cohen-Kim-Bartlett-Brown arXiv:2110.10794 §II.B.2 / Fig. 4 |Y⟩-ancilla
    # protocol). With |Y_+⟩^⊗w bridge init (RY) and MY detach, this term is
    # deterministic and produces a deterministic obs0 even in the degenerate
    # V_0 = ports regime (e.g. Steane × Steane), where rows_chi['l'] and
    # rows_chi['r'] are both empty.
    obs0_check_ids: list[int] = []

    # Compute layout-row → joint-row → virtual_cssc-row → check-id mappings.
    # layout.rows_chi[side] are indices into layout.H_X (if side's basis is X)
    # or layout.H_Z (if Z). to_quditcode stacks [H_X | 0] then [0 | H_Z] then
    # H_Y, so joint_row = layout_H_X_row for H_X rows or N_X + layout_H_Z_row
    # for H_Z rows. _split_quditcode_into_virtual_cssc then re-partitions
    # joint rows by Pauli type into x_row_idx / z_row_idx lists, where
    # qubit_ids.checks_x[i] corresponds to joint_row = x_row_idx[i]. We
    # invert that map here so a layout-row routes to the correct ancilla ID.
    N_X = layout.H_X.shape[0]
    x_inv = {jr: i for i, jr in enumerate(x_row_idx)}
    z_inv = {jr: i for i, jr in enumerate(z_row_idx)}

    def _layout_x_row_to_check_id(layout_row_idx: int) -> int | None:
        """Map layout.H_X row index to check_id via joint row + virtual_cssc split."""
        joint_row = layout_row_idx  # H_X rows are first in joint matrix
        if joint_row not in x_inv:
            return None  # row was not pure-X (split moved it somewhere else)
        return qubit_ids.checks_x[x_inv[joint_row]]

    def _layout_z_row_to_check_id(layout_row_idx: int) -> int | None:
        """Map layout.H_Z row index to check_id via joint row + virtual_cssc split."""
        joint_row = N_X + layout_row_idx  # H_Z rows come after H_X
        if joint_row not in z_inv:
            return None
        return qubit_ids.checks_z[z_inv[joint_row]]

    for row_idx in layout.rows_chi["l"]:
        if layout.basis_l is Pauli.X:
            cid = _layout_x_row_to_check_id(row_idx)
        else:
            cid = _layout_z_row_to_check_id(row_idx)
        if cid is not None:
            obs0_check_ids.append(cid)
    for row_idx in layout.rows_chi["r"]:
        if layout.basis_r is Pauli.X:
            cid = _layout_x_row_to_check_id(row_idx)
        else:
            cid = _layout_z_row_to_check_id(row_idx)
        if cid is not None:
            obs0_check_ids.append(cid)
    for y_idx in layout.rows_y:
        obs0_check_ids.append(y_ancilla_ids[y_idx])
    # ∏ Y_{a_q}: bridge destructive Y-basis readouts — closes the adapter
    # residual per main.tex §4.5 Eq. eq:obs0-corrected.
    for bid in bridge_ids:
        obs0_check_ids.append(bid)

    if obs0_check_ids:
        obs0_targets = [
            measurement_record.get_target_rec(cid) for cid in obs0_check_ids
        ]
        circuit.append("OBSERVABLE_INCLUDE", obs0_targets, 0)

    if noise_model is not None:
        circuit = noise_model.noisy_circuit(circuit)
    return circuit, joint_code


def _classify_reliable_round1_checks_joint(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    qubit_ids: QubitIDs,
    *,
    intercode: bool,
) -> tuple[int, ...]:
    """Joint-code variant: reliable checks across both gadgets + new cycle rows.

    basis=X (data |+⟩, ancilla + bridge |0⟩):
        H_X rows = [data S_X^(l), data S_X^(r), S'_meas^(l), S'_meas^(r)]
        H_Z rows = [data S_Z^(l) ext, data S_Z^(r) ext, S'_comp^(l)_aug,
                    S'_comp^(r)_aug, new cycle-Z]

      Reliable X: data S_X rows of both gadgets.
      Reliable Z: S'_comp_aug rows + new cycle-Z rows (all act on ancilla ∪ bridge, all |0⟩).

    basis=Z is the X↔Z dual.
    """
    m_X_l = g_l.code.matrix_x.shape[0]
    m_X_r = g_r.code.matrix_x.shape[0] if intercode else 0
    m_Z_l = g_l.code.matrix_z.shape[0]
    m_Z_r = g_r.code.matrix_z.shape[0] if intercode else 0
    if g_l.basis is Pauli.X:
        reliable_x = qubit_ids.checks_x[: m_X_l + m_X_r]  # data S_X^(l/r)
        reliable_z = qubit_ids.checks_z[m_Z_l + m_Z_r :]  # S'_comp_aug + new cycle-Z
    else:
        reliable_x = qubit_ids.checks_x[m_X_l + m_X_r :]  # S'_comp_aug + new cycle-X
        reliable_z = qubit_ids.checks_z[: m_Z_l + m_Z_r]  # data S_Z^(l/r)
    return tuple(reliable_x) + tuple(reliable_z)


def _surgery_qec_cycle_joint(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    joint_code: CSSCode,
    bridge: Bridge,
    num_rounds: int,
    qubit_ids: QubitIDs,
    *,
    intercode: bool,
) -> tuple[stim.Circuit, MeasurementRecord, DetectorRecord]:
    """Joint-code variant of _surgery_qec_cycle that classifies reliable checks
    across both gadgets + the bridge's new cycle-checks."""
    strategy = EdgeColoring()
    one_round, round_measurement_record = strategy.get_circuit(joint_code, qubit_ids)
    reliable = set(
        _classify_reliable_round1_checks_joint(
            g_l,
            g_r,
            qubit_ids,
            intercode=intercode,
        )
    )
    all_check_ids = qubit_ids.check
    lane_idx = _check_lane_index_map(
        g_l,
        qubit_ids,
        joint=(g_r, bridge, intercode),
    )

    circuit = stim.Circuit()
    measurement_record = MeasurementRecord()
    detector_record = DetectorRecord()

    circuit += one_round
    measurement_record.append(round_measurement_record)
    for check_id in all_check_ids:
        if check_id in reliable:
            lane, idx = lane_idx[check_id]
            circuit.append(
                "DETECTOR", [measurement_record.get_target_rec(check_id)], (idx, lane, 0)
            )
    reliable_in_order = [cid for cid in all_check_ids if cid in reliable]
    detector_record.append({cid: dd for dd, cid in enumerate(reliable_in_order)})

    if num_rounds > 1:
        repeat_circuit = one_round.copy()
        measurement_record.append(round_measurement_record)
        repeat_circuit.append("SHIFT_COORDS", [], (0, 0, 1))
        for check_id in all_check_ids:
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
            {cid: dd for dd, cid in enumerate(all_check_ids)},
            repeat=num_rounds - 1,
        )

    return circuit, measurement_record, detector_record


def _surgery_final_detectors_joint(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    joint_code: CSSCode,
    bridge: Bridge,
    qubit_ids: QubitIDs,
    *,
    measurement_record: MeasurementRecord,
    intercode: bool,
) -> stim.Circuit:
    """Joint-code variant of _surgery_final_detectors.

    Emits detectors for the same reliable stabilizers as the round-1 classifier:
    basis=X: data H_X rows from both gadgets + G_aug + new cycle-Z rows.
    basis=Z: data H_Z rows from both gadgets + G_aug + new cycle-X rows.
    """
    m_X_l = g_l.code.matrix_x.shape[0]
    m_X_r = g_r.code.matrix_x.shape[0] if intercode else 0
    m_Z_l = g_l.code.matrix_z.shape[0]
    m_Z_r = g_r.code.matrix_z.shape[0] if intercode else 0
    HX = np.asarray(joint_code.matrix_x).astype(np.uint8)
    HZ = np.asarray(joint_code.matrix_z).astype(np.uint8)

    circuit = stim.Circuit()
    lane_idx = _check_lane_index_map(
        g_l,
        qubit_ids,
        joint=(g_r, bridge, intercode),
    )

    def _emit_detector(stab_row: np.ndarray, check_id: int) -> None:
        supp = np.where(stab_row)[0]
        targets = [measurement_record.get_target_rec(qubit_ids.data[q]) for q in supp]
        targets.append(measurement_record.get_target_rec(check_id, -1))
        lane, idx = lane_idx[check_id]
        circuit.append("DETECTOR", targets, (idx, lane, 0))

    if g_l.basis is Pauli.X:
        for kk in range(m_X_l + m_X_r):
            _emit_detector(HX[kk], qubit_ids.checks_x[kk])
        for kk in range(m_Z_l + m_Z_r, HZ.shape[0]):
            _emit_detector(HZ[kk], qubit_ids.checks_z[kk])
    else:
        for kk in range(m_Z_l + m_Z_r):
            _emit_detector(HZ[kk], qubit_ids.checks_z[kk])
        for kk in range(m_X_l + m_X_r, HX.shape[0]):
            _emit_detector(HX[kk], qubit_ids.checks_x[kk])

    return circuit


def _classify_reliable_round1_checks(
    gadget: GadgetLayout,
    qubit_ids: QubitIDs,
) -> tuple[int, ...]:
    """Check ancillas with deterministic round-1 syndrome given surgery init state."""
    m_X, m_Z = gadget.code.matrix_x.shape[0], gadget.code.matrix_z.shape[0]
    if gadget.basis is Pauli.X:
        reliable_x = qubit_ids.checks_x[:m_X]  # data S_X rows (det. +1)
        reliable_z = qubit_ids.checks_z[m_Z:]  # gauge rows (= S'_comp) (det. +1)
    else:
        reliable_x = qubit_ids.checks_x[m_X:]  # gauge rows (= S'_comp)
        reliable_z = qubit_ids.checks_z[:m_Z]  # data S_Z rows

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
) -> tuple[stim.Circuit, MeasurementRecord, DetectorRecord]:
    """num_rounds of merged-code SE; round-1 detectors only for reliable checks."""
    strategy = EdgeColoring()
    one_round, round_measurement_record = strategy.get_circuit(merged_code, qubit_ids)
    reliable = set(_classify_reliable_round1_checks(gadget, qubit_ids))
    all_check_ids = qubit_ids.check
    lane_idx = _check_lane_index_map(gadget, qubit_ids)

    circuit = stim.Circuit()
    measurement_record = MeasurementRecord()
    detector_record = DetectorRecord()

    # Round 1: emit DETECTORs only for reliable checks.
    circuit += one_round
    measurement_record.append(round_measurement_record)
    for check_id in all_check_ids:
        if check_id in reliable:
            lane, idx = lane_idx[check_id]
            circuit.append(
                "DETECTOR", [measurement_record.get_target_rec(check_id)], (idx, lane, 0)
            )
    reliable_in_order = [cid for cid in all_check_ids if cid in reliable]
    detector_record.append({cid: dd for dd, cid in enumerate(reliable_in_order)})

    if num_rounds > 1:
        repeat_circuit = one_round.copy()
        measurement_record.append(round_measurement_record)
        repeat_circuit.append("SHIFT_COORDS", [], (0, 0, 1))
        for check_id in all_check_ids:
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
            {cid: dd for dd, cid in enumerate(all_check_ids)},
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
) -> stim.Circuit:
    """Emit two OBSERVABLE_INCLUDE entries (obs0, obs1) for the surgery PPM.

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
    data_targets = [measurement_record.get_target_rec(data_ids[i]) for i in support_indices]
    circuit.append("OBSERVABLE_INCLUDE", data_targets, 1)
    return circuit


def _surgery_final_detectors(
    gadget: GadgetLayout,
    merged_code: CSSCode,
    qubit_ids: QubitIDs,
    *,
    measurement_record: MeasurementRecord,
) -> stim.Circuit:
    """Emit DETECTORs for reliable stabs inferable from final readouts.

    For basis=X: data H_X (from Mx data) + G (from Mz κ).
    For basis=Z: data H_Z (from Mz data) + G (from Mx κ).
    Each DETECTOR XORs ⊕(final M-record on stab support) ⊕ last-round syndrome.
    """
    m_X = gadget.code.matrix_x.shape[0]
    m_Z = gadget.code.matrix_z.shape[0]
    HX = np.asarray(merged_code.matrix_x).astype(np.uint8)
    HZ = np.asarray(merged_code.matrix_z).astype(np.uint8)

    circuit = stim.Circuit()
    lane_idx = _check_lane_index_map(gadget, qubit_ids)

    def _emit_detector(stab_row: np.ndarray, check_id: int) -> None:
        supp = np.where(stab_row)[0]
        targets = [measurement_record.get_target_rec(qubit_ids.data[q]) for q in supp]
        targets.append(measurement_record.get_target_rec(check_id, -1))
        lane, idx = lane_idx[check_id]
        circuit.append("DETECTOR", targets, (idx, lane, 0))

    if gadget.basis is Pauli.X:
        for kk in range(m_X):
            _emit_detector(HX[kk], qubit_ids.checks_x[kk])
        for kk in range(m_Z, HZ.shape[0]):
            _emit_detector(HZ[kk], qubit_ids.checks_z[kk])
    else:  # Pauli.Z (symmetric: chi in HZ, G in HX)
        for kk in range(m_Z):
            _emit_detector(HZ[kk], qubit_ids.checks_z[kk])
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
) -> stim.Circuit:
    """Cain step 3 + final data measure. Mκ then SHIFT_COORDS then Mdata."""
    circuit = stim.Circuit()
    detach_qubits = list(ancilla_ids) + list(bridge_ids)
    ancilla_op = "M" if gadget.basis is Pauli.X else "MX"
    data_op = "MX" if gadget.basis is Pauli.X else "M"
    circuit.append(ancilla_op, detach_qubits)
    measurement_record.append({q: i for i, q in enumerate(detach_qubits)})
    circuit.append("SHIFT_COORDS", [], (0, 0, 1))
    circuit.append(data_op, list(data_ids))
    measurement_record.append({q: i for i, q in enumerate(data_ids)})
    return circuit
