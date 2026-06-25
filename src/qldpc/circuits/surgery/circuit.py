"""Stim surgery circuit construction (single-PPM and joint-PPM).

References:
    Cain et al. arXiv:2603.28627 §B.1  — single-PPM measurement protocol.
    Webster, Smith, Cohen arXiv:2511.15989  — gadget Eq. 1 observable.
"""

from __future__ import annotations

import numpy as np
import stim

from qldpc.circuits.bookkeeping import DetectorRecord, MeasurementRecord, QubitIDs
from qldpc.circuits.memory.syndrome_measurement import EdgeColoring
from qldpc.circuits.noise_model import NoiseModel
from qldpc.codes.common import CSSCode, QuditCode
from qldpc.objects import Pauli

from .bridge import Bridge
from .gadget import GadgetLayout
from .y_gadget import YGadgetLayout


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


def _steane_logical_y_eigenstate_prep(
    yg: YGadgetLayout,
    real_data_ids: tuple[int, ...],
    *,
    data_init: str,
    ancilla_base: int,
) -> stim.Circuit:
    """Prepare the EXACT logical-Y eigenstate codeword |Ȳ±⟩ = S̄|X̄±⟩.

    The correct preparation of a logical-Y eigenstate is the codeword |X̄+⟩
    followed by the transversal phase gate S̄ — NOT the physical product
    ``∏_i |Y_i+⟩`` (which is a +1 eigenstate of every single-qubit Y_i but is
    NOT a codeword: the original code stabilizers are random on it). The
    distinction is decisive for the Ȳ readout: on the physical product the bare
    Ȳ representative ``[x | z]`` (= the product of the merged code's new
    χ_X·χ_Z·y_v rows) is non-deterministic, because ``[x | z]`` differs from an
    all-Y representative by code stabilizers that are random there. On the
    proper codeword every code stabilizer is +1, so all Ȳ representatives agree
    and the in-circuit ``[x | z]`` readout is DETERMINISTIC. The bare support is
    the LITERAL Ȳ = iX̄Z̄: for Steane ``[x | z] = X₂X₄Z₁Z₃Y₅``, and
    ``iX̄Z̄ = +X₂X₄Z₁Z₃Y₅`` (the ``i`` cancels the ``X₅Z₅ = −iY₅`` phase), so
    ⟨[x | z]⟩ = ⟨Ȳ⟩ EXACTLY — no sign convention.

    Construction (exact, noiseless codeword injection):
        RX^n                 → |+⟩^n  (X̄ = +1, X-syndrome 0, Z-syndrome random)
        measure H_Z, correct → |X̄+⟩  (feedback X^{R·s} cancels the random
                                       Z-syndrome s, where ``H_Z R = I``; the
                                       X-corrections commute with X̄ = +1, so the
                                       logical is untouched)
        S†^n (Y+) / S^n (Y-)  → |Ȳ±⟩  (transversal S† maps X̄ → +Ȳ, so
                                       S†|X̄+⟩ = |Ȳ+⟩; transversal S maps X̄ → −Ȳ,
                                       so S|X̄+⟩ = |Ȳ-⟩ — verified in stim)

    The feedback measurement+correction PROJECTS |+⟩^n onto the syndrome-0
    codeword |X̄+⟩ deterministically (the random Z-syndrome is cancelled by a
    Pauli correction, not merely tracked), so the state entering the merge is an
    exact Ȳ eigenstate and the merge's original-code H_X/H_Z rows are +1
    deterministic on it (verified). ``data_init="Y+"`` applies ``S†`` (→ |Ȳ+⟩,
    ⟨Ȳ⟩ = +1, bare obs0 = +1 → bit 0); ``"Y-"`` applies ``S`` (→ |Ȳ-⟩, ⟨Ȳ⟩ = −1,
    bare obs0 = −1 → bit 1).

    Transversal S is a logical operation only for a self-dual CSS code (its X-
    and Z-stabilizer row spaces coincide); we restrict to that case and raise
    otherwise. Homological Ȳ = iX̄Z̄ measurement of Ide, Gowda, Nadkarni,
    Dauphinais arXiv:2410.02753 §III.C/§III.D (docs/superpowers/docs/main.tex
    §4); the eigenstate prep is the standard transversal-Clifford state
    injection of self-dual CSS codes.

    Args:
        yg: the Y-gadget layout (provides the underlying self-dual CSS ``code``).
        real_data_ids: the merged-code data-qubit IDs (first ``code.num_qudits``
            columns) the codeword is prepared on.
        data_init: ``"Y+"`` (apply S) or ``"Y-"`` (apply S†).
        ancilla_base: first free qubit ID; the ``m_z`` Z-stabilizer measurement
            ancillas occupy ``ancilla_base .. ancilla_base + m_z - 1`` (disjoint
            from the merge qubits). These prep measurements are emitted before
            any merge measurement, so the merge's end-relative ``target_rec``
            offsets are unaffected.
    """
    import galois as _galois

    F2 = _galois.GF(2)
    code = yg.code
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    # Self-dual ⇔ X- and Z-stabilizer row spaces coincide (transversal H is
    # then a logical operation; transversal S is logical S for the Steane-like
    # triorthogonal/self-dual family).
    if not (HX.shape == HZ.shape and np.array_equal(np.sort(HX, axis=0), np.sort(HZ, axis=0))):
        raise ValueError(
            "data_init='Y+'/'Y-' transversal prep requires a self-dual CSS code "
            "(transversal H/S logical); got a non-self-dual code. Provide an "
            "explicit Ȳ-eigenstate prep for this code (Ide, Gowda, Nadkarni, "
            "Dauphinais arXiv:2410.02753 §III.C)."
        )
    ids = list(real_data_ids)
    n = len(ids)
    m_z = HZ.shape[0]

    # Right inverse R (n×m_z) of H_Z over GF(2): H_Z @ R = I. The Pauli X^{R·s}
    # then has Z-syndrome s, so applying it cancels a measured Z-syndrome s.
    R = np.zeros((n, m_z), dtype=np.uint8)
    eye = np.eye(m_z, dtype=np.uint8)
    for i in range(m_z):
        aug = F2(np.hstack([HZ, eye[i].reshape(-1, 1)]))
        rref = np.asarray(aug.row_reduce()).astype(np.uint8)
        col = np.zeros(n, dtype=np.uint8)
        for row in rref:
            lead = np.flatnonzero(row[:n])
            if lead.size:
                col[lead[0]] = row[n]
        R[:, i] = col
    if not np.array_equal((HZ @ R) % 2, eye):
        raise ValueError("internal: H_Z has no right inverse (code not full Z-rank)")

    circuit = stim.Circuit()
    circuit.append("RX", ids)  # |+⟩^n  (X̄ = +1, X-syndrome 0)
    # Measure each Z-stabilizer onto a fresh ancilla (random Z-syndrome on |+⟩^n).
    anc_ids = [ancilla_base + i for i in range(m_z)]
    for i in range(m_z):
        anc = anc_ids[i]
        circuit.append("R", [anc])
        for q in np.flatnonzero(HZ[i]):
            circuit.append("CX", [ids[int(q)], anc])  # Z-parity of data → ancilla
        circuit.append("M", [anc])
    # Feedback X^{R·s} to cancel the syndrome → exact |X̄+⟩. Stabilizer i is the
    # (m_z - i)-th most recent measurement record.
    for i in range(m_z):
        rec = stim.target_rec(-(m_z - i))
        for q in np.flatnonzero(R[:, i]):
            circuit.append("CX", [rec, ids[int(q)]])  # classically-controlled X_q
    # Transversal S† maps X̄ → +Ȳ (= iX̄Z̄), so S†|X̄+⟩ = |Ȳ+⟩ (the Ȳ = +1
    # eigenstate); transversal S maps X̄ → −Ȳ, so S|X̄+⟩ = |Ȳ-⟩. Verified in stim:
    # ⟨Ȳ⟩ = ⟨[x|z]⟩ = +1 on S†|X̄+⟩ and −1 on S|X̄+⟩.
    circuit.append("S_DAG" if data_init == "Y+" else "S", ids)  # → |Ȳ+⟩ / |Ȳ-⟩
    return circuit


def build_single_y_ppm_circuit(
    yg: YGadgetLayout,
    *,
    rounds: int,
    noise_model: NoiseModel | None = None,
    data_init: str | None = None,
    memory_logical: int | None = None,
    force_obs0: bool = False,
    benchmark_y: bool = False,
) -> stim.Circuit:
    """Single logical-Y PPM measurement circuit (Ȳ = iX̄Z̄) for ``yg``.

    Builds the syndrome-extraction circuit for the homological Y-gadget merged
    code of Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.C/§III.D
    (docs/superpowers/docs/main.tex §4). The
    merged code ``yg.merged_code`` is a ``QuditCode`` whose stabilizers are the
    original code's X-checks and Z-checks (dual-extended onto the κ_x / κ_z
    ancillas) plus the ``|W|`` mixed Y-type checks ``y_v`` (= the ``yg.Y_stab``
    rows), the Webster, Smith, Cohen arXiv:2511.15989 §II.B.2 cross-merge of the
    χ_X and χ_Z rows anchored at each crossing qubit ``v ∈ W``.

    The emission runs the split X / Z / Y syndrome schedule (the X-ancillas
    collapse before the Z-phase fires; the mixed Y-checks are measured last)
    directly on the single merged code ``yg.merged_code``. This is the sole
    non-CSS surgery emitter — ``build_joint_ppm_circuit`` handles only
    same-basis CSS joints and rejects mixed-basis inputs.

    Pipeline (mirrors ``build_single_ppm_circuit``):
      1. QUBIT_COORDS for data + κ ancillas + Y-row ancilla.
      2. State prep: data via ``data_init``; κ_x ancillas |0⟩ (X-system), κ_z
         ancillas |+⟩ (Z-system) — the basis-complement eigenstate of each
         per-system gadget.
      3. Multi-round QEC with a SPLIT X-phase (CX) / Z-phase (CZ) / Y-phase
         (per-``Y_stab``-row CX/CY/CZ → MX) schedule. Splitting the X- and
         Z-type extractions into non-overlapping phases keeps the individual
         gauge outcomes deterministic in the subsystem-code sense (Cohen, Kim,
         Bartlett, Brown arXiv:2110.10794 §II.B.2; Ide, Gowda, Nadkarni,
         Dauphinais arXiv:2410.02753 §III.C/§III.D). Round-1 detectors are emitted only for
         stabilizer-center rows that are deterministic on the prepared state;
         later rounds get round-to-round difference detectors for all center
         rows.
      4. Detach + destructive readout: κ_x in Z (M), κ_z in X (MX), data in the
         basis matching ``data_init`` (Y-eigenstate data → MY).
      5. Final detectors for center rows reconstructable from the destructive
         readouts (single-row or readout-compatible null-space combinations,
         same construction as the mixed-basis final detectors).
      6. obs0 — the Ȳ eigenvalue as the §III.C readout product (a merged-code
         stabilizer equal to Ȳ = [x|z] on data), read off the IN-CIRCUIT last-
         QEC-round ancilla outcomes of the picked rows (the fault-tolerant
         readout); emitted ONLY when that in-circuit XOR is deterministic on the
         prepared state (gated by ``_observable_is_deterministic``). obs1 reads
         the SAME product off the final destructive readouts (``yg.obs0_readout``)
         as a noiseless cross-check.

    ``data_init`` options — the six logical Pauli-basis eigenstates, named by the
    operator they are a ± eigenstate of:
      * ``"Z+"`` / ``None``: |0̄⟩ (Z̄ = +1, R).     ``"Z-"``: |1̄⟩ (Z̄ = −1, R + X̄).
      * ``"X+"`` / ``"+"``:  |+̄⟩ (X̄ = +1, RX).    ``"X-"``: |-̄⟩ (X̄ = −1, RX + Z̄).
      * ``"Y+"`` / ``"Y-"``: the EXACT logical-Y eigenstate codeword |Ȳ±⟩ = S̄|X̄±⟩
        (inject |X̄+⟩ then transversal S†/S; self-dual CSS code only, see
        ``_steane_logical_y_eigenstate_prep``).
    ``None`` and ``"+"`` are backward-compatible aliases for ``"Z+"`` and ``"X+"``.
    The Ȳ readout (obs0) is deterministic only on the Ȳ eigenstates (Y±); on the
    Z̄ (Z±) and X̄ (X±) eigenstates Ȳ anticommutes with the prepared logical, so
    obs0 is a genuine 50/50 (read via ``force_obs0``).

    ``memory_logical`` (survivor-memory mode). The Ȳ-on-q0 measurement preserves
    the other logicals of the code; their Z̄ are deterministic on the |0̄…0̄⟩ prep
    (``data_init=None``). When ``memory_logical`` is an int and ``data_init is
    None``, this emits a single ``OBSERVABLE_INCLUDE`` (index 0 — obs0/obs1 are
    gated off in this mode, so the DEM carries exactly one observable) tracking
    the ``memory_logical``-th merged-code Z-logical as a surviving logical-memory
    observable read off the final destructive readouts.
    This is the standard logical-memory decodability check of the surgery (Ide,
    Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.C): it sidesteps the random
    Ȳ outcome (obs0 stays gated off — no Ȳ-eigenstate prep on |0̄…0̄⟩) by scoring
    a SURVIVING logical Z̄ instead. The chosen logical must be readout-compatible
    (pure-Z, no support on the κ_z ancillas, which are read in X); a non-compatible
    choice raises ``ValueError``. ``None`` (default) emits nothing, leaving the
    circuit byte-identical to the no-memory build.

    Bell/flag scope (Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.D;
    docs/superpowers/docs/main.tex §4). The brief's
    fault-tolerance refinement — splitting each ``y_v`` ancilla into a Bell/flag
    cell — is NOT built here. This function emits a straightforward Y-phase
    extraction of each ``y_v`` (one ancilla, RX → CX/CY/CZ → MX) that compiles to a
    DEM, which is the Task-4 acceptance. The Bell/flag cell is a distance
    refinement validated by the operational-distance task; it is added only if
    that task finds the operational distance has collapsed to 1.

    obs0 readout (§III.C product). Ide, Gowda, Nadkarni, Dauphinais
    arXiv:2410.02753 §III.C: the obs0 eigenvalue is the XOR of the IN-CIRCUIT
    ancilla records of the merged-code rows whose product equals Ȳ on the
    original data columns. The picker ``_ybar_obs0_rows`` solves over GF(2) for
    the product whose data restriction is the literal Ȳ support ``[x | z]`` (X on
    V_X, Z on V_Z, Y on W) and is eigenbasis-compatible on the κ ancillas (Z-only
    κ_x, X-only κ_z). obs0 is the FAULT-TOLERANT readout: the XOR of those rows'
    IN-CIRCUIT last-QEC-round ancilla outcomes (χ_X → ``checks_x`` M-record, χ_Z →
    ``checks_z`` M-record, q1 → ``y_ancilla`` MX-record), the same mechanism the
    X/Z-measurement sibling ``_surgery_observable`` uses on the final QEC round.

    Determinism gate. The bare ``[x | z]`` product carries Pauli-X on V_X /
    Pauli-Z on V_Z data qubits, so it is deterministic only on a PROPER Ȳ-
    eigenstate codeword — not on a non-eigenstate prep. With ``data_init`` in
    ``("Y+", "Y-")`` the data is the EXACT |Ȳ±⟩ codeword (state injection |X̄+⟩
    then transversal S), on which every code stabilizer is +1, so the bare
    product agrees with Ȳ and the in-circuit XOR is DETERMINISTIC; obs0 is
    emitted (DEM compiles). With the |0̄⟩/|+̄⟩ preps Ȳ anticommutes with the
    prepared logical, so the XOR is a genuine 50/50 and
    ``_observable_is_deterministic`` gates obs0 OFF unless ``force_obs0`` is set.
    The bare product equals Ȳ = iX̄Z̄ EXACTLY (for Steane ``[x | z] = X₂X₄Z₁Z₃Y₅``
    and ``iX̄Z̄ = +X₂X₄Z₁Z₃Y₅``), so the raw obs0 bit is the Ȳ eigenvalue bit:
    obs0 = 0 ↔ Ȳ = +1 (|Ȳ+⟩), obs0 = 1 ↔ Ȳ = −1 (|Ȳ-⟩). The complementary obs1
    reads the SAME product off the final destructive readouts (V_X data MX ⊕ V_Z
    data M ⊕ W data MY ⊕ κ_x M ⊕ κ_z MX) as a noiseless cross-check. The
    per-system Cheeger boost
    (Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.D;
    docs/superpowers/docs/main.tex §4.7) is the FAULT-DISTANCE refinement, applied
    inside ``build_y_gadget``; it is not needed for this readout.
    """
    merged_code = yg.merged_code
    field = merged_code.field
    n_q = merged_code.num_qudits
    n_code = yg.code.num_qudits
    k_x = len(yg.g_x.ancilla_qubits)
    k_z = len(yg.g_z.ancilla_qubits)
    assert n_q == n_code + k_x + k_z, (
        f"merged code width {n_q} != n_code {n_code} + k_x {k_x} + k_z {k_z}"
    )

    # Split the (possibly non-CSS) merged code into pure-X / pure-Z rows (driven
    # by EdgeColoring) and the mixed Y-type rows (driven by per-row CX/CY/CZ).
    virtual_cssc, HX, HZ, x_row_idx, z_row_idx, mixed_row_idx = (
        _split_quditcode_into_virtual_cssc(merged_code)
    )
    qubit_ids = QubitIDs.from_code(virtual_cssc)
    n_Y = len(mixed_row_idx)
    if n_Y:
        max_id = max(qubit_ids.all_qubits) if qubit_ids.all_qubits else -1
        y_ancilla_ids: tuple[int, ...] = tuple(range(max_id + 1, max_id + 1 + n_Y))
    else:
        y_ancilla_ids = ()

    # Merged-code column roles: [data (n_code) | κ_x (k_x) | κ_z (k_z)].
    data_cols = qubit_ids.data
    real_data_ids = data_cols[:n_code]
    kx_ids = data_cols[n_code : n_code + k_x]  # X-system ancillas → |0⟩, read Z
    kz_ids = data_cols[n_code + k_x :]  # Z-system ancillas → |+⟩, read X

    circuit = _mixed_basis_qubit_coords(n_q, qubit_ids, y_ancilla_ids)

    # --- State prep -----------------------------------------------------------
    # ``data_init`` is one of the six logical Pauli-basis eigenstates, named by
    # the operator they are a +/- eigenstate of:
    #   "Z+" (or None) → |0̄⟩ (Z̄ = +1)      "Z-" → |1̄⟩ (Z̄ = −1)
    #   "X+" (or "+")  → |+̄⟩ (X̄ = +1)      "X-" → |-̄⟩ (X̄ = −1)
    #   "Y+"           → |Ȳ+⟩ (Ȳ = +1)      "Y-" → |Ȳ-⟩ (Ȳ = −1)
    # The Ȳ measurement is deterministic ONLY on the Ȳ eigenstates (Y±); on the
    # Z̄/X̄ eigenstates Ȳ anticommutes with the prepared logical, so obs0 is a
    # genuine 50/50 (read via ``force_obs0``). None ("Z+") and "+" ("X+") are the
    # backward-compatible aliases for |0̄⟩ and |+̄⟩; internally |0̄⟩ is ``None``.
    data_init = {"Z+": None, "X+": "+"}.get(data_init, data_init)
    x_cols = tuple(int(q) for q in np.flatnonzero(np.asarray(yg.x).astype(np.uint8)))
    z_cols = tuple(int(q) for q in np.flatnonzero(np.asarray(yg.z).astype(np.uint8)))
    if data_init is None:
        circuit.append("R", list(real_data_ids))  # |0⟩^n → logical |0̄…0̄⟩
    elif data_init == "Z-":
        circuit.append("R", list(real_data_ids))  # |0⟩^n
        circuit.append("X", [real_data_ids[q] for q in x_cols])  # X̄ on supp(x) → |1̄⟩
    elif data_init == "+":
        circuit.append("RX", list(real_data_ids))  # |+⟩^n → logical |+̄…+̄⟩
    elif data_init == "X-":
        circuit.append("RX", list(real_data_ids))  # |+⟩^n
        circuit.append("Z", [real_data_ids[q] for q in z_cols])  # Z̄ on supp(z) → |-̄⟩
    elif data_init in ("Y+", "Y-"):
        prep_base = (
            max([*qubit_ids.all_qubits, *y_ancilla_ids])
            if (qubit_ids.all_qubits or y_ancilla_ids)
            else -1
        ) + 1
        circuit += _steane_logical_y_eigenstate_prep(
            yg, real_data_ids, data_init=data_init, ancilla_base=prep_base
        )
    else:
        raise ValueError(
            "data_init must be one of None/'Z+', 'Z-', '+'/'X+', 'X-', 'Y+', 'Y-' "
            f"for build_single_y_ppm_circuit; got {data_init!r}"
        )
    if sum([bool(force_obs0), memory_logical is not None, bool(benchmark_y)]) > 1:
        raise ValueError(
            "force_obs0, memory_logical, benchmark_y each use observable index 0; set at most one"
        )
    if benchmark_y and data_init in ("Y+", "Y-"):
        raise ValueError(
            "benchmark_y reads Ȳ without a Ȳ-eigenstate prep; use a Z̄/X̄ basis "
            "data_init (None/'Z+', 'Z-', '+'/'X+', 'X-')"
        )
    if kx_ids:
        circuit.append("R", list(kx_ids))  # X-system gadget ancilla |0⟩ (basis-complement)
    if kz_ids:
        circuit.append("RX", list(kz_ids))  # Z-system gadget ancilla |+⟩ (basis-complement)

    # --- Build the split X / Z / Y per-round circuit --------------------------
    # Determinism rationale: X-ancillas collapse before the Z-phase CZ gates
    # fire, so the data is in a definite X-stabilizer eigenstate when the
    # Z-phase starts.
    qubit_ids_x = QubitIDs(data=qubit_ids.data, check=qubit_ids.checks_x)
    qubit_ids_x.checks_x = qubit_ids.checks_x
    qubit_ids_z = QubitIDs(data=qubit_ids.data, check=qubit_ids.checks_z)
    qubit_ids_z.checks_z = qubit_ids.checks_z

    HX_only = HX if HX.shape[0] else np.zeros((0, n_q), dtype=np.uint8)
    HZ_only = HZ if HZ.shape[0] else np.zeros((0, n_q), dtype=np.uint8)
    virtual_cssc_X = CSSCode(
        field(HX_only),
        field(np.zeros((0, n_q), dtype=np.uint8)),
        is_subsystem_code=False,
    )
    virtual_cssc_Z = CSSCode(
        field(np.zeros((0, n_q), dtype=np.uint8)),
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

    # Y-row extraction phase: one |+⟩ ancilla per Y_stab row; CX/CY/CZ entangle
    # it with the data per the Pauli at each column; MX records the eigenvalue.
    H_full = np.asarray(merged_code.matrix).astype(np.int_)
    y_phase_circuit = stim.Circuit()
    y_phase_record = MeasurementRecord()
    if n_Y:
        y_phase_circuit.append("RX", list(y_ancilla_ids))
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

    # --- Map joint rows → check ancilla IDs, classify the stabilizer center ---
    center_mask = _compute_stabilizer_center_mask(H_full, n_q)
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

    # Per-qubit init Pauli + sign (used to find round-1 deterministic centers).
    #   real data: data_init → Pauli/sign; Y± is an Ȳ eigenstate, not a single-
    #     qubit product state, so we mark it None (no per-qubit single-stab is
    #     deterministic — only the Ȳ row product is, handled by obs0).
    #   κ_x: |0⟩ → +Z;  κ_z: |+⟩ → +X.
    qubit_init: dict[int, tuple[Pauli, int] | None] = {}
    if data_init in (None, "Z-"):
        flip = set(x_cols) if data_init == "Z-" else set()  # |1̄⟩ = X̄ on supp(x)
        for col, qid in enumerate(real_data_ids):
            qubit_init[qid] = (Pauli.Z, -1 if col in flip else +1)  # |0⟩ / |1⟩
    elif data_init in ("+", "X-"):
        flip = set(z_cols) if data_init == "X-" else set()  # |-̄⟩ = Z̄ on supp(z)
        for col, qid in enumerate(real_data_ids):
            qubit_init[qid] = (Pauli.X, -1 if col in flip else +1)  # |+⟩ / |-⟩
    else:  # "Y+"/"Y-": no single-qubit Pauli eigenstate per data qubit
        for qid in real_data_ids:
            qubit_init[qid] = None
    for qid in kx_ids:
        qubit_init[qid] = (Pauli.Z, +1)
    for qid in kz_ids:
        qubit_init[qid] = (Pauli.X, +1)

    # Per-qubit destructive readout basis (used by final detector emission):
    #   κ_x → Z (M), κ_z → X (MX), real data → basis matching data_init:
    #   Y± → Y;  X± ("+"/"X-") → X;  Z± (None/"1") → Z.
    data_final_pauli = (
        Pauli.Y
        if data_init in ("Y+", "Y-")
        else (Pauli.X if data_init in ("+", "X-") else Pauli.Z)
    )
    qubit_final_meas: dict[int, Pauli] = {}
    if benchmark_y:
        # Destructive Ȳ readout (obs1): the obs0-product support on data is
        # X on V_X (data_x), Z on V_Z (data_z), Y on W (data_y); other data → Z.
        _bx = set(yg.obs0_readout.data_x)
        _by = set(yg.obs0_readout.data_y)
        for col, qid in enumerate(real_data_ids):
            qubit_final_meas[qid] = (
                Pauli.X if col in _bx else (Pauli.Y if col in _by else Pauli.Z)
            )
    else:
        for qid in real_data_ids:
            qubit_final_meas[qid] = data_final_pauli
    for qid in kx_ids:
        qubit_final_meas[qid] = Pauli.Z
    for qid in kz_ids:
        qubit_final_meas[qid] = Pauli.X

    def _row_paulis(orig_row: int) -> dict[int, Pauli]:
        row = H_full[orig_row]
        out: dict[int, Pauli] = {}
        for q in range(n_q):
            xq, zq = int(row[q]), int(row[q + n_q])
            if xq == 0 and zq == 0:
                continue
            out[q] = Pauli.X if (xq, zq) == (1, 0) else (Pauli.Z if (xq, zq) == (0, 1) else Pauli.Y)
        return out

    # Round-1 reliable: center rows whose every non-I Pauli matches the init
    # eigenstate of that qubit, with net sign +1 (noiseless outcome 0).
    round1_reliable_check_ids: list[int] = []
    for orig in row_to_check:
        if not center_mask[orig]:
            continue
        cid = row_to_check[orig]
        sign = 1
        ok = True
        for q, pauli_q in _row_paulis(orig).items():
            init = qubit_init[qubit_ids.data[q]]
            if init is None or pauli_q is not init[0]:
                ok = False
                break
            sign *= init[1]
        if ok and sign == 1:
            round1_reliable_check_ids.append(cid)

    measurement_record = MeasurementRecord()

    # The merge (lattice surgery proper). On a Y±-eigenstate prep the data is the
    # exact |Ȳ±⟩ codeword (prepared by _steane_logical_y_eigenstate_prep above),
    # so the bare new-stabilizer product ∏(χ_X·χ_Z·y_v) = [x | z] first-measures
    # on a codeword and is the deterministic Ȳ readout (Ide, Gowda, Nadkarni,
    # Dauphinais arXiv:2410.02753 §III.C).
    circuit += one_round
    measurement_record.append(round_measurement_record)
    for cid in round1_reliable_check_ids:
        circuit.append(
            "DETECTOR", [measurement_record.get_target_rec(cid)], (cid, 0, 0)
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

    # --- Detach + destructive readout -----------------------------------------
    if kx_ids:
        circuit.append("M", list(kx_ids))  # X-system ancilla read in Z
        measurement_record.append({q: i for i, q in enumerate(kx_ids)})
    if kz_ids:
        circuit.append("MX", list(kz_ids))  # Z-system ancilla read in X
        measurement_record.append({q: i for i, q in enumerate(kz_ids)})
    circuit.append("SHIFT_COORDS", [], (0, 0, 1))
    if benchmark_y:
        # Per-qubit destructive Ȳ readout: MX on V_X, MY on W, M (Z) on the rest.
        _bx = set(yg.obs0_readout.data_x)
        _by = set(yg.obs0_readout.data_y)
        mx_qids = [qid for col, qid in enumerate(real_data_ids) if col in _bx]
        my_qids = [qid for col, qid in enumerate(real_data_ids) if col in _by]
        mz_qids = [
            qid for col, qid in enumerate(real_data_ids) if col not in _bx and col not in _by
        ]
        for op, qids in (("MX", mx_qids), ("MY", my_qids), ("M", mz_qids)):
            if qids:
                circuit.append(op, qids)
                measurement_record.append({q: i for i, q in enumerate(qids)})
    else:
        data_meas_op = (
            "MY"
            if data_init in ("Y+", "Y-")
            else ("MX" if data_init in ("+", "X-") else "M")
        )
        circuit.append(data_meas_op, list(real_data_ids))
        measurement_record.append({q: i for i, q in enumerate(real_data_ids)})

    # --- Final detectors (center rows reconstructable from destructive readouts)
    # Same construction as the mixed-basis final detectors: emit a detector for
    # each center row directly compatible with the destructive readout basis,
    # and for readout-compatible null-space combinations of the remaining rows
    # whose readout-incompatible parts cancel (Ide, Gowda, Nadkarni, Dauphinais
    # arXiv:2410.02753 §III.C/§III.D; docs/superpowers/docs/main.tex §4).
    center_idx = [orig for orig in row_to_check if center_mask[orig]]
    if center_idx:
        import galois as _galois

        F2 = _galois.GF(2)
        C = H_full[center_idx]  # (n_center, 2 n_q)

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

        emitted_for: set[int] = set()
        for slot, orig in enumerate(center_idx):
            if orig in emitted_for:
                continue
            if _row_destructive_compatible(C[slot]):
                c = np.zeros(len(center_idx), dtype=np.int_)
                c[slot] = 1
                _emit_combo_detector(c)
                emitted_for.add(orig)
                continue
            cands = [(int(v.sum()), v) for v in null_basis if int(v[slot]) == 1]
            if not cands:
                continue
            cands.sort(key=lambda x: x[0])
            best_c = cands[0][1].astype(np.int_)
            _emit_combo_detector(best_c)
            for s2, val in enumerate(best_c):
                if val:
                    emitted_for.add(center_idx[s2])

    # --- obs0: the Ȳ eigenvalue (§III.C IN-CIRCUIT readout product) ------------
    # Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.C: the obs0
    # eigenvalue is the XOR of the IN-CIRCUIT ancilla records of the merged-code
    # rows whose product equals Ȳ on the original data columns. The picker
    # ``_ybar_obs0_rows`` solves this over GF(2): the selected rows' product is
    # the BARE new-stabilizer product ∏(χ_X·χ_Z·y_v), restricting to the literal
    # Ȳ support ``[x | z]`` on data (X on V_X, Z on V_Z, Y on W) and eigenbasis-
    # compatible on the κ ancillas (Z-only κ_x, X-only κ_z). ``yg.obs0_xor_map``
    # records, per selected row, its merged-code (``H_sym``) row index plus its
    # Pauli family (``"X"`` χ_X row, ``"Z"`` χ_Z row, or the ``"Y"`` y_v row) and
    # its index within that family. The family→ancilla map is the same one the
    # round circuit uses (``_split_quditcode_into_virtual_cssc`` partitions
    # ``merged_code.matrix`` in the SAME row order as ``H_sym``, so family_index
    # is the slot in ``checks_x`` / ``checks_z`` / ``y_ancilla_ids``):
    #   family "X" → ``qubit_ids.checks_x[family_index]``   (X-phase, M record)
    #   family "Z" → ``qubit_ids.checks_z[family_index]``   (Z-phase, M record)
    #   family "Y" → ``y_ancilla_ids[family_index]``        (Y-phase, MX record)
    #
    # DETERMINISM GATE. The bare ``[x | z]`` product carries Pauli-X on V_X and
    # Pauli-Z on V_Z data qubits. On a Y±-eigenstate prep the data is the EXACT
    # |Ȳ±⟩ codeword (state injection |X̄+⟩ then transversal S, see
    # ``_steane_logical_y_eigenstate_prep``), where every code stabilizer is +1,
    # so the bare product agrees with Ȳ and the in-circuit XOR is DETERMINISTIC
    # (Y+ → bit 1, Y- → bit 0; the GF(2) product drops the ``i`` of iX̄Z̄, so the
    # raw bit reads −Ȳ). On a non-eigenstate prep (|0̄⟩/|+̄⟩) the bare product is a
    # genuine 50/50, so ``_observable_is_deterministic`` gates obs0 OFF unless
    # ``force_obs0`` is set (the 50/50 cross-check). The previous ALL-Y-on-data
    # representative — a physical ∏_i|Y_i⟩ prep artifact, infeasible on general
    # codes such as BB [[36,8,4]] — is no longer targeted: the exact codeword
    # prep is what makes the bare product deterministic.
    obs0_recs: list[stim.GateTarget] = []
    for row in yg.obs0_xor_map:
        if row.family == "X":
            cid = qubit_ids.checks_x[row.family_index]
        elif row.family == "Z":
            cid = qubit_ids.checks_z[row.family_index]
        else:  # "Y" — the mixed y_v check, read MX in the Y-phase
            cid = y_ancilla_ids[row.family_index]
        obs0_recs.append(measurement_record.get_target_rec(cid))
    if force_obs0:
        # Emit obs0 even when it is NON-deterministic. On a non-Ȳ-eigenstate prep
        # (|0̄⟩ via data_init=None, or |+̄⟩ via data_init="+"), Ȳ anticommutes with
        # the prepared logical so the outcome is a genuine 50/50 — the DEM will NOT
        # compile, but raw sampling (``circuit.compile_sampler``) reads obs0
        # directly. Demonstrates that the merge measures Ȳ (Ide, Gowda, Nadkarni,
        # Dauphinais arXiv:2410.02753 §III.C).
        if obs0_recs:
            circuit.append("OBSERVABLE_INCLUDE", obs0_recs, 0)
    elif data_init in ("Y+", "Y-"):
        if obs0_recs and _observable_is_deterministic(circuit, obs0_recs):
            circuit.append("OBSERVABLE_INCLUDE", obs0_recs, 0)

    # --- benchmark_y: obs0 ⊕ obs1 (surgery Ȳ readout vs destructive Ȳ readout) --
    # Direct benchmark of the Ȳ MEASUREMENT itself (Ide, Gowda, Nadkarni,
    # Dauphinais arXiv:2410.02753 §III.C): obs0 reads Ȳ off the in-circuit checks;
    # obs1 reads the SAME §III.C product (``yg.obs0_readout``) off the per-qubit
    # destructive Ȳ readout (data_x → MX, data_z → M, data_y → MY; κ_x → M, κ_z →
    # MX). Each alone is a random 50/50 outcome on a non-Ȳ-eigenstate prep, but
    # their XOR is DETERMINISTIC (both equal Ȳ) — so it compiles to a DEM and a
    # decoder scores P(obs0 ≠ obs1) = the measurement logical error rate. No Ȳ
    # eigenstate prep is required (any input state works).
    if benchmark_y:
        plan = yg.obs0_readout
        bench_recs: list[stim.GateTarget] = list(obs0_recs)
        for q in (*plan.data_x, *plan.data_z, *plan.data_y):
            bench_recs.append(measurement_record.get_target_rec(real_data_ids[q]))
        for q in plan.kx_z:
            bench_recs.append(measurement_record.get_target_rec(kx_ids[q - n_code]))
        for q in plan.kz_x:
            bench_recs.append(measurement_record.get_target_rec(kz_ids[q - n_code - k_x]))
        circuit.append("OBSERVABLE_INCLUDE", bench_recs, 0)

    # --- obs1: destructive cross-check (NOT a physical protocol) ----------------
    # Read the SAME §III.C product (``yg.obs0_readout``) off the FINAL DESTRUCTIVE
    # readouts. With the literal ``[x | z]`` Ȳ representative the data support is
    # MIXED: V_X data → MX, V_Z data → M, W data → MY; κ_x Z-support → M, κ_z
    # X-support → MX (Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.C).
    # This destructively collapses the data, so it is not the fault-tolerant
    # readout; it is the noiseless cross-check sibling of ``_surgery_observable``'s
    # obs1. It reads the SAME Ȳ = [x | z] product as obs0, so its raw bit is the
    # same Ȳ eigenvalue (Y+ → 0, Y- → 1). It is emitted only when its destructive
    # basis matches the readout — with the all-MY Y±-eigenstate readout the V_X/V_Z
    # records are measured in Y, not X/Z, so it is gated OFF (``benchmark_y`` is the
    # working destructive cross-check, splitting the readout MX/MY/M). Keep obs1
    # only as a cross-check; for any LER/noisy run keep ONLY obs0.
    if data_init in ("Y+", "Y-"):
        plan = yg.obs0_readout
        obs1_recs: list[stim.GateTarget] = []
        for q in (*plan.data_x, *plan.data_z, *plan.data_y):  # V_X→MX, V_Z→M, W→MY
            obs1_recs.append(measurement_record.get_target_rec(real_data_ids[q]))
        for q in plan.kx_z:  # κ_x column q, read M (Z)
            obs1_recs.append(measurement_record.get_target_rec(kx_ids[q - n_code]))
        for q in plan.kz_x:  # κ_z column q, read MX (X)
            obs1_recs.append(measurement_record.get_target_rec(kz_ids[q - n_code - k_x]))
        if obs1_recs and _observable_is_deterministic(circuit, obs1_recs):
            circuit.append("OBSERVABLE_INCLUDE", obs1_recs, 1)

    # --- survivor-memory observable --------------------------------------------
    # The Ȳ-on-q0 measurement preserves the other logicals; their Z̄ are
    # deterministic on the |0̄…0̄⟩ prep (data_init is None). Track one such
    # SURVIVING logical Z̄ off the final destructive readouts as a logical-memory
    # observable — the standard decodability check of the surgery (Ide, Gowda,
    # Nadkarni, Dauphinais arXiv:2410.02753 §III.C). This sidesteps the random Ȳ
    # outcome (obs0 stays gated off) by scoring a survivor instead. Gated on
    # ``data_init is None`` — mutually exclusive with the Y+/Y- obs0/obs1
    # emissions above — so index 0 is free; emitting there keeps the DEM at
    # exactly one observable (no phantom always-False obs0/obs1 padding). Gated by
    # determinism, emitted before the noise block so the noise model wraps it.
    if memory_logical is not None and data_init is None:
        LZ = np.asarray(merged_code.get_logical_ops(Pauli.Z)).astype(np.uint8)
        row = LZ[memory_logical]  # (2*n_q,) symplectic [x | z]
        xpart, zpart = row[:n_q], row[n_q:]
        # readout-compatible survivor: pure-Z, no κ_z support (κ_z is read in X).
        # ``xpart.any()`` is a defensive guard: a ``get_logical_ops(Pauli.Z)`` row
        # is already pure-Z, so it never fires in practice, but it pins the
        # contract should a non-pure-Z representative ever be passed.
        if xpart.any() or zpart[n_code + k_x :].any():
            raise ValueError(
                f"merged Z-logical {memory_logical} is not Z-readout-compatible"
            )
        mem_recs: list[stim.GateTarget] = []
        for q in range(n_code):  # data Z -> M record
            if zpart[q]:
                mem_recs.append(measurement_record.get_target_rec(real_data_ids[q]))
        for q in range(k_x):  # κ_x Z -> M record (detach)
            if zpart[n_code + q]:
                mem_recs.append(measurement_record.get_target_rec(kx_ids[q]))
        if mem_recs and _observable_is_deterministic(circuit, mem_recs):
            circuit.append("OBSERVABLE_INCLUDE", mem_recs, 0)

    if noise_model is not None:
        circuit = noise_model.noisy_circuit(circuit)

    return circuit


def _observable_is_deterministic(
    circuit: stim.Circuit, obs_targets: list[stim.GateTarget]
) -> bool:
    """Return True iff the XOR of ``obs_targets`` is deterministic in ``circuit``.

    Appends a probe OBSERVABLE_INCLUDE to a copy of ``circuit`` (which must be
    noiseless at this point) and asks stim whether it compiles: a
    non-deterministic observable makes ``detector_error_model()`` raise. We
    catch that to decide whether obs0 can be emitted. Used to gate obs0 in the
    regime where the Ȳ readout has an unpinned κ-gauge residual (Ide, Gowda,
    Nadkarni, Dauphinais arXiv:2410.02753 §III.C; docs/superpowers/docs/main.tex
    §4).
    """
    probe = circuit.copy()
    probe.append("OBSERVABLE_INCLUDE", obs_targets, 0)
    try:
        probe.detector_error_model()
        return True
    except ValueError:
        return False


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
        ancilla_ids=ancilla_ids,
        bridge_ids=bridge_ids,
        measurement_record=measurement_record,
    )
    circuit += _surgery_final_detectors(
        g_l,
        joint_code,
        qubit_ids,
        measurement_record=measurement_record,
        joint=(g_r, bridge, intercode),
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

    The merged code (Webster cross-merge in ``y_gadget``) has three row classes:
      * pure-X rows (HX block): data X-stabilizers + comp-side rows.
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
) -> tuple[stim.Circuit, MeasurementRecord, DetectorRecord]:
    """num_rounds of merged-code SE; round-1 detectors only for reliable checks.

    Single-PPM (``joint=None``) and joint-PPM (``joint=(g_r, bridge,
    intercode)``) share one round loop; only the reliable-check classifier and
    the check→lane map differ by whether the right gadget + bridge participate.
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
    joint: tuple[GadgetLayout, Bridge, bool] | None = None,
) -> stim.Circuit:
    """Emit DETECTORs for reliable stabs inferable from final readouts.

    For basis=X: data H_X (from Mx data) + G (from Mz κ).
    For basis=Z: data H_Z (from Mz data) + G (from Mx κ).
    Each DETECTOR XORs ⊕(final M-record on stab support) ⊕ last-round syndrome.
    Joint-PPM (``joint=(g_r, bridge, intercode)``) spans both gadgets' data rows.
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
