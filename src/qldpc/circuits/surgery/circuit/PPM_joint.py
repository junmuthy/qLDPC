"""Joint-PPM CSS surgery circuit builder (split from the former circuit.py).

Same-basis two-block logical measurement (L = X̄_l ⊗ X̄_r or Z̄_l ⊗ Z̄_r) over
the combined data code. Mixed-basis joints are rejected — joint PPMs are
same-type only (Cross, He, Rall, Yoder arXiv:2407.18393); use the single-Ȳ
path for mixed / non-CSS logical measurements.

References:
    Cain et al. arXiv:2603.28627 Appendix D  — joint-PPM measurement protocol.
"""

from __future__ import annotations

import numpy as np
import stim

from qldpc.circuits.bookkeeping import QubitIDs
from qldpc.circuits.noise_model import NoiseModel
from qldpc.codes.common import CSSCode, QuditCode
from qldpc.objects import Pauli, PauliXZ

from ..hmatrix.PPM_joint import Bridge, _joint_merged_dispatch
from ..hmatrix.PPM_XZ import GadgetLayout
from .engine import (
    _surgery_detach_and_readout,
    _surgery_final_detectors,
    _surgery_observable,
    _surgery_qec_cycle,
    _surgery_state_prep,
)
from .support import _surgery_qubit_coordinates


def _stitch_to_joint_code(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
) -> tuple[QuditCode, Bridge]:
    """Assemble merged CSSCode for same-basis two-PPM surgery.

    Delegates to ``_joint_merged_dispatch`` and returns the bridge
    unchanged. Mixed-basis joints are rejected upstream in
    ``build_joint_ppm_circuit`` (no valid CSS merged code exists).
    """
    return _joint_merged_dispatch(g_l, g_r, bridge), bridge


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
    experiment_basis: PauliXZ | None = None,
    noise_model: NoiseModel | None = None,
    data_init: str | tuple[str, ...] | list[str] | None = None,
    destructive_measure_data: bool = True,
    single_sector: bool = False,
) -> tuple[stim.Circuit, QuditCode]:
    """Cain et al. arXiv:2603.28627 Appendix D joint-PPM surgery experiment.

    Same-basis logical measurement L = X̄_l ⊗ X̄_r (or Z̄_l ⊗ Z̄_r) over the
    combined data code (``c_l ⊕ c_r`` intercode, shared ``c`` intracode).

    ``experiment_basis`` (default ``bridge.basis``): the data init/readout basis.
    Match basis (``experiment_basis is bridge.basis``) -> (k_l + k_r) + 1
    OBSERVABLE_INCLUDE entries (the combined block logicals + the time-like L);
    opposite basis -> (k_l + k_r) - 1 (the block logicals commuting with L). See
    ``_surgery_observable`` for the exact layout.

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
    data is left encoded, so no end-of-circuit observable set is emitted (the
    destructive final detectors are dropped too).

    ``single_sector`` (default False): emit DETECTORs for the measured-basis
    sector only (matching ``bridge.basis``), dropping the complementary sector.
    Valid for the match-basis experiment: every observable (the block logicals
    and the time-like L) is of the measured Pauli type, so it is flipped only by
    the opposite single-qubit error type, which fires the measured-basis sector —
    the complementary detectors carry no fault distance for this observable set.
    Shrinks the DEM ~2× with no loss. See ``build_single_ppm_circuit`` for the
    single-PPM analogue.
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
        rounds=rounds, experiment_basis=experiment_basis,
        noise_model=noise_model, data_init=data_init,
        destructive_measure_data=destructive_measure_data,
        single_sector=single_sector,
    )


def _build_joint_ppm_circuit_same_basis(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
    joint_code: CSSCode,
    *,
    rounds: int,
    experiment_basis: PauliXZ | None,
    noise_model: NoiseModel | None,
    data_init: str | tuple[str, ...] | list[str] | None,
    destructive_measure_data: bool = True,
    single_sector: bool = False,
) -> tuple[stim.Circuit, QuditCode]:
    """Original same-basis joint PPM pipeline (CSS merged code)."""
    if experiment_basis is None:
        experiment_basis = bridge.basis
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
    else:
        data_ids = qubit_ids.data[:n_l]
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
        experiment_basis=experiment_basis,
        data_init=expanded_data_init,
    )
    n_data = n_l + n_r  # data columns: n_l + n_r (intercode) or n_l (n_r = 0)
    qec_cycle, measurement_record, _ = _surgery_qec_cycle(
        g_l,
        joint_code,
        num_rounds=rounds,
        qubit_ids=qubit_ids,
        experiment_basis=experiment_basis,
        n_data=n_data,
        joint=(g_r, bridge, intercode),
        single_sector=single_sector,
    )
    circuit += qec_cycle
    circuit += _surgery_detach_and_readout(
        g_l,
        data_ids=data_ids,
        ancilla_ids=Q_prime_ids,
        bridge_ids=bridge_ids,
        measurement_record=measurement_record,
        experiment_basis=experiment_basis,
        destructive_measure_data=destructive_measure_data,
    )
    if destructive_measure_data:
        circuit += _surgery_final_detectors(
            g_l,
            joint_code,
            qubit_ids,
            measurement_record=measurement_record,
            experiment_basis=experiment_basis,
            n_data=n_data,
            joint=(g_r, bridge, intercode),
            single_sector=single_sector,
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

    if destructive_measure_data:
        # Combined data-code logicals + measured operator L over the combined data
        # columns. Intercode: block-diagonal over c_l ⊕ c_r data; L = concat of the
        # two seed supports. Intracode: shared data code; L = g_l.x XOR g_r.x.
        if intercode:
            lx_l = np.asarray(g_l.code.get_logical_ops(experiment_basis)).astype(np.uint8)
            lx_r = np.asarray(g_r.code.get_logical_ops(experiment_basis)).astype(np.uint8)
            logical_ops = np.zeros((lx_l.shape[0] + lx_r.shape[0], n_l + n_r), dtype=np.uint8)
            logical_ops[: lx_l.shape[0], :n_l] = lx_l
            logical_ops[lx_l.shape[0] :, n_l:] = lx_r
            L_support = np.zeros(n_l + n_r, dtype=np.uint8)
            L_support[:n_l] = np.asarray(g_l.x).astype(np.uint8)
            L_support[n_l:] = np.asarray(g_r.x).astype(np.uint8)
        else:  # intracode: shared data code
            logical_ops = np.asarray(g_l.code.get_logical_ops(experiment_basis)).astype(np.uint8)
            L_support = np.asarray(g_l.x).astype(np.uint8) ^ np.asarray(g_r.x).astype(np.uint8)

        circuit += _surgery_observable(
            g_l,
            experiment_basis=experiment_basis,
            merged_code=joint_code,
            meas_check_ids=meas_check_ids,
            logical_ops=logical_ops,
            L_support=L_support,
            n_data=n_data,
            data_ids=data_ids,
            qprime_ids=Q_prime_ids,
            bridge_ids=bridge_ids,
            measurement_record=measurement_record,
        )

    if noise_model is not None:
        circuit = noise_model.noisy_circuit(circuit)
    return circuit, joint_code
