"""Single-PPM CSS surgery circuit builder (split from the former circuit.py).

References:
    Cain et al. arXiv:2603.28627 Appendix D  — single-PPM measurement protocol.
"""

from __future__ import annotations

import numpy as np
import stim

from qldpc.circuits.bookkeeping import QubitIDs
from qldpc.circuits.noise_model import NoiseModel
from qldpc.objects import Pauli, PauliXZ

from ..hmatrix.PPM_X_Z import GadgetLayout
from .engine import (
    _surgery_detach_and_readout,
    _surgery_final_detectors,
    _surgery_observable,
    _surgery_qec_cycle,
    _surgery_state_prep,
)
from .support import _gadget_merged_csscode, _surgery_qubit_coordinates


def build_single_ppm_circuit(
    gadget: GadgetLayout,
    *,
    rounds: int,
    experiment_basis: PauliXZ | None = None,
    noise_model: NoiseModel | None = None,
    data_init: str | None = None,
    destructive_measure_data: bool = True,
    single_sector: bool = False,
) -> stim.Circuit:
    """Cain et al. arXiv:2603.28627 Appendix D single-PPM surgery experiment.

    ``experiment_basis`` (default ``gadget.basis``): the data init/readout basis,
    decoupled from ``gadget.basis``. Match basis (``experiment_basis is
    gadget.basis``) -> k+1 OBSERVABLE_INCLUDE entries (k frame-corrected block
    logicals + 1 time-like L); opposite basis -> k-1 (the block logicals commuting
    with L; L itself is not readable). See ``_surgery_observable`` for the exact
    layout and the design spec for full semantics.

    ``data_init`` (optional): per-data-qubit init override; see
    ``_surgery_state_prep`` for the character-to-state mapping.

    ``destructive_measure_data`` (default True): when True the data is read out
    destructively at the end (emitting the block observables + the destructive
    final detectors). When False, the circuit is **detach-only / non-destructive**
    — the κ ancillas are still measured (the split that restores the bare code) but
    the data qubits are left encoded, so no end-of-circuit observable can be
    formed; the observable set + destructive final detectors are dropped and the
    detector count drops accordingly.

    ``single_sector`` (default False): emit DETECTORs for the measured-basis sector
    only (X-checks for X̄, Z-checks for Z̄), dropping the complementary sector. The
    time-like L is flipped solely by the opposite single error type, which fires
    the measured-basis sector, so this preserves L's fault distance exactly while
    shrinking the DEM ~8× (the complementary detectors carried only correlated
    soft-info + off-basis correction). It does NOT correct the complementary error
    type on the data, so it is valid for an isolated readout/LER but not when the
    merged register must be handed back intact. CSS-type PPM only (X̄ or Z̄);
    inapplicable to Ȳ / mixed joints, which need both sectors.
    """
    if experiment_basis is None:
        experiment_basis = gadget.basis
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
        experiment_basis=experiment_basis,
        data_init=data_init,
    )
    qec_cycle, measurement_record, _ = _surgery_qec_cycle(
        gadget,
        merged_code,
        num_rounds=rounds,
        qubit_ids=qubit_ids,
        experiment_basis=experiment_basis,
        n_data=n_data,
        single_sector=single_sector,
    )
    circuit += qec_cycle
    circuit += _surgery_detach_and_readout(
        gadget,
        data_ids=data_ids,
        ancilla_ids=Q_prime_ids,
        bridge_ids=bridge_ids,
        measurement_record=measurement_record,
        experiment_basis=experiment_basis,
        destructive_measure_data=destructive_measure_data,
    )
    if destructive_measure_data:
        circuit += _surgery_final_detectors(
            gadget,
            merged_code,
            qubit_ids,
            measurement_record=measurement_record,
            experiment_basis=experiment_basis,
            n_data=n_data,
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

    if destructive_measure_data:
        logical_ops = np.asarray(gadget.code.get_logical_ops(experiment_basis)).astype(np.uint8)
        circuit += _surgery_observable(
            gadget,
            experiment_basis=experiment_basis,
            merged_code=merged_code,
            meas_check_ids=meas_check_ids,
            logical_ops=logical_ops,
            L_support=np.asarray(gadget.x).astype(np.uint8),
            n_data=n_data,
            data_ids=data_ids,
            qprime_ids=Q_prime_ids,
            bridge_ids=bridge_ids,
            measurement_record=measurement_record,
        )

    if noise_model is not None:
        circuit = noise_model.noisy_circuit(circuit)

    return circuit
