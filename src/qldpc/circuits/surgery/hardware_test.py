"""Matrix-based hardware selection includes extended block checks."""

import numpy as np
import pytest
import stim

from qldpc import codes
from qldpc.circuits import DepolarizingNoiseModel, QubitIDs
from qldpc.circuits.surgery import (
    AdapterHardware,
    JointPPMSelectiveBellPairSyndrome,
    build_bridge,
    build_gadget,
    build_preinitialized_joint_ppm_circuit,
    keep_only_observable,
)
from qldpc.objects import Pauli


def test_hardware_uses_matrix_columns_and_physical_id_mapping() -> None:
    # First two columns are block data, last two are new. The first check
    # survives on the block; the second is an extended block check.
    code = codes.CSSCode([[1, 1, 0, 0], [1, 1, 1, 1]], [[0, 0, 1, 1]])
    ids = QubitIDs([8, 2, 9, 1], [11, 15, 14], [20])
    hardware = AdapterHardware.from_code(code, ids, num_block_data_qubits=2)
    assert hardware.data == (9, 1)
    assert hardware.check == (15, 14)
    assert hardware.ancilla == (20,)
    with pytest.raises(ValueError, match="num_block_data_qubits"):
        AdapterHardware.from_code(code, ids, num_block_data_qubits=5)


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_hardware_window_keeps_boundaries_ideal_and_faults_on_interface_gates(basis: Pauli) -> None:
    left = codes.SteaneCode()
    # Construct a distinct generic CSS instance rather than relying on a surface-code layout.
    right = codes.CSSCode(left.matrix_x, left.matrix_z)
    gl = build_gadget(left, np.asarray(left.get_logical_ops(basis)[0]), basis=basis)
    gr = build_gadget(right, np.asarray(right.get_logical_ops(basis)[0]), basis=basis)
    bridge = build_bridge(gl, gr)
    strategy = JointPPMSelectiveBellPairSyndrome(
        gl, gr, bridge, independent_blocks=True, bell_pair_fidelity=0.999
    )
    circuit, joint = build_preinitialized_joint_ppm_circuit(
        gl,
        gr,
        bridge,
        preparation_rounds=2,
        surgery_rounds=3,
        adapter_noise_model=DepolarizingNoiseModel(0.001, include_idling_error=True),
        adapter_noise_targets="adapter_hardware",
        surgery_syndrome_measurement_strategy=strategy,
    )
    ids = QubitIDs.from_code(joint, num_ancillas=len(strategy.selected_check_indices(joint)))
    hardware = AdapterHardware.from_code(joint, ids, num_block_data_qubits=len(left) + len(right))
    active = set(hardware.all_qubits)
    block_data = set(ids.data[: len(left) + len(right)])
    measurements = 0
    crosses_boundary = False
    bell_noise = False
    for op in circuit.flattened():
        targets = [t.value for t in op.targets_copy() if t.qubit_value is not None]
        noisy_measure = op.name in {"M", "MX", "MR", "MRX"} and bool(op.gate_args_copy())
        is_noise = op.name in {
            "DEPOLARIZE1",
            "DEPOLARIZE2",
            "X_ERROR",
            "Z_ERROR",
            "PAULI_CHANNEL_1",
        }
        if is_noise or noisy_measure:
            assert measurements >= 2 * (left.num_checks + right.num_checks)
            if op.name == "DEPOLARIZE2":
                for first, second in zip(targets[::2], targets[1::2]):
                    assert {first, second} & active
                    crosses_boundary |= bool({first, second} & block_data)
            else:
                assert set(targets) <= active
            if op.name == "PAULI_CHANNEL_1":
                assert sum(op.gate_args_copy()) == pytest.approx(0.001)
                assert set(targets) <= set(hardware.ancilla)
                bell_noise = True
        measurements += op.num_measurements
    assert crosses_boundary and bell_noise
    terminal = next(op for op in reversed(list(circuit.flattened())) if op.name in {"M", "MX"})
    assert not terminal.gate_args_copy()
    dem = keep_only_observable(circuit, keep_idx=0).detector_error_model()
    assert isinstance(dem, stim.DetectorErrorModel) and dem.num_errors > 0
