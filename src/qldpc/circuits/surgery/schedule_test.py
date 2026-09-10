"""Controls must change ordering, never check/partner identity or gate content."""

from __future__ import annotations

import dataclasses

import pytest
import stim

from qldpc import codes
from qldpc.circuits import QubitIDs
from qldpc.objects import Pauli

from .bridge import build_bridge
from .circuit import JointPPMSelectiveBellPairSyndrome, _stitch_intercode
from .gadget import build_gadget
from .schedule import JointPPMSchedule, validate_local_layers


@pytest.mark.parametrize("field", ["local_order", "bell_order"])
def test_invalid_family_order(field: str) -> None:
    with pytest.raises(ValueError, match="exactly once"):
        JointPPMSchedule(**{field: (Pauli.X, Pauli.X)})


@pytest.mark.parametrize(
    "bad, message",
    [
        ("TICK\nCZ 0 1\nTICK", "inventory"),
        ("TICK\nCX 0 1 0 2\nTICK", "conflict"),
        ("TICK\nRX 0\nTICK", "only CX/CZ"),
        ("CX 0 1\nTICK", "start and end"),
        ("REPEAT 2 {\nTICK\nCX 0 1\nTICK\n}", "flat"),
    ],
)
def test_reject_invalid_local_layers(bad: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_local_layers(stim.Circuit(bad), stim.Circuit("TICK\nCX 0 1\nTICK"))


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_independent_order_controls_preserve_check_and_bell_partners(basis: Pauli) -> None:
    left, right = codes.SurfaceCode(3), codes.SteaneCode()
    gl = build_gadget(left, left.get_logical_ops(basis)[0], basis=basis)
    gr = build_gadget(right, right.get_logical_ops(basis)[0], basis=basis)
    bridge = build_bridge(gl, gr)
    code = _stitch_intercode(gl, gr, bridge)
    strategy = JointPPMSelectiveBellPairSyndrome(gl, gr, bridge)
    selected = strategy.selected_check_indices(code)
    ids = QubitIDs.from_code(code, num_ancillas=len(selected))
    baseline, _ = strategy.get_circuit(code, ids)
    local_x, local_z = strategy.local_interaction_circuits(code, ids)
    for schedule in (
        JointPPMSchedule(),
        JointPPMSchedule(bell_order=(Pauli.Z, Pauli.X)),
        JointPPMSchedule(local_order=(Pauli.Z, Pauli.X)),
        JointPPMSchedule(local_layers=(local_x, local_z)),
    ):
        variant = JointPPMSelectiveBellPairSyndrome(gl, gr, bridge, schedule=schedule)
        circuit, record = variant.get_circuit(code, ids)
        assert circuit.num_ticks == baseline.num_ticks
        measurements = [t.value for op in circuit if op.name == "MX" for t in op.targets_copy()]
        for check_index in range(code.num_checks):
            check = ids.check[check_index]
            expected = {check}
            if check_index in selected:
                expected.add(ids.ancilla[selected.index(check_index)])
            assert {measurements[i] for i in record[check][0]} == expected
        if schedule.bell_order == schedule.local_order == (Pauli.X, Pauli.Z):
            assert circuit == baseline
        else:
            assert circuit != baseline
        if schedule.local_order == (Pauli.Z, Pauli.X):
            assert str(local_z + local_x) in str(circuit)

    with pytest.raises(ValueError, match="inventory"):
        JointPPMSelectiveBellPairSyndrome(
            gl,
            gr,
            bridge,
            schedule=JointPPMSchedule(local_layers=(local_z, local_x)),
        ).get_circuit(code, ids)
    layers = JointPPMSchedule(local_layers=(local_x, local_z))
    assert layers.metadata() == dataclasses.replace(layers).metadata()
