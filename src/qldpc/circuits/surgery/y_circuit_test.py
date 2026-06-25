"""Snapshot + regression tests for y_circuit.py.

The characterization constant _EXPECTED_Y_CIRCUIT pins the EXACT Stim text
emitted by build_single_y_ppm_circuit(yg, rounds=3, data_init="Y+") before the
phase-extraction refactor (Task 3).  The extraction is PURE — no logic changes —
so the snapshot must stay byte-identical after every extraction step.
"""

from __future__ import annotations

_EXPECTED_Y_CIRCUIT = """\
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(1, 0) 1
QUBIT_COORDS(2, 0) 2
QUBIT_COORDS(3, 0) 3
QUBIT_COORDS(4, 0) 4
QUBIT_COORDS(5, 0) 5
QUBIT_COORDS(6, 0) 6
QUBIT_COORDS(7, 0) 7
QUBIT_COORDS(8, 0) 8
QUBIT_COORDS(9, 0) 9
QUBIT_COORDS(10, 0) 10
QUBIT_COORDS(11, 0) 11
QUBIT_COORDS(0, 2) 12
QUBIT_COORDS(1, 2) 13
QUBIT_COORDS(2, 2) 14
QUBIT_COORDS(3, 2) 15
QUBIT_COORDS(4, 2) 16
QUBIT_COORDS(0, 4) 17
QUBIT_COORDS(1, 4) 18
QUBIT_COORDS(2, 4) 19
QUBIT_COORDS(3, 4) 20
QUBIT_COORDS(4, 4) 21
QUBIT_COORDS(5, 4) 22
QUBIT_COORDS(0, 3) 23
RX 0 1 2 3 4 5 6
R 24
CX 3 24 4 24 5 24 6 24
M 24
R 25
CX 1 25 2 25 5 25 6 25
M 25
R 26
CX 0 26 2 26 4 26 6 26
M 26
CX rec[-3] 3 rec[-2] 1 rec[-1] 0
S_DAG 0 1 2 3 4 5 6
R 7 8 9
RX 10 11 12 13 14 15 16
TICK
CX 12 6 13 1 14 2 15 9 16 4
TICK
CX 12 5 13 2 14 6 15 8 16 9
TICK
CX 12 4 13 5 14 0 15 2 16 7
TICK
CX 12 10 13 11 14 4
TICK
CX 12 3 13 6
TICK
TICK
MX 12 13 14 15 16
RX 17 18 19 20 21 22
TICK
TICK
CZ 17 6 18 2 19 0 20 1 21 3 22 7
TICK
CZ 17 4 18 6 19 2 20 11 21 10 22 9
TICK
CZ 17 5 18 1 19 4 22 8
TICK
CZ 17 7 18 5 19 6
TICK
CZ 17 3 18 8 19 9
TICK
MX 17 18 19 20 21 22
RX 23
CX 23 7 23 8
CY 23 5
CZ 23 10 23 11
MX 23
DETECTOR(22, 0, 0) rec[-2]
REPEAT 2 {
    RX 12 13 14 15 16
    TICK
    CX 12 6 13 1 14 2 15 9 16 4
    TICK
    CX 12 5 13 2 14 6 15 8 16 9
    TICK
    CX 12 4 13 5 14 0 15 2 16 7
    TICK
    CX 12 10 13 11 14 4
    TICK
    CX 12 3 13 6
    TICK
    TICK
    MX 12 13 14 15 16
    RX 17 18 19 20 21 22
    TICK
    TICK
    CZ 17 6 18 2 19 0 20 1 21 3 22 7
    TICK
    CZ 17 4 18 6 19 2 20 11 21 10 22 9
    TICK
    CZ 17 5 18 1 19 4 22 8
    TICK
    CZ 17 7 18 5 19 6
    TICK
    CZ 17 3 18 8 19 9
    TICK
    MX 17 18 19 20 21 22
    RX 23
    CX 23 7 23 8
    CY 23 5
    CZ 23 10 23 11
    MX 23
    SHIFT_COORDS(0, 0, 1)
    DETECTOR(12, 0, 0) rec[-12] rec[-24]
    DETECTOR(13, 0, 0) rec[-11] rec[-23]
    DETECTOR(14, 0, 0) rec[-10] rec[-22]
    DETECTOR(15, 0, 0) rec[-9] rec[-21]
    DETECTOR(16, 0, 0) rec[-8] rec[-20]
    DETECTOR(17, 0, 0) rec[-7] rec[-19]
    DETECTOR(18, 0, 0) rec[-6] rec[-18]
    DETECTOR(19, 0, 0) rec[-5] rec[-17]
    DETECTOR(20, 0, 0) rec[-4] rec[-16]
    DETECTOR(21, 0, 0) rec[-3] rec[-15]
    DETECTOR(22, 0, 0) rec[-2] rec[-14]
    DETECTOR(23, 0, 0) rec[-1] rec[-13]
}
M 7 8 9
MX 10 11
SHIFT_COORDS(0, 0, 1)
MY 0 1 2 3 4 5 6
DETECTOR(0, 0, 0) rec[-4] rec[-3] rec[-2] rec[-1] rec[-12] rec[-9] rec[-24] rec[-19]
DETECTOR(0, 0, 0) rec[-6] rec[-5] rec[-2] rec[-1] rec[-11] rec[-8] rec[-23] rec[-18]
DETECTOR(0, 0, 0) rec[-7] rec[-5] rec[-3] rec[-1] rec[-10] rec[-22] rec[-17]
DETECTOR(0, 0, 0) rec[-5] rec[-3] rec[-2] rec[-12] rec[-11] rec[-21] rec[-20] rec[-19] rec[-18] rec[-16] rec[-15] rec[-13]
DETECTOR(0, 0, 0) rec[-12] rec[-11] rec[-10] rec[-14]
OBSERVABLE_INCLUDE(0) rec[-21] rec[-20] rec[-13] rec[-16] rec[-15]"""


def test_y_circuit_text_stable_under_decomposition() -> None:
    """Pin the exact Ȳ circuit so phase extraction stays byte-identical.

    Builds the Ȳ PPM circuit from a Steane-code pair and compares against the
    characterization constant _EXPECTED_Y_CIRCUIT captured before any refactoring.
    If this test fails after a phase extraction step, the extraction changed
    behaviour and must be fixed before continuing.
    """
    from qldpc.circuits.surgery import build_single_y_ppm_circuit
    from qldpc.circuits.surgery.y_gadget import _steane_y_pair, build_y_gadget

    code, x, z = _steane_y_pair()
    yg = build_y_gadget(code, x=x, z=z)
    circ = build_single_y_ppm_circuit(yg, rounds=3, data_init="Y+")
    assert circ.detector_error_model().num_detectors > 0
    assert str(circ) == _EXPECTED_Y_CIRCUIT
