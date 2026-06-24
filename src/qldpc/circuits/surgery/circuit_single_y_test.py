"""Tests for the single logical-Y measurement circuit (Ȳ = iX̄Z̄).

Task-4 acceptance: ``build_single_y_ppm_circuit`` builds a ``stim.Circuit``
for the homological Y-gadget merged code (Ide, Gowda, Nadkarni, Dauphinais
arXiv:2410.02753 §III.C/§III.D; docs/superpowers/docs/main.tex §4) and that
circuit compiles to a ``stim.DetectorErrorModel`` with the logical-Y
eigenstate prep ``data_init="Y+"``.
"""

from __future__ import annotations

import pytest
import stim

from qldpc.circuits.surgery.y_gadget import _steane_y_pair, build_y_gadget


def test_single_y_circuit_builds_and_compiles() -> None:
    """Single-Y PPM circuit builds and compiles to a DEM (Y+ prep)."""
    from qldpc.circuits.surgery import build_single_y_ppm_circuit

    code, x, z = _steane_y_pair()
    yg = build_y_gadget(code, x=x, z=z)
    circuit = build_single_y_ppm_circuit(yg, rounds=3, data_init="Y+")
    assert isinstance(circuit, stim.Circuit)
    dem = circuit.detector_error_model()
    assert dem is not None


@pytest.mark.parametrize("data_init", [None, "Y+", "Y-"])
@pytest.mark.parametrize("rounds", [1, 3])
def test_single_y_circuit_compiles_all_inits(data_init: str | None, rounds: int) -> None:
    """The circuit compiles to a DEM for the default and both Y eigenstate preps."""
    from qldpc.circuits.surgery import build_single_y_ppm_circuit

    code, x, z = _steane_y_pair()
    yg = build_y_gadget(code, x=x, z=z)
    circuit = build_single_y_ppm_circuit(yg, rounds=rounds, data_init=data_init)
    # Noiseless DEM compiles (all detectors deterministic on the prepared state).
    dem = circuit.detector_error_model()
    assert dem.num_detectors > 0


def test_single_y_circuit_rejects_product_state_init() -> None:
    """A product-state data_init (e.g. "0") is rejected: Ȳ prep needs Y+/Y-."""
    from qldpc.circuits.surgery import build_single_y_ppm_circuit

    code, x, z = _steane_y_pair()
    yg = build_y_gadget(code, x=x, z=z)
    with pytest.raises(ValueError, match="data_init must be None, 'Y\\+' or 'Y-'"):
        build_single_y_ppm_circuit(yg, rounds=3, data_init="0")


def test_single_y_survivor_memory_observable_compiles() -> None:
    """memory_logical=0 emits exactly one deterministic survivor-Z̄ observable.

    The Ȳ measurement preserves the other 7 logicals; their Z̄ are deterministic
    on |0̄…0̄⟩. Emitting one as an OBSERVABLE_INCLUDE gives a DEM with a single
    observable (Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.C).
    """
    from qldpc.circuits.surgery import build_single_y_ppm_circuit
    from qldpc.circuits.surgery.y_gadget import _bb_y_pair

    code, x, z = _bb_y_pair(overlap=1)
    yg = build_y_gadget(code, x=x, z=z)
    circuit = build_single_y_ppm_circuit(
        yg, rounds=3, data_init=None, memory_logical=0
    )
    dem = circuit.detector_error_model()
    assert dem.num_observables == 1, (
        f"expected exactly the survivor-Z̄ observable, got {dem.num_observables}"
    )


def test_single_y_memory_logical_none_unchanged() -> None:
    """memory_logical=None (default) emits no observable: byte-identical circuit."""
    from qldpc.circuits.surgery import build_single_y_ppm_circuit
    from qldpc.circuits.surgery.y_gadget import _bb_y_pair

    code, x, z = _bb_y_pair(overlap=1)
    yg = build_y_gadget(code, x=x, z=z)
    default = build_single_y_ppm_circuit(yg, rounds=3, data_init=None)
    explicit_none = build_single_y_ppm_circuit(
        yg, rounds=3, data_init=None, memory_logical=None
    )
    assert str(default) == str(explicit_none)
    assert default.detector_error_model().num_observables == 0


@pytest.mark.slow
def test_bb_survivor_memory_ler_monotone_in_p() -> None:
    """BB Ȳ-surgery preserves a surviving logical: survivor-memory LER grows with p.

    obs0 (the random Ȳ outcome) stays gated off; we score decoding against a
    surviving logical Z̄ (deterministic on |0̄…0̄⟩), demonstrating the merge is
    decodable and fault-tolerant for the other logicals (Ide, Gowda, Nadkarni,
    Dauphinais arXiv:2410.02753 §III.C).
    """
    import sinter

    from qldpc import decoders
    from qldpc.circuits import DepolarizingNoiseModel
    from qldpc.circuits.surgery import build_single_y_ppm_circuit
    from qldpc.circuits.surgery.y_gadget import _bb_y_pair

    code, x, z = _bb_y_pair(overlap=1)
    yg = build_y_gadget(code, x=x, z=z)
    error_rates = [0.001, 0.005, 0.02]
    tasks = []
    for p in error_rates:
        circ = build_single_y_ppm_circuit(
            yg,
            rounds=3,
            data_init=None,
            memory_logical=0,
            noise_model=DepolarizingNoiseModel(p),
        )
        tasks.append(sinter.Task(circuit=circ, json_metadata={"p": float(p)}))
    results = sinter.collect(
        tasks=tasks,
        decoders=["custom"],
        custom_decoders={"custom": decoders.SinterDecoder()},
        num_workers=4,
        max_shots=2000,
        max_errors=30,
        print_progress=False,
    )
    by_p = {r.json_metadata["p"]: r.errors / max(r.shots, 1) for r in results}
    sorted_p = sorted(by_p)
    ler = [by_p[p] for p in sorted_p]
    print("survivor-memory LER:", list(zip(sorted_p, ler)))
    for i in range(len(ler) - 1):
        assert ler[i] <= ler[i + 1] * 1.5, (
            f"LER not monotone: {list(zip(sorted_p, ler))}"
        )


# RETIRED (Task 5a): test_single_y_noiseless_truth_table and
# test_single_y_dem_has_no_undetectable_observable_error are superseded by the BB
# DEM test (Task 6). Both asserted on the Steane IN-CIRCUIT obs0, which was only
# deterministic for the ALL-Y-on-data representative of Ȳ = iX̄Z̄. That
# representative does not exist on a general code (BB [[36,8,4]] → ValueError), so
# `_ybar_obs0_rows` now targets the literal Ȳ support `[x | z]` (Ide, Gowda,
# Nadkarni, Dauphinais arXiv:2410.02753 §III.C). The `[x | z]` product carries
# bare Pauli-X on V_X / Pauli-Z on V_Z data qubits, which are NOT measurable
# in-circuit on a |Y±⟩ prep (verified: every selected row's ancilla record is
# individually non-deterministic; stim reports "The circuit contains
# non-deterministic observables"). build_single_y_ppm_circuit therefore gates obs0
# off (via _observable_is_deterministic), so no obs0 OBSERVABLE_INCLUDE is emitted
# for the Steane fixture and these two Steane-obs0 tests no longer have a subject.
# The fault-tolerant Ȳ readout / sign convention is validated on BB by the Task 6
# DEM test instead. (build_single_y_ppm_circuit / DEM compilation for the Steane
# fixture is still covered by the four tests above.)
