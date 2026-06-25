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


def test_single_y_circuit_compiles_with_noise() -> None:
    """The single-Y circuit instruments under a noise model and still compiles
    to a DEM. Deterministic structural check (no sampling) covering the
    ``noise_model`` branch of ``build_single_y_ppm_circuit``.
    """
    from qldpc.circuits import DepolarizingNoiseModel
    from qldpc.circuits.surgery import build_single_y_ppm_circuit

    code, x, z = _steane_y_pair()
    yg = build_y_gadget(code, x=x, z=z)
    circuit = build_single_y_ppm_circuit(
        yg, rounds=2, data_init="Y+", noise_model=DepolarizingNoiseModel(0.01)
    )
    assert isinstance(circuit, stim.Circuit)
    dem = circuit.detector_error_model()
    assert dem.num_detectors > 0


def test_single_y_circuit_rejects_unsupported_init() -> None:
    """An unsupported data_init (e.g. "0") is rejected; valid: the six Pauli
    eigenstates None/'Z+', 'Z-', '+'/'X+', 'X-', 'Y+', 'Y-'."""
    from qldpc.circuits.surgery import build_single_y_ppm_circuit

    code, x, z = _steane_y_pair()
    yg = build_y_gadget(code, x=x, z=z)
    with pytest.raises(ValueError, match="data_init must be one of"):
        build_single_y_ppm_circuit(yg, rounds=3, data_init="0")
    # the six Pauli-basis eigenstates (and the None/"+" aliases) are all accepted
    for di in (None, "Z+", "Z-", "+", "X+", "X-", "Y+", "Y-"):
        build_single_y_ppm_circuit(yg, rounds=3, data_init=di).detector_error_model()


def test_single_y_steane_obs0_deterministic_on_eigenstate() -> None:
    """Steane bare obs0 = ∏(χ_X·χ_Z·y_v) is the deterministic Ȳ readout on |Ȳ±⟩.

    The Ȳ-eigenstate prep is the EXACT codeword |Ȳ±⟩ (state injection |X̄+⟩ then
    transversal S†/S), NOT the physical product ∏_i|Y_i±⟩. On the proper codeword
    every Steane stabilizer is +1, so the bare new-stabilizer product [x | z]
    agrees with Ȳ and its in-circuit XOR is deterministic. The bare support is
    the LITERAL Ȳ = iX̄Z̄ ([x | z] = X₂X₄Z₁Z₃Y₅ = +iX̄Z̄ exactly), so the raw obs0
    bit IS the Ȳ eigenvalue bit: |Ȳ+⟩ → 0, |Ȳ-⟩ → 1 (opposite, both
    deterministic). Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.C.
    """
    import numpy as np

    from qldpc.circuits.surgery import build_single_y_ppm_circuit

    code, x, z = _steane_y_pair()
    yg = build_y_gadget(code, x=x, z=z)
    expected = {"Y+": 0.0, "Y-": 1.0}
    for data_init, want in expected.items():
        circuit = build_single_y_ppm_circuit(yg, rounds=3, data_init=data_init)
        # Deterministic ⇒ the noiseless DEM compiles with exactly the obs0 observable.
        assert circuit.detector_error_model().num_observables == 1
        obs_lines = [
            ln for ln in str(circuit).splitlines() if ln.startswith("OBSERVABLE_INCLUDE")
        ]
        assert len(obs_lines) == 1, "obs0 must be emitted on a deterministic Ȳ eigenstate"
        # Deterministic ⇒ a handful of shots suffices (every shot is identical).
        raw = circuit.compile_sampler().sample(shots=8).astype(np.uint8)
        n_meas = raw.shape[1]
        offs = [int(t.strip("rec[]")) for t in obs_lines[0].split() if t.startswith("rec[")]
        obs0 = np.bitwise_xor.reduce(raw[:, [n_meas + o for o in offs]], axis=1)
        frac = float(obs0.mean())
        assert frac == want, f"data_init={data_init!r}: P(obs0=1)={frac}, expected {want}"


def test_single_y_outcome_nondeterministic_on_0_and_plus() -> None:
    """Noiseless Ȳ on |0̄⟩ / |+̄⟩ is a genuine (non-deterministic) measurement.

    Ȳ = iX̄Z̄ anticommutes with Z̄ (on |0̄⟩) and with X̄ (on |+̄⟩), so the noiseless
    readout outcome is maximally random. Deterministic proof: build the circuit
    with the forced Ȳ readout observable (``force_obs0``) and assert stim REFUSES
    to compile a detector_error_model — it raises ``ValueError`` ("non-deterministic
    observables"). This is the structural signature of a 50/50 measurement (no
    sampling). Contrast with a Ȳ-eigenstate prep (``data_init="Y±"``), on which
    the same ``force_obs0`` path DOES compile — see the Steane obs0 test above —
    so the assertion below is not trivially true. Ide, Gowda, Nadkarni,
    Dauphinais arXiv:2410.02753 §III.C.
    """
    from qldpc.circuits.surgery import build_single_y_ppm_circuit
    from qldpc.circuits.surgery.y_gadget import _bb_y_pair

    code, x, z = _bb_y_pair(overlap=1)
    yg = build_y_gadget(code, x=x, z=z)
    for data_init in (None, "+"):
        circuit = build_single_y_ppm_circuit(
            yg, rounds=3, data_init=data_init, force_obs0=True
        )
        obs_lines = [
            ln for ln in str(circuit).splitlines() if ln.startswith("OBSERVABLE_INCLUDE")
        ]
        assert len(obs_lines) == 1, "force_obs0 should emit exactly one obs0 observable"
        # Non-deterministic Ȳ outcome ⇒ stim refuses the observable in the DEM.
        with pytest.raises(ValueError, match="non-deterministic observable"):
            circuit.detector_error_model()


def test_single_y_force_obs0_conflicts_with_memory_logical() -> None:
    """``force_obs0`` and ``memory_logical`` both claim observable index 0."""
    from qldpc.circuits.surgery import build_single_y_ppm_circuit
    from qldpc.circuits.surgery.y_gadget import _bb_y_pair

    code, x, z = _bb_y_pair(overlap=1)
    yg = build_y_gadget(code, x=x, z=z)
    with pytest.raises(ValueError, match="index 0"):
        build_single_y_ppm_circuit(
            yg, rounds=3, data_init=None, memory_logical=0, force_obs0=True
        )


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


# obs0 representative & prep (supersedes the retired all-Y truth-table tests).
# `_ybar_obs0_rows` targets the BARE literal Ȳ support `[x | z]` — the product of
# the new merge stabilizers ∏(χ_X·χ_Z·y_v) (Ide, Gowda, Nadkarni, Dauphinais
# arXiv:2410.02753 §III.C). That product carries Pauli-X on V_X / Pauli-Z on V_Z
# data qubits, so it is deterministic ONLY on a proper Ȳ-eigenstate CODEWORD —
# NOT on the physical product ∏_i|Y_i±⟩ (where the code stabilizers are random).
# `_steane_logical_y_eigenstate_prep` therefore prepares the EXACT codeword
# |Ȳ±⟩ = S̄|X̄±⟩ (inject |X̄+⟩ by RX^n + Z-syndrome feedback, then transversal S),
# on which the bare obs0 is deterministic and opposite for Y+/Y- — asserted by
# test_single_y_steane_obs0_deterministic_on_eigenstate above. The same bare
# `[x | z]` representative is the only feasible one on a general code (BB
# [[36,8,4]], where the all-Y representative does not exist); its 50/50 behaviour
# on a non-eigenstate prep is covered by the BB force_obs0 test above.
