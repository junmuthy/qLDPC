"""Tests for the single logical-Y measurement circuit (Ȳ = iX̄Z̄).

Task-4 acceptance: ``build_single_y_ppm_circuit`` builds a ``stim.Circuit``
for the single-overlap Y-gadget merged code (Cross, He, Rall, Yoder
arXiv:2407.18393 §3.7) and that circuit compiles to a
``stim.DetectorErrorModel`` with the logical-Y eigenstate prep
``data_init="Y+"``.
"""

from __future__ import annotations

import numpy as np
import pytest
import stim

from qldpc.circuits.surgery.y_gadget import _steane_y_pair, build_y_gadget


def _raw_observables(circuit: stim.Circuit, shots: int) -> np.ndarray:
    """Raw +-1 obs values via compile_sampler + manual XOR.

    Truth-table sign checks REQUIRE this API — detector_sampler.sample(
    separate_observables=True) returns flips relative to the noiseless
    reference (always 0 on a noiseless run), hiding the sign. See
    examples/lattice_surgery.ipynb section 0 ``raw_observables`` and the
    sibling helper in circuit_mixed_test.py that this mirrors.
    """
    sampler = circuit.compile_sampler()
    raw = sampler.sample(shots=shots).astype(np.uint8)
    n_meas = raw.shape[1]
    obs_lines = [ln for ln in str(circuit).splitlines() if ln.startswith("OBSERVABLE_INCLUDE")]
    cols = []
    for line in obs_lines:
        offsets = [int(t.strip("rec[]")) for t in line.split() if t.startswith("rec[")]
        meas_idx = [n_meas + off for off in offsets]
        cols.append(np.bitwise_xor.reduce(raw[:, meas_idx], axis=1))
    return np.stack(cols, axis=1)


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


def test_single_y_noiseless_truth_table() -> None:
    """Noiseless Steane Ȳ truth table: Y+ → obs0=1, Y- → obs0=0 (−Ȳ convention).

    Ȳ = iX̄Z̄ measured by the single-overlap §3.7 merged code (Cross, He, Rall,
    Yoder arXiv:2407.18393 §3.7), NO bridge — the readout does not need it.

    obs0 is the §3.2 readout product (a single merged-code stabilizer equal to
    Ȳ), read off the IN-CIRCUIT last-QEC-round ancilla outcomes of the rows the
    GF(2) picker selected — the fault-tolerant physical readout (same mechanism
    as the X/Z-measurement sibling ``_surgery_observable``). It is fully
    DETERMINISTIC but measures the SIGNED Pauli product of those rows, which for
    Ȳ = iX̄Z̄ is −Ȳ: the GF(2) picker drops the ``i`` phase, so the signed product
    carries sign −1. Hence the raw obs0 bit is the eigenvalue bit of −Ȳ =
    NOT(Ȳ bit): on |+i⟩_L (Ȳ = +1) obs0 = 1; on |−i⟩_L (Ȳ = −1) obs0 = 0. We
    assert these real, deterministic values (no faked sign). The complementary
    obs1 (destructive cross-check, kept at index 1) carries the un-inverted
    eigenvalue Y+ → 0, Y- → 1.

    Verified via the raw compile_sampler + manual XOR (``_raw_observables``),
    NOT ``detector_sampler.sample(separate_observables=True)`` — the latter
    reports flips relative to the noiseless baseline (always 0) and hides the
    sign.
    """
    from qldpc.circuits.surgery import build_single_y_ppm_circuit, keep_only_observable

    code, x, z = _steane_y_pair()
    yg = build_y_gadget(code, x=x, z=z)
    # obs0 measures −Ȳ (intrinsic iX̄Z̄ sign), so the eigenvalue bit is inverted.
    for data_init, expected in (("Y+", 1), ("Y-", 0)):
        circuit = build_single_y_ppm_circuit(yg, rounds=3, data_init=data_init)
        circuit = keep_only_observable(circuit, keep_idx=0)
        obs = _raw_observables(circuit, shots=64)
        assert obs.shape[1] == 1, f"expected exactly 1 observable, got {obs.shape[1]}"
        assert (obs[:, 0] == expected).all(), (
            f"data_init={data_init!r} expected obs0={expected}, got "
            f"{np.bincount(obs[:, 0], minlength=2).tolist()}"
        )


@pytest.mark.xfail(
    reason=(
        "Measurement FAULT DISTANCE requires the §3.7 bridge, which is a "
        "SEPARATE deferred task (Cross, He, Rall, Yoder arXiv:2407.18393 Remark "
        "23). The no-bridge construction here gives a correct, deterministic Ȳ "
        "readout (see test_single_y_noiseless_truth_table) but collapses the "
        "measurement fault distance to 1 — exactly the constant-weight "
        "undetectable operator Remark 23 predicts. Folding the bridge (adapter "
        "qubits B + gauge checks Uᴮ) is what restores detectability and flips "
        "this test to PASS."
    ),
    strict=True,
)
def test_single_y_dem_has_no_undetectable_observable_error() -> None:
    """No single-fault DEM term flips obs0 without firing a detector.

    A DEM entry ``error(p) L0`` (observable flip, no detector flip) is an
    undetectable logical error — the measurement fault distance has collapsed
    to 1. This is exactly the constant-weight undetectable operator
    ``P = Z̄_M · q1 · ∏ H_Z(odd-layer V)`` that Cross, He, Rall, Yoder
    arXiv:2407.18393 Remark 23 proves arises WHEN THE BRIDGE IS OMITTED; the
    §3.7 bridge (qubits B + gauge checks Uᴮ) is what restores detectability.
    This is the DEFERRED §3.7-bridge / fault-distance task — out of scope for
    the obs0 readout fix (which only needs determinism, verified separately).
    """
    from qldpc.circuits.noise_model import DepolarizingNoiseModel
    from qldpc.circuits.surgery import build_single_y_ppm_circuit, keep_only_observable

    code, x, z = _steane_y_pair()
    yg = build_y_gadget(code, x=x, z=z)
    circuit = build_single_y_ppm_circuit(
        yg,
        rounds=3,
        data_init="Y+",
        noise_model=DepolarizingNoiseModel(0.001, include_idling_error=False),
    )
    circuit = keep_only_observable(circuit, keep_idx=0)
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    offenders = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        det_targets = [t for t in inst.targets_copy() if t.is_relative_detector_id()]
        obs_targets = [t for t in inst.targets_copy() if t.is_logical_observable_id()]
        if obs_targets and not det_targets:
            offenders.append(inst)
            if len(offenders) >= 3:
                break
    assert not offenders, (
        f"{len(offenders)}+ DEM error term(s) flip obs0 with no detector firing "
        f"(measurement fault distance = 1; bridge omitted?). Examples: {offenders}"
    )
