"""Tests for the single logical-Y measurement circuit (Ȳ = iX̄Z̄).

Task-4 acceptance: ``build_single_y_ppm_circuit`` builds a ``stim.Circuit``
for the single-overlap Y-gadget merged code (Cross, He, Rall, Yoder
arXiv:2407.18393 §3.7) and that circuit compiles to a
``stim.DetectorErrorModel`` with the logical-Y eigenstate prep
``data_init="Y+"``.
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
