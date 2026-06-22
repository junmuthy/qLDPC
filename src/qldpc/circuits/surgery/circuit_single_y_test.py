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
