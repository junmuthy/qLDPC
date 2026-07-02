# edge_expanded_e2e_test.py
"""End-to-end DEM validity for the edge-expanded homological gadget.

Steane [[7,1,3]] X̄ = X0X1X2 gadget, built via the edge-expanded homological
measurement of Benjamin Ide, Manoj G. Gowda, Priya J. Nadkarni, Guillaume
Dauphinais, "Fault-Tolerant Logical Measurements via Homological Measurement",
arXiv:2410.02753 (Algorithm 3, mapping cone Eq 12/13), run through the
single-PPM surgery experiment of Cain et al., "Fast correlated decoding of
transversal logical algorithms", arXiv:2603.28627 (Appendix B.1 / D):

- the circuit compiles to a detector error model with exactly one observable
  after ``keep_only_observable`` (only the target X̄ is measured, arXiv:2410.02753
  Remark 3);
- the merged Z-check weight stays within native+1 (the low-weight-∂_0 win;
  on Steane ∂_0 is empty).
"""
import numpy as np, galois
from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli
from qldpc.circuits.surgery import build_gadget, build_single_ppm_circuit, keep_only_observable

_STEANE = np.array([[0,0,0,1,1,1,1],[0,1,1,0,0,1,1],[1,0,1,0,1,0,1]], dtype=np.uint8)

def _steane():
    GF2 = galois.GF(2)
    return CSSCode(GF2(_STEANE), GF2(_STEANE), is_subsystem_code=False)

def test_dem_compiles_and_one_observable():
    # Steane [[7,1,3]], X̄ = X0 X1 X2. End-to-end: build gadget -> circuit -> DEM.
    code = _steane()
    x = np.array([1,1,1,0,0,0,0], dtype=np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    circ = build_single_ppm_circuit(g, rounds=3)
    circ = keep_only_observable(circ, 0)
    dem = circ.detector_error_model(decompose_errors=False)
    assert dem.num_observables == 1                         # only X̄ measured (Remark 3)

def test_merged_z_check_weight_bounded():
    # Structural weight guard: merged Z-check weight stays near the native weight
    # (the low-weight-∂0 win). On Steane ∂0 is empty, so this is native+1.
    code = _steane()
    x = np.array([1,1,1,0,0,0,0], dtype=np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    native = int(np.asarray(code.matrix_z).sum(axis=1).max())
    assert int(np.asarray(g.HZ_merged).astype(int).sum(axis=1).max()) <= native + 1
