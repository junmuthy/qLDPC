"""Webster (arXiv:2511.15989) Appendix A seed-set test fixture.

Private to the surgery test suite + the `examples/lattice_surgery.ipynb`
demo. Not part of the public API; the leading `_test_` prefix matches the
convention used by `_test_helpers.py` and the `_test_<feature>.py` modules
(pytest collects this file, finds no `def test_*`, and moves on).

Provides:

* `load_webster_seed_set(code_index)` — read the JSON fixture for code
  index 0..3.
* `build_generalised_bicycle_code(l, A_set, B_set)` — build a CSS code
  from cyclic exponent sets per Kovalev-Pryadko arXiv:1212.6703.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qldpc.codes.common import CSSCode

_WEBSTER_APP_A_PATH = Path(__file__).resolve().parent / "_test_webster_app_a.json"


def load_webster_seed_set(code_index: int) -> dict:
    """Load Webster (arXiv:2511.15989) Appendix A data for code index 0..3.

    Returns:
        A dict matching the JSON schema.

    Raises:
        IndexError: if code_index is not in 0..3.
        FileNotFoundError: if the JSON fixture is missing.
    """
    if not 0 <= code_index <= 3:
        raise IndexError(f"code_index must be in 0..3, got {code_index}")
    with _WEBSTER_APP_A_PATH.open() as fh:
        data = json.load(fh)
    return data["codes"][code_index]


def build_generalised_bicycle_code(l: int, A_set: list[int], B_set: list[int]) -> CSSCode:
    """Build a generalised bicycle code from cyclic exponent sets A, B.

    Per Kovalev-Pryadko (arXiv:1212.6703) and Swaroop's reference
    implementation (https://github.com/eswaroop/adapters-LDPC-surgery,
    ext/bivariate_bicyclic.py): given subsets A, B of Z_l, let A(x) =
    sum(x^a for a in A_set) and B(x) = sum(x^b for b in B_set) as cyclic
    matrices in F_2[Z_l]. Then H_X = [A | B] and H_Z = [B^T | A^T] define
    the bicycle code on 2l data qubits.

    Args:
        l: cyclic group order.
        A_set, B_set: subsets of {0, 1, ..., l-1}.

    Returns:
        CSSCode on 2l data qubits with check matrices [A | B] and
        [B^T | A^T] over GF(2).
    """
    I_l = np.eye(l, dtype=np.int_)
    S = np.roll(I_l, shift=-1, axis=0)
    A = np.zeros((l, l), dtype=np.int_)
    for a in A_set:
        A = (A + np.linalg.matrix_power(S, a)) % 2
    B = np.zeros((l, l), dtype=np.int_)
    for b in B_set:
        B = (B + np.linalg.matrix_power(S, b)) % 2

    H_X = np.hstack([A, B])
    H_Z = np.hstack([B.T, A.T])

    return CSSCode(H_X, H_Z, is_subsystem_code=False)
