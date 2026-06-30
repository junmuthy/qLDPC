"""Webster (arXiv:2511.15989) Appendix A seed-set fixture + helpers.

Private to the surgery test suite and the `examples/lattice_surgery.ipynb`
demo. Not part of the public API. Pytest's default discovery skips this
file (no `def test_*` + leading underscore).

Provides:

* `load_webster_seed_set(code_index)` — read the Webster Appendix A seed
  set for code index 0..3.
* `build_generalised_bicycle_code(l, A_set, B_set)` — build a CSS code
  from cyclic exponent sets per Kovalev-Pryadko arXiv:1212.6703.
* `_webster_x_bar_operator(data, name, pauli_type)` — extract a named
  logical operator from a seed-set dict as a length-2l bit-vector.
* `_webster_z_bar_operator(data, name)` — Z-type convenience wrapper.
* `_steane_y_pair()` / `_bb_y_pair(overlap)` — logical (x, z) pair fixtures
  whose supports overlap on a chosen number of qubits, for Ȳ = iX̄Z̄ tests.
"""

from __future__ import annotations

import itertools
from typing import Any

import galois
import numpy as np

from qldpc import codes
from qldpc.circuits.surgery.y_gadget import _overlap_size
from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli

GF2 = galois.GF(2)

# Inlined verbatim from the former _webster_app_a.json (Webster App. A seeds).
_WEBSTER_APP_A: dict[str, Any] = {
    'codes': [{'A': [0, 6, 15],
               'B': [0, 5, 7],
               'distance': 6,
               'expected_bare_gadget_qubits_per_seed': 19,
               'expected_bridge_qubits_per_pair': 11,
               'expected_cheeger_boost_qubits': 0,
               'k_logical': 10,
               'l': 31,
               'n_data_qubits': 62,
               'name': '62_10_6',
               'seeds': [{'L_support': [1, 6, 8, 10],
                          'R_support': [11, 26],
                          'name': 'X_bar_1',
                          'pauli_type': 'X'},
                         {'L_support': [3, 12, 18, 19],
                          'R_support': [11, 18],
                          'name': 'Z_bar_1',
                          'pauli_type': 'Z'},
                         {'L_support': [16, 23],
                          'R_support': [0, 15, 16, 22],
                          'name': 'X_bar_k2p1',
                          'pauli_type': 'X'},
                         {'L_support': [0, 16],
                          'R_support': [1, 3, 5, 10],
                          'name': 'Z_bar_k2p1',
                          'pauli_type': 'Z'}]},
              {'A': [0, 4, 37],
               'B': [0, 29, 49],
               'distance': 10,
               'expected_bare_gadget_qubits_per_seed': 31,
               'expected_bridge_qubits_per_pair': 19,
               'expected_cheeger_boost_qubits': 0,
               'k_logical': 12,
               'l': 63,
               'n_data_qubits': 126,
               'name': '126_12_10',
               'seeds': [{'L_support': [7, 12, 36, 41, 56],
                          'R_support': [1, 27, 31, 38, 61],
                          'name': 'X_bar_1',
                          'pauli_type': 'X'},
                         {'L_support': [5, 15, 28, 35, 45, 61],
                          'R_support': [1, 11, 54, 57],
                          'name': 'Z_bar_1',
                          'pauli_type': 'Z'},
                         {'L_support': [9, 19, 26, 29],
                          'R_support': [5, 15, 22, 38, 48, 55],
                          'name': 'X_bar_k2p1',
                          'pauli_type': 'X'},
                         {'L_support': [2, 25, 32, 36, 62],
                          'R_support': [7, 22, 27, 51, 56],
                          'name': 'Z_bar_k2p1',
                          'pauli_type': 'Z'}]},
              {'A': [0, 32, 100],
               'B': [0, 28, 49],
               'distance': 16,
               'expected_bare_gadget_qubits_per_seed': 49,
               'expected_bridge_qubits_per_pair': 31,
               'expected_cheeger_boost_qubits': 8,
               'k_logical': 14,
               'l': 127,
               'n_data_qubits': 254,
               'name': '254_14_16',
               'seeds': [{'L_support': [28, 47, 55, 75, 103, 114, 124],
                          'R_support': [4, 14, 15, 23, 50, 77, 83, 109, 123],
                          'name': 'X_bar_1',
                          'pauli_type': 'X'},
                         {'L_support': [1, 24, 33, 51, 60, 65, 107, 119, 124],
                          'R_support': [7, 8, 36, 85, 106, 114, 124],
                          'name': 'Z_bar_1',
                          'pauli_type': 'Z'},
                         {'L_support': [3, 31, 32, 42, 52, 60, 81],
                          'R_support': [6, 15, 38, 42, 47, 59, 101, 106, 115],
                          'name': 'X_bar_k2p1',
                          'pauli_type': 'X'},
                         {'L_support': [0, 8, 9, 19, 27, 41, 67, 73, 100],
                          'R_support': [26, 36, 47, 75, 95, 103, 122],
                          'name': 'Z_bar_k2p1',
                          'pauli_type': 'Z'}]},
              {'A': [0, 39, 55],
               'B': [0, 70, 127],
               'distance': 24,
               'expected_bare_gadget_qubits_per_seed': 79,
               'expected_bridge_qubits_per_pair': 51,
               'expected_cheeger_boost_qubits': 20,
               'k_logical': 16,
               'l': 255,
               'n_data_qubits': 510,
               'name': '510_16_24',
               'seeds': [{'L_support': [18,
                                        31,
                                        35,
                                        36,
                                        91,
                                        126,
                                        146,
                                        163,
                                        164,
                                        180,
                                        196,
                                        216,
                                        233,
                                        253],
                          'R_support': [48,
                                        52,
                                        87,
                                        101,
                                        103,
                                        106,
                                        107,
                                        125,
                                        140,
                                        156,
                                        179,
                                        211],
                          'name': 'X_bar_1',
                          'pauli_type': 'X'},
                         {'L_support': [38,
                                        54,
                                        57,
                                        93,
                                        112,
                                        148,
                                        164,
                                        185,
                                        197,
                                        203,
                                        213,
                                        238,
                                        240,
                                        252],
                          'R_support': [18,
                                        55,
                                        59,
                                        73,
                                        129,
                                        130,
                                        142,
                                        182,
                                        187,
                                        199,
                                        244,
                                        252],
                          'name': 'Z_bar_1',
                          'pauli_type': 'Z'},
                         {'L_support': [6,
                                        27,
                                        35,
                                        80,
                                        92,
                                        97,
                                        137,
                                        149,
                                        150,
                                        206,
                                        220,
                                        224],
                          'R_support': [27,
                                        39,
                                        41,
                                        66,
                                        76,
                                        82,
                                        94,
                                        115,
                                        131,
                                        167,
                                        186,
                                        222,
                                        225,
                                        241],
                          'name': 'X_bar_k2p1',
                          'pauli_type': 'X'},
                         {'L_support': [10,
                                        11,
                                        14,
                                        16,
                                        30,
                                        65,
                                        69,
                                        161,
                                        193,
                                        216,
                                        232,
                                        247],
                          'R_support': [26,
                                        81,
                                        82,
                                        86,
                                        99,
                                        119,
                                        139,
                                        156,
                                        176,
                                        192,
                                        208,
                                        209,
                                        226,
                                        246],
                          'name': 'Z_bar_k2p1',
                          'pauli_type': 'Z'}]}],
    'source': 'Webster, Smith, Cohen, arXiv:2511.15989 Appendix A'}


def load_webster_seed_set(code_index: int) -> dict[str, Any]:
    """Load Webster (arXiv:2511.15989) Appendix A data for code index 0..3.

    Returns:
        The Webster Appendix A seed dict for ``code_index``.

    Raises:
        IndexError: if code_index is not in 0..3.
    """
    if not 0 <= code_index <= 3:
        raise IndexError(f"code_index must be in 0..3, got {code_index}")
    data = _WEBSTER_APP_A
    return data["codes"][code_index]


def build_generalised_bicycle_code(ell: int, A_set: list[int], B_set: list[int]) -> CSSCode:
    """Build a generalised bicycle code from cyclic exponent sets A, B.

    Per Kovalev-Pryadko (arXiv:1212.6703) and Swaroop's reference
    implementation (https://github.com/eswaroop/adapters-LDPC-surgery,
    ext/bivariate_bicyclic.py): given subsets A, B of Z_ell, let A(x) =
    sum(x^a for a in A_set) and B(x) = sum(x^b for b in B_set) as cyclic
    matrices in F_2[Z_ell]. Then H_X = [A | B] and H_Z = [B^T | A^T] define
    the bicycle code on 2*ell data qubits.

    Args:
        ell: cyclic group order.
        A_set, B_set: subsets of {0, 1, ..., ell-1}.

    Returns:
        CSSCode on 2*ell data qubits with check matrices [A | B] and
        [B^T | A^T] over GF(2).
    """
    I_ell = np.eye(ell, dtype=np.int_)
    S = np.roll(I_ell, shift=-1, axis=0)
    A = np.zeros((ell, ell), dtype=np.int_)
    for a in A_set:
        A = (A + np.linalg.matrix_power(S, a)) % 2
    B = np.zeros((ell, ell), dtype=np.int_)
    for b in B_set:
        B = (B + np.linalg.matrix_power(S, b)) % 2

    H_X = np.hstack([A, B])
    H_Z = np.hstack([B.T, A.T])

    return CSSCode(H_X, H_Z, is_subsystem_code=False)


def _webster_x_bar_operator(
    data: dict[str, Any],
    name: str = "X_bar_1",
    pauli_type: str = "X",
) -> np.ndarray:
    """Extract the named logical operator from a Webster seed_set dict.

    L_support and R_support are sparse index lists (positions within each ell-half
    that are set to 1). Returns a dense binary vector of length 2*ell.

    Args:
        data: Webster seed set dict (from load_webster_seed_set).
        name: Seed name, e.g. "X_bar_1", "Z_bar_1".
        pauli_type: "X" or "Z"; filters seeds by pauli_type field.
    """
    ell = data["l"]
    for seed in data["seeds"]:
        if seed["name"] == name and seed["pauli_type"] == pauli_type:
            v_L = np.zeros(ell, dtype=np.uint8)
            v_L[seed["L_support"]] = 1
            v_R = np.zeros(ell, dtype=np.uint8)
            v_R[seed["R_support"]] = 1
            return np.concatenate([v_L, v_R])
    raise ValueError(f"{name!r} (pauli_type={pauli_type!r}) seed not found")


def _webster_z_bar_operator(data: dict[str, Any], name: str = "Z_bar_1") -> np.ndarray:
    """Extract the named Z-type logical operator from a Webster seed_set dict.

    Convenience wrapper around _webster_x_bar_operator with pauli_type="Z".
    """
    return _webster_x_bar_operator(data, name, pauli_type="Z")


def _steane_y_pair() -> tuple[CSSCode, np.ndarray, np.ndarray]:
    """Return a Steane code and a logical (x, z) pair overlapping on exactly one qubit.

    This is the clean single-overlap (``|W|=1``) fixture for the Ȳ-overlap
    ``W = supp(x) ∩ supp(z)`` of Ȳ = iX̄Z̄ (Ide, Gowda, Nadkarni, Dauphinais
    arXiv:2410.02753 §III.D). The canonical
    weight-3 logical-X and logical-Z
    representatives of the [[7,1,3]] Steane code already cross on a single data
    qubit; if a given representative pair does not, it is reduced over GF(2) by
    adding stabilizer rows (a logical-X support stays logical-X after XOR-ing any
    row of ``matrix_x``; a logical-Z support stays logical-Z after XOR-ing any row
    of ``matrix_z``) until the supports overlap on exactly one qubit.
    """
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)

    if _overlap_size(x, z) == 1:
        return code, x, z

    # Fallback: search small GF(2) combinations of stabilizer rows added to x and/or
    # z. Adding an X-stabilizer row to x leaves it a logical-X representative;
    # adding a Z-stabilizer row to z leaves it a logical-Z representative.
    rows_x = GF2(np.asarray(code.matrix_x).astype(np.uint8))
    rows_z = GF2(np.asarray(code.matrix_z).astype(np.uint8))
    x_gf = GF2(x)
    z_gf = GF2(z)
    n_rx = rows_x.shape[0]
    n_rz = rows_z.shape[0]

    for k_x in range(n_rx + 1):
        for cols_x in itertools.combinations(range(n_rx), k_x):
            x_cand = x_gf + (sum((rows_x[i] for i in cols_x), GF2.Zeros(x_gf.shape)))
            for k_z in range(n_rz + 1):
                for cols_z in itertools.combinations(range(n_rz), k_z):
                    z_cand = z_gf + (sum((rows_z[j] for j in cols_z), GF2.Zeros(z_gf.shape)))
                    x_arr = np.asarray(x_cand).astype(np.uint8)
                    z_arr = np.asarray(z_cand).astype(np.uint8)
                    if _overlap_size(x_arr, z_arr) == 1:
                        return code, x_arr, z_arr

    raise ValueError(
        "BLOCKED: no overlap-1 logical (x, z) pair reachable on the Steane code by "
        "stabilizer-row reduction; the multi-overlap (|W|≥2) merge would be required "
        "(Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.D)"
    )


def _bb_y_pair(overlap: int = 1) -> tuple[CSSCode, np.ndarray, np.ndarray]:
    """BB [[36,8,4]] fixture for Ȳ on logical qubit 0 with chosen |W|.

    overlap=1: canonical reps (already single-overlap). overlap=3: add stabilizer
    rows (deterministic seed) until supp(x)∩supp(z) has size 3 — the |W|≥2
    crossing-cycle regime of arXiv:2410.02753 §III.D.
    """
    import sympy

    xs, ys = sympy.symbols("x y")
    code = codes.BBCode({xs: 3, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)
    n = code.num_qudits
    LX = np.asarray(code.get_logical_ops(Pauli.X)).astype(np.uint8)
    LZ = np.asarray(code.get_logical_ops(Pauli.Z)).astype(np.uint8)
    wide = LX.shape[1] == 2 * n
    x = (LX[0][:n] if wide else LX[0]).astype(np.uint8)
    z = (LZ[0][n:] if wide else LZ[0]).astype(np.uint8)
    if overlap == 1:
        return code, x, z
    if overlap == 3:
        HX = np.asarray(code.matrix_x).astype(np.uint8)
        HZ = np.asarray(code.matrix_z).astype(np.uint8)
        rng = np.random.default_rng(0)
        for _ in range(20000):
            ax = (
                rng.integers(0, 2, HX.shape[0])
                if rng.random() < 0.5
                else np.zeros(HX.shape[0], int)
            )
            az = rng.integers(0, 2, HZ.shape[0])
            xc = (x ^ (ax @ HX % 2)).astype(np.uint8)
            zc = (z ^ (az @ HZ % 2)).astype(np.uint8)
            if not xc.any() or not zc.any():
                continue
            if (
                int(np.count_nonzero(xc.astype(bool) & zc.astype(bool))) == 3
                and int(xc.sum()) <= 12
                and int(zc.sum()) <= 12
            ):
                return code, xc, zc
        raise ValueError("BLOCKED: no |W|=3 BB representative found in budget")
    raise ValueError(f"overlap must be 1 or 3, got {overlap}")
