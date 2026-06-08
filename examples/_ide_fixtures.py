"""Load Ide et al. (arXiv:2410.03628) Zenodo supplementary matrices.

These matrices live under tests/fixtures/ide_zenodo/ and serve as
ground truth for stab-group assertions and as a direct source of joint
deformed codes for the two paper examples (BB-LP inter-code §VII B and
BB-BB intra-code §VII C).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
from scipy.io import mmread

from qldpc.codes.common import CSSCode

_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "ide_zenodo"


def fixtures_available() -> bool:
    sentinel = _FIXTURE_ROOT / "BB_98_LP_200_adapter" / "Hx_intercode_BB_LP_adapter-Z_1_Z_2_deformed-code.mtx"
    return sentinel.exists()


def load_mtx(rel_path: str) -> np.ndarray:
    return mmread(_FIXTURE_ROOT / rel_path).toarray().astype(np.int_)


def load_ide_joint_BB_LP() -> tuple[np.ndarray, np.ndarray]:
    """Return Ide's HX, HZ for the [[355, 25, 10]] BB-LP joint code."""
    HX = load_mtx("BB_98_LP_200_adapter/Hx_intercode_BB_LP_adapter-Z_1_Z_2_deformed-code.mtx")
    HZ = load_mtx("BB_98_LP_200_adapter/Hz_intercode_BB_LP_adapter-Z_1_Z_2_deformed-code.mtx")
    return HX, HZ


def load_ide_joint_BB_intracode() -> tuple[np.ndarray, np.ndarray]:
    """Return Ide's HX, HZ for the [[150, 5, 12]] BB Z_1 Z_3 joint code."""
    HX = load_mtx("BB_98_intracode_adapter/Hx_BB_intracode_Z_1_Z_3_adapted-code.mtx")
    HZ = load_mtx("BB_98_intracode_adapter/Hz_BB_intracode_Z_1_Z_3_adapted-code.mtx")
    return HX, HZ


# =============================================================================
# Reverse-engineered structural data from Ide BB-LP joint mtx (Zenodo).
# See docs/superpowers/notes/2026-06-07-ide-reverse-engineering.md for full
# decode notes. These constants encode the §VII B port function and the
# BB-side X-stab deformation rule for the specific BB_1 Z̄_1 × LP_2 Z̄_2
# example. They are not derivable from the paper text alone.
# =============================================================================

#: Port function (BB V_0_1 qubit → LP qubit at corresponding label) for the
#: BB_1 Z̄_1 × LP_2 Z̄_2 inter-code joint. Recovered from the 14 cross-block
#: HZ rows in the published joint matrix.
IDE_BB_LP_PORT = {
    6: 0, 8: 1, 32: 2, 33: 3, 93: 4, 35: 5, 36: 6,
    41: 7, 17: 8, 37: 9, 13: 10, 50: 11, 51: 12, 31: 13,
}

#: LP V_0^(2) qubits for Ide's §VII.B Z̄_2 single PPM. Recovered from the
#: 14 cross-block Vl rows in `Hz_LP_200_20_10_aux-graph-Z_2-deformed-code.mtx`
#: (each row has data-weight 1; their support is V_0^(2)). These are NOT
#: the same as `IDE_BB_LP_PORT.values()` — the port relabels them to
#: {0..13} on the LP side of the joint code.
IDE_LP_V0_2 = (24, 25, 26, 29, 30, 56, 58, 59, 60, 61, 90, 93, 94, 121)

#: BB κ_1 edges in Ide's cellulated G_1 (23 edges = 21 spanning-tree + 2 cellulation).
#: Maps κ_1 ancilla index → (v_a, v_b) ∈ V_0_1 × V_0_1 (sorted).
IDE_BB_KAPPA1_EDGES = {
    0: (6, 50),   1: (6, 51),   2: (13, 93),  3: (13, 31),  4: (8, 32),
    5: (8, 33),   6: (17, 35),  7: (36, 50),  8: (37, 51),  9: (17, 41),
    10: (31, 32), 11: (32, 33), 12: (33, 93), 13: (6, 31),  14: (8, 50),
    15: (41, 51), 16: (35, 41), 17: (35, 36), 18: (36, 37), 19: (13, 37),
    20: (17, 93), 21: (33, 37), 22: (8, 13),
}


def build_joint_from_ide_fixture(example: str) -> CSSCode:
    """Return a CSSCode for one of Ide's two published joint deformed codes.

    Use when you need the paper-exact joint code as a CSSCode object
    (e.g., for distance / decoder experiments). The algorithmic
    construction in build_joint_measurement_code does NOT yet reproduce
    Ide's inter-code joint (see docs/superpowers/specs/2026-06-07-
    skiptree-bridge-v3-design.md §1 "future work").

    Args:
        example: "BB_LP" for [[355, 25, 10]] (§VII B) or "BB_BB" for
            [[150, 5, 12]] (§VII C).

    Returns:
        CSSCode with the paper's HX, HZ as F_2 matrices.

    Raises:
        ValueError: unknown example name.
        FileNotFoundError: Zenodo fixtures not installed.
    """
    if not fixtures_available():
        raise FileNotFoundError(
            f"Zenodo fixtures not found at {_FIXTURE_ROOT}; install them "
            f"from https://zenodo.org/records/17527545 (data_qLDPC_surgery.zip)."
        )
    if example == "BB_LP":
        HX, HZ = load_ide_joint_BB_LP()
    elif example == "BB_BB":
        HX, HZ = load_ide_joint_BB_intracode()
    else:
        raise ValueError(f"unknown example '{example}'; want 'BB_LP' or 'BB_BB'")
    import galois
    GF2 = galois.GF(2)
    return CSSCode(GF2(HX), GF2(HZ), is_subsystem_code=False)


def load_ide_BB_input_with_operator() -> tuple[CSSCode, np.ndarray]:
    """Return BB INPUT code (n=98) + pinned Z̄_1 logical operator (Ide §VII.B).

    The operator support V_0^(1) is the set of vertices appearing in
    `IDE_BB_KAPPA1_EDGES`. Z̄_1 is a Z-type logical, so its support
    commutes with the X-stabilizers (HX @ V_0 = 0).

    Use with ``build_gadget(code, x, basis=Pauli.Z)``.
    """
    if not fixtures_available():
        raise FileNotFoundError(
            f"Zenodo fixtures not found at {_FIXTURE_ROOT}."
        )
    HX = load_mtx("BB_98_6_12/original_codes/Hx_BB_98_6_12_original-code-canonicalbasis.mtx")
    HZ = load_mtx("BB_98_6_12/original_codes/Hz_BB_98_6_12_original-code-canonicalbasis.mtx")
    import galois
    GF2 = galois.GF(2)
    code = CSSCode(GF2(HX.tolist()), GF2(HZ.tolist()), is_subsystem_code=False)

    V0 = sorted({v for edge in IDE_BB_KAPPA1_EDGES.values() for v in edge})
    x = np.zeros(code.num_qudits, dtype=np.uint8)
    for v in V0:
        x[v] = 1
    return code, x


def load_ide_LP_input_with_operator() -> tuple[CSSCode, np.ndarray]:
    """Return LP INPUT code (n=200) + pinned Z̄_2 logical operator (Ide §VII.B).

    The operator support V_0^(2) is the constant `IDE_LP_V0_2` — the data
    qubits appearing on the 14 Vl rows of Ide's published LP Z̄_2 single
    deformed code. ``IDE_BB_LP_PORT.values()`` are *not* V_0^(2); they
    are the relabelled positions {0..13} used on the LP side of the joint
    code.

    Use with ``build_gadget(code, x, basis=Pauli.Z)``.
    """
    if not fixtures_available():
        raise FileNotFoundError(
            f"Zenodo fixtures not found at {_FIXTURE_ROOT}."
        )
    HX = load_mtx("LP_200_20_10/original_codes/Hx_LP_200_20_10_original-code.mtx")
    HZ = load_mtx("LP_200_20_10/original_codes/Hz_LP_200_20_10_original-code.mtx")
    import galois
    GF2 = galois.GF(2)
    code = CSSCode(GF2(HX.tolist()), GF2(HZ.tolist()), is_subsystem_code=False)

    x = np.zeros(code.num_qudits, dtype=np.uint8)
    for v in IDE_LP_V0_2:
        x[v] = 1
    return code, x


def load_ide_skiptree_TPG(path: str) -> dict[str, np.ndarray]:
    """Parse one of Ide's *_GTP.txt files.

    Args:
        path: relative path from ``tests/fixtures/ide_zenodo/`` (e.g.
            ``"BB_98_LP_200_adapter/skipTree_transformations/BB_98_6_12_Z_1_GTP.txt"``).

    Returns:
        Dict mapping variable name (e.g. ``"G_mat_1"``, ``"T_1"``, ``"P_1"``)
        to its numpy int array, parsed from the Python array literals in the file.
    """
    text = (_FIXTURE_ROOT / path).read_text()
    out: dict[str, np.ndarray] = {}
    pattern = re.compile(r"^([A-Za-z_0-9]+)\s*=\s*np\.array\((.*?)\)\s*$",
                          re.MULTILINE | re.DOTALL)
    for match in pattern.finditer(text):
        name = match.group(1)
        # trusted Zenodo input, parsed via ast.literal_eval
        inner = match.group(2).strip()
        arr = ast.literal_eval(inner)
        out[name] = np.array(arr, dtype=int)
    return out
