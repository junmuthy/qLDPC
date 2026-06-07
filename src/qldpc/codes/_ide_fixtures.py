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

_FIXTURE_ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "ide_zenodo"


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
