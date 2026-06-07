"""Load Ide et al. (arXiv:2410.03628) Zenodo supplementary matrices.

These matrices live under tests/fixtures/ide_zenodo/ and serve as
ground truth for stab-group equality assertions in joint tests.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.io import mmread

_FIXTURE_ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "ide_zenodo"


def fixtures_available() -> bool:
    return _FIXTURE_ROOT.exists() and any(_FIXTURE_ROOT.iterdir())


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


def load_ide_skiptree_TPG(path: str) -> dict[str, np.ndarray]:
    """Parse one of Ide's *_GTP.txt files. Returns dict with keys G_mat_*, T_*, P_*."""
    text = (_FIXTURE_ROOT / path).read_text()
    out: dict[str, np.ndarray] = {}
    pattern = re.compile(r"^([A-Za-z_0-9]+)\s*=\s*np\.array\((.*?)\)\s*$",
                          re.MULTILINE | re.DOTALL)
    for match in pattern.finditer(text):
        name = match.group(1)
        # safe eval: only floats, lists, np.array allowed
        arr = eval(match.group(2), {"np": np}, {})
        out[name] = np.array(arr, dtype=int)
    return out
