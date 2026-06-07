"""Stim surgery circuit construction.

build_single_ppm_circuit  — single-PPM measurement (gadget alone)
build_joint_ppm_circuit   — two-PPM joint measurement (gadget + gadget + bridge)
"""

from __future__ import annotations

import numpy as np
import stim

from qldpc.codes.common import CSSCode
from qldpc.circuits.memory.memory import get_memory_experiment
from qldpc.objects import Pauli

from .gadget import GadgetLayout


def _gadget_merged_csscode(g: GadgetLayout) -> CSSCode:
    return CSSCode(
        g.HX_merged.astype(np.int_),
        g.HZ_merged.astype(np.int_),
        is_subsystem_code=False,
    )


def build_single_ppm_circuit(
    gadget: GadgetLayout,
    *,
    rounds: int,
    noise_model=None,
) -> stim.Circuit:
    """Stim circuit for single-PPM measurement using `gadget`.

    Builds the merged CSS code (data + κ ancillas) and delegates to the
    existing memory experiment infrastructure.
    """
    merged = _gadget_merged_csscode(gadget)
    return get_memory_experiment(merged, basis=Pauli.X, num_rounds=rounds, noise_model=noise_model)
