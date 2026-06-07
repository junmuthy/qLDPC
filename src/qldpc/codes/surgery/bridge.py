"""Standalone bridge adapter for two-PPM joint surgery (math.md §2).

Handles both intra-code (g1.code is g2.code) and inter-code joints. SkipTree
and cellulation helpers are private to this module.
"""

from __future__ import annotations

import dataclasses

import galois
import numpy as np

from .gadget import GadgetLayout

GF2 = galois.GF(2)


@dataclasses.dataclass(frozen=True, eq=False)
class Bridge:
    width: int
    qubits: tuple[int, ...]
    U_B: np.ndarray
    chi_endpoint_extensions: dict[int, np.ndarray]
    intercode: bool
    aux_graph_edges: tuple[tuple[int, int], ...] | None
    z_extensions: dict[int, np.ndarray] | None


def _build_path_graph_U_B(w: int) -> np.ndarray:
    """math.md §2.2 — path-graph X-stabilizers on w bridge qubits."""
    if w < 2:
        raise ValueError(f"bridge width must be >= 2, got {w}")
    U_B = np.zeros((w - 1, w), dtype=np.uint8)
    for i in range(w - 1):
        U_B[i, i] = 1
        U_B[i, i + 1] = 1
    return U_B


def build_bridge(g1: GadgetLayout, g2: GadgetLayout) -> "Bridge":
    """Two-PPM bridge between gadgets. Auto-dispatches intra vs inter-code.

    math.md §2: bridge data qubits + path-graph U_B + chi endpoint extensions.
    """
    intercode = g1.code is not g2.code
    w = min(len(g1.V0), len(g2.V0))
    if w < 2:
        raise ValueError(f"bridge width must be >= 2, got {w}")

    qubits = tuple(range(w))  # relative offsets; circuit.py rebases.

    U_B = _build_path_graph_U_B(w)

    # math.md §2.3 χ-extension
    # gadget 1's χ_0 row → X on bridge[0]
    # gadget 2's χ_0 row → X on bridge[w-1]
    chi_endpoint_extensions: dict[int, np.ndarray] = {
        0: np.array([0], dtype=np.uint8),
    }

    if not intercode:
        return Bridge(
            width=w, qubits=qubits, U_B=U_B,
            chi_endpoint_extensions=chi_endpoint_extensions,
            intercode=False,
            aux_graph_edges=None,
            z_extensions=None,
        )

    # Inter-code path added in Task 12.
    raise NotImplementedError("inter-code bridge added in Task 12")
