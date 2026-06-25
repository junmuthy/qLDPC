"""Byte-identical golden-master for the §2/§3 paper-notation refactor.

These sha256 hashes pin the EXACT merged check matrices and emitted circuits of the
single- and joint-PPM construction. The refactor (docs/superpowers/plans/
2026-06-25-single-joint-ppm-paper-notation.md) is a pure rename + π-form re-expression
with NO behavioral change, so every hash below MUST stay identical.

If a hash changes, the refactor altered a matrix or circuit — that is a BUG.
Do NOT update the hashes to make this pass.
"""

from __future__ import annotations

import hashlib

import numpy as np

from qldpc import codes
from qldpc.objects import Pauli

from ._webster_fixture import (
    _webster_x_bar_operator,
    build_generalised_bicycle_code,
    load_webster_seed_set,
)


def _h(a) -> str:
    arr = np.ascontiguousarray(np.asarray(a).astype(np.uint8))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _hs(s) -> str:
    return hashlib.sha256(str(s).encode()).hexdigest()


GOLDEN = {
    "single_steane_X_HX": "5e071db6d229df9861a7e0221dcb74ed7ec9e3effcad43cbc84e82d1844a6f64",
    "single_steane_X_HZ": "0483c2c21a1593e299dbf1038aad2c1426de8e193fda18a3eefdda30531e80d8",
    "single_steane_Z_HX": "c68a2b6cc52492445c1c1544b3d3fb1774b49bd59b22e8937cb6d21896ad8713",
    "single_steane_Z_HZ": "0a149bfde7f21a3049b9cb6b2540d105a6c45ec4dd91ea5cafe488163612fbb2",
    "single_gb_X_HX": "a83ba1b89872bab854f0a225b97ee1338a88ee8fec1f829fd49144d460c16892",
    "single_gb_X_HZ": "35d3a73c3ff7b6ab49b5ae824a7cb77f806122bc51a51bb08d6e5811b95ac78c",
    "joint_inter_X_HX": "d2a9209574889d3a64ac62178afc7340911a69076e986623d2cea49c9a8cc639",
    "joint_inter_X_HZ": "90d71ef17ea8620eb8617dfd85618cea6f2780af7253fff9345e0a26eb8308c2",
    "joint_intra_X_HX": "aeeba0cbc1577efc3dda18c275d67a361fc228311baf335f27decbdfa923b5ed",
    "joint_intra_X_HZ": "f70ae66ab3f20920b4d69b249f99360829ee01da27fb5920844de6ef1665d0cb",
    "circ_single_X": "2a1f149a0958f0e36a33ccf5395e41539330048a6a4a437d681737acc4a6d19b",
    "circ_joint_intra": "ff00e74186550ccdc8d747693b15e952a9c6e335c9cb890693becc7f6a61d598",
}


def _steane_x():
    code = codes.SteaneCode()
    return code, np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)


def test_single_steane_basis_x_matrices_byte_identical() -> None:
    from qldpc.circuits.surgery.gadget import build_gadget

    code, x = _steane_x()
    g = build_gadget(code, x, basis=Pauli.X)
    assert _h(g.HX_merged) == GOLDEN["single_steane_X_HX"]
    assert _h(g.HZ_merged) == GOLDEN["single_steane_X_HZ"]


def test_single_steane_basis_z_matrices_byte_identical() -> None:
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    assert _h(g.HX_merged) == GOLDEN["single_steane_Z_HX"]
    assert _h(g.HZ_merged) == GOLDEN["single_steane_Z_HZ"]


def test_single_gb_basis_x_matrices_byte_identical() -> None:
    """Generalised-bicycle code: nontrivial incidence exercises the π-form construction."""
    from qldpc.circuits.surgery.gadget import build_gadget

    data = load_webster_seed_set(0)
    gb = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x = _webster_x_bar_operator(data, "X_bar_1", "X").astype(np.uint8)
    g = build_gadget(gb, x, basis=Pauli.X)
    assert _h(g.HX_merged) == GOLDEN["single_gb_X_HX"]
    assert _h(g.HZ_merged) == GOLDEN["single_gb_X_HZ"]


def test_joint_intercode_basis_x_matrices_byte_identical() -> None:
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_csscode
    from qldpc.circuits.surgery.gadget import build_gadget

    _, x = _steane_x()
    g_l = build_gadget(codes.SteaneCode(), x, basis=Pauli.X)
    g_r = build_gadget(codes.SteaneCode(), x, basis=Pauli.X)
    merged = _stitch_to_joint_csscode(g_l, g_r, build_bridge(g_l, g_r))
    assert _h(merged.matrix_x) == GOLDEN["joint_inter_X_HX"]
    assert _h(merged.matrix_z) == GOLDEN["joint_inter_X_HZ"]


def test_joint_intracode_basis_x_matrices_byte_identical() -> None:
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_csscode
    from qldpc.circuits.surgery.gadget import build_gadget

    code, x = _steane_x()
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(code, x, basis=Pauli.X)
    merged = _stitch_to_joint_csscode(g_l, g_r, build_bridge(g_l, g_r))
    assert _h(merged.matrix_x) == GOLDEN["joint_intra_X_HX"]
    assert _h(merged.matrix_z) == GOLDEN["joint_intra_X_HZ"]


def test_single_circuit_text_byte_identical() -> None:
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

    code, x = _steane_x()
    g = build_gadget(code, x, basis=Pauli.X)
    circuit = build_single_ppm_circuit(g, rounds=2, noise_model=None)
    assert _hs(circuit) == GOLDEN["circ_single_X"]


def test_joint_circuit_text_byte_identical() -> None:
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

    code, x = _steane_x()
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(code, x, basis=Pauli.X)
    circuit, _ = build_joint_ppm_circuit(g_l, g_r, build_bridge(g_l, g_r), rounds=2, noise_model=None)
    assert _hs(circuit) == GOLDEN["circ_joint_intra"]
