"""Golden regression for the joint-PPM merged check matrices.

Pins CSSCode.matrix_x / matrix_z (shape-aware SHA-256) for a 6-case basket
(intra/inter, X/Z bases, mixed codes) against an inlined baseline. This proves
the closed-form np.block joint assembly (hmatrix/PPM_joint.py) is byte-identical
to the prior `_stitch_*` output. Regenerate `_GOLDEN` by pasting the output of
`_regenerate_golden()` after an INTENTIONAL change.

Joint construction: Swaroop, Jochym-O'Connor, Yoder arXiv:2410.03628 §III.
"""

from __future__ import annotations

import hashlib

import numpy as np
import sympy

from qldpc import codes
from qldpc.circuits.surgery import build_bridge

# Task 3 rewires this import to the closed form:
#   from qldpc.circuits.surgery.hmatrix.PPM_joint import _joint_merged_dispatch as _joint_csscode
from qldpc.circuits.surgery.hmatrix.PPM_joint import _joint_merged_dispatch as _joint_csscode
from qldpc.circuits.surgery.hmatrix.PPM_X_Z import build_gadget
from qldpc.objects import Pauli


def _canon(arr: np.ndarray) -> str:
    """Shape-aware digest: distinguishes same-bytes/different-shape matrices."""
    a = np.asarray(arr).astype(np.int_)
    h = hashlib.sha256()
    h.update(a.tobytes())
    h.update(repr(a.shape).encode())
    return h.hexdigest()


def _cases() -> list[tuple[str, object, object, object]]:
    """(name, g_l, g_r, bridge) for the verified basket."""
    out = []
    steane = codes.SteaneCode()
    xs = np.asarray(steane.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    zs = np.asarray(steane.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    out.append(("intra Steane X", steane, steane, xs, xs, Pauli.X))
    out.append(("intra Steane Z", steane, steane, zs, zs, Pauli.Z))

    xsym, ysym = sympy.symbols("x y")
    bb36 = codes.BBCode({xsym: 3, ysym: 6}, xsym**3 + ysym + ysym**2, ysym**3 + xsym + xsym**2)
    z_ops = bb36.get_logical_ops(Pauli.Z)
    z0 = np.asarray(z_ops[0]).astype(np.uint8)
    z1 = np.asarray(z_ops[1]).astype(np.uint8)
    out.append(("intra BB36 Z", bb36, bb36, z0, z1, Pauli.Z))

    c1, c2 = codes.SteaneCode(), codes.SteaneCode()
    x1 = np.asarray(c1.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    x2 = np.asarray(c2.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    out.append(("inter Steane X", c1, c2, x1, x2, Pauli.X))

    c3, c4 = codes.SteaneCode(), codes.SteaneCode()
    z3 = np.asarray(c3.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    z4 = np.asarray(c4.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    out.append(("inter Steane Z", c3, c4, z3, z4, Pauli.Z))

    sc, st = codes.SurfaceCode(3), codes.SteaneCode()
    xsc = np.asarray(sc.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    xst = np.asarray(st.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    out.append(("inter Surface3xSteane X", sc, st, xsc, xst, Pauli.X))

    built = []
    for name, cl, cr, xl, xr, basis in out:
        g_l = build_gadget(cl, xl, basis=basis)
        g_r = build_gadget(cr, xr, basis=basis)
        built.append((name, g_l, g_r, build_bridge(g_l, g_r)))
    return built


def _hashes() -> dict[str, dict[str, str]]:
    result = {}
    for name, g_l, g_r, bridge in _cases():
        merged = _joint_csscode(g_l, g_r, bridge)
        result[name] = {"matrix_x": _canon(merged.matrix_x), "matrix_z": _canon(merged.matrix_z)}
    return result


_GOLDEN: dict[str, dict[str, str]] = {'inter Steane X': {'matrix_x': '52483653350257be9e8a2c502c372c12837d624d0d0e684bea6da92cfdcbe604',
                    'matrix_z': '5962040d195066388740c9f4fb70851762f5f5b44d21b4b812bfd66f8fbf42be'},
 'inter Steane Z': {'matrix_x': 'e2f5b46b375d8323a8ed872f489b9679bcd0d7ea6ebe1a17b3a443c19eab7712',
                    'matrix_z': '4387ea4d6966ca8666732807436545c13f69ab496f0431d3a168c6ed76ab4770'},
 'inter Surface3xSteane X': {'matrix_x': '22f411647406920fcaba9301b71acd35413d4f6f01de5514cfd3fd68895e0c2f',
                             'matrix_z': '47c449d93e1524346fa39f5fe1e69869541dd944bd6b165a2d067bc71cff584a'},
 'intra BB36 Z': {'matrix_x': '9952e1a9d032965806f7647a10e1675379c0793113d2950c04e444193d203be1',
                  'matrix_z': '9e080edee296e3d7cfe42a84991989e1a43ef87741f8dd7260efe7585098a8a7'},
 'intra Steane X': {'matrix_x': 'a59a11b70b5efcc704e62861ba12a5b15df17cebf17d8f7711d75d475ee6325f',
                    'matrix_z': '2d1398a22945a06d89459ad5b4ad91db85d745c675b6db100a26c5b40050d087'},
 'intra Steane Z': {'matrix_x': 'a118a85a2214b34f266ce40784edda414f6dd8a582691b59423972c104e200ed',
                    'matrix_z': '36b4198d331a01be71edaf15d2ab66e51d96f24e337649fdfa36f881ec798bf1'}}


def test_joint_merged_matrices_match_golden() -> None:
    assert _hashes() == _GOLDEN


def _regenerate_golden() -> None:  # pragma: no cover - maintenance helper
    import pprint

    print("_GOLDEN: dict[str, dict[str, str]] = " + pprint.pformat(_hashes(), sort_dicts=True, width=100))
