"""Golden byte-identity regression for the single-gadget builders.

Freezes build_gadget / build_gadget_augmented output (Cain et al.
arXiv:2603.28627 §B.1) across a fixed basket as SHA-256 hashes, so the
closed-form refactor of gadget.py is proven byte-identical to the pre-refactor
implementation. Regenerate _gadget_golden.json only via _regenerate_golden()
against a known-good tree.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from collections.abc import Iterator
from typing import Any

import numpy as np

from qldpc import codes
from qldpc.circuits.surgery._webster_fixture import (
    build_generalised_bicycle_code,
    load_webster_seed_set,
)
from qldpc.circuits.surgery.gadget import GadgetLayout, build_gadget, build_gadget_augmented
from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli, PauliXZ

_GOLDEN = pathlib.Path(__file__).with_name("_gadget_golden.json")
_FIELDS = (
    "support", "data_checks", "incidence", "partial_0",
    "HX_merged", "HZ_merged", "Q_prime",
)


def _canon(value: Any) -> str:
    """Shape-aware canonical SHA-256 of an array/tuple field (int64 bytes)."""
    arr = np.ascontiguousarray(np.asarray(value, dtype=np.int64))
    return hashlib.sha256(arr.tobytes() + repr(arr.shape).encode()).hexdigest()


def _golden_extra(support_len: int) -> np.ndarray:
    """Deterministic weight-2 incidence_extra: rows (0,1) and (1,2) mod L."""
    rows = []
    for j in (0, 1):
        r = np.zeros(support_len, dtype=np.uint8)
        r[j % support_len] = 1
        r[(j + 1) % support_len] = 1
        rows.append(r)
    return np.array(rows, dtype=np.uint8)


def _golden_cases() -> Iterator[tuple[str, CSSCode, np.ndarray, PauliXZ, np.ndarray | None]]:
    """Yield (tag, code, x, basis, incidence_extra | None)."""
    entries: list[tuple[str, CSSCode]] = [("Steane", codes.SteaneCode())]
    for ci in range(4):
        d = load_webster_seed_set(ci)
        entries.append((f"Webster{ci}", build_generalised_bicycle_code(d["l"], d["A"], d["B"])))
    for name, code in entries:
        for basis in (Pauli.X, Pauli.Z):
            x = np.asarray(code.get_logical_ops(basis)[0]).astype(np.uint8)
            yield (f"{name}|{basis.name}|plain", code, x, basis, None)
            extra = _golden_extra(int(np.count_nonzero(x)))
            yield (f"{name}|{basis.name}|aug", code, x, basis, extra)


def _layout_for(
    code: CSSCode, x: np.ndarray, basis: PauliXZ, extra: np.ndarray | None
) -> GadgetLayout:
    if extra is None:
        return build_gadget(code, x, basis=basis)
    return build_gadget_augmented(code, x, extra, basis=basis)


def _hashes() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for tag, code, x, basis, extra in _golden_cases():
        g = _layout_for(code, x, basis, extra)
        out[tag] = {f: _canon(getattr(g, f)) for f in _FIELDS}
    return out


def test_gadget_builders_byte_identical_to_golden() -> None:
    expected = json.loads(_GOLDEN.read_text())
    actual = _hashes()
    assert actual.keys() == expected.keys(), (
        f"case set drift: extra={set(actual) - set(expected)}, "
        f"missing={set(expected) - set(actual)}"
    )
    for tag in expected:
        for field in _FIELDS:
            assert actual[tag][field] == expected[tag][field], (
                f"byte mismatch at {tag}.{field}"
            )


def _regenerate_golden() -> None:  # pragma: no cover - manual, run on good tree
    _GOLDEN.write_text(json.dumps(_hashes(), indent=2, sort_keys=True) + "\n")
