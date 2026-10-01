"""Narrow, reviewed source-hash compatibility for the directory-only reorganization.

Unknown source edits and all non-source identity changes still fail closed. A
receipt does not replace the caller's circuit, DEM, runtime, and catalogue checks.
"""

import json
from pathlib import Path
from typing import Any


def matches_relocated_audit(
    saved: dict[str, Any],
    expected: dict[str, Any],
    directory: Path,
) -> bool:
    """Accept only the pinned old/new single-fault source pair, logging acceptance."""
    if saved == expected:
        return True
    key = "audit_source_sha256"
    compatibility = json.loads(Path(__file__).with_name("source_relocations.json").read_text())
    approved = compatibility["single_fault_audit.py"]
    if (
        saved.get(key) != approved["old_sha256"]
        or expected.get(key) != approved["new_sha256"]
        or {k: v for k, v in saved.items() if k != key}
        != {k: v for k, v in expected.items() if k != key}
    ):
        return False
    receipt = directory / "source_relocation.json"
    content = {"baseline_commit": "30c6a55f", "source": "single_fault_audit.py", **approved}
    if receipt.exists():
        if json.loads(receipt.read_text()) != content:
            raise ValueError("Source-relocation receipt changed")
    else:
        # Never rewrite the original checkpoint's identity or provenance.
        with receipt.open("x") as stream:
            stream.write(json.dumps(content, indent=2, sort_keys=True) + "\n")
    return True
