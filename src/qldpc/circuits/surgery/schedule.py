"""Independent controls for joint-PPM extraction; defaults retain legacy circuits."""

from __future__ import annotations

import dataclasses
import hashlib
from collections import Counter
from typing import Any

import stim

from qldpc.objects import Pauli


@dataclasses.dataclass(frozen=True)
class JointPPMSchedule:
    """Order families independently; optionally replace local interaction layers.

    ``local_layers`` is in canonical (X, Z) order and uses physical qubit IDs.
    It may contain only controlled-Pauli gates and TICKs, never preparation,
    measurements, noise, or detector annotations. The syndrome strategy checks
    its gate inventory against the code before using it. These controls affect
    merged rounds only when supplied as the surgery measurement strategy.
    """

    local_order: tuple[Pauli, Pauli] = (Pauli.X, Pauli.Z)
    bell_order: tuple[Pauli, Pauli] = (Pauli.X, Pauli.Z)
    local_layers: tuple[stim.Circuit, stim.Circuit] | None = None

    def __post_init__(self) -> None:
        for order in (self.local_order, self.bell_order):
            if len(order) != 2 or set(order) != {Pauli.X, Pauli.Z}:
                raise ValueError("Family order must contain X and Z exactly once")
        if self.local_layers is not None:
            if len(self.local_layers) != 2:
                raise ValueError("local_layers must contain canonical X and Z circuits")
            object.__setattr__(self, "local_layers", tuple(c.copy() for c in self.local_layers))

    def metadata(self) -> dict[str, Any]:
        """Serializable, content-addressed schedule identity (no large circuit strings)."""
        return {
            "local_order": [p.name for p in self.local_order],
            "bell_order": [p.name for p in self.bell_order],
            "local_layers_sha256": None
            if self.local_layers is None
            else [hashlib.sha256(str(c).encode()).hexdigest() for c in self.local_layers],
        }


def validate_local_layers(replacement: stim.Circuit, reference: stim.Circuit) -> None:
    """Reject changed gates, noisy operations, or qubit conflicts within a layer."""

    def inventory(circuit: stim.Circuit) -> Counter[tuple[str, int, int]]:
        result: Counter[tuple[str, int, int]] = Counter()
        occupied: set[int] = set()
        for op in circuit:
            if isinstance(op, stim.CircuitRepeatBlock) or op.tag:
                raise ValueError("Local layers must be flat and untagged")
            if op.name == "TICK" and not op.targets_copy() and not op.gate_args_copy():
                occupied.clear()
                continue
            if op.name not in {"CX", "CZ"} or op.gate_args_copy():
                raise ValueError("Local layers may contain only CX/CZ interactions and TICKs")
            targets = op.targets_copy()
            if any(not t.is_qubit_target or t.is_inverted_result_target for t in targets):
                raise ValueError("Local layers require plain qubit targets")
            for a, b in zip(targets[::2], targets[1::2]):
                if a.value == b.value or occupied & {a.value, b.value}:
                    raise ValueError("Local interaction layer has a qubit conflict")
                occupied.update((a.value, b.value))
                result[(op.name, a.value, b.value)] += 1
        if result and (circuit[0].name != "TICK" or circuit[-1].name != "TICK"):
            raise ValueError("Local interaction circuits must start and end with TICK")
        return result

    if inventory(replacement) != inventory(reference):
        raise ValueError("Local replacement changes the check-data gate inventory")
