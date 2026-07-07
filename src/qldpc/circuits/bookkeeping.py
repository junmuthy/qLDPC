"""Helper objects to keep track of qubits, measurements, and detectors

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import itertools
from collections.abc import Hashable, ItemsView, Iterable, Iterator, Mapping, Sequence

import numpy as np
import stim
from typing_extensions import Self

from qldpc import codes


@dataclasses.dataclass
class QubitIDs:
    """Container to keep track of the indices of qubits in a circuit."""

    data: tuple[int, ...]  # data qubits in an error-correcting code
    check: tuple[int, ...]  # qubits used to measure parity checks in an error-correcting code
    ancilla: tuple[int, ...]  # miscellaneous ancilla qubits

    # identify X-check and Z-check qubits for CSS codes
    checks_x: tuple[int, ...] = ()
    checks_z: tuple[int, ...] = ()

    def __init__(
        self, data: Sequence[int], check: Sequence[int] = (), ancilla: Sequence[int] = ()
    ) -> None:
        self.data = tuple(data)
        self.check = tuple(check)
        self.ancilla = tuple(ancilla)

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        """Iterate over the collections of qubits tracked by this QubitIDs object."""
        yield from (self.data, self.check, self.ancilla)

    @property
    def all_qubits(self) -> tuple[int, ...]:
        """Serialized tuple of all qubits tracked by this QubitIDs object."""
        return self.data + self.check + self.ancilla

    @staticmethod
    def from_code(code: codes.QuditCode, *, num_ancillas: int = 0, shift: int = 0) -> QubitIDs:
        """Initialize from an error-correcting code with specific parity checks."""
        data = tuple(range(len(code)))
        check = tuple(range(len(code), len(code) + code.num_checks))
        ancilla = tuple(range(check[-1] + 1, check[-1] + 1 + num_ancillas))
        qubit_ids = QubitIDs(data, check, ancilla)
        qubit_ids.checks_x = check[: code.num_checks_x] if isinstance(code, codes.CSSCode) else ()
        qubit_ids.checks_z = check[code.num_checks_x :] if isinstance(code, codes.CSSCode) else ()
        qubit_ids.shift(shift)
        return qubit_ids

    @staticmethod
    def validated(qubit_ids: QubitIDs, code: codes.QuditCode) -> QubitIDs:
        """Validate qubit IDs for the given code and return."""
        if len(qubit_ids.data) != len(code) or len(qubit_ids.check) != code.num_checks:
            raise ValueError("Qubit IDs are invalid for the given code")
        if isinstance(code, codes.CSSCode):
            qubit_ids.checks_x = tuple(qubit_ids.check[: code.num_checks_x])
            qubit_ids.checks_z = tuple(qubit_ids.check[code.num_checks_x :])
        return qubit_ids

    def max(self) -> int:
        """The largest index of any tracked qubit."""
        return max(itertools.chain(*self))

    def shift(self, shift: int) -> QubitIDs:
        """Shift all qubit indices by the given amount and return self."""
        self.data = tuple(qq + shift for qq in self.data)
        self.check = tuple(qq + shift for qq in self.check)
        self.ancilla = tuple(qq + shift for qq in self.ancilla)
        self.checks_x = tuple(qq + shift for qq in self.checks_x)
        self.checks_z = tuple(qq + shift for qq in self.checks_z)
        return self

    def shifted(self, shift: int) -> QubitIDs:
        """New QubitIDs object with shifted qubit indices."""
        qubit_ids = QubitIDs(self.data, self.check, self.ancilla)
        qubit_ids.checks_x = self.checks_x
        qubit_ids.checks_z = self.checks_z
        return qubit_ids.shift(shift)

    def add_ancillas(self, number: int) -> None:
        """Add ancilla qubits."""
        if number > 0:
            start = self.max() + 1
            self.ancilla += tuple(range(start, start + number))


class Record(Mapping[Hashable, list[int]]):
    """An organized record of events in a Stim circuit.

    A record is essentially a dictionary that maps some key (such as a qubit index) to an ordered
    list of the events (such as measurements or detectors) associated with that key.  The events that
    a Record keeps track of are assumed to be indexed from zero.

    Record is subclassed by MeasurementRecord to keep track of measurements in a circuit, and
    by DetectorRecord to keep track of the detectors in a circuit.
    """

    num_events: int
    key_to_events: dict[Hashable, list[int]]

    def __init__(
        self, initial_record: Mapping[Hashable, Iterable[int] | int] | None = None
    ) -> None:
        self.key_to_events = collections.defaultdict(list)
        if initial_record:
            _record = {  # convert initial_record into dict[Hashable, list[int]]
                key: list(events) if isinstance(events, Iterable) else [events]
                for key, events in initial_record.items()
            }
            self.key_to_events |= _record
        self.num_events = sum(len(events) for events in self.key_to_events.values())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({dict(self.key_to_events)})"

    def __str__(self) -> str:
        return repr(self)

    def __len__(self) -> int:
        """The number of keys associated with events in this record."""
        return len(self.key_to_events)

    def __iter__(self) -> Iterator[Hashable]:
        """Iterator over the keys associated with events in this record."""
        yield from self.key_to_events.keys()

    def __getitem__(self, key: Hashable) -> list[int]:
        """The events associated with a key."""
        return self.key_to_events[key]

    def items(self) -> ItemsView[Hashable, list[int]]:
        """Iterator over keys and their associated events."""
        return self.key_to_events.items()

    def get_events(self, *keys: Hashable) -> list[int]:
        """All events associated with the given keys."""
        return [event for key in keys for event in self.key_to_events.get(key, [])]

    def copy(self) -> Self:
        """A copy of this Record."""
        return type(self)(
            {copy.deepcopy(key): copy.deepcopy(events) for key, events in self.items()}
        )

    def append(self, record: Mapping[Hashable, Iterable[int] | int], repeat: int = 1) -> None:
        """Append the given record to this one.

        All event numbers in the appended record are increased by the number of events in the current
        record.  That is, if the current record holds n events numbered from 0 to n - 1, then events
        (0, 1, ...) in the appended record are added to the current record as (n, n+1, ...).
        """
        assert repeat >= 0
        _record = {  # convert input record into dict[Hashable, list[int]]
            key: list(events) if isinstance(events, Iterable) else [events]
            for key, events in record.items()
        }
        num_events_in_record = sum(len(events) for _, events in _record.items())
        for key, events in _record.items():
            self.key_to_events[key].extend(
                [
                    self.num_events + measurement + repetition * num_events_in_record
                    for repetition in range(repeat)
                    for measurement in events
                ]
            )
        self.num_events += num_events_in_record * repeat

    def __iadd__(self, other: Mapping[Hashable, Iterable[int] | int]) -> Self:
        """Append the given record to this one.  See help(qldpc.circuits.Record.append)."""
        self.append(other)
        return self

    def __add__(self, other: Self) -> Self:
        """Combine two records."""
        record = self.copy()
        record.append(other)
        return record


# class MeasurementRecord(Record):
#     """An organized record of measurements in a Stim circuit."""

#     def get_target_rec(self, qubit: Hashable, measurement_index: int = -1) -> stim.target_rec:
#         """Retrieve a Stim measurement record target for the given qubit.

#         Args:
#             qubit: The qubit whose measurement record we want.
#             measurement_index: An index specifying which measurement of the specified qubit we want.
#                 A measurement_index of 0 would be the first measurement of the qubit, while a
#                 measurement_index of -1 would be the most recent measurement.  Default value: -1.

#         Returns:
#             stim.target_rec: A Stim measurement record target.
#         """
#         measurements = self.get_events(qubit)
#         if not -len(measurements) <= measurement_index < len(measurements):
#             raise ValueError(
#                 f"Invalid measurement index {measurement_index} for qubit {qubit} with "
#                 f"{len(measurements)} measurements"
#             )
#         return stim.target_rec(measurements[measurement_index] - self.num_events)
class MeasurementRecord(Record):
    """An organized record of measurements in a circuit."""

    def get_target_rec(self, qubit: Hashable, measurement_index: int = -1) -> stim.target_rec:
        """Retrieve a Stim measurement record target for the given qubit."""
        measurements = self.get_events(qubit)
        if not -len(measurements) <= measurement_index < len(measurements):
            raise ValueError(
                f"Invalid measurement index {measurement_index} for qubit {qubit} with "
                f"{len(measurements)} measurements"
            )
        return stim.target_rec(measurements[measurement_index] - self.num_events)

    def get_target_recs(
        self,
        qubit: Hashable,
        measurement_index: int = -1,
    ) -> list[stim.GateTarget]:
        """Retrieve Stim record targets whose parity defines this measurement.

        For a normal MeasurementRecord, each measurement is just one raw measurement bit.
        This method exists so memory.py can consume either MeasurementRecord or
        ParityMeasurementRecord uniformly.
        """
        return [self.get_target_rec(qubit, measurement_index)]


class ParityMeasurementRecord(Mapping[Hashable, list[tuple[int, ...]]]):
    """Record of parity-valued measurements.

    Each key maps to a list of measurement groups.  Each group is a tuple of raw
    measurement indices, and the parity/XOR of that group is the effective
    measurement value.

    Examples:
        {check_id: (0, 1)}
            means the check syndrome is m0 XOR m1.

        {check_id: [(0, 1), (6, 7)]}
            means the same check was measured twice, first as m0 XOR m1 and then
            as m6 XOR m7.
    """

    num_events: int
    key_to_event_groups: dict[Hashable, list[tuple[int, ...]]]

    def __init__(
        self,
        initial_record: (
            Mapping[Hashable, int | Iterable[int] | Iterable[Iterable[int]]]
            | MeasurementRecord
            | None
        ) = None,
        *,
        num_events: int | None = None,
    ) -> None:
        self.key_to_event_groups = collections.defaultdict(list)
        self.num_events = 0

        if initial_record:
            self.append(initial_record)

        if num_events is not None:
            if num_events < self.num_events:
                raise ValueError(
                    f"Provided num_events={num_events} is smaller than the "
                    f"{self.num_events} events inferred from the record"
                )
            self.num_events = num_events

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({dict(self.key_to_event_groups)})"

    def __str__(self) -> str:
        return repr(self)

    def __len__(self) -> int:
        return len(self.key_to_event_groups)

    def __iter__(self) -> Iterator[Hashable]:
        yield from self.key_to_event_groups.keys()

    def __getitem__(self, key: Hashable) -> list[tuple[int, ...]]:
        return self.key_to_event_groups[key]

    def items(self) -> ItemsView[Hashable, list[tuple[int, ...]]]:
        return self.key_to_event_groups.items()

    def copy(self) -> Self:
        return type(self)(
            {
                copy.deepcopy(key): copy.deepcopy(groups)
                for key, groups in self.items()
            },
            num_events=self.num_events,
        )

    @staticmethod
    def _normalize_event_groups(
        events: int | Iterable[int] | Iterable[Iterable[int]],
    ) -> list[tuple[int, ...]]:
        if isinstance(events, int):
            return [(events,)]

        events_list = list(events)
        if not events_list:
            return []

        # A flat iterable of ints means one parity group.
        # For example, (0, 1) means m0 XOR m1.
        if all(isinstance(event, int) for event in events_list):
            return [tuple(int(event) for event in events_list)]

        # Otherwise interpret as an iterable of parity groups.
        groups: list[tuple[int, ...]] = []
        for group in events_list:
            if isinstance(group, int):
                groups.append((group,))
            else:
                groups.append(tuple(int(event) for event in group))

        return groups

    @staticmethod
    def _get_groups_and_num_events(
        record: (
            Mapping[Hashable, int | Iterable[int] | Iterable[Iterable[int]]]
            | MeasurementRecord
            | ParityMeasurementRecord
        ),
    ) -> tuple[dict[Hashable, list[tuple[int, ...]]], int]:
        if isinstance(record, ParityMeasurementRecord):
            return dict(record.key_to_event_groups), record.num_events

        if isinstance(record, MeasurementRecord):
            groups = {
                key: [(event,) for event in events]
                for key, events in record.items()
            }
            return groups, record.num_events

        groups = {
            key: ParityMeasurementRecord._normalize_event_groups(events)
            for key, events in record.items()
        }

        max_event = -1
        for event_groups in groups.values():
            for group in event_groups:
                if group:
                    max_event = max(max_event, max(group))

        return groups, max_event + 1

    def append(
        self,
        record: (
            Mapping[Hashable, int | Iterable[int] | Iterable[Iterable[int]]]
            | MeasurementRecord
            | ParityMeasurementRecord
        ),
        repeat: int = 1,
    ) -> None:
        """Append a parity record to this one.

        Event indices in the appended record are shifted by the number of raw
        measurements already in this record.
        """
        assert repeat >= 0

        groups, num_events_in_record = self._get_groups_and_num_events(record)

        for key, event_groups in groups.items():
            self.key_to_event_groups[key].extend(
                [
                    tuple(
                        self.num_events + event + repetition * num_events_in_record
                        for event in group
                    )
                    for repetition in range(repeat)
                    for group in event_groups
                ]
            )

        self.num_events += num_events_in_record * repeat

    def __iadd__(
        self,
        other: (
            Mapping[Hashable, int | Iterable[int] | Iterable[Iterable[int]]]
            | MeasurementRecord
            | ParityMeasurementRecord
        ),
    ) -> Self:
        self.append(other)
        return self

    def __add__(self, other: Self) -> Self:
        record = self.copy()
        record.append(other)
        return record

    def get_event_groups(self, *keys: Hashable) -> list[tuple[int, ...]]:
        """All parity groups associated with the given keys."""
        return [
            group
            for key in keys
            for group in self.key_to_event_groups.get(key, [])
        ]

    def get_target_recs(
        self,
        qubit: Hashable,
        measurement_index: int = -1,
    ) -> list[stim.GateTarget]:
        """Retrieve Stim record targets whose parity defines this measurement."""
        groups = self.get_event_groups(qubit)

        if not -len(groups) <= measurement_index < len(groups):
            raise ValueError(
                f"Invalid measurement index {measurement_index} for qubit {qubit} with "
                f"{len(groups)} parity measurements"
            )

        group = groups[measurement_index]
        return [stim.target_rec(event - self.num_events) for event in group]

    def get_target_rec(
        self,
        qubit: Hashable,
        measurement_index: int = -1,
    ) -> stim.GateTarget:
        """Retrieve a single measurement record target.

        This exists for compatibility with code paths that genuinely expect a
        singleton measurement.  It intentionally errors on multi-bit parities.
        """
        targets = self.get_target_recs(qubit, measurement_index)

        if len(targets) != 1:
            raise ValueError(
                f"Measurement for qubit/check {qubit} is a parity of {len(targets)} "
                "raw measurements. Use get_target_recs instead."
            )

        return targets[0]
    

class DetectorRecord(Record):
    """An organized record of detectors in a Stim circuit."""

    def get_detector(self, key: Hashable, detection_index: int = -1) -> int:
        """Retrieve a Stim detector (by index) assoiated with the given key.

        Args:
            key: The name associated with a sequence of detectors in the record.
            detection_index: An index specifying which detector in the specified sequence we want.
                A detection_index of 0 would be the first detector in the sequence, while a
                detection_index of -1 would be the last detector.  Default value: -1.

        Returns:
            int: The index of a detector.
        """
        detectors = self.get_events(key)
        if not -len(detectors) <= detection_index < len(detectors):
            raise ValueError(
                f"Invalid detection index {detection_index} for key '{key}' with {len(detectors)}"
                " detectors"
            )
        return detectors[detection_index]

    def after_post_selection(self, key: Hashable) -> DetectorRecord:
        """A record of the detectors remaining after post-selecting on the detectors of a key.

        If "detector_record" is the record of the detectors in circuit whose detector error model is
        represented by the qldpc.decoders.DetectorErrorModelArrays object "dem_arrays", the record
            new_detector_record = detector_record.after_post_selection(key)
        is the record of the detectors in
            new_dem_arrays = dem_arrays.post_selected_on(detector_record.get_events(key))
        See help(qldpc.decoders.DetectorErrorModelArrays).
        """
        # identify the indices of all detectors, and the detectors to remove
        last_detector = max(max(detectors) for detectors in self.values() if detectors)
        detector_indices = np.arange(last_detector + 1)
        detectors_to_remove = sorted(self.get_events(key))

        # for each detector D, find how many of the detectors_to_remove are <= D
        index_shift = np.searchsorted(detectors_to_remove, detector_indices, side="left")

        # shift detector indices down and remove the post-selection key
        detector_indices -= index_shift
        return DetectorRecord(
            {
                other_key: detector_indices[detectors].tolist()
                for other_key, detectors in self.items()
                if other_key != key
            }
        )
