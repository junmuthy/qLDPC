#!/usr/bin/env python3
"""Build and sample generic CSS joint-PPM circuits; CLI defaults to Surface-3/Steane."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import sinter
import stim

from qldpc import codes, decoders
from qldpc.circuits import DepolarizingNoiseModel
from qldpc.circuits.surgery import (
    JointPPMSelectiveBellPairSyndrome,
    build_bridge,
    build_gadget,
    build_preinitialized_joint_ppm_circuit,
    keep_only_observable,
)
from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli

DEFAULT_PHYSICAL_ERROR_RATE = 0.001
DEFAULT_BELL_PAIR_INFIDELITY = 0.0
DEFAULT_ROUNDS = (3, 5)
DEFAULT_PREPARATION_ROUNDS = 3
DEFAULT_BASES = (Pauli.X, Pauli.Z)
NoiseScope = Literal["intrinsic", "operational", "end-to-end"]
DEFAULT_NOISE_SCOPE: NoiseScope = "intrinsic"


@dataclasses.dataclass(frozen=True)
class BuiltExperiment:
    """One generated circuit and the metadata needed to sample and identify it."""

    name: str
    circuit: stim.Circuit
    detector_error_model: stim.DetectorErrorModel
    metadata: dict[str, Any]


def _validate_probability(value: float, name: str) -> float:
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1 inclusive, got {value}")
    return value


def _probability_slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _name_slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip(
        "_"
    )


def _validate_css_input(code: CSSCode, logical_index: int, name: str) -> None:
    if not isinstance(code, CSSCode):
        raise TypeError(f"{name} must be a CSSCode, got {type(code).__name__}")
    if getattr(code.field, "order", None) != 2:
        raise ValueError(f"{name} must be a binary CSSCode")
    if code.is_subsystem_code:
        raise ValueError(f"{name} must be a stabilizer CSSCode, not a subsystem code")
    if not 0 <= logical_index < code.dimension:
        raise IndexError(f"{name} logical index {logical_index} is outside [0, {code.dimension})")


def build_experiment_for_codes(
    *,
    left_code: CSSCode,
    right_code: CSSCode,
    basis: Pauli,
    preparation_rounds: int,
    surgery_rounds: int,
    left_logical_index: int = 0,
    right_logical_index: int = 0,
    left_code_name: str | None = None,
    right_code_name: str | None = None,
    physical_error_rate: float = DEFAULT_PHYSICAL_ERROR_RATE,
    bell_pair_infidelity: float = DEFAULT_BELL_PAIR_INFIDELITY,
    include_idling_error: bool = False,
    noise_scope: NoiseScope = DEFAULT_NOISE_SCOPE,
) -> BuiltExperiment:
    """Build a phased joint-PPM experiment for any two binary stabilizer CSS codes."""
    if basis not in (Pauli.X, Pauli.Z):
        raise ValueError(f"basis must be Pauli.X or Pauli.Z, got {basis!r}")
    if preparation_rounds < 1:
        raise ValueError(f"preparation_rounds must be positive, got {preparation_rounds}")
    if surgery_rounds < 1:
        raise ValueError(f"surgery_rounds must be positive, got {surgery_rounds}")
    if noise_scope not in ("intrinsic", "operational", "end-to-end"):
        raise ValueError(
            f"noise_scope must be 'intrinsic', 'operational', or 'end-to-end', got {noise_scope!r}"
        )
    _validate_css_input(left_code, left_logical_index, "left_code")
    _validate_css_input(right_code, right_logical_index, "right_code")

    physical_error_rate = _validate_probability(physical_error_rate, "physical_error_rate")
    bell_pair_infidelity = _validate_probability(
        bell_pair_infidelity,
        "bell_pair_infidelity",
    )
    left_logical = np.asarray(
        left_code.get_logical_ops(basis)[left_logical_index],
        dtype=np.uint8,
    )
    right_logical = np.asarray(
        right_code.get_logical_ops(basis)[right_logical_index],
        dtype=np.uint8,
    )

    left_gadget = build_gadget(left_code, left_logical, basis=basis)
    right_gadget = build_gadget(right_code, right_logical, basis=basis)
    bridge = build_bridge(left_gadget, right_gadget)
    syndrome_strategy = JointPPMSelectiveBellPairSyndrome(
        left_gadget,
        right_gadget,
        bridge,
        independent_blocks=True,
        bell_pair_fidelity=1 - bell_pair_infidelity,
    )
    noise_model = DepolarizingNoiseModel(
        physical_error_rate,
        include_idling_error=include_idling_error,
    )
    preparation_noise_model = noise_model if noise_scope == "end-to-end" else None
    closure_noise_model = noise_model if noise_scope in ("operational", "end-to-end") else None

    circuit, joint_code = build_preinitialized_joint_ppm_circuit(
        left_gadget,
        right_gadget,
        bridge,
        preparation_rounds=preparation_rounds,
        surgery_rounds=surgery_rounds,
        adapter_noise_model=noise_model,
        preparation_noise_model=preparation_noise_model,
        closure_noise_model=closure_noise_model,
        surgery_syndrome_measurement_strategy=syndrome_strategy,
    )
    circuit = keep_only_observable(circuit, keep_idx=0)
    detector_error_model = circuit.detector_error_model()
    selected_checks = syndrome_strategy.selected_check_indices(joint_code)

    left_label = left_code_name or type(left_code).__name__
    right_label = right_code_name or type(right_code).__name__
    name = (
        f"{_name_slug(left_label)}_{_name_slug(right_label)}"
        f"_joint_{basis.name.lower()}"
        f"_prep{preparation_rounds}_surg{surgery_rounds}_{noise_scope.replace('-', '_')}"
        f"_p{_probability_slug(physical_error_rate)}"
        f"_bell{_probability_slug(bell_pair_infidelity)}"
    )
    metadata = {
        "name": name,
        "left_code": left_label,
        "right_code": right_label,
        "left_code_parameters": list(left_code.get_code_params()),
        "right_code_parameters": list(right_code.get_code_params()),
        "left_logical_index": left_logical_index,
        "right_logical_index": right_logical_index,
        "logical_basis": basis.name,
        "preparation_rounds": preparation_rounds,
        "surgery_rounds": surgery_rounds,
        "rounds": surgery_rounds,
        "noise_scope": noise_scope,
        "physical_error_rate": physical_error_rate,
        "bell_pair_infidelity": bell_pair_infidelity,
        "bell_pair_fidelity": 1 - bell_pair_infidelity,
        "include_idling_error": include_idling_error,
        "decoder": "qldpc_bp_osd",
        "joint_code_parameters": list(joint_code.get_code_params()),
        "num_qubits": circuit.num_qubits,
        "num_detectors": circuit.num_detectors,
        "num_observables": circuit.num_observables,
        "num_dem_errors": detector_error_model.num_errors,
        "num_bell_pair_checks_per_round": len(selected_checks),
    }
    return BuiltExperiment(name, circuit, detector_error_model, metadata)


def build_experiment(
    *,
    basis: Pauli,
    rounds: int,
    physical_error_rate: float = DEFAULT_PHYSICAL_ERROR_RATE,
    bell_pair_infidelity: float = DEFAULT_BELL_PAIR_INFIDELITY,
    include_idling_error: bool = False,
    preparation_rounds: int = DEFAULT_PREPARATION_ROUNDS,
    noise_scope: NoiseScope = DEFAULT_NOISE_SCOPE,
) -> BuiltExperiment:
    """Surface-3/Steane preset for the generic CSS adapter experiment.

    ``rounds`` is retained as the preset's surgery-round count for compatibility.
    """
    surface_code = codes.SurfaceCode(3)
    steane_code = codes.SteaneCode()
    return build_experiment_for_codes(
        left_code=surface_code,
        right_code=steane_code,
        left_code_name="rotated_surface_d3",
        right_code_name="steane_d3",
        basis=basis,
        preparation_rounds=preparation_rounds,
        surgery_rounds=rounds,
        physical_error_rate=physical_error_rate,
        bell_pair_infidelity=bell_pair_infidelity,
        include_idling_error=include_idling_error,
        noise_scope=noise_scope,
    )


def build_experiment_matrix(
    *,
    bases: tuple[Pauli, ...] = DEFAULT_BASES,
    rounds: tuple[int, ...] = DEFAULT_ROUNDS,
    physical_error_rate: float = DEFAULT_PHYSICAL_ERROR_RATE,
    bell_pair_infidelity: float = DEFAULT_BELL_PAIR_INFIDELITY,
    include_idling_error: bool = False,
    preparation_rounds: int = DEFAULT_PREPARATION_ROUNDS,
    noise_scope: NoiseScope = DEFAULT_NOISE_SCOPE,
) -> list[BuiltExperiment]:
    """Build every requested basis/round combination."""
    return [
        build_experiment(
            basis=basis,
            rounds=num_rounds,
            physical_error_rate=physical_error_rate,
            bell_pair_infidelity=bell_pair_infidelity,
            include_idling_error=include_idling_error,
            preparation_rounds=preparation_rounds,
            noise_scope=noise_scope,
        )
        for basis in bases
        for num_rounds in rounds
    ]


def write_artifacts(experiments: list[BuiltExperiment], output_dir: Path) -> None:
    """Write Stim circuits, detector error models, and a shared manifest."""
    circuits_dir = output_dir / "circuits"
    dems_dir = output_dir / "dems"
    circuits_dir.mkdir(parents=True, exist_ok=True)
    dems_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for experiment in experiments:
        circuit_path = circuits_dir / f"{experiment.name}.stim"
        dem_path = dems_dir / f"{experiment.name}.dem"
        circuit_path.write_text(f"{experiment.circuit}\n", encoding="utf-8")
        dem_path.write_text(f"{experiment.detector_error_model}\n", encoding="utf-8")
        manifest.append(
            {
                **experiment.metadata,
                "circuit_path": str(circuit_path.relative_to(output_dir)),
                "dem_path": str(dem_path.relative_to(output_dir)),
            }
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _wilson_interval(num_hits: int, num_trials: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson score interval for a binomial rate."""
    if num_trials <= 0:
        return 0.0, 1.0
    z = 1.959963984540054
    rate = num_hits / num_trials
    denominator = 1 + z**2 / num_trials
    center = (rate + z**2 / (2 * num_trials)) / denominator
    margin = (
        z * math.sqrt(rate * (1 - rate) / num_trials + z**2 / (4 * num_trials**2)) / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def sample_experiments(
    experiments: list[BuiltExperiment],
    *,
    output_dir: Path,
    num_workers: int,
    max_shots: int,
    max_errors: int,
) -> list[sinter.TaskStats]:
    """Collect resumable decoded logical-error statistics with BP+OSD."""
    tasks = [
        sinter.Task(circuit=experiment.circuit, json_metadata=experiment.metadata)
        for experiment in experiments
    ]
    decoder = decoders.SinterDecoder()
    return sinter.collect(
        tasks=tasks,
        decoders=["qldpc_bp_osd"],
        custom_decoders={"qldpc_bp_osd": decoder},
        num_workers=num_workers,
        max_shots=max_shots,
        max_errors=max_errors,
        print_progress=True,
        save_resume_filepath=output_dir / "sinter_progress.csv",
    )


def write_summary(stats: list[sinter.TaskStats], output_dir: Path) -> None:
    """Write one concise result row per Sinter task."""
    fieldnames = [
        "name",
        "logical_basis",
        "preparation_rounds",
        "surgery_rounds",
        "rounds",
        "noise_scope",
        "physical_error_rate",
        "bell_pair_infidelity",
        "shots",
        "discards",
        "errors",
        "logical_error_rate",
        "ler_95ci_low",
        "ler_95ci_high",
        "seconds",
        "decoder",
    ]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted(
            stats,
            key=lambda item: (
                item.json_metadata["logical_basis"],
                item.json_metadata["rounds"],
            ),
        ):
            kept_shots = result.shots - result.discards
            rate = result.errors / kept_shots if kept_shots else float("nan")
            interval_low, interval_high = _wilson_interval(result.errors, kept_shots)
            writer.writerow(
                {
                    "name": result.json_metadata["name"],
                    "logical_basis": result.json_metadata["logical_basis"],
                    "preparation_rounds": result.json_metadata["preparation_rounds"],
                    "surgery_rounds": result.json_metadata["surgery_rounds"],
                    "rounds": result.json_metadata["rounds"],
                    "noise_scope": result.json_metadata["noise_scope"],
                    "physical_error_rate": result.json_metadata["physical_error_rate"],
                    "bell_pair_infidelity": result.json_metadata["bell_pair_infidelity"],
                    "shots": result.shots,
                    "discards": result.discards,
                    "errors": result.errors,
                    "logical_error_rate": rate,
                    "ler_95ci_low": interval_low,
                    "ler_95ci_high": interval_high,
                    "seconds": result.seconds,
                    "decoder": result.decoder,
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    parser.add_argument("--rounds", type=int, nargs="+", default=list(DEFAULT_ROUNDS))
    parser.add_argument(
        "--preparation-rounds",
        type=int,
        default=DEFAULT_PREPARATION_ROUNDS,
    )
    parser.add_argument("--bases", choices=("X", "Z"), nargs="+", default=["X", "Z"])
    parser.add_argument("--physical-error-rate", type=float, default=DEFAULT_PHYSICAL_ERROR_RATE)
    parser.add_argument("--bell-pair-infidelity", type=float, default=DEFAULT_BELL_PAIR_INFIDELITY)
    parser.add_argument("--include-idling-error", action="store_true")
    parser.add_argument(
        "--noise-scope",
        choices=("intrinsic", "operational", "end-to-end"),
        default=DEFAULT_NOISE_SCOPE,
    )
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--max-shots", type=int, default=100_000)
    parser.add_argument("--max-errors", type=int, default=200)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    bases = tuple(Pauli[basis] for basis in args.bases)
    rounds = tuple(args.rounds)
    experiments = build_experiment_matrix(
        bases=bases,
        rounds=rounds,
        physical_error_rate=args.physical_error_rate,
        bell_pair_infidelity=args.bell_pair_infidelity,
        include_idling_error=args.include_idling_error,
        preparation_rounds=args.preparation_rounds,
        noise_scope=args.noise_scope,
    )
    write_artifacts(experiments, args.output_dir)

    for experiment in experiments:
        metadata = experiment.metadata
        print(
            f"built {experiment.name}: "
            f"qubits={metadata['num_qubits']}, detectors={metadata['num_detectors']}, "
            f"dem_errors={metadata['num_dem_errors']}, "
            f"bell_checks/round={metadata['num_bell_pair_checks_per_round']}"
        )

    if args.build_only:
        print(f"artifacts written to {args.output_dir}")
        return

    stats = sample_experiments(
        experiments,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        max_shots=args.max_shots,
        max_errors=args.max_errors,
    )
    write_summary(stats, args.output_dir)
    print(f"results written to {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
