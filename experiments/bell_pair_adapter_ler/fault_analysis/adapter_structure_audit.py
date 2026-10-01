"""Small exact gadget-connectivity diagnostics, without a merged-code distance search."""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import time
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from qldpc.circuits.surgery.bridge import build_bridge
from qldpc.circuits.surgery.circuit import _stitch_intercode
from qldpc.circuits.surgery.gadget import build_gadget
from qldpc.objects import Pauli, PauliXZ

from ..common.run_experiment import build_experiment_for_codes
from ..pilots.run_ppm_basis_first import code_pair
from .effective_distance_audit import atomic_json, digest, load_source


def connectivity(incidence: np.ndarray[Any, Any]) -> dict[str, Any]:
    """Exact GF(2) boundary expansion on small supports; no spectral surrogate."""
    f = np.asarray(incidence, dtype=np.uint8)
    n = f.shape[1]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for row in f:
        support = np.flatnonzero(row).tolist()
        graph.add_edges_from(itertools.combinations(support, 2))
    result: dict[str, Any] = {
        "shape": list(f.shape),
        "sha256": digest(f.tobytes()),
        "connected_components": [sorted(c) for c in nx.connected_components(graph)],
        "row_weight_counts": dict(collections.Counter(int(w) for w in f.sum(axis=1))),
        "boundary_cheeger_exact": None,
        "expansion_condition_met": None,
    }
    if not 2 <= n <= 20:
        result["status"] = "not_computed_support_outside_2_to_20"
        return result
    cols = [sum(int(f[r, c]) << r for r in range(f.shape[0])) for c in range(n)]
    boundary = mask = best_mask = 0
    best_boundary, best_size = f.shape[0] + 1, 1
    for step in range(1, 1 << n):
        bit = step & -step
        mask ^= bit
        boundary ^= cols[bit.bit_length() - 1]
        size = mask.bit_count()
        if 0 < size <= n // 2 and boundary.bit_count() * best_size < best_boundary * size:
            best_mask, best_boundary, best_size = mask, boundary.bit_count(), size
    cut = [v for v in range(n) if (best_mask >> v) & 1]
    result.update(
        status="exact",
        boundary_cheeger_exact=best_boundary / best_size,
        expansion_condition_met=best_boundary >= best_size,
        minimizing_vertices=cut,
        boundary_rows=np.flatnonzero(f[:, cut].sum(axis=1) % 2).tolist(),
        boundary_size=best_boundary,
        subset_size=best_size,
    )
    return result


def audit(source: Path, output: Path, distance: int, basis: str) -> dict[str, Any]:
    start = time.perf_counter()
    archived, _, frozen = load_source(source / f"d{distance}_r4_{basis}")
    meta = frozen["source_identity"]["metadata"]
    left, right, left_name, right_name = code_pair(
        "hgp-simplex-surface", simplex_m=3 if distance == 4 else 4
    )
    if basis not in ("X", "Z"):
        raise ValueError("Expected X or Z")
    b: PauliXZ = Pauli.X if basis == "X" else Pauli.Z
    experiment = build_experiment_for_codes(
        left_code=left,
        right_code=right,
        left_code_name=left_name,
        right_code_name=right_name,
        basis=b,
        left_logical_index=meta["left_logical_index"],
        right_logical_index=meta["right_logical_index"],
        preparation_rounds=4,
        surgery_rounds=4,
        physical_error_rate=0.001,
        bell_pair_infidelity=0,
        include_idling_error=False,
        noise_scope="intrinsic",
        adapter_noise_targets="adapter_hardware",
        reduce_detectors=True,
        compute_exact_distances=False,
        schedule_policy="ppm_basis_first",
    )
    if experiment.circuit.flattened() != archived.flattened():
        raise ValueError("Reconstructed geometry is not bound to the archived experiment")
    gl = build_gadget(left, left.get_logical_ops(b)[meta["left_logical_index"]], basis=b)
    gr = build_gadget(right, right.get_logical_ops(b)[meta["right_logical_index"]], basis=b)
    bridge = build_bridge(gl, gr)
    joint = _stitch_intercode(gl, gr, bridge)
    matrix_l = left.matrix_x if b is Pauli.X else left.matrix_z
    matrix_r = right.matrix_x if b is Pauli.X else right.matrix_z
    matrix = np.asarray(joint.matrix_x if b is Pauli.X else joint.matrix_z, dtype=np.uint8)
    chi = matrix[matrix_l.shape[0] + matrix_r.shape[0] :, len(left) + len(right) :].T
    report = {
        "distance_label": distance,
        "basis": basis,
        "source_identity": frozen,
        "archived_circuit_reproduced": True,
        "left_support": list(gl.support),
        "right_support": list(gr.support),
        "bridge_width": bridge.width,
        "joint_code_parameters": [len(joint), int(joint.dimension), joint.get_distance_if_known()],
        "left_raw": connectivity(gl.incidence),
        "right_raw": connectivity(gr.incidence),
        "left_augmented": connectivity(bridge.g_l_aug.incidence),
        "right_augmented": connectivity(bridge.g_r_aug.incidence),
        "bridged": connectivity(chi),
        "elapsed_seconds": time.perf_counter() - start,
        "caveat": "Expansion is a structural diagnostic, not a computation of merged-code or PPM circuit distance. h<1 fails a sufficient condition, not proof of distance loss; other code-specific preservation arguments may apply. No boost or distance search was performed.",
    }
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"d{distance}_{basis}.json"
    if target.exists():
        raise ValueError("Do not overwrite an existing structural report")
    atomic_json(target, report)
    print(
        json.dumps({k: report[k] for k in ("distance_label", "basis", "bridge_width", "bridged")}),
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for d, b in itertools.product((4, 8), ("X", "Z")):
        audit(args.source_dir, args.output_dir, d, b)


if __name__ == "__main__":
    main()
