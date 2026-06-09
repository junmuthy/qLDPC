"""Surgery PPM LER vs Memory LER on Webster code 0 [[62, 10, 6]].

Closer to Cain et al.'s bb_18 figure: a real BB code rather than Steane.

For a fair "Idling vs Surgery" comparison (mirroring Cain Fig 1b):
- Memory baseline (idling): bb_like code alone running τ_s SE rounds
- Surgery PPM: merged code (data + κ ancillas) running surgery protocol

Plots both LER curves on a log-log axis.

Run: python examples/scripts/single_ppm_vs_memory_ler_webster.py
"""

from __future__ import annotations

import time
import numpy as np
import sinter
import matplotlib.pyplot as plt
import stim

from qldpc import circuits, decoders
import sys
from pathlib import Path

from qldpc.circuits.surgery import build_gadget, build_single_ppm_circuit, keep_only_observable
from qldpc.circuits.noise_model import DepolarizingNoiseModel
from qldpc.objects import Pauli

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _webster_seed_set import (  # noqa: E402
    build_generalised_bicycle_code,
    load_webster_seed_set,
)


def _x_bar_1_operator(d: dict) -> np.ndarray:
    l = d["l"]
    for seed in d["seeds"]:
        if seed["name"] == "X_bar_1" and seed["pauli_type"] == "X":
            L = np.zeros(l, dtype=np.uint8)
            R = np.zeros(l, dtype=np.uint8)
            for i in seed["L_support"]:
                L[i] = 1
            for i in seed["R_support"]:
                R[i] = 1
            return np.concatenate([L, R])
    raise ValueError("X_bar_1 not found")


def main() -> None:
    print("Surgery PPM LER vs Memory (Idling) LER on Webster code 0 [[62, 10, 6]]")
    print("=" * 70)

    d0 = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(d0["l"], d0["A"], d0["B"])
    x = _x_bar_1_operator(d0)
    g = build_gadget(code, x)
    rounds = 3

    p_values = list(np.logspace(-3, -1.8, 5))
    print(f"  code         : Webster 0 [[{code.num_qudits}, {code.dimension}, 6]]")
    print(f"  merged code  : [[{code.num_qudits + len(g.kappa_qubits)}, ?]]")
    print(f"  rounds       : {rounds}")
    print(f"  p values     : {[round(p, 5) for p in p_values]}")

    surgery_tasks = []
    memory_tasks = []
    for p in p_values:
        noise = DepolarizingNoiseModel(p, include_idling_error=False)

        surg = build_single_ppm_circuit(g, rounds=rounds, noise_model=noise)
        surg_obs0 = keep_only_observable(surg, keep_idx=0)
        surgery_tasks.append(sinter.Task(
            circuit=surg_obs0,
            json_metadata={"p": float(p), "kind": "surgery"},
        ))

        mem = circuits.get_memory_experiment(
            code, basis=Pauli.X, num_rounds=rounds, noise_model=noise,
        )
        memory_tasks.append(sinter.Task(
            circuit=mem,
            json_metadata={"p": float(p), "kind": "idling"},
        ))

    decoder = decoders.SinterDecoder()
    print(f"\n  collecting sinter samples (max_shots=3000, max_errors=40, progress on)...")
    t0 = time.time()
    results = sinter.collect(
        tasks=surgery_tasks + memory_tasks,
        decoders=["custom"],
        custom_decoders={"custom": decoder},
        num_workers=4,
        max_shots=3_000,
        max_errors=40,
        print_progress=True,
    )
    t_collect = time.time() - t0
    print(f"  collected {len(results)} task results in {t_collect:.1f}s")

    surgery_lers = {}
    memory_lers = {}
    for r in results:
        p = r.json_metadata["p"]
        kind = r.json_metadata["kind"]
        ler = r.errors / max(r.shots, 1)
        if kind == "surgery":
            surgery_lers[p] = (ler, r.errors, r.shots)
        else:
            memory_lers[p] = (ler, r.errors, r.shots)

    print(f"\n  {'p':>10} | {'surgery PPM LER':>22} | {'idling LER':>22}")
    print(f"  {'-' * 10} | {'-' * 22} | {'-' * 22}")
    for p in sorted(p_values):
        s_ler, s_err, s_shots = surgery_lers.get(p, (np.nan, 0, 0))
        m_ler, m_err, m_shots = memory_lers.get(p, (np.nan, 0, 0))
        print(
            f"  {p:>10.5f} | {s_ler:>10.5f} ({s_err:>4}/{s_shots:>5})"
            f" | {m_ler:>10.5f} ({m_err:>4}/{m_shots:>5})"
        )

    fig, ax = plt.subplots(figsize=(7, 5))
    ps_sorted = sorted(p_values)
    ax.loglog(
        ps_sorted, [surgery_lers[p][0] for p in ps_sorted],
        "o--", label="Surgery PPM (obs0 = Webster Eq.1)",
        markersize=8, linewidth=2,
    )
    ax.loglog(
        ps_sorted, [memory_lers[p][0] for p in ps_sorted],
        "s-", label=f"Idling (memory, {rounds} rounds)",
        markersize=8, linewidth=2,
    )
    ax.set_xlabel("physical error rate $p$")
    ax.set_ylabel("logical error rate")
    ax.set_title(
        f"Webster 0 [[{code.num_qudits}, {code.dimension}, 6]] — "
        f"Surgery PPM vs Idling ({rounds} rounds, BP+OSD)"
    )
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    out_path = "examples/scripts/single_ppm_vs_memory_ler_webster.png"
    plt.savefig(out_path, dpi=120)
    print(f"\n  plot saved → {out_path}")


if __name__ == "__main__":
    main()
