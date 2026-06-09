"""Surgery PPM LER vs Idling LER on bb_18 [[18, 8, ?]].

Mirrors Cain Fig 1b's comparison axis. Uses the small BB code:
  A = 1 + x + x^2,  B = 1 + y + y^2,  orders (R_x=3, R_y=3)  →  n=18

Idling  = memory experiment on data code (no ancillas).
Surgery = full Cain §III.A PPM protocol on merged code (with κ ancillas).

Run: python examples/scripts/single_ppm_vs_memory_ler_bb18.py
"""

from __future__ import annotations

import time
import numpy as np
import sinter
import matplotlib.pyplot as plt
import stim
import sympy

from qldpc import codes, circuits, decoders
from qldpc.codes.surgery import build_gadget, build_single_ppm_circuit, keep_only_observable
from qldpc.circuits.noise_model import DepolarizingNoiseModel
from qldpc.objects import Pauli


def main() -> None:
    print("Surgery PPM LER vs Idling LER on bb_18 [[18, 8]]")
    print("=" * 65)

    x, y = sympy.symbols("x y")
    code = codes.BBCode({x: 3, y: 3}, 1 + x + x**2, 1 + y + y**2)
    x_op = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x_op)
    rounds = 3

    p_values = [0.005, 0.008, 0.012, 0.018, 0.028]
    print(f"  code         : bb_18 [[{code.num_qudits}, {code.dimension}]]")
    print(f"  merged code  : [[{code.num_qudits + len(g.kappa_qubits)}, ?]]")
    print(f"  rounds       : {rounds}")
    print(f"  p values     : {p_values}")

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
    print(f"\n  collecting sinter samples (max_shots=2000, max_errors=30, 8 workers)...")
    t0 = time.time()
    results = sinter.collect(
        tasks=surgery_tasks + memory_tasks,
        decoders=["custom"],
        custom_decoders={"custom": decoder},
        num_workers=8,
        max_shots=2_000,
        max_errors=30,
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
            f"  {p:>10.4f} | {s_ler:>10.5f} ({s_err:>3}/{s_shots:>5})"
            f" | {m_ler:>10.5f} ({m_err:>3}/{m_shots:>5})"
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
        f"bb_18 [[{code.num_qudits}, {code.dimension}]] — "
        f"Surgery PPM vs Idling ({rounds} rounds, BP+OSD)"
    )
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    out_path = "examples/scripts/single_ppm_vs_memory_ler_bb18.png"
    plt.savefig(out_path, dpi=120)
    print(f"\n  plot saved → {out_path}")


if __name__ == "__main__":
    main()
