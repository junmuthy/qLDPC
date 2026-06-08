"""Surgery PPM LER vs Memory LER on the same code.

For a chosen code, runs two parallel sinter sweeps over physical error rates:
  1. Surgery PPM circuit (obs0 = Webster Eq.1 only) — measures X̄_M via syndromes
  2. Standard memory experiment of the data code — preserves logical X̄

Plots both LER curves on a log-log axis for direct comparison. The two
quantities are different physical observables on different circuits, but
both measure "how often a logical operation fails on this code at noise
rate p" and so are meaningful to compare.

Run: python examples/scripts/single_ppm_vs_memory_ler.py
"""

from __future__ import annotations

import time
import numpy as np
import sinter
import matplotlib.pyplot as plt
import stim

from qldpc import codes, circuits, decoders
from qldpc.codes.surgery import build_gadget, build_single_ppm_circuit
from qldpc.circuits.noise_model import DepolarizingNoiseModel
from qldpc.objects import Pauli


def _strip_observable(circuit: stim.Circuit, keep_idx: int) -> stim.Circuit:
    """Return a new circuit keeping only OBSERVABLE_INCLUDE(keep_idx)."""
    out = stim.Circuit()
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            out.append(stim.CircuitRepeatBlock(
                op.repeat_count, _strip_observable(op.body_copy(), keep_idx),
            ))
            continue
        if op.name == "OBSERVABLE_INCLUDE":
            args = list(op.gate_args_copy())
            if int(args[0]) != keep_idx:
                continue
        out.append(op)
    return out


def main() -> None:
    print("Surgery PPM LER vs Memory LER on Steane [[7, 1, 3]]")
    print("=" * 65)

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    rounds = 3

    p_values = list(np.logspace(-3, -1.5, 5))
    p_values_rounded = [round(p, 5) for p in p_values]
    print(f"  rounds       : {rounds}")
    print(f"  p values     : {p_values_rounded}")
    print(f"  decoder      : qldpc.decoders.SinterDecoder (BP + OSD)")

    surgery_tasks = []
    memory_tasks = []
    for p in p_values:
        noise = DepolarizingNoiseModel(p, include_idling_error=False)

        # Surgery PPM: keep only obs0 = Webster Eq.1
        surg = build_single_ppm_circuit(g, rounds=rounds, noise_model=noise)
        surg_obs0 = _strip_observable(surg, keep_idx=0)
        surgery_tasks.append(sinter.Task(
            circuit=surg_obs0,
            json_metadata={"p": float(p), "kind": "surgery"},
        ))

        # Memory: data code, X-basis
        mem = circuits.get_memory_experiment(
            code, basis=Pauli.X, num_rounds=rounds, noise_model=noise,
        )
        memory_tasks.append(sinter.Task(
            circuit=mem,
            json_metadata={"p": float(p), "kind": "memory"},
        ))

    decoder = decoders.SinterDecoder()
    print(f"\n  collecting sinter samples (max_shots=10000, max_errors=100)...")
    t0 = time.time()
    results = sinter.collect(
        tasks=surgery_tasks + memory_tasks,
        decoders=["custom"],
        custom_decoders={"custom": decoder},
        num_workers=4,
        max_shots=10_000,
        max_errors=100,
        print_progress=False,
    )
    t_collect = time.time() - t0
    print(f"  collected {len(results)} task results in {t_collect:.1f}s")

    # Aggregate
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

    # Print summary
    print(f"\n  {'p':>10} | {'surgery LER':>15} | {'memory LER':>15}")
    print(f"  {'-' * 10} | {'-' * 15} | {'-' * 15}")
    for p in sorted(p_values):
        s_ler, s_err, s_shots = surgery_lers.get(p, (np.nan, 0, 0))
        m_ler, m_err, m_shots = memory_lers.get(p, (np.nan, 0, 0))
        print(
            f"  {p:>10.4f} | {s_ler:>9.4f} ({s_err:>3}/{s_shots:<5})"
            f" | {m_ler:>9.4f} ({m_err:>3}/{m_shots:<5})"
        )

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ps_sorted = sorted(p_values)
    ax.loglog(
        ps_sorted, [surgery_lers[p][0] for p in ps_sorted],
        "o-", label="Surgery PPM (obs0 = Webster Eq.1)",
        markersize=8, linewidth=2,
    )
    ax.loglog(
        ps_sorted, [memory_lers[p][0] for p in ps_sorted],
        "s-", label=f"Memory X̄ ({rounds} rounds)",
        markersize=8, linewidth=2,
    )
    ax.set_xlabel("physical error rate $p$")
    ax.set_ylabel("logical error rate")
    ax.set_title(f"Steane [[7, 1, 3]] — Surgery PPM vs Memory ({rounds} rounds, BP+OSD)")
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    out_path = "examples/scripts/single_ppm_vs_memory_ler.png"
    plt.savefig(out_path, dpi=120)
    print(f"\n  plot saved → {out_path}")


if __name__ == "__main__":
    main()
