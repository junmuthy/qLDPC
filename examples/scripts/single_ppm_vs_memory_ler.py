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
from qldpc.codes.surgery import (
    build_gadget, build_single_ppm_circuit, boost_gadget, cheeger_constant,
    keep_only_observable,
)
from qldpc.circuits.noise_model import DepolarizingNoiseModel
from qldpc.objects import Pauli

import sympy

def main() -> None:
    print("Surgery PPM LER vs Memory LER on BBCode [[72, 12]]")
    print("=" * 65)

    code = codes.SteaneCode()
    xs, ys = sympy.symbols("x y")
    code = codes.BBCode(
          {xs: 6, ys: 6},
          xs**3 + ys + ys**2,
          ys**3 + xs + xs**2)
    # code = codes.BBCode(
    #       {xs: 31, ys: 4},
    #       1 + xs**6 * ys + xs**27,
    #       ys**2 + xs**15 * ys*3 + xs**24)
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    # Check the boundary Cheeger constant; only boost if below Webster's
    # distance-preservation threshold (h ≥ 1).
    h = cheeger_constant(g)
    print(f"  code         : [[{code.num_qudits}, {code.dimension}]]")
    print(f"  Cheeger h(F) : {h:.3f}  (Webster threshold = 1.0)")
    if h < 1.0:
        print(f"  → boosting...")
        g = boost_gadget(
            g, method='combinatorial', target=1.0, max_extra_qubits=20, seed=3,
        )
        print(f"  → boosted F: {g.F.shape}, h(F_aug) = {cheeger_constant(g):.3f}")
    else:
        print(f"  → already ≥ 1.0, skipping boost")
    rounds = 9

    # Cain Fig 1b range: linear p ∈ [0.003, 0.008], 6 points
    p_values = list(np.linspace(0.003, 0.008, 6))
    p_values_rounded = [round(p, 5) for p in p_values]
    print(f"  rounds       : {rounds}")
    print(f"  p values     : {p_values_rounded}")
    print(f"  decoder      : qldpc.decoders.SinterDecoder (BP + LSD, min-sum)")

    surgery_tasks = []
    memory_tasks = []
    for p in p_values:
        noise = DepolarizingNoiseModel(p, include_idling_error=False)

        # Surgery PPM: keep only obs0 = Webster Eq.1
        surg = build_single_ppm_circuit(g, rounds=rounds, noise_model=noise)
        surg_obs0 = keep_only_observable(surg, keep_idx=0)
        surgery_tasks.append(sinter.Task(
            circuit=surg_obs0,
            json_metadata={"p": float(p), "kind": "surgery"},
        ))

        # Memory: data code, X-basis. get_memory_experiment emits one
        # OBSERVABLE_INCLUDE per X-logical (k of them); we keep only X̄_0 to
        # match surgery's `code.get_logical_ops(Pauli.X)[0]`. Without this
        # filter, "any of the k X̄_i flipped" inflates the apparent LER by ~k.
        mem = circuits.get_memory_experiment(
            code, basis=Pauli.X, num_rounds=rounds, noise_model=noise,
        )
        mem_obs0 = keep_only_observable(mem, keep_idx=0)
        memory_tasks.append(sinter.Task(
            circuit=mem_obs0,
            json_metadata={"p": float(p), "kind": "memory"},
        ))

    decoder = decoders.SinterDecoder(
        with_BP_LSD=True, max_iter=20, bp_method="ms", lsd_method="lsd_cs", lsd_order=3,
    )
    # Cain Fig 1b reaches LER ~10⁻⁷ at p=0.003 — needs many shots to resolve.
    # We use 1e5 shots as a compromise (low-p points may still hit 0 errors).
    # Crank max_shots higher (e.g. 1e6 or 1e7) and num_workers up for publication
    # quality.
    max_shots = 2000
    max_errors = 100
    print(f"\n  collecting sinter samples (max_shots={max_shots:,}, max_errors={max_errors})...")
    t0 = time.time()
    results = sinter.collect(
        tasks=surgery_tasks + memory_tasks,
        decoders=["custom"],
        custom_decoders={"custom": decoder},
        num_workers=4,
        max_shots=max_shots,
        max_errors=max_errors,
        print_progress=True,
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
        "o-", label=f"Surgery PPM (boost +{extra} κ)",
        markersize=8, linewidth=2,
    )
    ax.loglog(
        ps_sorted, [memory_lers[p][0] for p in ps_sorted],
        "s-", label=f"Memory X̄ ({rounds} rounds)",
        markersize=8, linewidth=2,
    )
    ax.set_xlabel("physical error rate $p$")
    ax.set_ylabel("logical error rate")
    ax.set_title(
        f"BBCode [[{code.num_qudits}, {code.dimension}]] — "
        f"Surgery PPM vs Memory ({rounds} rounds, BP+LSD)"
    )
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    out_path = "examples/scripts/single_ppm_vs_memory_ler.png"
    plt.savefig(out_path, dpi=120)
    print(f"\n  plot saved → {out_path}")


if __name__ == "__main__":
    main()
