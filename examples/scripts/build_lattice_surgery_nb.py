"""Builds examples/lattice_surgery.ipynb (lean demo). Regenerate with:

    .venv/bin/python examples/scripts/build_lattice_surgery_nb.py
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
        examples/lattice_surgery.ipynb --ExecutePreprocessor.timeout=1800
"""
from __future__ import annotations
import pathlib
import nbformat

CELLS: list[tuple[str, str]] = []


def md(src: str) -> None:
    CELLS.append(("markdown", src.strip("\n")))


def code(src: str) -> None:
    CELLS.append(("code", src.strip("\n")))


# ─────────────────────────────── §0 Setup ───────────────────────────────
md(r"""
# Lattice Surgery on qLDPC Codes

End-to-end demo of `qldpc.circuits.surgery`:
- **§1 Correctness** — noiseless determinism + circuit diagrams on Steane [[7,1,3]]:
  single-qubit X̄/Ȳ/Z̄ PPM and the joint Z̄⊗Z̄.
- **§2 LER** — logical-error-rate vs a memory baseline: X̄ and Z̄⊗Z̄ on BB [[36,8]],
  Ȳ on Steane.

## §0 Setup
""")

code('''
from __future__ import annotations

import time
import numpy as np
import sinter
import stim
import sympy
import matplotlib.pyplot as plt

from qldpc import codes, circuits, decoders
from qldpc.circuits.surgery import (
    build_gadget,
    build_bridge,
    build_single_ppm_circuit,
    build_joint_ppm_circuit,
    build_y_gadget,
    build_single_y_ppm_circuit,
    boost_gadget,
    cheeger_constant,
    keep_only_observable,
    logical_state_init,
)
from qldpc.circuits.noise_model import DepolarizingNoiseModel
from qldpc.objects import Pauli
''')

code('''
def raw_observables(circuit: stim.Circuit, shots: int) -> np.ndarray:
    """Raw ±1 observable values (XOR of the bits each OBSERVABLE_INCLUDE designates).

    Uses compile_sampler (NOT compile_detector_sampler), so the result IS the raw
    logical eigenvalue (0 ↔ +1, 1 ↔ −1) and the truth-table sign is visible. The
    detector sampler returns flips vs the noiseless reference (always 0 here), which
    hides the sign — right for LER, wrong for correctness.
    """
    sampler = circuit.compile_sampler()
    raw = sampler.sample(shots=shots).astype(np.uint8)
    n_meas = raw.shape[1]
    obs_lines = [ln for ln in str(circuit).splitlines() if ln.startswith("OBSERVABLE_INCLUDE")]
    cols = []
    for line in obs_lines:
        offsets = [int(t.strip("rec[]")) for t in line.split() if t.startswith("rec[")]
        meas_idx = [n_meas + off for off in offsets]
        cols.append(np.bitwise_xor.reduce(raw[:, meas_idx], axis=1))
    return np.stack(cols, axis=1)


def run_ler_sweep(surgery_factory, memory_factory, p_values, *, num_workers, max_shots, max_errors):
    """Sweep p over a surgery circuit and a memory baseline; return {'surgery':{p:ler}, 'memory':{p:ler}}.

    `*_factory(p)` returns a stim.Circuit already reduced to its keep_idx=0 observable.
    """
    tasks = []
    for p in p_values:
        tasks.append(sinter.Task(circuit=surgery_factory(p),
                                 json_metadata={"p": float(p), "kind": "surgery"}))
        tasks.append(sinter.Task(circuit=memory_factory(p),
                                 json_metadata={"p": float(p), "kind": "memory"}))
    decoder = decoders.SinterDecoder(with_BP_LSD=True, max_iter=20, bp_method="ms",
                                     lsd_method="lsd_cs", lsd_order=5)
    t0 = time.time()
    results = sinter.collect(tasks=tasks, decoders=["custom"], custom_decoders={"custom": decoder},
                             num_workers=num_workers, max_shots=max_shots, max_errors=max_errors,
                             print_progress=False)
    out = {"surgery": {}, "memory": {}}
    for r in results:
        out[r.json_metadata["kind"]][r.json_metadata["p"]] = r.errors / max(r.shots, 1)
    print(f"collected {len(results)} task results in {time.time() - t0:.1f}s")
    return out


def plot_ler(sweep, p_values, *, title, surgery_label, memory_label):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ps = sorted(p_values)
    ax.loglog(ps, [sweep["surgery"][p] for p in ps], "o-", label=surgery_label, markersize=8, linewidth=2)
    ax.loglog(ps, [sweep["memory"][p] for p in ps], "s-", label=memory_label, markersize=8, linewidth=2)
    ax.set_xlabel("physical error rate $p$")
    ax.set_ylabel("logical error rate")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()
''')


def build() -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        nbformat.v4.new_markdown_cell(s) if t == "markdown" else nbformat.v4.new_code_cell(s)
        for t, s in CELLS
    ]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out = pathlib.Path(__file__).resolve().parents[2] / "examples" / "lattice_surgery.ipynb"
    nbformat.write(nb, out)
    print(f"wrote {out} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    build()
