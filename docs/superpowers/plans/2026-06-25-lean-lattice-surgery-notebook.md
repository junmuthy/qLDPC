# Lean `lattice_surgery.ipynb` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `examples/lattice_surgery.ipynb` with a lean 3-section demo (Setup · Correctness · LER) and archive the current one.

**Architecture:** The notebook is authored by a committed builder script `examples/scripts/build_lattice_surgery_nb.py` that assembles cells with `nbformat`, then executes them with `jupyter nbconvert --execute --inplace` (no jupytext in the venv). Each task appends its cells to the builder, regenerates + executes the notebook, and verifies it runs clean with in-cell asserts passing. The `.ipynb` is the committed deliverable; the script makes it regenerable.

**Tech Stack:** Python, `nbformat`, `jupyter nbconvert`, `stim`, `sinter`, `qldpc.circuits.surgery`, `matplotlib`. Use the venv interpreter `/.venv/bin/python` and `/.venv/bin/jupyter`.

## Global Constraints

- Correctness uses **Steane [[7,1,3]]**; LER X̄/Z̄⊗Z̄ use **BB `BBCode({x:3, y:6}, x³+y+y², y³+x+x²)` = [[36,8]]**; Ȳ LER uses **Steane** (the |Ȳ₊⟩ prep needs transversal S̄ → doubly-even self-dual CSS; BB does not qualify).
- Truth tables use the `raw_observables` helper (raw `compile_sampler` + manual XOR), **never** `compile_detector_sampler(separate_observables=True)` — the latter hides the logical sign.
- LER tasks use `keep_only_observable(circuit, keep_idx=0)` and decoder `decoders.SinterDecoder(with_BP_LSD=True, max_iter=20, bp_method="ms", lsd_method="lsd_cs", lsd_order=5)`; noise is `DepolarizingNoiseModel(p, include_idling_error=False)`; memory baseline is `circuits.get_memory_experiment(code, basis=..., num_rounds=..., noise_model=noise)`.
- Diagrams use `circuit.diagram("timeslice-svg")` rendered as the **last expression** of a cell; diagram circuits use `rounds=2` (compact picture), truth tables use `rounds=3` (odd → Webster Eq.1 ≡ logical).
- LER-in-notebook is intentional and distinct from the no-LER-in-*tests* rule (`memory/feedback_no_ler_tests.md`).
- Run everything from repo root `/Users/tgzhou/Project/qLDPC`.
- **All subagents (implementers, task reviewers, final review) are dispatched on the `opus` model.**
- **Markdown cells are terse:** a section header plus at most one short clarifying line — no descriptive paragraphs (applies §1 onward; §0 intro is already minimal).

---

## File Structure

- **Create:** `examples/scripts/build_lattice_surgery_nb.py` — builder; a list `CELLS` of `("markdown"|"code", source)` tuples written to the notebook via `nbformat`.
- **Create (generated):** `examples/lattice_surgery.ipynb` — the lean notebook (overwrites the archived original).
- **Move:** `examples/lattice_surgery.ipynb` → `examples/archive/lattice_surgery_full.ipynb` (Task 1, before regeneration).

---

### Task 1: Builder scaffold + §0 Setup + archive original

**Files:**
- Create: `examples/scripts/build_lattice_surgery_nb.py`
- Move: `examples/lattice_surgery.ipynb` → `examples/archive/lattice_surgery_full.ipynb`
- Generate: `examples/lattice_surgery.ipynb`

**Interfaces:**
- Produces: builder module with module-level list `CELLS` (appended by later tasks) and a `build()` that writes `examples/lattice_surgery.ipynb`. Helpers available to all later code cells once executed: `raw_observables(circuit, shots)`, `DepolarizingNoiseModel`, all surgery builders, `codes`, `circuits`, `decoders`, `Pauli`, `np`, `sinter`, `stim`, `sympy`, `plt`, `time`.

- [ ] **Step 1: Archive the current notebook**

```bash
cd /Users/tgzhou/Project/qLDPC
mkdir -p examples/archive
git mv examples/lattice_surgery.ipynb examples/archive/lattice_surgery_full.ipynb
```

- [ ] **Step 2: Write the builder scaffold with §0 cells**

Create `examples/scripts/build_lattice_surgery_nb.py`:

```python
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
```

- [ ] **Step 3: Generate and execute the notebook**

```bash
cd /Users/tgzhou/Project/qLDPC
.venv/bin/python examples/scripts/build_lattice_surgery_nb.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    examples/lattice_surgery.ipynb --ExecutePreprocessor.timeout=1800
```
Expected: `wrote .../lattice_surgery.ipynb (3 cells)` then nbconvert finishes with no traceback.

- [ ] **Step 4: Verify it executed cleanly**

```bash
.venv/bin/python -c "import nbformat; nb=nbformat.read('examples/lattice_surgery.ipynb', as_version=4); errs=[o for c in nb.cells for o in c.get('outputs',[]) if o.get('output_type')=='error']; print('errors:', errs); assert not errs"
```
Expected: `errors: []`.

- [ ] **Step 5: Commit**

```bash
git add examples/scripts/build_lattice_surgery_nb.py examples/lattice_surgery.ipynb examples/archive/lattice_surgery_full.ipynb
git commit -m "feat(examples): scaffold lean lattice_surgery notebook + archive original"
```

---

### Task 2: §1.1 Single-qubit PPM correctness (X̄/Ȳ/Z̄ on Steane)

**Files:** Modify `examples/scripts/build_lattice_surgery_nb.py` (append cells after §0).

**Interfaces:**
- Consumes: `raw_observables`, `build_gadget`, `build_single_ppm_circuit`, `build_y_gadget`, `build_single_y_ppm_circuit`, `logical_state_init` from §0.
- Produces: nothing later tasks depend on (self-contained section).

- [ ] **Step 1: Append §1.1 cells to the builder** (insert immediately before `def build()`)

```python
# ─────────────────── §1 Correctness — Steane [[7,1,3]] ───────────────────
md(r"""
## §1 Correctness — Steane [[7,1,3]]

`obs0` is deterministic when the prepared logical commutes with the measured one, 50/50 when it anticommutes.

### §1.1 Single-qubit PPM — X̄, Ȳ, Z̄
""")

code('''
# X̄ single-PPM gadget circuit on Steane (compact rounds=2 picture).
steane = codes.SteaneCode()
x_steane = np.asarray(steane.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
g_x = build_gadget(steane, x_steane, basis=Pauli.X)
build_single_ppm_circuit(g_x, rounds=2, noise_model=None, data_init="+").diagram("timeslice-svg")
''')

code('''
# X̄ / Z̄ truth tables: scan the 4 CSS-basis logical inits; assert determinism.
def single_css_truth_table(code, basis, label):
    log = np.asarray(code.get_logical_ops(basis)[0]).astype(np.uint8)
    g = build_gadget(code, log, basis=basis)
    commuting = "+" if basis == Pauli.X else "0"      # prep that commutes with measured logical
    anticomm = "0" if basis == Pauli.X else "+"       # prep that anticommutes → 50/50
    print(f"{label}:  measured {basis.name}̄")
    print(f"{'data':>6} | {'expected':>9} | {'obs0 frac=1':>12} | obs0==obs1 | ok")
    ok_all = True
    for state, kind in [(commuting, "det"), (anticomm, "rand")]:
        for sgn in ("+", "-") if kind == "det" else (state,):
            di = logical_state_init(code, sgn if kind == "det" else state, log_idx=0)
            circ = build_single_ppm_circuit(g, rounds=3, noise_model=None, data_init=di)
            obs = raw_observables(circ, 4000)
            rate, agree = float(obs[:, 0].mean()), float((obs[:, 0] == obs[:, 1]).mean())
            if kind == "det":
                ok = rate in (0.0, 1.0) and agree == 1.0
                exp = "0% or 100%"
            else:
                ok = 0.4 < rate < 0.6 and agree == 1.0
                exp = "~50%"
            ok_all &= ok
            print(f"{(sgn if kind=='det' else state):>6} | {exp:>9} | {rate:>10.2%}  | {agree:>7.1%}   | {'✓' if ok else '✗'}")
    assert ok_all, f"{label} truth table failed"
    print()

single_css_truth_table(codes.SteaneCode(), Pauli.X, "X̄ single-PPM")
single_css_truth_table(codes.SteaneCode(), Pauli.Z, "Z̄ single-PPM")
print("✓ X̄ and Z̄ single-PPM measure the logical correctly")
''')

code('''
# Ȳ single-PPM gadget circuit on Steane (Ȳ = iX̄Z̄, non-CSS merged code).
s_yg_diag = build_y_gadget(codes.SteaneCode(),
                           x=np.asarray(codes.SteaneCode().get_logical_ops(Pauli.X)[0]).astype(np.uint8),
                           z=np.asarray(codes.SteaneCode().get_logical_ops(Pauli.Z)[0]).astype(np.uint8))
build_single_y_ppm_circuit(s_yg_diag, rounds=2, data_init="Y+", force_obs0=True).diagram("timeslice-svg")
''')

code('''
# Ȳ truth table: deterministic on the Ȳ-eigenstates |Ȳ±⟩, 50/50 on X̄/Z̄ eigenstates.
s_code = codes.SteaneCode()
s_x = np.asarray(s_code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
s_z = np.asarray(s_code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
s_yg = build_y_gadget(s_code, x=s_x, z=s_z)
print(f"{'data_init':>10} | {'state':>6} | {'P(obs0=1)':>10} | outcome")
ok_all = True
for di, name, kind in [("Y+", "|Ȳ+⟩", "det"), ("Y-", "|Ȳ-⟩", "det"),
                       ("X+", "|+̄⟩", "rand"), ("X-", "|-̄⟩", "rand"),
                       ("Z+", "|0̄⟩", "rand"), ("Z-", "|1̄⟩", "rand")]:
    circ = build_single_y_ppm_circuit(s_yg, rounds=3, data_init=di, force_obs0=True)
    p = float(raw_observables(circ, 4000)[:, 0].mean())
    if kind == "det":
        ok = p < 0.02 or p > 0.98
        outcome = f"deterministic → Ȳ={'+1' if p < 0.5 else '-1'}"
    else:
        ok = 0.4 < p < 0.6
        outcome = "random 50/50 (Ȳ ⟂ prep)"
    ok_all &= ok
    print(f"{di:>10} | {name:>6} | {p:>10.3f} | {outcome}")
assert ok_all, "Ȳ truth table failed"
print("\\n✓ Ȳ deterministic on |Ȳ±⟩, 50/50 on X̄/Z̄ eigenstates")
''')
```

- [ ] **Step 2: Regenerate + execute**

```bash
.venv/bin/python examples/scripts/build_lattice_surgery_nb.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace examples/lattice_surgery.ipynb --ExecutePreprocessor.timeout=1800
```
Expected: `wrote ... (8 cells)`, nbconvert no traceback.

- [ ] **Step 3: Verify no error outputs and asserts passed**

```bash
.venv/bin/python -c "import nbformat; nb=nbformat.read('examples/lattice_surgery.ipynb', as_version=4); errs=[o.get('evalue') for c in nb.cells for o in c.get('outputs',[]) if o.get('output_type')=='error']; print('errors:', errs); assert not errs; txt=''.join(o.get('text','') for c in nb.cells for o in c.get('outputs',[]) if o.get('output_type')=='stream'); assert '✓ X̄ and Z̄' in txt and '✓ Ȳ deterministic' in txt, 'missing pass markers'; print('correctness §1.1 OK')"
```
Expected: `errors: []` then `correctness §1.1 OK`.

- [ ] **Step 4: Verify the two diagram cells produced SVG output**

```bash
.venv/bin/python -c "import nbformat; nb=nbformat.read('examples/lattice_surgery.ipynb', as_version=4); svg=sum(1 for c in nb.cells for o in c.get('outputs',[]) if 'image/svg+xml' in o.get('data',{})); print('svg outputs:', svg); assert svg >= 2"
```
Expected: `svg outputs: 2` (or more).

- [ ] **Step 5: Commit**

```bash
git add examples/scripts/build_lattice_surgery_nb.py examples/lattice_surgery.ipynb
git commit -m "feat(examples): §1.1 single-qubit X̄/Ȳ/Z̄ PPM correctness + diagrams"
```

---

### Task 3: §1.2 Joint Z̄⊗Z̄ correctness (Steane×Steane)

**Files:** Modify `examples/scripts/build_lattice_surgery_nb.py` (append before `def build()`).

**Interfaces:**
- Consumes: `build_gadget`, `build_bridge`, `build_joint_ppm_circuit`, `logical_state_init`, `raw_observables`.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Append §1.2 cells**

```python
md(r"""
### §1.2 Joint PPM — Z̄ ⊗ Z̄ (Steane × Steane, inter-code)
""")

code('''
# Joint Z̄₁⊗Z̄₂ gadget circuit on two Steane blocks (compact rounds=2 picture).
c1, c2 = codes.SteaneCode(), codes.SteaneCode()
z1 = np.asarray(c1.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
z2 = np.asarray(c2.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
g1 = build_gadget(c1, z1, basis=Pauli.Z)
g2 = build_gadget(c2, z2, basis=Pauli.Z)
bridge = build_bridge(g1, g2)
build_joint_ppm_circuit(g1, g2, bridge, rounds=2, noise_model=None)[0].diagram("timeslice-svg")
''')

code('''
# Joint truth table: Z̄₁⊗Z̄₂ parity is deterministic on |s₁⟩_L|s₂⟩_L.
SHOTS_JOINT = 1000
print(f"{'state':>14} | expected | obs0 frac=1 | ok")
joint_pass = True
for s1, s2, expected in [("0", "0", 0), ("0", "1", 1), ("1", "0", 1), ("1", "1", 0)]:
    circuit, _ = build_joint_ppm_circuit(
        g1, g2, bridge, rounds=3, noise_model=None,
        data_init=(logical_state_init(c1, s1, log_idx=0), logical_state_init(c2, s2, log_idx=0)),
    )
    rate = float(raw_observables(circuit, SHOTS_JOINT)[:, 0].mean())
    ok = rate == float(expected)
    joint_pass &= ok
    print(f"{'|'+s1+'⟩|'+s2+'⟩':>14} | {expected:>8} | {rate:>10.2%} | {'✓' if ok else '✗'}")
assert joint_pass, "joint-PPM correctness failed"
print("\\n✓ joint Z̄₁⊗Z̄₂ matches expected parity deterministically")
''')
```

- [ ] **Step 2: Regenerate + execute**

```bash
.venv/bin/python examples/scripts/build_lattice_surgery_nb.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace examples/lattice_surgery.ipynb --ExecutePreprocessor.timeout=1800
```
Expected: `wrote ... (11 cells)`, no traceback.

- [ ] **Step 3: Verify**

```bash
.venv/bin/python -c "import nbformat; nb=nbformat.read('examples/lattice_surgery.ipynb', as_version=4); errs=[o.get('evalue') for c in nb.cells for o in c.get('outputs',[]) if o.get('output_type')=='error']; print('errors:', errs); assert not errs; txt=''.join(o.get('text','') for c in nb.cells for o in c.get('outputs',[]) if o.get('output_type')=='stream'); assert 'joint Z̄₁⊗Z̄₂ matches' in txt; print('correctness §1.2 OK')"
```
Expected: `errors: []` then `correctness §1.2 OK`.

- [ ] **Step 4: Commit**

```bash
git add examples/scripts/build_lattice_surgery_nb.py examples/lattice_surgery.ipynb
git commit -m "feat(examples): §1.2 joint Z̄⊗Z̄ correctness on Steane×Steane"
```

---

### Task 4: §2.1 X̄ single LER on BB [[36,8]]

**Files:** Modify `examples/scripts/build_lattice_surgery_nb.py`.

**Interfaces:**
- Consumes: `build_gadget`, `build_single_ppm_circuit`, `boost_gadget`, `cheeger_constant`, `keep_only_observable`, `run_ler_sweep`, `plot_ler`, `DepolarizingNoiseModel`, `circuits.get_memory_experiment`.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Append §2 header + §2.1 cells**

```python
# ─────────────────── §2 LER — surgery PPM vs memory ───────────────────
md(r"""
## §2 LER — surgery PPM vs memory baseline

### §2.1 X̄ single — BB [[36,8]]
""")

code('''
# BB [[36,8]] X̄ gadget, Cheeger-boosted to h(F) ≥ 1 (Webster threshold).
xs, ys = sympy.symbols("x y")
bbcode = codes.BBCode({xs: 3, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)
x_bbcode = np.asarray(bbcode.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
g_bbcode = build_gadget(bbcode, x_bbcode, basis=Pauli.X)
h = cheeger_constant(g_bbcode)
print(f"code [[{bbcode.num_qudits}, {bbcode.dimension}]]  Cheeger h(F) = {h:.3f}")
if h < 1.0:
    g_bbcode = boost_gadget(g_bbcode, method="combinatorial", target=1.0, max_extra_qubits=20, seed=3)
    print(f"→ boosted to h(F_aug) = {cheeger_constant(g_bbcode):.3f}")
''')

code('''
LER_ROUNDS = 9
LER_P_VALUES = list(np.linspace(0.004, 0.008, 5))

def _surg(p):
    noise = DepolarizingNoiseModel(p, include_idling_error=False)
    return keep_only_observable(build_single_ppm_circuit(g_bbcode, rounds=LER_ROUNDS, noise_model=noise), keep_idx=0)

def _mem(p):
    noise = DepolarizingNoiseModel(p, include_idling_error=False)
    return keep_only_observable(circuits.get_memory_experiment(bbcode, basis=Pauli.X, num_rounds=LER_ROUNDS, noise_model=noise), keep_idx=0)

sweep_x = run_ler_sweep(_surg, _mem, LER_P_VALUES, num_workers=4, max_shots=10000, max_errors=100)
plot_ler(sweep_x, LER_P_VALUES,
         title=f"BB [[{bbcode.num_qudits}, {bbcode.dimension}]] — X̄ surgery PPM vs memory ({LER_ROUNDS} rounds, BP+LSD)",
         surgery_label="Surgery PPM (obs0 = Webster Eq.1)", memory_label=f"Memory X̄ ({LER_ROUNDS} rounds)")
''')
```

- [ ] **Step 2: Regenerate + execute**

```bash
.venv/bin/python examples/scripts/build_lattice_surgery_nb.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace examples/lattice_surgery.ipynb --ExecutePreprocessor.timeout=1800
```
Expected: `wrote ... (14 cells)`, no traceback (the sweep prints `collected 10 task results in ...s`).

- [ ] **Step 3: Verify LER ran and produced a plot**

```bash
.venv/bin/python -c "import nbformat; nb=nbformat.read('examples/lattice_surgery.ipynb', as_version=4); errs=[o.get('evalue') for c in nb.cells for o in c.get('outputs',[]) if o.get('output_type')=='error']; print('errors:', errs); assert not errs; png=sum(1 for c in nb.cells for o in c.get('outputs',[]) if 'image/png' in o.get('data',{})); print('png plots:', png); assert png >= 1"
```
Expected: `errors: []` then `png plots: 1` (or more).

- [ ] **Step 4: Commit**

```bash
git add examples/scripts/build_lattice_surgery_nb.py examples/lattice_surgery.ipynb
git commit -m "feat(examples): §2.1 X̄ single-PPM LER on BB [[36,8]]"
```

---

### Task 5: §2.2 Z̄⊗Z̄ joint LER on BB [[36,8]] (intra-code)

**Files:** Modify `examples/scripts/build_lattice_surgery_nb.py`.

**Interfaces:**
- Consumes: same surgery/LER helpers as Task 4 plus `build_bridge`, `build_joint_ppm_circuit`.
- Produces: nothing later tasks depend on.

**Note (intra- vs inter-code):** the joint builder supports an **intra-code** joint when
`g_l.code is g_r.code` — two logicals of one [[36,8]] block, staying at 36 data qubits. Step 1 smoke-tests
this build. If `build_bridge`/`build_joint_ppm_circuit` raises on the intra-code inputs, fall back to the
proven **inter-code** form: `code_l, code_r = make(), make()` with the *same* logical index on each, and a
single-block memory baseline (the rest of the cell is unchanged).

- [ ] **Step 1: Smoke-test the intra-code joint build**

```bash
.venv/bin/python -c "
import numpy as np, sympy
from qldpc import codes
from qldpc.objects import Pauli
from qldpc.circuits.surgery import build_gadget, build_bridge, build_joint_ppm_circuit, boost_gadget, cheeger_constant
xs, ys = sympy.symbols('x y')
code = codes.BBCode({xs:3, ys:6}, xs**3+ys+ys**2, ys**3+xs+xs**2)
z0 = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
z1 = np.asarray(code.get_logical_ops(Pauli.Z)[1]).astype(np.uint8)
gl = build_gadget(code, z0, basis=Pauli.Z); gr = build_gadget(code, z1, basis=Pauli.Z)
for g in (gl, gr):
    pass
br = build_bridge(gl, gr)
c, jc = build_joint_ppm_circuit(gl, gr, br, rounds=3, noise_model=None)
print('intracode joint OK: merged', f'[[{jc.num_qudits}, {jc.dimension}]]', 'data_qubits stays', code.num_qudits)
"
```
Expected: `intracode joint OK: merged [[...]] data_qubits stays 36`. If it raises, use the inter-code fallback in Step 2.

- [ ] **Step 2: Append §2.2 cells** (intra-code form)

```python
md(r"""
### §2.2 Z̄ ⊗ Z̄ joint — BB [[36,8]] (intra-code)
""")

code('''
# Intra-code joint: two Z-logicals of ONE BB [[36,8]] block (same `code` object → shared data lane).
xs, ys = sympy.symbols("x y")
jcode = codes.BBCode({xs: 3, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)
zj0 = np.asarray(jcode.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
zj1 = np.asarray(jcode.get_logical_ops(Pauli.Z)[1]).astype(np.uint8)
gj_l = build_gadget(jcode, zj0, basis=Pauli.Z)
gj_r = build_gadget(jcode, zj1, basis=Pauli.Z)
for name, g in (("l", gj_l), ("r", gj_r)):
    if cheeger_constant(g) < 1.0:
        boosted = boost_gadget(g, method="combinatorial", target=1.0, max_extra_qubits=20, seed=3)
        if name == "l":
            gj_l = boosted
        else:
            gj_r = boosted
        print(f"side {name}: boosted to h = {cheeger_constant(boosted):.3f}")
bridge_j = build_bridge(gj_l, gj_r)
_, jcode_merged = build_joint_ppm_circuit(gj_l, gj_r, bridge_j, rounds=3, noise_model=None)
print(f"merged joint code: [[{jcode_merged.num_qudits}, {jcode_merged.dimension}]]  (data stays {jcode.num_qudits})")
''')

code('''
LER_ROUNDS_J = 9
LER_P_VALUES_J = list(np.linspace(0.002, 0.006, 3))

def _surg_j(p):
    noise = DepolarizingNoiseModel(p, include_idling_error=False)
    return keep_only_observable(build_joint_ppm_circuit(gj_l, gj_r, bridge_j, rounds=LER_ROUNDS_J, noise_model=noise)[0], keep_idx=0)

def _mem_j(p):
    noise = DepolarizingNoiseModel(p, include_idling_error=False)
    return keep_only_observable(circuits.get_memory_experiment(jcode, basis=Pauli.Z, num_rounds=LER_ROUNDS_J, noise_model=noise), keep_idx=0)

sweep_j = run_ler_sweep(_surg_j, _mem_j, LER_P_VALUES_J, num_workers=3, max_shots=2000, max_errors=200)
plot_ler(sweep_j, LER_P_VALUES_J,
         title=f"BB [[{jcode.num_qudits}, {jcode.dimension}]] — Z̄⊗Z̄ joint PPM vs Z̄ memory ({LER_ROUNDS_J} rounds, BP+LSD)",
         surgery_label="Joint surgery PPM (obs0 = Webster joint)", memory_label=f"Memory Z̄ ({LER_ROUNDS_J} rounds)")
''')
```

- [ ] **Step 3: Regenerate + execute**

```bash
.venv/bin/python examples/scripts/build_lattice_surgery_nb.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace examples/lattice_surgery.ipynb --ExecutePreprocessor.timeout=1800
```
Expected: `wrote ... (17 cells)`, no traceback.

- [ ] **Step 4: Verify**

```bash
.venv/bin/python -c "import nbformat; nb=nbformat.read('examples/lattice_surgery.ipynb', as_version=4); errs=[o.get('evalue') for c in nb.cells for o in c.get('outputs',[]) if o.get('output_type')=='error']; print('errors:', errs); assert not errs; png=sum(1 for c in nb.cells for o in c.get('outputs',[]) if 'image/png' in o.get('data',{})); print('png plots:', png); assert png >= 2"
```
Expected: `errors: []` then `png plots: 2` (or more).

- [ ] **Step 5: Commit**

```bash
git add examples/scripts/build_lattice_surgery_nb.py examples/lattice_surgery.ipynb
git commit -m "feat(examples): §2.2 Z̄⊗Z̄ joint-PPM LER on BB [[36,8]] (intra-code)"
```

---

### Task 6: §2.3 Ȳ single LER on Steane

**Files:** Modify `examples/scripts/build_lattice_surgery_nb.py`.

**Interfaces:**
- Consumes: `build_y_gadget`, `build_single_y_ppm_circuit`, `keep_only_observable`, `run_ler_sweep`, `plot_ler`, `DepolarizingNoiseModel`, `circuits.get_memory_experiment`.
- Produces: nothing.

- [ ] **Step 1: Append §2.3 cells**

```python
md(r"""
### §2.3 Ȳ single — Steane [[7,1,3]]  (|Ȳ₊⟩ = S̄|X̄₊⟩ needs transversal S̄ → self-dual code)
""")

code('''
yler_code = codes.SteaneCode()
yler_x = np.asarray(yler_code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
yler_z = np.asarray(yler_code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
yler_yg = build_y_gadget(yler_code, x=yler_x, z=yler_z)

YLER_ROUNDS = 3
YLER_P_VALUES = list(np.linspace(0.002, 0.008, 4))

def _surg_y(p):
    noise = DepolarizingNoiseModel(p, include_idling_error=False)
    return keep_only_observable(build_single_y_ppm_circuit(yler_yg, rounds=YLER_ROUNDS, data_init="Y+", noise_model=noise), keep_idx=0)

def _mem_y(p):
    noise = DepolarizingNoiseModel(p, include_idling_error=False)
    return keep_only_observable(circuits.get_memory_experiment(yler_code, basis=Pauli.Z, num_rounds=YLER_ROUNDS, noise_model=noise), keep_idx=0)

sweep_y = run_ler_sweep(_surg_y, _mem_y, YLER_P_VALUES, num_workers=4, max_shots=4000, max_errors=120)
plot_ler(sweep_y, YLER_P_VALUES,
         title="Steane [[7,1,3]] — Ȳ measurement vs Z̄ memory (BP+LSD)",
         surgery_label="Ȳ measurement (|Ȳ+⟩ prep)", memory_label=f"Memory Z̄ ({YLER_ROUNDS} rounds)")
''')
```

- [ ] **Step 2: Regenerate + execute**

```bash
.venv/bin/python examples/scripts/build_lattice_surgery_nb.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace examples/lattice_surgery.ipynb --ExecutePreprocessor.timeout=1800
```
Expected: `wrote ... (19 cells)`, no traceback.

- [ ] **Step 3: Verify (final structural check)**

```bash
.venv/bin/python -c "
import nbformat
nb = nbformat.read('examples/lattice_surgery.ipynb', as_version=4)
errs = [o.get('evalue') for c in nb.cells for o in c.get('outputs',[]) if o.get('output_type')=='error']
svg = sum(1 for c in nb.cells for o in c.get('outputs',[]) if 'image/svg+xml' in o.get('data',{}))
png = sum(1 for c in nb.cells for o in c.get('outputs',[]) if 'image/png' in o.get('data',{}))
print('cells', len(nb.cells), '| errors', errs, '| svg', svg, '| png', png)
assert not errs and svg >= 3 and png >= 3, 'final checks failed'
print('§2.3 + full notebook OK')
"
```
Expected: `cells 19 | errors [] | svg 3 | png 3` then `§2.3 + full notebook OK`.

- [ ] **Step 4: Commit**

```bash
git add examples/scripts/build_lattice_surgery_nb.py examples/lattice_surgery.ipynb
git commit -m "feat(examples): §2.3 Ȳ single-PPM LER on Steane"
```

---

### Task 7: Full clean rebuild + acceptance

**Files:** none (verification + final commit only).

- [ ] **Step 1: Clean rebuild end-to-end from scratch**

```bash
cd /Users/tgzhou/Project/qLDPC
.venv/bin/python examples/scripts/build_lattice_surgery_nb.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace examples/lattice_surgery.ipynb --ExecutePreprocessor.timeout=1800
```
Expected: `wrote ... (19 cells)`, nbconvert completes with no traceback.

- [ ] **Step 2: Acceptance check (no errors; 3+ SVG diagrams; 3 LER plots; ~12–19 cells; archive present)**

```bash
.venv/bin/python -c "
import nbformat, pathlib
nb = nbformat.read('examples/lattice_surgery.ipynb', as_version=4)
errs = [o.get('evalue') for c in nb.cells for o in c.get('outputs',[]) if o.get('output_type')=='error']
svg = sum(1 for c in nb.cells for o in c.get('outputs',[]) if 'image/svg+xml' in o.get('data',{}))
png = sum(1 for c in nb.cells for o in c.get('outputs',[]) if 'image/png' in o.get('data',{}))
assert not errs, errs
assert svg >= 3, f'svg={svg}'
assert png == 3, f'png={png}'
assert pathlib.Path('examples/archive/lattice_surgery_full.ipynb').exists()
print(f'ACCEPTED: {len(nb.cells)} cells, {svg} diagrams, {png} LER plots, original archived')
"
```
Expected: `ACCEPTED: 19 cells, 3 diagrams, 3 LER plots, original archived`.

- [ ] **Step 3: Confirm §3/§4.1/superposition material is gone from the new notebook**

```bash
.venv/bin/python -c "
import nbformat
nb = nbformat.read('examples/lattice_surgery.ipynb', as_version=4)
src = '\n'.join(''.join(c['source']) for c in nb.cells)
for banned in ('Webster Table', 'Cain Table', 'bb_18', 'Cross Thm', 'superposition', '248'):
    assert banned not in src, f'leftover: {banned}'
print('clean: no Cain/Webster/Cross/superposition/bb_18 content')
"
```
Expected: `clean: no Cain/Webster/Cross/superposition/bb_18 content`.

- [ ] **Step 4: Final commit (if anything changed in the rebuild)**

```bash
git add examples/lattice_surgery.ipynb
git commit -m "chore(examples): clean rebuild of lean lattice_surgery notebook" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage:**
- §0 Setup + helpers → Task 1. ✓
- §1.1 X̄/Ȳ/Z̄ correctness + diagrams → Task 2. ✓
- §1.2 joint Z̄⊗Z̄ correctness + diagram → Task 3. ✓
- §2.1 X̄ LER on BB [[36,8]] → Task 4. ✓
- §2.2 Z̄⊗Z̄ LER on BB [[36,8]] (intra-code, with inter-code fallback) → Task 5. ✓
- §2.3 Ȳ LER on Steane → Task 6. ✓
- Drop §3/§4.1/superposition/cached-(rep,seed) → enforced by Task 7 Step 3. ✓
- Archive original → Task 1 Step 1, asserted in Task 7. ✓
- raw_observables for truth tables; keep_only_observable + BP+LSD + DepolarizingNoiseModel for LER; timeslice-svg diagrams → Global Constraints, used in every relevant cell. ✓

**Placeholder scan:** every code step contains complete, runnable cell source lifted/adapted from the verified original notebook; no TBD/TODO. ✓

**Type/name consistency:** `run_ler_sweep`/`plot_ler` signatures defined in Task 1 are called with matching keyword args in Tasks 4–6; `build_*` calls match the real signatures (`build_gadget(code, x, *, basis)`, `build_single_ppm_circuit(g, *, rounds, noise_model, data_init)`, `build_joint_ppm_circuit(...) -> (circuit, code)`, `build_y_gadget(code, *, x, z)`, `build_single_y_ppm_circuit(yg, *, rounds, noise_model, data_init, force_obs0)`). ✓

**Open risk:** §2.2 intra-code joint is smoke-tested in Task 5 Step 1 with a documented inter-code fallback, so the task cannot dead-end.
