# Surgery `data_init` Kwarg Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `data_init: str | None = None` kwarg to `build_single_ppm_circuit` and `build_joint_ppm_circuit` so callers can specify per-data-qubit initial state at construction time, then delete the three post-process helpers in `examples/lattice_surgery.ipynb`.

**Architecture:** Two atomic commits on `feat/surgery-construction`. Commit 1 modifies `_surgery_state_prep` to consume the kwarg and threads it through the two public entry points, with 5 new tests pinning the contract (default-preservation + each notebook helper's behavior). Commit 2 updates the notebook to use the new kwarg and drops the three helpers; the notebook is edited via a `nbformat`-based Python script.

**Tech Stack:** Python 3, `stim`, `numpy`, `pytest`, `nbformat` (for notebook editing), `nbclient` (for §0-§3 re-execution).

**Spec:** `docs/superpowers/specs/2026-06-10-surgery-data-init-kwarg-design.md`

---

## File Structure

```
src/qldpc/circuits/surgery/
├── circuit.py              (modify: _surgery_state_prep + 2 build_*_ppm_circuit kwarg)
└── _test_circuit.py        (add: 5 new tests)

examples/
└── lattice_surgery.ipynb   (notebook: drop 3 helpers, use data_init kwarg, re-execute §0-§3)
```

No new files. Two existing files modified by Task 1, one notebook modified by Task 2.

---

## Task 1: Commit 1 — add `data_init` kwarg + tests

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py` (`_surgery_state_prep`, `build_single_ppm_circuit`, `build_joint_ppm_circuit`)
- Modify: `src/qldpc/circuits/surgery/_test_circuit.py` (add 5 tests)

The pattern: write each test first, run to confirm it FAILS (test specifies the new behavior), implement, re-run to confirm PASS. The first test is the most important — it pins the default-preservation contract.

- [ ] **Step 1: Read current `_surgery_state_prep` to confirm baseline**

Read `src/qldpc/circuits/surgery/circuit.py` around line 534. Confirm the function signature is exactly:

```python
def _surgery_state_prep(
    gadget: GadgetLayout,
    data_ids: tuple[int, ...],
    kappa_ids: tuple[int, ...],
    bridge_ids: tuple[int, ...] = (),
) -> stim.Circuit:
```

and the body matches:

```python
    circuit = stim.Circuit()
    if gadget.basis is Pauli.X:
        circuit.append("RX", list(data_ids))
        circuit.append("R", list(kappa_ids) + (list(bridge_ids) if bridge_ids else []))
    else:
        circuit.append("R", list(data_ids))
        circuit.append("RX", list(kappa_ids) + (list(bridge_ids) if bridge_ids else []))
    return circuit
```

If the function differs, halt — the rewrite below assumes this exact baseline.

- [ ] **Step 2: Write the default-preservation test**

Append to `src/qldpc/circuits/surgery/_test_circuit.py`:

```python
def test_single_ppm_data_init_default_matches_pre_kwarg():
    """build_single_ppm_circuit(g, rounds=3) ≡ data_init=None ≡ data_init='+' for basis=X."""
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    c_no_kwarg = build_single_ppm_circuit(g, rounds=3, noise_model=None)
    c_none = build_single_ppm_circuit(g, rounds=3, noise_model=None, data_init=None)
    c_plus = build_single_ppm_circuit(g, rounds=3, noise_model=None, data_init="+")
    assert str(c_no_kwarg) == str(c_none), "data_init=None must match no-kwarg call"
    assert str(c_no_kwarg) == str(c_plus), "data_init='+' broadcast must match default for basis=X"
```

- [ ] **Step 3: Run the test — expect FAIL**

Run:
```bash
.venv/bin/python -m pytest src/qldpc/circuits/surgery/_test_circuit.py::test_single_ppm_data_init_default_matches_pre_kwarg -xvs
```

Expected: `FAILED` with `TypeError: build_single_ppm_circuit() got an unexpected keyword argument 'data_init'`.

- [ ] **Step 4: Rewrite `_surgery_state_prep` to accept `data_init`**

Replace the entire `_surgery_state_prep` function in `src/qldpc/circuits/surgery/circuit.py` with this:

```python
def _surgery_state_prep(
    gadget: GadgetLayout,
    data_ids: tuple[int, ...],
    kappa_ids: tuple[int, ...],
    bridge_ids: tuple[int, ...] = (),
    *,
    data_init: str | None = None,
) -> stim.Circuit:
    """Init data/κ/bridge qubits at the start of a surgery PPM circuit.

    Default (``data_init=None``):
      basis=X → data |+⟩ (RX), κ + bridge |0⟩ (R)
      basis=Z → data |0⟩ (R),  κ + bridge |+⟩ (RX)

    Optional ``data_init`` overrides per-data-qubit initial state. Each
    character selects a state for the data qubit at the same position:

      "0" → |0⟩  (R)
      "1" → |1⟩  (R + post-init X)
      "+" → |+⟩  (RX)
      "-" → |-⟩  (RX + post-init Z)

    A length-1 string broadcasts to all data qubits; otherwise length must
    equal ``len(data_ids)``.  κ + bridge init is independent of ``data_init``
    and always follows the protocol default (basis-complement +1 eigenstate).
    """
    if data_init is None:
        default_char = "+" if gadget.basis is Pauli.X else "0"
        per_qubit = default_char * len(data_ids)
    else:
        if len(data_init) == 1:
            data_init = data_init * len(data_ids)
        if len(data_init) != len(data_ids):
            raise ValueError(
                f"data_init length {len(data_init)} does not match num data "
                f"qubits {len(data_ids)}; pass a length-1 string to broadcast"
            )
        invalid = sorted(set(data_init) - set("01+-"))
        if invalid:
            raise ValueError(
                f"data_init must contain only '0', '1', '+', '-'; "
                f"got invalid chars {invalid}"
            )
        per_qubit = data_init

    r_data: list[int] = []
    rx_data: list[int] = []
    x_after: list[int] = []
    z_after: list[int] = []
    for q, c in zip(data_ids, per_qubit):
        if c == "0":
            r_data.append(q)
        elif c == "1":
            r_data.append(q)
            x_after.append(q)
        elif c == "+":
            rx_data.append(q)
        else:  # "-"
            rx_data.append(q)
            z_after.append(q)

    circuit = stim.Circuit()
    if r_data:
        circuit.append("R", r_data)
    if rx_data:
        circuit.append("RX", rx_data)
    if x_after:
        circuit.append("X", x_after)
    if z_after:
        circuit.append("Z", z_after)

    anc_ids = list(kappa_ids) + (list(bridge_ids) if bridge_ids else [])
    if anc_ids:
        anc_init = "R" if gadget.basis is Pauli.X else "RX"
        circuit.append(anc_init, anc_ids)

    return circuit
```

- [ ] **Step 5: Thread the kwarg through `build_single_ppm_circuit`**

In `src/qldpc/circuits/surgery/circuit.py`, locate the `build_single_ppm_circuit` definition. Add the `data_init` kwarg to its signature and forward to `_surgery_state_prep`. Specifically, change the signature from:

```python
def build_single_ppm_circuit(
    gadget: GadgetLayout,
    *,
    rounds: int,
    noise_model=None,
) -> stim.Circuit:
```

to:

```python
def build_single_ppm_circuit(
    gadget: GadgetLayout,
    *,
    rounds: int,
    noise_model=None,
    data_init: str | None = None,
) -> stim.Circuit:
```

And change the call to `_surgery_state_prep` inside its body from:

```python
    circuit += _surgery_state_prep(gadget, data_ids, kappa_ids, bridge_ids)
```

to:

```python
    circuit += _surgery_state_prep(
        gadget, data_ids, kappa_ids, bridge_ids, data_init=data_init,
    )
```

The docstring already mentions obs0/obs1. Append a one-line note above the closing `"""`:

```
    ``data_init`` (optional): per-data-qubit init override; see
    ``_surgery_state_prep`` for the character-to-state mapping.
```

- [ ] **Step 6: Thread the kwarg through `build_joint_ppm_circuit`**

Same pattern: change signature from:

```python
def build_joint_ppm_circuit(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
    *,
    rounds: int,
    noise_model=None,
) -> tuple[stim.Circuit, CSSCode]:
```

to:

```python
def build_joint_ppm_circuit(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
    *,
    rounds: int,
    noise_model=None,
    data_init: str | None = None,
) -> tuple[stim.Circuit, CSSCode]:
```

Change the body's `_surgery_state_prep` call from:

```python
    circuit += _surgery_state_prep(g_l, data_ids, kappa_ids, bridge_ids)
```

to:

```python
    circuit += _surgery_state_prep(
        g_l, data_ids, kappa_ids, bridge_ids, data_init=data_init,
    )
```

Append the same docstring note:

```
    ``data_init`` (optional): per-data-qubit init override. For inter-code,
    positions [0:n_l) are left, [n_l:n_l+n_r) are right; for intra-code,
    length is n_l. See ``_surgery_state_prep`` for the character-to-state mapping.
```

- [ ] **Step 7: Run the default-preservation test — expect PASS**

Run:
```bash
.venv/bin/python -m pytest src/qldpc/circuits/surgery/_test_circuit.py::test_single_ppm_data_init_default_matches_pre_kwarg -xvs
```

Expected: `PASSED`. If `FAILED`, the implementation in Step 4 broke default behavior — diff the `str(c_no_kwarg)` and `str(c_plus)` to find the mismatch and fix Step 4.

- [ ] **Step 8: Add the random-outcome test for `data_init="0"` on basis=X**

Append to `_test_circuit.py`:

```python
def test_single_ppm_data_init_zero_random_outcome():
    """data_init='0' on basis=X gadget → logical |0⟩, obs0 50% flip, obs0 ≡ obs1."""
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    circuit = build_single_ppm_circuit(g, rounds=3, noise_model=None, data_init="0")
    sampler = circuit.compile_detector_sampler()
    _, observables = sampler.sample(shots=4000, separate_observables=True)
    obs0, obs1 = observables[:, 0], observables[:, 1]
    rate0, rate1 = float(obs0.mean()), float(obs1.mean())
    agree = float((obs0 == obs1).mean())
    assert 0.40 < rate0 < 0.60, f"obs0 flip rate {rate0:.2%} not in (40%, 60%)"
    assert 0.40 < rate1 < 0.60, f"obs1 flip rate {rate1:.2%} not in (40%, 60%)"
    assert agree == 1.0, f"obs0 vs obs1 disagree on {int((1-agree)*4000)} of 4000 shots"
```

- [ ] **Step 9: Run the random-outcome test — expect PASS**

Run:
```bash
.venv/bin/python -m pytest src/qldpc/circuits/surgery/_test_circuit.py::test_single_ppm_data_init_zero_random_outcome -xvs
```

Expected: `PASSED`. The rates should report close to ~50%.

- [ ] **Step 10: Add the joint-PPM truth-table test**

Append to `_test_circuit.py`:

```python
def test_joint_ppm_data_init_truth_table():
    """Joint Z̄⊗Z̄ on two Steane copies: 4 |a⟩|b⟩ inits give expected parity."""
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit
    c1, c2 = codes.SteaneCode(), codes.SteaneCode()
    z1 = np.asarray(c1.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    z2 = np.asarray(c2.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g1 = build_gadget(c1, z1, basis=Pauli.Z)
    g2 = build_gadget(c2, z2, basis=Pauli.Z)
    bridge = build_bridge(g1, g2)
    n1 = c1.num_qudits
    cases = [
        ("0" * n1 + "0" * n1, 0),
        ("0" * n1 + "1" * n1, 1),
        ("1" * n1 + "0" * n1, 1),
        ("1" * n1 + "1" * n1, 0),
    ]
    for data_init, expected in cases:
        circuit, _ = build_joint_ppm_circuit(
            g1, g2, bridge, rounds=3, noise_model=None, data_init=data_init,
        )
        sampler = circuit.compile_sampler()
        raw = sampler.sample(shots=200).astype(np.uint8)
        n_meas = raw.shape[1]
        obs_lines = [ln for ln in str(circuit).splitlines() if ln.startswith("OBSERVABLE_INCLUDE")]
        offsets = [int(t.strip("rec[]")) for t in obs_lines[0].split() if t.startswith("rec[")]
        meas_idx = [n_meas + off for off in offsets]
        obs0 = np.bitwise_xor.reduce(raw[:, meas_idx], axis=1)
        rate = float(obs0.mean())
        assert rate == float(expected), (
            f"data_init={data_init!r} gave obs0 rate {rate:.3f}, expected {expected}"
        )
```

- [ ] **Step 11: Run the truth-table test — expect PASS**

Run:
```bash
.venv/bin/python -m pytest src/qldpc/circuits/surgery/_test_circuit.py::test_joint_ppm_data_init_truth_table -xvs
```

Expected: `PASSED`. All 4 cases give deterministic outcomes.

- [ ] **Step 12: Add the joint-PPM superposition test**

Append to `_test_circuit.py`:

```python
def test_joint_ppm_data_init_superposition():
    """c1 |0⟩ × c2 |+⟩: Z̄_2 random → obs0 ~50%, obs0 ≡ obs1 every shot."""
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit
    c1, c2 = codes.SteaneCode(), codes.SteaneCode()
    z1 = np.asarray(c1.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    z2 = np.asarray(c2.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g1 = build_gadget(c1, z1, basis=Pauli.Z)
    g2 = build_gadget(c2, z2, basis=Pauli.Z)
    bridge = build_bridge(g1, g2)
    n = c1.num_qudits
    circuit, _ = build_joint_ppm_circuit(
        g1, g2, bridge, rounds=3, noise_model=None,
        data_init="0" * n + "+" * n,
    )
    sampler = circuit.compile_sampler()
    raw = sampler.sample(shots=1000).astype(np.uint8)
    n_meas = raw.shape[1]
    obs_lines = [ln for ln in str(circuit).splitlines() if ln.startswith("OBSERVABLE_INCLUDE")]
    cols = []
    for line in obs_lines:
        offsets = [int(t.strip("rec[]")) for t in line.split() if t.startswith("rec[")]
        meas_idx = [n_meas + off for off in offsets]
        cols.append(np.bitwise_xor.reduce(raw[:, meas_idx], axis=1))
    obs = np.stack(cols, axis=1)
    rate0 = float(obs[:, 0].mean())
    rate1 = float(obs[:, 1].mean())
    agree = float((obs[:, 0] == obs[:, 1]).mean())
    assert 0.40 < rate0 < 0.60, f"obs0 rate {rate0:.2%} not random"
    assert 0.40 < rate1 < 0.60, f"obs1 rate {rate1:.2%} not random"
    assert agree == 1.0, f"obs0 vs obs1 disagree on {int((1-agree)*1000)} of 1000 shots"
```

- [ ] **Step 13: Run the superposition test — expect PASS**

Run:
```bash
.venv/bin/python -m pytest src/qldpc/circuits/surgery/_test_circuit.py::test_joint_ppm_data_init_superposition -xvs
```

Expected: `PASSED`. obs0 and obs1 both ~50% flip rate, 100% agreement.

- [ ] **Step 14: Add the validation test (parametrized)**

Append to `_test_circuit.py`:

```python
@pytest.mark.parametrize("bad_init,error_substr", [
    ("00", "does not match num data qubits"),    # wrong length (Steane n=7)
    ("@" * 7, "invalid chars"),                   # invalid character
    ("0123456", "invalid chars"),                 # mixed valid + invalid
])
def test_data_init_validation(bad_init, error_substr):
    """Bad data_init raises ValueError with informative message."""
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    with pytest.raises(ValueError, match=error_substr):
        build_single_ppm_circuit(g, rounds=3, noise_model=None, data_init=bad_init)
```

- [ ] **Step 15: Run the validation test — expect PASS**

Run:
```bash
.venv/bin/python -m pytest src/qldpc/circuits/surgery/_test_circuit.py::test_data_init_validation -xvs
```

Expected: `PASSED` (3 parametrized cases).

- [ ] **Step 16: Run the full surgery suite to confirm no regressions**

Run:
```bash
.venv/bin/python -m pytest src/qldpc/circuits/surgery/ -q
```

Expected: `93 passed` (88 existing + 5 new). If any pre-existing test now fails, the implementation in Step 4 broke something — diff and fix.

- [ ] **Step 17: Smoke-test the Cain reproduction script (default path)**

Run:
```bash
.venv/bin/python -m pytest src/qldpc/circuits/surgery/_test_cheeger.py -q
```

Expected: 6 passed (no regressions in the cheeger suite — sanity check that the joint-PPM default path still works since some bridge tests indirectly use it).

- [ ] **Step 18: Commit**

```bash
git add src/qldpc/circuits/surgery/circuit.py src/qldpc/circuits/surgery/_test_circuit.py
git commit -m "$(cat <<'EOF'
feat(surgery): data_init kwarg for build_single/joint_ppm_circuit

Add an optional `data_init: str | None = None` kwarg to the two public
PPM-circuit builders, threaded through _surgery_state_prep. The string
selects per-data-qubit initial state from {"0", "1", "+", "-"}
(mapping to R / R+X / RX / RX+Z gate sequences). A length-1 string
broadcasts; otherwise length must equal num data qubits.

Default (None) reproduces today's protocol-default init exactly
(test_single_ppm_data_init_default_matches_pre_kwarg pins this).

Add 5 tests covering the default-preservation contract, single-PPM
random outcome on |0⟩^n, joint-PPM truth table, joint-PPM superposition,
and validation errors. Surgery suite: 88 → 93.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Commit 2 — notebook update

**Files:**
- Modify: `examples/lattice_surgery.ipynb` (drop 3 helpers, use `data_init` kwarg)

The notebook has no jupytext source — it was deleted in commit `82cb075`. Edit cells programmatically via `nbformat`. The plan first identifies cells by their content, then replaces them, then re-executes §0-§3 to refresh outputs.

- [ ] **Step 1: Inventory the notebook cells**

Run this one-off inspection script:

```bash
.venv/bin/python -c "
import nbformat
nb = nbformat.read('examples/lattice_surgery.ipynb', as_version=4)
for i, c in enumerate(nb.cells):
    head = c.source.split(chr(10))[0][:80]
    print(f'  cell {i:>2} ({c.cell_type:>8}) | {head}')
"
```

Identify the indices of:
- the cell whose first line is `def _swap_data_init_to_zero(...)` (this also contains `verify_single_ppm`)
- the cell whose first line starts `# Run the single-PPM check on 3 codes.`
- the cell whose first line is `def _mutate_init(...)`
- the cell whose first line starts `# Inter-code joint Z̄_1 ⊗ Z̄_2 on two Steane copies.`
- the cell whose first line is `def _switch_init_basis(...)` (this also runs the superposition test)

Record these 5 indices. The edit script in Step 2 references them by name; if needed, adjust the indices in the script to match the inventory.

- [ ] **Step 2: Write the notebook-edit script**

Create `/tmp/update_notebook.py`:

```python
"""Edit examples/lattice_surgery.ipynb to use the new data_init kwarg."""
import nbformat

PATH = "/Users/tgzhou/Project/qLDPC/examples/lattice_surgery.ipynb"
nb = nbformat.read(PATH, as_version=4)


def find_cell(starts_with):
    """Return the index of the first code cell whose source starts with `starts_with`."""
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code" and c.source.lstrip().startswith(starts_with):
            return i
    raise RuntimeError(f"no cell found starting with {starts_with!r}")


# --- Cell §1 helper: kill _swap_data_init_to_zero, simplify verify_single_ppm.
i_swap = find_cell("def _swap_data_init_to_zero")
nb.cells[i_swap].source = (
    'def verify_single_ppm(name: str, code, x_logical: np.ndarray, rounds: int = 3,\n'
    '                      shots: int = 4000) -> dict:\n'
    '    """Build PPM with data init |0⟩^n (random-outcome test), sample, check randomness + consistency."""\n'
    '    g = build_gadget(code, x_logical)\n'
    '    circuit = build_single_ppm_circuit(g, rounds=rounds, noise_model=None, data_init="0")\n'
    '    n_data = code.num_qudits\n'
    '\n'
    '    sampler = circuit.compile_detector_sampler()\n'
    '    _, observables = sampler.sample(shots=shots, separate_observables=True)\n'
    '    obs0, obs1 = observables[:, 0], observables[:, 1]\n'
    '    rate0, rate1 = float(obs0.mean()), float(obs1.mean())\n'
    '    agree = float((obs0 == obs1).mean())\n'
    '\n'
    '    out = {\n'
    '        "name": name, "n_data": n_data,\n'
    '        "kappa": len(g.kappa_qubits), "chi": len(g.V0),\n'
    '        "rate0_eq1": rate0, "rate1_xbar": rate1, "obs_agree": agree,\n'
    '        "passes": 0.40 < rate0 < 0.60 and 0.40 < rate1 < 0.60 and agree == 1.0,\n'
    '    }\n'
    '    return out\n'
)
nb.cells[i_swap].outputs = []
nb.cells[i_swap].execution_count = None


# --- Cell §2 helper: kill _mutate_init entirely (cell becomes a small comment-only stub
#     so its surrounding markdown still flows; we'll re-execute later).
i_mutate = find_cell("def _mutate_init")
nb.cells[i_mutate].source = (
    "# (No init-mutation helpers needed; the joint truth-table cell below\n"
    "# passes data_init strings directly to build_joint_ppm_circuit.)\n"
    "raw_observables_doc = (\n"
    '    "raw_observables(circuit, shots): sample raw measurement bits and "\n'
    '    "reconstruct each OBSERVABLE_INCLUDE column."\n'
    ")\n\n\n"
    "def raw_observables(circuit: stim.Circuit, shots: int) -> np.ndarray:\n"
    "    sampler = circuit.compile_sampler()\n"
    "    raw = sampler.sample(shots=shots).astype(np.uint8)\n"
    "    n_meas = raw.shape[1]\n"
    "    obs_lines = [ln for ln in str(circuit).splitlines() if ln.startswith(\"OBSERVABLE_INCLUDE\")]\n"
    "    cols = []\n"
    "    for line in obs_lines:\n"
    "        offsets = [int(t.strip(\"rec[]\")) for t in line.split() if t.startswith(\"rec[\")]\n"
    "        meas_idx = [n_meas + off for off in offsets]\n"
    "        cols.append(np.bitwise_xor.reduce(raw[:, meas_idx], axis=1))\n"
    "    return np.stack(cols, axis=1)\n"
)
nb.cells[i_mutate].outputs = []
nb.cells[i_mutate].execution_count = None


# --- Cell §2 truth-table: rebuild joint circuit per init via data_init string.
i_truth = find_cell("# Inter-code joint Z")
nb.cells[i_truth].source = (
    "# Inter-code joint Z̄_1 ⊗ Z̄_2 on two Steane copies.\n"
    "c1, c2 = codes.SteaneCode(), codes.SteaneCode()\n"
    "z1 = np.asarray(c1.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)\n"
    "z2 = np.asarray(c2.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)\n"
    "g1 = build_gadget(c1, z1, basis=Pauli.Z)\n"
    "g2 = build_gadget(c2, z2, basis=Pauli.Z)\n"
    "bridge = build_bridge(g1, g2)\n"
    "rounds = 3  # odd → Webster Eq.1 ≡ Z̄_1 ⊗ Z̄_2\n"
    "\n"
    "circuit_default, joint_code = build_joint_ppm_circuit(\n"
    "    g1, g2, bridge, rounds=rounds, noise_model=None,\n"
    ")\n"
    "print(f\"joint code           : [[{joint_code.num_qudits}, {joint_code.dimension}]]\")\n"
    "print(f\"bridge.width         : {bridge.width}\")\n"
    "print(f\"extra κ_l, κ_r       : {bridge.extra_kappa_l.shape[0]}, {bridge.extra_kappa_r.shape[0]}\")\n"
    "print(f\"T_l, H_R shapes      : {bridge.T_l.shape}, {bridge.H_R.shape}\")\n"
    "print()\n"
    "\n"
    "n1 = c1.num_qudits\n"
    "n2 = c2.num_qudits\n"
    "SHOTS_JOINT = 1000\n"
    "print(f\"{'state':>16} | Z̄_1  Z̄_2 | expected obs0 | measured obs0 (frac=1)  | ok\")\n"
    "print(\"-\" * 86)\n"
    "joint_pass = True\n"
    "for label, data_init, expected in [\n"
    "    (\"|0⟩_L|0⟩_L\", \"0\" * n1 + \"0\" * n2, 0),\n"
    "    (\"|0⟩_L|1⟩_L\", \"0\" * n1 + \"1\" * n2, 1),\n"
    "    (\"|1⟩_L|0⟩_L\", \"1\" * n1 + \"0\" * n2, 1),\n"
    "    (\"|1⟩_L|1⟩_L\", \"1\" * n1 + \"1\" * n2, 0),\n"
    "]:\n"
    "    circuit, _ = build_joint_ppm_circuit(\n"
    "        g1, g2, bridge, rounds=rounds, noise_model=None, data_init=data_init,\n"
    "    )\n"
    "    obs = raw_observables(circuit, SHOTS_JOINT)\n"
    "    rate = float(obs[:, 0].mean())\n"
    "    ok = rate == float(expected)\n"
    "    flag = \"✓\" if ok else \"✗\"\n"
    "    sign1 = \"-1\" if \"1\" in data_init[:n1] else \"+1\"\n"
    "    sign2 = \"-1\" if \"1\" in data_init[n1:] else \"+1\"\n"
    "    print(f\"{label:>16} | {sign1:>4}  {sign2:>4} | \"\n"
    "          f\"{expected:>13} | {rate:>10.2%} ({int(obs[:, 0].sum()):>3}/{SHOTS_JOINT})    | {flag}\")\n"
    "    joint_pass = joint_pass and ok\n"
    "\n"
    "assert joint_pass, \"joint-PPM correctness failed\"\n"
    "print(\"\\n✓ joint Z̄_1 ⊗ Z̄_2 observable matches expected parity deterministically\")\n"
    "# Bind c2.num_qudits to n2 (used by superposition cell below); also bind data1/data2 ranges if needed.\n"
    "data1, data2 = list(range(n1)), list(range(n1, n1 + n2))\n"
)
nb.cells[i_truth].outputs = []
nb.cells[i_truth].execution_count = None


# --- Cell §2 superposition: kill _switch_init_basis, rebuild via data_init.
i_super = find_cell("def _switch_init_basis")
nb.cells[i_super].source = (
    "circuit_super, _ = build_joint_ppm_circuit(\n"
    "    g1, g2, bridge, rounds=rounds, noise_model=None,\n"
    "    data_init=\"0\" * n1 + \"+\" * n2,\n"
    ")\n"
    "obs_super = raw_observables(circuit_super, SHOTS_JOINT)\n"
    "rate0_super = float(obs_super[:, 0].mean())\n"
    "rate1_super = float(obs_super[:, 1].mean())\n"
    "agree_super = float((obs_super[:, 0] == obs_super[:, 1]).mean())\n"
    "print(f\"c1 in |0⟩_L, c2 in |+⟩_L (data_init=`{'0'*n1}` + `{'+'*n2}`):\")\n"
    "print(f\"  obs0 (Eq.1)         : {rate0_super:>6.1%} flips  (expected ~50% — Z̄_2 random)\")\n"
    "print(f\"  obs1 (cross-check)  : {rate1_super:>6.1%} flips  (expected ~50%)\")\n"
    "print(f\"  obs0 == obs1        : {agree_super:>6.1%}        (expected 100%)\")\n"
    "assert 0.4 < rate0_super < 0.6 and 0.4 < rate1_super < 0.6 and agree_super == 1.0\n"
    "print(\"\\n✓ joint observable is random (Z̄_2 ⟂ |+⟩) but obs0/obs1 agree on every shot\")\n"
)
nb.cells[i_super].outputs = []
nb.cells[i_super].execution_count = None


nbformat.write(nb, PATH)
print(f"updated {PATH}")
print(f"  cells edited: §1 helper ({i_swap}), §2 helper ({i_mutate}), "
      f"§2 truth-table ({i_truth}), §2 superposition ({i_super})")
```

- [ ] **Step 3: Run the edit script**

Run:
```bash
.venv/bin/python /tmp/update_notebook.py
```

Expected output: `updated /Users/tgzhou/Project/qLDPC/examples/lattice_surgery.ipynb` followed by the cell index summary.

- [ ] **Step 4: Verify no helper names remain in the notebook**

Run:
```bash
grep -c "_swap_data_init_to_zero\|_mutate_init\|_switch_init_basis" examples/lattice_surgery.ipynb
```

Expected: `0`.

- [ ] **Step 5: Re-execute §0-§3 cells**

Use the existing `/tmp/exec_notebook.py` from the earlier session. If it no longer exists, recreate it:

```bash
cat > /tmp/exec_notebook.py << 'EOF'
import nbformat
from nbclient import NotebookClient

PATH = "/Users/tgzhou/Project/qLDPC/examples/lattice_surgery.ipynb"
nb = nbformat.read(PATH, as_version=4)
ler_idx = next(
    i for i, c in enumerate(nb.cells)
    if c.cell_type == "markdown" and "## §4" in c.source
)
print(f"executing cells [0:{ler_idx}], preserving §4 cells [{ler_idx}:{len(nb.cells)}]")
non_ler = nb.cells[:ler_idx]
ler_cells = nb.cells[ler_idx:]
nb.cells = non_ler
client = NotebookClient(
    nb, timeout=600,
    resources={"metadata": {"path": "/Users/tgzhou/Project/qLDPC/examples"}},
)
client.execute()
nb.cells = list(nb.cells) + list(ler_cells)
nbformat.write(nb, PATH)
print("done — §0-§3 executed, §4 cells preserved unexecuted")
EOF
```

Then run:
```bash
.venv/bin/python /tmp/exec_notebook.py
```

Expected: `done — §0-§3 executed, §4 cells preserved unexecuted` after a few seconds. If a cell errors, the script halts at the failing cell — read the traceback and fix the cell source in `/tmp/update_notebook.py`, re-run Step 3 then Step 5.

- [ ] **Step 6: Verify the §0-§3 outputs look sensible**

Run:
```bash
.venv/bin/python -c "
import nbformat
nb = nbformat.read('examples/lattice_surgery.ipynb', as_version=4)
for cell_idx, label in [(6, '§1 single-PPM table'), (9, '§2 truth-table'), (11, '§2 superposition')]:
    if cell_idx < len(nb.cells):
        outputs = nb.cells[cell_idx].get('outputs', [])
        for o in outputs:
            if o.get('output_type') == 'stream':
                text = ''.join(o.get('text', '')).strip()
                last_line = text.splitlines()[-1] if text else ''
                print(f'{label} last line: {last_line[:90]}')
                break
"
```

Note: the cell indices `6`, `9`, `11` are the typical positions for §1 result table, §2 truth-table result, §2 superposition result. If `update_notebook.py` inserted/removed cells, adjust the indices. Each output's last line should begin with `✓`.

- [ ] **Step 7: Commit**

```bash
git add examples/lattice_surgery.ipynb
git commit -m "$(cat <<'EOF'
chore(examples): use data_init kwarg, drop 3 init-mutation helpers

The notebook's three private helpers (_swap_data_init_to_zero,
_switch_init_basis, _mutate_init) post-processed the generated stim
circuit to change the data init state — awkward, since init is a
construction-time concern.

Now that build_single_ppm_circuit and build_joint_ppm_circuit accept
`data_init: str | None = None`, rewrite the §1 single-PPM correctness
test and §2 joint truth-table + superposition tests to pass the
appropriate string directly. Delete the three helpers.

§0-§3 cells re-executed; outputs reflect the new code path. §4 LER
cells remain unexecuted (run on demand).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification

- [ ] **Run the full surgery test suite one more time**

```bash
.venv/bin/python -m pytest src/qldpc/circuits/surgery/ -q
```

Expected: `93 passed` (5 new tests + 88 pre-existing).

- [ ] **Confirm 2-commit branch shape**

```bash
git log --oneline -5
```

Expected: top two commits are exactly `chore(examples): use data_init kwarg…` and `feat(surgery): data_init kwarg…`, in that order.

- [ ] **Push (optional)**

```bash
git push origin feat/surgery-construction
```

---

## Self-review notes

Spec coverage:

- §"API" `_surgery_state_prep` signature → Task 1 Step 4 with full code.
- §"API" char-to-gate mapping → Task 1 Step 4 (the for-loop body).
- §"API" `build_single_ppm_circuit` signature → Task 1 Step 5.
- §"API" `build_joint_ppm_circuit` signature → Task 1 Step 6.
- §"API" validation rules → embedded in `_surgery_state_prep` body in Step 4 + exercised by `test_data_init_validation` in Step 14.
- §"Testing" 5 tests → Task 1 Steps 2, 8, 10, 12, 14.
- §"Notebook update" 3 helper removals + call-site rewrites → Task 2 Step 2.
- §"Compatibility" default-preservation contract → `test_single_ppm_data_init_default_matches_pre_kwarg` in Step 2 + pre-existing 88 tests in Step 16.
- §"Net effect" 88 → 93 surgery test count → verified in Step 16 + Final.

Type / signature consistency:

- `data_init: str | None = None` matches in all 3 functions (`_surgery_state_prep`, `build_single_ppm_circuit`, `build_joint_ppm_circuit`).
- Character set `"01+-"` consistent in the validation error string and the for-loop body.
- All test function names mirror the spec's listing.

Known constraint:

- Step 1 of Task 2 asks the implementer to inventory cell indices because the notebook may have had small shifts since the spec was written (the citation-cleanup commits modified some cell sources). The `find_cell()` helper in `/tmp/update_notebook.py` resolves cells by content prefix so the script is robust to index drift.
