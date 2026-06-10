# Logical state init helper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `logical_state_init(code, state) → str` to `qldpc.circuits.surgery` so callers can prepare any of the four Pauli logical states (`|0⟩_L`, `|1⟩_L`, `|+⟩_L`, `|-⟩_L`) correctly on any CSS code.

**Architecture:** A pure data-transform helper that lives next to `keep_only_observable` in `surgery/circuit.py`. Reads `code.get_logical_ops(Pauli.X)[0]` / `get_logical_ops(Pauli.Z)[0]` to find the X̄_0 / Z̄_0 support, then builds a length-n per-qubit `data_init` string suitable for `build_*_ppm_circuit`. No changes to `build_*_ppm_circuit` or `_surgery_state_prep`. Notebook §1 and §2 truth-table cells get re-pointed at the helper.

**Tech Stack:** Python 3.12, numpy (already imported), qldpc CSSCode + Pauli (already imported), stim (only for end-to-end tests). Test framework: pytest.

**Spec:** `docs/superpowers/specs/2026-06-10-logical-state-init-design.md`

---

## File Structure

| File | Action | Lines | Responsibility |
|---|---|---|---|
| `src/qldpc/circuits/surgery/circuit.py` | Modify | +~30 | Add `logical_state_init` next to `keep_only_observable` |
| `src/qldpc/circuits/surgery/__init__.py` | Modify | +2 | Add one import + one `__all__` entry |
| `src/qldpc/circuits/surgery/_test_circuit.py` | Modify | +~110 | Add 6 pytest cases |
| `examples/lattice_surgery.ipynb` cell `84449394` | Modify | unchanged size | Use helper for all 4 inits |
| `examples/lattice_surgery.ipynb` cell `b8340c26` | Modify | shorter | Delete inline `prep`, use helper, revert to `0/1/1/0` |

---

## Task 1: Helper function + 4 unit tests

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py:30` (insert after `keep_only_observable`)
- Modify: `src/qldpc/circuits/surgery/_test_circuit.py` (append at end-of-file)

- [ ] **Step 1: Write the 4 unit tests**

Append to `src/qldpc/circuits/surgery/_test_circuit.py`:

```python
def test_logical_state_init_zero_and_plus_broadcast():
    """'0' and '+' return length-n broadcast strings — trivial CSS prep."""
    from qldpc.circuits.surgery.circuit import logical_state_init
    code = codes.SteaneCode()
    n = code.num_qudits
    assert logical_state_init(code, "0") == "0" * n
    assert logical_state_init(code, "+") == "+" * n


def test_logical_state_init_one_flips_x_bar_support():
    """'1' = X̄_0 |0⟩_L: '1' on supp(X̄_0), '0' elsewhere."""
    from qldpc.circuits.surgery.circuit import logical_state_init
    code = codes.SteaneCode()
    n = code.num_qudits
    x_bar = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    s = logical_state_init(code, "1")
    assert len(s) == n
    expected_ones = set(int(i) for i in np.where(x_bar)[0])
    actual_ones = {i for i, c in enumerate(s) if c == "1"}
    actual_zeros = {i for i, c in enumerate(s) if c == "0"}
    assert actual_ones == expected_ones
    assert actual_zeros == set(range(n)) - expected_ones


def test_logical_state_init_minus_flips_z_bar_support():
    """'-' = Z̄_0 |+⟩_L: '-' on supp(Z̄_0), '+' elsewhere."""
    from qldpc.circuits.surgery.circuit import logical_state_init
    code = codes.SteaneCode()
    n = code.num_qudits
    z_bar = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    s = logical_state_init(code, "-")
    assert len(s) == n
    expected_minus = set(int(i) for i in np.where(z_bar)[0])
    actual_minus = {i for i, c in enumerate(s) if c == "-"}
    actual_plus = {i for i, c in enumerate(s) if c == "+"}
    assert actual_minus == expected_minus
    assert actual_plus == set(range(n)) - expected_minus


@pytest.mark.parametrize("bad", ["2", "x", "", "01", "0 ", " 0"])
def test_logical_state_init_invalid_state_raises(bad):
    """Anything outside {'0', '1', '+', '-'} raises ValueError."""
    from qldpc.circuits.surgery.circuit import logical_state_init
    code = codes.SteaneCode()
    with pytest.raises(ValueError, match="state"):
        logical_state_init(code, bad)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/qldpc/circuits/surgery/_test_circuit.py -k logical_state_init -v`
Expected: 8 failures (4 functions × 1; parametrized one runs 5 cases) with `ImportError` on `logical_state_init`.

- [ ] **Step 3: Write minimal implementation**

Insert into `src/qldpc/circuits/surgery/circuit.py` just after the `keep_only_observable` function (around line 56):

```python
def logical_state_init(code: CSSCode, state: str) -> str:
    """Per-qubit ``data_init`` string preparing a Pauli logical state on
    logical qubit 0 of a CSS code.

    ``state`` ∈ {"0", "1", "+", "-"}:
      * "0" → ``"0" * n``  — |0⟩^n projects to |0⟩_L for any CSS code
      * "1" → "1" on supp(X̄_0), "0" elsewhere — X̄_0 |0⟩_L = |1⟩_L
      * "+" → ``"+" * n``  — |+⟩^n projects to |+⟩_L for any CSS code
      * "-" → "-" on supp(Z̄_0), "+" elsewhere — Z̄_0 |+⟩_L = |-⟩_L

    X̄_0 and Z̄_0 are taken from ``code.get_logical_ops(Pauli.X)[0]`` and
    ``code.get_logical_ops(Pauli.Z)[0]``; qldpc guarantees they form an
    anti-commuting symplectic pair on logical qubit 0, so the prep is
    correct for ANY CSS code regardless of the parity of wt(X̄_0) /
    wt(Z̄_0). Naive broadcast ``data_init = "1" * n`` (or ``"-" * n``)
    is correct only when those weights are odd, and silently produces
    the wrong logical state on codes where they are even (e.g. BBCode
    [[36, 8]] with wt(Z̄_0) = 8).

    The returned string has length ``code.num_qudits``. Plug it straight
    into ``build_single_ppm_circuit(..., data_init=...)`` or wrap with a
    tuple for ``build_joint_ppm_circuit(..., data_init=(s_l, s_r))``.

    Raises
    ------
    ValueError
        If ``state`` is not one of "0", "1", "+", "-".
    """
    if state not in ("0", "1", "+", "-"):
        raise ValueError(
            f"state must be one of '0', '1', '+', '-'; got {state!r}"
        )
    n = code.num_qudits
    if state in ("0", "+"):
        return state * n
    if state == "1":
        flip = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
        flip_char, base_char = "1", "0"
    else:  # state == "-"
        flip = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
        flip_char, base_char = "-", "+"
    return "".join(flip_char if flip[i] else base_char for i in range(n))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/qldpc/circuits/surgery/_test_circuit.py -k logical_state_init -v`
Expected: 8 PASSED (4 functions; parametrized one runs 5 cases).

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/circuit.py src/qldpc/circuits/surgery/_test_circuit.py
git commit -m "$(cat <<'EOF'
feat(surgery): logical_state_init helper + unit tests

Pure data-transform helper that builds a per-qubit data_init string
for one of the four Pauli logical states on logical qubit 0 of any
CSS code. Uses get_logical_ops(Pauli.X)[0] / get_logical_ops(Pauli.Z)[0]
to flip exactly the right support, so the result is correct regardless
of wt(X̄_0) / wt(Z̄_0) parity.

4 unit tests: broadcast for "0"/"+", supp flip for "1"/"-", validation
of invalid state strings.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: End-to-end Steane test (legacy "naive-works" sanity)

**Files:**
- Modify: `src/qldpc/circuits/surgery/_test_circuit.py`

- [ ] **Step 1: Write the failing test**

Append to `src/qldpc/circuits/surgery/_test_circuit.py`:

```python
@pytest.mark.parametrize("state,expected_obs0", [("0", 0), ("1", 1)])
def test_logical_state_init_end_to_end_steane_basis_z(state, expected_obs0):
    """Steane single-PPM (basis=Z) reads obs0 = int(state) deterministically.

    Steane has wt(Z̄_0) = 3 (odd), so naive broadcast `"1" * n` ALSO works
    — this test pins the helper to the textbook expectation on the
    historically-working code, catching any regression where the helper
    accidentally diverges from naive on this code.
    """
    from qldpc.circuits.surgery.circuit import (
        build_single_ppm_circuit, logical_state_init,
    )
    from qldpc.circuits.surgery.gadget import build_gadget
    code = codes.SteaneCode()
    z_bar = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z_bar, basis=Pauli.Z)
    circuit = build_single_ppm_circuit(
        g, rounds=3, noise_model=None,
        data_init=logical_state_init(code, state),
    )
    # Raw measurement records — see lattice_surgery.ipynb §0 raw_observables.
    raw = circuit.compile_sampler().sample(shots=200).astype(np.uint8)
    n_meas = raw.shape[1]
    obs0_recs = []
    for ln in str(circuit).splitlines():
        if ln.startswith("OBSERVABLE_INCLUDE(0)"):
            obs0_recs = [
                int(t.strip("rec[]")) for t in ln.split() if t.startswith("rec[")
            ]
            break
    obs0 = np.bitwise_xor.reduce(
        raw[:, [n_meas + off for off in obs0_recs]], axis=1
    )
    rate = float(obs0.mean())
    assert rate == float(expected_obs0), (
        f"state={state!r}: obs0 rate {rate:.3f} != expected {expected_obs0}"
    )
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest src/qldpc/circuits/surgery/_test_circuit.py::test_logical_state_init_end_to_end_steane_basis_z -v`
Expected: 2 PASSED.

- [ ] **Step 3: Commit**

```bash
git add src/qldpc/circuits/surgery/_test_circuit.py
git commit -m "$(cat <<'EOF'
test(surgery): end-to-end logical_state_init on Steane (basis=Z)

Steane has wt(Z̄_0)=3 (odd), so naive broadcast "1"*n ALSO produces
|1⟩_L on this code. This test pins the helper to the textbook obs0
expectation on Steane, regressing any future drift where the helper
disagrees with naive on the historically-working code.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: End-to-end BBCode test (the "naive-breaks" regression)

**Files:**
- Modify: `src/qldpc/circuits/surgery/_test_circuit.py`

- [ ] **Step 1: Write the failing test**

Append to `src/qldpc/circuits/surgery/_test_circuit.py`:

```python
@pytest.mark.parametrize("state,expected_obs0", [("0", 0), ("1", 1)])
def test_logical_state_init_end_to_end_bbcode_basis_z(state, expected_obs0):
    """BBCode [[36, 8]] single-PPM (basis=Z): regression for even-weight Z̄.

    For BBCode (l=3, m=6) the chosen Z̄_0 has weight 8 (even), so naive
    broadcast `"1"*36` produces logical |0⟩_L (NOT |1⟩_L) and obs0=0,
    silently failing any truth table that hardcodes expected=1 for "1".

    The helper uses X̄_0 to flip the correct support, so obs0 tracks the
    textbook expectation. If this test ever returns obs0=0 for state="1",
    the helper has regressed to naive broadcast.
    """
    import sympy
    from qldpc.circuits.surgery.circuit import (
        build_single_ppm_circuit, logical_state_init,
    )
    from qldpc.circuits.surgery.gadget import build_gadget
    xs, ys = sympy.symbols("x y")
    code = codes.BBCode({xs: 3, ys: 6},
                        xs**3 + ys + ys**2, ys**3 + xs + xs**2)
    z_bar = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    assert int(z_bar.sum()) % 2 == 0, (
        "test premise broken: this BBCode should have even-wt Z̄_0"
    )
    g = build_gadget(code, z_bar, basis=Pauli.Z)
    circuit = build_single_ppm_circuit(
        g, rounds=3, noise_model=None,
        data_init=logical_state_init(code, state),
    )
    raw = circuit.compile_sampler().sample(shots=200).astype(np.uint8)
    n_meas = raw.shape[1]
    obs0_recs = []
    for ln in str(circuit).splitlines():
        if ln.startswith("OBSERVABLE_INCLUDE(0)"):
            obs0_recs = [
                int(t.strip("rec[]")) for t in ln.split() if t.startswith("rec[")
            ]
            break
    obs0 = np.bitwise_xor.reduce(
        raw[:, [n_meas + off for off in obs0_recs]], axis=1
    )
    rate = float(obs0.mean())
    assert rate == float(expected_obs0), (
        f"state={state!r}: obs0 rate {rate:.3f} != expected {expected_obs0}. "
        f"This is the BBCode even-wt regression test — failure here means "
        f"logical_state_init is no better than naive '{state}' * n broadcast."
    )
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest src/qldpc/circuits/surgery/_test_circuit.py::test_logical_state_init_end_to_end_bbcode_basis_z -v`
Expected: 2 PASSED. (If state="1" fails with rate=0.0 instead of 1.0, the helper is broken.)

- [ ] **Step 3: Commit**

```bash
git add src/qldpc/circuits/surgery/_test_circuit.py
git commit -m "$(cat <<'EOF'
test(surgery): end-to-end logical_state_init on BBCode even-wt regression

BBCode (l=3, m=6) has wt(Z̄_0) = 8 (even), so naive data_init="1"*36
puts the state in the Z̄_0 = +1 sector = logical |0⟩_L, NOT |1⟩_L.
This test asserts the helper correctly prepares |1⟩_L by flipping
supp(X̄_0); any drift back toward naive broadcast surfaces as obs0=0
on state="1".

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Export from `__init__.py`

**Files:**
- Modify: `src/qldpc/circuits/surgery/__init__.py`

- [ ] **Step 1: Add the import and `__all__` entry**

Edit `src/qldpc/circuits/surgery/__init__.py`. Change:

```python
from .circuit import build_single_ppm_circuit, build_joint_ppm_circuit, keep_only_observable
```

to:

```python
from .circuit import (
    build_single_ppm_circuit, build_joint_ppm_circuit,
    keep_only_observable, logical_state_init,
)
```

And add `"logical_state_init",` to `__all__` (between `"keep_only_observable",` and `"boost_gadget",`):

```python
__all__ = [
    "GadgetLayout",
    "Bridge",
    "build_gadget",
    "build_bridge",
    "build_single_ppm_circuit",
    "build_joint_ppm_circuit",
    "keep_only_observable",
    "logical_state_init",
    "boost_gadget",
    "cheeger_constant",
]
```

- [ ] **Step 2: Verify the export with a smoke test**

Run: `uv run python -c "from qldpc.circuits.surgery import logical_state_init; from qldpc import codes; print(logical_state_init(codes.SteaneCode(), '0'))"`
Expected: prints `0000000` (length-7 zero string).

- [ ] **Step 3: Run the full surgery test suite**

Run: `uv run pytest src/qldpc/circuits/surgery/ -q`
Expected: all tests PASS (12 new + existing).

- [ ] **Step 4: Commit**

```bash
git add src/qldpc/circuits/surgery/__init__.py
git commit -m "$(cat <<'EOF'
feat(surgery): export logical_state_init from package __init__

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Notebook §1 (single PPM truth table) — use helper

**Files:**
- Modify: `examples/lattice_surgery.ipynb` cell id `84449394`

- [ ] **Step 1: Inspect the current cell**

Run: `python -c "import json; nb = json.load(open('examples/lattice_surgery.ipynb')); print(next(''.join(c['source']) for c in nb['cells'] if c.get('id') == '84449394'))"`

Expected: prints the current §1 Steane truth-table code, which currently has hardcoded `"0" * n` etc. in the `table = [...]` literal.

- [ ] **Step 2: Replace the cell source**

Use NotebookEdit to replace the entire `84449394` cell source with:

```python
steane = codes.SteaneCode()
x_steane = np.asarray(steane.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
wt = int(x_steane.sum())  # weight of X̄ support (for the |−⟩_L commentary below)
g_steane = build_gadget(steane, x_steane, basis=Pauli.X)

SHOTS = 4000
# logical_state_init(code, state) returns the right per-qubit data_init for
# any CSS code — see surgery.logical_state_init docstring for the asymmetry
# explanation. Plug straight into data_init=.
table = [
    ("|0⟩_L", "0", "random",                         0.5, True),
    ("|1⟩_L", "1", "random",                         0.5, True),
    ("|+⟩_L", "+", "+1",                             0.0, False),
    ("|−⟩_L", "-", f"(-1)^{wt} = {(-1)**wt:+d}",     1.0, False),
]
print(f"X̄ support weight = {wt}  →  X̄|−⟩_L = ({-1 if wt % 2 else +1:+d})|−⟩_L\n")
print(f"{'data':<6} | {'X̄ eigenvalue':<16} | {'expected obs0':<14} | "
      f"{'measured obs0 (frac=1)':<24} | obs0==obs1 | ok")
print("-" * 96)
for label, state, eig, expected, stochastic in table:
    circuit = build_single_ppm_circuit(
        g_steane, rounds=3, noise_model=None,
        data_init=logical_state_init(steane, state),
    )
    # raw_observables (NOT compile_detector_sampler) — see §0 helper docstring.
    obs = raw_observables(circuit, SHOTS)
    rate0 = float(obs[:, 0].mean())
    agree = float((obs[:, 0] == obs[:, 1]).mean())
    if stochastic:
        ok = 0.4 < rate0 < 0.6 and agree == 1.0
        exp_str = "~50%"
    else:
        ok = rate0 == expected and agree == 1.0
        exp_str = f"{expected:>4.0%}"
    flag = "✓" if ok else "✗"
    print(f"{label:<6} | {eig:<16} | {exp_str:>14} | "
          f"{rate0:>10.2%} ({int(obs[:, 0].sum()):>4}/{SHOTS})    | "
          f"{agree:>7.1%}   | {flag}")
    assert ok, f"failed for state={state!r}"
print("\n✓ Steane single-PPM measures X̄ correctly under all 4 logical inits")
```

- [ ] **Step 3: Also add `logical_state_init` to the §0 imports**

Find cell id `66661ed0` (the §0 import cell). Use NotebookEdit to replace the surgery import line:

```python
from qldpc.circuits.surgery import (
    build_gadget, build_bridge,
    build_single_ppm_circuit, build_joint_ppm_circuit,
    boost_gadget, cheeger_constant, keep_only_observable,
)
```

with:

```python
from qldpc.circuits.surgery import (
    build_gadget, build_bridge,
    build_single_ppm_circuit, build_joint_ppm_circuit,
    boost_gadget, cheeger_constant, keep_only_observable,
    logical_state_init,
)
```

- [ ] **Step 4: Re-execute the notebook cells §0–§1 to verify**

```bash
python -c "
import nbformat
from nbclient import NotebookClient
nb = nbformat.read('examples/lattice_surgery.ipynb', as_version=4)
end_idx = next(i for i, c in enumerate(nb.cells)
               if c.cell_type == 'markdown' and '## §2' in ''.join(c.source))
ler_cells = nb.cells[end_idx:]
nb.cells = nb.cells[:end_idx]
NotebookClient(nb, timeout=120, resources={'metadata': {'path': 'examples'}}).execute()
nb.cells = list(nb.cells) + list(ler_cells)
nbformat.write(nb, 'examples/lattice_surgery.ipynb')
print('done')
"
```

Expected: prints `done`. Open the notebook (or `jq` the cell outputs) and confirm the §1 truth table shows `4/4 ✓`.

- [ ] **Step 5: Commit**

```bash
git add examples/lattice_surgery.ipynb
git commit -m "$(cat <<'EOF'
docs(notebook): §1 truth table uses logical_state_init

Replaces hardcoded "0"*n / "1"*n / "+"*n / "-"*n in the Steane single-PPM
truth table with logical_state_init(steane, state). For Steane the four
strings happen to match the helper output (wt(X̄)=3 odd), but using the
helper keeps §1 honest and consistent with §2's BBCode-paired flow.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Notebook §2 (joint PPM truth table) — use helper, restore textbook expected

**Files:**
- Modify: `examples/lattice_surgery.ipynb` cell id `b8340c26`

- [ ] **Step 1: Inspect current cell**

Run: `python -c "import json; nb = json.load(open('examples/lattice_surgery.ipynb')); print(next(''.join(c['source']) for c in nb['cells'] if c.get('id') == 'b8340c26'))"`

Expected: prints the current §2 joint truth-table code, which currently has the inline `prep(label, x_bar)` helper and a parity-derived `expected_obs0` calculation that we are about to remove.

- [ ] **Step 2: Replace the cell source**

Use NotebookEdit to replace the entire `b8340c26` cell source with:

```python
# Inter-code joint Z̄_1 ⊗ Z̄_2, code-agnostic via logical_state_init.
xs, ys = sympy.symbols("x y")
c1 = codes.SteaneCode()
c2 = codes.BBCode({xs: 3, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)
z1 = np.asarray(c1.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
z2 = np.asarray(c2.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
g1 = build_gadget(c1, z1, basis=Pauli.Z)
g2 = build_gadget(c2, z2, basis=Pauli.Z)
bridge = build_bridge(g1, g2)
rounds = 3  # odd → Webster Eq.1 ≡ Z̄_1 ⊗ Z̄_2

circuit_default, joint_code = build_joint_ppm_circuit(
    g1, g2, bridge, rounds=rounds, noise_model=None,
)
wt1, wt2 = int(z1.sum()), int(z2.sum())
print(f"joint code           : [[{joint_code.num_qudits}, {joint_code.dimension}]]")
print(f"bridge.width         : {bridge.width}")
print(f"extra κ_l, κ_r       : {bridge.extra_kappa_l.shape[0]}, {bridge.extra_kappa_r.shape[0]}")
print(f"wt(Z̄_1) = {wt1} ({'odd' if wt1 % 2 else 'even'}),  "
      f"wt(Z̄_2) = {wt2} ({'odd' if wt2 % 2 else 'even'})  "
      f"— logical_state_init handles either parity")
print()

SHOTS_JOINT = 1000
print(f"{'state':>16} | Z̄_1  Z̄_2 | expected obs0 | measured obs0 (frac=1)  | ok")
print("-" * 86)
joint_pass = True
for s1, s2, sign1, sign2, expected in [
    ("0", "0", "+1", "+1", 0),
    ("0", "1", "+1", "-1", 1),
    ("1", "0", "-1", "+1", 1),
    ("1", "1", "-1", "-1", 0),
]:
    label = f"|{s1}⟩_L|{s2}⟩_L"
    circuit, _ = build_joint_ppm_circuit(
        g1, g2, bridge, rounds=rounds, noise_model=None,
        data_init=(
            logical_state_init(c1, s1),
            logical_state_init(c2, s2),
        ),
    )
    obs = raw_observables(circuit, SHOTS_JOINT)
    rate = float(obs[:, 0].mean())
    ok = rate == float(expected)
    flag = "✓" if ok else "✗"
    print(f"{label:>16} | {sign1:>4}  {sign2:>4} | "
          f"{expected:>13} | {rate:>10.2%} ({int(obs[:, 0].sum()):>3}/{SHOTS_JOINT})    | {flag}")
    joint_pass = joint_pass and ok

assert joint_pass, "joint-PPM correctness failed"
print("\n✓ joint Z̄_1 ⊗ Z̄_2 observable matches expected parity deterministically")
```

- [ ] **Step 3: Re-execute §0–§3 to verify**

```bash
python -c "
import nbformat
from nbclient import NotebookClient
nb = nbformat.read('examples/lattice_surgery.ipynb', as_version=4)
end_idx = next(i for i, c in enumerate(nb.cells)
               if c.cell_type == 'markdown' and '## §4' in ''.join(c.source))
ler_cells = nb.cells[end_idx:]
nb.cells = nb.cells[:end_idx]
NotebookClient(nb, timeout=300, resources={'metadata': {'path': 'examples'}}).execute()
nb.cells = list(nb.cells) + list(ler_cells)
nbformat.write(nb, 'examples/lattice_surgery.ipynb')
print('done')
"
```

Expected: prints `done`. Inspect the §2 cell output:

```bash
python -c "
import json
nb = json.load(open('examples/lattice_surgery.ipynb'))
for c in nb['cells']:
    if c.get('id') == 'b8340c26':
        for o in c.get('outputs', []):
            if o.get('output_type') == 'stream':
                print(''.join(o.get('text', [])))
"
```

Expected output should include:
```
       |0⟩_L|0⟩_L |   +1    +1 |             0 |      0.00% ...    | ✓
       |0⟩_L|1⟩_L |   +1    -1 |             1 |    100.00% ...    | ✓
       |1⟩_L|0⟩_L |   -1    +1 |             1 |    100.00% ...    | ✓
       |1⟩_L|1⟩_L |   -1    -1 |             0 |      0.00% ...    | ✓
```

(All four rows must say `✓`. Steane × BBCode worked under the previous "compute expected from parity" fix but now also passes with textbook `0/1/1/0` because the helper actually prepares the right logical state.)

- [ ] **Step 4: Commit**

```bash
git add examples/lattice_surgery.ipynb
git commit -m "$(cat <<'EOF'
docs(notebook): §2 joint truth table uses logical_state_init

Replace the inline prep(label, x_bar) helper introduced in the previous
"compute expected from wt parity" fix with the public logical_state_init
helper. The expected obs0 column reverts to the textbook 0/1/1/0 because
the helper now correctly prepares logical |1⟩_L on the BBCode side
(supp(X̄_0) flipped) instead of physical |1⟩^36 (which lands in the
Z̄_0 = +1 = |0⟩_L sector since wt(Z̄_0) = 8 is even).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Final full-suite verification

**Files:** No code changes.

- [ ] **Step 1: Run the full surgery test suite**

Run: `uv run pytest src/qldpc/circuits/surgery/ -q`
Expected: all tests PASS, including the 8 + 2 + 2 = 12 new ones from Tasks 1–3.

- [ ] **Step 2: Run the full repo test suite to catch any unexpected regression**

Run: `uv run pytest src/qldpc/ -q -x --ignore=src/qldpc/codes/lifted_product_test.py 2>&1 | tail -20`

(The `--ignore` on lifted_product_test is a workaround for a known long-running test unrelated to this work — drop if not needed.)

Expected: all tests PASS or report skipped.

- [ ] **Step 3: Verify branch state and prepare for ship**

```bash
git log --oneline -10
git status
```

Expected: 6 new commits from Tasks 1–6, working tree clean.

This task has no commit — it is a verification gate.

---

## How to test the whole feature manually

After all 7 tasks land, do one end-to-end smoke check:

```bash
uv run python -c "
from qldpc import codes
from qldpc.objects import Pauli
from qldpc.circuits.surgery import build_gadget, build_single_ppm_circuit, logical_state_init
import sympy, numpy as np

xs, ys = sympy.symbols('x y')
code = codes.BBCode({xs: 3, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)
z_bar = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
print(f'wt(Z̄_0) = {int(z_bar.sum())}  (even → naive \"1\"*n FAILS, helper works)')
g = build_gadget(code, z_bar, basis=Pauli.Z)

for state in ['0', '1']:
    init = logical_state_init(code, state)
    circuit = build_single_ppm_circuit(g, rounds=3, noise_model=None, data_init=init)
    raw = circuit.compile_sampler().sample(shots=100).astype(np.uint8)
    n_meas = raw.shape[1]
    for ln in str(circuit).splitlines():
        if ln.startswith('OBSERVABLE_INCLUDE(0)'):
            recs = [int(t.strip('rec[]')) for t in ln.split() if t.startswith('rec[')]
            break
    obs0 = np.bitwise_xor.reduce(raw[:, [n_meas + off for off in recs]], axis=1)
    print(f'  state={state!r}: obs0 rate = {obs0.mean():.2%}  (expected {state})')
"
```

Expected:
```
wt(Z̄_0) = 8  (even → naive "1"*n FAILS, helper works)
  state='0': obs0 rate = 0.00%  (expected 0)
  state='1': obs0 rate = 100.00%  (expected 1)
```

If state='1' shows `0.00%` (not `100.00%`), the helper is broken.

---

## After plan completes

Use `superpowers:finishing-a-development-branch` to merge / push / cleanup the feature branch.
