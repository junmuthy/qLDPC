# Logical state init helper for surgery PPM testing

**Date:** 2026-06-10
**Goal:** Add `logical_state_init(code, state) → str` to `qldpc.circuits.surgery` so notebook truth tables and pytest end-to-end tests can prepare any of the four Pauli logical states (`|0⟩_L`, `|1⟩_L`, `|+⟩_L`, `|-⟩_L`) correctly on **any CSS code**, not just codes where `wt(X̄)` / `wt(Z̄)` are odd.

## Motivation

`build_single_ppm_circuit` and `build_joint_ppm_circuit` accept a `data_init` per-qubit string of characters `{0, 1, +, -}`. For notebook truth tables and pytest sanity tests we need to prep specific logical states from this physical-init string. Two of the four states are trivial; the other two are easy to silently get wrong:

| Target | Naive prep | When correct | When wrong |
|---|---|---|---|
| `|0⟩_L` | `"0" * n` | Always (CSS code with all-Z-stab) | Never |
| `|+⟩_L` | `"+" * n` | Always (CSS code with all-X-stab) | Never |
| `|1⟩_L` | `"1" * n` | When `wt(Z̄_0)` is odd | When `wt(Z̄_0)` is even |
| `|-⟩_L` | `"-" * n` | When `wt(X̄_0)` is odd | When `wt(X̄_0)` is even |

The asymmetry is real, not cosmetic. CSS code structure trivially places `|0⟩^n` in the `|0⟩_L^{⊗k}` sector (every Z-product, including every Z̄_i, acts as `+1` on `|0⟩^n`), and `|+⟩^n` in the `|+⟩_L^{⊗k}` sector. Reaching `|1⟩_L` from `|0⟩_L` requires applying the logical X̄_0, which physically means flipping exactly `supp(X̄_0)` — *not* flipping every qubit. For self-dual codes with odd-weight logicals (e.g. Steane), `X^{⊗n}` happens to coincide with X̄_0 modulo stabilizers, so `"1" * n` works by coincidence. For BB code `[[36, 8]]` with `wt(Z̄_0) = 8`, `"1" * 36` puts the state in the `Z̄_0 = +1` sector (= logical `|0⟩_L` on qubit 0), and a hardcoded truth-table expectation of `1` silently fails.

The notebook truth tables (§1 single PPM, §2 joint PPM) need correct prep for testing the surgery protocol's correctness on arbitrary CSS code combinations, not just Steane × Steane. The helper is also useful for any future pytest end-to-end test that wants to assert deterministic obs0 on a specific logical input.

## Scope

**In scope:**
- One helper function `logical_state_init(code: CSSCode, state: str) → str`
- Four states: `"0"`, `"1"`, `"+"`, `"-"`
- Logical qubit 0 only (other logical qubits left in trivial `|0⟩_L` or `|+⟩_L` sector)
- Lives in `src/qldpc/circuits/surgery/circuit.py`
- Exported via `qldpc/circuits/surgery/__init__.py`
- 6 pytest cases in `_test_circuit.py`
- Notebook §1 and §2 updated to use the helper

**Out of scope:**
- General logical superposition states (`α|0⟩_L + β|1⟩_L` with arbitrary α, β) — requires magic states / non-Clifford prep
- Multi-logical-qubit specs (e.g. `state = "01+"` for k = 3) — current `data_init` API broadcasts at the code level, not the logical-qubit level
- Logical qubit selection beyond qubit 0 — could be a future kwarg `log_idx: int = 0` but no current consumer needs it
- Helpers for non-CSS / qudit codes — surgery module is GF(2) only
- Changes to `build_*_ppm_circuit` or `data_init` parsing — helper output plugs into the existing API unchanged

## API

```python
def logical_state_init(code: CSSCode, state: str) -> str:
    """Per-qubit data_init string preparing a Pauli logical state on
    logical qubit 0 of a CSS code.

    state ∈ {"0", "1", "+", "-"}:
      "0" → "0" * n          (|0⟩^n projects to |0⟩_L for any CSS code)
      "1" → "1" on supp(X̄_0), "0" elsewhere     (X̄_0 |0⟩_L = |1⟩_L)
      "+" → "+" * n          (|+⟩^n projects to |+⟩_L for any CSS code)
      "-" → "-" on supp(Z̄_0), "+" elsewhere     (Z̄_0 |+⟩_L = |-⟩_L)

    The returned string has length code.num_qudits and is intended to be
    passed directly as data_init= to build_single_ppm_circuit, or wrapped
    per-side and passed as a tuple to build_joint_ppm_circuit.

    Examples
    --------
    Single PPM, measure Z̄_0 on logical |1⟩_L:
        z_bar = code.get_logical_ops(Pauli.Z)[0]
        g = build_gadget(code, z_bar, basis=Pauli.Z)
        circuit = build_single_ppm_circuit(
            g, rounds=3, noise_model=None,
            data_init=logical_state_init(code, "1"),
        )

    Joint PPM, c1 in |0⟩_L, c2 in |1⟩_L:
        circuit, _ = build_joint_ppm_circuit(
            g1, g2, bridge, rounds=3, noise_model=None,
            data_init=(
                logical_state_init(c1, "0"),
                logical_state_init(c2, "1"),
            ),
        )

    Notes
    -----
    X̄_0, Z̄_0 are taken from code.get_logical_ops(Pauli.X)[0] /
    code.get_logical_ops(Pauli.Z)[0]; qldpc guarantees they anti-commute
    (form a symplectic pair on logical qubit 0). The construction is
    therefore correct for any CSS code regardless of the parity of
    wt(X̄_0) or wt(Z̄_0). Naive broadcast (data_init = state * n) is
    correct only when those weights are odd, and silently produces the
    wrong logical state on codes where they are even (e.g. BBCode with
    even-weight logicals).

    Raises
    ------
    ValueError
        If state is not one of "0", "1", "+", "-".
    """
```

The function is small and self-contained (≈ 25 LOC including docstring). It performs:
1. Validate `state` against the four-element set; raise `ValueError` otherwise.
2. Read `n = code.num_qudits`.
3. For `"0"` / `"+"`: return `state * n` directly.
4. For `"1"` / `"-"`: fetch the appropriate logical operator support
   (`get_logical_ops(Pauli.X)[0]` for `"1"`, `get_logical_ops(Pauli.Z)[0]` for `"-"`)
   and build the per-qubit string with the flip character on supp positions
   and the base character ("0" for `"1"`, "+" for `"-"`) elsewhere.

## Implementation location

`src/qldpc/circuits/surgery/circuit.py`. The file already hosts the `build_single_ppm_circuit` and `build_joint_ppm_circuit` public functions; the helper conceptually belongs alongside them as part of the "user-facing PPM construction toolkit." No new file is introduced.

The function is added to `src/qldpc/circuits/surgery/__init__.py` `__all__` and `from .circuit import` statement so callers can write:

```python
from qldpc.circuits.surgery import logical_state_init
```

## Testing

Add the following pytest cases to `src/qldpc/circuits/surgery/_test_circuit.py`:

1. `test_logical_state_init_zero_and_plus_broadcast`
   - For Steane and a BBCode, assert `logical_state_init(code, "0") == "0" * code.num_qudits` and similarly for `"+"`.

2. `test_logical_state_init_one_flips_x_bar_support`
   - For Steane and a BBCode, build the helper output for `"1"`, parse out the positions where the character is `"1"`, and assert it equals `set(np.where(get_logical_ops(Pauli.X)[0])[0])`.

3. `test_logical_state_init_minus_flips_z_bar_support`
   - Same as above but for `"-"` and `Pauli.Z`.

4. `test_logical_state_init_invalid_state_raises`
   - Parametrized over `state ∈ {"2", "x", "", "01"}`, assert `pytest.raises(ValueError)`.

5. `test_logical_state_init_end_to_end_steane_basis_z`
   - Steane single-PPM with `basis=Pauli.Z`, `rounds=3`, noiseless. For each `state ∈ {"0", "1"}`, build circuit with `data_init=logical_state_init(steane, state)`, sample, assert `obs0` rate equals `int(state)` deterministically (matches textbook Z̄ eigenvalue).
   - Steane has odd-weight Z̄, so this is the "naive happens to work" case — verifies helper agrees with naive for the legacy case.

6. `test_logical_state_init_end_to_end_bbcode_basis_z`
   - BBCode `[[36, 8]]` (l=3, m=6) single-PPM with `basis=Pauli.Z`, `rounds=3`, noiseless. For each `state ∈ {"0", "1"}`, assert `obs0` rate equals `int(state)`.
   - BBCode has even-weight Z̄_0, so this is the "naive silently breaks" case — failing here would mean the helper is no better than naive broadcast. This is the regression test that catches the original notebook bug.

The end-to-end tests also serve as the spec for `build_single_ppm_circuit` correctness on the helper's output, so any future change to either side gets caught.

## Notebook integration

After the helper lands, update the notebook to consume it:

**`examples/lattice_surgery.ipynb` cell `84449394` (§1 Steane single-PPM truth table):**
- Replace the hardcoded `"0" * n` / `"+" * n` strings with `logical_state_init(steane, state)`.
- For `"-"` row, this is the cleanup: currently the cell relies on Steane's coincidence that `wt(X̄) = 3` is odd; the helper makes the prep correct for any code.

**`examples/lattice_surgery.ipynb` cell `b8340c26` (§2 joint PPM truth table):**
- Delete the inline `prep(label, x_bar)` helper that the previous fix introduced.
- Use `logical_state_init(c1, state)` and `logical_state_init(c2, state)` in the `data_init=(...)` tuple.
- Expected `obs0` column reverts to the textbook `0/1/1/0`. No more "compute expected from `wt` parity" logic; the prep handles it.

The markdown above each cell stays largely unchanged — it already explains the symmetry between trivial states (`|0⟩_L`, `|+⟩_L`) and operator-induced states (`|1⟩_L`, `|-⟩_L`).

## Architecture / dependencies

The helper sits at the boundary between `qldpc.codes.CSSCode` (for `get_logical_ops`) and the existing `data_init` string contract of `_surgery_state_prep`. It is **pure data transformation** — no circuit construction, no stim dependency. This keeps it cheap to call from anywhere (notebook, test, user code) without paying the cost of building a circuit.

It does not modify any existing surgery function. `build_single_ppm_circuit` and `build_joint_ppm_circuit` continue to receive a `data_init` string and pass it through to `_surgery_state_prep` exactly as today.

## Risks and edge cases

**`code.dimension == 0`**: The code encodes no logical qubits and `get_logical_ops` returns an empty array. State `"0"` / `"+"` still works (returns the broadcast string). State `"1"` / `"-"` should raise — accessing index `[0]` of an empty logical-ops array raises `IndexError` naturally; we do not catch and rewrap. Document that the helper requires `code.dimension >= 1`.

**Non-CSS code passed in**: `code.get_logical_ops(Pauli.X)` may behave differently or error. The helper is documented as taking a `CSSCode`; type annotations enforce this at static-check time. No runtime check beyond the natural error from `get_logical_ops`.

**Custom Z̄ representative used by gadget**: If the user supplies a Z̄ to `build_gadget` that is not `code.get_logical_ops(Pauli.Z)[0]` (e.g. a BP+OSD-found low-weight rep, as in §3.2 of the notebook), then the X̄_0 picked by this helper may not anti-commute with that specific representative. The same-coset rule (`get_logical_ops(Pauli.X)[0]` anti-commutes with any vector in the same logical-qubit-0 Z-logical coset) means the helper is still correct *if* the custom Z̄ rep stays in the qubit-0 coset. For arbitrary custom rep on an unspecified logical qubit, the helper is out of scope; document this clearly and refer the user to constructing `data_init` manually if needed.

**State characters beyond `{0, 1, +, -}`**: Out of scope. The four characters cover the four Pauli eigenstates; anything else requires non-Clifford prep that the underlying `_surgery_state_prep` already rejects.

## Success criteria

- `pytest src/qldpc/circuits/surgery/` passes all existing tests plus the 6 new ones.
- Notebook §1 truth table prints `4/4 ✓` for Steane with each of `{0, 1, +, -}` and the expected `obs0` matches the textbook prediction.
- Notebook §2 joint truth table prints `4/4 ✓` for Steane × BBCode `[[36, 8]]` with the textbook `0/1/1/0` expected column (no parity-dependent expected formula).
- A reader of the notebook can swap `c1` / `c2` to any CSS code pair and the truth table either still passes or fails with a clear error pointing at the gadget construction (not at the prep).
