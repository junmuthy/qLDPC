# Surgery `data_init` kwarg — Design

**Status**: Draft (2026-06-10)
**Scope**: `src/qldpc/circuits/surgery/circuit.py` (public API +1 kwarg) + `examples/lattice_surgery.ipynb`
**Author**: tgzhou (with Claude)

## Background

The surgery demo notebook (`examples/lattice_surgery.ipynb`) uses three private helper functions to post-process the circuits produced by `build_single_ppm_circuit` and `build_joint_ppm_circuit`, just to override the data-qubit initial state for correctness tests:

- `_swap_data_init_to_zero(circuit, data_ids)` — rewrite `RX(data) → R(data)` so basis=X gadgets get data initialized as $|0\rangle$
- `_switch_init_basis(circuit, data_ids)` — rewrite `R(data) → RX(data)` so basis=Z gadgets get data initialized as $|+\rangle$
- `_mutate_init(circuit, data_ids)` — append `X(data)` immediately after the first `R` layer to flip data $|0\rangle \to |1\rangle$

Together these cover the four single-qubit initial states $\{|0\rangle, |1\rangle, |+\rangle, |-\rangle\}$ via a mix of init-gate substitution and post-init Pauli application.

The pattern is awkward: re-parsing a generated `stim.Circuit` to inject a different initial state requires the notebook to know surgery's internal gate-emission conventions (which gate name is used for init, where it appears, that it's nested inside REPEAT blocks for some operations but not others). This is "changing initial state after construction" — conceptually it belongs at construction time.

The init layer in `_surgery_state_prep` (circuit.py:534) is already isolated — exposing a `data_init` parameter there and forwarding it through the public API is small and clean.

## Goal

Add a `data_init: str | None = None` kwarg to the public `build_single_ppm_circuit` and `build_joint_ppm_circuit`, threaded through `_surgery_state_prep`. The kwarg lets callers specify per-data-qubit initial states without post-processing. The notebook then uses the new kwarg and the three post-process helpers go away.

## Non-goals

- **No change to κ / bridge initialization.** Ancilla qubits stay on the protocol default (Z-basis +1 eigenstate when surgery basis=X, X-basis +1 eigenstate when surgery basis=Z). Caller cannot override.
- **No new noise-model behavior.** `data_init` is composed *before* `noise_model.noisy_circuit(circuit)` is applied (same place as today's protocol-default init).
- **No change to default behavior.** When `data_init=None`, the emitted circuit is bit-identical to today's output.
- **No change to other public functions** (`build_gadget`, `build_bridge`, `boost_gadget`, `keep_only_observable`, `cheeger_constant`).

## API

### `_surgery_state_prep` — private helper, new kwarg

```python
def _surgery_state_prep(
    gadget: GadgetLayout,
    data_ids: tuple[int, ...],
    kappa_ids: tuple[int, ...],
    bridge_ids: tuple[int, ...] = (),
    *,
    data_init: str | None = None,
) -> stim.Circuit:
```

### Character → state → gate sequence

| `data_init[i]` | Initial state of data qubit `i` | Emitted gates |
|---|---|---|
| `"0"` | $\|0\rangle$ | `R(q)` |
| `"1"` | $\|1\rangle$ | `R(q); X(q)` |
| `"+"` | $\|+\rangle$ | `RX(q)` |
| `"-"` | $\|-\rangle$ | `RX(q); Z(q)` |

### `data_init` interpretation

- `None` (default) — Protocol default:
  - basis = `Pauli.X` → `RX` for all data qubits (every data qubit in $|+\rangle$)
  - basis = `Pauli.Z` → `R` for all data qubits (every data qubit in $|0\rangle$)
- Length-1 string (e.g. `"0"`, `"+"`) — Broadcast to all `len(data_ids)` data qubits.
- String of length exactly `len(data_ids)` — Per-qubit init; `data_init[i]` controls the init of `data_ids[i]`.

### Validation (inside `_surgery_state_prep`)

```python
if data_init is not None:
    if len(data_init) == 1:
        data_init = data_init * len(data_ids)
    if len(data_init) != len(data_ids):
        raise ValueError(
            f"data_init length {len(data_init)} does not match num data qubits "
            f"{len(data_ids)}; pass a length-1 string to broadcast"
        )
    invalid = sorted(set(data_init) - set("01+-"))
    if invalid:
        raise ValueError(
            f"data_init must contain only '0', '1', '+', '-'; got invalid chars {invalid}"
        )
```

### Emission order

For each `data_init[i]`:

1. If char ∈ `"01"` → append to a `R` group; `"1"` queues an `X` to apply after the `R` layer.
2. If char ∈ `"+-"` → append to an `RX` group; `"-"` queues a `Z` to apply after the `RX` layer.

After both `R` and `RX` groups are emitted, apply queued `X` and `Z` gates. κ + bridge init follows on the same basis as the protocol default (unchanged from today).

This yields the same gate order whenever `data_init=None`: all qubits go into the basis-appropriate single group (`RX` or `R`), no post-init Pauli is queued.

### Public API: `build_single_ppm_circuit`

```python
def build_single_ppm_circuit(
    gadget: GadgetLayout,
    *,
    rounds: int,
    noise_model=None,
    data_init: str | None = None,
) -> stim.Circuit:
```

The kwarg is forwarded straight to `_surgery_state_prep`. `len(data_ids) = gadget.code.num_qudits`.

### Public API: `build_joint_ppm_circuit`

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

Joint `data_init` length convention:

| Joint type | `data_ids` length | `data_init` length (per-qubit) |
|---|---|---|
| intra-code (`g_l.code is g_r.code`) | `n_l` (data shared) | `n_l` |
| inter-code (`g_l.code is not g_r.code`) | `n_l + n_r` | `n_l + n_r`; positions `[0 : n_l)` are left, `[n_l : n_l + n_r)` are right |

(Per-qubit string only; broadcast `"0"` etc. still works in both cases.)

## Testing

5 new tests in `_test_circuit.py`:

1. **`test_single_ppm_data_init_zero_random_outcome`** — Build Steane gadget (basis=X). Call `build_single_ppm_circuit(g, rounds=3, data_init="0")`. Sample noiselessly; assert obs0 flip rate ∈ (0.40, 0.60), obs0 ≡ obs1 on every shot. Mirrors §1 notebook semantics.

2. **`test_single_ppm_data_init_default_matches_pre_kwarg`** — Build Steane gadget (basis=X). Compare `build_single_ppm_circuit(g, rounds=3)` to `build_single_ppm_circuit(g, rounds=3, data_init=None)` and to a `data_init="+"` broadcast; all three must produce bit-identical `stim.Circuit` strings.

3. **`test_joint_ppm_data_init_truth_table`** — Two Steane copies, basis=Z, joint Z̄⊗Z̄. Sweep 4 inits — `"00"×7+"00"×7`, `"00"×7+"11"×7`, `"11"×7+"00"×7`, `"11"×7+"11"×7` (more concisely: `"0"*14`, `"0"*7+"1"*7`, `"1"*7+"0"*7`, `"1"*14`). For each, sample noiselessly and assert obs0 matches expected parity.

4. **`test_joint_ppm_data_init_superposition`** — Steane × Steane basis=Z, `data_init="0"*7 + "+"*7`. Assert obs0 flip rate ∈ (0.40, 0.60) AND `obs0 == obs1` on every shot. Mirrors §2 superposition notebook test.

5. **`test_data_init_validation`** — Three sub-cases (parametrized): wrong length (Steane n=7, pass `"00"`), invalid character (`"@" * 7`), all should raise `ValueError`.

After landing: surgery suite goes from 88 → 93 (5 new tests).

## Notebook update

A second commit drops the three post-process helpers and rewrites the call sites:

- §1 single-PPM correctness: `verify_single_ppm` calls `build_single_ppm_circuit(g, rounds=3, noise_model=None, data_init="0")` directly. Delete `_swap_data_init_to_zero`.
- §2 truth-table call sites: `_mutate_init(circuit_joint, [flip ids])` → reconstruct `data_init` string from `flip_ids` and call `build_joint_ppm_circuit(..., data_init=...)`. Delete `_mutate_init`.
- §2 superposition: `_switch_init_basis(circuit_joint, data2)` → `build_joint_ppm_circuit(..., data_init="0"*n + "+"*n)`. Delete `_switch_init_basis`.

The notebook is updated via direct ipynb cell edit (no jupytext source). After edits, re-execute §0-§3 cells so outputs reflect the new code. §4 LER cells stay unexecuted.

## Compatibility

Existing callers of `build_*_ppm_circuit` keep working with no change. The default `data_init=None` reproduces today's exact gate stream. All existing tests (88 of them) should pass unchanged.

External consumers (the surgery demo notebook is the only one in this repo) get updated as part of this PR.

## Net effect

| File | Δ |
|---|---|
| `circuit.py` | +~30 lines (`_surgery_state_prep` validation + per-char gate emission + 2 public-API kwarg forwarding + docstring updates) |
| `_test_circuit.py` | +~80 lines (5 new tests) |
| `lattice_surgery.ipynb` | −~90 lines net (drop 3 helpers, ~5 call-site simplifications) |
| Surgery test count | 88 → 93 |
| Public API surface | +1 kwarg per affected function; default preserves prior behavior |

## Success criteria

- `pytest src/qldpc/circuits/surgery/ -q` reports **93 passed**.
- `python -c "from qldpc.circuits.surgery import build_single_ppm_circuit; help(build_single_ppm_circuit)" | grep data_init` finds the new kwarg in the docstring.
- `grep -n "_swap_data_init_to_zero\|_mutate_init\|_switch_init_basis" examples/lattice_surgery.ipynb` returns empty after the notebook commit.
- Re-executing notebook §0-§3 succeeds (all assertions pass: §1 obs0/obs1 agree, §2 truth-table matches, §2 superposition obs0 random + obs0==obs1).
- `build_single_ppm_circuit(g, rounds=3)` on Steane returns a `stim.Circuit` whose string representation is bit-identical to the same call pre-refactor (the `_default_matches_pre_kwarg` test pins this).

## Open questions

None — API shape and char-to-gate mapping are explicit above. Notebook helpers' joint usage at §2 truth-table is straightforward (just construct the right string from `flip_ids`).
