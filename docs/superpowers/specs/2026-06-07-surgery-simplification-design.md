# Surgery Module Simplification — Design Spec

**Date:** 2026-06-07
**Status:** approved (brainstorming) → awaiting writing-plans
**Branch:** `feat/surgery-construction`

## Motivation

The `src/qldpc/codes/surgery/` package grew to 2,518 lines across 8 files
covering Webster L-layered gadgets (L ≥ 1), single-PPM and multi-PPM
constructions, intra-code and inter-code joint measurements via bridges,
SetValuedPort overlap logic, Cheeger distance boost, SkipTree adapter,
and cellulation helpers. Only a subset is load-bearing for the user's
current research direction. This spec narrows the package to the minimum
needed for: (1) Webster L=1 gadget construction, (2) single-PPM
measurement circuit generation, (3) bridge construction between two
gadgets, (4) joint-PPM (intra-code or inter-code) measurement circuit
generation, plus Cheeger distance boost as a utility.

## Goals

1. Five public symbols only: `build_gadget`, `build_bridge`,
   `build_single_ppm_circuit`, `build_joint_ppm_circuit`,
   `boost_gadget` — plus the `GadgetLayout` and `Bridge` dataclasses
   they return/consume.
2. `build_gadget(code, x)` is **deterministic**: same `(code, x)` ⇒
   byte-identical `GadgetLayout`. Ancilla ordering (κ-block,
   χ-rows, gauge-fix rows) is canonical and reproducible.
3. The Webster gadget construction is implemented as three explicit
   named steps that map 1:1 to math.md §1.1–1.4.
4. Webster Table I (κ+χ+r ∈ {19,31,49,79}, 2w−1 ∈ {11,19,31,51}) is
   recovered exactly by an automated test. Ide BB↔LP inter-code joint
   recovers (n=355, k=25, d=10) exactly.
5. Stim surgery circuit generation (single-PPM and joint-PPM) is a
   first-class API. Noise model layer and DETECTOR / OBSERVABLE_INCLUDE
   construction included.
6. No backwards-compatibility aliases. Old names are removed cleanly.
7. **Final code is minimal and reviewable end-to-end.** Each module
   stays small enough to hold in head at once. Target line budget
   (excluding `_test.py` and `cheeger.py`): `gadget.py` ≤ 200,
   `bridge.py` ≤ 350, `circuit.py` ≤ 300, `__init__.py` ≤ 30. No
   helper functions that exist only to be called once unless they
   carry a named paper step (the three `_stepN_*` in `gadget.py` are
   the explicit exception). No defensive checks beyond what
   `build_gadget` / `build_bridge` / `build_*_circuit` need at the
   API boundary. Docstrings cite math.md sections rather than
   re-deriving math.

## Non-goals

- Multi-PPM / SetValuedPort / overlap measurements (multi.py + port.py).
- Webster L ≥ 3 layered structure.
- New algorithms — this is a simplification + clean-rename + a Stim
  circuit API surfacing existing inline circuit-building logic.

## 1. Target module layout

```
src/qldpc/codes/surgery/
  __init__.py        # public re-exports only (5 symbols + 2 dataclasses)
  gadget.py          # L=1 Webster gadget, 3 explicit steps
  bridge.py          # standalone bridge adapter (intra + inter)
  cheeger.py         # distance boost
  circuit.py         # NEW — Stim circuit + noise + detectors + observables
  _test.py           # replaces src/qldpc/codes/surgery_test.py
```

**Deleted:** `layered.py`, `joint.py`, `multi.py`, `port.py`,
`skiptree.py`, `cellulation.py`. The last two get folded into
`bridge.py` as private helpers.

## 2. Public API

```python
# === gadget.py ===
@dataclass(frozen=True)
class GadgetLayout:
    code: CSSCode
    x: np.ndarray                      # logical-X support, length n
    V0: tuple[int, ...]                # supp(x), sorted ascending
    C0: tuple[int, ...]                # Z-checks touching V_0, sorted ascending
    F: np.ndarray                      # H_Z[C_0, V_0]
    G: np.ndarray                      # gauge-fix basis, row-reduced over GF(2)
    HX_merged: np.ndarray              # (m_X + |V_0|) × (n + |C_0|)
    HZ_merged: np.ndarray              # (m_Z + r) × (n + |C_0|)
    kappa_qubits: tuple[int, ...]      # κ-ancilla indices in merged register

def build_gadget(code: CSSCode, x: np.ndarray) -> GadgetLayout: ...
```

`build_gadget` is the three explicit steps from math.md §1:

```python
def build_gadget(code, x):
    V0, C0, F = _step1_restriction(code, x)             # math.md §1.1
    G        = _step2_gauge_fix(F)                       # math.md §1.2
    HX_m, HZ_m = _step3_assemble(code, V0, C0, F, G)     # math.md §1.4
    return GadgetLayout(code, x, V0, C0, F, G, HX_m, HZ_m,
                        kappa_qubits=tuple(range(code.num_qudits,
                                                 code.num_qudits + len(C0))))
```

Each `_stepN_*` helper is exported privately (underscore prefix) so the
three-step structure is browsable but not part of the public surface.

```python
# === bridge.py ===
@dataclass(frozen=True)
class Bridge:
    width: int                                     # w = min(|V_0^(1)|, |V_0^(2)|)
    qubits: tuple[int, ...]                        # bridge data qubit indices
    U_B: np.ndarray                                # (w-1) × w path-graph X-stabilizers
    chi_endpoint_extensions: dict[int, np.ndarray] # math.md §2.3
    intercode: bool                                # True ⇒ inter-code path
    aux_graph_edges: tuple[tuple[int, int], ...] | None  # inter only
    z_extensions: dict[int, np.ndarray] | None           # inter only (chi-Z bridge)

def build_bridge(g1: GadgetLayout, g2: GadgetLayout) -> Bridge: ...
    # Auto-dispatches: g1.code is g2.code  ⇒ intra-code
    #                  else                ⇒ inter-code (cellulation + SkipTree)
```

```python
# === circuit.py ===
def build_single_ppm_circuit(
    gadget: GadgetLayout, *,
    rounds: int,
    noise_model: NoiseModel | None = None,
) -> tuple[stim.Circuit, MeasurementRecord]: ...

def build_joint_ppm_circuit(
    g1: GadgetLayout, g2: GadgetLayout, bridge: Bridge, *,
    rounds: int,
    noise_model: NoiseModel | None = None,
) -> tuple[stim.Circuit, MeasurementRecord, CSSCode]: ...
    # Third return is the merged joint surgery CSSCode so callers can
    # inspect distance / run decoders on it.
```

```python
# === cheeger.py ===
def boost_gadget(
    gadget: GadgetLayout, *,
    method: Literal['spectral', 'combinatorial', 'distance'],
    target: float,
    seed: int | None = None,
    **kwargs,
) -> GadgetLayout: ...
    # Single entry point. Returns a NEW GadgetLayout with boosted F (and
    # consequently new HX_merged / HZ_merged). Seed-deterministic when
    # method is 'spectral'; 'combinatorial' is deterministic regardless
    # of seed; 'distance' is seed-deterministic.
```

**Determinism contract.** `build_gadget` and `build_bridge` are
deterministic in their inputs. `build_joint_ppm_circuit` is
deterministic given `(g1, g2, bridge, rounds, noise_model)`.
`boost_gadget` is deterministic in `(gadget, method, target, seed,
kwargs)`.

## 3. Migration

### Files

| File | Action |
|---|---|
| `src/qldpc/codes/surgery/layered.py` | Replace with `gadget.py` |
| `src/qldpc/codes/surgery/joint.py` | Split: bridge code → `bridge.py`; joint code assembly → `circuit.py` |
| `src/qldpc/codes/surgery/skiptree.py` | Delete (folded into `bridge.py`) |
| `src/qldpc/codes/surgery/cellulation.py` | Delete (folded into `bridge.py`) |
| `src/qldpc/codes/surgery/multi.py` | Delete |
| `src/qldpc/codes/surgery/port.py` | Delete |
| `src/qldpc/codes/surgery/cheeger.py` | Edit: add `boost_gadget` dispatcher |
| `src/qldpc/codes/surgery/__init__.py` | Rewrite to 5 public symbols + 2 dataclasses |
| `src/qldpc/codes/surgery_test.py` | Move + rewrite → `src/qldpc/codes/surgery/_test.py` |
| `src/qldpc/codes/__init__.py` | Drop re-exports of removed symbols |
| `examples/webster_table1_verify.py` | Delete (becomes a unit test) |

### Example scripts (`examples/scripts/`)

| Script | Migration |
|---|---|
| `cain_bb18_*`, `cain_lp_*`, `cain_lp20_*`, `cain_lp24_*` | Rename `build_layered_surgery_code` → `build_gadget`; `boost_gadget_cheeger_combinatorial` → `boost_gadget(method='combinatorial')` |
| `cain_fig1b_circuit_level.py`, `cain_fig1b_full_protocol.py` | Replace `_stitch_gadgets_with_bridge` + inline Stim with `build_bridge` + `build_joint_ppm_circuit` |
| `cain_fig1b_webster_surgery.py` | Rename calls |
| `ide_table_ii_*`, `ide_skiptree_verification.py`, `ide_lemma10_prototype.py` | Rename calls |
| `verify_cain_table_iii.py`, `cain_table_iii_summary.py` | Rename calls |
| `_9_lattice_surgery_cain_fig1b_source.py` | Drop `num_layers=1` arg; rename |

No backwards-compat shims. Old names removed.

## 4. Test plan

Tests live in `src/qldpc/codes/surgery/_test.py`. Estimated ~25 tests,
~600 lines.

### A. Determinism (core constraint)

- `test_build_gadget_deterministic` — same input ⇒ byte-identical layout.
- `test_build_gadget_canonical_ordering` — V_0, C_0 sorted; G row-reduced.
- `test_build_bridge_deterministic`.
- `test_build_joint_ppm_circuit_deterministic`.
- `test_boost_gadget_seed_reproducible` — `seed=42` twice ⇒ identical.
- `test_boost_gadget_seed_differs` — `seed=42` vs `seed=43` differ.

### B. Webster Table I exact-match ground truth

```python
# Webster Appendix A code index → (κ+χ+r, 2w-1)
WEBSTER_TABLE_I = [
    (0, 19, 11),
    (1, 31, 19),
    (2, 49, 31),
    (3, 79, 51),
]

@pytest.mark.parametrize('code_index,n_anc,bridge_w', WEBSTER_TABLE_I)
def test_webster_table_i_exact(code_index, n_anc, bridge_w):
    # load_webster_seed_set returns a dict with the GB code params + the
    # 4 seed operators (X_bar_1, Z_bar_1, X_bar_{k/2+1}, Z_bar_{k/2+1}).
    data = load_webster_seed_set(code_index)
    code = _build_generalised_bicycle_code(data['l'], data['A_set'], data['B_set'])
    x1 = np.asarray(data['operators']['X_bar_1'], dtype=np.uint8)
    x2 = np.asarray(data['operators']['X_bar_kh1'], dtype=np.uint8)
    g1 = build_gadget(code, x1)
    g2 = build_gadget(code, x2)
    assert len(g1.kappa_qubits) + int(g1.x.sum()) + g1.G.shape[0] == n_anc
    bridge = build_bridge(g1, g2)
    assert 2 * bridge.width - 1 == bridge_w
```

`load_webster_seed_set` (existing — currently in `layered.py`) and
`_build_generalised_bicycle_code` (existing) move to a small fixtures
module (e.g. `gadget.py` keeps the GB code builder; the JSON loader
either stays in `gadget.py` or moves to a `_fixtures.py`).

### C. CSS commutation + dimension invariants (math.md §1.5, §2.7, §2.8)

- `test_gadget_css_commutation` — H_X^merged · H_Z^merged.T == 0.
- `test_gadget_preserves_k` — dim(merged) == dim(data).
- `test_joint_css_commutation`.
- `test_joint_dimension_minus_one` — k_joint == k_data − 1.
- `test_joint_protocol_formula_alpha_star` — math.md §2.7.
- `test_joint_singleton_logicals_excluded` — math.md Table 4.

### D. Bridge correctness

- `test_bridge_telescoping` — Σ U_B rows == e_0 + e_{w−1}.
- `test_bridge_intercode_skiptree_invariant`.

### E. Stim circuit smoke + detector sanity

- `test_single_ppm_circuit_compiles`.
- `test_single_ppm_circuit_noiseless_no_detector_fires`.
- `test_single_ppm_circuit_with_noise_detectors_fire`.
- `test_joint_ppm_circuit_compiles`.
- `test_joint_ppm_circuit_noiseless_clean`.
- `test_joint_ppm_circuit_returns_merged_csscode` — third return is a
  valid CSSCode.

### F. Ide BB↔LP inter-code joint exact-match ground truth

```python
def test_intercode_joint_bb_lp_exact():
    # Loaders to be added to _ide_fixtures.py (do not exist yet):
    #   load_ide_BB_input_with_operator() → (CSSCode, x_bb)
    #   load_ide_LP_input_with_operator() → (CSSCode, x_lp)
    # These return the BB and LP INPUT codes (before joining) plus the
    # pinned logical-X operator that Ide §4.7 uses to define V_0^(1), V_0^(2).
    bb, x_bb = load_ide_BB_input_with_operator()
    lp, x_lp = load_ide_LP_input_with_operator()
    g1 = build_gadget(bb, x_bb)
    g2 = build_gadget(lp, x_lp)
    bridge = build_bridge(g1, g2)
    _, _, joint = build_joint_ppm_circuit(g1, g2, bridge, rounds=1)
    assert joint.num_qudits == 355
    assert joint.dimension == 25
    assert joint.get_distance() == 10
```

**Required new infrastructure** for this test:

1. `load_ide_BB_input_with_operator()` and
   `load_ide_LP_input_with_operator()` in `_ide_fixtures.py`. The
   Ide Zenodo bundle (`data_qLDPC_surgery.zip`) contains the input
   codes; the operator vector needs to be derived from the
   `IDE_BB_KAPPA1_EDGES` table (which encodes V_0^(1)) and the
   analogous LP table.
2. If our `build_bridge` cellulation does not match Ide's by default,
   a `cellulation_override=` kwarg on `build_bridge` is added so the
   test can feed Ide's published cellulation choice and still pass.

### G. Boost behavior

- `test_boost_method_spectral_monotone_lambda` — λ_2 non-decreasing.
- `test_boost_method_combinatorial_monotone`.
- `test_boost_method_distance_monotone`.
- `test_boost_preserves_css_commutation` — boosted gadget is valid CSS.

### Excluded from tests (policy)

Cain Table III reproductions (bb_18, lp_20, lp_24) are **not** used as
ground-truth tests because the paper does not pin the logical operator
— our choice is algorithmic. They remain in `examples/scripts/` for
reproduction work.

## 5. Risks

- **Webster Table I seed fixture compatibility** — `load_webster_seed_set`
  must continue to return inputs that produce the published numbers
  under the new deterministic `build_gadget`. If any non-determinism
  was hidden in the old `_compute_gauge_fix`, those numbers may shift.
  Mitigation: lock in the canonical row-reduced gauge-fix basis as the
  determinism contract; verify against Webster Table I before deleting
  the old code.
- **Ide BB↔LP exact-match** — requires the operator to be pinned in
  fixtures. `_ide_fixtures.py` has `IDE_BB_KAPPA1_EDGES` which suggests
  the operator IS pinned (κ_1 edges match Ide's cellulated G_1), but
  the loader may need a small extension to expose the operator vector
  explicitly. If our cellulation algorithm picks different cycles than
  Ide's, the exact-match test fails. Mitigation: if needed, fall back
  to feeding Ide's published cellulation choice to `build_bridge` via
  an optional `cellulation_override=` kwarg.
- **External script migration churn** — 16 scripts touched, all by
  rename. Risk is low (no semantic changes for single-PPM scripts);
  joint scripts (cain_fig1b_*) get a heavier rewrite to use the new
  circuit API.

## 6. Out of scope (explicitly)

- Multi-PPM / overlap PPM measurement.
- L ≥ 3 layered structure.
- New noise models.
- Decoder integration beyond what's already in `qldpc.decoders`.
- Inter-code cellulation algorithm changes.
