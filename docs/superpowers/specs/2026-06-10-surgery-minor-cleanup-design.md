# Surgery Module Minor Cleanup — Design

**Status**: Draft (2026-06-10, revised to include doc citation cleanup)
**Scope**: `src/qldpc/circuits/surgery/` — small mechanical edits + doc citation rewrites
**Author**: tgzhou (with Claude)

## Background

After the recent 4-commit surgery cleanup and the examples consolidation, a follow-up audit of the surgery module turned up:

1. Three small mechanical wins (one redundant test, two unused imports, one single-caller helper to inline).
2. A pervasive doc-style issue: docstrings and comments reference an internal `math.md` doc and use shorthand author surnames like `Cain §III.A` or `Webster §II.A` instead of full citations. Several existing arXiv IDs are also misattributed (e.g. `Cain et al. arXiv:2503.10390` — but arXiv:2503.10390 is actually He et al.'s *Extractors* paper).

This commit bundles both — they share the theme "tighten up surgery docstrings without changing behavior."

Larger consolidation candidates surfaced by the audit (single/joint helper dedup; `boost_gadget_cheeger_combinatorial` internal extraction) are out of scope here.

## Goal

Pay down three small code debts plus the doc-citation debt across the surgery module in one mechanical commit. No public API change, no algorithm change, no behavior change.

## Non-goals

- No public API change. The 7 public surface symbols (`build_gadget`, `build_bridge`, `build_single_ppm_circuit`, `build_joint_ppm_circuit`, `keep_only_observable`, `boost_gadget`, `cheeger_constant`) are bit-identical.
- No algorithm change.
- No new tests. We remove one redundant test that's strictly covered by another.
- No single/joint helper dedup in `circuit.py`. Deferred.
- No rewrite of inline math derivations — only paper citations (the labels in front of equations).
- No edits to non-surgery code, even if `math.md` is referenced elsewhere (out of scope).

## Canonical surgery-paper citations (user-authoritative)

Used throughout the doc-citation cleanup. From the user's authoritative mapping (2026-06-10):

| arXiv ID | Authors | Short cite (used in docstrings) |
|---|---|---|
| arXiv:2407.18393 | Cross, He, Rall, Yoder | `Cross et al. arXiv:2407.18393` |
| arXiv:2511.15989 | Webster, Smith, Cohen | `Webster, Smith, Cohen arXiv:2511.15989` |
| arXiv:2410.02213 | Williamson & Yoder | `Williamson & Yoder arXiv:2410.02213` |
| arXiv:2410.03628 | Swaroop, Jochym-O'Connor, Yoder | `Swaroop et al. arXiv:2410.03628` |
| arXiv:2503.10390 | He, Cowtan, Williamson, Yoder | `He et al. arXiv:2503.10390` |
| arXiv:2603.28627 | Cain et al. | `Cain et al. arXiv:2603.28627` |
| arXiv:1212.6703 | Kovalev & Pryadko | `Kovalev & Pryadko arXiv:1212.6703` |

## Part 1 — Three mechanical code edits

### Edit 1A — Delete redundant test `test_gadget_layout_has_basis_field`

`src/qldpc/circuits/surgery/_test_gadget.py:276-279`:

```python
def test_gadget_layout_has_basis_field():
    from qldpc.circuits.surgery.gadget import GadgetLayout
    fields = {f.name for f in dataclasses.fields(GadgetLayout)}
    assert "basis" in fields, f"basis field missing; got {fields}"
```

Already strictly covered by `test_gadget_layout_is_frozen_dataclass` (line 24-40) which asserts the full field-set equality `fields == {..., "basis"}`. Delete the 4 lines.

### Edit 1B — Drop unused Webster helper imports in `_test_gadget.py`

Edit `_test_gadget.py:13-19` from:

```python
from ._test_helpers import (
    load_webster_seed_set,
    build_generalised_bicycle_code,
    _webster_x_bar_operator,
    _webster_z_bar_operator,
    _webster_x_bar_1_operator,
)
```

to:

```python
from ._test_helpers import (
    load_webster_seed_set,
    build_generalised_bicycle_code,
    _webster_x_bar_1_operator,
)
```

`_webster_x_bar_operator` and `_webster_z_bar_operator` are never referenced in this file (only `_webster_x_bar_1_operator` at line 265).

### Edit 1C — Inline `_label_inverse` into `_skip_tree_fullrank`

In `bridge.py`, delete the `_label_inverse` function (lines 105-114). In `_skip_tree_fullrank` at line 136, replace:

```python
    label = _label_inverse(P)
```

with the inlined body:

```python
    # P is a permutation matrix from _skip_tree; recover label[l] = v such that P[v, l] = 1.
    label = [-1] * n
    for v in range(n):
        for l in range(n):
            if P[v, l] == 1:
                label[l] = v
                break
```

## Part 2 — Doc citation cleanup

All `math.md` references → paper section references. All shorthand author surnames → full citation. Existing misattributions corrected per the canonical mapping above.

### Edit 2A — `gadget.py` module + step docstrings

The module is a Webster-style L=1 gadget construction. All `math.md §1.X` references map to Webster, Smith, Cohen arXiv:2511.15989 §II.A subsections.

Specific edits:

- **Line 1** (module docstring): `"""L=1 Webster gadget construction (see math.md §1, spec §2)."""` → `"""L=1 gadget construction (Webster, Smith, Cohen arXiv:2511.15989 §II.A)."""`
- **Lines 4-6** (step list in module docstring):
  ```
  _step1_restriction  — math.md §1.1
  _step2_gauge_fix    — math.md §1.2
  _step3_assemble     — math.md §1.4
  ```
  →
  ```
  _step1_restriction  — Webster §II.A step 1 (restriction)
  _step2_gauge_fix    — Webster §II.A step 2 (gauge fix)
  _step3_assemble     — Webster §II.A step 3 (block assembly)
  ```
- **Line 39** (`_step1_restriction` docstring): `"""math.md §1.1 — V_0 = supp(x); ..."""` → `"""Webster §II.A step 1 — V_0 = supp(x); C_0 = checks touching V_0; F = H_complement[C_0, V_0]."""`
- **Line 66** (`_step2_gauge_fix` docstring): `"""math.md §1.2 — G whose rows form a canonical basis of ker(F.T) over GF(2)."""` → `"""Webster §II.A step 2 — G whose rows form a canonical basis of ker(F.T) over GF(2)."""`
- **Line 115** (`_step3_assemble` docstring): `"""math.md §1.4 — block assembly of HX_merged, HZ_merged."""` → `"""Webster §II.A step 3 — block assembly of HX_merged, HZ_merged."""`
- **Line 160** (`build_gadget` docstring): `"""Webster L=1 gadget = steps 1+2+3 composed. ..."""` — already says "Webster", but the module docstring will now provide the arXiv. No edit needed if module docstring is correct.

The convention: the **module docstring** establishes the full citation `Webster, Smith, Cohen arXiv:2511.15989 §II.A` once. **Function docstrings** within the same file can then use the short form `Webster §II.A step N`.

### Edit 2B — `cheeger.py` citation fixes

Module-level: there is no module docstring currently. Add one establishing the relevant papers, then per-function docstrings can use short forms.

Specific edits:

- **Line 1** (top of file): currently `"""Cheeger and distance boost transformations for surgery gadgets."""`. Replace with:
  ```
  """Cheeger and distance boost transformations for surgery gadgets.

  References:
      Webster, Smith, Cohen arXiv:2511.15989  — boundary Cheeger constant,
          combinatorial boost.
      Cross et al. arXiv:2407.18393  — Cheeger-based distance preservation
          (§III Thm 6).
      Williamson & Yoder arXiv:2410.02213  — distance-verifying random
          augmentation boost.
  """
  ```
- **Line 13** (`_exact_boundary_cheeger` docstring): `"""Exact boundary Cheeger constant of F per Webster §II.A Definition 1."""` → `"""Exact boundary Cheeger constant of F per Webster §II.A Definition 1 (arXiv:2511.15989)."""` — actually leave the inline arXiv off since module docstring has it; just keep `Webster §II.A Definition 1`. No edit.
- **Line 16** (docstring continuation): `(which follows Williamson-Yoder / Webster: random edge addition + distance ...)` → `(which follows Williamson & Yoder arXiv:2410.02213 / Webster, Smith, Cohen arXiv:2511.15989: random edge addition + distance ...)` — first occurrence in file, give full cite even though module docstring covers Webster, since Williamson & Yoder is new here.

Actually, simpler convention: once full cites are in the module docstring, function docstrings use short cite (surname + section). So:

- **Line 16**: `Williamson-Yoder / Webster` → `Williamson & Yoder / Webster, Smith, Cohen` (short forms; arXivs are in module docstring).
- **Line 106** (`cheeger_constant` docstring): `Webster §II.A Def 1` — already short form, no edit needed.
- **Line 112** (`cheeger_constant` docstring): `Webster Lemma 9` — already short form, no edit.
- **Line 180** (`boost_gadget_cheeger_combinatorial` docstring): `Webster Def 1 / Cross Def 3` — already short form, no edit. (Cross is now in module docstring.)
- **Line 185**: `By Cross §III Thm 6` — short form, no edit.
- **Line 187**: `Webster's family up to l=255` — code-family label, not a citation, no edit.
- **Line 191**: `Cross Thm 6 threshold` — short form, no edit.
- **Line 318** (`boost_gadget_distance` docstring): `"""Williamson-Yoder / Webster distance-verifying gadget boost."""` → `"""Williamson & Yoder / Webster distance-verifying gadget boost (arXiv:2410.02213 / arXiv:2511.15989)."""` — keep full cite here because this function has heavy paper dependence.
- **Line 320** (`boost_gadget_distance` docstring): `Per Cain et al. arXiv:2503.10390 / Webster` → `Per Williamson & Yoder arXiv:2410.02213 / Webster, Smith, Cohen arXiv:2511.15989`. **This is a misattribution fix**: arXiv:2503.10390 is He et al. (Extractors), not Cain et al. The distance-verifying boost protocol described here is from Williamson & Yoder.

### Edit 2C — `circuit.py` citation fixes

- **Line 1** (no module docstring currently): add a one-line module docstring with paper anchors:
  ```python
  """Surgery measurement circuit assembly (Cain et al. arXiv:2603.28627 §III.A protocol)."""
  ```
- **Line 54** (`build_single_ppm_circuit` docstring): `"""Cain §III.A single-PPM measurement circuit for `gadget`."""` — already short form. Keep.
- **Line 569** (`_surgery_observable` docstring): `"""Obs 0 = ⊕ chi-XOR over rounds (Webster Eq. 1); Obs 1 = data on V_0."""` — first Webster ref in the file. Replace with `Webster, Smith, Cohen Eq. 1 (arXiv:2511.15989)` OR add `Webster, Smith, Cohen arXiv:2511.15989` to module docstring and keep short form. Choose the latter for consistency: update module docstring to list both Cain + Webster:
  ```python
  """Surgery measurement circuit assembly.

  References:
      Cain et al. arXiv:2603.28627 §III.A — single-PPM measurement protocol.
      Webster, Smith, Cohen arXiv:2511.15989 — gadget Eq. 1 observable.
  """
  ```
  Then line 569 stays as `Webster Eq. 1`. No further edit at line 569.
- **Line 634** (`_surgery_detach_and_readout` docstring): `"""Cain step 3 + final data measure. Mκ then SHIFT_COORDS then Mdata."""` — short form. Keep.

### Edit 2D — `bridge.py` citation fixes

- **Line 1** (module docstring): `"""Standalone bridge adapter for two-PPM joint surgery (arXiv:2410.03628 §IV / §VII)."""` → `"""Standalone bridge adapter for two-PPM joint surgery (Swaroop et al. arXiv:2410.03628 §IV / §VII)."""`
- **Line 20** (`Bridge` class docstring): `"""Universal adapter between two GadgetLayouts (arXiv:2410.03628 §IV / §VII)."""` — already has the arXiv, just add authors: → `"""Universal adapter between two GadgetLayouts (Swaroop et al. arXiv:2410.03628 §IV / §VII)."""`
- **Line 44** (`_skip_tree` docstring): `"""SkipTree basis transform (arXiv:2410.03628 §III). Returns T, P."""` → `"""SkipTree basis transform (Swaroop et al. arXiv:2410.03628 §III). Returns T, P."""`
- **Line 161** (`_cellulate_port_subgraph` docstring): `Theorem 7 (arXiv:2410.03628) already bounds T_s row weight at ≤ 3` → `Theorem 7 (Swaroop et al. arXiv:2410.03628) already bounds T_s row weight at ≤ 3`
- **Line 350** (`build_bridge` docstring): `"""Universal-adapter bridge between two gadgets (arXiv:2410.03628 §IV)."""` → `"""Universal-adapter bridge between two gadgets (Swaroop et al. arXiv:2410.03628 §IV)."""`

### Edit 2E — `_test_gadget.py` math.md comment fixes

- **Line 60**: `# F @ 1_{V0} == 0 (math.md §1.1 invariant)` → `# F @ 1_{V0} == 0 (Webster §II.A step 1 invariant)`
- **Line 71**: `# math.md §1.2: G F = 0 over GF(2)` → `# Webster §II.A step 2: G F = 0 over GF(2)`
- **Line 127**: `# math.md §1.5(a): H_X^merged @ H_Z^merged.T == 0 over GF(2)` → `# Webster §II.A: H_X^merged @ H_Z^merged.T == 0 over GF(2) (CSS commutation)` — the `§1.5(a)` was math.md-specific; replace with a self-describing label.
- **Line 305**: `# math.md §1.1 invariant: F @ 1_{V0} = 0 (since H_X @ z = 0 for a logical Z)` → `# Webster §II.A step 1 invariant: F @ 1_{V0} = 0 (since H_X @ z = 0 for a logical Z)`

These are inline test comments, not docstrings — full arXiv ID is overkill; `Webster §II.A step N` is unambiguous in surgery test context.

## Approach summary

One commit titled `chore(surgery): drop dead code + fix doc citations`. All edits mechanical, no behavior change, no test changes other than the deletion of the redundant test.

## Net effect

| Aspect | Δ |
|---|---|
| `_test_gadget.py` LOC | −6 (4-line test + 2 unused imports) + 4 inline-comment rewrites |
| `bridge.py` LOC | −6 (`_label_inverse` inline) + 5 citation rewrites |
| `cheeger.py` LOC | +7 (replace 1-line module docstring with 7-line block) + several citation edits |
| `gadget.py` LOC | ±0 (5 in-place citation rewrites) |
| `circuit.py` LOC | +6 (add multi-line module docstring with 2-paper anchor) |
| Surgery test count | −1 (the deleted redundant test) |
| `grep -rn "math\.md" src/qldpc/circuits/surgery/` after | empty |
| `grep -rn "_label_inverse" src/qldpc/circuits/surgery/` after | empty |
| `grep -rn "Cain et al. arXiv:2503.10390" src/qldpc/circuits/surgery/` after | empty (misattribution fixed) |

## Success criteria

- `pytest src/qldpc/circuits/surgery/ -q` reports **88 passed** (one fewer than before).
- `grep -rn "math\.md" src/qldpc/circuits/surgery/` returns empty.
- `grep -n "_label_inverse" src/qldpc/circuits/surgery/bridge.py` returns empty.
- `grep -n "_webster_x_bar_operator\b\|_webster_z_bar_operator\b" src/qldpc/circuits/surgery/_test_gadget.py` returns empty.
- `grep -rn "Cain et al. arXiv:2503.10390" src/qldpc/circuits/surgery/` returns empty (misattribution fixed).
- Python import smoke test: `python -c "from qldpc.circuits.surgery import build_gadget, build_bridge, build_single_ppm_circuit, build_joint_ppm_circuit, keep_only_observable, boost_gadget, cheeger_constant"` succeeds.

## Open questions

None — citation mapping is user-authoritative; all edits have a single correct form shown above.
