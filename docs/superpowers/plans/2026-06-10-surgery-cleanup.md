# Surgery Module Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pay down four debts in `src/qldpc/circuits/surgery/` (legacy `SurgeryLayout` adapter, unused 'spectral' boost strategy, monolithic `_test.py`, 4 stitch helpers across the basis duality) in one PR with four atomic commits, no public-API change.

**Architecture:** Each commit is independently reviewable and testable. Test split first to narrow blast radius; pure deletion second; substantive refactor third (boost paths consume `GadgetLayout` natively via the existing `build_gadget_augmented`); stitch dedup last (independent of the cheeger work).

**Tech Stack:** Python 3, `numpy`, `galois` (GF(2)), `pytest`, `stim`. Existing dependencies — nothing new.

**Spec:** `docs/superpowers/specs/2026-06-10-surgery-cleanup-design.md`

---

## File Structure (after all four commits)

```
src/qldpc/circuits/surgery/
├── __init__.py            (unchanged)
├── gadget.py              (unchanged)
├── bridge.py              (unchanged)
├── cheeger.py             (-290 LOC: drop SurgeryLayout, 2 Result classes,
│                          _build_layout, _compute_gauge_fix,
│                          _reassemble_gadget_with_new_F, _gadget_to_legacy_layout,
│                          _legacy_to_gadget, boost_gadget_cheeger)
├── circuit.py             (-76 LOC: collapse 4 stitch helpers to 2)
├── _test_helpers.py       (NEW: ~30 LOC — sys.path injection + Webster operators)
├── _test_gadget.py        (NEW: ~280 LOC — 21 tests on gadget.py)
├── _test_bridge.py        (NEW: ~440 LOC — 18 tests on bridge.py)
├── _test_circuit.py       (NEW: ~900 LOC — 30 tests on circuit.py)
└── _test_cheeger.py       (NEW: ~85 LOC — 5 cheeger tests + 1 basis=Z safety-net)
```

`_test.py` is deleted by commit 1.

---

## Task 1: Commit 1 — Split `_test.py` by feature

**Files:**
- Create: `src/qldpc/circuits/surgery/_test_helpers.py`
- Create: `src/qldpc/circuits/surgery/_test_gadget.py`
- Create: `src/qldpc/circuits/surgery/_test_bridge.py`
- Create: `src/qldpc/circuits/surgery/_test_circuit.py`
- Create: `src/qldpc/circuits/surgery/_test_cheeger.py`
- Delete: `src/qldpc/circuits/surgery/_test.py`

- [ ] **Step 1: Snapshot test IDs before split**

Run:
```bash
cd /Users/tgzhou/Project/qLDPC && \
  pytest src/qldpc/circuits/surgery/_test.py --collect-only -q 2>&1 \
  | grep "::test_" | sort > /tmp/surgery_tests_before.txt && \
  wc -l /tmp/surgery_tests_before.txt
```

Expected: a number around 74 (test functions in `_test.py`; parametrized tests show one line per case, so the count may be higher than 74).

- [ ] **Step 2: Create `_test_helpers.py` with the shared bits**

Write `src/qldpc/circuits/surgery/_test_helpers.py`:

```python
"""Shared fixtures + helpers for the split surgery test files.

Importing this module injects `<repo>/examples` onto sys.path so the
Webster JSON seed-set loader is available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Webster seed-set helpers live under examples/ (the JSON fixture is there too).
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "examples"))
from _webster_seed_set import (  # noqa: E402
    load_webster_seed_set,
    build_generalised_bicycle_code,
)


def _webster_x_bar_operator(
    data: dict, name: str = "X_bar_1", pauli_type: str = "X",
) -> np.ndarray:
    """Extract the named logical operator from a Webster seed_set dict.

    L_support and R_support are sparse index lists (positions within each l-half
    that are set to 1). Returns a dense binary vector of length 2l.

    Args:
        data: Webster seed set dict (from load_webster_seed_set).
        name: Seed name, e.g. "X_bar_1", "Z_bar_1".
        pauli_type: "X" or "Z"; filters seeds by pauli_type field.
    """
    l = data["l"]
    for seed in data["seeds"]:
        if seed["name"] == name and seed["pauli_type"] == pauli_type:
            v_L = np.zeros(l, dtype=np.uint8)
            v_L[seed["L_support"]] = 1
            v_R = np.zeros(l, dtype=np.uint8)
            v_R[seed["R_support"]] = 1
            return np.concatenate([v_L, v_R])
    raise ValueError(f"{name!r} (pauli_type={pauli_type!r}) seed not found")


def _webster_z_bar_operator(data: dict, name: str = "Z_bar_1") -> np.ndarray:
    """Extract the named Z-type logical operator from a Webster seed_set dict.

    Convenience wrapper around _webster_x_bar_operator with pauli_type="Z".
    """
    return _webster_x_bar_operator(data, name, pauli_type="Z")


def _webster_x_bar_1_operator(data: dict) -> np.ndarray:
    """Back-compat: returns X_bar_1; prefer _webster_x_bar_operator."""
    return _webster_x_bar_operator(data, "X_bar_1")
```

- [ ] **Step 3: Create `_test_gadget.py` with the 21 gadget tests**

Write `src/qldpc/circuits/surgery/_test_gadget.py`. Header:

```python
"""Tests for src/qldpc/circuits/surgery/gadget.py."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from qldpc import codes
from qldpc.objects import Pauli

from ._test_helpers import (
    load_webster_seed_set,
    build_generalised_bicycle_code,
    _webster_x_bar_operator,
    _webster_z_bar_operator,
    _webster_x_bar_1_operator,
)
```

Then COPY VERBATIM these 21 functions from `_test.py` into this file (find each by name with grep, copy the whole `def ... :` block including all internal imports):

1. `test_gadget_layout_is_frozen_dataclass`
2. `test_step1_restriction_steane`
3. `test_step2_gauge_fix_basis_property`
4. `test_step2_gauge_fix_deterministic`
5. `test_step3_assemble_basis_z_places_chi_in_HZ_merged_and_G_in_HX_merged`
6. `test_step3_assemble_steane_css_commutes`
7. `test_step3_assemble_csscode_with_distinct_nV_nC`
8. `test_build_gadget_steane_returns_valid_layout`
9. `test_build_gadget_deterministic`
10. `test_build_gadget_rejects_non_x_logical`
11. `test_load_webster_seed_set_returns_known_shape`
12. `test_build_generalised_bicycle_code_constructs_css`
13. `test_webster_table_i_kappa_chi_r_exact`
14. `test_gadget_layout_has_basis_field`
15. `test_gadget_layout_basis_defaults_to_x_via_build_gadget`
16. `test_step1_restriction_basis_z_uses_HX`
17. `test_build_gadget_z_basis_css_commutation`
18. `test_build_gadget_z_basis_rejects_non_z_logical`
19. `test_build_gadget_z_basis_dual_matches_x_basis_on_dual_code`
20. `test_webster_table_i_z_basis_kappa_chi_r_exact`
21. `test_build_gadget_augmented_extends_F_and_recomputes_G`

Inside any function that has `from qldpc.circuits.surgery.gadget import ...`, leave that import unchanged — the in-function import pattern is the existing style.

- [ ] **Step 4: Create `_test_bridge.py` with the 18 bridge tests**

Write `src/qldpc/circuits/surgery/_test_bridge.py`. Header:

```python
"""Tests for src/qldpc/circuits/surgery/bridge.py."""

from __future__ import annotations

import numpy as np
import pytest

from qldpc import codes
from qldpc.objects import Pauli

from ._test_helpers import (
    load_webster_seed_set,
    build_generalised_bicycle_code,
    _webster_x_bar_1_operator,
)
```

Copy verbatim these 18 functions from `_test.py`:

1. `test_skip_tree_fullrank_on_K4_matches_H_R`
2. `test_build_aux_graph_weight2_rows_become_edges`
3. `test_build_aux_graph_filters_hyperedges`
4. `test_build_aux_graph_rejects_weight1_row`
5. `test_connect_induced_subgraph_no_op_when_connected`
6. `test_connect_induced_subgraph_adds_edges_to_disconnected_components`
7. `test_cellulate_caps_cycle_length`
8. `test_cellulate_no_op_when_already_short`
9. `test_cellulate_raises_when_port_cycle_has_no_available_chord`
10. `test_cellulate_port_subgraph_breaks_long_port_cycle`
11. `test_cellulate_port_subgraph_skips_non_port_cycle`
12. `test_bridge_dataclass_fields_universal_adapter`
13. `test_build_bridge_smoke_steane_intracode`
14. `test_build_bridge_skiptree_invariant_holds`
15. `test_build_bridge_rejects_basis_mismatch`
16. `test_build_bridge_bb18_hyperedge_and_long_cycle`
17. `test_adapter_cycle_check_weight_bounded`
18. `test_cellulation_caps_aug_aux_cycle_length_on_webster`

- [ ] **Step 5: Create `_test_circuit.py` with the 30 circuit tests**

Write `src/qldpc/circuits/surgery/_test_circuit.py`. Header:

```python
"""Tests for src/qldpc/circuits/surgery/circuit.py (single + joint PPM)."""

from __future__ import annotations

import numpy as np
import pytest
import stim

from qldpc import codes
from qldpc.objects import Pauli

from ._test_helpers import (
    load_webster_seed_set,
    build_generalised_bicycle_code,
    _webster_x_bar_1_operator,
)
```

Copy verbatim these 30 functions from `_test.py`:

1. `test_build_single_ppm_circuit_noiseless_compiles`
2. `test_build_single_ppm_circuit_noiseless_no_detectors_fire`
3. `test_build_single_ppm_circuit_with_noise_detectors_fire`
4. `test_classify_reliable_round1_checks_basis_x`
5. `test_classify_reliable_round1_checks_basis_z`
6. `test_surgery_state_prep_basis_x_resets`
7. `test_surgery_state_prep_basis_z_resets`
8. `test_surgery_qec_cycle_round_1_detectors_classified`
9. `test_surgery_detach_and_readout_basis_x_measures_kappa_then_data`
10. `test_surgery_detach_and_readout_basis_z_measures_kappa_in_x_then_data_in_z`
11. `test_surgery_observable_emits_two_observable_include`
12. `test_build_single_ppm_circuit_noiseless_observables_zero`
13. `test_single_ppm_circuit_noise_flips_observable_at_high_p`
14. `test_surgery_final_detectors_count_matches_reliable_round1`
15. `test_build_single_ppm_circuit_noiseless_no_detector_fires`
16. `test_single_ppm_ler_monotone_in_p`
17. `test_single_ppm_ler_with_final_detectors_below_threshold`
18. `test_stitch_intercode_basis_x_css_commutation`
19. `test_stitch_intercode_basis_x_k_reduces_by_one`
20. `test_stitch_intercode_basis_x_joint_logical_in_stabilizer`
21. `test_stitch_intercode_basis_x_singletons_excluded`
22. `test_stitch_intracode_basis_x_css_commutation`
23. `test_stitch_intracode_basis_x_k_reduces_by_one`
24. `test_stitch_intercode_both_bases_commute_and_singletons_excluded`
25. `test_stitch_intracode_both_bases_commute`
26. `test_build_joint_ppm_circuit_chi_check_ids_no_UB`
27. `test_build_joint_ppm_circuit_intercode_noiseless_observables_zero`
28. `test_joint_ppm_ler_monotone_steane_intercode`
29. `test_joint_xx_in_stabilizer_on_webster_intracode`
30. `test_build_joint_ppm_circuit_intracode_noiseless_observables_zero`

If any function uses `import stim` or `from stim import ...` inline, leave those imports in place.

- [ ] **Step 6: Create `_test_cheeger.py` with the 5 cheeger tests**

Write `src/qldpc/circuits/surgery/_test_cheeger.py`. Header:

```python
"""Tests for src/qldpc/circuits/surgery/cheeger.py (cheeger_constant + boost_gadget)."""

from __future__ import annotations

import numpy as np
import pytest

from qldpc import codes
from qldpc.objects import Pauli

from ._test_helpers import (
    load_webster_seed_set,
    build_generalised_bicycle_code,
    _webster_x_bar_1_operator,
)
```

Copy verbatim these 5 functions from `_test.py`:

1. `test_cheeger_constant_matches_boost_target`
2. `test_boost_gadget_dispatches_to_three_methods`
3. `test_boost_gadget_seed_reproducible`
4. `test_boost_gadget_preserves_css_commutation`
5. `test_boost_gadget_preserves_css_commutation_both_bases`

- [ ] **Step 7: Delete `_test.py`**

Run:
```bash
git rm src/qldpc/circuits/surgery/_test.py
```

- [ ] **Step 8: Verify test ID set is unchanged**

Run:
```bash
cd /Users/tgzhou/Project/qLDPC && \
  pytest src/qldpc/circuits/surgery/ --collect-only -q 2>&1 \
  | grep "::test_" | sort > /tmp/surgery_tests_after.txt && \
  diff <(awk -F'::' '{print $NF}' /tmp/surgery_tests_before.txt) \
       <(awk -F'::' '{print $NF}' /tmp/surgery_tests_after.txt)
```

Expected: empty diff (test names match modulo their file path).

- [ ] **Step 9: Run the full surgery suite**

Run: `pytest src/qldpc/circuits/surgery/ -v`

Expected: all tests pass with the same outcomes as before (count should match `/tmp/surgery_tests_before.txt`).

- [ ] **Step 10: Commit**

```bash
git add src/qldpc/circuits/surgery/_test_helpers.py \
        src/qldpc/circuits/surgery/_test_gadget.py \
        src/qldpc/circuits/surgery/_test_bridge.py \
        src/qldpc/circuits/surgery/_test_circuit.py \
        src/qldpc/circuits/surgery/_test_cheeger.py
git commit -m "refactor(surgery/tests): split _test.py by feature"
```

(`git rm` from Step 7 is already staged.)

---

## Task 2: Commit 2 — Drop unused 'spectral' boost strategy

**Files:**
- Modify: `src/qldpc/circuits/surgery/cheeger.py`
- Modify: `src/qldpc/circuits/surgery/_test_cheeger.py`

- [ ] **Step 1: Delete `boost_gadget_cheeger` (~80 LOC)**

In `src/qldpc/circuits/surgery/cheeger.py`, delete the entire function `boost_gadget_cheeger` (currently starts at line 198 with `def boost_gadget_cheeger(`, ends with `return boosted_merged, boosted_layout, BoostResult(...)` and the closing parenthesis).

Do NOT delete `_spectral_cheeger_lower_bound` — it stays as the `|V_0| > 26` fallback for `cheeger_constant`.

- [ ] **Step 2: Remove 'spectral' branch from `boost_gadget` dispatcher**

In `boost_gadget` (currently at line 774 of `cheeger.py`), edit the body to remove the `if method == "spectral":` branch and update the error message. The new dispatcher body (between the docstring and `return _legacy_to_gadget(...)`):

```python
    merged0, layout0 = _gadget_to_legacy_layout(gadget)
    if method == "combinatorial":
        boosted_merged, boosted_layout, _ = boost_gadget_cheeger_combinatorial(
            merged0, layout0, target_h=target, seed=seed, **kwargs,
        )
    elif method == "distance":
        boosted_merged, boosted_layout, _ = boost_gadget_distance(
            merged0, layout0, target_distance=int(target), seed=seed, **kwargs,
        )
    else:
        raise ValueError(
            f"unknown method: {method!r}. Allowed: 'combinatorial', 'distance'."
        )
    return _legacy_to_gadget(boosted_merged, boosted_layout, gadget)
```

(`_gadget_to_legacy_layout` and `_legacy_to_gadget` are kept here — they go away in Task 3.)

- [ ] **Step 3: Update docstring on `boost_gadget`**

Inside the docstring of `boost_gadget`, change the `method` description from:

```
method: 'spectral' | 'combinatorial' | 'distance'.
```

to:

```
method: 'combinatorial' | 'distance'.
```

- [ ] **Step 4: Update `_test_cheeger.py` — drop 'spectral' parametrize values**

In `src/qldpc/circuits/surgery/_test_cheeger.py`, make three edits:

(a) Rename `test_boost_gadget_dispatches_to_three_methods` → `test_boost_gadget_dispatches_to_two_methods`. Inside the body, change the loop:

```python
for method in ("spectral", "combinatorial", "distance"):
```

to:

```python
for method in ("combinatorial", "distance"):
```

(b) In `test_boost_gadget_seed_reproducible`, change both occurrences of `method="spectral"` to `method="combinatorial"`. (The combinatorial boost is also deterministic given seed; this test still validates determinism.)

(c) In `test_boost_gadget_preserves_css_commutation`, change the `@pytest.mark.parametrize` value list from:

```python
@pytest.mark.parametrize("method", ["spectral", "combinatorial", "distance"])
```

to:

```python
@pytest.mark.parametrize("method", ["combinatorial", "distance"])
```

`test_boost_gadget_preserves_css_commutation_both_bases` already only exercises combinatorial — no change.

- [ ] **Step 5: Run the cheeger tests**

Run: `pytest src/qldpc/circuits/surgery/_test_cheeger.py -v`

Expected: all 4 (renamed `_two_methods`) + 2 basis variants pass. No `spectral` test cases.

- [ ] **Step 6: Run the full surgery suite as a sanity check**

Run: `pytest src/qldpc/circuits/surgery/`

Expected: green. No other test depends on `boost_gadget_cheeger` or `method="spectral"`.

- [ ] **Step 7: Verify 'spectral' is gone from `cheeger.py` (helper retained)**

Run:
```bash
grep -n "spectral" src/qldpc/circuits/surgery/cheeger.py
```

Expected: 2-3 matches — only the `_spectral_cheeger_lower_bound` definition and its references inside `cheeger_constant`. NO `boost_gadget_cheeger` function definition, NO `method == "spectral"` branch.

- [ ] **Step 8: Commit**

```bash
git add src/qldpc/circuits/surgery/cheeger.py src/qldpc/circuits/surgery/_test_cheeger.py
git commit -m "refactor(surgery/cheeger): drop unused 'spectral' boost strategy"
```

---

## Task 3: Commit 3 — Boost paths consume `GadgetLayout` natively

**Files:**
- Modify: `src/qldpc/circuits/surgery/cheeger.py`
- Modify: `src/qldpc/circuits/surgery/_test_cheeger.py`

This is the substantive commit. Strategy:
1. Add safety-net tests first (they pass under the current code — a sanity check that nothing was broken in commit 2).
2. Rewrite `boost_gadget_cheeger_combinatorial` and `boost_gadget_distance` with new signatures that take/return `GadgetLayout` directly, using `build_gadget_augmented` as the rebuild path.
3. Rewrite `boost_gadget` dispatcher to call the new signatures with no translation layer.
4. Delete all the now-unused symbols (8 of them).
5. Run everything; one atomic commit.

- [ ] **Step 1: Add 1 safety-net test to `_test_cheeger.py`**

Append this test to `src/qldpc/circuits/surgery/_test_cheeger.py`:

```python
def test_boost_gadget_combinatorial_basis_z_preserves_chi_carrier():
    """After basis=Z combinatorial boost, χ rows must live in HZ_merged.

    The legacy adapter handled basis=Z by swapping HX↔HZ on entry and back on
    exit; the GadgetLayout-native path delegates basis routing to
    build_gadget_augmented. This test catches a regression where χ rows end
    up in HX_merged instead of HZ_merged.

    Distance-strategy basis=Z is not tested here because the Webster JSON
    fixture only ships X̄ operators; the basis=X path of distance boost is
    covered by test_boost_gadget_preserves_css_commutation[distance].
    """
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.cheeger import boost_gadget

    code = codes.SteaneCode()
    z_op = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z_op, basis=Pauli.Z)

    boosted = boost_gadget(g, method="combinatorial", target=1.0, seed=42)

    assert boosted.basis is Pauli.Z, (
        f"basis dropped through boost: got {boosted.basis!r}, expected Pauli.Z"
    )
    n_chi = len(boosted.V0)
    n_z_data = code.matrix_z.shape[0]
    chi_block = boosted.HZ_merged[n_z_data : n_z_data + n_chi, :]
    assert chi_block.any(), (
        "χ rows missing from HZ_merged; basis=Z boost path likely swapped HX/HZ."
    )
```

- [ ] **Step 2: Verify the safety-net test passes under current (legacy-adapter) code**

Run: `pytest src/qldpc/circuits/surgery/_test_cheeger.py::test_boost_gadget_combinatorial_basis_z_preserves_chi_carrier -v`

Expected: PASS. (The current legacy adapter correctly handles basis=Z via HX↔HZ swap; the test confirms the invariant we want preserved across the refactor.)

- [ ] **Step 3: Rewrite `boost_gadget_cheeger_combinatorial` with `GadgetLayout` signature**

In `src/qldpc/circuits/surgery/cheeger.py`, replace the entire `boost_gadget_cheeger_combinatorial` function (currently at line 417, ending around line 582) with this version:

```python
def boost_gadget_cheeger_combinatorial(
    g: GadgetLayout,
    *,
    target_h: float = 1.0,
    max_extra_qubits: int = 50,
    seed: int | None = None,
) -> GadgetLayout:
    """Greedy combinatorial Cheeger boost — deterministic distance guarantee.

    Computes the exact boundary Cheeger constant h(F) via subset enumeration
    (Webster Def 1 / Cross Def 3). When h < target_h, identifies the worst
    cut v* and adds a κ qubit (degree-2 row of F) with one endpoint in v*
    and one outside, which monotonically increases |∂v*| by 1 without
    decreasing any other |∂v|.

    By Cross §III Thm 6, h(F) >= 1 implies d_merged >= d_data, so reaching
    target_h = 1.0 GUARANTEES distance preservation. Tractable for
    |V_0| <= 26 (Webster's family up to l=255).

    Args:
        g: input gadget produced by build_gadget.
        target_h: Cheeger target. Default 1.0 (Cross Thm 6 threshold).
        max_extra_qubits: cap on additions. Default 50.
        seed: RNG seed for tie-breaking in edge selection.

    Returns:
        A new GadgetLayout with F augmented to reach target_h, rebuilt via
        build_gadget_augmented (basis=X/Z handled symmetrically).

    Raises:
        ValueError: |V_0| > 26 (enumeration infeasible) or target_h <= 0.
    """
    from .gadget import build_gadget_augmented

    if target_h <= 0:
        raise ValueError(f"target_h must be positive, got {target_h}.")
    if max_extra_qubits < 0:
        raise ValueError(f"max_extra_qubits must be >= 0, got {max_extra_qubits}.")

    rng = np.random.default_rng(seed)
    field = galois.GF(2)
    F = np.asarray(g.F).astype(np.int_).copy()
    n_orig_rows = F.shape[0]
    n_V = F.shape[1]
    if n_V > 26:
        raise ValueError(
            f"|V_0| = {n_V} > 26; exact Cheeger enumeration infeasible. "
            f"Use boost_gadget_distance (BP+OSD) instead."
        )
    if n_V < 2:
        # Nothing to boost; return identity GadgetLayout (no F_extra rows).
        return build_gadget_augmented(
            g.code, g.x, np.zeros((0, n_V), dtype=np.uint8), basis=g.basis,
        )

    half = n_V // 2
    F_col_ints = [
        int.from_bytes(
            np.packbits(F[:, i][::-1]).tobytes()[::-1], "little"
        ) for i in range(n_V)
    ]
    total = 1 << n_V
    masks_buf: list[int] = []
    sizes_buf: list[int] = []
    cuts_buf: list[int] = []
    boundary_int = 0
    subset_mask = 0
    for k in range(1, total):
        bit = (k & -k).bit_length() - 1
        subset_mask ^= 1 << bit
        boundary_int ^= F_col_ints[bit]
        size = subset_mask.bit_count()
        if 1 <= size <= half:
            masks_buf.append(subset_mask)
            sizes_buf.append(size)
            cuts_buf.append(boundary_int.bit_count())

    masks = np.array(masks_buf, dtype=np.uint64)
    sizes = np.array(sizes_buf, dtype=np.int32)
    cuts = np.array(cuts_buf, dtype=np.int32)

    def _existing_pairs(arr: np.ndarray) -> set[tuple[int, int]]:
        pairs: set[tuple[int, int]] = set()
        for row in arr:
            ones = np.flatnonzero(row)
            for a in range(len(ones)):
                for b in range(a + 1, len(ones)):
                    pairs.add((int(ones[a]), int(ones[b])))
        return pairs

    extra = 0
    while True:
        h_num = cuts.astype(np.int64)
        h_den = sizes.astype(np.int64)
        idx = int(np.argmin(h_num / h_den))
        h = float(h_num[idx] / h_den[idx])
        worst_mask = int(masks[idx])

        if h >= target_h:
            break
        if extra >= max_extra_qubits:
            break

        v_star_arr = np.array(
            [(worst_mask >> i) & 1 for i in range(n_V)], dtype=np.int8
        )
        inside = np.flatnonzero(v_star_arr).tolist()
        outside = np.flatnonzero(1 - v_star_arr).tolist()
        if not inside or not outside:
            break

        rng.shuffle(inside)
        rng.shuffle(outside)
        pairs = _existing_pairs(F)
        chosen = None
        for i in inside:
            for j in outside:
                a, b = (i, j) if i < j else (j, i)
                if (a, b) not in pairs:
                    chosen = (a, b)
                    break
            if chosen is not None:
                break
        if chosen is None:
            break

        new_row = np.zeros(n_V, dtype=np.int_)
        new_row[chosen[0]] = 1
        new_row[chosen[1]] = 1
        F = np.vstack([F, new_row])
        extra += 1

        bit_i = ((masks >> chosen[0]) & np.uint64(1)).astype(np.int32)
        bit_j = ((masks >> chosen[1]) & np.uint64(1)).astype(np.int32)
        cuts += (bit_i ^ bit_j)

    F_extra = F[n_orig_rows:].astype(np.uint8)
    return build_gadget_augmented(g.code, g.x, F_extra, basis=g.basis)
```

The algorithm body (Phase 1 Gray-code enumeration + greedy loop + vectorized cut update) is preserved verbatim from the prior implementation; only the data plumbing changed (read from `g.F`, return via `build_gadget_augmented`, no `BoostResult`).

- [ ] **Step 4: Rewrite `boost_gadget_distance` with `GadgetLayout` signature**

In `src/qldpc/circuits/surgery/cheeger.py`, replace the entire `boost_gadget_distance` function with this version:

```python
def boost_gadget_distance(
    g: GadgetLayout,
    *,
    target_distance: int,
    max_extra_qubits: int = 30,
    num_trials_per_step: int = 20,
    decoder_trials: int = 10,
    seed: int | None = None,
) -> GadgetLayout:
    """Williamson-Yoder / Webster distance-verifying gadget boost.

    Per Cain et al. arXiv:2503.10390 / Webster: iteratively add small random
    batches of degree-2 edges to F, use BP+OSD upper bound on merged code
    distance to fast-reject any augmentation whose deformed code falls below
    target. Starts from n_extra = 0 (verify bare gadget already meets target).

    Args:
        g: input gadget produced by build_gadget.
        target_distance: minimum X- and Z-distance required for acceptance
            (usually d_data, the data code's distance).
        max_extra_qubits: cap on number of new κ' qubits to consider.
        num_trials_per_step: random augmentations per n_extra value.
        decoder_trials: trials for each get_distance_bound_with_decoder call.
        seed: RNG seed for reproducibility.

    Returns:
        A new GadgetLayout whose merged code passes BP+OSD at target_distance,
        or the bare input gadget if max_extra_qubits is exhausted.

    Raises:
        ValueError: target_distance <= 0 or max_extra_qubits < 0.

    Notes:
        BP+OSD gives an UPPER bound on distance. ``d_bound >= target_distance``
        is a strong heuristic but not a proof. For exact verification,
        post-process accepted candidates with ``merged.get_distance_exact()``.
    """
    from qldpc.objects import Pauli as _Pauli
    from .gadget import build_gadget_augmented

    if target_distance <= 0:
        raise ValueError(f"target_distance must be positive, got {target_distance}.")
    if max_extra_qubits < 0:
        raise ValueError(f"max_extra_qubits must be >= 0, got {max_extra_qubits}.")

    rng = np.random.default_rng(seed)
    F_base = np.asarray(g.F).astype(np.int_)
    n_V = F_base.shape[1]

    def _passes_decoder(layout: GadgetLayout) -> bool:
        # Reconstruct the merged CSSCode from layout.HX_merged / HZ_merged
        # to feed the existing decoder.
        merged = CSSCode(
            galois.GF(2)(np.asarray(layout.HX_merged).astype(np.int_).tolist()),
            galois.GF(2)(np.asarray(layout.HZ_merged).astype(np.int_).tolist()),
            is_subsystem_code=False,
        )
        bx = merged.get_distance_bound_with_decoder(_Pauli.X, num_trials=decoder_trials)
        if bx < target_distance:
            return False
        bz = merged.get_distance_bound_with_decoder(_Pauli.Z, num_trials=decoder_trials)
        return bz >= target_distance

    # n_extra = 0: bare gadget first.
    bare = build_gadget_augmented(g.code, g.x, np.zeros((0, n_V), dtype=np.uint8), basis=g.basis)
    if _passes_decoder(bare):
        return bare

    for n_extra in range(1, max_extra_qubits + 1):
        for _trial in range(num_trials_per_step):
            F_extra = _augment_F_with_random_edges(F_base, n_extra, rng)
            if F_extra is None:
                continue
            # _augment_F_with_random_edges returns F_aug = F_base + extra rows;
            # extract just the new rows for build_gadget_augmented.
            F_extra_rows = np.asarray(F_extra[F_base.shape[0]:]).astype(np.uint8)
            try:
                candidate = build_gadget_augmented(
                    g.code, g.x, F_extra_rows, basis=g.basis,
                )
            except Exception:
                continue
            if _passes_decoder(candidate):
                return candidate

    # Exhausted: return bare gadget unchanged.
    return bare
```

The algorithm (n_extra outer loop, num_trials inner loop, _passes_decoder fast-reject) is preserved verbatim; only the plumbing changed. `_augment_F_with_random_edges` is reused unchanged (it operates on raw numpy F + rng).

- [ ] **Step 5: Rewrite `boost_gadget` dispatcher with no translation layer**

In `src/qldpc/circuits/surgery/cheeger.py`, replace the `boost_gadget` function body so it calls the new native signatures directly:

```python
def boost_gadget(
    gadget,
    *,
    method: str,
    target: float,
    seed: int | None = None,
    **kwargs,
):
    """Single entry point for Cheeger / distance boost.

    Args:
        gadget: a GadgetLayout from build_gadget.
        method: 'combinatorial' | 'distance'.
        target: target Cheeger constant (for combinatorial) or
            target distance (for distance method; cast via int(target)).
        seed: RNG seed.
        **kwargs: forwarded to the underlying boost function.

    Returns:
        A NEW GadgetLayout with boosted F, G, HX_merged, HZ_merged,
        kappa_qubits.
    """
    if method == "combinatorial":
        return boost_gadget_cheeger_combinatorial(
            gadget, target_h=target, seed=seed, **kwargs,
        )
    if method == "distance":
        return boost_gadget_distance(
            gadget, target_distance=int(target), seed=seed, **kwargs,
        )
    raise ValueError(
        f"unknown method: {method!r}. Allowed: 'combinatorial', 'distance'."
    )
```

- [ ] **Step 6: Delete the legacy adapter symbols**

In `src/qldpc/circuits/surgery/cheeger.py`, delete (in order from top to bottom of file) ALL of:

1. The `SurgeryLayout` class (currently lines 15-44 — `@dataclasses.dataclass` decorator + class body).
2. `_compute_gauge_fix` (currently lines 47-49).
3. `_build_layout` (currently lines 52-73).
4. The `BoostResult` class (currently lines 76-83).
5. The `DistanceBoostResult` class (currently lines 305-313 — search for `class DistanceBoostResult:`).
6. `_reassemble_gadget_with_new_F` (currently lines 316-368).
7. `_gadget_to_legacy_layout` (currently lines 700-744).
8. `_legacy_to_gadget` (currently lines 747-771).

After these deletions, also remove the now-unused imports at the top of `cheeger.py`:

- `import numpy.typing as npt` (was only used by `SurgeryLayout.v0_indices` annotation).
- `from qldpc.codes.common import CSSCode` MUST STAY — it's used by `boost_gadget_distance._passes_decoder` to construct the candidate merged code for the decoder.
- `import dataclasses` STAYS only if any remaining function uses it; otherwise remove. (`BoostResult` / `DistanceBoostResult` / `SurgeryLayout` deleted; remaining `boost_gadget_cheeger_combinatorial`, `boost_gadget_distance`, `boost_gadget`, `cheeger_constant`, `_exact_boundary_cheeger`, `_spectral_cheeger_lower_bound`, `_augment_F_with_random_edges` — none decorate with `@dataclasses.dataclass`. Remove the import.)

Keep these imports unchanged:
- `import galois`
- `import numpy as np`
- `from qldpc.codes.common import CSSCode`
- `from .gadget import GadgetLayout, _assemble_HX_L1`

`_assemble_HX_L1` is no longer used in cheeger.py (it was used by `_reassemble_gadget_with_new_F`). Remove `_assemble_HX_L1` from the import. The final top-of-file imports should read:

```python
"""Cheeger and distance boost transformations for surgery gadgets."""

from __future__ import annotations

import galois
import numpy as np

from qldpc.codes.common import CSSCode
from .gadget import GadgetLayout
```

- [ ] **Step 7: Run cheeger tests + safety nets**

Run: `pytest src/qldpc/circuits/surgery/_test_cheeger.py -v`

Expected: all 5 original cheeger tests + the 2 new safety-net tests pass (one skipped, one real PASS).

- [ ] **Step 8: Run full surgery suite**

Run: `pytest src/qldpc/circuits/surgery/`

Expected: all tests pass. Watch especially for any failures in `_test_gadget.py::test_build_gadget_augmented_extends_F_and_recomputes_G` (sanity check on the helper that boost now leans on).

- [ ] **Step 9: Run the Cain reproduction script**

Run: `python examples/scripts/cain_bb18_resource_exact_match.py`

Expected: prints `(Qubits=39, X-checks=20, Z-checks=20)` (exact bb_18 resource match unchanged).

- [ ] **Step 10: Verify symbol cleanup is complete**

Run:
```bash
grep -n "SurgeryLayout\|_gadget_to_legacy_layout\|_legacy_to_gadget\|_reassemble_gadget_with_new_F\|_build_layout\|_compute_gauge_fix\|class BoostResult\|class DistanceBoostResult" src/qldpc/circuits/surgery/
```

Expected: empty output.

- [ ] **Step 11: Commit**

```bash
git add src/qldpc/circuits/surgery/cheeger.py src/qldpc/circuits/surgery/_test_cheeger.py
git commit -m "refactor(surgery/cheeger): boost paths consume GadgetLayout natively"
```

---

## Task 4: Commit 4 — Consolidate 4 stitch helpers via (intercode, basis) dispatch

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py`

Strategy: the intercode/intracode axis stays as two functions (real structural difference); the basis=X/Z axis folds into "χ-carrier vs co-carrier" within each function. Net: 4 functions → 2 + a 4-line dispatcher.

- [ ] **Step 1: Write `_stitch_intercode` (replaces inter-X and inter-Z)**

In `src/qldpc/circuits/surgery/circuit.py`, ADD this new function above the current `_stitch_to_joint_csscode` (i.e., after the `keep_only_observable` definition, around line 48 area; pick a clean insertion point — anywhere above `_stitch_to_joint_csscode` works):

```python
def _stitch_intercode(g_l, g_r, bridge):
    """Inter-code joint stitch (g_l.code is not g_r.code). Handles both bases."""
    assert g_l.code is not g_r.code
    field = g_l.code.field
    g_l_aug, g_r_aug = bridge.g_l_aug, bridge.g_r_aug

    # χ-carrier abstraction: M_chi holds χ rows; M_co holds the dual cycle rows.
    if bridge.basis is Pauli.X:
        M_chi_l_src, M_co_l_src = g_l_aug.HX_merged, g_l_aug.HZ_merged
        M_chi_r_src, M_co_r_src = g_r_aug.HX_merged, g_r_aug.HZ_merged
        m_chi_l_data = g_l.code.matrix_x.shape[0]
        m_chi_r_data = g_r.code.matrix_x.shape[0]
        m_co_l_data  = g_l.code.matrix_z.shape[0]
        m_co_r_data  = g_r.code.matrix_z.shape[0]
    else:
        M_chi_l_src, M_co_l_src = g_l_aug.HZ_merged, g_l_aug.HX_merged
        M_chi_r_src, M_co_r_src = g_r_aug.HZ_merged, g_r_aug.HX_merged
        m_chi_l_data = g_l.code.matrix_z.shape[0]
        m_chi_r_data = g_r.code.matrix_z.shape[0]
        m_co_l_data  = g_l.code.matrix_x.shape[0]
        m_co_r_data  = g_r.code.matrix_x.shape[0]

    M_chi_l = np.asarray(M_chi_l_src).astype(np.int_)
    M_chi_r = np.asarray(M_chi_r_src).astype(np.int_)
    M_co_l  = np.asarray(M_co_l_src).astype(np.int_)
    M_co_r  = np.asarray(M_co_r_src).astype(np.int_)

    n_l, n_r = g_l.code.num_qudits, g_r.code.num_qudits
    k_l, k_r = g_l_aug.F.shape[0], g_r_aug.F.shape[0]
    w = bridge.width
    n_merged = n_l + n_r + k_l + k_r + w
    r_l, r_r = g_l_aug.G.shape[0], g_r_aug.G.shape[0]

    cl_data   = slice(0, n_l)
    cr_data   = slice(n_l, n_l + n_r)
    cl_kappa  = slice(n_l + n_r, n_l + n_r + k_l)
    cr_kappa  = slice(n_l + n_r + k_l, n_l + n_r + k_l + k_r)
    c_adapter = slice(n_l + n_r + k_l + k_r, n_merged)

    # Build M_chi: data χ-carrier rows (left & right) + χ rows + adapter Π labels.
    M_chi = np.zeros(
        (m_chi_l_data + m_chi_r_data + len(g_l.V0) + len(g_r.V0), n_merged),
        dtype=np.int_,
    )
    M_chi[: m_chi_l_data, cl_data] = M_chi_l[: m_chi_l_data, : n_l]
    M_chi[m_chi_l_data : m_chi_l_data + m_chi_r_data, cr_data] = M_chi_r[: m_chi_r_data, : n_r]
    chi_l_rows = M_chi_l[m_chi_l_data :, :]
    chi_r_rows = M_chi_r[m_chi_r_data :, :]
    chi_start = m_chi_l_data + m_chi_r_data
    M_chi[chi_start : chi_start + len(g_l.V0), cl_data] = chi_l_rows[:, : n_l]
    M_chi[chi_start : chi_start + len(g_l.V0), cl_kappa] = chi_l_rows[:, n_l :]
    M_chi[chi_start + len(g_l.V0) :, cr_data] = chi_r_rows[:, : n_r]
    M_chi[chi_start + len(g_l.V0) :, cr_kappa] = chi_r_rows[:, n_r :]
    for v_idx, lab in enumerate(bridge.label_l):
        if lab >= 0:
            M_chi[chi_start + v_idx, c_adapter.start + lab] = 1
    for v_idx, lab in enumerate(bridge.label_r):
        if lab >= 0:
            M_chi[chi_start + len(g_l.V0) + v_idx, c_adapter.start + lab] = 1

    # Build M_co: co-carrier data rows (with κ extension) + G_aug + new cycle.
    M_co = np.zeros(
        (m_co_l_data + m_co_r_data + r_l + r_r + (w - 1), n_merged),
        dtype=np.int_,
    )
    M_co[: m_co_l_data, cl_data]  = M_co_l[: m_co_l_data, : n_l]
    M_co[: m_co_l_data, cl_kappa] = M_co_l[: m_co_l_data, n_l :]
    M_co[m_co_l_data : m_co_l_data + m_co_r_data, cr_data]  = M_co_r[: m_co_r_data, : n_r]
    M_co[m_co_l_data : m_co_l_data + m_co_r_data, cr_kappa] = M_co_r[: m_co_r_data, n_r :]
    g_start = m_co_l_data + m_co_r_data
    M_co[g_start : g_start + r_l, cl_kappa] = M_co_l[m_co_l_data :, n_l :]
    M_co[g_start + r_l : g_start + r_l + r_r, cr_kappa] = M_co_r[m_co_r_data :, n_r :]
    cyc_start = g_start + r_l + r_r
    M_co[cyc_start :, cl_kappa]  = bridge.T_l
    M_co[cyc_start :, cr_kappa]  = bridge.T_r
    M_co[cyc_start :, c_adapter] = bridge.H_R

    if bridge.basis is Pauli.X:
        return CSSCode(field(M_chi), field(M_co), is_subsystem_code=False)
    return CSSCode(field(M_co), field(M_chi), is_subsystem_code=False)
```

- [ ] **Step 2: Write `_stitch_intracode` (replaces intra-X and intra-Z)**

In `src/qldpc/circuits/surgery/circuit.py`, ADD this function right below `_stitch_intercode`:

```python
def _stitch_intracode(g_l, g_r, bridge):
    """Intra-code joint stitch (g_l.code is g_r.code). Handles both bases.

    Differences from _stitch_intercode:
      - Shared data check rows (count = m_chi/co_data once, not l+r).
      - Shared data column block (n columns, not n_l + n_r).
      - χ rows from both sides write into the SAME data-column slice.
    """
    assert g_l.code is g_r.code
    field = g_l.code.field
    g_l_aug, g_r_aug = bridge.g_l_aug, bridge.g_r_aug

    if bridge.basis is Pauli.X:
        M_chi_l_src, M_co_l_src = g_l_aug.HX_merged, g_l_aug.HZ_merged
        M_chi_r_src, M_co_r_src = g_r_aug.HX_merged, g_r_aug.HZ_merged
        m_chi_data = g_l.code.matrix_x.shape[0]
        m_co_data  = g_l.code.matrix_z.shape[0]
    else:
        M_chi_l_src, M_co_l_src = g_l_aug.HZ_merged, g_l_aug.HX_merged
        M_chi_r_src, M_co_r_src = g_r_aug.HZ_merged, g_r_aug.HX_merged
        m_chi_data = g_l.code.matrix_z.shape[0]
        m_co_data  = g_l.code.matrix_x.shape[0]

    M_chi_l = np.asarray(M_chi_l_src).astype(np.int_)
    M_chi_r = np.asarray(M_chi_r_src).astype(np.int_)
    M_co_l  = np.asarray(M_co_l_src).astype(np.int_)
    M_co_r  = np.asarray(M_co_r_src).astype(np.int_)

    n = g_l.code.num_qudits
    k_l, k_r = g_l_aug.F.shape[0], g_r_aug.F.shape[0]
    w = bridge.width
    n_merged = n + k_l + k_r + w
    r_l, r_r = g_l_aug.G.shape[0], g_r_aug.G.shape[0]

    c_data    = slice(0, n)
    cl_kappa  = slice(n, n + k_l)
    cr_kappa  = slice(n + k_l, n + k_l + k_r)
    c_adapter = slice(n + k_l + k_r, n_merged)

    # Build M_chi: shared data check rows + χ rows (both sides into shared data).
    M_chi = np.zeros(
        (m_chi_data + len(g_l.V0) + len(g_r.V0), n_merged),
        dtype=np.int_,
    )
    M_chi[: m_chi_data, c_data] = M_chi_l[: m_chi_data, : n]  # shared
    chi_l_rows = M_chi_l[m_chi_data :, :]
    chi_r_rows = M_chi_r[m_chi_data :, :]
    M_chi[m_chi_data : m_chi_data + len(g_l.V0), c_data]  = chi_l_rows[:, : n]
    M_chi[m_chi_data : m_chi_data + len(g_l.V0), cl_kappa] = chi_l_rows[:, n :]
    M_chi[m_chi_data + len(g_l.V0) :, c_data]  = chi_r_rows[:, : n]
    M_chi[m_chi_data + len(g_l.V0) :, cr_kappa] = chi_r_rows[:, n :]
    for v_idx, lab in enumerate(bridge.label_l):
        if lab >= 0:
            M_chi[m_chi_data + v_idx, c_adapter.start + lab] = 1
    for v_idx, lab in enumerate(bridge.label_r):
        if lab >= 0:
            M_chi[m_chi_data + len(g_l.V0) + v_idx, c_adapter.start + lab] = 1

    # Build M_co: shared data co-carrier rows with κ extension on BOTH sides,
    # then G_l, G_r, then new cycle.
    M_co = np.zeros(
        (m_co_data + r_l + r_r + (w - 1), n_merged),
        dtype=np.int_,
    )
    M_co[: m_co_data, c_data]    = M_co_l[: m_co_data, : n]
    M_co[: m_co_data, cl_kappa]  = M_co_l[: m_co_data, n :]
    M_co[: m_co_data, cr_kappa]  = M_co_r[: m_co_data, n :]
    M_co[m_co_data : m_co_data + r_l, cl_kappa] = M_co_l[m_co_data :, n :]
    M_co[m_co_data + r_l : m_co_data + r_l + r_r, cr_kappa] = M_co_r[m_co_data :, n :]
    cyc_start = m_co_data + r_l + r_r
    M_co[cyc_start :, cl_kappa]  = bridge.T_l
    M_co[cyc_start :, cr_kappa]  = bridge.T_r
    M_co[cyc_start :, c_adapter] = bridge.H_R

    if bridge.basis is Pauli.X:
        return CSSCode(field(M_chi), field(M_co), is_subsystem_code=False)
    return CSSCode(field(M_co), field(M_chi), is_subsystem_code=False)
```

- [ ] **Step 3: Replace the old `_stitch_to_joint_csscode` with a thin dispatcher**

In `src/qldpc/circuits/surgery/circuit.py`, REPLACE the existing `_stitch_to_joint_csscode` function (the one that currently contains both the early dispatch and the inter-X assembly inline) with this minimal version:

```python
def _stitch_to_joint_csscode(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
) -> CSSCode:
    """Assemble merged CSSCode for two-PPM surgery (spec §3 block tables).

    Dispatches on the structural axis (g_l.code is g_r.code → intra-code
    shares data; otherwise inter-code).  Each branch handles both
    bridge.basis values internally via the χ-carrier abstraction.
    """
    if g_l.code is g_r.code:
        return _stitch_intracode(g_l, g_r, bridge)
    return _stitch_intercode(g_l, g_r, bridge)
```

- [ ] **Step 4: Delete the 3 obsolete stitch helpers**

In `src/qldpc/circuits/surgery/circuit.py`, DELETE the following 3 functions in their entirety:

1. `_stitch_intracode_joint_csscode` (currently at line 176)
2. `_stitch_intercode_joint_csscode_basis_z` (currently at line 235)
3. `_stitch_intracode_joint_csscode_basis_z` (currently at line 302)

Each is `def _stitch_..._csscode(...):` followed by ~50-60 lines ending in `return CSSCode(field(HX), field(HZ), is_subsystem_code=False)`.

- [ ] **Step 5: Run stitch and joint-PPM tests**

Run:
```bash
pytest src/qldpc/circuits/surgery/_test_circuit.py -v \
  -k "stitch or joint or adapter_cycle"
```

Expected: every stitch test passes (basis=X and basis=Z, intra and inter), plus `test_build_joint_ppm_circuit_*` (intracode + intercode noiseless observables zero), plus `test_joint_xx_in_stabilizer_*` and `test_adapter_cycle_check_weight_bounded`.

- [ ] **Step 6: Run full surgery suite**

Run: `pytest src/qldpc/circuits/surgery/`

Expected: green.

- [ ] **Step 7: Verify basis-suffix function names are gone**

Run:
```bash
grep -n "basis_z" src/qldpc/circuits/surgery/circuit.py
```

Expected: empty.

- [ ] **Step 8: Commit**

```bash
git add src/qldpc/circuits/surgery/circuit.py
git commit -m "refactor(surgery/circuit): consolidate 4 stitch helpers via (intercode, basis) dispatch"
```

---

## Final verification (all four commits)

- [ ] **Run the full surgery suite one more time**

Run: `pytest src/qldpc/circuits/surgery/ -v`

Expected: every test that was green on `main` is still green.

- [ ] **Run both example scripts that exercise the public API**

Run:
```bash
python examples/scripts/cain_bb18_resource_exact_match.py
```
Expected: `(Qubits=39, X-checks=20, Z-checks=20)`.

Run (smoke-strength — let it run for a minute then ctrl-C if it's progressing normally):
```bash
python examples/scripts/single_ppm_vs_memory_ler.py --help 2>/dev/null \
  || head -100 examples/scripts/single_ppm_vs_memory_ler.py
```

Confirm the import line `from qldpc.circuits.surgery import build_gadget, build_single_ppm_circuit, boost_gadget, cheeger_constant` resolves without error.

- [ ] **Push the branch and open the PR**

```bash
git log --oneline main..HEAD
```

Expected output: exactly 4 commits, in order:

```
<sha4> refactor(surgery/circuit): consolidate 4 stitch helpers via (intercode, basis) dispatch
<sha3> refactor(surgery/cheeger): boost paths consume GadgetLayout natively
<sha2> refactor(surgery/cheeger): drop unused 'spectral' boost strategy
<sha1> refactor(surgery/tests): split _test.py by feature
```

If that's the shape, the branch is ready to push. PR body should reference `docs/superpowers/specs/2026-06-10-surgery-cleanup-design.md`.

---

## Self-review notes

Coverage check against spec:
- §"Commit 1" → Task 1, 10 steps with snapshot-and-diff invariant.
- §"Commit 2" → Task 2, 8 steps; helper retention for `cheeger_constant` called out in Step 1.
- §"Commit 3" → Task 3, 11 steps; new boost bodies shown in full; safety-net test added per spec; symbol cleanup explicit in Step 6.
- §"Commit 4" → Task 4, 8 steps; new function bodies shown in full; safety-net tests are existing parametrized tests (no new tests added per spec).
- §"Verification matrix" → Final-verification block.

Type/name consistency check:
- `boost_gadget_cheeger_combinatorial` keeps its existing name across Tasks 2-3 (Task 2 leaves it untouched; Task 3 rewrites the body but keeps the symbol).
- `boost_gadget_distance` likewise.
- `_stitch_intercode` and `_stitch_intracode` (new names in Task 4) match the dispatcher.
- New test `test_boost_gadget_combinatorial_basis_z_preserves_chi_carrier` — referenced once in Task 3 Step 1 (added) and once in Step 2 (verified).

Known gap: a distance-strategy basis=Z safety-net is not added because the Webster JSON ships only X̄ operators. Distance basis=X is already covered by `test_boost_gadget_preserves_css_commutation[distance]`; distance has no external user, so this gap is acceptable per spec.
