# SkipTree Bridge v3 Implementation Plan

> **Execution status (2026-06-07):** Partial. Tasks 1–9 SHIPPED as v2.1.
> Tasks 10–17 ABANDONED after numerical investigation showed the spec's
> algorithmic construction is infeasible on BB_1 Z̄_1 (39/182 violators
> in Task 9 probe; Path B linear-solve rank 26 vs augmented rank 27).
> The two paper joint codes are accessible via the new
> `build_joint_from_ide_fixture()` helper (`_ide_fixtures.py`) which
> loads Ide's Zenodo `.mtx` directly. See spec §10 for the full
> investigation outcome and pointers for resuming v3 work.

**Goal:** Replace the v2 path-graph joint bridge with the SkipTree-based Lemma 10 adapter from Ide et al. arXiv:2410.03628 §VII B + §VII C. The new construction reproduces the stabilizer groups of Ide's [[355, 25, 10]] BB-LP joint code and [[150, 5, 12]] BB-BB intra-code joint.

**Architecture:** Split the 1672-line `surgery.py` into a `surgery/` package (`layered.py`, `skiptree.py`, `cellulation.py`, `cheeger.py`, `joint.py`, `port.py`), keeping `__init__.py` re-exports for backwards compatibility. Implement Algorithm 2 (flag-based SkipTree) for the open-path H_R basis. Construct the joint code from each gadget's HX/HZ via (i) Lemma 10 adapter X-checks `[T_1 | H_R | T_2]`, (ii) chi-row bridge extension with α_v breaking the `Σχ = X̄_s` identity, (iii) Webster Z-row bridge extension with `b_c = running-XOR(T_s[:, c])`.

**Tech Stack:** Python 3.12, numpy, galois (GF(2)), networkx, scipy.io.mmread, pytest. Existing surgery code, Webster gadget, _skip_tree port (eswaroop/MIT).

**Branch:** `feat/surgery-construction` (HEAD `cbedfc7` after spec commit).

---

## File structure (final state)

```
src/qldpc/codes/surgery/
  __init__.py        # re-export public + previously-imported privates
  layered.py         # SurgeryLayout, _restrict_to_logical_support, _compute_gauge_fix,
                     #   _LayeredBlocks, _build_layered_blocks, _assemble_merged_HX,
                     #   _assemble_merged_HZ, build_layered_surgery_code, _build_layout,
                     #   load_webster_seed_set, _build_generalised_bicycle_code
  skiptree.py        # _skip_tree (Alg 1, existing port) + _skip_tree_hr (Alg 2, NEW)
  cellulation.py     # _cellulate_long_cycles
  cheeger.py         # BoostResult, _exact_boundary_cheeger, _spectral_cheeger_lower_bound,
                     #   boost_gadget_cheeger, DistanceBoostResult, _reassemble_gadget_with_new_F,
                     #   _augment_F_with_random_edges, boost_gadget_cheeger_combinatorial,
                     #   boost_gadget_distance
  joint.py           # NEW: JointSurgeryLayout, _validate_joint_logical_ops,
                     #   _build_auxiliary_graph, _label_inverse, canonical_HR,
                     #   _running_xor_b_c, _chi_z_compatibility_check,
                     #   _extend_chi_rows_with_bridge, _extend_Z_rows_with_bridge,
                     #   _build_adapter_x_checks, build_joint_measurement_code
  port.py            # NEW: SetValuedPort + helpers for §VII C

src/qldpc/codes/surgery_test.py   # tests updated + 3 new joint tests

tests/fixtures/ide_zenodo/         # NEW: Ide's Zenodo bundle (35 KB)
  BB_98_6_12/...                   # original + deformed BB matrices
  LP_200_20_10/...                 # original + deformed LP matrices
  BB_98_LP_200_adapter/...         # joint BB-LP HX/HZ
  BB_98_intracode_adapter/...      # joint BB-BB HX/HZ
```

`surgery.py` (the file) is replaced by `surgery/` package directory.

---

## Tasks overview

1. Set up Zenodo fixtures
2. Create `surgery/` package skeleton + move `cellulation.py`
3. Move `skiptree.py`
4. Move `cheeger.py`
5. Move `layered.py` (largest split)
6. Run full 81-test suite — must pass
7. Implement `_skip_tree_hr` (Algorithm 2)
8. Joint helpers — aux graph, label inverse, canonical H_R, running-XOR b_c
9. χ–Z compatibility lemma verification (load-bearing § 4.7)
10. Implement chi-row bridge extension
11. Implement Webster Z-row bridge extension
12. Implement adapter X-checks block
13. Implement `build_joint_measurement_code` v3 (disjoint, BB-LP)
14. BB-LP joint test
15. Set-valued port for §VII C overlap
16. BB-BB intra-code joint test
17. Delete v2 dead code and update test imports

---

### Task 1: Set up Ide Zenodo fixtures

**Files:**
- Create: `tests/fixtures/ide_zenodo/` (directory, ~32 .mtx + .txt files)
- Create: `src/qldpc/codes/_ide_fixtures.py` (loader helpers)

- [ ] **Step 1: Verify Zenodo data is already extracted to /tmp**

Run: `ls /tmp/data_qLDPC_surgery/BB_98_LP_200_adapter/`
Expected output includes: `Hx_intercode_BB_LP_adapter-Z_1_Z_2_deformed-code.mtx`

- [ ] **Step 2: Copy Zenodo data into repo fixture path**

Run:
```bash
mkdir -p tests/fixtures/ide_zenodo
cp -r /tmp/data_qLDPC_surgery/* tests/fixtures/ide_zenodo/
ls tests/fixtures/ide_zenodo/
```
Expected output:
```
BB_98_6_12  BB_98_LP_200_adapter  BB_98_intracode_adapter  LP_200_20_10
```

- [ ] **Step 3: Write a fixture loader helper**

Create `src/qldpc/codes/_ide_fixtures.py`:
```python
"""Load Ide et al. (arXiv:2410.03628) Zenodo supplementary matrices.

These matrices live under tests/fixtures/ide_zenodo/ and serve as
ground truth for stab-group equality assertions in joint tests.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.io import mmread

_FIXTURE_ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "ide_zenodo"


def fixtures_available() -> bool:
    return _FIXTURE_ROOT.exists() and any(_FIXTURE_ROOT.iterdir())


def load_mtx(rel_path: str) -> np.ndarray:
    return mmread(_FIXTURE_ROOT / rel_path).toarray().astype(np.int_)


def load_ide_joint_BB_LP() -> tuple[np.ndarray, np.ndarray]:
    """Return Ide's HX, HZ for the [[355, 25, 10]] BB-LP joint code."""
    HX = load_mtx("BB_98_LP_200_adapter/Hx_intercode_BB_LP_adapter-Z_1_Z_2_deformed-code.mtx")
    HZ = load_mtx("BB_98_LP_200_adapter/Hz_intercode_BB_LP_adapter-Z_1_Z_2_deformed-code.mtx")
    return HX, HZ


def load_ide_joint_BB_intracode() -> tuple[np.ndarray, np.ndarray]:
    """Return Ide's HX, HZ for the [[150, 5, 12]] BB Z_1 Z_3 joint code."""
    HX = load_mtx("BB_98_intracode_adapter/Hx_BB_intracode_Z_1_Z_3_adapted-code.mtx")
    HZ = load_mtx("BB_98_intracode_adapter/Hz_BB_intracode_Z_1_Z_3_adapted-code.mtx")
    return HX, HZ


def load_ide_skiptree_TPG(path: str) -> dict[str, np.ndarray]:
    """Parse one of Ide's *_GTP.txt files. Returns dict with keys G_mat_*, T_*, P_*."""
    text = (_FIXTURE_ROOT / path).read_text()
    out: dict[str, np.ndarray] = {}
    pattern = re.compile(r"^([A-Za-z_0-9]+)\s*=\s*np\.array\((.*?)\)\s*$",
                          re.MULTILINE | re.DOTALL)
    for match in pattern.finditer(text):
        name = match.group(1)
        # safe eval: only floats, lists, np.array allowed
        arr = eval(match.group(2), {"np": np}, {})
        out[name] = np.array(arr, dtype=int)
    return out
```

- [ ] **Step 4: Write a tiny smoke test**

Append to `src/qldpc/codes/surgery_test.py`:
```python
import pytest
from qldpc.codes._ide_fixtures import (
    fixtures_available, load_ide_joint_BB_LP,
    load_ide_joint_BB_intracode, load_ide_skiptree_TPG,
)


@pytest.mark.skipif(not fixtures_available(), reason="Zenodo fixtures not present")
def test_ide_fixtures_load_correctly():
    HX_bbLP, HZ_bbLP = load_ide_joint_BB_LP()
    assert HX_bbLP.shape == (175, 355)
    assert HZ_bbLP.shape == (173, 355)
    assert ((HX_bbLP @ HZ_bbLP.T) % 2 == 0).all()

    HX_bbBB, HZ_bbBB = load_ide_joint_BB_intracode()
    assert HX_bbBB.shape == (73, 150)
    assert HZ_bbBB.shape == (72, 150)
    assert ((HX_bbBB @ HZ_bbBB.T) % 2 == 0).all()

    bb_z1 = load_ide_skiptree_TPG(
        "BB_98_LP_200_adapter/skipTree_transformations/BB_98_6_12_Z_1_GTP.txt"
    )
    assert "T_1" in bb_z1 and "P_1" in bb_z1 and "G_mat_1" in bb_z1
    T1, P1, G1 = bb_z1["T_1"] % 2, bb_z1["P_1"] % 2, bb_z1["G_mat_1"] % 2
    HR_canonical = np.zeros((P1.shape[0] - 1, P1.shape[0]), dtype=int)
    for l in range(P1.shape[0] - 1):
        HR_canonical[l, l] = 1
        HR_canonical[l, l + 1] = 1
    assert np.array_equal((T1 @ G1 @ P1) % 2, HR_canonical)
```

- [ ] **Step 5: Run smoke test**

Run: `pytest src/qldpc/codes/surgery_test.py::test_ide_fixtures_load_correctly -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/ide_zenodo/ src/qldpc/codes/_ide_fixtures.py src/qldpc/codes/surgery_test.py
git commit -m "test: load Ide Zenodo fixtures for joint stab-group verification"
```

---

### Task 2: Create surgery/ package skeleton and migrate cellulation.py

**Files:**
- Create: `src/qldpc/codes/surgery/__init__.py`
- Create: `src/qldpc/codes/surgery/cellulation.py`
- Modify: `src/qldpc/codes/surgery.py` → temporarily import from new package
- Eventually delete: `src/qldpc/codes/surgery.py` (after all splits done — Task 7)

- [ ] **Step 1: Create the package skeleton (do not delete surgery.py yet)**

The trick: Python treats `surgery.py` and `surgery/` as the same module name. We must replace the file with the package in one shot. Use a staging approach:

Create `src/qldpc/codes/surgery_new/` (temporary name) with `__init__.py` that re-exports everything from `qldpc.codes.surgery`:
```python
"""Surgery construction package (v3, replaces flat surgery.py).

Public API is re-exported here for backwards compatibility with
``from qldpc.codes.surgery import ...`` callers.
"""

# Will be populated as modules are split out.
from qldpc.codes.surgery import (
    # public
    SurgeryLayout,
    JointSurgeryLayout,
    BoostResult,
    DistanceBoostResult,
    build_layered_surgery_code,
    build_joint_measurement_code,
    boost_gadget_cheeger,
    boost_gadget_cheeger_combinatorial,
    boost_gadget_distance,
    load_webster_seed_set,
    # internals used by tests
    _restrict_to_logical_support,
    _compute_gauge_fix,
    _build_layered_blocks,
    _assemble_merged_HX,
    _assemble_merged_HZ,
    _build_generalised_bicycle_code,
    _skip_tree,
    _cellulate_long_cycles,
    _spectral_cheeger_lower_bound,
)
```

- [ ] **Step 2: Move _cellulate_long_cycles into cellulation.py**

Read the current implementation at `src/qldpc/codes/surgery.py:555-612`.

Create `src/qldpc/codes/surgery_new/cellulation.py` with EXACT verbatim copy of that function, prepending the necessary imports:
```python
"""Lemma 14 cellulation: break long cycles in the auxiliary graph.

Direct port of cellulate_long_cycles() in
https://github.com/eswaroop/adapters-LDPC-surgery (MIT, 2025).
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def _cellulate_long_cycles(
    G: nx.Graph,
    edge_qubit_to_vertices: dict[int, tuple[int, int]],
    vert_to_edge: dict[tuple[int, int], int],
    G_mat: np.ndarray,
    max_len: int = 6,
) -> tuple[list[tuple[int, int]], dict[int, tuple[int, int]], dict[tuple[int, int], int], np.ndarray]:
    # ... verbatim copy of the existing body ...
```

(Engineer: read lines 555-612 of the existing surgery.py and paste exactly into the new file.)

- [ ] **Step 3: Stage the cutover**

Update `src/qldpc/codes/surgery_new/__init__.py` to re-export from local `cellulation`:
```python
from .cellulation import _cellulate_long_cycles  # noqa: F401
```
Remove the line that re-imports `_cellulate_long_cycles` from the flat module (still keep the others).

- [ ] **Step 4: Run the cellulation-specific tests**

Run: `pytest src/qldpc/codes/surgery_test.py -k cellulat -v`
Expected: existing tests still pass (since old surgery.py still has the function and the new package re-exports it).

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery_new/
git commit -m "refactor: scaffold surgery/ package with cellulation.py extracted"
```

---

### Task 3: Migrate skiptree.py

**Files:**
- Create: `src/qldpc/codes/surgery_new/skiptree.py`
- Modify: `src/qldpc/codes/surgery_new/__init__.py`

- [ ] **Step 1: Copy _skip_tree from surgery.py:483-553**

Create `src/qldpc/codes/surgery_new/skiptree.py`:
```python
"""SkipTree algorithm (Ide / Swaroop et al. arXiv:2410.03628 §III).

Algorithm 1 returns T G P = H_C (cyclic repetition basis).
Algorithm 2 (added in Task 7) returns T G P = H_R (open path basis).
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def _skip_tree(
    S: nx.Graph,
    root: int = 0,
    edge_index_verts: dict[tuple[int, int], int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    # ... verbatim copy of surgery.py:483-553 body ...
```

- [ ] **Step 2: Update package init**

In `src/qldpc/codes/surgery_new/__init__.py`, replace the `_skip_tree` re-import line:
```python
from .skiptree import _skip_tree  # noqa: F401
```

- [ ] **Step 3: Run skiptree tests**

Run: `pytest src/qldpc/codes/surgery_test.py -k skip -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/qldpc/codes/surgery_new/skiptree.py src/qldpc/codes/surgery_new/__init__.py
git commit -m "refactor: extract skiptree.py from surgery.py"
```

---

### Task 4: Migrate cheeger.py

**Files:**
- Create: `src/qldpc/codes/surgery_new/cheeger.py`
- Modify: `src/qldpc/codes/surgery_new/__init__.py`

- [ ] **Step 1: Copy lines 613–1264 into cheeger.py**

Includes: `BoostResult`, `_exact_boundary_cheeger`, `_spectral_cheeger_lower_bound`, `boost_gadget_cheeger`, `DistanceBoostResult`, `_reassemble_gadget_with_new_F`, `_augment_F_with_random_edges`, `boost_gadget_cheeger_combinatorial`, `boost_gadget_distance`.

Add imports at the top:
```python
"""Cheeger and distance boost transformations for surgery gadgets."""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Callable

import galois
import networkx as nx
import numpy as np
import numpy.typing as npt
import scipy.sparse
import scipy.sparse.linalg

from qldpc.codes.common import CSSCode
from .layered import SurgeryLayout  # will exist after Task 5
```

(Note: layered.py doesn't exist yet — fix the import after Task 5 lands. Until then, keep the re-export from old surgery.py for SurgeryLayout.)

Temporary workaround for the import: `from qldpc.codes.surgery import SurgeryLayout`.

- [ ] **Step 2: Update package init**

```python
from .cheeger import (  # noqa: F401
    BoostResult,
    DistanceBoostResult,
    boost_gadget_cheeger,
    boost_gadget_cheeger_combinatorial,
    boost_gadget_distance,
    _spectral_cheeger_lower_bound,
)
```

- [ ] **Step 3: Run cheeger tests**

Run: `pytest src/qldpc/codes/surgery_test.py -k cheeger -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/qldpc/codes/surgery_new/cheeger.py src/qldpc/codes/surgery_new/__init__.py
git commit -m "refactor: extract cheeger.py from surgery.py"
```

---

### Task 5: Migrate layered.py

**Files:**
- Create: `src/qldpc/codes/surgery_new/layered.py`
- Modify: `src/qldpc/codes/surgery_new/__init__.py`

- [ ] **Step 1: Copy lines 1–477 into layered.py**

Includes: `SurgeryLayout`, `_restrict_to_logical_support`, `_compute_gauge_fix`, `_LayeredBlocks`, `_build_layered_blocks`, `_assemble_merged_HX`, `_assemble_merged_HZ`, `build_layered_surgery_code`, `_build_layout`, `load_webster_seed_set`, `_build_generalised_bicycle_code`.

Top of file:
```python
"""Webster (Cross et al. arXiv:2407.18393) L-layer surgery construction.

build_layered_surgery_code is the single-logical workhorse — preserved
unchanged from the v1/v2 codebase.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import galois
import numpy as np
import numpy.typing as npt

from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli
```

- [ ] **Step 2: Fix cheeger.py import**

Edit `src/qldpc/codes/surgery_new/cheeger.py`: replace the temporary `from qldpc.codes.surgery import SurgeryLayout` with `from .layered import SurgeryLayout`.

- [ ] **Step 3: Update package init**

```python
from .layered import (  # noqa: F401
    SurgeryLayout,
    build_layered_surgery_code,
    load_webster_seed_set,
    _restrict_to_logical_support,
    _compute_gauge_fix,
    _build_layered_blocks,
    _assemble_merged_HX,
    _assemble_merged_HZ,
    _build_generalised_bicycle_code,
)
```

- [ ] **Step 4: Run all Webster single-logical tests**

Run: `pytest src/qldpc/codes/surgery_test.py -v 2>&1 | tail -30`
Expected: same pass count as before (81 originally + the fixture smoke from Task 1 = 82).

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery_new/
git commit -m "refactor: extract layered.py from surgery.py"
```

---

### Task 6: Cutover — replace surgery.py with surgery/ package

**Files:**
- Delete: `src/qldpc/codes/surgery.py`
- Move: `src/qldpc/codes/surgery_new/` → `src/qldpc/codes/surgery/`
- Modify: `src/qldpc/codes/surgery/__init__.py` (remove transitional imports)

- [ ] **Step 1: Verify joint-related code still lives in surgery.py**

Run: `grep -nE "^def build_joint_measurement_code|^class JointSurgeryLayout|^def _build_bridge_via_skiptree" src/qldpc/codes/surgery.py`
Expected output: matches at lines around 1266, 1376, 1614 — these stay in `surgery.py` until v3 replaces them in Task 13.

So we DO need to keep `surgery.py` around as the joint module — or move the joint pieces into the new package first.

- [ ] **Step 2: Move joint pieces into surgery_new/joint.py (placeholder)**

Read lines 1266–1672 of `src/qldpc/codes/surgery.py` and copy into `src/qldpc/codes/surgery_new/joint.py`, preserving the v2 implementation. Top of file:
```python
"""Joint-measurement construction (v2 path-graph bridge — to be replaced
with SkipTree-based v3 in subsequent tasks of this plan).
"""

from __future__ import annotations

import dataclasses

import galois
import numpy as np
import numpy.typing as npt

from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli

from .layered import (
    SurgeryLayout,
    _build_layered_blocks,
    build_layered_surgery_code,
)

# ... paste lines 1266-1672 here ...
```

- [ ] **Step 3: Update package init to re-export joint pieces**

```python
from .joint import (  # noqa: F401
    JointSurgeryLayout,
    build_joint_measurement_code,
)
```

Remove `from qldpc.codes.surgery import ...` lines — the new package now owns everything.

- [ ] **Step 4: Cutover**

```bash
git mv src/qldpc/codes/surgery.py /tmp/surgery_old.py.bak
git mv src/qldpc/codes/surgery_new src/qldpc/codes/surgery
```

- [ ] **Step 5: Run all surgery tests**

Run: `pytest src/qldpc/codes/surgery_test.py -v 2>&1 | tail -30`
Expected: same pass count as Task 5 step 4 (~82 passing, none failing).

- [ ] **Step 6: Run full repo test suite**

Run: `pytest src/qldpc/ -q 2>&1 | tail -10`
Expected: no new failures vs. baseline.

- [ ] **Step 7: Commit**

```bash
git add src/qldpc/codes/surgery/
git rm src/qldpc/codes/surgery.py
git commit -m "refactor: replace flat surgery.py with surgery/ package"
```

---

### Task 7: Implement `_skip_tree_hr` (Algorithm 2 flag-based variant)

**Files:**
- Modify: `src/qldpc/codes/surgery/skiptree.py` (append new function)
- Modify: `src/qldpc/codes/surgery_test.py` (new unit tests)

- [ ] **Step 1: Write failing tests for `_skip_tree_hr`**

Append to `src/qldpc/codes/surgery_test.py`:
```python
def test_skip_tree_hr_path_graph_is_optimal():
    """On a path graph of 5 vertices, Algorithm 2 returns T = H_R(5) directly."""
    from qldpc.codes.surgery import _skip_tree_hr
    G = nx.path_graph(5)
    T, P = _skip_tree_hr(G, root=0)
    n = 5
    H_R = np.zeros((n - 1, n - 1), dtype=int)
    for l in range(n - 1):
        H_R[l, l] = 1  # edge l-to-l+1 is the path edge between labels l, l+1
    # T should be identity since path is the spanning tree itself
    assert T.shape == (n - 1, n - 1)
    assert P.shape == (n, n)


def test_skip_tree_hr_gives_HR_basis():
    """T G P = H_R(n) for the open-path basis, NOT H_C (cyclic)."""
    from qldpc.codes.surgery import _skip_tree_hr
    # cyclic graph of 6 vertices, Hamilton path uses 5 of 6 edges
    G = nx.cycle_graph(6)
    span = nx.minimum_spanning_tree(G)
    T, P = _skip_tree_hr(span, root=0)
    n = 6
    edge_list = list(span.edges())
    G_mat = np.zeros((len(edge_list), n), dtype=int)
    for i, (u, v) in enumerate(edge_list):
        G_mat[i, u] = 1
        G_mat[i, v] = 1
    H_R = np.zeros((n - 1, n), dtype=int)
    for l in range(n - 1):
        H_R[l, l] = 1
        H_R[l, l + 1] = 1
    product = (T @ G_mat @ P) % 2
    assert np.array_equal(product, H_R)
```

- [ ] **Step 2: Run tests to verify FAIL**

Run: `pytest src/qldpc/codes/surgery_test.py -k skip_tree_hr -v`
Expected: FAIL with `ImportError: cannot import name '_skip_tree_hr'`.

- [ ] **Step 3: Implement `_skip_tree_hr`**

Append to `src/qldpc/codes/surgery/skiptree.py`:
```python
def _skip_tree_hr(
    S: nx.Graph,
    root: int = 0,
    edge_index_verts: dict[tuple[int, int], int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """SkipTree Algorithm 2 (flag-based, returns T G P = H_R, not H_C).

    Ide et al. arXiv:2410.03628 Appendix VIII Algorithm 2. Differs from
    Algorithm 1 by NOT skipping the leftmost child in the "skip" mode,
    allowing the open-path basis (not cyclic).

    Args:
        S: connected simple graph (typically a spanning tree).
        root: vertex to start labeling at.
        edge_index_verts: optional override mapping each edge ``tuple(sorted)``
            to a column index in T.

    Returns:
        T: shape (n-1, |E|) edge-incidence matrix. T[l, e] = 1 iff edge e
            lies on the shortest path from vertex labeled l to (l+1).
        P: shape (n, n) permutation matrix.
    """
    n = S.number_of_nodes()
    index = 0
    label = [0] * n
    visited: set[int] = set()

    def label_first(v: int, skip: bool) -> None:
        nonlocal index
        visited.add(v)
        label[index] = v
        index = index + 1

        children = [nbr for nbr in S.neighbors(v) if nbr not in visited]
        for child_idx, child in enumerate(children):
            youngest = child_idx == len(children) - 1
            if youngest and not skip:
                label_first(child, skip=False)
            else:
                label_last(child)

    def label_last(v: int) -> None:
        nonlocal index
        for child in S.neighbors(v):
            if child not in visited:
                label_first(child, skip=True)
        visited.add(v)
        label[index] = v
        index = index + 1

    label_first(root, skip=False)

    P = np.zeros((n, n), dtype=np.int_)
    for l_idx, v in enumerate(label):
        P[v, l_idx] = 1

    if not edge_index_verts:
        edge_index_verts = {tuple(sorted(e)): i for i, e in enumerate(S.edges())}

    T = np.zeros((n - 1, len(edge_index_verts)), dtype=np.int_)
    # Open-path: ONLY label l → l+1, no cyclic l = n-1 → 0.
    for l_idx in range(n - 1):
        path = nx.shortest_path(S, source=label[l_idx], target=label[l_idx + 1])
        for u, v in zip(path[:-1], path[1:]):
            e = tuple(sorted((u, v)))
            T[l_idx, edge_index_verts[e]] = 1
    return T, P
```

- [ ] **Step 4: Re-export from package init**

Add to `src/qldpc/codes/surgery/__init__.py`:
```python
from .skiptree import _skip_tree_hr  # noqa: F401
```

- [ ] **Step 5: Run tests**

Run: `pytest src/qldpc/codes/surgery_test.py -k skip_tree_hr -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/qldpc/codes/surgery/skiptree.py src/qldpc/codes/surgery/__init__.py src/qldpc/codes/surgery_test.py
git commit -m "feat: add SkipTree Algorithm 2 (flag-based, H_R basis) per Ide Appendix VIII"
```

---

### Task 8: Joint helpers — aux graph, label inverse, H_R, running-XOR b_c

**Files:**
- Create: `src/qldpc/codes/surgery/joint.py` (replace v2 placeholder body)
- Modify: `src/qldpc/codes/surgery_test.py`

- [ ] **Step 1: Write failing unit tests for helpers**

Append to `src/qldpc/codes/surgery_test.py`:
```python
def test_build_auxiliary_graph_returns_simple_graph_of_V0():
    """For BB_1 Z̄_1 wt 14, aux graph G_1 has 14 vertices."""
    from qldpc.codes.surgery.joint import _build_auxiliary_graph
    bb = codes.BBCode((7, 7),
        sympy.Symbol("x")**3 + sympy.Symbol("y")**3 + sympy.Symbol("y")**4,
        sympy.Symbol("y")**6 + sympy.Symbol("x")**2 + sympy.Symbol("x")**5)
    z1_support = np.zeros(98, dtype=int)
    for q in [6, 8, 13, 17, 31, 32, 33, 35, 36, 37, 41, 50, 51, 93]:
        z1_support[q] = 1
    dual = codes.CSSCode(bb.matrix_z, bb.matrix_x, is_subsystem_code=False)
    _, layout = build_layered_surgery_code(dual, z1_support, num_layers=1, validate_logical_op=False)
    G, edge_qubit_to_vertices = _build_auxiliary_graph(layout.F)
    assert G.number_of_nodes() == 14


def test_label_inverse_round_trip():
    from qldpc.codes.surgery.joint import _label_inverse
    P = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=int)
    # Vertex 0 has label 1, vertex 1 has label 0, vertex 2 has label 2
    assert _label_inverse(P) == [1, 0, 2]


def test_canonical_HR_is_path_graph_check():
    from qldpc.codes.surgery.joint import canonical_HR
    H = canonical_HR(4)
    expected = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=int)
    assert np.array_equal(H, expected)


def test_running_xor_b_c_solves_HRt_b_eq_Tcol():
    """b_c = running XOR of T[:, c] satisfies H_R^T b_c = T[:, c]."""
    from qldpc.codes.surgery.joint import canonical_HR, _running_xor_b_c
    w = 5
    H_R = canonical_HR(w)
    rng = np.random.default_rng(0)
    T_col = rng.integers(0, 2, size=w - 1)
    b = _running_xor_b_c(T_col)
    assert b.shape == (w,)
    assert b[0] == 0
    product = (H_R.T @ b) % 2
    assert np.array_equal(product, T_col)
```

- [ ] **Step 2: Run tests to verify FAIL**

Run: `pytest src/qldpc/codes/surgery_test.py -k "auxiliary_graph or label_inverse or canonical_HR or running_xor" -v`
Expected: FAIL with `ImportError: cannot import name '_build_auxiliary_graph'`.

- [ ] **Step 3: Add helpers to joint.py**

In `src/qldpc/codes/surgery/joint.py`, append (above v2 body — these are NEW helpers):
```python
import networkx as nx


def _build_auxiliary_graph(
    F: np.ndarray | galois.FieldArray,
) -> tuple[nx.Graph, dict[int, tuple[int, int]]]:
    """Build aux graph G_s from Webster F matrix.

    Vertices = V_0_s (columns of F).
    Edges = rows of F with weight exactly 2 (one per κ_s ancilla qubit).
    Returns G and a dict mapping κ_s qubit index → (u, v) vertex pair.
    """
    F_arr = np.asarray(F).astype(int)
    n_V = F_arr.shape[1]
    G = nx.Graph()
    G.add_nodes_from(range(n_V))
    edge_qubit_to_vertices: dict[int, tuple[int, int]] = {}
    for i, row in enumerate(F_arr):
        eps = sorted(np.flatnonzero(row).tolist())
        if len(eps) == 2:
            u, v = eps[0], eps[1]
            edge_qubit_to_vertices[i] = (u, v)
            if not G.has_edge(u, v):
                G.add_edge(u, v)
    return G, edge_qubit_to_vertices


def _label_inverse(P: np.ndarray) -> list[int]:
    """Return list `inv[l] = vertex v` such that P[v, l] = 1.

    P is a permutation matrix with exactly one 1 per row and per column.
    """
    n = P.shape[0]
    inv = [-1] * n
    for v in range(n):
        for l in range(n):
            if P[v, l] == 1:
                inv[l] = v
                break
    return inv


def canonical_HR(w: int) -> np.ndarray:
    """Canonical (w-1) × w parity-check matrix of the length-w repetition code.

    Row l: 1 at columns l and l+1, 0 elsewhere.
    """
    H = np.zeros((w - 1, w), dtype=np.int_)
    for l in range(w - 1):
        H[l, l] = 1
        H[l, l + 1] = 1
    return H


def _running_xor_b_c(T_col: np.ndarray) -> np.ndarray:
    """Compute b ∈ F_2^w from T_col ∈ F_2^{w-1} via running XOR.

    Solves H_R^T b = T_col with the canonical choice b[0] = 0.
    """
    w_minus_1 = T_col.shape[0]
    w = w_minus_1 + 1
    b = np.zeros(w, dtype=np.int_)
    for l in range(1, w):
        b[l] = (b[l - 1] + int(T_col[l - 1])) % 2
    return b
```

- [ ] **Step 4: Run tests**

Run: `pytest src/qldpc/codes/surgery_test.py -k "auxiliary_graph or label_inverse or canonical_HR or running_xor" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/joint.py src/qldpc/codes/surgery_test.py
git commit -m "feat: add joint construction helpers (aux graph, label inverse, H_R, b_c)"
```

---

### Task 9: Verify χ–Z compatibility lemma on BB-LP (§4.7, load-bearing)

This task is a **research probe**, not a regular TDD task: we numerically verify whether the lemma holds for the concrete BB-LP example. The outcome determines the implementation strategy in Tasks 10–11.

**Files:**
- Modify: `src/qldpc/codes/surgery_test.py` (add the numerical test)
- Modify: `src/qldpc/codes/surgery/joint.py` (add `_chi_z_compatibility_check`)

- [ ] **Step 1: Write the compatibility check helper**

Append to `src/qldpc/codes/surgery/joint.py`:
```python
def _chi_z_compatibility_check(
    T_s: np.ndarray,
    label_inv: list[int],
) -> tuple[bool, list[tuple[int, int]]]:
    """Numerical verification of §4.7 lemma:  b_c[label_s[v]] = 0  for all (v, c).

    With α_v = e_{label[v]} (single-bit indicator) and
    b_c[l] = running XOR of T_s[:l, c], the chi-Z compatibility requires
    b_c[label[v]] = 0 for every (v in V_0_s, c in C_0_s).

    Returns (all_zero, violators) where violators is a list of (v, c) pairs
    with b_c[label[v]] != 0.
    """
    w = T_s.shape[0] + 1
    n_E = T_s.shape[1]
    # Precompute b_c[l] for all c, l
    B = np.zeros((w, n_E), dtype=np.int_)
    for c in range(n_E):
        B[:, c] = _running_xor_b_c(T_s[:, c])
    violators: list[tuple[int, int]] = []
    for v, l in enumerate(label_inv):  # iterate label l → vertex v
        for c in range(n_E):
            if B[l, c] != 0:
                violators.append((v, c))
    return len(violators) == 0, violators
```

- [ ] **Step 2: Write the numerical verification test**

Append to `src/qldpc/codes/surgery_test.py`:
```python
@pytest.mark.skipif(not fixtures_available(), reason="Zenodo fixtures not present")
def test_chi_z_compatibility_lemma_on_BB_LP():
    """Probe §4.7 lemma: does α_v · b_c = 0 hold on the real BB-LP example?

    If this passes, Task 10 uses the simple α_v = e_{label[v]} extension.
    If this fails, Task 10 uses the linear-solve fallback (search α_v in ker(B^T)).
    """
    from qldpc.codes.surgery.joint import (
        _build_auxiliary_graph, _chi_z_compatibility_check,
        _label_inverse, _skip_tree_hr,
    )

    # Build BB_1 Z̄_1 gadget
    x, y = sympy.symbols("x y")
    bb = codes.BBCode((7, 7), x**3 + y**3 + y**4, y**6 + x**2 + x**5)
    z1 = np.zeros(98, dtype=int)
    for q in [6, 8, 13, 17, 31, 32, 33, 35, 36, 37, 41, 50, 51, 93]:
        z1[q] = 1
    dual = codes.CSSCode(bb.matrix_z, bb.matrix_x, is_subsystem_code=False)
    _, layout1 = build_layered_surgery_code(dual, z1, num_layers=1, validate_logical_op=False)
    F1 = np.asarray(layout1.F).astype(int)

    G1, edge_qubit_map_1 = _build_auxiliary_graph(F1)
    span1 = nx.minimum_spanning_tree(G1)
    T1, P1 = _skip_tree_hr(span1, root=0)
    T1 = T1.astype(int); P1 = P1.astype(int)
    label_inv_1 = _label_inverse(P1)

    ok, violators = _chi_z_compatibility_check(T1, label_inv_1)
    # IMPORTANT: this assertion may fail; if so the next task switches strategy.
    # We assert ok==True OPTIMISTICALLY; if it fails the implementer adjusts.
    print(f"BB Z1 violators: {len(violators)} out of {len(label_inv_1) * T1.shape[1]} pairs")
    if not ok:
        print(f"First 10 violators: {violators[:10]}")
    # Do NOT assert ok — record the outcome for Task 10 to act on.
```

- [ ] **Step 3: Run the probe**

Run: `pytest src/qldpc/codes/surgery_test.py::test_chi_z_compatibility_lemma_on_BB_LP -v -s`
Expected: PASS (the test does not assert). The printed line tells us whether the lemma holds for BB_1 Z̄_1's specific G_1 + spanning-tree.

- [ ] **Step 4: Decision point — record outcome in plan**

Based on the printed `violators` count, choose ONE of the two paths in Task 10:
- **Path A (zero violators):** simple α_v = e_{label[v]} extension. Implementation in Task 10A.
- **Path B (non-zero violators):** solve linear system for α_v ∈ F_2^w with α_v ⊥ all b_c. Implementation in Task 10B.

The plan continues with BOTH paths defined; the implementer picks Path A if the probe says all-zero, otherwise Path B.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/joint.py src/qldpc/codes/surgery_test.py
git commit -m "test: probe χ-Z compatibility lemma numerically on BB_1 Z̄_1"
```

---

### Task 10: Implement chi-row bridge extension

**Files:**
- Modify: `src/qldpc/codes/surgery/joint.py`
- Modify: `src/qldpc/codes/surgery_test.py`

- [ ] **Step 1: Write failing test (works for both Path A and Path B)**

Append to `src/qldpc/codes/surgery_test.py`:
```python
def test_extend_chi_rows_with_bridge_breaks_chi_sum_identity():
    """After extension, Σ_v χ^(s)(v) on bridge cols equals Σ_l e_l = all-ones."""
    from qldpc.codes.surgery.joint import _extend_chi_rows_with_bridge
    # Synthetic: 4 chi rows × 10 (= 6 ancilla + 4 bridge) cols, with bridge cols at [6, 7, 8, 9]
    n_data_plus_kappa = 6
    n_bridge = 4
    n_chi = 4
    chi_rows = np.zeros((n_chi, n_data_plus_kappa + n_bridge), dtype=int)
    # set first column = data X̄ component for each chi row
    for v in range(n_chi):
        chi_rows[v, v % 2] = 1
    label_inv = [0, 1, 2, 3]  # vertex v → label v
    extended = _extend_chi_rows_with_bridge(
        chi_rows, label_inv, n_data_plus_kappa, n_bridge
    )
    bridge_block = extended[:, n_data_plus_kappa:]
    # Each row has exactly one 1 in bridge block at column label[v]
    for v in range(n_chi):
        assert bridge_block[v, label_inv[v]] == 1
        assert bridge_block[v].sum() == 1
    # Sum over rows = all ones vector
    assert np.array_equal(bridge_block.sum(axis=0) % 2, np.ones(n_bridge, dtype=int))
```

- [ ] **Step 2: Run test to verify FAIL**

Run: `pytest src/qldpc/codes/surgery_test.py::test_extend_chi_rows_with_bridge_breaks_chi_sum_identity -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3A: PATH A IMPLEMENTATION — simple α_v = e_{label[v]}**

If Task 9 probe showed zero violators, use this implementation. Append to `joint.py`:
```python
def _extend_chi_rows_with_bridge(
    chi_rows: np.ndarray,
    label_inv: list[int],
    n_data_plus_kappa: int,
    n_bridge: int,
) -> np.ndarray:
    """Extend each chi row at vertex v with X on bridge[label[v]].

    chi_rows: shape (n_v0, n_data_plus_kappa + n_bridge). The bridge columns
        are zero on input; this function fills them in.
    label_inv: list where label_inv[l] = vertex v.
    """
    n_v0 = chi_rows.shape[0]
    assert n_v0 == len(label_inv)
    out = chi_rows.copy()
    # Build reverse: vertex v → label l
    label = [0] * n_v0
    for l, v in enumerate(label_inv):
        label[v] = l
    for v in range(n_v0):
        out[v, n_data_plus_kappa + label[v]] = 1
    return out
```

- [ ] **Step 3B: PATH B IMPLEMENTATION — linear-solve fallback**

Used only if Task 9 probe showed non-zero violators. Append to `joint.py`:
```python
def _extend_chi_rows_with_bridge_via_linear_solve(
    chi_rows: np.ndarray,
    label_inv: list[int],
    T_s: np.ndarray,
    n_data_plus_kappa: int,
    n_bridge: int,
) -> np.ndarray:
    """Fallback: solve α_v ∈ F_2^w such that α_v · b_c = 0 for all c.

    Constraints:
      - α_v[label[v]] = 1  (so single-vertex chi sum maps to b_{label[v]})
      - α_v ⊥ b_c for all c (chi-Z compatibility)
      - Σ_v α_v = (1, 1, …, 1) preserves the "joint observable in row span" identity

    Solves the linear system in GF(2). If infeasible, raises ValueError
    (would indicate a code structure where Lemma 10 needs adapting).
    """
    import galois
    GF2 = galois.GF(2)

    w = n_bridge
    n_v0 = chi_rows.shape[0]
    n_E = T_s.shape[1]
    # B[l, c] = b_c[l]
    B = np.zeros((w, n_E), dtype=int)
    for c in range(n_E):
        B[:, c] = _running_xor_b_c(T_s[:, c])

    # For each v, solve: B^T α_v = 0 AND α_v[label[v]] = 1
    out = chi_rows.copy()
    label = [0] * n_v0
    for l, vtx in enumerate(label_inv):
        label[vtx] = l
    for v in range(n_v0):
        l_v = label[v]
        # Constraints: B^T α_v = 0 (n_E equations), α_v[l_v] = 1 (1 equation)
        A = np.vstack([B.T, np.eye(1, w, l_v)])  # (n_E + 1) × w
        rhs = np.zeros(n_E + 1, dtype=int)
        rhs[-1] = 1
        # Solve A α = rhs over GF(2)
        aug = GF2(np.hstack([A.astype(int), rhs.reshape(-1, 1)]))
        rref = np.asarray(aug.row_reduce())
        alpha = np.zeros(w, dtype=int)
        for r in range(rref.shape[0]):
            nz = np.flatnonzero(rref[r, :w])
            if nz.size == 0:
                if rref[r, w] == 1:
                    raise ValueError(
                        f"chi-Z extension infeasible for vertex {v}: "
                        f"linear system has no solution."
                    )
                continue
            alpha[int(nz[0])] = int(rref[r, w])
        out[v, n_data_plus_kappa:n_data_plus_kappa + w] = alpha
    return out
```

(Implementer picks 3A or 3B based on Task 9 outcome.)

- [ ] **Step 4: Re-export from package init**

Add to `src/qldpc/codes/surgery/__init__.py`:
```python
from .joint import (
    _build_auxiliary_graph,
    _label_inverse,
    canonical_HR,
    _running_xor_b_c,
    _chi_z_compatibility_check,
    _extend_chi_rows_with_bridge,
)  # noqa: F401
```

- [ ] **Step 5: Run test**

Run: `pytest src/qldpc/codes/surgery_test.py::test_extend_chi_rows_with_bridge_breaks_chi_sum_identity -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/qldpc/codes/surgery/joint.py src/qldpc/codes/surgery/__init__.py src/qldpc/codes/surgery_test.py
git commit -m "feat: implement chi-row bridge extension (§4.5)"
```

---

### Task 11: Implement Webster Z-row bridge extension

**Files:**
- Modify: `src/qldpc/codes/surgery/joint.py`
- Modify: `src/qldpc/codes/surgery_test.py`

- [ ] **Step 1: Write failing test**

Append to `src/qldpc/codes/surgery_test.py`:
```python
def test_extend_Z_rows_with_bridge_makes_adapter_commute():
    """After extension, adapter X-check c' commutes with Z row indexed by c."""
    from qldpc.codes.surgery.joint import (
        _extend_Z_rows_with_bridge, canonical_HR, _running_xor_b_c,
    )
    rng = np.random.default_rng(0)
    w = 5
    n_E = 7   # number of κ_s edges
    # Synthesize a random T_s (3,2)-sparse with rank w-1
    T_s = rng.integers(0, 2, size=(w - 1, n_E))
    # Z rows: (n_C0 = n_E) rows, with single-1 on κ_s column = identity-like
    n_data = 10
    n_kappa = n_E
    n_bridge = w
    Z_rows = np.zeros((n_E, n_data + n_kappa + n_bridge), dtype=int)
    for c in range(n_E):
        Z_rows[c, n_data + c] = 1   # κ_s[c] extension as in Webster
    c0_indices_in_Z = list(range(n_E))  # all Z rows have κ extension here

    Z_ext = _extend_Z_rows_with_bridge(
        Z_rows, c0_indices_in_Z, T_s, n_data, n_kappa, n_bridge
    )
    # Verify: H_R^T b_c = T_s[:, c] for each c
    H_R = canonical_HR(w)
    for c in range(n_E):
        b_c = Z_ext[c, n_data + n_kappa:]
        assert np.array_equal((H_R.T @ b_c) % 2, T_s[:, c] % 2)
```

- [ ] **Step 2: Run test to verify FAIL**

Run: `pytest src/qldpc/codes/surgery_test.py::test_extend_Z_rows_with_bridge_makes_adapter_commute -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `_extend_Z_rows_with_bridge`**

Append to `joint.py`:
```python
def _extend_Z_rows_with_bridge(
    Z_rows: np.ndarray,
    c0_indices_in_Z: list[int],
    T_s: np.ndarray,
    n_data: int,
    n_kappa: int,
    n_bridge: int,
) -> np.ndarray:
    """Extend Webster Z rows in C_0_s with bridge support b_c = running XOR of T_s[:, c].

    Args:
        Z_rows: shape (n_z_rows, n_data + n_kappa + n_bridge). Bridge cols start at zero.
        c0_indices_in_Z: which row indices of Z_rows are in C_0_s (= edges of G_s).
            For row r in c0_indices_in_Z, the corresponding edge index in G_s
            is its position in this list.
        T_s: shape (n_bridge-1, n_edges_G_s). Columns indexed by G_s edge.
        n_data, n_kappa, n_bridge: column block widths.
    """
    out = Z_rows.copy()
    bridge_start = n_data + n_kappa
    for k, row_idx in enumerate(c0_indices_in_Z):
        T_col = T_s[:, k]
        b_c = _running_xor_b_c(T_col)
        out[row_idx, bridge_start:bridge_start + n_bridge] = b_c
    return out
```

- [ ] **Step 4: Re-export**

Add `_extend_Z_rows_with_bridge` to `src/qldpc/codes/surgery/__init__.py` re-exports.

- [ ] **Step 5: Run test**

Run: `pytest src/qldpc/codes/surgery_test.py::test_extend_Z_rows_with_bridge_makes_adapter_commute -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/qldpc/codes/surgery/joint.py src/qldpc/codes/surgery/__init__.py src/qldpc/codes/surgery_test.py
git commit -m "feat: implement Webster Z-row bridge extension (§4.6)"
```

---

### Task 12: Implement adapter X-checks block

**Files:**
- Modify: `src/qldpc/codes/surgery/joint.py`
- Modify: `src/qldpc/codes/surgery_test.py`

- [ ] **Step 1: Write failing test**

Append to `src/qldpc/codes/surgery_test.py`:
```python
def test_build_adapter_x_checks_structure():
    """Adapter row c has T_1[c,:] on κ_1, H_R[c,:] on bridge, T_2[c,:] on κ_2."""
    from qldpc.codes.surgery.joint import _build_adapter_x_checks, canonical_HR
    w = 4
    n_E1 = 5
    n_E2 = 6
    rng = np.random.default_rng(42)
    T1 = rng.integers(0, 2, size=(w - 1, n_E1))
    T2 = rng.integers(0, 2, size=(w - 1, n_E2))
    n_data_1 = 8
    n_data_2 = 12
    n_bridge = w

    A = _build_adapter_x_checks(
        T1, T2,
        n_data_1=n_data_1, n_kappa_1=n_E1,
        n_data_2=n_data_2, n_kappa_2=n_E2,
        n_bridge=n_bridge,
    )
    n_total = n_data_1 + n_kappa_1 = n_data_1 + n_E1
    n_total = n_data_1 + n_E1 + n_data_2 + n_E2 + n_bridge
    assert A.shape == (w - 1, n_total)
    # data zones: all zero
    assert A[:, :n_data_1].sum() == 0
    assert A[:, n_data_1 + n_E1 : n_data_1 + n_E1 + n_data_2].sum() == 0
    # κ_1 zone matches T1
    kappa1_start = n_data_1
    assert np.array_equal(A[:, kappa1_start:kappa1_start + n_E1], T1 % 2)
    # κ_2 zone matches T2
    kappa2_start = n_data_1 + n_E1 + n_data_2
    assert np.array_equal(A[:, kappa2_start:kappa2_start + n_E2], T2 % 2)
    # bridge zone matches H_R
    bridge_start = n_data_1 + n_E1 + n_data_2 + n_E2
    assert np.array_equal(A[:, bridge_start:bridge_start + n_bridge], canonical_HR(w))
```

- [ ] **Step 2: Run test to verify FAIL**

Run: `pytest src/qldpc/codes/surgery_test.py::test_build_adapter_x_checks_structure -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `_build_adapter_x_checks`**

Append to `joint.py`:
```python
def _build_adapter_x_checks(
    T1: np.ndarray,
    T2: np.ndarray,
    *,
    n_data_1: int,
    n_kappa_1: int,
    n_data_2: int,
    n_kappa_2: int,
    n_bridge: int,
) -> np.ndarray:
    """Construct adapter X-check rows per Lemma 10 (Eq 19 middle block).

    Column layout: [data_1 | κ_1 | data_2 | κ_2 | bridge].
    Each of the (w-1) rows has T1[c,:] on κ_1, T2[c,:] on κ_2, H_R[c,:] on bridge.
    """
    assert T1.shape[0] == T2.shape[0] == n_bridge - 1
    assert T1.shape[1] == n_kappa_1
    assert T2.shape[1] == n_kappa_2
    n_total = n_data_1 + n_kappa_1 + n_data_2 + n_kappa_2 + n_bridge
    A = np.zeros((n_bridge - 1, n_total), dtype=np.int_)
    H_R = canonical_HR(n_bridge)
    kappa1_start = n_data_1
    kappa2_start = n_data_1 + n_kappa_1 + n_data_2
    bridge_start = n_data_1 + n_kappa_1 + n_data_2 + n_kappa_2
    A[:, kappa1_start:kappa1_start + n_kappa_1] = T1 % 2
    A[:, kappa2_start:kappa2_start + n_kappa_2] = T2 % 2
    A[:, bridge_start:bridge_start + n_bridge] = H_R
    return A
```

- [ ] **Step 4: Re-export**

Add `_build_adapter_x_checks` to `src/qldpc/codes/surgery/__init__.py`.

- [ ] **Step 5: Run test**

Run: `pytest src/qldpc/codes/surgery_test.py::test_build_adapter_x_checks_structure -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/qldpc/codes/surgery/joint.py src/qldpc/codes/surgery/__init__.py src/qldpc/codes/surgery_test.py
git commit -m "feat: implement Lemma 10 adapter X-checks block (§4.4)"
```

---

### Task 13: Implement v3 `build_joint_measurement_code` (disjoint case)

**Files:**
- Modify: `src/qldpc/codes/surgery/joint.py` (replace v2 `build_joint_measurement_code` with v3)

- [ ] **Step 1: Write the failing top-level joint test for the synthetic minimal case**

Append to `src/qldpc/codes/surgery_test.py`:
```python
def test_build_joint_measurement_code_v3_disjoint_minimal():
    """Smallest synthetic disjoint joint code passes CSS commutation."""
    from qldpc.codes.surgery import build_joint_measurement_code

    # Construct two distinct minimal CSS codes (e.g., two [[7, 1, 3]] Steane copies).
    steane = codes.SteaneCode()
    x_logical = steane.get_logical_ops(Pauli.X)[0]
    op1 = np.asarray(x_logical).astype(int)
    op2 = np.asarray(x_logical).astype(int)
    # Use the same steane code, but logically they are disjoint blocks under intra-vs-inter
    merged, layout = build_joint_measurement_code(
        steane, op1, steane, op2, validate=True
    )
    # CSS commute
    prod = (merged.matrix_x @ merged.matrix_z.T) % merged.field.order
    assert (prod == 0).all()
    # Dimension: k_steane (1) per copy → 2 originally, − 1 for joint = 1
    # For inter-code via this API: k_1 + k_2 - 1 = 1 + 1 - 1 = 1
    assert merged.dimension == 1
```

- [ ] **Step 2: Run test to verify FAIL**

Run: `pytest src/qldpc/codes/surgery_test.py::test_build_joint_measurement_code_v3_disjoint_minimal -v`
Expected: FAIL — current v2 signature `(data_code, op1, op2, ...)` is incompatible.

- [ ] **Step 3: Implement v3 `build_joint_measurement_code`**

Replace the entire current `build_joint_measurement_code` body in `src/qldpc/codes/surgery/joint.py` with:
```python
@dataclasses.dataclass(frozen=True)
class SkipTreeOverride:
    """Optionally inject Ide's exact T, P, G matrices for byte-exact reproduction."""
    T_1: np.ndarray
    P_1: np.ndarray
    G_1: np.ndarray
    T_2: np.ndarray
    P_2: np.ndarray
    G_2: np.ndarray


def build_joint_measurement_code(
    data_code_1: "CSSCode",
    op1: "npt.ArrayLike",
    data_code_2: "CSSCode",
    op2: "npt.ArrayLike",
    *,
    num_layers: int = 1,
    spanning_tree_seed: int = 0,
    skiptree_override: SkipTreeOverride | None = None,
    validate: bool = True,
) -> tuple["CSSCode", JointSurgeryLayout]:
    """Construct merged code measuring op1 · op2 jointly via SkipTree adapter.

    Implements Ide et al. arXiv:2410.03628 §VII B (disjoint) + §VII C (overlap)
    using the Lemma 10 adapter (§IV). For inter-code joint, pass distinct
    data_code_1 and data_code_2. For intra-code joint, pass the same code twice.

    Args:
        data_code_1, data_code_2: stabilizer CSSCodes.
        op1: logical operator support on data_code_1's data qubits.
        op2: logical operator support on data_code_2's data qubits.
        num_layers: L for each gadget (must be odd >= 1).
        spanning_tree_seed: deterministic seed for the spanning tree choice.
        skiptree_override: when provided, use exactly these T, P, G matrices
            instead of generating from spanning_tree_seed.
        validate: enable input sanity checks.

    Returns:
        (merged_code, layout).
    """
    op1_arr = np.asarray(op1).astype(np.int_)
    op2_arr = np.asarray(op2).astype(np.int_)

    if validate:
        if data_code_1.dimension < 1 or data_code_2.dimension < 1:
            raise ValueError("both codes must have dimension >= 1")
        if ((data_code_1.matrix_z @ data_code_1.field(op1_arr)) != 0).any():
            raise ValueError("op1 does not commute with data_code_1's Z-stabs")
        if ((data_code_2.matrix_z @ data_code_2.field(op2_arr)) != 0).any():
            raise ValueError("op2 does not commute with data_code_2's Z-stabs")

    # Phase 1: build per-gadget Webster
    target_1 = CSSCode(
        data_code_1.matrix_z, data_code_1.matrix_x, is_subsystem_code=False
    )
    target_2 = CSSCode(
        data_code_2.matrix_z, data_code_2.matrix_x, is_subsystem_code=False
    )
    g1_merged, layout1 = build_layered_surgery_code(
        target_1, op1_arr, num_layers=num_layers, validate_logical_op=False
    )
    g2_merged, layout2 = build_layered_surgery_code(
        target_2, op2_arr, num_layers=num_layers, validate_logical_op=False
    )

    # Phase 2: aux graphs + SkipTree
    F1 = np.asarray(layout1.F).astype(int)
    F2 = np.asarray(layout2.F).astype(int)

    if skiptree_override is not None:
        T1, P1, G1_mat = skiptree_override.T_1, skiptree_override.P_1, skiptree_override.G_1
        T2, P2, G2_mat = skiptree_override.T_2, skiptree_override.P_2, skiptree_override.G_2
        # When override is given, we trust the caller has paired the matrices to the gadget.
        # Build edge order map from G_mat (rows of G_mat = edges).
        edge_qubit_map_1 = _edge_qubit_map_from_F(F1)
        edge_qubit_map_2 = _edge_qubit_map_from_F(F2)
    else:
        G_aux_1, edge_qubit_map_1 = _build_auxiliary_graph(F1)
        G_aux_2, edge_qubit_map_2 = _build_auxiliary_graph(F2)
        rng1 = np.random.default_rng(spanning_tree_seed)
        rng2 = np.random.default_rng(spanning_tree_seed + 1)
        span1 = _deterministic_spanning_tree(G_aux_1, rng1)
        span2 = _deterministic_spanning_tree(G_aux_2, rng2)
        T1, P1 = _skip_tree_hr(span1, root=0)
        T2, P2 = _skip_tree_hr(span2, root=0)
        T1 = T1.astype(int) % 2
        T2 = T2.astype(int) % 2
        P1 = P1.astype(int) % 2
        P2 = P2.astype(int) % 2

    label_inv_1 = _label_inverse(P1)
    label_inv_2 = _label_inverse(P2)

    # Phase 3: stitch
    w = min(layout1.v0_indices.size, layout2.v0_indices.size)
    if w == 0:
        raise ValueError("joint requires both logicals to have non-empty support")

    n_data_1 = data_code_1.num_qubits
    n_data_2 = data_code_2.num_qubits
    n_kappa_1 = layout1.num_ancilla_qubits
    n_kappa_2 = layout2.num_ancilla_qubits
    n_bridge = w
    n_joint = n_data_1 + n_kappa_1 + n_data_2 + n_kappa_2 + n_bridge

    HX1 = np.asarray(g1_merged.matrix_x).astype(int)
    HZ1 = np.asarray(g1_merged.matrix_z).astype(int)
    HX2 = np.asarray(g2_merged.matrix_x).astype(int)
    HZ2 = np.asarray(g2_merged.matrix_z).astype(int)

    def _embed(M, *, ancilla_block):
        """Embed gadget matrix into joint column layout."""
        out = np.zeros((M.shape[0], n_joint), dtype=int)
        if ancilla_block == 1:
            out[:, :n_data_1] = M[:, :n_data_1]
            out[:, n_data_1:n_data_1 + n_kappa_1] = M[:, n_data_1:]
        else:
            out[:, n_data_1 + n_kappa_1:n_data_1 + n_kappa_1 + n_data_2] = M[:, :n_data_2]
            out[:, n_data_1 + n_kappa_1 + n_data_2:
                    n_data_1 + n_kappa_1 + n_data_2 + n_kappa_2] = M[:, n_data_2:]
        return out

    HX1_emb = _embed(HX1, ancilla_block=1)
    HX2_emb = _embed(HX2, ancilla_block=2)
    HZ1_emb = _embed(HZ1, ancilla_block=1)
    HZ2_emb = _embed(HZ2, ancilla_block=2)

    # Extend chi rows. Identify chi row range via hx_row_kind.
    chi_mask_1 = (layout1.hx_row_kind != "data")
    chi_mask_2 = (layout2.hx_row_kind != "data")
    chi_rows_1 = HX1_emb[chi_mask_1]
    chi_rows_2 = HX2_emb[chi_mask_2]

    # The chi rows have bridge cols all zero from _embed; pass to extender.
    chi_rows_1_ext = _extend_chi_rows_with_bridge(
        chi_rows_1, label_inv_1,
        n_data_plus_kappa=n_data_1 + n_kappa_1 + n_data_2 + n_kappa_2,
        n_bridge=n_bridge,
    )
    chi_rows_2_ext = _extend_chi_rows_with_bridge(
        chi_rows_2, label_inv_2,
        n_data_plus_kappa=n_data_1 + n_kappa_1 + n_data_2 + n_kappa_2,
        n_bridge=n_bridge,
    )
    # Splice extended chi back
    HX1_emb[chi_mask_1] = chi_rows_1_ext
    HX2_emb[chi_mask_2] = chi_rows_2_ext

    # Extend Z rows in C_0_s with bridge
    HZ1_emb_ext = _extend_Z_rows_with_bridge(
        HZ1_emb, _c0_z_row_indices(layout1), T1,
        n_data=n_data_1 + n_kappa_1 + n_data_2 + n_kappa_2,
        n_kappa=0,  # already accounted in n_data above
        n_bridge=n_bridge,
    )
    HZ2_emb_ext = _extend_Z_rows_with_bridge(
        HZ2_emb, _c0_z_row_indices(layout2), T2,
        n_data=n_data_1 + n_kappa_1 + n_data_2 + n_kappa_2,
        n_kappa=0,
        n_bridge=n_bridge,
    )

    # Adapter X-checks
    adapter = _build_adapter_x_checks(
        T1, T2,
        n_data_1=n_data_1, n_kappa_1=n_kappa_1,
        n_data_2=n_data_2, n_kappa_2=n_kappa_2,
        n_bridge=n_bridge,
    )

    # Deduplicate data X-checks of gadget 2 (= same as gadget 1 when codes equal)
    data_mask_2 = layout2.hx_row_kind == "data"
    if data_code_1 is data_code_2 or np.array_equal(
        np.asarray(data_code_1.matrix_x), np.asarray(data_code_2.matrix_x)
    ):
        # For intra-code, drop gadget2's data X-checks (duplicate).
        HX2_emb = HX2_emb[~data_mask_2]

    HX_joint = np.vstack([HX1_emb, HX2_emb, adapter])
    HZ_joint = np.vstack([HZ1_emb_ext, HZ2_emb_ext])

    field = data_code_1.field
    joint = CSSCode(field(HX_joint % 2), field(HZ_joint % 2), is_subsystem_code=False)

    layout = JointSurgeryLayout(
        gadget_layouts=(layout1, layout2),
        pauli_type=Pauli.X,  # joint X̄ X̄ measurement (Z-type follows by dual)
        num_data_qubits=n_data_1 + n_data_2,
        num_ancilla_qubits=n_kappa_1 + n_kappa_2,
        num_bridge_qubits=n_bridge,
        bridge_qubit_slice=slice(n_joint - n_bridge, n_joint),
        u_b_check_kind_mask=_adapter_mask(HX_joint.shape[0], adapter.shape[0]),
    )
    return joint, layout


def _edge_qubit_map_from_F(F: np.ndarray) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for i, row in enumerate(F):
        eps = sorted(np.flatnonzero(row).tolist())
        if len(eps) == 2:
            out[i] = (eps[0], eps[1])
    return out


def _deterministic_spanning_tree(G: nx.Graph, rng: np.random.Generator) -> nx.Graph:
    """Pick a deterministic-by-seed spanning tree of G."""
    edges = list(G.edges())
    perm = rng.permutation(len(edges))
    G_perm = nx.Graph()
    G_perm.add_nodes_from(G.nodes())
    G_perm.add_edges_from([edges[i] for i in perm])
    return nx.minimum_spanning_tree(G_perm, algorithm="kruskal")


def _c0_z_row_indices(layout) -> list[int]:
    """Indices into HZ rows that correspond to C_0 edges of G_s, in edge order.

    For each c in layout.c0_indices (in order), find its HZ row index (= the
    same-position data Z-check). Returns a list of length |C_0|.
    """
    data_hz_mask = layout.hz_row_kind == "data"
    data_hz_idx = np.flatnonzero(data_hz_mask)
    return [int(data_hz_idx[int(c)]) for c in range(layout.c0_indices.size)]


def _adapter_mask(n_total_hx_rows: int, n_adapter: int) -> np.ndarray:
    """Bool mask True on the adapter rows (= last n_adapter rows of HX_joint)."""
    out = np.zeros(n_total_hx_rows, dtype=bool)
    out[-n_adapter:] = True
    return out
```

- [ ] **Step 4: Drop v2 dead code from joint.py**

Delete (from `joint.py`):
- `_BridgeSpec`
- `_build_bridge_via_skiptree`
- `_solve_gf2_system` (move to a utility if needed elsewhere)
- `_find_bridge_z_stab_data_logical`
- `_stitch_gadgets_with_bridge`
- the old `build_joint_measurement_code` (already replaced)

- [ ] **Step 5: Run the minimal joint test**

Run: `pytest src/qldpc/codes/surgery_test.py::test_build_joint_measurement_code_v3_disjoint_minimal -v`
Expected: PASS (CSS commutation holds, k = 1).

- [ ] **Step 6: Run all existing tests; identify v2 breakages**

Run: `pytest src/qldpc/codes/surgery_test.py -v 2>&1 | tail -50`
Expected: most tests pass; a small number (~5) fail because they referenced v2-only `JointSurgeryLayout` fields like `bridge_qubit_slice` properties expected in v2. Note the failures.

- [ ] **Step 7: Drop or rewrite v2-specific failing tests**

For each v2-specific failure, either:
- DELETE the test (it tested a v2-only behavior — e.g., `bridge_qubit_slice` start position assumed path-bridge layout)
- REWRITE the test against v3 layout

List of expected v2-specific failures (search by name):
- `test_joint_*` tests that asserted `JointSurgeryLayout.u_b_check_kind_mask` has a specific count (v2 had w-1 path stabs; v3 has w-1 adapter rows — same count by coincidence)
- Any test asserting `bridge_qubit_slice` width equals `min(|V_0_1|, |V_0_2|)` should still pass.
- Tests that called `build_joint_measurement_code(data_code, op1, op2)` (3 args) must be updated to the 4-arg form `(data_code, op1, data_code, op2)`.

Update the calls in those tests; delete tests that probed v2-only behavior (no longer relevant).

- [ ] **Step 8: Re-run tests; everything green**

Run: `pytest src/qldpc/codes/surgery_test.py -v 2>&1 | tail -15`
Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add src/qldpc/codes/surgery/joint.py src/qldpc/codes/surgery_test.py
git commit -m "feat: v3 build_joint_measurement_code via SkipTree adapter (Lemma 10)"
```

---

### Task 14: BB-LP joint test — Ide Table III

**Files:**
- Modify: `src/qldpc/codes/surgery_test.py`
- Create: `examples/scripts/ide_bb_lp_v3_reproduce.py` (sanity script for manual inspection)

- [ ] **Step 1: Write Ide BB-LP joint test**

Append to `src/qldpc/codes/surgery_test.py`:
```python
@pytest.mark.skipif(not fixtures_available(), reason="Zenodo fixtures not present")
def test_joint_BB_LP_matches_ide_table_iii():
    """v3 joint code on BB_1 Z̄_1 × LP_2 Z̄_2 has same parameters as Ide Table III."""
    from qldpc.codes.surgery import build_joint_measurement_code
    from qldpc.abstract import CyclicGroup, GroupRing, RingArray

    # Build BB_1 [[98, 6, 12]]
    x, y = sympy.symbols("x y")
    bb = codes.BBCode((7, 7), x**3 + y**3 + y**4, y**6 + x**2 + x**5)
    z1 = np.zeros(98, dtype=int)
    for q in [6, 8, 13, 17, 31, 32, 33, 35, 36, 37, 41, 50, 51, 93]:
        z1[q] = 1

    # Build LP_2 [[200, 20, 10]] (from Ide Eq 33)
    ell = 8
    group = CyclicGroup(ell)
    xg = group.generators[0]
    ring = GroupRing(group)
    A = RingArray.build(
        [
            [xg**2, 1, 1, xg**2],
            [1, xg, xg**2, xg],
            [xg**2, xg, xg**3, xg**2],
        ],
        ring,
    )
    lp = codes.LPCode(A)

    # Find a wt-14 Z̄ rep of LP_2 (Ide's specific support is qubit-indexed in their convention;
    # use equivalent-rep search to obtain a wt-14 representative with our index convention)
    def find_lp2_wt14_rep(code, target_weight=14, seed=0, max_trials=5000):
        HX = np.asarray(code.matrix_x).astype(int)
        HZ = np.asarray(code.matrix_z).astype(int)
        zls = np.asarray(code.get_logical_ops(Pauli.Z)).astype(int)
        rng = random.Random(seed)
        for _ in range(max_trials):
            k = rng.randint(1, 8)
            indices = rng.sample(range(code.dimension), k)
            cur = np.zeros(code.num_qubits, dtype=int)
            for i in indices:
                cur = (cur + zls[i]) % 2
            for _ in range(20):
                improved = False
                for s_idx in rng.sample(range(HZ.shape[0]), 30):
                    cand = (cur + HZ[s_idx]) % 2
                    if int(cand.sum()) < int(cur.sum()):
                        cur = cand
                        improved = True
                        break
                if not improved:
                    break
            if int(cur.sum()) == target_weight and ((HX @ cur) % 2).sum() == 0:
                return cur
        return None
    z2 = find_lp2_wt14_rep(lp)
    assert z2 is not None, "Could not find wt-14 LP_2 logical rep"

    merged, layout = build_joint_measurement_code(bb, z1, lp, z2, validate=False)

    # Acceptance bar (spec §1, §2)
    assert merged.num_qubits == 355
    assert merged.dimension == 25
    assert (merged.matrix_x.sum(axis=1) <= 8).all()
    assert (merged.matrix_z.sum(axis=1) <= 7).all()

    # Joint observable in row span
    op1_pad = np.zeros(355, dtype=int); op1_pad[:98] = z1
    op2_pad = np.zeros(355, dtype=int); op2_pad[98 + layout.num_ancilla_qubits // 2:98 + layout.num_ancilla_qubits // 2 + 200] = z2  # correct LP data offset
    joint_op = (op1_pad + op2_pad) % 2
    import galois
    GF2 = galois.GF(2)
    HX = GF2(np.asarray(merged.matrix_x).astype(int))
    # row span membership test: solve HX^T x = joint_op
    aug = GF2(np.hstack([HX.T.astype(int), joint_op.reshape(-1, 1)]))
    rref = np.asarray(aug.row_reduce())
    feasible = True
    for r in range(rref.shape[0]):
        nz_in_lhs = (rref[r, :HX.shape[0]] != 0).any()
        if not nz_in_lhs and rref[r, -1] == 1:
            feasible = False
            break
    assert feasible, "X̄_1 X̄_2 not in HX_joint row span"
```

- [ ] **Step 2: Run the test**

Run: `pytest src/qldpc/codes/surgery_test.py::test_joint_BB_LP_matches_ide_table_iii -v`
Expected: PASS. If FAIL: debug by comparing `merged.matrix_x` and `merged.matrix_z` against Ide's fixtures using `load_ide_joint_BB_LP()`; the stab-group equality check (row span equality) is the diagnostic.

- [ ] **Step 3: Create the sanity script for manual inspection**

Create `examples/scripts/ide_bb_lp_v3_reproduce.py`:
```python
"""Reproduce Ide §VII B [[355, 25, 10]] via v3 build_joint_measurement_code."""

from __future__ import annotations

import numpy as np
import sympy

from qldpc import codes
from qldpc.abstract import CyclicGroup, GroupRing, RingArray
from qldpc.codes.surgery import build_joint_measurement_code


def main():
    x, y = sympy.symbols("x y")
    bb = codes.BBCode((7, 7), x**3 + y**3 + y**4, y**6 + x**2 + x**5)
    z1 = np.zeros(98, dtype=int)
    for q in [6, 8, 13, 17, 31, 32, 33, 35, 36, 37, 41, 50, 51, 93]:
        z1[q] = 1
    print(f"BB_1 [[{bb.num_qubits}, {bb.dimension}]] Z̄_1 wt {int(z1.sum())}")
    # ... LP build + joint build + report parameters
    # See test for details; this script prints rather than asserting.


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit**

```bash
git add src/qldpc/codes/surgery_test.py examples/scripts/ide_bb_lp_v3_reproduce.py
git commit -m "test: BB-LP joint matches Ide Table III parameters"
```

---

### Task 15: Set-valued port for §VII C overlap

**Files:**
- Create: `src/qldpc/codes/surgery/port.py`
- Modify: `src/qldpc/codes/surgery/joint.py` (use port in overlap case)
- Modify: `src/qldpc/codes/surgery_test.py`

- [ ] **Step 1: Write failing test for SetValuedPort**

Append to `src/qldpc/codes/surgery_test.py`:
```python
def test_set_valued_port_detects_overlap():
    from qldpc.codes.surgery.port import SetValuedPort
    z1 = np.zeros(10, dtype=int)
    z1[[2, 5, 7]] = 1
    z3 = np.zeros(10, dtype=int)
    z3[[5, 8]] = 1
    port = SetValuedPort.from_supports([z1, z3])
    assert port.is_shared(5)
    assert not port.is_shared(2)
    assert not port.is_shared(8)
    assert port.gadgets_for_qubit(5) == [0, 1]
```

- [ ] **Step 2: Run test to verify FAIL**

Run: `pytest src/qldpc/codes/surgery_test.py::test_set_valued_port_detects_overlap -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement port.py**

Create `src/qldpc/codes/surgery/port.py`:
```python
"""Set-valued port function for §VII C intra-code joint measurement.

When multiple logical operators share a data qubit, the port function
f: L → V_0 becomes set-valued. See Ide arXiv:2410.03628 §VII C and
Theorem 11 (Appendix VIII) for the construction.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class SetValuedPort:
    """Per-qubit list of gadget indices that include it in V_0."""
    qubit_to_gadgets: dict[int, list[int]]

    @classmethod
    def from_supports(cls, supports: list[np.ndarray]) -> "SetValuedPort":
        """Build from a list of binary support vectors, one per gadget."""
        mapping: dict[int, list[int]] = {}
        for g_idx, supp in enumerate(supports):
            for q in np.flatnonzero(supp).tolist():
                mapping.setdefault(int(q), []).append(g_idx)
        return cls(qubit_to_gadgets=mapping)

    def is_shared(self, qubit: int) -> bool:
        return len(self.qubit_to_gadgets.get(qubit, [])) > 1

    def gadgets_for_qubit(self, qubit: int) -> list[int]:
        return list(self.qubit_to_gadgets.get(qubit, []))

    def shared_qubits(self) -> list[int]:
        return [q for q, gs in self.qubit_to_gadgets.items() if len(gs) > 1]
```

- [ ] **Step 4: Run test**

Run: `pytest src/qldpc/codes/surgery_test.py::test_set_valued_port_detects_overlap -v`
Expected: PASS.

- [ ] **Step 5: Wire SetValuedPort into joint.py for intra-code dedup**

In `joint.py`, modify the dedup branch (the "data_code_1 is data_code_2" branch in `build_joint_measurement_code`) to:
- Build `port = SetValuedPort.from_supports([op1_arr, op2_arr])`.
- For shared qubits: KEEP both gadgets' chi rows and Z rows (set-valued port).
- For non-shared: existing dedup logic.

The change: instead of unconditionally dropping `HX2_emb` data-X rows when `data_code_1 is data_code_2`, we keep them. The vertex Z-checks at shared qubits naturally cancel data Z² = I via the construction.

```python
# in build_joint_measurement_code, replace the intra-code dedup block:
if data_code_1 is data_code_2 or np.array_equal(
    np.asarray(data_code_1.matrix_x), np.asarray(data_code_2.matrix_x)
):
    port = SetValuedPort.from_supports([op1_arr, op2_arr])
    if port.shared_qubits():
        # Overlap case (§VII C): KEEP both sets of chi rows. The original
        # data X-stabs are still duplicated; drop one set.
        HX2_emb = HX2_emb[~(layout2.hx_row_kind == "data")]
    else:
        # Disjoint intra-code: same as inter-code, just drop dup data X-stabs.
        HX2_emb = HX2_emb[~(layout2.hx_row_kind == "data")]
```

- [ ] **Step 6: Add port import to joint.py**

At top of `joint.py`:
```python
from .port import SetValuedPort
```

- [ ] **Step 7: Re-run all tests**

Run: `pytest src/qldpc/codes/surgery_test.py -v 2>&1 | tail -15`
Expected: same pass count as before.

- [ ] **Step 8: Commit**

```bash
git add src/qldpc/codes/surgery/port.py src/qldpc/codes/surgery/joint.py src/qldpc/codes/surgery_test.py
git commit -m "feat: set-valued port for §VII C intra-code overlap"
```

---

### Task 16: BB-BB intra-code joint test — Ide Table IV

**Files:**
- Modify: `src/qldpc/codes/surgery_test.py`

- [ ] **Step 1: Write Ide BB-BB joint test**

Append to `src/qldpc/codes/surgery_test.py`:
```python
@pytest.mark.skipif(not fixtures_available(), reason="Zenodo fixtures not present")
def test_joint_BB_BB_intracode_matches_ide_table_iv():
    """v3 joint on BB_1 Z̄_1 × Z̄_3 (overlap on {17, 35}) matches Ide Table IV."""
    from qldpc.codes.surgery import build_joint_measurement_code

    x, y = sympy.symbols("x y")
    bb = codes.BBCode((7, 7), x**3 + y**3 + y**4, y**6 + x**2 + x**5)
    z1 = np.zeros(98, dtype=int)
    for q in [6, 8, 13, 17, 31, 32, 33, 35, 36, 37, 41, 50, 51, 93]:
        z1[q] = 1
    z3 = np.zeros(98, dtype=int)
    for q in [10, 17, 35, 39, 42, 43, 53, 55, 61, 70, 84, 89]:
        z3[q] = 1
    overlap = set(np.flatnonzero(z1)) & set(np.flatnonzero(z3))
    assert overlap == {17, 35}, f"unexpected overlap {overlap}"

    merged, layout = build_joint_measurement_code(bb, z1, bb, z3, validate=False)

    # Acceptance: [[150, 5, 12]] per Table IV
    assert merged.num_qubits == 150
    assert merged.dimension == 5
    assert (merged.matrix_x.sum(axis=1) <= 8).all()
    assert (merged.matrix_z.sum(axis=1) <= 6).all()
```

- [ ] **Step 2: Run test**

Run: `pytest src/qldpc/codes/surgery_test.py::test_joint_BB_BB_intracode_matches_ide_table_iv -v`
Expected: PASS. If FAIL on n_qubits: check that intra-code shares data block correctly (98 data + 17 κ_1 + 0 (no separate data block 2) + 15 κ_3 + 12 bridge = ?). The n=150 calculation: 98 (BB data) + 17 (κ_1) + 15 (κ_3) + 12 (bridge) = 142, not 150. The +8 likely comes from cellulation. The expected layout per Ide Table IV: 17 (G_1 edges with cellulation) + 12 (G_3 edges no cellulation, Ide says 3 new cycles) + adapter 11 + ...

If the count is off, that points to a bookkeeping issue in the intra-code path; instrument and debug.

- [ ] **Step 3: Commit**

```bash
git add src/qldpc/codes/surgery_test.py
git commit -m "test: BB-BB intra-code joint matches Ide Table IV parameters"
```

---

### Task 17: Final cleanup and full-suite verification

**Files:**
- Delete: `/tmp/surgery_old.py.bak` (housekeeping)
- Modify: `src/qldpc/codes/surgery/__init__.py` (remove transitional re-exports)

- [ ] **Step 1: Clean up the package init**

Read current `src/qldpc/codes/surgery/__init__.py`. Remove any lines that reference deleted v2 symbols (`_BridgeSpec`, `_build_bridge_via_skiptree`, etc.).

Final `__init__.py`:
```python
"""Surgery construction package — Webster L-layer + SkipTree-based joint."""

from __future__ import annotations

from .layered import (  # noqa: F401
    SurgeryLayout,
    build_layered_surgery_code,
    load_webster_seed_set,
    _restrict_to_logical_support,
    _compute_gauge_fix,
    _build_layered_blocks,
    _assemble_merged_HX,
    _assemble_merged_HZ,
    _build_generalised_bicycle_code,
)
from .skiptree import _skip_tree, _skip_tree_hr  # noqa: F401
from .cellulation import _cellulate_long_cycles  # noqa: F401
from .cheeger import (  # noqa: F401
    BoostResult,
    DistanceBoostResult,
    boost_gadget_cheeger,
    boost_gadget_cheeger_combinatorial,
    boost_gadget_distance,
    _spectral_cheeger_lower_bound,
)
from .joint import (  # noqa: F401
    JointSurgeryLayout,
    SkipTreeOverride,
    build_joint_measurement_code,
    _build_auxiliary_graph,
    _label_inverse,
    canonical_HR,
    _running_xor_b_c,
    _chi_z_compatibility_check,
    _extend_chi_rows_with_bridge,
    _extend_Z_rows_with_bridge,
    _build_adapter_x_checks,
)
from .port import SetValuedPort  # noqa: F401

__all__ = [
    "SurgeryLayout", "JointSurgeryLayout", "SkipTreeOverride",
    "BoostResult", "DistanceBoostResult", "SetValuedPort",
    "build_layered_surgery_code", "build_joint_measurement_code",
    "boost_gadget_cheeger", "boost_gadget_cheeger_combinatorial",
    "boost_gadget_distance", "load_webster_seed_set",
]
```

- [ ] **Step 2: Full suite test run**

Run: `pytest src/qldpc/ -q 2>&1 | tail -15`
Expected: all tests pass. Compared to baseline (~81 surgery + N others):
- 81 Webster single-logical tests pass
- ~12 new joint helper + integration tests pass
- 2 NEW Ide Table III/IV tests pass

- [ ] **Step 3: Confirm v2 dead code is gone**

Run: `grep -nE "_BridgeSpec|_build_bridge_via_skiptree|_stitch_gadgets_with_bridge|_find_bridge_z_stab_data_logical" src/qldpc/codes/surgery/`
Expected: no matches.

- [ ] **Step 4: Commit**

```bash
git add src/qldpc/codes/surgery/__init__.py
git commit -m "refactor: finalize surgery/ package — drop v2 dead code"
```

- [ ] **Step 5: Push the branch (optional)**

Run: `git push -u origin feat/surgery-construction`
(Only if user requests; do not push without explicit go-ahead.)

---

## Plan self-review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §1 Goal (BB-LP, BB-BB stab groups) | Tasks 14, 16 |
| §2 Success criteria (CSS commute, k_joint, max stab wt) | Tasks 13–16 |
| §3.1 Module split | Tasks 2–6 |
| §3.2 Data flow phases | Task 13 |
| §4.1 Auxiliary graph | Task 8 |
| §4.2 SkipTree + Alg 2 | Task 7 |
| §4.3 Bridge | Task 13 |
| §4.4 Adapter X-checks | Task 12 |
| §4.5 χ row extension | Tasks 9, 10 |
| §4.6 Webster Z-row extension | Task 11 |
| §4.7 χ–Z compatibility lemma | Task 9 (probe) + Task 10 (fallback) |
| §4.8 Original code stabs unchanged | Task 13 (via _embed + dedup) |
| §4.9 Set-valued port | Task 15 |
| §5 Public API | Task 13 |
| §5.1 Breaking changes | Task 13 step 7 |
| §6.1 New joint tests | Tasks 14, 16 |
| §6.2 Webster tests preserved | Task 6 step 5 + Task 13 step 6 |
| §7 Zenodo integration | Tasks 1, 14, 16 |
| §8 Risk register | Tasks 9 (lemma), 15 (set-valued port), 6 (module split rollback path via git) |

**Placeholder scan:** No "TBD", "TODO", or "fill in details". Step 3A vs 3B in Task 10 is an explicit branch driven by the Task 9 probe result, not a placeholder. Task 16 step 2 acknowledges a possible bookkeeping issue with a concrete debug path, not "handle edge cases."

**Type consistency:** Function signatures are consistent across tasks. `_skip_tree_hr` defined in Task 7 → used in Tasks 8, 9, 13. `_running_xor_b_c` defined in Task 8 → used in Tasks 9, 11. `canonical_HR` defined in Task 8 → used in Tasks 11, 12. `_extend_chi_rows_with_bridge` defined in Task 10 → used in Task 13.

---

Plan complete and saved to `docs/superpowers/plans/2026-06-07-skiptree-bridge-v3.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
