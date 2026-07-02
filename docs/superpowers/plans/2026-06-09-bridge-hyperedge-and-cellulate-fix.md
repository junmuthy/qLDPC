# Bridge fix — hyperedge skip + cellulate on port subgraph — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `build_bridge` work on real BB codes (Cain `bb_18` and any user-chosen Z-logical representative) by (a) silently skipping hyperedge rows in the auxiliary graph and (b) cellulating only the port subgraph.

**Architecture:** Two surgical changes in `src/qldpc/codes/surgery/bridge.py`. `_build_aux_graph_strict` swaps its `raise NotImplementedError` on hyperedge rows for a `continue` — hyperedge κ qubits stay in `F_aug`, T_s automatically gets zero columns there, CSS commutation holds via the SkipTree key identity restricted to weight-2 sub-incidence. `_cellulate_strict` renamed to `_cellulate_port_subgraph`; runs `nx.cycle_basis` on the port-induced subgraph (where SkipTree actually operates) instead of the full graph.

**Tech Stack:** Python 3.x, NumPy, NetworkX, pytest, the existing `qldpc` package (do not touch `gadget.py`, `circuit.py`, `cheeger.py`).

**Reference spec:** `docs/superpowers/specs/2026-06-09-bridge-hyperedge-and-cellulate-fix-design.md`

---

## File Structure

| File | Role | Touched by |
|---|---|---|
| `src/qldpc/codes/surgery/bridge.py` | Both fixes live here (lines 200-234 and 150-197). One call site in `build_bridge` (line 373). | Tasks 1, 2, 3 |
| `src/qldpc/codes/surgery/_test.py` | Unit tests for the two functions + one end-to-end BB18 smoke test. | Tasks 1, 2, 4 |
| `examples/scripts/joint_ppm_z_with_superposition_demo.py` | Manual verification — should run as-is post-fix. | Task 5 |
| `docs/superpowers/math.md` | Append paragraph to §2.2 noting hyperedge handling. | Task 6 |
| `docs/superpowers/specs/2026-06-09-joint-ppm-bridge-design.md` | Update §3 "Hyperedge handling" note to point at the new design doc. | Task 6 |

---

## Task 1: Bug 1 — `_build_aux_graph_strict` skips hyperedge rows

**Files:**
- Modify: `src/qldpc/codes/surgery/bridge.py:200-234`
- Modify: `src/qldpc/codes/surgery/_test.py:995-1000`
- Test: `src/qldpc/codes/surgery/_test.py` (new test, near the existing aux-graph tests around line 1000)

- [ ] **Step 1.1: Rewrite the existing rejection test as a positive "filters" test**

In `src/qldpc/codes/surgery/_test.py`, replace lines 995-1000 with:

```python
def test_build_aux_graph_filters_hyperedges():
    """F rows of weight >= 3 (hyperedges) are silently skipped; weight-2 rows survive."""
    from qldpc.codes.surgery.bridge import _build_aux_graph_strict
    F = np.array([
        [1, 1, 0, 0, 0],  # weight-2 → edge (0,1)
        [1, 1, 1, 1, 0],  # weight-4 hyperedge → skipped
        [0, 0, 1, 1, 0],  # weight-2 → edge (2,3)
        [0, 0, 0, 1, 1],  # weight-2 → edge (3,4)
    ], dtype=np.uint8)
    G_nx, edge_idx = _build_aux_graph_strict(F)
    assert set(G_nx.nodes) == {0, 1, 2, 3, 4}
    # Three weight-2 rows → three edges; hyperedge row contributes nothing
    assert G_nx.number_of_edges() == 3
    assert (0, 1) in edge_idx
    assert (2, 3) in edge_idx
    assert (3, 4) in edge_idx
    # Hyperedge would have produced edges (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    # but only edges from weight-2 rows are present
    assert (0, 2) not in edge_idx
    assert (0, 3) not in edge_idx
    assert (1, 3) not in edge_idx
```

- [ ] **Step 1.2: Run test to verify it currently fails**

Run: `pytest src/qldpc/codes/surgery/_test.py::test_build_aux_graph_filters_hyperedges -v`

Expected: FAIL with `NotImplementedError: F row 1 has weight 4 (hyperedge). ...`

- [ ] **Step 1.3: Modify `_build_aux_graph_strict` to skip hyperedges**

Edit `src/qldpc/codes/surgery/bridge.py:200-234`. Replace the function body with:

```python
def _build_aux_graph_strict(F: np.ndarray) -> tuple[nx.Graph, dict[tuple[int, int], int]]:
    """Build auxiliary graph from F; weight-2 rows become edges, hyperedges are skipped.

    Vertices: range(|V_0|) = range(F.shape[1]).
    Edges: one per weight-2 row of F, between the two columns where the row has 1s.

    Weight-≥3 rows (hyperedges) are silently ignored — they remain in F_aug so
    the gadget structure (G_aug = ker(F_aug^T), deformed check c → c · X(κ_r))
    is unchanged, but T_s assigns them zero columns (existing skip at
    _run_skiptree_on_port_subgraph). CSS commutation, κ-cancellation, joint
    observable, and dimension counting all hold by direct calculation;
    see docs/superpowers/specs/2026-06-09-bridge-hyperedge-and-cellulate-fix-design.md
    §correctness-proof. Paper Eq. 9's perfect-matching decomposition (§II.C)
    is not applied; structural Theorem 12 distance argument is substituted
    by empirical LER smoke tests.

    Raises:
        ValueError: if any row of F has weight 1 (defensive — F · 1 = 0 mod 2
        forbids odd weights for a valid logical).
    """
    F_arr = np.asarray(F).astype(int)
    G = nx.Graph()
    G.add_nodes_from(range(F_arr.shape[1]))
    edge_index: dict[tuple[int, int], int] = {}
    for i, row in enumerate(F_arr):
        eps = np.flatnonzero(row).tolist()
        if len(eps) == 0 or len(eps) >= 3:
            continue
        if len(eps) == 1:
            raise ValueError(
                f"F row {i} has weight 1 (column {eps[0]}). "
                f"Auxiliary-graph edges require exactly 2 endpoints "
                f"(F · 1 = 0 mod 2 forbids odd weights — invalid logical?)."
            )
        u, v = sorted(eps)
        if (u, v) not in edge_index:
            edge_index[(u, v)] = len(edge_index)
            G.add_edge(u, v)
    return G, edge_index
```

- [ ] **Step 1.4: Run filters test — should now pass**

Run: `pytest src/qldpc/codes/surgery/_test.py::test_build_aux_graph_filters_hyperedges -v`

Expected: PASS

- [ ] **Step 1.5: Run the other aux-graph tests to verify no regression**

Run: `pytest src/qldpc/codes/surgery/_test.py::test_build_aux_graph_weight2_rows_become_edges src/qldpc/codes/surgery/_test.py::test_build_aux_graph_rejects_weight1_row -v`

Expected: 2 passed.

- [ ] **Step 1.6: Commit**

```bash
git add src/qldpc/codes/surgery/bridge.py src/qldpc/codes/surgery/_test.py
git commit -m "$(cat <<'EOF'
fix: silently skip hyperedge rows in bridge aux graph

Weight-≥3 rows of F (hyperedges) used to raise NotImplementedError pointing
at paper §II.C decomposition. Switch to skip — the hyperedge κ qubit stays
in F_aug so the gadget is unchanged, and T_s already gets zero columns on
hyperedge rows (bridge.py:312). CSS commutation, κ-cancellation, joint
observable, and dim −1 all hold; see spec for proof.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Bug 2 — `_cellulate_strict` renamed and scoped to port subgraph

**Files:**
- Modify: `src/qldpc/codes/surgery/bridge.py:150-197` (rename + new body)
- Modify: `src/qldpc/codes/surgery/_test.py:1037-1066` (3 existing tests, rename function reference)
- Test: `src/qldpc/codes/surgery/_test.py` (1 new positive test, 1 new "non-port cycle skipped" test)

- [ ] **Step 2.1: Add the new positive test (cellulate operates on port subgraph)**

In `src/qldpc/codes/surgery/_test.py`, after `test_cellulate_raises_when_no_port_chord_available`, add:

```python
def test_cellulate_port_subgraph_breaks_long_port_cycle():
    """A cycle > max_len in the port subgraph gets broken by adding a chord."""
    import networkx as nx
    from qldpc.codes.surgery.bridge import _cellulate_port_subgraph
    # 10-cycle on port vertices only
    G = nx.cycle_graph(10)
    ports = tuple(range(10))
    added = _cellulate_port_subgraph(G, ports, max_len=6)
    assert len(added) >= 1
    # All port-subgraph basis cycles now bounded
    sub = G.subgraph(ports)
    for c in nx.cycle_basis(sub):
        assert len(c) <= 6


def test_cellulate_port_subgraph_skips_non_port_cycle():
    """Long cycle entirely on non-port vertices is ignored; no edges added."""
    import networkx as nx
    from qldpc.codes.surgery.bridge import _cellulate_port_subgraph
    G = nx.Graph()
    # Long non-port cycle: 10-11-12-...-17-10 (length 8)
    G.add_edges_from([(10, 11), (11, 12), (12, 13), (13, 14),
                      (14, 15), (15, 16), (16, 17), (17, 10)])
    # Short port cycle: triangle on 0,1,2
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    ports = (0, 1, 2)
    n_edges_before = G.number_of_edges()
    added = _cellulate_port_subgraph(G, ports, max_len=6)
    assert added == []
    assert G.number_of_edges() == n_edges_before
```

- [ ] **Step 2.2: Update the three existing cellulate tests to use the new name**

In `src/qldpc/codes/surgery/_test.py`, edit lines 1037-1066:

- Line 1040: `from qldpc.codes.surgery.bridge import _cellulate_strict` → `from qldpc.codes.surgery.bridge import _cellulate_port_subgraph`
- Line 1043: `added = _cellulate_strict(...)` → `added = _cellulate_port_subgraph(...)`
- Line 1046-1047: replace
  ```python
      cycles = nx.cycle_basis(G_aux)
      assert max(len(c) for c in cycles) <= 6
  ```
  with
  ```python
      sub = G_aux.subgraph(tuple(range(10)))
      assert max((len(c) for c in nx.cycle_basis(sub)), default=0) <= 6
  ```
- Line 1053: `from qldpc.codes.surgery.bridge import _cellulate_strict` → `from qldpc.codes.surgery.bridge import _cellulate_port_subgraph`
- Line 1055: `added = _cellulate_strict(...)` → `added = _cellulate_port_subgraph(...)`
- Line 1059-1066: replace the whole test with:
  ```python
  def test_cellulate_raises_when_port_cycle_has_no_available_chord():
      """RuntimeError when a port-subgraph cycle exists but every (i, j) pair
      is already an edge — i.e. the port subgraph is complete on those vertices."""
      import networkx as nx
      from qldpc.codes.surgery.bridge import _cellulate_port_subgraph
      # 7-cycle 0-1-2-3-4-5-6-0 plus ALL chords among {0..6} → complete graph K_7.
      # cycle_basis still surfaces cycles of length > max_len in K_7 (basis cycles
      # are length-3 triangles), so no long cycle exists in this case.
      # Instead: make a 7-cycle without any extra edges, then call with max_len=2.
      G = nx.cycle_graph(7)
      ports = tuple(range(7))
      # Already a complete graph K_7? No — cycle_graph(7) has only 7 edges.
      # Pre-saturate with all possible chords so no chord can be added:
      for i in range(7):
          for j in range(i + 2, 7):
              if not G.has_edge(i, j) and (i, j) != (0, 6):
                  G.add_edge(i, j)
      # Now every (i, j) with j >= i+2 in the 7-cycle is already an edge.
      # A length-7 basis cycle no longer exists (it's broken into triangles),
      # so max_len=6 finds no long cycle and returns []. Use max_len=2 to force
      # the failure path:
      with pytest.raises(RuntimeError, match=r"No chord found"):
          _cellulate_port_subgraph(G, ports, max_len=2)
  ```

- [ ] **Step 2.3: Run the new + updated tests to verify they fail (function not renamed yet)**

Run: `pytest src/qldpc/codes/surgery/_test.py::test_cellulate_port_subgraph_breaks_long_port_cycle src/qldpc/codes/surgery/_test.py::test_cellulate_port_subgraph_skips_non_port_cycle src/qldpc/codes/surgery/_test.py::test_cellulate_caps_cycle_length src/qldpc/codes/surgery/_test.py::test_cellulate_no_op_when_already_short src/qldpc/codes/surgery/_test.py::test_cellulate_raises_when_port_cycle_has_no_available_chord -v`

Expected: 5 FAIL with `ImportError: cannot import name '_cellulate_port_subgraph'`.

- [ ] **Step 2.4: Replace `_cellulate_strict` with `_cellulate_port_subgraph`**

Edit `src/qldpc/codes/surgery/bridge.py`. Replace lines 150-197 (the entire `_cellulate_strict` definition) with:

```python
def _cellulate_port_subgraph(
    G_aux: nx.Graph,
    ports: tuple[int, ...],
    *,
    max_len: int = 6,
) -> list[tuple[int, int]]:
    """Break port-subgraph cycles longer than ``max_len`` by adding chords.

    SkipTree runs on G_aux.subgraph(ports); cycles entirely outside the
    port subgraph never enter T_s, so we cellulate only there. The full-graph
    version (the previous _cellulate_strict) failed spuriously on real BB
    codes when the only long cycles threaded through non-port vertices.

    Chords are added to ``G_aux`` (the full graph). For port-subgraph cycles,
    chord endpoints are necessarily ports (cycle vertices = port vertices).

    Returns the list of added (u, v) edges in insertion order. Idempotent
    once all port-subgraph basis cycles fit under the cap.
    """
    added: list[tuple[int, int]] = []
    while True:
        sub = G_aux.subgraph(ports)
        long_cycles = [c for c in nx.cycle_basis(sub) if len(c) > max_len]
        if not long_cycles:
            return added
        cycle = long_cycles[0]
        n = len(cycle)
        chord_found = False
        for i in range(n):
            if chord_found:
                break
            for j in range(i + 2, n):
                u, v = sorted((cycle[i], cycle[j]))
                if G_aux.has_edge(u, v):
                    continue
                G_aux.add_edge(u, v)
                added.append((u, v))
                chord_found = True
                break
        if not chord_found:
            raise RuntimeError(
                f"No chord found to cellulate port-subgraph cycle of length {n}; "
                f"cycle={cycle!r}"
            )
```

- [ ] **Step 2.5: Update the one caller in `build_bridge`**

Edit `src/qldpc/codes/surgery/bridge.py` lines 372-373. Replace:

```python
    extras_l_cell = _cellulate_strict(G_l_aux, port_l, max_len=cellulate_max_len)
    extras_r_cell = _cellulate_strict(G_r_aux, port_r, max_len=cellulate_max_len)
```

with:

```python
    extras_l_cell = _cellulate_port_subgraph(G_l_aux, port_l, max_len=cellulate_max_len)
    extras_r_cell = _cellulate_port_subgraph(G_r_aux, port_r, max_len=cellulate_max_len)
```

- [ ] **Step 2.6: Run the cellulate tests — should now all pass**

Run: `pytest src/qldpc/codes/surgery/_test.py::test_cellulate_port_subgraph_breaks_long_port_cycle src/qldpc/codes/surgery/_test.py::test_cellulate_port_subgraph_skips_non_port_cycle src/qldpc/codes/surgery/_test.py::test_cellulate_caps_cycle_length src/qldpc/codes/surgery/_test.py::test_cellulate_no_op_when_already_short src/qldpc/codes/surgery/_test.py::test_cellulate_raises_when_port_cycle_has_no_available_chord -v`

Expected: 5 passed.

- [ ] **Step 2.7: Update `test_cellulation_caps_aug_aux_cycle_length_on_webster` to inspect the port subgraph**

This test (line 1434 of `_test.py`) currently inspects the FULL augmented aux graph's cycle basis. After the rename + scope-down, only port-subgraph cycles are capped. Update the assertion accordingly.

Edit `src/qldpc/codes/surgery/_test.py:1447-1453`. Replace:

```python
    # Build aux graph from g_l_aug.F and check cycle basis
    G_aux, _ = _build_aux_graph_strict(bridge.g_l_aug.F)
    cycles = nx.cycle_basis(G_aux)
    if cycles:
        assert max(len(c) for c in cycles) <= 6, (
            f"max cycle length {max(len(c) for c in cycles)} > 6"
        )
```

with:

```python
    # Cellulation is now scoped to the port subgraph (where SkipTree runs).
    # Inspect cycles on the induced port subgraph, not the full graph.
    G_aux, _ = _build_aux_graph_strict(bridge.g_l_aug.F)
    sub = G_aux.subgraph(bridge.port_l)
    cycles = nx.cycle_basis(sub)
    if cycles:
        assert max(len(c) for c in cycles) <= 6, (
            f"max port-subgraph cycle length {max(len(c) for c in cycles)} > 6"
        )
```

- [ ] **Step 2.8: Run the whole surgery test module to catch any other reference to `_cellulate_strict`**

Run: `pytest src/qldpc/codes/surgery/_test.py -v --co 2>&1 | grep -i cellulat`

Expected: only cellulate tests listed (5 unit tests + 1 Webster integration test).

Run: `grep -rn "_cellulate_strict" src/ examples/ docs/ 2>/dev/null`

Expected: empty output. If anything appears, edit it to use `_cellulate_port_subgraph`.

- [ ] **Step 2.9: Commit**

```bash
git add src/qldpc/codes/surgery/bridge.py src/qldpc/codes/surgery/_test.py
git commit -m "$(cat <<'EOF'
fix: cellulate only the port subgraph, not the full aux graph

SkipTree runs on G_aux.subgraph(ports), so cycles outside the port subgraph
never enter T_s. The old _cellulate_strict (cycle-basis on the full graph,
port-port chord requirement) failed spuriously on real BB codes — cycles
threading through non-port vertices with too few port endpoints had no
available chord. Rename to _cellulate_port_subgraph and run cycle_basis
on the induced port subgraph; chord endpoints are then necessarily ports
by construction.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: End-to-end smoke test on `bb_18`

**Files:**
- Test: `src/qldpc/codes/surgery/_test.py` (add new test after the existing build_bridge tests)

- [ ] **Step 3.1: Find the right insertion point**

Run: `grep -n "^def test_build_bridge\|^def test_stitch" src/qldpc/codes/surgery/_test.py | tail -10`

Note the line of the last `test_build_bridge_*` or `test_stitch_*` test. The new test goes right after it. (Existing test locations: line 1417 area has `_stitch_to_joint_csscode` calls; insert after the last stitch test.)

- [ ] **Step 3.2: Add the BB18 hyperedge-and-long-cycle smoke test**

Add to `src/qldpc/codes/surgery/_test.py` after the last existing stitch/bridge test:

```python
def test_build_bridge_bb18_hyperedge_and_long_cycle():
    """End-to-end: Cain bb_18 BBCode triggers both Bug 1 (hyperedge) and
    Bug 2 (long port-subgraph cycle). build_bridge must succeed and produce
    a merged code with k_merged = 2 * k_orig - 1 (intra-code joint Z̄ ⊗ Z̄)."""
    import sympy
    from qldpc import codes
    from qldpc.objects import Pauli
    from qldpc.codes.surgery import build_gadget, build_bridge
    from qldpc.codes.surgery.circuit import _stitch_to_joint_csscode

    x, y = sympy.symbols("x y")
    code = codes.BBCode(
        {x: 31, y: 4},
        1 + x**6 * y + x**27,
        y**2 + x**15 * y**3 + x**24,
    )
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    # Confirm we are actually exercising both bugs in this case:
    row_weights = np.asarray(g.F.sum(axis=1)).ravel().astype(int).tolist()
    assert max(row_weights) >= 4, "Test no longer triggers Bug 1 (no hyperedge)"
    # Build bridge (this used to raise NotImplementedError or RuntimeError)
    bridge = build_bridge(g, g)
    # Merged code construction must succeed
    merged = _stitch_to_joint_csscode(g, g, bridge)
    # Intra-code joint: k_merged == k_orig − 1
    assert merged.dimension == code.dimension - 1
    # CSS commutation on merged code
    HX = np.asarray(merged.matrix_x).astype(np.uint8)
    HZ = np.asarray(merged.matrix_z).astype(np.uint8)
    assert not ((HX @ HZ.T) % 2).any(), "CSS commutation broken on merged code"
```

- [ ] **Step 3.3: Run the new test to verify it currently passes (Tasks 1, 2 already merged it should)**

Run: `pytest src/qldpc/codes/surgery/_test.py::test_build_bridge_bb18_hyperedge_and_long_cycle -v`

Expected: PASS. (If FAIL, investigate — Tasks 1 and 2 should have made it green. Check the failure mode against the spec's correctness proof.)

- [ ] **Step 3.4: Run the full surgery test module to confirm no regression**

Run: `pytest src/qldpc/codes/surgery/_test.py -v 2>&1 | tail -40`

Expected: all tests pass (the surgery module's tests; ignore pytest-randomly plugin errors elsewhere in the project, which are pre-existing).

- [ ] **Step 3.5: Commit**

```bash
git add src/qldpc/codes/surgery/_test.py
git commit -m "$(cat <<'EOF'
test: bb_18 build_bridge succeeds despite hyperedge + long port cycles

End-to-end smoke test confirming the two surgical bridge fixes cover Cain's
bb_18 BBCode (l=31, m=4, polys 1+x^6 y+x^27, y^2+x^15 y^3+x^24). The chosen
Z-logical has a weight-4 F row (Bug 1) AND a port-subgraph cycle whose
chords are not port-port (Bug 2 once a different logical is picked).
Asserts merged code dimension drops by exactly 1 and CSS commutation holds.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Manual verification of the demo

**Files:**
- No source changes. Just runs `examples/scripts/joint_ppm_z_with_superposition_demo.py` to confirm.

- [ ] **Step 4.1: Run the demo**

Run: `/Users/tgzhou/Project/qLDPC/.venv/bin/python examples/scripts/joint_ppm_z_with_superposition_demo.py`

Expected output ends with:

```
  obs0 bit=1 rate      : 4X.XX%  (expected ~50%)
  obs1 bit=1 rate      : 4X.XX%  (expected ~50%)
  obs0 == obs1 per shot: 100.00%  (expected 100%)

  ✓ Genuine quantum randomness (Z̄_2 on |+⟩_L collapses 50/50)
  ✓ Webster Eq.1 obs0 and final-Mz cross-check obs1 agree on EVERY shot
    → both paths really measure the same Z̄_1 ⊗ Z̄_2 operator
```

- [ ] **Step 4.2: If the demo fails, diagnose before proceeding**

If any of the three asserts fail (`rate0`, `rate1`, `obs0 == obs1`):

1. Check the stack trace; if it's a Python error from `build_bridge`, Tasks 1-3 are incomplete — go back.
2. If it's an assertion error on the rates/agreement, the construction is producing the wrong observable. Re-read spec §correctness-proof and verify the χ row count and adapter Π_s matrix match the design.
3. Do not proceed to Task 5 (docs) until the demo passes.

- [ ] **Step 4.3: No commit** — demo run is verification only.

---

## Task 5: Documentation updates

**Files:**
- Modify: `docs/superpowers/math.md` (append a paragraph to §2.2)
- Modify: `docs/superpowers/specs/2026-06-09-joint-ppm-bridge-design.md` (§3 "Hyperedge handling")

- [ ] **Step 5.1: Append the hyperedge paragraph to math.md §2.2**

Open `docs/superpowers/math.md` and find the section "2.2 Auxiliary graph augmentation" (around line 105 — the one ending with `Cellulate basis cycles to length ≤ max_len.`).

Append, immediately after that paragraph:

```
  When F has rows of weight ≥ 4 (hyperedges, even-weight forced by F·1_{V_0}=0),
  they are kept in F_aug so the gadget structure is unchanged but skipped in the
  auxiliary graph 𝒢_s. SkipTree assigns T_s zero columns to hyperedge rows
  (existing skip at _run_skiptree_on_port_subgraph), so the SkipTree key identity
  T_s · F_aug · P_s = H_R reduces to its restriction onto the weight-2 sub-
  incidence and holds automatically. CSS commutation, κ-cancellation, joint
  observable, and dim−1 all hold by direct calculation. Paper Eq. 9's perfect-
  matching decomposition (§II.C) is not applied; the structural distance
  argument of Theorem 12 is replaced by empirical LER smoke tests, per the
  paper's own remark at the end of §IV.
```

- [ ] **Step 5.2: Update §3 of the old joint-ppm-bridge design doc**

Open `docs/superpowers/specs/2026-06-09-joint-ppm-bridge-design.md`. Find the section "Hyperedge handling" (around line 193-198).

Replace its body with:

```
The build silently skips F rows of weight ≥ 4; T_s gets zero columns there
automatically (existing logic in `_run_skiptree_on_port_subgraph`). CSS
commutation, κ-cancellation, joint observable, and dim−1 are preserved.
See `docs/superpowers/specs/2026-06-09-bridge-hyperedge-and-cellulate-fix-design.md`
for the proof and the rationale for choosing this over paper-faithful §II.C
decomposition.
```

- [ ] **Step 5.3: Commit**

```bash
git add docs/superpowers/math.md docs/superpowers/specs/2026-06-09-joint-ppm-bridge-design.md
git commit -m "$(cat <<'EOF'
docs: note bridge hyperedge skip + cellulate scope-down in math.md

Cross-reference the bridge fix design doc from math.md §2.2 and from the
original joint-ppm-bridge spec's §3 hyperedge-handling note. Replaces the
"raises NotImplementedError" wording with a pointer to the proof.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Final sanity sweep

**Files:** None — this is a check pass.

- [ ] **Step 6.1: Verify no stray references**

Run: `grep -rn "NotImplementedError.*hyperedge\|_cellulate_strict" src/ examples/ docs/ 2>/dev/null`

Expected: empty output. If anything appears outside the spec docs themselves, edit it.

- [ ] **Step 6.2: Run the full surgery test suite one final time**

Run: `pytest src/qldpc/codes/surgery/_test.py -v 2>&1 | tail -20`

Expected: all surgery tests pass.

- [ ] **Step 6.3: Optional — run the demo one more time**

Run: `/Users/tgzhou/Project/qLDPC/.venv/bin/python examples/scripts/joint_ppm_z_with_superposition_demo.py 2>&1 | tail -15`

Expected: same success output as Task 4 Step 1.

- [ ] **Step 6.4: No commit** — sanity sweep is verification only.

---

## Self-review (filled in by plan author)

**Spec coverage:**

- Spec §1 (Bug 1 fix) → Task 1 (steps 1.1-1.6). ✓
- Spec §2 (Bug 2 fix, rename + behavior) → Task 2 (steps 2.1-2.8). ✓
- Spec §3 (build_bridge integration) → Task 2 step 2.5 (one-line call-site change). ✓
- Spec §4 (existing tests updated + 5 new tests):
  - `test_build_aux_graph_filters_hyperedges` (replaces rejects) → Task 1 step 1.1. ✓
  - `test_build_aux_graph_skips_hyperedge_row` → folded into the same test (Task 1 step 1.1 covers both "filter exists" and "skip happens"). Distinct test would be redundant. ✓
  - `test_cellulate_port_subgraph_breaks_long_port_cycle` → Task 2 step 2.1. ✓
  - `test_cellulate_port_subgraph_skips_non_port_cycle` → Task 2 step 2.1. ✓
  - `test_cellulate_raises_when_no_port_chord_available` (rewritten) → Task 2 step 2.2 (renamed to `_when_port_cycle_has_no_available_chord`). ✓
  - `test_cellulate_caps_cycle_length` (updated) → Task 2 step 2.2. ✓
  - `test_cellulate_no_op_when_already_short` (unchanged) → Task 2 step 2.2 (just rename function reference). ✓
  - `test_cellulation_caps_aug_aux_cycle_length_on_webster` (update to use port subgraph) → covered by Task 6 step 6.2 running the full module; the existing test will need the same line-edit as the others (the call to `_cellulate_strict` if any). Add a note: if this test breaks in Task 6, add a Task 6.4 to update it the same way as Task 2 step 2.2.
  - `test_build_bridge_bb18_hyperedge_and_long_cycle` → Task 3. ✓
  - `test_build_bridge_bbcode_k_reduces_by_one` → folded into Task 3 step 3.2 (the BB18 test asserts dim = k_orig − 1 already). ✓
- Spec §5 (docs) → Task 5. ✓
- Spec §6 (migration / blast radius) → covered implicitly by Tasks 1-5 (no gadget/circuit/cheeger touches; demo script untouched). ✓
- Spec §7 (out of scope) → not in plan; correct. ✓

**Placeholder scan:** None — every step has concrete code or commands. The "if this test breaks in Task 6" note is contingent verification, not a placeholder.

**Type consistency:** Function name `_cellulate_port_subgraph` used consistently across Task 2 and Task 6. `_build_aux_graph_strict` keeps its name (behavior change, name unchanged — keep churn low).
