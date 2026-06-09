# Joint PPM Bridge — Universal Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current path-graph bridge in `src/qldpc/codes/surgery/bridge.py` with the Swaroop–Jochym-O'Connor–Yoder universal adapter (arXiv:2410.03628 §IV+§VII), so `build_joint_ppm_circuit` measures `X̄_l ⊗ X̄_r` (or `Z̄_l ⊗ Z̄_r`) with merged-code distance ≥ `min(d_l, d_r)`.

**Architecture:** SkipTree-driven bijection adapter between two `GadgetLayout` auxiliary graphs. Bridge introduces `w = min(|V_0^(l)|, |V_0^(r)|)` adapter data qubits, `(w-1)` new Z-type cycle-check rows `[T_l | H_R | T_r]`, optional `extra_κ` qubits for induced-subgraph connectivity and cellulation. Joint observable's α* selects only `χ^(l) ∪ χ^(r)` — no separate `U_B` block.

**Tech Stack:** Python 3, NumPy, NetworkX, galois (GF(2)), stim, pytest.

**Spec:** `docs/superpowers/specs/2026-06-09-joint-ppm-bridge-design.md`.

**Paper section refs:** §III SkipTree (Algorithm 1), §IV adapter (Lemma 9–10, Theorem 11–12), §V.A "expansion-free for ports", §VII.B/C BB-LP and BB-BB examples.

---

## File structure

| File | Role |
|---|---|
| `src/qldpc/codes/surgery/bridge.py` | Rewritten. Owns `Bridge` dataclass, SkipTree, auxiliary graph builder, port/connectivity/cellulation augmentation, and `build_bridge` orchestrator. |
| `src/qldpc/codes/surgery/gadget.py` | Add `build_gadget_augmented(code, x, F_extra, basis)` — rebuilds a `GadgetLayout` over an augmented F (= F stacked with extra weight-2 rows from the bridge). |
| `src/qldpc/codes/surgery/circuit.py` | Modify `_stitch_to_joint_csscode` to consume new `Bridge` fields; switch `chi_check_ids` to `χ⁽¹⁾ ∪ χ⁽²⁾` only (no `ub_ids`); extend `_surgery_state_prep` / `_surgery_detach_and_readout` / `_surgery_final_detectors` to cover `extra_κ` + adapter qubits. |
| `src/qldpc/codes/surgery/_test.py` | Delete path-graph tests (`test_path_graph_U_B_telescoping`, `test_webster_table_i_bridge_width_exact`, `test_build_bridge_intracode_chi_endpoint_extensions`, `test_skip_tree_path_graph_returns_identity`, `test_alpha_star_yields_joint_op_on_webster`). Add tests called out in the spec §5. |
| `docs/superpowers/math.md` | Replace §2.2–2.7 with the universal-adapter math (block tables + SkipTree key identity + α* derivation). Keep old §2 as a historical footnote. |
| `examples/scripts/joint_ppm_correctness_z_demo.py`, `joint_ppm_z_with_superposition_demo.py`, `cain_fig1b_full_protocol.py`, `cain_bb18_resource_exact_match.py` | Update to new `Bridge` field names; drop `U_B` references. |
| `examples/test_ide_bb_lp.py` | Update `build_bridge` call site. |
| `examples/logical_error_rates/9_lattice_surgery.ipynb` + `_9_lattice_surgery_source.py` | Refresh narrative + code cells. |

---

## Task 1: Refactor `_skip_tree` to expose a full-rank variant

**Files:**
- Modify: `src/qldpc/codes/surgery/bridge.py` (current `_skip_tree` at line 44)
- Test: `src/qldpc/codes/surgery/_test.py`

The current `_skip_tree` returns a cyclic `T` (rows close back from `n-1 → 0`), giving an `n × n` matrix that equals the cyclic `H_C` after `T G P = H_C`. The new variant drops the last row to produce a full-rank `(n-1) × n` matrix matching `H_R` from the design spec.

- [ ] **Step 1: Write failing test**

Add to `src/qldpc/codes/surgery/_test.py` after the existing `_skip_tree` test (line 364 area, BEFORE deleting old `test_skip_tree_path_graph_returns_identity`):

```python
def test_skip_tree_fullrank_on_K4_matches_H_R():
    """SkipTree full-rank: T_ind · G · P_ind = H_R for the complete graph K_4."""
    import networkx as nx
    from qldpc.codes.surgery.bridge import _skip_tree_fullrank, _canonical_H_R

    G_nx = nx.complete_graph(4)
    n = 4
    edges = sorted(tuple(sorted(e)) for e in G_nx.edges())
    edge_index = {e: i for i, e in enumerate(edges)}
    G_mat = np.zeros((len(edges), n), dtype=np.int_)
    for (u, v), i in edge_index.items():
        G_mat[i, u] = 1
        G_mat[i, v] = 1

    T_ind, P_ind = _skip_tree_fullrank(G_nx, root=0, edge_index=edge_index)
    H_R = _canonical_H_R(n)

    assert T_ind.shape == (n - 1, len(edges))
    assert P_ind.shape == (n, n)
    # SkipTree key identity: T_ind · G · P_ind == H_R over GF(2)
    product = (T_ind @ G_mat @ P_ind) % 2
    assert np.array_equal(product, H_R), f"got\n{product}\nwant\n{H_R}"
    # (3,2)-sparsity
    assert T_ind.sum(axis=1).max() <= 3
    assert T_ind.sum(axis=0).max() <= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/qldpc/codes/surgery/_test.py::test_skip_tree_fullrank_on_K4_matches_H_R -v`
Expected: FAIL with `ImportError` (`_skip_tree_fullrank` does not exist, `_canonical_H_R` does not exist).

- [ ] **Step 3: Implement `_canonical_H_R` and `_skip_tree_fullrank`**

Add to `src/qldpc/codes/surgery/bridge.py` (just after `_skip_tree` definition):

```python
def _canonical_H_R(w: int) -> np.ndarray:
    """Full-rank canonical rep-code parity check matrix, shape (w-1) × w.

    Row i has 1s in columns i and i+1. rank == w-1; column 0 and column w-1 have
    weight 1, other columns weight 2.
    """
    if w < 2:
        raise ValueError(f"H_R requires w >= 2, got {w}")
    H = np.zeros((w - 1, w), dtype=np.int_)
    for i in range(w - 1):
        H[i, i] = 1
        H[i, i + 1] = 1
    return H


def _skip_tree_fullrank(
    S: nx.Graph,
    root: int = 0,
    edge_index: dict[tuple[int, int], int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop last row of cyclic SkipTree to get T · G · P == H_R (full-rank rep code).

    Returns (T_ind, P_ind) of shapes (n-1, |E|) and (n, n).
    """
    T_cyc, P = _skip_tree(S, root=root, edge_index_verts=edge_index)
    # _skip_tree returns (T_cyc, P) with T_cyc shape (n-1, |E|) already
    # (the last path from label[n-1] to label[0] wraps cyclically).  The cyclic
    # T satisfies T · G · P = H_C cyclic.  To match H_R, we use the *first n-1*
    # rows directly — _skip_tree already produces n-1 rows because we treat the
    # last cyclic path as redundant in the loop body.  Re-check shape:
    assert T_cyc.shape[0] == S.number_of_nodes() - 1
    return T_cyc.astype(np.int_), P.astype(np.int_)
```

Note: inspecting the existing `_skip_tree` shows the loop produces `n - 1` rows already (`for l_idx in range(n - 1)`). So the "full-rank" version is just a thin wrapper plus the renamed signature. Keep both `_skip_tree` (cyclic naming) and `_skip_tree_fullrank` (clear name, the version used downstream).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/qldpc/codes/surgery/_test.py::test_skip_tree_fullrank_on_K4_matches_H_R -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/bridge.py src/qldpc/codes/surgery/_test.py
git commit -m "$(cat <<'EOF'
feat: _skip_tree_fullrank wrapper + canonical H_R helper

Adds the (3,2)-sparse full-rank variant called for in
docs/superpowers/specs/2026-06-09-joint-ppm-bridge-design.md §2,
plus the canonical rep-code parity matrix H_R consumed by the
new adapter cycle-checks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Auxiliary-graph builder (weight-2 only, raise on hyperedges)

**Files:**
- Modify: `src/qldpc/codes/surgery/bridge.py`
- Test: `src/qldpc/codes/surgery/_test.py`

Replace the current `_build_auxiliary_graph_from_F` (which silently skips weight ≥ 3 rows) with a stricter version that **raises** on hyperedges per spec §3 "Hyperedge handling".

- [ ] **Step 1: Write failing test**

Add to `_test.py`:

```python
def test_build_aux_graph_weight2_rows_become_edges():
    """F rows of weight 2 → graph edges; vertex set = {0, ..., |V_0|-1}."""
    from qldpc.codes.surgery.bridge import _build_aux_graph_strict
    F = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.uint8)
    G_nx, edge_idx = _build_aux_graph_strict(F)
    assert set(G_nx.nodes) == {0, 1, 2, 3}
    assert set(tuple(sorted(e)) for e in G_nx.edges) == {(0, 1), (1, 2), (2, 3)}
    assert edge_idx[(0, 1)] == 0
    assert edge_idx[(1, 2)] == 1
    assert edge_idx[(2, 3)] == 2


def test_build_aux_graph_rejects_hyperedge():
    """F rows of weight >= 3 raise NotImplementedError with §II.C reference."""
    from qldpc.codes.surgery.bridge import _build_aux_graph_strict
    F = np.array([[1, 1, 1, 0], [0, 1, 1, 0]], dtype=np.uint8)
    with pytest.raises(NotImplementedError, match=r"hyperedge.*§II\.C"):
        _build_aux_graph_strict(F)
```

- [ ] **Step 2: Run to verify fails**

Run: `pytest src/qldpc/codes/surgery/_test.py::test_build_aux_graph_weight2_rows_become_edges src/qldpc/codes/surgery/_test.py::test_build_aux_graph_rejects_hyperedge -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement `_build_aux_graph_strict`**

Add to `bridge.py`:

```python
def _build_aux_graph_strict(F: np.ndarray) -> tuple[nx.Graph, dict[tuple[int, int], int]]:
    """Build auxiliary graph from F; raise on hyperedges.

    Vertices: range(|V_0|) = range(F.shape[1]).
    Edges: one per weight-2 row of F, between the two columns where the row has 1s.

    Raises:
        NotImplementedError: if any row of F has weight >= 3 (hyperedge), pointing to
        paper §II.C decomposition.
    """
    F_arr = np.asarray(F).astype(int)
    n_V = F_arr.shape[1]
    G = nx.Graph()
    G.add_nodes_from(range(n_V))
    edge_index: dict[tuple[int, int], int] = {}
    for i, row in enumerate(F_arr):
        eps = np.flatnonzero(row).tolist()
        if len(eps) == 0:
            continue
        if len(eps) >= 3:
            raise NotImplementedError(
                f"F row {i} has weight {len(eps)} (hyperedge). "
                f"Universal-adapter construction here requires weight-2 rows. "
                f"To handle hyperedges, decompose them per arXiv:2410.03628 §II.C."
            )
        u, v = sorted(eps)
        if (u, v) not in edge_index:
            edge_index[(u, v)] = len(edge_index)
            G.add_edge(u, v)
    return G, edge_index
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest src/qldpc/codes/surgery/_test.py::test_build_aux_graph_weight2_rows_become_edges src/qldpc/codes/surgery/_test.py::test_build_aux_graph_rejects_hyperedge -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/bridge.py src/qldpc/codes/surgery/_test.py
git commit -m "$(cat <<'EOF'
feat: strict auxiliary graph builder (weight-2 only)

Hyperedges now raise NotImplementedError pointing at §II.C
decomposition; matches universal-adapter spec §2 step 1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Induced-subgraph connectivity augmentation

**Files:**
- Modify: `src/qldpc/codes/surgery/bridge.py`
- Test: `src/qldpc/codes/surgery/_test.py`

Spec §2 step 3 — when `𝒢[𝒫*]` is disconnected, add edges between components by going through the full `𝒢` graph. Each added edge becomes one new `extra_κ` row (weight 2, endpoints at the bridged vertices).

- [ ] **Step 1: Write failing test**

```python
def test_connect_induced_subgraph_no_op_when_connected():
    """If induced subgraph is already connected, no edges are added."""
    import networkx as nx
    from qldpc.codes.surgery.bridge import _connect_induced_subgraph
    G_aux = nx.path_graph(4)  # 0-1-2-3
    extra = _connect_induced_subgraph(G_aux, ports=(0, 1, 2, 3))
    assert extra == []
    assert set(tuple(sorted(e)) for e in G_aux.edges) == {(0, 1), (1, 2), (2, 3)}


def test_connect_induced_subgraph_adds_edges_to_disconnected_components():
    """Disconnected induced subgraph gets one bridging edge per missing connection."""
    import networkx as nx
    from qldpc.codes.surgery.bridge import _connect_induced_subgraph
    # G_aux: 0-1   2-3 (two separate components)
    G_aux = nx.Graph()
    G_aux.add_edges_from([(0, 1), (2, 3)])
    extra = _connect_induced_subgraph(G_aux, ports=(0, 1, 2, 3))
    assert len(extra) == 1  # exactly one bridge needed
    (u, v) = extra[0]
    # Endpoints must come from different original components
    assert {u, v} & {0, 1} and {u, v} & {2, 3}
    # G_aux mutated: induced subgraph now connected
    assert nx.is_connected(G_aux.subgraph((0, 1, 2, 3)))
```

- [ ] **Step 2: Run to verify fails**

Run: `pytest src/qldpc/codes/surgery/_test.py::test_connect_induced_subgraph_no_op_when_connected src/qldpc/codes/surgery/_test.py::test_connect_induced_subgraph_adds_edges_to_disconnected_components -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement `_connect_induced_subgraph`**

Add to `bridge.py`:

```python
def _connect_induced_subgraph(
    G_aux: nx.Graph,
    ports: tuple[int, ...],
) -> list[tuple[int, int]]:
    """Add edges to G_aux so that G_aux.subgraph(ports) is connected.

    Mutates G_aux. Each added edge has both endpoints in ``ports`` so it
    contributes a weight-2 row to the augmented F matrix downstream.

    Returns the list of added edges in insertion order.
    """
    ports_set = set(ports)
    added: list[tuple[int, int]] = []
    while True:
        sub = G_aux.subgraph(ports)
        comps = list(nx.connected_components(sub))
        if len(comps) <= 1:
            return added
        # Pick lowest-indexed vertex of first component and lowest of second
        c0 = sorted(comps[0])
        c1 = sorted(comps[1])
        u, v = sorted((c0[0], c1[0]))
        assert u in ports_set and v in ports_set
        if not G_aux.has_edge(u, v):
            G_aux.add_edge(u, v)
            added.append((u, v))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest src/qldpc/codes/surgery/_test.py::test_connect_induced_subgraph_no_op_when_connected src/qldpc/codes/surgery/_test.py::test_connect_induced_subgraph_adds_edges_to_disconnected_components -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/bridge.py src/qldpc/codes/surgery/_test.py
git commit -m "$(cat <<'EOF'
feat: _connect_induced_subgraph (Sec. V.A connectivity trick)

Adds weight-2 edges between components of the induced port subgraph
so SkipTree can run on a connected graph. Each added edge becomes
one extra_κ qubit downstream.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Cellulation refactor with new accounting

**Files:**
- Modify: `src/qldpc/codes/surgery/bridge.py` (replace existing `_cellulate_long_cycles`)
- Test: `src/qldpc/codes/surgery/_test.py`

The existing `_cellulate_long_cycles` returns a 4-tuple based on the old `(F, edge_qubit_to_vertices)` plumbing. Rewrite it to mutate the NetworkX graph in place and return only the list of added edges (same return shape as `_connect_induced_subgraph`).

- [ ] **Step 1: Write failing test**

```python
def test_cellulate_caps_cycle_length():
    """After cellulation, every basis cycle has length <= cap."""
    import networkx as nx
    from qldpc.codes.surgery.bridge import _cellulate_strict
    # 10-cycle: 0-1-2-...-9-0 has one length-10 basis cycle
    G_aux = nx.cycle_graph(10)
    added = _cellulate_strict(G_aux, ports=tuple(range(10)), max_len=6)
    assert len(added) >= 1
    # All basis cycles now bounded
    cycles = nx.cycle_basis(G_aux)
    assert max(len(c) for c in cycles) <= 6


def test_cellulate_no_op_when_already_short():
    """If all basis cycles are short, no edges are added."""
    import networkx as nx
    from qldpc.codes.surgery.bridge import _cellulate_strict
    G_aux = nx.cycle_graph(4)  # one 4-cycle
    added = _cellulate_strict(G_aux, ports=(0, 1, 2, 3), max_len=6)
    assert added == []
```

- [ ] **Step 2: Run to verify fails**

Run: `pytest src/qldpc/codes/surgery/_test.py::test_cellulate_caps_cycle_length src/qldpc/codes/surgery/_test.py::test_cellulate_no_op_when_already_short -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement `_cellulate_strict`**

Add to `bridge.py` (replacing the old `_cellulate_long_cycles` function body):

```python
def _cellulate_strict(
    G_aux: nx.Graph,
    ports: tuple[int, ...],
    *,
    max_len: int = 6,
) -> list[tuple[int, int]]:
    """Break cycles longer than ``max_len`` by adding chord edges.

    Mutates G_aux. Only adds chords whose endpoints lie in ``ports`` so the
    added edges remain valid weight-2 rows of the augmented F matrix.

    Returns the list of added edges in insertion order. Idempotent once all
    basis cycles fit under the cap.
    """
    ports_set = set(ports)
    added: list[tuple[int, int]] = []
    while True:
        long_cycles = [c for c in nx.cycle_basis(G_aux) if len(c) > max_len]
        if not long_cycles:
            return added
        cycle = long_cycles[0]
        n = len(cycle)
        # Try chord at antipodal position; otherwise scan for any port-port chord
        for offset in range(1, n):
            u = cycle[0]
            v = cycle[offset % n]
            if u not in ports_set or v not in ports_set:
                continue
            u, v = sorted((u, v))
            if G_aux.has_edge(u, v):
                continue
            G_aux.add_edge(u, v)
            added.append((u, v))
            break
        else:
            raise RuntimeError(
                f"No port-port chord found to cellulate cycle of length {n}; "
                f"ports={ports!r}, cycle={cycle!r}"
            )
```

Delete the old `_cellulate_long_cycles` function (the one with the 4-tuple return).

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest src/qldpc/codes/surgery/_test.py::test_cellulate_caps_cycle_length src/qldpc/codes/surgery/_test.py::test_cellulate_no_op_when_already_short -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/bridge.py src/qldpc/codes/surgery/_test.py
git commit -m "$(cat <<'EOF'
refactor: _cellulate_strict returns added-edge list, in-place mutation

Aligns with _connect_induced_subgraph's API; drops the
F-matrix-accounting tuple form (the new build_bridge does its own
accounting via extra_kappa_s rows).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `build_gadget_augmented` helper in `gadget.py`

**Files:**
- Modify: `src/qldpc/codes/surgery/gadget.py`
- Test: `src/qldpc/codes/surgery/_test.py`

Spec §2 step 7: after we know the extra weight-2 F rows from connectivity + cellulation, rebuild a `GadgetLayout` whose `F`, `G`, `HX_merged`, `HZ_merged` reflect the augmentation. Reuses `_step2_gauge_fix` and `_step3_assemble`.

- [ ] **Step 1: Write failing test**

```python
def test_build_gadget_augmented_extends_F_and_recomputes_G():
    """Augmenting with one weight-2 row adds a column to merged matrices and recomputes G."""
    from qldpc.codes.surgery.gadget import build_gadget, build_gadget_augmented
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    # Pick two ports in V_0; create one extra weight-2 row connecting them
    v0_a, v0_b = g.V0[0], g.V0[1]
    extra_F = np.zeros((1, len(g.V0)), dtype=np.uint8)
    idx_a = g.V0.index(v0_a)
    idx_b = g.V0.index(v0_b)
    extra_F[0, idx_a] = 1
    extra_F[0, idx_b] = 1
    g_aug = build_gadget_augmented(code, x, extra_F, basis=Pauli.X)

    # F_aug = [F | extra_F] vertically stacked
    assert g_aug.F.shape == (g.F.shape[0] + 1, g.F.shape[1])
    assert np.array_equal(g_aug.F[: g.F.shape[0]], g.F)
    assert np.array_equal(g_aug.F[g.F.shape[0]:], extra_F)
    # HX_merged has one extra column (one extra κ qubit); same number of rows
    assert g_aug.HX_merged.shape == (g.HX_merged.shape[0], g.HX_merged.shape[1] + 1)
    # CSS commutation
    assert np.array_equal((g_aug.HX_merged @ g_aug.HZ_merged.T) % 2, 0)
```

- [ ] **Step 2: Run to verify fails**

Run: `pytest src/qldpc/codes/surgery/_test.py::test_build_gadget_augmented_extends_F_and_recomputes_G -v`
Expected: FAIL (`ImportError`).

- [ ] **Step 3: Implement `build_gadget_augmented`**

Add to `src/qldpc/codes/surgery/gadget.py` (after `build_gadget`):

```python
def build_gadget_augmented(
    code: CSSCode,
    x: np.ndarray,
    F_extra: np.ndarray,
    *,
    basis: PauliXZ = Pauli.X,
) -> GadgetLayout:
    """Rebuild a GadgetLayout with F augmented by extra weight-2 rows.

    Each row of ``F_extra`` has weight 2 and corresponds to a new κ qubit not
    backed by any original Z-check (basis=X) or X-check (basis=Z). The function:

    1. Stacks F_aug = [F; F_extra].
    2. Recomputes G_aug = ker(F_aug^T) via _step2_gauge_fix.
    3. Calls _step3_assemble with the original V_0 / C_0 plus the new κ rows.
       The extra columns of tilde_F are all zero (no original check sits on the
       new κ qubits).

    The returned ``GadgetLayout.C0`` and ``kappa_qubits`` are extended to cover
    the new κ qubits; the new κ indices come after the original ones.
    """
    x = np.asarray(x).astype(np.uint8)
    V0, C0, F = _step1_restriction(code, x, basis=basis)
    F_extra = np.asarray(F_extra).astype(np.uint8)
    if F_extra.shape[1] != len(V0):
        raise ValueError(
            f"F_extra has {F_extra.shape[1]} columns; expected {len(V0)} (= |V_0|)"
        )
    if F_extra.size and not np.all(F_extra.sum(axis=1) == 2):
        bad = np.flatnonzero(F_extra.sum(axis=1) != 2).tolist()
        raise ValueError(f"F_extra rows {bad} have weight != 2; required weight 2.")

    F_aug = np.vstack([F, F_extra]).astype(np.uint8)
    G_aug = _step2_gauge_fix(F_aug)

    # _step3_assemble computes tilde_F by indexing into C_0; we need an extended
    # C_0_aug that has the new rows as sentinels (their tilde_F columns must be 0).
    # Trick: pass C_0_aug = C_0 + (-1, -1, ...) sentinels which fall outside [0, mZ),
    # so the tilde_F loop sets nothing for those positions.
    n_extra = F_extra.shape[0]
    C0_aug = tuple(C0) + tuple([-1] * n_extra)
    HX_aug, HZ_aug = _step3_assemble(
        code, V0, C0_aug, F_aug, G_aug, basis=basis,
    )
    kappa_qubits_aug = tuple(range(code.num_qudits, code.num_qudits + len(C0_aug)))
    return GadgetLayout(
        code=code, x=x, V0=V0, C0=C0_aug, F=F_aug, G=G_aug,
        HX_merged=HX_aug, HZ_merged=HZ_aug, kappa_qubits=kappa_qubits_aug,
        basis=basis,
    )
```

Also update `_step3_assemble` (in the same file) to skip the `tilde_F[j, k] = 1` write when `j < 0` (sentinel for extra rows). Locate the loop at line ~111:

```python
    for k, j in enumerate(C0):
        if j < 0:
            continue        # NEW — sentinel for extra-κ rows from build_gadget_augmented
        F_tilde[j, k] = 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/qldpc/codes/surgery/_test.py::test_build_gadget_augmented_extends_F_and_recomputes_G -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/gadget.py src/qldpc/codes/surgery/_test.py
git commit -m "$(cat <<'EOF'
feat: build_gadget_augmented(code, x, F_extra, basis)

Rebuilds a GadgetLayout on top of F stacked with extra weight-2
rows from the bridge augmentation step. Recomputes G_aug via
_step2_gauge_fix; tilde_F leaves the new κ columns at zero via a
sentinel (-1) in C0_aug.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: New `Bridge` dataclass; delete obsolete helpers

**Files:**
- Modify: `src/qldpc/codes/surgery/bridge.py` (Bridge dataclass + helper deletions)
- Modify: `src/qldpc/codes/surgery/_test.py` (delete old dataclass-fields tests)

Replace the existing `Bridge` dataclass with the new shape from spec §1. Delete `_build_path_graph_U_B`, `_solve_chi_z_bridge_choices`, `_running_xor_b_c`, `_label_inverse` (no longer used). Don't yet rewrite `build_bridge` — that's Task 7.

- [ ] **Step 1: Write failing test**

```python
def test_bridge_dataclass_fields_universal_adapter():
    """Bridge dataclass exposes the universal-adapter fields from spec §1."""
    import dataclasses
    from qldpc.codes.surgery.bridge import Bridge
    fields = {f.name for f in dataclasses.fields(Bridge)}
    assert fields == {
        "width", "basis",
        "port_l", "port_r",
        "label_l", "label_r",
        "extra_kappa_l", "extra_kappa_r",
        "T_l", "T_r", "H_R",
        "g_l_aug", "g_r_aug",
    }
```

Also delete (from `_test.py`):
- `test_bridge_dataclass_fields` (around line 290 — old field set)
- `test_path_graph_U_B_telescoping`
- `test_webster_table_i_bridge_width_exact`
- `test_build_bridge_intracode_chi_endpoint_extensions`
- `test_skip_tree_path_graph_returns_identity`
- `test_alpha_star_yields_joint_op_on_webster`
- `test_bridge_has_basis_field_and_inherits_from_gadgets` (will be re-added in Task 7)
- `test_build_bridge_rejects_basis_mismatch` (will be re-added in Task 7)
- `test_build_bridge_intercode_two_different_codes` (will be re-added in Task 7)

Capture the line ranges first by running:

```bash
grep -n "^def test_bridge_dataclass_fields\b\|^def test_path_graph_U_B_telescoping\b\|^def test_webster_table_i_bridge_width_exact\b\|^def test_build_bridge_intracode_chi_endpoint_extensions\b\|^def test_skip_tree_path_graph_returns_identity\b\|^def test_alpha_star_yields_joint_op_on_webster\b\|^def test_bridge_has_basis_field_and_inherits_from_gadgets\b\|^def test_build_bridge_rejects_basis_mismatch\b\|^def test_build_bridge_intercode_two_different_codes\b" src/qldpc/codes/surgery/_test.py
```

Then delete each function body (def line through the next `^def ` or `@pytest`).

- [ ] **Step 2: Run to verify fails**

Run: `pytest src/qldpc/codes/surgery/_test.py::test_bridge_dataclass_fields_universal_adapter -v`
Expected: FAIL — current Bridge has different fields.

- [ ] **Step 3: Replace `Bridge` dataclass and remove obsolete helpers**

Replace the `Bridge` dataclass in `bridge.py` (around line 22):

```python
@dataclasses.dataclass(frozen=True, eq=False)
class Bridge:
    """Universal adapter between two GadgetLayouts (arXiv:2410.03628 §IV / §VII).

    Attributes match docs/superpowers/specs/2026-06-09-joint-ppm-bridge-design.md §1.
    """
    width: int                                  # w = |𝒜| (adapter qubits)
    basis: PauliXZ                              # X or Z (symmetric dual)
    port_l: tuple[int, ...]                     # 𝒫_l* ⊆ V_0^(l), length w
    port_r: tuple[int, ...]                     # 𝒫_r* ⊆ V_0^(r), length w
    label_l: tuple[int, ...]                    # label_l[i] = SkipTree label of V_0^(l)[i]; -1 if i ∉ 𝒫_l*
    label_r: tuple[int, ...]                    # same for right side
    extra_kappa_l: np.ndarray                   # (e_l, |V_0^(l)|) F_2; weight-2 rows added
    extra_kappa_r: np.ndarray                   # (e_r, |V_0^(r)|) F_2
    T_l: np.ndarray                             # (w-1, |C_0^(l)| + e_l) F_2 (3,2)-sparse
    T_r: np.ndarray                             # (w-1, |C_0^(r)| + e_r) F_2
    H_R: np.ndarray                             # (w-1, w) canonical rep code parity
    g_l_aug: GadgetLayout                       # gadget rebuilt over F_aug^(l)
    g_r_aug: GadgetLayout                       # gadget rebuilt over F_aug^(r)
```

Delete from `bridge.py`:
- `_build_path_graph_U_B` (line 33–41)
- `_label_inverse` (line 151–160)
- `_running_xor_b_c` (line 163–169)
- `_solve_chi_z_bridge_choices` (line 172–228)
- The old `_build_auxiliary_graph_from_F` (line 132–148) — replaced by `_build_aux_graph_strict` from Task 2.

Also delete the old `_cellulate_long_cycles` if Task 4's edit didn't already remove it.

The old `build_bridge` (line 231–287) will be replaced in Task 7; for now, replace its body with `raise NotImplementedError("rewritten in Task 7")` so the module imports clean.

- [ ] **Step 4: Run dataclass test + smoke import**

Run:
```bash
pytest src/qldpc/codes/surgery/_test.py::test_bridge_dataclass_fields_universal_adapter -v
python -c "from qldpc.codes.surgery.bridge import Bridge; print(Bridge)"
```
Expected: PASS on the test; import prints `<class 'qldpc.codes.surgery.bridge.Bridge'>`.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/bridge.py src/qldpc/codes/surgery/_test.py
git commit -m "$(cat <<'EOF'
refactor: replace Bridge dataclass + drop path-graph helpers

Drops _build_path_graph_U_B, chi_endpoint_extensions,
_solve_chi_z_bridge_choices, _label_inverse, _running_xor_b_c,
_build_auxiliary_graph_from_F.  New Bridge dataclass exposes the
universal-adapter fields (port_l/r, label_l/r, extra_kappa_l/r,
T_l/r, H_R, g_l/r_aug); build_bridge body temporarily raises and
will be rewritten in the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: New `build_bridge` orchestrator

**Files:**
- Modify: `src/qldpc/codes/surgery/bridge.py`
- Test: `src/qldpc/codes/surgery/_test.py`

Implement the 7-step algorithm from spec §2.

- [ ] **Step 1: Write failing tests**

```python
def test_build_bridge_smoke_steane_intracode():
    """Steane × Steane intra-code joint X̄ X̄: build_bridge returns valid Bridge."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    code = codes.SteaneCode()
    x1 = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    x2 = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)  # same logical
    g_l = build_gadget(code, x1, basis=Pauli.X)
    g_r = build_gadget(code, x2, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    assert bridge.width == min(len(g_l.V0), len(g_r.V0))
    assert bridge.basis is Pauli.X
    assert len(bridge.port_l) == bridge.width
    assert len(bridge.port_r) == bridge.width
    assert bridge.T_l.shape == (bridge.width - 1, bridge.g_l_aug.F.shape[0])
    assert bridge.T_r.shape == (bridge.width - 1, bridge.g_r_aug.F.shape[0])
    assert bridge.H_R.shape == (bridge.width - 1, bridge.width)


def test_build_bridge_skiptree_invariant_holds():
    """T_s · G_s_aug · P_s = H_R for both sides on Steane × Steane."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(code, x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)

    for side in ("l", "r"):
        T = getattr(bridge, f"T_{side}")
        g_aug = getattr(bridge, f"g_{side}_aug")
        label = getattr(bridge, f"label_{side}")
        port = getattr(bridge, f"port_{side}")
        # G_aug = F_aug (incidence: rows = edges = κ qubits, cols = V_0 vertices)
        G_aug = g_aug.F.astype(np.int_)
        # P_s: |V_0^(s)| × w; P_s[v, k] = 1 iff v ∈ port AND label[v] == k
        P = np.zeros((G_aug.shape[1], bridge.width), dtype=np.int_)
        for v_idx, lab in enumerate(label):
            if lab >= 0:
                P[v_idx, lab] = 1
        lhs = (T @ G_aug @ P) % 2
        assert np.array_equal(lhs, bridge.H_R), f"side {side}:\n{lhs}\nvs\n{bridge.H_R}"


def test_build_bridge_rejects_basis_mismatch():
    """Bridge requires g_l.basis == g_r.basis."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(code, z, basis=Pauli.Z)
    with pytest.raises(ValueError, match=r"basis"):
        build_bridge(g_l, g_r)
```

- [ ] **Step 2: Run to verify fails**

Run: `pytest src/qldpc/codes/surgery/_test.py -k "build_bridge_smoke_steane or build_bridge_skiptree_invariant or build_bridge_rejects_basis" -v`
Expected: FAIL on all three (current `build_bridge` raises `NotImplementedError`).

- [ ] **Step 3: Implement `build_bridge`**

Replace the body of `build_bridge` in `bridge.py`:

```python
def build_bridge(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    *,
    port_subset_l: tuple[int, ...] | None = None,
    port_subset_r: tuple[int, ...] | None = None,
    spanning_tree_root_l: int = 0,
    spanning_tree_root_r: int = 0,
    cellulate_max_len: int = 6,
) -> Bridge:
    """Universal-adapter bridge between two gadgets (arXiv:2410.03628 §IV).

    See docs/superpowers/specs/2026-06-09-joint-ppm-bridge-design.md §2 for the
    7-step recipe.
    """
    if g_l.basis is not g_r.basis:
        raise ValueError(
            f"build_bridge requires g_l.basis == g_r.basis, "
            f"got {g_l.basis!r} vs {g_r.basis!r}"
        )
    basis = g_l.basis

    # Step 1: auxiliary graphs
    G_l_aux, _ = _build_aux_graph_strict(g_l.F)
    G_r_aux, _ = _build_aux_graph_strict(g_r.F)

    # Step 2: port subsets + width
    port_l_all = tuple(port_subset_l) if port_subset_l is not None else tuple(range(len(g_l.V0)))
    port_r_all = tuple(port_subset_r) if port_subset_r is not None else tuple(range(len(g_r.V0)))
    width = min(len(port_l_all), len(port_r_all))
    if width < 2:
        raise ValueError(f"bridge width must be >= 2, got {width}")
    port_l = port_l_all[:width]
    port_r = port_r_all[:width]

    # Step 3: induced-subgraph connectivity augmentation
    extras_l_conn = _connect_induced_subgraph(G_l_aux, port_l)
    extras_r_conn = _connect_induced_subgraph(G_r_aux, port_r)

    # Step 4: cellulation
    extras_l_cell = _cellulate_strict(G_l_aux, port_l, max_len=cellulate_max_len)
    extras_r_cell = _cellulate_strict(G_r_aux, port_r, max_len=cellulate_max_len)

    # Collect extra weight-2 rows (one per added edge) for each side
    def _edges_to_F_extra(edges: list[tuple[int, int]], n_V0: int) -> np.ndarray:
        out = np.zeros((len(edges), n_V0), dtype=np.uint8)
        for r, (u, v) in enumerate(edges):
            out[r, u] = 1
            out[r, v] = 1
        return out

    extras_l_edges = extras_l_conn + extras_l_cell
    extras_r_edges = extras_r_conn + extras_r_cell
    extra_kappa_l = _edges_to_F_extra(extras_l_edges, len(g_l.V0))
    extra_kappa_r = _edges_to_F_extra(extras_r_edges, len(g_r.V0))

    # Step 7 (early): rebuild augmented gadgets so we have F_aug + G_aug + tilde_F
    from .gadget import build_gadget_augmented
    g_l_aug = build_gadget_augmented(g_l.code, g_l.x, extra_kappa_l, basis=basis)
    g_r_aug = build_gadget_augmented(g_r.code, g_r.x, extra_kappa_r, basis=basis)

    # Step 5: SkipTree on induced subgraph (relabel to [0, |port|) first so the
    # internal n×n P allocation in _skip_tree is square and valid); embed back.
    def _run_skiptree(
        G_aux_full: nx.Graph,
        port: tuple[int, ...],
        root_port_idx: int,
        F_aug: np.ndarray,
    ) -> tuple[np.ndarray, list[int]]:
        sub_orig = G_aux_full.subgraph(port).copy()
        port_sorted = sorted(port)
        new_of_orig = {orig: new for new, orig in enumerate(port_sorted)}
        orig_of_new = {new: orig for orig, new in new_of_orig.items()}
        sub_relab = nx.relabel_nodes(sub_orig, new_of_orig, copy=True)
        # Take a spanning tree (Algorithm 1 of paper expects a tree input). MST
        # is deterministic; for unweighted graphs nx returns a BFS-like tree.
        sub_tree = nx.minimum_spanning_tree(sub_relab)
        tree_edges = sorted(tuple(sorted(e)) for e in sub_tree.edges())
        edge_idx_tree = {e: i for i, e in enumerate(tree_edges)}
        root_orig = port[root_port_idx]
        root_relab = new_of_orig[root_orig]
        T_relab, P_relab = _skip_tree_fullrank(sub_tree, root=root_relab, edge_index=edge_idx_tree)
        # labels[orig_v_idx] = k iff orig_v ∈ port  (else -1)
        labels = [-1] * F_aug.shape[1]
        for new_v in range(len(port)):
            orig_v = orig_of_new[new_v]
            nz = np.flatnonzero(P_relab[new_v])
            assert len(nz) == 1, f"vertex {orig_v} (relab {new_v}) has {len(nz)} labels"
            labels[orig_v] = int(nz[0])
        # T_relab columns are spanning-tree edges (relabeled). Map each F_aug
        # row to a tree-edge column if applicable; F_aug rows that are non-tree
        # edges or that touch a non-port vertex stay zero in T_full.
        T_full = np.zeros((T_relab.shape[0], F_aug.shape[0]), dtype=np.int_)
        for r in range(F_aug.shape[0]):
            cols = np.flatnonzero(F_aug[r])
            if len(cols) != 2:
                continue
            u_orig, v_orig = sorted(int(x) for x in cols)
            if u_orig not in new_of_orig or v_orig not in new_of_orig:
                continue
            e_relab = tuple(sorted((new_of_orig[u_orig], new_of_orig[v_orig])))
            if e_relab in edge_idx_tree:
                T_full[:, r] = T_relab[:, edge_idx_tree[e_relab]]
        return T_full.astype(np.int_), labels

    T_l, label_l = _run_skiptree(G_l_aux, port_l, 0, g_l_aug.F)
    T_r, label_r = _run_skiptree(G_r_aux, port_r, 0, g_r_aug.F)

    return Bridge(
        width=width,
        basis=basis,
        port_l=port_l,
        port_r=port_r,
        label_l=tuple(label_l),
        label_r=tuple(label_r),
        extra_kappa_l=extra_kappa_l.astype(np.uint8),
        extra_kappa_r=extra_kappa_r.astype(np.uint8),
        T_l=T_l,
        T_r=T_r,
        H_R=_canonical_H_R(width).astype(np.int_),
        g_l_aug=g_l_aug,
        g_r_aug=g_r_aug,
    )
```

`spanning_tree_root_l` and `spanning_tree_root_r` are accepted but unused for now (default-0 implicit through `port[0]`); future Hamiltonian-path optimization (Remark 20) hooks into them.

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest src/qldpc/codes/surgery/_test.py -k "build_bridge_smoke_steane or build_bridge_skiptree_invariant or build_bridge_rejects_basis" -v`
Expected: PASS on all three.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/bridge.py src/qldpc/codes/surgery/_test.py
git commit -m "$(cat <<'EOF'
feat: build_bridge — SkipTree-driven universal adapter

Implements docs/superpowers/specs/2026-06-09-joint-ppm-bridge-design.md
§2 steps 1-7: aux graph build, port subset, induced-subgraph
connectivity augment, cellulation, SkipTree, label_s, augmented
gadget rebuild. SkipTree invariant T·G·P=H_R verified by test.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `_stitch_to_joint_csscode` — inter-code, basis=X

**Files:**
- Modify: `src/qldpc/codes/surgery/circuit.py` (lines 76–175 currently)
- Test: `src/qldpc/codes/surgery/_test.py`

Replace the current path-graph stitching with the §3 block construction. Inter-code path only; intra-code in Task 9.

- [ ] **Step 1: Write failing tests**

```python
def test_stitch_intercode_basis_x_css_commutation():
    """Inter-code Steane × Steane joint X̄X̄ merged code commutes."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import _stitch_to_joint_csscode
    code1 = codes.SteaneCode()
    code2 = codes.SteaneCode()
    x1 = np.asarray(code1.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    x2 = np.asarray(code2.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code1, x1, basis=Pauli.X)
    g_r = build_gadget(code2, x2, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    HZ = np.asarray(merged.matrix_z).astype(np.int_)
    assert np.array_equal((HX @ HZ.T) % 2, np.zeros((HX.shape[0], HZ.shape[0]), dtype=np.int_))


def test_stitch_intercode_basis_x_k_reduces_by_one():
    """k_joint = k_l + k_r - 1 for inter-code Steane × Steane joint X̄X̄."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import _stitch_to_joint_csscode
    code1 = codes.SteaneCode()
    code2 = codes.SteaneCode()
    x1 = np.asarray(code1.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    x2 = np.asarray(code2.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code1, x1, basis=Pauli.X)
    g_r = build_gadget(code2, x2, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    assert merged.dimension == code1.dimension + code2.dimension - 1


def test_stitch_intercode_basis_x_joint_logical_in_stabilizer():
    """(x_1, x_2, 0, 0, 0) lies in rowspan(H_X^merged) — joint X̄_l X̄_r is a stabilizer."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import _stitch_to_joint_csscode
    code1 = codes.SteaneCode()
    code2 = codes.SteaneCode()
    x1 = np.asarray(code1.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    x2 = np.asarray(code2.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code1, x1, basis=Pauli.X)
    g_r = build_gadget(code2, x2, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    import galois
    GF2 = galois.GF(2)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    n_l = code1.num_qudits
    n_r = code2.num_qudits
    joint = np.zeros(HX.shape[1], dtype=np.int_)
    joint[:n_l] = x1
    joint[n_l : n_l + n_r] = x2
    augmented = np.vstack([HX, joint.reshape(1, -1)])
    assert np.linalg.matrix_rank(GF2(HX.tolist())) == np.linalg.matrix_rank(GF2(augmented.tolist()))


def test_stitch_intercode_basis_x_singletons_excluded():
    """(x_1, 0, ...) and (0, x_2, ...) alone are NOT in rowspan(H_X^merged)."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import _stitch_to_joint_csscode
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(codes.SteaneCode(), x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    import galois
    GF2 = galois.GF(2)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    n_l = code.num_qudits
    base = np.linalg.matrix_rank(GF2(HX.tolist()))
    for which in ("left", "right"):
        single = np.zeros(HX.shape[1], dtype=np.int_)
        if which == "left":
            single[:n_l] = x
        else:
            single[n_l : 2 * n_l] = x
        augmented = np.vstack([HX, single.reshape(1, -1)])
        assert np.linalg.matrix_rank(GF2(augmented.tolist())) == base + 1, which
```

- [ ] **Step 2: Run to verify fails**

Run: `pytest src/qldpc/codes/surgery/_test.py -k "stitch_intercode_basis_x" -v`
Expected: FAIL because current `_stitch_to_joint_csscode` reads bridge fields (`.U_B`, etc.) that no longer exist.

- [ ] **Step 3: Rewrite `_stitch_to_joint_csscode` for inter-code basis=X**

Replace function in `src/qldpc/codes/surgery/circuit.py` (lines 76–175). The new function uses `bridge.g_l_aug`, `bridge.g_r_aug`, `bridge.T_l`, `bridge.T_r`, `bridge.H_R`, and label arrays.

```python
def _stitch_to_joint_csscode(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
) -> CSSCode:
    """Assemble merged CSSCode for two-PPM surgery (spec §3 block tables)."""
    intercode = g_l.code is not g_r.code
    if not intercode:
        return _stitch_intracode_joint_csscode(g_l, g_r, bridge)  # added in Task 9
    if bridge.basis is not Pauli.X:
        return _stitch_intercode_joint_csscode_basis_z(g_l, g_r, bridge)  # added in Task 10

    field = g_l.code.field
    g_l_aug, g_r_aug = bridge.g_l_aug, bridge.g_r_aug
    n_l = g_l.code.num_qudits
    n_r = g_r.code.num_qudits
    k_l = g_l_aug.F.shape[0]      # |C_0^(l)| + e_l
    k_r = g_r_aug.F.shape[0]
    w = bridge.width
    n_merged = n_l + n_r + k_l + k_r + w
    mX_l = g_l.code.matrix_x.shape[0]
    mX_r = g_r.code.matrix_x.shape[0]
    mZ_l = g_l.code.matrix_z.shape[0]
    mZ_r = g_r.code.matrix_z.shape[0]
    r_l = g_l_aug.G.shape[0]      # r_l_aug
    r_r = g_r_aug.G.shape[0]

    # Column ranges
    cl_data = slice(0, n_l)
    cr_data = slice(n_l, n_l + n_r)
    cl_kappa = slice(n_l + n_r, n_l + n_r + k_l)
    cr_kappa = slice(n_l + n_r + k_l, n_l + n_r + k_l + k_r)
    c_adapter = slice(n_l + n_r + k_l + k_r, n_merged)

    HX = np.zeros((mX_l + mX_r + len(g_l.V0) + len(g_r.V0), n_merged), dtype=np.int_)
    HZ = np.zeros((mZ_l + mZ_r + r_l + r_r + (w - 1), n_merged), dtype=np.int_)

    HX_l = np.asarray(g_l_aug.HX_merged).astype(np.int_)
    HX_r = np.asarray(g_r_aug.HX_merged).astype(np.int_)
    HZ_l = np.asarray(g_l_aug.HZ_merged).astype(np.int_)
    HZ_r = np.asarray(g_r_aug.HZ_merged).astype(np.int_)

    # H_X rows: data H_X^(l), data H_X^(r), χ^(l), χ^(r)
    HX[: mX_l, cl_data] = HX_l[: mX_l, : n_l]
    HX[mX_l : mX_l + mX_r, cr_data] = HX_r[: mX_r, : n_r]
    # χ^(l) = HX_l[mX_l:, :]: cols [: n_l] -> cl_data, cols [n_l:] -> cl_kappa
    chi_l_rows = HX_l[mX_l :, :]
    HX[mX_l + mX_r : mX_l + mX_r + len(g_l.V0), cl_data] = chi_l_rows[:, : n_l]
    HX[mX_l + mX_r : mX_l + mX_r + len(g_l.V0), cl_kappa] = chi_l_rows[:, n_l :]
    chi_r_rows = HX_r[mX_r :, :]
    HX[mX_l + mX_r + len(g_l.V0) :, cr_data] = chi_r_rows[:, : n_r]
    HX[mX_l + mX_r + len(g_l.V0) :, cr_kappa] = chi_r_rows[:, n_r :]
    # Π_l, Π_r adapter extensions on χ rows
    for v_idx, lab in enumerate(bridge.label_l):
        if lab >= 0:
            HX[mX_l + mX_r + v_idx, c_adapter.start + lab] = 1
    for v_idx, lab in enumerate(bridge.label_r):
        if lab >= 0:
            HX[mX_l + mX_r + len(g_l.V0) + v_idx, c_adapter.start + lab] = 1

    # H_Z rows: data H_Z^(l) extended, data H_Z^(r) extended, G^(l)_aug, G^(r)_aug, new cycle-Z
    HZ[: mZ_l, cl_data] = HZ_l[: mZ_l, : n_l]
    HZ[: mZ_l, cl_kappa] = HZ_l[: mZ_l, n_l :]
    HZ[mZ_l : mZ_l + mZ_r, cr_data] = HZ_r[: mZ_r, : n_r]
    HZ[mZ_l : mZ_l + mZ_r, cr_kappa] = HZ_r[: mZ_r, n_r :]
    HZ[mZ_l + mZ_r : mZ_l + mZ_r + r_l, cl_kappa] = HZ_l[mZ_l :, n_l :]
    HZ[mZ_l + mZ_r + r_l : mZ_l + mZ_r + r_l + r_r, cr_kappa] = HZ_r[mZ_r :, n_r :]
    # New cycle Z-checks: [T_l | T_r | H_R]
    new_z_start = mZ_l + mZ_r + r_l + r_r
    HZ[new_z_start :, cl_kappa] = bridge.T_l
    HZ[new_z_start :, cr_kappa] = bridge.T_r
    HZ[new_z_start :, c_adapter] = bridge.H_R

    return CSSCode(field(HX), field(HZ), is_subsystem_code=False)
```

You'll need stubs for the not-yet-implemented branches so the function imports cleanly:

```python
def _stitch_intracode_joint_csscode(g_l, g_r, bridge):
    raise NotImplementedError("intra-code stitching — Task 9")

def _stitch_intercode_joint_csscode_basis_z(g_l, g_r, bridge):
    raise NotImplementedError("basis=Z stitching — Task 10")
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest src/qldpc/codes/surgery/_test.py -k "stitch_intercode_basis_x" -v`
Expected: PASS on all four.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/circuit.py src/qldpc/codes/surgery/_test.py
git commit -m "$(cat <<'EOF'
feat: _stitch_to_joint_csscode inter-code basis=X (universal adapter)

Implements spec §3 block tables for the inter-code path: new cycle
Z-checks [T_l | T_r | H_R] go into H_Z^merged (not H_X) per the
SkipTree key identity.  k_joint = k_l + k_r - 1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `_stitch_to_joint_csscode` — intra-code, basis=X

**Files:**
- Modify: `src/qldpc/codes/surgery/circuit.py`
- Test: `src/qldpc/codes/surgery/_test.py`

Spec §3 "Intra-code, basis=X": single shared `data` column block, χ^(l) and χ^(r) extend onto the SAME data columns. Overlap auto-handled by row distinctness.

- [ ] **Step 1: Write failing tests**

```python
def test_stitch_intracode_basis_x_css_commutation():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import _stitch_to_joint_csscode
    code = codes.SteaneCode()
    x1 = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    # Use Pauli.X logical 0 for both (same V_0); intra-code test
    g_l = build_gadget(code, x1, basis=Pauli.X)
    g_r = build_gadget(code, x1, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    HZ = np.asarray(merged.matrix_z).astype(np.int_)
    assert np.array_equal((HX @ HZ.T) % 2, 0)


def test_stitch_intracode_basis_x_k_reduces_by_one():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import _stitch_to_joint_csscode
    code = codes.SteaneCode()
    x1 = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x1, basis=Pauli.X)
    g_r = build_gadget(code, x1, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    assert merged.dimension == code.dimension - 1
```

- [ ] **Step 2: Run to verify fails**

Run: `pytest src/qldpc/codes/surgery/_test.py -k "stitch_intracode_basis_x" -v`
Expected: FAIL (`NotImplementedError`).

- [ ] **Step 3: Implement `_stitch_intracode_joint_csscode`**

In `circuit.py`:

```python
def _stitch_intracode_joint_csscode(g_l, g_r, bridge):
    """Spec §3 intra-code: shared data column block, χ^(l/r) extend onto same data cols."""
    assert g_l.code is g_r.code
    assert bridge.basis is Pauli.X  # basis=Z handled separately
    field = g_l.code.field
    g_l_aug, g_r_aug = bridge.g_l_aug, bridge.g_r_aug
    n = g_l.code.num_qudits
    k_l = g_l_aug.F.shape[0]
    k_r = g_r_aug.F.shape[0]
    w = bridge.width
    n_merged = n + k_l + k_r + w
    mX = g_l.code.matrix_x.shape[0]
    mZ = g_l.code.matrix_z.shape[0]
    r_l = g_l_aug.G.shape[0]
    r_r = g_r_aug.G.shape[0]

    c_data = slice(0, n)
    cl_kappa = slice(n, n + k_l)
    cr_kappa = slice(n + k_l, n + k_l + k_r)
    c_adapter = slice(n + k_l + k_r, n_merged)

    HX_l = np.asarray(g_l_aug.HX_merged).astype(np.int_)
    HX_r = np.asarray(g_r_aug.HX_merged).astype(np.int_)
    HZ_l = np.asarray(g_l_aug.HZ_merged).astype(np.int_)
    HZ_r = np.asarray(g_r_aug.HZ_merged).astype(np.int_)

    HX = np.zeros((mX + len(g_l.V0) + len(g_r.V0), n_merged), dtype=np.int_)
    HZ = np.zeros((mZ + r_l + r_r + (w - 1), n_merged), dtype=np.int_)

    # H_X: shared data H_X (from g_l_aug or g_r_aug, same)
    HX[: mX, c_data] = HX_l[: mX, : n]
    # χ^(l), χ^(r) onto shared data
    chi_l = HX_l[mX :, :]
    chi_r = HX_r[mX :, :]
    HX[mX : mX + len(g_l.V0), c_data] = chi_l[:, : n]
    HX[mX : mX + len(g_l.V0), cl_kappa] = chi_l[:, n :]
    HX[mX + len(g_l.V0) :, c_data] = chi_r[:, : n]
    HX[mX + len(g_l.V0) :, cr_kappa] = chi_r[:, n :]
    for v_idx, lab in enumerate(bridge.label_l):
        if lab >= 0:
            HX[mX + v_idx, c_adapter.start + lab] = 1
    for v_idx, lab in enumerate(bridge.label_r):
        if lab >= 0:
            HX[mX + len(g_l.V0) + v_idx, c_adapter.start + lab] = 1

    # H_Z: shared data H_Z with κ extension on BOTH sides
    HZ[: mZ, c_data] = HZ_l[: mZ, : n]
    HZ[: mZ, cl_kappa] = HZ_l[: mZ, n :]
    HZ[: mZ, cr_kappa] = HZ_r[: mZ, n :]
    HZ[mZ : mZ + r_l, cl_kappa] = HZ_l[mZ :, n :]
    HZ[mZ + r_l : mZ + r_l + r_r, cr_kappa] = HZ_r[mZ :, n :]
    new_z_start = mZ + r_l + r_r
    HZ[new_z_start :, cl_kappa] = bridge.T_l
    HZ[new_z_start :, cr_kappa] = bridge.T_r
    HZ[new_z_start :, c_adapter] = bridge.H_R

    return CSSCode(field(HX), field(HZ), is_subsystem_code=False)
```

- [ ] **Step 4: Run tests**

Run: `pytest src/qldpc/codes/surgery/_test.py -k "stitch_intracode_basis_x" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/circuit.py src/qldpc/codes/surgery/_test.py
git commit -m "$(cat <<'EOF'
feat: _stitch_to_joint_csscode intra-code basis=X

Shared data column; χ^(l) and χ^(r) extend onto SAME data cols.
Overlap auto-handled (V_0 intersection contributes two distinct
χ rows; α* sum cancels at overlap qubits).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: basis=Z symmetric dual (inter-code + intra-code)

**Files:**
- Modify: `src/qldpc/codes/surgery/circuit.py`
- Test: `src/qldpc/codes/surgery/_test.py`

Swap X ↔ Z roles everywhere. χ rows live in `H_Z^merged`; new cycle-X-checks `[T_l | T_r | H_R]` live in `H_X^merged`.

- [ ] **Step 1: Write failing tests (parametrized)**

```python
@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_stitch_intercode_both_bases_commute_and_singletons_excluded(basis):
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import _stitch_to_joint_csscode
    code = codes.SteaneCode()
    if basis is Pauli.X:
        x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    else:
        x = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=basis)
    g_r = build_gadget(codes.SteaneCode(), x, basis=basis)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    HZ = np.asarray(merged.matrix_z).astype(np.int_)
    assert np.array_equal((HX @ HZ.T) % 2, 0)
    assert merged.dimension == 2 * code.dimension - 1


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_stitch_intracode_both_bases_commute(basis):
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import _stitch_to_joint_csscode
    code = codes.SteaneCode()
    if basis is Pauli.X:
        x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    else:
        x = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=basis)
    g_r = build_gadget(code, x, basis=basis)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    HZ = np.asarray(merged.matrix_z).astype(np.int_)
    assert np.array_equal((HX @ HZ.T) % 2, 0)
```

- [ ] **Step 2: Run to verify fails on basis=Z**

Run: `pytest src/qldpc/codes/surgery/_test.py -k "intercode_both_bases or intracode_both_bases" -v`
Expected: PASS on basis=X (from Tasks 8 + 9), FAIL on basis=Z (`NotImplementedError`).

- [ ] **Step 3: Implement basis=Z duals**

In `circuit.py`, factor out a common helper and dispatch:

```python
def _stitch_intercode_joint_csscode_basis_z(g_l, g_r, bridge):
    """Symmetric dual of inter-code basis=X: χ → H_Z, new cycle → H_X."""
    # Implementation mirror of _stitch_to_joint_csscode inter-code path with X ↔ Z
    assert g_l.code is not g_r.code
    assert bridge.basis is Pauli.Z
    field = g_l.code.field
    g_l_aug, g_r_aug = bridge.g_l_aug, bridge.g_r_aug
    n_l = g_l.code.num_qudits
    n_r = g_r.code.num_qudits
    k_l = g_l_aug.F.shape[0]
    k_r = g_r_aug.F.shape[0]
    w = bridge.width
    n_merged = n_l + n_r + k_l + k_r + w
    mX_l = g_l.code.matrix_x.shape[0]
    mX_r = g_r.code.matrix_x.shape[0]
    mZ_l = g_l.code.matrix_z.shape[0]
    mZ_r = g_r.code.matrix_z.shape[0]
    r_l = g_l_aug.G.shape[0]
    r_r = g_r_aug.G.shape[0]

    cl_data = slice(0, n_l); cr_data = slice(n_l, n_l + n_r)
    cl_kappa = slice(n_l + n_r, n_l + n_r + k_l)
    cr_kappa = slice(n_l + n_r + k_l, n_l + n_r + k_l + k_r)
    c_adapter = slice(n_l + n_r + k_l + k_r, n_merged)

    HX_l = np.asarray(g_l_aug.HX_merged).astype(np.int_)
    HX_r = np.asarray(g_r_aug.HX_merged).astype(np.int_)
    HZ_l = np.asarray(g_l_aug.HZ_merged).astype(np.int_)
    HZ_r = np.asarray(g_r_aug.HZ_merged).astype(np.int_)

    # Roles swapped: chi rows live in HZ; G rows + new cycle live in HX.
    HZ = np.zeros((mZ_l + mZ_r + len(g_l.V0) + len(g_r.V0), n_merged), dtype=np.int_)
    HX = np.zeros((mX_l + mX_r + r_l + r_r + (w - 1), n_merged), dtype=np.int_)

    # H_Z rows: data H_Z + chi
    HZ[: mZ_l, cl_data] = HZ_l[: mZ_l, : n_l]
    HZ[mZ_l : mZ_l + mZ_r, cr_data] = HZ_r[: mZ_r, : n_r]
    chi_l = HZ_l[mZ_l :, :]
    chi_r = HZ_r[mZ_r :, :]
    HZ[mZ_l + mZ_r : mZ_l + mZ_r + len(g_l.V0), cl_data] = chi_l[:, : n_l]
    HZ[mZ_l + mZ_r : mZ_l + mZ_r + len(g_l.V0), cl_kappa] = chi_l[:, n_l :]
    HZ[mZ_l + mZ_r + len(g_l.V0) :, cr_data] = chi_r[:, : n_r]
    HZ[mZ_l + mZ_r + len(g_l.V0) :, cr_kappa] = chi_r[:, n_r :]
    for v_idx, lab in enumerate(bridge.label_l):
        if lab >= 0:
            HZ[mZ_l + mZ_r + v_idx, c_adapter.start + lab] = 1
    for v_idx, lab in enumerate(bridge.label_r):
        if lab >= 0:
            HZ[mZ_l + mZ_r + len(g_l.V0) + v_idx, c_adapter.start + lab] = 1

    # H_X rows: data H_X extended, G_aug, new cycle-X-checks
    HX[: mX_l, cl_data] = HX_l[: mX_l, : n_l]
    HX[: mX_l, cl_kappa] = HX_l[: mX_l, n_l :]
    HX[mX_l : mX_l + mX_r, cr_data] = HX_r[: mX_r, : n_r]
    HX[mX_l : mX_l + mX_r, cr_kappa] = HX_r[: mX_r, n_r :]
    HX[mX_l + mX_r : mX_l + mX_r + r_l, cl_kappa] = HX_l[mX_l :, n_l :]
    HX[mX_l + mX_r + r_l : mX_l + mX_r + r_l + r_r, cr_kappa] = HX_r[mX_r :, n_r :]
    new_x_start = mX_l + mX_r + r_l + r_r
    HX[new_x_start :, cl_kappa] = bridge.T_l
    HX[new_x_start :, cr_kappa] = bridge.T_r
    HX[new_x_start :, c_adapter] = bridge.H_R

    return CSSCode(field(HX), field(HZ), is_subsystem_code=False)


def _stitch_intracode_joint_csscode_basis_z(g_l, g_r, bridge):
    """Symmetric dual of intra-code basis=X."""
    assert g_l.code is g_r.code
    assert bridge.basis is Pauli.Z
    field = g_l.code.field
    g_l_aug, g_r_aug = bridge.g_l_aug, bridge.g_r_aug
    n = g_l.code.num_qudits
    k_l = g_l_aug.F.shape[0]; k_r = g_r_aug.F.shape[0]
    w = bridge.width
    n_merged = n + k_l + k_r + w
    mX = g_l.code.matrix_x.shape[0]
    mZ = g_l.code.matrix_z.shape[0]
    r_l = g_l_aug.G.shape[0]; r_r = g_r_aug.G.shape[0]

    c_data = slice(0, n)
    cl_kappa = slice(n, n + k_l)
    cr_kappa = slice(n + k_l, n + k_l + k_r)
    c_adapter = slice(n + k_l + k_r, n_merged)

    HX_l = np.asarray(g_l_aug.HX_merged).astype(np.int_)
    HX_r = np.asarray(g_r_aug.HX_merged).astype(np.int_)
    HZ_l = np.asarray(g_l_aug.HZ_merged).astype(np.int_)
    HZ_r = np.asarray(g_r_aug.HZ_merged).astype(np.int_)

    HZ = np.zeros((mZ + len(g_l.V0) + len(g_r.V0), n_merged), dtype=np.int_)
    HX = np.zeros((mX + r_l + r_r + (w - 1), n_merged), dtype=np.int_)

    HZ[: mZ, c_data] = HZ_l[: mZ, : n]
    chi_l = HZ_l[mZ :, :]
    chi_r = HZ_r[mZ :, :]
    HZ[mZ : mZ + len(g_l.V0), c_data] = chi_l[:, : n]
    HZ[mZ : mZ + len(g_l.V0), cl_kappa] = chi_l[:, n :]
    HZ[mZ + len(g_l.V0) :, c_data] = chi_r[:, : n]
    HZ[mZ + len(g_l.V0) :, cr_kappa] = chi_r[:, n :]
    for v_idx, lab in enumerate(bridge.label_l):
        if lab >= 0:
            HZ[mZ + v_idx, c_adapter.start + lab] = 1
    for v_idx, lab in enumerate(bridge.label_r):
        if lab >= 0:
            HZ[mZ + len(g_l.V0) + v_idx, c_adapter.start + lab] = 1

    HX[: mX, c_data] = HX_l[: mX, : n]
    HX[: mX, cl_kappa] = HX_l[: mX, n :]
    HX[: mX, cr_kappa] = HX_r[: mX, n :]
    HX[mX : mX + r_l, cl_kappa] = HX_l[mX :, n :]
    HX[mX + r_l : mX + r_l + r_r, cr_kappa] = HX_r[mX :, n :]
    new_x_start = mX + r_l + r_r
    HX[new_x_start :, cl_kappa] = bridge.T_l
    HX[new_x_start :, cr_kappa] = bridge.T_r
    HX[new_x_start :, c_adapter] = bridge.H_R

    return CSSCode(field(HX), field(HZ), is_subsystem_code=False)
```

Update the dispatch in `_stitch_to_joint_csscode` to call the intra-code basis=Z helper too:

```python
def _stitch_to_joint_csscode(g_l, g_r, bridge):
    intercode = g_l.code is not g_r.code
    if not intercode and bridge.basis is Pauli.X:
        return _stitch_intracode_joint_csscode(g_l, g_r, bridge)
    if not intercode and bridge.basis is Pauli.Z:
        return _stitch_intracode_joint_csscode_basis_z(g_l, g_r, bridge)
    if intercode and bridge.basis is Pauli.Z:
        return _stitch_intercode_joint_csscode_basis_z(g_l, g_r, bridge)
    # intercode + basis=X — body already in Task 8
    ...  # existing implementation
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest src/qldpc/codes/surgery/_test.py -k "intercode_both_bases or intracode_both_bases" -v`
Expected: PASS on all parametrizations.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/circuit.py src/qldpc/codes/surgery/_test.py
git commit -m "$(cat <<'EOF'
feat: basis=Z dual for both inter-code and intra-code stitching

X ↔ Z swap: χ rows go into HZ; G_aug + new cycle (T_l|T_r|H_R)
go into HX.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Update `build_joint_ppm_circuit` and related circuit helpers

**Files:**
- Modify: `src/qldpc/codes/surgery/circuit.py` (build_joint_ppm_circuit lines 178–251; _surgery_state_prep, _surgery_detach_and_readout, _surgery_final_detectors)
- Test: `src/qldpc/codes/surgery/_test.py`

The χ-check ID computation must follow the new H_X row layout: data H_X rows first, then χ rows. No `ub_ids`. `_surgery_state_prep` / `_surgery_detach_and_readout` already accept arbitrary `bridge_ids`; the change is in row-offset arithmetic.

- [ ] **Step 1: Write failing tests**

```python
def test_build_joint_ppm_circuit_chi_check_ids_no_UB():
    """build_joint_ppm_circuit's chi_check_ids equals χ^(l) ∪ χ^(r) only."""
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import build_joint_ppm_circuit
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(codes.SteaneCode(), x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    circuit, merged = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=2)
    # noiseless: all detectors must NOT fire on first sample
    sampler = circuit.compile_detector_sampler()
    dets, _ = sampler.sample(8, separate_observables=True)
    assert dets.sum() == 0


def test_build_joint_ppm_circuit_intercode_noiseless_observables_zero():
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import build_joint_ppm_circuit
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(codes.SteaneCode(), x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    circuit, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=2)
    sampler = circuit.compile_detector_sampler()
    _, obs = sampler.sample(8, separate_observables=True)
    # basis=X init: data in |+⟩, κ in |0⟩, adapter in |0⟩
    # Joint observable 0 (chi XOR over rounds) deterministic = 0
    # Observable 1 (final M-X on V_0 ∪) deterministic = 0
    assert obs.sum() == 0
```

- [ ] **Step 2: Run to verify fails**

Run: `pytest src/qldpc/codes/surgery/_test.py -k "joint_ppm_circuit_chi or joint_ppm_circuit_intercode_noiseless" -v`
Expected: FAIL — the current `build_joint_ppm_circuit` still references old `bridge.U_B` and old field names.

- [ ] **Step 3: Update `build_joint_ppm_circuit`**

In `src/qldpc/codes/surgery/circuit.py`:

```python
def build_joint_ppm_circuit(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
    *,
    rounds: int,
    noise_model=None,
) -> tuple[stim.Circuit, CSSCode]:
    """Joint-PPM circuit per spec §4 (universal adapter, no U_B in α*)."""
    joint_code = _stitch_to_joint_csscode(g_l, g_r, bridge)
    qubit_ids = QubitIDs.from_code(joint_code)
    intercode = g_l.code is not g_r.code

    g_l_aug, g_r_aug = bridge.g_l_aug, bridge.g_r_aug
    n_l = g_l.code.num_qudits
    n_r = g_r.code.num_qudits if intercode else 0
    k_l = g_l_aug.F.shape[0]
    k_r = g_r_aug.F.shape[0]
    w = bridge.width

    if intercode:
        data_ids = qubit_ids.data[: n_l + n_r]
        v0_combined = tuple(g_l.V0) + tuple(n_l + i for i in g_r.V0)
    else:
        data_ids = qubit_ids.data[: n_l]
        v0_combined = tuple(g_l.V0) + tuple(g_r.V0)
    kappa_ids = qubit_ids.data[n_l + n_r : n_l + n_r + k_l + k_r]
    bridge_ids = qubit_ids.data[n_l + n_r + k_l + k_r :]
    assert len(bridge_ids) == w

    circuit = get_qubit_coordinates(qubit_ids.data, qubit_ids.check)
    circuit += _surgery_state_prep(g_l, data_ids, kappa_ids, bridge_ids)
    qec_cycle, measurement_record, _ = _surgery_qec_cycle(
        g_l, joint_code, num_rounds=rounds, qubit_ids=qubit_ids,
    )
    circuit += qec_cycle
    circuit += _surgery_detach_and_readout(
        g_l, data_ids=data_ids, kappa_ids=kappa_ids, bridge_ids=bridge_ids,
        measurement_record=measurement_record,
    )
    circuit += _surgery_final_detectors(
        g_l, joint_code, qubit_ids,
        measurement_record=measurement_record,
    )

    # χ check IDs per spec §4: data H_X^(l) rows occupy first mX_l indices in
    # qubit_ids.checks_x, then m_X_r (inter-code), then χ^(l), then χ^(r).
    if bridge.basis is Pauli.X:
        check_ids = qubit_ids.checks_x
        m_l = g_l.code.matrix_x.shape[0]
        m_r = g_r.code.matrix_x.shape[0] if intercode else 0
    else:
        check_ids = qubit_ids.checks_z
        m_l = g_l.code.matrix_z.shape[0]
        m_r = g_r.code.matrix_z.shape[0] if intercode else 0
    n_V_l = len(g_l.V0)
    n_V_r = len(g_r.V0)
    chi_l_offset = m_l + m_r
    chi_r_offset = chi_l_offset + n_V_l
    chi_l_ids = tuple(check_ids[chi_l_offset : chi_l_offset + n_V_l])
    chi_r_ids = tuple(check_ids[chi_r_offset : chi_r_offset + n_V_r])
    chi_check_ids = chi_l_ids + chi_r_ids   # NO U_B / no adapter cycle-check ids

    circuit += _surgery_observable(
        g_l,
        chi_check_ids=chi_check_ids,
        data_ids=data_ids,
        v0_indices=v0_combined,
        num_rounds=rounds,
        measurement_record=measurement_record,
    )

    if noise_model is not None:
        circuit = noise_model.noisy_circuit(circuit)
    return circuit, joint_code
```

The intra-code branch shares the same code path (with `n_r = 0`, `m_r = 0`). No special case needed beyond setting `intercode`.

Verify `_surgery_state_prep` already handles `bridge_ids` correctly (`circuit.py:270`). It does — it batches all κ + bridge qubits under the same R/RX. No changes needed unless `extra_κ` qubits need a different init basis (they don't: they're new κ qubits, same as original κ).

`_surgery_detach_and_readout` already iterates over `bridge_ids` — no change.

`_surgery_final_detectors` reads `merged_code.matrix_x` and `merged_code.matrix_z` from the joint code; row count matches the new stitch shape automatically.

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest src/qldpc/codes/surgery/_test.py -k "joint_ppm_circuit_chi or joint_ppm_circuit_intercode_noiseless" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/codes/surgery/circuit.py src/qldpc/codes/surgery/_test.py
git commit -m "$(cat <<'EOF'
feat: build_joint_ppm_circuit consumes universal-adapter Bridge

Drops U_B-based α* (chi_check_ids = χ^(l) ∪ χ^(r) only). Layout
arithmetic follows new HX^merged ordering. Inter/intra-code share
the same path with n_r=m_r=0 for intra-code.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Adapter cycle-check weight bound + structural smoke tests

**Files:**
- Test: `src/qldpc/codes/surgery/_test.py`

These structural invariants (from spec §5) should already pass given Tasks 1–11. We add them now as guards against regression.

- [ ] **Step 1: Add tests**

```python
def test_adapter_cycle_check_weight_bounded():
    """Each new cycle-Z row has weight <= 8 (SkipTree (3,2) + H_R weight 2)."""
    from qldpc.codes.surgery.gadget import (
        build_gadget, load_webster_seed_set, _build_generalised_bicycle_code,
    )
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import _stitch_to_joint_csscode
    data = load_webster_seed_set(0)
    code = _build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x = np.asarray(data["seed_ops"]["Z_bar_1"]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.Z)
    g_r = build_gadget(code, x, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    new_x_rows = HX[-(bridge.width - 1):, :]   # basis=Z: new cycle goes into HX
    max_w = int(new_x_rows.sum(axis=1).max())
    assert max_w <= 8, f"max new cycle-X weight {max_w}"


def test_cellulation_caps_aug_aux_cycle_length_on_webster():
    """After cellulation, every basis cycle in the augmented aux graph has length <= 6."""
    import networkx as nx
    from qldpc.codes.surgery.gadget import (
        build_gadget, load_webster_seed_set, _build_generalised_bicycle_code,
    )
    from qldpc.codes.surgery.bridge import build_bridge, _build_aux_graph_strict
    data = load_webster_seed_set(0)
    code = _build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x = np.asarray(data["seed_ops"]["Z_bar_1"]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.Z)
    g_r = build_gadget(code, x, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r, cellulate_max_len=6)
    # Build aux graph from g_l_aug.F and check cycle basis
    G_aux, _ = _build_aux_graph_strict(bridge.g_l_aug.F)
    cycles = nx.cycle_basis(G_aux)
    assert all(len(c) <= 6 for c in cycles), f"max len {max(len(c) for c in cycles)}"
```

- [ ] **Step 2: Run**

Run: `pytest src/qldpc/codes/surgery/_test.py -k "adapter_cycle_check_weight or cellulation_caps_aug_aux" -v`
Expected: PASS (these are smoke tests on existing functionality).

- [ ] **Step 3: Commit**

```bash
git add src/qldpc/codes/surgery/_test.py
git commit -m "$(cat <<'EOF'
test: adapter check weight ≤ 8 + cellulation invariants on Webster BB

Pins spec §5 structural invariants for the universal-adapter
construction.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: LER smoke test (Steane × Steane)

**Files:**
- Test: `src/qldpc/codes/surgery/_test.py`

- [ ] **Step 1: Add test**

```python
@pytest.mark.slow
def test_joint_ppm_ler_monotone_steane_intercode():
    """LER non-increasing in p across {1e-4, 3e-4, 1e-3} for Steane × Steane."""
    from qldpc.circuits.noise_model import DepolarizingNoiseModel
    from qldpc.codes.surgery.gadget import build_gadget
    from qldpc.codes.surgery.bridge import build_bridge
    from qldpc.codes.surgery.circuit import build_joint_ppm_circuit
    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(codes.SteaneCode(), x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    lers = []
    shots = 2000
    for p in (1e-3, 3e-4, 1e-4):
        nm = DepolarizingNoiseModel(p)
        circuit, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=3, noise_model=nm)
        sampler = circuit.compile_detector_sampler()
        _, obs = sampler.sample(shots, separate_observables=True)
        # logical error rate of OBS 0 (joint χ XOR)
        ler = (obs[:, 0] != 0).mean()
        lers.append(ler)
    assert lers[0] >= lers[1] / 1.3, f"LER not monotone: {lers}"
    assert lers[1] >= lers[2] / 1.3, f"LER not monotone: {lers}"
```

(If `DepolarizingNoiseModel` is named differently in this repo — `grep -rn "DepolarizingNoiseModel\|class .*Noise" src/qldpc/circuits/noise_model.py` to confirm. Use the actual name.)

- [ ] **Step 2: Run**

Run: `pytest src/qldpc/codes/surgery/_test.py::test_joint_ppm_ler_monotone_steane_intercode -v --runslow`
(or whatever the slow-test marker invocation is in this project — check `conftest.py` / `pyproject.toml`).
Expected: PASS (LER decreases as p decreases).

If the project doesn't have a `--runslow` flag yet, just run without:
Run: `pytest src/qldpc/codes/surgery/_test.py::test_joint_ppm_ler_monotone_steane_intercode -v`

- [ ] **Step 3: Commit**

```bash
git add src/qldpc/codes/surgery/_test.py
git commit -m "$(cat <<'EOF'
test: joint PPM LER monotone in p on Steane × Steane (slow)

Smoke test guarding against silent regressions in joint observable
correctness. Allows 1.3× tolerance to absorb sampling noise.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Migrate demo scripts and examples

**Files:**
- Modify: `examples/scripts/joint_ppm_correctness_z_demo.py`
- Modify: `examples/scripts/joint_ppm_z_with_superposition_demo.py`
- Modify: `examples/scripts/cain_fig1b_full_protocol.py`
- Modify: `examples/scripts/cain_bb18_resource_exact_match.py`
- Modify: `examples/test_ide_bb_lp.py`
- Modify: `examples/logical_error_rates/_9_lattice_surgery_source.py`
- Regenerate: `examples/logical_error_rates/9_lattice_surgery.ipynb` (if there's a script-to-notebook tool; otherwise hand-edit the JSON)

These all consume the old `Bridge` API (`bridge.U_B`, `bridge.qubits`, etc.). The fix per script: replace `.U_B` references with `.T_l, .T_r, .H_R` printouts; replace any `chi_endpoint_extensions` walking with the new `bridge.label_l`, `bridge.label_r`, `bridge.port_l`, `bridge.port_r`; drop "path-graph" narrative; replace with "universal adapter (arXiv:2410.03628)".

- [ ] **Step 1: Survey call sites**

Run:
```bash
grep -rn "\.U_B\|\.qubits\|chi_endpoint_extensions\|aux_graph_edges\|z_extensions\|_build_path_graph\|_solve_chi_z" examples/ | tee /tmp/migrate.txt
```

Read each location and decide whether the script just prints diagnostic info (replace with new fields) or actually consumes the structure (rebuild around new fields).

- [ ] **Step 2: Migrate `examples/scripts/cain_fig1b_full_protocol.py`**

Update the docstring comment block (around the file head):

```python
"""Cain Fig. 1b joint X̄X̄ protocol on BB_18 (post-Swaroop universal adapter).

Picks first chi-XOR sum across rounds 1..rounds as the joint observable.
For the universal adapter (arXiv:2410.03628), α* = χ^(l) ∪ χ^(r) only — there
is no separate U_B telescoping path. The old path-graph version that used to
live in this file has been removed.
"""
```

Update the `build_bridge` call site if any keyword args changed (none did). If the script reads `bridge.U_B.shape`, replace with:

```python
print(f"adapter width     : {bridge.width}")
print(f"T_l shape         : {bridge.T_l.shape}")
print(f"T_r shape         : {bridge.T_r.shape}")
print(f"extra_κ_l rows    : {bridge.extra_kappa_l.shape[0]}")
print(f"extra_κ_r rows    : {bridge.extra_kappa_r.shape[0]}")
```

Run after editing:

```bash
python examples/scripts/cain_fig1b_full_protocol.py
```
Expected: runs to completion, prints joint observable summary.

- [ ] **Step 3: Migrate the other scripts (same pattern)**

Apply the same kind of edit to:
- `joint_ppm_correctness_z_demo.py`
- `joint_ppm_z_with_superposition_demo.py`
- `cain_bb18_resource_exact_match.py`
- `examples/test_ide_bb_lp.py`

For each, after editing:
```bash
python <script>
```
should run without errors. If a script reports unexpected LER, eyeball the comparison and confirm it's within the predicted ≤ 2× regime vs single-PPM baseline.

- [ ] **Step 4: Migrate `_9_lattice_surgery_source.py` + regenerate the notebook**

Edit `examples/logical_error_rates/_9_lattice_surgery_source.py` — drop the section that prints `bridge.U_B`, `col_sum = bridge.U_B.sum(axis=0) % 2`, and the "path telescoping" narrative. Replace with a "universal adapter" subsection that shows:

```python
print(f"Bridge width                  : {bridge.width}")
print(f"Adapter qubits                : {bridge.width}")
print(f"extra_κ_l (connectivity+cell) : {bridge.extra_kappa_l.shape[0]} new κ qubits")
print(f"extra_κ_r                     : {bridge.extra_kappa_r.shape[0]} new κ qubits")
print(f"T_l shape                     : {bridge.T_l.shape} (SkipTree, (3,2)-sparse)")
print(f"H_R shape                     : {bridge.H_R.shape} (canonical rep code parity)")
```

Replace the "U_B telescoping" check by the SkipTree key identity check from Task 7:

```python
# Verify T_l · G_l_aug · P_l == H_R on the augmented aux graph
import numpy as np
P_l = np.zeros((bridge.g_l_aug.F.shape[1], bridge.width), dtype=np.int_)
for v_idx, lab in enumerate(bridge.label_l):
    if lab >= 0:
        P_l[v_idx, lab] = 1
lhs = (bridge.T_l @ bridge.g_l_aug.F.astype(np.int_) @ P_l) % 2
print("SkipTree identity (left): ", "PASS" if np.array_equal(lhs, bridge.H_R) else "FAIL")
```

Regenerate the .ipynb:

```bash
# If a script-to-nb conversion is part of project tooling, invoke it:
jupytext --to ipynb examples/logical_error_rates/_9_lattice_surgery_source.py \
    --output examples/logical_error_rates/9_lattice_surgery.ipynb
```

(If `jupytext` isn't installed, fall back to hand-editing the JSON cells of the notebook. Search for `bridge.U_B` cells and edit them.)

- [ ] **Step 5: Commit**

```bash
git add examples/
git commit -m "$(cat <<'EOF'
chore: migrate demos to universal-adapter Bridge API

Drops bridge.U_B / chi_endpoint_extensions references; switches
diagnostics to bridge.T_l/T_r/H_R/label_l/label_r/extra_κ; updates
narrative in 9_lattice_surgery to cite arXiv:2410.03628.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Update `docs/superpowers/math.md`

**Files:**
- Modify: `docs/superpowers/math.md`

Replace §2.2–2.7 (path-graph derivation) with the universal-adapter derivation. Keep §1 (gadget steps) and §3 (Cheeger boost) untouched. Add §2.8 as a brief historical note.

- [ ] **Step 1: Write the new §2 content**

Edit `docs/superpowers/math.md`. Replace lines `2. Joint measurement — build_joint_measurement_code` through the end of the joint-measurement subsection with:

```markdown
2. Joint measurement — universal adapter (arXiv:2410.03628 §IV / §VII)

2.1 Adapter width and port subsets

Given two GadgetLayouts (g_l, g_r) measuring X̄_l, X̄_r (or Z̄_l, Z̄_r) we pick port subsets
𝒫_l* ⊆ V_0^(l), 𝒫_r* ⊆ V_0^(r) of equal size w = min(|V_0^(l)|, |V_0^(r)|), and a bijection
𝒜: 𝒫_l* → 𝒫_r*. Each adapter edge ∈ 𝒜 is one new "adapter" data qubit.

2.2 Auxiliary graph augmentation

For each side build 𝒢_s = (V_0^(s), {edges = weight-2 F_s rows}). When 𝒢_s[𝒫_s*] is
disconnected, add weight-2 edges (= new κ qubits) until connected. Cellulate basis cycles
to length ≤ max_len.

2.3 SkipTree key identity

Run SkipTree (paper §III Algorithm 1) on 𝒢_s_aug to obtain T_s ∈ F_2^{(w-1) × |E_aug|},
P_s ∈ F_2^{|V_0^(s)| × w} satisfying

  T_s · G_s_aug · P_s = H_R    (canonical full-rank rep-code parity, (w-1) × w)

where G_s_aug = F_aug^(s) is the auxiliary-graph incidence matrix.

2.4 Merged check matrices (basis=X, inter-code)

H_X^merged blocks (rows × support):
   data H_X^(l)     : m_X^(l) × data_l
   data H_X^(r)     : m_X^(r) × data_r
   χ^(l)            : |V_0^(l)| × (data_l + κ_l_aug + adapter)  via E_V0^T, F_aug^(l)^T, Π_l
   χ^(r)            : |V_0^(r)| × (data_r + κ_r_aug + adapter)  via E_V0^T, F_aug^(r)^T, Π_r

H_Z^merged blocks:
   data H_Z^(l) ext : m_Z^(l) × (data_l + κ_l_aug)
   data H_Z^(r) ext : m_Z^(r) × (data_r + κ_r_aug)
   G^(l)_aug        : r_l × κ_l_aug
   G^(r)_aug        : r_r × κ_r_aug
   new cycle-Z      : (w-1) × (κ_l_aug + κ_r_aug + adapter)  via [T_l | T_r | H_R]

Π_s ∈ F_2^{|V_0^(s)| × w} satisfies Π_s[v, k] = 1 iff v ∈ 𝒫_s* and label_s(v) = k.

basis=Z is the symmetric X↔Z dual; intra-code merges the two data column blocks.

2.5 Commutation (CSS)

The only non-trivial pairing is χ^(s) vs new cycle-Z^(s'). For s = s':
  (χ_v on κ + adapter) · (cycle_c on κ + adapter)
    = (T_s · F_aug^(s))[c, v] + H_R[c, label_s(v)] · [v ∈ 𝒫_s*]
    = (T_s · G_s_aug · P_s)[c, v]                       (by SkipTree identity)
    = H_R[c, k] − H_R[c, k]                              (when label_s(v) = k)
    = 0.
For v ∉ 𝒫_s*, both halves are zero (T_s has zero columns on edges outside 𝒢_s[𝒫_s*]).

2.6 Joint observable (α* derivation)

α* picks Σ χ^(l) + Σ χ^(r). On the merged register:
   data side: 1_{V_0^(l)} + 1_{V_0^(r)} = x_l + x_r (XOR support, joint X̄_l X̄_r).
   κ side:    F_aug^(s)^T · 1_{V_0^(s)} = 0          (κ-cancellation).
   adapter:   Σ_{v ∈ 𝒫_l*} e_{label_l(v)} + Σ_{v ∈ 𝒫_r*} e_{label_r(v)} = 1_𝒜 + 1_𝒜 = 0.

New cycle-Z rows are not in α* (they're Z-type; orthogonal to the X-type joint observable).
```

(Path-graph §2 is replaced. Don't keep the old subsections — `git log` preserves history if anyone needs them.)

- [ ] **Step 2: Verify markdown renders cleanly**

Run:
```bash
python -c "import pathlib; print(pathlib.Path('docs/superpowers/math.md').read_text()[:500])"
```
Expected: prints the new §0/§1 prefix without errors.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/math.md
git commit -m "$(cat <<'EOF'
docs: math.md §2 rewritten for universal adapter

Replaces the path-graph derivation with the SkipTree-driven
construction (block tables, key identity, α* derivation).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: Final verification — full test suite + example walkthrough

**Files:** (none)

- [ ] **Step 1: Run the full surgery test suite**

```bash
pytest src/qldpc/codes/surgery/_test.py -v
```
Expected: all tests pass; no XFAIL skips beyond pre-existing skips.

- [ ] **Step 2: Run all migrated demos**

```bash
for f in examples/scripts/joint_ppm_correctness_z_demo.py \
         examples/scripts/joint_ppm_z_with_superposition_demo.py \
         examples/scripts/cain_fig1b_full_protocol.py \
         examples/scripts/cain_bb18_resource_exact_match.py \
         examples/test_ide_bb_lp.py; do
    echo "=== $f ==="
    python "$f" || echo "FAILED: $f"
done
```
Expected: each prints its summary and exits 0.

- [ ] **Step 3: Run full repo tests as a regression check**

```bash
pytest src/qldpc/ -x -q
```
Expected: all pass.

- [ ] **Step 4: Tag the milestone**

```bash
git tag -a universal-adapter-bridge -m "joint PPM bridge replaced by Swaroop universal adapter"
git log --oneline -20
```

(No `git push` — leave that to the human.)

---

## Self-Review

Run through this checklist after the plan is complete.

- **Spec §1 (Architecture)**: covered by Task 6 (Bridge dataclass) + Task 7 (build_bridge wiring).
- **Spec §2 (build_bridge 7 steps)**: covered by Tasks 1–7. Step 1 → Task 2. Step 2 → Task 7. Step 3 → Task 3. Step 4 → Task 4. Step 5 → Task 1 (SkipTree) + Task 7 (wiring). Step 6 → Task 7. Step 7 → Task 5 + Task 7.
- **Spec §3 (block tables)**: covered by Tasks 8 (inter-code basis=X), 9 (intra-code basis=X), 10 (basis=Z duals).
- **Spec §4 (joint observable α*)**: covered by Task 11.
- **Spec §5 (testing)**: covered by structural tests sprinkled through Tasks 1–10 + Task 12 (weight bound) + Task 13 (LER smoke).
- **Spec §6 (migration)**: covered by Task 14 (demos) + Task 15 (math.md).
- **Placeholder scan**: no "TBD", "TODO", "similar to Task N" entries. All code blocks are concrete.
- **Type/name consistency**: `Bridge` field set defined in Task 6 and used identically across Tasks 7–11. `_skip_tree_fullrank`, `_canonical_H_R`, `_build_aux_graph_strict`, `_connect_induced_subgraph`, `_cellulate_strict`, `build_gadget_augmented`, `_stitch_to_joint_csscode`, `_stitch_intracode_joint_csscode`, `_stitch_intercode_joint_csscode_basis_z`, `_stitch_intracode_joint_csscode_basis_z` — all referenced consistently. `bridge.label_l[v]` uses `-1` sentinel for non-port vertices; usage at Task 8/9/10 row-fill loops respects the sentinel via `if lab >= 0`.
- **Risks called out in spec § "Out of scope"**: hyperedge decomposition (raise NotImplementedError per Task 2), Hamiltonian-path SkipTree (default-0 root), decongestion, relative-expansion certificates, BP-OSD — none included; LER monotonicity test (Task 13) is the empirical guard.
