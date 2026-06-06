# Design: v2 — Bridges (SkipTree), Cheeger Boost, and Webster Table I Verification

**Date**: 2026-06-06
**Status**: Approved (brainstorming phase)
**Branch**: `feat/surgery-construction` (continues from v1)
**Builds on**: `docs/superpowers/specs/2026-06-05-cross-layered-ancilla-design.md` (v1, commit `8e58f77`).
**Primary references**:
- Cross, He, Rall, Yoder. *Improved QLDPC Surgery: Logical Measurements and Bridging Codes*. arXiv:2407.18393 §3.6 (bridges for joint X-type measurements), §3.7 (Y measurements — OOS).
- Swaroop, Jochym-O'Connor, Yoder. *Universal adapters between quantum LDPC codes*. arXiv:2410.03628 §III (SkipTree basis transformation), Lemma 14 (cellulation / decongestion).
- **Swaroop reference implementation**: https://github.com/eswaroop/adapters-LDPC-surgery (MIT, 2025). Contains the SkipTree (`skip_tree_algorithm.py`, ~50 lines) and cellulation (`cellulation.py`, ~40 lines) primitives ported here verbatim with attribution. The high-level joint-measurement adapter assembly is "coming soon" in that repo and is implemented from scratch in this spec.
- Webster, Smith, Cohen. *Explicit construction of low-overhead gadgets for gates on quantum LDPC codes*. arXiv:2511.15989 §II.A (gadget construction — v1), Table I (overhead numbers — v2 verification target), Appendix A (4 generalised bicycle code instances with seed operators).

## 1. Motivation

v1 implements the bare gadget construction (Webster §II.A Steps 1–3 = Cross L=1) and verifies the Webster Eq. (1) algebraic identity. It is mathematically complete for single logical-X measurements but cannot reproduce the concrete numerical overhead numbers in Webster Table I, which require:

1. **Bare gadget construction on the 4 generalised bicycle codes from Webster Appendix A** (gadget qubit counts: 19, 31, 49, 79).
2. **Cheeger augmentation for codes 3 and 4** (+8 and +20 qubits respectively, lifting boundary Cheeger constant h to ≥ 1).
3. **Bridges for joint X̄X̄' and Z̄Z̄' measurements** (bridge qubits: 11, 19, 31, 51).

The bare gadget numbers (item 1) are the strongest non-trivial empirical test of v1's correctness. Items 2 and 3 are engineering features that complete the practical surgery toolkit. Together they make `qldpc.codes.surgery` the first public reference implementation of Webster's full overhead reduction toolkit.

## 2. Scope

**In scope**:
- A function `build_joint_measurement_code(data_code, op1, op2, *, num_layers=1, validate=True)` for joint X̄X̄' and Z̄Z̄' measurements on a single CSS code block.
- A function `boost_gadget_cheeger(merged, layout, *, target_h=1.0, max_extra_qubits=None, seed=None)` implementing the random-search Cheeger augmentation heuristic.
- Internal helpers for SkipTree (port of Swaroop `skip_tree_algorithm.py`) and cellulation (port of `cellulation.py`) with MIT attribution.
- A JSON data file `examples/webster_app_a.json` containing the (l, A, B) polynomial supports and the 4 seed operators per code from Webster Appendix A.
- A verification script `examples/webster_table1_verify.py` that builds the 4 codes × 4 seeds and prints a comparison table against Webster Table I.
- New `JointSurgeryLayout` and `BoostResult` dataclasses.
- Unit tests for each new helper plus the Table I bare-gadget acceptance test.

**Out of scope (future specs)**:
- Y measurement bridges (Cross §3.7) and the R(xy^5) qubit splitting trick (Cross §IV.1).
- Cross-block joint measurements between two different code instances (Swaroop universal adapter framework's full generality).
- Constructive cellulation for arbitrary graphs beyond what is needed for same-block X̄X̄' bridges.
- Automatic seed-operator search via automorphism enumeration (Webster §II.B).
- Constructive Cheeger boost algorithms beyond random search (e.g. integer programming, deterministic expander constructions).
- The Cain Fig 1b notebook circuit-build wiring (`memory.build_x_memory_circuit` call, sinter sweep, plot) — remains TODO for the notebook author.

## 3. Public API

**Module**: existing `src/qldpc/codes/surgery.py`. v2 appends three new public symbols and one optional data loader.

```python
@dataclasses.dataclass(frozen=True, eq=False)
class JointSurgeryLayout:
    """Provenance of qubits and checks in a merged joint-measurement code.

    Returned by ``build_joint_measurement_code``. Captures the two individual
    gadget layouts (with their own SurgeryLayout fields) plus bridge-specific
    metadata: bridge qubit indices and U_B gauge-check row indices in the
    merged code.

    Attributes:
        gadget_layouts: Pair of SurgeryLayout instances, one per input
            logical operator.
        pauli_type: Pauli.X if the joint measurement is X̄_1 X̄_2; Pauli.Z if
            Z̄_1 Z̄_2. Auto-detected by ``build_joint_measurement_code``.
        num_data_qubits: Number of qubits in the original data code.
        num_ancilla_qubits: gadget1.num_ancilla + gadget2.num_ancilla.
        num_bridge_qubits: Bridge qubits introduced by SkipTree.
        bridge_qubit_slice: Column slice of bridge qubits in the merged
            qubit register (after data + both gadget ancillas).
        u_b_check_kind_mask: Boolean mask over merged H_Z rows marking the
            U_B gauge-fix bridge checks.
    """
    gadget_layouts: tuple[SurgeryLayout, SurgeryLayout]
    pauli_type: Pauli  # qldpc.objects.Pauli — X or Z
    num_data_qubits: int
    num_ancilla_qubits: int
    num_bridge_qubits: int
    bridge_qubit_slice: slice
    u_b_check_kind_mask: npt.NDArray[np.bool_]


def build_joint_measurement_code(
    data_code: CSSCode,
    op1: npt.ArrayLike,
    op2: npt.ArrayLike,
    *,
    num_layers: int = 1,
    validate: bool = True,
) -> tuple[CSSCode, JointSurgeryLayout]:
    """Construct a merged stabilizer code that measures op1 · op2 by lattice surgery.

    Both op1 and op2 must be logical Pauli operators of the same type (both
    X-type or both Z-type) on the same ``data_code``. The merged code
    encodes ``data_code.dimension - 2`` logical qubits (two logicals are
    consumed by the joint measurement).

    Implementation: build two single-operator gadgets via
    ``build_layered_surgery_code``, connect their interfaces with a bridge
    constructed via SkipTree (Swaroop arXiv:2410.03628 §III), with bridge
    cycles bounded in length via cellulation (Lemma 14).

    Args:
        data_code: The data CSSCode (stabilizer, not subsystem).
        op1, op2: Binary row vectors of length ``data_code.num_qubits``
            indicating the support of the two logical operators. Both must
            be the same Pauli type.
        num_layers: Layer count for each individual gadget (passed to
            ``build_layered_surgery_code``).
        validate: If True, run all v1 validation plus the joint-specific
            type-consistency check.

    Returns:
        (merged_code, joint_layout).

    Raises:
        ValueError: if op1 and op2 are different Pauli types, either op
            fails v1 validation, or the SkipTree procedure cannot connect
            the two gadget interfaces (degenerate input).
    """


@dataclasses.dataclass(frozen=True, eq=False)
class BoostResult:
    """Statistics about a Cheeger boost run."""
    extra_qubits_added: int
    final_h_lower_bound: float       # spectral lower bound on Cheeger at termination
    iterations: int
    terminated_by: str               # "target_reached", "max_qubits_exhausted", "no_progress"


def boost_gadget_cheeger(
    merged: CSSCode,
    layout: SurgeryLayout,
    *,
    target_h: float = 1.0,
    max_extra_qubits: int | None = None,
    seed: int | None = None,
) -> tuple[CSSCode, SurgeryLayout, BoostResult]:
    """Heuristically lift the gadget's boundary Cheeger constant via random
    edge augmentation (Webster arXiv:2511.15989 §II.A end).

    Iteratively adds new degree-2 κ' qubits — each new κ' connects a
    randomly chosen pair of X-checks (χ_i, χ_j) not already directly
    connected via another κ — and checks whether the spectral lower bound
    ``λ_2(F^T F) / 2`` on the boundary Cheeger constant has reached
    ``target_h``. Terminates on success, on hitting ``max_extra_qubits``,
    or if no new edge candidates remain.

    Args:
        merged: The merged CSSCode returned by ``build_layered_surgery_code``.
        layout: The associated SurgeryLayout. The function reads
            ``layout.F`` to compute the augmentation candidates.
        target_h: Target boundary Cheeger lower bound. Default 1.0 matches
            Webster's distance-preservation threshold.
        max_extra_qubits: Cap on the number of κ' qubits added. None means
            no cap; iteration runs until ``target_h`` or no-progress.
        seed: Random seed for reproducibility.

    Returns:
        (boosted_merged, boosted_layout, result) where ``boosted_layout`` has
        an extended F matrix and a larger num_ancilla_qubits, and ``result``
        records what happened.

    Raises:
        ValueError: if the input layout has empty F or non-binary entries.
    """


def load_webster_seed_set(code_index: int) -> dict:
    """Load Webster Appendix A code data for verification.

    Args:
        code_index: 0, 1, 2, or 3, selecting one of the four codes l ∈
            {31, 63, 127, 255} respectively.

    Returns:
        A dict with keys ``l``, ``A`` (list of monomial exponents),
        ``B`` (same), and ``seeds``: a list of 4 dicts with keys ``name``
        ("X_bar_1", "Z_bar_1", "X_bar_k2p1", "Z_bar_k2p1"), ``L_support``
        (list of qubit indices on the L block), and ``R_support`` (same on
        the R block).
    """
```

Re-exported from `qldpc.codes.__init__.py`: `JointSurgeryLayout`, `build_joint_measurement_code`, `BoostResult`, `boost_gadget_cheeger`.

## 4. Algorithm structure

All linear algebra runs over GF(2) via `galois.GF(2)`. Graph theory uses `networkx` (which is already a qldpc dependency).

```
build_joint_measurement_code (top-level)
├── _validate_joint_logical_ops(op1, op2, data_code)
│       - Detect Pauli type (X or Z) from non-commutation with the opposite parity check matrix.
│       - Reject mixed-type, mismatched-shape, etc.
│       - Each op individually validated via the v1 _restrict_to_logical_support contract.
├── gadget1 = build_layered_surgery_code(data_code, op1, num_layers, ...)
├── gadget2 = build_layered_surgery_code(data_code, op2, num_layers, ...)
├── bridge = _build_bridge_via_skiptree(gadget1, gadget2)
│       Step 1: Build interface graph S = (V, E)
│           - V = κ_j_1 vertices (layer-1 ancillas of gadget1) ∪ κ_j_2 vertices (gadget2)
│           - E = pairs (κ_j_1, κ_k_2) where the data Z-checks S_j (in gadget1's C_0) and
│             S_k (in gadget2's C_0) share at least one qubit in V_0_1 ∩ V_0_2.
│           - Note: V_0_1 = supp(op1), V_0_2 = supp(op2). Overlap V_0_1 ∩ V_0_2 is the
│             physical region where the two gadgets share data qubits.
│       Step 2: S' = _cellulate_long_cycles(S, max_len=6)
│           - Adds chord edges to keep every cycle in the cycle basis ≤ 6 edges,
│             which is the threshold used by Webster/Swaroop for bicycle codes.
│       Step 3: T, P = _skip_tree(S', root=0)
│           - T: (|V|−1) × |E(S')| edge-incidence matrix of shortest-paths.
│             Each row of T is the support (as edges) of one bridge stabilizer.
│           - P: |V| × |V| vertex permutation matrix.
│       Step 4: Translate T into bridge qubits + U_B stabilizers
│           - num_bridge_qubits = |V(S')| − 1
│           - U_B rows: for each row of T, the corresponding bridge stabilizer is a
│             Z-check on (interface qubits selected by the edges in that row of T)
│             plus the two endpoint bridge qubits.
│       Returns: BridgeSpec(num_bridge_qubits, u_b_rows, vertex_to_bridge_qubit_map)
├── _stitch_gadgets_with_bridge(gadget1, gadget2, bridge)
│       - Concatenates the two merged codes' qubit registers + bridge qubits.
│       - Builds H_X^joint and H_Z^joint with appropriate zero-padding.
│       - Adds U_B rows to H_Z^joint.
│       - Constructs CSSCode and JointSurgeryLayout.
└── return (joint_merged_code, joint_layout)


_skip_tree(S: nx.Graph, root: int = 0) → (T, P)
       - Verbatim port of Swaroop skip_tree_algorithm.py (MIT, 2025).
       - Recursive label_first / label_last functions assign labels 0..n-1 to vertices.
       - T[l, e] = 1 iff edge e is on the shortest path from vertex labeled l to vertex
         labeled (l+1) mod n.
       - P is the permutation matrix mapping label → vertex.


_cellulate_long_cycles(G, edge_qubit_to_vertices, vert_to_edge, G_mat, max_len=6)
       - Verbatim port of Swaroop cellulation.py (MIT, 2025).
       - Iterates the cycle basis; for each cycle longer than max_len, adds a chord
         edge between cycle[0] and cycle[n//2].
       - Updates G, the dictionaries, and G_mat (the incidence matrix) in place.
       - Returns (new_edges_added, updated_dicts, updated_G_mat).


boost_gadget_cheeger (top-level)
├── F = layout.F.copy()
├── while True:
│       lambda2 = _spectral_cheeger_lower_bound(F)
│       h_estimate = lambda2 / 2
│       if h_estimate >= target_h: break ("target_reached")
│       if max_extra_qubits and extra >= max_extra_qubits: break ("max_qubits_exhausted")
│       candidate = _random_unused_chi_pair(F, rng)
│       if candidate is None: break ("no_progress")
│       new_row = e_i + e_j  # one-hot at the two chosen χ indices
│       F = vstack(F, new_row)
│       extra += 1
├── boosted = _rebuild_with_augmented_F(F, original_data_code, layout, ...)
│       - Conceptually rebuilds the merged code using the augmented F.
│       - Reuses v1 _assemble_merged_HX / _assemble_merged_HZ with the new F.
└── return (boosted_merged, boosted_layout, BoostResult(...))


_spectral_cheeger_lower_bound(F: galois.FieldArray) → float
       - Cast F to numpy float, form M = F.astype(float) @ F.astype(float).T
       - λ_2 = numpy.linalg.eigvalsh(M)[1]   # second-smallest eigenvalue
       - h_lb = λ_2 / 2  # Cheeger inequality, simple form
       - Returns h_lb as Python float.
```

### Paper traceability

| Construction element | Swaroop ref | Cross ref | Webster ref |
|---|---|---|---|
| Interface graph S from gadget layer-1 ancillas | (custom, not in repo) | §3.6 implicit | n/a |
| SkipTree basis transformation `_skip_tree` | §III, `skip_tree_algorithm.py` | n/a | n/a |
| Cellulation for cycle length bound `_cellulate_long_cycles` | Lemma 14, `cellulation.py` | n/a | n/a |
| Joint-measurement merged code structure | (custom; "coming soon" in repo) | §3.6, Eq. 35-40 | n/a |
| Spectral Cheeger lower bound | (folk-classical inequality h ≤ √(2 λ_2)) | §3.3 footnote | §II.A end (mentioned without algorithm) |
| Random degree-2 edge augmentation | n/a | n/a | §II.A end, Table I "+n" column |

## 5. Validation

All `ValueError` with explicit messages.

**`build_joint_measurement_code` (always run)**:
1. `data_code.dimension >= 2` (otherwise joint measurement of two independent logicals is degenerate). ValueError: "joint measurement requires at least 2 logical qubits, got data_code.dimension={k}".
2. op1 and op2 both pass v1 `_restrict_to_logical_support` validation independently.
3. Both ops are the same Pauli type: detected by checking whether op1 commutes with `data_code.matrix_z` (X-type) or with `data_code.matrix_x` (Z-type). Mixed → ValueError("op1 and op2 must be the same Pauli type").
4. op1 and op2 are in different logical equivalence classes (so their joint measurement is non-trivial). Checked by verifying that `op1 ⊕ op2` is not in the row span of `H_X` (for X-type) or `H_Z` (for Z-type).
5. `num_layers` is odd ≥ 1.
6. Output merged code is a stabilizer CSSCode (commutation holds).
7. Logical count: `merged.dimension == data_code.dimension - 2`.

**`boost_gadget_cheeger`**:
1. `target_h > 0`.
2. `max_extra_qubits is None or max_extra_qubits >= 0`.
3. `layout.F.shape[1] >= 2` (need at least 2 X-checks to add a degree-2 edge).

No silent fallback. The boost function can return `terminated_by="no_progress"` if no valid edge candidates remain; this is a legitimate (non-error) termination.

## 6. Testing

Unit tests in `src/qldpc/codes/surgery_test.py` (appended to existing 21 tests). All use `galois.GF(2)`.

| Test | Code | What it asserts |
|---|---|---|
| `test_skip_tree_small_graph` | Hand-built 5-vertex tree graph | T has correct shape (4, |E|), P is a permutation matrix, each row of T sums to the length of the corresponding shortest-path (verifiable by hand) |
| `test_cellulate_reduces_cycle_length` | Hand-built 8-vertex cycle | After `_cellulate_long_cycles(G, max_len=4)`, no cycle in `nx.cycle_basis(G)` has length > 4 |
| `test_spectral_cheeger_lower_bound_known_value` | F = [[1,1,0],[0,1,1]] | h_lb computed; compare to manually computed λ_2 |
| `test_boost_gadget_cheeger_increases_lambda2` | Steane + X̄ | After boost, `result.final_h_lower_bound > initial_h_lower_bound` |
| `test_boost_gadget_cheeger_reproducible_with_seed` | Steane | Two runs with `seed=42` produce identical `result.extra_qubits_added` and same final layout |
| `test_boost_gadget_cheeger_respects_max_extra_qubits` | Steane + small max_extra_qubits=3 | `result.extra_qubits_added <= 3`, `result.terminated_by` ∈ {"max_qubits_exhausted", "target_reached"} |
| `test_build_joint_rejects_low_k_data` | Steane (k=1) | Calling with two ops when `data_code.dimension < 2` raises ValueError matching "at least 2 logical qubits" |
| `test_build_joint_small_hgp_X_css_valid` | HGPCode (k ≥ 2) | merged is CSS, `k_merged = k_data - 2`, layout has both gadget layouts populated and bridge data |
| `test_build_joint_invalid_mixed_type_raises` | HGPCode | op1 = logical X, op2 = logical Z → ValueError matching "same Pauli type" |
| `test_joint_webster_observable_X` | small HGP | Π χ_i^(1) ⊕ Π χ_i^(2) restricted to data qubits == (op1 ⊕ op2), restricted to ancilla and bridge == 0 |
| `test_webster_table1_bare_gadget` | 4 generalised bicycle codes l ∈ {31, 63, 127, 255} × 4 seeds each (loaded from JSON) | For each (code, seed) pair: `layout.num_ancilla_qubits` equals Webster's bare gadget number from Table I. **This is v2's acceptance test.** |

Bridge qubit count (Webster's 11/19/31/51) and Cheeger boost +n (Webster's +8/+20) are **NOT** verified to specific numerical values — bridges are SkipTree-deterministic but the spec doesn't pin the algorithm uniquely enough to expect exact match, and Cheeger boost is heuristic. The verification script (§7) reports these as informational.

## 7. Webster Table I verification script

**Path**: `examples/webster_table1_verify.py`.

This is the v2 deliverable that proves the construction matches the paper's Table I numbers for the bare gadget column. It is a Python script (not a notebook) intended to be run directly: `python examples/webster_table1_verify.py`.

Structure:
1. Load `examples/webster_app_a.json` (4 codes × 4 seeds).
2. For each code:
   - Construct `qldpc.codes.BBCode` with `orders=(l, 1)` and polynomial expressions encoding the (l, A, B) data from the JSON.
   - For each seed (X̄_1, Z̄_1, X̄_{k/2+1}, Z̄_{k/2+1}):
     - Build the X̄_1 support vector (length 2lm).
     - Run `build_layered_surgery_code(code, support, num_layers=1)`.
     - Record `layout.num_ancilla_qubits`.
   - Sum gadget qubits across the 4 seeds; multiply by 1 (each seed has its own gadget) to get the per-code gadget-qubit total.
3. For each code: build the joint measurement code for one X̄ pair and one Z̄ pair, recording bridge qubit counts.
4. For each code with non-trivial Cheeger augmentation (codes 3 and 4): run `boost_gadget_cheeger` on each seed gadget with `target_h=1.0` and a fixed seed for reproducibility, recording `result.extra_qubits_added`.
5. Print a markdown table comparing observed numbers to Webster Table I:

```
| Code         | Bare gadget (paper) | Bare gadget (ours) | +n (paper) | +n (ours) | Bridge (paper) | Bridge (ours) |
|--------------|---------------------|--------------------|------------|-----------|----------------|---------------|
| J62,10,6K   | 19                  | XX                 | 0          | YY        | 11             | ZZ            |
| J126,12,10K  | 31                  | XX                 | 0          | YY        | 19             | ZZ            |
| J254,14,16K  | 49                  | XX                 | 8          | YY        | 31             | ZZ            |
| J510,16,24K  | 79                  | XX                 | 20         | YY        | 51             | ZZ            |
```

6. Exit 0 if all bare-gadget numbers match exactly; exit 1 otherwise. This is the script-level acceptance gate.

Implementation note on `orders=(l, 1)`: the qldpc `BBCode` expects bivariate polynomials A(x, y) and B(x, y). For Webster's single-variable codes, we encode `A(x) = sum(x^a for a in A_set)` (no y dependence) and `B(x) = sum(x^b for b in B_set)`. The y direction has order 1 so the resulting code is equivalent to the single-variable cyclic bicycle code Webster uses. Validate this equivalence in Task v2.2 (see §8).

## 8. Implementation order

The v2 plan continues on branch `feat/surgery-construction` (already at HEAD `f8ab407`). Tasks are sequenced so that the most impactful test (Webster Table I bare gadget verification) runs early, exercising v1 directly.

| v2 Task | Subject | Depends on |
|---|---|---|
| 1 | JSON data file `examples/webster_app_a.json` + reader `load_webster_seed_set` | v1 |
| 2 | BBCode `orders=(l, 1)` sanity test: build, count qubits, verify dimension | v1 |
| **3** | **Webster Table I bare gadget verification test** `test_webster_table1_bare_gadget`. If this fails, v1 has a bug — must fix before continuing. | v1, tasks 1, 2 |
| 4a | Port `_skip_tree` from Swaroop (MIT) + tests | networkx |
| 4b | Port `_cellulate_long_cycles` from Swaroop (MIT) + tests | networkx |
| 5 | `_spectral_cheeger_lower_bound` helper + test | numpy |
| 6 | `BoostResult` dataclass + `boost_gadget_cheeger` + tests | task 5 |
| 7 | `JointSurgeryLayout` dataclass | v1 |
| 8 | `_validate_joint_logical_ops` + invalid-input tests | v1 |
| 9 | `_build_bridge_via_skiptree`: interface graph construction + SkipTree integration + tests | tasks 4a, 4b, 7 |
| 10 | `_stitch_gadgets_with_bridge` + tests | tasks 7, 9 |
| 11 | Top-level `build_joint_measurement_code` + integration tests including Webster Eq. (1) joint extension | tasks 8, 9, 10 |
| 12 | Re-export new symbols from `qldpc.codes` `__init__.py` | tasks 6, 7, 11 |
| 13 | `examples/webster_table1_verify.py` script | tasks 3, 6, 11 |

Total: ~13 tasks. Estimated 1–2 weeks of focused implementation.

## 9. References

**Primary** (v2 additions):
- Cross, He, Rall, Yoder. *Improved QLDPC Surgery*. arXiv:2407.18393 §3.6 — bridge construction for joint X̄X̄' measurements (the antecedent of Swaroop's universal adapter framework).
- Swaroop, Jochym-O'Connor, Yoder. *Universal adapters between quantum LDPC codes*. arXiv:2410.03628 §III (SkipTree), Lemma 14 (cellulation).
- **Swaroop reference implementation**: https://github.com/eswaroop/adapters-LDPC-surgery (MIT, 2025). The `_skip_tree` and `_cellulate_long_cycles` helpers in our `surgery.py` are direct ports with attribution comments.
- Webster, Smith, Cohen. *Explicit construction of low-overhead gadgets*. arXiv:2511.15989 §II.A end (Cheeger augmentation), Table I (verification target), Appendix A (4-code, 4-seed-each fixture data).

**Inherited from v1**: all references in `2026-06-05-cross-layered-ancilla-design.md` §9.

**MIT attribution note**: Following MIT license requirements, the ported `_skip_tree` and `_cellulate_long_cycles` functions in `src/qldpc/codes/surgery.py` carry a comment block crediting `eswaroop/adapters-LDPC-surgery` (2025) and noting both the qLDPC (Apache 2.0) and Swaroop (MIT) licenses are compatible for redistribution.
