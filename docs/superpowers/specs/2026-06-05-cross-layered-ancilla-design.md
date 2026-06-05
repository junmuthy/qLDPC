# Design: Cross et al. 2024 Layered Ancilla Construction for QLDPC Surgery

**Date**: 2026-06-05
**Status**: Approved (brainstorming phase)
**Reference**: Cross, He, Rall, Yoder. *Improved QLDPC Surgery: Logical Measurements and Bridging Codes*. arXiv:2407.18393, §III, Lemma 4–Theorem 6, §IV.1.

## 1. Motivation

The qLDPC repository currently has no implementation of generalized lattice surgery for QLDPC codes. A prior in-tree attempt (a Cohen-style L1 toy implementation) fails BP decoding for bivariate-bicycle codes because the chosen ancilla graph does not expand sufficiently — Cohen / CKBB requires `L = 2d − 1` layers in the worst case, which is impractical (Table 1 of the paper: 1380 ancilla qubits for the [[144,12,12]] gross code).

Cross et al. 2024 §III replaces the random expander with a *deterministic* layered Tanner graph built from `F = H_Z[C_0, V_0]` and proves that `⌈L/2⌉ ≥ 1/β` (β = boundary Cheeger constant of F) suffices, where in practice L ∈ {1, 3, 5} covers all cases of interest. For the [[144,12,12]] gross code with hand-chosen polynomial logical operators the authors achieve L = 1 with 103 ancilla qubits.

This spec adds the layered ancilla construction to qLDPC as a public API plus an end-to-end Cain Fig 1b reproduction notebook.

## 2. Scope

**In scope**:
- A function `build_layered_surgery_code(data_code, logical_op, *, num_layers=1)` that returns a merged `CSSCode` plus a `SurgeryLayout` describing qubit/check provenance.
- Internal helpers for the four algorithmic steps (restriction, layered block construction, gauge fix, matrix assembly).
- Unit tests for CSS validity, helper correctness, and layout consistency.
- A reproduction notebook `examples/logical_error_rates/9_lattice_surgery_cain_fig1b.ipynb`.

**Out of scope (future work)**:
- Logical Z measurement merge (symmetric construction; trivial extension once X works).
- Bridge / joint measurement systems (paper §3.6–3.8).
- Automatic Cheeger constant estimation.
- Distance verification (caller's responsibility; paper uses CPLEX).
- Syndrome extraction circuit synthesis for the merged code.

## 3. Public API

**Module**: `src/qldpc/codes/surgery.py` (new file, sibling of `quantum.py`, `common.py`).
**Re-exports** from `src/qldpc/codes/__init__.py`: `build_layered_surgery_code`, `SurgeryLayout`.

```python
@dataclass(frozen=True)
class SurgeryLayout:
    """Provenance of qubits and checks in a merged surgery code."""

    num_data_qubits: int
    num_ancilla_qubits: int
    num_layers: int

    # Qubit columns: layer index per merged-code qubit column.
    # 0 = data qubit, i ∈ {1, ..., L} = ancilla qubit in layer i.
    qubit_layer: npt.NDArray[np.int_]      # shape (n_merged,)

    # Data-side restriction (Step 1 of Cross §III).
    v0_indices: npt.NDArray[np.int_]       # qubit indices in supp(X̄_M)
    c0_indices: npt.NDArray[np.int_]       # Z-check row indices adjacent to V_0

    # Step 1 restriction matrix and Step 4 gauge-fix basis.
    F: npt.NDArray[np.int_]                # shape (|C_0|, |V_0|)
    G: npt.NDArray[np.int_]                # shape (rank(null(F)), |C_0|)

    # Check row provenance for the merged code.
    # Values: "data", "ancilla_L1", "ancilla_L2", ..., "gauge_fix".
    hx_row_kind: npt.NDArray               # shape (n_x_checks_merged,), dtype=object/str
    hz_row_kind: npt.NDArray               # shape (n_z_checks_merged,), dtype=object/str


def build_layered_surgery_code(
    data_code: CSSCode,
    logical_op: npt.ArrayLike,
    *,
    num_layers: int = 1,
    validate_logical_op: bool = True,
) -> tuple[CSSCode, SurgeryLayout]:
    """Construct a merged stabilizer code that measures logical_op by lattice surgery.

    Implements the layered ancilla construction of Cross et al. 2024 §III (arXiv:2407.18393).
    Given a CSSCode and the binary support vector of a logical X operator X̄_M, this builds
    L ancilla layers (L = num_layers, must be odd) with intra-layer wiring from F = H_Z[C_0, V_0]
    and inter-layer identity wiring, plus rank(null(F)) gauge-fixing Z-checks on the top layer.
    The result is a stabilizer CSSCode encoding (k_data − 1) logical qubits.

    Args:
        data_code: The data CSSCode (stabilizer, not subsystem).
        logical_op: Binary row vector of length data_code.num_qubits indicating supp(X̄_M).
        num_layers: Number of ancilla layers L (odd, ≥ 1). Default 1 follows the [[144,12,12]]
            gross code example in the paper (Table 1). For arbitrary logical_op, distance
            preservation may require num_layers ∈ {3, 5}; this function does NOT verify
            distance — caller must check separately if needed. See paper §IV.1 for selection
            guidance.
        validate_logical_op: If True, verifies logical_op commutes with Z-stabilizers and is
            not in the row span of H_X. Skip with False for performance on large codes if
            the caller has already validated.

    Returns:
        (merged_code, layout):
            merged_code: CSSCode on (n_data + num_ancilla) qubits, with logical dimension
                k_data − 1.
            layout: SurgeryLayout describing the qubit/check partition for downstream use
                (circuit synthesis, decoder configuration, plotting).

    Raises:
        ValueError: if num_layers is even or < 1, logical_op has wrong shape or is not binary,
            V_0 is empty, logical_op does not commute with Z-stabilizers, logical_op is a
            stabilizer (only checked when validate_logical_op=True), or data_code is a
            subsystem code.
    """
```

## 4. Algorithm structure

The top-level function orchestrates five pure helpers. All linear algebra runs over GF(2) using `galois.GF(2)` (consistent with `qldpc.codes.common`).

```
build_layered_surgery_code
├── _restrict_to_logical_support(data_code, logical_op)
│     → returns (V_0_indices, C_0_indices, F)
│     Step 1: V_0 = supp(logical_op); C_0 = {row indices in H_Z whose support meets V_0};
│             F = H_Z[C_0, V_0].
│     Performs all input validation (§5).
│
├── _build_layered_blocks(F, num_layers)
│     → returns LayeredBlocks (internal dataclass; per-layer block matrices)
│     Step 3: builds the merged-Tanner-graph block matrices layer by layer.
│             Even layers (i ≥ 2): contribute |V_0| new ancilla qubits and |C_0| new Z-checks.
│             Odd layers (i ≥ 1):  contribute |V_0| new X-checks and |C_0| new ancilla qubits.
│             Intra-layer block alternates F (even) / F^T (odd).
│             Inter-layer blocks are identity wiring on same-index vertices.
│
├── _compute_gauge_fix(F)
│     → returns G of shape (rank(null(F)), |C_0|)
│     Step 4: G = basis matrix for null(F) via galois `null_space()`.
│
├── _assemble_merged_HX(data_code, blocks, V_0_indices)
│     → returns H_X^merged (galois FieldArray)
│
└── _assemble_merged_HZ(data_code, blocks, G, C_0_indices)
      → returns H_Z^merged (galois FieldArray)
```

### 4.1 Qubit column order in the merged code

```
[ data qubits | C_1 ancilla | V_2 ancilla | C_3 ancilla | V_4 ancilla | ... | C_L ancilla ]
   n_data       |C_0|         |V_0|         |C_0|         |V_0|              |C_0|
```

For odd i ≥ 1, layer i contributes |C_0| ancilla qubits (these are `C_i` in Cross's notation).
For even i ≥ 2, layer i contributes |V_0| ancilla qubits (these are `V_i`).
L is always odd, so the top layer contributes |C_0| qubits.

### 4.2 Block-matrix structure (L = 1)

```
                    data         C_1
                  ┌─────────┬───────────┐
H_X (X-stabs):    │ H_X^d   │    0      │  ← old data X-checks (n_x_data rows)
                  ├─────────┼───────────┤
                  │  Π_V0   │   F^T     │  ← V_1 new X-checks (|V_0| rows)
                  └─────────┴───────────┘

H_Z (Z-stabs):    ┌─────────┬───────────┐
                  │ H_Z^¬C0 │    0      │  ← non-C_0 data Z-checks (untouched)
                  ├─────────┼───────────┤
                  │ H_Z^C0  │    I      │  ← C_0 data Z-checks + identity-attached C_1
                  ├─────────┼───────────┤
                  │   0     │    G      │  ← gauge-fix rows (rank(null(F)) rows)
                  └─────────┴───────────┘
```

`Π_V0` is the |V_0| × n_data sparse-injection matrix: row v has a 1 at column `v0_indices[v]`.

### 4.3 Block-matrix structure (L = 3)

```
                    data       C_1         V_2         C_3
H_X:              ┌─────────┬─────────┬─────────┬─────────┐
                  │ H_X^d   │    0    │    0    │    0    │  ← data X-checks
                  ├─────────┼─────────┼─────────┼─────────┤
                  │  Π_V0   │   F^T   │    I    │    0    │  ← V_1 X-checks
                  ├─────────┼─────────┼─────────┼─────────┤
                  │    0    │    0    │    I    │   F^T   │  ← V_3 X-checks
                  └─────────┴─────────┴─────────┴─────────┘

H_Z:              ┌─────────┬─────────┬─────────┬─────────┐
                  │ H_Z^¬C0 │    0    │    0    │    0    │  ← non-C_0 data Z-checks
                  ├─────────┼─────────┼─────────┼─────────┤
                  │ H_Z^C0  │    I    │    0    │    0    │  ← C_0 data Z-checks
                  ├─────────┼─────────┼─────────┼─────────┤
                  │    0    │    I    │    F    │    I    │  ← C_2 Z-checks
                  ├─────────┼─────────┼─────────┼─────────┤
                  │    0    │    0    │    0    │    G    │  ← gauge-fix on C_3
                  └─────────┴─────────┴─────────┴─────────┘
```

### 4.4 General odd L (loop body specification)

| Row source | Block pattern |
|---|---|
| Old data X-check | `[H_X^d, 0, 0, ..., 0]` |
| V_1 new X-check | `[Π_V0, F^T, I, 0, ..., 0]` (last `I` absent if L = 1) |
| V_{2k+1} new X-check (k ≥ 1) | zeros except `I` on V_{2k} block, `F^T` on C_{2k+1} block, `I` on V_{2k+2} block if present |
| Old data Z-check ∈ ¬C_0 | `[H_Z row, 0, ..., 0]` |
| Old data Z-check ∈ C_0 | `[H_Z row, I, 0, ..., 0]` (extension on C_1) |
| C_{2k} new Z-check (k ≥ 1) | zeros except `I` on C_{2k−1} block, `F` on V_{2k} block, `I` on C_{2k+1} block |
| Gauge-fix U_L | zeros except `G` on C_L block |

`_build_layered_blocks` is one loop from i = 1 to L; even/odd branch selects intra-layer F vs F^T and which adjacent blocks get identity wiring.

### 4.5 Paper traceability

Every construction element above maps to a specific passage in arXiv:2407.18393. Line numbers refer to the rendered PDF text extraction; section/equation numbers are paper-canonical.

**Notation convention**: the paper writes inter-layer wiring with the higher-layer vertex on the left of the arrow (e.g. `I : V_1 →_X V_0`, `I : C_2 →_Z C_1`). Identity is self-transpose, so directionality is cosmetic, but implementations should match the paper for grep-ability.

| Construction element | Paper citation |
|---|---|
| `V_0 = supp(X̄_M)`, `C_0 = {Z-checks neighboring V_0}`, `F = J_{C_0}^⊤ H^Z J_{V_0}` (= `H_Z[C_0, V_0]`) | §2.2, paragraph beginning "Key to their construction" |
| Vertex roles per layer (odd i → X-check/qubit; even i → qubit/Z-check) | §2.2, Eq. (1)–(2) |
| Intra-layer wiring: `F : C_i →_Z V_i` for even i, `F^⊤ : V_i →_X C_i` for odd i | §2.2, sentence beginning "Layers are connected identically via F" |
| Inter-layer identities `I : C_0 →_Z C_1`, `I : V_1 →_X V_0`, `I : V_1 →_X V_2`, `I : C_2 →_Z C_1`, etc. | §2.2, next sentence ("adjacent layers are connected by the identity") |
| Layers extended to L total: `(C_1, V_1), ..., (C_L, V_L)` with the same wiring rules | §3.1, first paragraph |
| Gauge-fix: `G` spans `null(F)` (`G F = 0`), and `G : U_L →_Z C_L` introduces `rank(null(F))` new Z-checks; "Minimizing row and column weight of G minimizes the degrees added" | §3.1, paragraph beginning "To find a set of additional Z checks" |
| Merged code G_X is a non-subsystem stabilizer code; `X̄_M` becomes a stabilizer; `k(G_X) = k(G) − 1` | §3.1, Theorem 1 |
| Layer-count condition `⌈L/2⌉ ≥ 1/β` (β = boundary Cheeger constant of F); `L = 1` reference example for [[144,12,12]] gross code | §3.3, Lemma 4 / Theorem 6; §IV.1 mono-layer construction; Table 1 |
| Distance preservation verified numerically (CPLEX), not from β | §IV.1, paragraph beginning "We verify numerically using CPLEX" |
| Practical L ∈ {1, 3, 5} for small-to-medium codes (worst-case bound is pessimistic) | §3.3, paragraph beginning "We also consider Theorem 6 a worst case upper bound" (cites [Cow24]) |

The block-matrix expansions in §4.2 / §4.3 / §4.4 are mechanical specializations of these rules to concrete L; no additional algorithmic content beyond what is stated in §2.2 + §3.1.

## 5. Validation

All validation lives in `_restrict_to_logical_support` at function entry. Raises `ValueError` with explicit messages.

**Always run (cheap)**:
1. `logical_op.shape == (n_data,)` and values ⊂ {0, 1}.
2. `logical_op.sum() > 0` (V_0 non-empty).
3. `num_layers >= 1` and `num_layers % 2 == 1`.
4. `(H_Z @ logical_op.T) % 2 == 0` — commutes with Z-stabilizers.
5. `data_code` is a stabilizer CSSCode (not subsystem). Detected via the public property `data_code.is_subsystem_code` (defined at `src/qldpc/codes/common.py:880`).

**Default-on, skippable** (`validate_logical_op=False` to skip):
6. `logical_op` is not in the row span of H_X. Verified by `rank(H_X) == rank(vstack([H_X, logical_op]))` over GF(2).

No silent fallback or auto-correction.

## 6. Testing

Unit tests in `src/qldpc/codes/surgery_test.py`. All use `galois.GF(2)` for deterministic verification.

| Test | Code | num_layers | Asserts |
|---|---|---|---|
| `test_steane_L1_css_valid` | Steane [[7,1,3]] | 1 | `(H_X^m @ H_Z^m.T) % 2 == 0`, k_merged = 0 |
| `test_steane_L3_css_valid` | Steane | 3 | Same; exercises ≥ 1 odd layer in loop body |
| `test_F_equals_HZ_restriction` | Steane | 1 | `layout.F == H_Z[C_0, V_0]` elementwise |
| `test_G_is_null_basis_of_F` | Steane | 1 | `(G @ F.T) % 2 == 0`; `G.shape[0] == |C_0| − rank(F)` |
| `test_layout_partition_consistency` | Steane | 1, 3 | `qubit_layer` block sizes match merged column count; `hx_row_kind` / `hz_row_kind` lengths match row counts |
| `test_small_hgp_L1` | small HGPCode | 1 | CSS validity, logical count on a non-Steane code |
| `test_invalid_inputs_raise` | Steane | — | Wrong shape, even L, L=0, stabilizer input (rejected with `validate_logical_op=True`), trivial input, non-binary, subsystem code — each raises `ValueError` with a distinct message |

The internal helper sanity tests (`F`, `G`, layout consistency) go beyond the user-required validation surface (stabilizer + logical count) but are inexpensive and isolate helper bugs that the top-level CSS check would mask.

No distance computation or Cain Table III matching in unit tests — those belong in the notebook.

## 7. End-to-end notebook

**Path**: `examples/logical_error_rates/9_lattice_surgery_cain_fig1b.ipynb`.

Sections:
1. Imports; construct the bb_18 BBCode (polynomial / orders specified inline).
2. Choose X̄_M (default `code.get_logical_ops(Pauli.X)[0]`; hook to swap representative).
3. Call `build_layered_surgery_code(bb18, X_bar, num_layers=1)`.
4. Print ancilla qubit count, new X-check count, new Z-check count from `layout`. Qualitative comparison against Cain Table III (189, 104, 86); exact match not expected (Cain construction may include bridge qubits beyond Cross's bare layered ancilla).
5. Build memory experiment circuit on `merged_code` via existing `qldpc.circuits.memory` API.
6. Configure BP-LSD decoder following Cain Appendix D settings.
7. Sweep physical error rate; produce LER curve.
8. Plot alongside Cain Fig 1b for visual comparison.

Notebook follows the established `NUM_WORKERS` constant and `num_workers` parameter conventions in this repo. It is not run in CI; it is a manual reproduction artifact.

## 8. Implementation order

1. `SurgeryLayout` dataclass.
2. `_restrict_to_logical_support` + validation logic; tests `test_invalid_inputs_raise`, `test_F_equals_HZ_restriction`.
3. `_compute_gauge_fix`; test `test_G_is_null_basis_of_F`.
4. `_build_layered_blocks` (loop body); covered indirectly via the assembly tests.
5. `_assemble_merged_HX`, `_assemble_merged_HZ`.
6. Top-level `build_layered_surgery_code`; tests `test_steane_L1_css_valid`, `test_steane_L3_css_valid`, `test_layout_partition_consistency`, `test_small_hgp_L1`.
7. Re-exports in `codes/__init__.py`.
8. Notebook.

## 9. References

- Cross, He, Rall, Yoder. *Improved QLDPC Surgery: Logical Measurements and Bridging Codes*. arXiv:2407.18393 (2024). §III, Lemma 4, Theorem 6; §IV.1 for the L = 1 mono-layer example on the [[144,12,12]] gross code; Table 1 for L comparison with CKBB and [Cow24].
- Cohen, Kim, Bartlett, Brown. *Low-overhead fault-tolerant quantum computing using long-range connectivity*. Sci. Adv. 8, eabn1717 (2022). Original CKBB scheme with L = 2d − 1.
- Cowtan. *SSIP: Automated Surgery with QLDPC Codes*. arXiv:2407.09423 (2024). Heuristic L = 5 for the gross code.
- Bravyi, Cross, Gambetta, et al. *High-threshold and low-overhead fault-tolerant quantum memory*. Nature 627 (2024). The [[144,12,12]] code and notation.
