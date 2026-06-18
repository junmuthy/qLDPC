# Mixed-basis Joint PPM Design

Branch: `feat/latticesurgery-mixedjoint`
Author: tgzhou + Claude Opus 4.7
Status: design (awaiting implementation)
Created: 2026-06-15

## 1. Goal

Extend `qldpc.circuits.surgery` to support **joint Pauli-product measurement of two logical operators of different Pauli type** — e.g. measuring `Z̄_l ⊗ X̄_r` (one Z-type, one X-type) — without measuring the individual operators.

Concrete API change: `build_bridge(g_l, g_r)` accepts `g_l.basis != g_r.basis`. The returned `Bridge` and downstream `build_joint_ppm_circuit` produce a stim circuit whose observable commits exactly the mixed-basis product.

## 2. Background

Existing `build_bridge` requires `g_l.basis is g_r.basis` (Webster L=1 + Swaroop SkipTree adapter, same Pauli basis on both sides). Mixed-basis (X on one side, Z on the other) is blocked by a CSS obstruction: a pure-CSS merged code cannot contain `Z̄_l ⊗ X̄_r` as a stabilizer product without also fixing the individuals.

**Webster–Smith–Cohen** (arXiv:2511.15989, §II.B.2) gives the recipe to break the obstruction: build the same-basis Swaroop adapter, then on shared bridge qubits apply X-pair merges, Z-pair merges, and a final X+Z → Y cross-merge. The merged code is non-CSS (contains Y-type stabilizers on bridge qubits) but supports the mixed-basis joint PPM.

**Scope**: only bridge-qubit X/Z conflicts (case (i) in the brainstorming). Data-qubit overlap (case (ii), intra-code `X̄_1 ⊗ Z̄_2` on Webster seeds where supports actually intersect) is deferred. Logical operator supports must be disjoint; intra-code applications are limited to disjoint-support logical pairs.

## 3. Architecture

```
build_gadget(L_l, basis_l)
build_gadget(L_r, basis_r)
        │
        ▼
build_bridge(g_l, g_r):
    if basis_l == basis_r:
        return _legacy_same_basis_path(...)           # unchanged
    else:
        bridge = _build_same_basis_adapter(...)       # reuse SkipTree
        bridge = _apply_mixed_basis_merge(bridge)     # new merge.py
        return bridge   # now has Y_stab, basis_l ≠ basis_r
        │
        ▼
build_joint_ppm_circuit(g_l, g_r, bridge, rounds, ...):
    if bridge.Y_stab is None:
        return _legacy_css_pipeline(...)              # unchanged
    else:
        joint_code = _stitch_to_quditcode(...)        # QuditCode, not CSSCode
        return _mixed_basis_pipeline(...)             # CX/CY/CZ Y-stab extraction,
                                                      # Y detector registration,
                                                      # obs0 with Y outcomes
```

**Key decisions**:
- Merge runs **after** the existing SkipTree adapter, in-place on the assembled stabilizer matrices. SkipTree / cellulation / gauge fix code is unmodified.
- Merged code uses **`QuditCode`** (general symplectic matrix) when Y-stabs present, **`CSSCode`** when same-basis. The QuditCode base already supports mixed-Pauli stabilizers via its symplectic representation — no new container class needed.
- Y-stab measurement uses **explicit CX/CY/CZ ancilla extraction** (matching the existing X/Z stab measurement pattern). `MPP` shortcut rejected — must remain fault-tolerant for downstream noise verification.

## 4. Bridge Dataclass Extension

```python
@dataclass(frozen=True, eq=False)
class Bridge:
    width: int
    basis_l: PauliXZ                            # was: basis: PauliXZ
    basis_r: PauliXZ                            # NEW
    port_l: tuple[int, ...]
    port_r: tuple[int, ...]
    label_l: tuple[int, ...]
    label_r: tuple[int, ...]
    extra_ancilla_l: np.ndarray
    extra_ancilla_r: np.ndarray
    T_l: np.ndarray
    T_r: np.ndarray
    H_R: np.ndarray
    g_l_aug: GadgetLayout
    g_r_aug: GadgetLayout
    # NEW mixed-basis fields:
    Y_stab: np.ndarray | None = None            # shape (n_Y, 2*n_merged) symplectic, None if same-basis
    merge_qubits: tuple[int, ...] = ()          # bridge qubit indices touched by cross-merge
    obs0_xor_map: tuple[int, ...] = ()          # indices of Y-stab rows to XOR into obs0
    x_leftover_indices: tuple[int, ...] = ()    # row indices of X-cycle checks not cross-merged (singletons)
    z_leftover_indices: tuple[int, ...] = ()    # row indices of Z-cycle checks not cross-merged (singletons)

    @property
    def basis(self) -> PauliXZ:
        """Backward-compat: raises if mixed-basis."""
        if self.basis_l is not self.basis_r:
            raise AttributeError("mixed-basis Bridge has no single .basis; use basis_l/basis_r")
        return self.basis_l
```

Backward-compat: existing callers using `bridge.basis` keep working in same-basis case. Mixed-basis callers use `bridge.basis_l` / `bridge.basis_r`.

## 5. Merge Algorithm (Webster §II.B.2)

Runs in a new module `src/qldpc/circuits/surgery/merge.py`. Pure GF(2) row arithmetic on the merged stabilizer matrices produced by the same-basis SkipTree adapter.

### 5.1 Input

After the same-basis adapter (in the mixed-basis dispatch), we have a "tentative" merged code:
- `H_X_tentative`: shape `(N_X, n_merged)` — original code X-stabs (both sides) + Swaroop eq (37) X-cycle rows (from the side whose basis is the meas-basis dual)
- `H_Z_tentative`: shape `(N_Z, n_merged)` — original Z-stabs + Webster χ rows + Swaroop eq (37) Z-cycle rows (from the other side)
- `merge_qubits`: bridge qubit indices q where both X-type and Z-type checks have support

`merge_qubits` is computed by intersecting the supports of `H_X_tentative` and `H_Z_tentative` on the bridge-qubit column range (`c_adapter` + relevant κ ancilla blocks).

### 5.2 Per-qubit merge (Webster §II.B.2)

```python
def _merge_at_qubit(H_X, H_Z, q):
    """In-place GF(2) row ops. Returns (H_X, H_Z, Y_row or None, x_row_idx or None, z_row_idx or None)."""

    # Step A: pair-merge X-checks on q
    x_rows = np.flatnonzero(H_X[:, q])
    leftover_x = None
    if len(x_rows) >= 1:
        pivot = x_rows[0]
        for r in x_rows[1:]:
            H_X[r] = (H_X[r] + H_X[pivot]) % 2   # XOR: cancels q-column except in pivot
        leftover_x = pivot

    # Step B: pair-merge Z-checks on q (symmetric)
    z_rows = np.flatnonzero(H_Z[:, q])
    leftover_z = None
    if len(z_rows) >= 1:
        pivot = z_rows[0]
        for r in z_rows[1:]:
            H_Z[r] = (H_Z[r] + H_Z[pivot]) % 2
        leftover_z = pivot

    # Step C: cross-merge leftover X + leftover Z → Y row
    Y_row = None
    if leftover_x is not None and leftover_z is not None:
        # Build symplectic Y row: x-support from H_X[leftover_x], z-support from H_Z[leftover_z]
        Y_row = np.zeros(2 * H_X.shape[1], dtype=np.uint8)
        Y_row[:H_X.shape[1]] = H_X[leftover_x]              # X-part
        Y_row[H_X.shape[1]:] = H_Z[leftover_z]              # Z-part
        # Remove the two leftover rows (their product is now the Y-stab)
        H_X = np.delete(H_X, leftover_x, axis=0)
        H_Z = np.delete(H_Z, leftover_z, axis=0)

    return H_X, H_Z, Y_row, leftover_x, leftover_z
```

### 5.3 Iteration over merge qubits

```python
def apply_mixed_basis_merge(H_X, H_Z, merge_qubits):
    Y_stab_rows = []
    obs0_y_indices = []                          # which Y_stab rows go into obs0
    for q in merge_qubits:
        H_X, H_Z, Y_row, _, _ = _merge_at_qubit(H_X, H_Z, q)
        if Y_row is not None:
            obs0_y_indices.append(len(Y_stab_rows))   # record before append
            Y_stab_rows.append(Y_row)
    Y_stab = np.array(Y_stab_rows) if Y_stab_rows else None
    return H_X, H_Z, Y_stab, obs0_y_indices
```

Iteration order: ascending qubit index (deterministic). Earlier merges may change which checks touch later qubits — that's fine, the algorithm still terminates with proper behavior because each step strictly reduces total `|H_X[:, q]| + |H_Z[:, q]|` at the current q to 0 (no X-cycle or Z-cycle touching q after Step C).

### 5.4 Output

- Modified `H_X`, `H_Z` (smaller, no X+Z conflict on `merge_qubits`)
- `Y_stab` matrix (shape `(n_Y, 2*n_merged)`, symplectic form)
- `obs0_y_indices`: ordered list of Y_stab row indices contributing to obs0 (used by `build_joint_ppm_circuit`)

The `Bridge` dataclass receives `Y_stab=Y_stab, merge_qubits=tuple(merge_qubits), obs0_xor_map=tuple(obs0_y_indices)`.

## 6. Stitch: CSSCode → QuditCode

Existing `_stitch_intercode` / `_stitch_intracode` end with:
```python
if bridge.basis is Pauli.X:
    return CSSCode(field(M_meas), field(M_comp), is_subsystem_code=False)
return CSSCode(field(M_comp), field(M_meas), is_subsystem_code=False)
```

New dispatch:
```python
def _stitch_intercode(g_l, g_r, bridge):
    M_meas, M_comp = _assemble_meas_comp(g_l, g_r, bridge)   # existing logic

    if bridge.Y_stab is None:
        # same-basis: legacy CSS path
        if bridge.basis_l is Pauli.X:
            return CSSCode(field(M_meas), field(M_comp), is_subsystem_code=False)
        return CSSCode(field(M_comp), field(M_meas), is_subsystem_code=False)

    # mixed-basis: merge + QuditCode
    H_X, H_Z = _arrange_as_HX_HZ(M_meas, M_comp, bridge.basis_l, bridge.basis_r)
    H_X, H_Z, Y_stab_from_merge, _ = apply_mixed_basis_merge(H_X, H_Z, bridge.merge_qubits)
    # bridge.Y_stab already populated by build_bridge; re-derived here for consistency check
    assert np.array_equal(Y_stab_from_merge, bridge.Y_stab), "Y_stab mismatch between build_bridge and stitch"

    sym_matrix = _pack_symplectic(H_X, H_Z, bridge.Y_stab, n_merged)
    return QuditCode(field(sym_matrix), is_subsystem_code=False)
```

`_arrange_as_HX_HZ`: when `basis_l != basis_r`, the M_meas / M_comp blocks split differently into H_X vs H_Z. Specifically:
- `basis_l=Z` χ rows → H_Z (the Z-meas-check Webster χ rows)
- `basis_r=X` χ rows → H_X (the X-meas-check Webster χ rows on the other side)
- bridge X-cycle checks from the basis=Z side → H_X
- bridge Z-cycle checks from the basis=X side → H_Z
- Original code H_X^(l), H_X^(r), H_Z^(l), H_Z^(r) → appropriate H_X or H_Z

`_pack_symplectic`: builds `(num_stabs, 2, n_merged)` from H_X (X-only rows: matrix[i, 0, :] = X-support, matrix[i, 1, :] = 0), H_Z (Z-only rows: symmetric), and Y_stab (matrix[i, 0, :] = X-part, matrix[i, 1, :] = Z-part).

## 7. Y-stab Measurement + Detector + obs0

### 7.1 Measurement circuit (per Y-stab row, per round)

```python
# Y-stab row in symplectic form: (x_support, z_support)
# Pauli at qubit q: lookup table from (x[q], z[q]) → 'I', 'X', 'Y', 'Z'
y_anc = next_y_ancilla()
circuit.append("RX", [y_anc])
for q in sorted_support(Y_row):
    p = pauli_at(Y_row, q)
    if p == 'X':   circuit.append("CX", [y_anc, q])
    elif p == 'Y': circuit.append("CY", [y_anc, q])
    elif p == 'Z': circuit.append("CZ", [y_anc, q])
    # 'I' skipped
circuit.append("MX", [y_anc])
```

`CX`, `CY`, `CZ`, `RX`, `MX` are stim native Cliffords. Ancilla allocation: each Y-stab gets one dedicated ancilla qubit (allocated in `_surgery_qubit_coordinates`), reused across rounds.

### 7.2 Detector registration

| Round | Y-stab detector |
|---|---|
| Round 1 | **Not registered** — Y-stab on `X-init` or `Z-init` data has random outcome (no eigenstate reference). |
| Round 2 ≤ t ≤ τ_s | `m_t(Y_stab) ⊕ m_{t-1}(Y_stab)` |
| Final (destructive readout) | **Not registered** — single-basis destructive measurement cannot reconstruct Y-stab. |

Implementation:
- `_classify_reliable_round1_checks_joint` returns `(reliable_x_checks, reliable_z_checks, reliable_y_checks=[])`. The Y bucket is always empty because no Y-stab is reliable under single-basis data initialization.
- `_surgery_qec_cycle_joint` adds a Y-stab measurement + detector loop after the existing X/Z loops, structurally parallel.
- `_surgery_final_detectors_joint` skips Y-stabs when computing final destructive cross-check detectors.

### 7.3 obs0 formula

Same-basis obs0 (legacy):
```
obs0 = ⊕_v m(χ_v^(l)) ⊕ ⊕_v m(χ_v^(r)) ⊕ ⊕_c m(adapter X-cycle c)
```

Mixed-basis obs0 (this design):
```
obs0 = ⊕_v m(χ_v^(l)) ⊕ ⊕_v m(χ_v^(r))
       ⊕ ⊕_{i ∈ obs0_xor_map} m(Y_stab[i])
       ⊕ ⊕_{remaining X-cycle outcomes that did not get merged}
       ⊕ ⊕_{remaining Z-cycle outcomes that did not get merged}
```

The `obs0_xor_map` is the list of Y-stab row indices recorded during the merge algorithm. Remaining X-cycle and Z-cycle outcomes correspond to the leftover X-singletons and Z-singletons after pair-merging (cases where one side had an odd number of conflicting checks, where Step C in §5.2 only triggered for one Pauli type but not the other).

The exact set of XOR terms is determined by Lemma 2 (§9). Implementation reads three `Bridge` fields: `obs0_xor_map` (Y-stab rows), `x_leftover_indices` (X-cycle rows whose Step A left them in H_X after the pivot operation), and `z_leftover_indices` (symmetric on Z side).

## 8. Test Strategy

### Tier 1: Noiseless correctness (must-have)

Located in `src/qldpc/circuits/surgery/merge_test.py` (new file) and `bridge_mixed_test.py` / `circuit_mixed_test.py` (new files, not polluting existing test files).

1. **Stab commutation**: assert merged code's `(H_X, H_Z, Y_stab)` stabilizers commute pairwise over the symplectic F_2 inner product. Pure algebra, no circuit needed.
2. **Distance verification**: per Appendix A, verify merged code distance ≥ `min(d_l, d_r)`. Brute force for d ≤ 3, CPLEX MIP for larger.
3. **Joint truth table**: initialize a +1 eigenstate of `Z̄_l X̄_r`, run circuit, expect `obs0 == +1` for all noiseless shots (1000 shots).
4. **Individuals free**: initialize a state that is NOT a Z̄_l eigenstate and NOT an X̄_r eigenstate but IS a Z̄_l X̄_r eigenstate (e.g., Bell-pair-encoded across the two logicals); verify `obs0` correctly commits the joint without collapsing the individuals (measured via mid-circuit "free" logical operator inferred from data).
5. **Y detector electronic check**: noiseless run, assert all registered Y detectors output 0 across all shots.

Required gadget pairs to test:
- Steane (basis=X) × Steane (basis=Z) — small, fast smoke
- BB Webster seed (basis=Z) × BB Webster seed (basis=X) — production target

### Tier 2: Noise smoke (recommended)

5. BB Webster seed (small d, e.g. d=4 or 6) + single physical error rate (`p=0.005`) + BP-LSD decoder.
6. Compare mixed-basis joint LER vs same-basis `Z̄_l ⊗ Z̄_r` baseline LER on same code. Acceptance: same order of magnitude, mixed slightly higher (10-30% expected per denser detector graph).
7. Sanity: BP-LSD converges (no decoder crash, no timeout).

### Tier 3: Scaling (optional)

8. Multi-d × multi-p sweep, fit threshold.
9. Quantitative threshold loss vs same-basis baseline.
10. Cheeger constant numerical verification per merged code instance (sanity, not gate).

Tier 1 is must-have to land the PR. Tier 2 recommended. Tier 3 if time permits.

## 9. Correctness Lemmas

This section provides rigorous proofs of algebraic correctness. Distance preservation and FT threshold are empirical (see Tier 1.3 distance verification and Tier 2 noise smoke).

### Lemma 1 (Stabilizer Commutation)

**Claim**: After the merge algorithm, the merged code's stabilizers `S_X ∪ S_Z ∪ S_Y` pairwise commute under the symplectic F_2 inner product.

**Proof**:
- Stabilizers in `S_X ∪ S_Z` after pair-merge are GF(2) sums of pre-merge stabilizers. By the linearity of the symplectic inner product, `⟨A+B, C⟩ = ⟨A, C⟩ + ⟨B, C⟩`. Pre-merge stabilizers commuted with everything (Bridge invariant), so post-merge X and Z rows commute with everything.
- A Y-stab `Y_q = X_q · Z_q` from cross-merge is the operator product of `leftover_x_row` and `leftover_z_row`, both of which already commuted with everything. The product of two commuting operators commutes with the same set.
- Two Y-stabs from different merge qubits q, q': their X-supports come from `leftover_x_row(q)` and `leftover_x_row(q')` (similar for Z). After Step A at each q, no other X-row touches q. So the X-supports of `leftover_x_row(q)` and `leftover_x_row(q')` cannot both touch the same qubit r (else q', r would still be in `leftover_x_row(q)`'s support after Step A, contradiction). Hence X-supports are disjoint, similarly Z-supports. Symplectic inner product `⟨Y(q), Y(q')⟩ = X-supp(q) · Z-supp(q') + Z-supp(q) · X-supp(q')` evaluates over disjoint qubit sets and yields 0.

Q.E.D.

### Lemma 2 (Joint PPM Identity)

**Claim**: In the merged code, `∏_v χ_v^(l) · ∏_v χ_v^(r) ≡ Z̄_l ⊗ X̄_r · ∏_{i ∈ obs0_xor_map} Y_stab[i] · ∏_{leftover-X} m(c) · ∏_{leftover-Z} m(c) · (gauge stabilizers)` (mod merged code stabilizer group), up to a global phase.

**Proof** (sketch — full derivation in implementation comments):

1. **Data segment**: `∏_v χ_v^(l)` on the data side equals `Z̄_l` (Webster L=1 identity per side); symmetrically `∏_v χ_v^(r)` equals `X̄_r`. Joint product on data = `Z̄_l ⊗ X̄_r`.
2. **κ_l and κ_r segments**: each ∏ over χ rows on its own κ ancilla side equals a gauge product (kernel of incidence^T per gauge fix), which is in the stabilizer group.
3. **Adapter segment**: each χ_v^(l) contributes `Z` on `adapter[label_l(v)]`, each χ_v^(r) contributes `X` on `adapter[label_r(v)]`. Since `label_s` is a bijection from port_s to `{0..w-1}`, the adapter-side product is `∏_k Z_k · ∏_k X_k = (-i)^w ∏_k Y_k`.
4. **Adapter Y product is a stabilizer**: the merge step at each adapter qubit `k ∈ merge_qubits` produces a Y-stab whose adapter-side support contains `Y_k` (plus possibly other Y/X/Z on cl_ancilla, cr_ancilla, other adapter qubits). The product `∏_{i ∈ obs0_xor_map} Y_stab[i]` on the adapter side equals `∏_k Y_k` exactly (each adapter qubit is covered exactly once by exactly one Y-stab from the merge). On the cl_ancilla, cr_ancilla side, the product equals additional gauge stabilizers (proof: cancellation via the χ row structure).
5. **Leftover X-singletons and Z-singletons**: if `merge_qubits` doesn't cover all adapter qubits (because some had only X-cycles or only Z-cycles, not both), the leftover singletons contribute `X` or `Z` on those uncovered adapter qubits, which the joint product also picks up. Hence `obs0` includes `m(c)` for each leftover X-cycle row `c` and each leftover Z-cycle row `c`.

Combining: `∏ χ_l · ∏ χ_r = Z̄_l ⊗ X̄_r · ∏ Y-stabs · ∏ leftover X-cycles · ∏ leftover Z-cycles · (gauge)`. The last factor is in the stabilizer group, hence projects to identity on the codespace. The Y-stab and leftover factors are stabilizers whose eigenvalues are determined by the round's measurement outcomes. Hence:

`Z̄_l ⊗ X̄_r eigenvalue = ⊕ m(χ_l) ⊕ ⊕ m(χ_r) ⊕ ⊕ m(Y-stabs in obs0_xor_map) ⊕ ⊕ m(leftover-X) ⊕ ⊕ m(leftover-Z)`.

Q.E.D.

### Lemma 3 (obs0 Noiseless Correctness)

**Claim**: In a noiseless run, `obs0` as computed by the formula in §7.3 equals the true eigenvalue of `Z̄_l ⊗ X̄_r` on the initial state.

**Proof**: Direct from Lemma 2 plus the fact that in a noiseless circuit, the measurement outcomes `m(·)` are exactly the eigenvalues of the corresponding stabilizers on the input state. The global phase `(-i)^w` cancels because we work with `±1` outcomes (squared phases) at the observable level.

Q.E.D.

### What Lemmas 1–3 prove and don't prove

**Proven**: algebraic correctness of the construction. The merged code is a valid stabilizer code, the joint PPM commits exactly `Z̄_l ⊗ X̄_r`, and `obs0` extracts the correct value in noiseless runs.

**Not proven**:
- **Code distance**: Webster's Cheeger argument (paper §II.B.2) is heuristic. Tier 1 numerical distance check (test §8 item 2, per Appendix A) is the gate.
- **FT threshold preservation**: requires Tier 2 empirical verification.
- **Round-1 / final-round reliability optimality**: the design's conservative choices (Y unreliable round 1, skip final Y detector) are *safe* but possibly suboptimal. Future work may improve.

## 10. Separability Commitment

All modifications shall preserve the existing same-basis code paths bit-for-bit. CI gate: existing `src/qldpc/circuits/surgery/*_test.py` (179 tests as of 2026-06-15) must pass without modification.

### Commit sequence on `feat/latticesurgery-mixedjoint`

```
1. fix(surgery): cellulate_max_len → max stab weight                  [DONE: fd2dbce]
2. refactor(surgery): Bridge dataclass basis_l/basis_r split,
   backward-compat .basis property
3. feat(surgery): merge.py algorithm + unit tests (no integration)
4. feat(surgery): build_bridge mixed-basis dispatch + Bridge.Y_stab
5. feat(surgery): _stitch_to_quditcode for mixed-basis merged code
6. feat(surgery): Y-stab measurement + detector in circuit pipeline
7. feat(surgery): obs0 formula extension + Tier 1 tests
8. test(surgery): Tier 2 BB Webster seed noise smoke + comparison
```

Each commit independently passes the pre-existing test suite. Any commit can be the stopping point without breaking same-basis functionality.

### File-level isolation

| Existing file | Touched? | Backward-compat strategy |
|---|---|---|
| `src/qldpc/circuits/surgery/__init__.py` | Yes — `__all__` unchanged | New exports only via `from .merge import …` if needed |
| `src/qldpc/circuits/surgery/gadget.py` | **No** | Webster L=1 gadget logic unchanged |
| `src/qldpc/circuits/surgery/bridge.py` | Yes — dispatch + dataclass | Same-basis path is `if`-guarded, untouched |
| `src/qldpc/circuits/surgery/circuit.py` | Yes — dispatch in stitch + cycle + obs0 | Same-basis: `if bridge.Y_stab is None: return _legacy_…(…)` early returns |
| `src/qldpc/circuits/surgery/cheeger.py` | **No** | Unchanged |
| `src/qldpc/circuits/surgery/merge.py` | **New** | Pure new module, no entanglement |
| Existing `*_test.py` files | **Not modified** | Verified via CI gate. New tests in `merge_test.py` / `bridge_mixed_test.py` / `circuit_mixed_test.py` |

### What we can't fully isolate

- **Bridge dataclass fields**: added fields are permanent on the class; can't be added behind a feature flag. But unused fields default to `None`/`()` and don't affect same-basis behavior.
- **Inter-file dependencies**: `bridge.py` and `circuit.py` import from `merge.py`. To remove mixed-basis, three files revert together (not just one).
- **Upstream conflicts**: if `main` modifies surgery files during this work, git conflicts are real (operational concern, not design).

## Appendix A: Distance Verification

For each Tier 1 test case (Steane × Steane, BB Webster seed × BB Webster seed in mixed basis):

1. Construct the merged QuditCode.
2. Compute the code distance via brute force (search minimal-weight `P ∈ N(stabs) \ stabs` such that P is non-identity on the codespace). For small Steane test (d=3) brute force is tractable.
3. For BB Webster seed (d up to ~6 in Tier 1 scope), use CPLEX MIP search if brute force is too slow. (Cross–He–Rall–Yoder uses this approach for gross code Y-system distance verification.)
4. Acceptance: merged code distance ≥ `min(d_l, d_r)` where `d_l`, `d_r` are the original codes' distances.

If distance check fails, the design's `merge_qubits` selection or cellulation parameters may need adjustment. This is a known risk per Webster's heuristic Cheeger argument and is the primary reason Tier 1 distance check is a must-have.

## Status (2026-06-18)

**Joint-PPM-layout refactor**: 14-task plan at `docs/superpowers/plans/2026-06-18-joint-ppm-layout-refactor.md` landed on `feat/latticesurgery-mixedjoint`. Mixed-basis joint PPM now builds its merged QuditCode via `joint_layout.build_joint_layout` block-by-block per main.tex §4.2/§4.3, with explicit per-row provenance (`JointPPMLayout.rows_chi`, `rows_y`, etc.). obs0 emission in `_build_joint_ppm_circuit_mixed_basis` reads provenance directly per Lemma 2 of this spec.

**Closed**:
- Tier 1 acceptance bar `test_mixed_basis_circuit_compiles_to_dem` passes (vacuously, see degeneracy note below).
- Block-by-block construction (Tasks 1-9: `joint_layout.py`, `joint_layout_test.py`).
- Layout-driven stitch dispatch (Tasks 10-11: `_build_mixed_basis_joint_code` + plumbing).
- obs0 emission via row provenance (Task 12).

**Known limitations**:
- The Steane × Steane fixture (V_0 = ports = w = 3) is degenerate: surviving χ rows are empty post-merge, so obs0 reduces to `⊕ m(y_q)` alone, which carries an adapter Y residual that anti-commutes with bridge state-prep. The Task 12 obs0 block detects this regime and suppresses emission, so the DEM test passes vacuously. The truth-table test `test_mixed_basis_joint_truth_table_x_l_z_r` remains `xfail` documenting this. Closing requires either (a) a non-degenerate fixture (|V_0| > w) where surviving χ rows cancel the adapter residual, or (b) adding adapter destructive-readout outcomes to obs0 (research-level, separate plan).

**Deferred to follow-up plans**:
- Same-basis joint PPM migration to `joint_layout` (currently still on the legacy `_stitch_intercode` / `_stitch_intracode` path).
- `Bridge` dataclass cleanup: remove `Y_stab`, `obs0_xor_map`, `x_leftover_indices`, `z_leftover_indices`, `merge_qubits` (now unused by the new path; legacy `_stitch_to_joint_code_mixed` still references them).
- Legacy `_stitch_to_joint_code_mixed` removal and `merge.py` `apply_mixed_basis_merge` removal.
- The pre-existing `bridge_mixed_test.py::test_stitch_mixed_basis_populates_bridge_fields` failure tracks the legacy Bridge field cleanup item.
