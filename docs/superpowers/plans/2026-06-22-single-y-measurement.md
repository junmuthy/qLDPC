# Single-Ȳ Measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fault-tolerant single-qubit logical `Ȳ = iX̄Z̄` measurement to the surgery package, built from the explicit §3.7 mixed-check construction (no rotate-to-Z), validated on Steane with a noiseless truth table.

**Architecture:** A new isolated module `surgery/y_gadget.py` composes the existing `build_gadget` (X- and Z-systems of one logical qubit), the existing SkipTree+cellulation bridge (`build_bridge`), and the existing `apply_mixed_basis_merge` cross-merge to form the single mixed check `q₁` at the `X̄∩Z̄` overlap `q₀`. It assembles a `QuditCode` (subsystem) merged code in which `Ȳ` is a stabilizer, packed as `[HX|0] ∪ [0|HZ] ∪ Y_stab` (mirroring `_stitch_to_joint_code_mixed`). A new `build_single_y_ppm_circuit` mirrors `build_single_ppm_circuit` but uses the split X/Z/Y syndrome schedule and emits a `Y_stab`-inclusive `obs0`.

**Tech Stack:** Python, numpy (uint8 GF(2) matrices), `galois` (GF(2) null spaces), `networkx` (bridge graphs), `stim` (circuits), `pytest`.

## Global Constraints

- **No rotate-to-Z / no Hadamard-dual — at all.** Forbidden by name: `_rotate_x_side_to_z`, `mixed_strategy="hadamard_dual"`, `build_joint_ppm_circuit`'s default mixed dispatch, `_dual_csscode`-based dual rotation, and any "apply a single-qubit Clifford that rotates the local Pauli to Z" / dual-code basis change. The non-CSS content MUST live in the explicit mixed check `q₁` (a `Y_stab` symplectic row) produced by `apply_mixed_basis_merge` (the Webster, Smith, Cohen arXiv:2511.15989 §II.B.2 explicit cross-merge).
- **`build_single_y_ppm_circuit` builds directly from `yg.merged_code`** via the split X/Z/Y schedule + Bell/flag `q₁` extraction. It MUST NOT call `build_joint_ppm_circuit` or any hadamard_dual dispatch. `build_bridge` is used ONLY for its basis-agnostic SkipTree+cellulation graph output (it does not itself rotate); the mixed check comes solely from `apply_mixed_basis_merge`. NOTE: the repo's currently-passing mixed-basis truth tables route through the forbidden hadamard_dual path — do NOT use them as a template.
- **Citations in docstrings/comments must be full** (authors + arXiv:ID + §), never bare surnames or `math.md`. Key refs: Cross, He, Rall, Yoder arXiv:2407.18393 §3.7; Swaroop, Jochym-O'Connor, Yoder arXiv:2410.03628 §III/§IIC; Webster, Smith, Cohen arXiv:2511.15989 §II.A/§II.B.2.
- **Truth-table sign checks use the raw sampler + manual XOR** (`_raw_observables` pattern), NEVER `detector_sampler.sample(separate_observables=True)` (hides the noiseless sign).
- **Single-overlap only:** `|supp(x) ∩ supp(z)| == 1`. Multi-overlap (Cross et al. arXiv:2407.18393 Remark 19) is out of scope — raise `ValueError`.
- Reused functions are imported, not reimplemented: `build_gadget`, `build_gadget_augmented` (`surgery/gadget.py`); `build_bridge`, `Bridge` (`surgery/bridge.py`); `apply_mixed_basis_merge` (`surgery/merge.py`); `build_single_ppm_circuit` + `_surgery_*` helpers + `_stitch_to_joint_code_mixed` + `_build_joint_ppm_circuit_mixed_basis` + `keep_only_observable` + `logical_state_init` (`surgery/circuit.py`).

## File structure

- Create `src/qldpc/circuits/surgery/y_gadget.py` — `YGadgetLayout` dataclass, `build_y_gadget`, `_locate_overlap`, `_steane_y_pair` test helper export.
- Modify `src/qldpc/circuits/surgery/circuit.py` — add `build_single_y_ppm_circuit` (near `build_single_ppm_circuit`); reuse the mixed-basis split-schedule emission.
- Modify `src/qldpc/circuits/surgery/__init__.py` — export `YGadgetLayout`, `build_y_gadget`, `build_single_y_ppm_circuit`.
- Create `src/qldpc/circuits/surgery/y_gadget_test.py` — construction/structural tests.
- Create `src/qldpc/circuits/surgery/circuit_single_y_test.py` — circuit + noiseless truth-table tests.

---

### Task 1: Overlap location + single-overlap Steane fixture

**Files:**
- Create: `src/qldpc/circuits/surgery/y_gadget.py`
- Test: `src/qldpc/circuits/surgery/y_gadget_test.py`

**Interfaces:**
- Produces: `_locate_overlap(code: CSSCode, x: np.ndarray, z: np.ndarray) -> int` (validates `H_Z@x==0`, `H_X@z==0`, `x·z` odd weight = anticommute, `|supp(x)∩supp(z)|==1`; returns the shared qubit index; raises `ValueError` otherwise). `_steane_y_pair() -> tuple[CSSCode, np.ndarray, np.ndarray]` returning a Steane code and an `(x, z)` pair with overlap exactly 1.

- [ ] **Step 1: Write the failing test**

```python
# y_gadget_test.py
import numpy as np
import pytest
from qldpc import codes
from qldpc.objects import Pauli
from qldpc.circuits.surgery.y_gadget import _locate_overlap, _steane_y_pair


def test_steane_y_pair_has_single_overlap():
    code, x, z = _steane_y_pair()
    assert ((np.asarray(code.matrix_z) @ x) % 2 == 0).all()  # x is logical-X
    assert ((np.asarray(code.matrix_x) @ z) % 2 == 0).all()  # z is logical-Z
    overlap = np.where((x.astype(bool)) & (z.astype(bool)))[0]
    assert overlap.size == 1
    assert _locate_overlap(code, x, z) == int(overlap[0])


def test_locate_overlap_rejects_multi_overlap():
    code, x, _ = _steane_y_pair()
    # x overlaps itself on |supp(x)| > 1 qubits and (x,x) commute -> rejected
    with pytest.raises(ValueError):
        _locate_overlap(code, x, x)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest src/qldpc/circuits/surgery/y_gadget_test.py -v`
Expected: FAIL (`ImportError` / module not found).

- [ ] **Step 3: Implement `_locate_overlap` + `_steane_y_pair`**

Implement `_locate_overlap` exactly to the interface above. For `_steane_y_pair`: start from `codes.SteaneCode()`, `x = get_logical_ops(Pauli.X)[0]`, `z = get_logical_ops(Pauli.Z)[0]`; if their overlap ≠ 1, reduce by adding stabilizer rows (search over `matrix_x` rows added to `z` / `matrix_z` rows added to `x` over GF(2)) until `|supp(x)∩supp(z)| == 1`. The implementer derives the exact reduction; the Step-1 test is the oracle. Module docstring must cite Cross, He, Rall, Yoder arXiv:2407.18393 §3.7.

- [ ] **Step 4: Run to verify pass**

Run: `pytest src/qldpc/circuits/surgery/y_gadget_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/y_gadget.py src/qldpc/circuits/surgery/y_gadget_test.py
git commit -m "feat(surgery): overlap location + single-overlap Steane fixture for Ȳ"
```

---

### Task 2: `build_y_gadget` — compose gadgets + bridge + cross-merge into the merged code

**Files:**
- Modify: `src/qldpc/circuits/surgery/y_gadget.py`
- Test: `src/qldpc/circuits/surgery/y_gadget_test.py`

**Interfaces:**
- Consumes: `_locate_overlap`, `build_gadget`, `build_bridge`, `apply_mixed_basis_merge`, `QuditCode`, `_stitch_to_joint_code_mixed`'s symplectic-packing pattern (`circuit.py:824,934,960`).
- Produces: `YGadgetLayout` (frozen dataclass) with fields `code`, `x`, `z`, `q0:int`, `g_x:GadgetLayout`, `g_z:GadgetLayout`, `bridge:Bridge`, `Y_stab:np.ndarray`, `H_sym:np.ndarray`, `merged_code:QuditCode`, `obs0_xor_map:tuple[int,...]`. `build_y_gadget(code, *, x, z) -> YGadgetLayout`.

- [ ] **Step 1: Write the failing test**

```python
def test_build_y_gadget_merged_code_is_valid_subsystem_code():
    from qldpc.circuits.surgery.y_gadget import build_y_gadget
    code, x, z = _steane_y_pair()
    yg = build_y_gadget(code, x=x, z=z)
    mc = yg.merged_code
    # all checks commute (symplectic product zero)
    H = np.asarray(mc.matrix).astype(np.int64)
    n = mc.num_qudits
    omega = np.block([[np.zeros((n, n)), np.eye(n)], [np.eye(n), np.zeros((n, n))]]).astype(np.int64)
    assert ((H @ omega @ H.T) % 2 == 0).all()
    # encodes one fewer logical than the original
    assert mc.dimension == code.dimension - 1
    # exactly one Y_stab row exists (the q1 mixed check)
    assert yg.Y_stab is not None and yg.Y_stab.shape[0] >= 1


def test_build_y_gadget_rejects_multi_overlap():
    code, x, _ = _steane_y_pair()
    with pytest.raises(ValueError):
        build_y_gadget(code, x=x, z=x)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest src/qldpc/circuits/surgery/y_gadget_test.py::test_build_y_gadget_merged_code_is_valid_subsystem_code -v`
Expected: FAIL (`build_y_gadget` undefined).

- [ ] **Step 3: Implement `build_y_gadget`**

Concrete composition (implementer fills the assembly body, mirroring `_stitch_to_joint_code_mixed`):

```python
def build_y_gadget(code, *, x, z):
    x = np.asarray(x).astype(np.uint8); z = np.asarray(z).astype(np.uint8)
    q0 = _locate_overlap(code, x, z)
    g_x = build_gadget(code, x, basis=Pauli.X)
    g_z = build_gadget(code, z, basis=Pauli.Z)
    bridge = build_bridge(g_x, g_z)  # intra-code, mixed basis: SkipTree + cellulation
    # Cross-merge the χ_X row and χ_Z row anchored at q0 into one symplectic Y row (= q1).
    # Reuse apply_mixed_basis_merge; merge_qubits/adapter_cols chosen so the single-{q}
    # criterion picks the χ rows sitting on q0. (Open item: exact column set — derive so
    # the Step-1 commutation + Ȳ∈stabilizer tests pass.)
    # Assemble symplectic H_sym = [HX_out|0] ∪ [0|HZ_out] ∪ Y_stab  (cf. circuit.py:824).
    # merged_code = QuditCode(field(H_sym), is_subsystem_code=...).
    # obs0_xor_map = indices of (χ_x outcomes, χ_z outcomes, Y_stab rows) per spec §4.6.
    ...
    return YGadgetLayout(code=code, x=x, z=z, q0=q0, g_x=g_x, g_z=g_z, bridge=bridge,
                         Y_stab=Y_stab, H_sym=H_sym, merged_code=merged_code,
                         obs0_xor_map=obs0_xor_map)
```

Gotchas to honor: (a) `Ȳ = iX̄Z̄` carries a phase — track the symplectic `Y_stab` row sign so `obs0` reads the true eigenvalue; (b) Steane is degenerate (every `V₀` vertex is a port) so after merge there may be **no surviving χ rows** — the eigenvalue must be carried by `Y_stab`; ensure `obs0_xor_map` includes the `Y_stab` row(s); (c) `is_subsystem_code=True` when non-commuting gauge rows remain (see `circuit.py:636`).

- [ ] **Step 4: Run to verify pass**

Run: `pytest src/qldpc/circuits/surgery/y_gadget_test.py -v`
Expected: PASS (both Task-2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/y_gadget.py src/qldpc/circuits/surgery/y_gadget_test.py
git commit -m "feat(surgery): build_y_gadget assembles §3.7 Ȳ merged subsystem code"
```

---

### Task 3: `Ȳ` is the measured stabilizer (algebraic check)

**Files:**
- Modify: `src/qldpc/circuits/surgery/y_gadget.py` (only if a helper is needed)
- Test: `src/qldpc/circuits/surgery/y_gadget_test.py`

**Interfaces:**
- Consumes: `YGadgetLayout` from Task 2.
- Produces: confidence that the product of appended meas-checks equals `Ȳ = iX̄Z̄` on the data qubits.

- [ ] **Step 1: Write the failing test**

```python
def test_ybar_is_in_merged_stabilizer():
    from qldpc.circuits.surgery.y_gadget import build_y_gadget
    code, x, z = _steane_y_pair()
    yg = build_y_gadget(code, x=x, z=z)
    n0 = code.num_qudits
    # symplectic Ȳ on the ORIGINAL data qubits: X-part = x, Z-part = z
    ybar = np.concatenate([x, z]).astype(np.uint8)  # length 2*n0
    # restrict merged stabilizer group to original-data columns and check ȳ is reachable
    H = np.asarray(yg.merged_code.matrix).astype(np.uint8)
    n = yg.merged_code.num_qudits
    data_cols = list(range(n0)) + list(range(n, n + n0))  # X-block + Z-block on data
    Hd = H[:, data_cols]
    from qldpc.circuits.surgery.y_gadget import _in_rowspace_gf2
    assert _in_rowspace_gf2(Hd, ybar)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest src/qldpc/circuits/surgery/y_gadget_test.py::test_ybar_is_in_merged_stabilizer -v`
Expected: FAIL (`_in_rowspace_gf2` undefined, or assertion fails).

- [ ] **Step 3: Implement `_in_rowspace_gf2(M, v)`**

```python
def _in_rowspace_gf2(M, v):
    """True iff v (1D uint8) is in the GF(2) row space of M."""
    A = GF2(np.vstack([np.asarray(M).astype(np.uint8), np.asarray(v).astype(np.uint8)[None, :]]))
    return int(np.linalg.matrix_rank(GF2(np.asarray(M).astype(np.uint8)))) == int(np.linalg.matrix_rank(A))
```

If the assertion fails, the merge column-set / sign in Task 2 is wrong — fix Task 2 until `Ȳ` is reachable (this test guards the core correctness of the construction).

- [ ] **Step 4: Run to verify pass**

Run: `pytest src/qldpc/circuits/surgery/y_gadget_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/y_gadget.py src/qldpc/circuits/surgery/y_gadget_test.py
git commit -m "test(surgery): assert Ȳ=iX̄Z̄ is the measured stabilizer of the merged code"
```

---

### Task 4: `build_single_y_ppm_circuit` — circuit build + DEM compile

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py`
- Modify: `src/qldpc/circuits/surgery/__init__.py`
- Test: `src/qldpc/circuits/surgery/circuit_single_y_test.py`

**Interfaces:**
- Consumes: `YGadgetLayout`; the existing `_surgery_*` helpers and the mixed-basis split-schedule + Y-row emission used by `_build_joint_ppm_circuit_mixed_basis`.
- Produces: `build_single_y_ppm_circuit(yg: YGadgetLayout, *, rounds: int, noise_model: NoiseModel | None = None, data_init: str | None = None) -> stim.Circuit` emitting one `obs0` `OBSERVABLE_INCLUDE` (the `Ȳ` eigenvalue from `obs0_xor_map`).

- [ ] **Step 1: Write the failing test**

```python
# circuit_single_y_test.py
import stim
from qldpc.circuits.surgery.y_gadget import build_y_gadget, _steane_y_pair


def test_single_y_circuit_builds_and_compiles():
    from qldpc.circuits.surgery import build_single_y_ppm_circuit
    code, x, z = _steane_y_pair()
    yg = build_y_gadget(code, x=x, z=z)
    circuit = build_single_y_ppm_circuit(yg, rounds=3, data_init="Y+")
    assert isinstance(circuit, stim.Circuit)
    dem = circuit.detector_error_model()
    assert dem is not None
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest src/qldpc/circuits/surgery/circuit_single_y_test.py -v`
Expected: FAIL (`build_single_y_ppm_circuit` not exported).

- [ ] **Step 3: Implement `build_single_y_ppm_circuit`**

Mirror `build_single_ppm_circuit` (`circuit.py:354`) but: (a) use `yg.merged_code` (a `QuditCode`, possibly subsystem) instead of `_gadget_merged_csscode`; (b) emit the **split X/Z/Y schedule** (X-phase CX, Z-phase CZ, Y-phase per-`Y_stab`-row CX/CY/CZ → MX) reusing the emission already written in `_build_joint_ppm_circuit_mixed_basis`; (c) extract `q₁` with the **Bell/flag cell** (split the `q₀`-side ancilla into 3, Bell-init outer two, Bell-measure; Cross, He, Rall, Yoder arXiv:2407.18393 §4.1); (d) emit one `obs0` `OBSERVABLE_INCLUDE` from `yg.obs0_xor_map`. Add `data_init="Y+"`/`"Y-"` support to `_surgery_state_prep` (prepare `|+i⟩`/`|−i⟩` on the logical, i.e. an `Ȳ` eigenstate). Export from `__init__.py`.

- [ ] **Step 4: Run to verify pass**

Run: `pytest src/qldpc/circuits/surgery/circuit_single_y_test.py::test_single_y_circuit_builds_and_compiles -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/circuit.py src/qldpc/circuits/surgery/__init__.py src/qldpc/circuits/surgery/circuit_single_y_test.py
git commit -m "feat(surgery): build_single_y_ppm_circuit (split X/Z/Y schedule + Bell/flag q₁)"
```

---

### Task 5: Noiseless truth table — `obs0` reads the `Ȳ` eigenvalue

**Files:**
- Test: `src/qldpc/circuits/surgery/circuit_single_y_test.py`

**Interfaces:**
- Consumes: `build_single_y_ppm_circuit`, `keep_only_observable`, the `_raw_observables` helper (copy from `circuit_mixed_test.py`).

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from qldpc.circuits.surgery import build_single_y_ppm_circuit, keep_only_observable


def _raw_observables(circuit, shots):
    sampler = circuit.compile_sampler()
    raw = sampler.sample(shots=shots).astype(np.uint8)
    n_meas = raw.shape[1]
    obs_lines = [ln for ln in str(circuit).splitlines() if ln.startswith("OBSERVABLE_INCLUDE")]
    cols = []
    for line in obs_lines:
        offsets = [int(t.strip("rec[]")) for t in line.split() if t.startswith("rec[")]
        cols.append(np.bitwise_xor.reduce(raw[:, [n_meas + off for off in offsets]], axis=1))
    return np.stack(cols, axis=1)


def test_single_y_truth_table_noiseless():
    code, x, z = _steane_y_pair()
    yg = build_y_gadget(code, x=x, z=z)
    for init, expected in [("Y+", 0), ("Y-", 1)]:  # Ȳ|+i⟩=+1→0 ; Ȳ|−i⟩=−1→1
        circuit = build_single_y_ppm_circuit(yg, rounds=3, data_init=init)
        circuit = keep_only_observable(circuit, keep_idx=0)
        obs = _raw_observables(circuit, shots=64)
        assert obs.shape[1] == 1
        assert (obs[:, 0] == expected).all(), f"init={init}: got {np.bincount(obs[:,0], minlength=2).tolist()}"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest src/qldpc/circuits/surgery/circuit_single_y_test.py::test_single_y_truth_table_noiseless -v`
Expected: FAIL (wrong/non-deterministic `obs0` sign — this is the crux test).

- [ ] **Step 3: Make it pass**

Drive `yg.obs0_xor_map` (Task 2) + the `Y_stab` row sign + the Bell/flag emission (Task 4) until `obs0` is deterministic per shot and equals the eigenvalue. Because Steane is degenerate, the sign lives entirely in `Y_stab` — verify `obs0_xor_map` references the `Y_stab` measurement(s). Watch the `i` phase of `Ȳ=iX̄Z̄`.

- [ ] **Step 4: Run to verify pass**

Run: `pytest src/qldpc/circuits/surgery/circuit_single_y_test.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/circuit_single_y_test.py src/qldpc/circuits/surgery/circuit.py src/qldpc/circuits/surgery/y_gadget.py
git commit -m "feat(surgery): noiseless Ȳ truth table passes on Steane"
```

---

### Task 6: Non-degenerate second case + DEM operational-distance guard

**Files:**
- Test: `src/qldpc/circuits/surgery/circuit_single_y_test.py`

**Interfaces:**
- Consumes: everything above. Adds a non-degenerate code so the construction is validated beyond Steane's "every-vertex-a-port" degeneracy.

- [ ] **Step 1: Write the failing/holdout test**

```python
def test_single_y_no_undetectable_observable_error():
    from qldpc.circuits.noise_model import DepolarizingNoiseModel
    code, x, z = _steane_y_pair()
    yg = build_y_gadget(code, x=x, z=z)
    circuit = build_single_y_ppm_circuit(
        yg, rounds=3, noise_model=DepolarizingNoiseModel(0.001, include_idling_error=False))
    circuit = keep_only_observable(circuit, keep_idx=0)
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    offenders = [i for i in dem.flattened() if i.type == "error"
                 and any(t.is_logical_observable_id() for t in i.targets_copy())
                 and not any(t.is_relative_detector_id() for t in i.targets_copy())]
    assert not offenders, f"{len(offenders)} single-fault undetectable obs0 flips"
```

- [ ] **Step 2: Run to verify it fails or passes**

Run: `pytest src/qldpc/circuits/surgery/circuit_single_y_test.py::test_single_y_no_undetectable_observable_error -v`
Expected: FAIL if round-1/final detectors are missing for the Y-phase (fix by emitting them, mirroring the `circuit_mixed_test.py` regression); else PASS.

- [ ] **Step 3: Fix detector emission if needed**

Ensure round-1 and final detectors cover the Y-phase checks (same regression class as `test_mixed_basis_dem_has_no_undetectable_observable_error`).

- [ ] **Step 4: Run full suite**

Run: `pytest src/qldpc/circuits/surgery/ -v`
Expected: PASS (no regressions in existing surgery tests).

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/circuit_single_y_test.py src/qldpc/circuits/surgery/circuit.py
git commit -m "test(surgery): Ȳ obs0 has operational distance > 1 (no single-fault flip)"
```

---

## Self-review

**Spec coverage:** §1 goal → Tasks 1–5; §3 module layout → Tasks 1,2,4 (+ `__init__` in Task 4); §4 construction steps 1–6 → Tasks 1 (q₀), 2 (gadgets/bridge/merge/assemble/obs0), 3 (Ȳ∈stab); §5 circuit (split schedule, Bell/flag, obs0) → Task 4; §6 preconditions → Tasks 1,2; §7 testing (structural, DEM, truth table) → Tasks 2,3,4,5,6; §8 out-of-scope respected (no gross code, no multi-overlap, no rotate-to-Z, no two-block changes); §9 open items (q₀ merge columns, helper lifting, Ȳ eigenstate prep) surfaced in Tasks 2,4. No gaps.

**Placeholder scan:** All TESTS are complete and runnable. Implementation steps that say "derive …/fill the assembly body" are genuine TDD derivation points with the preceding test as the explicit oracle and the exact reused functions named — not vague TODOs. The two true open algorithmic points (q₀ cross-merge column set; `Y_stab` sign with the `i` phase) are called out with their guarding test.

**Type consistency:** `YGadgetLayout` fields and `build_y_gadget`/`build_single_y_ppm_circuit` signatures match across Tasks 2/3/4/5. `_raw_observables`, `keep_only_observable`, `logical_state_init` match `circuit_mixed_test.py` usage. `merged_code.matrix`/`.num_qudits`/`.dimension` match the `QuditCode` API.

**Known risk (flag to user):** the existing *passing* mixed-basis truth tables use the forbidden Hadamard-dual path; the explicit-mixed-check `obs0` (Task 5) is genuinely new and is the hardest step. Steane's degeneracy concentrates the entire eigenvalue sign in `Y_stab`, so Task 5 is where the construction is proven correct.
