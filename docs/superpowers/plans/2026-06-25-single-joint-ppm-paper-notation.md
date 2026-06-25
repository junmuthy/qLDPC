# Single & joint PPM paper-notation refactor — Implementation Plan

**Status:** implemented (2026-06-25)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename §2/§3 surgery internals (`gadget.py`, `circuit.py`, `bridge.py`) so the code reads against the parity-check-matrix formulas — `π_{V₀}`, `π_{C₀}`, `H_X'`, `f_X'`, `f_Z`, `H_Z'`, `Q'`, `S_X'` — with **zero behavioral change**.

**Architecture:** Pure rename + re-express the gadget construction in explicit `π`-projection form. Both §2 and §3 are already in paper block-row order, so no matrix reorder. A byte-identical golden-master test (Task 1) pins every merged matrix and circuit; every later task must keep it green.

**Tech Stack:** Python, numpy, galois (GF(2)), stim, pytest.

## Global Constraints

- **Byte-identical output.** Every `HX_merged`/`HZ_merged` (single + joint inter + joint intra) and every emitted circuit MUST be unchanged. The golden test (Task 1) is authoritative. **If a golden hash changes, that is a BUG in the rename — revert and fix; NEVER update the golden hashes.**
- **Citations:** cite primary papers fully (`Authors arXiv:ID §`), NEVER the internal `main.tex`. Verified strings: §2 gadget = `Webster, Smith, Cohen arXiv:2511.15989 §II.A` + `Cain et al. arXiv:2603.28627 §B.1`; §3 joint = `Swaroop et al. (Swaroop, Jochym-O'Connor, Yoder) arXiv:2410.03628 §III` + `Cross et al. arXiv:2407.18393 Thm 6`; boost = `Williamson & Yoder arXiv:2410.02213`. No bare surnames; no `math.md`.
- **Naming:** no `χ` (→ `S_X'`/`S_prime`), no `κ`/`ancilla_qubits` (→ `Q'`/`Q_prime`). §4 (`y_gadget.py`/`y_circuit.py`) keeps its own `∂₁/∂₀` internal names; only its *reads* of the renamed field are touched.
- **No LER / `sinter` sampling tests.** Deterministic checks only.
- Run the suite with: `cd /Users/tgzhou/Project/qLDPC && python -m pytest src/qldpc/circuits/surgery/ -q`

---

### Task 1: Golden-master characterization test (the safety net)

Write a test that pins the current matrices + circuits by sha256. It passes against the **current** (pre-refactor) code; it stays green through every later task. This is the inverse-TDD characterization step for a no-behavior-change refactor.

**Files:**
- Create: `src/qldpc/circuits/surgery/refactor_snapshot_test.py`

**Interfaces:**
- Consumes (public API, unchanged by this refactor): `build_gadget(code, x, *, basis)`, `build_bridge(g_l, g_r)`, `_stitch_to_joint_csscode(g_l, g_r, bridge)`, `build_single_ppm_circuit(g, *, rounds, noise_model)`, `build_joint_ppm_circuit(g_l, g_r, bridge, *, rounds, noise_model)`, fields `.HX_merged`/`.HZ_merged`, fixtures from `_webster_fixture`.
- Produces: the golden test module — later tasks rely on it as the regression gate.

- [ ] **Step 1: Write the golden test**

```python
"""Byte-identical golden-master for the §2/§3 paper-notation refactor.

These sha256 hashes pin the EXACT merged check matrices and emitted circuits of the
single- and joint-PPM construction. The refactor (docs/superpowers/plans/
2026-06-25-single-joint-ppm-paper-notation.md) is a pure rename + π-form re-expression
with NO behavioral change, so every hash below MUST stay identical.

If a hash changes, the refactor altered a matrix or circuit — that is a BUG.
Do NOT update the hashes to make this pass.
"""

from __future__ import annotations

import hashlib

import numpy as np

from qldpc import codes
from qldpc.objects import Pauli

from ._webster_fixture import (
    _webster_x_bar_operator,
    build_generalised_bicycle_code,
    load_webster_seed_set,
)


def _h(a) -> str:
    arr = np.ascontiguousarray(np.asarray(a).astype(np.uint8))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _hs(s) -> str:
    return hashlib.sha256(str(s).encode()).hexdigest()


GOLDEN = {
    "single_steane_X_HX": "5e071db6d229df9861a7e0221dcb74ed7ec9e3effcad43cbc84e82d1844a6f64",
    "single_steane_X_HZ": "0483c2c21a1593e299dbf1038aad2c1426de8e193fda18a3eefdda30531e80d8",
    "single_steane_Z_HX": "c68a2b6cc52492445c1c1544b3d3fb1774b49bd59b22e8937cb6d21896ad8713",
    "single_steane_Z_HZ": "0a149bfde7f21a3049b9cb6b2540d105a6c45ec4dd91ea5cafe488163612fbb2",
    "single_gb_X_HX": "a83ba1b89872bab854f0a225b97ee1338a88ee8fec1f829fd49144d460c16892",
    "single_gb_X_HZ": "35d3a73c3ff7b6ab49b5ae824a7cb77f806122bc51a51bb08d6e5811b95ac78c",
    "joint_inter_X_HX": "d2a9209574889d3a64ac62178afc7340911a69076e986623d2cea49c9a8cc639",
    "joint_inter_X_HZ": "90d71ef17ea8620eb8617dfd85618cea6f2780af7253fff9345e0a26eb8308c2",
    "joint_intra_X_HX": "aeeba0cbc1577efc3dda18c275d67a361fc228311baf335f27decbdfa923b5ed",
    "joint_intra_X_HZ": "f70ae66ab3f20920b4d69b249f99360829ee01da27fb5920844de6ef1665d0cb",
    "circ_single_X": "2a1f149a0958f0e36a33ccf5395e41539330048a6a4a437d681737acc4a6d19b",
    "circ_joint_intra": "ff00e74186550ccdc8d747693b15e952a9c6e335c9cb890693becc7f6a61d598",
}


def _steane_x():
    code = codes.SteaneCode()
    return code, np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)


def test_single_steane_basis_x_matrices_byte_identical() -> None:
    from qldpc.circuits.surgery.gadget import build_gadget

    code, x = _steane_x()
    g = build_gadget(code, x, basis=Pauli.X)
    assert _h(g.HX_merged) == GOLDEN["single_steane_X_HX"]
    assert _h(g.HZ_merged) == GOLDEN["single_steane_X_HZ"]


def test_single_steane_basis_z_matrices_byte_identical() -> None:
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    assert _h(g.HX_merged) == GOLDEN["single_steane_Z_HX"]
    assert _h(g.HZ_merged) == GOLDEN["single_steane_Z_HZ"]


def test_single_gb_basis_x_matrices_byte_identical() -> None:
    """Generalised-bicycle code: nontrivial incidence exercises the π-form construction."""
    from qldpc.circuits.surgery.gadget import build_gadget

    data = load_webster_seed_set(0)
    gb = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x = _webster_x_bar_operator(data, "X_bar_1", "X").astype(np.uint8)
    g = build_gadget(gb, x, basis=Pauli.X)
    assert _h(g.HX_merged) == GOLDEN["single_gb_X_HX"]
    assert _h(g.HZ_merged) == GOLDEN["single_gb_X_HZ"]


def test_joint_intercode_basis_x_matrices_byte_identical() -> None:
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_csscode
    from qldpc.circuits.surgery.gadget import build_gadget

    _, x = _steane_x()
    g_l = build_gadget(codes.SteaneCode(), x, basis=Pauli.X)
    g_r = build_gadget(codes.SteaneCode(), x, basis=Pauli.X)
    merged = _stitch_to_joint_csscode(g_l, g_r, build_bridge(g_l, g_r))
    assert _h(merged.matrix_x) == GOLDEN["joint_inter_X_HX"]
    assert _h(merged.matrix_z) == GOLDEN["joint_inter_X_HZ"]


def test_joint_intracode_basis_x_matrices_byte_identical() -> None:
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import _stitch_to_joint_csscode
    from qldpc.circuits.surgery.gadget import build_gadget

    code, x = _steane_x()
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(code, x, basis=Pauli.X)
    merged = _stitch_to_joint_csscode(g_l, g_r, build_bridge(g_l, g_r))
    assert _h(merged.matrix_x) == GOLDEN["joint_intra_X_HX"]
    assert _h(merged.matrix_z) == GOLDEN["joint_intra_X_HZ"]


def test_single_circuit_text_byte_identical() -> None:
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

    code, x = _steane_x()
    g = build_gadget(code, x, basis=Pauli.X)
    circuit = build_single_ppm_circuit(g, rounds=2, noise_model=None)
    assert _hs(circuit) == GOLDEN["circ_single_X"]


def test_joint_circuit_text_byte_identical() -> None:
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit
    from qldpc.circuits.surgery.gadget import build_gadget

    code, x = _steane_x()
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(code, x, basis=Pauli.X)
    circuit, _ = build_joint_ppm_circuit(g_l, g_r, build_bridge(g_l, g_r), rounds=2, noise_model=None)
    assert _hs(circuit) == GOLDEN["circ_joint_intra"]
```

- [ ] **Step 2: Run the golden test against current code — verify it PASSES**

Run: `cd /Users/tgzhou/Project/qLDPC && python -m pytest src/qldpc/circuits/surgery/refactor_snapshot_test.py -q`
Expected: 7 passed. (If any fail now, the embedded hash is stale — recompute from current code before proceeding; do NOT start the refactor against a red baseline.)

- [ ] **Step 3: Commit**

```bash
git add src/qldpc/circuits/surgery/refactor_snapshot_test.py
git commit -m "test(surgery): golden-master pinning §2/§3 merged matrices + circuits"
```

---

### Task 2: `gadget.py` — `_projection` helper, π-form construction, local renames

Introduce `_projection` (`π_S`), re-express the initial `incidence`/`f_Z` construction in explicit `π`-products, rename the non-paper locals (`incidence_tilde→f_Z`; name `pi_V0`/`pi_C0`/`H_X_prime`/`f_X_prime`/`S_X_prime`), and rewrite docstrings to cite primary papers. **No dataclass field rename in this task.**

**Files:**
- Modify: `src/qldpc/circuits/surgery/gadget.py`
- Modify: `src/qldpc/circuits/surgery/gadget_test.py` (add `_projection` unit tests)

**Interfaces:**
- Produces: `_projection(indices: Sequence[int], N: int) -> np.ndarray` returning a `(len(indices), N)` uint8 matrix; row `i` is `e_{indices[i]}` if `0 <= indices[i] < N`, else an all-zero row (sentinel handling for boost-added `Q'`).
- Consumes: nothing new.

- [ ] **Step 1: Write failing unit tests for `_projection`**

Add to `src/qldpc/circuits/surgery/gadget_test.py`:

```python
def test_projection_basic_selection() -> None:
    from qldpc.circuits.surgery.gadget import _projection

    pi = _projection((0, 2), 4)
    assert pi.dtype == np.uint8
    assert pi.shape == (2, 4)
    assert np.array_equal(pi, np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.uint8))


def test_projection_identity_pi_M_piT_is_submatrix() -> None:
    """π_S M π_T^T == M[S, T] (numpy-style index), the helper's defining identity."""
    from qldpc.circuits.surgery.gadget import _projection

    rng = np.random.default_rng(0)
    M = rng.integers(0, 2, size=(5, 6), dtype=np.uint8)
    S, T = (1, 3), (0, 2, 5)
    pi_S, pi_T = _projection(S, 5), _projection(T, 6)
    lhs = (pi_S @ M @ pi_T.T) % 2
    assert np.array_equal(lhs, M[np.ix_(S, T)])


def test_projection_empty_indices() -> None:
    from qldpc.circuits.surgery.gadget import _projection

    assert _projection((), 4).shape == (0, 4)


def test_projection_negative_sentinel_is_zero_row() -> None:
    """Sentinel index (-1, from boost-added Q' with no backing check) → all-zero row."""
    from qldpc.circuits.surgery.gadget import _projection

    pi = _projection((0, -1, 2), 3)
    assert np.array_equal(pi, np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.uint8))
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest src/qldpc/circuits/surgery/gadget_test.py -k projection -q`
Expected: FAIL — `ImportError: cannot import name '_projection'`.

- [ ] **Step 3: Implement `_projection`**

Add to `src/qldpc/circuits/surgery/gadget.py` (after the `GF2 = galois.GF(2)` line):

```python
def _projection(indices, N: int) -> np.ndarray:
    """π_S ∈ F₂^{|S|×N}: row i is the unit vector e_{indices[i]}.

    (π_S)_{i,j} = δ_{j, indices[i]}, so π_S M π_T^T = M[S, T] (numpy-style index).
    Entries outside [0, N) (e.g. the -1 sentinels build_gadget_augmented uses for
    boost-added Q' qubits with no backing check) give an all-zero row.
    """
    pi = np.zeros((len(indices), N), dtype=np.uint8)
    for i, s in enumerate(indices):
        if 0 <= s < N:
            pi[i, s] = 1
    return pi
```

- [ ] **Step 4: Run to verify the `_projection` tests pass**

Run: `python -m pytest src/qldpc/circuits/surgery/gadget_test.py -k projection -q`
Expected: 4 passed.

- [ ] **Step 5: Re-express `_step1_restriction` in π-form (byte-identical)**

Replace the `incidence = ...` computation in `_step1_restriction` so the formula `H_X' = π_{V₀}H_Z^Tπ_{C₀}^T` is visible. New body (keep the signature, the `support`/`data_checks` computation, and the validation above it):

```python
    # H_X' = π_{V₀} H_Z^T π_{C₀}^T  (Webster, Smith, Cohen arXiv:2511.15989 §II.A;
    # Cain et al. arXiv:2603.28627 §B.1).  The stored `incidence` is its transpose —
    # the |C₀|×|V₀| vertex-edge incidence (§4 y_gadget.py calls this ∂₁ˣ).
    n = code.num_qudits
    pi_V0 = _projection(support, n)                  # π_{V₀} = f_X' ∈ F₂^{|V₀|×n}
    pi_C0 = _projection(data_checks, H_complement.shape[0])   # π_{C₀} ∈ F₂^{|C₀|×m_comp}
    H_X_prime = (pi_V0 @ H_complement.T @ pi_C0.T) % 2        # |V₀|×|C₀|
    incidence = H_X_prime.T.astype(np.uint8)         # |C₀|×|V₀|
    return support, data_checks, incidence
```

(Delete the old `incidence = H_complement[np.ix_(...)] if ... else np.zeros(...)` line and its trailing `.astype` in the return.)

- [ ] **Step 6: Rename `incidence_tilde→f_Z` (π-form) in `_step3_assemble`**

In `_step3_assemble`, replace the `incidence_tilde` block (the `if basis is Pauli.X: incidence_tilde = np.zeros(...)` ... `for k, j in enumerate(data_checks): ... incidence_tilde[j, k] = 1` loop) with:

```python
    # f_Z = π_{C₀}^T : extends the original Z-checks (basis=X) onto the new Q' ancillas.
    # _projection's sentinel rule zeroes the columns of boost-added Q' (data_checks == -1).
    m_comp = mZ if basis is Pauli.X else mX
    f_Z = _projection(data_checks, m_comp).T.astype(np.uint8)   # (m_comp, |C₀|)
```

Then in the `np.block(...)` assemblies, replace `incidence_tilde` with `f_Z` (both the `basis is Pauli.X` and `else` branches).

- [ ] **Step 7: Rename the `S_X'` block locals in `_assemble_HX_L1`**

In `_assemble_HX_L1`, make the bottom block read as `S_X' = [f_X' | H_X']`:

```python
    mX, n = HX_data.shape
    n_v0, n_c0 = int(incidence.shape[1]), int(incidence.shape[0])
    n_merged = n + n_c0
    top = np.hstack([HX_data, np.zeros((mX, n_c0), dtype=np.uint8)]).astype(np.uint8)
    # S_X' rows = [f_X' | H_X'] : f_X' = π_{V₀} on data, H_X' = incidence.T on Q'.
    f_X_prime = np.zeros((n_v0, n), dtype=np.uint8)
    f_X_prime[np.arange(n_v0), np.asarray(support_indices)] = 1
    H_X_prime = incidence.T.astype(np.uint8)
    S_X_prime = np.hstack([f_X_prime, H_X_prime]).astype(np.uint8)
    return np.vstack([top, S_X_prime]).astype(np.uint8)
```

- [ ] **Step 8: Rewrite docstrings (citations + symbol mapping), drop "Cain mapping" bare-surname lines**

In the module docstring and the `_step1/2/3`, `_assemble_HX_L1`, `build_gadget`, `build_gadget_augmented`, `_projection`, and `GadgetLayout` docstrings: replace each `Cain mapping: X → Y` line and the `Webster ... step N` headers with full-citation + symbol-mapping text per the Global Constraints (e.g. `"""Single-gadget restriction (Webster, Smith, Cohen arXiv:2511.15989 §II.A; Cain et al. arXiv:2603.28627 §B.1). V₀ = support = supp(x); C₀ = data_checks; H_X' = π_{V₀}H_Z^Tπ_{C₀}^T (stored transposed as incidence)."""`). Document the `GadgetLayout` fields: `incidence` = `(H_X')^T`; `gauge` = `H_Z' = ker(H_X')`; `support` = `V₀`; `data_checks` = `C₀`. No reference to `main.tex`.

- [ ] **Step 9: Run gadget tests + golden — verify PASS (byte-identical)**

Run: `python -m pytest src/qldpc/circuits/surgery/gadget_test.py src/qldpc/circuits/surgery/refactor_snapshot_test.py -q`
Expected: all pass, including every `*_byte_identical` test.
**If a golden hash test FAILS, the π-form re-expression changed a matrix — fix the code (check the `% 2` and `.astype(np.uint8)` and shape on empty support). Do NOT edit the golden hashes.**

- [ ] **Step 10: Commit**

```bash
git add src/qldpc/circuits/surgery/gadget.py src/qldpc/circuits/surgery/gadget_test.py
git commit -m "refactor(surgery): π-form gadget construction + S_X'/f_Z names; cite papers"
```

---

### Task 3: Rename field `ancilla_qubits → Q_prime` (cross-file mechanical)

Rename the `GadgetLayout` field to the paper symbol `Q'`. Mechanical; update every read-site.

**Files:**
- Modify: `src/qldpc/circuits/surgery/gadget.py` (field decl + constructor kwargs in `build_gadget`, `build_gadget_augmented`)
- Modify: `src/qldpc/circuits/surgery/circuit.py:172`
- Modify: `src/qldpc/circuits/surgery/y_gadget.py` (lines ~530, 628 docstrings; 693–694 code)
- Modify: `src/qldpc/circuits/surgery/y_circuit.py:199–200`
- Modify: `src/qldpc/circuits/surgery/gadget_test.py`, `circuit_test.py`, `y_gadget_test.py`

**Interfaces:**
- Produces: `GadgetLayout.Q_prime` (was `.ancilla_qubits`) — `tuple[int, ...]`, the `Q'` ancilla qubit IDs. All consumers updated.

- [ ] **Step 1: Find every reference**

Run: `cd /Users/tgzhou/Project/qLDPC && grep -rn 'ancilla_qubits' src/qldpc/circuits/surgery/`
Expected: the field decl + `ancilla_qubits=` constructor kwargs in `gadget.py`, plus the 15 read-sites enumerated in Files.

- [ ] **Step 2: Rename**

In `gadget.py`: the dataclass field `ancilla_qubits: tuple[int, ...]` → `Q_prime: tuple[int, ...]`, and both `ancilla_qubits=ancilla_qubits_…` constructor kwargs → `Q_prime=…` (rename the local `ancilla_qubits`/`ancilla_qubits_aug` too, with a `# Q' ancilla qubit IDs` comment). In every other file replace `.ancilla_qubits` with `.Q_prime`. The `y_gadget.py` docstrings at lines ~530/628 (`len(g_x.ancilla_qubits)`) → `len(g_x.Q_prime)`. Do NOT touch y_gadget's internal `∂₁/∂₀` names.

- [ ] **Step 3: Verify no stragglers**

Run: `grep -rn 'ancilla_qubits' src/qldpc/circuits/surgery/`
Expected: no output.

- [ ] **Step 4: Run the full surgery suite + golden — verify PASS**

Run: `python -m pytest src/qldpc/circuits/surgery/ -q`
Expected: all pass; golden hashes unchanged. **If golden FAILS, you changed behavior — a rename should not. Revert and find the typo.**

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/
git commit -m "refactor(surgery): rename GadgetLayout.ancilla_qubits -> Q_prime (Q')"
```

---

### Task 4: `circuit.py` — joint + single local renames + block-row documentation

Rename the `χ`/ancilla locals and document `M_meas`/`M_comp` against the joint formula. Pure rename.

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py` (`_stitch_intercode` ~433–515, `_stitch_intracode` ~518–594, and the single-PPM emitter)

**Interfaces:**
- Consumes: `GadgetLayout.Q_prime`, `.HX_merged`, `.HZ_merged`, `Bridge.T_l/T_r/H_R/label_l/label_r/width`. No signature changes.
- Produces: nothing new (private helpers, identical return values).

- [ ] **Step 1: Rename locals in `_stitch_intercode` and `_stitch_intracode`**

Apply consistently in both functions:
- `meas_l_rows` → `S_prime_l`, `meas_r_rows` → `S_prime_r` (these are the `S_X'^l`/`S_X'^r` ancilla-stabilizer rows; the old `# χ rows` comments → `# S_X'^s rows = [f_X'^s | H_X'^s | port labels]`).
- column slices `cl_ancilla` → `Ql_prime`, `cr_ancilla` → `Qr_prime` (the `Q'_l`/`Q'_r` columns). Keep `c_adapter` (= adapter 𝒜) and `cl_data`/`cr_data`/`c_data`.

- [ ] **Step 2: Document the block-rows against the joint formula**

Above the `M_meas` assembly add a comment block mapping the 4 rows to `H̃_X^joint` and above `M_comp` the 5 rows to `H̃_Z^joint` (the bridge cycle row `[T_l T_r H_R]`), citing `Swaroop et al. (Swaroop, Jochym-O'Connor, Yoder) arXiv:2410.03628 §III`. Update the function docstrings to name `M_meas`/`M_comp` as the measured/complementary merged check matrices and remove any `χ` wording. No `main.tex` reference.

- [ ] **Step 3: Mirror the rename in the single-PPM emitter**

In the single-PPM circuit/observable helpers, rename any `χ`/`ancilla`-flavored locals to `S_prime`/`Q_prime` wording consistent with Task 2–3 (verify by `grep -n 'χ\|chi\|ancilla' src/qldpc/circuits/surgery/circuit.py` and rewording each non-paper occurrence; leave genuinely-descriptive ones like `ancilla_ids` if they read clearly, but prefer `Q_prime_ids`).

- [ ] **Step 4: Run circuit tests + golden — verify PASS**

Run: `python -m pytest src/qldpc/circuits/surgery/circuit_test.py src/qldpc/circuits/surgery/refactor_snapshot_test.py -q`
Expected: all pass; golden unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/circuit.py
git commit -m "refactor(surgery): joint stitch reads as H̃^joint blocks (S_X', Q'_l/r)"
```

---

### Task 5: `bridge.py` — docstring polish + citations

`bridge.py` identifiers are already paper-faithful; only docstrings change.

**Files:**
- Modify: `src/qldpc/circuits/surgery/bridge.py`

**Interfaces:** none change.

- [ ] **Step 1: Polish docstrings**

In the `Bridge` dataclass and `build_bridge`/`_skip_tree`/`_run_skiptree_on_port_subgraph` docstrings: document the port-label block `π_{𝒫_s}^T P_{σ_s}`, the SkipTree identity `T_s (H_X'^{s,aug})^T π_{𝒫_s}^T = H_R P_{σ_s}^T`, and full citations: `Swaroop et al. (Swaroop, Jochym-O'Connor, Yoder) arXiv:2410.03628 §III` (SkipTree), `Cross et al. arXiv:2407.18393 Thm 6` (Cheeger), `Williamson & Yoder arXiv:2410.02213` (cellulation/boost). Remove any bare surnames / `math.md` refs. Verify `grep -n 'math.md\|Cain mapping' src/qldpc/circuits/surgery/bridge.py` is empty.

- [ ] **Step 2: Run bridge tests + golden — verify PASS**

Run: `python -m pytest src/qldpc/circuits/surgery/bridge_test.py src/qldpc/circuits/surgery/refactor_snapshot_test.py -q`
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add src/qldpc/circuits/surgery/bridge.py
git commit -m "docs(surgery): cite Swaroop/Cross/Williamson-Yoder fully in bridge"
```

---

### Task 6: Final verification sweep

**Files:** none (verification + status update only).

- [ ] **Step 1: Full surgery suite**

Run: `python -m pytest src/qldpc/circuits/surgery/ -q`
Expected: all pass, zero failures, golden green.

- [ ] **Step 2: Jargon sweep — confirm no `χ` / `ancilla_qubits` / `math.md` / bare-surname `mapping` left**

Run: `grep -rn 'χ\|ancilla_qubits\|math.md\|Cain mapping' src/qldpc/circuits/surgery/ | grep -v refactor_snapshot_test`
Expected: no output (or only genuinely-descriptive `ancilla_ids`-style names you consciously kept).

- [ ] **Step 3: Lint / type check (if configured)**

Run: `cd /Users/tgzhou/Project/qLDPC && ruff check src/qldpc/circuits/surgery/ && mypy src/qldpc/circuits/surgery/gadget.py 2>/dev/null || true`
Expected: clean (or unchanged from baseline).

- [ ] **Step 4: Mark spec/plan done & commit**

Append a `**Status:** implemented (2026-06-25)` line to the spec and this plan; commit.

```bash
git add docs/superpowers/
git commit -m "docs(surgery): mark §2/§3 paper-notation refactor implemented"
```

---

## Self-Review (completed by plan author)

- **Spec coverage:** `_projection`+π-form (Task 2) ✓; no-χ/`S_X'` (Tasks 2,4) ✓; `Q'` field (Task 3) ✓; joint block-row docs (Task 4) ✓; bridge citations (Task 5) ✓; byte-identical guarantee (Task 1 + every task's golden gate) ✓; citation strategy/no-main.tex (all tasks) ✓; keep `incidence`/`gauge`/`support`/`data_checks` fields (Task 2 docstrings only) ✓; no-LER (Global Constraints) ✓.
- **Placeholders:** none — every code step shows the code; every run step shows the command + expected output.
- **Type consistency:** `_projection(indices, N) -> np.ndarray (uint8)` used identically in Tasks 2 steps 5–7; `GadgetLayout.Q_prime: tuple[int,...]` defined in Task 3 and consumed in Task 4.
