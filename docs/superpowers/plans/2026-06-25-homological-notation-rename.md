# Surgery homological-notation rename — Implementation Plan

**Status:** implemented (2026-06-25)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-align surgery identifiers + docstrings to `main.tex`'s unified homological symbols (`f_1`/`∂_1`/`f_0`/`∂_0`), with zero behavioral change.

**Architecture:** Pure rename + docstring edits across `gadget.py`, `circuit.py`, `bridge.py`, `y_gadget.py`, `y_circuit.py`. The existing golden-master test `src/qldpc/circuits/surgery/refactor_snapshot_test.py` (12 sha256 hashes, committed in the prior refactor) is the regression gate — matrices/circuits are unchanged, so every hash must stay identical.

**Tech Stack:** Python, numpy, galois (GF(2)), stim, pytest.

## Global Constraints

- **Byte-identical output.** `refactor_snapshot_test.py`'s 12 golden hashes MUST NOT change. If a hash changes, a rename altered behavior — that is a BUG; revert and fix, NEVER edit the hashes.
- **Mapping:** `f_X'→f_1` (data attachment `f_1^T = π_{V₀}`), `H_X'→∂_1` (vertex×edge `π_{V₀}H_Z^Tπ_{C₀}^T`), `f_Z→f_0` (`π_{C₀}^T`), `H_Z'→∂_0` (`ker ∂_1`). §4 extension blocks `f_X/f_Z→f_0^z/f_0^x`.
- **Field decisions:** rename field `gauge → partial_0`; KEEP field `incidence` (= `∂_1^T`; `∂_1 = incidence.T`).
- **Generators keep primes:** `S_X'`/`S_Z'` (`S_X_prime`, `S_prime_l/r`), `Q'` (`Q_prime`, `Ql/Qr_prime`) unchanged.
- **No `χ`** anywhere in surgery source (extends to §4 now): `chi_x/chi_z → SX_prime/SZ_prime`, `χ_X·χ_Z → S_X'·S_Z'`.
- **§4 depth:** keep `d1x/d1z/D1/partial0` (already read as `∂_1^x/∂_1^z/∂_1/∂_0`).
- **Citations:** no internal `docs/superpowers/docs/main.tex` refs in source (drop the 21 in y_gadget/y_circuit, keep the arXiv:2410.02753 cite); cite primary papers fully.
- **No LER / sinter tests.**
- Suite: `cd /Users/tgzhou/Project/qLDPC && python -m pytest src/qldpc/circuits/surgery/ -q`

---

### Task 1: `gadget.py` internal locals + docstrings (no field rename)

Rename function-local identifiers and docstring symbols to the homological set. Do NOT rename any dataclass field yet.

**Files:**
- Modify: `src/qldpc/circuits/surgery/gadget.py`

**Interfaces:**
- Consumes: `_projection`, field `incidence`/`gauge` (unchanged this task).
- Produces: no public signature change (locals are private; fields untouched).

- [ ] **Step 1: Rename locals in `_step1_restriction`**

Replace the incidence-construction block with (keep surrounding code):

```python
    # ∂_1 = π_{V₀} H_Z^T π_{C₀}^T  (vertex×edge incidence; Webster, Smith, Cohen
    # arXiv:2511.15989 §II.A; Cain et al. arXiv:2603.28627 §B.1; arXiv:2410.02753 Eq.(62)).
    # The stored `incidence` is its transpose ∂_1^T — the |C₀|×|V₀| edge×vertex form
    # (§4 y_gadget.py recovers ∂_1^x = incidence.T).
    pi_V0 = _projection(support, n)                          # π_{V₀} = f_1^T ∈ F₂^{|V₀|×n}
    pi_C0 = _projection(data_checks, H_complement.shape[0])  # π_{C₀} ∈ F₂^{|C₀|×m_comp}
    partial_1 = (pi_V0 @ H_complement.T @ pi_C0.T) % 2       # ∂_1, |V₀|×|C₀|
    incidence = partial_1.T.astype(np.uint8)                # ∂_1^T, |C₀|×|V₀|
    return support, data_checks, incidence
```

- [ ] **Step 2: Rename locals in `_assemble_HX_L1`**

Replace the bottom-block assembly with:

```python
    # S_X' rows = [f_1^T | ∂_1] : f_1^T = π_{V₀} on data, ∂_1 = incidence.T on Q'.
    f_1_T = np.zeros((n_v0, n), dtype=np.uint8)
    f_1_T[np.arange(n_v0), np.asarray(support_indices)] = 1
    partial_1 = incidence.T.astype(np.uint8)
    S_X_prime = np.hstack([f_1_T, partial_1]).astype(np.uint8)
    return np.vstack([top, S_X_prime]).astype(np.uint8)
```

- [ ] **Step 3: Rename `f_Z → f_0` in `_step3_assemble`**

Replace the `f_Z` construction:

```python
    # f_0 = π_{C₀}^T : extends the original Z-checks (basis=X) onto the new Q' ancillas.
    # _projection's sentinel rule zeroes the columns of boost-added Q' (data_checks == -1).
    m_comp = mZ if basis is Pauli.X else mX
    f_0 = _projection(data_checks, m_comp).T.astype(np.uint8)   # (m_comp, |C₀|)
```

Then in BOTH `np.block(...)` branches replace the `incidence_tilde`/`f_Z` entry with `f_0` (the variable feeding the `[HZ, f_0]` / `[HX, f_0]` top block-row).

- [ ] **Step 4: Update remaining docstrings to homological symbols**

In the module docstring and the `_step1_restriction`, `_step2_gauge_fix`, `_assemble_HX_L1`, `_step3_assemble`, `build_gadget`, `build_gadget_augmented`, and `GadgetLayout` docstrings, apply `H_X'→∂_1`, `f_X'→f_1` (note `f_1^T = π_{V₀}`), `f_Z→f_0`, `H_Z'→∂_0`. In particular fix the `_step2_gauge_fix` doc slip — replace `ker((H_X')^T)` with the correct:

```
∂_0 = canonical row basis of ker(∂_1) = left_null_space(incidence) = ker(incidence^T) over GF(2).
`incidence` = ∂_1^T (|C₀|×|V₀|); returns ∂_0 of shape (r, |C₀|) where r = |C₀| − rank(∂_1).
```

Document `GadgetLayout` fields: `incidence` = `∂_1^T` (edge×vertex; `∂_1 = incidence.T`), `gauge` = `∂_0 = ker ∂_1` (field renamed next task).

- [ ] **Step 5: Verify no stale §2 symbols + golden green**

Run: `grep -nE "H_X_prime|f_X_prime|\bf_Z\b|H_X'|H_Z'|f_X'" src/qldpc/circuits/surgery/gadget.py`
Expected: no output (all renamed; `S_X_prime` and `f_1_T`/`f_0`/`partial_1` remain, which is correct — they don't match the pattern).
Run: `python -m pytest src/qldpc/circuits/surgery/gadget_test.py src/qldpc/circuits/surgery/refactor_snapshot_test.py -q`
Expected: all pass; golden hashes unchanged. **If a golden test fails, a rename changed a value — fix the code, NEVER the hashes.**

- [ ] **Step 6: Commit**

```bash
git add src/qldpc/circuits/surgery/gadget.py
git commit -m "refactor(surgery): gadget locals/docstrings → ∂_1/f_1/f_0/∂_0"
```

---

### Task 2: Rename field `gauge → partial_0` (atomic cross-file)

Rename the `GadgetLayout` field and every reader in one commit (else the suite breaks mid-rename).

**Files:**
- Modify: `gadget.py` (field decl + `gauge=`/`gauge_aug` constructor kwargs + locals feeding them), `circuit.py` (6 reads), `cheeger.py` (if it reads `.gauge`), and tests: `gadget_test.py`, `circuit_test.py`, `cheeger_test.py`, `bridge_test.py`, `y_gadget_test.py` (only where `.gauge` is read).

**Interfaces:**
- Produces: `GadgetLayout.partial_0` (was `.gauge`), `np.ndarray` `∂_0 = ker ∂_1`. All readers updated.

- [ ] **Step 1: Enumerate every reference**

Run: `cd /Users/tgzhou/Project/qLDPC && grep -rn '\bgauge\b\|\.gauge\b\|gauge=' src/qldpc/circuits/surgery/`
Note which are the field (decl `gauge: np.ndarray`; constructor kwargs `gauge=...`; reads `.gauge`) vs prose-word "gauge" in docstrings (those may stay or become "∂_0" — see Step 3).

- [ ] **Step 2: Rename the field + reads**

In `gadget.py`: dataclass field `gauge: np.ndarray` → `partial_0: np.ndarray`; the local `gauge`/`gauge_aug` (from `_step2_gauge_fix`) and both constructor kwargs `gauge=gauge`/`gauge=gauge_aug` → `partial_0=...`. (Keep `_step2_gauge_fix`'s return value; just rename the receiving local to `partial_0`/`partial_0_aug`.) In every other file replace `.gauge` reads with `.partial_0` (`circuit.py` lines ~170, 180, 186, 303, 312, 469; tests). Keep the surrounding count-local NAMES (`n_gauge_*`, `r_l`, `r_r`) unchanged.

- [ ] **Step 3: Prose "gauge" word (docstrings)**

Where "gauge" appears as a prose noun (e.g. `cheeger.py:453` "boosted incidence, gauge, ...", `bridge.py:35/396`), update to "∂_0" or "gauge (∂_0)" for consistency. These are doc-only.

- [ ] **Step 4: Verify + golden green**

Run: `grep -rn '\.gauge\b\|gauge:\s*np\|gauge=' src/qldpc/circuits/surgery/`
Expected: no output (field + kwargs + reads all renamed).
Run: `python -m pytest src/qldpc/circuits/surgery/ -q`
Expected: all pass; golden unchanged. **Golden fail ⇒ typo changed behavior; fix code, not hashes.**

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/
git commit -m "refactor(surgery): rename GadgetLayout.gauge -> partial_0 (∂_0)"
```

---

### Task 3: `circuit.py` + `bridge.py` docstring/comment symbol rename

Pure docstring/comment edits — no code/logic change.

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py`, `src/qldpc/circuits/surgery/bridge.py`

**Interfaces:** none change.

- [ ] **Step 1: circuit.py comments/docstrings**

Apply `H_X'→∂_1`, `f_X'→f_1`, `f_Z→f_0`, `H_Z'→∂_0` to the block-row comments and docstrings (e.g. lines ~131, 480–481, 488–489, 503–504, 576–577, 583–584, 597). The `S_X'^s rows = [f_X'^s | H_X'^s | …]` comments become `S_X'^s rows = [f_1^{s,T} | ∂_1^s | …]`. Keep `S_prime_l/r`, `Ql/Qr_prime`, `M_meas/M_comp` identifiers.

- [ ] **Step 2: bridge.py docstrings**

Replace `H_X'^{s,aug} → ∂_1^{s,aug}` in the SkipTree identity and surrounding text (lines ~29, 31, 103, 356, 359, 452, 454): the identity reads `T_s (∂_1^{s,aug})^T π_{𝒫_s}^T = H_R P_{σ_s}^T`. No executable line changes.

- [ ] **Step 3: Verify + golden green**

Run: `grep -nE "H_X'|H_Z'|f_X'|\bf_Z\b" src/qldpc/circuits/surgery/circuit.py src/qldpc/circuits/surgery/bridge.py`
Expected: no output.
Run: `python -m pytest src/qldpc/circuits/surgery/circuit_test.py src/qldpc/circuits/surgery/bridge_test.py src/qldpc/circuits/surgery/refactor_snapshot_test.py -q`
Expected: all pass; golden unchanged.

- [ ] **Step 4: Commit**

```bash
git add src/qldpc/circuits/surgery/circuit.py src/qldpc/circuits/surgery/bridge.py
git commit -m "docs(surgery): circuit/bridge docstrings → ∂_1/f_1/f_0/∂_0"
```

---

### Task 4: §4 `y_gadget.py` / `y_circuit.py` — remove χ + align docstrings + drop main.tex refs

**Files:**
- Modify: `src/qldpc/circuits/surgery/y_gadget.py`, `src/qldpc/circuits/surgery/y_circuit.py`, and `y_gadget_test.py`/`y_circuit_test.py` if they reference `chi_x`/`chi_z`.

**Interfaces:** no public signature change (`chi_x`/`chi_z` are function-locals).

- [ ] **Step 1: Remove the χ identifiers**

In `y_gadget.py`, rename the function-locals `chi_x → SX_prime`, `chi_z → SZ_prime` (lines ~698, 700, 717, 724) — these hold the `S_X'`/`S_Z'` rows (`g_x.HX_merged[m_x:]` / `g_z.HZ_merged[m_z:]`). Update their use sites in the same function. (Verify by reading the function; rename every occurrence of the two locals.)

- [ ] **Step 2: Remove χ from prose**

In `y_circuit.py` replace `χ_X·χ_Z·y_v` (and any `χ_X`/`χ_Z`) in docstrings/comments (lines ~76, 490, 688, 692) with `S_X'·S_Z'·y_v` (generators keep primes). Keep the `y_v` symbol.

- [ ] **Step 3: Align residual block symbols + KEEP d1x/D1/partial0**

Where §4 docstrings still write `H_X'`/`f_X`/`f_Z`, change to `∂_1`/`f_0^x`/`f_0^z`. Do NOT rename `d1x/d1z/D1/partial0` or `∂_0^{(X)}/∂_0^{(Z)}` — they already read homologically.

- [ ] **Step 4: Drop internal-doc references**

Remove every `docs/superpowers/docs/main.tex §X` tail from y_gadget/y_circuit docstrings (14 + 7 occurrences), keeping the primary citation `Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §…`. Do not remove the arXiv cite; remove only the `main.tex` pointer.

- [ ] **Step 5: Verify + golden green**

Run: `grep -nE "χ|chi_x|chi_z|main\.tex|H_X'" src/qldpc/circuits/surgery/y_gadget.py src/qldpc/circuits/surgery/y_circuit.py`
Expected: no output.
Run: `python -m pytest src/qldpc/circuits/surgery/y_gadget_test.py src/qldpc/circuits/surgery/y_circuit_test.py src/qldpc/circuits/surgery/refactor_snapshot_test.py -q`
Expected: all pass; golden unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/qldpc/circuits/surgery/y_gadget.py src/qldpc/circuits/surgery/y_circuit.py src/qldpc/circuits/surgery/y_gadget_test.py src/qldpc/circuits/surgery/y_circuit_test.py
git commit -m "refactor(surgery): §4 drop χ + align ∂_1/∂_0; remove main.tex refs"
```

---

### Task 5: Final verification sweep

**Files:** none (verification + status only).

- [ ] **Step 1: Full suite**

Run: `python -m pytest src/qldpc/circuits/surgery/ -q`
Expected: all pass; golden green.

- [ ] **Step 2: Jargon + symbol sweep**

Run: `grep -rnE "χ|chi_x|chi_z|H_X'|H_Z'|f_X'|main\.tex" src/qldpc/circuits/surgery/gadget.py src/qldpc/circuits/surgery/circuit.py src/qldpc/circuits/surgery/bridge.py src/qldpc/circuits/surgery/y_gadget.py src/qldpc/circuits/surgery/y_circuit.py`
Expected: no output. (Report any hit verbatim; do not fix silently.)
Run: `grep -rn '\.gauge\b' src/qldpc/circuits/surgery/` → expected no output.

- [ ] **Step 3: Mark status (working-tree only, no commit)**

Append `**Status:** implemented (2026-06-25)` near the top of this plan and the design spec `docs/superpowers/specs/2026-06-25-homological-notation-rename-design.md`. Do NOT `git add`/commit them.

---

## Self-Review (completed by plan author)

- **Spec coverage:** gadget locals+∂_0 doc-fix (Task 1) ✓; field gauge→partial_0 (Task 2) ✓; circuit/bridge docstrings (Task 3) ✓; §4 χ-removal + docstring alignment + keep d1x/D1 + drop main.tex (Task 4) ✓; keep `incidence` field / generators' primes (Global Constraints, all tasks) ✓; byte-identical golden gate (every task) ✓; no-LER (Global Constraints) ✓.
- **Placeholders:** none — substantive edits shown verbatim for gadget.py; mechanical renames give exact target strings + grep verification.
- **Type consistency:** `GadgetLayout.partial_0: np.ndarray` defined in Task 2 and read in Task 2's circuit.py updates; `partial_1`/`f_1_T`/`f_0` locals introduced and used within the same functions in Task 1; `SX_prime`/`SZ_prime` locals introduced and used within one y_gadget function in Task 4.
