# Eq (68) Homological Y / mixed-Pauli PPM — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `build_y_gadget` to produce the arXiv:2410.02753 Eq (68)
(`eq:joint_final`) merged code for **general overlap `|W|`**, with per-system
Cheeger boost for distance and no SkipTree/bridge, and run it end-to-end
(merged code → circuit → DEM) on a BB `[[36,8,4]]` fixture.

**Architecture:** Two single-operator homological gadgets (`build_gadget` for
`X̄`, dual for `Z̄`), each Cheeger-boosted to `≥1`; merge the χ pairs sharing
each qubit `v ∈ W = supp(x)∩supp(z)` into mixed `y_v` rows; append
`∂_0 = ker(merged ∂_1)` (the merged cycle basis — `0` extra rows at `|W|=1`,
`|W|−1` non-CSS crossing-cycle rows at `|W|≥2`) in place of the old per-system
gauge blocks. The existing circuit already routes every non-CSS row through a
per-column CX/CY/CZ syndrome ancilla, so it consumes the new merged code
unchanged.

**Tech Stack:** Python, numpy, `galois` (GF(2)), `stim`, the in-repo
`qldpc.codes` / `qldpc.circuits.surgery` modules.

## Global Constraints

- **No SkipTree / no §3.7 bridge** in this path: do not import or build
  `bridge.Bridge` / `build_bridge` from `y_gadget.py`; remove the `bridge`
  field from `YGadgetLayout` (verified unused: `yg.bridge` is read nowhere).
- **Compute `W = supp(x)∩supp(z)`**; never assume `|W|=1`. `|W|` is odd for an
  anticommuting same-qubit `(x, z)` pair.
- **`∂_0 = ker(merged ∂_1)`** always; the `|W|=1` pure-CSS split is the
  degenerate output, not a separate code path.
- **`∂_0` cross-typing (Eq 68 row 6):** a cycle's κ_X support → Z-part on κ_X
  (`∂_0^(X)`); its κ_Z support → X-part on κ_Z (`∂_0^(Z)`). One crossing cycle
  is a single symplectic (non-CSS) row.
- **Per-system Cheeger boost ≥ 1** (`cheeger.boost_gadget(method="combinatorial",
  target=1.0)`) on each gadget before merging; store the **boosted** gadgets in
  the layout so `k_x = len(g_x.ancilla_qubits)` stays consistent with the merged
  code width.
- **Citations:** arXiv:2410.02753 §III.C/§III.D + `docs/superpowers/docs/main.tex`
  §4. Remove arXiv:2407.18393 §3.7/§3.2 and arXiv:2410.03628 (SkipTree) from this
  path's code and docstrings. Use full author+arXiv-ID+§ citation form.
- **Fixture:** BB `BBCode({x:3, y:6}, x³+y+y², y³+x+x²)` = `[[36,8,4]]`.

---

## File Structure

- `src/qldpc/circuits/surgery/y_gadget.py` — **primary**. Generalize overlap
  helper; add merged-incidence/`∂_0` helpers; rebuild `build_y_gadget` (boost +
  `∂_0` + drop bridge); add BB fixture. Remove `Bridge`/`build_bridge` import and
  the `bridge` field.
- `src/qldpc/circuits/surgery/circuit.py` — small: `build_single_y_ppm_circuit`
  docstring/citations; confirm mixed-`∂_0` rows flow through the Y-phase loop
  (already generic). No structural change expected.
- `src/qldpc/circuits/surgery/y_gadget_test.py` — re-target to BB, parametrize
  `|W| ∈ {1, 3}`, assert Eq-68 structure + crossing-cycle counts + distance.
- `src/qldpc/circuits/surgery/circuit_single_y_test.py` — re-target to BB;
  DEM-compiles + measurement-fault-distance check; drop the Steane xfail.

---

### Task 1: Generalize overlap helper to `_locate_overlaps`

**Files:**
- Modify: `src/qldpc/circuits/surgery/y_gadget.py` (`_locate_overlap` → add
  `_locate_overlaps`)
- Test: `src/qldpc/circuits/surgery/y_gadget_test.py`

**Interfaces:**
- Produces: `_locate_overlaps(code: CSSCode, x: np.ndarray, z: np.ndarray) ->
  tuple[int, ...]` — sorted tuple `W = supp(x)∩supp(z)`; raises `ValueError`
  unless `H_Z x = 0`, `H_X z = 0`, and `x·z` is odd (anticommuting).

- [ ] **Step 1: Write the failing test**

```python
# y_gadget_test.py
import numpy as np
from qldpc.circuits.surgery.y_gadget import _locate_overlaps, _steane_y_pair

def test_locate_overlaps_steane_is_singleton():
    code, x, z = _steane_y_pair()
    W = _locate_overlaps(code, x, z)
    assert isinstance(W, tuple)
    assert W == tuple(int(i) for i in np.where(x.astype(bool) & z.astype(bool))[0])
    assert len(W) % 2 == 1  # anticommuting ⇒ odd overlap
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/qldpc/circuits/surgery/y_gadget_test.py::test_locate_overlaps_steane_is_singleton -v`
Expected: FAIL with `ImportError: cannot import name '_locate_overlaps'`

- [ ] **Step 3: Add `_locate_overlaps` (keep `_locate_overlap` as a thin wrapper for now)**

```python
def _locate_overlaps(code: CSSCode, x: np.ndarray, z: np.ndarray) -> tuple[int, ...]:
    """Return W = supp(x) ∩ supp(z), the physical Pauli-Y qubits of Ȳ = iX̄Z̄.

    Validates (Ide, Gowda, Nadkarni, Dauphinais arXiv:2410.02753 §III.D;
    docs/superpowers/docs/main.tex §4.1):
      * x is a logical-X representative: H_Z @ x == 0 (mod 2),
      * z is a logical-Z representative: H_X @ z == 0 (mod 2),
      * x and z anticommute: x · z is odd (so |W| is odd, ≥ 1).
    """
    x = np.asarray(x).astype(np.uint8)
    z = np.asarray(z).astype(np.uint8)
    n = code.num_qudits
    if x.shape != (n,) or z.shape != (n,):
        raise ValueError(f"x/z must have shape ({n},); got {x.shape}, {z.shape}")
    if ((np.asarray(code.matrix_z).astype(np.uint8) @ x) % 2 != 0).any():
        raise ValueError("x is not a logical-X representative: H_Z @ x != 0 (mod 2)")
    if ((np.asarray(code.matrix_x).astype(np.uint8) @ z) % 2 != 0).any():
        raise ValueError("z is not a logical-Z representative: H_X @ z != 0 (mod 2)")
    if int(np.dot(x.astype(np.int64), z.astype(np.int64))) % 2 == 0:
        raise ValueError("x and z commute (x · z even); they cannot form Ȳ = iX̄Z̄")
    return tuple(int(i) for i in np.where(x.astype(bool) & z.astype(bool))[0])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/qldpc/circuits/surgery/y_gadget_test.py::test_locate_overlaps_steane_is_singleton -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/y_gadget.py src/qldpc/circuits/surgery/y_gadget_test.py
git commit -m "feat(surgery): _locate_overlaps returns full W=supp(x)∩supp(z)"
```

---

### Task 2: Merged incidence `∂_1` and `∂_0 = ker(∂_1)` helpers

**Files:**
- Modify: `src/qldpc/circuits/surgery/y_gadget.py`
- Test: `src/qldpc/circuits/surgery/y_gadget_test.py`

**Interfaces:**
- Consumes: `GadgetLayout` (`.incidence`, `.support`, `.ancilla_qubits`).
- Produces:
  - `_merged_incidence(g_x, g_z, x, z) -> tuple[np.ndarray, int, int]` — returns
    `(∂_1, k_x, k_z)` with `∂_1` shape `(|V_X|+|W|+|V_Z|, k_x+k_z)`, columns
    ordered `(κ_X | κ_Z)`.
  - `_partial0_symplectic_rows(g_x, g_z, x, z, *, n, k_x, k_z) -> np.ndarray` —
    `∂_0 = ker(∂_1)` embedded as symplectic rows, shape `(r, 2*(n+k_x+k_z))`;
    each cycle's κ_X support → Z-part on κ_X, κ_Z support → X-part on κ_Z.

- [ ] **Step 1: Write the failing test (|W|=1 ⇒ 0 crossing cycles, ∂_0 = G_x⊕G_z)**

```python
# y_gadget_test.py
import galois, numpy as np
from qldpc.objects import Pauli
from qldpc.circuits.surgery.gadget import build_gadget
from qldpc.circuits.surgery.y_gadget import (
    _merged_incidence, _partial0_symplectic_rows, _steane_y_pair,
)
GF2 = galois.GF(2)

def test_partial0_steane_w1_is_pure_css_no_crossing():
    code, x, z = _steane_y_pair()
    g_x = build_gadget(code, x, basis=Pauli.X)
    g_z = build_gadget(code, z, basis=Pauli.Z)
    n = code.num_qudits
    k_x = len(g_x.ancilla_qubits); k_z = len(g_z.ancilla_qubits)
    D1, kx, kz = _merged_incidence(g_x, g_z, x, z)
    assert (kx, kz) == (k_x, k_z)
    # |W|=1 wedge: dim ker(merged) == dim ker(∂1x) + dim ker(∂1z)  (no crossing cycle)
    rank = lambda M: int(np.linalg.matrix_rank(GF2(np.asarray(M).astype(int) % 2)))
    d1x = np.asarray(g_x.incidence).astype(int).T
    d1z = np.asarray(g_z.incidence).astype(int).T
    n_merged_cyc = np.asarray(GF2(D1.astype(int)).null_space()).shape[0]
    n_sep = (np.asarray(GF2(d1x).null_space()).shape[0]
             + np.asarray(GF2(d1z).null_space()).shape[0])
    assert n_merged_cyc == n_sep  # verified fact for |W|=1
    rows = _partial0_symplectic_rows(g_x, g_z, x, z, n=n, k_x=k_x, k_z=k_z)
    nm = n + k_x + k_z
    assert rows.shape[1] == 2 * nm
    # every ∂_0 row is pure-CSS at |W|=1: no row has BOTH an X-part and a Z-part bit
    for r in rows:
        has_x = r[:nm].any(); has_z = r[nm:].any()
        assert not (has_x and has_z), "unexpected crossing (non-CSS) row at |W|=1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/qldpc/circuits/surgery/y_gadget_test.py::test_partial0_steane_w1_is_pure_css_no_crossing -v`
Expected: FAIL with `ImportError: cannot import name '_merged_incidence'`

- [ ] **Step 3: Implement the helpers**

```python
def _merged_incidence(
    g_x: GadgetLayout, g_z: GadgetLayout, x: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """Merged graph incidence ∂_1 (arXiv:2410.02753 Eq.(66); main.tex §4.4).

    Rows = vertices V_X ⊔ W ⊔ V_Z (data qubits of supp(x)/supp(z)); columns =
    edges (κ_X | κ_Z). ∂_1^x = g_x.incidence.T (rows=support, cols=κ_X); dual for
    ∂_1^z. The W rows stack the X- and Z-system incidences side by side, gluing
    the two graphs at the shared Y-qubits.
    """
    x = np.asarray(x).astype(np.uint8); z = np.asarray(z).astype(np.uint8)
    d1x = np.asarray(g_x.incidence).astype(np.uint8).T  # (|supp x|, k_x)
    d1z = np.asarray(g_z.incidence).astype(np.uint8).T  # (|supp z|, k_z)
    supx = list(g_x.support); supz = list(g_z.support)
    W = sorted(set(int(i) for i in np.where(x)[0]) & set(int(i) for i in np.where(z)[0]))
    VX = [v for v in supx if v not in W]
    VZ = [v for v in supz if v not in W]
    k_x = d1x.shape[1]; k_z = d1z.shape[1]

    def rows_of(d: np.ndarray, sup: list[int], sel: list[int]) -> np.ndarray:
        idx = [sup.index(v) for v in sel]
        return d[idx] if idx else np.zeros((0, d.shape[1]), dtype=np.uint8)

    top = np.hstack([rows_of(d1x, supx, VX), np.zeros((len(VX), k_z), np.uint8)])
    mid = np.hstack([rows_of(d1x, supx, W), rows_of(d1z, supz, W)])
    bot = np.hstack([np.zeros((len(VZ), k_x), np.uint8), rows_of(d1z, supz, VZ)])
    return np.vstack([top, mid, bot]).astype(np.uint8), k_x, k_z


def _partial0_symplectic_rows(
    g_x: GadgetLayout, g_z: GadgetLayout, x: np.ndarray, z: np.ndarray,
    *, n: int, k_x: int, k_z: int,
) -> np.ndarray:
    """∂_0 = ker(merged ∂_1) as symplectic rows (arXiv:2410.02753 Eq.(67)/(68)).

    Column layout per half: [data (n) | κ_X | κ_Z]. A cycle c = (c_X | c_Z):
    its κ_X support enters as Z-part on κ_X (∂_0^(X)); its κ_Z support enters as
    X-part on κ_Z (∂_0^(Z)). A crossing cycle (|W|≥2) populates both → one
    non-CSS row.
    """
    D1, kx, kz = _merged_incidence(g_x, g_z, x, z)
    if (kx, kz) != (k_x, k_z):
        raise ValueError(f"incidence κ sizes {(kx, kz)} != ({k_x}, {k_z})")
    ker = np.asarray(GF2(D1.astype(int)).null_space()).astype(np.uint8)  # rows over (κ_X|κ_Z)
    nm = n + k_x + k_z
    out = np.zeros((ker.shape[0], 2 * nm), dtype=np.uint8)
    for i, c in enumerate(ker):
        c_x = c[:k_x]; c_z = c[k_x:]
        out[i, n + k_x : nm] = c_z              # X-part on κ_Z  (∂_0^(Z))
        out[i, nm + n : nm + n + k_x] = c_x      # Z-part on κ_X  (∂_0^(X))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/qldpc/circuits/surgery/y_gadget_test.py::test_partial0_steane_w1_is_pure_css_no_crossing -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/y_gadget.py src/qldpc/circuits/surgery/y_gadget_test.py
git commit -m "feat(surgery): merged-incidence ∂_1 and ∂_0=ker(∂_1) helpers"
```

---

### Task 3: Rebuild `build_y_gadget` (boost + `∂_0` block + drop bridge)

**Files:**
- Modify: `src/qldpc/circuits/surgery/y_gadget.py` (`YGadgetLayout` dataclass and
  `build_y_gadget`; remove `from .bridge import Bridge, build_bridge`)
- Test: `src/qldpc/circuits/surgery/y_gadget_test.py`

**Interfaces:**
- Consumes: `_locate_overlaps`, `_partial0_symplectic_rows`,
  `gadget.build_gadget`, `cheeger.boost_gadget`, `merge.apply_mixed_basis_merge`.
- Produces: `build_y_gadget(code, *, x, z) -> YGadgetLayout`. `YGadgetLayout`
  has **no `bridge` field**; fields: `code, x, z, W: tuple[int,...], g_x, g_z,
  Y_stab, H_sym, merged_code, obs0_xor_map, obs0_readout`.

- [ ] **Step 1: Write the failing test**

```python
# y_gadget_test.py
import dataclasses
from qldpc.objects import Pauli
from qldpc.circuits.surgery.cheeger import cheeger_constant
from qldpc.circuits.surgery.y_gadget import build_y_gadget, _steane_y_pair, YGadgetLayout

def test_build_y_gadget_no_bridge_field_and_boosted():
    code, x, z = _steane_y_pair()
    yg = build_y_gadget(code, x=x, z=z)
    names = {f.name for f in dataclasses.fields(YGadgetLayout)}
    assert "bridge" not in names
    assert yg.W == (int(__import__("numpy").where(x.astype(bool) & z.astype(bool))[0][0]),)
    assert cheeger_constant(yg.g_x) >= 1.0 and cheeger_constant(yg.g_z) >= 1.0
    # Ȳ in stabilizer center: symplectic [x|z] in rowspace(H_sym) restricted to data cols
    import numpy as np, galois
    GF2 = galois.GF(2); n = code.num_qudits
    nm = yg.merged_code.num_qudits
    H = np.asarray(yg.H_sym).astype(int)
    data_cols = list(range(n)) + list(range(nm, nm + n))
    M = GF2(H[:, data_cols] % 2)
    v = GF2(np.concatenate([np.asarray(x), np.asarray(z)]).astype(int) % 2)
    assert int(np.linalg.matrix_rank(M)) == int(np.linalg.matrix_rank(
        GF2(np.vstack([np.asarray(M), np.asarray(v)[None, :]]))))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/qldpc/circuits/surgery/y_gadget_test.py::test_build_y_gadget_no_bridge_field_and_boosted -v`
Expected: FAIL (`YGadgetLayout` still has `bridge`; `yg.W` missing)

- [ ] **Step 3: Edit `YGadgetLayout` and `build_y_gadget`**

In `y_gadget.py`: remove `from .bridge import Bridge, build_bridge`. In the
`YGadgetLayout` dataclass, **delete** the `bridge: Bridge` field and the `q0: int`
field; **add** `W: tuple[int, ...]`. Replace the body of `build_y_gadget` (from
the `q0 = _locate_overlap(...)` line through the `return YGadgetLayout(...)`)
with:

```python
    x = np.asarray(x).astype(np.uint8)
    z = np.asarray(z).astype(np.uint8)
    W = _locate_overlaps(code, x, z)

    # Per-system L=1 gadgets, each Cheeger-boosted to ≥1 (main.tex §4.7 /
    # arXiv:2410.02753 §III.D: per-system distance argument).
    from .cheeger import boost_gadget, cheeger_constant
    g_x = build_gadget(code, x, basis=Pauli.X)
    g_z = build_gadget(code, z, basis=Pauli.Z)
    if cheeger_constant(g_x) < 1.0:
        g_x = boost_gadget(g_x, method="combinatorial", target=1.0, seed=0)
    if cheeger_constant(g_z) < 1.0:
        g_z = boost_gadget(g_z, method="combinatorial", target=1.0, seed=0)

    field = code.field
    n = code.num_qudits
    m_x = int(np.asarray(code.matrix_x).shape[0])
    m_z = int(np.asarray(code.matrix_z).shape[0])
    k_x = len(g_x.ancilla_qubits)
    k_z = len(g_z.ancilla_qubits)
    n_merged = n + k_x + k_z

    # χ / extension blocks (Webster §II.A step 3 decomposition).
    chi_x = np.asarray(g_x.HX_merged[m_x:]).astype(np.uint8)          # (|V0x|, n+k_x)
    hz_ext_kx = np.asarray(g_x.HZ_merged[:m_z]).astype(np.uint8)      # [H_Z | F̃_x]
    chi_z = np.asarray(g_z.HZ_merged[m_z:]).astype(np.uint8)          # (|V0z|, n+k_z)
    hx_ext_kz = np.asarray(g_z.HX_merged[:m_x]).astype(np.uint8)      # [H_X | F̃_z]

    def _embed(rows, *, data=None, kx=None, kz=None):
        out = np.zeros((rows, n_merged), dtype=np.uint8)
        if data is not None: out[:, :n] = data
        if kx is not None:   out[:, n : n + k_x] = kx
        if kz is not None:   out[:, n + k_x :] = kz
        return out

    # X-type rows: H_X extended onto κ_z; χ_X on [data | κ_x]. (NO per-system gauge.)
    HX_all = np.vstack([
        _embed(m_x, data=hx_ext_kz[:, :n], kz=hx_ext_kz[:, n:]),
        _embed(chi_x.shape[0], data=chi_x[:, :n], kx=chi_x[:, n:]),
    ]).astype(np.uint8)
    # Z-type rows: H_Z extended onto κ_x; χ_Z on [data | κ_z].
    HZ_all = np.vstack([
        _embed(m_z, data=hz_ext_kx[:, :n], kx=hz_ext_kx[:, n:]),
        _embed(chi_z.shape[0], data=chi_z[:, :n], kz=chi_z[:, n:]),
    ]).astype(np.uint8)

    # Merge χ_X@v / χ_Z@v into one mixed y_v row for every v ∈ W (§III.D / §4.3).
    HX_out, HZ_out, Y_stab, _obs0_y, _xl, _zl = apply_mixed_basis_merge(
        HX_all, HZ_all, merge_qubits=W, adapter_cols=tuple(range(n)),
    )
    HX_out = np.asarray(HX_out).astype(np.int_)
    HZ_out = np.asarray(HZ_out).astype(np.int_)
    if Y_stab is None or Y_stab.shape[0] < len(W):
        raise ValueError(
            f"BLOCKED: cross-merge produced {0 if Y_stab is None else Y_stab.shape[0]} "
            f"y_v rows, expected |W|={len(W)} (arXiv:2410.02753 §III.D)"
        )

    # ∂_0 = ker(merged ∂_1): cycle basis of the glued graph (§III.D Eq.66/67),
    # replacing the per-system gauge blocks. 0 rows at |W|=1, |W|−1 crossing rows
    # at |W|≥2.
    partial0 = _partial0_symplectic_rows(g_x, g_z, x, z, n=n, k_x=k_x, k_z=k_z)

    rows_sym: list[np.ndarray] = []
    for r in HX_out:
        rows_sym.append(np.concatenate([r, np.zeros(n_merged, dtype=np.int_)]))
    for r in HZ_out:
        rows_sym.append(np.concatenate([np.zeros(n_merged, dtype=np.int_), r]))
    for r in Y_stab:
        rows_sym.append(r.astype(np.int_))
    for r in partial0:
        rows_sym.append(r.astype(np.int_))
    H_sym = (np.array(rows_sym, dtype=np.int_) if rows_sym
             else np.zeros((0, 2 * n_merged), dtype=np.int_))

    Hx = H_sym[:, :n_merged]; Hz = H_sym[:, n_merged:]
    comm = (Hx @ Hz.T + Hz @ Hx.T) % 2
    np.fill_diagonal(comm, 0)
    merged_code = QuditCode(field(H_sym), is_subsystem_code=bool(comm.any()))

    obs0_rows, obs0_readout = _ybar_obs0_rows(
        H_sym, code, x, z, n0=n, n_merged=n_merged, k_x=k_x
    )
    return YGadgetLayout(
        code=code, x=x, z=z, W=W, g_x=g_x, g_z=g_z,
        Y_stab=Y_stab, H_sym=H_sym, merged_code=merged_code,
        obs0_xor_map=obs0_rows, obs0_readout=obs0_readout,
    )
```

Also remove the now-unused `_locate_overlap` body if nothing references it (keep
`_overlap_size` and `_steane_y_pair`). Update the module + `YGadgetLayout`
docstrings to cite arXiv:2410.02753 §III.C/§III.D + main.tex §4 and drop §3.7
bridge / Remark-23 narrative.

- [ ] **Step 4: Run the test + the full y_gadget suite**

Run: `pytest src/qldpc/circuits/surgery/y_gadget_test.py -v`
Expected: PASS (fix any test still importing `_locate_overlap`/`q0`/`bridge`).

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/y_gadget.py src/qldpc/circuits/surgery/y_gadget_test.py
git commit -m "feat(surgery): build_y_gadget = Eq68 general-|W| (boost + ∂_0, no bridge)"
```

---

### Task 4: BB fixture `_bb_y_pair`

**Files:**
- Modify: `src/qldpc/circuits/surgery/y_gadget.py`
- Test: `src/qldpc/circuits/surgery/y_gadget_test.py`

**Interfaces:**
- Produces: `_bb_y_pair(overlap: int = 1) -> tuple[CSSCode, np.ndarray,
  np.ndarray]` — `BBCode({x:3,y:6}, x³+y+y², y³+x+x²)` and an `(x, z)` pair for
  logical qubit 0 with `|supp(x)∩supp(z)| == overlap` (`overlap ∈ {1, 3}`).

- [ ] **Step 1: Write the failing test**

```python
# y_gadget_test.py
import numpy as np
from qldpc.circuits.surgery.y_gadget import _bb_y_pair, _locate_overlaps

def test_bb_y_pair_overlaps():
    for ov in (1, 3):
        code, x, z = _bb_y_pair(overlap=ov)
        assert code.num_qudits == 36
        W = _locate_overlaps(code, x, z)
        assert len(W) == ov
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest src/qldpc/circuits/surgery/y_gadget_test.py::test_bb_y_pair_overlaps -v`
Expected: FAIL (`ImportError: cannot import name '_bb_y_pair'`)

- [ ] **Step 3: Implement `_bb_y_pair`** (derive the `|W|=3` vector at build time
  from the cached stabilizer-row offsets; see the search recipe in the design
  doc — add ≤3 `H_X` rows to `x` and ≤3 `H_Z` rows to `z` until `|W|=3` with both
  gadget graphs cyclic, seed=0). Implementation:

```python
def _bb_y_pair(overlap: int = 1) -> tuple[CSSCode, np.ndarray, np.ndarray]:
    """BB [[36,8,4]] fixture for Ȳ on logical qubit 0 with chosen |W|.

    overlap=1: canonical reps (already single-overlap). overlap=3: add stabilizer
    rows (deterministic seed) until supp(x)∩supp(z) has size 3 — the |W|≥2
    crossing-cycle regime of arXiv:2410.02753 §III.D / main.tex §4.
    """
    import sympy
    xs, ys = sympy.symbols("x y")
    code = codes.BBCode({xs: 3, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)
    n = code.num_qudits
    LX = np.asarray(code.get_logical_ops(Pauli.X)).astype(np.uint8)
    LZ = np.asarray(code.get_logical_ops(Pauli.Z)).astype(np.uint8)
    wide = LX.shape[1] == 2 * n
    x = (LX[0][:n] if wide else LX[0]).astype(np.uint8)
    z = (LZ[0][n:] if wide else LZ[0]).astype(np.uint8)
    if overlap == 1:
        return code, x, z
    if overlap == 3:
        HX = np.asarray(code.matrix_x).astype(np.uint8)
        HZ = np.asarray(code.matrix_z).astype(np.uint8)
        rng = np.random.default_rng(0)
        for _ in range(20000):
            ax = (rng.integers(0, 2, HX.shape[0]) if rng.random() < 0.5
                  else np.zeros(HX.shape[0], int))
            az = rng.integers(0, 2, HZ.shape[0])
            xc = (x ^ (ax @ HX % 2)).astype(np.uint8)
            zc = (z ^ (az @ HZ % 2)).astype(np.uint8)
            if not xc.any() or not zc.any():
                continue
            if int(np.count_nonzero(xc.astype(bool) & zc.astype(bool))) == 3 \
               and int(xc.sum()) <= 12 and int(zc.sum()) <= 12:
                return code, xc, zc
        raise ValueError("BLOCKED: no |W|=3 BB representative found in budget")
    raise ValueError(f"overlap must be 1 or 3, got {overlap}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest src/qldpc/circuits/surgery/y_gadget_test.py::test_bb_y_pair_overlaps -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/y_gadget.py src/qldpc/circuits/surgery/y_gadget_test.py
git commit -m "test(surgery): BB [[36,8,4]] Ȳ fixture for |W| in {1,3}"
```

---

### Task 5: BB merged-code structure + distance assertions

**Files:**
- Test: `src/qldpc/circuits/surgery/y_gadget_test.py`

**Interfaces:**
- Consumes: `build_y_gadget`, `_bb_y_pair`, `_partial0_symplectic_rows`.

- [ ] **Step 1: Write the failing/asserting tests**

```python
# y_gadget_test.py
import numpy as np, galois, pytest
from qldpc.circuits.surgery.y_gadget import build_y_gadget, _bb_y_pair, _merged_incidence
GF2 = galois.GF(2)

def _kerdim(M):
    M = np.asarray(M).astype(int)
    return 0 if M.size == 0 else np.asarray(GF2(M).null_space()).shape[0]

# crossing_dim = dim(ker merged ∂_1) − dim(ker ∂_1^x) − dim(ker ∂_1^z) is the
# BASIS-INDEPENDENT count of genuine crossing cycles. (The number of mixed ROWS
# in any particular ∂_0 basis is NOT invariant — do not assert on it.)
@pytest.mark.parametrize("overlap, crossing_dim_expected", [(1, 0), (3, 2)])
def test_bb_merged_structure_and_crossing_dim(overlap, crossing_dim_expected):
    code, x, z = _bb_y_pair(overlap=overlap)
    yg = build_y_gadget(code, x=x, z=z)
    assert yg.merged_code.dimension == 7  # 8 logicals − 1 measured
    D1, k_x, k_z = _merged_incidence(yg.g_x, yg.g_z, x, z)
    d1x = np.asarray(yg.g_x.incidence).astype(int).T
    d1z = np.asarray(yg.g_z.incidence).astype(int).T
    crossing_dim = _kerdim(D1) - _kerdim(d1x) - _kerdim(d1z)
    assert crossing_dim == crossing_dim_expected  # 0 at |W|=1, |W|−1 at |W|=3

def test_bb_w1_distance_not_collapsed():
    code, x, z = _bb_y_pair(overlap=1)
    yg = build_y_gadget(code, x=x, z=z)
    d = yg.merged_code.get_distance(bound=12)  # decoder upper bound
    assert d >= 4  # collapse below d_data=4 would make the bound return < 4
```

(Verified seed-0 invariants: `|W|=1 → dim_merged 9, crossing_dim 0`; `|W|=3 →
dim_merged 16, crossing_dim 2`; both `k=7`, `d≤4`, genuine stabilizer code.)

- [ ] **Step 2: Run to verify**

Run: `pytest src/qldpc/circuits/surgery/y_gadget_test.py -k "bb_merged or bb_w1_distance" -v`
Expected: First run may reveal an off-by-one in `∂_0`/merge sizing; iterate on
Task 2/3 until PASS. (`overlap=1 → 0` crossing, `overlap=3 → 2` crossing, `k=7`,
`d≥4` are the verified targets.)

- [ ] **Step 3: (only if a test fails) Fix in Task 2/3 source, not the test**

Re-run the failing assertion; adjust `_merged_incidence` row/column ordering or
the `∂_0` embedding cross-typing until green. Do not weaken the asserted numbers.

- [ ] **Step 4: Run the full y_gadget suite**

Run: `pytest src/qldpc/circuits/surgery/y_gadget_test.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/y_gadget_test.py
git commit -m "test(surgery): BB Eq68 merged structure, crossing-cycle counts, distance"
```

---

### Task 6: Circuit compiles + measurement-fault-distance on BB

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py` (docstring/citations of
  `build_single_y_ppm_circuit`; no structural change expected — the Y-phase loop
  at lines ~669-693 already handles every non-CSS row)
- Test: `src/qldpc/circuits/surgery/circuit_single_y_test.py`

**Interfaces:**
- Consumes: `build_single_y_ppm_circuit(yg, *, rounds, data_init=None,
  noise_model=None)`, `_bb_y_pair`, `build_y_gadget`.

- [ ] **Step 1: Write the failing tests (re-target to BB; drop Steane fixture)**

```python
# circuit_single_y_test.py  (replace Steane fixture usage)
import numpy as np, pytest, stim
from qldpc.circuits.surgery.y_gadget import _bb_y_pair, build_y_gadget

@pytest.mark.parametrize("overlap", [1, 3])
def test_bb_single_y_circuit_compiles_dem(overlap):
    from qldpc.circuits.surgery import build_single_y_ppm_circuit
    code, x, z = _bb_y_pair(overlap=overlap)
    yg = build_y_gadget(code, x=x, z=z)
    circuit = build_single_y_ppm_circuit(yg, rounds=4, data_init=None)
    assert isinstance(circuit, stim.Circuit)
    dem = circuit.detector_error_model()
    assert dem.num_detectors > 0

@pytest.mark.parametrize("overlap", [1, 3])
def test_bb_single_y_no_undetectable_obs_flip(overlap):
    from qldpc.circuits.noise_model import DepolarizingNoiseModel
    from qldpc.circuits.surgery import build_single_y_ppm_circuit, keep_only_observable
    code, x, z = _bb_y_pair(overlap=overlap)
    yg = build_y_gadget(code, x=x, z=z)
    circuit = build_single_y_ppm_circuit(
        yg, rounds=4, data_init=None,
        noise_model=DepolarizingNoiseModel(0.001, include_idling_error=False),
    )
    circuit = keep_only_observable(circuit, keep_idx=0)
    dem = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
    offenders = [
        inst for inst in dem.flattened()
        if inst.type == "error"
        and any(t.is_logical_observable_id() for t in inst.targets_copy())
        and not any(t.is_relative_detector_id() for t in inst.targets_copy())
    ]
    assert not offenders, f"{len(offenders)} undetectable obs0-flip term(s): {offenders[:3]}"
```

- [ ] **Step 2: Run to verify compilation first**

Run: `pytest src/qldpc/circuits/surgery/circuit_single_y_test.py::test_bb_single_y_circuit_compiles_dem -v`
Expected: PASS (the circuit already consumes `yg.merged_code` generically; if it
raises on `data_init=None` or on a mixed `∂_0` row, fix in `circuit.py` — but the
Y-phase loop already covers per-column CX/CY/CZ for any mixed row).

- [ ] **Step 3: Run the fault-distance check; if it fails, debug the readout**

Run: `pytest src/qldpc/circuits/surgery/circuit_single_y_test.py::test_bb_single_y_no_undetectable_obs_flip -v`
Expected: PASS. **If it fails**, the offending minimal chain is a
circuit-readout issue (obs0 definition / detector coverage), NOT a missing
bridge: print `offenders[:3]`, identify whether the flip rides a single
`∂_0`/`y_v` measurement, and add the missing detector or fix the obs0 row set in
`_ybar_obs0_rows`. Keep iterating until green; do not re-introduce SkipTree.

- [ ] **Step 4: Update `build_single_y_ppm_circuit` docstring/citations**

Replace arXiv:2407.18393 §3.2/§3.7 + Remark-23 narrative with arXiv:2410.02753
§III.C (product of new stabilizers = `L`; `R≥d` rounds + decode) + main.tex §4.6.

- [ ] **Step 5: Commit**

```bash
git add src/qldpc/circuits/surgery/circuit.py src/qldpc/circuits/surgery/circuit_single_y_test.py
git commit -m "test(surgery): BB Ȳ circuit compiles + no undetectable obs0-flip (Eq68)"
```

---

### Task 7: Citation + dead-reference cleanup sweep

**Files:**
- Modify: `src/qldpc/circuits/surgery/y_gadget.py`,
  `src/qldpc/circuits/surgery/circuit_single_y_test.py`,
  `src/qldpc/circuits/surgery/y_gadget_test.py`

- [ ] **Step 1: Find stragglers**

Run:
```bash
grep -rn "2407.18393\|SkipTree\|2410.03628\|Remark 23\|§3.7\|3.2 readout\|q1\b\|q0\b" \
  src/qldpc/circuits/surgery/y_gadget.py \
  src/qldpc/circuits/surgery/circuit_single_y_test.py \
  src/qldpc/circuits/surgery/y_gadget_test.py
```
Expected: only intended hits remain.

- [ ] **Step 2: Rewrite remaining docstrings/comments** to arXiv:2410.02753
  §III.C/§III.D + main.tex §4 (Ide, Gowda, Nadkarni, Dauphinais — full form).
  Rename leftover `q0`/`q1` prose to `W` / `y_v`.

- [ ] **Step 3: Run the whole surgery suite**

Run: `pytest src/qldpc/circuits/surgery/ -q`
Expected: PASS (note: `circuit_single_y_test.py` Steane tests are replaced;
ensure no test still imports `_locate_overlap`, `_steane_logical_y_eigenstate_prep`
for BB, or `yg.bridge`/`yg.q0`).

- [ ] **Step 4: Commit**

```bash
git add src/qldpc/circuits/surgery/
git commit -m "docs(surgery): unify Ȳ-PPM citations to arXiv:2410.02753 + main.tex §4"
```

---

## Self-Review

**Spec coverage:**
- Core principle (compute `W`) → Task 1. ✓
- `∂_0 = ker(merged ∂_1)`, cross-typed, general `|W|` → Task 2 + Task 3. ✓
- Per-system Cheeger boost (§4.7) → Task 3 (boost step). ✓
- Drop bridge/SkipTree → Task 3 (field + import removal) + Task 7 (sweep). ✓
- BB fixture, both `|W|` → Task 4. ✓
- Merged `k=7`, distance, crossing-cycle counts → Task 5. ✓
- Circuit compiles + measurement-fault-distance (Acceptance #3/#4) → Task 6. ✓
- Citation unification (Acceptance #5) → Task 6 + Task 7. ✓

**Placeholder scan:** Task 6 Step 3 is conditional debugging, but it names the
concrete failure mode and fix location (no vague "handle errors"). All code steps
contain real code. ✓

**Type consistency:** `_locate_overlaps` (tuple) used consistently; `_merged_incidence`
returns `(∂_1, k_x, k_z)` consumed by `_partial0_symplectic_rows`; `YGadgetLayout`
field set (`W`, no `bridge`/`q0`) consistent across Tasks 3–6; `build_y_gadget`
signature unchanged (`code, *, x, z`). ✓

**Known risk:** Task 5/6 assert distance / fault-distance numbers verified only
at the *code* level so far (`k=7`, `d≤4`, `|W|=3→2` crossing cycles). The DEM
measurement-fault-distance (Task 6 Step 3) is the one genuinely unproven gate; if
it cannot be met it is documented as a circuit-readout bug per the spec, not a
silent pass.
