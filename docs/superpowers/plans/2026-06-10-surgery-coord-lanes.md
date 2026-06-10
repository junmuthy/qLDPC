# Surgery QUBIT_COORDS + DETECTOR Coord Lanes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace surgery's opaque `(0, 0, kk)` DETECTOR coords and 2-row QUBIT_COORDS layout with semantic 6-lane encoding (7 lanes for joint PPM): y=0 data, y=1 data H_X ancillas, y=2 data H_Z ancillas, y=3 κ, y=4 χ, y=5 G, y=6 bridge.

**Architecture:** Four TDD cycles in a single atomic commit. Add two new private helpers to `circuit.py` (`_surgery_qubit_coordinates`, `_check_lane_index_map`), replace the `get_qubit_coordinates(...)` call sites in `build_single_ppm_circuit` / `build_joint_ppm_circuit`, and update the 6 DETECTOR emit sites to use `(t, lane, idx)` coords inherited from the lane map. Default behavior preserved: same observables, same detectors, same circuits — only the coord *tuple values* change.

**Tech Stack:** Python 3, `stim` (circuit and coord conventions), `pytest`, `numpy`.

**Spec:** `docs/superpowers/specs/2026-06-10-surgery-coord-lanes-design.md`

---

## File Structure

```
src/qldpc/circuits/surgery/
├── circuit.py              (modify: add 2 helpers, swap 2 call sites, update 6 DETECTOR emit sites)
└── _test_circuit.py        (add 4 new tests pinning the new coord scheme)
```

No new files. Two existing files modified, single atomic commit at the end of Task 5.

**IMPORTANT:** Tasks 1-4 do NOT commit individually. Only Task 5 (final verification) commits the entire change. The earlier reviews of this branch established a one-commit scope for this feature.

---

## Task 1: Single PPM QUBIT_COORDS — write test, then implement

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py` (add `_surgery_qubit_coordinates`, swap call site at `build_single_ppm_circuit`)
- Modify: `src/qldpc/circuits/surgery/_test_circuit.py` (add `test_qubit_coords_layout_steane`)

- [ ] **Step 1: Write the failing test for Steane single-PPM QUBIT_COORDS layout**

Append to `src/qldpc/circuits/surgery/_test_circuit.py`:

```python
def test_qubit_coords_layout_steane():
    """Steane single-PPM circuit emits QUBIT_COORDS in 6 semantic lanes.

    y=0 data (Steane ids 0..6), y=1 data H_X ancillas (3), y=2 data H_Z
    ancillas (3), y=3 κ ancillas (3), y=4 χ ancillas (3), y=5 G ancilla (1).
    """
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    circuit = build_single_ppm_circuit(g, rounds=1, noise_model=None)

    # Parse QUBIT_COORDS lines: each line is "QUBIT_COORDS(x, y) qubit_id"
    coord_map: dict[int, tuple[int, int]] = {}
    for line in str(circuit).splitlines():
        line = line.strip()
        if not line.startswith("QUBIT_COORDS"):
            continue
        # "QUBIT_COORDS(x, y) qid" — parse "(x, y)" and qid
        head, qid_str = line.rsplit(" ", 1)
        tup = head[len("QUBIT_COORDS("):-1]
        x_str, y_str = [t.strip() for t in tup.split(",")]
        coord_map[int(qid_str)] = (int(x_str), int(y_str))

    expected = {
        # data qubits on y=0
        0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0),
        4: (4, 0), 5: (5, 0), 6: (6, 0),
        # κ ancillas on y=3
        7: (0, 3), 8: (1, 3), 9: (2, 3),
        # data H_X ancillas on y=1 (Steane has 3 X-checks)
        10: (0, 1), 11: (1, 1), 12: (2, 1),
        # χ ancillas on y=4 (basis=X gadget: χ in checks_x[m_X:])
        13: (0, 4), 14: (1, 4), 15: (2, 4),
        # data H_Z ancillas on y=2
        16: (0, 2), 17: (1, 2), 18: (2, 2),
        # G ancilla on y=5 (gauge-fix, basis=X: G in checks_z[m_Z:])
        19: (0, 5),
    }
    assert coord_map == expected, f"\nexpected: {expected}\ngot:      {coord_map}"
```

- [ ] **Step 2: Run the test — expect FAIL**

Run:
```bash
.venv/bin/python -m pytest src/qldpc/circuits/surgery/_test_circuit.py::test_qubit_coords_layout_steane -xvs
```

Expected: `FAILED`. The existing `get_qubit_coordinates` puts all data ids on y=0 and all check ids on y=1, so the assertion mismatches.

- [ ] **Step 3: Add `_surgery_qubit_coordinates` to `circuit.py`**

In `src/qldpc/circuits/surgery/circuit.py`, ADD this function above `build_single_ppm_circuit` (placement: between `keep_only_observable` and `build_single_ppm_circuit`). It must work for both single PPM (joint=None) and joint PPM (joint != None, used in Task 4):

```python
def _surgery_qubit_coordinates(
    gadget: GadgetLayout,
    qubit_ids: QubitIDs,
    *,
    joint: tuple[GadgetLayout, Bridge, bool] | None = None,
) -> stim.Circuit:
    """Emit QUBIT_COORDS in surgery's 6/7-lane semantic layout.

    Lanes:
      y=0  data qubits         (originally data + κ + bridge in qubit_ids.data
                                slot; we split them across y=0/3/6 here).
      y=1  data H_X ancillas   (checks_x[:m_X])
      y=2  data H_Z ancillas   (checks_z[:m_Z])
      y=3  κ ancillas
      y=4  χ ancillas          (basis=X: checks_x[m_X:]; basis=Z: checks_z[m_Z:])
      y=5  G ancillas          (basis=X: checks_z[m_Z:]; basis=Z: checks_x[m_X:])
      y=6  bridge data + bridge cycle ancillas (joint PPM only)

    `joint=None` → single PPM. Otherwise pass (g_r, bridge, intercode).
    """
    circuit = stim.Circuit()

    if joint is None:
        g_l = gadget
        g_r = None
        bridge = None
        intercode = False
    else:
        g_l = gadget
        g_r, bridge, intercode = joint

    # Sizes for left side (always present).
    n_l = g_l.code.num_qudits
    m_X_l = g_l.code.matrix_x.shape[0]
    m_Z_l = g_l.code.matrix_z.shape[0]
    chi_l = len(g_l.V0)
    G_l = g_l.G.shape[0]
    k_l = len(g_l.kappa_qubits)

    # Sizes for right side (joint+intercode only — intracode shares data).
    if joint is not None and intercode:
        n_r = g_r.code.num_qudits
        m_X_r = g_r.code.matrix_x.shape[0]
        m_Z_r = g_r.code.matrix_z.shape[0]
        chi_r = len(g_r.V0)
        G_r = g_r.G.shape[0]
        k_r = len(g_r.kappa_qubits)
    elif joint is not None:  # intracode: data shared, ancillas separate per gadget
        n_r = 0
        m_X_r = m_Z_r = 0  # data checks not duplicated for intracode
        chi_r = len(g_r.V0)
        G_r = g_r.G.shape[0]
        # for intracode bridge code, kappa is the augmented one
        k_l = bridge.g_l_aug.F.shape[0]
        k_r = bridge.g_r_aug.F.shape[0]
    else:
        n_r = 0
        m_X_r = m_Z_r = 0
        chi_r = G_r = k_r = 0

    # For joint cases, kappa counts come from the augmented gadgets carried
    # by the bridge — they may exceed the bare gadget's |C_0| if cellulation
    # added extra rows.
    if joint is not None:
        k_l = bridge.g_l_aug.F.shape[0]
        k_r = bridge.g_r_aug.F.shape[0]

    n_data_total = n_l + n_r
    w = bridge.width if joint is not None else 0

    # y=0 data
    for i in range(n_data_total):
        circuit.append("QUBIT_COORDS", qubit_ids.data[i], (i, 0))

    # y=3 κ
    for i in range(k_l + k_r):
        circuit.append("QUBIT_COORDS", qubit_ids.data[n_data_total + i], (i, 3))

    # y=6 bridge data (joint PPM only)
    for i in range(w):
        circuit.append(
            "QUBIT_COORDS",
            qubit_ids.data[n_data_total + k_l + k_r + i],
            (i, 6),
        )

    # X-check ancillas: data H_X on y=1, then either χ on y=4 (basis=X) or G on y=5 (basis=Z).
    is_basis_x = g_l.basis is Pauli.X
    m_X_total = m_X_l + m_X_r
    chi_total = chi_l + chi_r
    G_total = G_l + G_r

    for i in range(m_X_total):
        circuit.append("QUBIT_COORDS", qubit_ids.checks_x[i], (i, 1))
    if is_basis_x:
        # χ rows on y=4 (within checks_x)
        for i in range(chi_total):
            circuit.append(
                "QUBIT_COORDS", qubit_ids.checks_x[m_X_total + i], (i, 4),
            )
    else:
        # G rows on y=5 (within checks_x for basis=Z)
        for i in range(G_total):
            circuit.append(
                "QUBIT_COORDS", qubit_ids.checks_x[m_X_total + i], (i, 5),
            )

    # Z-check ancillas: data H_Z on y=2, then either G on y=5 (basis=X) or χ on y=4 (basis=Z).
    m_Z_total = m_Z_l + m_Z_r
    for i in range(m_Z_total):
        circuit.append("QUBIT_COORDS", qubit_ids.checks_z[i], (i, 2))
    if is_basis_x:
        for i in range(G_total):
            circuit.append(
                "QUBIT_COORDS", qubit_ids.checks_z[m_Z_total + i], (i, 5),
            )
    else:
        for i in range(chi_total):
            circuit.append(
                "QUBIT_COORDS", qubit_ids.checks_z[m_Z_total + i], (i, 4),
            )

    # Joint PPM: bridge cycle ancillas on y=6 (sharing the row with bridge data).
    if joint is not None and w > 1:
        # The new cycle checks live at the end of checks_x (basis=Z) or
        # checks_z (basis=X). They're (w - 1) of them.
        if is_basis_x:
            cycle_check_ids = qubit_ids.checks_z[m_Z_total + G_total:]
        else:
            cycle_check_ids = qubit_ids.checks_x[m_X_total + G_total:]
        for i, cid in enumerate(cycle_check_ids):
            circuit.append("QUBIT_COORDS", cid, (i, 6))

    return circuit
```

- [ ] **Step 4: Replace the call site in `build_single_ppm_circuit`**

In `src/qldpc/circuits/surgery/circuit.py`, locate the `build_single_ppm_circuit` function (around line 59-95). Change line ~88 from:

```python
    circuit = get_qubit_coordinates(qubit_ids.data, qubit_ids.check)
```

to:

```python
    circuit = _surgery_qubit_coordinates(gadget, qubit_ids)
```

Do NOT change the import statement yet (the import is still used by `build_joint_ppm_circuit` until Task 4).

- [ ] **Step 5: Run the test — expect PASS**

Run:
```bash
.venv/bin/python -m pytest src/qldpc/circuits/surgery/_test_circuit.py::test_qubit_coords_layout_steane -xvs
```

Expected: `PASSED`.

- [ ] **Step 6: Verify no surgery suite regression**

Run:
```bash
.venv/bin/python -m pytest src/qldpc/circuits/surgery/ -q | tail -3
```

Expected: 97 passed (96 pre-existing + 1 new). NO commits.

---

## Task 2: Single PPM DETECTOR coords — write test, then implement

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py` (add `_check_lane_index_map`, update DETECTOR emit sites in `_surgery_qec_cycle` and `_surgery_final_detectors`)
- Modify: `src/qldpc/circuits/surgery/_test_circuit.py` (add `test_detector_coords_steane_round_1_reliable`)

- [ ] **Step 1: Write the failing test for DETECTOR coord lanes**

Append to `src/qldpc/circuits/surgery/_test_circuit.py`:

```python
def test_detector_coords_steane_round_1_reliable():
    """Steane single-PPM round-1 reliable detectors have lane ∈ {1, 5}.

    Round-1 reliable for basis=X gadget: 3 data H_X checks (lane=1) + 1 G
    check (lane=5). No χ or data H_Z because those aren't deterministic
    on the protocol-default |+⟩ init.
    """
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x)
    circuit = build_single_ppm_circuit(g, rounds=1, noise_model=None)

    detector_coords: set[tuple[int, int, int]] = set()
    for line in str(circuit).splitlines():
        line = line.strip()
        if not line.startswith("DETECTOR"):
            continue
        # "DETECTOR(t, lane, idx) rec[-N] ..." — extract the tuple
        head = line.split(")")[0]
        tup = head[len("DETECTOR("):]
        parts = [int(p.strip()) for p in tup.split(",")]
        detector_coords.add(tuple(parts))

    expected = {(0, 1, 0), (0, 1, 1), (0, 1, 2), (0, 5, 0)}
    assert detector_coords == expected, (
        f"\nexpected: {expected}\ngot:      {detector_coords}"
    )
```

- [ ] **Step 2: Run the test — expect FAIL**

Run:
```bash
.venv/bin/python -m pytest src/qldpc/circuits/surgery/_test_circuit.py::test_detector_coords_steane_round_1_reliable -xvs
```

Expected: `FAILED`. Current detector coords are `(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 9)`.

- [ ] **Step 3: Add `_check_lane_index_map` to `circuit.py`**

In `src/qldpc/circuits/surgery/circuit.py`, ADD this function right after `_surgery_qubit_coordinates` (added in Task 1):

```python
def _check_lane_index_map(
    gadget: GadgetLayout,
    qubit_ids: QubitIDs,
    *,
    joint: tuple[GadgetLayout, Bridge, bool] | None = None,
) -> dict[int, tuple[int, int]]:
    """Build a {check_id: (lane, idx)} map matching the QUBIT_COORDS layout.

    Lanes for checks (idx is x position within lane):
      lane=1: data H_X check ancillas (checks_x[:m_X_total])
      lane=2: data H_Z check ancillas (checks_z[:m_Z_total])
      lane=4: χ check ancillas (basis=X: checks_x[m_X:]; basis=Z: checks_z[m_Z:])
      lane=5: G check ancillas (basis=X: checks_z[m_Z:]; basis=Z: checks_x[m_X:])
      lane=6: bridge cycle check ancillas (joint PPM only).
    """
    is_basis_x = gadget.basis is Pauli.X

    if joint is None:
        m_X_total = gadget.code.matrix_x.shape[0]
        m_Z_total = gadget.code.matrix_z.shape[0]
        chi_total = len(gadget.V0)
        G_total = gadget.G.shape[0]
    else:
        g_r, bridge, intercode = joint
        m_X_total = gadget.code.matrix_x.shape[0]
        m_Z_total = gadget.code.matrix_z.shape[0]
        if intercode:
            m_X_total += g_r.code.matrix_x.shape[0]
            m_Z_total += g_r.code.matrix_z.shape[0]
        chi_total = len(gadget.V0) + len(g_r.V0)
        G_total = gadget.G.shape[0] + g_r.G.shape[0]

    result: dict[int, tuple[int, int]] = {}

    # data H_X on lane=1
    for i in range(m_X_total):
        result[qubit_ids.checks_x[i]] = (1, i)
    # data H_Z on lane=2
    for i in range(m_Z_total):
        result[qubit_ids.checks_z[i]] = (2, i)

    if is_basis_x:
        # χ on lane=4 in checks_x[m_X:]; G on lane=5 in checks_z[m_Z:]
        for i in range(chi_total):
            result[qubit_ids.checks_x[m_X_total + i]] = (4, i)
        for i in range(G_total):
            result[qubit_ids.checks_z[m_Z_total + i]] = (5, i)
    else:
        # G on lane=5 in checks_x[m_X:]; χ on lane=4 in checks_z[m_Z:]
        for i in range(G_total):
            result[qubit_ids.checks_x[m_X_total + i]] = (5, i)
        for i in range(chi_total):
            result[qubit_ids.checks_z[m_Z_total + i]] = (4, i)

    # Joint PPM bridge cycle ancillas on lane=6.
    if joint is not None:
        if is_basis_x:
            cycle_ids = qubit_ids.checks_z[m_Z_total + G_total:]
        else:
            cycle_ids = qubit_ids.checks_x[m_X_total + G_total:]
        for i, cid in enumerate(cycle_ids):
            result[cid] = (6, i)

    return result
```

- [ ] **Step 4: Thread the lane map into `_surgery_qec_cycle` and `_surgery_final_detectors`**

In `src/qldpc/circuits/surgery/circuit.py`:

(a) Locate `_surgery_qec_cycle` (currently around line 615). Build the lane map once at the start of the function:

Find this block (around line 635-650):
```python
    strategy = EdgeColoring()
    one_round, round_measurement_record = strategy.get_circuit(merged_code, qubit_ids)
    reliable = set(_classify_reliable_round1_checks(gadget, qubit_ids))
    all_check_ids = qubit_ids.check

    circuit = stim.Circuit()
    measurement_record = MeasurementRecord()
    detector_record = DetectorRecord()

    # Round 1: emit DETECTORs only for reliable checks.
    circuit += one_round
    measurement_record.append(round_measurement_record)
    for kk, check_id in enumerate(all_check_ids):
        if check_id in reliable:
            circuit.append("DETECTOR", [measurement_record.get_target_rec(check_id)], (0, 0, kk))
```

Modify to:
```python
    strategy = EdgeColoring()
    one_round, round_measurement_record = strategy.get_circuit(merged_code, qubit_ids)
    reliable = set(_classify_reliable_round1_checks(gadget, qubit_ids))
    all_check_ids = qubit_ids.check
    lane_idx = _check_lane_index_map(gadget, qubit_ids)

    circuit = stim.Circuit()
    measurement_record = MeasurementRecord()
    detector_record = DetectorRecord()

    # Round 1: emit DETECTORs only for reliable checks.
    circuit += one_round
    measurement_record.append(round_measurement_record)
    for check_id in all_check_ids:
        if check_id in reliable:
            lane, idx = lane_idx[check_id]
            circuit.append("DETECTOR", [measurement_record.get_target_rec(check_id)], (0, lane, idx))
```

(b) Update the repeat-block detector emission. Find this block (around line 655-665):
```python
        repeat_circuit.append("SHIFT_COORDS", [], (1, 0, 0))
        for kk, check_id in enumerate(all_check_ids):
            repeat_circuit.append("DETECTOR", [
                measurement_record.get_target_rec(check_id, -1),
                measurement_record.get_target_rec(check_id, -2),
            ], (0, 0, kk))
```

Modify to:
```python
        repeat_circuit.append("SHIFT_COORDS", [], (1, 0, 0))
        for check_id in all_check_ids:
            lane, idx = lane_idx[check_id]
            repeat_circuit.append("DETECTOR", [
                measurement_record.get_target_rec(check_id, -1),
                measurement_record.get_target_rec(check_id, -2),
            ], (0, lane, idx))
```

(c) Locate `_surgery_final_detectors` (currently around line 713). Build the lane map at the start and update both emit sites. Find the `_emit_detector` inner function (around line 733-737):

```python
    def _emit_detector(stab_row: np.ndarray, check_id: int, det_idx: int) -> None:
        supp = np.where(stab_row)[0]
        targets = [measurement_record.get_target_rec(qubit_ids.data[q]) for q in supp]
        targets.append(measurement_record.get_target_rec(check_id, -1))
        circuit.append("DETECTOR", targets, (0, 0, det_idx))
```

Modify to:
```python
    lane_idx = _check_lane_index_map(gadget, qubit_ids)

    def _emit_detector(stab_row: np.ndarray, check_id: int) -> None:
        supp = np.where(stab_row)[0]
        targets = [measurement_record.get_target_rec(qubit_ids.data[q]) for q in supp]
        targets.append(measurement_record.get_target_rec(check_id, -1))
        lane, idx = lane_idx[check_id]
        circuit.append("DETECTOR", targets, (0, lane, idx))
```

(d) Update the 4 callers of `_emit_detector` inside `_surgery_final_detectors` (currently around lines 741-754). Drop the `det_idx` argument. The block becomes:

Find:
```python
    if gadget.basis is Pauli.X:
        # data H_X rows (X-checks indices [:m_X])
        for kk in range(m_X):
            _emit_detector(HX[kk], qubit_ids.checks_x[kk], kk)
        # G rows (Z-checks indices [m_Z:])
        for offset, kk in enumerate(range(m_Z, HZ.shape[0])):
            _emit_detector(HZ[kk], qubit_ids.checks_z[kk], m_X + offset)
    else:  # Pauli.Z (symmetric: chi in HZ, G in HX)
        for kk in range(m_Z):
            _emit_detector(HZ[kk], qubit_ids.checks_z[kk], kk)
        for offset, kk in enumerate(range(m_X, HX.shape[0])):
            _emit_detector(HX[kk], qubit_ids.checks_x[kk], m_Z + offset)
```

Modify to:
```python
    if gadget.basis is Pauli.X:
        for kk in range(m_X):
            _emit_detector(HX[kk], qubit_ids.checks_x[kk])
        for kk in range(m_Z, HZ.shape[0]):
            _emit_detector(HZ[kk], qubit_ids.checks_z[kk])
    else:  # Pauli.Z (symmetric: chi in HZ, G in HX)
        for kk in range(m_Z):
            _emit_detector(HZ[kk], qubit_ids.checks_z[kk])
        for kk in range(m_X, HX.shape[0]):
            _emit_detector(HX[kk], qubit_ids.checks_x[kk])
```

- [ ] **Step 5: Run the test — expect PASS**

Run:
```bash
.venv/bin/python -m pytest src/qldpc/circuits/surgery/_test_circuit.py::test_detector_coords_steane_round_1_reliable -xvs
```

Expected: `PASSED`.

- [ ] **Step 6: Run the full single-PPM-related tests to confirm no regression**

Run:
```bash
.venv/bin/python -m pytest src/qldpc/circuits/surgery/_test_circuit.py -q | tail -3
```

Expected: still all passing (no joint test broken yet; joint PPM updated in Task 4). NO commits.

---

## Task 3: basis=Z preserves χ/G lanes — verify

**Files:**
- Modify: `src/qldpc/circuits/surgery/_test_circuit.py` (add `test_detector_coords_basis_z_preserves_lane_semantics`)

- [ ] **Step 1: Write the basis=Z lane-preservation test**

Append to `src/qldpc/circuits/surgery/_test_circuit.py`:

```python
def test_detector_coords_basis_z_preserves_lane_semantics():
    """For basis=Z gadget on Steane, round-1 reliable detectors are on lane 2 (data H_Z) + lane 5 (G).

    Steane logical-Z has weight 3, so basis=Z gadget has m_Z=3 data H_Z
    checks + ≥1 G check. Reliable round-1 checks are deterministic given
    data |0⟩ init: data H_Z (Z-stabilizers always +1 on |0⟩^n) and G
    (gauge-fix Z-stabilizer on κ qubits).

    Critically, the χ/G lane numbers stay stable across basis swap:
    G is always lane=5 even though for basis=Z it lives in checks_x[m_X:],
    not checks_z[m_Z:].
    """
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.circuit import build_single_ppm_circuit

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    circuit = build_single_ppm_circuit(g, rounds=1, noise_model=None)

    detector_lanes: set[int] = set()
    for line in str(circuit).splitlines():
        line = line.strip()
        if not line.startswith("DETECTOR"):
            continue
        head = line.split(")")[0]
        tup = head[len("DETECTOR("):]
        parts = [int(p.strip()) for p in tup.split(",")]
        detector_lanes.add(parts[1])  # the lane component

    # Expect only lane 2 (data H_Z) and lane 5 (G); NO lane 1, 4, or other.
    assert detector_lanes == {2, 5}, (
        f"basis=Z round-1 reliable lanes: expected {{2, 5}}, got {detector_lanes}"
    )
```

- [ ] **Step 2: Run the test**

Run:
```bash
.venv/bin/python -m pytest src/qldpc/circuits/surgery/_test_circuit.py::test_detector_coords_basis_z_preserves_lane_semantics -xvs
```

Expected: `PASSED`. (The implementation in Task 2 already handles basis=Z; this test just verifies that the lane numbers are basis-symmetric.)

If FAILED, the lane-map computation in Task 2 Step 3 is incorrect for basis=Z. Re-check `_check_lane_index_map`: for `is_basis_x = False`, the χ ancillas are at `checks_z[m_Z_total:]` (lane=4) and G ancillas are at `checks_x[m_X_total:]` (lane=5).

- [ ] **Step 3: NO commits.**

---

## Task 4: Joint PPM — write test, then implement

**Files:**
- Modify: `src/qldpc/circuits/surgery/circuit.py` (swap call site in `build_joint_ppm_circuit`, update `_surgery_qec_cycle_joint` + `_surgery_final_detectors_joint` DETECTOR emit sites, drop unused `get_qubit_coordinates` import if no longer used)
- Modify: `src/qldpc/circuits/surgery/_test_circuit.py` (add `test_joint_ppm_qubit_coords_intercode_layout`)

- [ ] **Step 1: Write the joint-PPM layout test**

Append to `src/qldpc/circuits/surgery/_test_circuit.py`:

```python
def test_joint_ppm_qubit_coords_intercode_layout():
    """Intercode joint Z̄⊗Z̄ on two Steane copies: QUBIT_COORDS lanes correct.

    n_l = n_r = 7; left data on y=0 at x=0..6; right data on y=0 at x=7..13.
    κ ancillas on y=3. Bridge data + cycle ancillas on y=6.
    """
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.circuits.surgery.bridge import build_bridge
    from qldpc.circuits.surgery.circuit import build_joint_ppm_circuit

    c1, c2 = codes.SteaneCode(), codes.SteaneCode()
    z1 = np.asarray(c1.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    z2 = np.asarray(c2.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g1 = build_gadget(c1, z1, basis=Pauli.Z)
    g2 = build_gadget(c2, z2, basis=Pauli.Z)
    bridge = build_bridge(g1, g2)
    circuit, _ = build_joint_ppm_circuit(
        g1, g2, bridge, rounds=1, noise_model=None,
    )

    # Parse QUBIT_COORDS and group qubit ids by y.
    by_y: dict[int, list[tuple[int, int]]] = {}
    for line in str(circuit).splitlines():
        line = line.strip()
        if not line.startswith("QUBIT_COORDS"):
            continue
        head, qid_str = line.rsplit(" ", 1)
        tup = head[len("QUBIT_COORDS("):-1]
        x_str, y_str = [t.strip() for t in tup.split(",")]
        x, y = int(x_str), int(y_str)
        qid = int(qid_str)
        by_y.setdefault(y, []).append((x, qid))

    # y=0 must have n_l + n_r = 14 qubits at x=0..13.
    y0 = sorted(by_y.get(0, []))
    assert len(y0) == 14, f"y=0 expected 14 data qubits, got {len(y0)}"
    assert [x for x, _ in y0] == list(range(14)), (
        f"y=0 x positions: expected 0..13, got {[x for x, _ in y0]}"
    )

    # y=3 must have κ_l + κ_r qubits (depends on bridge augmentation).
    y3 = sorted(by_y.get(3, []))
    assert len(y3) >= 2, f"y=3 expected at least 2 κ qubits, got {len(y3)}"

    # y=6 must have bridge data (= bridge.width) at x=0..w-1, plus
    # cycle ancillas (= bridge.width - 1) at x=0..w-2.
    y6 = sorted(by_y.get(6, []))
    w = bridge.width
    expected_y6_count = w + max(0, w - 1)  # bridge data + cycle ancillas
    assert len(y6) == expected_y6_count, (
        f"y=6 expected {expected_y6_count} qubits (w={w} bridge data + w-1 cycle ancillas), got {len(y6)}"
    )
```

- [ ] **Step 2: Run the test — expect FAIL**

Run:
```bash
.venv/bin/python -m pytest src/qldpc/circuits/surgery/_test_circuit.py::test_joint_ppm_qubit_coords_intercode_layout -xvs
```

Expected: `FAILED`. `build_joint_ppm_circuit` still calls the old `get_qubit_coordinates`, so y=0 has 14 + κ + bridge qubits all squashed together rather than split across lanes.

- [ ] **Step 3: Replace the call site in `build_joint_ppm_circuit`**

In `src/qldpc/circuits/surgery/circuit.py`, locate `build_joint_ppm_circuit` (around line 303). Find this line (around line 351):

```python
    circuit = get_qubit_coordinates(qubit_ids.data, qubit_ids.check)
```

Modify to:
```python
    circuit = _surgery_qubit_coordinates(
        g_l, qubit_ids, joint=(g_r, bridge, intercode),
    )
```

(The `g_r`, `bridge`, `intercode` variables are already defined in `build_joint_ppm_circuit`'s scope.)

- [ ] **Step 4: Update DETECTOR emit sites in `_surgery_qec_cycle_joint`**

In `src/qldpc/circuits/surgery/circuit.py`, locate `_surgery_qec_cycle_joint` (around line 433). Build the lane map at the start. Find this block (around line 453-471):

```python
    reliable = set(_classify_reliable_round1_checks_joint(g_l, g_r, qubit_ids, intercode=intercode))
    all_check_ids = qubit_ids.check

    circuit = stim.Circuit()
    measurement_record = MeasurementRecord()
    detector_record = DetectorRecord()

    # Round 1: emit DETECTORs only for reliable checks.
    circuit += one_round
    measurement_record.append(round_measurement_record)
    for kk, check_id in enumerate(all_check_ids):
        if check_id in reliable:
            circuit.append("DETECTOR", [measurement_record.get_target_rec(check_id)], (0, 0, kk))
```

Modify to (insert `lane_idx` and replace both DETECTOR emit sites):

```python
    reliable = set(_classify_reliable_round1_checks_joint(g_l, g_r, qubit_ids, intercode=intercode))
    all_check_ids = qubit_ids.check
    lane_idx = _check_lane_index_map(
        g_l, qubit_ids, joint=(g_r, bridge, intercode),
    )

    circuit = stim.Circuit()
    measurement_record = MeasurementRecord()
    detector_record = DetectorRecord()

    # Round 1: emit DETECTORs only for reliable checks.
    circuit += one_round
    measurement_record.append(round_measurement_record)
    for check_id in all_check_ids:
        if check_id in reliable:
            lane, idx = lane_idx[check_id]
            circuit.append("DETECTOR", [measurement_record.get_target_rec(check_id)], (0, lane, idx))
```

**IMPORTANT:** The `_surgery_qec_cycle_joint` function signature must have access to `bridge`. Check the current signature (around line 433-439) — it currently takes `(g_l, g_r, joint_code, *, num_rounds, qubit_ids, intercode)`. Add `bridge: Bridge` as a parameter:

Change signature from:
```python
def _surgery_qec_cycle_joint(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    joint_code: CSSCode,
    *,
    num_rounds: int,
    qubit_ids: QubitIDs,
    intercode: bool,
) -> tuple[stim.Circuit, MeasurementRecord, DetectorRecord]:
```

to:

```python
def _surgery_qec_cycle_joint(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    joint_code: CSSCode,
    bridge: Bridge,
    *,
    num_rounds: int,
    qubit_ids: QubitIDs,
    intercode: bool,
) -> tuple[stim.Circuit, MeasurementRecord, DetectorRecord]:
```

In `build_joint_ppm_circuit`, find the call to `_surgery_qec_cycle_joint` (around line 356-359):

```python
    qec_cycle, measurement_record, _ = _surgery_qec_cycle_joint(
        g_l, g_r, joint_code, num_rounds=rounds, qubit_ids=qubit_ids,
        intercode=intercode,
    )
```

Update to:

```python
    qec_cycle, measurement_record, _ = _surgery_qec_cycle_joint(
        g_l, g_r, joint_code, bridge, num_rounds=rounds, qubit_ids=qubit_ids,
        intercode=intercode,
    )
```

Now update the repeat-block detector emission inside `_surgery_qec_cycle_joint` (around line 466-471):

```python
        repeat_circuit.append("SHIFT_COORDS", [], (1, 0, 0))
        for kk, check_id in enumerate(all_check_ids):
            repeat_circuit.append("DETECTOR", [
                measurement_record.get_target_rec(check_id, -1),
                measurement_record.get_target_rec(check_id, -2),
            ], (0, 0, kk))
```

to:

```python
        repeat_circuit.append("SHIFT_COORDS", [], (1, 0, 0))
        for check_id in all_check_ids:
            lane, idx = lane_idx[check_id]
            repeat_circuit.append("DETECTOR", [
                measurement_record.get_target_rec(check_id, -1),
                measurement_record.get_target_rec(check_id, -2),
            ], (0, lane, idx))
```

- [ ] **Step 5: Update `_surgery_final_detectors_joint`**

In `src/qldpc/circuits/surgery/circuit.py`, locate `_surgery_final_detectors_joint` (around line 482). Add `bridge: Bridge` parameter and build lane map:

Find signature:
```python
def _surgery_final_detectors_joint(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    joint_code: CSSCode,
    qubit_ids: QubitIDs,
    *,
    measurement_record: MeasurementRecord,
    intercode: bool,
) -> stim.Circuit:
```

Modify to add `bridge`:
```python
def _surgery_final_detectors_joint(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    joint_code: CSSCode,
    bridge: Bridge,
    qubit_ids: QubitIDs,
    *,
    measurement_record: MeasurementRecord,
    intercode: bool,
) -> stim.Circuit:
```

In `build_joint_ppm_circuit`, find the call (around line 372-376):

```python
    circuit += _surgery_final_detectors_joint(
        g_l, g_r, joint_code, qubit_ids,
        measurement_record=measurement_record, intercode=intercode,
    )
```

Update to:

```python
    circuit += _surgery_final_detectors_joint(
        g_l, g_r, joint_code, bridge, qubit_ids,
        measurement_record=measurement_record, intercode=intercode,
    )
```

Now update the inner `_emit_detector` in `_surgery_final_detectors_joint` (around line 506-510). Find:

```python
    circuit = stim.Circuit()

    def _emit_detector(stab_row: np.ndarray, check_id: int, det_idx: int) -> None:
        supp = np.where(stab_row)[0]
        targets = [measurement_record.get_target_rec(qubit_ids.data[q]) for q in supp]
        targets.append(measurement_record.get_target_rec(check_id, -1))
        circuit.append("DETECTOR", targets, (0, 0, det_idx))
```

Modify to:

```python
    circuit = stim.Circuit()
    lane_idx = _check_lane_index_map(
        g_l, qubit_ids, joint=(g_r, bridge, intercode),
    )

    def _emit_detector(stab_row: np.ndarray, check_id: int) -> None:
        supp = np.where(stab_row)[0]
        targets = [measurement_record.get_target_rec(qubit_ids.data[q]) for q in supp]
        targets.append(measurement_record.get_target_rec(check_id, -1))
        lane, idx = lane_idx[check_id]
        circuit.append("DETECTOR", targets, (0, lane, idx))
```

And update the 4 callers inside `_surgery_final_detectors_joint` (around lines 512-526). Find:

```python
    if g_l.basis is Pauli.X:
        # data H_X rows from both gadgets
        for kk in range(m_X_l + m_X_r):
            _emit_detector(HX[kk], qubit_ids.checks_x[kk], kk)
        # G_aug rows + new cycle-Z rows: indices [m_Z_l + m_Z_r : HZ.shape[0])
        det_offset = m_X_l + m_X_r
        for offset, kk in enumerate(range(m_Z_l + m_Z_r, HZ.shape[0])):
            _emit_detector(HZ[kk], qubit_ids.checks_z[kk], det_offset + offset)
    else:
        # data H_Z rows from both gadgets
        for kk in range(m_Z_l + m_Z_r):
            _emit_detector(HZ[kk], qubit_ids.checks_z[kk], kk)
        det_offset = m_Z_l + m_Z_r
        for offset, kk in enumerate(range(m_X_l + m_X_r, HX.shape[0])):
            _emit_detector(HX[kk], qubit_ids.checks_x[kk], det_offset + offset)
```

Modify to:

```python
    if g_l.basis is Pauli.X:
        for kk in range(m_X_l + m_X_r):
            _emit_detector(HX[kk], qubit_ids.checks_x[kk])
        for kk in range(m_Z_l + m_Z_r, HZ.shape[0]):
            _emit_detector(HZ[kk], qubit_ids.checks_z[kk])
    else:
        for kk in range(m_Z_l + m_Z_r):
            _emit_detector(HZ[kk], qubit_ids.checks_z[kk])
        for kk in range(m_X_l + m_X_r, HX.shape[0]):
            _emit_detector(HX[kk], qubit_ids.checks_x[kk])
```

- [ ] **Step 6: Drop the now-unused `get_qubit_coordinates` import**

Run:
```bash
grep -c "get_qubit_coordinates" src/qldpc/circuits/surgery/circuit.py
```

Expected: 1 (only the import line). If 1, remove the import. Find at the top of `circuit.py` (around line 9):

```python
from qldpc.circuits.memory.memory import get_qubit_coordinates
```

Delete this entire line.

Run the grep again:
```bash
grep -c "get_qubit_coordinates" src/qldpc/circuits/surgery/circuit.py
```

Expected: 0.

- [ ] **Step 7: Run the joint test — expect PASS**

Run:
```bash
.venv/bin/python -m pytest src/qldpc/circuits/surgery/_test_circuit.py::test_joint_ppm_qubit_coords_intercode_layout -xvs
```

Expected: `PASSED`.

- [ ] **Step 8: Run the full surgery suite — expect 100 passed**

Run:
```bash
.venv/bin/python -m pytest src/qldpc/circuits/surgery/ -q | tail -3
```

Expected: `100 passed` (96 baseline + 4 new in Tasks 1-4). NO commits yet.

---

## Task 5: Final verification + single atomic commit

**Files:** None modified — just verification.

- [ ] **Step 1: Smoke-test the Cain reproduction script (default path)**

Run:
```bash
.venv/bin/python examples/scripts/find_bbcode_layouts.py 2>&1 | head -3
```

This is the only remaining surgery-adjacent script in `examples/scripts/`. It doesn't use the surgery module, so output should be unaffected. Mostly a sanity check that imports still resolve.

If the script has its own runtime errors unrelated to surgery, ignore them — we only care that `from qldpc.circuits.surgery import ...` doesn't break.

Actually a more direct API smoke test:

```bash
.venv/bin/python -c "
from qldpc.circuits.surgery import (
    build_gadget, build_bridge,
    build_single_ppm_circuit, build_joint_ppm_circuit,
    keep_only_observable, boost_gadget, cheeger_constant,
)
import numpy as np
from qldpc import codes
from qldpc.objects import Pauli
s = codes.SteaneCode()
x = np.asarray(s.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
g = build_gadget(s, x)
c = build_single_ppm_circuit(g, rounds=3, noise_model=None)
# Verify the new coord scheme is in the emitted circuit.
qc_lines = [l for l in str(c).splitlines() if 'QUBIT_COORDS' in l]
d_lines = [l for l in str(c).splitlines() if l.strip().startswith('DETECTOR')]
print(f'QUBIT_COORDS count: {len(qc_lines)}')
print(f'DETECTOR count: {len(d_lines)}')
# Confirm new format
sample_d = [l for l in d_lines if l.strip().startswith('DETECTOR')][:4]
for line in sample_d:
    print(f'  {line.strip()}')
"
```

Expected: `QUBIT_COORDS count: 20` (10 data + 6 X-check + 4 Z-check ancillas), and the first 4 DETECTOR lines should have coords like `(0, 1, 0)`, `(0, 1, 1)`, `(0, 1, 2)`, `(0, 5, 0)` (the round-1 reliable ones).

- [ ] **Step 2: Confirm timeline-svg renders the new labels**

Run:
```bash
.venv/bin/python -c "
import numpy as np
from qldpc import codes
from qldpc.circuits.surgery import build_gadget, build_single_ppm_circuit
from qldpc.objects import Pauli
s = codes.SteaneCode()
x = np.asarray(s.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
g = build_gadget(s, x)
c = build_single_ppm_circuit(g, rounds=3, noise_model=None, data_init='0')
svg = c.diagram('timeline-svg')
# Search for new coord pattern (0, 5, 0) in the SVG — proves G lane label exists.
svg_text = str(svg)
assert '(0, 5, 0)' in svg_text or '(0,5,0)' in svg_text, (
    'Expected (0, 5, 0) coord label in timeline-svg output; new lane encoding may not be active'
)
print('timeline-svg contains G-lane coord label as expected')
"
```

Expected: `timeline-svg contains G-lane coord label as expected`.

- [ ] **Step 3: Final full-suite run**

Run:
```bash
.venv/bin/python -m pytest src/qldpc/circuits/surgery/ -q | tail -3
```

Expected: `100 passed`.

- [ ] **Step 4: Single atomic commit**

```bash
git add src/qldpc/circuits/surgery/circuit.py src/qldpc/circuits/surgery/_test_circuit.py
git commit -m "$(cat <<'EOF'
feat(surgery): semantic-lane QUBIT_COORDS + DETECTOR coords

Replace surgery's opaque (0, 0, kk) DETECTOR coords and 2-row QUBIT_COORDS
layout with a 6/7-lane semantic encoding:

  y=0  data qubits
  y=1  data H_X ancillas (original X-stabilizer ancillas)
  y=2  data H_Z ancillas (original Z-stabilizer ancillas)
  y=3  κ ancillas
  y=4  χ ancillas        (basis-symmetric: χ stays on y=4 even when it
                          lives in checks_z[m_Z:] for basis=Z)
  y=5  G ancillas        (basis-symmetric: G stays on y=5 even when it
                          lives in checks_x[m_X:] for basis=Z)
  y=6  bridge data + bridge cycle ancillas (joint PPM only)

DETECTOR coords become (t, lane, idx) where lane ∈ {1, 2, 4, 5, 6}
inherits the y-position of the measured ancilla and idx matches the
ancilla's x position. Pure visualization improvement — same circuit
semantics, same observables, same detectors as before. detslice-svg
renders 6-7 stripes by role; timeline-svg labels become self-documenting
(e.g., coords=(0, 5, 0) immediately identifies the round-0 G check 0).

Add two new private helpers (_surgery_qubit_coordinates,
_check_lane_index_map) and drop the dependency on
qldpc.circuits.memory.memory.get_qubit_coordinates (memory module
unaffected). Public API unchanged. Surgery test count: 96 → 100.

Per docs/superpowers/specs/2026-06-10-surgery-coord-lanes-design.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: Confirm commit landed**

Run:
```bash
git log -1 --oneline
```

Expected: top commit is `feat(surgery): semantic-lane QUBIT_COORDS + DETECTOR coords`.

NO push — keeping local-only per the user's prior preference on this branch.

---

## Self-review notes

Spec coverage check (compared to `docs/superpowers/specs/2026-06-10-surgery-coord-lanes-design.md`):

- §"Lane layout (single PPM)" 6-lane table → Task 1 Step 3 (`_surgery_qubit_coordinates`) emits exactly these lanes for single PPM.
- §"Lane layout (joint PPM)" with bridge at y=6 → Task 4 Step 3 (extends helper) + Task 4 Step 5 (extends helper) + Step 7 (joint test verifies y=6 has bridge data + cycle ancillas).
- §"DETECTOR coord convention" — (t, lane, idx) format pinned by `test_detector_coords_steane_round_1_reliable` (Task 2 Step 1).
- §"χ stays on y=4, G stays on y=5 across basis swap" — pinned by `test_detector_coords_basis_z_preserves_lane_semantics` (Task 3 Step 1).
- §"DETECTOR emit sites" — all 6 listed sites updated: 2 in `_surgery_qec_cycle` (Task 2 Step 4), 2 in `_surgery_qec_cycle_joint` (Task 4 Step 4), 1 in `_surgery_final_detectors` (Task 2 Step 4d), 1 in `_surgery_final_detectors_joint` (Task 4 Step 5).
- §"Call sites" — both `build_single_ppm_circuit` (Task 1 Step 4) and `build_joint_ppm_circuit` (Task 4 Step 3) swap to `_surgery_qubit_coordinates`.
- §"Drop import" — Task 4 Step 6 removes the `get_qubit_coordinates` import.
- §"Test plan" 4 tests — Task 1 Step 1, Task 2 Step 1, Task 3 Step 1, Task 4 Step 1.

Type / signature consistency check:

- `_surgery_qubit_coordinates(gadget, qubit_ids, *, joint=None)` — same signature used in Task 1 Step 4 (single PPM call) and Task 4 Step 3 (joint PPM call with `joint=(g_r, bridge, intercode)`).
- `_check_lane_index_map(gadget, qubit_ids, *, joint=None)` — same signature used in Task 2 Step 4 (single) and Task 4 Steps 4-5 (joint).
- `_surgery_qec_cycle_joint` and `_surgery_final_detectors_joint` both gain a `bridge` parameter in Task 4 Steps 4-5; the corresponding call sites in `build_joint_ppm_circuit` are updated in the same steps.
- All test function names match the spec.

Known constraint:

- The `_surgery_qubit_coordinates` helper builds in Task 1 with `joint=None`. The signature already includes the optional `joint` parameter so Task 4 doesn't need to refactor the function — it only adds bridge-specific QUBIT_COORDS emission. Reading Task 1 Step 3 in isolation, the bridge-cycle branch at the bottom is dead code in the single-PPM case (gated by `joint is not None`), but Task 4 exercises it.
