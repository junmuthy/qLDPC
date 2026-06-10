# Surgery QUBIT_COORDS + DETECTOR Coord Lane Layout — Design

**Status**: Draft (2026-06-10)
**Scope**: `src/qldpc/circuits/surgery/circuit.py` + `src/qldpc/circuits/surgery/_test_circuit.py`
**Author**: tgzhou (with Claude)

## Background

Surgery currently emits `QUBIT_COORDS` and `DETECTOR` annotations with coordinate values that have very low information density:

- `QUBIT_COORDS` comes from `qldpc.circuits.memory.memory.get_qubit_coordinates`, which puts ALL data-slot qubits at `(0, kk)` and ALL check-slot qubits at `(1, kk)`. For surgery this conflates true data qubits with κ ancillas (both live in the merged-code "data" slot) and conflates original-code stabilizer ancillas with surgery-added χ / G ancillas. Net result: 2 rows of qubits, with no semantic structure visible in `detslice-svg`.
- `DETECTOR` coords are hard-coded `(0, 0, kk)` at every emit site. `timeline-svg` displays this as detector labels but the `kk` integer has no semantic meaning — you have to remember the qubit_ids layout to decode it.

For someone debugging the surgery construction (e.g., checking which detectors fire under a given init state), the current encoding is opaque. The user observed that the four Steane round-1 reliable detectors all share `coords=(0, 0, kk)` with `kk ∈ {0,1,2,9}`, and the `kk=9` jump to the G-row gauge-fix detector is invisible without code inspection.

## Goal

Replace both annotations with a coordinate scheme that encodes the **semantic role** of every qubit and detector. Lane numbers map 1:1 between QUBIT_COORDS (the `y` axis position) and DETECTOR coord arg `lane` (the second tuple element). After the change:

- `detslice-svg` shows qubits in 7 horizontal stripes by semantic role.
- `timeline-svg` detector labels read `(t, lane, idx)` where `lane ∈ {2, 3, 4, 5, 6}` immediately identifies which check class fired.
- Programmatic post-processing (DEM grouping, decoder debugging, fault-trace analysis) can filter detectors by lane in one line.

This is purely a debugging visualization improvement — it does NOT model hardware geometry. Neutral-atom architectures use dynamic zone-based layouts (storage / entangling / readout) that this scheme cannot represent. The lane layout is a semantic convention chosen for clarity.

## Non-goals

- **No public API change.** `build_gadget`, `build_bridge`, `build_single_ppm_circuit`, `build_joint_ppm_circuit`, `boost_gadget`, `cheeger_constant`, `keep_only_observable` keep their current signatures and behavior.
- **No algorithm change.** Same circuits, same observables, same detectors as before — only the `QUBIT_COORDS` and `DETECTOR` coord tuples change.
- **No change to `qldpc.circuits.memory.memory.get_qubit_coordinates`.** That helper is shared with the memory module; we add a surgery-private replacement instead.
- **No change to the lane scheme for non-PPM contexts** (memory circuits unaffected).
- **No new tests for unchanged behavior.** Existing tests that assert `compile_detector_sampler` / `compile_sampler` output shapes / values still pass with the new coords. We add ~4 small tests that pin the new coord scheme directly.

## Lane layout (single PPM)

| `y` | Content | Source slot | X range |
|---|---|---|---|
| 0 | data qubits | `qubit_ids.data[:n_data]` | `0 .. n_data - 1` |
| 1 | κ ancillas | `qubit_ids.data[n_data : n_data + k]` (k = len(gadget.kappa_qubits)) | `0 .. k - 1` |
| 2 | data H_X ancillas (original X-stabilizer ancillas) | `qubit_ids.checks_x[:m_X]` | `0 .. m_X - 1` |
| 3 | χ ancillas (surgery-added) | basis=X: `qubit_ids.checks_x[m_X:]`; basis=Z: `qubit_ids.checks_z[m_Z:]` | `0 .. \|V_0\| - 1` |
| 4 | data H_Z ancillas (original Z-stabilizer ancillas) | `qubit_ids.checks_z[:m_Z]` | `0 .. m_Z - 1` |
| 5 | G ancillas (gauge-fix) | basis=X: `qubit_ids.checks_z[m_Z:]`; basis=Z: `qubit_ids.checks_x[m_X:]` | `0 .. r - 1` |

Rationale: y is monotonic in qubit ID for basis=X (ids 0..6→y=0, 7..9→y=1, 10..12→y=2, 13..15→y=3, 16..18→y=4, 19→y=5), so QUBIT_COORDS lines in the circuit text dump appear in increasing y order. basis=Z breaks monotonicity because χ and G swap matrix slots, but the lane numbers remain stable: χ always y=3, G always y=5.

## Lane layout (joint PPM)

Joint PPM adds bridge qubits + bridge cycle check ancillas. Both go on `y = 6`:

| `y` | Joint PPM content | X range |
|---|---|---|
| 0 | data: left then right (intercode) or shared (intracode) | intercode: `0 .. n_l + n_r - 1`; intracode: `0 .. n_l - 1` |
| 1 | κ ancillas: left then right | `0 .. k_l + k_r - 1` |
| 2 | data H_X ancillas: left then right (intercode) | intercode: `0 .. m_X_l + m_X_r - 1`; intracode: `0 .. m_X - 1` |
| 3 | χ ancillas: left then right | `0 .. \|V_0_l\| + \|V_0_r\| - 1` |
| 4 | data H_Z ancillas: left then right (intercode) | analogous |
| 5 | G ancillas: left then right | `0 .. r_l + r_r - 1` |
| 6 | bridge data qubits + bridge cycle check ancillas (sharing the row) | data: `0 .. w - 1`; cycle ancillas: `0 .. w - 2` |

Bridge data qubits and bridge cycle ancillas have **different qubit ids** so sharing the row is unambiguous; visually they occupy the same vertical band in `detslice-svg` — bridge data at integer x positions `0..w-1`, bridge cycle ancillas at integer x positions `0..w-2`. (The overlap at x ∈ [0, w-2] is acceptable: bridge data and bridge cycle ancillas are conceptually paired by index.)

## DETECTOR coord convention

DETECTOR coords become `(t, lane, idx)` where:

- `t` = round index. `SHIFT_COORDS (1, 0, 0)` after each round (unchanged from today).
- `lane` = the y-position of the ancilla being measured, so `lane ∈ {2, 3, 4, 5, 6}` for the 5 lanes that contain check ancillas (lane=0 = data and lane=1 = κ have no DETECTOR; κ is final-measured but not a syndrome check).
- `idx` = position within that lane, matching the `x` coordinate of the ancilla being measured.

Lane / x mapping (single PPM, basis=X):

| Check role | lane | idx |
|---|---|---|
| data H_X row `j` (j ∈ [0, m_X)) | 2 | `j` |
| data H_Z row `j` (j ∈ [0, m_Z)) | 4 | `j` |
| χ row `j` (j ∈ [0, \|V_0\|)) | 3 | `j` |
| G row `j` (j ∈ [0, r)) | 5 | `j` |

For basis=Z, the source-matrix slot swaps but lane is preserved.

For joint PPM, the bridge cycle check (basis=X: new Z-check at `qubit_ids.checks_z[index]`; basis=Z: new X-check) maps to lane=6.

**Steane example before/after** (basis=X, all 4 round-1 reliable detectors):

| Detector | Old `(0, 0, kk)` | New `(0, lane, idx)` | Decoded |
|---|---|---|---|
| D0 | (0, 0, 0) | (0, 2, 0) | round 0, data H_X check 0 |
| D1 | (0, 0, 1) | (0, 2, 1) | round 0, data H_X check 1 |
| D2 | (0, 0, 2) | (0, 2, 2) | round 0, data H_X check 2 |
| D3 | (0, 0, 9) | (0, 5, 0) | round 0, G check 0 |

## Implementation surface

### New helpers in `circuit.py`

**`_surgery_qubit_coordinates(gadget, qubit_ids, *, joint=None) -> stim.Circuit`**

Emits `QUBIT_COORDS` per the lane layout above. Replaces today's `get_qubit_coordinates(qubit_ids.data, qubit_ids.check)` call at the top of `build_single_ppm_circuit` and `build_joint_ppm_circuit`. The optional `joint` parameter (None for single PPM; otherwise a small bundle holding `g_l`, `g_r`, `bridge`, `intercode` for joint emission) drives whether the right-side / bridge lanes get coords. For single PPM, `n_data = gadget.code.num_qudits`, `k = len(gadget.kappa_qubits)`, and the rest derives from `qubit_ids`.

**`_check_lane_index(gadget, qubit_ids, check_id, *, joint=None) -> tuple[int, int]`**

Pure mapping: given a check ancilla `check_id`, return `(lane, idx)` matching that ancilla's `QUBIT_COORDS`. Used by the 4 DETECTOR emit sites in place of `kk`. Implementation: precompute a `dict[int, tuple[int, int]]` once per circuit build (returned alongside or attached to a small struct passed into `_surgery_qec_cycle` and `_surgery_final_detectors`), so the per-detector emit cost is O(1) dict lookup.

### Modified emit sites in `circuit.py`

Five emit sites change from `(0, 0, kk)` (or `(0, 0, det_idx)`) to `(0, lane, idx)`:

- `_surgery_qec_cycle_joint` round-1 detectors (currently line 459).
- `_surgery_qec_cycle_joint` repeat-block detectors (currently line 468).
- `_surgery_final_detectors_joint` (currently line 510).
- `_surgery_qec_cycle` round-1 detectors (currently line 647).
- `_surgery_qec_cycle` repeat-block detectors (currently line 656).
- `_surgery_final_detectors` (currently line 737).

(`SHIFT_COORDS (1, 0, 0)` calls — currently at lines 466, 654, 770 — are unchanged.)

### Modified call sites in `circuit.py`

Two call sites swap `get_qubit_coordinates(qubit_ids.data, qubit_ids.check)` for `_surgery_qubit_coordinates(gadget, qubit_ids, ...)`:

- `build_single_ppm_circuit` (currently around line 70).
- `build_joint_ppm_circuit` (currently around line 320).

The `from qldpc.circuits.memory.memory import get_qubit_coordinates` import is dropped if no other surgery code still uses it.

### No external API change

`build_*_ppm_circuit` signatures, return types, and all internal detector indices into `MeasurementRecord` / `DetectorRecord` stay identical. The only observable difference is the coord values inside `QUBIT_COORDS` and `DETECTOR` instructions in the emitted `stim.Circuit`.

## Tests

Add to `_test_circuit.py` (~4 tests, ~80 lines):

**1. `test_qubit_coords_layout_steane`** — Build Steane single-PPM circuit (basis=X). Parse `QUBIT_COORDS` lines from `str(circuit)`. Assert:
- Qubits 0..6 (Steane data) all on y=0, x=0..6.
- Qubits 7..9 (κ) all on y=1, x=0..2.
- Qubits 10..12 (data H_X ancillas) on y=2.
- Qubits 13..15 (χ ancillas) on y=3.
- Qubits 16..18 (data H_Z ancillas) on y=4.
- Qubit 19 (G ancilla) on y=5.

**2. `test_detector_coords_steane_round_1_reliable`** — Build Steane single-PPM circuit (basis=X), 1 round. Walk DETECTOR instructions; assert exactly 4 detectors with coords `{(0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 5, 0)}` (the 3 data H_X reliable checks + the 1 G reliable check). Index-set match (order-insensitive).

**3. `test_detector_coords_basis_z_preserves_lane_semantics`** — Build Steane gadget with `basis=Pauli.Z` (need a logical-Z support — use `get_logical_ops(Pauli.Z)[0]`). Round-1 reliable detector lanes should be `{4, 5}` (data H_Z = lane 4; G = lane 5). The χ would be lane 3 — but χ is NOT a round-1 reliable check, so it doesn't show up. Asserts the χ/G lane numbers stay stable under basis swap (i.e., `lane=5` is G, not χ, regardless of which matrix slot G lives in).

**4. `test_joint_ppm_qubit_coords_intercode_layout`** — Build intercode joint PPM (e.g., two Steane copies, basis=Pauli.Z). Parse `QUBIT_COORDS`. Assert:
- left+right data on y=0, x = 0..n_l + n_r - 1, with x = 0..n_l-1 being left and x = n_l..n_l+n_r-1 being right.
- κ ancillas on y=1, similarly left-then-right.
- bridge data on y=6 at x=0..w-1.
- bridge cycle ancillas on y=6 at x=0..w-2.

(The bridge ancilla / data overlap on y=6 is intentional per the layout above.)

## Net effect

| File | Δ |
|---|---|
| `circuit.py` | +~80 (new `_surgery_qubit_coordinates`, `_check_lane_index`) − ~5 (drop the `get_qubit_coordinates` import + 2 call sites) + ~5 (touch 6 DETECTOR emit sites) ≈ **+80 net** |
| `_test_circuit.py` | +~80 (4 new tests) |
| Surgery test count | 96 → 100 |
| Public API | unchanged |
| Algorithm | unchanged |

## Success criteria

- `.venv/bin/python -m pytest src/qldpc/circuits/surgery/ -q | tail -3` reports **100 passed**.
- `python -c "from qldpc.circuits.surgery import build_single_ppm_circuit; import numpy as np; from qldpc import codes; from qldpc.objects import Pauli; s = codes.SteaneCode(); x = np.asarray(s.get_logical_ops(Pauli.X)[0]).astype(np.uint8); from qldpc.circuits.surgery import build_gadget; g = build_gadget(s, x); c = build_single_ppm_circuit(g, rounds=1, noise_model=None); print([l for l in str(c).splitlines() if 'DETECTOR' in l])"` shows DETECTOR lines whose coord args are `{(0, 2, 0), (0, 2, 1), (0, 2, 2), (0, 5, 0)}` — i.e., `lane ∈ {2, 5}`, not `lane=0`.
- `circuit.diagram("timeline-svg")` rendered on the same example shows detector labels `coords=(0, 2, 0) / (0, 2, 1) / (0, 2, 2) / (0, 5, 0)` instead of `(0, 0, 0..2, 9)`.
- `circuit.diagram("detslice-svg", tick=...)` rendered on the same example shows qubit dots in 4 distinct y-rows (0, 2, 1, 5 in this small case; or all 6 rows once χ ancillas are present at their position).
- No existing test fails. No public function's signature changes.

## Open questions

None — lane layout, DETECTOR coord convention, joint PPM bridge handling, and basis symmetry are all explicit above. The "bridge data + cycle ancillas share y=6" choice is documented and intentional.
