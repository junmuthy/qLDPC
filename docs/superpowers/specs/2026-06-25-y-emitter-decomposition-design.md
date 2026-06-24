# Ȳ emitter decomposition — design

**Date:** 2026-06-25
**Status:** approved (design)
**Scope:** `src/qldpc/circuits/surgery/` — the single logical-Ȳ measurement emitter.

## Motivation

After the Z̄⊗X̄ deletion and the `_surgery_*_joint` merge (done earlier on this
branch — mixed-basis joint code, `joint_layout.py`, and the forbidden
`hadamard_dual`/`cohen` paths removed; the joint and single CSS emitters now
share unified `_surgery_*` helpers via a `joint=` parameter), the surgery
module has three emitters:

| Emitter | Lines | Shape |
|---|---|---|
| `build_single_ppm_circuit` (X̄/Z̄) | ~85 | CSS, thin — calls `_surgery_*` helpers |
| `_build_joint_ppm_circuit_same_basis` (X̄X̄/Z̄Z̄) | ~98 | CSS, thin — same `_surgery_*` helpers |
| `build_single_y_ppm_circuit` (Ȳ) | ~635 | non-CSS, fully inline |

The two CSS emitters are already small and share everything. The outlier is the
**635-line Ȳ emitter**, a single function in a 2003-line `circuit.py`. Its
internal phases parallel the CSS `_surgery_*` helpers, but its algorithm — a
split X/Z/Y syndrome schedule over a non-CSS merged code — is genuinely
different from the CSS single-pass.

This refactor decomposes that one function into named phases and isolates the
non-CSS path in its own module. It is **behaviour-preserving** except for the
deliberate removal of two untested bring-up observables (§3).

## Key finding: the Ȳ subsystem is already self-contained

Every private helper the Ȳ emitter uses is called **only** by the Ȳ emitter
(verified by grep over `circuit.py`):

- `_steane_logical_y_eigenstate_prep`
- `_split_quditcode_into_virtual_cssc`
- `_mixed_basis_qubit_coords`
- `_compute_stabilizer_center_mask`
- `_observable_is_deterministic`

The emitter imports **nothing** from the CSS path — no `_surgery_*`, no
`_check_lane_index_map`. The state-prep (|Ȳ±⟩ injection + transversal S) and
detach (mixed-basis MX/M/MY destructive readout) are genuinely Ȳ-specific, not
shared structure. Therefore the right move is **extract + decompose**, not
force-merge with CSS (which would be the rejected "CSS as degenerate no-Y case"
option A).

## §1 — Module structure

Create `src/qldpc/circuits/surgery/y_circuit.py`. Move the whole Ȳ subsystem
there: `build_single_y_ppm_circuit`, the five private helpers above, the
currently-nested `_row_paulis` helper, and the new phase helpers (§2).

Update the single re-export in `surgery/__init__.py`
(`build_single_y_ppm_circuit` moves from `.circuit` to `.y_circuit`).

Result:
- `circuit.py`: 2003 → ~1100 lines (pure CSS: single + joint + stitch +
  shared `_surgery_*`).
- `y_circuit.py`: ~640 lines (the non-CSS path, isolated).

`y_circuit.py` imports its low-level deps directly from their sources
(`qldpc.circuits.bookkeeping`, `qldpc.circuits.memory.syndrome_measurement`,
`qldpc.circuits.noise_model`, `qldpc.codes.common`, `qldpc.objects`,
`.y_gadget`) — not from `.circuit`.

## §2 — Phase decomposition

`build_single_y_ppm_circuit` becomes a thin orchestrator (~40 lines) that wires
together named phase functions. Phase boundaries follow the existing inline
section comments, so each becomes a function with an explicit signature:

```
build_single_y_ppm_circuit(yg, *, rounds, noise_model, data_init,
                           memory_logical, force_obs0) -> stim.Circuit
  ├─ _y_state_prep            setup (split virtual CSSc, qubit-id arrays,
  │                           coords) + |Ȳ±⟩ injection + κ_x/κ_z init
  ├─ _y_qec_cycle             split X/Z/Y schedule + round-1 reliable
  │                           detectors + repeat block   ← non-CSS strategy
  ├─ _y_detach_and_readout    mixed-basis MX/M/MY destructive readout
  ├─ _y_final_detectors       stabilizer-center rows reconstructed from
  │                           final destructive readouts
  ├─ _y_emit_obs0             Ȳ eigenvalue product (force_obs0 /
  │                           Y±-deterministic gate)
  └─ _y_emit_survivor_memory  survivor-Z̄ logical-memory observable
```

The orchestrator owns the shared state that crosses phases (the merged code,
`QubitIDs`, the id arrays `real_data_ids`/`kx_ids`/`kz_ids`/`y_ancilla_ids`,
`center_mask`, the `MeasurementRecord`) and threads it explicitly into each
phase — no module-level or closure state. Each phase appends to and returns
`stim.Circuit` fragments (matching the `_surgery_*` convention).

`_y_qec_cycle` is the explicit non-CSS counterpart to `_surgery_qec_cycle`; the
two stay separate functions (the user's "two explicit syndrome-schedule
strategies" decision). No attempt is made to share a round-loop skeleton.

## §3 — Prune (the only behaviour change)

Remove, during extraction:

- the `benchmark_y` parameter and its emission block (obs0 ⊕ obs1 LER
  benchmark) — not referenced by any surviving test;
- the `obs1` destructive-cross-check block (documented "NOT a physical
  protocol") — not referenced by any surviving test.

Keep `obs0` (real FT readout) and the survivor-memory observable (tested,
real decodability check). `_observable_is_deterministic` stays — still used by
`_y_emit_obs0` and `_y_emit_survivor_memory`.

The CSS emitters' own `obs1` (in `_surgery_observable`, tested by
`circuit_test.py::test_build_joint_ppm_circuit_intercode_noiseless_observables_zero`)
is **out of scope** and untouched.

## §4 — Success criteria & testing

- Behaviour-preserving for all kept paths: X̄/Z̄/X̄X̄/Z̄Z̄ circuits byte-identical
  (CSS path untouched); Ȳ circuits identical except the removed obs1/benchmark_y
  observables.
- The surviving 219-test surgery suite stays green. Only adjustment: any test
  asserting on `obs1`/`benchmark_y` for the Ȳ emitter (currently none — verified
  by grep). Net Ȳ test count unchanged.
- Tests in `circuit_single_y_test.py` that import the Ȳ helpers update their
  import path from `.circuit` to `.y_circuit`.
- `ruff check` clean; no new public API.

## Out of scope

- Any change to the CSS emitters (single/joint) or their `_surgery_*` helpers.
- Sharing code between the CSS and Ȳ paths (the analysis shows minimal genuine
  overlap; forcing it is rejected option A).
- The Ȳ algorithm itself, the merged-code construction in `y_gadget.py`, or
  `apply_mixed_basis_merge` in `merge.py`.
- `docs/superpowers/docs/main.tex` cleanup of deleted Z̄⊗X̄ sections (separate
  follow-up).
