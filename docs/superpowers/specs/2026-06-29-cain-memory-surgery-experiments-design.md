# Cain-faithful memory & surgery experiments — design

**Date:** 2026-06-29
**Status:** approved (pending written-spec review)
**Scope:** `src/qldpc/circuits/surgery/circuit.py` (+ tests). Reuse of
`src/qldpc/circuits/memory/memory.py` for the memory experiment.

## 1. Motivation

Implement the numerical experiments of **Cain, Xu, King, Picard, Levine, Endres,
Preskill, Huang, Bluvstein, arXiv:2603.28627, Appendix D ("Numerical
simulations")** — both the **Memory experiment** and the **Surgery experiment** —
with the observable sets defined exactly as the paper states them, so a
surgery-vs-memory comparison is apples-to-apples.

The surgery gadget here is an **L = 1 gadget** (`build_gadget` measures the single
logical operator `gadget.x`). So the paper's `t` (number of measured target
operators `L = {P̄_i}`) is **`t = 1`** for both single-PPM and joint-PPM (the joint
measures the one product `X̄_l ⊗ X̄_r`). Therefore `k + t = k + 1` and
`k − t = k − 1` throughout.

### Paper definitions (Appendix D)

- **Memory experiment** — initialize `|+⟩^⊗k`, run QEC, transversal X readout;
  logical observables = the `k` logical X̄ operators of the code.
- **Surgery experiment** — a gadget measures `L` (here `t = 1`). Ancilla `Q'`
  always init/measured in Z; data `Q` init/measured in **either X or Z**. Two
  experiments per gadget:
  - **X-basis:** `k + t` observables = the `k` final logical X̄ of the data code
    (final transversal X parities on `Q`) **plus** the `t` outcomes of `L`
    (parities of the merged-code X-checks in the **first** stabilizer cycle).
    Catches space-like Z̄ errors + time-like errors.
  - **Z-basis:** `k − t` observables = the logical Z̄ of the data code **that
    commute with `L`**. Catches space-like X̄ errors.

## 2. Definitions used in this design

Let `experiment_basis ∈ {Pauli.X, Pauli.Z}` be the basis the **data** qubits `Q`
are initialized/measured in — **decoupled** from `gadget.basis` (the type of `L`).
Let `k` = data-code dimension (single: `gadget.code.dimension`; joint:
`k_l + k_r` over the combined data code `c_l ⊕ c_r`).

- **Match-basis** ≡ `experiment_basis == gadget.basis`.
- **Opposite-basis** ≡ `experiment_basis != gadget.basis`.

"Block observables" = the logical operators of the bare data code in
`experiment_basis` (the memory convention; same machinery as
`qldpc.circuits.memory.memory.get_observables`).

## 3. Observable sets to emit

### 3.1 Match-basis → `k + 1`

- **`k` space-like** = the `k` logical-`experiment_basis` operators of the data
  code, read from the **final transversal data measurement**. These data-only
  representatives are already valid logicals of the merged code (e.g. for an
  X-gadget the X̄_i commute with the X-type `S'_meas` and the Z-type gauge on
  `Q'`), so **no frame correction is needed**.
- **`1` time-like** = the value of `L`, defined as the XOR of the **first-cycle**
  `S'_meas` merge-check outcomes (`MeasurementRecord.get_target_rec(cid, 0)`),
  per Cain et al. arXiv:2603.28627 Appendix D ("first stabilizer measurement
  cycle").

### 3.2 Opposite-basis → `k − 1`

The `k − 1` logical-`experiment_basis` operators of the data code that **commute
with `L`**, each emitted as a **Pauli-frame-corrected** observable:

```
observable_i = (final data parity on supp(z_i))  ⊕  (Q'-split parity on supp(c_i))
```

where `c_i` is the Z-type Pauli-frame correction on `Q'` (§4). The correction is
folded into `OBSERVABLE_INCLUDE` as measurement records — **no physical gates**
(see §6 decision). The single anticommuting logical is dropped (§4 shows its
deformation is unsolvable), giving `k − 1`.

### 3.3 Memory experiment

Reuse `get_memory_experiment(code, basis=Pauli.X, num_rounds=...)` directly — it
already emits exactly the `k` logical-X block observables via `get_observables`.
**No new code; do NOT wrap with `keep_only_observable`** (that helper exists to
collapse to a single observable for sinter LER sweeps, which we do not run).

### 3.4 Folded-in cross-check (replaces obs1)

The old noiseless `obs0 == obs1` cross-check is removed. Its replacement is a
**test assertion**, not a circuit observable: in any noiseless shot the time-like
`L` observable equals the matching GF(2) combination of the space-like block
observables. (Lives in the test file, §8.)

## 4. Pauli-frame correction (the GF(2) construction)

For an opposite-basis block observable with bare-code logical support `z_i`
(embedded in the merged register with zeros on `Q'`):

1. **Syndrome** `s = HX_merged · z_i` (for an X-gadget; symmetric for Z). Because
   `z_i` is a bare-code logical, the original X-rows `[HX_data, 0]` contribute 0;
   only the `S'_meas = [f₁ᵀ | ∂₁]` rows fire, so
   `s = f₁ᵀ · z_i|_data = z_i restricted to V₀ (= supp L) ∈ GF(2)^{|V₀|}`.
2. **Solve** `∂₁ · c = s` over GF(2) for `c ∈ GF(2)^{|C₀|}` supported on `Q'`
   (`∂₁ = incidence.T`, shape `|V₀|×|C₀|`). Then the deformed representative
   `v_i = z_i ⊕ lift(c)` satisfies `HX_merged · v_i = 0`, i.e. commutes with every
   merged X-check.
3. **Targets** = end-of-circuit measurement records on `supp(v_i)`: final data `M`
   records for data columns, `Q'`-split `M` records for `Q'` columns.

**Solvability ⇔ commutes with `L`.** `∂₁ · c = s` is solvable iff `s ∈ im(∂₁)`;
the all-ones vector spans the left null space of an incidence operator, so
solvable ⇔ `Σ s = 0` ⇔ `x · z_i = 0` ⇔ `z_i` commutes with `L`. Thus the same
linear solve (a) selects the `k − 1` commuting logicals and (b) produces each
one's frame-correction records. The dropped logical has no solution.

This construction generalizes to joint PPM (`L = X̄_l ⊗ X̄_r` over the combined
data code; `∂₁` is the joint gadget's incidence; `c` lives on the combined `Q'`).

## 5. API changes

Add `experiment_basis: PauliXZ` to `build_single_ppm_circuit` and
`build_joint_ppm_circuit`:

- Default = the measured basis (`gadget.basis` / `bridge.basis`) → the natural
  **`k + 1`** experiment. Pass the opposite basis for the **`k − 1`** experiment.
- The emitted observable set is **always** the paper set (§3). The default
  `block_observables` flag and the obs0/obs1 path are **removed** from the public
  builders. (An explicit `data_init` override may remain for targeted tests, but
  it no longer changes which observable set is emitted.)
- Y circuits (`build_single_y_ppm_circuit`, `y_circuit.py`, `y_gadget.py`) are
  **out of scope** — there is no direct Y measurement here.

## 6. Internal refactor

Principle: **the data side follows `experiment_basis`; the `Q'`/merge side follows
`gadget.basis`.**

- `_surgery_state_prep`: data default init from `experiment_basis`
  (`|+⟩` for X, `|0⟩` for Z); `Q'`/bridge init unchanged (complement of
  `gadget.basis`).
- `_surgery_detach_and_readout`: data readout op from `experiment_basis`
  (`MX`/`M`); `Q'`/bridge detach op unchanged.
- `_classify_reliable_round1_checks` + `_surgery_final_detectors`: collapse to one
  unifying rule — a merged check is **round-1-deterministic ⇔ final-reconstructable
  ⇔ its support lies entirely within matching-basis qubits**, where data qubits are
  in `experiment_basis`, `Q'` qubits are in complement(`gadget.basis`), and (joint
  PPM) bridge qubits are in complement(`bridge.basis`). (For the
  current match-basis X-gadget case this reproduces today's behavior: reliable =
  `H_X` data rows + Z-gauge on `Q'`; final detectors = `H_X` from `MX` data +
  gauge from `M` `Q'`.)
- `_surgery_observable`: rewritten to emit the §3 sets. New pieces:
  - first-cycle `L` reader (`get_target_rec(cid, 0)` over `S'_meas` ids);
  - commuting-subset selector + frame-correction builder (§4): a small GF(2)
    routine returning, per kept logical, the list of end-record targets.

The `S'_meas` (meas-check) id computation already present in
`build_single_ppm_circuit` / `_build_joint_ppm_circuit_same_basis` is reused to
locate the first-cycle `L` checks.

## 7. Scope / out of scope

- **In:** single-PPM and same-basis joint-PPM X/Z experiments; memory via existing
  helper; frame correction via record folding.
- **Out:** Y / mixed-basis surgery; LER / sinter sweeps; `keep_only_observable` in
  the memory path; any physical conditional-Pauli feedback gates.

## 8. Test migration

`circuit_test.py` (~200 tests) currently assert on obs0/obs1. Migrate to:

- **Observable counts:** match-basis emits `k + 1`; opposite emits `k − 1`
  (single and joint).
- **Noiseless determinism:** detector sampler predicts all observables (no noise →
  every emitted observable is deterministic).
- **Commute-with-`L`:** every opposite-basis observable's logical support commutes
  with `L`; the dropped one anticommutes.
- **Folded cross-check (§3.4):** time-like `L` == matching GF(2) combination of the
  block observables, noiseless.
- **Frame-correction correctness:** an opposite-basis observable that omits the
  `Q'`-split records is non-deterministic (sanity check that the correction is
  load-bearing), while the corrected one is deterministic.
- Keep DEM-compile / structural-matrix / coordinate tests. No LER/sinter.

## 9. References

- Cain, Xu, King, Picard, Levine, Endres, Preskill, Huang, Bluvstein,
  arXiv:2603.28627 — Appendix D (Memory experiment, Surgery experiment),
  Appendix B.1 (single-PPM protocol).
- Webster, Smith, Cohen, arXiv:2511.15989 §II.A — L=1 gadget, Eq. 1 readout.
- Swaroop, Jochym-O'Connor, Yoder, arXiv:2410.03628 §III — joint stitch
  (`H̃_X^joint` / `H̃_Z^joint` block structure).
- Homological measurement, arXiv:2410.02753 — gauge `∂_0 = ker ∂_1`, fault
  distance via edge expansion.
