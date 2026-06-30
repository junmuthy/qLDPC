"""End-to-end / truth-table / x-error-locality tests for
qldpc.circuits.surgery.circuit.PPM_XZ (build_single_ppm_circuit).

These are the heavy sampling tests: logical_state_init end-to-end truth tables,
multi-round invariance, the single-qubit X-error locality regression, the
even-rounds time-like-L truth table, the frame-correction determinism check, and
the BB [[36, 8]] boost DEM contract. The unit/structural tests live in
PPM_XZ_test.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import stim

from qldpc import codes
from qldpc.circuits.surgery.circuit.conftest import _bb_36_8_code
from qldpc.objects import Pauli, PauliXZ


@pytest.mark.parametrize("state,eigenvalue", [("+", 0), ("-", 1)])
def test_single_ppm_match_basis_block_and_L_equal_prepared_eigenvalue(
    state: str, eigenvalue: int
) -> None:
    """Match-basis single PPM with an experiment_basis eigenstate prep: both the
    block logical (index 0) and the time-like L (index 1 = k) read the prepared
    eigenvalue deterministically — the §3.4 folded cross-check.

    Steane basis=X gadget, experiment_basis=X (match): data |+⟩→X̄=+1 (bit 0),
    |-⟩→X̄=-1 (bit 1). The block X̄ readout and the time-like L (XOR of the
    first-cycle merge X-checks) both equal that bit, replacing the old obs0==obs1
    cross-check (which read flips-vs-baseline and was always 0 for any init).
    """
    from qldpc.circuits.surgery.circuit.PPM_XZ import build_single_ppm_circuit
    from qldpc.circuits.surgery.circuit.support import logical_state_init
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    circuit = build_single_ppm_circuit(
        g, rounds=3, noise_model=None, data_init=logical_state_init(code, state, log_idx=0)
    )
    assert circuit.num_observables == code.dimension + 1  # k+1, k=1
    raw = circuit.compile_sampler().sample(shots=16).astype(np.uint8)
    n_meas = raw.shape[1]
    obs_lines = [ln for ln in str(circuit).splitlines() if ln.startswith("OBSERVABLE_INCLUDE")]
    vals = []
    for ln in obs_lines:
        offs = [int(t.strip("rec[]")) for t in ln.split() if t.startswith("rec[")]
        vals.append(np.bitwise_xor.reduce(raw[:, [n_meas + o for o in offs]], axis=1))
    block_x, time_L = vals[0], vals[code.dimension]
    assert (block_x == eigenvalue).all(), f"state={state!r}: block X̄ != {eigenvalue}"
    assert (time_L == eigenvalue).all(), f"state={state!r}: time-like L != {eigenvalue}"


@pytest.mark.parametrize("state,expected_obs0", [("0", 0), ("1", 1)])
def test_logical_state_init_end_to_end_steane_basis_z(state: str, expected_obs0: int) -> None:
    """Steane single-PPM (basis=Z) reads obs0 = int(state) deterministically.

    Steane has wt(Z̄_0) = 3 (odd), so naive broadcast `"1" * n` ALSO works
    — this test pins the helper to the textbook expectation on the
    historically-working code, catching any regression where the helper
    accidentally diverges from naive on this code.
    """
    from qldpc.circuits.surgery.circuit.PPM_XZ import build_single_ppm_circuit
    from qldpc.circuits.surgery.circuit.support import logical_state_init
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    z_bar = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z_bar, basis=Pauli.Z)
    circuit = build_single_ppm_circuit(
        g,
        rounds=3,
        noise_model=None,
        data_init=logical_state_init(code, state, log_idx=0),
    )
    # Raw measurement records — see lattice_surgery.ipynb §0 raw_observables.
    raw = circuit.compile_sampler().sample(shots=16).astype(np.uint8)
    n_meas = raw.shape[1]
    obs0_recs = []
    for ln in str(circuit).splitlines():
        if ln.startswith("OBSERVABLE_INCLUDE(0)"):
            obs0_recs = [int(t.strip("rec[]")) for t in ln.split() if t.startswith("rec[")]
            break
    obs0 = np.bitwise_xor.reduce(raw[:, [n_meas + off for off in obs0_recs]], axis=1)
    rate = float(obs0.mean())
    assert rate == float(expected_obs0), (
        f"state={state!r}: obs0 rate {rate:.3f} != expected {expected_obs0}"
    )


@pytest.mark.parametrize("state,expected_obs0", [("0", 0), ("1", 1)])
def test_logical_state_init_end_to_end_bbcode_basis_z(state: str, expected_obs0: int) -> None:
    """BBCode [[36, 8]] single-PPM (basis=Z): regression for even-weight Z̄.

    For BBCode (l=3, m=6) the chosen Z̄_0 has weight 8 (even), so naive
    broadcast `"1"*36` produces logical |0⟩_L (NOT |1⟩_L) and obs0=0,
    silently failing any truth table that hardcodes expected=1 for "1".

    The helper uses X̄_0 to flip the correct support, so obs0 tracks the
    textbook expectation. If this test ever returns obs0=0 for state="1",
    the helper has regressed to naive broadcast.
    """
    import sympy

    from qldpc.circuits.surgery.circuit.PPM_XZ import build_single_ppm_circuit
    from qldpc.circuits.surgery.circuit.support import logical_state_init
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    xs, ys = sympy.symbols("x y")
    code = codes.BBCode({xs: 3, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)
    z_bar = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    assert int(z_bar.sum()) % 2 == 0, "test premise broken: this BBCode should have even-wt Z̄_0"
    g = build_gadget(code, z_bar, basis=Pauli.Z)
    circuit = build_single_ppm_circuit(
        g,
        rounds=3,
        noise_model=None,
        data_init=logical_state_init(code, state, log_idx=0),
    )
    raw = circuit.compile_sampler().sample(shots=200).astype(np.uint8)
    n_meas = raw.shape[1]
    obs0_recs = []
    for ln in str(circuit).splitlines():
        if ln.startswith("OBSERVABLE_INCLUDE(0)"):
            obs0_recs = [int(t.strip("rec[]")) for t in ln.split() if t.startswith("rec[")]
            break
    obs0 = np.bitwise_xor.reduce(raw[:, [n_meas + off for off in obs0_recs]], axis=1)
    rate = float(obs0.mean())
    assert rate == float(expected_obs0), (
        f"state={state!r}: obs0 rate {rate:.3f} != expected {expected_obs0}. "
        f"This is the BBCode even-wt regression test — failure here means "
        f"logical_state_init is no better than naive '{state}' * n broadcast."
    )


@pytest.mark.parametrize("rounds", [1, 2, 3, 5, 10])
@pytest.mark.parametrize("state", ["0", "1"])
def test_multi_round_invariance_steane_basis_z(rounds: int, state: str) -> None:
    """The block Z̄ logical (observable index 0) reads the prepared eigenvalue
    independently of R.

    In match-basis (experiment_basis == gadget.basis == Z) the index-0 observable
    is the block Z̄ logical read from the FINAL destructive data measurement (Cain
    et al. arXiv:2603.28627 Appendix D), so it equals the prepared eigenvalue for
    every R ≥ 1:
      * state="0" (|0⟩^n → Z̄=+1): index 0 = 0
      * state="1" (|1⟩^n → Z̄=−1, wt(Z̄_Steane)=3 odd): index 0 = 1

    R-invariance guards _surgery_qec_cycle / _surgery_observable /
    MeasurementRecord.get_target_rec against round-index drift. (The companion
    time-like L at index k uses the FIRST-cycle merge-check product, per Webster,
    Smith, Cohen arXiv:2511.15989 §II.A Z̄ = ∏_v A_v; the earlier
    XOR-across-R-rounds bug silently zeroed that L for every even R.)
    """
    from qldpc.circuits.surgery.circuit.PPM_XZ import build_single_ppm_circuit
    from qldpc.circuits.surgery.circuit.support import logical_state_init
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    z_bar = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z_bar, basis=Pauli.Z)
    circuit = build_single_ppm_circuit(
        g,
        rounds=rounds,
        noise_model=None,
        data_init=logical_state_init(code, state, log_idx=0),
    )
    raw = circuit.compile_sampler().sample(shots=200).astype(np.uint8)
    n_meas = raw.shape[1]
    obs0_recs = []
    for ln in str(circuit).splitlines():
        if ln.startswith("OBSERVABLE_INCLUDE(0)"):
            obs0_recs = [int(t.strip("rec[]")) for t in ln.split() if t.startswith("rec[")]
            break
    obs0 = np.bitwise_xor.reduce(raw[:, [n_meas + off for off in obs0_recs]], axis=1)
    rate = float(obs0.mean())
    # index-0 block Z̄ (final destructive data readout) = eigenvalue bit of Z̄,
    # independent of R.
    expected_obs0 = int(state)
    assert rate == float(expected_obs0), (
        f"rounds={rounds}, state={state!r}: index-0 block Z̄ rate {rate:.3f} != "
        f"expected {expected_obs0} (block logical = prepared eigenvalue for any R)"
    )


@pytest.mark.parametrize("error_qubit", list(range(7)))
def test_single_qubit_x_error_triggers_only_neighboring_z_checks_steane(
    error_qubit: int,
) -> None:
    """Inject X_ERROR(1.0) on data qubit ``error_qubit`` between state
    prep and the first QEC round of the Steane basis=Z PPM. Assert
    exactly the round-1 Z-stab detectors whose support contains
    ``error_qubit`` fire (by row index, not just count).

    Why X_ERROR (not data_init):
    * Stim's detector sampler reports ``actual XOR tableau-predicted``.
      A state-prep-only change is already known to the tableau, so
      detectors stay 0 (no deviation from prediction).
    * X_ERROR(1.0) is a noise channel — the tableau prediction is
      computed without noise, so applying X always deviates the
      measured Z-stab parities from the prediction, firing the
      affected detectors.

    Why this catches stim wiring bugs:
    * Round-1 reliable Z-checks compare measured syndrome to +1.
    * An X error on data qubit i flips the parity of every Z-stab whose
      support contains i — exactly those detectors must fire, no others.
    * CX target/control swap, wrong measurement basis, or EdgeColoring
      delaying a check to a later round all break this exact-match
      pattern loudly.
    * The assertion checks the FIRED SET against the expected set of
      Z-stab row indices (not just the count) — a bug that swaps rows
      while preserving cardinality is caught.
    """
    from qldpc.circuits.surgery.circuit.PPM_XZ import build_single_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    z_bar = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z_bar, basis=Pauli.Z)
    clean_circuit = build_single_ppm_circuit(
        g,
        rounds=1,
        noise_model=None,
        data_init="0" * 7,
    )

    # Splice X_ERROR(1.0) at the boundary between state prep and QEC.
    # _surgery_state_prep emits only R, RX, X, Z instructions (closed
    # set) before the QEC cycle begins. Scan for the LAST such op and
    # insert immediately after — this is robust to future QEC ops
    # (MPP, XCX, etc.) that an open-set heuristic would misclassify.
    lines = str(clean_circuit).splitlines()
    prep_ops = ("R", "RX", "X", "Z")
    last_prep_idx = -1
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            continue  # pragma: no cover  -- stim's str() never emits blank lines today
        op = s.split()[0].split("(")[0]
        if op in prep_ops:
            last_prep_idx = i
    assert last_prep_idx >= 0, "could not locate any prep op (R/RX/X/Z) in Steane PPM circuit"
    injected_lines = (
        lines[: last_prep_idx + 1] + [f"X_ERROR(1.0) {error_qubit}"] + lines[last_prep_idx + 1 :]
    )
    injected_circuit = stim.Circuit("\n".join(injected_lines))

    sampler = injected_circuit.compile_detector_sampler()
    detection_events, _ = sampler.sample(
        shots=1,
        separate_observables=True,
    )
    events = detection_events[0]

    # Identify ROUND-1 reliable Z-side detectors via the clean reference:
    # deterministic-0 detectors emitted in the round-1 slab (time-coord
    # 0, before SHIFT_COORDS). Steane basis=Z rounds=1 emits 6 such
    # detectors total — 3 reliable round-1 Z-checks (time=0) and 3
    # final-readout cross-checks (time=1, after SHIFT_COORDS). We want
    # only the round-1 set: those are the ones flipped by X errors
    # injected before the first CZ extraction (the post-SHIFT detectors
    # check (round-1 syndrome) XOR (data-derived syndrome), which is
    # invariant under prep-time X errors and therefore stays at 0).
    #
    # The round-1 reliable detectors are emitted in data-H_Z row order
    # (set by _reliable_checks iterating qubit_ids.checks_z in row order,
    # of which the reliable ones are the first m_Z data H_Z rows), so
    # deterministic_zero_round1[j] corresponds to H_Z row j.
    clean_sampler = clean_circuit.compile_detector_sampler()
    clean_events, _ = clean_sampler.sample(
        shots=256,
        separate_observables=True,
    )
    all_det_zero = np.where(clean_events.sum(axis=0) == 0)[0]
    det_coords = clean_circuit.get_detector_coordinates()
    deterministic_zero = np.array(
        [d for d in all_det_zero if det_coords[d][2] == 0.0],
        dtype=int,
    )

    HZ = np.asarray(code.matrix_z).astype(int)
    n_reliable_z = HZ.shape[0]  # 3 for Steane
    assert len(deterministic_zero) == n_reliable_z, (
        f"expected exactly {n_reliable_z} round-1 deterministic-zero "
        f"detectors on clean Steane basis=Z PPM (rounds=1), got "
        f"{len(deterministic_zero)} — reliable-check emission order may "
        f"have changed"
    )

    # Steane Z-stabs touching error_qubit (row indices)
    z_stabs_touching = set(int(j) for j in np.where(HZ[:, error_qubit] == 1)[0])
    # Map each round-1 deterministic-zero detector position (sorted by
    # emission order) to its corresponding Z-stab row index. The fired
    # set is the set of row indices whose detector fired.
    fired_z_stab_rows = {j for j in range(len(deterministic_zero)) if events[deterministic_zero[j]]}
    assert fired_z_stab_rows == z_stabs_touching, (
        f"X_ERROR on qubit {error_qubit}: expected Z-stab rows "
        f"{sorted(z_stabs_touching)} to fire, got "
        f"{sorted(fired_z_stab_rows)}. This is the syndrome-extraction "
        f"wiring regression: CX swap, wrong measurement basis, "
        f"EdgeColoring schedule bug, or a stabilizer row that was "
        f"reordered/replaced. The set comparison catches bugs that "
        f"swap detector contents while preserving cardinality."
    )


def test_single_ppm_even_rounds_truth_table() -> None:
    """The time-like L must encode single-patch X̄ (or Z̄) parity at EVEN rounds.

    Same regression as test_joint_ppm_even_rounds_truth_table but for the
    single-patch PPM construction. Sweeps "+" and "-" data inits in basis=X (and
    "0"/"1" in basis=Z) — each is an experiment_basis eigenstate, so in match
    basis the block logical (index 0) and the time-like L (index k=1) both read
    the prepared parity bit. The XOR-across-rounds bug would silence the
    time-like L at even R. Uses compile_sampler + manual XOR.
    """
    from qldpc.circuits.surgery.circuit.PPM_XZ import build_single_ppm_circuit
    from qldpc.circuits.surgery.circuit.support import logical_state_init
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    code = codes.SteaneCode()
    basis_cases: list[tuple[PauliXZ, list[tuple[str, int]]]] = [
        (Pauli.X, [("+", 0), ("-", 1)]),
        (Pauli.Z, [("0", 0), ("1", 1)]),
    ]
    k = code.dimension  # time-like L at index k=1
    for basis, cases in basis_cases:
        op = (
            code.get_logical_ops(Pauli.X)[0]
            if basis is Pauli.X
            else code.get_logical_ops(Pauli.Z)[0]
        )
        op_arr = np.asarray(op).astype(np.uint8)
        g = build_gadget(code, op_arr, basis=basis)
        for state, expected in cases:
            data_init = logical_state_init(code, state=state, log_idx=0)
            circuit = build_single_ppm_circuit(
                g,
                rounds=2,
                noise_model=None,
                data_init=data_init,
            )
            raw = circuit.compile_sampler().sample(shots=16).astype(np.uint8)
            n_meas = raw.shape[1]
            obs_lines = [
                ln for ln in str(circuit).splitlines() if ln.startswith("OBSERVABLE_INCLUDE")
            ]
            vals = []
            for ln in obs_lines:
                offs = [int(t.strip("rec[]")) for t in ln.split() if t.startswith("rec[")]
                vals.append(np.bitwise_xor.reduce(raw[:, [n_meas + o for o in offs]], axis=1))
            block, time_L = vals[0], vals[k]
            assert (time_L == expected).all(), (
                f"basis={basis!r} state={state!r}: time-like L has "
                f"{(time_L != expected).sum()}/16 shots disagreeing with "
                f"expected parity bit {expected}"
            )
            # §3.4: block logical and time-like L agree on the eigenstate prep.
            assert (block == time_L).all(), (
                f"basis={basis!r} state={state!r}: block != time-like L in noiseless run"
            )


def test_single_ppm_dem_ok_bb_36_8_with_boost() -> None:
    """Single-PPM DEM constructs cleanly on BB [[36, 8]] with boost.

    Contract test: single-PPM does NOT call build_bridge / SkipTree, so the
    joint-PPM boost-drop and duplicate-edge bugs (fixed in bridge.py) cannot
    affect it. This regression locks that property in — both BB [[36, 8]]
    (duplicate weight-2 incidence rows on Z̄_0) AND boost (Cheeger h<1)
    simultaneously, the double-boundary case for the bridge bugs. If a future
    refactor accidentally routes single-PPM through bridge code, this test
    will catch it via stim's non-deterministic-detector rejection.
    """
    import sympy

    from qldpc.circuits.noise_model import DepolarizingNoiseModel
    from qldpc.circuits.surgery.circuit.PPM_XZ import build_single_ppm_circuit
    from qldpc.circuits.surgery.circuit.support import keep_only_observable
    from qldpc.circuits.surgery.hmatrix.cheeger import boost_gadget, cheeger_constant
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    xs, ys = sympy.symbols("x y")
    code = codes.BBCode({xs: 3, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    # Premise: restricted incidence has duplicate weight-2 rows.
    assert g.incidence.shape[0] > np.unique(g.incidence, axis=0).shape[0], (
        "test premise broken: BB [[36, 8]] Z̄_0 restriction should have duplicate κ rows"
    )
    if cheeger_constant(g) < 1.0:
        g = boost_gadget(g, method="combinatorial", target=1.0, max_extra_qubits=20, seed=3)

    noise = DepolarizingNoiseModel(1e-3, include_idling_error=False)
    circuit = build_single_ppm_circuit(g, rounds=3, noise_model=noise)
    stripped = keep_only_observable(circuit, keep_idx=0)
    dem = stripped.detector_error_model(approximate_disjoint_errors=True)
    assert dem.num_detectors > 0


def test_frame_correction_is_load_bearing_opposite_basis() -> None:
    """Opposite-basis (k-1) frame-corrected observables are noiselessly deterministic.

    The existing tests cover the opposite-basis observable COUNT
    (test_single_ppm_opposite_basis_emits_k_minus_1_observables); this asserts the
    Pauli-frame correction is load-bearing and correct end-to-end. Each of the k-1
    block observables is ``(final data parity) ⊕ (Q'-split parity)`` (design §3.2/§4);
    without folding in the Q'-split records the data-only readout would be random, so
    a noiseless ``not obs.any()`` confirms the frame correction folds the Q'-split
    records correctly. Uses the k=8 BBCode [[36,8]] with an X-gadget and
    experiment_basis=Z (the opposite basis), the same construction as the count test.
    """
    from qldpc.circuits.surgery.circuit.PPM_XZ import build_single_ppm_circuit
    from qldpc.circuits.surgery.hmatrix.PPM_XZ import build_gadget

    bb = _bb_36_8_code()  # dimension 8 -> k-1 = 7 frame-corrected observables
    xop = np.asarray(bb.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    gbb = build_gadget(bb, xop, basis=Pauli.X)
    circ = build_single_ppm_circuit(gbb, rounds=3, experiment_basis=Pauli.Z)

    # Non-vacuous: there must actually be k-1 = 7 frame-corrected observables to check.
    assert circ.num_observables == bb.dimension - 1 > 0

    _, obs = circ.compile_detector_sampler().sample(shots=128, separate_observables=True)
    assert obs.shape == (128, bb.dimension - 1)
    assert not obs.any()  # every frame-corrected opposite-basis observable is deterministic
