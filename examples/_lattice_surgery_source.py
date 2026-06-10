"""Source for lattice_surgery.ipynb.

End-to-end demo of qldpc.circuits.surgery:
  §1 Single-PPM correctness (noiseless determinism on multiple codes)
  §2 Joint-PPM correctness (inter-code Z̄⊗Z̄ + superposition variant)
  §3 Gadget construction vs published Webster Table I + Cain Table III
  §4 [[72, 12]] BB code logical-error-rate (surgery PPM vs memory baseline)

Edit this .py file (jupytext "percent" format) and run
    jupytext --to ipynb _lattice_surgery_source.py
to regenerate lattice_surgery.ipynb.
"""

# %% [markdown]
# # Lattice Surgery on qLDPC Codes
#
# This notebook demonstrates the public API of `qldpc.circuits.surgery`:
#
# | Function | Returns | Purpose |
# |---|---|---|
# | `build_gadget(code, x, basis=Pauli.X)` | `GadgetLayout` | Webster §II.A 3-step gadget for measuring logical X̄ (or Z̄ via `basis=Pauli.Z`) |
# | `build_bridge(g_l, g_r)` | `Bridge` | Universal adapter (arXiv:2410.03628 §IV) joining two gadgets for joint measurement |
# | `build_single_ppm_circuit(g, rounds, noise_model)` | `stim.Circuit` | Single-shot logical Pauli measurement circuit |
# | `build_joint_ppm_circuit(g_l, g_r, bridge, rounds, noise_model)` | `(stim.Circuit, joint_code)` | Two-code joint logical measurement |
# | `boost_gadget(g, method, target, seed)` | `GadgetLayout` | Cheeger / distance-verifying ancilla augmentation |
# | `cheeger_constant(g)` | `float` | Pre-boost Cheeger h(F) check |
# | `keep_only_observable(circuit, keep_idx)` | `stim.Circuit` | Strip all but one `OBSERVABLE_INCLUDE` for clean LER plots |
#
# References: Cain et al. arXiv:2603.28627; Webster, Smith, Cohen arXiv:2511.15989;
# Cross arXiv:2406.16294; Ide et al. arXiv:2410.03628.

# %% [markdown]
# ## §0 Setup

# %%
from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np
import sinter
import stim
import sympy
import matplotlib.pyplot as plt

from qldpc import codes, circuits, decoders
from qldpc.circuits.surgery import (
    build_gadget, build_bridge,
    build_single_ppm_circuit, build_joint_ppm_circuit,
    boost_gadget, cheeger_constant, keep_only_observable,
)
from qldpc.circuits.noise_model import DepolarizingNoiseModel
from qldpc.objects import Pauli

# Webster seed-set JSON lives under examples/; import its loader.
sys.path.insert(0, str(Path("..").resolve()))
sys.path.insert(0, str(Path(".").resolve()))
from _webster_seed_set import (  # noqa: E402
    load_webster_seed_set,
    build_generalised_bicycle_code,
)


# %% [markdown]
# ## §1 Single-PPM correctness
#
# The single-PPM protocol measures a logical X̄ (or Z̄ for `basis=Pauli.Z`).
# Correctness check on initial state |0⟩ⁿ (logical |0⟩ for a code where
# |0⟩ⁿ is a +1 eigenstate of all Z-stabilizers):
#
#   1. **Random outcome** — measuring X̄ on |0⟩_L gives ±1 with 50/50 odds.
#   2. **Consistency** — the Webster Eq.1 observable (`obs0`) must agree with
#      the cross-check X̄_M observable (`obs1`) on every shot.
#
# We test on 3 codes spanning increasing scale: Steane, Webster code 0, Gross BBCode.

# %%
def _swap_data_init_to_zero(circuit: stim.Circuit, data_ids: list[int]) -> stim.Circuit:
    """Replace RX(data_ids) with R(data_ids); preserve everything else.

    The single-PPM circuit initializes data qubits to |+⟩ via RX (basis=X).
    Swapping to R(data) → |0⟩ⁿ lets us test the protocol on a logical-Z
    eigenstate, where the measurement outcome should be random.
    """
    data_set = set(data_ids)
    out = stim.Circuit()
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            out.append(stim.CircuitRepeatBlock(
                op.repeat_count,
                _swap_data_init_to_zero(op.body_copy(), data_ids),
            ))
            continue
        if op.name == "RX":
            targets = [t.value for t in op.targets_copy()]
            data_targets = [t for t in targets if t in data_set]
            other_targets = [t for t in targets if t not in data_set]
            if data_targets:
                out.append("R", data_targets)
            if other_targets:
                out.append("RX", other_targets)
        else:
            out.append(op)
    return out


def verify_single_ppm(name: str, code, x_logical: np.ndarray, rounds: int = 3,
                      shots: int = 4000) -> dict:
    """Build PPM, swap |+⟩→|0⟩ on data, sample noiselessly, check randomness + consistency."""
    g = build_gadget(code, x_logical)
    circuit_plus = build_single_ppm_circuit(g, rounds=rounds, noise_model=None)
    n_data = code.num_qudits
    circuit_zero = _swap_data_init_to_zero(circuit_plus, list(range(n_data)))

    sampler = circuit_zero.compile_detector_sampler()
    _, observables = sampler.sample(shots=shots, separate_observables=True)
    obs0, obs1 = observables[:, 0], observables[:, 1]
    rate0, rate1 = float(obs0.mean()), float(obs1.mean())
    agree = float((obs0 == obs1).mean())

    out = {
        "name": name, "n_data": n_data,
        "kappa": len(g.kappa_qubits), "chi": len(g.V0),
        "rate0_eq1": rate0, "rate1_xbar": rate1, "obs_agree": agree,
        "passes": 0.40 < rate0 < 0.60 and 0.40 < rate1 < 0.60 and agree == 1.0,
    }
    return out


# %%
# Run the single-PPM check on 3 codes.
single_results = []

# 1. Steane [[7, 1, 3]]
steane = codes.SteaneCode()
x_steane = np.asarray(steane.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
single_results.append(verify_single_ppm("Steane [[7, 1, 3]]", steane, x_steane))

# 2. Webster code 0 [[62, 10, 6]] with X̄_1
d0 = load_webster_seed_set(0)
w0 = build_generalised_bicycle_code(d0["l"], d0["A"], d0["B"])
def _seed_op(d, name):
    pauli = name[0]
    l = d["l"]
    for seed in d["seeds"]:
        if seed["name"] == name and seed["pauli_type"] == pauli:
            L = np.zeros(l, dtype=np.uint8); R = np.zeros(l, dtype=np.uint8)
            for i in seed["L_support"]: L[i] = 1
            for i in seed["R_support"]: R[i] = 1
            return np.concatenate([L, R])
    raise ValueError(name)
x_w0 = _seed_op(d0, "X_bar_1")
single_results.append(verify_single_ppm("Webster 0 [[62, 10, 6]]", w0, x_w0))

# 3. Gross / IBM bivariate-bicycle [[144, 12, 12]]
xs, ys = sympy.symbols("x y")
gross = codes.BBCode({xs: 12, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)
x_gross = np.asarray(gross.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
single_results.append(verify_single_ppm("Gross BB [[144, 12, 12]]", gross, x_gross, shots=2000))

print(f"{'code':<26} | n_data  κ   χ   | obs0 flip  obs1 flip  agree  | ok")
print("-" * 92)
for r in single_results:
    flag = "✓" if r["passes"] else "✗"
    print(f"{r['name']:<26} | {r['n_data']:>6} {r['kappa']:>3} {r['chi']:>3}  "
          f"|   {r['rate0_eq1']:>6.1%}     {r['rate1_xbar']:>6.1%}    {r['obs_agree']:>5.1%}  | {flag}")

assert all(r["passes"] for r in single_results), "single-PPM correctness failed"
print("\n✓ noiseless single-PPM is genuine X̄_M measurement on all 3 codes")


# %% [markdown]
# ## §2 Joint-PPM correctness
#
# Two independent Steane copies, inter-code joint Z̄_1 ⊗ Z̄_2 measurement
# (`basis=Pauli.Z`). The joint observable should agree with the parity of
# the two individual Z̄ eigenvalues on every shot:
#
# | init | Z̄_1 | Z̄_2 | parity (obs0) |
# |---|---|---|---|
# | $\|0\rangle\|0\rangle$ | +1 | +1 | 0 |
# | $\|0\rangle\|1\rangle$ | +1 | -1 | 1 |
# | $\|1\rangle\|0\rangle$ | -1 | +1 | 1 |
# | $\|1\rangle\|1\rangle$ | -1 | -1 | 0 |

# %%
def _mutate_init(circuit: stim.Circuit, x_data_ids: list[int]) -> stim.Circuit:
    """Append X on listed data ids right after the first R(data) layer."""
    if not x_data_ids:
        return circuit
    x_set = set(x_data_ids)
    out = stim.Circuit()
    applied = False
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            out.append(stim.CircuitRepeatBlock(
                op.repeat_count, _mutate_init(op.body_copy(), x_data_ids),
            ))
            continue
        out.append(op)
        if not applied and op.name == "R":
            r_targets = [t.value for t in op.targets_copy()]
            x_targets = [q for q in r_targets if q in x_set]
            if x_targets:
                out.append("X", x_targets)
                applied = True
    return out


def raw_observables(circuit: stim.Circuit, shots: int) -> np.ndarray:
    """Sample raw measurement bits and reconstruct each OBSERVABLE_INCLUDE column."""
    sampler = circuit.compile_sampler()
    raw = sampler.sample(shots=shots).astype(np.uint8)
    n_meas = raw.shape[1]
    obs_lines = [ln for ln in str(circuit).splitlines() if ln.startswith("OBSERVABLE_INCLUDE")]
    cols = []
    for line in obs_lines:
        offsets = [int(t.strip("rec[]")) for t in line.split() if t.startswith("rec[")]
        meas_idx = [n_meas + off for off in offsets]
        cols.append(np.bitwise_xor.reduce(raw[:, meas_idx], axis=1))
    return np.stack(cols, axis=1)


# %%
# Inter-code joint Z̄_1 ⊗ Z̄_2 on two Steane copies.
c1, c2 = codes.SteaneCode(), codes.SteaneCode()
z1 = np.asarray(c1.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
z2 = np.asarray(c2.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
g1 = build_gadget(c1, z1, basis=Pauli.Z)
g2 = build_gadget(c2, z2, basis=Pauli.Z)
bridge = build_bridge(g1, g2)
rounds = 3  # odd → Webster Eq.1 ≡ Z̄_1 ⊗ Z̄_2

circuit_joint, joint_code = build_joint_ppm_circuit(
    g1, g2, bridge, rounds=rounds, noise_model=None,
)
print(f"joint code           : [[{joint_code.num_qudits}, {joint_code.dimension}]]")
print(f"bridge.width         : {bridge.width}")
print(f"extra κ_l, κ_r       : {bridge.extra_kappa_l.shape[0]}, {bridge.extra_kappa_r.shape[0]}")
print(f"T_l, H_R shapes      : {bridge.T_l.shape}, {bridge.H_R.shape}")
print()

n1 = c1.num_qudits
data1, data2 = list(range(n1)), list(range(n1, n1 + c2.num_qudits))
SHOTS_JOINT = 1000
print(f"{'state':>16} | Z̄_1  Z̄_2 | expected obs0 | measured obs0 (frac=1)  | ok")
print("-" * 86)
joint_pass = True
for label, flip_ids, expected in [
    ("|0⟩_L|0⟩_L", [],              0),
    ("|0⟩_L|1⟩_L", data2,           1),
    ("|1⟩_L|0⟩_L", data1,           1),
    ("|1⟩_L|1⟩_L", data1 + data2,   0),
]:
    obs = raw_observables(_mutate_init(circuit_joint, flip_ids), SHOTS_JOINT)
    rate = float(obs[:, 0].mean())
    ok = rate == float(expected)
    flag = "✓" if ok else "✗"
    print(f"{label:>16} | {'+1' if not flip_ids or flip_ids == data2 else '-1':>4}  "
          f"{'+1' if not flip_ids or flip_ids == data1 else '-1':>4} | "
          f"{expected:>13} | {rate:>10.2%} ({int(obs[:, 0].sum()):>3}/{SHOTS_JOINT})    | {flag}")
    joint_pass = joint_pass and ok

assert joint_pass, "joint-PPM correctness failed"
print("\n✓ joint Z̄_1 ⊗ Z̄_2 observable matches expected parity deterministically")

# %% [markdown]
# **Superposition variant.** Default `basis=Z` PPM initializes data via `R`
# (|0⟩, a Z eigenstate). To put `c2` in $|+\rangle_L$ we move its data ids
# from the `R` group into a separate `RX` group. The joint observable then
# becomes random (Z̄_2 random), and the two `OBSERVABLE_INCLUDE` paths
# (Webster Eq.1 + cross-check) must still agree on every shot.

# %%
def _switch_init_basis(circuit: stim.Circuit, plus_ids: list[int]) -> stim.Circuit:
    """Move `plus_ids` from the R group into a separate RX group → |+⟩^⊗ init."""
    if not plus_ids:
        return circuit
    plus_set = set(plus_ids)
    out = stim.Circuit()
    applied = False
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            out.append(stim.CircuitRepeatBlock(
                op.repeat_count, _switch_init_basis(op.body_copy(), plus_ids),
            ))
            continue
        if not applied and op.name == "R":
            targets = [t.value for t in op.targets_copy()]
            remain = [q for q in targets if q not in plus_set]
            move = [q for q in targets if q in plus_set]
            if remain:
                out.append("R", remain)
            if move:
                out.append("RX", move)
                applied = True
            continue
        out.append(op)
    return out


circuit_super = _switch_init_basis(circuit_joint, data2)
obs_super = raw_observables(circuit_super, SHOTS_JOINT)
rate0_super = float(obs_super[:, 0].mean())
rate1_super = float(obs_super[:, 1].mean())
agree_super = float((obs_super[:, 0] == obs_super[:, 1]).mean())
print(f"c1 in |0⟩_L (R), c2 in |+⟩_L (R→RX swap):")
print(f"  obs0 (Eq.1)         : {rate0_super:>6.1%} flips  (expected ~50% — Z̄_2 random)")
print(f"  obs1 (cross-check)  : {rate1_super:>6.1%} flips  (expected ~50%)")
print(f"  obs0 == obs1        : {agree_super:>6.1%}        (expected 100%)")
assert 0.4 < rate0_super < 0.6 and 0.4 < rate1_super < 0.6 and agree_super == 1.0
print("\n✓ joint observable is random (Z̄_2 ⟂ |+⟩) but obs0/obs1 agree on every shot")


# %% [markdown]
# ## §3 Gadget construction vs published results
#
# ### §3.1 Webster Table I — exact (κ, χ, r) match
#
# Webster, Smith & Cohen arXiv:2511.15989 Table I lists the ancilla counts
# (κ + χ + r) for X̄_1 on 4 generalised-bicycle codes. We reproduce them with
# `build_gadget` and verify each row.

# %%
WEBSTER_TABLE_I_KAPPA_CHI_R = [
    # (code_index, code_label,    X̄,     published κ+χ+r)
    (0, "[[62, 10]]",  "X̄_1", 19),
    (1, "[[126, 12]]", "X̄_1", 31),
    (2, "[[254, 14]]", "X̄_1", 49),
    (3, "[[510, 16]]", "X̄_1", 79),
]
print(f"{'Webster code':<22} {'X̄':<6} | κ   χ   r   κ+χ+r  | published")
print("-" * 80)
for code_index, label, op, published in WEBSTER_TABLE_I_KAPPA_CHI_R:
    d = load_webster_seed_set(code_index)
    code = build_generalised_bicycle_code(d["l"], d["A"], d["B"])
    x = _seed_op(d, "X_bar_1")
    g = build_gadget(code, x)
    kappa, chi, r = len(g.kappa_qubits), len(g.V0), g.G.shape[0]
    total = kappa + chi + r
    flag = "✓" if total == published else "✗"
    print(f"{label:<22} {op:<6} | {kappa:>3} {chi:>3} {r:>3}  {total:>5}    | "
          f"{published:>5}  {flag}")
    assert total == published, f"Webster code {code_index} ancilla mismatch"
print("\n✓ all 4 Webster Table I rows match exactly")

# %% [markdown]
# ### §3.2 Cain Table III bb_18 — exact (39, 20, 20)
#
# Cain et al. arXiv:2603.28627 Extended Data Table III row "bb_18 Resource"
# reports `(Qubits, X-checks, Z-checks) = (39, 20, 20)`. We reproduce this
# in 4 steps:
#
# 1. Build `bb_18` from Cain App. A Eq A11 polynomials.
# 2. Use BP+OSD + greedy stab reduction to find a weight-20 Z̄ representative.
# 3. Run `build_gadget` for the gadget construction.
# 4. Run `boost_gadget(method='combinatorial')` to add κ' ancillas until
#    the Cheeger constant `h(F) ≥ 1` (Webster's distance-preservation threshold).

# %%
from qldpc.codes.common import get_random_array

def find_weight_20_z_logical_rep(bb18: codes.CSSCode) -> np.ndarray:
    """BP+OSD + greedy stabilizer reduction to find wt(Z̄)=20."""
    HZ = np.asarray(bb18.matrix_z).astype(int)
    eff_check = np.vstack([bb18.get_matrix(Pauli.X), bb18.get_logical_ops(Pauli.Z)])
    decoder = decoders.get_decoder(eff_check, with_BP_OSD=True, max_iter=200)
    eff_syndrome = np.zeros(len(eff_check), dtype=int)
    field = bb18.field
    for _ in range(100_000):
        eff_syndrome[-bb18.dimension:] = get_random_array(
            field, bb18.dimension, satisfy=lambda v: v.any(),
        )
        cand = decoder.decode(eff_syndrome)
        if not np.array_equal(eff_check @ cand.view(field), eff_syndrome):
            continue
        cur = np.asarray(cand).astype(int)
        for _ in range(20):
            best = None
            for s_idx in range(HZ.shape[0]):
                nxt = (cur + HZ[s_idx]) % 2
                if int(nxt.sum()) < int(cur.sum()):
                    best = nxt; break
            if best is None:
                break
            cur = best
        if int(cur.sum()) == 20:
            return cur
    raise RuntimeError("Failed to find weight-20 Z̄ rep in 100K trials")


# %%
print("Cain App. A Eq A11: bb_18 with l=31, m=4")
print("  a = 1 + x^6 y + x^27")
print("  b = y^2 + x^15 y^3 + x^24")
xs, ys = sympy.symbols("x y")
bb18 = codes.BBCode((31, 4),
                    1 + xs**6 * ys + xs**27,
                    ys**2 + xs**15 * ys**3 + xs**24)
print(f"  built: [[{bb18.num_qubits}, {bb18.dimension}]]  (expected [[248, 10]])")
assert (bb18.num_qubits, bb18.dimension) == (248, 10)

print("\nStep 2: find weight-20 Z̄ rep …")
vec_20 = find_weight_20_z_logical_rep(bb18)
print(f"  wt(Z̄) = {int(vec_20.sum())}")

print("\nStep 3: build_gadget")
target_code = codes.CSSCode(bb18.matrix_z, bb18.matrix_x, is_subsystem_code=False)
g_bb = build_gadget(target_code, vec_20)
print(f"  bare gadget: (κ={len(g_bb.kappa_qubits)}, χ={len(g_bb.V0)}, G={g_bb.G.shape[0]})")

print("\nStep 4: boost_gadget (combinatorial, target h=1.0)")
g_bb_boosted = boost_gadget(g_bb, method="combinatorial", target=1.0,
                            max_extra_qubits=20, seed=3)
kappa_b, chi_b, gauge_b = (
    len(g_bb_boosted.kappa_qubits), len(g_bb_boosted.V0), g_bb_boosted.G.shape[0],
)
print(f"  boost added +{kappa_b - len(g_bb.kappa_qubits)} κ' qubits")
print(f"  boosted gadget: (κ={kappa_b}, χ={chi_b}, G={gauge_b})")
print(f"\n  Cain Table III target: (39, 20, 20)")
match = (kappa_b, chi_b, gauge_b) == (39, 20, 20)
print(f"  {'✓ EXACT MATCH' if match else '✗ mismatch (stochastic BP+OSD upstream)'}")


# %% [markdown]
# ## §4 LER comparison on `[[72, 12]]` BB code
#
# Compare two protocols on the same code under a depolarizing noise model:
#
# 1. **Surgery PPM** — `build_single_ppm_circuit` measuring X̄, keeping only
#    `obs0` (Webster Eq.1).
# 2. **Memory baseline** — `circuits.get_memory_experiment` idling for the
#    same number of rounds, keeping only X̄_0.
#
# Both decode with BP+LSD. We sweep `p ∈ [0.003, 0.008]` and plot LER vs p.

# %%
xs, ys = sympy.symbols("x y")
bb72 = codes.BBCode({xs: 6, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)
x_bb72 = np.asarray(bb72.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
g_bb72 = build_gadget(bb72, x_bb72)

h_bb72 = cheeger_constant(g_bb72)
print(f"code         : [[{bb72.num_qudits}, {bb72.dimension}]]")
print(f"Cheeger h(F) : {h_bb72:.3f}  (Webster threshold = 1.0)")
if h_bb72 < 1.0:
    print(f"→ boosting (h < 1.0)")
    g_bb72 = boost_gadget(g_bb72, method="combinatorial", target=1.0,
                          max_extra_qubits=20, seed=3)
    print(f"→ boosted F shape: {g_bb72.F.shape}, h(F_aug) = {cheeger_constant(g_bb72):.3f}")
else:
    print(f"→ h ≥ 1.0, no boost needed")


# %%
LER_ROUNDS = 9
LER_P_VALUES = list(np.linspace(0.003, 0.008, 6))
LER_MAX_SHOTS = 2000
LER_MAX_ERRORS = 100
LER_NUM_WORKERS = 4

surgery_tasks, memory_tasks = [], []
for p in LER_P_VALUES:
    noise = DepolarizingNoiseModel(p, include_idling_error=False)
    surg = build_single_ppm_circuit(g_bb72, rounds=LER_ROUNDS, noise_model=noise)
    surgery_tasks.append(sinter.Task(
        circuit=keep_only_observable(surg, keep_idx=0),
        json_metadata={"p": float(p), "kind": "surgery"},
    ))
    mem = circuits.get_memory_experiment(
        bb72, basis=Pauli.X, num_rounds=LER_ROUNDS, noise_model=noise,
    )
    memory_tasks.append(sinter.Task(
        circuit=keep_only_observable(mem, keep_idx=0),
        json_metadata={"p": float(p), "kind": "memory"},
    ))

decoder = decoders.SinterDecoder(
    with_BP_LSD=True, max_iter=20, bp_method="ms",
    lsd_method="lsd_cs", lsd_order=3,
)

print(f"sweeping p ∈ [{LER_P_VALUES[0]:.4f}, {LER_P_VALUES[-1]:.4f}] "
      f"({len(LER_P_VALUES)} points, max_shots={LER_MAX_SHOTS}, max_errors={LER_MAX_ERRORS})")
print(f"  decoder = BP+LSD (qldpc.decoders.SinterDecoder)")

t0 = time.time()
results = sinter.collect(
    tasks=surgery_tasks + memory_tasks,
    decoders=["custom"],
    custom_decoders={"custom": decoder},
    num_workers=LER_NUM_WORKERS,
    max_shots=LER_MAX_SHOTS,
    max_errors=LER_MAX_ERRORS,
    print_progress=False,
)
print(f"collected {len(results)} task results in {time.time() - t0:.1f}s")


# %%
surgery_lers, memory_lers = {}, {}
for r in results:
    p = r.json_metadata["p"]; kind = r.json_metadata["kind"]
    ler = r.errors / max(r.shots, 1)
    (surgery_lers if kind == "surgery" else memory_lers)[p] = (ler, r.errors, r.shots)

print(f"{'p':>10} | {'surgery LER':>22} | {'memory LER':>22}")
print(f"{'-'*10} | {'-'*22} | {'-'*22}")
for p in sorted(LER_P_VALUES):
    s, se, ss = surgery_lers.get(p, (np.nan, 0, 0))
    m, me, ms = memory_lers.get(p, (np.nan, 0, 0))
    print(f"{p:>10.4f} | {s:>10.4f} ({se:>3}/{ss:<6}) | "
          f"{m:>10.4f} ({me:>3}/{ms:<6})")

# %%
fig, ax = plt.subplots(figsize=(7.5, 5))
ps_sorted = sorted(LER_P_VALUES)
ax.loglog(ps_sorted, [surgery_lers[p][0] for p in ps_sorted],
          "o-", label="Surgery PPM (obs0 = Webster Eq.1)", markersize=8, linewidth=2)
ax.loglog(ps_sorted, [memory_lers[p][0] for p in ps_sorted],
          "s-", label=f"Memory X̄ ({LER_ROUNDS} rounds idling)", markersize=8, linewidth=2)
ax.set_xlabel("physical error rate $p$")
ax.set_ylabel("logical error rate")
ax.set_title(f"BBCode [[{bb72.num_qudits}, {bb72.dimension}]] — Surgery PPM vs Memory "
             f"({LER_ROUNDS} rounds, BP+LSD)")
ax.legend(loc="upper left")
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()
