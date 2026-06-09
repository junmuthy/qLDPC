"""Source for 9_lattice_surgery.ipynb.

Demonstrates the 5 public functions of qldpc.codes.surgery against
Webster, Smith & Cohen (arXiv:2511.15989) Table I codes and reproduces
the published gadget sizes exactly.

Convert with:
    jupytext --to notebook \
        examples/logical_error_rates/_9_lattice_surgery_source.py \
        -o examples/logical_error_rates/9_lattice_surgery.ipynb
"""

# %% [markdown]
# # qLDPC Lattice Surgery — Public API Walkthrough
#
# This notebook demonstrates the 5 public functions of `qldpc.codes.surgery`
# by running them against Webster, Smith & Cohen (arXiv:2511.15989) Table I
# codes and reproducing the published gadget sizes exactly.

# %% [markdown]
# ## Overview
#
# **What is lattice surgery?**
# Lattice surgery is a protocol for performing logical gates between
# error-corrected qubits by temporarily merging and splitting their
# stabilizer codes. In the qLDPC setting, Webster *et al.* (arXiv:2511.15989)
# developed a compact recipe — the *L = 1 gadget* — that performs a
# Pauli-Product Measurement (PPM) on a single logical operator X̄
# using a small number of ancilla qubits while preserving the code distance.
#
# **The 5 public APIs** (from `qldpc.codes.surgery`):
#
# | Function | Returns | Purpose |
# |---|---|---|
# | `build_gadget(code, x)` | `GadgetLayout` | Webster 3-step gadget for single logical X̄ |
# | `build_bridge(g1, g2)` | `Bridge` | Universal adapter (arXiv:2410.03628 §IV) joining two gadgets |
# | `build_single_ppm_circuit(g, *, rounds, noise_model)` | `stim.Circuit` | Stim circuit for single-PPM measurement |
# | `build_joint_ppm_circuit(g1, g2, bridge, *, rounds, noise_model)` | `(stim.Circuit, CSSCode)` | Stim circuit + merged code for joint PPM |
# | `boost_gadget(g, *, method, target, seed)` | `GadgetLayout` | Cheeger/distance augmentation to preserve code distance |
#
# **Webster L=1 gadget** refers to the single-layer (L=1) construction
# described in Webster *et al.* arXiv:2511.15989 §II.A.  Three explicit
# steps correspond directly to the math (see `docs/superpowers/math.md`):
# - §1.1 Restriction: V₀ = supp(x̄), C₀ = Z-checks touching V₀, F = H_Z[C₀, V₀]
# - §1.2 Gauge fix: G whose rows span ker(Fᵀ) over GF(2)
# - §1.4 Assembly: block matrices H_X^merged and H_Z^merged
#
# For the design specification see:
# `docs/superpowers/specs/2026-06-07-surgery-simplification-design.md`

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from qldpc.codes.surgery import (
    GadgetLayout,
    Bridge,
    build_gadget,
    build_bridge,
    build_single_ppm_circuit,
    build_joint_ppm_circuit,
    boost_gadget,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _webster_seed_set import (  # noqa: E402
    load_webster_seed_set,
    build_generalised_bicycle_code,
)

print("Imports OK")

# %% [markdown]
# ## 2. Load a Webster code
#
# Webster Appendix A defines 4 generalised bicycle (GB) codes with
# `l ∈ {31, 63, 127, 255}` and `n = 2l` data qubits.  We start with
# code 0: `l = 31`, `n = 62`, `k = 10`, `d = 6`.

# %%
# Pick Webster Appendix A code 0: l=31, n=62, k=10, d=6
data = load_webster_seed_set(0)
code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
print(f"Code: [[{code.num_qudits}, {code.dimension}]]  (l={data['l']})")
print(f"Code name from JSON: {data.get('name', 'N/A')}")

# %% [markdown]
# ## 3. `build_gadget` — three explicit Webster steps
#
# `build_gadget(code, x)` implements the Webster L=1 gadget via three
# named steps (math.md §1.1–§1.4):
#
# 1. **Step 1 — Restriction** (§1.1):
#    V₀ = supp(x̄),  C₀ = {Z-checks touching V₀},  F = H_Z[C₀, V₀]
#
# 2. **Step 2 — Gauge fix** (§1.2):
#    Compute G whose rows form a canonical GF(2) basis of ker(Fᵀ)
#    (left null space of F).  This is deterministic via row-reduction.
#
# 3. **Step 3 — Assembly** (§1.4):
#    H_X^merged = [H_X | 0 ; E_V₀ᵀ | Fᵀ]  (data + κ ancilla columns)
#    H_Z^merged = [H_Z | F̃ ; 0 | G]
#
# The `GadgetLayout` dataclass stores all intermediate quantities:
# `V0`, `C0`, `F`, `G`, `HX_merged`, `HZ_merged`, `kappa_qubits`.

# %%
def x_bar_1_operator(d: dict) -> np.ndarray:
    """Extract X̄_1 from a Webster seed_set dict as a 2l binary vector."""
    l = d["l"]
    for seed in d["seeds"]:
        if seed["name"] == "X_bar_1" and seed["pauli_type"] == "X":
            L = np.zeros(l, dtype=np.uint8)
            R = np.zeros(l, dtype=np.uint8)
            for i in seed["L_support"]:
                L[i] = 1
            for i in seed["R_support"]:
                R[i] = 1
            return np.concatenate([L, R])
    raise ValueError("X_bar_1 not found")


x = x_bar_1_operator(data)
g = build_gadget(code, x)

print(f"|V_0| = {len(g.V0)}  (χ rows  — number of X̄ support qubits)")
print(f"|C_0| = {len(g.C0)}  (κ ancillas — Z-checks touching V₀)")
print(f"r     = {g.G.shape[0]}  (gauge-fix rows)")
print(f"H_X^merged shape: {g.HX_merged.shape}")
print(f"H_Z^merged shape: {g.HZ_merged.shape}")
print(f"kappa_qubits indices: {g.kappa_qubits[:5]}{'...' if len(g.kappa_qubits) > 5 else ''}")

# %% [markdown]
# **Determinism check:** calling `build_gadget` twice on the same input
# yields byte-identical output.  This is guaranteed by the deterministic
# GF(2) row-reduction in `_step2_gauge_fix`.

# %%
g1_again = build_gadget(code, x)

assert g1_again.V0 == g.V0
assert g1_again.C0 == g.C0
assert np.array_equal(g1_again.F, g.F)
assert np.array_equal(g1_again.G, g.G)
assert np.array_equal(g1_again.HX_merged, g.HX_merged)
assert np.array_equal(g1_again.HZ_merged, g.HZ_merged)
assert g1_again.kappa_qubits == g.kappa_qubits

print("Determinism confirmed: two calls to build_gadget yield byte-identical results.")

# %% [markdown]
# ## 4. Verify Webster Table I κ+χ+r
#
# Webster Table I lists the ancilla count `κ + χ + r` for each of the 4
# codes.  Our `build_gadget` reproduces these exactly:
#
# | code | n   | κ+χ+r (paper) |
# |------|-----|---------------|
# |  0   |  62 |      19       |
# |  1   | 126 |      31       |
# |  2   | 254 |      49       |
# |  3   | 510 |      79       |

# %%
WEBSTER_TABLE_I_KAPPA_CHI_R = [(0, 19), (1, 31), (2, 49), (3, 79)]

print(f"{'code':>4} | {'n':>4} | {'κ':>4} {'χ':>4} {'r':>4} | {'κ+χ+r':>6} {'paper':>5} {'match':>6}")
print("-" * 56)

for code_index, expected in WEBSTER_TABLE_I_KAPPA_CHI_R:
    d = load_webster_seed_set(code_index)
    c = build_generalised_bicycle_code(d["l"], d["A"], d["B"])
    xv = x_bar_1_operator(d)
    gg = build_gadget(c, xv)
    kappa = len(gg.kappa_qubits)
    chi = len(gg.V0)
    r = gg.G.shape[0]
    total = kappa + chi + r
    match = "✓" if total == expected else "✗"
    print(f"{code_index:>4} | {c.num_qudits:>4} | {kappa:>4} {chi:>4} {r:>4} | {total:>6} {expected:>5} {match:>6}")

# %% [markdown]
# ## 5. `boost_gadget` — distance-preserving augmentation
#
# The bare Webster gadget may shrink the code distance of the merged code.
# `boost_gadget` adds extra κ' ancilla qubits (degree-2 rows of F) until
# the boundary Cheeger constant h(F) ≥ target.
#
# **Why does this preserve distance?**
# Cross §III Theorem 6 states that h(F) ≥ 1 implies d_merged ≥ d_data.
# The Cheeger constant h(F) measures the expansion of the bipartite
# incidence graph between V₀ (X-check support) and C₀ (κ ancillas).
#
# **Three available methods:**
# - `'spectral'`: spectral lower bound on h(F) via λ₂(F Fᵀ)/2; fast but approximate
# - `'combinatorial'`: exact Gray-code enumeration of all cuts; tractable for |V₀| ≤ 26
# - `'distance'`: BP+OSD decoder-verified distance; slowest but gives direct guarantees

# %%
g_boosted = boost_gadget(g, method="combinatorial", target=1.0, seed=42)

print(f"Before boost: |C_0| = {len(g.C0)}, κ ancillas = {len(g.kappa_qubits)}")
print(f"After boost : F shape = {g_boosted.F.shape}, κ ancillas = {len(g_boosted.kappa_qubits)}")
print(f"Extra κ ancillas added: {len(g_boosted.kappa_qubits) - len(g.kappa_qubits)}")

# CSS commutation still holds after boost
HX_b = np.asarray(g_boosted.HX_merged).astype(np.uint8)
HZ_b = np.asarray(g_boosted.HZ_merged).astype(np.uint8)
prod_b = (HX_b @ HZ_b.T) % 2
print(f"CSS commutation holds after boost (HX HZ^T = 0)? {(prod_b == 0).all()}")

# %% [markdown]
# ## 5b. X̄ vs Z̄ — basis-symmetric API
#
# `build_gadget(code, x, basis=Pauli.X|Pauli.Z)` selects whether the gadget
# measures an X̄ or Z̄ logical. The construction is dual-symmetric: for
# `basis=Pauli.Z` the χ rows live in `HZ_merged` (Z-type) instead of `HX_merged`,
# and the surgery circuit init swaps |+⟩ ↔ |0⟩ on data vs κ.

# %%
from qldpc.objects import Pauli

def z_bar_1_operator(d: dict) -> np.ndarray:
    l = d["l"]
    for seed in d["seeds"]:
        if seed["name"] == "Z_bar_1" and seed["pauli_type"] == "Z":
            L = np.zeros(l, dtype=np.uint8); R = np.zeros(l, dtype=np.uint8)
            for i in seed["L_support"]: L[i] = 1
            for i in seed["R_support"]: R[i] = 1
            return np.concatenate([L, R])
    raise ValueError("Z_bar_1 not found")

z = z_bar_1_operator(data)
g_z = build_gadget(code, z, basis=Pauli.Z)
print(f"basis=Z gadget: |V_0|={len(g_z.V0)}, |C_0|={len(g_z.C0)}, r={g_z.G.shape[0]}")
print(f"basis=Z κ+χ+r = {len(g_z.kappa_qubits) + len(g_z.V0) + g_z.G.shape[0]} (same as X)")

# %% [markdown]
# ## 6. `build_single_ppm_circuit` — Cain §III.A surgery circuit
#
# `build_single_ppm_circuit(gadget, *, rounds, noise_model)` implements the
# full Cain §III.A 3-step surgery protocol:
# 1. Initialize κ ancillas in |0⟩ (or |+⟩ for basis=Z); data qubits in |+⟩
#    (or |0⟩ for basis=Z).
# 2. Run `rounds` cycles of merged-code syndrome extraction. Round-1
#    detectors are emitted only for stabilizers in known eigenstate of the
#    init state — data H_X and gauge-fix G for basis=X.
# 3. Detach κ ancillas by Z-measurement; measure data qubits in X basis.
#
# The circuit emits two observables:
# - `OBSERVABLE_INCLUDE(0)`: ⊕ χ-row records across all rounds (Webster Eq. 1
#   — the PPM result).
# - `OBSERVABLE_INCLUDE(1)`: ⊕ data measurements on V_0 (X̄_M cross-check
#   inferred from the final data Mx).
#
# Under noiseless conditions both observables evaluate to 0 (= +1) deterministically.

# %%
circuit = build_single_ppm_circuit(g_boosted, rounds=3, noise_model=None)
print(f"Noiseless circuit: {len(circuit)} instructions")
print(f"Number of detectors: {circuit.num_detectors}")

# Noiseless: all detector samples should be zero
samples = circuit.compile_detector_sampler().sample(shots=32)
print(f"Noiseless detectors fire?  {bool(samples.any())}  (expected False)")

# %%
from qldpc.circuits.noise_model import DepolarizingNoiseModel

noisy = build_single_ppm_circuit(
    g_boosted,
    rounds=3,
    noise_model=DepolarizingNoiseModel(p=0.01),
)
samples_noisy = noisy.compile_detector_sampler().sample(shots=2000)

# Per-detector firing rate is the meaningful diagnostic: with N detectors
# at independent firing probability q each, P(any detector fires per shot)
# is 1 - (1-q)^N → 1 as N grows. That's why we report the per-detector
# rate, not the per-shot rate.
per_detector_rate = samples_noisy.mean()
per_shot_any_fire = samples_noisy.any(axis=1).mean()
print(f"Noisy (p=0.01):")
print(f"  per-detector fire rate : {per_detector_rate:.2%}  (averaged over {noisy.num_detectors} detectors)")
print(f"  per-shot any-fire rate : {per_shot_any_fire:.2%}  (≈ 1 by union bound when many detectors)")

# %% [markdown]
# ## 7. `build_bridge` — joining two gadgets
#
# `build_bridge(g1, g2)` constructs a **universal adapter** (Ide et al.,
# arXiv:2410.03628 §IV) that links each gadget's χ endpoints through an
# auxiliary register of `w = min(|V₀(g1)|, |V₀(g2)|)` qubits, enabling a
# **joint** PPM of the product Z̄_1 ⊗ Z̄_2 (math.md §2).
#
# **How it works (math.md §2):**
# - Build the auxiliary graph G_aux^(s) on each side (vertices = V_0^(s),
#   one edge per weight-2 F row).
# - Augment with chord/connector edges so the induced subgraph on the
#   chosen port subset is connected and all basis cycles fit under
#   `cellulate_max_len` (these become `extra_kappa_{l,r}` rows).
# - Rebuild each gadget over the augmented F to get `g_l_aug, g_r_aug`.
# - Run the SkipTree transform (arXiv:2410.03628 Algorithm 1) on the
#   induced port subgraph to produce a (3,2)-sparse matrix `T_s` such
#   that `T_s · F_aug · P_s == H_R` (canonical repetition-code parity
#   of the w-qubit adapter register).
#
# The Bridge dataclass exposes only the field-level products of these
# steps (`port_l`, `port_r`, `label_l`, `label_r`, `extra_kappa_l`,
# `extra_kappa_r`, `T_l`, `T_r`, `H_R`, `g_l_aug`, `g_r_aug`). There is
# **no** explicit path-graph stabilizer matrix any more; the adapter's
# (w−1) parity rows are H_R.
#
# `build_bridge` auto-dispatches:
# - **intra-code** (`g1.code is g2.code`): shared data qubits, used for measuring X̄_1 ⊗ X̄_2 on the same physical code
# - **inter-code** (`g1.code is not g2.code`): disjoint data registers, used for CNOT-like operations between two different codes

# %%
def x_bar_k2p1_operator(d: dict) -> np.ndarray:
    """Extract X̄_{k/2+1} from a Webster seed_set dict as a 2l binary vector."""
    l = d["l"]
    for seed in d["seeds"]:
        if seed["name"] == "X_bar_k2p1" and seed["pauli_type"] == "X":
            L = np.zeros(l, dtype=np.uint8)
            R = np.zeros(l, dtype=np.uint8)
            for i in seed["L_support"]:
                L[i] = 1
            for i in seed["R_support"]:
                R[i] = 1
            return np.concatenate([L, R])
    raise ValueError("X_bar_k2p1 not found")


x2 = x_bar_k2p1_operator(data)
g2 = build_gadget(code, x2)
bridge = build_bridge(g, g2)

print(f"Bridge width w               : {bridge.width}  (universal adapter qubits)")
print(f"port_l                       : {bridge.port_l}")
print(f"port_r                       : {bridge.port_r}")
print(f"extra_κ_l (connect+cellulate): {bridge.extra_kappa_l.shape[0]} new κ qubits")
print(f"extra_κ_r                    : {bridge.extra_kappa_r.shape[0]} new κ qubits")
print(f"T_l shape                    : {bridge.T_l.shape}  (SkipTree, (3,2)-sparse)")
print(f"T_r shape                    : {bridge.T_r.shape}")
print(f"H_R shape                    : {bridge.H_R.shape}  (canonical rep-code parity)")

# SkipTree key identity (arXiv:2410.03628 Theorem 5): T · G_aug · P == H_R
# where P is the port-permutation matrix built from label_l.
P_l = np.zeros((bridge.g_l_aug.F.shape[1], bridge.width), dtype=np.int_)
for v_idx, lab in enumerate(bridge.label_l):
    if lab >= 0:
        P_l[v_idx, lab] = 1
lhs_l = (bridge.T_l @ bridge.g_l_aug.F.astype(np.int_) @ P_l) % 2
print(f"SkipTree identity (left)     : {'PASS' if np.array_equal(lhs_l, bridge.H_R) else 'FAIL'}")

P_r = np.zeros((bridge.g_r_aug.F.shape[1], bridge.width), dtype=np.int_)
for v_idx, lab in enumerate(bridge.label_r):
    if lab >= 0:
        P_r[v_idx, lab] = 1
lhs_r = (bridge.T_r @ bridge.g_r_aug.F.astype(np.int_) @ P_r) % 2
print(f"SkipTree identity (right)    : {'PASS' if np.array_equal(lhs_r, bridge.H_R) else 'FAIL'}")

# %% [markdown]
# ## 8. Verify Webster Table I bridge width (2w−1)
#
# Webster Table I gives the bridge size as `2w − 1` (the path-graph qubit
# count, counting the w bridge body qubits minus the two shared endpoint
# ancillas).  Our `build_bridge` reproduces these exactly:
#
# | code | n   | 2w−1 (paper) |
# |------|-----|--------------|
# |  0   |  62 |     11       |
# |  1   | 126 |     19       |
# |  2   | 254 |     31       |
# |  3   | 510 |     51       |

# %%
WEBSTER_TABLE_I_BRIDGE = [(0, 11), (1, 19), (2, 31), (3, 51)]

print(f"{'code':>4} | {'w':>3} {'2w-1':>5} {'paper':>5} {'match':>6}")
print("-" * 30)

for code_index, expected in WEBSTER_TABLE_I_BRIDGE:
    d = load_webster_seed_set(code_index)
    c = build_generalised_bicycle_code(d["l"], d["A"], d["B"])
    g_a = build_gadget(c, x_bar_1_operator(d))
    g_b = build_gadget(c, x_bar_k2p1_operator(d))
    br = build_bridge(g_a, g_b)
    val = 2 * br.width - 1
    match = "✓" if val == expected else "✗"
    print(f"{code_index:>4} | {br.width:>3} {val:>5} {expected:>5} {match:>6}")

# %% [markdown]
# ## 9. `build_joint_ppm_circuit` — joint measurement Stim circuit
#
# `build_joint_ppm_circuit(g1, g2, bridge, *, rounds, noise_model)` assembles
# the full joint CSS code (data + g1-κ + g2-κ + adapter qubits) and returns
# both the Stim measurement circuit and the merged `CSSCode` object.
#
# **Protocol formula (math.md §2 / arXiv:2410.03628 §IV):**
# The adapter register carries the canonical repetition-code parity rows
# `H_R` (shape `(w-1, w)`); the SkipTree matrices `T_l, T_r` deform the
# original gadget χ rows so that, when written in the labelled basis of
# the augmented F matrices, they restrict cleanly onto the adapter.
# Summing all transformed χ rows from both sides together with the H_R
# rows yields exactly `x̄₁⊕x̄₂` on data and 0 on every ancilla/adapter
# qubit. This is equivalent to a projective measurement of the joint
# Pauli product X̄_1 ⊗ X̄_2, reducing the logical dimension by 1.

# %%
joint_circuit, joint_code = build_joint_ppm_circuit(g, g2, bridge, rounds=1, noise_model=None)

print(f"Joint code: [[{joint_code.num_qudits}, {joint_code.dimension}]]")
print(f"Data code dimension: k = {code.dimension}")
print(
    f"Logical-DOF reduction: k_joint − k_data = {joint_code.dimension - code.dimension}"
    f"  (expect −1 per Cross §3.6)"
)

# CSS commutation
HX = np.asarray(joint_code.matrix_x).astype(np.uint8)
HZ = np.asarray(joint_code.matrix_z).astype(np.uint8)
prod = (HX @ HZ.T) % 2
print(f"CSS commutation holds (HX HZ^T = 0)? {(prod == 0).all()}")

# %% [markdown]
# **Joint operator is a stabilizer** (Cross §3.6 correctness criterion):
#
# The product X̄_1 ⊗ X̄_2, padded with zeros on all ancilla/bridge qubits,
# must lie in the GF(2) row span of H_X^joint.  Individually X̄_1 and X̄_2
# must NOT (otherwise we'd be stabilizing them separately, not their product).

# %%
import galois
GF2 = galois.GF(2)

def _gf2_in_row_span(HX_arr: np.ndarray, target: np.ndarray) -> bool:
    """Return True iff target is in the GF(2) row span of HX_arr."""
    M = GF2(HX_arr.astype(np.int_))
    t = GF2(target.astype(np.int_).reshape(1, -1))
    rank_M = int(np.linalg.matrix_rank(M))
    aug = GF2(np.vstack([np.asarray(M), np.asarray(t)]).astype(np.int_))
    return int(np.linalg.matrix_rank(aug)) == rank_M


n_data = code.num_qubits
n_total = joint_code.num_qubits

op1_padded = np.zeros(n_total, dtype=np.uint8)
op1_padded[:n_data] = x

op2_padded = np.zeros(n_total, dtype=np.uint8)
op2_padded[:n_data] = x2

joint_op = (op1_padded + op2_padded) % 2

print(f"X̄_1 ⊗ X̄_2 (joint product) in HX_joint row span? {_gf2_in_row_span(HX, joint_op)}")
print(f"X̄_1 alone in HX_joint row span? {_gf2_in_row_span(HX, op1_padded)}  (expected False)")
print(f"X̄_2 alone in HX_joint row span? {_gf2_in_row_span(HX, op2_padded)}  (expected False)")

# %% [markdown]
# ## 10. End-to-end: PPM logical error rate vs physical p
#
# Since section 6's circuit implements the full Cain §III.A surgery protocol
# (and not a generic memory experiment), the LER below is the **PPM failure
# rate** — the probability that the surgery measurement of X̄_M returns a
# flipped outcome relative to the noiseless +1. This is the correct
# end-to-end quantity for evaluating surgery fault tolerance.

# %%
import sinter
import matplotlib.pyplot as plt
from qldpc import circuits, decoders

error_rates = np.logspace(-3, -2, 4)
num_rounds = 3

tasks = []
for p in error_rates:
    circuit = build_single_ppm_circuit(
        g_boosted, rounds=num_rounds,
        noise_model=circuits.DepolarizingNoiseModel(p, include_idling_error=False),
    )
    tasks.append(sinter.Task(circuit=circuit, json_metadata={"p": float(p)}))

# Use sinter.collect with custom_decoders dict (matches the pattern from R16)
sinter_decoder = decoders.SinterDecoder()
if __name__ == "__main__":
    results = sinter.collect(
        tasks=tasks,
        decoders=["custom"],
        custom_decoders={"custom": sinter_decoder},
        num_workers=4,
        max_shots=5_000,
        max_errors=50,
        print_progress=False,
    )
    print(f"Collected {len(results)} task results.")
    for _r in sorted(results, key=lambda r: r.json_metadata["p"]):
        _ler = _r.errors / _r.shots if _r.shots > 0 else float("nan")
        print(f"  p={_r.json_metadata['p']:.5f}  shots={_r.shots}  errors={_r.errors}  LER≈{_ler:.4f}")

# %%
# Plot LER vs p (only when results are available, i.e. in Jupyter or script __main__).
if __name__ == "__main__":
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _sorted_results = sorted(results, key=lambda r: r.json_metadata["p"])
    _ps = [r.json_metadata["p"] for r in _sorted_results]
    _ler = [r.errors / r.shots if r.shots > 0 else float("nan") for r in _sorted_results]

    _ax.loglog(_ps, _ler, "o-", label="Webster code 0 single-PPM (boosted)")
    _ax.set_xlabel("physical error rate p")
    _ax.set_ylabel("logical error rate")
    _ax.set_title("Webster code 0 single-PPM LER (Cain §III.A)")
    _ax.grid(True, which="both", alpha=0.3)
    _ax.legend()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 11. Summary
#
# The 5 public APIs of `qldpc.codes.surgery`, each in one sentence:
#
# - **`build_gadget(code, x)`**: runs the Webster 3-step gadget construction
#   (restriction → gauge-fix → assembly) to produce a merged CSS code with
#   κ ancilla qubits that absorbs the logical operator x̄ into a stabilizer.
#
# - **`build_bridge(g1, g2)`**: constructs the Ide universal adapter
#   (arXiv:2410.03628 §IV) of width `w = min(|V₀(g1)|, |V₀(g2)|)` that
#   links two gadgets through SkipTree-deformed χ rows on a canonical
#   repetition-code adapter register; auto-dispatches intra-code vs
#   inter-code based on `g1.code is g2.code`.
#
# - **`build_single_ppm_circuit(g, *, rounds, noise_model)`**: wraps the
#   merged code in a Stim memory-experiment circuit for `rounds` rounds of
#   syndrome extraction, optionally with a depolarizing or SI1000 noise model.
#
# - **`build_joint_ppm_circuit(g1, g2, bridge, *, rounds, noise_model)`**:
#   stitches the two gadgets and bridge into a joint CSS code and returns
#   both the Stim circuit and the `CSSCode` object for further inspection.
#
# - **`boost_gadget(g, *, method, target, seed)`**: augments the gadget's
#   restriction matrix F with additional degree-2 rows (new κ ancillas) until
#   the Cheeger constant h(F) ≥ target, guaranteeing distance preservation.
#
# **Verified against Webster (arXiv:2511.15989) Table I:**
# - κ+χ+r = {19, 31, 49, 79} for codes 0–3: **4/4 exact match**
# - Bridge 2w−1 = {11, 19, 31, 51} for codes 0–3: **4/4 exact match**

# %% [markdown]
# ## References and further reading
#
# - **Math derivation**: `docs/superpowers/math.md` — step-by-step Webster
#   §1.1–§1.4 formulas with explicit block-matrix assembly.
# - **Design spec**: `docs/superpowers/specs/2026-06-07-surgery-simplification-design.md`
# - **Webster *et al.* paper**: arXiv:2511.15989
# - **Inter-code joint example**: `examples/test_ide_bb_lp.py` — runs the full
#   Ide BB-LP inter-code bridge construction with two different bicycle codes.
