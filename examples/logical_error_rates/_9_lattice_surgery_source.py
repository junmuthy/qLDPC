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
# | `build_bridge(g1, g2)` | `Bridge` | Path-graph bridge joining two gadgets |
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

import numpy as np

from qldpc.codes.surgery import (
    GadgetLayout,
    Bridge,
    build_gadget,
    build_bridge,
    build_single_ppm_circuit,
    build_joint_ppm_circuit,
    boost_gadget,
    load_webster_seed_set,
)
from qldpc.codes.surgery.gadget import _build_generalised_bicycle_code

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
code = _build_generalised_bicycle_code(data["l"], data["A"], data["B"])
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
    c = _build_generalised_bicycle_code(d["l"], d["A"], d["B"])
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
# ## 6. `build_single_ppm_circuit` — Stim measurement circuit
#
# `build_single_ppm_circuit(gadget, *, rounds, noise_model)` builds a Stim
# circuit that performs `rounds` of syndrome extraction on the merged CSS
# code (data + κ ancillas).  An optional `noise_model` injects Pauli errors.
#
# Under noiseless conditions all detectors should remain silent.

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
samples_noisy = noisy.compile_detector_sampler().sample(shots=256)
fire_frac = samples_noisy.any(axis=1).mean()
print(f"Noisy (p=0.01) detector firing fraction: {fire_frac:.2%}")

# %% [markdown]
# ## 7. `build_bridge` — joining two gadgets
#
# `build_bridge(g1, g2)` constructs the path-graph bridge that links the
# χ endpoints of two gadgets, enabling a **joint** PPM of the product
# X̄_1 ⊗ X̄_2 (Cross §3.6 / math.md §2).
#
# **How it works (math.md §2):**
# - The bridge is a path graph on `w` qubits, where `w = min(|V₀(g1)|, |V₀(g2)|)`.
# - `w - 1` path-graph X-stabilizers U_B form a telescoping sum (e₀ + e_{w-1}).
# - The χ endpoint rows of each gadget get an X on their respective bridge endpoint.
# - XOR-ing all χ rows from g1, all χ rows from g2, and all U_B rows yields
#   exactly (x̄₁ + x̄₂) on data qubits and 0 on all ancillas/bridge qubits.
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

print(f"Bridge width w = {bridge.width}")
print(f"Intercode flag : {bridge.intercode}  (False = intra-code joint)")
print(f"U_B shape      : {bridge.U_B.shape}  ((w-1) × w path-graph X-stabilizers)")
print(f"Path telescoping: XOR of U_B rows = e_0 + e_(w-1)? ", end="")
col_sum = bridge.U_B.sum(axis=0) % 2
expected_ends = np.zeros(bridge.width, dtype=np.uint8)
expected_ends[0] = 1
expected_ends[-1] = 1
print(np.array_equal(col_sum, expected_ends))

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
    c = _build_generalised_bicycle_code(d["l"], d["A"], d["B"])
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
# the full joint CSS code (data + g1-κ + g2-κ + bridge qubits) and returns
# both the Stim measurement circuit and the merged `CSSCode` object.
#
# **Protocol formula (math.md §2.7 / Cross §3.6):**
# The canonical measurement vector α* has 1 on every χ row from both
# gadgets and every U_B bridge-path row, and 0 on data X-check rows:
#
#   Σ(χ₁ rows) = x̄₁ on data | 0 on g1-κ | X on bridge[0]
#   Σ(χ₂ rows) = x̄₂ on data | 0 on g2-κ | X on bridge[w−1]
#   Σ(U_B rows)= 0 on data   | 0 on κ    | e₀ + e_{w−1}  (path telescoping)
#   ─────────────────────────────────────────────────────
#   Total      = (x̄₁⊕x̄₂) on data | 0 on ancillas | 0 on bridge
#
# This means measuring this circuit is equivalent to a projective measurement
# of the joint Pauli product X̄_1 ⊗ X̄_2, reducing the logical dimension by 1
# (Cross §3.6 / math.md §2.8).

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
# ## 10. Summary
#
# The 5 public APIs of `qldpc.codes.surgery`, each in one sentence:
#
# - **`build_gadget(code, x)`**: runs the Webster 3-step gadget construction
#   (restriction → gauge-fix → assembly) to produce a merged CSS code with
#   κ ancilla qubits that absorbs the logical operator x̄ into a stabilizer.
#
# - **`build_bridge(g1, g2)`**: constructs a path-graph bridge of width
#   `w = min(|V₀(g1)|, |V₀(g2)|)` that links two gadgets for a joint PPM;
#   auto-dispatches intra-code vs inter-code based on `g1.code is g2.code`.
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
