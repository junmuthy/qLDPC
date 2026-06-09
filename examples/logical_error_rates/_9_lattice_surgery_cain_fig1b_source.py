"""Source for 9_lattice_surgery_cain_fig1b.ipynb.

Reproduces Cain et al. 2024 Fig 1b using qldpc.circuits.surgery.build_gadget
(Webster, Smith, Cohen 2025 §II.A 3-step gadget; equivalent to Cross 2024
§III at L=1). The measurement circuit follows Webster Eq. (1): with gadget
qubits initialized to |0⟩, the logical observable is the product of new
X-check (χ_i) outcomes across all rounds.

Convert with:
    jupytext --to notebook \
        examples/logical_error_rates/_9_lattice_surgery_cain_fig1b_source.py \
        -o examples/logical_error_rates/9_lattice_surgery_cain_fig1b.ipynb
"""

# %% [markdown]
# # Lattice Surgery for bb_18 — Reproducing Cain et al. 2024 Fig 1b
#
# Builds a merged surgery code for a bivariate-bicycle code via the explicit
# Webster–Smith–Cohen 3-step gadget recipe (arXiv:2511.15989 §II.A), runs a
# fault-tolerant measurement circuit on top with Webster's Eq. (1) observable
# (product of new X-check outcomes), decodes via BP-LSD, and compares the
# resulting LER curve against Cain et al. Fig 1b.

# %%
from __future__ import annotations

import numpy as np

from qldpc import codes
from qldpc.circuits.surgery import build_gadget
from qldpc.objects import Pauli

NUM_WORKERS = 8  # adjust to your machine; matches conventions in this directory.

# %% [markdown]
# ## 1. Construct the bb_18 BBCode

# %%
import sympy

x, y = sympy.symbols("x y")
# TODO(notebook author): replace with the bb_18 polynomials used in your
# reproduction target. See Cain App. D or your project notes.
poly_a = 1 + x + x**2  # PLACEHOLDER
poly_b = 1 + y + y**2  # PLACEHOLDER
orders = (3, 3)         # PLACEHOLDER

data_code = codes.BBCode(orders, poly_a, poly_b)
print(f"Data code: [[{data_code.num_qubits}, {data_code.dimension}]]")

# %% [markdown]
# ## 2. Pick a logical X representative

# %%
logical_x = np.asarray(data_code.get_logical_ops(Pauli.X)[0]).astype(np.int_)
print(f"|supp(X̄_M)| = {int(logical_x.sum())}")

# %% [markdown]
# ## 3. Build the gadget (merged surgery code)
#
# ``build_gadget`` implements Webster Steps 1–3 verbatim (L=1).

# %%
g = build_gadget(data_code, logical_x)
merged_hx = g.HX_merged
merged_hz = g.HZ_merged
print(f"Gadget kappa qubits: {len(g.kappa_qubits)}, V0 size: {len(g.V0)}")

# %% [markdown]
# ## 4. Sanity print against Cain Table III
#
# Cain Table III lists (ancilla qubits, X-checks, Z-checks) = (189, 104, 86)
# for bb_18. Exact match is not expected — Cain likely includes bridges and
# Cheeger-augmentation qubits beyond the bare 3-step gadget. The orders of
# magnitude should be comparable.

# %%
ancilla_qubits = len(g.kappa_qubits)
new_x_checks = len(g.V0)
new_z_checks = g.G.shape[0]
print(f"Ancilla qubits      : {ancilla_qubits}")
print(f"New X-checks (χ_i)  : {new_x_checks}")
print(f"New Z-checks (G)    : {new_z_checks}")
print(f"Cain Table III ref  : (189, 104, 86) — qualitative comparison only")

# %% [markdown]
# ## 5. Build the Webster minimal surgery measurement circuit
#
# The circuit differs from a standard X-memory experiment in only one place:
# the logical observable. Per Webster Eq. (1), the observable is the product
# across all R = d rounds of the χ_i outcomes — these are the last len(g.V0)
# rows of the merged H_X (rows mX_data+0 .. mX_data+|V_0|-1). Because the
# gadget qubits κ_j (columns at index ≥ g.code.num_qudits) are initialized
# to |0⟩, the χ_i are reliable from round 1 — no Cross §3.2 unreliable/D_0
# bookkeeping is needed.

# %%
from qldpc.circuits import memory

# TODO(notebook author): pick num_rounds matching Cain App. D. Typical
# choice is num_rounds = d (the code distance).
num_rounds = 12  # PLACEHOLDER — set per target code distance

# Identify which columns of the merged code are gadget qubits (Webster κ_j)
# vs data qubits — needed for the per-qubit initial state.
n_data = data_code.num_qudits
total_qubits = merged_hx.shape[1]
ancilla_qubit_mask = np.array(
    [False] * n_data + [True] * len(g.kappa_qubits), dtype=bool
)
data_qubit_mask = ~ancilla_qubit_mask

# Identify which H_X rows are the χ_i (new X-checks added by the gadget).
# In GadgetLayout, the merged HX has data-code X-checks first, then |V0| new rows.
n_data_x_checks = data_code.matrix_x.shape[0]
chi_row_indices = np.arange(n_data_x_checks, n_data_x_checks + len(g.V0))
print(f"#chi_i rows: {chi_row_indices.size}")

# TODO(notebook author): build the surgery circuit. The pattern is:
#
#     circuit = memory.build_x_memory_circuit(
#         code=merged,
#         num_rounds=num_rounds,
#         initial_state="logical_plus",   # data qubits = logical |+⟩ of bb_18
#         ancilla_initial_state={i: "0" for i in np.flatnonzero(ancilla_qubit_mask)},
#         noise_model=...,                # standard depolarizing per Cain App. D
#     )
#
# The exact builder API depends on the `qldpc.circuits.memory` interface in
# this repo — adapt to whichever helper notebooks 2/3 already use. The
# critical detail is that gadget qubits start in |0⟩ (Z=+1 eigenstate).
#
# Then OVERRIDE the logical observable on the resulting stim circuit so it
# becomes the product of all chi_i measurement outcomes across all rounds,
# instead of the default data-code logical X̄.
#
# Pseudo-code:
#
#     circuit = ...build base circuit as above...
#     # Remove the default OBSERVABLE_INCLUDE that targets the data logical:
#     circuit = strip_default_observable(circuit)
#     # Add an OBSERVABLE_INCLUDE that XORs every chi_i measurement record
#     # across all num_rounds rounds:
#     chi_measurement_records = [
#         stim.target_rec(-offset_for(round_idx, chi_idx))
#         for round_idx in range(num_rounds)
#         for chi_idx in chi_row_indices
#     ]
#     circuit.append("OBSERVABLE_INCLUDE", chi_measurement_records, 0)

# %% [markdown]
# ## 6. Configure the BP-LSD decoder per Cain App. D
#
# Copy the exact decoder configuration from your prior
# `reproduce_cain_bb18_*.py` scripts. The key Cain App. D parameters are BP
# iteration count and LSD post-processing settings.

# %%
from qldpc import decoders

decoder_kwargs = {
    "max_iter": 30,        # PLACEHOLDER — set to Cain App. D value
    "osd_method": "OSD_0", # PLACEHOLDER — set to Cain App. D value
}

# %% [markdown]
# ## 7. Sweep and produce the LER curve

# %%
import sinter

# TODO(notebook author): fill in the sinter.collect sweep over physical
# error rates, using NUM_WORKERS workers. The circuit factory must wire the
# Webster observable defined in section 5; the decoder runs on the DEM
# derived from that circuit so that "logical error" = wrong χ_i product.

p_values = [1e-3, 2e-3, 3e-3, 5e-3, 7e-3]
results = []  # placeholder for sinter.collect output

# %% [markdown]
# ## 8. Plot alongside Cain Fig 1b

# %%
import matplotlib.pyplot as plt

# TODO(notebook author): extract LER from results, overlay Cain Fig 1b
# data if available, save figure to examples/logical_error_rates/figures/.
