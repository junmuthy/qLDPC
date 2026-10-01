# Physical PPM fault-distance audit

This diagnostic reads the fixed-p single-fault audit without changing its
circuits, detector models, decoder checkpoints, or Monte Carlo results. It
searches for the minimum number of **allowed physical faulty locations** that
produce no detector events and flip PPM observable zero. It does not compute the
static distance of the input codes or the merged stabilizer code, and does not
infer the minimum number of faults that BP+OSD misdecodes.

## Scope and assumptions

The archived eight cases are distances 4/8, rounds 4/8, and X/Z PPMs, with
physical p=0.001, Bell-source fidelity 1, four ideal preparation rounds, ideal
closure, hardware-only noise, no idling, and matching-basis local extraction.
Circuit fault distance depends on the allowed nonzero noise outcomes, not their
particular positive probabilities. Probabilities remain relevant to the
separate failure-probability accounting.

All full detector signatures are retained, including hyperedges. No graphlike
decomposition or heuristic symptom-weight cutoff is used. Primitive channel
outcomes at the same instruction/target location are mutually exclusive. A
two-qubit Pauli from one DEPOLARIZE2 location counts as one fault.

## Search and coverage guarantees

`effective_distance_audit.py` groups equivalent single-fault detector/observable
signatures while retaining **every physical origin** for compatibility checking.
The zero signature is omitted from a minimum-weight search: adding it cannot
create an undetected PPM flip. Repeated equal signatures can also be omitted
from a minimum-weight witness because they cancel in pairs; removing both
locations leaves a lower-weight, still physically valid history.

1. Test every nonzero signature and every complementary-logical same-syndrome
   signature for weights one and two.
2. Generate every unordered pair of distinct signatures. Match pairs to single
   signatures for weight three.
3. Partition pair records by a linear detector fingerprint. Match pairs with
   equal detector patterns but opposite PPM flips for weight four.
4. For each candidate, compare the **complete** detector/observable bit strings
   and find a realization on distinct physical locations. Exhaustive backtracking
   retains alternative origins when one representative conflicts.
5. Independently inject every accepted witness into the idealized saved Stim
   circuit and validate its full detector and observable output. Also validate
   the two ambiguous histories obtained by splitting the witness.

The fingerprint is a lookup accelerator only. Collisions are retained and
checked with full signatures, so hash collisions cannot create a false distance
certificate or exclude a genuine match. Tests force every detector fingerprint
to collide and compare against brute force over physical faults.

A verified weight-w witness proves an upper bound. Excluding all lower weights
makes that bound exact. Exhausting the search through weight four without a
witness proves only distance >=5, not distance eight. Time-budget exhaustion
proves nothing beyond the completely searched weights. The search stops after
finding a minimum: it does **not** enumerate all minimum-weight mechanisms.

## Running and resuming

From the repository root:

```bash
env PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.fault_analysis.effective_distance_audit \
  --source-dir experiments/bell_pair_adapter_ler/outputs/single_fault_hgp_d4_d8_p0p001 \
  --output-dir experiments/bell_pair_adapter_ler/outputs/effective_distance_hgp_d4_d8_p0p001 \
  --workers 2 --seconds-per-condition 600
```

Repeat the same command to resume. `--conditions d8_r8_X d8_r8_Z` limits the
conditions. Worker count and time budget may change; source identities and
algorithm identity may not. Changing the algorithm requires a new output root.

Each condition holds an advisory writer lock. Checkpoints use flush/fsync and
atomic replacement. Pair chunks commit before progress advances. Interrupted
uncommitted chunks can be regenerated. On resume, committed pair-file hashes
and the source circuit, DEM, catalogue, and identity hashes are verified.

The time budget is per condition per invocation and is checked between pair
chunks and buckets. A current chunk/bucket and validation may exceed the budget.
It is not a hard process-kill timeout. Complete conditions are not searched again.

The largest current condition stores approximately 1.34 GB of pair checkpoints;
all eight together store 3.94 GB. Chunks and bucket reads bound working memory.
These are reproducible search intermediates, not Monte Carlo shots, and remain
available for resume/integrity checks. They are not suitable as git result assets.

## Geometry and decoder follow-ups

`adapter_structure_audit.py` reconstructs the matching four-round circuits and
requires exact equality with the archives before inspecting their geometry. It
reports logical supports, bridge width, and exact boundary expansion for the
raw gadgets, augmented gadgets, and complete bridged incidence matrix. It
does not call a distance solver or add qubits. Exact subset searches are capped
at 20 vertices; larger cases remain uncomputed rather than using an uncertified
spectral approximation. Use a fresh structure output directory:

```bash
env PYTHONPATH=src:. MPLBACKEND=Agg MPLCONFIGDIR=/tmp/qldpc-mpl \
  NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.fault_analysis.adapter_structure_audit \
  --source-dir experiments/bell_pair_adapter_ler/outputs/single_fault_hgp_d4_d8_p0p001 \
  --output-dir experiments/bell_pair_adapter_ler/outputs/effective_distance_hgp_d4_d8_p0p001/structure
```

Boundary expansion >=1 is a sufficient structural condition only when the
construction's other hypotheses apply. Expansion <1 is not proof of distance
loss; code-specific alternatives may preserve distance. Neither outcome
certifies the noisy extraction circuit's PPM fault distance.

`explain_distance_witnesses.py <condition-directories...>` enumerates compatible
physical realizations of the two saved witness halves (default limit 200,000
per half), verifies alternative representatives, and tests the unchanged
production decoder on their common syndrome. Its probability sums include no
fault at any other physical site. The smaller opposite-parity mass is a lower
bound for any detector-only decoder; the mass opposite BP+OSD's prediction is
that decoder's contribution from these families only. Neither is the full LER,
an optimal full-noise decoder computation, or a count of all failure mechanisms.

## Outputs

- `summary.csv` / `summary.json`: exact distances or explicit lower/upper bounds,
  coverage, timing, and checkpoint size.
- Per-condition `identity.json`, `progress.json`, `pairs_*.npy`: provenance and
  resumable coverage. Indices in physical fault records are zero-based.
- `witness.json`: combined undetected error, its two ambiguous histories,
  physical locations, phases, rounds, and verified probabilities.
- `witness_analysis.json`: additional physical realizations and production
  decoder behavior, with probability-accounting caveats.
- `structure/*.json`: exact connectivity diagnostics and archival equality checks.
- `RESULTS.md`: interpretation of the completed pilot and its limitations.
