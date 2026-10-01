# Fixed-noise single-fault audit

This diagnostic compares HGP-simplex / unrotated-surface distances 4 and 8,
with four and eight adapter rounds, separately for XX and ZZ. It does not run
Monte Carlo sampling, two-fault searches, parameter sweeps, or exact-distance
calculations. All configurations use physical p=0.001, ideal Bell-source
fidelity, four ideal preparation rounds, ideal diagnostic closure, no idling,
hardware-only noise, inactive-detector reduction, and matching-basis local
order. Bell order remains X then Z for both PPM bases.

Six configurations are loaded from the archived circuits and undecomposed
DEMs without modifying them. Distance 4 / eight rounds is built with the same
generic builder and settings, for both bases. Source paths, canonical hashes,
metadata, decoder-source hashes, package versions, and the audit source hash
are saved with each condition. A changed identity is rejected on resume.

## Complete physical propagation catalogue

Every independent primitive noise location is expanded into its mutually
exclusive physical outcomes: three Pauli outcomes for DEPOLARIZE1, fifteen for
DEPOLARIZE2, one for a measurement flip or a specified Pauli error. Different
outcomes of the same channel are not multiple independent faults. Identity is
the no-fault outcome; nonidentity but detector/observable-silent outcomes are
still counted and saved.

Stim's error explanations associate each outcome with a detector pattern and
PPM observable flip. All explanations are retained, not one representative per
DEM term. Unsupported noise channels and nondeterministic detectors raise.
Twelve selected outcomes per circuit are independently cross-checked by
injecting the deterministic physical fault into an otherwise noiseless circuit
and sampling detectors and the observable with normal stabilizer randomization.

The catalogue records adapter initialization, surgery round, or detachment;
the original operation producing the noise; qubit IDs; Pauli outcome;
probability; and detector/observable signature. Circuit instruction and target
indices are **zero-based** in the saved flattened `circuit.stim`.

Six illustrative large-detector-footprint faults are also traced with Stim's
flip simulator. Stabilizer randomization is disabled only for this trace,
not for the independent validation. Reported block Pauli weight is a frame
representative at TICK boundaries, not minimized modulo stabilizers. A large
weight is not proof of a logical error, reduced code distance, or a harmful hook.

## Bounded decoder audit

The production SinterDecoder (BP+OSD, unchanged defaults and p=0.001 priors) is
compiled once per condition. Identical detector patterns are decoded only once
and the prediction is checked against all associated physical outcomes.
This includes the all-zero syndrome and silent faults.

The first pass tests at most 128 distinct syndromes per condition, with a
180-second wall budget per condition and eight workers total. An in-flight
decode can exceed the budget. The deterministic order alternates shuffled
logical-flipping and non-flipping signatures after the zero syndrome. This is
a diagnostic selection, **not** a probability-weighted sample or an unbiased LER
estimate. Time limits can produce unequal coverage between circuits.

Each decoded syndrome is appended and flushed to `decoder_progress.jsonl`.
Increasing the limit/time budget reuses saved predictions. BP convergence and
iteration counts are recorded; failure of BP to converge does not imply a
logical failure, since OSD may still correct the result.

From the repository root:

```bash
env PYTHONPATH=src:. MPLBACKEND=Agg MPLCONFIGDIR=/tmp/qldpc-mpl \
    NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    ../.venv/bin/python -m experiments.bell_pair_adapter_ler.single_fault_audit \
    --output-dir experiments/bell_pair_adapter_ler/outputs/single_fault_hgp_d4_d8_p0p001 \
    --decode-limit 128 --seconds-per-condition 180 --workers 8
```

Use `--decode-limit 0` for catalogue-only generation or re-summarization.
Do not change audited source files or decoder dependencies during an audit;
choose a new output directory if those identities change. No archived Monte
Carlo checkpoint is read as new sampling input or modified.

## Interpretation

`summary.csv` and `summary.json` distinguish complete propagation coverage from
partial decoder coverage. Full physical catalogues are in `catalogue.json.gz`;
decoder failures retain their physical origins in `decoder_failures.json`.

Two probability summaries have different meanings:

- `known_single_fault_failure_marginal_sum_not_ler`: the sum of individual
  physical probabilities of known misdecoded outcomes. It is not a full LER;
  it ignores the requirement that other locations be fault-free.
- `known_single_fault_sector_failure_probability`: the probability of those
  misdecoded outcomes **and no fault at any other location**, computed as
  `p_outcome * product(1-p_other_location)`. With incomplete decoder coverage it
  includes only already-identified single-fault failures. It excludes all
  multiple-fault events and is not a replacement for Monte Carlo LER.

The decoder settings are frozen at p=0.001. A coefficient measured with this
decoder is not automatically the asymptotic slope of a decoder retuned at other p.
No failures in the tested set do not prove fault distance four or eight, and
two same-syndrome faults with different observables identify ambiguity rather
than necessarily a software bug.
