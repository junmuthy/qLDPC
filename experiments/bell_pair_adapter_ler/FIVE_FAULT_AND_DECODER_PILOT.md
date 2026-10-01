# Five-fault search and paired decoder timing pilot

These are separate investigations, using the archived matching-basis,
hardware-only PPM circuits. No physical schedule, noise model, decoder default,
or old result is modified. The initial paired pilot is **100 shots per basis**;
larger shot counts require a separate invocation.

## Five-fault search

`five_fault_search.py` reads the completed effective-distance audit and its pair
checkpoints. The source audit must have exhaustively excluded weights one
through four. Each new output freezes the source identities, parent progress,
and new search implementation hash.

The algorithm searches connected triples against all saved pairs:

1. Construct the graph whose vertices are distinct single-fault D/L signatures,
   with an edge when two signatures share a detector.
2. Enumerate connected triples as wedges: a central vertex and an unordered
   pair of its neighbors. Triangles can be visited three times, so the coverage
   counter is **wedges**, not distinct triples or physical fault histories.
3. Query the complete pair index for an equal detector pattern with opposite
   logical parity. Compare full signatures after fingerprint lookup, retain all
   physical origins, and require five distinct primitive noise locations.
4. Independently replay each accepted physical witness and its two/three-fault
   halves in the saved Stim circuit with other noise disabled.

Why connected triples suffice **if the search completes**: a minimum undetected
logical error cannot have disconnected detector-overlap components. Each such
component would independently have zero detector boundary, and at least one
would flip the observable, contradicting the exclusion of smaller weights.
Every connected five-vertex graph contains a connected three-vertex subgraph,
which has a spanning two-edge wedge. The complementary two vertices are in the
complete pair index. Repeated identical signatures are unnecessary in a minimum
error because they cancel. This argument depends on the validated smaller-weight
exclusion and applies to the supported physical Pauli-noise model.

The order prioritizes logical-flipping signatures and smaller neighborhoods;
it changes search order, not eventual coverage. Full completion without a witness
would establish distance >=6. A verified five-fault witness establishes exactly
five. **A time-limited negative result leaves the bound at >=5.**

```bash
env PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.five_fault_search \
  --distance-audit-dir experiments/bell_pair_adapter_ler/outputs/effective_distance_hgp_d4_d8_p0p001 \
  --output-dir experiments/bell_pair_adapter_ler/outputs/five_fault_hgp_d4_d8_p0p001 \
  --workers 2 --seconds-per-condition 600
```

The default order is distance-eight/eight-round XX and ZZ, then distance-four/
eight-round XX and ZZ controls. `--conditions d8_r8_X d8_r8_Z` restricts cases.
Repeat the command to resume; changing worker count/time budget is allowed, but
changing a frozen source or algorithm is rejected. The 600 s budget includes
setup/indexing and is checked between buckets and central vertices; an in-flight
unit can exceed it. Sorted pair indexes are cached under the **new** output
directory, hash checked on resume, and excluded from git. Old pair files remain
read-only. Progress is flushed/fsynced and atomically replaced after each unit.

## Paired-decoder timing check

`paired_decoder_check.py` uses direct Stim **circuit** sampling, not independently
sampled DEM columns. It stores bit-packed detector outcomes and true observable
flips in immutable 100-shot chunks. Fixed per-chunk seeds allow extension without
replacing earlier chunks. Saved arrays and hashes, not seeds alone, identify the
sample bank. This bank does not record the underlying physical fault histories.

The two configurations are:

- `production`: unchanged `SinterDecoder()` using the archived default settings.
- `osd_cs2`: same wrapper and priors, with only `osd_method="OSD_CS"` and
  `osd_order=2` changed.

Resolved settings are checked against the archived decoder and against each
other. Both use the same undecomposed DEM and its ordinary simplification.
The true observable is never supplied to the decoder; scoring occurs separately.
Every requested shot is scored, including shots where production succeeds, so
new failures introduced by the alternative are not hidden.

```bash
env PYTHONPATH=src:. MPLBACKEND=Agg MPLCONFIGDIR=/tmp/qldpc-mpl \
  NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.paired_decoder_check \
  --source-dir experiments/bell_pair_adapter_ler/outputs/single_fault_hgp_d4_d8_p0p001 \
  --output-dir experiments/bell_pair_adapter_ler/outputs/paired_decoder_hgp_d8_r8_p0p001_100 \
  --shots 100 --workers 4 --seconds-per-decoder 600
```

Do not run the heavy distance search concurrently when measuring decoder timing.
Each of the four condition/decoder tasks compiles one decoder and checkpoints
each prediction to an atomic per-shot JSON file. Existing predictions are reused
on resume. An in-flight status identifies an interrupted shot; it is not counted
as completed. The soft budget is checked between shots, and **timeouts are not
silently removed from the denominator**. Incomplete conditions receive no LER
or paired significance claim. Runtime/source/configuration changes are rejected.

Timing includes per-shot process CPU time and elapsed time, separately from
decoder setup; medians, p95, maxima, and process peak RSS are reported. CPU times
help separate decoder work from scheduling contention between the four workers.
Peak RSS is the Linux process high-water mark, not aggregate system RAM.

Statistics include exact binomial LER intervals and both rescued and introduced
failures. The LER difference is `(introduced - rescued) / shots`. Its conservative
95% paired interval subtracts exact 97.5% marginal bounds for the two discordant
categories (Bonferroni), rather than using a bootstrap that would misleadingly
give a zero-width interval when no failures were observed. An exact McNemar test
is also reported. A 100-shot pilot is a correctness/runtime check, not a precise
LER experiment or evidence of equal decoder performance.

Tests cover known-distance and random brute-force five-fault cases, forced
fingerprint collisions, exact physical replay, immutable sample extension,
decoder resume, incomplete-run handling, and both directions of paired errors.

## Initial deliverables

- Separate search coverage/bounds and sample-bank/decoder reports.
- Explicit failure witnesses if found, without claiming missing coverage.
- 100-shot-per-basis timing and pilot error counts.
- A runtime-based recommendation before any 1,000- or 10,000-shot extension.

## Initial pass reports

- [Five-fault search](outputs/five_fault_hgp_d4_d8_p0p001/RESULTS.md): complete
  distance-four controls exclude weight five; distance-eight searches reached
  their budgets with about 30% coverage and no witness.
- [Paired decoder pilot](outputs/paired_decoder_hgp_d8_r8_p0p001_100/RESULTS.md):
  XX completed 100 shots per decoder; ZZ checkpointed 93 production and 92
  alternative shots before the initial time limit. The
  [September 16 resume](outputs/paired_decoder_hgp_d8_r8_p0p001_100/COMPLETION_2026-09-16.md)
  completed the remaining ZZ predictions: both bases now have 100 shots per
  decoder and zero observed failures. The resume took 32.1 seconds and preserved
  all earlier samples and predictions. No increased sample count was launched.

## Authorized 5,000-shot extension

On September 16, a separate 5,000-shot-per-basis comparison was authorized and
launched with four workers. It reuses copies of the completed 100-shot pilot
checkpoints in a new directory; the original pilot remains untouched. See
[run, checkpoint, stop, and resume instructions](outputs/paired_decoder_hgp_d8_r8_p0p001_5000/RUN_5000.md).
Every completed prediction is saved; the target remains 5,000 shots for each
basis, decoded by both configurations. No distance search was restarted.
