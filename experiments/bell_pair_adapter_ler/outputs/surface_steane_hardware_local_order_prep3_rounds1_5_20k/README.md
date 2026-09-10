# Surface–Steane hardware-only local-order comparison

Implementation branch: `surface-steane-local-order`, based on the preserved HGP
commit `b28a981d`. Runner and tests were committed as `736aa2fc` before sampling.
All legacy datasets are read-only references.

## Completed results

All ten new ZZ tasks finished at exactly 20,000 shots and zero discards: 200,000
new shots total. The uninterrupted invocation took 137.13 seconds (2 minutes
17 seconds), including construction, pilots, continuations, and plotting. Pilot
collection accounted for 10,000 of those shots; it is not an extra dataset.
Cumulative worker time was 869.81 seconds. All 46 reference-file hashes were
verified unchanged after sampling; all ten checkpoint identities and summary
counts were cross-checked against the saved audit.

The table gives LER in percent; parentheses show the **new** failure count out
of 20,000. Legacy counts and all intervals are retained in `comparison.csv`.

| Adapter rounds | Legacy ZZ, `F=1` | New ZZ, `F=1` | Legacy ZZ, `F=0.999` | New ZZ, `F=0.999` |
|---|---:|---:|---:|---:|
| 1 | 3.2414% | 3.695% (739) | 3.4284% | 3.600% (720) |
| 2 | 2.6108% | 2.475% (495) | 2.3282% | 2.715% (543) |
| 3 | 0.240% | 0.140% (28) | 0.190% | 0.165% (33) |
| 4 | 0.450% | 0.400% (80) | 0.290% | 0.345% (69) |
| 5 | 0.060% | 0.035% (7) | 0.080% | 0.065% (13) |

All ten new ZZ estimates are statistically consistent with their corresponding
legacy ZZ estimates at this precision. Every nominal difference interval includes
zero. The smallest unadjusted Fisher p-value is 0.0552; after Holm adjustment over
ten comparisons the smallest p-value is 0.552. No resolved degradation or improvement
is observed. In particular, local-order reversal preserves the old round-dependent
pattern, including the rise from three to four rounds; it does not make XX and ZZ
identical or establish monotonic improvement with more rounds.

This supports the simpler PPM-basis-first **local** rule for these hardware-only
settings without changing adapter geometry. It is not a proof of equivalence or a
guarantee for other noise models, codes, or round counts. The nominal-statistics
caveat concerning legacy early stopping is described below.

New-run worker seconds per 10,000 shots (historical workload, not a controlled
benchmark) were:

| Adapter rounds | `F=1` | `F=0.999` |
|---|---:|---:|
| 1 | 0.19 | 0.18 |
| 2 | 4.35 | 8.64 |
| 3 | 20.13 | 27.76 |
| 4 | 66.96 | 53.56 |
| 5 | 118.64 | 134.53 |

## Question and fixed settings

Does changing **only ZZ's local-family order** retain legacy-quality performance
across syndrome rounds and Bell fidelities, without changing logical supports or
adapter geometry? XX stays at legacy local X-then-Z; ZZ changes to local Z-then-X.
Thus each uses PPM-basis-first local extraction, but the circuits are not exact
X/Z duals. No claim of equal XX and ZZ error rates is built into this construction.

- Rotated distance-3 surface code on the left, Steane on the right; logical index
  zero, original supports. ZZ retains 23 merged data qubits, not XX's 24.
- Three noiseless independent preparation rounds; 1–5 noisy adapter rounds.
- Physical p=0.001; Bell state fidelity parameter F=1 or F=0.999; no idling noise.
- Hardware-only noise mask, including complete two-qubit errors on boundary gates.
  Block-only gates remain ideal, but their qubits and error propagation are retained.
- Original adapter initialization, Bell order and partners, within-check gate
  sequences, detachment, and ideal terminal readout.
- Same BP+OSD defaults and exact undecomposed DEM/inactive-detector reduction policy.
- Ten fresh ZZ tasks, each with a fixed 20,000-shot cap; eight workers. Each task's
  first 1,000 shots are a pilot included in the 20,000 total, not added again.
- No new XX sampling, no full-surgery simulation, and no exact-distance searches.

## Which results are reused?

The 20 archived Surface–Steane **hardware-only** configurations are stored in
`../adapter_hardware_prep3_round_sweep_1_5_comparison/bell0/` and `bell0p001/`:
two bases × five round counts × two Bell fidelities. They are neither the HGP
datasets nor the full-surgery references in that directory's other subfolders.

The old runs stopped at 10,000 shots or roughly 200 failures (batch overshoot was
possible). Their actual shot counts are retained. All default canonical serialized
Stim circuits and ordered detector-error models were checked against these archives.
Stim's text format rounds gate-argument floats, so bitwise equality of unrounded
in-memory probabilities is not required; serialized circuit text and ordered DEMs
must each match exactly, including DEM probabilities.

## Validation

The reusable HGP audit checks full gate/noise inventories, ideal preparation,
canonical detector and PPM-observable meanings, unchanged Bell partners and
within-check gates, and noiseless correctness. Additional Surface–Steane checks
require exactly the original Z-family layers followed by the original X-family
layers, unchanged complete tick depth, and identical quantum detachment/readout
operations. Ideal one-round templates check the gate order; full noisy circuits
are also checked separately at both actual Bell fidelities.

43 focused tests passed, covering the ten variants, all 20 references, decoder
smoke checks, noise placement, artifact/strong-ID round trips, archive protection,
invalid summaries, and zero-failure plotting. Ruff and mypy checks pass.
Reference circuits, DEMs, manifests, configurations, and summaries are SHA-256
hashed in `schedule_audit.json` and rechecked before each analysis.

## Files and statistical interpretation

- `summary.csv`: only the ten newly sampled ZZ estimates.
- `comparison.csv`: 20 archived and ten new estimates, actual counts, pointwise
  Wilson 95% intervals, gate depths, model sizes, and worker times.
- `differences.csv`: new-minus-legacy ZZ LER differences, Newcombe/Wilson intervals,
  nominal two-sided Fisher tests, and Holm-adjusted p-values across ten comparisons.
- `adapter_ler_vs_rounds_local_order.pdf` / `.png`: two panels for Bell F=1 and
  F=0.999, each showing reused XX, legacy ZZ, and new ZZ. Zero failures are shown
  as upper-limit markers rather than as log-axis zeros.
- `manifest.json`, `schedule_audit.json`, `run_config.json`: settings, hashes, and IDs.
- `invocations.jsonl`: cumulative pilot/final counts and stage wall times.
- `sinter_progress.csv`, `circuits/`, `dems/`: resumable/generated local artifacts,
  ignored by Git but retained on disk.

These are per-PPM failure probabilities, not per-round rates. Error bars and
difference intervals are pointwise, not simultaneous. The old error-count stopping
rule means the nominal tests/intervals do not claim stopping-rule-adjusted coverage.
Holm adjustment accounts for the ten planned comparisons but does not correct that
stopping-rule issue. Consistency is not proof of equivalence; no equivalence margin
was specified. The unchanged XX curve is contextual, not another independently
sampled outcome of this study. Counts are not pooled across curves or fidelities.
Historical worker times are workload records, not controlled speed benchmarks.

## Reproduce or resume

```bash
PYTHONPATH=src:. NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache MPLBACKEND=Agg \
  MPLCONFIGDIR=/tmp/qldpc-matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.run_surface_local_order \
  --max-shots 20000 --pilot-shots 1000 --num-workers 8
```

Use `--analyze-only` to regenerate figures without adding shots. The shared
checkpoint carries ten distinct, recursively canonicalized task identities.
