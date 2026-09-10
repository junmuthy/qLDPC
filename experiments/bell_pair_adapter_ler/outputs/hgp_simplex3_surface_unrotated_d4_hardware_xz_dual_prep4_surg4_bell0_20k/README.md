# Exact X/Z-dual ZZ schedule: hardware-only HGP–surface

The pre-existing implementation and results were checkpointed in commit `aa284059`
before this addition. This directory contains a new, independent ZZ experiment;
no archived full-surgery or hardware-only checkpoints were changed.

## Construction and validation

The new ZZ circuit is the exact X/Z-transformed, qubit-relabelled counterpart of
the archived hardware-only XX circuit. This transports the entire schedule and
its noise, including ideal preparation/closure, coupling order, Bell-check order,
record references, and detector/observable order. No extra physical Hadamard
operations are inserted. The data-column permutation, all opposite-basis check
supports, Bell partners, physical-block partition, hardware noise mask, and final
ZZ PPM readout are validated. Unsupported mappings or instructions fail closed.

The reduced detector-error model equals the reference XX model **exactly**,
including error probabilities and detector/error-column ordering: 340 detectors,
2,803 error terms, and 332 scheduling ticks. Consequently the model and BP+OSD
decoder predict the same LER distribution as reference XX. Independent samples
are not expected to give identical counts. This is a symmetry control, not an
assertion that the new schedule improves hardware-only LER or explains the
full-surgery ZZ excess.

Settings: HGP simplex `m=3`, `[[98,18,4]]`, and unrotated surface distance 4;
logical index zero in each block; four ideal preparation rounds and four adapter
rounds; `p=0.001`; Bell `F=1`; no idling noise; ideal terminal closure;
adapter-hardware-only noise; inactive detector reduction; exact undecomposed DEM
analysis; existing BP+OSD settings. No exact-distance search was performed.

## Results

| Hardware-only experiment | Failures / shots | LER | 95% Wilson interval |
|---|---:|---:|---:|
| Archived `XX`, reference schedule | 22 / 20,000 | 0.1100% | 0.0727%–0.1665% |
| Archived `ZZ`, legacy schedule | 16 / 20,000 | 0.0800% | 0.0493%–0.1299% |
| New `ZZ`, exact dual of `XX` | 19 / 20,000 | 0.0950% | 0.0608%–0.1483% |

The new estimate is compatible with both archived hardware estimates. Nominal
two-sided Fisher exact p-values are 0.755 against reference XX and 0.736 against
legacy ZZ; these are pointwise comparisons, not equivalence tests or corrections
for multiple/repeated comparisons. There is no statistically resolved hardware-only
schedule improvement in these samples.

The 1,000-shot pilot had zero failures and was resumed with the **same task ID**
to 20,000 total shots. It is part of the final estimate, not an independent sample
to pool again. Task ID:
`92f475cf23e91416fbe133fed29c59d11bd68ca673681de2c5483d557c28e41f`.

## Runtime and provenance

The invocation took 221.67 seconds (about 3 minutes 42 seconds), including
construction, pilot, production, and plotting, with eight workers. The pilot
stage took 20.97 seconds; the continuation/analysis stage took 200.18 seconds.
Cumulative worker time was 1,542.43 seconds, or 771.21 worker seconds per 10,000
shots. Archived XX and ZZ values were 823.39 and 745.33 worker seconds per 10,000
shots. These historical timings are not a controlled speed benchmark; regression
tests also ran concurrently with parts of this invocation.

Validation: 68 focused regression tests passed, including the 13 new dual-schedule
tests, plus Ruff and production-module mypy checks. Tests cover generic CSS
mapping, mixed data/ancilla gate and noise transformations, unsupported operations,
noiseless behavior, hardware-only noise placement, decoder equivalence on identical
syndromes, saved-artifact task-identity round trips, and archive preservation.

- `hgp_hardware_dual_schedule_comparison.pdf` / `.png`: three-estimate comparison.
- `summary.csv`: the new ZZ estimate only.
- `comparison.csv`: estimates, intervals, source paths, and normalized worker times.
- `schedule_audit.json`: complete mapping, validation results, hashes, and task ID.
- `manifest.json`, `run_config.json`: experiment identity and runtime configuration.
- `invocations.jsonl`: pilot and final cumulative counts and stage timings.
- `sinter_progress.csv`: resumable batch checkpoint (ignored by Git).
- `circuits/`, `dems/`: generated simulation artifacts (ignored by Git).

From the repository root, reproduce/resume with:

```bash
PYTHONPATH=src:. NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache MPLBACKEND=Agg \
  MPLCONFIGDIR=/tmp/qldpc-matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.run_hgp_dual_schedule \
  --max-shots 20000 --pilot-shots 1000 --num-workers 8
```

Use `--analyze-only` to regenerate the plot without sampling. The experiment runner
requires the existing local hardware-reference circuit/DEM/summary artifacts.
