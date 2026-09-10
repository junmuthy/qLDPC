# HGP simplex–surface: extension to 20,000 shots per basis

This extends the previous hardware-only run by 10,000 additional shots in each basis.
The original 10,000-shot directory is unchanged. Its Sinter checkpoint was copied to this
new directory and resumed; the cumulative estimates therefore include the original shots.
The checkpoint hash and original counts are recorded in `extension_config.json`, and both
task strong IDs were verified unchanged. The archived full-surgery reference was not extended.

Settings are unchanged: HGP simplex `m=3`, `[[98,18,4]]`, paired with an unrotated distance-4
surface block; logical index zero on each side; four ideal preparation rounds and four adapter
rounds; physical noise `p=0.001`, perfect Bell pairs, no idling, ideal diagnostic closure;
adapter-hardware-only noise, inactive detector reduction, and undecomposed DEM analysis.

## Counts and LER

| Measurement | Archived full surgery | First hardware 10,000 | Additional hardware 10,000 | Cumulative hardware 20,000 |
|---|---:|---:|---:|---:|
| `XX` | 9 / 10,014 (0.0899%) | 11 (0.1100%) | 11 (0.1100%) | 22 (0.1100%) |
| `ZZ` | 30 / 10,022 (0.2993%) | 13 (0.1300%) | 3 (0.0300%) | 16 (0.0800%) |

Cumulative hardware 95% Wilson intervals:

- `XX`: 0.0727%–0.1665%, versus full surgery 0.0473%–0.1707%.
- `ZZ`: 0.0493%–0.1299%, versus full surgery 0.2098%–0.4270%.

`XX` remains consistent with the archived full-surgery result (approximately 0.53 combined
estimated standard errors apart). `ZZ` moved lower rather than closer to the full-surgery
estimate; the separation is approximately 3.77 combined estimated standard errors.
Two-sided Fisher exact p-values against the full-surgery reference are approximately 0.705
(`XX`) and 0.0000121 (`ZZ`). These are nominal pointwise tests, not adjusted for repeated looks
or multiple comparisons. The quoted standard-error separation is not a conversion of the
Fisher p-value into Gaussian sigma; the two calculations differ for these unequal sample sizes.

The 10,000- and 20,000-shot hardware estimates are correlated and must not be treated as
independent replications or pooled into 30,000 shots. The two disjoint 10,000-shot batches are
listed separately above. There are still only 16 cumulative `ZZ` failures, so the hardware
LER estimate remains imprecise despite its separation from the archived reference.

More shots do not force equality: full-surgery and hardware-only noise define different
experiments. Both use matched DEM-analysis settings, as recorded in `baseline_audit.json`.

## Runtime and files

The extension completed in 218.29 seconds (about 3 minutes 38 seconds) with eight workers.
This is the time for the additional 10,000 shots per basis, plus construction and analysis,
not the combined elapsed time of both production invocations. Worker times in `summary.csv`
include the inherited checkpoint's processing time and are not wall-clock elapsed time.

- `hgp_adapter_hardware_comparison.pdf` / `.png`: archived reference versus cumulative 20,000-shot estimates.
- `summary.csv`: cumulative hardware statistics.
- `comparison.csv`: reference and cumulative estimates, intervals, and recorded worker times.
- `extension_config.json`: checkpoint provenance and original counts.
- `run_config.json`, `manifest.json`, `baseline_audit.json`: circuit settings and baseline validation.
- `sinter_progress.csv`: cumulative resumable checkpoint, ignored by Git.
- `invocations.jsonl`: elapsed time of the extension invocation.

Resume or regenerate the completed run with:

```bash
PYTHONPATH=src:. NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache \
  MPLCONFIGDIR=/tmp/qldpc-matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.run_hgp_hardware_comparison \
  --max-shots 20000 --num-workers 8 \
  --output-dir experiments/bell_pair_adapter_ler/outputs/hgp_simplex3_surface_unrotated_d4_hardware_prep4_surg4_bell0_20k
```
