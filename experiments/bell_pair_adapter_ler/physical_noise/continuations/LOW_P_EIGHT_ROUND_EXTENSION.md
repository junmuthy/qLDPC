# Targeted p=0.0001, eight-round XX extension

Only two conditions are extended: simplex HGP m=3 / unrotated surface d=4 and
simplex HGP m=4 / unrotated surface d=8. Both measure XX, with eight adapter
rounds, four noiseless preparation rounds, ideal terminal closure, physical
p=0.0001, Bell-source fidelity 1, and no idling noise. Adapter-hardware-only
noise, inactive-detector reduction, logical indices 0/0, and the matching-basis
local/Bell X-then-Z schedule are unchanged. The production BP+OSD decoder remains
OSD_0 with default settings; its priors come from each saved DEM. Optional exact
distance searches are not performed.

Each original checkpoint contains 100,000 shots and four failures. The target is
**200,000 cumulative shots per point**, i.e. **100,000 additional per point** and
200,000 additional shots total. The other seven points in the nine-point plot
are not sampled or changed. Eight worker processes are shared across these two
tasks in one Sinter pool; once one finishes, the pool can work on the remaining
task. No extension past the requested target is automatic.

## Start / resume

From the repository root in a persistent host terminal:

```bash
env PYTHONPATH=src:. NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache \
  MPLCONFIGDIR=/tmp/qldpc-mpl MPLBACKEND=Agg \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -u -m experiments.bell_pair_adapter_ler.physical_noise.continuations.run_low_p_extension \
  --max-shots 200000 --num-workers 8 --max-batch-size 32 --max-batch-seconds 5
```

New default output:
`outputs/ppm_basis_first_hgp_d4_d8_xx_p1e4_surg8_extension_200000/`.

`--build-only` validates the original circuits, DEMs, decoder settings, source
checkpoint identities, and 32 noiseless shots per selected task. It installs the
two original aggregates into the new checkpoint but samples no new shots.
`--analyze-only --max-shots 200000` regenerates reports only after both targets
are complete. Repeating the normal command resumes to 200,000, not another
200,000 or another 100,000 shots. Worker and batch settings are performance
options and do not change the experiment identity.

## Checkpoint and provenance safeguards

- Original directories and plots are read-only. Canonical circuits/DEMs and full
  metadata are copied to the new directory with identical Sinter strong IDs.
- The new `sinter_progress.csv` starts with exactly two aggregate seed rows.
  `seed_receipt.json` records their counts and hash; `identity.json` records the
  source checkpoint hashes, selected IDs, decoder-source hashes and dependencies.
- Seeding is atomic and happens once. Resumption verifies the seed prefix and
  receipt and never appends the seed again. Loss of an already initialized
  checkpoint is an error, not permission to silently reset to 100,000.
- The collector receives only the two selected tasks. No separate
  `existing_data_filepaths` import is used, so seeded shots cannot also be imported
  through that route. All comparisons replace the corresponding original rows;
  they do not add seeded aggregates to the original 100,000 a second time.
- An exclusive writer lock prevents concurrent collectors or analysis. A resumed
  target below saved counts is refused. Source-data or decoder changes require a
  new, explicitly reviewed provenance chain.
- Sampling batches are capped at 32 shots, with a five-second target reporting
  interval. Sinter may combine batches into checkpoint rows. Slow shots can
  exceed the target interval. A hard stop loses only unflushed/in-flight work.

Before stopping a detached run, verify its recorded PID/process group, boot ID,
start ticks, command and working directory, then signal only that process group.
After confirming it has stopped, the command above resumes flushed counts.
Laptop shutdown does not delete the checkpoint. Each detached launch should have
its own process identity record in the output directory.

## Analysis and timing

At completion, `analysis_200000/` contains the nine-point comparison. The two
p=0.0001/eight-round points have 200,000 shots. The other low-p points remain at
100,000. The p=0.001 points retain 20,000, 20,000 and 10,000 shots. Actual counts
are recorded in the CSV; high-p counts are also labeled beside the points. All
measured points are filled circles, distinguished by color; the x offsets only
separate overlapping estimates. Data-only plots omit the bottom explanatory
caption and reclaim its whitespace; fitted plots retain their captions.

Both the requested `_jackknife_1sigma` PDF/PNG/CSV and the separate exact-95%
version are written automatically. Jackknife bars use the delete-one-shot SE
`sqrt(LER*(1-LER)/(N-1))` on the cumulative retained shots, not on batches or
workers. They are standard errors, not exact confidence intervals. Existing seed
failures make the jackknife estimate nondegenerate for this extension.

Add `--log-log-fits` to `--analyze-only --max-shots 200000` to write separate
`_log_log_fits` PDF/PNG/CSV files without replacing the unfitted plots. Each
color gets a weighted least-squares fit of `log10(LER)` against `log10(p)`:
`LER = A * p**alpha`. The legend gives `alpha`; `fit_parameters*.csv` records
both the exponent and intercept/prefactor. Fits use actual physical probabilities,
not the cosmetic horizontal offsets; fitted lines receive the same offsets as
their data. Fit weights are `1 / sigma_log10**2`, with the one-sigma jackknife
SE propagated as `sigma_log10 = sigma_LER / (LER * ln(10))`. This first-order
conversion is approximate, especially for small failure counts; it is not an
exact binomial likelihood fit. These same jackknife weights are used in both
plot variants (even when exact-95% intervals are displayed). Three points
per color give descriptive slopes, not a measurement of circuit distance or
evidence for asymptotic scaling. Zero-rate points are refused, not replaced by
pseudocounts. No new simulation shots are generated in analyze-only mode.

For a focused comparison, the plotting helper `write_comparison_rows` also
accepts `only_rounds=8, log_log_fits=True, show_fit_caption=False`. This retains
only the d=4/eight-round and d=8/eight-round points and their weighted fit lines,
preserves the original orange/green colors, omits the bottom caption, and writes
separate `_log_log_fits_rounds8` reports. It can use a saved snapshot's rows without
re-reading or interrupting a live checkpoint.

Alternatively, `--analyze-only --max-shots 200000 --corrected-log-log-fits`
writes separate `_corrected_log_log_fits` reports for `LER = A * p**w * (1+b*p)`.
These fits have three free real parameters and three data points per color,
with the same propagated jackknife inverse-variance weights in log space.
For a fixed `b`, the other two parameters are fitted linearly; a curvature
profile finds the global optimum and both exact branches when they exist.
`A > 0` and `1+b*p > 0` throughout the measured interval. Curves are not
extrapolated, and `w` is not constrained to an integer or interpreted as distance.
The model has limited upward log-log curvature, so three free parameters do
not guarantee exact interpolation. If two exact solutions exist, the one with
smaller `abs(b)` is shown; the alternative is retained in the fit-parameter CSV.
The CSV also records weighted residuals and the largest `abs(b*p)` in the data
range. Values above one mean that this is not a small perturbative correction.
No parameter confidence bands or goodness-of-fit p-values are asserted from
the zero residual degrees of freedom. Original straight-line/unfitted reports
and all checkpoints are preserved.

`batch_samples.jsonl` records newly collected batches only. `invocations.jsonl`
records before/after totals and newly sampled shots/worker seconds per invocation.
The final timing report separates seeded and additional work. Worker seconds
include sampling/decoding and are not pure CPU time. Hard-killed invocations lack
final elapsed time, but their flushed worker statistics remain available.

The original per-shot worker times were approximately 0.03585 s (m=3) and
0.22515 s (m=4). Another 100,000 shots each projects to about 54 minutes at ideal
eight-worker scaling; allow roughly 1–1.5 hours for startup, tails, and decoder
variability. Runtime estimates must use additional work, not the imported seed.
