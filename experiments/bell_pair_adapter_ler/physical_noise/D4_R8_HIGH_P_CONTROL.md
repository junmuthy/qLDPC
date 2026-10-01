# Missing distance-four, eight-round control

This samples exactly one new condition: simplex HGP `m=3` with unrotated surface
`d=4`, XX, four ideal preparation rounds, eight adapter rounds, physical
`p=0.001`, supplied Bell-pair fidelity `F=1`. Intrinsic adapter-hardware-only
noise, inactive-detector reduction, noiseless terminal closure, logical indices
0/0, and no idling noise match the completed physical-noise sweep. Both local and
Bell extraction are X then Z (matching-basis, not exact dual).

The production `qldpc_bp_osd` decoder is unchanged: default `SinterDecoder()`,
OSD_0/order zero, probabilities from this condition's own DEM. No exact-distance
search runs. Preflight compares all metadata except name/physical p and the
noise-rescaled operation/target sequence to both saved low-p eight-round circuits.
It also checks 32 noiseless shots and archives resolved decoder settings.

## Run or resume

From the repository root in a persistent terminal:

```bash
env PYTHONPATH=src:. NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache \
  MPLCONFIGDIR=/tmp/qldpc-mpl MPLBACKEND=Agg \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -u -m experiments.bell_pair_adapter_ler.physical_noise.run_d4_r8_control \
  --max-shots 20000 --pilot-shots 1000 --num-workers 8 \
  --max-batch-size 32 --max-batch-seconds 5
```

Default new output directory:
`outputs/ppm_basis_first_hgp_m3_d4_xx_prep4_surg8_p0p001/`.

The first stage ends at 1,000 cumulative shots, saves timing and a provisional
comparison, then automatically continues to the approved **20,000 total**.
Those 1,000 shots count toward the total. No further extension is automatic.
Repeating the identical command retains all flushed shots and skips any completed
pilot stage. Eight workers are the total across this one condition, not eight per
phase; native BLAS/OpenMP threads are restricted to one.

Use `--build-only` for validation without sampling; `--analyze-only --max-shots N`
only regenerates the comparison when the checkpoint contains exactly N shots.
The writer lock also excludes simultaneous analysis during collection.

## Checkpoints, stopping, and output

`sinter_progress.csv` is authoritative. Sampling batches are capped at 32 shots;
the target report/flush interval is five seconds, not a hard bound for slow shots.
Sinter can combine several completed sampling batches into one checkpoint row.
Interrupting can lose in-flight work and results not yet flushed by each worker;
all checkpointed shots remain usable.
For a foreground run, Ctrl-C stops collection. For a detached run, verify its
recorded PID, process group, boot ID, and start ticks before signaling only that
process group. A laptop shutdown does not delete the checkpoint. Never launch a
second writer against the same directory; the lock rejects it.

Resume checks metadata, circuit/DEM, strong ID, decoder-source hashes and package
versions, plus a frozen hash of all eight reference result rows. Changing a
reference dataset requires a new output directory, not silent mixing. Old output
directories and their plots are strictly read-only.

`analysis_1000/` and `analysis_20000/` hold independent nine-point CSV/PDF/PNG
comparisons. The six low-p points have 100,000 shots each; the old high-p d4/r4
point has 20,000 and d8/r8 has 10,000. Actual high-p counts are printed on the plot.
Intervals are exact two-sided 95% Clopper-Pearson; zero failures are represented
by upper-limit arrows. All measured points use filled circles, with color
distinguishing the distance/round-count series. Unequal counts are
not pooled and no LER is divided by the round count. Root `summary.csv` retains
the existing generic runner's Wilson intervals; use the comparison CSV for the
exact intervals shown on the new plot.

### Optional one-sigma delete-one jackknife plot

```bash
env PYTHONPATH=src:. NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache \
  MPLCONFIGDIR=/tmp/qldpc-mpl MPLBACKEND=Agg \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -u -m experiments.bell_pair_adapter_ler.physical_noise.run_d4_r8_control \
  --analyze-only --max-shots 20000 --error-bars jackknife1sigma
```

This writes `adapter_ler_vs_physical_error_rate_jackknife_1sigma.pdf` and `.png`,
plus `comparison_jackknife_1sigma.csv`, without replacing the exact-95% outputs.
It uses existing counts and never samples additional shots. The CSV retains the
named exact-95% columns for reference and adds explicitly named jackknife SE and
plus/minus-one-SE columns, with `plotted_error_bars` identifying the plotted method.

For N independent retained shots, k failures, and r=k/N, the leave-one-shot-out
jackknife standard error is exactly `sqrt(r*(1-r)/(N-1))`. Each shot (not each
checkpoint batch or worker) is a resampling unit. This follows from the standard
delete-one variance formula `(N-1)/N * sum((r_leave_one_out - r)**2)`; see
[Shalizi's jackknife notes, Eq. 5](https://stat.cmu.edu/~cshalizi/490/23/2023-11-29-jackknife.pdf).
No per-shot record is required for this Bernoulli sample mean. N excludes discards.

Bars are symmetric in LER, not in log(LER), so they look asymmetric on log axes.
They estimate one standard error; they are not exact 68% confidence intervals,
especially with very few observed failures. For all-success/all-failure samples
the jackknife SE is zero, which does not establish zero uncertainty about the true
rate. This optional log plot refuses those degenerate cases and one-failure bars
whose lower endpoint is zero; use the exact95 mode for those samples instead.

`timing_1000.json` estimates the remaining runtime from measured elapsed
throughput. `timing_20000.json`, `invocations.jsonl`, and `batch_samples.jsonl`
record final timing and worker time. Worker seconds include sampling/decoding,
not pure CPU time. A hard kill can leave an unfinished invocation event; the
checkpoint still governs resumption. The older d8/r8 archive has dependency
versions but no decoder-source hashes; its weaker provenance remains explicitly
recorded in `archived_references.json`.
