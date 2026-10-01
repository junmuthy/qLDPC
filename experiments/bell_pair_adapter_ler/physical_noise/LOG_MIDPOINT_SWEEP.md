# Eight-round XX logarithmic-midpoint sweep

Four new independent conditions: m=3 HGP / unrotated surface d=4 and m=4 HGP /
unrotated surface d=8 at `sqrt(1e-4 * 3e-4)` and `sqrt(3e-4 * 1e-3)`.
Full binary floating-point probabilities are saved in metadata; displayed labels
may be rounded. Each probability bisects its neighboring existing log interval.

Each condition has 4 ideal preparation rounds, 8 adapter rounds, perfect Bell
pairs, adapter-hardware-only noise, inactive detector reduction, matching-basis
local/Bell X-then-Z, no idling, ideal closure, and unchanged production BP+OSD
OSD_0 settings. Decoder priors are computed from each probability's DEM.
Preflight checks scaled-noise signatures, all other metadata, decoder settings,
and noiseless behavior against the archived eight-round circuits. No optional
exact-distance search is performed. There are no ZZ or four-round tasks.

## Launch or resume

From the worktree root in a persistent host terminal:

```bash
env PYTHONPATH=src:. NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache \
  MPLCONFIGDIR=/tmp/qldpc-mpl MPLBACKEND=Agg \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -u -m experiments.bell_pair_adapter_ler.physical_noise.run_log_midpoint_sweep \
  --max-shots 100000 --pilot-shots 1000 --num-workers 8 \
  --max-batch-size 32 --max-batch-seconds 5
```

The same command resumes toward **100,000 cumulative shots per condition**,
not another 100,000 each restart. The first 1,000 shots per condition form a
timing pilot and are reused in the final counts. The script automatically
continues from the pilot to the authorized total, then exits. Only eight worker
processes run at once, plus the collector/helper. BLAS/OpenMP thread limits are
set to one; libraries may retain additional idle helper threads. Five seconds is a batch-duration
target, not a hard timeout on slow shots.

Use `--build-only` for preflight and `--analyze-only` for completed-target reports.
The default output directory is
`outputs/ppm_basis_first_hgp_d4_d8_xx_surg8_log_midpoints_100000`.
Checkpoint CSVs are appended and reused; an exclusive writer lock prevents
duplicate collectors. Completed flushed batches survive interruption; in-flight
work may be lost. Do not signal a historical PID without verifying its identity.

## References and reports

The six previous eight-round points are frozen at the initial preflight. This
includes the stopped d8/p=1e-4 checkpoint (533,500 shots / 25 failures at setup),
not the earlier 500k total. Their archive/seed provenance is validated; the
reference snapshot hash is checked on resume. No reference samples are imported
into the four fresh tasks, and old source datasets are not written.

Pilot and final reports each contain ten points, shared circle markers, orange
d4 and green d8, with the existing small cosmetic horizontal offset for d8.
Actual probabilities are retained in the CSV. There are no fits or bottom text.
The requested jackknife plot uses delete-one-shot 1-sigma standard errors.
Because a log plot cannot show a zero rate or a nonpositive lower endpoint,
zero/one-failure pilots instead get the separately labeled exact-95% plot;
jackknife output is produced when every point supports it. Exact intervals are
also retained for the final report. Repeated analysis creates a new timestamped
directory instead of overwriting plots.

`sinter_progress.csv` is the sampling checkpoint. `invocations.jsonl`,
`batch_samples.jsonl`, and `resource_samples.jsonl` record timing and resources.
Pilot `timing.json` estimates remaining wall time from worker seconds / shot;
this assumes representative shots and full utilization, and excludes future
startup/tail losses, machine load, and laptop sleep.
