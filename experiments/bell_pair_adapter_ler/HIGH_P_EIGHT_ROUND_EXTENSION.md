# Extend both eight-round XX p=0.001 points

Add 20,000 shots per point, preserving the existing data:

- d=4 (m=3 HGP / unrotated surface): 20k/21-failure seed, 40k cumulative target.
- d=8 (m=4 HGP / unrotated surface): 10k/21-failure seed, 30k cumulative target.

Same eight adapter rounds, four noiseless preparation rounds, XX observable,
matching-basis local/Bell X-then-Z scheduling, perfect Bell source, no idling,
adapter-hardware-only noise, inactive-detector reduction, ideal terminal closure,
logical indices 0/0, and default BP+OSD / OSD_0 decoder. Archived circuits and
DEMs are used verbatim with their original Sinter strong IDs. No distance search
or additional lower-p/ZZ/four-round sampling is performed.

## Start/resume

From the repository root in a persistent host terminal:

```bash
env PYTHONPATH=src:. NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache \
  MPLCONFIGDIR=/tmp/qldpc-mpl MPLBACKEND=Agg \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -u -m experiments.bell_pair_adapter_ler.run_high_p_extension \
  --additional-shots 20000 --num-workers 8 --max-batch-size 32 --max-batch-seconds 5
```

Output: `outputs/ppm_basis_first_hgp_d4_d8_xx_p1e3_surg8_add20000/`.
Additional shots are relative to the fixed 20k/10k seeds. Repeating the command
resumes to the same 40k/30k cumulative targets, not another 20k on every restart.
One shared pool has eight worker processes, with independent per-task shot caps.

`--build-only` validates/installs the checkpoint without sampling noisy shots.
`--analyze-only` requires both targets complete and regenerates reports only.
The collector never extends beyond the requested totals automatically.

## Preservation and verification

All source directories are read-only. Seeds are imported once into the new
checkpoint with a receipt and source hashes. An exclusive lock prevents duplicate
collectors; missing initialized checkpoints and foreign tasks are refused.
The latest low-p data (d8: 500k at 1e-4, 300k at 3e-4; d4: 300k at 1e-4) are
validated through the preceding continuation and kept as fixed plot references.
Seeds replace existing aggregates in plots, never get added twice.

The older d8 high-p archive recorded dependency versions/default settings but not
decoder-source hashes. This limitation is explicitly retained in the new identity
and preflight records. Current decoder-source hashes and resolved settings are
recorded for all new shots. Source circuits/noise locations, metadata, strong IDs,
32 noiseless validation shots, and current decoder settings are checked against
the matched low-p controls. No claim of historical source-hash verification is made
for that older archive.

Completed batches are saved to `sinter_progress.csv`; timing/resource data go to
`invocations.jsonl`, `batch_samples.jsonl` and `resource_samples.jsonl`. Batch cap
32, target duration five seconds; a slow shot can take longer. A hard interruption
may lose in-flight/unflushed work but not completed checkpoints. Verify the current
launch record (boot ID, PID/start ticks, arguments, working directory and process
group) before stopping; confirm the collector exited before resuming.

## Plot and timing

Completion writes `analysis_add20000/`: eight-round-only data plots with common
circle markers, original colors, one-sigma delete-one-shot jackknife bars, no fits,
and no bottom caption. Exact-95% reports and full nine-point source JSON are also
saved separately. Only the two high-p rows change. Older plots remain intact.

Archived costs project 0.38 hours (d4) plus 3.05 hours (d8) at ideal eight-worker
scaling, about 3.5–4.5 hours total allowing for startup, decoding and scheduling
variability. Worker time includes sampling/decoding and is not pure CPU time.
