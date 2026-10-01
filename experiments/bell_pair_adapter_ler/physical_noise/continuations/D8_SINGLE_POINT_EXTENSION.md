# One-million-shot d=8 XX point at p=1e-4

Add **500,000 shots to only one condition**, using the fixed 500,000-shot/
23-failure seed. Target: **1,000,000 cumulative shots**. No other condition is
sampled or changed.

Same m=4 simplex HGP / unrotated surface d=8 pair, XX, physical p=1e-4,
eight adapter rounds, four ideal preparation rounds, perfect Bell source, no
idling, ideal output closure, adapter-hardware-only noise, inactive-detector
reduction, logical indices 0/0, matching-basis local/Bell X-then-Z scheduling,
and production BP+OSD/OSD_0 decoder. Archived circuit, DEM, priors, metadata and
Sinter strong ID are reused exactly. No optional distance calculation runs.

## Start/resume

From the repository root in a persistent host terminal:

```bash
env PYTHONPATH=src:. NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache \
  MPLCONFIGDIR=/tmp/qldpc-mpl MPLBACKEND=Agg \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -u -m experiments.bell_pair_adapter_ler.physical_noise.continuations.run_d8_single_point_extension \
  --additional-shots 500000 --num-workers 8 --max-batch-size 32 --max-batch-seconds 5
```

Output: `outputs/ppm_basis_first_hgp_d8_xx_p1e4_surg8_add500000/`.
Eight workers share this single task. Additional shots are relative to the fixed
imported 500k seed, not current counts. Repeating the command resumes to the same
one-million target; it does not add 500k each time. `--build-only` validates and
seeds without noisy sampling; `--analyze-only` requires the target complete.

All source directories are read-only. The full source/decoder/checkpoint chain
is validated, including the newly completed high-p extension. Seeding is atomic
and happens once, with a receipt and hashes. A missing initialized checkpoint,
changed identity, foreign task or lower target is refused. An exclusive lock
prevents duplicate writers. Only one selected Sinter task reaches the collector.

Batches are capped at 32 shots, targeting five seconds per batch; a slow shot may
take longer. Flushed data survive interruption; in-flight/unflushed shots can be
lost on a hard stop. Verify PID/process group, boot ID, start ticks, command and
working directory against the launch record before signaling, and confirm exit
before resuming. Logs: `sinter_progress.csv`, `invocations.jsonl`,
`batch_samples.jsonl`, `resource_samples.jsonl`.

## Reports

Completion writes `analysis_add500000/`: eight-round-only data plots with common
circle markers, original colors, one-sigma jackknife bars, no fits and no bottom
caption. Separate exact-95% plots, full source rows and timing reports are saved.
Only this one estimate is replaced, never added again to its seed. Other points
retain their latest totals: d4 high-p 40k/38 failures; d8 high-p 30k/54; d8
p=3e-4 300k/54; d4 p=1e-4 300k/21; d4 p=3e-4 100k/22. Older plots are preserved.

Historical worker time for this point is about 0.234 seconds/shot, projecting
roughly 4.1 hours for 500k new shots at ideal eight-worker scaling. Allow 4–5
hours while the laptop remains awake, subject to contention/decoding variability.
No automatic extension beyond one million shots follows.
