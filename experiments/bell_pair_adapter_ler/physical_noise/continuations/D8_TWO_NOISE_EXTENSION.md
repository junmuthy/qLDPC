# d=8 XX extension at p=1e-4 and 3e-4

Only the m=4 simplex HGP / unrotated surface d=8, eight-adapter-round XX points
are sampled. Add **200,000 shots per point** (400,000 new shots total):

- p=1e-4: seed 300,000 shots / 12 failures; target 500,000 shots.
- p=3e-4: seed 100,000 shots / 12 failures; target 300,000 shots.

Four ideal preparation rounds, ideal closure, Bell-source fidelity 1, no idling,
adapter-hardware-only noise, inactive-detector reduction, logical indices 0/0,
and matching-basis local/Bell X-then-Z scheduling are unchanged. The production
BP+OSD decoder retains OSD_0 and its original priors/iteration limits. Optional
exact-distance calculations are not run. No d=4 or p=0.001 point is sampled.

## Start and resume

From the repository root in a persistent host terminal:

```bash
env PYTHONPATH=src:. NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache \
  MPLCONFIGDIR=/tmp/qldpc-mpl MPLBACKEND=Agg \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -u -m experiments.bell_pair_adapter_ler.physical_noise.continuations.run_d8_noise_extension \
  --additional-shots 200000 --num-workers 8 --max-batch-size 32 --max-batch-seconds 5
```

Default output: `outputs/ppm_basis_first_hgp_d8_xx_p1e4_3e4_surg8_add200000/`.
**Repeating this command resumes to 500k and 300k, not another 200k per restart.**
`--additional-shots` is relative to the fixed imported seeds, not current counts.
One pool shares eight worker processes total. BLAS/OMP limits are set to one;
other libraries may still create auxiliary OS threads.

`--build-only` validates sources and installs the seed checkpoint without new
noisy samples. `--analyze-only` requires both targets complete and only regenerates
reports. There is no automatic extension beyond these targets.

## Checkpoint and provenance

Original checkpoints/plots are read-only. The p=1e-4 seed is the latest 300k
aggregate, not that aggregate PLUS its original 100k seed. The p=3e-4 seed comes
from the original six-condition 100k sweep. The previous continuation's receipt,
identity, reference rows, circuit/DEM artifacts and counts are checked read-only.
All source versions/hashes and new seed hashes/counts are recorded.

The new checkpoint is seeded once atomically. Missing initialized checkpoints
are refused, never silently recreated. An exclusive writer lock prevents two
collectors. Per-task collection limits preserve the original Sinter strong IDs
and stop each task at its own cumulative target. Regression below saved counts,
foreign tasks and changed decoder identities are rejected.

Batches are capped at 32 shots, with a five-second target batch duration. Slow
shots may exceed that target. Completed flushed batches survive stopping or
shutdown; abrupt interruption can lose unflushed/in-flight work. Before stopping,
verify the launch record's PID, process group, boot ID, start time and command,
then signal only that run's process group. Confirm exit before resuming.

## Reports and cost

`analysis_add200000/` contains the focused d=4/eight-round and d=8/eight-round
data-only comparison (PDF, PNG, CSV), with common circle markers, existing colors,
one-sigma delete-one-shot jackknife bars, no fits, and no bottom caption. A separate
exact-95% interval plot and full nine-row source JSON are retained. The d=4
p=1e-4 reference keeps its latest 300k/21-failure result. Only the selected two
d=8 rows are replaced; no seed is added twice.

`invocations.jsonl`, `batch_samples.jsonl`, and `resource_samples.jsonl` preserve
progress/timing. Timing reports distinguish imported shots from new work. Worker
seconds cover sampling/decoding, not pure decoder CPU time.

Archived costs of approximately 0.23 and 0.91 worker-seconds per shot project to
about eight hours at ideal eight-worker scaling for this extension. Budget roughly
8–10 hours for startup, decoder variability, contention and scheduling tails.
