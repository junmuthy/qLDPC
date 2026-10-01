# XX-only physical-noise comparison

This experiment asks whether lowering physical noise reveals a distance advantage
for the HGP-simplex/surface adapter. It does not estimate an asymptotic threshold.

## Conditions

At each of `p=0.0003` and `p=0.0001`, collect XX only for:

| Simplex seed | Unrotated surface distance | Adapter rounds | Purpose                          |
|--------------|----------------------------|----------------|----------------------------------|
| `m=3`        | `d=4`                      | `4`            | Smaller-code baseline            |
| `m=3`        | `d=4`                      | `8`            | Equal-rounds control             |
| `m=4`        | `d=8`                      | `8`            | Larger code with scaled duration |

All six use logical indices 0/0, four noiseless preparation rounds, ideal terminal
closure, intrinsic adapter-hardware noise, inactive-detector reduction, no idling
noise, and Bell-source fidelity 1. Ordinary adapter gates/reset/readout still have
physical noise. Local and Bell extraction are both X then Z for this XX-only run.
This is the matching-basis policy, not the exact X/Z dual construction.

Production `qldpc_bp_osd` settings are unchanged: `SinterDecoder()` with no decoder
overrides, OSD_0/order zero. Decoder probabilities come from each condition's own
DEM. The default BP iteration limit scales with the DEM size; it is the same
policy, not an artificially fixed iteration count for differently sized codes.
Resolved settings are recorded in `preflight.json`. No exact-distance search runs.

## Start/resume the pilot

From the repository root, in a persistent host terminal:

```bash
env PYTHONPATH=src:. NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache \
  MPLCONFIGDIR=/tmp/qldpc-mpl MPLBACKEND=Agg \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -u -m experiments.bell_pair_adapter_ler.physical_noise.run_physical_noise_comparison \
  --output-dir experiments/bell_pair_adapter_ler/outputs/ppm_basis_first_hgp_d4_d8_xx_physical_noise \
  --physical-error-rates 0.0003 0.0001 \
  --max-shots 1000 --num-workers 4 --max-batch-size 32 --max-batch-seconds 5
```

The target is **1,000 cumulative shots per condition (6,000 total)**, not 1,000
additional shots on each restart. There is no automatic extension. After reviewing
timing and uncertainties, `--max-shots 10000` would extend these same checkpoints
to 60,000 total shots while retaining the pilot data. That extension requires a
separate decision. Four workers are shared across conditions; the `p=0.0003`
group finishes before collection starts for `p=0.0001`.

`--build-only` validates the whole matrix and archives circuits/DEMs without
sampling. `--analyze-only --max-shots N` regenerates the report only when every
condition has exactly N saved shots. It cannot reconstruct an earlier prefix
after extending the counts; earlier completed `analysis_N` directories are kept.

## Checkpoints and safety

- `sinter_progress.csv` is the authoritative live checkpoint, flushed after each
  completed batch. At most 32 shots are in a batch, with a target five-second batch
  duration. A slow shot cannot be forcibly interrupted at that target duration.
- A stop/crash may lose an in-flight batch per worker. Flushed batches are reused;
  the interrupted random shots themselves are not a saved immutable sample bank.
- A writer lock rejects a duplicate collector. Resuming verifies complete task
  metadata, circuits, DEMs, Sinter strong IDs, dependency versions, and decoder
  source hashes. A different noise grid requires a new output directory.
- New sampling never imports historical shots. Old experiment directories are
  read-only references and are not modified.
- For a detached run, record PID, process group, boot ID, and process start ticks.
  Verify that identity before stopping **only that run's entire process group**;
  do not terminate unrelated simulations. Laptop shutdown stops computation but
  does not remove completed checkpoints.

The preflight also verifies that dividing all physical-noise probabilities by p
leaves exactly the same instruction/target sequence for both p values, and checks
32 noiseless detector/observable samples per circuit. Thus noisy locations and
the adapter-hardware mask, not just the noise-free schedule, are held fixed.

## Reports

At completion, `analysis_1000/` contains a CSV and PDF/PNG plot, plus `timing.json`.
Later `analysis_10000/` does not replace the pilot report. Counts are per complete
PPM, never divided by the number of rounds. Plot intervals are exact two-sided
95% Clopper-Pearson intervals; zero-failure points are upper-bound arrows, not
measured zero LERs. Slight horizontal offsets separate overlapping bounds.
The generic root `summary.csv` retains the existing runner's Wilson intervals;
use `analysis_N/comparison.csv` for the exact intervals shown in these new plots.

Compatible archived `p=0.001` XX controls (m=3/r=4 and m=4/r=8) are shown with open
markers. They are independently validated against current metadata, scaled noise
locations, saved circuit/DEM/checkpoint identity and software versions. The older
m=4 archive lacks decoder-source hashes, so its provenance check is weaker; this
is recorded in `archived_references.json`. There is no archived m=3/r=8 point and
none is fabricated or silently simulated. `--without-archived-references` omits
both controls if running in a checkout without those archives.

`batch_samples.jsonl` records sampling/decoding worker seconds, and
`invocations.jsonl` records elapsed collection time for each invocation. Worker
seconds are not pure decoder CPU time, and summed worker time is not wall time.
The pilot-to-10k timing projection is approximate: hard syndromes, pool startup,
and scheduling tails affect throughput. A hard-killed invocation has no final
elapsed-time record, though its checkpointed worker times remain available.

At these low noise rates, 1,000 shots primarily test timing and plumbing. Even
zero failures in 10,000 shots only gives an exact two-sided 95% upper bound of
about 0.000369. Do not claim code-size equivalence or a threshold from overlapping
bounds or too few failures. Any conclusions here apply to XX, not automatically ZZ.
