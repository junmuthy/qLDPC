# HGP simplex–surface adapter-hardware comparison

Completed a new independent 10,000-shot run for each of `XX` and `ZZ` using the hardware-only
noise model and inactive-detector reduction. The archived full-surgery data were not modified.

Settings: `m=3` simplex-seeded HGP `[[98,18,4]]` on the left, unrotated distance-4 surface
`[[25,1,4]]` on the right, logical index zero in each block, four ideal preparation rounds,
four adapter rounds, physical noise `p=0.001`, ideal Bell source (`F=1`), no idling noise,
and ideal diagnostic closure. No exact-distance searches were performed.

The new model selects 43 adapter-hardware qubits for noise. All 279 circuit qubits remain;
block-only operations are ideal but can propagate faults introduced at adapter interactions.
Detectors decrease from 1,030 to 340 (`XX`) or 335 (`ZZ`). DEM error terms decrease from
20,045 to 2,803 (`XX`) and from 18,127 to 2,683 (`ZZ`).

## Results

LERs and intervals below are percentages. Intervals are two-sided 95% Wilson intervals.

| Measurement | Full-surgery failures | Full-surgery LER | Hardware failures | Hardware LER |
|---|---:|---:|---:|---:|
| `XX` | 9 / 10,014 | 0.0899% [0.0473%, 0.1707%] | 11 / 10,000 | 0.1100% [0.0614%, 0.1969%] |
| `ZZ` | 30 / 10,022 | 0.2993% [0.2098%, 0.4270%] | 13 / 10,000 | 0.1300% [0.0760%, 0.2223%] |

The `XX` estimates are compatible (approximately 0.45 combined standard errors apart).
The `ZZ` hardware estimate is lower by approximately 2.59 combined standard errors, with
two-sided Fisher exact p-value approximately 0.0136 before multiple-comparison adjustments.
This is suggestive evidence, not a high-precision result; only 13 new `ZZ` failures occurred.
Overlap between individual 95% intervals is not itself a test of equality of the two rates.

The full-surgery circuits match the archived circuits in canonical serialized form. In both
bases, Sinter's legacy automatic DEM analysis falls back to an undecomposed model, which
agrees with the explicit undecomposed model used in the new protocol. Unlike the earlier
Surface–Steane comparison, there is no need for another full-surgery reference to match DEM
analysis. The checks are recorded in `baseline_audit.json`.

## Runtime and provenance

The production invocation completed in 217.85 seconds (about 3 minutes 38 seconds), using
eight workers with BLAS/OpenMP thread counts set to one. This includes construction,
collection, and initial analysis. New Sinter worker-time totals were about 794.63 seconds
for `XX` and 759.09 seconds for `ZZ`; their sum is not elapsed wall time.

Normalized archived worker times were roughly 251 times (`XX`) and 182 times (`ZZ`) the new
recorded times per shot. These are historical workload ratios, not controlled single-machine
speed benchmarks: scheduling, worker configuration, and previous interruptions can differ.
The smaller fault model also changes the physical experiment; this is not an equivalent
acceleration of full-surgery noise simulation.

Files:

- `hgp_adapter_hardware_comparison.pdf` / `.png`: side-by-side LER comparison and intervals.
- `summary.csv`: new hardware-only estimates and counts.
- `comparison.csv`: archived/new estimates, confidence intervals, and recorded worker times.
- `manifest.json`, `run_config.json`, `baseline_audit.json`: settings, resources, and audit.
- `sinter_progress.csv`: resumable production checkpoints (ignored by Git).
- `invocations.jsonl`: elapsed wall time for completed production invocations.

The initial 32-shot diagnostic per basis was not pooled into the production statistics.
See `run_hgp_hardware_comparison.py` and the parent experiment README for rerun/resume commands.
