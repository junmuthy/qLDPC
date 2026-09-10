# Surface–Steane hardware-only round comparison

The new runs retain the original Surface-3/Steane orientation and logical indices: rotated
Surface-3 on the left, Steane on the right, logical index zero, with both `XX` and `ZZ`.
They use three ideal preparation rounds, 1–5 adapter syndrome rounds, physical depolarizing
parameter `p=0.001`, no idling noise, and ideal diagnostic closure. Bell-pair infidelity is
either `0` or `0.001` (fidelity `1` or `0.999`). Each production point stops at 10,000 shots
or approximately 200 failures; batch collection can overshoot the failure threshold.

## Three LER curves

- Archived full-surgery results: reused unchanged from the two older round-sweep directories.
  These used Sinter's automatic detector-error-model (DEM) analysis.
- Matched full-surgery reference: new samples with the same physical circuit and an explicitly
  undecomposed DEM, stored in `full_surgery_exact/`.
- Adapter-hardware noise: new samples with an explicitly undecomposed DEM and inactive detector
  rows removed, stored in `bell0/` and `bell0p001/`.

The extra full-surgery reference matters: Sinter's automatic DEM analysis tries a graphlike
decomposition. Decomposition suggestions can change the error-column representation used by
the current BP+OSD wrapper, even without changing the sampled quantum circuit. The matched
reference and hardware curves use the same DEM-analysis policy. Comparing only the archived
and hardware curves would mix noise-policy and decoder-representation effects.

Hardware-only noise is a different physical experiment, not an equivalent shortcut for the
original full-surgery noise model. The code blocks and their ideal operations are still present;
cross-boundary gates retain their full error channels. This LER is conditional on ideal block-only
operations, not the practical failure rate of blocks subject to their own gate and memory noise.

## Files and timings

- `adapter_ler_vs_rounds_comparison.pdf` / `.png`: LER curves and 95% Wilson intervals.
- `comparison.csv`: counts, LER intervals, detector counts, and source paths for all curves.
- `adapter_decode_time_vs_rounds.pdf` / `.png`: fresh serial decoder timing, median with
  min–max range across three independent 256-shot batches per condition.
- `timing_samples.csv`: detailed setup, sampling, conversion, CPU and wall-clock decoder times.
- `timing_summary.csv`: per-shot timing summaries and speed ratios.
- `benchmark/`: per-condition resumable timing checkpoints and dependency/seed configuration.

Timing compares full-surgery noise, hardware noise with all detectors, and hardware noise with
reduced detectors, all with explicitly undecomposed DEMs. The hardware variants decode the
same measurement records; their results must not be pooled as independent samples. Timing
shots are not included in the production LER estimates. Full-surgery timing samples are separate.
Execution order rotates between batches. Build and decoder-compilation times are measured once
per condition and repeated as setup metadata in batch rows. Reduced sampling time is inherited
from the shared hardware records; it does not imply faster quantum-circuit sampling.

The archived and production Sinter worker times in `comparison.csv` are historical workload
measurements, not comparable elapsed times or a controlled benchmark. Use the fresh serial
timing files to assess speed. Ratios above one indicate a speedup. The reduction-only ratio
compares the same noise model, whereas the full-surgery-to-hardware ratio also changes the
fault model being decoded. Three-batch min–max timing ranges are not confidence intervals.

See the parent experiment README for production, resume, analysis, and benchmark commands.

## LER observations

All 20 hardware points and all 20 matched-reference points reached their production stopping
criterion. The 20 archived points are retained separately; none were overwritten or resampled
into the historical datasets. The plot contains 60 estimates in total.

For four adapter rounds, all fresh estimates below used 10,000 shots:

| Bell fidelity | Logical measurement | Matched full-surgery LER | Hardware-only LER |
|---|---|---:|---:|
| 1 | `XX` | 0.25% | 0.16% |
| 1 | `ZZ` | 0.24% | 0.45% |
| 0.999 | `XX` | 0.18% | 0.07% |
| 0.999 | `ZZ` | 0.18% | 0.29% |

These are point estimates, not claims of statistically significant improvement; the plot and
CSV give their confidence intervals. Hardware-only estimates are not uniformly smaller.
The four-round `ZZ` behavior and nonmonotonic `XX` round dependence merit decoder/noise-model
investigation before drawing broad scaling conclusions.

The importance of a matched decoder representation is particularly visible at three `XX`
rounds with ideal Bell pairs: archived full-surgery LER is 0.42%, new matched full-surgery LER
is 1.32%, and hardware-only LER is 0.79%. The canonical serialized full-surgery circuits match
the archive; the comparison is sensitive to more than the physical qubit count alone.

## Timing observations

All 20 conditions completed three 256-shot benchmark batches for each representation. The
hardware representations produced identical logical predictions on all 15,360 paired shots.

Across the 20 conditions, the median full-surgery-to-reduced-hardware decoder speed ratio
was approximately 3.64, with ratios from 1.23 to 7.55. These are condition-by-condition ratios
of median decoder times, not a pooled end-to-end speedup or a confidence interval.
The median reduction-only ratio was 1.02 (range 0.77–1.39): removing inactive detector rows
did not produce a consistent additional speed benefit in this small-code benchmark.

Four-round median decoder wall times, in milliseconds per shot:

| Bell fidelity | Measurement | Full surgery | Hardware, all detectors | Hardware, reduced detectors |
|---|---|---:|---:|---:|
| 1 | `XX` | 33.85 | 9.75 | 10.11 |
| 1 | `ZZ` | 15.55 | 4.55 | 4.54 |
| 0.999 | `XX` | 31.55 | 9.10 | 11.83 |
| 0.999 | `ZZ` | 12.98 | 3.36 | 3.27 |

The primary gain here is the smaller adapter-hardware fault model. It changes the experiment
being decoded and cannot be advertised as an equivalent acceleration of the original
full-surgery-noise calculation. Detector reduction preserves the hardware noise experiment,
but does not remove quantum gates or the code blocks themselves.
