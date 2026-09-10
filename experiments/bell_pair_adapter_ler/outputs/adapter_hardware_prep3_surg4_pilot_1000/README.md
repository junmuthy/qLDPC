# Adapter hardware comparison pilot

Surface-3/Steane, three ideal independent preparation rounds, four adapter rounds,
physical depolarizing parameter `p=0.001`, Bell source fidelity `F=1`, no idling noise,
and ideal terminal closure. Each noise policy uses 1,000 shots per basis. The configuration
records the seed and dependency versions.

| Measurement | Full surgery failures | Hardware failures | Full detector rows | Retained rows |
|---|---:|---:|---:|---:|
| `XX` | 2 | 0 | 131 | 80 |
| `ZZ` | 5 | 2 | 126 | 86 |

Hardware rows in `summary.csv` with and without detector reduction decode the same
measurement records and are paired validation data, not additional independent shots.
Both representations predicted identical outcomes for every shared shot. The plot shows
only one hardware estimate per basis, with 95% Wilson intervals. Zero `XX` failures does
not establish a zero LER: its Wilson upper bound is approximately 0.383%.

Measured decoder time for 1,000 shots in a single process:

| Measurement | Full surgery noise | Hardware noise, full detectors | Hardware noise, reduced detectors |
|---|---:|---:|---:|
| `XX` | 30.40 s | 11.02 s | 11.00 s |
| `ZZ` | 14.43 s | 4.77 s | 3.14 s |

Changing the noise selection changes the quantity being estimated. Only the last two
columns compare representations of the same noise model on identical samples. These are
single-pass timing measurements, not guaranteed speedup factors. Build, conversion, and
decoder compilation times are recorded separately in `timing.csv`. The reduced conversion
reuses existing measurement records, so it does not measure an independent sampling speedup.

Use a fresh output directory to rerun the command in the parent experiment README.
