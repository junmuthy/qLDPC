# HGP–surface hardware-only schedule ablations

This study changes one scheduling control at a time from the corresponding
legacy XX or ZZ circuit. It does **not** start from the complete exact-dual ZZ
circuit. Code and old results were preserved in `9c4ab967` on
`adapter-hardware-only-ler`; this study is on `hgp-hardware-schedule-ablations`.
Implementation commits: `33555f89` (opt-in controls) and `918cee54` (runner/tests).

## Completed results

All six new tasks finished at exactly 20,000 shots and zero discards: 120,000 new
shots total. Every percentage is a per-PPM failure probability, not a per-round rate.
Parentheses give failures out of 20,000; intervals are pointwise 95% Wilson intervals.

| Schedule | XX LER | XX interval | ZZ LER | ZZ interval |
|---|---:|---:|---:|---:|
| Legacy | 0.110% (22) | 0.0727–0.1665% | 0.080% (16) | 0.0493–0.1299% |
| A: Bell order | 0.345% (69) | 0.2727–0.4363% | 0.165% (33) | 0.1175–0.2316% |
| B: Local order | 0.085% (17) | 0.0531–0.1361% | 0.105% (21) | 0.0687–0.1605% |
| C: Local layers | 0.270% (54) | 0.2070–0.3521% | 0.265% (53) | 0.2027–0.3464% |

The archived combined exact-dual ZZ control is 0.095% (19/20,000), with interval
0.0608–0.1483%. It is shown separately in gray and is not a one-factor experiment.

Against each basis's legacy baseline, three increases remain significant at a
familywise 0.05 threshold after Holm correction over the six preplanned tests:

- XX Bell order: adjusted p=0.00000478; LER increase 0.235 percentage points.
- XX local layers: adjusted p=0.00124; increase 0.160 percentage points.
- ZZ local layers: adjusted p=0.0000447; increase 0.185 percentage points.

ZZ Bell order rises by 0.085 percentage points, with nominal p=0.0212 but adjusted
p=0.0636: suggestive, not significant at that familywise threshold. Both local-order
comparisons have adjusted p=1.0. This does not prove equivalence or exclude smaller
effects. No statistically resolved improvement was observed.

The complete dual result cannot be attributed to a single independently beneficial
change. Interactions between the scheduling changes, ideal preparation ordering,
and decoding representation would require further controlled experiments; this
suite does not isolate those interactions or explain the full-surgery noise excess.

## Runtime

The uninterrupted invocation took 1,334.03 seconds (22 minutes 14 seconds), including
construction, six pilots, six continuations, and intermediate plots. Historical worker
seconds normalized to 10,000 shots are below; these are not controlled speed measurements.

| Schedule | XX worker seconds / 10k | ZZ worker seconds / 10k |
|---|---:|---:|
| Legacy | 823.4 | 745.3 |
| A: Bell order | 834.4 | 620.7 |
| B: Local order | 611.6 | 834.3 |
| C: Local layers | 803.0 | 705.7 |

Both reference-summary hashes were rechecked after the run and are unchanged.

## Fixed settings

- HGP simplex m=3, [[98,18,4]], and unrotated distance-4 surface code; logical
  index zero in each block, with their original logical supports and adapter geometry.
- Four noiseless independent preparation rounds; four noisy merged adapter rounds.
- Physical p=0.001, Bell state fidelity F=1, no idling noise.
- Hardware-only noise selection, including complete two-qubit errors at the
  block/adapter boundary. Block-only faults remain disabled.
- Original detachment and ideal terminal readout; original physical PPM observable.
- BP+OSD defaults; exact undecomposed DEM; inactive-detector reduction.
- 20,000 shots per new task, including its first 1,000-shot pilot; eight workers.
- No exact-distance searches and no new full-surgery simulation.

## Independent variants

| Policy | Change from that basis's legacy schedule | Unchanged controls |
|---|---|---|
| A: `bell_order` | Bell-assisted `Z` checks before `X` checks | Local schedule, Bell partners, within-family and within-check orders |
| B: `local_order` | Local `Z` layers before `X` layers | Individual local family layers and all Bell extraction |
| C: `local_layers` | Local gate layers from the mapped *opposite legacy basis* | Target legacy `X`-then-`Z` family order and all Bell extraction |

C changes local coupling sequences and parallel grouping together, not just
one gate reversal. Its total tick count changes from 328 to 332 for ZZ, and from
332 to 328 for XX. A and B retain their respective legacy total tick counts.
Ideal gate-only extraction depths in the audit exclude the noise model's extra
moment boundaries; use `variant_total_ticks` for the complete simulation circuit.

Unlike the earlier exact-dual control, preparation schedules are **not** transported.
Detector meanings and full detector ordering remain tied to legacy physical checks
and epochs. Only measurement-record references are regenerated. The active detector
selection and DEM size can change as consequences of changing fault propagation.

## Validation and provenance

All six variants pass the physical gate/noise inventory, ideal preparation,
canonical detector/observable identity, Bell partner/within-check sequence, and
noiseless sampling checks. Default circuits and ordered DEMs match the archived
hardware baselines exactly. None of the six new ordered DEMs is identical to an
existing reference or another variant, so all six receive independent samples.

158 focused tests passed, including saved-artifact task-identity round trips,
invalid-layer rejection, baseline preservation, decoder smoke checks, and
comparison plotting with zero failures. Ruff and mypy checks on the new schedule
and experiment modules pass. The legacy `circuit.py` retains three pre-existing
mypy diagnostics (the measurement-record annotation and two ndarray annotations).

- `schedule_audit.json`: assertions, hashes, gate depths, Bell orders, and strong IDs.
- `manifest.json`, `run_config.json`: experiment identity, configuration, versions.
- `Z/<policy>/`, `X/<policy>/`: independent resumable checkpoints and summaries.
- `comparison.csv`: legacy baselines, new estimates, and combined exact-dual reference.
- `differences.csv`: differences from each basis's legacy baseline, pointwise
  Newcombe/Wilson 95% intervals, Fisher exact p-values, and Holm-adjusted p-values
  across the six planned tests. Shared baseline observations are never double-counted
  as independent shots or pooled across curves.
- `hgp_schedule_ablation_comparison.pdf` / `.png`: pointwise Wilson 95% error bars.
- `invocations.jsonl`: pilot and final cumulative counts and stage wall times.

Pilot and final counts are cumulative, not independent datasets to add together.
Intermediate plots are descriptive only; the shot total is fixed, not chosen
based on repeated significance tests. Historical worker times are workload records,
not a controlled performance benchmark. Pointwise confidence intervals are not
simultaneous intervals, and overlap does not establish equivalence.

## Reproduce or resume

```bash
PYTHONPATH=src:. NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache MPLBACKEND=Agg \
  MPLCONFIGDIR=/tmp/qldpc-matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.run_hgp_schedule_ablations \
  --max-shots 20000 --pilot-shots 1000 --num-workers 8
```

Use `--analyze-only` to regenerate the comparison without collecting new shots.
Existing legacy and exact-dual reference archives are read-only inputs.
