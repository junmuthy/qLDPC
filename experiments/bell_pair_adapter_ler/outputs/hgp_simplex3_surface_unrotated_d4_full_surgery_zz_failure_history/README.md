# Archived full-surgery ZZ failure history

This is an analysis of existing data, not a new simulation. The original checkpoint
and previous comparison plots are unchanged.

Source: `../hgp_simplex3_surface_unrotated_d4_intrinsic_prep4_surg4_bell0_10k/sinter_progress.csv`.
Only original ZZ task ID
`c95edd6d15d1242ba996a4636541712add7f4ab78d4f50d7d76015b9e6d23ab6`
is plotted: 10,000 shots, 30 failures, 878 recorded batches.
The accidental fresh-task restart (ID
`26bde915a2ccfd33debf613c964bc32a1d6f24ac43526de6c502b81c9304971f`)
contributed 22 shots and zero failures and is excluded.

The horizontal axis concatenates original-task batches in CSV append order.
Parallel workers finish batches asynchronously, so this is neither exact
shot-execution order nor wall-clock time. Individual failed-shot indices were not
recorded. Red bar widths show entire failure-containing batch ranges; they do not
mean that all shots inside the bar failed. The lower curve updates the cumulative
failure count at each batch end. Its dashed line is a constant-rate reference
using the final observed rate, not a confidence band or an independent prediction.

The stop/resume milestones were recovered from the conversation and verified
against cumulative original-task counts in the checkpoint:

- Corrected restart: 22 shots, zero failures.
- Unexpected stop: 1,614 shots, four failures (source CSV line 369).
- Overnight pause: 5,789 shots, 16 failures (source CSV line 1010).
- Completion: 10,000 shots, 30 failures (source CSV line 1669).

`zz_failure_batches.csv` exports all 878 original-task batches, including those with
zero failures. Shot-range endpoints are inclusive. `source_csv_line` refers to
the original checkpoint, with its header counted as line 1.

To regenerate from the repository root:

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/qldpc-matplotlib ../.venv/bin/python -m experiments.bell_pair_adapter_ler.plot_hgp_failure_history
```
