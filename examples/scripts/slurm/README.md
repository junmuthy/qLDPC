# Cluster-friendly surgery LER sweep

Runs Webster code 0 [[62, 10, 6]] surgery PPM vs idling LER sweep on a Slurm cluster.

## Files

- `run_webster0_ler_sweep.py` — main Python driver
- `submit.sbatch` — example Slurm submission script
- `README.md` — this file

## What it does

For 6 physical error rates `p`, runs two parallel sinter sweeps:
1. **Surgery PPM** — `build_single_ppm_circuit` with `obs1` stripped (only Webster Eq.1 PPM observable kept)
2. **Idling** — `get_memory_experiment(code, basis=Pauli.X, num_rounds=3)`

Decodes with `qldpc.decoders.SinterDecoder` (BP + OSD).
Saves results to a CSV (resumable), prints a summary table, and writes a log-log plot.

## Running locally

```bash
python examples/scripts/slurm/run_webster0_ler_sweep.py \
    --out-dir results/local_run \
    --workers 8 \
    --max-shots 5000 \
    --max-errors 50
```

## Running on Slurm

1. Edit `submit.sbatch` to set `PROJECT_DIR` and `VENV_DIR` for your environment.
2. Submit:

```bash
mkdir -p logs results
sbatch examples/scripts/slurm/submit.sbatch
```

3. Check status: `squeue -u $USER`, logs at `logs/webster0_ler_<jobid>.out`.

## Running on PBS / Torque

1. Edit `submit.pbs` to set `PROJECT_DIR` and `VENV_DIR`.
2. Adjust the `#PBS -l select=...` line for your site (some sites use the older
   `#PBS -l nodes=1:ppn=N` syntax — the file has both with one commented out).
3. Submit:

```bash
mkdir -p logs results
qsub examples/scripts/slurm/submit.pbs
```

4. Check status: `qstat -u $USER`, logs at `logs/webster0_ler.out`.

## Resuming

The script uses `sinter.collect(save_resume_filepath=...)`. If interrupted,
**re-run the same command** — completed work is skipped.

```bash
sbatch examples/scripts/slurm/submit.sbatch     # first submit
# (job times out / preempted)
sbatch examples/scripts/slurm/submit.sbatch     # same script → resumes
```

## Outputs

In `results/webster0_run_<jobid>/`:

- `results.csv`     — raw sinter stats (resumable, machine-readable)
- `summary.txt`     — human-readable LER table
- `ler_curve.png`   — log-log plot

## Resource estimates (rough)

- Merged code: ~71 qubits, ~280 detectors at rounds=3
- BP+OSD decoding: ~1–2 s per shot on a single core
- 6 p × 2 kinds × 200 000 shots = 2.4M shots worst case
- With `--cpus-per-task=32`, expect ~4 hours wall (varies with p — high p finishes early via `max_errors`)

## Tuning

- `--max-shots`: shot count cap per task (lower = faster, less precision at low p)
- `--max-errors`: stop early once this many errors are seen (lower at low-p typically rare)
- `--p-values`: pass in custom values, e.g. `--p-values 0.001 0.003 0.01`
- `--rounds`: surgery / memory SE rounds (3 by default; increase if you want longer experiments)

## Notes

- Required Python packages: `qldpc`, `stim`, `sinter`, `numpy`, `matplotlib`, `sympy`, `galois`, `ldpc` (decoder)
- The script uses `matplotlib.use("Agg")` so no display is needed.
- Stdout is line-buffered (`python -u`) so Slurm log files update in near-real-time.
