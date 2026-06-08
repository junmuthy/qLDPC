#!/bin/bash
#SBATCH --job-name=webster0_ler
#SBATCH --output=logs/webster0_ler_%j.out
#SBATCH --error=logs/webster0_ler_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G

# --- adjust these for your cluster ---
# Example: source your virtual environment.
# Replace with your project path / venv location.
PROJECT_DIR="${PROJECT_DIR:-$HOME/qLDPC}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
# -------------------------------------

cd "$PROJECT_DIR"
mkdir -p logs results

if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

# Show config for debugging
echo "PROJECT_DIR=$PROJECT_DIR"
echo "VENV_DIR=$VENV_DIR"
python --version
which python

# Each sinter task gets one CPU; SLURM_CPUS_PER_TASK is auto-picked by the script.
python -u examples/scripts/slurm/run_webster0_ler_sweep.py \
    --out-dir "results/webster0_run_${SLURM_JOB_ID:-local}" \
    --rounds 3 \
    --max-shots 200000 \
    --max-errors 200 \
    --p-values 0.001 0.002 0.004 0.008 0.015 0.025
