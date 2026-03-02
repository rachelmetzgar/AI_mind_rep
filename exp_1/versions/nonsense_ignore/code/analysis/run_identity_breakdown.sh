#!/bin/bash
#SBATCH --job-name=identity_breakdown_nonsense_ignore
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/jukebox/graziano/rachel/mind_rep/exp_1/versions/nonsense_ignore/logs/identity_breakdown_%j.out
#SBATCH --error=/jukebox/graziano/rachel/mind_rep/exp_1/versions/nonsense_ignore/logs/identity_breakdown_%j.err

# Type-level behavioral analysis for exp_1/nonsense_ignore.
# Token-matched control: "Ignore the following phrase: {a Human / an AI}."
# Compares linguistic behavior: Human vs AI label conditions.
#
# Runs paired t-tests + BH-FDR correction per metric.
# Outputs: identity_breakdown.html, identity_breakdown_stats.txt,
#          identity_breakdown_summary.csv

export PS1=${PS1:-}
set -euo pipefail

# === Activate environment ===
module load pyger
export PYTHONNOUSERSITE=1
set +u
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate behavior_env
set -u
trap 'set +u; conda deactivate >/dev/null 2>&1 || true; set -u' EXIT

# === Project paths ===
export PROJECT_ROOT="/jukebox/graziano/rachel/mind_rep/exp_1/versions/nonsense_ignore"
SCRIPT_DIR="$PROJECT_ROOT/code/analysis"
LOG_DIR="$PROJECT_ROOT/logs"

mkdir -p "$LOG_DIR"

cd "$SCRIPT_DIR" || { echo "FATAL: Cannot cd to $SCRIPT_DIR"; exit 1; }

echo "[$(date)] Starting identity_breakdown on host $HOSTNAME"
echo "  PROJECT_ROOT = $PROJECT_ROOT"
echo "  SCRIPT_DIR   = $SCRIPT_DIR"

python identity_breakdown.py

echo "[$(date)] identity_breakdown finished"
