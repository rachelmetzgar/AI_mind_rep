#!/bin/bash
#SBATCH --job-name=llm_data_gen
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-49
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_1/logs/llm_data_gen_%A_%a.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_1/logs/llm_data_gen_%A_%a.err

# -------------------------------------------------------------
# SLURM ARRAY SCRIPT — each task runs one subject
# -------------------------------------------------------------

# Prevent conda / module PS1 bug under -u
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
PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_1/code/data_gen"
PY_SCRIPT="$PROJECT_ROOT/llm_data_generation.py"
LOG_DIR="$PROJECT_ROOT/logs"

# Make sure log dir exists
mkdir -p "$LOG_DIR"

# Always work from project root
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

# === Run one subject per array task ===
IDX=${SLURM_ARRAY_TASK_ID:-0}
echo "[$(date)] Starting subject index $IDX"

python "$PY_SCRIPT" "$IDX"

echo "[$(date)] Finished subject index $IDX"
