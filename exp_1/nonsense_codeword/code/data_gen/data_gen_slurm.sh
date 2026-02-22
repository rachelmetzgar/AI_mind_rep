#!/bin/bash
#SBATCH --job-name=nonsense_codeword_data_gen
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-49
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_1/nonsense_codeword/logs/llm_data_gen_%A_%a.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_1/nonsense_codeword/logs/llm_data_gen_%A_%a.err

# SLURM ARRAY SCRIPT — Exp 1 (nonsense_codeword control)
# Token-matched control: "Your assigned session code word is {a Human / an AI}."
# Each task runs one subject (50 subjects total)

# Prevent conda / module PS1 bug under -u
export PS1=${PS1:-}
set -euo pipefail

# === Activate environment ===
module load pyger
export PYTHONNOUSERSITE=1
set +u
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate llama2_env
set -u
trap 'set +u; conda deactivate >/dev/null 2>&1 || true; set -u' EXIT

# === Project paths ===
PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_1/nonsense_codeword"
SCRIPT_DIR="$PROJECT_ROOT/code/data_gen"
PY_SCRIPT="$SCRIPT_DIR/llm_data_generation.py"
LOG_DIR="$PROJECT_ROOT/logs"

# Make sure log dir exists
mkdir -p "$LOG_DIR"

# cd to script dir so relative paths (utils/, logs/) resolve correctly
cd "$SCRIPT_DIR" || { echo "FATAL: Cannot cd to $SCRIPT_DIR"; exit 1; }

# === Run one subject per array task ===
IDX=${SLURM_ARRAY_TASK_ID:-0}
echo "[$(date)] Starting subject index $IDX"

python "$PY_SCRIPT" "$IDX"

echo "[$(date)] Finished subject index $IDX"
