#!/bin/bash
#SBATCH --job-name=V2_steer
#SBATCH --array=0-49
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_2b-13B-chat/logs/causality_V2_%A_%a.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_2b-13B-chat/logs/causality_V2_%A_%a.err

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
PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_2b-13B-chat"
PY_SCRIPT="$PROJECT_ROOT/3b_causality_exp1_recreation.py"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] Starting V2 subject $SLURM_ARRAY_TASK_ID on $HOSTNAME"
echo "GPU available: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# === Run — pass array task ID as subject index ===
python "$PY_SCRIPT" "$SLURM_ARRAY_TASK_ID"

echo "[$(date)] Finished V2 subject $SLURM_ARRAY_TASK_ID"