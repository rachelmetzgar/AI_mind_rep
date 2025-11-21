#!/bin/bash
#SBATCH --job-name=train_probes
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_2a/logs/train_probes_%A.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_2a/logs/train_probes_%A.err

# -------------------------------------------------------------
# TRAIN PROBES — Human vs AI
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
conda activate llama2_env
set -u
trap 'set +u; conda deactivate >/dev/null 2>&1 || true; set -u' EXIT

# === Project paths ===
PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_2a"
PY_SCRIPT="$PROJECT_ROOT/2_train_and_read_controlling_probes.py"
LOG_DIR="$PROJECT_ROOT/logs"

# Make sure log dir exists
mkdir -p "$LOG_DIR"

# Always work from project root
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] Starting probe training job on $HOSTNAME"
echo "GPU available: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# === Run probe training ===
python "$PY_SCRIPT"

echo "[$(date)] Finished probe training job"
