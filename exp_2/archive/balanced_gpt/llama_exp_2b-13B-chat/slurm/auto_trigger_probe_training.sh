#!/bin/bash
#SBATCH --job-name=auto_train_probes
#SBATCH --partition=all
#SBATCH --dependency=aftercorr:3559400
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_2/balanced_gpt/llama_exp_2b-13B-chat/logs/train_probes/auto_train_probes_%j.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_2/balanced_gpt/llama_exp_2b-13B-chat/logs/train_probes/auto_train_probes_%j.err

# ---------------------------------------------------------------------------
# Auto-triggered probe training after Exp 1 data generation completes.
# Waits for ALL array tasks of job 3559400 to complete successfully (aftercorr).
# Trains reading and control probes across all layers.
#
# Submit:  sbatch slurm/auto_trigger_probe_training.sh
# ---------------------------------------------------------------------------

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
PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_2/balanced_gpt/llama_exp_2b-13B-chat"
PY_SCRIPT="$PROJECT_ROOT/2_train_and_read_controlling_probes.py"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] Auto-triggered probe training after Exp 1 data generation completion"
echo "  Data source: exp_1/balanced_gpt/data/meta-llama-Llama-2-13b-chat-hf/0.8/"
echo "  Host: $HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# === Check data exists ===
DATA_DIR="/jukebox/graziano/rachel/ai_mind_rep/exp_1/balanced_gpt/data/meta-llama-Llama-2-13b-chat-hf/0.8"
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

CSV_COUNT=$(ls "$DATA_DIR"/s???.csv 2>/dev/null | wc -l)
echo "Found $CSV_COUNT subject CSV files"

if [ "$CSV_COUNT" -lt 50 ]; then
    echo "WARNING: Expected 50 subject files, found only $CSV_COUNT"
fi

# === Run probe training ===
python "$PY_SCRIPT"

echo "[$(date)] Auto-triggered probe training finished"
