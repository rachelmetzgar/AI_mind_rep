#!/bin/bash
#SBATCH --job-name=probe_turn_cmp
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_2/balanced_names/llama_exp_2b-13B-chat/logs/probe_turn_cmp/turn_%a_%A.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_2/balanced_names/llama_exp_2b-13B-chat/logs/probe_turn_cmp/turn_%a_%A.err

# ----------------------------------------------------------------
# PROBE TURN COMPARISON — Turns 1-4 only
# Turn 5 probes already exist at data/probe_checkpoints/turn_5/
# Array index 0 → turn_index=0 (turn 1 only)
# Array index 1 → turn_index=1 (through turn 2)
# Array index 2 → turn_index=2 (through turn 3)
# Array index 3 → turn_index=3 (through turn 4)
# ----------------------------------------------------------------

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

# === Map array index to turn_index ===
TURN_INDICES=(0 1 2 3)
TURN_INDEX=${TURN_INDICES[$SLURM_ARRAY_TASK_ID]}

# === Project paths ===
PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_2/balanced_names/llama_exp_2b-13B-chat"
PY_SCRIPT="$PROJECT_ROOT/2b_train_probes_turn_comparison.py"
LOG_DIR="$PROJECT_ROOT/logs/probe_turn_cmp"
mkdir -p "$LOG_DIR"

cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] Starting probe training: turn_index=${TURN_INDEX} on $HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python "$PY_SCRIPT" --turn_index "$TURN_INDEX"

echo "[$(date)] Finished probe training: turn_index=${TURN_INDEX}"
