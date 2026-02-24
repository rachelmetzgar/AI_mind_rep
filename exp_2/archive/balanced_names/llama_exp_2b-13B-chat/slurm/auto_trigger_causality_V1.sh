#!/bin/bash
#SBATCH --job-name=auto_causal_V1
#SBATCH --partition=all
#SBATCH --dependency=afterok:3559458
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_2/balanced_names/llama_exp_2b-13B-chat/logs/causality_v1/auto_causal_V1_%j.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_2/balanced_names/llama_exp_2b-13B-chat/logs/causality_v1/auto_causal_V1_%j.err

# ---------------------------------------------------------------------------
# Auto-triggered V1 causality generation after probe training completes.
# Runs wide, peak_15, and all_70 strategies with strengths 1, 2, 4, 8.
# This job waits for probe training job 3559458 to complete successfully.
#
# Submit:  sbatch slurm/auto_trigger_causality_V1.sh
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
PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_2/balanced_names/llama_exp_2b-13B-chat"
PY_SCRIPT="$PROJECT_ROOT/3_causality_generate.py"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] Auto-triggered V1 generation after probe training completion"
echo "  Strategies: wide, peak_15, all_70"
echo "  Strengths: 1, 2, 4, 8"
echo "  Host: $HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# === Run generation for all three strategies ===
python "$PY_SCRIPT" --version V1 --layer_strategy wide peak_15 all_70 --strengths 1 2 4 8

echo "[$(date)] Auto-triggered V1 generation finished"
