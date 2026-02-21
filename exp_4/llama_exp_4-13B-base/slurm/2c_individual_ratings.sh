#!/bin/bash
#SBATCH --job-name=exp4_indiv
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/mnt/cup/labs/graziano/rachel/ai_mind_rep/exp_4/llama_exp_4-13B-base/logs/2c_individual_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/ai_mind_rep/exp_4/llama_exp_4-13B-base/logs/2c_individual_%j.err

# Experiment 4, Phase 2c: Individual Likert ratings — BASE model
# Only 216-234 forward passes (vs 2376-2808 for pairwise), so much faster.

export PS1=${PS1:-}
set -euo pipefail
module load pyger
export PYTHONNOUSERSITE=1

# Point to cached model on labs filesystem
export HF_HOME="/mnt/cup/labs/graziano/rachel/.cache_huggingface"
export HF_HUB_CACHE="/mnt/cup/labs/graziano/rachel/.cache_huggingface/hub"
export HF_HUB_OFFLINE=1

set +u
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate llama2_env
set -u
trap 'set +u; conda deactivate >/dev/null 2>&1 || true; set -u' EXIT

PROJECT_ROOT="/mnt/cup/labs/graziano/rachel/ai_mind_rep/exp_4/llama_exp_4-13B-base"
mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] Starting individual Likert ratings (BASE model)"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

echo ""
echo "=== Run 1: without self (12 entities) ==="
python 2c_individual_ratings.py

echo ""
echo "=== Run 2: with self (13 entities) ==="
python 2c_individual_ratings.py --include_self

echo "[$(date)] Done"
