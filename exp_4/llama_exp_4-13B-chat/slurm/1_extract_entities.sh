#!/bin/bash
#SBATCH --job-name=exp4_entities
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/mnt/cup/labs/graziano/rachel/ai_mind_rep/exp_4/llama_exp_4-13B-chat/logs/1_extract_entities_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/ai_mind_rep/exp_4/llama_exp_4-13B-chat/logs/1_extract_entities_%j.err

# Experiment 4, Phase 1: Extract entity representations
# Runs both conditions (without_self and with_self) in one job.
# Model is loaded once; both runs share it.

export PS1=${PS1:-}
set -euo pipefail
module load pyger
export PYTHONNOUSERSITE=1
set +u
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate llama2_env
set -u
trap 'set +u; conda deactivate >/dev/null 2>&1 || true; set -u' EXIT

PROJECT_ROOT="/mnt/cup/labs/graziano/rachel/ai_mind_rep/exp_4/llama_exp_4-13B-chat"
mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] Starting entity extraction"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

echo ""
echo "=== Run 1: without self ==="
python 1_extract_entity_representations.py

echo ""
echo "=== Run 2: with self ==="
python 1_extract_entity_representations.py --include_self

echo "[$(date)] Done"
