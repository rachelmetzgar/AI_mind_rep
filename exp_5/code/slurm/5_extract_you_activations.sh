#!/bin/bash
#SBATCH --job-name=exp5_extract_you
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/activations/extract_you_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/activations/extract_you_%j.err

# Phase 5: Extract activations for "You" variant stimuli (C2/C5 modified).
# GPU required. ~5-10 min for 336 short sentences.

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

PROJECT_ROOT="/mnt/cup/labs/graziano/rachel/mind_rep/exp_5"
mkdir -p "$PROJECT_ROOT/logs/activations"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

MODEL=${MODEL:-llama2_13b_chat}

echo "[$(date)] Starting 'You' variant activation extraction"
echo "  model=$MODEL"
echo "  host=$HOSTNAME"
echo "  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

python code/5_extract_you_activations.py --model "$MODEL"

echo "[$(date)] Done"
