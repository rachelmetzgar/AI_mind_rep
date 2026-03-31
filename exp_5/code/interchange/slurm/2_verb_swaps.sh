#!/bin/bash
#SBATCH --job-name=exp5_verb_swap
#SBATCH --partition=all
#SBATCH --gres=gpu:1 --mem=48G --time=10:00:00
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/interchange/verb_swap/verb_swap_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/interchange/verb_swap/verb_swap_%j.err

# Verb swap interventions

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
mkdir -p "$PROJECT_ROOT/logs/interchange/verb_swap"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

MODEL=${MODEL:-llama2_13b_chat}

echo "[$(date)] Starting Verb swap interventions"
echo "  model=$MODEL"
echo "  host=$HOSTNAME"
echo "  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

python code/interchange/2_verb_swap_interventions.py --model "$MODEL"

echo "[$(date)] Done"
