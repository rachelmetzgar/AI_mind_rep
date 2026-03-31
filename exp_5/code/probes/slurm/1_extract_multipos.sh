#!/bin/bash
#SBATCH --job-name=exp5_multipos
#SBATCH --partition=all
#SBATCH --gres=gpu:1 --mem=48G --time=1:00:00
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/activations/multipos/multipos_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/activations/multipos/multipos_%j.err

# Extract multi-position activations

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
mkdir -p "$PROJECT_ROOT/logs/activations/multipos"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

MODEL=${MODEL:-llama2_13b_chat}

echo "[$(date)] Starting Extract multi-position activations"
echo "  model=$MODEL"
echo "  host=$HOSTNAME"
echo "  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

python code/probes/1_extract_multipos_activations.py --model "$MODEL"

echo "[$(date)] Done"
