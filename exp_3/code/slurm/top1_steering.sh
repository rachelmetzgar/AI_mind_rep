#!/bin/bash
#SBATCH --job-name=top1_steer
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_3/logs/top1/steering_%A_%a.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_3/logs/top1/steering_%A_%a.err
# -------------------------------------------------------------
# Top-1 pipeline: concept steering generation for _1 variant
#
# Requires env vars: VERSION, DIM_ID
#
# Usage:
#   export VERSION=balanced_gpt DIM_ID=1; sbatch top1_steering.sh
# -------------------------------------------------------------
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

PROJECT_ROOT="/mnt/cup/labs/graziano/rachel/mind_rep/exp_3"
mkdir -p "$PROJECT_ROOT/logs/top1"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] Starting top-1 steering: VERSION=${VERSION}, DIM_ID=${DIM_ID}"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

python code/4_concept_steering_generate.py \
    --version "${VERSION}" --dim_id "${DIM_ID}" --variant _1 \
    --strategies exp2_peak upper_half --strengths 4

echo "[$(date)] Finished top-1 steering: VERSION=${VERSION}, DIM_ID=${DIM_ID}"
