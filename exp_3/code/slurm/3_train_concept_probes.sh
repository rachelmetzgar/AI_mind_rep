#!/bin/bash
#SBATCH --job-name=train_cprobes
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-19
#SBATCH --output=/mnt/cup/graziano/rachel/mind_rep/exp_3/logs/train_concept_probes/%A_%a.out
#SBATCH --error=/mnt/cup/graziano/rachel/mind_rep/exp_3/logs/train_concept_probes/%A_%a.err
# -------------------------------------------------------------
# Train concept probes on contrast activations (human vs AI)
# Parallelized: one dimension per SLURM array task
#
# Array indices map to dimension IDs:
#   0    = entity baseline
#   1-7  = mental properties
#   8-10 = physical/ontological
#   11-13 = alternative hypotheses
#   14   = biological control
#   15   = shapes (orthogonal control)
#
# Usage:
#   sbatch train_concept_probes.sh              # all dims
#   sbatch --array=1,5,7 train_concept_probes.sh  # subset
#
# Chain after elicitation:
#   sbatch --dependency=afterok:<ELICIT_JOBID> train_concept_probes.sh
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

PROJECT_ROOT="/mnt/cup/graziano/rachel/mind_rep/exp_3"
mkdir -p "$PROJECT_ROOT/logs/train_concept_probes"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

DIM_ID=${SLURM_ARRAY_TASK_ID}

echo "[$(date)] Starting concept probe training — dim_id=${DIM_ID}"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

python code/3_train_concept_probes.py \
    --dim_id ${DIM_ID}

echo "[$(date)] Finished concept probe training — dim_id=${DIM_ID}"