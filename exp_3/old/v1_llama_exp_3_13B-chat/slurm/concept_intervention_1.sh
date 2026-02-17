#!/bin/bash
#SBATCH --job-name=concept_v1
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-15
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_3/llama_exp_3-13B-chat/logs/concept_v1_%A_%a.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_3/llama_exp_3-13B-chat/logs/concept_v1_%A_%a.err
# -------------------------------------------------------------
# Concept injection V1: single-turn test questions
# Parallelized: one dimension per SLURM array task
#
# Array indices map to dimension IDs:
#   0  = entity baseline
#   1-7  = mental properties
#   8-10 = physical/ontological
#   11-13 = alternative hypotheses
#   14 = biological control
#   15 = shapes (orthogonal control)
#
# Usage:
#   sbatch concept_intervention_v1.sh              # all dims
#   sbatch --array=1,5,7,11 concept_intervention_v1.sh  # subset
#
# Chain after concept elicitation + probe training:
#   sbatch --dependency=afterok:<PROBE_JOBID> concept_intervention_v1.sh
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

PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_3/llama_exp_3-13B-chat"
mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

DIM_ID=${SLURM_ARRAY_TASK_ID}

echo "[$(date)] Starting concept intervention V1 — dim_id=${DIM_ID}"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

python 3_concept_intervention.py \
    --mode V1 \
    --dim_id ${DIM_ID} \
    --strengths 1 2 4 8

echo "[$(date)] Finished concept intervention V1 — dim_id=${DIM_ID}"