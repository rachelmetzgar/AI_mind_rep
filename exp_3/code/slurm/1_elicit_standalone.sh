#!/bin/bash
#SBATCH --job-name=elicit_standalone
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=1-19
#SBATCH --output=/mnt/cup/graziano/rachel/mind_rep/exp_3/logs/elicit/standalone_%A_%a.out
#SBATCH --error=/mnt/cup/graziano/rachel/mind_rep/exp_3/logs/elicit/standalone_%A_%a.err
# -------------------------------------------------------------
# Concept elicitation: STANDALONE mode (concept-only, no entity framing)
# Parallelized: one dimension per SLURM array task
#
# Array indices 1-19 (no dim 0 — entity baseline has no standalone version)
#   1-7  = mental properties
#   8-10 = physical/ontological
#   11-13 = alternative hypotheses
#   14   = biological control
#   15   = shapes (orthogonal control)
#   16   = standalone human (entity topic)
#   17   = standalone AI (entity topic)
#   18   = attention (cognitive marker)
#   19   = general mind (concatenation of dims 1-7, 280 prompts)
#
# Usage:
#   sbatch elicit_standalone.sh              # all dims
#   sbatch --array=1,5,7 elicit_standalone.sh  # subset
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
mkdir -p "$PROJECT_ROOT/logs/elicit"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

DIM_ID=${SLURM_ARRAY_TASK_ID}

echo "[$(date)] Starting concept elicitation (standalone) — dim_id=${DIM_ID}"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

python code/1_elicit_concept_vectors.py \
    --mode standalone \
    --dim_id ${DIM_ID} \
    --concepts_root concepts

echo "[$(date)] Finished concept elicitation (standalone) — dim_id=${DIM_ID}"