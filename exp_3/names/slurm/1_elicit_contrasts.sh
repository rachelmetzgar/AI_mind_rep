#!/bin/bash
#SBATCH --job-name=elicit_contrasts
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=16-17
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_3/llama_exp_3-13B-chat/logs/elicit/contrasts_%A_%a.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_3/llama_exp_3-13B-chat/logs/elicit/contrasts_%A_%a.err
# -------------------------------------------------------------
# Concept elicitation: CONTRASTS mode (human vs AI paired prompts)
# Parallelized: one dimension per SLURM array task
#
# Array indices map to dimension IDs:
#   0    = entity baseline
#   1-7  = mental properties
#   8-10 = physical/ontological
#   11-13 = alternative hypotheses
#   14   = biological control
#   15   = shapes (orthogonal control)
#   18   = attention (cognitive marker)
#   19   = general mind (concatenation of dims 1-10, 400+400 prompts)
#
# Usage:
#   sbatch elicit_contrasts.sh              # all dims
#   sbatch --array=1,5,7 elicit_contrasts.sh  # subset
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
mkdir -p "$PROJECT_ROOT/logs/elicit"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

DIM_ID=${SLURM_ARRAY_TASK_ID}

echo "[$(date)] Starting concept elicitation (contrasts) — dim_id=${DIM_ID}"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

python 1_elicit_concept_vectors.py \
    --mode contrasts \
    --dim_id ${DIM_ID} \
    --concepts_root concepts

echo "[$(date)] Finished concept elicitation (contrasts) — dim_id=${DIM_ID}"