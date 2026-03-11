#!/bin/bash
#SBATCH --job-name=elicit_other
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=1-19,25-27,30-32
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_3/logs/elicit/other_standalone_%A_%a.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_3/logs/elicit/other_standalone_%A_%a.err
# -------------------------------------------------------------
# Concept elicitation: STANDALONE mode with other-focused prompts
# Same dims as standalone (1-19, 25-27, 30-32), using concepts/standalone_other/
# Outputs get _other suffix via --variant _other
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
mkdir -p "$PROJECT_ROOT/logs/elicit"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

DIM_ID=${SLURM_ARRAY_TASK_ID}

echo "[$(date)] Starting concept elicitation (standalone_other) — dim_id=${DIM_ID}"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

python code/1_elicit_concept_vectors.py \
    --mode standalone \
    --dim_id ${DIM_ID} \
    --concepts_root concepts/other \
    --variant _other

echo "[$(date)] Finished concept elicitation (standalone_other) — dim_id=${DIM_ID}"
