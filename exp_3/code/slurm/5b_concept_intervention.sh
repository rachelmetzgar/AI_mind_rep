#!/bin/bash
#SBATCH --job-name=concept_v2
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-49
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_3/logs/concept_v2_%A_%a.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_3/logs/concept_v2_%A_%a.err
# -------------------------------------------------------------
# Concept injection V2: multi-turn Exp 1 recreation
# One subject per array task, using concept probe vectors
# Chain after concept elicitation:
#   sbatch --dependency=afterok:<ELICIT_JOBID> concept_intervention_v2.sh
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

PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_3"
mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

IDX=${SLURM_ARRAY_TASK_ID:-0}
echo "[$(date)] Starting concept intervention V2 — subject index $IDX"

python code/pipeline/5_concept_intervention.py --mode v2 --vector_source probe --subject_idx $IDX

echo "[$(date)] Finished subject $IDX"