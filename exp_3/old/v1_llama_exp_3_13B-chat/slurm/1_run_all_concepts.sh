#!/bin/bash
#SBATCH --job-name=exp3_concepts
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-15
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_3/llama_exp_3-13B-chat/logs/concept_dim%a_%A_%a.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_3/llama_exp_3-13B-chat/logs/concept_dim%a_%A_%a.err
# -------------------------------------------------------------
# Experiment 3: Concept Dimension Analysis (parallel)
#
# SLURM array job: each task handles one concept dimension.
# Task ID ($SLURM_ARRAY_TASK_ID) = dimension ID (0-14).
#
# Dimensions:
#   1  phenomenal_experience      8  embodiment
#   2  emotions_affect            9  functional_roles
#   3  agency                    10  animacy
#   4  intentions_goals          11  formality_register
#   5  prediction_anticipation   12  expertise_knowledge
#   6  cognitive_processes       13  helpfulness_service
#   7  social_cognition          14  biological
#
# Each task:
#   Phase 1: Extract concept activations (~80 forward passes)
#   Phase 2: Train concept probes + alignment with Exp 2b
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

echo "============================================================"
echo "[$(date)] Exp 3 — Dimension ${DIM_ID} on ${HOSTNAME}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================================"

# Phase 1: Extract concept activations
echo "[$(date)] Phase 1: Extracting concept activations (dim=${DIM_ID})..."
python 1_elicit_concept_vectors.py --dim_id "${DIM_ID}"

# Phase 2: Train probes + alignment
echo "[$(date)] Phase 2: Training probes + alignment (dim=${DIM_ID})..."
python 2_train_concept_probes.py --dim_id "${DIM_ID}"

echo "[$(date)] ✅ Dimension ${DIM_ID} complete."