#!/bin/bash
#SBATCH --job-name=exp3_V1
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-19
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_3/logs/concept_V1/%A_%a.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_3/logs/concept_V1/%A_%a.err
# ---------------------------------------------------------------------------
# Experiment 3: Concept Injection V1 (single-turn, 60 questions)
#
# One SLURM array job per concept dimension.
# Each job sweeps strengths 1, 2, 4 for its dimension.
# Model loads once per job, baseline generated once, then 3 strengths × 2
# conditions = 6 generation passes per dimension.
#
# Dimensions:
#   0    = entity baseline
#   1-7  = mental properties
#   8-10 = physical/ontological
#   11-13 = alternative hypotheses
#   14   = biological control
#   15   = shapes (orthogonal control)
#   16-17 = standalone only (no probes, skipped automatically)
#   18   = mind
#   19   = attention
#
# Estimated time: ~1-2 hours per dimension (60 questions × 3 strengths × 3
# conditions = 540 generations, but baseline is shared across strengths).
#
# Submit:
#   sbatch 5a_concept_intervention.sh
#   sbatch --array=7,3,4,11 5a_concept_intervention.sh   # subset
# ---------------------------------------------------------------------------

export PS1=${PS1:-}
set -euo pipefail

# === Activate environment ===
module load pyger
export PYTHONNOUSERSITE=1
set +u
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate llama2_env
set -u
trap 'set +u; conda deactivate >/dev/null 2>&1 || true; set -u' EXIT

# === Project paths ===
PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_3"
mkdir -p "$PROJECT_ROOT/logs/concept_V1"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

# === Dim ID comes directly from array index ===
DIM_ID=${SLURM_ARRAY_TASK_ID}

echo "[$(date)] Exp3 V1 concept intervention | dim_id=$DIM_ID | array_idx=$SLURM_ARRAY_TASK_ID | host=$HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# === Run V1 generation: all 3 strengths for this dimension ===
python code/pipeline/5_concept_intervention.py \
    --mode V1 \
    --dim_id "$DIM_ID" \
    --strengths 1 2 4

echo "[$(date)] Exp3 V1 concept intervention finished | dim_id=$DIM_ID"