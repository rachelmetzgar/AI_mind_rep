#!/bin/bash
#SBATCH --job-name=exp3_ctrl_sa
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_3/logs/new_controls/standalone_%A_%a.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_3/logs/new_controls/standalone_%A_%a.err
# ---------------------------------------------------------------------------
# Experiment 3: New Control Dimensions — Standalone Concept Extraction
#
# Array job (3 tasks):
#   0 = 32_horizontal_vertical
#   1 = 30_granite_sandstone
#   2 = 31_squares_triangles
#
# Submit:
#   sbatch code/slurm/4e_new_controls_standalone.sh
# ---------------------------------------------------------------------------

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
mkdir -p "$PROJECT_ROOT/logs/new_controls"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

DIM_IDS=(32 30 31)
DIM_ID=${DIM_IDS[$SLURM_ARRAY_TASK_ID]}

echo "[$(date)] Standalone concept extraction — dim_id=$DIM_ID"
echo "  host=$HOSTNAME"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python code/1_elicit_concept_vectors.py \
    --mode standalone \
    --dim_id "$DIM_ID" \
    --concepts_root concepts

echo "[$(date)] Standalone extraction complete | dim_id=$DIM_ID"
