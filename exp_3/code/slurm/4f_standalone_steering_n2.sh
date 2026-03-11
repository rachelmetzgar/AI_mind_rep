#!/bin/bash
#SBATCH --job-name=exp3_sa_n2
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_3/logs/concept_steering_v1/standalone_n2_%A_%a.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_3/logs/concept_steering_v1/standalone_n2_%A_%a.err
# ---------------------------------------------------------------------------
# Experiment 3: Standalone Concept Vector Steering — N=2
#
# Same dims as 4f_standalone_steering.sh but with strength N=2.
# Existing N=4 results are preserved (script auto-skips completed).
#
# Array job (5 tasks):
#   0 = 18_attention
#   1 = 16_human (mind holistic)
#   2 = 1_phenomenology
#   3 = 15_shapes
#   4 = 14_biological
#
# Submit:
#   VERSION=balanced_gpt sbatch code/slurm/4f_standalone_steering_n2.sh
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

VERSION=${VERSION:?ERROR: VERSION is required. Use VERSION=balanced_gpt sbatch ...}
PROJECT_ROOT="/mnt/cup/labs/graziano/rachel/mind_rep/exp_3"
mkdir -p "$PROJECT_ROOT/logs/concept_steering_v1"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

DIM_IDS=(18 16 1 15 14)
ARRAY_IDX=${SLURM_ARRAY_TASK_ID}

if [[ $ARRAY_IDX -ge ${#DIM_IDS[@]} ]]; then
    echo "FATAL: SLURM_ARRAY_TASK_ID=$ARRAY_IDX exceeds dimension list"
    exit 1
fi

DIM_ID=${DIM_IDS[$ARRAY_IDX]}

echo "[$(date)] Standalone concept steering (N=2)"
echo "  version=$VERSION | dim_id=$DIM_ID | array_idx=$ARRAY_IDX"
echo "  host=$HOSTNAME"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python code/4_concept_steering_generate.py \
    --version "$VERSION" \
    --dim_id "$DIM_ID" \
    --mode standalone \
    --strategies exp2_peak upper_half \
    --strengths 2

echo "[$(date)] Standalone steering (N=2) complete | dim_id=$DIM_ID"
