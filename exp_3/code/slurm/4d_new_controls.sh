#!/bin/bash
#SBATCH --job-name=exp3_ctrl
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_3/logs/new_controls/%A_%a.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_3/logs/new_controls/%A_%a.err
# ---------------------------------------------------------------------------
# Experiment 3: New Control Dimensions — Elicit + Steer
#
# Array job (3 tasks):
#   0 = 32_horizontal_vertical
#   1 = 30_granite_sandstone
#   2 = 31_squares_triangles
#
# Each task:
#   1. Extracts concept vectors (forward passes on 80 prompts)
#   2. Runs steering generation (exp2_peak + upper_half, N=4)
#
# Submit:
#   VERSION=balanced_gpt sbatch code/slurm/4d_new_controls.sh
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
mkdir -p "$PROJECT_ROOT/logs/new_controls"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

DIM_IDS=(32 30 31)
ARRAY_IDX=${SLURM_ARRAY_TASK_ID}

if [[ $ARRAY_IDX -ge ${#DIM_IDS[@]} ]]; then
    echo "FATAL: SLURM_ARRAY_TASK_ID=$ARRAY_IDX exceeds dimension list"
    exit 1
fi

DIM_ID=${DIM_IDS[$ARRAY_IDX]}

echo "[$(date)] New control dimension pipeline"
echo "  version=$VERSION | dim_id=$DIM_ID | array_idx=$ARRAY_IDX"
echo "  host=$HOSTNAME"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# === Step 1: Extract concept vectors ===
echo ""
echo "====== Step 1: Concept vector extraction ======"
python code/1_elicit_concept_vectors.py \
    --mode contrasts \
    --dim_id "$DIM_ID" \
    --concepts_root concepts

# === Step 2: Concept steering (exp2_peak + upper_half, N=4) ===
echo ""
echo "====== Step 2: Concept steering generation ======"
python code/4_concept_steering_generate.py \
    --version "$VERSION" \
    --dim_id "$DIM_ID" \
    --strategies exp2_peak upper_half \
    --strengths 4

echo ""
echo "[$(date)] New control pipeline complete | dim_id=$DIM_ID"
