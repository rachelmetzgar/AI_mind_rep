#!/bin/bash
#SBATCH --job-name=exp3_cvs_v1
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=/jukebox/graziano/rachel/mind_rep/exp_3/logs/concept_steering_v1/%A_%a.out
#SBATCH --error=/jukebox/graziano/rachel/mind_rep/exp_3/logs/concept_steering_v1/%A_%a.err
# ---------------------------------------------------------------------------
# Experiment 3: Concept Vector Steering V1 (mean-vector, single-turn)
#
# Array job: one task per concept dimension.
# Each task runs all 3 layer strategies x 2 strengths x 2 directions + baseline
# = 12 steered + 1 baseline = 13 generation passes (60 questions each = 780 total).
#
# Initial dimensions (4 array tasks):
#   0 = entity baseline     (positive control — broad shift expected)
#   1 = biological           (partial control — humans are biological)
#   2 = shapes               (negative control — minimal shift expected)
#   3 = attention            (core mental property — tests specificity)
#
# Prerequisites:
#   - Concept vectors extracted (1_elicit_concept_vectors.py)
#   - Exp 2 metacognitive probes trained (for exp2_peak strategy)
#   - Concept-aligned layers computed (2h_concept_aligned_layers.py)
#
# Submit:
#   VERSION=balanced_gpt sbatch exp_3/code/slurm/4a_concept_steering_v1.sh
#   VERSION=balanced_gpt sbatch --array=0 exp_3/code/slurm/4a_concept_steering_v1.sh  # baseline only
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

# === Config ===
VERSION=${VERSION:?ERROR: VERSION is required. Use VERSION=balanced_gpt sbatch ...}
PROJECT_ROOT="/jukebox/graziano/rachel/mind_rep/exp_3"
mkdir -p "$PROJECT_ROOT/logs/concept_steering_v1"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

# === Map array index to dimension ID ===
DIM_IDS=(0 14 15 17)
ARRAY_IDX=${SLURM_ARRAY_TASK_ID}

if [[ $ARRAY_IDX -ge ${#DIM_IDS[@]} ]]; then
    echo "FATAL: SLURM_ARRAY_TASK_ID=$ARRAY_IDX exceeds dimension list (${#DIM_IDS[@]} entries)"
    exit 1
fi

DIM_ID=${DIM_IDS[$ARRAY_IDX]}

echo "[$(date)] Exp3 Concept Vector Steering V1"
echo "  version=$VERSION | dim_id=$DIM_ID | array_idx=$ARRAY_IDX"
echo "  host=$HOSTNAME"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# === Run V1 generation ===
STRENGTHS=${STRENGTHS:-"2 4"}
python code/pipeline/4_concept_steering_generate.py \
    --version "$VERSION" \
    --dim_id "$DIM_ID" \
    --strategies exp2_peak upper_half concept_aligned \
    --strengths $STRENGTHS

echo "[$(date)] Exp3 Concept Vector Steering V1 finished | dim_id=$DIM_ID"
