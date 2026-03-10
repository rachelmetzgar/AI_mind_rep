#!/bin/bash
#SBATCH --job-name=exp3_flip
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_3/logs/concept_steering_v1/flip_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_3/logs/concept_steering_v1/flip_%j.err
# ---------------------------------------------------------------------------
# Experiment 3: Shapes Flip Sanity Check — Steering
#
# Runs concept steering with the NEGATED shapes vector (angular = +1 pole).
# If behavioral effects reverse vs original shapes, the effects are from
# shape semantics in activation space, not steering procedure artifacts.
#
# Prerequisites:
#   python code/4c_shapes_flip_test.py --mode setup --version balanced_gpt
#
# Submit:
#   VERSION=balanced_gpt sbatch code/slurm/4c_shapes_flip.sh
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

echo "[$(date)] Shapes flip sanity check — steering"
echo "  version=$VERSION"
echo "  host=$HOSTNAME"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# dim_id=29 = 29_shapes_flip (negated shapes vector)
# exp2_peak only — same layers as original, just flipped direction
python code/4_concept_steering_generate.py \
    --version "$VERSION" \
    --dim_id 29 \
    --strategies exp2_peak upper_half \
    --strengths 4

echo "[$(date)] Shapes flip steering complete"
