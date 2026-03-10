#!/bin/bash
#SBATCH --job-name=exp3_conv_align
#SBATCH --partition=all
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_3/logs/conv_alignment_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_3/logs/conv_alignment_%j.err
# ---------------------------------------------------------------------------
# Experiment 3, Phase 9b: Concept-conversation alignment analysis
#
# Submit:
#   VERSION=balanced_gpt sbatch exp_3/code/slurm/9b_concept_conversation_alignment.sh
#   VERSION=balanced_gpt TURN=3 sbatch exp_3/code/slurm/9b_concept_conversation_alignment.sh
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
TURN=${TURN:-5}
DIM_IDS=${DIM_IDS:-}
PROJECT_ROOT="/mnt/cup/labs/graziano/rachel/mind_rep/exp_3"
mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] Concept-conversation alignment analysis"
echo "  version=$VERSION turn=$TURN"
echo "  host=$HOSTNAME"
[ -n "${DIM_IDS}" ] && echo "  dim_ids=${DIM_IDS}"

python code/9b_concept_conversation_alignment.py \
    --version "$VERSION" \
    --turn "$TURN" \
    ${DIM_IDS:+--dim_ids ${DIM_IDS}}

echo "[$(date)] Done"
