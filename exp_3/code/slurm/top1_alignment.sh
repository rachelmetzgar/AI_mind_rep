#!/bin/bash
#SBATCH --job-name=top1_align
#SBATCH --partition=all
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_3/logs/top1/alignment_%A_%a.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_3/logs/top1/alignment_%A_%a.err
# -------------------------------------------------------------
# Top-1 pipeline: alignment analysis for _1 variant
#
# Requires env vars: VERSION, TURN
#
# Usage:
#   export VERSION=balanced_gpt TURN=5; sbatch top1_alignment.sh
#   # Or for all 5 turns x 2 versions, submit 10 jobs
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
mkdir -p "$PROJECT_ROOT/logs/top1"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] Starting top-1 alignment: VERSION=${VERSION}, TURN=${TURN}"
echo "  Node: $(hostname)"

python code/2a_alignment_analysis.py \
    --version "${VERSION}" --turn "${TURN}" \
    --analysis all --variant _1

echo "[$(date)] Finished top-1 alignment: VERSION=${VERSION}, TURN=${TURN}"
