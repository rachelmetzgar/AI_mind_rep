#!/bin/bash
#SBATCH --job-name=simple_align
#SBATCH --partition=all
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_3/logs/simple/alignment_%A_%a.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_3/logs/simple/alignment_%A_%a.err
# -------------------------------------------------------------
# Simple pipeline: alignment analysis for _simple variant
#
# Requires env vars: VERSION, TURN
#
# Usage:
#   export VERSION=balanced_gpt TURN=5; sbatch simple_alignment.sh
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
mkdir -p "$PROJECT_ROOT/logs/simple"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] Starting simple alignment: VERSION=${VERSION}, TURN=${TURN}"
echo "  Node: $(hostname)"

python code/2a_alignment_analysis.py \
    --version "${VERSION}" --turn "${TURN}" \
    --analysis standalone --variant _simple

echo "[$(date)] Finished simple alignment: VERSION=${VERSION}, TURN=${TURN}"
