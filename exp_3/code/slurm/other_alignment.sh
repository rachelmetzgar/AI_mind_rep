#!/bin/bash
#SBATCH --job-name=align_other
#SBATCH --partition=all
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_3/logs/alignment/other_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_3/logs/alignment/other_%j.err
# -------------------------------------------------------------
# Alignment analysis for other-focused standalone variant
# CPU-only — no GPU needed.
#
# Usage:
#   export VERSION=balanced_gpt TURN=5; sbatch other_alignment.sh
#   export VERSION=nonsense_codeword TURN=5; sbatch other_alignment.sh
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

VERSION=${VERSION:?ERROR: VERSION is required}
TURN=${TURN:-5}
PROJECT_ROOT="/mnt/cup/labs/graziano/rachel/mind_rep/exp_3"
mkdir -p "$PROJECT_ROOT/logs/alignment"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] Starting alignment analysis (other variant) — version=${VERSION}, turn=${TURN}"
echo "  Node: $(hostname)"

python code/2a_alignment_analysis.py \
    --version "${VERSION}" \
    --turn "${TURN}" \
    --analysis standalone \
    --variant _other

echo "[$(date)] Finished alignment analysis (other variant) — version=${VERSION}, turn=${TURN}"
