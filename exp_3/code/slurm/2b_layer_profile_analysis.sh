#!/bin/bash
#SBATCH --job-name=layer_profiles
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/cup/graziano/rachel/mind_rep/exp_3/logs/alignment/%j.out
#SBATCH --error=/mnt/cup/graziano/rachel/mind_rep/exp_3/logs/alignment/%j.err
# -------------------------------------------------------------
# Layer profile analysis: per-layer alignment patterns from 2a output
# No GPU needed — reads .npz files and computes summary statistics.
#
# Runs all three analyses by default: raw, residual, standalone.
# To run a single analysis, override via environment variable:
#   sbatch --export=VERSION=labels,ANALYSIS=raw 2b_layer_profile_analysis.sh
#   sbatch --export=VERSION=labels,ANALYSIS=standalone 2b_layer_profile_analysis.sh
#
# Chain after alignment:
#   sbatch --dependency=afterok:<ALIGN_JOBID> --export=VERSION=labels 2b_layer_profile_analysis.sh
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

VERSION=${VERSION:?ERROR: VERSION is required. Use --export=VERSION=labels}
PROJECT_ROOT="/mnt/cup/graziano/rachel/mind_rep/exp_3"
mkdir -p "$PROJECT_ROOT/logs/alignment"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

ANALYSIS=${ANALYSIS:-all}
TURN=${TURN:-5}

echo "[$(date)] Starting layer profile analysis — version=${VERSION}, turn=${TURN}, mode=${ANALYSIS}"
echo "  Node: $(hostname)"

python code/2b_layer_profile_analysis.py --version "${VERSION}" --turn "${TURN}" --analysis "${ANALYSIS}"

echo "[$(date)] Finished layer profile analysis — version=${VERSION}, turn=${TURN}, mode=${ANALYSIS}"
