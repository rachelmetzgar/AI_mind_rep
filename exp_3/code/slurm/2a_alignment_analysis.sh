#!/bin/bash
#SBATCH --job-name=alignment
#SBATCH --partition=all
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/jukebox/graziano/rachel/mind_rep/exp_3/logs/alignment/%j.out
#SBATCH --error=/jukebox/graziano/rachel/mind_rep/exp_3/logs/alignment/%j.err
# -------------------------------------------------------------
# Alignment analysis: concept vectors vs conversational probes
# No GPU needed — CPU-only vector math + bootstrap.
#
# Runs all three analyses by default: raw, residual, standalone.
# To run a single analysis, override via environment variable:
#   sbatch --export=VERSION=labels,ANALYSIS=raw 2a_alignment_analysis.sh
#   sbatch --export=VERSION=labels,ANALYSIS=standalone 2a_alignment_analysis.sh
#
# Chain after elicitation:
#   sbatch --dependency=afterok:<ELICIT_JOBID> --export=VERSION=labels 2a_alignment_analysis.sh
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
PROJECT_ROOT="/jukebox/graziano/rachel/mind_rep/exp_3"
mkdir -p "$PROJECT_ROOT/logs/alignment"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

ANALYSIS=${ANALYSIS:-all}
TURN=${TURN:-5}

echo "[$(date)] Starting alignment analysis — version=${VERSION}, turn=${TURN}, mode=${ANALYSIS}"
echo "  Node: $(hostname)"

python code/analysis/alignment/2a_alignment_analysis.py --version "${VERSION}" --turn "${TURN}" --analysis "${ANALYSIS}"

echo "[$(date)] Finished alignment analysis — version=${VERSION}, turn=${TURN}, mode=${ANALYSIS}"
