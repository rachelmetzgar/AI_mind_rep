#!/bin/bash
#SBATCH --job-name=probe_pipeline
#SBATCH --partition=all
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_3/logs/probe_pipeline/%j.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_3/logs/probe_pipeline/%j.err
# -------------------------------------------------------------
# Concept-probe alignment pipeline: stats then figures.
# No GPU needed — CPU-only permutation tests + bootstrap + plotting.
#
# Usage:
#   sbatch --export=VERSION=labels slurm/4a_contrast_pipeline.sh           # run both
#   sbatch --export=VERSION=labels,STEP=stats slurm/4a_contrast_pipeline.sh   # stats only
#   sbatch --export=VERSION=labels,STEP=figures slurm/4a_contrast_pipeline.sh # figures only
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
PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_3"
mkdir -p "$PROJECT_ROOT/logs/probe_pipeline"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

STEP=${STEP:-all}

echo "[$(date)] Starting concept-probe pipeline — version=${VERSION}, step=${STEP}"
echo "  Node: $(hostname)"

if [[ "$STEP" == "all" || "$STEP" == "stats" ]]; then
    echo ""
    echo "=== Step 1: Statistical analysis ==="
    python code/analysis/probes/4a_compute_alignment_stats.py --version "${VERSION}" --mode both
fi

if [[ "$STEP" == "all" || "$STEP" == "figures" ]]; then
    echo ""
    echo "=== Step 2: Generating figures ==="
    python code/analysis/probes/4b_generate_alignment_figures.py --version "${VERSION}" --mode both
fi

echo ""
echo "[$(date)] Pipeline complete — version=${VERSION}, step=${STEP}"
echo "Results in: results/concept_probe_alignment/${VERSION}/"
