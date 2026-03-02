#!/bin/bash
#SBATCH --job-name=standalone_pipeline
#SBATCH --partition=all
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/jukebox/graziano/rachel/mind_rep/exp_3/logs/standalone_pipeline/%j.out
#SBATCH --error=/jukebox/graziano/rachel/mind_rep/exp_3/logs/standalone_pipeline/%j.err
# -------------------------------------------------------------
# Standalone concept activation alignment pipeline:
#   stats → figures → HTML summary.
# No GPU needed — CPU-only bootstrap + plotting.
#
# Usage:
#   sbatch --export=VERSION=labels slurm/4b_standalone_pipeline.sh                          # run all
#   sbatch --export=VERSION=labels,STEP=stats slurm/4b_standalone_pipeline.sh      # stats only
#   sbatch --export=VERSION=labels,STEP=figures slurm/4b_standalone_pipeline.sh    # figures only
#   sbatch --export=VERSION=labels,STEP=html slurm/4b_standalone_pipeline.sh       # html only
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
mkdir -p "$PROJECT_ROOT/logs/standalone_pipeline"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

STEP=${STEP:-all}

echo "[$(date)] Starting standalone alignment pipeline — version=${VERSION}, step=${STEP}"
echo "  Node: $(hostname)"

if [[ "$STEP" == "all" || "$STEP" == "stats" ]]; then
    echo ""
    echo "=== Step 1: Statistical analysis ==="
    python code/analysis/probes/4a_compute_alignment_stats.py --version "${VERSION}" --mode standalone
fi

if [[ "$STEP" == "all" || "$STEP" == "figures" ]]; then
    echo ""
    echo "=== Step 2: Generating figures ==="
    python code/analysis/probes/4b_generate_alignment_figures.py --version "${VERSION}" --mode standalone
fi

if [[ "$STEP" == "all" || "$STEP" == "html" ]]; then
    echo ""
    echo "=== Step 3: Building HTML summary ==="
    python results/standalone_alignment/build_html_summary.py
fi

echo ""
echo "[$(date)] Pipeline complete — version=${VERSION}, step=${STEP}"
echo "Results in: results/standalone_alignment/${VERSION}/"
