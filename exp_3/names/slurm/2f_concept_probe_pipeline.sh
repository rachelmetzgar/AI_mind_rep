#!/bin/bash
#SBATCH --job-name=probe_pipeline
#SBATCH --partition=all
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_3/llama_exp_3-13B-chat/logs/probe_pipeline/%j.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_3/llama_exp_3-13B-chat/logs/probe_pipeline/%j.err
# -------------------------------------------------------------
# Concept-probe alignment pipeline: stats then figures.
# No GPU needed — CPU-only permutation tests + bootstrap + plotting.
#
# Usage:
#   sbatch slurm/2f_concept_probe_pipeline.sh           # run both
#   sbatch --export=STEP=stats slurm/2f_concept_probe_pipeline.sh   # stats only
#   sbatch --export=STEP=figures slurm/2f_concept_probe_pipeline.sh # figures only
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

PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_3/llama_exp_3-13B-chat"
mkdir -p "$PROJECT_ROOT/logs/probe_pipeline"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

STEP=${STEP:-all}

echo "[$(date)] Starting concept-probe pipeline — step=${STEP}"
echo "  Node: $(hostname)"

if [[ "$STEP" == "all" || "$STEP" == "stats" ]]; then
    echo ""
    echo "=== Step 1: Statistical analysis ==="
    python 2d_concept_probe_stats.py
fi

if [[ "$STEP" == "all" || "$STEP" == "figures" ]]; then
    echo ""
    echo "=== Step 2: Generating figures ==="
    python 2e_concept_probe_figures.py
fi

echo ""
echo "[$(date)] Pipeline complete — step=${STEP}"
echo "Results in: results/concept_probe_alignment/"
