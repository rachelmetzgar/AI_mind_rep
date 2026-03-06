#!/bin/bash
#SBATCH --job-name=concept_overlap
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/jukebox/graziano/rachel/mind_rep/exp_3/logs/alignment/%j_concept_overlap.out
#SBATCH --error=/jukebox/graziano/rachel/mind_rep/exp_3/logs/alignment/%j_concept_overlap.err
# -------------------------------------------------------------
# Contrast concept overlap analysis + report generation
# CPU only — loads contrast activations, runs 1000-iter bootstrap,
# then generates HTML/MD report with figures.
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

PROJECT_ROOT="/jukebox/graziano/rachel/mind_rep/exp_3"
mkdir -p "$PROJECT_ROOT/logs/alignment"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] Starting contrast concept overlap analysis"
echo "  Node: $(hostname)"

python code/analysis/alignment/2f_concept_overlap.py --exclude-dims 10 16

echo "[$(date)] Generating contrast overlap report"

python code/analysis/alignment/2f_concept_overlap_report.py

echo "[$(date)] Done — contrast concept overlap"
