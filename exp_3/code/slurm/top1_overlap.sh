#!/bin/bash
#SBATCH --job-name=top1_overlap
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_3/logs/top1/overlap_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_3/logs/top1/overlap_%j.err
# -------------------------------------------------------------
# Top-1 pipeline: concept overlap analysis for _1 variant
#
# Usage:
#   sbatch top1_overlap.sh
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

echo "[$(date)] Starting top-1 concept overlap analysis"
echo "  Node: $(hostname)"

python code/2f_concept_overlap.py --variant _1

echo "[$(date)] Finished top-1 concept overlap analysis"
