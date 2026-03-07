#!/bin/bash
#SBATCH --job-name=sysprompt_align
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/cup/graziano/rachel/mind_rep/exp_3/logs/align/sysprompt_%A.out
#SBATCH --error=/mnt/cup/graziano/rachel/mind_rep/exp_3/logs/align/sysprompt_%A.err
# -------------------------------------------------------------
# Phase 2d: System prompt ↔ concept alignment analysis
#
# CPU only — no GPU needed (just cosine similarities + bootstrap).
# Requires 2c outputs in results/{model}/concept_activations/.
#
# Output: data/alignment_results/sysprompt/
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

PROJECT_ROOT="/mnt/cup/graziano/rachel/mind_rep/exp_3"
mkdir -p "$PROJECT_ROOT/logs/align"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] Starting system prompt ↔ concept alignment"
echo "  Node: $(hostname)"

python code/2d_sysprompt_alignment.py --analysis all

echo "[$(date)] Phase 2d complete"