#!/bin/bash
#SBATCH --job-name=sysprompt_align
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_3/llama_exp_3-13B-chat/logs/align/sysprompt_%A.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_3/llama_exp_3-13B-chat/logs/align/sysprompt_%A.err
# -------------------------------------------------------------
# Phase 1e: System prompt ↔ concept alignment analysis
#
# CPU only — no GPU needed (just cosine similarities + bootstrap).
# Requires 1d outputs in data/concept_activations/.
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

PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_3/llama_exp_3-13B-chat"
mkdir -p "$PROJECT_ROOT/logs/align"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] Starting system prompt ↔ concept alignment"
echo "  Node: $(hostname)"

python code/analysis/alignment/1e_sysprompt_alignment.py --analysis all

echo "[$(date)] Phase 1e complete"