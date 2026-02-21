#!/bin/bash
#SBATCH --job-name=layer_profiles
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_3/llama_exp_3-13B-chat/logs/alignment/%j.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_3/llama_exp_3-13B-chat/logs/alignment/%j.err
# -------------------------------------------------------------
# Layer profile analysis: per-layer alignment patterns from 1b output
# No GPU needed — reads .npz files and computes summary statistics.
#
# Runs all three analyses by default: raw, residual, standalone.
# To run a single analysis, override via environment variable:
#   sbatch --export=ANALYSIS=raw layer_profile_analysis.sh
#   sbatch --export=ANALYSIS=standalone layer_profile_analysis.sh
#
# Chain after alignment:
#   sbatch --dependency=afterok:<ALIGN_JOBID> layer_profile_analysis.sh
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
mkdir -p "$PROJECT_ROOT/logs/alignment"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }
ANALYSIS=${ANALYSIS:-all}
echo "[$(date)] Starting layer profile analysis — mode=${ANALYSIS}"
echo "  Node: $(hostname)"
python 1c_layer_profile_analysis.py --analysis "${ANALYSIS}"
echo "[$(date)] Finished layer profile analysis — mode=${ANALYSIS}"