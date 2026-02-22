#!/bin/bash
#SBATCH --job-name=concept_behav
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_3-13B-chat_mind/logs/behav_%A.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_3-13B-chat_mind/logs/behav_%A.err
# -------------------------------------------------------------
# Behavioral analysis on concept injection outputs
# Chain after intervention:
#   sbatch --dependency=afterok:<V1_JOBID> behavior_analysis.sh
# No GPU needed.
# -------------------------------------------------------------

export PS1=${PS1:-}
set -euo pipefail

module load pyger
export PYTHONNOUSERSITE=1
set +u
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate behavior_env
set -u
trap 'set +u; conda deactivate >/dev/null 2>&1 || true; set -u' EXIT

PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_3-13B-chat_mind"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] Starting behavioral analysis"

# V1 probe-based
python 4_behavior_analysis.py \
    --input data/intervention_results/concept_probe_v1/intervention_responses.csv \
    --version v1

# V1 mean-based
python 4_behavior_analysis.py \
    --input data/intervention_results/concept_mean_v1/intervention_responses.csv \
    --version v1

echo "[$(date)] Finished behavioral analysis"