#!/bin/bash
#SBATCH --job-name=concept_behav
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_3/logs/behav_%A.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_3/logs/behav_%A.err
# -------------------------------------------------------------
# Behavioral analysis on concept injection outputs (Experiment 3)
# Runs all dimensions × all strengths (1, 2, 4, 8).
# No GPU needed.
#
# Submit:
#   sbatch 4_behavioral_analysis.sh
# Or chain after intervention:
#   sbatch --dependency=afterok:<V1_JOBID> 4_behavioral_analysis.sh
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
PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_3"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }
echo "[$(date)] Starting Exp 3 behavioral analysis | host=$HOSTNAME"
# Run all dimensions, all strengths (1, 2, 4, 8), V1
python code/pipeline/6_behavior_analysis.py \
    --version v1 \
    --all \
    --strengths 1 2 4 8
echo "[$(date)] Finished Exp 3 behavioral analysis"