#!/bin/bash
#SBATCH --job-name=behav_analysis
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_2b-13B-chat/logs/behav_analysis_%A.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_2b-13B-chat/logs/behav_analysis_%A.err
# -------------------------------------------------------------
# Behavioral analysis on V1 causality test output.
# No GPU needed — CPU only (regex + stats).
#
# Expects:
#   V1/control_probes/intervention_responses.csv
#   V1/reading_probes/intervention_responses.csv
#
# Submit chained after causality test:
#   sbatch --dependency=afterok:<CAUSALITY_JOBID> behavior_analysis.sh
# -------------------------------------------------------------
export PS1=${PS1:-}
set -euo pipefail

# === Activate environment ===
module load pyger
export PYTHONNOUSERSITE=1
set +u
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate behavior_env
set -u
trap 'set +u; conda deactivate >/dev/null 2>&1 || true; set -u' EXIT

# === Project paths ===
PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_2b-13B-chat"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] Starting behavioral analysis"

python "$PROJECT_ROOT/4_behavior_analysis.py" \
    --input "$PROJECT_ROOT/data/intervention_results/V2/per_subject" \
    --version v2 \
    --topics "$PROJECT_ROOT/data/conds/topics.csv"

echo "[$(date)] Finished behavioral analysis"