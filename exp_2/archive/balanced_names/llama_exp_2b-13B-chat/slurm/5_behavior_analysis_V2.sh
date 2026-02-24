#!/bin/bash
#SBATCH --job-name=behav_V2
#SBATCH --partition=all
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_2/balanced_names/llama_exp_2b-13B-chat/logs/behavior_v2/behavior_V2_%A_%a.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_2/balanced_names/llama_exp_2b-13B-chat/logs/behavior_v2/behavior_V2_%A_%a.err

# ---------------------------------------------------------------------------
# V2 Behavioral analysis: parallelized by intervention strength.
# Array index: 0→2, 1→4, 2→8, 3→16
# Each job loads ALL subjects for both probe types at one strength,
# runs feature computation and RM-ANOVAs.
# No GPU needed, but needs more memory for multi-subject data.
#
# Submit after V2 generation completes:
#   sbatch 4_behavior_analysis_V2.sh
# ---------------------------------------------------------------------------

#STRENGTHS=(2 4 8 16)
STRENGTHS=(6)
N=${STRENGTHS[$SLURM_ARRAY_TASK_ID]}

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

PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_2/balanced_names/llama_exp_2b-13B-chat"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

# Optional: pass topics file for social/nonsocial classification
TOPICS_FILE="$PROJECT_ROOT/data/topics.csv"
TOPICS_FLAG=""
if [ -f "$TOPICS_FILE" ]; then
    TOPICS_FLAG="--topics $TOPICS_FILE"
fi

echo "[$(date)] V2 behavioral analysis | strength=$N | host=$HOSTNAME"

python 5_behavior_analysis.py --version v2 --strength $N --input data/intervention_results/V2/peak_15 $TOPICS_FLAG

echo "[$(date)] V2 behavioral analysis finished | strength=$N"