#!/bin/bash
#SBATCH --job-name=behav_V1_all
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --array=0-27
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_2/names/llama_exp_2b-13B-chat/logs/behavior_v1/behavior_V1_all_%A_%a.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_2/names/llama_exp_2b-13B-chat/logs/behavior_v1/behavior_V1_all_%A_%a.err
# ---------------------------------------------------------------------------
# V1 Behavioral analysis: ALL strategies × ALL strengths
# Array: 4 strategies × 7 strengths = 28 jobs (0-27)
#   strategy_idx = TASK_ID / 7
#   strength_idx = TASK_ID % 7
# No GPU needed.
# ---------------------------------------------------------------------------

STRATEGIES=(narrow wide peak_15 all_70)
STRENGTHS=(1 2 3 4 5 6 8)

N_STRENGTHS=${#STRENGTHS[@]}
STRATEGY_IDX=$(( SLURM_ARRAY_TASK_ID / N_STRENGTHS ))
STRENGTH_IDX=$(( SLURM_ARRAY_TASK_ID % N_STRENGTHS ))

STRATEGY=${STRATEGIES[$STRATEGY_IDX]}
N=${STRENGTHS[$STRENGTH_IDX]}

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

PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_2/names/llama_exp_2b-13B-chat"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] V1 behavioral analysis | strategy=$STRATEGY | strength=$N | host=$HOSTNAME"

python 5_behavior_analysis.py \
    --version v1 \
    --strength "$N" \
    --input "data/intervention_results/V1/${STRATEGY}"

echo "[$(date)] V1 behavioral analysis finished | strategy=$STRATEGY | strength=$N"
