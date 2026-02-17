#!/bin/bash
#SBATCH --job-name=behav_V1
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --array=0-4
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_2b/llama_exp_2b-13B-chat/logs/behavior_V1_%A_%a.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_2b/llama_exp_2b-13B-chat/logs/behavior_V1_%A_%a.err

# ---------------------------------------------------------------------------
# V1 Behavioral analysis: parallelized by intervention strength.
# Array index: 0→2, 1→4, 2→8, 3→16
# Each job analyzes BOTH probe types at one strength.
# No GPU needed.
#
# Submit after V1 generation completes:
#   sbatch 4_behavior_analysis_V1.sh
# ---------------------------------------------------------------------------

STRENGTHS=(2 4 6 8 16)
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

PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_2b/llama_exp_2b-13B-chat"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] V1 behavioral analysis | strength=$N | host=$HOSTNAME"

python 5_behavior_analysis.py --version v1 --strength $N

echo "[$(date)] V1 behavioral analysis finished | strength=$N"