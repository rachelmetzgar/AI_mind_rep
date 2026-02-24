#!/bin/bash
#SBATCH --job-name=behav_V1
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --array=0-1
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_2/logs/behavior_V1_%A_%a.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_2/logs/behavior_V1_%A_%a.err

# ---------------------------------------------------------------------------
# V1 Behavioral analysis: peak_15 strategy only.
# Array: 0->4, 1->5
# Usage: VERSION=labels sbatch 4a_behavior_analysis.sh
# ---------------------------------------------------------------------------

VERSION=${VERSION:?"ERROR: VERSION env var required (e.g. labels)"}

STRENGTHS=(4 5)
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

PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_2"
mkdir -p "$PROJECT_ROOT/logs/$VERSION"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] V1 behavioral analysis | version=$VERSION | strength=$N | host=$HOSTNAME"

python code/pipeline/4_behavior_analysis.py \
    --version "$VERSION" \
    --mode v1 \
    --strength "$N"

echo "[$(date)] V1 behavioral analysis finished | version=$VERSION | strength=$N"
