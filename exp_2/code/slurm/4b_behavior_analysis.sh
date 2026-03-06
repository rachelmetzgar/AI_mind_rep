#!/bin/bash
#SBATCH --job-name=behav_V2
#SBATCH --partition=all
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-1

# ---------------------------------------------------------------------------
# V2 Behavioral analysis.
# Usage: VERSION=labels sbatch 4b_behavior_analysis.sh
# ---------------------------------------------------------------------------

VERSION=${VERSION:?"ERROR: VERSION env var required (e.g. labels)"}
MODEL=${MODEL:-llama2_13b_chat}

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

EXP2_DIR="/mnt/cup/labs/graziano/rachel/mind_rep/exp_2"
LOG_DIR="${EXP2_DIR}/logs/${MODEL}/${VERSION}"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/behavior_v2_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.out" 2> "${LOG_DIR}/behavior_v2_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.err"
cd "$EXP2_DIR" || { echo "FATAL: Cannot cd to $EXP2_DIR"; exit 1; }

echo "[$(date)] V2 behavioral analysis | version=$VERSION | strength=$N | host=$HOSTNAME"

STRATEGY="${STRATEGY:-peak_15}"

python code/4_behavior_analysis.py \
    --version "$VERSION" \
    --mode v2 \
    --layer_strategy "$STRATEGY" \
    --strength "$N" \
    --model "$MODEL"

echo "[$(date)] V2 behavioral analysis finished | version=$VERSION | strength=$N"
