#!/bin/bash
#SBATCH --job-name=behav_V1
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --array=0-1

# ---------------------------------------------------------------------------
# V1 Behavioral analysis: peak_15 strategy only.
# Array: 0->4, 1->5
# Usage: VERSION=labels sbatch 4a_behavior_analysis.sh
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
LOG_DIR="${EXP2_DIR}/logs/${MODEL}/${VERSION}/V1_causality"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/behavior_v1_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.out" 2> "${LOG_DIR}/behavior_v1_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.err"
cd "$EXP2_DIR" || { echo "FATAL: Cannot cd to $EXP2_DIR"; exit 1; }

echo "[$(date)] V1 behavioral analysis | version=$VERSION | strength=$N | host=$HOSTNAME"

STRATEGY="${STRATEGY:-peak_15}"

python code/4_behavior_analysis.py \
    --version "$VERSION" \
    --mode v1 \
    --layer_strategy "$STRATEGY" \
    --strength "$N" \
    --model "$MODEL"

echo "[$(date)] V1 behavioral analysis finished | version=$VERSION | strength=$N"
