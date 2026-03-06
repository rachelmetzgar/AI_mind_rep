#!/bin/bash
#SBATCH --job-name=alt_new_sfx
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-14

# ----------------------------------------------------------------
# ADDITIONAL IRRELEVANT SUFFIX PROBES — ALL 5 TURNS
# Array index encodes (turn × condition):
#   index = turn_idx * 3 + condition_idx
#   turn_idx:  0-4 → turns 1-5 (turn_index 0-3, -1 for turn 5)
#   condition_idx: 0=reading_sky, 1=reading_nextword, 2=reading_short
#
# Usage: VERSION=labels sbatch 1f_alt_position_new_suffixes.sh
# ----------------------------------------------------------------

VERSION=${VERSION:?"ERROR: VERSION env var required (e.g. labels)"}
MODEL=${MODEL:-llama2_13b_chat}

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

CONDITIONS=(reading_sky reading_nextword reading_short)
TURN_INDICES=(0 1 2 3 -1)

TURN_IDX_POS=$(( SLURM_ARRAY_TASK_ID / 3 ))
COND_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))

TURN_INDEX=${TURN_INDICES[$TURN_IDX_POS]}
CONDITION=${CONDITIONS[$COND_IDX]}
TURN_LABEL=$(( TURN_IDX_POS + 1 ))

EXP2_DIR="/mnt/cup/labs/graziano/rachel/mind_rep/exp_2"
LOG_DIR="${EXP2_DIR}/logs/${MODEL}/${VERSION}"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/alt_new_sfx_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.out" 2> "${LOG_DIR}/alt_new_sfx_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.err"
cd "$EXP2_DIR" || { echo "FATAL: Cannot cd to $EXP2_DIR"; exit 1; }

echo "[$(date)] New suffix probe: version=$VERSION condition=$CONDITION turn=$TURN_LABEL (index=$TURN_INDEX) host=$HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python code/1f_alternative_position_probes.py \
    --version "$VERSION" \
    --condition "$CONDITION" \
    --turn_index "$TURN_INDEX" \
    --model "$MODEL"

echo "[$(date)] Finished: version=$VERSION condition=$CONDITION turn=$TURN_LABEL"
