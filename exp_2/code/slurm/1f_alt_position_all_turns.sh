#!/bin/bash
#SBATCH --job-name=alt_pos_all
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-19
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_2/logs/alt_pos_all_%a_%A.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_2/logs/alt_pos_all_%a_%A.err

# ----------------------------------------------------------------
# ALTERNATIVE POSITION PROBES — ALL 5 TURNS
# Array index encodes (turn × condition):
#   index = turn_idx * 4 + condition_idx
#   turn_idx:  0-4 → turns 1-5 (turn_index 0-3, -1 for turn 5)
#   condition_idx: 0=control_first, 1=control_random,
#                  2=control_eos, 3=reading_irrelevant
#
# Usage: VERSION=labels sbatch 1f_alt_position_all_turns.sh
# ----------------------------------------------------------------

VERSION=${VERSION:?"ERROR: VERSION env var required (e.g. labels)"}

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

CONDITIONS=(control_first control_random control_eos reading_irrelevant)
TURN_INDICES=(0 1 2 3 -1)

TURN_IDX_POS=$(( SLURM_ARRAY_TASK_ID / 4 ))
COND_IDX=$(( SLURM_ARRAY_TASK_ID % 4 ))

TURN_INDEX=${TURN_INDICES[$TURN_IDX_POS]}
CONDITION=${CONDITIONS[$COND_IDX]}
TURN_LABEL=$(( TURN_IDX_POS + 1 ))

PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_2"
mkdir -p "$PROJECT_ROOT/logs/$VERSION"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] Alt position probe: version=$VERSION condition=$CONDITION turn=$TURN_LABEL (index=$TURN_INDEX) host=$HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python code/pipeline/1f_alternative_position_probes.py \
    --version "$VERSION" \
    --condition "$CONDITION" \
    --turn_index "$TURN_INDEX"

echo "[$(date)] Finished: version=$VERSION condition=$CONDITION turn=$TURN_LABEL"
