#!/bin/bash
#SBATCH --job-name=probe_turn_cmp
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-3

# ----------------------------------------------------------------
# PROBE TURN COMPARISON — Turns 1-4 only
# Turn 5 probes already exist at data/{VERSION}/probe_checkpoints/turn_5/
# Usage: VERSION=labels sbatch 1b_train_probes_turn_comparison.sh
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

TURN_INDICES=(0 1 2 3)
TURN_INDEX=${TURN_INDICES[$SLURM_ARRAY_TASK_ID]}

EXP2_DIR="/mnt/cup/labs/graziano/rachel/mind_rep/exp_2"
LOG_DIR="${EXP2_DIR}/logs/${MODEL}/${VERSION}"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/probe_turn_cmp_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.out" 2> "${LOG_DIR}/probe_turn_cmp_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.err"
cd "$EXP2_DIR" || { echo "FATAL: Cannot cd to $EXP2_DIR"; exit 1; }

echo "[$(date)] Probe training: version=$VERSION turn_index=$TURN_INDEX host=$HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python code/1b_train_probes_turn_comparison.py --version "$VERSION" --turn_index "$TURN_INDEX" --model "$MODEL"

echo "[$(date)] Finished: version=$VERSION turn_index=$TURN_INDEX"
