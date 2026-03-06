#!/bin/bash
#SBATCH --job-name=alt_pos_probe
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-3

# ----------------------------------------------------------------
# ALTERNATIVE POSITION PROBES
# Array 0: control_first       (probe at BOS <s> token, position 0)
# Array 1: control_random      (probe at random mid-sequence token)
# Array 2: control_eos         (probe at </s> ending first exchange)
# Array 3: reading_irrelevant  (probe with "I think the weather...")
#
# Baselines (control_last, reading_real) already exist in
# data/{VERSION}/probe_checkpoints/turn_5/
#
# Usage: VERSION=labels sbatch 1f_alternative_position_probes.sh
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

CONDITIONS=(control_first control_random control_eos reading_irrelevant)
CONDITION=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}

EXP2_DIR="/mnt/cup/labs/graziano/rachel/mind_rep/exp_2"
LOG_DIR="${EXP2_DIR}/logs/${MODEL}/${VERSION}/probe_training"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/alt_pos_probes_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.out" 2> "${LOG_DIR}/alt_pos_probes_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.err"
cd "$EXP2_DIR" || { echo "FATAL: Cannot cd to $EXP2_DIR"; exit 1; }

echo "[$(date)] Alt position probe: version=$VERSION condition=$CONDITION host=$HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python code/1f_alternative_position_probes.py --version "$VERSION" --condition "$CONDITION" --model "$MODEL"

echo "[$(date)] Finished: version=$VERSION condition=$CONDITION"
