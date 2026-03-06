#!/bin/bash
#SBATCH --job-name=causal_V1
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-3

# ---------------------------------------------------------------------------
# V1: Single-prompt causality generation.
# Array index maps to intervention strength: 0->2, 1->4, 2->8, 3->16
# Each job runs BOTH probe types (control + reading) at one strength.
#
# Usage: VERSION=labels sbatch 2a_causality_generate.sh
# ---------------------------------------------------------------------------

VERSION=${VERSION:?"ERROR: VERSION env var required (e.g. labels)"}
MODEL=${MODEL:-llama2_13b_chat}

STRENGTHS=(2 4 8 16)
N=${STRENGTHS[$SLURM_ARRAY_TASK_ID]}

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

EXP2_DIR="/mnt/cup/labs/graziano/rachel/mind_rep/exp_2"
LOG_DIR="${EXP2_DIR}/logs/${MODEL}/${VERSION}"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/causality_generate_v1_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.out" 2> "${LOG_DIR}/causality_generate_v1_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.err"
cd "$EXP2_DIR" || { echo "FATAL: Cannot cd to $EXP2_DIR"; exit 1; }

echo "[$(date)] V1 generation | version=$VERSION | strength=$N | host=$HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python code/2_causality_generate.py --version "$VERSION" --mode V1 --strength $N --layer_strategy peak_15 --model "$MODEL"

echo "[$(date)] V1 generation finished | version=$VERSION | strength=$N"
