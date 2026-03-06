#!/bin/bash
#SBATCH --job-name=causal_V2
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-99

# ---------------------------------------------------------------------------
# V2: Multi-turn Exp1 recreation — peak_15 strategy, N=4 and N=5.
# 2D array: 50 subjects x 2 strengths = 100 jobs (array 0-99)
#   subject_idx = TASK_ID / 2    (0-49)
#   strength    = STRENGTHS[TASK_ID % 2]   (4, 5)
#
# Usage: VERSION=labels sbatch 2b_causality_generate.sh
# ---------------------------------------------------------------------------

VERSION=${VERSION:?"ERROR: VERSION env var required (e.g. labels)"}
MODEL=${MODEL:-llama2_13b_chat}

STRENGTHS=(4 5)
N_STRENGTHS=${#STRENGTHS[@]}

SUBJECT_IDX=$(( SLURM_ARRAY_TASK_ID / N_STRENGTHS ))
STRENGTH_IDX=$(( SLURM_ARRAY_TASK_ID % N_STRENGTHS ))
N=${STRENGTHS[$STRENGTH_IDX]}

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
LOG_DIR="${EXP2_DIR}/logs/${MODEL}/${VERSION}/V2_causality"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/causality_generate_v2_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.out" 2> "${LOG_DIR}/causality_generate_v2_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}.err"
cd "$EXP2_DIR" || { echo "FATAL: Cannot cd to $EXP2_DIR"; exit 1; }

SUBJECT_ID=$(printf "s%03d" $((SUBJECT_IDX + 1)))

echo "[$(date)] V2 generation | version=$VERSION | subject=$SUBJECT_ID | strength=$N | host=$HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python code/2_causality_generate.py --version "$VERSION" --mode V2 --subject_idx $SUBJECT_IDX --strength $N --layer_strategy peak_15 --model "$MODEL"

echo "[$(date)] V2 generation finished | version=$VERSION | subject=$SUBJECT_ID | strength=$N"
