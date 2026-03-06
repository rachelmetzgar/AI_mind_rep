#!/bin/bash
#SBATCH --job-name=exp1_datagen
#SBATCH --array=0-49
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4

# Usage:
#   VERSION=balanced_gpt MODEL=llama2_13b_chat sbatch 1_generate_conversations.sh

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

VERSION=${VERSION:?'VERSION must be set (e.g., balanced_gpt)'}
MODEL=${MODEL:-llama2_13b_chat}

EXP1_DIR="/mnt/cup/labs/graziano/rachel/mind_rep/exp_1"
LOG_DIR="${EXP1_DIR}/logs/${MODEL}/${VERSION}"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/data_gen_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2> "${LOG_DIR}/data_gen_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

echo "=== Exp 1 Data Generation ==="
echo "Version : $VERSION"
echo "Model   : $MODEL"
echo "Subject : ${SLURM_ARRAY_TASK_ID}"
echo "Node    : $(hostname)"
echo "GPU     : $CUDA_VISIBLE_DEVICES"
echo "Started : $(date)"

cd "$EXP1_DIR"

python code/data_gen/1_generate_conversations.py \
    --version "$VERSION" \
    --model "$MODEL" \
    --subject "${SLURM_ARRAY_TASK_ID}"

echo "=== Done: $(date) ==="
