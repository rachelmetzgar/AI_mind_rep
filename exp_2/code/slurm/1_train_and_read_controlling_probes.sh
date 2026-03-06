#!/bin/bash
#SBATCH --job-name=train_probes
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

# -------------------------------------------------------------
# TRAIN PROBES — Human vs AI
# VERSION must be set: labels, balanced_names, balanced_gpt, names,
#                      nonsense_codeword, nonsense_ignore
# Usage: VERSION=labels sbatch 1_train_and_read_controlling_probes.sh
# -------------------------------------------------------------

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

EXP2_DIR="/mnt/cup/labs/graziano/rachel/mind_rep/exp_2"
LOG_DIR="${EXP2_DIR}/logs/${MODEL}/${VERSION}/probe_training"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/train_probes_${SLURM_JOB_ID}.out" 2> "${LOG_DIR}/train_probes_${SLURM_JOB_ID}.err"
cd "$EXP2_DIR" || { echo "FATAL: Cannot cd to $EXP2_DIR"; exit 1; }

echo "[$(date)] Starting probe training | version=$VERSION | host=$HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python code/1_train_probes.py --version "$VERSION" --model "$MODEL"

echo "[$(date)] Finished probe training | version=$VERSION"
