#!/bin/bash
#SBATCH --job-name=weather_inv
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1

# ----------------------------------------------------------------
# WEATHER SUFFIX INVESTIGATION
# Runs C1 (cosine similarity analysis) + C3 (accuracy comparison)
# and generates the investigation report.
#
# Usage: VERSION=labels sbatch weather_suffix_investigation.sh
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

EXP2_DIR="/mnt/cup/labs/graziano/rachel/mind_rep/exp_2"
LOG_DIR="${EXP2_DIR}/logs/${MODEL}/${VERSION}"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/weather_inv_${SLURM_JOB_ID}.out" 2> "${LOG_DIR}/weather_inv_${SLURM_JOB_ID}.err"
cd "$EXP2_DIR" || { echo "FATAL: Cannot cd to $EXP2_DIR"; exit 1; }

echo "[$(date)] Weather suffix investigation: version=$VERSION host=$HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python code/1f_investigate_weather_suffix.py \
    --run-c1 \
    --version "$VERSION" \
    --n-samples 50

echo "[$(date)] Finished weather suffix investigation"
