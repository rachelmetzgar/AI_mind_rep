#!/bin/bash
#SBATCH --job-name=exp5_5pred_llama2_13b_base
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/rsa/5_predictors_llama2_13b_base_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/rsa/5_predictors_llama2_13b_base_%j.err

# Reduced 1-4 RSA: 5-predictor regression for llama2_13b_base

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

PROJECT_ROOT="/mnt/cup/labs/graziano/rachel/mind_rep/exp_5"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

mkdir -p "$PROJECT_ROOT/logs/rsa"

echo "[$(date)] Starting 5-predictor RSA — model=llama2_13b_base host=$HOSTNAME"

python code/rsa/5_predictors/1_reduced_1_4_rsa.py --model llama2_13b_base --resume

echo "[$(date)] Done"
