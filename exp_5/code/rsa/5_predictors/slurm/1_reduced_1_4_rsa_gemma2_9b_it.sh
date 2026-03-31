#!/bin/bash
#SBATCH --job-name=exp5_5pred_gemma2_9b_it
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/rsa/5_predictors_gemma2_9b_it_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/rsa/5_predictors_gemma2_9b_it_%j.err

# Reduced 1-4 RSA: 5-predictor regression for gemma2_9b_it

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

echo "[$(date)] Starting 5-predictor RSA — model=gemma2_9b_it host=$HOSTNAME"

python code/rsa/5_predictors/1_reduced_1_4_rsa.py --model gemma2_9b_it --resume

echo "[$(date)] Done"
