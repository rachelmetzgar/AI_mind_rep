#!/bin/bash
#SBATCH --job-name=exp5_pos_qwen25_7b_instruct
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/rsa/positional_qwen25_7b_instruct_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/rsa/positional_qwen25_7b_instruct_%j.err

# Positional 5-predictor RSA (verb + object) for qwen25_7b_instruct

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

echo "[$(date)] Starting positional RSA — model=qwen25_7b_instruct host=$HOSTNAME"

python code/rsa/5_predictors/1b_positional_rsa.py --model qwen25_7b_instruct

echo "[$(date)] Done"
