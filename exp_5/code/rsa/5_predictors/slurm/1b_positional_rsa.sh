#!/bin/bash
#SBATCH --job-name=exp5_5pred_positional
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/rsa/5_predictors_positional_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/rsa/5_predictors_positional_%j.err

# Positional 5-predictor RSA: verb + object token positions, C1-C4 only.

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
mkdir -p "$PROJECT_ROOT/logs/rsa"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd"; exit 1; }

MODEL=${MODEL:-llama2_13b_chat}
echo "[$(date)] Starting Positional 5-predictor RSA — model=$MODEL host=$HOSTNAME"

python code/rsa/5_predictors/1b_positional_rsa.py --model "$MODEL" --resume

echo "[$(date)] Done"
