#!/bin/bash
#SBATCH --job-name=exp5_9_predictors_rsa
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/rsa/9_predictors_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/rsa/9_predictors_%j.err

# Full 9-predictor RSA (A-I), both correlation and cosine metrics.
# ~4-8 hrs for 41 layers x 10K perms x 2 metrics.

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
METRIC=${METRIC:-both}
echo "[$(date)] Starting Full Model RSA — model=$MODEL metric=$METRIC host=$HOSTNAME"

python code/rsa/9_predictors/1_9_predictors_rsa.py --model "$MODEL" --metric "$METRIC" --resume

echo "[$(date)] Done"
