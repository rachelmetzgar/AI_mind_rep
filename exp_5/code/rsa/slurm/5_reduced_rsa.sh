#!/bin/bash
#SBATCH --job-name=exp5_reduced_rsa
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/rsa/reduced_4cond_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/rsa/reduced_4cond_%j.err

# Reduced 4-condition RSA: drops C5/C6, keeps A,B,C,D,E,F only.
# All 3 analyses (simple, partial A+E, category). ~4-8 hrs for 41 layers × 10K perms.

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
ANALYSIS=${ANALYSIS:-all}
echo "[$(date)] Starting reduced 4-cond RSA — model=$MODEL analysis=$ANALYSIS host=$HOSTNAME"

python code/rsa/5_reduced_rsa.py --model "$MODEL" --analysis "$ANALYSIS" --resume

echo "[$(date)] Done"
