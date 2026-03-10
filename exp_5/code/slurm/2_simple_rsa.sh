#!/bin/bash
#SBATCH --job-name=exp5_simple_rsa
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/rsa/simple_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/rsa/simple_%j.err

# Analysis 1: Simple RSA (Model A). ~15-30 min for 41 layers x 10K perms.

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
echo "[$(date)] Starting simple RSA — model=$MODEL host=$HOSTNAME"

python code/2_simple_rsa.py --model "$MODEL"

echo "[$(date)] Done"
