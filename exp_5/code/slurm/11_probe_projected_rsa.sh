#!/bin/bash
#SBATCH --job-name=exp5_proj_rsa
#SBATCH --partition=all
#SBATCH --mem=16G --time=2:00:00
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/probe_training/projected_rsa/projected_rsa_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/probe_training/projected_rsa/projected_rsa_%j.err

# Probe-projected RSA

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
mkdir -p "$PROJECT_ROOT/logs/probe_training/projected_rsa"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

MODEL=${MODEL:-llama2_13b_chat}

echo "[$(date)] Starting Probe-projected RSA"
echo "  model=$MODEL"
echo "  host=$HOSTNAME"

python code/11_probe_projected_rsa.py --model "$MODEL"

echo "[$(date)] Done"
