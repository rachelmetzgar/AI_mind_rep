#!/bin/bash
#SBATCH --job-name=exp5_resid_probes
#SBATCH --partition=all
#SBATCH --mem=32G --time=24:00:00
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/probe_training/residual/residual_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/probe_training/residual/residual_%j.err

# Residual activation probes (OLS deconfound + logistic probe)

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
mkdir -p "$PROJECT_ROOT/logs/probe_training/residual"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

MODEL=${MODEL:-llama2_13b_chat}

echo "[$(date)] Starting Residual activation probes"
echo "  model=$MODEL"
echo "  host=$HOSTNAME"

python code/probes/8_residual_probes.py --model "$MODEL"

echo "[$(date)] Done"
