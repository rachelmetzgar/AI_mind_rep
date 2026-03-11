#!/bin/bash
#SBATCH --job-name=exp5_partial_rsa
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/rsa/partial_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/rsa/partial_%j.err

# Analysis 2: Partial RSA (Model A + Model E). Heaviest analysis.
# ~1-3 hrs for 41 layers x 10K perms x 2 analyses.

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
ANALYSIS=${ANALYSIS:-both}
echo "[$(date)] Starting partial RSA — model=$MODEL analysis=$ANALYSIS host=$HOSTNAME"

python code/rsa/3_partial_rsa.py --model "$MODEL" --analysis "$ANALYSIS" --resume

echo "[$(date)] Done"
