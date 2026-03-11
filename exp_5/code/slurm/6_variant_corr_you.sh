#!/bin/bash
#SBATCH --job-name=exp5_rsa_corr_you
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/rsa/variant_corr_you_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/rsa/variant_corr_you_%j.err

# Variant RSA: correlation distance, "You" stimuli. ~2-4 hrs for all 3 analyses.
# Depends on 5_extract_you_activations.sh completing first.

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
echo "[$(date)] Starting variant RSA: correlation / you — model=$MODEL host=$HOSTNAME"

python code/6_variant_rsa.py --model "$MODEL" --metric correlation --variant you

echo "[$(date)] Done"
