#!/bin/bash
#SBATCH --job-name=exp5_multipos_gemma2_9b
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/activations/multipos/multipos_gemma2_9b_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_5/logs/activations/multipos/multipos_gemma2_9b_%j.err

# Extract multi-position activations for gemma2_9b

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

mkdir -p "$PROJECT_ROOT/logs/activations/multipos"

echo "[$(date)] Starting multi-position activation extraction"
echo "  model=gemma2_9b"
echo "  host=$HOSTNAME"
echo "  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

python code/probes/1_extract_multipos_activations.py --model gemma2_9b

echo "[$(date)] Done"
