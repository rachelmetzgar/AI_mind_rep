#!/bin/bash
#SBATCH --job-name=exp3_conv_acts
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_3/logs/conv_activations_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_3/logs/conv_activations_%j.err
# ---------------------------------------------------------------------------
# Experiment 3, Phase 9a: Extract conversation activations
#
# Submit:
#   VERSION=balanced_gpt sbatch exp_3/code/slurm/9a_extract_conversation_activations.sh
#   VERSION=balanced_gpt TURN=3 sbatch exp_3/code/slurm/9a_extract_conversation_activations.sh
# ---------------------------------------------------------------------------

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

VERSION=${VERSION:?ERROR: VERSION is required. Use VERSION=balanced_gpt sbatch ...}
TURN=${TURN:-5}
PROJECT_ROOT="/mnt/cup/labs/graziano/rachel/mind_rep/exp_3"
mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] Extracting conversation activations"
echo "  version=$VERSION turn=$TURN"
echo "  host=$HOSTNAME"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python code/9a_extract_conversation_activations.py \
    --version "$VERSION" \
    --turn "$TURN"

echo "[$(date)] Done"
