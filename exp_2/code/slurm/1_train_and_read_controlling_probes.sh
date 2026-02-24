#!/bin/bash
#SBATCH --job-name=train_probes
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_2/logs/%x_%A.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_2/logs/%x_%A.err

# -------------------------------------------------------------
# TRAIN PROBES — Human vs AI
# VERSION must be set: labels, balanced_names, balanced_gpt, names,
#                      nonsense_codeword, nonsense_ignore
# Usage: VERSION=labels sbatch 1_train_and_read_controlling_probes.sh
# -------------------------------------------------------------

VERSION=${VERSION:?"ERROR: VERSION env var required (e.g. labels)"}

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

PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_2"
mkdir -p "$PROJECT_ROOT/logs/$VERSION"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] Starting probe training | version=$VERSION | host=$HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python code/pipeline/1_train_and_read_controlling_probes.py --version "$VERSION"

echo "[$(date)] Finished probe training | version=$VERSION"
