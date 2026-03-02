#!/bin/bash
#SBATCH --job-name=causal_V1
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-3
#SBATCH --output=/jukebox/graziano/rachel/mind_rep/exp_2/logs/causality_V1_%A_%a.out
#SBATCH --error=/jukebox/graziano/rachel/mind_rep/exp_2/logs/causality_V1_%A_%a.err

# ---------------------------------------------------------------------------
# V1: Single-prompt causality generation.
# Array index maps to intervention strength: 0->2, 1->4, 2->8, 3->16
# Each job runs BOTH probe types (control + reading) at one strength.
#
# Usage: VERSION=labels sbatch 2a_causality_generate.sh
# ---------------------------------------------------------------------------

VERSION=${VERSION:?"ERROR: VERSION env var required (e.g. labels)"}

STRENGTHS=(2 4 8 16)
N=${STRENGTHS[$SLURM_ARRAY_TASK_ID]}

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

PROJECT_ROOT="/jukebox/graziano/rachel/mind_rep/exp_2"
mkdir -p "$PROJECT_ROOT/logs/$VERSION"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] V1 generation | version=$VERSION | strength=$N | host=$HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python code/pipeline/2_causality_generate.py --version "$VERSION" --mode V1 --strength $N --layer_strategy peak_15

echo "[$(date)] V1 generation finished | version=$VERSION | strength=$N"
