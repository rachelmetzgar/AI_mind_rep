#!/bin/bash
#SBATCH --job-name=probe_turn_cmp
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=/jukebox/graziano/rachel/mind_rep/exp_2/logs/probe_turn_cmp_%a_%A.out
#SBATCH --error=/jukebox/graziano/rachel/mind_rep/exp_2/logs/probe_turn_cmp_%a_%A.err

# ----------------------------------------------------------------
# PROBE TURN COMPARISON — Turns 1-4 only
# Turn 5 probes already exist at data/{VERSION}/probe_checkpoints/turn_5/
# Usage: VERSION=labels sbatch 1b_train_probes_turn_comparison.sh
# ----------------------------------------------------------------

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

TURN_INDICES=(0 1 2 3)
TURN_INDEX=${TURN_INDICES[$SLURM_ARRAY_TASK_ID]}

PROJECT_ROOT="/jukebox/graziano/rachel/mind_rep/exp_2"
mkdir -p "$PROJECT_ROOT/logs/$VERSION"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] Probe training: version=$VERSION turn_index=$TURN_INDEX host=$HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python code/pipeline/1b_train_probes_turn_comparison.py --version "$VERSION" --turn_index "$TURN_INDEX"

echo "[$(date)] Finished: version=$VERSION turn_index=$TURN_INDEX"
