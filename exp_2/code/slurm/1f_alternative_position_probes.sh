#!/bin/bash
#SBATCH --job-name=alt_pos_probe
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=/jukebox/graziano/rachel/mind_rep/exp_2/logs/alt_pos_probes_%a_%A.out
#SBATCH --error=/jukebox/graziano/rachel/mind_rep/exp_2/logs/alt_pos_probes_%a_%A.err

# ----------------------------------------------------------------
# ALTERNATIVE POSITION PROBES
# Array 0: control_first       (probe at BOS <s> token, position 0)
# Array 1: control_random      (probe at random mid-sequence token)
# Array 2: control_eos         (probe at </s> ending first exchange)
# Array 3: reading_irrelevant  (probe with "I think the weather...")
#
# Baselines (control_last, reading_real) already exist in
# data/{VERSION}/probe_checkpoints/turn_5/
#
# Usage: VERSION=labels sbatch 1f_alternative_position_probes.sh
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

CONDITIONS=(control_first control_random control_eos reading_irrelevant)
CONDITION=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}

PROJECT_ROOT="/jukebox/graziano/rachel/mind_rep/exp_2"
mkdir -p "$PROJECT_ROOT/logs/$VERSION"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] Alt position probe: version=$VERSION condition=$CONDITION host=$HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python code/pipeline/1f_alternative_position_probes.py --version "$VERSION" --condition "$CONDITION"

echo "[$(date)] Finished: version=$VERSION condition=$CONDITION"
