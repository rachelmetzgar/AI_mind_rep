#!/bin/bash
#SBATCH --job-name=weather_inv
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/jukebox/graziano/rachel/mind_rep/exp_2/logs/weather_inv_%j.out
#SBATCH --error=/jukebox/graziano/rachel/mind_rep/exp_2/logs/weather_inv_%j.err

# ----------------------------------------------------------------
# WEATHER SUFFIX INVESTIGATION
# Runs C1 (cosine similarity analysis) + C3 (accuracy comparison)
# and generates the investigation report.
#
# Usage: VERSION=labels sbatch weather_suffix_investigation.sh
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

PROJECT_ROOT="/jukebox/graziano/rachel/mind_rep/exp_2"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] Weather suffix investigation: version=$VERSION host=$HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python code/analysis/investigate_weather_suffix.py \
    --run-c1 \
    --version "$VERSION" \
    --n-samples 50

echo "[$(date)] Finished weather suffix investigation"
