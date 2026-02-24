#!/bin/bash
#SBATCH --job-name=deg_probe_corr
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_2/logs/deg_probe_corr_%j.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_2/logs/deg_probe_corr_%j.err

# ----------------------------------------------------------------
# Degradation-Probe Correlation: Extract per-conversation probe
# confidence and text degradation metrics across all 5 turns.
# Usage: VERSION=labels sbatch 1d_degradation_probe_correlation.sh
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

PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_2"
mkdir -p "$PROJECT_ROOT/logs/$VERSION"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] Degradation-probe correlation | version=$VERSION | host=$HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python code/pipeline/1d_degradation_probe_correlation.py --version "$VERSION"

echo "[$(date)] Finished. Run 1e_analyze_degradation_results.py --version $VERSION for analysis."
