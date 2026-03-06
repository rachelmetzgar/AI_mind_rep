#!/bin/bash
#SBATCH --job-name=deg_probe_corr
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1

# ----------------------------------------------------------------
# Degradation-Probe Correlation: Extract per-conversation probe
# confidence and text degradation metrics across all 5 turns.
# Usage: VERSION=labels sbatch 1d_degradation_probe_correlation.sh
# ----------------------------------------------------------------

VERSION=${VERSION:?"ERROR: VERSION env var required (e.g. labels)"}
MODEL=${MODEL:-llama2_13b_chat}

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

EXP2_DIR="/mnt/cup/labs/graziano/rachel/mind_rep/exp_2"
LOG_DIR="${EXP2_DIR}/logs/${MODEL}/${VERSION}/probe_training"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/deg_probe_corr_${SLURM_JOB_ID}.out" 2> "${LOG_DIR}/deg_probe_corr_${SLURM_JOB_ID}.err"
cd "$EXP2_DIR" || { echo "FATAL: Cannot cd to $EXP2_DIR"; exit 1; }

echo "[$(date)] Degradation-probe correlation | version=$VERSION | host=$HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python code/1d_degradation_probe_correlation.py --version "$VERSION" --model "$MODEL"

echo "[$(date)] Finished. Run 1e_analyze_degradation_results.py --version $VERSION for analysis."
