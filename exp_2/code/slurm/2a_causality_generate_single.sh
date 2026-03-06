#!/bin/bash
#SBATCH --job-name=causal_V1_s
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# ---------------------------------------------------------------------------
# V1: Single-strength causality generation.
# Usage: VERSION=nonsense_codeword STRENGTH=5 sbatch 2a_causality_generate_single.sh
# ---------------------------------------------------------------------------

VERSION=${VERSION:?"ERROR: VERSION env var required (e.g. nonsense_codeword)"}
STRENGTH=${STRENGTH:?"ERROR: STRENGTH env var required (e.g. 5)"}
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
LOG_DIR="${EXP2_DIR}/logs/${MODEL}/${VERSION}/V1_causality"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/causality_generate_v1_single_${SLURM_JOB_ID}.out" 2> "${LOG_DIR}/causality_generate_v1_single_${SLURM_JOB_ID}.err"
cd "$EXP2_DIR" || { echo "FATAL: Cannot cd to $EXP2_DIR"; exit 1; }

echo "[$(date)] V1 generation | version=$VERSION | strength=$STRENGTH | host=$HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python code/2_causality_generate.py --version "$VERSION" --mode V1 --strength $STRENGTH --layer_strategy peak_15 --model "$MODEL"

echo "[$(date)] V1 generation finished | version=$VERSION | strength=$STRENGTH"
