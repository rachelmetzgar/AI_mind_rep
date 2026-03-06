#!/bin/bash
#SBATCH --job-name=exp1_analysis
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4

# Usage:
#   VERSION=balanced_gpt MODEL=llama2_13b_chat sbatch run_analysis.sh
#
# Runs the full analysis pipeline:
#   1a. Combine per-subject CSVs
#   1.  Extract linguistic features
#   2.  Identity breakdown analysis

export PS1=${PS1:-}
set -euo pipefail
module load pyger
export PYTHONNOUSERSITE=1
set +u
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate behavior_env
set -u
trap 'set +u; conda deactivate >/dev/null 2>&1 || true; set -u' EXIT

VERSION=${VERSION:?'VERSION must be set (e.g., balanced_gpt)'}
MODEL=${MODEL:-llama2_13b_chat}

EXP1_DIR="/mnt/cup/labs/graziano/rachel/mind_rep/exp_1"
LOG_DIR="${EXP1_DIR}/logs/${MODEL}/${VERSION}"
mkdir -p "$LOG_DIR"
exec > "${LOG_DIR}/analysis_${SLURM_JOB_ID}.out" 2> "${LOG_DIR}/analysis_${SLURM_JOB_ID}.err"

echo "=== Exp 1 Analysis Pipeline ==="
echo "Version : $VERSION"
echo "Model   : $MODEL"
echo "Node    : $(hostname)"
echo "Started : $(date)"

cd "$EXP1_DIR"

echo ""
echo "--- Step 1a: Combine text data ---"
python code/data_gen/1a_combine_text_data.py --version "$VERSION" --model "$MODEL"

echo ""
echo "--- Step 1: Extract features ---"
python code/analysis/1_extract_features.py --version "$VERSION" --model "$MODEL"

echo ""
echo "--- Step 2: Identity breakdown ---"
python code/analysis/2_identity_breakdown.py --version "$VERSION" --model "$MODEL"

echo ""
echo "=== Analysis pipeline complete: $(date) ==="
