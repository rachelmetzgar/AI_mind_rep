#!/bin/bash
#SBATCH --job-name=causal_V1
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-3
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_2/labels/llama_exp_2b-13B-chat/logs/causality_v1/causality_V1_%A_%a.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_2/labels/llama_exp_2b-13B-chat/logs/causality_v1/causality_V1_%A_%a.err

# ---------------------------------------------------------------------------
# V1: Single-prompt causality generation.
# Array index maps to intervention strength: 0→2, 1→4, 2→8, 3→16
# Each job runs BOTH probe types (control + reading) at one strength.
# Model loads once per job; probes are lightweight.
#
# Submit:  sbatch 3b_causality_generate_V1.sh
# ---------------------------------------------------------------------------

# Strength lookup from array index
#STRENGTHS=(2 4 8 16)
STRENGTHS=(2 4 5 6)
N=${STRENGTHS[$SLURM_ARRAY_TASK_ID]}

export PS1=${PS1:-}
set -euo pipefail

# === Activate environment ===
module load pyger
export PYTHONNOUSERSITE=1
set +u
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate llama2_env
set -u
trap 'set +u; conda deactivate >/dev/null 2>&1 || true; set -u' EXIT

# === Project paths ===
PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_2/labels/llama_exp_2b-13B-chat"
PY_SCRIPT="$PROJECT_ROOT/3_causality_generate.py"
mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] V1 generation | strength=$N | array_idx=$SLURM_ARRAY_TASK_ID | host=$HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# === Run generation (both probe types, one strength) ===
python "$PY_SCRIPT" --version V1 --strength $N --layer_strategy peak_15

echo "[$(date)] V1 generation finished | strength=$N"