#!/bin/bash
#SBATCH --job-name=causal_V2
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-49
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_2b/llama_exp_2b-13B-chat/logs/causality_V2_%A_%a.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_2b/llama_exp_2b-13B-chat/logs/causality_V2_%A_%a.err

# ---------------------------------------------------------------------------
# V2: Multi-turn Exp1 recreation.
# 2D array: 50 subjects × 4 strengths = 200 jobs (array 0-199)
#   subject_idx = TASK_ID / 4    (0-49)
#   strength    = STRENGTHS[TASK_ID % 4]   (2, 4, 8, 16)
# Each job runs BOTH probe types (control + reading) at one subject × strength.
#
# Adjust --array range if fewer subjects:
#   20 subjects: --array=0-79
#   10 subjects: --array=0-39
#
# Submit:  sbatch 3b_causality_generate_V2.sh
# ---------------------------------------------------------------------------

#STRENGTHS=(2 4 8 16)
STRENGTHS=(4)
N_STRENGTHS=${#STRENGTHS[@]}

SUBJECT_IDX=$(( SLURM_ARRAY_TASK_ID / N_STRENGTHS ))
STRENGTH_IDX=$(( SLURM_ARRAY_TASK_ID % N_STRENGTHS ))
N=${STRENGTHS[$STRENGTH_IDX]}

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
PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_2b/llama_exp_2b-13B-chat"
PY_SCRIPT="$PROJECT_ROOT/3_causality_generate.py"
mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

SUBJECT_ID=$(printf "s%03d" $((SUBJECT_IDX + 1)))
echo "[$(date)] V2 generation | subject=$SUBJECT_ID | strength=$N | array_idx=$SLURM_ARRAY_TASK_ID | host=$HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# === Run generation (both probe types, one subject, one strength) ===
python "$PY_SCRIPT" --version V2 --subject_idx $SUBJECT_IDX --strength $N

echo "[$(date)] V2 generation finished | subject=$SUBJECT_ID | strength=$N"