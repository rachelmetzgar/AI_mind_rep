#!/bin/bash
#SBATCH --job-name=judge_V2
#SBATCH --partition=all
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --array=0-99
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_2b/llama_exp_2b-13B-chat/logs/causality_judge_v2/judge_V2_%A_%a.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_2b/llama_exp_2b-13B-chat/causality_judge_v2/logs/judge_V2_%A_%a.err

# ---------------------------------------------------------------------------
# V2 Judge: parallelized by subject × strength.
# 2D array: 50 subjects × 4 strengths = 200 jobs (array 0-199)
#   subject_idx = TASK_ID / 4    (0-49)  → s001-s050
#   strength    = STRENGTHS[TASK_ID % 4] (2, 4, 8, 16)
# Each job judges BOTH probe types for one subject at one strength.
# No GPU needed.
#
# Adjust --array for fewer subjects (e.g., 20 subjects: --array=0-79)
#
# Submit after V2 generation completes:
#   sbatch 3b_causality_judge_V2.sh
# ---------------------------------------------------------------------------

#STRENGTHS=(2 4 8 16)
STRENGTHS=(4 6)
N_STRENGTHS=${#STRENGTHS[@]}

SUBJECT_IDX=$(( SLURM_ARRAY_TASK_ID / N_STRENGTHS ))
STRENGTH_IDX=$(( SLURM_ARRAY_TASK_ID % N_STRENGTHS ))
N=${STRENGTHS[$STRENGTH_IDX]}
SUBJECT_ID=$(printf "s%03d" $((SUBJECT_IDX + 1)))

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

# === OpenAI API key ===
if [ -f "$HOME/.openai_key" ]; then
    export OPENAI_API_KEY="$(cat "$HOME/.openai_key")"
elif [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[ERROR] OPENAI_API_KEY not set"; exit 1
fi

# === Project paths ===
PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_2b/llama_exp_2b-13B-chat"
PY_SCRIPT="$PROJECT_ROOT/4_causality_judge.py"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] V2 judge | subject=$SUBJECT_ID | strength=$N | host=$HOSTNAME"

# Judge both probe types for this subject × strength
for PROBE_TYPE in control_probes reading_probes; do
    RESULT_DIR="$PROJECT_ROOT/data/intervention_results/V2/${PROBE_TYPE}/is_${N}"
    if [ -d "$RESULT_DIR/per_subject" ] && [ -f "$RESULT_DIR/per_subject/${SUBJECT_ID}.csv" ]; then
        echo "[$(date)] Judging $PROBE_TYPE/is_$N/$SUBJECT_ID ..."
        python "$PY_SCRIPT" --version V2 --result_dir "$RESULT_DIR" --subject "$SUBJECT_ID"
    else
        echo "[SKIP] $RESULT_DIR/per_subject/${SUBJECT_ID}.csv does not exist"
    fi
done

echo "[$(date)] V2 judge finished | subject=$SUBJECT_ID | strength=$N"