#!/bin/bash
#SBATCH --job-name=judge_V1
#SBATCH --partition=all
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --array=0-3
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_2b/llama_exp_2b-13B-chat/logs/judge_V1_%A_%a.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_2b/llama_exp_2b-13B-chat/logs/judge_V1_%A_%a.err

# ---------------------------------------------------------------------------
# V1 Judge: parallelized by intervention strength.
# Array index maps to strength: 0→2, 1→4, 2→8, 3→16
# Each job judges BOTH probe types at one strength.
# No GPU needed — just OpenAI API calls.
#
# Submit after V1 generation completes:
#   sbatch 3b_causality_judge_V1.sh
# ---------------------------------------------------------------------------

STRENGTHS=(2 4 8 16)
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

echo "[$(date)] V1 judge | strength=$N | host=$HOSTNAME"

# Judge both probe types at this strength
for PROBE_TYPE in control_probes reading_probes; do
    RESULT_DIR="$PROJECT_ROOT/data/intervention_results/V1/${PROBE_TYPE}/is_${N}"
    if [ -d "$RESULT_DIR" ]; then
        echo "[$(date)] Judging $PROBE_TYPE/is_$N ..."
        python "$PY_SCRIPT" --version V1 --result_dir "$RESULT_DIR"
    else
        echo "[SKIP] $RESULT_DIR does not exist"
    fi
done

echo "[$(date)] V1 judge finished | strength=$N"