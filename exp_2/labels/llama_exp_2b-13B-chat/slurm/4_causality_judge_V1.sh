#!/bin/bash
#SBATCH --job-name=judge_V1_peak15
#SBATCH --partition=all
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_2/labels/llama_exp_2b-13B-chat/logs/causality_judge_v1/judge_V1_%j.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_2/labels/llama_exp_2b-13B-chat/logs/causality_judge_v1/judge_V1_%j.err
# ---------------------------------------------------------------------------
# V1 Judge: peak_15 strategy, N=4, control_probes + reading_probes_peak only.
# No array needed — just two directories to judge.
#
# Submit:
#   sbatch 3b_causality_judge_V1.sh
# ---------------------------------------------------------------------------

export PS1=${PS1:-}
set -euo pipefail

# === Activate environment ===
module load pyger
export PYTHONNOUSERSITE=1
set +u
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate behavior_env
set -u
trap 'set +u; conda deactivate >/dev/null 2>&1 || true; set -u' EXIT

# === OpenAI API key ===
if [ -f "$HOME/.openai_key" ]; then
    export OPENAI_API_KEY="$(cat "$HOME/.openai_key")"
elif [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[ERROR] OPENAI_API_KEY not set"; exit 1
fi

# === Project paths ===
PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_2/labels/llama_exp_2b-13B-chat"
PY_SCRIPT="$PROJECT_ROOT/4_causality_judge.py"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

STRATEGY="peak_15"
N=4
RESULT_BASE="$PROJECT_ROOT/data/intervention_results/V1/${STRATEGY}"

echo "[$(date)] V1 judge | strategy=$STRATEGY | N=$N | host=$HOSTNAME"

# === Judge control_probes and reading_probes_peak at peak_15/is_4 ===
FOUND=0
for PROBE_TYPE in control_probes reading_probes_peak; do
    PROBE_DIR="${RESULT_BASE}/${PROBE_TYPE}/is_${N}"
    if [ -d "$PROBE_DIR" ]; then
        echo "[$(date)] Judging ${STRATEGY}/${PROBE_TYPE}/is_${N} ..."
        python "$PY_SCRIPT" \
            --version V1 \
            --layer_strategy "$STRATEGY" \
            --result_dir "$PROBE_DIR"
        FOUND=$((FOUND + 1))
    else
        echo "[WARNING] Directory not found: $PROBE_DIR"
    fi
done

if [ "$FOUND" -eq 0 ]; then
    echo "[ERROR] No probe directories found for ${STRATEGY}/*/is_${N}"
    exit 1
fi

echo "[$(date)] V1 judge finished | strategy=$STRATEGY | N=$N | judged $FOUND probe configs"