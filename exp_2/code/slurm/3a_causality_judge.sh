#!/bin/bash
#SBATCH --job-name=judge_V1
#SBATCH --partition=all
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=/jukebox/graziano/rachel/mind_rep/exp_2/logs/judge_V1_%j.out
#SBATCH --error=/jukebox/graziano/rachel/mind_rep/exp_2/logs/judge_V1_%j.err

# ---------------------------------------------------------------------------
# V1 Judge: peak_15 strategy, operational + metacognitive_peak.
# Usage: VERSION=labels sbatch 3a_causality_judge.sh
# ---------------------------------------------------------------------------

VERSION=${VERSION:?"ERROR: VERSION env var required (e.g. labels)"}

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

# API keys
if [ -f "$HOME/.anthropic_key" ]; then
    export ANTHROPIC_API_KEY="$(cat "$HOME/.anthropic_key")"
fi
if [ -f "$HOME/.openai_key" ]; then
    export OPENAI_API_KEY="$(cat "$HOME/.openai_key")"
fi

# Validate based on backend
JUDGE_BACKEND="${JUDGE_BACKEND:-claude}"
if [ "$JUDGE_BACKEND" = "claude" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "[ERROR] ANTHROPIC_API_KEY not set"; exit 1
elif [ "$JUDGE_BACKEND" = "gpt" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[ERROR] OPENAI_API_KEY not set"; exit 1
fi

PROJECT_ROOT="/jukebox/graziano/rachel/mind_rep/exp_2"
mkdir -p "$PROJECT_ROOT/logs/$VERSION"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

STRATEGY="peak_15"
N=4
RESULT_BASE="$PROJECT_ROOT/data/$VERSION/intervention_results/V1/${STRATEGY}"

echo "[$(date)] V1 judge | version=$VERSION | strategy=$STRATEGY | N=$N | backend=$JUDGE_BACKEND | host=$HOSTNAME"

FOUND=0
for PROBE_TYPE in operational metacognitive_peak; do
    PROBE_DIR="${RESULT_BASE}/${PROBE_TYPE}/is_${N}"
    if [ -d "$PROBE_DIR" ]; then
        echo "[$(date)] Judging ${STRATEGY}/${PROBE_TYPE}/is_${N} ..."
        python code/pipeline/3_causality_judge.py \
            --version "$VERSION" \
            --mode V1 \
            --layer_strategy "$STRATEGY" \
            --result_dir "$PROBE_DIR" \
            --judge_backend "$JUDGE_BACKEND"
        FOUND=$((FOUND + 1))
    else
        echo "[WARNING] Directory not found: $PROBE_DIR"
    fi
done

if [ "$FOUND" -eq 0 ]; then
    echo "[ERROR] No probe directories found for ${STRATEGY}/*/is_${N}"
    exit 1
fi

echo "[$(date)] V1 judge finished | version=$VERSION | strategy=$STRATEGY | N=$N"
