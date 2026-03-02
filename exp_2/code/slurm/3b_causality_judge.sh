#!/bin/bash
#SBATCH --job-name=judge_V2
#SBATCH --partition=all
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --array=0-99
#SBATCH --output=/jukebox/graziano/rachel/mind_rep/exp_2/logs/judge_V2_%A_%a.out
#SBATCH --error=/jukebox/graziano/rachel/mind_rep/exp_2/logs/judge_V2_%A_%a.err

# ---------------------------------------------------------------------------
# V2 Judge: parallelized by subject x strength.
# 2D array: 50 subjects x 2 strengths = 100 jobs (array 0-99)
# Usage: VERSION=labels sbatch 3b_causality_judge.sh
# ---------------------------------------------------------------------------

VERSION=${VERSION:?"ERROR: VERSION env var required (e.g. labels)"}

STRENGTHS=(4 5)
N_STRENGTHS=${#STRENGTHS[@]}

SUBJECT_IDX=$(( SLURM_ARRAY_TASK_ID / N_STRENGTHS ))
STRENGTH_IDX=$(( SLURM_ARRAY_TASK_ID % N_STRENGTHS ))
N=${STRENGTHS[$STRENGTH_IDX]}
SUBJECT_ID=$(printf "s%03d" $((SUBJECT_IDX + 1)))

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

if [ -f "$HOME/.openai_key" ]; then
    export OPENAI_API_KEY="$(cat "$HOME/.openai_key")"
elif [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[ERROR] OPENAI_API_KEY not set"; exit 1
fi

PROJECT_ROOT="/jukebox/graziano/rachel/mind_rep/exp_2"
mkdir -p "$PROJECT_ROOT/logs/$VERSION"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] V2 judge | version=$VERSION | subject=$SUBJECT_ID | strength=$N | host=$HOSTNAME"

STRATEGY="peak_15"

for PROBE_TYPE in control_probes reading_probes_peak; do
    RESULT_DIR="$PROJECT_ROOT/data/$VERSION/intervention_results/V2/${STRATEGY}/${PROBE_TYPE}/is_${N}"
    if [ -d "$RESULT_DIR/per_subject" ] && [ -f "$RESULT_DIR/per_subject/${SUBJECT_ID}.csv" ]; then
        echo "[$(date)] Judging ${STRATEGY}/$PROBE_TYPE/is_$N/$SUBJECT_ID ..."
        python code/pipeline/3_causality_judge.py \
            --version "$VERSION" \
            --mode V2 \
            --layer_strategy "$STRATEGY" \
            --result_dir "$RESULT_DIR" \
            --subject "$SUBJECT_ID"
    else
        echo "[SKIP] $RESULT_DIR/per_subject/${SUBJECT_ID}.csv does not exist"
    fi
done

echo "[$(date)] V2 judge finished | version=$VERSION | subject=$SUBJECT_ID | strength=$N"
