#!/bin/bash
#SBATCH --job-name=ya_bgpt_analysis
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_1/you_are_balanced_gpt/logs/analysis_pipeline_%j.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_1/you_are_balanced_gpt/logs/analysis_pipeline_%j.err

# ---------------------------------------------------------------------------
# Automated analysis pipeline for you_are_balanced_gpt experiment.
# This job should be submitted with --dependency=afterok:<data_gen_job_id>
# to run automatically after data generation completes successfully.
#
# Pipeline:
#   1. combine_text_data.py - Aggregate per-subject CSVs
#   2. identity_breakdown.py - Agent-level statistical analysis + HTML report
#
# Usage:
#   sbatch --dependency=afterok:3559400 run_pipeline.sh
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

# === Project paths ===
export PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_1/you_are_balanced_gpt"
ANALYSIS_DIR="$PROJECT_ROOT/code/analysis"
DATA_DIR="$PROJECT_ROOT/data/meta-llama-Llama-2-13b-chat-hf/0.8"
RESULTS_DIR="$PROJECT_ROOT/results/meta-llama-Llama-2-13b-chat-hf/0.8"
LOG_DIR="$PROJECT_ROOT/logs"

mkdir -p "$DATA_DIR" "$RESULTS_DIR" "$LOG_DIR"

echo "=================================================================="
echo "YOU_ARE_BALANCED_GPT ANALYSIS PIPELINE"
echo "Started: $(date)"
echo "Host: $HOSTNAME"
echo "Project root: $PROJECT_ROOT"
echo "=================================================================="

# === Step 1: Check data completeness ===
echo ""
echo "[1/3] Checking data generation completeness..."
cd "$DATA_DIR" || { echo "FATAL: Cannot cd to $DATA_DIR"; exit 1; }

EXPECTED_COUNT=50
ACTUAL_COUNT=$(ls -1 s*.csv 2>/dev/null | wc -l)

echo "  Expected subjects: $EXPECTED_COUNT"
echo "  Found CSVs: $ACTUAL_COUNT"

if [ "$ACTUAL_COUNT" -lt "$EXPECTED_COUNT" ]; then
    echo "  [WARN] Missing $(($EXPECTED_COUNT - $ACTUAL_COUNT)) subject files!"
    echo "  Continuing with available data..."
else
    echo "  [OK] All subjects present"
fi

# === Step 2: Combine text data ===
echo ""
echo "[2/3] Combining per-subject CSVs..."
cd "$ANALYSIS_DIR" || { echo "FATAL: Cannot cd to $ANALYSIS_DIR"; exit 1; }

if [ ! -f "combine_text_data.py" ]; then
    echo "  [ERROR] combine_text_data.py not found in $ANALYSIS_DIR"
    exit 1
fi

python combine_text_data.py --config "$PROJECT_ROOT/configs/behavior.json" --use_clean=False \
    2>&1 | tee "$LOG_DIR/combine_text_data_$(date +%Y%m%d_%H%M%S).log"

COMBINE_EXIT=${PIPESTATUS[0]}
if [ $COMBINE_EXIT -ne 0 ]; then
    echo "  [ERROR] combine_text_data.py failed with exit code $COMBINE_EXIT"
    exit 1
fi

# Check if combined CSV was created
if [ ! -f "$DATA_DIR/combined_text_data.csv" ]; then
    echo "  [ERROR] combined_text_data.csv not created"
    exit 1
fi

COMBINED_ROWS=$(wc -l < "$DATA_DIR/combined_text_data.csv")
echo "  [OK] Combined CSV created: $COMBINED_ROWS rows"

# === Step 3: Cross-experiment comparison ===
echo ""
echo "[3/4] Running cross-experiment comparison..."

if [ ! -f "$ANALYSIS_DIR/cross_experiment_comparison.py" ]; then
    echo "  [ERROR] cross_experiment_comparison.py not found in $ANALYSIS_DIR"
    exit 1
fi

# cross_experiment_comparison uses relative paths (data/, results/) so it
# must run from the project root, not the analysis directory.
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

python "$ANALYSIS_DIR/cross_experiment_comparison.py" --config "$PROJECT_ROOT/configs/behavior.json" \
    2>&1 | tee "$LOG_DIR/cross_experiment_$(date +%Y%m%d_%H%M%S).log"

CROSSEXP_EXIT=${PIPESTATUS[0]}
if [ $CROSSEXP_EXIT -ne 0 ]; then
    echo "  [ERROR] cross_experiment_comparison.py failed with exit code $CROSSEXP_EXIT"
    exit 1
fi
echo "  [OK] Cross-experiment comparison complete"

# Return to analysis dir for remaining steps
cd "$ANALYSIS_DIR" || { echo "FATAL: Cannot cd to $ANALYSIS_DIR"; exit 1; }

# === Step 4: Identity breakdown analysis ===
echo ""
echo "[4/4] Running identity breakdown analysis..."

if [ ! -f "identity_breakdown.py" ]; then
    echo "  [ERROR] identity_breakdown.py not found in $ANALYSIS_DIR"
    exit 1
fi

python identity_breakdown.py \
    2>&1 | tee "$LOG_DIR/identity_breakdown_$(date +%Y%m%d_%H%M%S).log"

BREAKDOWN_EXIT=${PIPESTATUS[0]}
if [ $BREAKDOWN_EXIT -ne 0 ]; then
    echo "  [ERROR] identity_breakdown.py failed with exit code $BREAKDOWN_EXIT"
    exit 1
fi

# === Final summary ===
echo ""
echo "=================================================================="
echo "PIPELINE COMPLETE"
echo "Finished: $(date)"
echo "=================================================================="
echo ""
echo "Output files:"
echo "  Data:"
echo "    $DATA_DIR/combined_text_data.csv"
echo ""
echo "  Results:"
if [ -f "$RESULTS_DIR/identity_breakdown.html" ]; then
    HTML_SIZE=$(du -h "$RESULTS_DIR/identity_breakdown.html" | cut -f1)
    echo "    $RESULTS_DIR/identity_breakdown.html ($HTML_SIZE)"
else
    echo "    [WARN] identity_breakdown.html not found"
fi

if [ -f "$RESULTS_DIR/identity_breakdown_stats.txt" ]; then
    STATS_SIZE=$(du -h "$RESULTS_DIR/identity_breakdown_stats.txt" | cut -f1)
    echo "    $RESULTS_DIR/identity_breakdown_stats.txt ($STATS_SIZE)"
fi

if [ -f "$RESULTS_DIR/identity_breakdown_summary.csv" ]; then
    CSV_SIZE=$(du -h "$RESULTS_DIR/identity_breakdown_summary.csv" | cut -f1)
    echo "    $RESULTS_DIR/identity_breakdown_summary.csv ($CSV_SIZE)"
fi

echo ""
echo "View HTML report:"
echo "  Open in browser: file://$RESULTS_DIR/identity_breakdown.html"
echo "=================================================================="

exit 0
