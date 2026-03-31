#!/bin/bash
# =============================================================================
# Monitor new model jobs, submit dependent waves, run post-processing & reports
#
# Checks every 30 min for 5 hours, then every 60 min.
# Logs to exp_4/logs/monitor_new_models.log
#
# Usage: nohup bash monitor_new_models.sh &
# =============================================================================

set -u
export PS1=${PS1:-}

LOG="/mnt/cup/labs/graziano/rachel/mind_rep/exp_4/logs/monitor_new_models.log"
SLURM_DIR="/mnt/cup/labs/graziano/rachel/mind_rep/exp_4/code/slurm"
CODE_DIR="/mnt/cup/labs/graziano/rachel/mind_rep/exp_4/code"
RESULTS_DIR="/mnt/cup/labs/graziano/rachel/mind_rep/exp_4/results"

# Track what we've already submitted/done
WAVE2_SUBMITTED_GEM2=0
WAVE2_SUBMITTED_QW3=0
WAVE2_SUBMITTED_QW25=1   # already submitted
POSTPROC_DONE_GEM2=0
POSTPROC_DONE_QW25=0
POSTPROC_DONE_QW3=0
REPORTS_DONE=0

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

# Check if all jobs matching a pattern have completed (none pending/running)
all_done() {
    local pattern="$1"
    local running=$(squeue -u "$USER" --format="%j" --noheader | grep -c "$pattern" || true)
    echo "$running"
}

# Check if any jobs matching a pattern failed
check_failures() {
    local pattern="$1"
    sacct -u "$USER" --starttime=2026-03-28 --format="JobName%30,State%15,ExitCode" --noheader \
        | grep "$pattern" | grep -v "\.batch" | grep -v "\.extern" | grep -v "COMPLETED" | grep -v "RUNNING" | grep -v "PENDING" || true
}

# Check if specific data files exist for a model (basic sanity check)
check_data_exists() {
    local model="$1"
    local branch="$2"
    local file="$3"
    local base="$RESULTS_DIR/$model/$branch"
    find "$base" -name "$file" 2>/dev/null | head -1
}

submit_wave2() {
    local model_suffix="$1"
    local model_key="$2"
    log "Submitting Wave 2 for $model_key"

    # Individual ratings
    local f="$SLURM_DIR/gray_replication/3_individual_${model_suffix}.sh"
    if [ -f "$f" ]; then
        local jid=$(sbatch "$f" 2>&1 | awk '{print $4}')
        log "  Submitted 3_individual: job $jid"
    fi

    # Neural PCA (CPU)
    f="$SLURM_DIR/gray_simple/2_neural_pca_${model_suffix}.sh"
    if [ -f "$f" ]; then
        local jid=$(sbatch "$f" 2>&1 | awk '{print $4}')
        log "  Submitted 2_neural_pca: job $jid"
    fi

    # Names only
    f="$SLURM_DIR/human_ai_adaptation/2_gray_names_only_${model_suffix}.sh"
    if [ -f "$f" ]; then
        local jid=$(sbatch "$f" 2>&1 | awk '{print $4}')
        log "  Submitted 2_gray_names_only: job $jid"
    fi
}

run_postprocessing() {
    local model="$1"
    log "Running CPU post-processing for $model"

    cd "$CODE_DIR"

    # Setup conda
    module load pyger 2>/dev/null || true
    export PYTHONNOUSERSITE=1
    set +u
    CONDA_BASE="$(conda info --base)"
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate llama2_env
    set -u

    # Gray replication post-processing
    local pairwise_data=$(check_data_exists "$model" "gray_replication" "pairwise_pca_results.npz")
    if [ -n "$pairwise_data" ]; then
        log "  Running compute_excl_pca..."
        python gray_replication/behavior/compute_excl_pca.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: compute_excl_pca failed"

        log "  Running compute_human_comparisons..."
        python gray_replication/behavior/compute_human_comparisons.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: compute_human_comparisons failed"

        for cond in with_self without_self; do
            log "  Running make_loadings_bar_chart ($cond)..."
            python gray_replication/behavior/make_loadings_bar_chart.py --model "$model" --condition "$cond" >> "$LOG" 2>&1 || log "  WARN: make_loadings_bar_chart ($cond) failed"

            log "  Running make_condition_reports ($cond)..."
            python gray_replication/behavior/make_condition_reports.py --model "$model" --condition "$cond" >> "$LOG" 2>&1 || log "  WARN: make_condition_reports ($cond) failed"
        done
    else
        log "  SKIP: No pairwise data found for gray_replication"
    fi

    # Gray simple report
    local rsa_data=$(check_data_exists "$model" "gray_simple" "rsa_results.json")
    if [ -n "$rsa_data" ]; then
        log "  Running gray_simple RSA report..."
        python gray_simple/internals/1a_rsa_report_generator.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: RSA report failed"
    fi

    local pca_data=$(check_data_exists "$model" "gray_simple" "neural_pca_results.npz")
    if [ -n "$pca_data" ]; then
        log "  Running neural PCA report..."
        python gray_simple/internals/2a_neural_pca_report_generator.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: neural PCA report failed"
    fi

    # Human-AI adaptation reports
    local chars_data=$(check_data_exists "$model" "human_ai_adaptation" "pairwise_pca_results.npz")
    if [ -n "$chars_data" ]; then
        log "  Running human_ai_adaptation reports..."
        python human_ai_adaptation/behavior/1a_gray_chars_pca_report_generator.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: PCA report failed"
        python human_ai_adaptation/behavior/1b_gray_chars_detailed_report_generator.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: detailed report failed"
        python human_ai_adaptation/behavior/1c_gray_chars_rsa_report_generator.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: RSA report failed"
    fi

    # Expanded mental concepts reports
    local bpca_data=$(check_data_exists "$model" "expanded_mental_concepts" "pairwise_pca_results.npz")
    if [ -n "$bpca_data" ]; then
        log "  Running expanded_mental_concepts behavioral reports..."
        python expanded_mental_concepts/behavior/pca/behavioral_pca_report_generator.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: behavioral PCA report failed"
        python expanded_mental_concepts/behavior/pca/matched_behavioral_pca.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: matched behavioral PCA failed"
        python expanded_mental_concepts/behavior/pca/matched_behavioral_pca_report_generator.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: matched behavioral PCA report failed"
        python expanded_mental_concepts/behavior/pca/behavioral_attribution_report_generator.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: behavioral attribution report failed"
        python expanded_mental_concepts/behavior/pca/detailed_response_report_generator.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: detailed response report failed"
    fi

    local arsa_data=$(check_data_exists "$model" "expanded_mental_concepts" "rsa_results.json")
    if [ -n "$arsa_data" ]; then
        log "  Running expanded_mental_concepts internals reports..."
        python expanded_mental_concepts/internals/rsa/activation_rsa_report_generator.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: activation RSA report failed"
        python expanded_mental_concepts/internals/pca/activation_pca.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: activation PCA failed"
        python expanded_mental_concepts/internals/pca/activation_pca_report_generator.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: activation PCA report failed"
    fi

    local salign_data=$(check_data_exists "$model" "expanded_mental_concepts" "standalone_alignment_report.html")
    local salign_data2=$(check_data_exists "$model" "expanded_mental_concepts" "alignment_results.json")
    if [ -n "$salign_data2" ]; then
        log "  Running standalone/contrast alignment reports..."
        python expanded_mental_concepts/internals/standalone_alignment/standalone_alignment_report_generator.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: standalone alignment report failed"
        python expanded_mental_concepts/internals/contrast_alignment/contrast_alignment_report_generator.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: contrast alignment report failed"
    fi

    log "  Post-processing complete for $model"
}

run_comparison_reports() {
    log "Regenerating cross-model comparison reports..."

    cd "$CODE_DIR"

    module load pyger 2>/dev/null || true
    export PYTHONNOUSERSITE=1
    set +u
    CONDA_BASE="$(conda info --base)"
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate llama2_env
    set -u

    log "  Running status report..."
    python comparisons/2_status_report_generator.py >> "$LOG" 2>&1 || log "  WARN: status report failed"

    log "  Running gray replication summary..."
    python comparisons/3_gray_replication_summary_generator.py >> "$LOG" 2>&1 || log "  WARN: gray replication summary failed"

    log "  Running gray simple summary..."
    python comparisons/4_gray_simple_summary_generator.py >> "$LOG" 2>&1 || log "  WARN: gray simple summary failed"

    log "  Running human-AI summary..."
    python comparisons/5_human_ai_summary_generator.py >> "$LOG" 2>&1 || log "  WARN: human-AI summary failed"

    log "  Running expanded concepts summary..."
    python comparisons/6_expanded_concepts_summary_generator.py >> "$LOG" 2>&1 || log "  WARN: expanded concepts summary failed"

    log "  Running behavioral summary figures..."
    python comparisons/1_behavioral_summary_figures_generator.py >> "$LOG" 2>&1 || log "  WARN: behavioral summary figures failed"

    log "  Running behavioral summary report..."
    python comparisons/1a_behavioral_summary_report_generator.py >> "$LOG" 2>&1 || log "  WARN: behavioral summary report failed"

    log "Comparison reports complete."
}

# =============================================================================
# MAIN LOOP
# =============================================================================

log "=========================================="
log "MONITOR STARTED"
log "Tracking 3 new models: gemma2_9b_it, qwen25_7b_instruct, qwen3_8b"
log "=========================================="

CHECK_NUM=0
MAX_CHECKS=20  # Safety limit

while [ $CHECK_NUM -lt $MAX_CHECKS ]; do
    CHECK_NUM=$((CHECK_NUM + 1))

    # Determine sleep interval: 30 min for first 10 checks (5 hours), then 60 min
    if [ $CHECK_NUM -le 10 ]; then
        INTERVAL=1800
    else
        INTERVAL=3600
    fi

    log "--- Check #$CHECK_NUM ---"

    # Count running/pending jobs per model
    GEM2_ACTIVE=$(all_done "gem2")
    QW25_ACTIVE=$(all_done "qw25")
    QW3_ACTIVE=$(all_done "qw3")

    log "Active jobs: gemma2=$GEM2_ACTIVE, qwen25=$QW25_ACTIVE, qwen3=$QW3_ACTIVE"

    # Check for failures in resubmitted jobs
    FAILURES=$(sacct -u "$USER" --starttime=2026-03-28 --format="JobName%30,State%15" --noheader \
        | grep -E "gem2|qw25|qw3" | grep -v "\.batch" | grep -v "\.extern" \
        | grep -E "FAILED|OUT_OF_ME|CANCELLED|TIMEOUT" | grep -v "413463[4-9]" | grep -v "41346[4-5][0-9]" || true)
    if [ -n "$FAILURES" ]; then
        log "FAILURES DETECTED (post-resubmit):"
        echo "$FAILURES" | while read line; do log "  $line"; done
    fi

    # --- Gemma-2 Wave 2 ---
    if [ $WAVE2_SUBMITTED_GEM2 -eq 0 ]; then
        # Check if all Wave 1 Gemma-2 jobs are done (pairwise + resubmitted)
        GEM2_W1=$(squeue -u "$USER" --format="%j" --noheader | grep -E "(pairwise|entities|gchars|bpca|arsa|crsa|salign|calign)_gem2" | wc -l || true)
        if [ "$GEM2_W1" -eq 0 ]; then
            # Verify at least some data landed
            if [ -n "$(check_data_exists gemma2_9b_it gray_replication pairwise_pca_results.npz)" ]; then
                submit_wave2 "gemma2_it" "gemma2_9b_it"
                WAVE2_SUBMITTED_GEM2=1
            else
                log "Gemma-2 Wave 1 jobs gone but no data found — may have all failed"
            fi
        fi
    fi

    # --- Qwen3 Wave 2 ---
    if [ $WAVE2_SUBMITTED_QW3 -eq 0 ]; then
        QW3_W1=$(squeue -u "$USER" --format="%j" --noheader | grep -E "(pairwise|entities|gchars|bpca|arsa|crsa|salign|calign)_qw3" | wc -l || true)
        if [ "$QW3_W1" -eq 0 ]; then
            if [ -n "$(check_data_exists qwen3_8b gray_replication pairwise_pca_results.npz)" ]; then
                submit_wave2 "qwen3" "qwen3_8b"
                WAVE2_SUBMITTED_QW3=1
            else
                log "Qwen3 Wave 1 jobs gone but no data found — may have all failed"
            fi
        fi
    fi

    # --- Post-processing for each model (when ALL jobs including Wave 2 are done) ---
    if [ $POSTPROC_DONE_QW25 -eq 0 ] && [ "$QW25_ACTIVE" -eq 0 ] && [ $WAVE2_SUBMITTED_QW25 -eq 1 ]; then
        run_postprocessing "qwen25_7b_instruct"
        POSTPROC_DONE_QW25=1
    fi

    if [ $POSTPROC_DONE_GEM2 -eq 0 ] && [ "$GEM2_ACTIVE" -eq 0 ] && [ $WAVE2_SUBMITTED_GEM2 -eq 1 ]; then
        run_postprocessing "gemma2_9b_it"
        POSTPROC_DONE_GEM2=1
    fi

    if [ $POSTPROC_DONE_QW3 -eq 0 ] && [ "$QW3_ACTIVE" -eq 0 ] && [ $WAVE2_SUBMITTED_QW3 -eq 1 ]; then
        run_postprocessing "qwen3_8b"
        POSTPROC_DONE_QW3=1
    fi

    # --- Comparison reports (when ALL models are post-processed) ---
    if [ $REPORTS_DONE -eq 0 ] && [ $POSTPROC_DONE_GEM2 -eq 1 ] && [ $POSTPROC_DONE_QW25 -eq 1 ] && [ $POSTPROC_DONE_QW3 -eq 1 ]; then
        run_comparison_reports
        REPORTS_DONE=1
        log "=========================================="
        log "ALL DONE. Reports regenerated."
        log "=========================================="
        break
    fi

    # If everything is done, exit
    if [ $REPORTS_DONE -eq 1 ]; then
        break
    fi

    log "Sleeping ${INTERVAL}s until next check..."
    sleep $INTERVAL
done

if [ $REPORTS_DONE -eq 0 ]; then
    log "WARNING: Monitor reached max checks ($MAX_CHECKS) without completing all work."
    log "Status: postproc gem2=$POSTPROC_DONE_GEM2, qw25=$POSTPROC_DONE_QW25, qw3=$POSTPROC_DONE_QW3"

    # Run comparison reports with whatever we have
    log "Running comparison reports with available data..."
    run_comparison_reports
fi

log "Monitor script exiting."
