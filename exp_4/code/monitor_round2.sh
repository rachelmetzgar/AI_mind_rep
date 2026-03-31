#!/bin/bash
# =============================================================================
# Monitor round 2 new model jobs (gemma2_2b_it, gemma2_2b, gemma2_9b, qwen25_7b)
# Submit Wave 2 when Wave 1 completes, then run post-processing and reports.
# =============================================================================

set -u
export PS1=${PS1:-}

LOG="/mnt/cup/labs/graziano/rachel/mind_rep/exp_4/logs/monitor_round2.log"
SLURM_DIR="/mnt/cup/labs/graziano/rachel/mind_rep/exp_4/code/slurm"
CODE_DIR="/mnt/cup/labs/graziano/rachel/mind_rep/exp_4/code"
RESULTS_DIR="/mnt/cup/labs/graziano/rachel/mind_rep/exp_4/results"

# Job IDs from this round (4138488-4138519)
ROUND2_START=4138488

declare -A WAVE2_SUBMITTED
declare -A POSTPROC_DONE
REPORTS_DONE=0

MODELS=("gemma2_2b_it" "gemma2_2b" "gemma2_9b" "qwen25_7b")
MODEL_KEYS=("gemma2_2b_it" "gemma2_2b" "gemma2_9b" "qwen25_7b")
JOB_TAGS=("g2bit" "g2b" "g9b" "q25b")

for m in "${MODELS[@]}"; do
    WAVE2_SUBMITTED[$m]=0
    POSTPROC_DONE[$m]=0
done

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

count_active_jobs() {
    local tag="$1"
    squeue -u "$USER" --format="%j" --noheader | grep -c "$tag" 2>/dev/null || echo 0
}

check_data() {
    local model="$1" file="$2"
    find "$RESULTS_DIR/$model" -name "$file" 2>/dev/null | head -1
}

setup_conda() {
    module load pyger 2>/dev/null || true
    export PYTHONNOUSERSITE=1
    set +u
    CONDA_BASE="$(conda info --base)"
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate llama2_env
    set -u
}

submit_wave2() {
    local model="$1"
    log "Submitting Wave 2 for $model"
    for f in "$SLURM_DIR/gray_replication/3_individual_${model}.sh" \
             "$SLURM_DIR/gray_simple/2_neural_pca_${model}.sh" \
             "$SLURM_DIR/human_ai_adaptation/2_gray_names_only_${model}.sh"; do
        if [ -f "$f" ]; then
            local jid=$(sbatch "$f" 2>&1 | awk '{print $4}')
            log "  Submitted $(basename $f): job $jid"
        fi
    done
}

run_postprocessing() {
    local model="$1"
    log "Running post-processing for $model"
    cd "$CODE_DIR"
    setup_conda

    # Gray replication
    if [ -n "$(check_data $model pairwise_pca_results.npz)" ]; then
        log "  gray_replication post-processing..."
        python gray_replication/behavior/compute_excl_pca.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: compute_excl_pca failed"
        python gray_replication/behavior/compute_human_comparisons.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: compute_human_comparisons failed"
        for cond in with_self without_self; do
            python gray_replication/behavior/make_loadings_bar_chart.py --model "$model" --condition "$cond" >> "$LOG" 2>&1 || log "  WARN: make_loadings ($cond) failed"
            python gray_replication/behavior/make_condition_reports.py --model "$model" --condition "$cond" >> "$LOG" 2>&1 || log "  WARN: make_reports ($cond) failed"
        done
    fi

    # Gray simple reports
    if [ -n "$(check_data $model rsa_results.json)" ]; then
        python gray_simple/internals/1a_rsa_report_generator.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: RSA report failed"
    fi
    if [ -n "$(check_data $model neural_pca_results.npz)" ]; then
        python gray_simple/internals/2a_neural_pca_report_generator.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: neural PCA report failed"
    fi

    # Human-AI adaptation reports
    if [ -n "$(check_data $model pairwise_categorical_analysis.json)" ]; then
        python human_ai_adaptation/behavior/1a_gray_chars_pca_report_generator.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: PCA report failed"
        python human_ai_adaptation/behavior/1b_gray_chars_detailed_report_generator.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: detailed report failed"
        python human_ai_adaptation/behavior/1c_gray_chars_rsa_report_generator.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: RSA report failed"
    fi

    # Expanded mental concepts
    if [ -n "$(check_data $model pairwise_pca_results.npz)" ]; then
        python expanded_mental_concepts/behavior/pca/behavioral_pca_report_generator.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: bpca report failed"
        python expanded_mental_concepts/behavior/pca/matched_behavioral_pca.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: matched bpca failed"
        python expanded_mental_concepts/behavior/pca/matched_behavioral_pca_report_generator.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: matched bpca report failed"
        python expanded_mental_concepts/behavior/pca/behavioral_attribution_report_generator.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: attribution report failed"
        python expanded_mental_concepts/behavior/pca/detailed_response_report_generator.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: detailed report failed"
    fi
    if [ -n "$(check_data $model all_character_activations.npz)" ]; then
        python expanded_mental_concepts/internals/rsa/activation_rsa_report_generator.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: arsa report failed"
        python expanded_mental_concepts/internals/pca/activation_pca.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: activation pca failed"
        python expanded_mental_concepts/internals/pca/activation_pca_report_generator.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: activation pca report failed"
    fi
    if [ -n "$(check_data $model alignment_results.json)" ]; then
        python expanded_mental_concepts/internals/standalone_alignment/standalone_alignment_report_generator.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: salign report failed"
        python expanded_mental_concepts/internals/contrast_alignment/contrast_alignment_report_generator.py --model "$model" >> "$LOG" 2>&1 || log "  WARN: calign report failed"
    fi

    log "  Post-processing complete for $model"
}

run_comparison_reports() {
    log "Regenerating cross-model comparison reports..."
    cd "$CODE_DIR"
    setup_conda

    for script in \
        comparisons/2_status_report_generator.py \
        comparisons/3_gray_replication_summary_generator.py \
        comparisons/4_gray_simple_summary_generator.py \
        comparisons/5_human_ai_summary_generator.py \
        comparisons/6_expanded_concepts_summary_generator.py \
        comparisons/1_behavioral_summary_figures_generator.py \
        comparisons/1a_behavioral_summary_report_generator.py; do
        log "  Running $(basename $script)..."
        python "$script" >> "$LOG" 2>&1 || log "  WARN: $(basename $script) failed"
    done
    log "Comparison reports complete."
}

# =============================================================================
# MAIN LOOP
# =============================================================================

log "=========================================="
log "MONITOR ROUND 2 STARTED"
log "Models: ${MODELS[*]}"
log "=========================================="

CHECK_NUM=0
MAX_CHECKS=20

while [ $CHECK_NUM -lt $MAX_CHECKS ]; do
    CHECK_NUM=$((CHECK_NUM + 1))
    [ $CHECK_NUM -le 10 ] && INTERVAL=1800 || INTERVAL=3600

    log "--- Check #$CHECK_NUM ---"

    ALL_POSTPROC=1

    for i in "${!MODELS[@]}"; do
        m="${MODELS[$i]}"
        tag="${JOB_TAGS[$i]}"
        active=$(count_active_jobs "$tag")
        log "  $m: $active active jobs"

        # Submit Wave 2 when Wave 1 is done
        if [ "${WAVE2_SUBMITTED[$m]}" -eq 0 ] && [ "$active" -eq 0 ]; then
            if [ -n "$(check_data $m pairwise_pca_results.npz)" ]; then
                submit_wave2 "$m"
                WAVE2_SUBMITTED[$m]=1
            else
                log "  $m: Wave 1 done but no pairwise data — checking if entity data exists"
                if [ -n "$(check_data $m all_entity_activations.npz)" ]; then
                    submit_wave2 "$m"
                    WAVE2_SUBMITTED[$m]=1
                else
                    log "  $m: No data found yet, will recheck"
                fi
            fi
        fi

        # Post-process when all jobs done (including Wave 2)
        if [ "${POSTPROC_DONE[$m]}" -eq 0 ] && [ "${WAVE2_SUBMITTED[$m]}" -eq 1 ] && [ "$active" -eq 0 ]; then
            run_postprocessing "$m"
            POSTPROC_DONE[$m]=1
        fi

        [ "${POSTPROC_DONE[$m]}" -eq 0 ] && ALL_POSTPROC=0
    done

    # Comparison reports when all models done
    if [ $REPORTS_DONE -eq 0 ] && [ $ALL_POSTPROC -eq 1 ]; then
        run_comparison_reports
        REPORTS_DONE=1
        log "=========================================="
        log "ALL DONE. Reports regenerated with all 11 models."
        log "=========================================="
        break
    fi

    log "Sleeping ${INTERVAL}s..."
    sleep $INTERVAL
done

if [ $REPORTS_DONE -eq 0 ]; then
    log "WARNING: Max checks reached. Running reports with available data..."
    run_comparison_reports
fi

log "Monitor script exiting."
