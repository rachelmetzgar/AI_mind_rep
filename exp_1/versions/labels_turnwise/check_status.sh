#!/bin/bash
# Quick status checker for labels_turnwise experiment
# Usage: bash check_status.sh

echo "=================================================================="
echo "BALANCED_GPT STATUS CHECK"
echo "$(date)"
echo "=================================================================="

# Job status
echo ""
echo "SLURM Jobs:"
echo "  Data generation (3559400):"
squeue -j 3559400 -o "    %.18i %.8T" 2>&1 | tail -5
RUNNING=$(squeue -j 3559400 -h -t RUNNING 2>/dev/null | wc -l)
PENDING=$(squeue -j 3559400 -h -t PENDING 2>/dev/null | wc -l)
TOTAL=$((RUNNING + PENDING))
if [ $TOTAL -eq 0 ]; then
    echo "    Status: COMPLETED or not found"
else
    echo "    Status: $RUNNING running, $PENDING pending (out of 50 total)"
fi

echo ""
echo "  Analysis pipeline (3559471):"
PIPELINE_STATE=$(squeue -j 3559471 -h -o "%.8T" 2>/dev/null)
if [ -z "$PIPELINE_STATE" ]; then
    echo "    Status: COMPLETED or not found"
else
    echo "    Status: $PIPELINE_STATE"
fi

# Data files
echo ""
echo "Data Files:"
DATA_DIR="/jukebox/graziano/rachel/mind_rep/exp_1/versions/labels_turnwise/data/meta-llama-Llama-2-13b-chat-hf/0.8"
if [ -d "$DATA_DIR" ]; then
    CSV_COUNT=$(ls -1 "$DATA_DIR"/s*.csv 2>/dev/null | wc -l)
    echo "  Per-subject CSVs: $CSV_COUNT / 50"
    if [ -f "$DATA_DIR/combined_text_data.csv" ]; then
        ROWS=$(wc -l < "$DATA_DIR/combined_text_data.csv")
        SIZE=$(du -h "$DATA_DIR/combined_text_data.csv" | cut -f1)
        echo "  Combined CSV: ✓ ($ROWS rows, $SIZE)"
    else
        echo "  Combined CSV: Not yet created"
    fi
else
    echo "  Data directory not yet created"
fi

# Results
echo ""
echo "Results:"
RESULTS_DIR="/jukebox/graziano/rachel/mind_rep/exp_1/versions/labels_turnwise/results/meta-llama-Llama-2-13b-chat-hf/0.8"
if [ -d "$RESULTS_DIR" ]; then
    if [ -f "$RESULTS_DIR/identity_breakdown.html" ]; then
        SIZE=$(du -h "$RESULTS_DIR/identity_breakdown.html" | cut -f1)
        echo "  HTML report: ✓ ($SIZE)"
        echo "  Location: $RESULTS_DIR/identity_breakdown.html"
    else
        echo "  HTML report: Not yet generated"
    fi

    if [ -f "$RESULTS_DIR/identity_breakdown_stats.txt" ]; then
        echo "  Stats text: ✓"
    fi

    if [ -f "$RESULTS_DIR/identity_breakdown_summary.csv" ]; then
        echo "  Summary CSV: ✓"
    fi
else
    echo "  Results directory not yet created"
fi

# Recent logs
echo ""
echo "Recent Log Activity:"
LOG_DIR="/jukebox/graziano/rachel/mind_rep/exp_1/versions/labels_turnwise/logs"
if [ -d "$LOG_DIR" ]; then
    echo "  Last 3 modified files:"
    ls -lt "$LOG_DIR"/*.{out,err,log} 2>/dev/null | head -3 | awk '{print "    " $9 " (" $6, $7, $8 ")"}'

    # Check for errors
    ERROR_COUNT=$(grep -l "Error\|ERROR\|FATAL\|Traceback" "$LOG_DIR"/*.err 2>/dev/null | wc -l)
    if [ $ERROR_COUNT -gt 0 ]; then
        echo "  ⚠️  Found errors in $ERROR_COUNT log files"
    fi
fi

echo ""
echo "=================================================================="
echo "To monitor in real-time:"
echo "  watch -n 30 bash check_status.sh"
echo "=================================================================="
