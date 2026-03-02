#!/bin/bash
#SBATCH --job-name=identity_brkdn_ya_bgpt
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/jukebox/graziano/rachel/mind_rep/exp_1/versions/you_are_balanced_gpt/logs/identity_breakdown_%j.out
#SBATCH --error=/jukebox/graziano/rachel/mind_rep/exp_1/versions/you_are_balanced_gpt/logs/identity_breakdown_%j.err

# ---------------------------------------------------------------------------
# Identity breakdown analysis for exp_1/you_are_balanced_gpt.
# Compares linguistic behavior across individual partner agents:
#   Gregory, Rebecca (human), ChatGPT, GPT-4 (AI).
#
# Runs one-way RM-ANOVA + BH-FDR pairwise tests per metric.
# Outputs: identity_breakdown.html, identity_breakdown_stats.txt,
#          identity_breakdown_summary.csv
#
# Submit:
#   sbatch run_identity_breakdown.sh
# ---------------------------------------------------------------------------

# Prevent conda / module PS1 bug under -u
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
export PROJECT_ROOT="/jukebox/graziano/rachel/mind_rep/exp_1/versions/you_are_balanced_gpt"
SCRIPT_DIR="$PROJECT_ROOT/code/analysis"
LOG_DIR="$PROJECT_ROOT/logs"

mkdir -p "$LOG_DIR"

# cd to analysis dir so relative imports (utils/) resolve correctly
cd "$SCRIPT_DIR" || { echo "FATAL: Cannot cd to $SCRIPT_DIR"; exit 1; }

echo "[$(date)] Starting identity_breakdown (you_are_balanced_gpt) on host $HOSTNAME"
echo "  PROJECT_ROOT = $PROJECT_ROOT"
echo "  SCRIPT_DIR   = $SCRIPT_DIR"

python identity_breakdown.py

echo "[$(date)] identity_breakdown finished"
