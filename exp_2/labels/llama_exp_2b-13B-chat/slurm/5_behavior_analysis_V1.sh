#!/bin/bash
#SBATCH --job-name=behav_V1_peak15
#SBATCH --partition=all
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --array=0-1
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_2/labels/llama_exp_2b-13B-chat/logs/behavior_v1/behavior_V1_%A_%a.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_2/labels/llama_exp_2b-13B-chat/logs/behavior_v1/behavior_V1_%A_%a.err
# ---------------------------------------------------------------------------
# V1 Behavioral analysis: peak_15 strategy only, N=4 and N=5.
# Array: 0→4, 1→5
# Walks peak_15/{control_probes,reading_probes_peak}/is_{N}/
# No GPU needed.
#
# Submit:
#   sbatch 5_behavior_analysis_V1.sh
# ---------------------------------------------------------------------------

STRENGTHS=(4 5)
N=${STRENGTHS[$SLURM_ARRAY_TASK_ID]}

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

PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_2/labels/llama_exp_2b-13B-chat"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] V1 behavioral analysis | strategy=peak_15 | strength=$N | host=$HOSTNAME"

python 5_behavior_analysis.py \
    --version v1 \
    --strength "$N" \
    --input data/intervention_results/V1/peak_15

echo "[$(date)] V1 behavioral analysis finished | strategy=peak_15 | strength=$N"