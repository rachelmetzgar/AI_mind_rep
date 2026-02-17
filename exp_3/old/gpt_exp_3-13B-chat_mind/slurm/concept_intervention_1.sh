#!/bin/bash
#SBATCH --job-name=concept_v1
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_3-13B-chat_mind/logs/concept_v1_%A.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_3-13B-chat_mind/logs/concept_v1_%A.err
# -------------------------------------------------------------
# Concept injection V1: single-turn test questions + GPT judge
# Uses mean-difference vectors (more trustworthy than probe weights
# given the alignment analysis results).
#
# Chain after concept elicitation:
#   sbatch --dependency=afterok:<ELICIT_JOBID> concept_intervention_v1.sh
# -------------------------------------------------------------

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

PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_3-13B-chat_mind"
mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

# OpenAI API key for GPT judge
export OPENAI_API_KEY="$(cat /jukebox/graziano/rachel/.openai_key 2>/dev/null || echo '')"
if [ -z "$OPENAI_API_KEY" ]; then
    echo "[WARN] OPENAI_API_KEY not found — GPT judge will fail"
fi

echo "[$(date)] Starting concept intervention V1"

# Mean-difference vectors (primary — data-driven, no optimization artifacts)
python 3_concept_intervention.py --mode v1 --vector_source probe --judge local

echo "[$(date)] Finished concept intervention V1"