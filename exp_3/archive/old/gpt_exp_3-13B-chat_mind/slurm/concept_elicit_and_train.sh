#!/bin/bash
#SBATCH --job-name=concept_elicit
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_3-13B-chat_mind/logs/concept_elicit_%A.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_3-13B-chat_mind/logs/concept_elicit_%A.err
# -------------------------------------------------------------
# Phase 1: Extract concept activations (~48 forward passes)
# Phase 2: Train concept probes + alignment analysis
# Both are fast — run in a single job.
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

echo "[$(date)] Starting concept elicitation on $HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Phase 1: Extract concept activations
python 1_elicit_concept_vectors.py

# Phase 2: Train concept probes + alignment with Exp 2b
python 2_train_concept_probes.py

echo "[$(date)] Finished concept elicitation + probe training"