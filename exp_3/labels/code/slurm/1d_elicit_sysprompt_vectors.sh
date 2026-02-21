#!/bin/bash
#SBATCH --job-name=elicit_sysprompt
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_3/llama_exp_3-13B-chat/logs/elicit/sysprompt_%A.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_3/llama_exp_3-13B-chat/logs/elicit/sysprompt_%A.err
# -------------------------------------------------------------
# System prompt partner identity elicitation (Phase 1d)
#
# Extracts activations from system prompts ("You are talking to
# Sarah (a human)" / "You are talking to ChatGPT (an AI chatbot)")
# to test whether the model's initial partner representation
# already contains structured mental-property content.
#
# Runs both modes (contrasts + standalone) sequentially since
# there are only 28 prompts total — should complete in <30 min.
#
# Output:
#   data/concept_activations/contrasts/18_sysprompt_labeled/
#   data/concept_activations/standalone/20_sysprompt_talkto_human/
#   data/concept_activations/standalone/21_sysprompt_talkto_ai/
#   data/concept_activations/standalone/22_sysprompt_bare_human/
#   data/concept_activations/standalone/23_sysprompt_bare_ai/
#
# These integrate with the existing 1b alignment analysis pipeline.
# After running, re-run 1b_alignment_analysis.py to include dim 19.
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

PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_3/llama_exp_3-13B-chat"
mkdir -p "$PROJECT_ROOT/logs/elicit"
cd "$PROJECT_ROOT" || { echo "FATAL: Cannot cd to $PROJECT_ROOT"; exit 1; }

echo "[$(date)] Starting system prompt elicitation"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

# Contrast mode: human names vs AI names
echo ""
echo "[$(date)] Running contrasts mode..."
python code/analysis/alignment/1d_elicit_sysprompt_vectors.py --only contrasts

# Standalone mode: all prompts pooled
echo ""
echo "[$(date)] Running standalone mode..."
python code/analysis/alignment/1d_elicit_sysprompt_vectors.py --only standalone

echo ""
echo "[$(date)] System prompt elicitation complete"
echo "  Next: re-run 1b_alignment_analysis.py to include new dims"