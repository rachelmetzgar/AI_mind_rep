#!/bin/bash
#SBATCH --job-name=cg_internals_chat
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_4/logs/expanded_mental_concepts/activation_rsa_chat_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_4/logs/expanded_mental_concepts/activation_rsa_chat_%j.err
#SBATCH --gres=gpu:1 --mem=64G --time=2:00:00 --cpus-per-task=4

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

cd /mnt/cup/labs/graziano/rachel/mind_rep/exp_4/code

python human_ai_characters/neural/1_extract_character_activations.py --model llama2_13b_chat
