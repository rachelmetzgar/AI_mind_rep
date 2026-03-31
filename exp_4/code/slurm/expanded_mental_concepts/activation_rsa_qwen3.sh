#!/bin/bash
#SBATCH --job-name=exp4_arsa_qw3
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_4/logs/expanded_mental_concepts/activation_rsa_qwen3_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_4/logs/expanded_mental_concepts/activation_rsa_qwen3_%j.err
#SBATCH --gres=gpu:1 --mem=64G --time=4:00:00 --cpus-per-task=4

export HF_HOME="/mnt/cup/labs/graziano/rachel/.cache_huggingface"
export HF_HUB_CACHE="/mnt/cup/labs/graziano/rachel/.cache_huggingface/hub"
export HF_HUB_DISABLE_XET=1

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

python expanded_mental_concepts/internals/rsa/activation_rsa.py --model qwen3_8b
