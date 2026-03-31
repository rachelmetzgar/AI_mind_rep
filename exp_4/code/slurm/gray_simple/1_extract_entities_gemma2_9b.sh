#!/bin/bash
#SBATCH --job-name=exp4_entities_g9b
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_4/logs/gray_simple/1_extract_entities_gemma2_9b_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_4/logs/gray_simple/1_extract_entities_gemma2_9b_%j.err
#SBATCH --gres=gpu:1 --mem=64G --time=2:00:00 --cpus-per-task=4

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

python gray_simple/internals/1_extract_entity_representations.py --model gemma2_9b --both
